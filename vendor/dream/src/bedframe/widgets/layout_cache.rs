//! Bounded text-layout cache for widget text.
//!
//! Layouts are cached at *physical* size (`size * scale`, canonicalized to
//! 1/8 px so cache admission never depends on which fractional size was
//! asked first — the same canonical key always rasterizes identically).
//! Values are shared `Arc`s: a hit costs a bump, not a copy.
//!
//! Lookups borrow: a hit compares the caller's `&str` and plain fields
//! against the stored full key — no `String` is allocated until a miss
//! needs one. Hash-only equality is never trusted; a bucket holds every
//! field that decides the layout and compares all of them.
//!
//! Bounds are fixed: 2048 entries and a 2 MiB retained-payload cap, LRU
//! eviction. An oversized layout is drawn but not admitted. The cap is
//! admission-side, not a live-memory guarantee: `Arc`s held by a finished
//! frame keep their layout alive outside the budget.
//!
//! The entry bound tracks the byte bound rather than undercutting it. A
//! typical label layout retains about 1 KiB, so 1024 entries capped the
//! cache near 1 MiB and left half of the 2 MiB payload budget unused. That
//! mattered for dense screens: a 1236-layout frame overflowed 1024 slots,
//! and LRU then evicted exactly the entries the next frame would ask for
//! first, so every layout was rebuilt every frame. 2048 slots let the byte
//! cap decide eviction for layouts of ordinary size.

use super::super::font::{Align, FontId, TextLayout};
use std::collections::HashMap;
use std::sync::Arc;

const MAX_ENTRIES: usize = 2048;
const MAX_BYTES: usize = 2 * 1024 * 1024;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Everything that decides a layout, as a borrowed request — the owned
/// `String` exists only inside admitted entries.
pub(super) struct LayoutRequest<'a> {
    pub text: &'a str,
    pub font: FontId,
    /// Canonical physical size (1/8 px, see [`LayoutCache::canonical_eighth`]).
    pub px_eighth: u32,
    /// Physical wrap width in 1/8 px; `u32::MAX` = unwrapped.
    pub wrap_eighth: u32,
    pub align: Align,
    pub line_height_bits: u32,
    pub letter_spacing_bits: u32,
    /// `Fonts` registry revision (a new face changes metrics).
    pub fonts_rev: u64,
    /// Atlas epoch (glyph regions after a reset differ).
    pub atlas_epoch: u64,
}

impl LayoutRequest<'_> {
    fn hash(&self) -> u64 {
        let mut hash = FNV_OFFSET;
        hash = fnv_u64(hash, self.text.len() as u64);
        for byte in self.text.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        hash = fnv_u64(hash, self.font.0 as u64);
        hash = fnv_u64(hash, u64::from(self.px_eighth));
        hash = fnv_u64(hash, u64::from(self.wrap_eighth));
        hash = fnv_u64(hash, self.align as u64);
        hash = fnv_u64(hash, u64::from(self.line_height_bits));
        hash = fnv_u64(hash, u64::from(self.letter_spacing_bits));
        hash = fnv_u64(hash, self.fonts_rev);
        fnv_u64(hash, self.atlas_epoch)
    }

    /// Exact equality with a stored entry — never hash-only.
    fn matches(&self, entry: &Entry) -> bool {
        entry.text == self.text
            && entry.font == self.font
            && entry.px_eighth == self.px_eighth
            && entry.wrap_eighth == self.wrap_eighth
            && entry.align == self.align
            && entry.line_height_bits == self.line_height_bits
            && entry.letter_spacing_bits == self.letter_spacing_bits
            && entry.fonts_rev == self.fonts_rev
            && entry.atlas_epoch == self.atlas_epoch
    }
}

struct Entry {
    layout: Arc<TextLayout>,
    /// Retained-payload estimate: struct + glyph buffer + owned text.
    bytes: usize,
    hash: u64,
    hash_next: Option<usize>,
    /// LRU order in place of the old last-used tick: `prev` toward the
    /// eldest (`head`), `next` toward the most recently used (`tail`).
    prev: Option<usize>,
    next: Option<usize>,
    // Owned full key (allocated on miss).
    text: String,
    font: FontId,
    px_eighth: u32,
    wrap_eighth: u32,
    align: Align,
    line_height_bits: u32,
    letter_spacing_bits: u32,
    fonts_rev: u64,
    atlas_epoch: u64,
}

/// Bounded physical-layout cache. See the module docs.
#[derive(Default)]
pub(super) struct LayoutCache {
    slots: Vec<Option<Entry>>,
    free: Vec<usize>,
    /// Hash bucket -> head slot of its collision chain: hits compare
    /// every field, so a collision is just a slightly longer chain.
    buckets: HashMap<u64, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    entries: usize,
    bytes: usize,
    hits: u64,
    misses: u64,
}

impl LayoutCache {
    /// Physical `size` canonicalized to 1/8 px — the same eighth always
    /// means the same raster, so 16.000 and 16.001 do not fork the cache.
    pub(super) fn canonical_eighth(px: f32) -> u32 {
        (px.max(0.0) * 8.0).round() as u32
    }

    fn find(&mut self, request: &LayoutRequest<'_>, hash: u64) -> Option<Arc<TextLayout>> {
        let mut slot = self.buckets.get(&hash).copied();
        while let Some(index) = slot {
            let (next, hit) = {
                let entry = self.slots[index].as_ref().expect("live slot");
                (entry.hash_next, request.matches(entry))
            };
            if hit {
                self.unlink_lru(index);
                self.push_tail(index);
                self.hits += 1;
                return Some(Arc::clone(
                    &self.slots[index].as_ref().expect("live slot").layout,
                ));
            }
            slot = next;
        }
        None
    }

    /// A cached physical layout for `request`, or `None` on miss.
    pub(super) fn get(&mut self, request: &LayoutRequest<'_>) -> Option<Arc<TextLayout>> {
        let hash = request.hash();
        if let Some(layout) = self.find(request, hash) {
            return Some(layout);
        }
        self.misses += 1;
        None
    }

    /// Admit a freshly computed layout. Oversized layouts are skipped
    /// rather than admitted.
    pub(super) fn insert(&mut self, request: &LayoutRequest<'_>, layout: Arc<TextLayout>) {
        let bytes = std::mem::size_of::<Entry>()
            + request.text.len() * 2 // owned String content + spare headroom
            + request.text.len()
            + layout.glyphs.capacity() * std::mem::size_of::<super::super::font::PositionedGlyph>()
            + layout.stops.capacity() * std::mem::size_of::<(f32, usize)>()
            + std::mem::size_of::<TextLayout>();
        if bytes > MAX_BYTES / 8 {
            return;
        }
        let hash = request.hash();
        {
            let mut slot = self.buckets.get(&hash).copied();
            while let Some(index) = slot {
                let entry = self.slots[index].as_ref().expect("live slot");
                let next = entry.hash_next;
                if request.matches(entry) {
                    self.remove_slot(index);
                    break;
                }
                slot = next;
            }
        }
        // Retire eldest entries until the new one fits both bounds.
        while self.entries >= MAX_ENTRIES || self.bytes + bytes > MAX_BYTES {
            let Some(head) = self.head else {
                break;
            };
            self.remove_slot(head);
        }
        let slot = self.alloc_slot(Entry {
            layout,
            bytes,
            hash,
            hash_next: self.buckets.get(&hash).copied(),
            prev: None,
            next: None,
            text: request.text.to_string(),
            font: request.font,
            px_eighth: request.px_eighth,
            wrap_eighth: request.wrap_eighth,
            align: request.align,
            line_height_bits: request.line_height_bits,
            letter_spacing_bits: request.letter_spacing_bits,
            fonts_rev: request.fonts_rev,
            atlas_epoch: request.atlas_epoch,
        });
        self.buckets.insert(hash, slot);
        self.push_tail(slot);
        self.bytes += bytes;
        self.entries += 1;
    }

    fn alloc_slot(&mut self, entry: Entry) -> usize {
        if let Some(slot) = self.free.pop() {
            self.slots[slot] = Some(entry);
            slot
        } else {
            self.slots.push(Some(entry));
            self.slots.len() - 1
        }
    }

    fn unlink_lru(&mut self, slot: usize) {
        let entry = self.slots[slot].as_ref().expect("live slot");
        let (prev, next) = (entry.prev, entry.next);
        match prev {
            Some(prev) => self.slots[prev].as_mut().expect("live slot").next = next,
            None => self.head = next,
        }
        match next {
            Some(next) => self.slots[next].as_mut().expect("live slot").prev = prev,
            None => self.tail = prev,
        }
        let entry = self.slots[slot].as_mut().expect("live slot");
        entry.prev = None;
        entry.next = None;
    }

    fn push_tail(&mut self, slot: usize) {
        let tail = self.tail;
        {
            let entry = self.slots[slot].as_mut().expect("live slot");
            entry.prev = tail;
            entry.next = None;
        }
        match tail {
            Some(tail) => self.slots[tail].as_mut().expect("live slot").next = Some(slot),
            None => self.head = Some(slot),
        }
        self.tail = Some(slot);
    }

    fn unlink_hash(&mut self, slot: usize) {
        let hash = self.slots[slot].as_ref().expect("live slot").hash;
        let mut link = self.buckets.get(&hash).copied();
        let mut prev: Option<usize> = None;
        while let Some(index) = link {
            if index == slot {
                let next = self.slots[index].as_ref().expect("live slot").hash_next;
                match prev {
                    Some(prev) => {
                        self.slots[prev].as_mut().expect("live slot").hash_next = next;
                    }
                    None => match next {
                        Some(next) => {
                            self.buckets.insert(hash, next);
                        }
                        None => {
                            self.buckets.remove(&hash);
                        }
                    },
                }
                self.slots[slot].as_mut().expect("live slot").hash_next = None;
                return;
            }
            prev = link;
            link = self.slots[index].as_ref().expect("live slot").hash_next;
        }
    }

    fn remove_slot(&mut self, slot: usize) {
        self.unlink_hash(slot);
        self.unlink_lru(slot);
        let entry = self.slots[slot].take().expect("live slot");
        self.bytes = self.bytes.saturating_sub(entry.bytes);
        self.entries -= 1;
        self.free.push(slot);
    }

    /// (hits, misses, entries, retained byte estimate) — retained for the
    /// bench harness and the implementation report.
    pub(super) fn stats(&self) -> LayoutStats {
        LayoutStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries,
            retained_bytes: self.bytes,
        }
    }

    #[cfg(test)]
    fn insert_with_hash(
        &mut self,
        request: &LayoutRequest<'_>,
        layout: Arc<TextLayout>,
        hash: u64,
    ) {
        let bytes = std::mem::size_of::<Entry>() + request.text.len() * 3;
        let slot = self.alloc_slot(Entry {
            layout,
            bytes,
            hash,
            hash_next: self.buckets.get(&hash).copied(),
            prev: None,
            next: None,
            text: request.text.to_string(),
            font: request.font,
            px_eighth: request.px_eighth,
            wrap_eighth: request.wrap_eighth,
            align: request.align,
            line_height_bits: request.line_height_bits,
            letter_spacing_bits: request.letter_spacing_bits,
            fonts_rev: request.fonts_rev,
            atlas_epoch: request.atlas_epoch,
        });
        self.buckets.insert(hash, slot);
        self.push_tail(slot);
        self.bytes += bytes;
        self.entries += 1;
    }

    #[cfg(test)]
    fn probe(&mut self, request: &LayoutRequest<'_>, hash: u64) -> Option<Arc<TextLayout>> {
        self.find(request, hash)
    }

    #[cfg(test)]
    fn slot_of(&self, text: &str, hash: u64) -> Option<usize> {
        let mut link = self.buckets.get(&hash).copied();
        while let Some(index) = link {
            let entry = self.slots[index].as_ref().expect("live slot");
            if entry.text == text {
                return Some(index);
            }
            link = entry.hash_next;
        }
        None
    }
}

/// Retained layout-cache diagnostics (raw numbers, no interpretation).
#[derive(Clone, Copy, Debug)]
pub struct LayoutStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub retained_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(text: &str, px_eighth: u32) -> LayoutRequest<'_> {
        LayoutRequest {
            text,
            font: FontId(0),
            px_eighth,
            wrap_eighth: u32::MAX,
            align: Align::Left,
            line_height_bits: 0,
            letter_spacing_bits: 0,
            fonts_rev: 0,
            atlas_epoch: 0,
        }
    }

    fn layout() -> Arc<TextLayout> {
        Arc::new(TextLayout::default())
    }

    #[test]
    fn layout_cache_hit_is_exact_key_not_hash_only() {
        let mut cache = LayoutCache::default();
        let req = request("abc", 128);
        cache.insert(&req, layout());
        assert!(cache.get(&request("abc", 128)).is_some());
        assert!(cache.get(&request("abd", 128)).is_none());
        assert!(cache.get(&request("abc", 129)).is_none());
        let mut other = request("abc", 128);
        other.wrap_eighth = 400;
        assert!(cache.get(&other).is_none());
        let mut other = request("abc", 128);
        other.align = Align::Center;
        assert!(cache.get(&other).is_none());
        let mut other = request("abc", 128);
        other.line_height_bits = 1;
        assert!(cache.get(&other).is_none());
        let mut other = request("abc", 128);
        other.fonts_rev = 1;
        assert!(cache.get(&other).is_none());
        let mut other = request("abc", 128);
        other.atlas_epoch = 1;
        assert!(cache.get(&other).is_none());
        let mut other = request("abc", 128);
        other.font = FontId(1);
        assert!(cache.get(&other).is_none());
        let mut other = request("abc", 128);
        other.letter_spacing_bits = 1;
        assert!(cache.get(&other).is_none());
    }

    #[test]
    fn layout_cache_collision_chain_compares_full_keys() {
        let mut cache = LayoutCache::default();
        cache.insert_with_hash(&request("aaa", 128), layout(), 7);
        cache.insert_with_hash(&request("bbb", 128), layout(), 7);
        cache.insert_with_hash(&request("ccc", 128), layout(), 7);
        assert!(cache.probe(&request("ddd", 128), 7).is_none());
        let mid = cache.slot_of("bbb", 7).expect("bbb on the chain");
        cache.remove_slot(mid);
        assert_eq!(cache.entries, 2);
        assert!(cache.probe(&request("aaa", 128), 7).is_some());
        assert!(cache.probe(&request("ccc", 128), 7).is_some());
        assert!(cache.probe(&request("bbb", 128), 7).is_none());
        cache.remove_slot(cache.head.expect("an eldest"));
        cache.remove_slot(cache.tail.expect("a newest"));
        assert_eq!(cache.entries, 0);
        assert!(cache.buckets.is_empty());
        assert_eq!(cache.head, None);
        assert_eq!(cache.tail, None);
        assert_eq!(cache.bytes, 0);
    }

    #[test]
    fn layout_cache_hit_promotes_over_eviction() {
        let mut cache = LayoutCache::default();
        for i in 0..MAX_ENTRIES {
            cache.insert(&request(&format!("s{i}"), 128), layout());
        }
        assert_eq!(cache.entries, MAX_ENTRIES);
        assert!(cache.get(&request("s0", 128)).is_some());
        cache.insert(&request("new", 128), layout());
        assert_eq!(cache.entries, MAX_ENTRIES);
        assert!(cache.get(&request("s0", 128)).is_some(), "hit survives");
        assert!(cache.get(&request("s1", 128)).is_none(), "eldest evicted");
        assert!(cache.get(&request("new", 128)).is_some());
    }

    /// A dense frame of distinct labels must stay cached: 1236 layouts is
    /// above the old 1024-entry bound, where LRU evicted exactly the
    /// entries the next frame asked for first, so every layout was rebuilt
    /// every frame.
    #[test]
    fn layout_cache_dense_working_set_does_not_thrash() {
        let mut cache = LayoutCache::default();
        let texts: Vec<String> = (0..1236).map(|i| format!("Row {i:04}")).collect();
        for text in &texts {
            cache.insert(&request(text, 128), layout());
        }
        assert_eq!(cache.entries, texts.len());
        for text in &texts {
            assert!(
                cache.get(&request(text, 128)).is_some(),
                "{text} was evicted before its next use"
            );
        }
        let stats = cache.stats();
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hits as usize, texts.len());
    }

    #[test]
    fn layout_cache_byte_bound_evicts_and_oversize_is_refused() {
        let mut cache = LayoutCache::default();
        let big = "x".repeat(MAX_BYTES);
        let before = cache.stats();
        cache.insert(&request(&big, 128), layout());
        assert_eq!(cache.stats().entries, before.entries);
        let chunk_len = MAX_BYTES / 32;
        for i in 0..40 {
            let text = format!("{:05}{}", i, "y".repeat(chunk_len));
            cache.insert(&request(&text, 128), layout());
        }
        let stats = cache.stats();
        assert!(
            stats.retained_bytes <= MAX_BYTES,
            "retained {} over the cap",
            stats.retained_bytes
        );
        assert!(
            stats.entries > 0 && stats.entries < 40,
            "some admitted, some evicted: {}",
            stats.entries
        );
        let oldest = format!("{:05}{}", 0, "y".repeat(chunk_len));
        let newest = format!("{:05}{}", 39, "y".repeat(chunk_len));
        assert!(cache.get(&request(&oldest, 128)).is_none(), "evicted");
        assert!(cache.get(&request(&newest, 128)).is_some(), "retained");
    }

    #[test]
    fn layout_cache_held_arc_outlives_eviction() {
        let mut cache = LayoutCache::default();
        let held = Arc::new(TextLayout {
            width: 42.0,
            ..Default::default()
        });
        let req = request("held", 128);
        cache.insert(&req, Arc::clone(&held));
        for i in 0..MAX_ENTRIES + 8 {
            cache.insert(&request(&format!("f{i}"), 128), layout());
        }
        assert!(cache.get(&req).is_none(), "evicted");
        assert_eq!(held.width, 42.0, "the Arc keeps the layout alive");
    }

    #[test]
    fn layout_cache_slots_are_reused() {
        let mut cache = LayoutCache::default();
        for i in 0..MAX_ENTRIES {
            cache.insert(&request(&format!("s{i}"), 128), layout());
        }
        let high_water = cache.slots.len();
        cache.insert(&request("x", 128), layout());
        assert_eq!(cache.slots.len(), high_water);
        assert_eq!(cache.free.len(), 0);
        for i in 0..64 {
            cache.insert(&request(&format!("t{i}"), 128), layout());
        }
        assert_eq!(cache.slots.len(), high_water);
        assert_eq!(cache.entries, MAX_ENTRIES);
    }

    #[test]
    fn layout_cache_duplicate_insert_replaces() {
        let mut cache = LayoutCache::default();
        let req = request("dup", 128);
        cache.insert(&req, layout());
        cache.insert(&req, layout());
        assert_eq!(cache.entries, 1, "identical request is not duplicated");
        assert_eq!(cache.stats().hits, 0);
        assert!(cache.get(&req).is_some());
    }
}
