use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const FX_K: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add_word(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(FX_K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let chunk: [u8; 8] = bytes[..8].try_into().unwrap();
            self.add_word(u64::from_ne_bytes(chunk));
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let chunk: [u8; 4] = bytes[..4].try_into().unwrap();
            self.add_word(u64::from(u32::from_ne_bytes(chunk)));
            bytes = &bytes[4..];
        }
        if bytes.len() >= 2 {
            let chunk: [u8; 2] = bytes[..2].try_into().unwrap();
            self.add_word(u64::from(u16::from_ne_bytes(chunk)));
            bytes = &bytes[2..];
        }
        if let Some(&b) = bytes.first() {
            self.add_word(u64::from(b));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_word(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_word(i as u64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.add_word(u64::from(i as u32));
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.add_word(i as u64);
    }
}

pub(crate) type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub(crate) type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// Fast mixer for already-hashed `u64` keys (FNV-1a widget IDs and layout
/// request hashes). Multiplying by the 64-bit Fibonacci constant spreads
/// entropy across both low probe-index bits and top H2 metadata bits.
#[derive(Default)]
pub(crate) struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            let v = u64::from_ne_bytes(bytes.try_into().unwrap());
            self.write_u64(v);
        } else {
            let mut h = FxHasher::default();
            h.write(bytes);
            self.0 = h.finish();
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i.wrapping_mul(FX_K);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(u64::from(i));
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

pub(crate) type IdBuildHasher = BuildHasherDefault<IdHasher>;
pub(crate) type IdMap<V> = HashMap<u64, V, IdBuildHasher>;
pub(crate) type IdSet = HashSet<u64, IdBuildHasher>;
