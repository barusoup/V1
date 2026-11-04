//! CoreFoundation/CoreGraphics/CoreText FFI — raw extern decls plus the
//! small RAII wrappers. All `Create`/`Copy` handles are released through
//! [`CFRef`]; `Get` references are borrowed and never released.

#![allow(non_snake_case)]

use std::ffi::c_void;

pub(crate) type CFTypeRef = *const c_void;
pub(crate) type CFDataRef = *const c_void;
pub(crate) type CFStringRef = *const c_void;
pub(crate) type CFAttributedStringRef = *const c_void;
pub(crate) type CFArrayRef = *const c_void;
pub(crate) type CFDictionaryRef = *const c_void;
pub(crate) type CTFontRef = *const c_void;
pub(crate) type CTLineRef = *const c_void;
pub(crate) type CTTypesetterRef = *const c_void;
pub(crate) type CTRunRef = *const c_void;
pub(crate) type CGDataProviderRef = *const c_void;
pub(crate) type CGFontRef = *const c_void;
pub(crate) type CGContextRef = *const c_void;
pub(crate) type CGColorSpaceRef = *const c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CFRange {
    pub location: isize,
    pub length: isize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct CGPoint {
    pub x: f64,
    pub y: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct CGSize {
    pub width: f64,
    pub height: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct CGRect {
    pub origin: CGPoint,
    pub size: CGSize,
}

pub(crate) const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
pub(crate) const K_CT_FONT_ORIENTATION_HORIZONTAL: i32 = 1;
pub(crate) const K_CT_FONT_TRAIT_COLOR_GLYPHS: u32 = 1 << 13;
pub(crate) const K_CG_IMAGE_ALPHA_PREMULTIPLIED_LAST: u32 = 1;
pub(crate) const K_CG_BITMAP_BYTE_ORDER_32_BIG: u32 = 4 << 12;

// Three separate `#[link]` items are required — one per framework —
// clippy's duplicated_attributes counts the kind fragment as a dup.
#[allow(clippy::duplicated_attributes)]
#[link(name = "CoreFoundation", kind = "framework")]
#[link(name = "CoreGraphics", kind = "framework")]
#[link(name = "CoreText", kind = "framework")]
unsafe extern "C" {
    pub(crate) fn CFRelease(obj: CFTypeRef);
    /// CFNumberCreate(allocator, type, valuePtr) — returns a retained ref.
    pub(crate) fn CFNumberCreate(
        allocator: *const c_void,
        the_type: isize,
        value_ptr: *const c_void,
    ) -> CFTypeRef;
    pub(crate) fn CFDataCreate(
        allocator: *const c_void,
        bytes: *const u8,
        length: isize,
    ) -> CFDataRef;
    pub(crate) fn CFStringCreateWithCString(
        allocator: *const c_void,
        cstr: *const i8,
        encoding: u32,
    ) -> CFStringRef;
    pub(crate) fn CFStringCreateWithBytes(
        allocator: *const c_void,
        bytes: *const u8,
        len: isize,
        encoding: u32,
        external: u8,
    ) -> CFStringRef;
    pub(crate) fn CFStringGetLength(s: CFStringRef) -> isize;
    pub(crate) fn CFStringGetCString(
        s: CFStringRef,
        buffer: *mut u8,
        buffer_size: isize,
        encoding: u32,
    ) -> u8;
    pub(crate) fn CFAttributedStringCreate(
        allocator: *const c_void,
        string: CFStringRef,
        attributes: CFDictionaryRef,
    ) -> CFAttributedStringRef;
    pub(crate) fn CFDictionaryCreate(
        allocator: *const c_void,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        num_values: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFDictionaryRef;
    pub(crate) fn CFDictionaryGetValue(dict: CFDictionaryRef, key: CFTypeRef) -> CFTypeRef;
    pub(crate) fn CFArrayGetCount(array: CFArrayRef) -> isize;
    pub(crate) fn CFArrayGetValueAtIndex(array: CFArrayRef, index: isize) -> CFTypeRef;

    // CoreGraphics
    pub(crate) fn CGDataProviderCreateWithCFData(data: CFDataRef) -> CGDataProviderRef;
    pub(crate) fn CGFontCreateWithDataProvider(provider: CGDataProviderRef) -> CGFontRef;
    pub(crate) fn CGBitmapContextCreate(
        data: *mut c_void,
        width: usize,
        height: usize,
        bits_per_component: usize,
        bytes_per_row: usize,
        color_space: CGColorSpaceRef,
        bitmap_info: u32,
    ) -> CGContextRef;
    pub(crate) fn CGColorSpaceCreateWithName(name: CFStringRef) -> CGColorSpaceRef;
    pub(crate) static kCGColorSpaceSRGB: CFStringRef;

    // CoreText
    pub(crate) fn CTFontCreateWithGraphicsFont(
        font: CGFontRef,
        size: f64,
        matrix: *const c_void,
        attributes: CFDictionaryRef,
    ) -> CTFontRef;
    pub(crate) fn CTFontCreateWithName(
        name: CFStringRef,
        size: f64,
        matrix: *const c_void,
    ) -> CTFontRef;
    pub(crate) fn CTFontCopyPostScriptName(font: CTFontRef) -> CFStringRef;
    pub(crate) fn CTFontGetAscent(font: CTFontRef) -> f64;
    pub(crate) fn CTFontGetDescent(font: CTFontRef) -> f64;
    pub(crate) fn CTFontGetLeading(font: CTFontRef) -> f64;
    pub(crate) fn CTFontGetSymbolicTraits(font: CTFontRef) -> u32;
    pub(crate) fn CTFontGetGlyphsForCharacters(
        font: CTFontRef,
        characters: *const u16,
        glyphs: *mut u16,
        count: isize,
    ) -> u8;
    pub(crate) fn CTFontGetAdvancesForGlyphs(
        font: CTFontRef,
        orientation: i32,
        glyphs: *const u16,
        advances: *mut CGSize,
        count: isize,
    ) -> f64;
    pub(crate) fn CTFontGetBoundingRectsForGlyphs(
        font: CTFontRef,
        orientation: i32,
        glyphs: *const u16,
        bounding_rects: *mut CGRect,
        count: isize,
    ) -> CGRect;
    pub(crate) fn CTFontDrawGlyphs(
        font: CTFontRef,
        glyphs: *const u16,
        positions: *const CGPoint,
        count: usize,
        context: CGContextRef,
    );
    pub(crate) fn CTFontCreateCopyWithAttributes(
        font: CTFontRef,
        size: f64,
        matrix: *const c_void,
        attributes: CFDictionaryRef,
    ) -> CTFontRef;
    pub(crate) fn CTLineCreateWithAttributedString(attr: CFAttributedStringRef) -> CTLineRef;
    pub(crate) fn CTLineGetGlyphRuns(line: CTLineRef) -> CFArrayRef;
    pub(crate) fn CTLineGetTypographicBounds(
        line: CTLineRef,
        ascent: *mut f64,
        descent: *mut f64,
        leading: *mut f64,
    ) -> f64;
    pub(crate) fn CTRunGetGlyphCount(run: CTRunRef) -> isize;
    pub(crate) fn CTRunGetGlyphs(run: CTRunRef, range: CFRange, buffer: *mut u16);
    pub(crate) fn CTRunGetPositions(run: CTRunRef, range: CFRange, buffer: *mut CGPoint);
    pub(crate) fn CTRunGetAttributes(run: CTRunRef) -> CFDictionaryRef;
    pub(crate) fn CTRunGetStringIndices(run: CTRunRef, range: CFRange, buffer: *mut isize);

    pub(crate) static kCTFontAttributeName: CFStringRef;
    // NOTE: kCFTypeDictionary{Key,Value}CallBacks are C struct statics,
    // not pointer variables — declaring them as `*const c_void` reads
    // the struct's first field (version 0 = NULL). They are not needed
    // (CFAttributedStringSetAttribute retains its value), so no struct
    // extern is kept.
    pub(crate) static kCTKernAttributeName: CFStringRef;

    pub(crate) fn CFAttributedStringCreateMutable(
        allocator: *const c_void,
        max_length: isize,
    ) -> CFAttributedStringRef;
    pub(crate) fn CFAttributedStringReplaceString(
        attr: CFAttributedStringRef,
        range: CFRange,
        string: CFStringRef,
    );
    pub(crate) fn CFAttributedStringSetAttribute(
        attr: CFAttributedStringRef,
        range: CFRange,
        name: CFStringRef,
        value: CFTypeRef,
    );
    pub(crate) fn CTTypesetterCreateWithAttributedString(
        string: CFAttributedStringRef,
    ) -> CTTypesetterRef;
    pub(crate) fn CTTypesetterSuggestLineBreak(
        typesetter: CTTypesetterRef,
        start: isize,
        width: f64,
    ) -> isize;
    pub(crate) fn CTTypesetterCreateLine(typesetter: CTTypesetterRef, range: CFRange) -> CTLineRef;
    pub(crate) fn CTLineGetOffsetForStringIndex(
        line: CTLineRef,
        index: isize,
        secondary: *mut f64,
    ) -> f64;
    pub(crate) fn CTLineGetTrailingWhitespaceWidth(line: CTLineRef) -> f64;
}

// --- CF helpers ------------------------------------------------------------

/// A retained CF object released on drop (`Create`/`Copy` rule).
pub(crate) struct CFRef(pub(crate) CFTypeRef);

impl CFRef {
    pub(crate) fn raw(&self) -> CFTypeRef {
        self.0
    }
}

impl Drop for CFRef {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { CFRelease(self.0) }
        }
    }
}

pub(crate) fn cf_string(text: &str) -> Option<CFRef> {
    let cstr = std::ffi::CString::new(text).ok()?;
    let s = unsafe {
        CFStringCreateWithCString(std::ptr::null(), cstr.as_ptr(), K_CF_STRING_ENCODING_UTF8)
    };
    (!s.is_null()).then(|| CFRef(s))
}

/// CFString from UTF-8 bytes (for shaping; keeps interior NULs legal).
pub(crate) fn cf_text(text: &str) -> Option<CFRef> {
    let s = unsafe {
        CFStringCreateWithBytes(
            std::ptr::null(),
            text.as_ptr(),
            text.len() as isize,
            K_CF_STRING_ENCODING_UTF8,
            0,
        )
    };
    (!s.is_null()).then(|| CFRef(s))
}

pub(crate) fn cf_to_string(s: CFStringRef) -> Option<String> {
    unsafe {
        let len = CFStringGetLength(s);
        let max = (len * 4 + 1).clamp(1, 1 << 20) as usize;
        let mut buf = vec![0u8; max];
        let ok = CFStringGetCString(s, buf.as_mut_ptr(), max as isize, K_CF_STRING_ENCODING_UTF8);
        if ok == 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(max);
        String::from_utf8(buf[..end].to_vec()).ok()
    }
}
