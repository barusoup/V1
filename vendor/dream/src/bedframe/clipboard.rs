//! Cross-platform plain-text clipboard access for BedFrame apps.
//!
//! - **macOS:** reads and writes `NSPasteboard` (`public.utf8-plain-text`).
//! - **Windows:** reads and writes the Win32 clipboard (`CF_UNICODETEXT`,
//!   UTF-16LE), falling back to an in-process buffer if the clipboard cannot
//!   be opened (for example in a headless service session).
//! - **Other platforms:** in-process UTF-8 buffer.

use std::sync::Mutex;

static FALLBACK: Mutex<String> = Mutex::new(String::new());

fn fallback_read() -> Option<String> {
    let text = FALLBACK.lock().unwrap();
    (!text.is_empty()).then(|| text.clone())
}

fn fallback_write(text: &str) {
    *FALLBACK.lock().unwrap() = text.to_string();
}

#[cfg(target_os = "macos")]
#[allow(unexpected_cfgs)]
mod imp {
    use objc::runtime::Object;
    use objc::{class, msg_send, sel, sel_impl};

    struct NsString(*mut Object);

    impl NsString {
        fn new(text: &str) -> Self {
            unsafe {
                let cls = class!(NSString);
                let obj: *mut Object = msg_send![cls, alloc];
                let obj: *mut Object = msg_send![obj,
                    initWithBytes: text.as_ptr() as *const std::ffi::c_void
                    length: text.len()
                    encoding: 4usize
                ];
                Self(obj)
            }
        }

        fn raw(&self) -> *mut Object {
            self.0
        }
    }

    impl Drop for NsString {
        fn drop(&mut self) {
            unsafe {
                if !self.0.is_null() {
                    let _: () = msg_send![self.0, release];
                }
            }
        }
    }

    struct Pool(*mut Object);

    impl Pool {
        fn new() -> Self {
            unsafe { Self(msg_send![class!(NSAutoreleasePool), new]) }
        }
    }

    impl Drop for Pool {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.0, drain];
            }
        }
    }

    pub fn read() -> Option<String> {
        let _pool = Pool::new();
        unsafe {
            let board: *mut Object = msg_send![class!(NSPasteboard), generalPasteboard];
            if board.is_null() {
                return super::fallback_read();
            }
            let text: *mut Object =
                msg_send![board, stringForType: NsString::new("public.utf8-plain-text").raw()];
            if text.is_null() {
                return super::fallback_read();
            }
            let ptr: *const i8 = msg_send![text, UTF8String];
            if ptr.is_null() {
                return super::fallback_read();
            }
            Some(std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }

    pub fn write(text: &str) {
        super::fallback_write(text);
        let _pool = Pool::new();
        unsafe {
            let board: *mut Object = msg_send![class!(NSPasteboard), generalPasteboard];
            if board.is_null() {
                return;
            }
            let _cleared: isize = msg_send![board, clearContents];
            let str_obj = NsString::new(text);
            let type_obj = NsString::new("public.utf8-plain-text");
            let _ok: bool = msg_send![board, setString: str_obj.raw() forType: type_obj.raw()];
        }
    }
}

#[cfg(windows)]
mod imp {
    use core::ffi::c_void;

    const CF_UNICODETEXT: u32 = 13;
    const GMEM_MOVEABLE: u32 = 0x0002;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn OpenClipboard(hwnd: *mut c_void) -> i32;
        fn CloseClipboard() -> i32;
        fn EmptyClipboard() -> i32;
        fn GetClipboardData(format: u32) -> *mut c_void;
        fn SetClipboardData(format: u32, mem: *mut c_void) -> *mut c_void;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalAlloc(flags: u32, bytes: usize) -> *mut c_void;
        fn GlobalLock(mem: *mut c_void) -> *mut c_void;
        fn GlobalUnlock(mem: *mut c_void) -> i32;
        fn GlobalSize(mem: *mut c_void) -> usize;
        fn GlobalFree(mem: *mut c_void) -> *mut c_void;
    }

    pub fn read() -> Option<String> {
        unsafe {
            if OpenClipboard(core::ptr::null_mut()) == 0 {
                return super::fallback_read();
            }
            let handle = GetClipboardData(CF_UNICODETEXT);
            if handle.is_null() {
                CloseClipboard();
                return super::fallback_read();
            }
            let ptr = GlobalLock(handle) as *const u16;
            if ptr.is_null() {
                CloseClipboard();
                return super::fallback_read();
            }
            let max_u16 = GlobalSize(handle) / 2;
            let mut len = 0usize;
            while len < max_u16 && *ptr.add(len) != 0 {
                len += 1;
            }
            let slice = core::slice::from_raw_parts(ptr, len);
            let mut text = String::from_utf16_lossy(slice);
            if text.contains('\r') {
                text = text.replace("\r\n", "\n");
            }
            GlobalUnlock(handle);
            CloseClipboard();
            (!text.is_empty()).then_some(text)
        }
    }

    pub fn write(text: &str) {
        super::fallback_write(text);
        let mut utf16: Vec<u16> = text.encode_utf16().collect();
        utf16.push(0);
        let bytes = utf16.len() * 2;
        unsafe {
            if OpenClipboard(core::ptr::null_mut()) == 0 {
                return;
            }
            EmptyClipboard();
            let handle = GlobalAlloc(GMEM_MOVEABLE, bytes);
            if handle.is_null() {
                CloseClipboard();
                return;
            }
            let dst = GlobalLock(handle) as *mut u16;
            if dst.is_null() {
                GlobalFree(handle);
                CloseClipboard();
                return;
            }
            core::ptr::copy_nonoverlapping(utf16.as_ptr(), dst, utf16.len());
            GlobalUnlock(handle);
            if SetClipboardData(CF_UNICODETEXT, handle).is_null() {
                GlobalFree(handle);
            }
            CloseClipboard();
        }
    }
}

#[cfg(not(any(target_os = "macos", windows)))]
mod imp {
    pub fn read() -> Option<String> {
        super::fallback_read()
    }

    pub fn write(text: &str) {
        super::fallback_write(text);
    }
}

/// Read the current plain-text clipboard contents, if any.
pub fn get_text() -> Option<String> {
    imp::read().filter(|s| !s.is_empty())
}

/// Replace the plain-text clipboard contents with `text`.
pub fn set_text(text: &str) {
    imp::write(text);
}
