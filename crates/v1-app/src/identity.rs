// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Windows app identity.
//!
//! The taskbar and the media flyout identify a process by its AppUserModelID and resolve the
//! display name from the Start Menu shortcut that carries the same ID. An explicit ID without a
//! matching shortcut shows up as "Unknown app", so an installed copy repairs its own shortcut
//! before the window exists. Builds run from `target/` are left alone: installing a shortcut for a
//! development binary would put an entry in the Start Menu for something the user never installed.

#[cfg(windows)]
pub const APP_ID: &str = "Barusoup.V1";

#[cfg(windows)]
pub fn apply() {
    use windows::Win32::System::Com::{CoInitializeEx, COINIT_APARTMENTTHREADED};

    // The shortcut work below is COM; winit would initialize the apartment later, but the
    // identity has to be in place before the first window or media session is created.
    // SAFETY: CoInitializeEx takes no borrowed pointers; an already-initialized apartment is
    // reported as an error and is harmless.
    let _ = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
    set_process_id();
    if crate::update::installed_dir().is_some() {
        let _ = repair_shortcut();
    }
}

#[cfg(windows)]
fn set_process_id() {
    #[link(name = "shell32")]
    extern "system" {
        fn SetCurrentProcessExplicitAppUserModelID(app_id: *const u16) -> i32;
    }

    let id: Vec<u16> = APP_ID.encode_utf16().chain(Some(0)).collect();
    // SAFETY: `id` is a NUL-terminated UTF-16 string that outlives the call.
    unsafe {
        SetCurrentProcessExplicitAppUserModelID(id.as_ptr());
    }
}

#[cfg(windows)]
fn repair_shortcut() -> windows::core::Result<()> {
    use windows::core::{Interface, GUID, HSTRING, PROPVARIANT};
    use windows::Win32::Foundation::BOOL;
    use windows::Win32::System::Com::{
        CoCreateInstance, IPersistFile, CLSCTX_INPROC_SERVER, STGM_READWRITE,
    };
    use windows::Win32::UI::Shell::IShellLinkW;
    use windows::Win32::UI::Shell::PropertiesSystem::{IPropertyStore, PROPERTYKEY};

    /// `PKEY_AppUserModel_ID`: {9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}, 5.
    const APP_USER_MODEL_ID: PROPERTYKEY = PROPERTYKEY {
        fmtid: GUID::from_u128(0x9f4c_2855_9f79_4b39_a8d0_e1d4_2de1_d5f3),
        pid: 5,
    };
    /// `CLSID_ShellLink`.
    const SHELL_LINK: GUID = GUID::from_u128(0x0002_1401_0000_0000_c000_0000_0000_0046);

    let exe = std::env::current_exe()?;
    let Some(link_path) = start_menu_shortcut() else {
        return Ok(());
    };
    if let Some(parent) = link_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let path = hstring(&link_path);

    // SAFETY: every call below uses a live COM object created here; strings are NUL-terminated
    // HSTRINGs that outlive their calls.
    unsafe {
        let link: IShellLinkW = CoCreateInstance(&SHELL_LINK, None, CLSCTX_INPROC_SERVER)?;
        let persist: IPersistFile = link.cast()?;
        if link_path.is_file() {
            persist.Load(&path, STGM_READWRITE)?;
        } else {
            let target = hstring(&exe);
            link.SetPath(&target)?;
            link.SetDescription(&HSTRING::from("V1"))?;
            link.SetIconLocation(&target, 0)?;
            if let Some(dir) = exe.parent() {
                link.SetWorkingDirectory(&hstring(dir))?;
            }
        }

        let store: IPropertyStore = link.cast()?;
        let present = store
            .GetValue(&APP_USER_MODEL_ID)
            .ok()
            .is_some_and(|value| value.to_string() == APP_ID);
        if !present {
            let value = PROPVARIANT::from(APP_ID);
            store.SetValue(&APP_USER_MODEL_ID, &value)?;
            store.Commit()?;
            persist.Save(&path, BOOL(1))?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn start_menu_shortcut() -> Option<std::path::PathBuf> {
    let appdata = std::env::var_os("APPDATA")?;
    Some(std::path::PathBuf::from(appdata).join(r"Microsoft\Windows\Start Menu\Programs\V1\V1.lnk"))
}

#[cfg(windows)]
fn hstring(path: &std::path::Path) -> windows::core::HSTRING {
    windows::core::HSTRING::from(path.as_os_str().to_string_lossy().as_ref())
}
