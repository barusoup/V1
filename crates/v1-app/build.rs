// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Embeds the app icon into the Windows executable, so Explorer and the taskbar show it.

fn main() {
    println!("cargo:rerun-if-changed=assets/icon/app.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        winres::WindowsResource::new()
            .set_icon("assets/icon/app.ico")
            // Explorer, Task Manager and the media flyout read these. Without them the crate name
            // ("v1-app") is shown instead of the product name.
            .set("ProductName", "V1")
            .set("FileDescription", "V1")
            .set("CompanyName", "Barusoup")
            .set(
                "LegalCopyright",
                "Copyright (c) 2026 Hunter Lehr (Barusoup)",
            )
            .set("OriginalFilename", "v1.exe")
            .set("InternalName", "V1")
            .compile()
            .expect("embed the app icon and version information");
    }
}
