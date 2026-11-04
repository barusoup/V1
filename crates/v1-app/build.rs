// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Embeds the app icon into the Windows executable, so Explorer and the taskbar show it.

fn main() {
    println!("cargo:rerun-if-changed=assets/icon/app.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        winres::WindowsResource::new()
            .set_icon("assets/icon/app.ico")
            .compile()
            .expect("embed the app icon");
    }
}
