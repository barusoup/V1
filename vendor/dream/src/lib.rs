pub mod app;
pub mod audio;
pub mod bedframe;
pub mod camera;
pub mod fsr3;
pub mod gfx;
pub mod impostor;
pub mod lighting;
pub mod model;
pub mod png;
pub mod settings;
pub mod temporal;
pub mod texture;
pub mod threads;

pub use bedframe as ui;

/// The stock scene shader. [`gfx::Renderer`] requires a scene pipeline even
/// for an app that only draws UI; hosts that never shade a scene can pass
/// this instead of carrying a copy of the engine's WGSL.
pub const DEFAULT_SCENE_WGSL: &str = include_str!("app/scene.wgsl");

extern crate self as dream;
