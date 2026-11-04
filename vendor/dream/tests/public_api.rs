use std::mem::size_of;
use std::sync::Arc;

use glam::Mat4;

use dream::app::FramePacing;
use dream::bedframe::widgets::{
    ButtonStyle, Key, KeyFeed, LabelStyle, MenuItem, MenuStyle, NavInput, PanelStyle, ScrollAlign,
    ScrollBehavior, ScrollConfig, ScrollIntoViewOptions, ScrollMetrics, ScrollToOptions,
    SegmentedStyle, SliderStyle, TextFieldStyle, Toast, ToastKind, Toasts, Ui, UiInput,
};
use dream::fsr3::JitterSequence as Fsr3Jitter;
use dream::gfx::ColorVertex;
use dream::gfx::vulkan::{DynamicBuffers, GpuBuffer, GpuContext, GpuImage, PulledMesh, Swapchain};
use dream::model::render::{ModelVertex, PbrVertex, SkinnedVertex};
use dream::model::{AlphaMode, Material, Model, ModelInstance, TexelFilter, UvTransform, Wrap};
use dream::temporal::JitterSequence as TemporalJitter;
use dream::texture::{
    AddressMode, DecodedTexture, Filter, MipFilter, SamplerDesc, TextureFormat, TextureId,
    TextureManager, TextureOptions, TextureStats, decode_rgba8, mip_levels_for,
};

#[test]
fn opaque_gpu_types_referenced() {
    let _ = size_of::<GpuContext>();
    let _ = size_of::<GpuBuffer>();
    let _ = size_of::<GpuImage>();
    let _ = size_of::<DynamicBuffers>();
    let _ = size_of::<PulledMesh>();
    let _ = size_of::<Swapchain>();
    let _ = size_of::<Ui>();
    let _ = size_of::<dream::ui::widgets::Ui>();
    let _ = size_of::<dream::bedframe::widgets::Ui>();
    let _ = size_of::<TextureManager>();
    let _ = size_of::<TextureStats>();
    let _ = size_of::<TextureId>();
    let _ = size_of::<DecodedTexture>();
    let _ = size_of::<UiInput>();
    let _ = size_of::<NavInput>();
    let _ = size_of::<LabelStyle>();
    let _ = size_of::<ButtonStyle>();
    let _ = size_of::<TextFieldStyle>();
    let _ = size_of::<PanelStyle>();
    let _ = size_of::<SliderStyle>();
    let _ = size_of::<MenuItem>();
    let _ = size_of::<MenuStyle>();
    let _ = size_of::<dream::bedframe::widgets::ScrollbarStyle>();
    let _ = size_of::<ScrollConfig>();
    let _ = size_of::<ScrollMetrics>();
    let _ = ScrollToOptions::smooth(0.0).behavior;
    let _ = ScrollIntoViewOptions::default().block;
    let _ = ScrollBehavior::Auto;
    let _ = ScrollAlign::Nearest;
    let _ = ScrollConfig::no_momentum();
    // The `ui` alias resolves the same scroll paths.
    let _ = size_of::<dream::ui::widgets::ScrollConfig>();
    let _ = dream::ui::widgets::ScrollBehavior::Smooth;
    let _ = size_of::<dream::bedframe::tokens::Tokens>();
    let _ = size_of::<dream::bedframe::tokens::Space>();
    let _ = size_of::<dream::bedframe::tokens::Radius>();
    let _ = size_of::<dream::bedframe::tokens::ControlSize>();
    let _ = size_of::<dream::bedframe::tokens::Density>();
    let _ = size_of::<dream::bedframe::tokens::TypeRole>();
    let _ = size_of::<dream::bedframe::tokens::Elevation>();
    // The `ui` alias must resolve the same paths.
    let _ = size_of::<dream::ui::tokens::Tokens>();
    let _ = size_of::<dream::ui::theme::Theme>();
    let _ = dream::ui::theme::Theme::default();
    let _ = dream::ui::widgets::ScrollbarStyle::default();
    let _ = size_of::<Toast>();
    let _ = size_of::<ToastKind>();
    let _ = size_of::<Toasts>();
    let _ = TextureFormat::Srgb;
    let _ = TextureOptions::default();
    let _ = SamplerDesc::PIXEL;
    let _ = SamplerDesc::ANISOTROPIC;
    let _ = Filter::Nearest;
    let _ = MipFilter::Nearest;
    let _ = AddressMode::Repeat;
    let _ = AlphaMode::Opaque;
    let _ = Wrap::Repeat;
    let _ = TexelFilter::Nearest;
    let _ = UvTransform::default();
    let _ = Material::default();
}

#[test]
fn jitter_sequence_matches_halton_2_3() {
    let mut seq = TemporalJitter::default();
    let first = seq.next(100, 100);
    assert!((first[0] - 0.0).abs() < 1e-6);
    assert!((first[1] - (-1.0 / 6.0)).abs() < 1e-6);
    let second = seq.next(100, 100);
    assert!((second[0] - (-0.25)).abs() < 1e-6);
    assert!((second[1] - (1.0 / 6.0)).abs() < 1e-6);
}

#[test]
fn jitter_paths_agree_for_40_frames() {
    let mut a = Fsr3Jitter::default();
    let mut b = TemporalJitter::default();
    for _ in 0..40 {
        let ja = a.next(100, 200);
        let jb = b.next(100, 200);
        assert_eq!(ja, jb);
    }
}

#[test]
fn jitter_reset_reproduces_initial_sample() {
    let mut seq = TemporalJitter::default();
    let initial = seq.next(100, 100);
    for _ in 0..5 {
        seq.next(100, 100);
    }
    seq.reset();
    assert_eq!(seq.next(100, 100), initial);
}

#[test]
fn jitter_repeats_after_eight_native_phases() {
    let mut seq = TemporalJitter::default();
    let initial = seq.next(100, 100);
    for _ in 1..8 {
        seq.next(100, 100);
    }
    assert_eq!(seq.next(100, 100), initial);
}

#[test]
fn png_roundtrip_and_mip_counts() {
    let rgba: Vec<u8> = (0..16).map(|i| i as u8 * 16).collect();
    let png = dream::png::encode(2, 2, &rgba);
    let decoded = decode_rgba8(&png).expect("decodes");
    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.pixels, rgba);
    assert_eq!(mip_levels_for(1, 1), 1);
    assert_eq!(mip_levels_for(1, 256), 9);
}

#[test]
fn model_instance_tracks_motion() {
    let model = Arc::new(Model {
        meshes: Vec::new(),
        materials: Vec::new(),
        nodes: Vec::new(),
        skins: Vec::new(),
        clips: Vec::new(),
        roots: Vec::new(),
    });
    let mut instance = ModelInstance::new(model);
    let (current, previous) = instance.motion_matrices();
    assert_eq!(current, Mat4::IDENTITY);
    assert_eq!(previous, Mat4::IDENTITY);
    instance.set_transform(Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0)));
    instance.update(0.0);
    let (current, previous) = instance.motion_matrices();
    assert_eq!(
        current,
        Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0))
    );
    assert_eq!(previous, Mat4::IDENTITY);
}

#[test]
fn renderer_ui_only_signature() {
    use dream::gfx::{Renderer, SurfaceError};
    let _f: fn(&mut Renderer, Option<&dream::ui::draw::UiFrame>) -> Result<(), SurfaceError> =
        Renderer::render_ui_only;
}

#[test]
fn public_vertex_layouts() {
    assert_eq!(size_of::<ColorVertex>(), 16);
    assert_eq!(size_of::<ModelVertex>(), 32);
    assert_eq!(size_of::<SkinnedVertex>(), 56);
    assert_eq!(size_of::<PbrVertex>(), 56);
}

#[test]
fn default_styles_and_pacing() {
    let _ = UiInput::default();
    let _ = NavInput::default();
    let _ = size_of::<Key>();
    let _ = size_of::<KeyFeed>();
    let _ = KeyFeed::default();
    let _ = Key::Tab;
    let _ = Toasts::default();
    let _ = PanelStyle::default();
    let _ = dream::ui::theme::Theme::default().segmented(dream::ui::font::FontId(0));
    let _: SegmentedStyle =
        dream::ui::theme::Theme::default().segmented(dream::ui::font::FontId(0));
    let _ = size_of::<dream::bedframe::widgets::SegmentedStyle>();
    let _ = dream::ui::widgets::headless::State::default();
    let _ = dream::bedframe::widgets::headless::State::default();
    let _ = MenuItem::new("x");
    let mut pacing = FramePacing::default();
    pacing.display_hz = Some(360.0);
    pacing.cap_hz = Some(120.0);
    assert_eq!(pacing.target_hz(), Some(120.0));
    let unknown = FramePacing::default();
    assert_eq!(unknown.target_hz(), None);
}

/// Regression: `SegmentedBehavior::update` keeps its original 7-argument
/// public signature (the pooled view-layer path uses `update_with_cells`).
#[test]
fn segmented_behavior_public_update_signature() {
    use dream::bedframe::Rect;
    use dream::bedframe::widgets::Response;
    use dream::bedframe::widgets::headless::{SegmentedBehavior, State};
    use dream::ui::atlas::AtlasConfig;
    use dream::ui::font::Fonts;

    let mut ui = Ui::new(Fonts::new(), AtlasConfig::default());
    let mut selected = 0usize;
    let mut out: Vec<(Rect, Response, State)> = Vec::new();
    ui.begin(UiInput::default(), (400.0, 300.0));
    let _picked = SegmentedBehavior::new("s", &["a", "b"]).update(
        &mut ui,
        Rect::new(0.0, 0.0, 200.0, 24.0),
        &mut selected,
        1.0,
        2.0,
        &mut out,
    );
    ui.end();
    assert_eq!(out.len(), 2);
}

#[test]
fn hooks_and_presence_public_paths() {
    use dream::bedframe::widgets::{Presence, PresencePhase, State};
    let state: dream::ui::widgets::State<u32> = State::new(3);
    let mut ui = Ui::new(Default::default(), Default::default());
    ui.begin(Default::default(), (200.0, 100.0));
    ui.with_context(&state, |ui| {
        ui.use_context::<u32>().unwrap().set(4);
    });
    assert_eq!(state.get(), 4);
    let hook = ui.use_state("value", || 7u32);
    assert_eq!(hook.get(), 7);
    let presence: Presence = ui.presence("popup", false, 0.2);
    assert_eq!(presence.phase, PresencePhase::Hidden);
    ui.end();
}
