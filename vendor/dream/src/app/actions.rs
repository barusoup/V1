use super::main_pane::{DEBUG_SCENE_ID, MainPanePage, ManagedScene};
use super::{App, Command, scene};
use crate::settings::AaMode;

impl App {
    // --- Commands -------------------------------------------------------------

    pub fn undo(&mut self) {
        if let Some(Command::Camera(orbit)) = self.undo_stack.pop() {
            let current = self.scene.orbit.clone();
            self.scene.orbit = orbit;
            self.redo_stack.push(Command::Camera(current));
            self.dirty = true;
        }
    }

    pub fn redo(&mut self) {
        if let Some(Command::Camera(orbit)) = self.redo_stack.pop() {
            let current = self.scene.orbit.clone();
            self.scene.orbit = orbit;
            self.undo_stack.push(Command::Camera(current));
            self.dirty = true;
        }
    }

    // --- Actions ---------------------------------------------------------------

    pub fn capture_png(&mut self) {
        self.captures += 1;
        let dir = std::path::Path::new("target/dream-app");
        if std::fs::create_dir_all(dir).is_ok() {
            let path = dir.join(format!("capture-{}.png", self.captures));
            // A failed capture has no notification surface any more, so it
            // goes to stderr rather than being swallowed.
            if let Err(err) = self.renderer.save_png(&path) {
                eprintln!("[capture] failed to write {}: {err}", path.display());
            }
        }
        self.dirty = true;
    }

    /// Boot directly into the debug scene, configure spin and switch to viewport.
    pub fn load_debug_scene(&mut self) {
        self.scene.load_debug();
        self.settings.spin = 0.4;
        self.scene_manager.active_scene_id = Some(DEBUG_SCENE_ID);
        self.scene_manager.touch_scene(DEBUG_SCENE_ID);
        self.main_pane_page = MainPanePage::Viewport;
        self.dirty = true;
    }

    /// Remember the current viewport orbit on the active scene, so its card
    /// preview reopens from the viewpoint left behind.
    pub fn note_viewport_exit(&mut self) {
        if let Some(active) = self.scene_manager.active_scene_id
            && let Some(scene) = self
                .scene_manager
                .scenes
                .iter_mut()
                .find(|s| s.id == active)
        {
            scene.last_view = Some((self.scene.orbit.yaw, self.scene.orbit.pitch));
        }
        self.dirty = true;
    }

    /// Show `page` in the main pane. Leaving the viewport keeps its exit
    /// viewpoint for the card preview: the next frame's scene pass still
    /// renders the exit view (nothing mutates orbit on the way out), so its
    /// capture is a real 3D shot of where the user left — whether they left
    /// for the Scenes page or the Settings page.
    ///
    /// Opening Settings remembers the tab page underneath it: Settings is a
    /// dedicated page, and [`Self::close_settings`] returns there.
    pub fn show_page(&mut self, page: MainPanePage) {
        if self.main_pane_page == MainPanePage::Viewport && page != MainPanePage::Viewport {
            self.note_viewport_exit();
            if self.pending_thumb_capture.is_none() {
                self.pending_thumb_capture = self.scene_manager.active_scene_id;
            }
        }
        if page == MainPanePage::Settings && self.main_pane_page != MainPanePage::Settings {
            self.settings_return = self.main_pane_page;
        }
        self.main_pane_page = page;
        self.dirty = true;
    }

    /// Leave the Settings page for the tab page it was opened from. If that
    /// was a viewport that has since lost its binding, fall back to the
    /// Scenes page (the same fallback the main pane draws anyway).
    pub fn close_settings(&mut self) {
        let page = match self.settings_return {
            MainPanePage::Settings => MainPanePage::SceneManager,
            page => page,
        };
        let bound = self.scene.debug_loaded() || self.scene_manager.active_scene_id.is_some();
        self.show_page(if page == MainPanePage::Viewport && !bound {
            MainPanePage::SceneManager
        } else {
            page
        });
    }

    /// Leave the viewport for the Scenes page.
    pub fn show_scene_manager(&mut self) {
        self.show_page(MainPanePage::SceneManager);
    }

    /// Queue the pending exit view (if any) for the scene pass about to
    /// render. The renderer downsamples it inside that same frame, so the
    /// leave itself never waits on the GPU; the pixels land in
    /// [`Self::capture_pending_thumbnail`] a couple of frames later. Runs
    /// before `render_ui`, while the scene target still holds the frame's 3D
    /// pass. Warmup shoots stay synchronous: their state is restored right
    /// after the capture.
    pub fn queue_pending_thumbnail(&mut self) {
        let Some(id) = self.pending_thumb_capture else {
            return;
        };
        if self.thumb_warmup.is_some() || self.renderer.scene_thumbnail_busy() {
            return;
        }
        self.pending_thumb_capture = None;
        let viewport = self.physical_rect(self.layout.viewport);
        if self
            .renderer
            .request_scene_thumbnail(viewport, super::main_pane::THUMB_SIZE)
        {
            self.thumb_job = Some(id);
        } else {
            self.thumb_failed.insert(id);
        }
    }

    /// Store finished captures as card previews, and complete a hidden warmup
    /// shoot (capturing it now, then restoring the state it hid). Runs after
    /// `render_ui`.
    pub fn capture_pending_thumbnail(&mut self) {
        // Retired previews cool down; a few frames cover every in-flight UI.
        for (_, frames) in &mut self.retired_previews {
            *frames = frames.saturating_sub(1);
        }
        self.retired_previews.retain(|(_, frames)| *frames > 0);

        if let Some(warmup) = self.thumb_warmup.take() {
            self.pending_thumb_capture = None;
            let viewport = self.physical_rect(self.layout.viewport);
            match self
                .renderer
                .capture_scene_thumbnail(viewport, super::main_pane::THUMB_SIZE)
            {
                Some((pixels, w, _)) if w == super::main_pane::THUMB_SIZE => {
                    self.store_preview(warmup.scene_id, pixels);
                }
                _ => {
                    self.thumb_failed.insert(warmup.scene_id);
                }
            }
            self.scene.orbit = warmup.prev_orbit;
            if warmup.debug_was_loaded && !self.scene.debug_loaded() {
                self.scene.load_debug();
            } else if !warmup.debug_was_loaded {
                self.scene.unload_debug();
            }
            self.renderer.reset_history();
        } else if let Some(pixels) = self.renderer.take_scene_thumbnail()
            && let Some(id) = self.thumb_job.take()
        {
            self.store_preview(id, pixels);
        }
        // More scenes may still need their first shot; keep frames coming.
        if self.main_pane_page == MainPanePage::SceneManager
            && self.scene_manager.scenes.iter().any(|s| {
                !self.preview_pixels.contains_key(&s.id) && !self.thumb_failed.contains(&s.id)
            })
        {
            self.dirty = true;
        }
    }

    /// Keep a finished capture as the scene's card preview, unless the scene
    /// was deleted while its capture was in flight.
    fn store_preview(&mut self, id: u64, pixels: Vec<u8>) {
        if !self.scene_manager.scenes.iter().any(|s| s.id == id) {
            return;
        }
        self.preview_pixels
            .insert(id, super::main_pane::PreviewImage { pixels });
        self.upload_thumbnail(id);
    }

    /// Upload (or refresh) one scene's card preview texture.
    fn upload_thumbnail(&mut self, id: u64) {
        let Some(preview) = self.preview_pixels.get(&id) else {
            return;
        };
        let size = super::main_pane::THUMB_SIZE;
        debug_assert_eq!(preview.pixels.len(), size as usize * size as usize * 4);
        // Cards show only the top band (rows are top-first), so upload just
        // those rows: the card's rounded UVs then span the texture exactly.
        let rows = super::main_pane::CARD_H as u32;
        let texture = std::sync::Arc::new(crate::texture::GpuTexture::from_rgba8(
            self.renderer.context(),
            &format!("scene preview {id}"),
            &preview.pixels[..size as usize * rows as usize * 4],
            size,
            rows,
            crate::texture::TextureOptions {
                format: crate::texture::TextureFormat::Srgb,
                sampler: crate::texture::SamplerDesc::default(),
                mip_levels: Some(1),
            },
        ));
        if let Some(old) = self.preview_textures.insert(id, texture) {
            // The UI frames still in flight may read it; drop it later.
            self.retired_previews.push((old, 4));
        }
    }

    /// Hidden one-frame shoot for the first scene still missing a preview:
    /// stage its basic top-down 3D shot so the next scene pass captures it,
    /// then [`Self::capture_pending_thumbnail`] restores everything. Only
    /// runs while staying on the Scenes page with no capture already queued.
    pub fn maybe_arm_thumb_warmup(&mut self) {
        if self.main_pane_page != MainPanePage::SceneManager
            || self.pending_thumb_capture.is_some()
            || self.thumb_job.is_some()
            || self.thumb_warmup.is_some()
        {
            return;
        }
        let next = self.scene_manager.scenes.iter().find(|s| {
            !self.preview_pixels.contains_key(&s.id) && !self.thumb_failed.contains(&s.id)
        });
        let Some(scene) = next else { return };
        let (id, is_debug) = (scene.id, scene.is_debug);
        let debug_was_loaded = self.scene.debug_loaded();
        let prev_orbit = self.scene.orbit.clone();
        if is_debug {
            self.scene.load_debug();
            // Basic shot from above: keep the yaw, look straight down.
            if let Some((yaw, _)) = scene.last_view {
                self.scene.orbit.yaw = yaw;
            }
            self.scene.orbit.pitch = -1.35;
        } else if debug_was_loaded {
            // Empty scenes shoot the bare canvas; park the debug scene.
            self.scene.unload_debug();
        }
        self.scene.orbit.apply(&mut self.camera);
        self.thumb_warmup = Some(super::main_pane::ThumbWarmup {
            scene_id: id,
            prev_orbit,
            debug_was_loaded,
        });
        self.pending_thumb_capture = Some(id);
        self.dirty = true;
    }

    /// Load or unload the debug scene. The app boots with the scene manager;
    /// the debug scene is booted on demand.
    pub fn toggle_debug(&mut self) {
        if self.scene.debug_loaded() {
            self.note_viewport_exit();
            self.scene.unload_debug();
            self.settings.spin = 0.0;
            if self.scene_manager.active_scene_id == Some(DEBUG_SCENE_ID) {
                self.scene_manager.active_scene_id = None;
            }
            if self.scene_manager.active_scene_id.is_none()
                && self.main_pane_page == MainPanePage::Viewport
            {
                self.main_pane_page = MainPanePage::SceneManager;
            }
        } else {
            self.load_debug_scene();
        }
        self.dirty = true;
    }

    /// Compatibility wrapper for `toggle_debug`.
    pub fn toggle_demo(&mut self) {
        self.toggle_debug();
    }

    /// Create a new managed scene. With no pending name (the Scenes row's
    /// Create card), the scene takes the next unused `scene<number>`
    /// placeholder: one past the highest number already taken.
    pub fn create_new_scene(&mut self) {
        let name = if self.scene_manager.new_scene_name.trim().is_empty() {
            self.scene_manager.next_placeholder_name()
        } else {
            self.scene_manager.new_scene_name.trim().to_string()
        };
        let id = self.scene_manager.next_scene_id;
        self.scene_manager.next_scene_id += 1;
        self.scene_manager.scenes.push(ManagedScene {
            id,
            name,
            description: "Custom user scene".to_string(),
            object_count: 0,
            is_debug: false,
            last_view: None,
        });
        self.scene_manager.new_scene_name.clear();
        self.scene_manager.active_scene_id = Some(id);
        self.scene_manager.touch_scene(id);
        self.scene.unload_debug();
        self.settings.spin = 0.0;
        self.main_pane_page = MainPanePage::Viewport;
        self.dirty = true;
    }

    /// Open an existing managed scene by ID.
    pub fn open_managed_scene(&mut self, id: u64) {
        if let Some(scene) = self
            .scene_manager
            .scenes
            .iter()
            .find(|s| s.id == id)
            .cloned()
        {
            if scene.is_debug {
                self.load_debug_scene();
            } else {
                self.scene.unload_debug();
                self.settings.spin = 0.0;
                self.scene_manager.active_scene_id = Some(id);
                self.scene_manager.touch_scene(id);
                self.main_pane_page = MainPanePage::Viewport;
                self.dirty = true;
            }
        }
    }

    /// Duplicate a managed scene.
    pub fn duplicate_managed_scene(&mut self, id: u64) {
        if let Some(scene) = self
            .scene_manager
            .scenes
            .iter()
            .find(|s| s.id == id)
            .cloned()
        {
            let new_id = self.scene_manager.next_scene_id;
            self.scene_manager.next_scene_id += 1;
            let dup_name = format!("{} (Copy)", scene.name);
            self.scene_manager.scenes.push(ManagedScene {
                id: new_id,
                name: dup_name,
                description: format!("Copy of {}", scene.name),
                object_count: scene.object_count,
                is_debug: false,
                last_view: scene.last_view,
            });
            self.scene_manager.touch_scene(new_id);
            self.dirty = true;
        }
    }

    /// Delete a managed scene.
    pub fn delete_managed_scene(&mut self, id: u64) {
        if let Some(pos) = self
            .scene_manager
            .scenes
            .iter()
            .position(|s| s.id == id && !s.is_debug)
        {
            self.scene_manager.scenes.remove(pos);
            self.preview_pixels.remove(&id);
            if let Some(old) = self.preview_textures.remove(&id) {
                self.retired_previews.push((old, 4));
            }
            if self.scene_manager.active_scene_id == Some(id) {
                self.scene_manager.active_scene_id = None;
                if self.main_pane_page == MainPanePage::Viewport {
                    self.main_pane_page = MainPanePage::SceneManager;
                }
            }
            self.dirty = true;
        }
    }

    /// Return the orbit camera to its default pose, as one undo step.
    pub fn reset_camera(&mut self) {
        let before = self.scene.orbit.clone();
        if before != scene::OrbitCamera::default() {
            self.undo_stack.push(Command::Camera(before));
            self.redo_stack.clear();
        }
        self.scene.orbit = scene::OrbitCamera::default();
        self.dirty = true;
    }

    /// Toggle FSR on/off at runtime (the engine supports this live).
    pub fn toggle_antialias(&mut self) {
        self.settings.antialias = !self.settings.antialias;
        self.renderer.set_temporal(self.settings.antialias);
        self.engine_settings.aa = if self.settings.antialias {
            AaMode::Fsr
        } else {
            AaMode::None
        };
        self.dirty = true;
    }
}
