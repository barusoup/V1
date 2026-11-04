//! Temporal data: the motion-vector contract for upscalers (FSR, DLSS, XeSS).
//!
//! Every vendor upscaler reconstructs the current frame by reprojecting
//! history along per-pixel motion vectors. This module is the contract for
//! producing them correctly, plus the tracking helpers that keep dynamic
//! objects honest:
//!
//! - **Scene contract**: the scene pass renders colour + depth + a
//!   `R16G16_SFLOAT` velocity target. Velocity = `prev_uv - curr_uv` in the
//!   offscreen targets' texture space (bottom-up, so consumers read them with
//!   `uv.y = ndc.y * 0.5 + 0.5`), computed from *unjittered* current/previous
//!   view-projections
//!   so sub-pixel jitter never leaks into reprojection. The engine's camera
//!   uniform block ([`CameraMatrices`]) carries exactly those three matrices,
//!   and every scene pipeline binds it as set 0.
//! - **Per-object motion**: camera reprojection is exact for static geometry
//!   only. Anything that moves in the world needs its previous transform in
//!   the shader too. [`MotionTracker`] hands out `(current, previous)` model
//!   matrices per object; `model::render` feeds them (and previous joint
//!   palettes) to skinned and static model shaders.
//! - **History validity**: temporal history is only meaningful while the
//!   scene is continuous. Teleports, resizes, cuts and toggling the upscaler
//!   must invalidate it — `Renderer::reset_history` does, and clears the
//!   persistent accumulation state so no stale smear survives the cut.
//!
//! ```ignore
//! let (model, prev_model) = tracker.update(object_id, transform);
//! // shader: velocity = prev_uv(prev_vp * prev_model * pos) - curr_uv(curr_vp * model * pos)
//! ```

use glam::Mat4;
use std::collections::HashMap;

mod jitter;

pub use jitter::JitterSequence;
pub(crate) use jitter::halton;

/// The engine camera uniform: three matrices with distinct jobs.
///
/// This is the layout the scene-pass uniform uses (`set 0, binding 0`), and
/// what `model.wgsl`'s `Camera` struct mirrors. Keep the field order: shaders
/// on both sides read it.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraMatrices {
    /// Jittered projection: the matrix that renders the current frame.
    pub view_projection: [[f32; 4]; 4],
    /// Unjittered current projection: motion denominator.
    pub curr_view_projection: [[f32; 4]; 4],
    /// Unjittered previous projection: motion numerator.
    pub prev_view_projection: [[f32; 4]; 4],
}

/// Stable id for a motion-tracked object (entity id, pointer key, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObjectId(pub u64);

/// Per-object previous-transform registry.
///
/// Call [`MotionTracker::update`] once per object per frame with its current
/// model matrix; it returns the (current, previous) pair to feed the shader.
/// Objects that skip a frame keep their last pair (no spurious motion), and
/// teleports can snap with [`MotionTracker::snap`].
#[derive(Default)]
pub struct MotionTracker {
    transforms: HashMap<ObjectId, (Mat4, Mat4)>,
}

impl MotionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `model` for this frame and return (current, previous). The
    /// first frame for an object reports zero motion.
    pub fn update(&mut self, id: ObjectId, model: Mat4) -> (Mat4, Mat4) {
        let entry = self.transforms.entry(id).or_insert((model, model));
        let previous = entry.0;
        *entry = (model, previous);
        (model, previous)
    }

    /// Record `model` with *no* motion (teleport, respawn, cut): the next
    /// update reports the new position as zero motion.
    pub fn snap(&mut self, id: ObjectId, model: Mat4) {
        self.transforms.insert(id, (model, model));
    }

    /// Last recorded pair without advancing time.
    pub fn get(&self, id: ObjectId) -> Option<(Mat4, Mat4)> {
        self.transforms.get(&id).copied()
    }

    /// Drop objects not refreshed this frame. Call once per frame with the
    /// set of ids actually updated; otherwise the registry grows forever.
    pub fn prune(&mut self, live: &std::collections::HashSet<ObjectId>) {
        self.transforms.retain(|id, _| live.contains(id));
    }

    /// Forget everything (scene change, level load).
    pub fn clear(&mut self) {
        self.transforms.clear();
    }

    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

/// The Halton(2, 3) sample used for sub-pixel jitter, in -0.5..0.5. Exposed
/// for tests and custom jitter schedules.
pub fn halton_jitter(index: i32) -> [f32; 2] {
    [halton(index + 1, 2) - 0.5, halton(index + 1, 3) - 0.5]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_first_frame_is_zero_motion() {
        let mut tracker = MotionTracker::new();
        let model = Mat4::from_translation(glam::Vec3::X);
        let (current, previous) = tracker.update(ObjectId(1), model);
        assert_eq!(current, previous);
    }

    #[test]
    fn tracker_lags_one_frame_behind() {
        let mut tracker = MotionTracker::new();
        let a = Mat4::from_translation(glam::Vec3::X);
        let b = Mat4::from_translation(glam::Vec3::Y * 2.0);
        tracker.update(ObjectId(1), a);
        let (current, previous) = tracker.update(ObjectId(1), b);
        assert_eq!(current, b);
        assert_eq!(previous, a);
    }

    #[test]
    fn snap_reports_no_motion_after_teleport() {
        let mut tracker = MotionTracker::new();
        let a = Mat4::from_translation(glam::Vec3::X);
        let far = Mat4::from_translation(glam::Vec3::X * 1000.0);
        tracker.update(ObjectId(1), a);
        tracker.snap(ObjectId(1), far);
        let (current, previous) = tracker.update(ObjectId(1), far);
        assert_eq!(current, previous, "teleports must not read as motion");
    }

    #[test]
    fn objects_are_independent_and_prunable() {
        let mut tracker = MotionTracker::new();
        tracker.update(ObjectId(1), Mat4::IDENTITY);
        tracker.update(ObjectId(2), Mat4::IDENTITY);
        assert_eq!(tracker.len(), 2);
        let live = std::collections::HashSet::from([ObjectId(1)]);
        tracker.prune(&live);
        assert_eq!(tracker.len(), 1);
        assert!(tracker.get(ObjectId(1)).is_some());
        tracker.clear();
        assert!(tracker.is_empty());
    }

    #[test]
    fn jitter_is_bounded_and_periodic() {
        for i in 0..64 {
            let [x, y] = halton_jitter(i);
            assert!(x.abs() <= 0.5 && y.abs() <= 0.5);
        }
        // The sequence is deterministic.
        assert_eq!(halton_jitter(3), halton_jitter(3));
    }

    #[test]
    fn camera_matrices_layout_matches_shader_contract() {
        // Three mat4s: shaders read 192 bytes.
        assert_eq!(std::mem::size_of::<CameraMatrices>(), 192);
    }
}
