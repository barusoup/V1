use super::anim::{NodePose, Player};
use super::{Model, ModelInstance, Node};
use glam::Mat4;
use std::sync::Arc;

impl ModelInstance {
    /// Instance at the model's rest pose, identity transform.
    pub fn new(model: Arc<Model>) -> Self {
        let pose: Vec<NodePose> = model.nodes.iter().map(Node::rest_pose).collect();
        let globals = compute_globals(&model, &pose, Mat4::IDENTITY);
        Self {
            model,
            player: Player::new(),
            pose,
            prev_globals: globals.clone(),
            globals,
            transform: Mat4::IDENTITY,
            prev_transform: Mat4::IDENTITY,
            last_transform: Mat4::IDENTITY,
        }
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn player_mut(&mut self) -> &mut Player {
        &mut self.player
    }

    /// Place the instance. The previous transform is captured at the next
    /// [`Self::update`], which is what motion vectors need.
    pub fn set_transform(&mut self, transform: Mat4) {
        self.transform = transform;
    }

    pub fn transform(&self) -> Mat4 {
        self.transform
    }

    /// Advance animation and recompute the node globals. Call once per frame
    /// *after* setting the transform. The previous frame's transform and
    /// globals are latched here, so draws after this call see correct
    /// (current, previous) motion pairs.
    pub fn update(&mut self, dt: f32) {
        self.prev_transform = self.last_transform;
        self.last_transform = self.transform;
        self.player
            .update(dt, self.model.nodes.len(), &mut self.pose);
        self.prev_globals = self.globals.clone();
        self.globals = compute_globals(&self.model, &self.pose, self.transform);
    }

    /// Global (model-space) transform of a node.
    pub fn global(&self, node: usize) -> Mat4 {
        self.globals.get(node).copied().unwrap_or(Mat4::IDENTITY)
    }

    /// (current, previous) model matrices — the per-object motion input.
    pub fn motion_matrices(&self) -> (Mat4, Mat4) {
        (self.transform, self.prev_transform)
    }

    /// Skinning matrices for `skin`: `global(joint) * inverse_bind`, current
    /// and previous frame. Previous matrices track the animated pose only
    /// (the instance transform is applied by the shader).
    pub fn skinning_matrices(&self, skin: usize) -> (Vec<Mat4>, Vec<Mat4>) {
        let Some(skin_data) = self.model.skins.get(skin) else {
            return (Vec::new(), Vec::new());
        };
        let mut current = Vec::with_capacity(skin_data.joints.len());
        let mut previous = Vec::with_capacity(skin_data.joints.len());
        for (index, &joint) in skin_data.joints.iter().enumerate() {
            let inverse = skin_data
                .inverse_binds
                .get(index)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let global = self.globals.get(joint).copied().unwrap_or(Mat4::IDENTITY);
            let prev_global = self
                .prev_globals
                .get(joint)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            current.push(global * inverse);
            previous.push(prev_global * inverse);
        }
        (current, previous)
    }
}

/// Propagate local poses down the hierarchy, right-multiplying the instance
/// transform at the roots.
pub(super) fn compute_globals(model: &Model, pose: &[NodePose], transform: Mat4) -> Vec<Mat4> {
    let mut globals = vec![Mat4::IDENTITY; model.nodes.len()];
    let mut stack: Vec<(usize, Mat4)> = model.roots.iter().map(|&root| (root, transform)).collect();
    while let Some((index, parent)) = stack.pop() {
        let Some(node) = model.nodes.get(index) else {
            continue;
        };
        let local = pose
            .get(index)
            .copied()
            .unwrap_or_else(|| node.rest_pose())
            .to_mat4();
        let global = parent * local;
        if index < globals.len() {
            globals[index] = global;
        }
        for &child in &node.children {
            stack.push((child, global));
        }
    }
    globals
}
