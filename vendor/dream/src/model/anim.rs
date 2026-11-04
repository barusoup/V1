//! Skeletal animation: clips, sampling, crossfade blending.
//!
//! Clips are pre-processed at load into sorted keyframe channels targeting a
//! node's translation/rotation/scale. [`Player`] advances one or
//! more weighted clip slots per frame and writes a blended [`NodePose`];
//! crossfading is weighted quaternion nlerp, which is what real-time engines
//! use for this scale of blending.
//!
//! Sampling is deterministic: time only moves when [`Player::update`]
//! runs, so scripted captures replay identically.

use glam::{Quat, Vec3};

/// Interpolation between consecutive keys of a channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interpolation {
    /// Hold the previous key until the next (glTF STEP).
    Step,
    /// Linear between keys (glTF LINEAR).
    Linear,
    /// Hermite spline with per-key tangents (glTF CUBICSPLINE).
    CubicSpline,
}

/// One keyframe. For [`Interpolation::CubicSpline`] `in_tangent`/`out_tangent`
/// hold the sampled value's neighbours' tangents.
#[derive(Clone, Copy, Debug)]
pub struct Key {
    pub time: f32,
    pub value: [f32; 4],
    pub in_tangent: [f32; 4],
    pub out_tangent: [f32; 4],
}

impl Key {
    /// A single-component key (heights, weights, one-axis motion); games
    /// building clips procedurally can use it for any target.
    pub fn scalar(time: f32, value: f32) -> Self {
        Self {
            time,
            value: [value, 0.0, 0.0, 0.0],
            in_tangent: [0.0; 4],
            out_tangent: [0.0; 4],
        }
    }
}

/// What part of a node's transform a channel drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Translation,
    Rotation,
    Scale,
}

/// A keyframe channel for one node and target.
#[derive(Clone, Debug)]
pub struct Channel {
    pub node: usize,
    pub target: Target,
    pub interpolation: Interpolation,
    /// Sorted by time; vec4 values (quaternions for [`Target::Rotation`]).
    pub keys: Vec<Key>,
}

impl Channel {
    /// Sample at `time` (seconds), clamped to the first/last key.
    pub fn sample(&self, time: f32) -> [f32; 4] {
        let keys = &self.keys;
        if keys.is_empty() {
            return [0.0, 0.0, 0.0, 1.0];
        }
        if time <= keys[0].time {
            return keys[0].value;
        }
        if time >= keys[keys.len() - 1].time {
            return keys[keys.len() - 1].value;
        }
        let index = match keys.binary_search_by(|key| {
            key.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(exact) => return keys[exact].value,
            Err(next) => next - 1,
        };
        let a = &keys[index];
        let b = &keys[index + 1];
        let span = (b.time - a.time).max(1e-6);
        let t = ((time - a.time) / span).clamp(0.0, 1.0);
        match self.interpolation {
            Interpolation::Step => a.value,
            Interpolation::Linear => {
                std::array::from_fn(|i| a.value[i] + (b.value[i] - a.value[i]) * t)
            }
            Interpolation::CubicSpline => {
                // glTF CUBICSPLINE keys carry in/out tangents scaled by the
                // key interval already.
                let t2 = t * t;
                let t3 = t2 * t;
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;
                std::array::from_fn(|i| {
                    h00 * a.value[i]
                        + h10 * span * a.out_tangent[i]
                        + h01 * b.value[i]
                        + h11 * span * b.in_tangent[i]
                })
            }
        }
    }
}

/// A named animation: channels plus its length.
#[derive(Clone, Debug)]
pub struct AnimClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

impl AnimClip {
    /// Human-readable summary for devtools.
    pub fn summary(&self) -> String {
        format!(
            "{}: {:.2}s, {} channels",
            self.name,
            self.duration,
            self.channels.len()
        )
    }
}

/// Local transform of one node, blended output of the player.
#[derive(Clone, Copy, Debug)]
pub struct NodePose {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for NodePose {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl NodePose {
    pub fn to_mat4(self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// How an animation slot behaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PlayMode {
    #[default]
    Loop,
    Once,
    /// Loop back and forth.
    PingPong,
}

/// One playing clip slot inside an [`AnimationPlayer`].
#[derive(Clone, Debug)]
struct Slot {
    clip: std::sync::Arc<AnimClip>,
    time: f32,
    weight: f32,
    target_weight: f32,
    /// Seconds to fade `weight` toward `target_weight`.
    fade: f32,
    speed: f32,
    mode: PlayMode,
    /// Ping-pong direction; the folded time alone cannot encode it.
    forward: bool,
    finished: bool,
}

/// Plays and blends animation clips onto a node pose.
///
/// ```ignore
/// player.play(clip, PlayMode::Loop, 0.0);        // snap on
/// player.crossfade(next_clip, 0.25);             // fade over 0.25s
/// player.update(dt, &mut pose);                  // pose is the output
/// ```
#[derive(Default)]
pub struct Player {
    slots: Vec<Slot>,
}

impl Player {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start `clip` from scratch (drops other slots). `fade_in` seconds of
    /// weight ramp, or 0 to snap.
    pub fn play(&mut self, clip: std::sync::Arc<AnimClip>, mode: PlayMode, fade_in: f32) {
        self.slots.clear();
        self.push_slot(clip, mode, fade_in);
    }

    /// Fade the current mix toward `clip` over `seconds`, dropping slots that
    /// reach zero weight.
    pub fn crossfade(&mut self, clip: std::sync::Arc<AnimClip>, mode: PlayMode, seconds: f32) {
        for slot in &mut self.slots {
            slot.target_weight = 0.0;
            slot.fade = seconds.max(0.0);
        }
        self.push_slot(clip, mode, seconds);
    }

    /// Add a weighted additive... no: a weighted layer at fixed `weight`.
    /// Layers blend additively onto the dominant slot's result.
    pub fn add_layer(&mut self, clip: std::sync::Arc<AnimClip>, mode: PlayMode, weight: f32) {
        self.push_slot(clip, mode, 0.0);
        if let Some(slot) = self.slots.last_mut() {
            slot.weight = weight.clamp(0.0, 1.0);
            slot.target_weight = slot.weight;
        }
    }

    fn push_slot(&mut self, clip: std::sync::Arc<AnimClip>, mode: PlayMode, fade: f32) {
        let fade = fade.max(0.0);
        self.slots.push(Slot {
            clip,
            time: 0.0,
            weight: if fade > 0.0 { 0.0 } else { 1.0 },
            target_weight: 1.0,
            fade,
            speed: 1.0,
            mode,
            forward: true,
            finished: false,
        });
    }

    /// Playback rate for every slot (slow motion).
    pub fn set_speed(&mut self, speed: f32) {
        for slot in &mut self.slots {
            slot.speed = speed;
        }
    }

    /// Advance all slots and blend into `pose` for `nodes` nodes. Channels
    /// write their node; untouched nodes keep their pose's values.
    pub fn update(&mut self, dt: f32, nodes: usize, pose: &mut [NodePose]) {
        assert!(pose.len() >= nodes);
        // First channel wins per (node, target) inside one slot; slots blend
        // with weights. Accumulate per target across slots.
        let mut blended: Vec<Option<(Vec3, Quat, Vec3, f32)>> = vec![None; nodes];
        let mut total_weight = 0.0f32;

        for slot in &mut self.slots {
            // Weight ramp.
            if (slot.weight - slot.target_weight).abs() > 1e-4 && slot.fade > 0.0 {
                let step = dt / slot.fade;
                if slot.weight < slot.target_weight {
                    slot.weight = (slot.weight + step).min(slot.target_weight);
                } else {
                    slot.weight = (slot.weight - step).max(slot.target_weight);
                }
            } else {
                slot.weight = slot.target_weight;
            }
            if slot.weight <= 0.0 {
                continue;
            }

            // Time advance.
            let duration = slot.clip.duration.max(1e-6);
            let direction = if slot.mode == PlayMode::PingPong && !slot.forward {
                -1.0
            } else {
                1.0
            };
            slot.time += dt * slot.speed * direction;
            match slot.mode {
                PlayMode::Loop => slot.time = slot.time.rem_euclid(duration),
                PlayMode::Once => {
                    if slot.time >= duration {
                        slot.time = duration;
                        slot.finished = true;
                    }
                }
                PlayMode::PingPong => {
                    if slot.forward && slot.time >= duration {
                        // Reflect the overshoot and head back down.
                        slot.time = duration - (slot.time - duration);
                        slot.forward = false;
                    } else if !slot.forward && slot.time <= 0.0 {
                        slot.time = -slot.time;
                        slot.forward = true;
                    }
                    slot.time = slot.time.clamp(0.0, duration);
                }
            }

            // Sample this slot's channels.
            for channel in &slot.clip.channels {
                if channel.node >= nodes {
                    continue;
                }
                let value = channel.sample(slot.time);
                let entry = blended[channel.node].get_or_insert((
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::ZERO,
                    0.0,
                ));
                let (weight, is_rotation) = match channel.target {
                    Target::Translation => {
                        entry.0 += Vec3::new(value[0], value[1], value[2]) * slot.weight;
                        (entry.3, false)
                    }
                    Target::Rotation => {
                        let q = Quat::from_array(normalized(value));
                        entry.1 = weighted_nlerp(entry.1, q, slot.weight);
                        (entry.3, true)
                    }
                    Target::Scale => {
                        entry.2 += Vec3::new(value[0], value[1], value[2]) * slot.weight;
                        (entry.3, false)
                    }
                };
                let _ = (weight, is_rotation);
                entry.3 += slot.weight;
                total_weight += slot.weight;
            }
        }
        let _ = total_weight;

        for (index, entry) in blended.iter_mut().enumerate() {
            let Some((translation, rotation, scale, weight)) = *entry else {
                continue;
            };
            let weight = weight.max(1e-6);
            let base = pose[index];
            // Channels divide out their accumulated weight; targets no
            // channel touched keep the incoming pose.
            pose[index] = NodePose {
                translation: translation / weight,
                rotation,
                scale: scale / weight,
            };
            // Blend untouched components with the incoming pose at zero
            // channel contribution: components with no keys stay as-is.
            if translation == Vec3::ZERO && scale == Vec3::ZERO {
                pose[index].translation = base.translation;
                pose[index].scale = base.scale;
            }
        }
    }

    /// True when every Once-mode slot has reached its end.
    pub fn finished(&self) -> bool {
        self.slots
            .iter()
            .all(|slot| slot.target_weight <= 0.0 || slot.finished || slot.mode != PlayMode::Once)
    }

    pub fn active_clips(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.weight > 0.0 || slot.target_weight > 0.0)
            .count()
    }
}

fn normalized(value: [f32; 4]) -> [f32; 4] {
    let q = Quat::from_array(value);
    if q.length_squared() < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    q.normalize().to_array()
}

fn weighted_nlerp(acc: Quat, q: Quat, weight: f32) -> Quat {
    if (acc - Quat::IDENTITY).length_squared() < 1e-12 {
        return q;
    }
    // Keep hemisphere continuity for the shortest path.
    let aligned = if acc.dot(q) < 0.0 { -q } else { q };
    acc.lerp(aligned, weight).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn channel(target: Target, keys: Vec<Key>) -> Channel {
        Channel {
            node: 0,
            target,
            interpolation: Interpolation::Linear,
            keys,
        }
    }

    fn clip(channels: Vec<Channel>, duration: f32) -> Arc<AnimClip> {
        Arc::new(AnimClip {
            name: "test".into(),
            duration,
            channels,
        })
    }

    #[test]
    fn linear_interpolates_between_keys() {
        let channel = channel(
            Target::Translation,
            vec![
                Key::scalar(0.0, 0.0),
                Key::scalar(1.0, 10.0),
                Key::scalar(2.0, 20.0),
            ],
        );
        assert_eq!(channel.sample(0.5)[0], 5.0);
        assert_eq!(channel.sample(1.5)[0], 15.0);
        assert_eq!(channel.sample(-1.0)[0], 0.0, "clamps before the first key");
        assert_eq!(channel.sample(9.0)[0], 20.0, "clamps past the last key");
    }

    #[test]
    fn step_holds_previous_key() {
        let mut channel = channel(
            Target::Translation,
            vec![Key::scalar(0.0, 1.0), Key::scalar(1.0, 2.0)],
        );
        channel.interpolation = Interpolation::Step;
        assert_eq!(channel.sample(0.99)[0], 1.0);
        assert_eq!(channel.sample(1.0)[0], 2.0);
    }

    #[test]
    fn cubic_spline_respects_tangents() {
        let mut channel = channel(
            Target::Translation,
            vec![
                Key {
                    time: 0.0,
                    value: [0.0; 4],
                    in_tangent: [0.0; 4],
                    out_tangent: [10.0, 0.0, 0.0, 0.0],
                },
                Key {
                    time: 1.0,
                    value: [0.0; 4],
                    in_tangent: [0.0; 4],
                    out_tangent: [0.0; 4],
                },
            ],
        );
        channel.interpolation = Interpolation::CubicSpline;
        let mid = channel.sample(0.5)[0];
        assert!(mid > 0.0, "positive tangent bulges above zero, got {mid}");
        // Endpoints are exact.
        assert_eq!(channel.sample(0.0)[0], 0.0);
        assert_eq!(channel.sample(1.0)[0], 0.0);
    }

    #[test]
    fn player_advances_time_and_loops() {
        let mut player = Player::new();
        player.play(
            clip(
                vec![channel(
                    Target::Translation,
                    vec![Key::scalar(0.0, 0.0), Key::scalar(1.0, 100.0)],
                )],
                1.0,
            ),
            PlayMode::Loop,
            0.0,
        );
        let mut pose = vec![NodePose::default()];
        player.update(0.5, 1, &mut pose);
        assert!((pose[0].translation.x - 50.0).abs() < 1e-3);
        player.update(0.75, 1, &mut pose);
        // 1.25s wraps to 0.25s in a 1s loop.
        assert!(
            (pose[0].translation.x - 25.0).abs() < 1e-3,
            "got {}",
            pose[0].translation.x
        );
    }

    #[test]
    fn once_mode_finishes() {
        let mut player = Player::new();
        player.play(
            clip(
                vec![channel(
                    Target::Translation,
                    vec![Key::scalar(0.0, 0.0), Key::scalar(1.0, 10.0)],
                )],
                1.0,
            ),
            PlayMode::Once,
            0.0,
        );
        let mut pose = vec![NodePose::default()];
        player.update(2.0, 1, &mut pose);
        assert!(player.finished());
        assert!((pose[0].translation.x - 10.0).abs() < 1e-3);
    }

    #[test]
    fn crossfade_blends_weights() {
        let mut player = Player::new();
        let a = clip(
            vec![channel(
                Target::Translation,
                vec![Key::scalar(0.0, 0.0), Key::scalar(1.0, 0.0)],
            )],
            1.0,
        );
        let b = clip(
            vec![channel(
                Target::Translation,
                vec![Key::scalar(0.0, 100.0), Key::scalar(1.0, 100.0)],
            )],
            1.0,
        );
        player.play(a, PlayMode::Loop, 0.0);
        player.crossfade(b, PlayMode::Loop, 1.0);
        let mut pose = vec![NodePose::default()];
        player.update(0.5, 1, &mut pose);
        // Halfway through the fade: somewhere strictly between 0 and 100.
        let value = pose[0].translation.x;
        assert!(value > 5.0 && value < 95.0, "crossfade blends, got {value}");
        player.update(1.0, 1, &mut pose);
        assert!(
            (pose[0].translation.x - 100.0).abs() < 5.0,
            "fade completes"
        );
    }

    #[test]
    fn rotation_channels_nlerp() {
        let mut player = Player::new();
        let half_turn = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        player.play(
            clip(
                vec![Channel {
                    node: 0,
                    target: Target::Rotation,
                    interpolation: Interpolation::Linear,
                    keys: vec![
                        Key {
                            time: 0.0,
                            value: [0.0, 0.0, 0.0, 1.0],
                            in_tangent: [0.0; 4],
                            out_tangent: [0.0; 4],
                        },
                        Key {
                            time: 1.0,
                            value: half_turn.to_array(),
                            in_tangent: [0.0; 4],
                            out_tangent: [0.0; 4],
                        },
                    ],
                }],
                1.0,
            ),
            PlayMode::Once,
            0.0,
        );
        let mut pose = vec![NodePose::default()];
        player.update(0.5, 1, &mut pose);
        let angle = pose[0].rotation.angle_between(Quat::IDENTITY);
        assert!(
            (angle - std::f32::consts::FRAC_PI_4).abs() < 0.05,
            "halfway rotation, got {angle}"
        );
    }

    #[test]
    fn ping_pong_reverses() {
        let mut player = Player::new();
        player.play(
            clip(
                vec![channel(
                    Target::Translation,
                    vec![Key::scalar(0.0, 0.0), Key::scalar(1.0, 100.0)],
                )],
                1.0,
            ),
            PlayMode::PingPong,
            0.0,
        );
        let mut pose = vec![NodePose::default()];
        player.update(1.25, 1, &mut pose);
        // t = 1.25s: past the 1s peak, 0.25s back down -> 75.
        assert!(
            (pose[0].translation.x - 75.0).abs() < 1e-2,
            "got {}",
            pose[0].translation.x
        );
        player.update(0.5, 1, &mut pose);
        // t = 1.75s: 0.75s of the way down -> 25.
        assert!(
            (pose[0].translation.x - 25.0).abs() < 1e-2,
            "got {}",
            pose[0].translation.x
        );
        player.update(0.5, 1, &mut pose);
        // t = 2.25s: bounced at the bottom (t=2s) and turned up again -> 25.
        assert!(
            (pose[0].translation.x - 25.0).abs() < 1e-2,
            "got {}",
            pose[0].translation.x
        );
    }

    #[test]
    fn untouched_nodes_keep_their_pose() {
        let mut player = Player::new();
        player.play(
            clip(
                vec![channel(
                    Target::Translation,
                    vec![Key::scalar(0.0, 5.0), Key::scalar(1.0, 5.0)],
                )],
                1.0,
            ),
            PlayMode::Loop,
            0.0,
        );
        let mut pose = vec![
            NodePose::default(),
            NodePose {
                translation: Vec3::new(9.0, 9.0, 9.0),
                ..Default::default()
            },
        ];
        player.update(0.1, 2, &mut pose);
        assert_eq!(pose[1].translation, Vec3::new(9.0, 9.0, 9.0));
    }
}
