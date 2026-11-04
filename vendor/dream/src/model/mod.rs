//! Models: glTF 2.0 loading, node hierarchy, skinning, instances.
//!
//! A [`Model`] is pure CPU data (meshes, materials, nodes, skins, animation
//! clips) and loads without a GPU, so it unit tests on its own. GPU upload
//! lives in [`render`]; textures stay as decoded pixels on the material until
//! a renderer claims them.
//!
//! [`ModelInstance`] carries the animated state plus the *previous* frame's
//! transforms — the pair feeds motion vectors for temporal upscaling (see
//! [`crate::temporal`]).

pub mod anim;
pub mod render;

pub use anim::{AnimClip, Channel, Interpolation, Key, NodePose, PlayMode, Player, Target};

mod gltf;
mod instance;
mod material;

use instance::compute_globals;
pub use material::{AlphaMode, Material, TexelFilter, TextureRef, UvTransform, Wrap};

use glam::{Mat4, Quat, Vec3};
use std::sync::Arc;

/// One mesh primitive: indexed triangle soup with optional skin weights.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// Second UV set (`TEXCOORD_1`); empty when the mesh has none.
    pub uvs1: Vec<[f32; 2]>,
    /// Tangent + bitangent sign (`TANGENT`); empty when the file omits them.
    pub tangents: Vec<[f32; 4]>,
    /// Four joint indices per vertex; empty for static meshes.
    pub joints: Vec<[u16; 4]>,
    /// Four weights per vertex; empty for static meshes.
    pub weights: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    /// Index into [`Model::materials`].
    pub material: usize,
    /// Node this primitive is attached to (glTF nodes can share a mesh; each
    /// flattened mesh remembers the first referencing node). Needed to apply
    /// node transforms to static draws and bounds.
    pub node: Option<usize>,
}

impl Mesh {
    pub fn is_skinned(&self) -> bool {
        !self.joints.is_empty() && !self.weights.is_empty()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
}

/// One node in the hierarchy.
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub children: Vec<usize>,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
}

impl Node {
    pub fn rest_pose(&self) -> NodePose {
        NodePose {
            translation: self.translation,
            rotation: self.rotation,
            scale: self.scale,
        }
    }
}

/// A skin: joint nodes and their inverse bind matrices.
#[derive(Clone, Debug, Default)]
pub struct Skin {
    /// Node indices, in joint order.
    pub joints: Vec<usize>,
    pub inverse_binds: Vec<Mat4>,
}

/// The whole model. See the module docs.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub nodes: Vec<Node>,
    pub skins: Vec<Skin>,
    pub clips: Vec<AnimClip>,
    pub roots: Vec<usize>,
}

impl Model {
    /// Rest-pose global transform of every node (identity instance
    /// transform) — the matrix a static mesh primitive needs to land in
    /// object space.
    pub fn rest_globals(&self) -> Vec<Mat4> {
        let pose: Vec<NodePose> = self.nodes.iter().map(Node::rest_pose).collect();
        compute_globals(self, &pose, Mat4::IDENTITY)
    }

    /// The matrix a flattened mesh needs in object space (its node's rest
    /// global, or identity for meshes no node references).
    pub fn mesh_node_transform(&self, flattened_mesh_index: usize) -> Mat4 {
        let Some(node) = self.meshes[flattened_mesh_index].node else {
            return Mat4::IDENTITY;
        };
        let pose: Vec<NodePose> = self.nodes.iter().map(Node::rest_pose).collect();
        compute_globals(self, &pose, Mat4::IDENTITY)
            .get(node)
            .copied()
            .unwrap_or(Mat4::IDENTITY)
    }

    /// Object-space bounds over every mesh a node references, positions
    /// transformed by the node's rest-pose global — includes node rotation/
    /// scale (e.g. DamagedHelmet's root +90° X). `(min, max)`; degenerate
    /// models return `(0,0)` bounds.
    pub fn static_bounds(&self) -> (Vec3, Vec3) {
        let globals = self.rest_globals();
        let (mut mn, mut mx) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for mesh in self.meshes.iter() {
            let Some(node) = mesh.node else { continue };
            let global = globals.get(node).copied().unwrap_or(Mat4::IDENTITY);
            for &position in &mesh.positions {
                let world = global.transform_point3(Vec3::from(position));
                mn = mn.min(world);
                mx = mx.max(world);
            }
        }
        if mn.cmpgt(mx).any() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        (mn, mx)
    }

    /// Meshes referencing `mesh_index` in document order. Primitive meshes are
    /// flattened at load; this maps a glTF mesh back to its primitives.
    pub fn summary(&self) -> String {
        format!(
            "{} meshes, {} materials, {} nodes, {} skins, {} clips",
            self.meshes.len(),
            self.materials.len(),
            self.nodes.len(),
            self.skins.len(),
            self.clips.len()
        )
    }
}

/// Animated state of one model: pose, transforms and the previous frame's
/// values for motion vectors.
pub struct ModelInstance {
    model: Arc<Model>,
    player: Player,
    pose: Vec<NodePose>,
    globals: Vec<Mat4>,
    prev_globals: Vec<Mat4>,
    transform: Mat4,
    prev_transform: Mat4,
    /// Transform consumed by the previous [`Self::update`]; `set_transform`
    /// may overwrite `transform` before update runs, so the old value is
    /// parked here.
    last_transform: Mat4,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GLB builder: a skinned two-triangle-ish node with a
    /// translation animation. Exercises the real loader end to end.
    pub fn test_glb() -> Vec<u8> {
        // Buffer layout (little endian):
        // 0..36   positions (3 vertices, vec3 f32)
        // 36..42  indices (3 x u16)
        // 44..68  joints (3 x u16x4)
        // 68..116 weights (3 x vec4 f32)
        // 116..244 inverse bind matrices (2 x mat4)
        // 244..252 animation input (2 x f32)
        // 252..276 animation output (2 x vec3)
        let mut bin = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for value in v {
                bin.extend_from_slice(&value.to_le_bytes());
            }
        }
        for index in [0u16, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        bin.extend_from_slice(&[0u8; 2]); // pad to 4-byte alignment
        for joints in [[0u16, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]] {
            for joint in joints {
                bin.extend_from_slice(&joint.to_le_bytes());
            }
        }
        for weights in [
            [1.0f32, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ] {
            for weight in weights {
                bin.extend_from_slice(&weight.to_le_bytes());
            }
        }
        for _ in 0..2 {
            // Identity matrix, column-major.
            let identity = [
                1.0f32, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ];
            for value in identity {
                bin.extend_from_slice(&value.to_le_bytes());
            }
        }
        for time in [0.0f32, 1.0] {
            bin.extend_from_slice(&time.to_le_bytes());
        }
        for offset in [[0.0f32, 0.0, 0.0], [2.0, 0.0, 0.0]] {
            for value in offset {
                bin.extend_from_slice(&value.to_le_bytes());
            }
        }

        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},
            "scene":0,"scenes":[{{"nodes":[0]}}],
            "nodes":[
                {{"name":"root","mesh":0,"skin":0,"children":[1]}},
                {{"name":"j1","translation":[0,0,0],"children":[2]}},
                {{"name":"j2","translation":[0,1,0]}}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"JOINTS_0":2,"WEIGHTS_0":3}},"indices":1,"material":0}}]}}],
            "skins":[{{"inverseBindMatrices":4,"joints":[1,2]}}],
            "animations":[{{"name":"move","samplers":[{{"input":5,"output":6,"interpolation":"LINEAR"}}],
                "channels":[{{"sampler":0,"target":{{"node":1,"path":"translation"}}}}]}}],
            "materials":[{{"name":"red","pbrMetallicRoughness":{{"baseColorFactor":[1,0,0,1]}}}}],
            "accessors":[
                {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}},
                {{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}},
                {{"bufferView":2,"componentType":5123,"count":3,"type":"VEC4"}},
                {{"bufferView":3,"componentType":5126,"count":3,"type":"VEC4"}},
                {{"bufferView":4,"componentType":5126,"count":2,"type":"MAT4"}},
                {{"bufferView":5,"componentType":5126,"count":2,"type":"SCALAR"}},
                {{"bufferView":6,"componentType":5126,"count":2,"type":"VEC3"}}],
            "bufferViews":[
                {{"buffer":0,"byteOffset":0,"byteLength":36}},
                {{"buffer":0,"byteOffset":36,"byteLength":6}},
                {{"buffer":0,"byteOffset":44,"byteLength":24}},
                {{"buffer":0,"byteOffset":68,"byteLength":48}},
                {{"buffer":0,"byteOffset":116,"byteLength":128}},
                {{"buffer":0,"byteOffset":244,"byteLength":8}},
                {{"buffer":0,"byteOffset":252,"byteLength":24}}],
            "buffers":[{{"byteLength":{}}}]}}"#,
            bin.len()
        );
        let mut json_bytes = json.into_bytes();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::new();
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);
        glb
    }

    #[test]
    fn loads_glb_mesh_skin_and_clip() {
        let model = Model::load_gltf(&test_glb()).expect("load glb");
        assert_eq!(model.meshes.len(), 1);
        let mesh = &model.meshes[0];
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert!(mesh.is_skinned());
        assert_eq!(model.skins.len(), 1);
        assert_eq!(model.skins[0].joints, vec![1, 2]);
        assert_eq!(model.clips.len(), 1);
        assert_eq!(model.clips[0].name, "move");
        assert!((model.clips[0].duration - 1.0).abs() < 1e-5);
        assert_eq!(model.materials[0].base_color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn instance_animates_and_tracks_previous_transforms() {
        let model = Model::load_gltf(&test_glb()).expect("load glb");
        let mut instance = ModelInstance::new(model.clone());
        instance
            .player_mut()
            .play(model.clips[0].clone().into(), PlayMode::Once, 0.0);
        instance.update(0.5);
        // Halfway through the clip: joint 1 has moved +1 in x.
        let (skin, _) = instance.skinning_matrices(0);
        assert!(
            (skin[0].col(3).x - 1.0).abs() < 1e-3,
            "got {}",
            skin[0].col(3).x
        );

        // Motion matrices: first update snaps prev to current.
        let (current, previous) = instance.motion_matrices();
        assert_eq!(current, previous);

        instance.set_transform(Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));
        instance.update(0.0);
        let (current, previous) = instance.motion_matrices();
        assert_eq!(previous, Mat4::IDENTITY);
        assert_eq!(current, Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)));

        // Skinning previous matrices lag one update behind.
        let (_, previous_skin) = instance.skinning_matrices(0);
        assert!(
            (previous_skin[0].col(3).x - 1.0).abs() < 1e-3,
            "previous skin keeps the last pose"
        );
    }

    /// GLB wrapping a JSON doc + binary chunk.
    fn glb_of(json: &str, bin: &[u8]) -> Vec<u8> {
        let mut json_bytes = json.as_bytes().to_vec();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let mut bin = bin.to_vec();
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::new();
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);
        glb
    }

    /// A glTF whose material uses TEXCOORD_1 + KHR_texture_transform +
    /// KHR_materials_clearcoat (unsupported) on a masked base colour map.
    fn transform_glb() -> Vec<u8> {
        // buffer: 3 pos (36) + 3 uv0 (24) + 3 uv1 (24) + 3 idx u16 (6, pad 2)
        // + 1x1 png
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(1, 1, |_, _| {
            image::Rgba([10, 200, 30, 255])
        }))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png encode");
        let mut bin = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for x in v {
                bin.extend_from_slice(&x.to_le_bytes());
            }
        }
        for uv in [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
            for x in uv {
                bin.extend_from_slice(&x.to_le_bytes());
            }
        }
        for uv in [[0.5f32, 0.5], [0.25, 0.25], [0.75, 0.75]] {
            for x in uv {
                bin.extend_from_slice(&x.to_le_bytes());
            }
        }
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        bin.extend_from_slice(&[0; 2]);
        let png_off = bin.len();
        bin.extend_from_slice(&png);
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},
            "scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
            "extensionsUsed":["KHR_texture_transform","KHR_materials_clearcoat"],
            "images":[{{"mimeType":"image/png","bufferView":4}}],
            "samplers":[{{"wrapS":33071,"wrapT":33648}}],
            "textures":[{{"sampler":0,"source":0}}],
            "materials":[{{
                "doubleSided":true,"alphaMode":"MASK","alphaCutoff":0.7,
                "pbrMetallicRoughness":{{
                    "baseColorFactor":[0.5,0.5,0.5,1],
                    "metallicFactor":0.7,"roughnessFactor":0.3,
                    "baseColorTexture":{{"index":0,"texCoord":1,
                        "extensions":{{"KHR_texture_transform":{{
                            "offset":[0.1,0.2],"rotation":0.5,
                            "scale":[2.0,2.0],"texCoord":1}}}}}}
                }},
                "extensions":{{"KHR_materials_clearcoat":{{"clearcoatFactor":1.0}}}}
            }}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"TEXCOORD_0":1,"TEXCOORD_1":2}},"indices":3,"material":0}}]}}],
            "accessors":[
                {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
                {{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}},
                {{"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"}},
                {{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}}],
            "bufferViews":[
                {{"buffer":0,"byteOffset":0,"byteLength":36}},
                {{"buffer":0,"byteOffset":36,"byteLength":24}},
                {{"buffer":0,"byteOffset":60,"byteLength":24}},
                {{"buffer":0,"byteOffset":84,"byteLength":6}},
                {{"buffer":0,"byteOffset":{png_off},"byteLength":{}}}],
            "buffers":[{{"byteLength":{}}}]}}"#,
            png.len(),
            bin.len()
        );
        glb_of(&json, &bin)
    }

    #[test]
    fn damaged_helmet_loads_full_pbr_material() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/DamagedHelmet.glb");
        let model = Model::load_gltf_file(path).expect("load helmet");
        let material = &model.materials[0];
        for map in [
            material.base_color_map.as_ref(),
            material.metallic_roughness_texture.as_ref(),
            material.normal_texture.as_ref(),
            material.occlusion_texture.as_ref(),
            material.emissive_texture.as_ref(),
        ] {
            let map = map.expect("helmet material must carry all five maps");
            assert_eq!(map.uv_set(), 0, "helmet uses TEXCOORD_0");
            assert!(!map.texture.pixels.is_empty());
        }
        assert!(material.unsupported.is_empty());
        assert_eq!(material.unsupported_reason(), None);
        assert_eq!(material.alpha_mode, AlphaMode::Opaque);
        // base_colour mirror field intact for the simple path
        assert!(material.base_color_texture.is_some());
    }

    #[test]
    fn texture_transform_second_uv_and_unsupported_extension_parse() {
        let model = Model::load_gltf(&transform_glb()).expect("load transform glb");
        let mesh = &model.meshes[0];
        assert_eq!(mesh.uvs1.len(), 3, "TEXCOORD_1 must load");
        let material = &model.materials[0];
        let map = material.base_color_map.as_ref().expect("base colour map");
        assert_eq!(map.tex_coord, 1);
        let t = map.transform.expect("KHR_texture_transform");
        assert!((t.offset[0] - 0.1).abs() < 1e-6 && (t.offset[1] - 0.2).abs() < 1e-6);
        assert!((t.rotation - 0.5).abs() < 1e-6);
        assert_eq!(t.scale, [2.0, 2.0]);
        assert_eq!(t.tex_coord, Some(1));
        assert_eq!(map.wrap_s, Wrap::ClampToEdge);
        assert_eq!(map.wrap_t, Wrap::MirroredRepeat);
        assert_eq!(material.alpha_mode, AlphaMode::Mask);
        assert!((material.alpha_cutoff - 0.7).abs() < 1e-6);
        assert!(
            material
                .unsupported
                .iter()
                .any(|e| e == "KHR_materials_clearcoat")
        );
        assert!(material.unsupported_reason().is_some());
    }

    #[test]
    fn rejects_garbage() {
        let err = Model::load_gltf(b"not a gltf").expect_err("garbage must fail");
        assert!(err.contains("import gltf"));
    }

    #[test]
    fn globals_propagate_through_roots() {
        let model = Model::load_gltf(&test_glb()).expect("load glb");
        let pose: Vec<NodePose> = model.nodes.iter().map(Node::rest_pose).collect();
        let globals = compute_globals(&model, &pose, Mat4::from_translation(Vec3::X));
        // Node 0 is a root: global = transform * local.
        assert_eq!(globals[0], Mat4::from_translation(Vec3::X));
        // Joint 2 rests 1 unit up.
        assert!((globals[2].col(3).y - 1.0).abs() < 1e-6);
    }
}
