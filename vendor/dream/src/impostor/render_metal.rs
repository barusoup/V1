//! Native Metal impostor renderer: instanced bounding-box shells shaded as the baked mesh.

use super::atlas::AssetDebug;
use super::{ImpostorAsset, ImpostorBounds, VIEWS, projected_extent};
use crate::gfx::metal::pipeline::{create_depth_stencil_state, create_render_pipeline};
use crate::gfx::metal::shader::MslBindings;
use crate::gfx::{DrawCtx, GpuBuffer, GpuContext, align_up};
use crate::lighting::Lighting;
use glam::Mat4;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCompareFunction, MTLCullMode, MTLDepthStencilState, MTLDevice, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPipelineState, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState, MTLWinding,
};
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpostorInstanceError {
    NonFiniteMatrix,
    UnsupportedScale,
    InvalidOpacity,
    InvalidInverse,
}

impl fmt::Display for ImpostorInstanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteMatrix => {
                write!(f, "model or prev_model contains NaN or infinite values")
            }
            Self::UnsupportedScale => write!(
                f,
                "impostor transform must be proper rotation + positive uniform scale + translation"
            ),
            Self::InvalidOpacity => write!(f, "impostor opacity must be finite and in [0, 1]"),
            Self::InvalidInverse => write!(f, "impostor inv_model does not match model.inverse()"),
        }
    }
}

impl std::error::Error for ImpostorInstanceError {}

fn check_transform(m: &[[f32; 4]; 4]) -> Result<(), ImpostorInstanceError> {
    if !m.iter().flatten().all(|v| v.is_finite()) {
        return Err(ImpostorInstanceError::NonFiniteMatrix);
    }
    let eps = 1e-4;
    if m[0][3].abs() > eps
        || m[1][3].abs() > eps
        || m[2][3].abs() > eps
        || (m[3][3] - 1.0).abs() > eps
    {
        return Err(ImpostorInstanceError::UnsupportedScale);
    }
    let cols = [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ];
    let len = cols.map(|c| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt());
    if len
        .iter()
        .any(|l| !l.is_finite() || !(1e-6..=1e6).contains(l))
    {
        return Err(ImpostorInstanceError::UnsupportedScale);
    }
    let s0 = len[0];
    if !len.iter().all(|l| (l - s0).abs() <= 1e-4 * s0.max(1e-12)) {
        return Err(ImpostorInstanceError::UnsupportedScale);
    }
    let norm: [[f32; 3]; 3] = std::array::from_fn(|i| {
        [
            cols[i][0] / len[i],
            cols[i][1] / len[i],
            cols[i][2] / len[i],
        ]
    });
    for i in 0..3 {
        for j in (i + 1)..3 {
            let d = norm[i][0] * norm[j][0] + norm[i][1] * norm[j][1] + norm[i][2] * norm[j][2];
            if d.abs() > 1e-4 {
                return Err(ImpostorInstanceError::UnsupportedScale);
            }
        }
    }
    let det = cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1])
        - cols[0][1] * (cols[1][0] * cols[2][2] - cols[1][2] * cols[2][0])
        + cols[0][2] * (cols[1][0] * cols[2][1] - cols[1][1] * cols[2][0]);
    if !det.is_finite() || det <= 0.0 {
        return Err(ImpostorInstanceError::UnsupportedScale);
    }
    Ok(())
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImpostorInstance {
    pub model: [[f32; 4]; 4],
    pub prev_model: [[f32; 4]; 4],
    pub inv_model: [[f32; 4]; 4],
    pub misc: [f32; 4],
}

impl ImpostorInstance {
    pub fn try_new(
        model: Mat4,
        prev_model: Mat4,
        alpha: f32,
    ) -> Result<Self, ImpostorInstanceError> {
        let inv = model.inverse();
        let out = Self {
            model: model.to_cols_array_2d(),
            prev_model: prev_model.to_cols_array_2d(),
            inv_model: inv.to_cols_array_2d(),
            misc: [alpha, 0.0, 0.0, 0.0],
        };
        out.validate()?;
        Ok(out)
    }

    pub fn validate(&self) -> Result<(), ImpostorInstanceError> {
        check_transform(&self.model)?;
        check_transform(&self.prev_model)?;
        let alpha = self.misc[0];
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(ImpostorInstanceError::InvalidOpacity);
        }
        if !self.inv_model.iter().flatten().all(|v| v.is_finite()) {
            return Err(ImpostorInstanceError::InvalidInverse);
        }
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0f32;
                for k in 0..4 {
                    acc += self.model[k][i] * self.inv_model[j][k];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                if !acc.is_finite() || (acc - want).abs() > 1e-3 {
                    return Err(ImpostorInstanceError::InvalidInverse);
                }
            }
        }
        Ok(())
    }

    pub fn new(model: Mat4, prev_model: Mat4, alpha: f32) -> Self {
        Self::try_new(model, prev_model, alpha).expect("impostor instance transform")
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AssetParams {
    box_min: [f32; 4],
    box_max: [f32; 4],
    params: [f32; 4],
    screen: [f32; 4],
    material: [f32; 4],
    u_axes: [[f32; 4]; super::VIEW_COUNT],
    v_axes: [[f32; 4]; super::VIEW_COUNT],
    extents: [[f32; 4]; super::VIEW_COUNT],
    views: [[f32; 4]; super::VIEW_COUNT],
}

unsafe impl bytemuck::Zeroable for AssetParams {}
unsafe impl bytemuck::Pod for AssetParams {}

struct Slot {
    frame_id: u64,
    instance_buffer: GpuBuffer,
    instance_capacity: u64,
    instance_used: u64,
    params_buffer: GpuBuffer,
    params_capacity: u64,
    params_used: u64,
    retired: Vec<GpuBuffer>,
}

#[derive(Debug)]
pub enum ImpostorDrawError {
    InvalidInstance(ImpostorInstanceError),
    InvalidView,
    WrongDevice,
    TooManyInstances,
    StorageLimit,
}

impl fmt::Display for ImpostorDrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInstance(e) => write!(f, "invalid impostor instance: {e}"),
            Self::InvalidView => {
                write!(f, "view height/fov/grad_scale must be positive and finite")
            }
            Self::WrongDevice => write!(f, "impostor asset belongs to a different GPU context"),
            Self::TooManyInstances => write!(f, "impostor instance batch is too large"),
            Self::StorageLimit => write!(f, "instance batch exceeds maxStorageBufferRange"),
        }
    }
}

impl std::error::Error for ImpostorDrawError {}

pub struct ImpostorRenderer {
    ctx: Arc<GpuContext>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth_stencil: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    material_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    instance_align: u64,
    uniform_align: u64,
    slots: Vec<Slot>,
    retired_assets: Vec<(ImpostorAsset, u64)>,
    draws: u64,
    instances: u64,
    pub grad_scale: f32,
}

impl ImpostorRenderer {
    pub fn new(
        ctx: &Arc<GpuContext>,
        color_formats: &[MTLPixelFormat],
        depth_format: MTLPixelFormat,
        _lighting: &Lighting,
    ) -> Self {
        let bindings = MslBindings::new()
            // @group(0): camera
            .buffer(0, 0, 0, false)
            // @group(1): instances
            .buffer(1, 0, 1, false)
            // @group(2): asset, textures, samplers, voxels, material_table
            .buffer(2, 0, 2, false)
            .texture(2, 1, 0)
            .texture(2, 2, 1)
            .texture(2, 3, 2)
            .texture(2, 4, 3)
            .texture(2, 5, 4)
            .sampler(2, 6, 0)
            .buffer(2, 7, 3, false)
            .texture(2, 8, 5)
            .texture(2, 9, 6)
            .texture(2, 10, 7)
            .texture(2, 11, 8)
            .texture(2, 12, 9)
            .sampler(2, 13, 1)
            .buffer(2, 14, 4, false)
            // @group(3): lighting_env
            .buffer(3, 0, 5, false);

        let src = format!(
            "{}\n{}",
            include_str!("../lighting/lighting.wgsl"),
            include_str!("impostor.wgsl")
        );
        let pipeline = create_render_pipeline(
            ctx,
            "impostor",
            &src,
            &bindings,
            "vs_main",
            "fs_main",
            color_formats,
            Some(depth_format),
            None,
            false,
        );
        let depth_stencil =
            create_depth_stencil_state(ctx, true, true, MTLCompareFunction::LessEqual);

        let sampler = make_sampler(ctx, MTLSamplerAddressMode::ClampToEdge, 8.0);
        let material_sampler = make_sampler(ctx, MTLSamplerAddressMode::Repeat, 8.0);

        let instance_align = 256u64;
        let uniform_align = 256u64;
        let inst_cap = 256 * 1024u64;
        let params_cap = 64 * 1024u64;
        let slots = (0..3)
            .map(|_| Slot {
                frame_id: u64::MAX,
                instance_buffer: ctx.create_host_storage_buffer("impostor instances", inst_cap),
                instance_capacity: inst_cap,
                instance_used: 0,
                params_buffer: ctx.create_host_buffer("impostor asset params", params_cap),
                params_capacity: params_cap,
                params_used: 0,
                retired: Vec::new(),
            })
            .collect();

        Self {
            ctx: Arc::clone(ctx),
            pipeline,
            depth_stencil,
            sampler,
            material_sampler,
            instance_align,
            uniform_align,
            slots,
            retired_assets: Vec::new(),
            draws: 0,
            instances: 0,
            grad_scale: 0.5,
        }
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }

    pub fn instances_submitted(&self) -> u64 {
        self.instances
    }

    pub fn draw(
        &mut self,
        ctx: &mut DrawCtx<'_>,
        asset: &mut ImpostorAsset,
        instances: &[ImpostorInstance],
        lighting: &Lighting,
        screen_px: f32,
        fov_tan: f32,
        debug: AssetDebug,
    ) {
        self.try_draw(ctx, asset, instances, lighting, screen_px, fov_tan, debug)
            .expect("impostor draw");
    }

    pub fn try_draw(
        &mut self,
        ctx: &mut DrawCtx<'_>,
        asset: &mut ImpostorAsset,
        instances: &[ImpostorInstance],
        lighting: &Lighting,
        screen_px: f32,
        fov_tan: f32,
        debug: AssetDebug,
    ) -> Result<(), ImpostorDrawError> {
        if !screen_px.is_finite()
            || screen_px <= 0.0
            || !fov_tan.is_finite()
            || fov_tan <= 0.0
            || !self.grad_scale.is_finite()
            || self.grad_scale < 0.0
        {
            return Err(ImpostorDrawError::InvalidView);
        }
        if !Arc::ptr_eq(&asset.ctx, &self.ctx) {
            return Err(ImpostorDrawError::WrongDevice);
        }
        let inst_size = std::mem::size_of::<ImpostorInstance>() as u64;
        let stride = inst_size
            .checked_mul(instances.len() as u64)
            .ok_or(ImpostorDrawError::TooManyInstances)?;
        if instances.len() > u32::MAX as usize || stride > u64::from(u32::MAX) {
            return Err(ImpostorDrawError::TooManyInstances);
        }
        let sb_range = u64::from(self.ctx.max_storage_buffer_range());
        if stride > sb_range {
            return Err(ImpostorDrawError::StorageLimit);
        }
        if asset.voxels.size > sb_range || asset.material_table.size > sb_range {
            return Err(ImpostorDrawError::StorageLimit);
        }
        for inst in instances {
            inst.validate()
                .map_err(ImpostorDrawError::InvalidInstance)?;
        }
        if instances.is_empty() {
            return Ok(());
        }
        self.collect_retired();

        let frame_slot = ctx.frame_slot();
        let slot_index = frame_slot % self.slots.len();
        {
            let slot = &mut self.slots[slot_index];
            if slot.frame_id != ctx.frame_id() {
                slot.frame_id = ctx.frame_id();
                slot.instance_used = 0;
                slot.params_used = 0;
                slot.retired.clear();
            }
        }

        let (inst_offset, params_offset) = {
            let slot = &mut self.slots[slot_index];
            let inst_offset = align_up(slot.instance_used, self.instance_align);
            if inst_offset
                .checked_add(stride)
                .ok_or(ImpostorDrawError::TooManyInstances)?
                > slot.instance_capacity
            {
                let grown = self
                    .ctx
                    .create_host_storage_buffer("impostor instances", (inst_offset + stride) * 2);
                slot.retired
                    .push(std::mem::replace(&mut slot.instance_buffer, grown));
                slot.instance_capacity = (inst_offset + stride) * 2;
            }
            slot.instance_buffer
                .write_bytes(inst_offset, bytemuck::cast_slice(instances));
            slot.instance_used = inst_offset + stride;

            let params_entry = align_up(
                std::mem::size_of::<AssetParams>() as u64,
                self.uniform_align,
            );
            let params_offset = align_up(slot.params_used, self.uniform_align);
            if params_offset + params_entry > slot.params_capacity {
                let grown = self
                    .ctx
                    .create_host_buffer("impostor asset params", slot.params_capacity * 2);
                slot.retired
                    .push(std::mem::replace(&mut slot.params_buffer, grown));
                slot.params_capacity *= 2;
            }
            let params = asset_params(asset, self.grad_scale, screen_px, fov_tan, debug);
            slot.params_buffer
                .write_bytes(params_offset, bytemuck::bytes_of(&params));
            slot.params_used = params_offset + params_entry;
            (inst_offset, params_offset)
        };

        let cam_buf = ctx.scene_uniform_buffer();
        let lighting_buf = lighting.buffer(frame_slot);
        let enc = ctx.raw();
        enc.setRenderPipelineState(&self.pipeline);
        enc.setDepthStencilState(Some(&self.depth_stencil));
        enc.setFrontFacingWinding(MTLWinding::Clockwise);
        enc.setCullMode(MTLCullMode::Back);

        let slot = &self.slots[slot_index];
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(cam_buf), 0, 0);
            enc.setFragmentBuffer_offset_atIndex(Some(cam_buf), 0, 0);
            enc.setVertexBuffer_offset_atIndex(
                Some(&slot.instance_buffer.buffer),
                inst_offset as usize,
                1,
            );
            enc.setFragmentBuffer_offset_atIndex(
                Some(&slot.instance_buffer.buffer),
                inst_offset as usize,
                1,
            );
            enc.setVertexBuffer_offset_atIndex(
                Some(&slot.params_buffer.buffer),
                params_offset as usize,
                2,
            );
            enc.setFragmentBuffer_offset_atIndex(
                Some(&slot.params_buffer.buffer),
                params_offset as usize,
                2,
            );
            enc.setFragmentBuffer_offset_atIndex(Some(&asset.voxels.buffer), 0, 3);
            enc.setFragmentBuffer_offset_atIndex(Some(&asset.material_table.buffer), 0, 4);
            enc.setFragmentBuffer_offset_atIndex(Some(lighting_buf), 0, 5);

            enc.setFragmentTexture_atIndex(Some(&asset.albedo.texture), 0);
            enc.setFragmentTexture_atIndex(Some(&asset.normal_ao.texture), 1);
            enc.setFragmentTexture_atIndex(Some(&asset.metallic_rough.texture), 2);
            enc.setFragmentTexture_atIndex(Some(&asset.emissive.texture), 3);
            enc.setFragmentTexture_atIndex(Some(&asset.depth.texture), 4);
            enc.setFragmentTexture_atIndex(Some(&asset.material_maps[0].texture), 5);
            enc.setFragmentTexture_atIndex(Some(&asset.material_maps[1].texture), 6);
            enc.setFragmentTexture_atIndex(Some(&asset.material_maps[2].texture), 7);
            enc.setFragmentTexture_atIndex(Some(&asset.material_maps[3].texture), 8);
            enc.setFragmentTexture_atIndex(Some(&asset.material_maps[4].texture), 9);

            enc.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
            enc.setFragmentSamplerState_atIndex(Some(&self.material_sampler), 1);

            enc.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                36,
                instances.len(),
            );
        }

        self.draws += 1;
        self.instances += instances.len() as u64;
        Ok(())
    }

    pub fn retire_asset(&mut self, mut asset: ImpostorAsset) {
        let serial = self.ctx.latest_submit_serial();
        if self.ctx.is_serial_complete(serial) {
            asset.drop_without_wait = true;
            drop(asset);
        } else {
            self.retired_assets.push((asset, serial));
        }
    }

    pub fn collect_retired(&mut self) {
        let ctx = &self.ctx;
        self.retired_assets.retain_mut(|(asset, serial)| {
            if ctx.is_serial_complete(*serial) {
                asset.drop_without_wait = true;
                false
            } else {
                true
            }
        });
    }

    pub fn retired_asset_count(&self) -> usize {
        self.retired_assets.len()
    }

    pub fn buffer_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| s.instance_capacity as usize + s.params_capacity as usize)
            .sum()
    }
}

pub(crate) fn source_geometry_enabled(has_mask_materials: bool, voxel_resolution: u32) -> bool {
    !has_mask_materials && voxel_resolution > 0
}

fn asset_params(
    asset: &ImpostorAsset,
    grad: f32,
    screen_px: f32,
    fov_tan: f32,
    debug: AssetDebug,
) -> AssetParams {
    let bounds: ImpostorBounds = asset.bounds;
    let slab_mask: u32 = asset
        .slab_valid
        .iter()
        .enumerate()
        .map(|(i, v)| (*v as u32) << i)
        .sum();
    let mut params = AssetParams {
        box_min: bounds.min.extend(0.0).to_array(),
        box_max: bounds.max.extend(0.0).to_array(),
        params: [
            0.5,
            asset.resolution as f32,
            slab_mask as f32,
            debug.code() as f32,
        ],
        screen: [
            3.0 * bounds.extent().max_element() / asset.resolution as f32,
            fov_tan * 2.0 / screen_px.max(1.0),
            grad,
            asset.voxel_resolution as f32,
        ],
        material: [
            f32::from_bits(asset.triangle_base),
            asset.material_resolution as f32,
            asset.material_count as f32,
            if source_geometry_enabled(asset.has_mask_materials, asset.voxel_resolution) {
                1.0
            } else {
                0.0
            },
        ],
        ..bytemuck::Zeroable::zeroed()
    };
    for (f, view) in VIEWS.iter().enumerate() {
        let extent = bounds.extent();
        params.u_axes[f] = view.u.extend(0.0).to_array();
        params.v_axes[f] = view.v.extend(0.0).to_array();
        params.extents[f] = [
            projected_extent(extent, view.u),
            projected_extent(extent, view.v),
            projected_extent(extent, view.dir),
            0.0,
        ];
        params.views[f] = view.dir.extend(0.0).to_array();
    }
    params
}

fn make_sampler(
    ctx: &GpuContext,
    address: MTLSamplerAddressMode,
    max_aniso: f32,
) -> Retained<ProtocolObject<dyn MTLSamplerState>> {
    let desc = MTLSamplerDescriptor::new();
    desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
    desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
    desc.setMipFilter(MTLSamplerMipFilter::Linear);
    desc.setSAddressMode(address);
    desc.setTAddressMode(address);
    desc.setRAddressMode(address);
    desc.setMaxAnisotropy(max_aniso.max(1.0) as usize);
    desc.setLodMinClamp(0.0);
    desc.setLodMaxClamp(16.0);
    ctx.device
        .newSamplerStateWithDescriptor(&desc)
        .expect("impostor sampler")
}
