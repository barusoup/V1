//! The instanced card draw: one draw call per asset per frame.
//!
//! Instances stream through a per-frame-slot host storage buffer (same
//! ring/pool pattern as the model renderer's joint data), the vertex stage
//! expands each instance into a 12-triangle bounding-box shell, and the
//! fragment shader resolves geometry (source-triangle DDA for opaque
//! assets, visual-hull march otherwise) + projective shading, with exact
//! source-triangle material evaluation at subpixel range. Color blends (for
//! LoD crossfade); the velocity attachment stays unblended, matching the
//! model pipelines' contract.

use super::atlas::{AssetDebug, ImpostorAsset};
use super::{ImpostorBounds, VIEWS, projected_extent};
use crate::gfx::vulkan::{GpuContext, align_up};
use crate::gfx::{DrawCtx, GpuBuffer};
use crate::lighting::Lighting;
use crate::model::render::scene_set_layout;
use ash::vk;
use glam::Mat4;
use std::fmt;
use std::sync::Arc;

/// Validation failure for [`ImpostorInstance`]: the card shader supports
/// affine proper rotations with a positive uniform scale only. Use the mesh
/// when a selection needs a different transform — no engine override exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpostorInstanceError {
    /// Any matrix element is NaN or infinite.
    NonFinite,
    /// Last row is not (0,0,0,1) — projection or non-affine input.
    NonAffine,
    /// Axis lengths differ (non-uniform scale), an axis is zero, the basis
    /// is sheared, or the determinant is negative (reflection).
    UnsupportedScale,
    /// Alpha is not finite in `[0, 1]`.
    InvalidOpacity,
    /// `inv_model` is not finite or `model * inv_model` is not identity.
    InvalidInverse,
}

impl fmt::Display for ImpostorInstanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "instance transform is not finite"),
            Self::NonAffine => write!(f, "instance transform is not affine"),
            Self::UnsupportedScale => {
                write!(
                    f,
                    "instance transform needs non-uniform scale/shear/reflection — use the mesh"
                )
            }
            Self::InvalidOpacity => write!(f, "instance alpha must be finite in [0,1]"),
            Self::InvalidInverse => write!(f, "instance inverse is invalid"),
        }
    }
}

impl std::error::Error for ImpostorInstanceError {}

/// Check one 4x4 column-major matrix: finite, affine last row, orthogonal
/// basis with a positive uniform scale (rotation + uniform scale +
/// translation only).
fn check_transform(m: &[[f32; 4]; 4]) -> Result<(), ImpostorInstanceError> {
    if !m.iter().flatten().all(|v| v.is_finite()) {
        return Err(ImpostorInstanceError::NonFinite);
    }
    let row3 = [m[0][3], m[1][3], m[2][3], m[3][3]];
    if !row3
        .iter()
        .zip([0.0f32, 0.0, 0.0, 1.0])
        .all(|(a, b)| (a - b).abs() <= 1e-6)
    {
        return Err(ImpostorInstanceError::NonAffine);
    }
    let cols: [[f32; 3]; 3] = [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ];
    let len: [f32; 3] = cols.map(|c| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt());
    if len
        .iter()
        .any(|l| !l.is_finite() || !(1e-6..=1e6).contains(l))
    {
        return Err(ImpostorInstanceError::UnsupportedScale);
    }
    // Uniform scale within 1e-4 relative.
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
    // Positive determinant: reject reflections.
    let det = cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1])
        - cols[0][1] * (cols[1][0] * cols[2][2] - cols[1][2] * cols[2][0])
        + cols[0][2] * (cols[1][0] * cols[2][1] - cols[1][1] * cols[2][0]);
    if !det.is_finite() || det <= 0.0 {
        return Err(ImpostorInstanceError::UnsupportedScale);
    }
    Ok(())
}

/// One card instance: model + prev model for motion vectors, inverse model
/// for the object-space view ray, alpha for the crossfade weight.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImpostorInstance {
    pub model: [[f32; 4]; 4],
    pub prev_model: [[f32; 4]; 4],
    pub inv_model: [[f32; 4]; 4],
    pub misc: [f32; 4], // alpha, -, -, -
}

impl ImpostorInstance {
    /// Contract-checked constructor: affine proper rotation + positive
    /// uniform scale + translation, finite inverse, alpha in `[0, 1]`.
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

    /// Re-checks the contract on an instance (fields are public, so a raw
    /// struct can bypass `try_new`; `try_draw` calls this per instance).
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
        // model * inv ~= identity (column-major multiply, 1e-3 tolerance).
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

    /// Panicking constructor for the common contract-compliant case.
    pub fn new(model: Mat4, prev_model: Mat4, alpha: f32) -> Self {
        Self::try_new(model, prev_model, alpha).expect("impostor instance transform")
    }
}

/// Asset params uniform, mirroring `Asset` in impostor.wgsl.
#[repr(C)]
#[derive(Clone, Copy)]
struct AssetParams {
    box_min: [f32; 4],
    box_max: [f32; 4],
    params: [f32; 4],   // cutoff, res, slab_mask, debug
    screen: [f32; 4],   // tol, px-per-unit, grad scale, voxel res
    material: [f32; 4], // tri record base, material res, material count, -
    u_axes: [[f32; 4]; super::VIEW_COUNT],
    v_axes: [[f32; 4]; super::VIEW_COUNT],
    extents: [[f32; 4]; super::VIEW_COUNT],
    views: [[f32; 4]; super::VIEW_COUNT],
}

// All fields are f32/arrays of f32; the layout is dense with no padding.
unsafe impl bytemuck::Zeroable for AssetParams {}
unsafe impl bytemuck::Pod for AssetParams {}

/// Per-frame-slot draw state: instance + params host rings plus descriptor
/// pool chunks (2 sets per draw). Recycled on first use each frame — a
/// grown ring's replaced buffer lives in `retired` until the slot's frame
/// fence has recycled, since immutable draw sets still reference it.
struct Slot {
    frame_id: u64,
    instance_buffer: GpuBuffer,
    instance_capacity: u64,
    instance_used: u64,
    params_buffer: GpuBuffer,
    params_capacity: u64,
    params_used: u64,
    retired: Vec<GpuBuffer>,
    pools: Vec<vk::DescriptorPool>,
    pool_cursor: usize,
}

/// Draw-call failure from [`ImpostorRenderer::try_draw`]. `draw` panics on
/// these; they are developer-contract errors, not recoverable GPU states.
#[derive(Debug)]
pub enum ImpostorDrawError {
    /// An instance violates the transform contract.
    InvalidInstance(ImpostorInstanceError),
    /// View height / fov / grad_scale is non-positive or non-finite.
    InvalidView,
    /// The asset was baked on a different GPU context.
    WrongDevice,
    /// The batch cannot fit a u32 instance count or its byte stride.
    TooManyInstances,
    /// The instance stride exceeds `maxStorageBufferRange`.
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
            Self::StorageLimit => {
                write!(f, "instance batch exceeds maxStorageBufferRange")
            }
        }
    }
}

impl std::error::Error for ImpostorDrawError {}

/// One instanced draw per [`ImpostorAsset`]; all instances of that asset
/// share the call.
pub struct ImpostorRenderer {
    ctx: Arc<GpuContext>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsl_instances: vk::DescriptorSetLayout,
    dsl_atlas: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    material_sampler: vk::Sampler,
    instance_align: u64,
    uniform_align: u64,
    slots: Vec<Slot>,
    pool_sizes: [vk::DescriptorPoolSize; 4],
    /// Assets queued by [`Self::retire_asset`] with their submit serial;
    /// dropped (without an idle wait) once that submission completes.
    retired_assets: Vec<(ImpostorAsset, u64)>,
    draws: u64,
    instances: u64,
    /// Mip-gradient scale factor (0.5 = half-footprint); see impostor.wgsl.
    pub grad_scale: f32,
}

impl ImpostorRenderer {
    pub fn new(
        ctx: &Arc<GpuContext>,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
        lighting: &Lighting,
    ) -> Self {
        let dsl_instances = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
            unsafe {
                ctx.device().create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }
            .expect("impostor instance set layout")
        };
        let dsl_atlas = {
            let mut bindings = vec![
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX),
            ];
            for binding in 1..6u32 {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                );
            }
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(6)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(7)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
            for binding in 8..13u32 {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                );
            }
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(13)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(14)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
            unsafe {
                ctx.device().create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }
            .expect("impostor atlas set layout")
        };
        let set_layouts = [
            scene_set_layout(ctx),
            dsl_instances,
            dsl_atlas,
            lighting.set_layout(),
        ];
        let layout = unsafe {
            ctx.device().create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
        }
        .expect("impostor pipeline layout");
        let shader = ctx.compile_shader(
            &format!(
                "{}{}",
                crate::lighting::SHADER,
                include_str!("impostor.wgsl")
            ),
            "impostor",
        );
        let pipeline = create_impostor_pipeline(ctx, shader, layout, color_formats, depth_format);
        unsafe { ctx.device().destroy_shader_module(shader, None) };

        // Anisotropy only when the logical device enabled the feature.
        let aniso = unsafe {
            ctx.instance()
                .get_physical_device_features(ctx.physical_device())
        }
        .sampler_anisotropy
            == vk::TRUE;
        let max_aniso = unsafe {
            ctx.instance()
                .get_physical_device_properties(ctx.physical_device())
        }
        .limits
        .max_sampler_anisotropy
        .min(8.0);
        let sampler_info = |address: vk::SamplerAddressMode| {
            vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(address)
                .address_mode_v(address)
                .address_mode_w(address)
                .anisotropy_enable(aniso)
                .max_anisotropy(if aniso { max_aniso } else { 1.0 })
                .max_lod(vk::LOD_CLAMP_NONE)
        };
        let sampler = unsafe {
            ctx.device()
                .create_sampler(&sampler_info(vk::SamplerAddressMode::CLAMP_TO_EDGE), None)
        }
        .expect("impostor tile sampler");
        let material_sampler = unsafe {
            ctx.device()
                .create_sampler(&sampler_info(vk::SamplerAddressMode::REPEAT), None)
        }
        .expect("impostor material sampler");

        let limits = unsafe {
            ctx.instance()
                .get_physical_device_properties(ctx.physical_device())
                .limits
        };
        let instance_align = limits.min_storage_buffer_offset_alignment.max(64);
        let uniform_align = limits.min_uniform_buffer_offset_alignment.max(16);
        // 2 sets per draw: instance storage + asset (params + 10 images +
        // 2 samplers + voxel + material table).
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(64),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(192),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(640),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(128),
        ];
        let slots = (0..ctx.max_frames_in_flight())
            .map(|_| Slot {
                frame_id: u64::MAX,
                instance_buffer: ctx.create_host_storage_buffer("impostor instances", 1 << 20),
                instance_capacity: 1 << 20,
                instance_used: 0,
                params_buffer: ctx.create_host_buffer("impostor asset params", 1 << 16),
                params_capacity: 1 << 16,
                params_used: 0,
                retired: Vec::new(),
                pools: vec![new_draw_pool(ctx, &pool_sizes)],
                pool_cursor: 0,
            })
            .collect();

        Self {
            ctx: Arc::clone(ctx),
            pipeline,
            layout,
            dsl_instances,
            dsl_atlas,
            sampler,
            material_sampler,
            instance_align,
            uniform_align,
            slots,
            pool_sizes,
            retired_assets: Vec::new(),
            draws: 0,
            instances: 0,
            grad_scale: 0.35,
        }
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }

    pub fn instances_submitted(&self) -> u64 {
        self.instances
    }

    /// Draw all `instances` of `asset` in one instanced call.
    /// `screen_px` is render height in pixels and `fov_tan` is tan(fov_y/2),
    /// feeding the analytic mip gradient. `debug` selects the shader mode.
    ///
    /// Panics on contract violations; use [`Self::try_draw`] to get the
    /// [`ImpostorDrawError`] instead.
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
            .expect("impostor draw")
    }

    /// Validating version of [`Self::draw`]: every contract failure returns
    /// an [`ImpostorDrawError`] instead of panicking or producing invalid
    /// Vulkan state.
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
        // A zero batch is valid and touches nothing GPU-side.
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
                slot.pool_cursor = 0;
                for pool in &slot.pools {
                    unsafe {
                        self.ctx
                            .device()
                            .reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty())
                    }
                    .expect("impostor pool reset");
                }
            }
        }

        // Instance ring: bump allocation at the storage alignment, exact
        // range bound into a per-draw set — grown buffers are retired, not
        // copied (earlier draw sets keep their own valid slice).
        let (inst_buffer, params_buffer, inst_offset, params_offset) = {
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
                .write(inst_offset, bytemuck::cast_slice(instances));
            slot.instance_used = inst_offset + stride;

            // Params ring: one AssetParams entry per draw at uniform
            // alignment.
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
                .write(params_offset, bytemuck::bytes_of(&params));
            slot.params_used = params_offset + params_entry;
            (
                slot.instance_buffer.buffer,
                slot.params_buffer.buffer,
                inst_offset,
                params_offset,
            )
        };

        // Two fresh draw sets from the slot's pool chunks.
        let [instance_set, asset_set] = self.alloc_draw_sets(slot_index);
        self.write_draw_sets(
            instance_set,
            asset_set,
            inst_buffer,
            inst_offset,
            stride,
            params_buffer,
            params_offset,
            asset,
        );

        let (device, cmd) = ctx.raw();
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[
                    ctx.scene_descriptor_set(),
                    instance_set,
                    asset_set,
                    lighting.set(frame_slot),
                ],
                &[],
            );
            device.cmd_draw(cmd, 36, instances.len() as u32, 0, 0);
        }
        self.draws += 1;
        self.instances += instances.len() as u64;
        Ok(())
    }

    /// Two fresh sets (instance + asset) from the slot's pool chunks,
    /// growing a chunk on exhaustion.
    fn alloc_draw_sets(&mut self, slot_index: usize) -> [vk::DescriptorSet; 2] {
        let layouts = [self.dsl_instances, self.dsl_atlas];
        loop {
            let slot = &mut self.slots[slot_index];
            if slot.pool_cursor == slot.pools.len() {
                slot.pools.push(new_draw_pool(&self.ctx, &self.pool_sizes));
            }
            let pool = slot.pools[slot.pool_cursor];
            let result = unsafe {
                self.ctx.device().allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
            };
            match result {
                Ok(sets) => return [sets[0], sets[1]],
                Err(err) if crate::gfx::vulkan::pool_exhaustion_is_growable(err) => {
                    self.slots[slot_index].pool_cursor += 1;
                }
                Err(err) => panic!("impostor draw set allocation: {err}"),
            }
        }
    }

    /// One-time writes for a draw's instance + asset sets. The instance
    /// range is the exact batch stride (never WHOLE_SIZE, which would
    /// violate maxStorageBufferRange on the grown ring); the params buffer
    /// offset is this draw's uniform-aligned ring entry.
    fn write_draw_sets(
        &self,
        instance_set: vk::DescriptorSet,
        asset_set: vk::DescriptorSet,
        inst_buffer: vk::Buffer,
        inst_offset: u64,
        inst_range: u64,
        params_buffer: vk::Buffer,
        params_offset: u64,
        asset: &ImpostorAsset,
    ) {
        let inst_info = vk::DescriptorBufferInfo::default()
            .buffer(inst_buffer)
            .offset(inst_offset)
            .range(inst_range);
        let inst_write = vk::WriteDescriptorSet::default()
            .dst_set(instance_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&inst_info));

        let params_info = vk::DescriptorBufferInfo::default()
            .buffer(params_buffer)
            .offset(params_offset)
            .range(std::mem::size_of::<AssetParams>() as u64);
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(asset_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&params_info)),
        ];
        let image_infos: Vec<_> = [
            &asset.albedo,
            &asset.normal_ao,
            &asset.metallic_rough,
            &asset.emissive,
            &asset.depth,
        ]
        .iter()
        .map(|image| {
            vk::DescriptorImageInfo::default()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::GENERAL)
        })
        .collect();
        for (binding, info) in image_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(asset_set)
                    .dst_binding(1 + binding as u32)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(info)),
            );
        }
        let sampler_info = vk::DescriptorImageInfo::default().sampler(self.sampler);
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(asset_set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&sampler_info)),
        );
        let voxel_info = vk::DescriptorBufferInfo::default()
            .buffer(asset.voxels.buffer)
            .offset(0)
            .range(asset.voxels.size);
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(asset_set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&voxel_info)),
        );
        let material_infos: Vec<_> = asset
            .material_maps
            .iter()
            .map(|image| {
                vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();
        for (binding, info) in material_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(asset_set)
                    .dst_binding(8 + binding as u32)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(info)),
            );
        }
        let material_sampler_info =
            vk::DescriptorImageInfo::default().sampler(self.material_sampler);
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(asset_set)
                .dst_binding(13)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&material_sampler_info)),
        );
        let material_info = vk::DescriptorBufferInfo::default()
            .buffer(asset.material_table.buffer)
            .offset(0)
            .range(asset.material_table.size);
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(asset_set)
                .dst_binding(14)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&material_info)),
        );
        unsafe {
            self.ctx.device().update_descriptor_sets(&[inst_write], &[]);
            self.ctx.device().update_descriptor_sets(&writes, &[]);
        }
    }

    /// Queue `asset` for destruction once the submissions recorded so far
    /// complete. Must be called AFTER frame submission (outside the scene
    /// callback — no pending unsubmitted commands may reference the asset)
    /// and the asset must belong to this renderer's GPU context. This is the
    /// streaming unload path: `Drop` alone waits for device idle.
    pub fn retire_asset(&mut self, mut asset: ImpostorAsset) {
        assert!(
            Arc::ptr_eq(&asset.ctx, &self.ctx),
            "impostor asset belongs to a different GPU context"
        );
        asset.drop_without_wait = false;
        let serial = self.ctx.submitted_serial();
        self.retired_assets.push((asset, serial));
    }

    /// Drop retired assets whose submit serial has completed (also called
    /// at the start of every draw).
    pub fn collect_retired(&mut self) {
        let ctx = Arc::clone(&self.ctx);
        self.retired_assets.retain_mut(|(asset, serial)| {
            if ctx.submission_complete(*serial) {
                asset.drop_without_wait = true;
                false
            } else {
                true
            }
        });
    }

    /// Assets still queued by [`Self::retire_asset`] (diagnostics).
    pub fn retired_asset_count(&self) -> usize {
        self.retired_assets.len()
    }

    /// Bytes held by the instance/params rings and their retired buffers —
    /// memory observability only, not an asset-size accounting.
    pub fn buffer_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| {
                (s.instance_capacity + s.params_capacity) as usize
                    + s.retired.iter().map(|b| b.size as usize).sum::<usize>()
            })
            .sum()
    }
}

impl Drop for ImpostorRenderer {
    fn drop(&mut self) {
        unsafe {
            let device = self.ctx.device();
            // Hold the submission lock across the idle wait so a streamer
            // thread can't queue new work mid-teardown.
            let _queue = self.ctx.queue_lock.lock().unwrap();
            let _ = device.device_wait_idle();
            for (mut asset, _) in self.retired_assets.drain(..) {
                asset.drop_without_wait = true;
                drop(asset);
            }
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.dsl_instances, None);
            device.destroy_descriptor_set_layout(self.dsl_atlas, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.material_sampler, None);
            for slot in &self.slots {
                for pool in &slot.pools {
                    device.destroy_descriptor_pool(*pool, None);
                }
            }
        }
    }
}

/// One pool chunk: 128 sets = 64 draws (2 sets each).
fn new_draw_pool(ctx: &GpuContext, pool_sizes: &[vk::DescriptorPoolSize]) -> vk::DescriptorPool {
    unsafe {
        ctx.device().create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(128)
                .pool_sizes(pool_sizes),
            None,
        )
    }
    .expect("impostor draw pool")
}

/// Whether the authoritative triangle-DDA path may replace the visual hull:
/// opaque-only materials and a baked voxel grid are both required. MASK
/// materials keep the hull because cutout coverage is capture-driven.
fn source_geometry_enabled(has_mask_materials: bool, voxel_resolution: u32) -> bool {
    !has_mask_materials && voxel_resolution > 0
}

/// Fill the `Asset` uniform from the asset + frame parameters.
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
        // Visibility tolerance ~3 tile texels along the axis normal, plus
        // the pixel-footprint-per-unit term for mip gradients.
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
            // w: geometry fast-path flag — 1.0 = the triangle DDA owns
            // coverage/depth/motion; 0.0 = legacy visual hull (MASK or no
            // voxel grid).
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

fn create_impostor_pipeline(
    ctx: &GpuContext,
    shader: vk::ShaderModule,
    layout: vk::PipelineLayout,
    color_formats: &[vk::Format],
    depth_format: vk::Format,
) -> vk::Pipeline {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(c"vs_main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader)
            .name(c"fs_main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        // Entry shell faces only: the outward-wound box under the engine's
        // negative-height viewport — one hull fragment per covered pixel.
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
    // Colour blends for the crossfade; velocity must not blend (same contract
    // as the model pipelines' blended variants).
    let blend_attachments: Vec<_> = color_formats
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA);
            if index == 0 {
                attachment
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
            } else {
                attachment.blend_enable(false)
            }
        })
        .collect();
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = color_formats.to_vec();
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_state)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    unsafe {
        ctx.device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, err)| err)
            .expect("create impostor pipeline")[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn instance_layout_matches_wgsl() {
        // mat4 x3 + vec4 = 3*64 + 16.
        assert_eq!(std::mem::size_of::<ImpostorInstance>(), 208);
        // 2 box + params + screen + material + 4 view arrays of 26 vec4.
        assert_eq!(
            std::mem::size_of::<AssetParams>(),
            (5 + 4 * crate::impostor::VIEW_COUNT) * 16
        );
    }

    #[test]
    fn composed_shader_validates() {
        let source = format!(
            "{}{}",
            crate::lighting::SHADER,
            include_str!("impostor.wgsl")
        );
        let module = naga::front::wgsl::parse_str(&source).expect("impostor shader parses");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("impostor shader validates");
    }

    #[test]
    fn source_geometry_flag_matrix() {
        assert!(source_geometry_enabled(false, 128));
        assert!(!source_geometry_enabled(true, 128));
        assert!(!source_geometry_enabled(false, 0));
        assert!(!source_geometry_enabled(true, 0));
    }

    // ---- instance contract + ring/pool math ------------------------------

    fn rot_z(angle: f32) -> Mat4 {
        Mat4::from_rotation_z(angle)
    }

    #[test]
    fn instance_try_new_valid_contracts() {
        assert!(ImpostorInstance::try_new(Mat4::IDENTITY, Mat4::IDENTITY, 1.0).is_ok());
        for s in [0.001f32, 1.0, 1000.0] {
            let m = Mat4::from_scale(Vec3::splat(s)) * rot_z(0.7);
            let p = m * Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
            let inst = ImpostorInstance::try_new(m, p, 0.5).expect("uniform scale + rot");
            inst.validate().expect("raw fields pass");
        }
        assert!(ImpostorInstance::try_new(Mat4::IDENTITY, Mat4::IDENTITY, 0.0).is_ok());
    }

    #[test]
    fn instance_try_new_rejections() {
        use ImpostorInstanceError as E;
        let id = Mat4::IDENTITY;
        // NaN in model.
        let mut nan = [[0.0f32; 4]; 4];
        nan[0][0] = f32::NAN;
        nan[1][1] = 1.0;
        nan[2][2] = 1.0;
        nan[3][3] = 1.0;
        let inst = ImpostorInstance {
            model: nan,
            prev_model: id.to_cols_array_2d(),
            inv_model: id.to_cols_array_2d(),
            misc: [1.0, 0.0, 0.0, 0.0],
        };
        assert_eq!(inst.validate(), Err(E::NonFinite));
        // Zero scale axis.
        let zero = Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0));
        assert_eq!(
            ImpostorInstance::try_new(zero, id, 1.0).unwrap_err(),
            E::UnsupportedScale
        );
        // Singular (zero) -> inverse non-finite path is covered above; here a
        // near-singular but invertible non-uniform is still UnsupportedScale.
        let nonuniform = Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
        assert_eq!(
            ImpostorInstance::try_new(nonuniform, id, 1.0).unwrap_err(),
            E::UnsupportedScale
        );
        // Shear.
        let mut shear = id.to_cols_array_2d();
        shear[1][0] = 0.5;
        let inst = ImpostorInstance {
            model: shear,
            prev_model: id.to_cols_array_2d(),
            inv_model: Mat4::from_cols_array_2d(&shear)
                .inverse()
                .to_cols_array_2d(),
            misc: [1.0, 0.0, 0.0, 0.0],
        };
        assert_eq!(inst.validate(), Err(E::UnsupportedScale));
        // Reflection.
        let refl = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
        assert_eq!(
            ImpostorInstance::try_new(refl, id, 1.0).unwrap_err(),
            E::UnsupportedScale
        );
        // Non-affine last row (row index 3 of a column-major matrix).
        let mut nonaff = id.to_cols_array_2d();
        nonaff[0][3] = 0.1;
        let inst = ImpostorInstance {
            model: nonaff,
            prev_model: id.to_cols_array_2d(),
            inv_model: id.to_cols_array_2d(),
            misc: [1.0, 0.0, 0.0, 0.0],
        };
        assert_eq!(inst.validate(), Err(E::NonAffine));
        // Opacity outside [0,1] / NaN.
        for a in [-0.5f32, 1.5, f32::NAN] {
            let mut inst = ImpostorInstance::try_new(id, id, 1.0).unwrap();
            inst.misc[0] = a;
            assert_eq!(inst.validate(), Err(E::InvalidOpacity));
        }
        // prev_model invalid.
        let prev_bad = Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0));
        assert_eq!(
            ImpostorInstance::try_new(id, prev_bad, 1.0).unwrap_err(),
            E::UnsupportedScale
        );
        // Raw struct with a wrong inverse.
        let mut inst = ImpostorInstance::try_new(id, id, 1.0).unwrap();
        inst.inv_model[0][0] = 2.0;
        assert_eq!(inst.validate(), Err(E::InvalidInverse));
        // Finite but extreme: the model*inv product overflows to non-finite
        // — must still be InvalidInverse, never a false-pass on NaN.
        let mut inst = ImpostorInstance::try_new(id, id, 1.0).unwrap();
        inst.inv_model[0][0] = f32::MAX;
        inst.model = Mat4::from_scale(Vec3::splat(2.0)).to_cols_array_2d();
        assert_eq!(inst.validate(), Err(E::InvalidInverse));
    }

    #[test]
    fn triangle_base_packing_roundtrip() {
        // material.x carries the raw u32 word, not a float — offsets beyond
        // 2^24 must survive.
        for base in [0u32, 1, 17, 1 << 20, (1 << 24) + 7, (1 << 24) * 3 + 1] {
            let bits = f32::from_bits(base);
            assert_eq!(bits.to_bits(), base);
        }
        let wgsl = include_str!("impostor.wgsl");
        assert!(
            wgsl.contains("bitcast<u32>(asset.material.x)"),
            "record base must be bitcast, not float-rounded"
        );
        assert!(
            !wgsl.contains("asset.material.x + 0.5"),
            "float-rounded record base is a precision bug above 2^24"
        );
    }

    // Pure ring-math helpers mirrored from try_draw so the growth/alignment
    // contract is testable without a GPU.
    fn bump_offset(used: u64, align: u64) -> u64 {
        align_up(used, align)
    }

    #[test]
    fn ring_growth_math() {
        // Small capacity: grow retains the old buffer (capacity doubles the
        // required end, matching try_draw).
        let align = 256u64;
        let mut used = 0u64;
        let cap = 1024u64;
        for i in 0..4 {
            let off = bump_offset(used, align);
            assert_eq!(off % align, 0);
            let stride = 208u64 * 3;
            let end = off + stride;
            let new_cap = if end > cap { end * 2 } else { cap };
            assert!(new_cap >= end);
            used = end;
            let _ = i;
        }
        // Checked math: overflowing stride errors instead of wrapping.
        let n = u64::MAX / 208 + 1;
        assert!(208u64.checked_mul(n).is_none());
    }
}
