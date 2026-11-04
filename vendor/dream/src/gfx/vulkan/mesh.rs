use super::{GpuBuffer, GpuContext};
use ash::vk;
use std::sync::Arc;

/// Device-local vertex + index storage for geometry the CPU rewrites
/// every frame (ambient fields, particles, procedural grids).
///
/// One reused staging buffer feeds the device-local copy, so an update
/// never allocates: buffers only ever grow. Bind the pair, then
/// `draw_indexed`.
pub struct DynamicBuffers {
    ctx: Arc<GpuContext>,
    vertices: GpuBuffer,
    indices: GpuBuffer,
    staging: GpuBuffer,
    vertex_capacity: u64,
    index_capacity: u64,
}

impl DynamicBuffers {
    pub fn new(ctx: &Arc<GpuContext>, label: &str, vertex_bytes: u64, index_bytes: u64) -> Self {
        let usage = vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST;
        Self {
            vertices: ctx.create_gpu_buffer(&format!("{label} vertices"), vertex_bytes, usage),
            indices: ctx.create_gpu_buffer(&format!("{label} indices"), index_bytes, usage),
            staging: ctx.create_transfer_buffer(
                &format!("{label} staging"),
                vertex_bytes.max(index_bytes).max(4),
            ),
            vertex_capacity: vertex_bytes,
            index_capacity: index_bytes,
            ctx: Arc::clone(ctx),
        }
    }

    /// Upload this frame's geometry, growing the buffers when needed.
    pub fn update(&mut self, vertices: &[u8], indices: &[u8]) {
        let vertex_bytes = vertices.len().max(4) as u64;
        let index_bytes = indices.len().max(4) as u64;
        if vertex_bytes > self.vertex_capacity {
            self.vertex_capacity = vertex_bytes * 2;
            self.vertices = self.ctx.create_gpu_buffer(
                "dynamic vertices",
                self.vertex_capacity,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
        }
        if index_bytes > self.index_capacity {
            self.index_capacity = index_bytes * 2;
            self.indices = self.ctx.create_gpu_buffer(
                "dynamic indices",
                self.index_capacity,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
        }
        if vertex_bytes + index_bytes > self.staging.size {
            self.staging = self
                .ctx
                .create_transfer_buffer("dynamic staging", vertex_bytes + index_bytes);
        }
        if !vertices.is_empty() {
            self.staging.write(0, vertices);
        }
        if !indices.is_empty() {
            self.staging.write(vertices.len().max(4) as u64, indices);
        }
        let (vertex_buffer, index_buffer, staging) = (
            self.vertices.buffer,
            self.indices.buffer,
            self.staging.buffer,
        );
        let index_offset = vertices.len().max(4) as u64;
        let device = self.ctx.device().clone();
        self.ctx.one_shot(|cmd| unsafe {
            if !vertices.is_empty() {
                let region = vk::BufferCopy::default().size(vertices.len() as u64);
                device.cmd_copy_buffer(cmd, staging, vertex_buffer, &[region]);
            }
            if !indices.is_empty() {
                let region = vk::BufferCopy::default()
                    .src_offset(index_offset)
                    .size(indices.len() as u64);
                device.cmd_copy_buffer(cmd, staging, index_buffer, &[region]);
            }
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        });
    }

    pub fn vertices(&self) -> &GpuBuffer {
        &self.vertices
    }

    pub fn indices(&self) -> &GpuBuffer {
        &self.indices
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }
}

/// A packed-quad storage buffer bound for vertex pulling. Owns its
/// descriptor layout, pool and set so its teardown stays self-contained.
pub struct PulledMesh {
    ctx: Arc<GpuContext>,
    quads: Arc<GpuBuffer>,
    quad_count: u32,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl PulledMesh {
    pub(super) fn new(ctx: &Arc<GpuContext>, quads: Arc<GpuBuffer>, quad_count: u32) -> Self {
        assert!(
            Arc::ptr_eq(&quads.ctx, ctx),
            "pulled mesh buffer belongs to a different GpuContext"
        );
        assert!(
            quads.size >= quad_count as u64 * 8,
            "pulled mesh buffer holds fewer than {quad_count} quads"
        );
        let max_range = unsafe {
            ctx.instance()
                .get_physical_device_properties(ctx.physical_device())
                .limits
                .max_storage_buffer_range
        };
        assert!(
            quad_count as u64 * 8 <= max_range as u64,
            "pulled mesh range {} exceeds max_storage_buffer_range {max_range}",
            quad_count as u64 * 8
        );
        // `vertexIndex` expansion means quad_count * 6 must stay in u32.
        assert!(
            quad_count as u64 <= u32::MAX as u64 / 6,
            "pulled mesh of {quad_count} quads overflows u32 vertex indices"
        );
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let layout = unsafe {
            ctx.device().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[binding]),
                None,
            )
        }
        .expect("pulled mesh descriptor set layout");
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1);
        let pool = unsafe {
            ctx.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[pool_size]),
                None,
            )
        }
        .expect("pulled mesh descriptor pool");
        let set = unsafe {
            ctx.device().allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[layout]),
            )
        }
        .expect("pulled mesh descriptor set")[0];
        let info = vk::DescriptorBufferInfo::default()
            .buffer(quads.buffer)
            .offset(0)
            // Vulkan forbids a zero descriptor range; the buffer itself is
            // at least 4 bytes, so clamp rather than special-casing.
            .range((quad_count as u64 * 8).max(4));
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&info));
        unsafe { ctx.device().update_descriptor_sets(&[write], &[]) };
        Self {
            ctx: Arc::clone(ctx),
            quads,
            quad_count,
            layout,
            pool,
            set,
        }
    }

    /// Quads the descriptor range covers; draws must stay within this count.
    pub fn quad_count(&self) -> u32 {
        self.quad_count
    }

    /// The packed-quad storage buffer itself.
    pub fn quads(&self) -> &Arc<GpuBuffer> {
        &self.quads
    }

    /// Set-1 descriptor binding the quad storage buffer.
    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// Layout the set was allocated from; matches the pulling pipeline's set 1.
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for PulledMesh {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device().destroy_descriptor_pool(self.pool, None);
            self.ctx
                .device()
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
