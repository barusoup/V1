use super::{GpuBuffer, GpuContext};
use ash::vk;
use std::sync::Arc;

/// Host-visible vertex + index ring for geometry the CPU rewrites every frame
/// (ambient fields, particles, procedural grids).
///
/// Uses CPU-to-GPU host-visible buffers tracked by GPU submission serials, so
/// updating never stalls on `one_shot` or allocates temporary staging buffers.
/// Only grows extra ring slots (up to `max_frames_in_flight + 1`) when an
/// earlier slot is still in flight on the GPU.
struct DynamicSlot {
    vertices: GpuBuffer,
    indices: GpuBuffer,
    vertex_capacity: u64,
    index_capacity: u64,
    last_serial: u64,
}

pub struct DynamicBuffers {
    ctx: Arc<GpuContext>,
    label: String,
    slots: Vec<DynamicSlot>,
    active: usize,
}

impl DynamicBuffers {
    pub fn new(ctx: &Arc<GpuContext>, label: &str, vertex_bytes: u64, index_bytes: u64) -> Self {
        let vertex_capacity = vertex_bytes.max(4);
        let index_capacity = index_bytes.max(4);
        let initial = DynamicSlot {
            vertices: ctx.create_host_vertex_buffer(&format!("{label} vertices"), vertex_capacity),
            indices: ctx.create_host_index_buffer(&format!("{label} indices"), index_capacity),
            vertex_capacity,
            index_capacity,
            last_serial: 0,
        };
        Self {
            ctx: Arc::clone(ctx),
            label: label.to_owned(),
            slots: vec![initial],
            active: 0,
        }
    }

    /// Upload this frame's geometry, reusing an idle slot or growing the ring when needed.
    pub fn update(&mut self, vertices: &[u8], indices: &[u8]) {
        let vertex_bytes = (vertices.len() as u64).max(4);
        let index_bytes = (indices.len() as u64).max(4);
        let completed = self.ctx.completed_serial();
        let next_serial = self.ctx.submitted_serial().saturating_add(1);
        let max_slots = self.ctx.max_frames_in_flight() + 1;

        let chosen = if self.slots[self.active].last_serial == 0
            || self.slots[self.active].last_serial <= completed
        {
            self.active
        } else if let Some(idx) = self
            .slots
            .iter()
            .position(|slot| slot.last_serial <= completed)
        {
            idx
        } else if self.slots.len() < max_slots {
            let vcap = vertex_bytes.max(self.slots[0].vertex_capacity);
            let icap = index_bytes.max(self.slots[0].index_capacity);
            self.slots.push(DynamicSlot {
                vertices: self
                    .ctx
                    .create_host_vertex_buffer(&format!("{} vertices", self.label), vcap),
                indices: self
                    .ctx
                    .create_host_index_buffer(&format!("{} indices", self.label), icap),
                vertex_capacity: vcap,
                index_capacity: icap,
                last_serial: 0,
            });
            self.slots.len() - 1
        } else {
            let idx = self
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, slot)| slot.last_serial)
                .map_or(0, |(i, _)| i);
            let submitted = self.ctx.submitted_serial();
            if self.slots[idx].last_serial <= submitted && submitted > completed {
                unsafe { self.ctx.device().device_wait_idle() }
                    .expect("wait idle for dynamic buffer slot");
                self.ctx.confirm_completed(submitted);
                self.ctx.collect_retired_buffers();
            }
            idx
        };

        let slot = &mut self.slots[chosen];
        if vertex_bytes > slot.vertex_capacity {
            slot.vertex_capacity = vertex_bytes * 2;
            slot.vertices = self.ctx.create_host_vertex_buffer(
                &format!("{} vertices", self.label),
                slot.vertex_capacity,
            );
        }
        if index_bytes > slot.index_capacity {
            slot.index_capacity = index_bytes * 2;
            slot.indices = self
                .ctx
                .create_host_index_buffer(&format!("{} indices", self.label), slot.index_capacity);
        }
        if !vertices.is_empty() {
            slot.vertices.write(0, vertices);
        }
        if !indices.is_empty() {
            slot.indices.write(0, indices);
        }
        slot.last_serial = next_serial;
        self.active = chosen;
    }

    pub fn vertices(&self) -> &GpuBuffer {
        &self.slots[self.active].vertices
    }

    pub fn indices(&self) -> &GpuBuffer {
        &self.slots[self.active].indices
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
