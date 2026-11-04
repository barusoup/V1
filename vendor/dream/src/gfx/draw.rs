use super::{DrawCtx, GpuBuffer, PulledMesh};
use ash::vk;

impl<'a> DrawCtx<'a> {
    #[inline]
    pub(super) fn ensure_started(&self) {
        if let Some(starter) = self.starter.borrow_mut().take() {
            starter();
        }
    }

    /// Raw device + command buffer access: the escape hatch for custom
    /// pipelines (models, debug renderers) drawn inside the scene pass.
    /// Rebind pipeline state before drawing — the renderer binds its own
    /// state around each of its helpers.
    pub fn raw(&self) -> (&'a ash::Device, vk::CommandBuffer) {
        self.ensure_started();
        (self.device, self.cmd)
    }

    /// Frame slot being recorded. Host-visible per-frame resources (uniform
    /// rings, joint palettes) must be written to this slot only: its fence
    /// recycled before this frame started recording.
    pub fn frame_slot(&self) -> usize {
        self.frame_slot
    }

    /// Monotonic identifier of the frame being recorded. Unlike
    /// [`Self::frame_slot`], which repeats every `max_frames_in_flight`
    /// frames, this never repeats, so per-slot resources can tell "first use
    /// after the slot's fence recycled" apart from "second draw of the same
    /// frame" without comparing against a previous slot value.
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    /// The engine camera set (three matrices, `crate::temporal::CameraMatrices`),
    /// bindable at set 0 of any pipeline whose set-0 layout is one uniform
    /// buffer visible to the vertex stage.
    pub fn scene_descriptor_set(&self) -> vk::DescriptorSet {
        self.scene_set
    }

    /// Bind the indexed pipeline, the camera set, and `buffer` as vertex
    /// input. The pipeline/set rebind keeps this correct after pulled draws.
    pub fn bind_vertex_buffer(&mut self, buffer: &GpuBuffer) {
        self.bind_vertex_buffer_at(buffer, 0);
    }

    pub fn bind_index_buffer(&mut self, buffer: &GpuBuffer) {
        self.ensure_started();
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.cmd, buffer.buffer, 0, vk::IndexType::UINT32);
        }
    }

    pub fn draw_indexed(&mut self, index_count: u32) {
        self.ensure_started();
        unsafe {
            self.device
                .cmd_draw_indexed(self.cmd, index_count, 1, 0, 0, 0);
        }
    }

    /// Indexed draw inside shared arena banks: `vertices`/`indices` are big
    /// buffers, the span's byte offsets select where within them this tile
    /// lives. `vertex_offset` is a vertex index (elements, not bytes).
    pub fn bind_vertex_buffer_at(&mut self, buffer: &GpuBuffer, byte_offset: u64) {
        self.ensure_started();
        unsafe {
            self.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.scene_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.scene_layout,
                0,
                &[self.scene_set],
                &[],
            );
            self.device
                .cmd_bind_vertex_buffers(self.cmd, 0, &[buffer.buffer], &[byte_offset]);
        }
    }

    pub fn bind_index_buffer_at(&mut self, buffer: &GpuBuffer, byte_offset: u64) {
        self.ensure_started();
        unsafe {
            self.device.cmd_bind_index_buffer(
                self.cmd,
                buffer.buffer,
                byte_offset,
                vk::IndexType::UINT32,
            );
        }
    }

    /// `first_index` counts indices inside the bound index buffer;
    /// `vertex_offset` shifts indices into the vertex span (elements).
    pub fn draw_indexed_range(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        self.ensure_started();
        unsafe {
            self.device
                .cmd_draw_indexed(self.cmd, index_count, 1, first_index, vertex_offset, 0);
        }
    }

    /// Bind the vertex-pulling pipeline with the camera set at 0 and the
    /// mesh's quad set at 1, then push the mesh's world origin. Set 0 has
    /// the same layout in both pipelines; rebinding anyway keeps the state
    /// correct across pipeline switches and differing push-constant ranges.
    fn bind_pulling(&mut self, mesh: &PulledMesh, origin: [f32; 3]) {
        self.ensure_started();
        let Some((pipeline, layout)) = self.pulling else {
            panic!("draw_pulled requires Renderer::new_with_pulling");
        };
        unsafe {
            self.device
                .cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[self.scene_set, mesh.descriptor_set()],
                &[],
            );
            let constants = [origin[0], origin[1], origin[2], 0.0];
            self.device.cmd_push_constants(
                self.cmd,
                layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&constants),
            );
        }
    }

    /// Draw every quad of `mesh` expanded to six vertices each.
    pub fn draw_pulled(&mut self, mesh: &PulledMesh, origin: [f32; 3]) {
        self.draw_pulled_range(mesh, origin, 0, mesh.quad_count());
    }

    /// Draw `quad_count` quads starting at `first_quad`.
    pub fn draw_pulled_range(
        &mut self,
        mesh: &PulledMesh,
        origin: [f32; 3],
        first_quad: u32,
        quad_count: u32,
    ) {
        // Out-of-range stays invalid even when nothing would be drawn.
        assert!(
            first_quad <= mesh.quad_count() && quad_count <= mesh.quad_count() - first_quad,
            "pulled range {first_quad}..+{quad_count} exceeds {} quads",
            mesh.quad_count()
        );
        if quad_count == 0 {
            return;
        }
        self.bind_pulling(mesh, origin);
        unsafe {
            self.device
                .cmd_draw(self.cmd, quad_count * 6, 1, first_quad * 6, 0);
        }
    }

    /// Draw quads via `vkCmdDrawIndirect`: `indirect` holds a
    /// `VkDrawIndirectCommand` (vertexCount, instanceCount, firstVertex,
    /// firstInstance) at `byte_offset`.
    pub fn draw_pulled_indirect(
        &mut self,
        mesh: &PulledMesh,
        origin: [f32; 3],
        indirect: &GpuBuffer,
        byte_offset: u64,
    ) {
        // vkCmdDrawIndirect requires 4-byte-aligned offsets into a buffer
        // that still holds the whole command struct.
        assert!(
            byte_offset.is_multiple_of(4)
                && byte_offset
                    .checked_add(16)
                    .is_some_and(|end| end <= indirect.size),
            "indirect draw args at {byte_offset} exceed {} byte buffer",
            indirect.size
        );
        self.bind_pulling(mesh, origin);
        unsafe {
            self.device
                .cmd_draw_indirect(self.cmd, indirect.buffer, byte_offset, 1, 16);
        }
    }
}
