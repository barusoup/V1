use super::metal::shader::{PUSH_CONSTANT_BUFFER_SLOT, VERTEX_BUFFER_SLOT};
use super::{DrawCtx, GpuBuffer, PulledMesh};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDepthStencilState, MTLIndexType, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPipelineState,
};
use std::ptr::NonNull;

impl<'a> DrawCtx<'a> {
    #[inline]
    pub(crate) fn ensure_started(&self) {
        if let Some(starter) = self.starter.borrow_mut().take() {
            starter();
        }
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLRenderCommandEncoder> {
        self.ensure_started();
        self.encoder
    }

    pub fn frame_slot(&self) -> usize {
        self.frame_slot
    }

    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    pub fn scene_uniform_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.scene_uniform
    }

    pub fn scene_depth_stencil(&self) -> &ProtocolObject<dyn MTLDepthStencilState> {
        self.scene_depth_stencil
    }

    pub fn bind_vertex_buffer(&mut self, buffer: &GpuBuffer) {
        self.bind_vertex_buffer_at(buffer, 0);
    }

    pub fn bind_index_buffer(&mut self, buffer: &GpuBuffer) {
        self.bind_index_buffer_at(buffer, 0);
    }

    pub fn draw_indexed(&mut self, index_count: u32) {
        self.draw_indexed_range(index_count, 0, 0);
    }

    pub fn bind_vertex_buffer_at(&mut self, buffer: &GpuBuffer, byte_offset: u64) {
        self.ensure_started();
        self.encoder.setRenderPipelineState(self.scene_pipeline);
        self.encoder
            .setDepthStencilState(Some(self.scene_depth_stencil));
        unsafe {
            self.encoder
                .setVertexBuffer_offset_atIndex(Some(self.scene_uniform), 0, 0);
            self.encoder
                .setFragmentBuffer_offset_atIndex(Some(self.scene_uniform), 0, 0);
            self.encoder.setVertexBuffer_offset_atIndex(
                Some(&buffer.buffer),
                byte_offset as usize,
                VERTEX_BUFFER_SLOT as usize,
            );
        }
    }

    pub fn bind_index_buffer_at(&mut self, buffer: &GpuBuffer, byte_offset: u64) {
        self.ensure_started();
        self.index_buffer = Some((buffer.buffer.clone(), byte_offset));
    }

    pub fn draw_indexed_range(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        self.ensure_started();
        let (idx_buf, base_off) = self
            .index_buffer
            .as_ref()
            .expect("bind_index_buffer must be called before draw_indexed");
        let offset = *base_off as usize + first_index as usize * std::mem::size_of::<u32>();
        unsafe {
            self.encoder
                .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    MTLPrimitiveType::Triangle,
                    index_count as usize,
                    MTLIndexType::UInt32,
                    idx_buf,
                    offset,
                    1,
                    vertex_offset as isize,
                    0,
                );
        }
    }

    fn bind_pulling(&mut self, mesh: &PulledMesh, origin: [f32; 3]) {
        self.ensure_started();
        let pipeline: &ProtocolObject<dyn MTLRenderPipelineState> = self
            .pulling
            .expect("draw_pulled requires Renderer::new_with_pulling");
        self.encoder.setRenderPipelineState(pipeline);
        self.encoder
            .setDepthStencilState(Some(self.scene_depth_stencil));
        let constants = [origin[0], origin[1], origin[2], 0.0f32];
        unsafe {
            self.encoder
                .setVertexBuffer_offset_atIndex(Some(self.scene_uniform), 0, 0);
            self.encoder
                .setFragmentBuffer_offset_atIndex(Some(self.scene_uniform), 0, 0);
            self.encoder
                .setVertexBuffer_offset_atIndex(Some(&mesh.buffer().buffer), 0, 1);
            let bytes = bytemuck::bytes_of(&constants);
            self.encoder.setVertexBytes_length_atIndex(
                NonNull::new_unchecked(bytes.as_ptr() as *mut _),
                bytes.len(),
                PUSH_CONSTANT_BUFFER_SLOT as usize,
            );
        }
    }

    pub fn draw_pulled(&mut self, mesh: &PulledMesh, origin: [f32; 3]) {
        self.draw_pulled_range(mesh, origin, 0, mesh.quad_count());
    }

    pub fn draw_pulled_range(
        &mut self,
        mesh: &PulledMesh,
        origin: [f32; 3],
        first_quad: u32,
        quad_count: u32,
    ) {
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
            self.encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                (first_quad * 6) as usize,
                (quad_count * 6) as usize,
            );
        }
    }

    pub fn draw_pulled_indirect(
        &mut self,
        mesh: &PulledMesh,
        origin: [f32; 3],
        indirect: &GpuBuffer,
        byte_offset: u64,
    ) {
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
            self.encoder
                .drawPrimitives_indirectBuffer_indirectBufferOffset(
                    MTLPrimitiveType::Triangle,
                    &indirect.buffer,
                    byte_offset as usize,
                );
        }
    }
}
