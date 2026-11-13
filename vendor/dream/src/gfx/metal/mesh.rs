use super::{GpuBuffer, GpuContext};
use std::sync::Arc;

/// Dynamic vertex/index buffer pair for CPU-streamed geometry.
/// On Apple Silicon unified memory, writes go directly into `StorageModeShared`
/// buffers with zero staging copies.
pub struct DynamicBuffers {
    ctx: Arc<GpuContext>,
    label: String,
    vertices: GpuBuffer,
    indices: GpuBuffer,
    index_count: u32,
}

impl DynamicBuffers {
    pub fn new(ctx: &Arc<GpuContext>, label: &str, vertex_bytes: u64, index_bytes: u64) -> Self {
        let vertices = ctx.create_host_buffer(&format!("{label} vertices"), vertex_bytes.max(16));
        let indices = ctx.create_host_buffer(&format!("{label} indices"), index_bytes.max(16));
        Self {
            ctx: Arc::clone(ctx),
            label: label.to_string(),
            vertices,
            indices,
            index_count: 0,
        }
    }

    pub fn upload(&mut self, vertex_bytes: &[u8], index_bytes: &[u8]) {
        if vertex_bytes.len() as u64 > self.vertices.size {
            self.vertices = self.ctx.create_host_buffer(
                &format!("{} vertices", self.label),
                (vertex_bytes.len() as u64).next_power_of_two(),
            );
        }
        if index_bytes.len() as u64 > self.indices.size {
            self.indices = self.ctx.create_host_buffer(
                &format!("{} indices", self.label),
                (index_bytes.len() as u64).next_power_of_two(),
            );
        }
        if !vertex_bytes.is_empty() {
            self.vertices.write_bytes(0, vertex_bytes);
        }
        if !index_bytes.is_empty() {
            self.indices.write_bytes(0, index_bytes);
        }
        self.index_count = (index_bytes.len() / std::mem::size_of::<u32>()) as u32;
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    pub fn update(&mut self, vertex_bytes: &[u8], index_bytes: &[u8]) {
        self.upload(vertex_bytes, index_bytes);
    }

    pub fn vertices(&self) -> &GpuBuffer {
        &self.vertices
    }

    pub fn indices(&self) -> &GpuBuffer {
        &self.indices
    }

    pub fn vertex_buffer(&self) -> &GpuBuffer {
        &self.vertices
    }

    pub fn index_buffer(&self) -> &GpuBuffer {
        &self.indices
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    pub fn vertex_bytes(&self) -> u64 {
        self.vertices.size
    }

    pub fn index_bytes(&self) -> u64 {
        self.indices.size
    }
}

/// Packed quad storage buffer for vertex pulling.
pub struct PulledMesh {
    buffer: Arc<GpuBuffer>,
    quad_count: u32,
}

impl PulledMesh {
    pub fn new(_ctx: &Arc<GpuContext>, buffer: Arc<GpuBuffer>, quad_count: u32) -> Self {
        Self { buffer, quad_count }
    }

    pub fn quad_count(&self) -> u32 {
        self.quad_count
    }

    pub fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }
}
