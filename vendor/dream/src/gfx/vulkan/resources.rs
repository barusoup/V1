use super::barriers::is_depth;
use super::{GpuBuffer, GpuContext, GpuImage};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use std::sync::Arc;

impl GpuContext {
    // --- Buffers -----------------------------------------------------------

    /// A host-visible buffer for data written every frame (uniforms).
    pub fn create_host_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_buffer_inner(
            label,
            size,
            host_buffer_usage(HostBufferKind::Uniform),
            MemoryLocation::CpuToGpu,
        )
    }

    /// A host-visible vertex buffer rewritten every frame (dynamic UI
    /// geometry). Binding a uniform-only buffer as a vertex buffer is a
    /// usage violation, so this is not just [`Self::create_host_buffer`]
    /// with a different name.
    pub fn create_host_vertex_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_buffer_inner(
            label,
            size,
            host_buffer_usage(HostBufferKind::Vertex),
            MemoryLocation::CpuToGpu,
        )
    }

    /// A host-visible index buffer rewritten every frame (dynamic UI
    /// geometry). See [`Self::create_host_vertex_buffer`].
    pub fn create_host_index_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_buffer_inner(
            label,
            size,
            host_buffer_usage(HostBufferKind::Index),
            MemoryLocation::CpuToGpu,
        )
    }

    /// A device-local buffer uploaded once from CPU data (vertex/index data).
    pub fn upload_buffer(
        self: &Arc<Self>,
        label: &str,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> GpuBuffer {
        let size = data.len().max(4) as u64;
        let dst = self.create_buffer_inner(
            label,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        );

        // Staging buffer, filled from the CPU.
        let mut staging = self.create_buffer_inner(
            label,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );
        staging.write(0, data);

        if data.is_empty() {
            // A zero-byte vkCmdCopyBuffer is invalid; the minimal destination
            // buffer stays valid for binding.
            return dst;
        }
        self.one_shot(|cmd| unsafe {
            let region = vk::BufferCopy::default().size(data.len() as u64);
            self.device
                .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region]);
            // The copy's writes must be visible to whichever stage consumes
            // the buffer next: vertex attributes/indices, pulled vertices,
            // indirect args, or compute input.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                        | vk::AccessFlags::INDEX_READ
                        | vk::AccessFlags::INDIRECT_COMMAND_READ,
                );
            // ALL_COMMANDS expands to the stages this queue actually
            // supports: a graphics-only queue (all select_device guarantees)
            // has no COMPUTE_SHADER stage, and naming one is invalid.
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        });
        dst
    }

    /// Upload vertex data once into a device-local buffer.
    pub fn create_vertex_buffer(self: &Arc<Self>, label: &str, data: &[u8]) -> GpuBuffer {
        self.upload_buffer(label, data, vk::BufferUsageFlags::VERTEX_BUFFER)
    }

    /// Upload index data once into a device-local buffer.
    pub fn create_index_buffer(self: &Arc<Self>, label: &str, data: &[u8]) -> GpuBuffer {
        self.upload_buffer(label, data, vk::BufferUsageFlags::INDEX_BUFFER)
    }

    /// Device-local buffer written and read through storage bindings.
    pub fn create_storage_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_buffer_inner(
            label,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )
    }

    /// Host-visible TRANSFER_SRC staging buffer for caller-managed uploads.
    pub fn create_transfer_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_buffer_inner(
            label,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )
    }

    /// Device-local buffer with a caller-chosen usage mask (compute outputs,
    /// indirect args, pulled vertex storage).
    pub fn create_gpu_buffer(
        self: &Arc<Self>,
        label: &str,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> GpuBuffer {
        self.create_buffer_inner(label, size, usage, MemoryLocation::GpuOnly)
    }

    /// A host-visible buffer a shader reads through a storage binding and the
    /// CPU refills every frame (per-instance records for instanced draws).
    pub fn create_host_storage_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_buffer_inner(
            label,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
        )
    }

    /// Host-visible buffer used to read rendered pixels back to the CPU.
    pub fn create_readback_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_buffer_inner(
            label,
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
        )
    }

    /// Host-visible buffer allocated in CPU-cached memory, for large bulk
    /// readbacks where the CPU will actually read the data (uncached device
    /// memory reads are ~40MB/s on ReBAR systems).
    pub fn create_cached_readback_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_buffer_inner(
            label,
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )
    }

    fn create_buffer_inner(
        self: &Arc<Self>,
        label: &str,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> GpuBuffer {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&info, None) }
            .unwrap_or_else(|err| panic!("{label}: create buffer: {err:?}"));
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let allocation = {
            let mut allocator = self.allocator.lock().unwrap();
            let allocator = allocator.as_mut().expect("allocator present");
            allocator
                .allocate(&AllocationCreateDesc {
                    name: label,
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap_or_else(|err| panic!("{label}: allocate buffer memory: {err:?}"))
        };
        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .unwrap_or_else(|err| panic!("{label}: bind buffer memory: {err:?}"));

        GpuBuffer {
            ctx: Arc::clone(self),
            buffer,
            allocation: Some(allocation),
            size: size.max(4),
        }
    }

    // --- Images ------------------------------------------------------------

    pub fn create_image(
        self: &Arc<Self>,
        label: &str,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> GpuImage {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: width.max(1),
                height: height.max(1),
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&info, None) }
            .unwrap_or_else(|err| panic!("{label}: create image: {err:?}"));
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let allocation = {
            let mut allocator = self.allocator.lock().unwrap();
            let allocator = allocator.as_mut().expect("allocator present");
            allocator
                .allocate(&AllocationCreateDesc {
                    name: label,
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap_or_else(|err| panic!("{label}: allocate image memory: {err:?}"))
        };
        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .unwrap_or_else(|err| panic!("{label}: bind image memory: {err:?}"));

        let aspect = if is_depth(format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        self.finish_image(label, image, allocation, format, aspect, 1)
    }

    /// Like [`Self::create_image`], but with a full mip chain. The view covers
    /// every level; texture uploads generate the chain with blits after the
    /// level-0 copy.
    pub fn create_image_mips(
        self: &Arc<Self>,
        label: &str,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
    ) -> GpuImage {
        let mip_levels = mip_levels.max(1);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: width.max(1),
                height: height.max(1),
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&info, None) }
            .unwrap_or_else(|err| panic!("{label}: create image: {err:?}"));
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let allocation = {
            let mut allocator = self.allocator.lock().unwrap();
            let allocator = allocator.as_mut().expect("allocator present");
            allocator
                .allocate(&AllocationCreateDesc {
                    name: label,
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap_or_else(|err| panic!("{label}: allocate image memory: {err:?}"))
        };
        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .unwrap_or_else(|err| panic!("{label}: bind image memory: {err:?}"));
        let aspect = if is_depth(format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        self.finish_image(label, image, allocation, format, aspect, mip_levels)
    }

    /// 2D array image with a full mip chain — the impostor atlas keeps six
    /// view layers per channel in one image.
    pub fn create_image_array_mips(
        self: &Arc<Self>,
        label: &str,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        array_layers: u32,
        mip_levels: u32,
    ) -> GpuImage {
        self.create_image_impl(
            label,
            ImageSpec {
                width,
                height,
                format,
                usage,
                array_layers,
                mip_levels,
                swizzle: vk::ComponentMapping::default(),
                view_type: vk::ImageViewType::TYPE_2D_ARRAY,
            },
        )
    }

    /// Single-layer image whose view swizzles channels — e.g. an R8 coverage
    /// page sampled as (1,1,1,a) so coverage bytes read as white-with-alpha.
    pub(crate) fn create_image_swizzled(
        self: &Arc<Self>,
        label: &str,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        swizzle: vk::ComponentMapping,
    ) -> GpuImage {
        self.create_image_impl(
            label,
            ImageSpec {
                width,
                height,
                format,
                usage,
                array_layers: 1,
                mip_levels: 1,
                swizzle,
                view_type: vk::ImageViewType::TYPE_2D,
            },
        )
    }

    fn create_image_impl(self: &Arc<Self>, label: &str, spec: ImageSpec) -> GpuImage {
        let ImageSpec {
            width,
            height,
            format,
            usage,
            array_layers,
            mip_levels,
            swizzle,
            view_type,
        } = spec;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: width.max(1),
                height: height.max(1),
                depth: 1,
            })
            .mip_levels(mip_levels.max(1))
            .array_layers(array_layers.max(1))
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&info, None) }
            .unwrap_or_else(|err| panic!("{label}: create array image: {err:?}"));
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let allocation = {
            let mut allocator = self.allocator.lock().unwrap();
            let allocator = allocator.as_mut().expect("allocator present");
            allocator
                .allocate(&AllocationCreateDesc {
                    name: label,
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap_or_else(|err| panic!("{label}: allocate image memory: {err:?}"))
        };
        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .unwrap_or_else(|err| panic!("{label}: bind image memory: {err:?}"));

        let aspect = if is_depth(format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .components(swizzle)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: mip_levels.max(1),
                base_array_layer: 0,
                layer_count: array_layers.max(1),
            });
        let view = unsafe { self.device.create_image_view(&view_info, None) }
            .unwrap_or_else(|err| panic!("{label}: create array view: {err:?}"));

        GpuImage {
            ctx: Arc::clone(self),
            image,
            view,
            allocation: Some(allocation),
        }
    }

    fn finish_image(
        self: &Arc<Self>,
        label: &str,
        image: vk::Image,
        allocation: Allocation,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> GpuImage {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { self.device.create_image_view(&view_info, None) }
            .unwrap_or_else(|err| panic!("{label}: create image view: {err:?}"));

        GpuImage {
            ctx: Arc::clone(self),
            image,
            view,
            allocation: Some(allocation),
        }
    }
}

/// Creation parameters for a sampled/staged image (kept as a struct so
/// the array/single-view variants share one signature).
struct ImageSpec {
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    array_layers: u32,
    mip_levels: u32,
    swizzle: vk::ComponentMapping,
    view_type: vk::ImageViewType,
}

impl GpuBuffer {
    /// Bytes the allocator gave this buffer's backing allocation.
    pub fn allocation_size(&self) -> u64 {
        self.allocation.as_ref().map(|a| a.size()).unwrap_or(0)
    }

    /// Read the buffer's mapped bytes (host-visible buffers only).
    pub fn mapped(&self) -> Option<&[u8]> {
        self.allocation.as_ref().and_then(|a| a.mapped_slice())
    }

    /// Write bytes into a host-visible buffer. Panics if the buffer is
    /// device-local, which would be a programming error.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) {
        let allocation = self.allocation.as_mut().expect("buffer allocation present");
        let mapped = allocation
            .mapped_slice_mut()
            .expect("buffer is host visible");
        let start = offset as usize;
        mapped[start..start + bytes.len()].copy_from_slice(bytes);
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe { self.ctx.device.destroy_buffer(self.buffer, None) };
        if let Some(allocation) = self.allocation.take() {
            let mut allocator = self.ctx.allocator.lock().unwrap();
            if let Some(allocator) = allocator.as_mut() {
                let _ = allocator.free(allocation);
            }
        }
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_image_view(self.view, None);
            self.ctx.device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let mut allocator = self.ctx.allocator.lock().unwrap();
            if let Some(allocator) = allocator.as_mut() {
                let _ = allocator.free(allocation);
            }
        }
    }
}

/// Which binding a per-frame host-visible buffer will be used as. The
/// usage flags differ, and choosing the wrong one is a validation error
/// rather than a slowdown.
#[derive(Clone, Copy, Debug)]
enum HostBufferKind {
    Uniform,
    Vertex,
    Index,
}

/// Usage flags for [`HostBufferKind`] (pure, so tests can check the mapping
/// without a device).
fn host_buffer_usage(kind: HostBufferKind) -> vk::BufferUsageFlags {
    match kind {
        HostBufferKind::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
        HostBufferKind::Vertex => vk::BufferUsageFlags::VERTEX_BUFFER,
        HostBufferKind::Index => vk::BufferUsageFlags::INDEX_BUFFER,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_buffer_kinds_map_to_their_own_usages() {
        assert_eq!(
            host_buffer_usage(HostBufferKind::Uniform),
            vk::BufferUsageFlags::UNIFORM_BUFFER
        );
        assert_eq!(
            host_buffer_usage(HostBufferKind::Vertex),
            vk::BufferUsageFlags::VERTEX_BUFFER
        );
        assert_eq!(
            host_buffer_usage(HostBufferKind::Index),
            vk::BufferUsageFlags::INDEX_BUFFER
        );
    }
}
