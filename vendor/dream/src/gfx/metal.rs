//! Native macOS Metal context: device, command queue, CAMetalLayer, and
//! buffer/texture handles tailored for Apple Silicon unified memory (UMA) and
//! tile-based deferred rendering (TBDR).

use crate::texture::SamplerDesc;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{MainThreadMarker, msg_send};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLPixelFormat, MTLResource,
    MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerMipFilter, MTLSamplerState, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use winit::window::Window;

pub mod mesh;
pub mod pipeline;
pub mod shader;
pub mod swapchain;

pub use mesh::{DynamicBuffers, PulledMesh};
pub use objc2_metal::MTLPixelFormat as PixelFormat;
pub use swapchain::{PresentMode, Swapchain, SwapchainOptions};

/// Frames the CPU may have in flight at once. Triple buffering matches
/// `CAMetalLayer::maximumDrawableCount = 3` and the Vulkan backend.
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SamplerKey {
    filter: u8,
    mip: u8,
    address: u8,
    anisotropy: bool,
}

impl SamplerKey {
    fn from_desc(d: &SamplerDesc) -> Self {
        Self {
            filter: d.filter as u8,
            mip: d.mip as u8,
            address: d.address as u8,
            anisotropy: d.anisotropy,
        }
    }
}

struct RetiredBuffer {
    serial: u64,
    _buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Owns the Metal device, command queue, and `CAMetalLayer`. Cheaply shareable via `Arc`.
pub struct GpuContext {
    _window: Option<Arc<Window>>,
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    layer: Retained<CAMetalLayer>,
    has_unified_memory: bool,
    submitted_serial: AtomicU64,
    completed_serial: AtomicU64,
    retired_buffers: Mutex<Vec<RetiredBuffer>>,
    library_cache: Mutex<
        HashMap<
            (u64, usize, u64),
            (
                Retained<ProtocolObject<dyn MTLLibrary>>,
                Vec<(String, String)>,
            ),
        >,
    >,
    shared_samplers: Mutex<HashMap<SamplerKey, Retained<ProtocolObject<dyn MTLSamplerState>>>>,
    default_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    buffer_count: AtomicU64,
    blocking_submissions: AtomicU64,
    pub(crate) live_textures: AtomicUsize,
}

// Safety: MTLDevice, MTLCommandQueue, MTLLibrary, MTLSamplerState, and CAMetalLayer
// are thread-safe or synchronized via internal mutexes / main-thread creation.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl GpuContext {
    pub fn new(window: &Window) -> Arc<Self> {
        let device =
            MTLCreateSystemDefaultDevice().expect("Metal is not supported on this macOS device");
        let queue = device
            .newCommandQueue()
            .expect("create Metal command queue");
        let has_unified_memory = device.hasUnifiedMemory();

        let mtm = MainThreadMarker::new()
            .expect("GpuContext::new must be called on the main thread on macOS");
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        layer.setMaximumDrawableCount(MAX_FRAMES_IN_FLIGHT as _);
        layer.setFramebufferOnly(true);
        layer.setPresentsWithTransaction(false);
        layer.setContentsScale(window.scale_factor());

        let _ = mtm;
        attach_layer_to_window(window, &layer);

        let default_sampler = create_sampler_state(&device, &SamplerDesc::default());

        eprintln!(
            "Metal device: {} (unified_memory={}, max_frames={})",
            device.name(),
            has_unified_memory,
            MAX_FRAMES_IN_FLIGHT
        );

        Arc::new(Self {
            _window: None,
            device,
            queue,
            layer,
            has_unified_memory,
            submitted_serial: AtomicU64::new(0),
            completed_serial: AtomicU64::new(0),
            retired_buffers: Mutex::new(Vec::new()),
            library_cache: Mutex::new(HashMap::new()),
            shared_samplers: Mutex::new(HashMap::new()),
            default_sampler,
            buffer_count: AtomicU64::new(0),
            blocking_submissions: AtomicU64::new(0),
            live_textures: AtomicUsize::new(0),
        })
    }

    pub fn new_owned(window: Arc<Window>) -> Arc<Self> {
        let mut ctx = Self::new(&window);
        Arc::get_mut(&mut ctx).expect("fresh context")._window = Some(window);
        ctx
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    pub fn layer(&self) -> &CAMetalLayer {
        &self.layer
    }

    pub fn has_unified_memory(&self) -> bool {
        self.has_unified_memory
    }

    pub fn max_frames_in_flight(&self) -> usize {
        MAX_FRAMES_IN_FLIGHT
    }

    pub fn max_storage_buffer_range(&self) -> u32 {
        // Metal buffers on Apple Silicon support gigabytes; clamp to u32::MAX
        // for parity with Vulkan's u32 limit query.
        self.device.maxBufferLength().min(u32::MAX as usize) as u32
    }

    pub fn gpu_allocation_stats(&self) -> (u64, usize) {
        (
            self.device.currentAllocatedSize() as u64,
            self.buffer_count.load(Ordering::Relaxed) as usize,
        )
    }

    pub fn default_sampler(&self) -> &ProtocolObject<dyn MTLSamplerState> {
        &self.default_sampler
    }

    pub fn get_or_create_sampler(
        &self,
        desc: &SamplerDesc,
    ) -> Retained<ProtocolObject<dyn MTLSamplerState>> {
        let key = SamplerKey::from_desc(desc);
        let mut cache = self.shared_samplers.lock().unwrap();
        if let Some(s) = cache.get(&key) {
            return s.clone();
        }
        let s = create_sampler_state(&self.device, desc);
        cache.insert(key, s.clone());
        s
    }

    pub fn compile_library(
        &self,
        source: &str,
        label: &str,
        bindings: &shader::MslBindings,
    ) -> (
        Retained<ProtocolObject<dyn MTLLibrary>>,
        Vec<(String, String)>,
    ) {
        self.compile_library_opts(source, label, bindings, true)
    }

    pub fn compile_library_unflipped(
        &self,
        source: &str,
        label: &str,
        bindings: &shader::MslBindings,
    ) -> (
        Retained<ProtocolObject<dyn MTLLibrary>>,
        Vec<(String, String)>,
    ) {
        self.compile_library_opts(source, label, bindings, false)
    }

    fn compile_library_opts(
        &self,
        source: &str,
        label: &str,
        bindings: &shader::MslBindings,
        flip_y: bool,
    ) -> (
        Retained<ProtocolObject<dyn MTLLibrary>>,
        Vec<(String, String)>,
    ) {
        let key = (
            fnv1a_64(source.as_bytes()) ^ if flip_y { 0 } else { 0x9e37_79b9 },
            source.len(),
            bindings.hash_key(),
        );
        {
            let cache = self.library_cache.lock().unwrap();
            if let Some(entry) = cache.get(&key) {
                return entry.clone();
            }
        }
        let compiled = if flip_y {
            shader::compile_wgsl_to_msl(source, label, bindings)
        } else {
            shader::compile_wgsl_to_msl_unflipped(source, label, bindings)
        };
        let ns_src = NSString::from_str(&compiled.source);
        let lib = self
            .device
            .newLibraryWithSource_options_error(&ns_src, None)
            .unwrap_or_else(|err| {
                panic!(
                    "{label}: Metal shader compilation failed: {err:?}\n--- MSL ---\n{}",
                    compiled.source
                )
            });
        let val = (lib, compiled.entry_points);
        self.library_cache.lock().unwrap().insert(key, val.clone());
        val
    }

    // --- Buffers ---

    /// Host-visible buffer. On Apple Silicon UMA this uses `StorageModeShared`
    /// with `CPUCacheModeWriteCombined` for fast zero-copy CPU->GPU writes.
    pub fn create_host_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        let size = size.max(16);
        let options = if self.has_unified_memory {
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined
        } else {
            MTLResourceOptions::StorageModeShared
        };
        self.alloc_buffer(label, size, options)
    }

    /// Host-visible readback buffer with default CPU caching so CPU reads are fast.
    pub fn create_readback_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        let size = size.max(16);
        let options = MTLResourceOptions::StorageModeShared;
        self.alloc_buffer(label, size, options)
    }

    /// Device-local buffer. On Apple Silicon UMA, `StorageModeShared` or
    /// `StorageModePrivate` have the same physical RAM; `StorageModePrivate`
    /// allows GPU-only compression/caching when CPU never touches it.
    pub fn create_device_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        let size = size.max(16);
        let options = MTLResourceOptions::StorageModePrivate;
        self.alloc_buffer(label, size, options)
    }

    /// Upload initial bytes into a GPU buffer. On Apple Silicon UMA this writes
    /// directly into a `StorageModeShared` buffer with zero staging copy!
    pub fn upload_buffer(self: &Arc<Self>, label: &str, data: &[u8]) -> GpuBuffer {
        if self.has_unified_memory {
            let mut buf = self.create_host_buffer(label, data.len() as u64);
            buf.write_bytes(0, data);
            buf
        } else {
            let mut staging = self.create_host_buffer("upload staging", data.len() as u64);
            staging.write_bytes(0, data);
            let dst = self.create_device_buffer(label, data.len() as u64);
            self.one_shot(|cmd| {
                let blit = cmd.blitCommandEncoder().expect("blit encoder");
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &staging.buffer,
                        0,
                        &dst.buffer,
                        0,
                        data.len(),
                    );
                }
                blit.endEncoding();
            });
            dst
        }
    }

    fn alloc_buffer(
        self: &Arc<Self>,
        label: &str,
        size: u64,
        options: MTLResourceOptions,
    ) -> GpuBuffer {
        let buffer = self
            .device
            .newBufferWithLength_options(size as usize, options)
            .unwrap_or_else(|| panic!("{label}: failed to allocate Metal buffer ({size} bytes)"));
        let ns_label = NSString::from_str(label);
        buffer.setLabel(Some(&ns_label));
        self.buffer_count.fetch_add(1, Ordering::Relaxed);
        GpuBuffer {
            ctx: Arc::clone(self),
            buffer,
            size,
        }
    }

    // --- Images ---

    pub fn create_image(
        self: &Arc<Self>,
        label: &str,
        width: u32,
        height: u32,
        format: MTLPixelFormat,
        usage: MTLTextureUsage,
    ) -> GpuImage {
        self.create_image_mipmapped(label, width, height, 1, format, usage)
    }

    pub fn create_image_mipmapped(
        self: &Arc<Self>,
        label: &str,
        width: u32,
        height: u32,
        mip_levels: u32,
        format: MTLPixelFormat,
        usage: MTLTextureUsage,
    ) -> GpuImage {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                width.max(1) as usize,
                height.max(1) as usize,
                mip_levels > 1,
            )
        };
        unsafe {
            desc.setMipmapLevelCount(mip_levels.max(1) as usize);
        }
        desc.setTextureType(MTLTextureType::Type2D);
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setUsage(usage);
        let texture = self
            .device
            .newTextureWithDescriptor(&desc)
            .unwrap_or_else(|| panic!("{label}: failed to create Metal texture"));
        let ns_label = NSString::from_str(label);
        texture.setLabel(Some(&ns_label));
        GpuImage {
            ctx: Arc::clone(self),
            texture,
            format,
            width: width.max(1),
            height: height.max(1),
            mip_levels: mip_levels.max(1),
        }
    }

    // --- Submissions & Synchronization ---

    pub fn next_serial(&self) -> u64 {
        self.submitted_serial.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn confirm_completed(&self, serial: u64) {
        self.completed_serial.fetch_max(serial, Ordering::Release);
    }

    pub fn submission_complete(&self, serial: u64) -> bool {
        self.completed_serial.load(Ordering::Acquire) >= serial
    }

    pub fn retire_buffer(&self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let serial = self.submitted_serial.load(Ordering::Acquire);
        if self.submission_complete(serial) {
            return;
        }
        self.retired_buffers.lock().unwrap().push(RetiredBuffer {
            serial,
            _buffer: buffer,
        });
    }

    pub fn collect_retired_buffers(&self) {
        let completed = self.completed_serial.load(Ordering::Acquire);
        let mut retired = self.retired_buffers.lock().unwrap();
        retired.retain(|entry| entry.serial > completed);
    }

    pub fn blocking_submissions(&self) -> u64 {
        self.blocking_submissions.load(Ordering::Relaxed)
    }

    pub fn submitted_serial(&self) -> u64 {
        self.submitted_serial.load(Ordering::Acquire)
    }

    pub fn completed_serial(&self) -> u64 {
        self.completed_serial.load(Ordering::Acquire)
    }

    pub fn texture_object_stats(&self) -> super::vulkan::TextureObjectStats {
        super::vulkan::TextureObjectStats {
            samplers: self.shared_samplers.lock().unwrap().len(),
            descriptor_pools: 0,
            live_texture_sets: self.live_textures.load(Ordering::Relaxed),
        }
    }

    pub fn one_shot(&self, record: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>)) {
        self.blocking_submissions.fetch_add(1, Ordering::Relaxed);
        let cmd = self
            .queue
            .commandBuffer()
            .expect("create Metal command buffer");
        record(&cmd);
        let serial = self.next_serial();
        cmd.commit();
        cmd.waitUntilCompleted();
        self.confirm_completed(serial);
    }

    pub fn submit_async(&self, record: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>)) -> u64 {
        let cmd = self
            .queue
            .commandBuffer()
            .expect("create Metal command buffer");
        record(&cmd);
        let serial = self.next_serial();
        cmd.commit();
        serial
    }

    pub fn wait_idle(&self) {
        let cmd = self
            .queue
            .commandBuffer()
            .expect("create Metal command buffer for wait_idle");
        cmd.commit();
        cmd.waitUntilCompleted();
        let serial = self.submitted_serial.load(Ordering::Acquire);
        self.confirm_completed(serial);
        self.collect_retired_buffers();
    }

    pub fn create_host_storage_buffer(self: &Arc<Self>, label: &str, size: u64) -> GpuBuffer {
        self.create_host_buffer(label, size)
    }

    pub fn create_upload_buffer(self: &Arc<Self>, label: &str, size: usize) -> GpuBuffer {
        self.create_host_buffer(label, size.max(1) as u64)
    }

    pub fn create_storage_buffer(self: &Arc<Self>, label: &str, data: &[u8]) -> GpuBuffer {
        self.upload_buffer(label, data)
    }

    pub fn create_uniform_buffer(self: &Arc<Self>, label: &str, data: &[u8]) -> GpuBuffer {
        self.upload_buffer(label, data)
    }

    pub fn create_vertex_buffer(self: &Arc<Self>, label: &str, data: &[u8]) -> GpuBuffer {
        self.upload_buffer(label, data)
    }

    pub fn create_index_buffer(self: &Arc<Self>, label: &str, data: &[u8]) -> GpuBuffer {
        self.upload_buffer(label, data)
    }

    pub fn latest_submit_serial(&self) -> u64 {
        self.submitted_serial.load(Ordering::Acquire)
    }

    pub fn is_serial_complete(&self, serial: u64) -> bool {
        self.submission_complete(serial)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_texture_2d_array(
        self: &Arc<Self>,
        label: &str,
        width: u32,
        height: u32,
        layers: u32,
        mip_levels: u32,
        format: MTLPixelFormat,
        render_target: bool,
    ) -> GpuImage {
        let desc = MTLTextureDescriptor::new();
        desc.setTextureType(MTLTextureType::Type2DArray);
        desc.setPixelFormat(format);
        unsafe {
            desc.setWidth(width.max(1) as usize);
            desc.setHeight(height.max(1) as usize);
            desc.setArrayLength(layers.max(1) as usize);
            desc.setMipmapLevelCount(mip_levels.max(1) as usize);
        }
        desc.setStorageMode(MTLStorageMode::Private);
        let mut usage = MTLTextureUsage::ShaderRead;
        if render_target {
            usage |= MTLTextureUsage::RenderTarget;
        }
        desc.setUsage(usage);
        let texture = self
            .device
            .newTextureWithDescriptor(&desc)
            .expect("create Metal 2DArray texture");
        texture.setLabel(Some(&NSString::from_str(label)));
        GpuImage {
            ctx: Arc::clone(self),
            texture,
            width,
            height,
            mip_levels: mip_levels.max(1),
            format,
        }
    }

    pub fn create_depth_image(
        self: &Arc<Self>,
        width: u32,
        height: u32,
        format: MTLPixelFormat,
    ) -> GpuImage {
        self.create_image(
            "depth",
            width,
            height,
            format,
            MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
        )
    }

    pub fn create_pulled_mesh(
        self: &Arc<Self>,
        label: &str,
        data: &[u8],
        quad_count: u32,
    ) -> PulledMesh {
        assert_eq!(
            data.len(),
            quad_count as usize * 8,
            "{label}: packed quads are 8 bytes each"
        );
        let quads = Arc::new(self.upload_buffer(label, data));
        PulledMesh::new(self, quads, quad_count)
    }

    pub fn create_pulled_mesh_from_buffer(
        self: &Arc<Self>,
        buffer: Arc<GpuBuffer>,
        quad_count: u32,
    ) -> PulledMesh {
        assert!(
            Arc::ptr_eq(&buffer.ctx, self),
            "pulled mesh buffer belongs to a different GpuContext"
        );
        PulledMesh::new(self, buffer, quad_count)
    }
}

fn attach_layer_to_window(window: &Window, layer: &CAMetalLayer) {
    let handle = window
        .window_handle()
        .expect("window must have a valid handle")
        .as_raw();
    match handle {
        RawWindowHandle::AppKit(appkit) => {
            let ns_view: NonNull<AnyObject> = appkit.ns_view.cast();
            unsafe {
                let view = ns_view.as_ptr();
                let _: () = msg_send![view, setWantsLayer: true];
                let layer_ptr: *const CAMetalLayer = Retained::as_ptr(&Retained::from(layer));
                let _: () = msg_send![view, setLayer: layer_ptr];
            }
        }
        _ => panic!("expected AppKit window handle on macOS"),
    }
}

fn create_sampler_state(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &SamplerDesc,
) -> Retained<ProtocolObject<dyn MTLSamplerState>> {
    use crate::texture::{AddressMode, Filter, MipFilter};

    let d = MTLSamplerDescriptor::new();
    let f = match desc.filter {
        Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
        Filter::Linear => MTLSamplerMinMagFilter::Linear,
    };
    d.setMagFilter(f);
    d.setMinFilter(f);
    d.setMipFilter(match desc.mip {
        MipFilter::Nearest => MTLSamplerMipFilter::Nearest,
        MipFilter::Linear => MTLSamplerMipFilter::Linear,
    });
    let addr = match desc.address {
        AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
    };
    d.setSAddressMode(addr);
    d.setTAddressMode(addr);
    d.setRAddressMode(addr);
    d.setMaxAnisotropy(if desc.anisotropy { 16 } else { 1 });
    device
        .newSamplerStateWithDescriptor(&d)
        .expect("create Metal sampler state")
}

// --- Handles ---

pub struct GpuBuffer {
    ctx: Arc<GpuContext>,
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub size: u64,
}

impl GpuBuffer {
    pub fn mapped(&self) -> Option<&[u8]> {
        if self.buffer.storageMode() == MTLStorageMode::Private {
            return None;
        }
        let ptr = self.buffer.contents().as_ptr() as *const u8;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.size as usize) })
    }

    pub fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        if self.buffer.storageMode() == MTLStorageMode::Private {
            return None;
        }
        let ptr = self.buffer.contents().as_ptr() as *mut u8;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.size as usize) })
    }

    pub fn read_bytes(&self, len: usize) -> &[u8] {
        let mapped = self
            .mapped()
            .expect("read_bytes requires host-accessible buffer");
        &mapped[..len.min(mapped.len())]
    }

    pub fn write_bytes(&mut self, offset: u64, data: &[u8]) {
        let offset_usize = usize::try_from(offset).expect("GpuBuffer::write_bytes offset overflow");
        let end = offset_usize
            .checked_add(data.len())
            .expect("GpuBuffer::write_bytes length overflow");
        assert!(
            end <= self.size as usize,
            "GpuBuffer::write_bytes out of bounds"
        );
        if let Some(dst) = self.mapped_mut() {
            dst[offset as usize..end].copy_from_slice(data);
            if self.buffer.storageMode() == MTLStorageMode::Managed {
                self.buffer
                    .didModifyRange(NSRange::new(offset as usize, data.len()));
            }
        }
    }

    pub fn write(&mut self, offset: u64, data: &[u8]) {
        self.write_bytes(offset, data);
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        self.ctx.buffer_count.fetch_sub(1, Ordering::Relaxed);
        self.ctx.retire_buffer(self.buffer.clone());
    }
}

pub struct GpuImage {
    #[allow(dead_code)]
    ctx: Arc<GpuContext>,
    pub texture: Retained<ProtocolObject<dyn MTLTexture>>,
    pub format: MTLPixelFormat,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
}

impl GpuImage {
    pub fn allocation_size(&self) -> u64 {
        self.texture.allocatedSize() as u64
    }
}

pub fn is_depth(format: MTLPixelFormat) -> bool {
    matches!(
        format,
        MTLPixelFormat::Depth32Float
            | MTLPixelFormat::Depth16Unorm
            | MTLPixelFormat::Depth32Float_Stencil8
            | MTLPixelFormat::Depth24Unorm_Stencil8
    )
}

pub fn is_srgb(format: MTLPixelFormat) -> bool {
    matches!(
        format,
        MTLPixelFormat::BGRA8Unorm_sRGB | MTLPixelFormat::RGBA8Unorm_sRGB
    )
}

pub(crate) fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}
