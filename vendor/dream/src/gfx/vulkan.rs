//! Native Vulkan (ash) context: instance, device, allocator, swapchain, and the
//! buffer/image handles the engine and game share.
//!
//! Everything here is deliberately low-level. Higher-level orchestration lives in
//! `gfx.rs`. Nothing in this module knows about cameras, passes, or shaders
//! beyond compiling WGSL to SPIR-V.

use crate::gfx::shader;
use ash::{Entry, Instance, khr, vk};
use gpu_allocator::vulkan::{Allocation, Allocator, AllocatorCreateDesc};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::{CStr, c_char};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use winit::window::Window;

mod barriers;
mod mesh;
mod resources;
mod submission;
mod swapchain;

pub use barriers::{
    buffer_barrier, image_barrier, image_barrier_full, image_barrier_mips, is_depth, is_srgb,
};
pub use mesh::{DynamicBuffers, PulledMesh};
pub use swapchain::Swapchain;

/// Frames the CPU may have in flight at once. Three slots give the loop
/// enough slack to ride out a presentation-engine hiccup (an older frame's
/// fence still pending) without blocking on `wait_for_fences`. The swapchain
/// image count tracks it (`frames in flight + 1`), and every per-slot
/// resource sizes itself through [`GpuContext::max_frames_in_flight`].
const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Owns the Vulkan device. Cheaply shareable via `Arc`: buffers and images keep
/// it alive, so they can free their own memory on drop without ordering hazards.
pub struct GpuContext {
    /// Only held so the Vulkan loader stays loaded for the process lifetime;
    /// dropping it would unload the library the device still needs.
    _entry: Entry,
    /// When created through `new_owned`, keeps the window alive until after
    /// `Drop` destroys the surface that was created on it. Fields release
    /// after `Drop::drop` runs, so this stays alive long enough.
    _window: Option<Arc<Window>>,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    surface: vk::SurfaceKHR,
    surface_loader: khr::surface::Instance,
    pub swapchain_loader: khr::swapchain::Device,
    // `Option` so the teardown can free the allocator before the device goes
    // away: `gpu-allocator` touches the device on drop, and doing that after
    // `vkDestroyDevice` crashes.
    allocator: Mutex<Option<Allocator>>,
    /// Frame resources we allocate once and cycle through.
    transient_pool: vk::CommandPool,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Whether dynamic rendering is core (1.3) or the KHR extension.
    pub core_1_3: bool,
    /// Monotonic counters advanced by `submit_tracked`/`confirm_completed`.
    /// On this in-order queue, confirming serial N implies every earlier
    /// submission finished, so `fetch_max` is sound.
    submitted_serial: AtomicU64,
    completed_serial: AtomicU64,
    blocking_submissions: AtomicU64,
    /// Serializes every queue submission AND presentation -- serial order
    /// must match queue order. Raw `queue_submit`/`queue_present` calls
    /// outside `submit_tracked`/`present_tracked` must hold this lock.
    pub queue_lock: Mutex<()>,
    /// Guards `transient_pool`: command-buffer alloc/reset/free on a shared
    /// pool is not externally synchronized in Vulkan.
    transient_lock: Mutex<()>,
    /// Small ring of command buffer + fence pairs for submissions the caller
    /// does not wait on. Reused; a full ring means backpressure, not a wait.
    async_ring: Mutex<AsyncRing>,
    /// Queue capability flags picked at device selection (GRAPHICS+PRESENT
    /// guaranteed; COMPUTE only on some devices).
    queue_flags: vk::QueueFlags,
    /// Non-zero when the queue family supports timestamp queries.
    timestamp_valid_bits: u32,
    /// Limits used for clamping arena pages/descriptor ranges.
    max_storage_buffer_range: u32,
}

struct AsyncSlot {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    serial: Option<u64>,
}

const ASYNC_RING_SLOTS: usize = 3;

#[derive(Default)]
struct AsyncRing {
    slots: [Option<AsyncSlot>; ASYNC_RING_SLOTS],
}

// Safety: Vulkan queues/devices are externally synchronized. We only mutate the
// allocator behind a mutex and treat the rest as read-only after init.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Create a context on `window`. The caller must keep the window alive
    /// until this context and every resource it owns (buffers, images,
    /// surfaces) have been dropped; use [`Self::new_owned`] to hand that
    /// lifetime over instead.
    pub fn new(window: &Window) -> Arc<Self> {
        unsafe { Self::new_unchecked(window) }
    }

    /// `new` for callers handing the window over: the context keeps the
    /// handle alive until its own `Drop` has destroyed the surface built on
    /// it, so resources can't outlive the native window.
    pub fn new_owned(window: Arc<Window>) -> Arc<Self> {
        let mut ctx = Self::new(&window);
        // The fresh Arc is uniquely owned, so the field can be attached here.
        Arc::get_mut(&mut ctx).expect("fresh context")._window = Some(window);
        ctx
    }

    unsafe fn new_unchecked(window: &Window) -> Arc<Self> {
        let (entry, loader_note) = load_vulkan_entry();
        eprintln!("Vulkan loader: {loader_note}");

        let surface_exts =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .expect("query required surface extensions");
        let mut ext_ptrs: Vec<*const c_char> = surface_exts.to_vec();

        // MoltenVK on macOS enumerates through portability mode: ask for it
        // only when the loader advertises the extension, so other loaders
        // are unaffected.
        let mut instance_flags = vk::InstanceCreateFlags::empty();
        let instance_exts =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let portability = instance_exts.iter().any(|ext| {
            ext.extension_name_as_c_str()
                .map(|name| name == c"VK_KHR_portability_enumeration")
                .unwrap_or(false)
        });
        if portability {
            ext_ptrs.push(c"VK_KHR_portability_enumeration".as_ptr());
            instance_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            eprintln!("Vulkan: VK_KHR_portability_enumeration enabled");
        }

        let app_name = c"Dream";
        let engine_name = c"Dream Engine";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(engine_name)
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let instance_info = vk::InstanceCreateInfo::default()
            .flags(instance_flags)
            .application_info(&app_info)
            .enabled_extension_names(&ext_ptrs);
        let instance =
            unsafe { entry.create_instance(&instance_info, None) }.expect("create Vulkan instance");

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
        }
        .expect("create window surface");

        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family, portability_subset) =
            unsafe { pick_physical_device(&instance, &surface_loader, surface) };

        let mut device_ext_names = vec![c"VK_KHR_swapchain".as_ptr()];
        // Required on MoltenVK: the subset extension gates the non-portable
        // features we use. It is only ever enabled when the device lists it.
        if portability_subset {
            device_ext_names.push(c"VK_KHR_portability_subset".as_ptr());
        }
        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let queue_infos = [queue_info];
        // Anisotropy is opt-in per sampler (texture::SamplerDesc); enable the
        // feature only when the adapter supports it so those samplers are
        // valid to create.
        let supported_features = unsafe { instance.get_physical_device_features(physical_device) };
        let base_features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(supported_features.sampler_anisotropy == vk::TRUE)
            // The scene pass blends colour attachment 0 while leaving the
            // velocity attachment unblended, which requires a different blend
            // state per attachment. Without this feature those pipelines are
            // invalid to create; pick_physical_device already rejects
            // adapters that lack it.
            .independent_blend(true);

        // Dynamic rendering is required from Vulkan 1.3 core — device
        // selection already rejected anything that cannot provide it.
        let mut features_13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_ext_names)
            .enabled_features(&base_features)
            .push_next(&mut features_13);

        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .expect("create logical device");
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::new(8 * 1024 * 1024, 2 * 1024 * 1024),
        })
        .expect("create GPU allocator");

        let transient_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
        }
        .expect("create transient command pool");

        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        let family_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                [queue_family as usize];
        let limits = unsafe { instance.get_physical_device_properties(physical_device) }.limits;

        let ctx = Arc::new(Self {
            _entry: entry,
            _window: None,
            instance,
            physical_device,
            device,
            queue,
            queue_family,
            surface,
            surface_loader,
            swapchain_loader,
            allocator: Mutex::new(Some(allocator)),
            transient_pool,
            memory_properties,
            // Vulkan 1.3 is a hard requirement: see `pick_physical_device`.
            core_1_3: true,
            submitted_serial: AtomicU64::new(0),
            completed_serial: AtomicU64::new(0),
            blocking_submissions: AtomicU64::new(0),
            queue_lock: Mutex::new(()),
            transient_lock: Mutex::new(()),
            async_ring: Mutex::new(AsyncRing::default()),
            queue_flags: family_props.queue_flags,
            timestamp_valid_bits: family_props.timestamp_valid_bits,
            max_storage_buffer_range: limits.max_storage_buffer_range,
        });
        ctx.print_adapter();
        ctx
    }

    fn print_adapter(&self) {
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "unknown".to_string());
        let version = (props.api_version >> 22) & 0x7f;
        let minor = (props.api_version >> 12) & 0x3ff;
        let patch = props.api_version & 0xfff;
        eprintln!(
            "GPU: {name} (Vulkan {version}.{minor}.{patch}, {:?}, queue flags {:?})",
            props.device_type, self.queue_flags
        );
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Live GPU allocation bytes and count, from the allocator's own books.
    pub fn gpu_allocation_stats(&self) -> (u64, usize) {
        let allocator = self.allocator.lock().unwrap();
        allocator.as_ref().map_or((0, 0), |a| {
            let report = a.generate_report();
            let count = report.allocations.len();
            (report.total_allocated_bytes, count)
        })
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Capability flags of the single device queue (GRAPHICS+PRESENT always;
    /// COMPUTE on compute-capable adapters).
    pub fn queue_flags(&self) -> vk::QueueFlags {
        self.queue_flags
    }

    /// `vkPhysicalDeviceLimits::timestampValidBits` -- 0 means timestamps
    /// are unsupported on this queue.
    pub fn timestamp_valid_bits(&self) -> u32 {
        self.timestamp_valid_bits
    }

    /// `vkPhysicalDeviceLimits::maxStorageBufferRange`.
    pub fn max_storage_buffer_range(&self) -> u32 {
        self.max_storage_buffer_range
    }

    /// Nanoseconds per timestamp tick, for converting query results.
    pub fn timestamp_period(&self) -> f32 {
        unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        }
        .limits
        .timestamp_period
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn surface_loader(&self) -> &khr::surface::Instance {
        &self.surface_loader
    }

    pub fn max_frames_in_flight(&self) -> usize {
        MAX_FRAMES_IN_FLIGHT
    }

    pub fn create_shader_module(&self, spv: &[u32], label: &str) -> vk::ShaderModule {
        let code = vk::ShaderModuleCreateInfo::default().code(spv);
        unsafe { self.device.create_shader_module(&code, None) }
            .unwrap_or_else(|err| panic!("{label}: create shader module: {err:?}"))
    }

    /// Compile WGSL source and upload it as a shader module.
    pub fn compile_shader(&self, source: &str, label: &str) -> vk::ShaderModule {
        if let Some(spv) = shader::precompiled_wgsl(source) {
            return self.create_shader_module(spv, label);
        }
        let spv = shader::compile_wgsl(source, label);
        self.create_shader_module(&spv, label)
    }

    pub fn create_sampler(&self, label: &str) -> vk::Sampler {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        unsafe { self.device.create_sampler(&info, None) }
            .unwrap_or_else(|err| panic!("{label}: create sampler: {err:?}"))
    }

    /// Upload packed quad data into a storage buffer and bind it for vertex
    /// pulling. `data` must be exactly `quad_count * 8` bytes.
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
        let quads = Arc::new(self.upload_buffer(label, data, vk::BufferUsageFlags::STORAGE_BUFFER));
        PulledMesh::new(self, quads, quad_count)
    }

    /// Bind an existing storage buffer (for example compute output) for
    /// vertex pulling without re-uploading it to the device.
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

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // Async ring slots own their fences; the command pool destroys
            // the buffers, but every slot's fence needs an explicit
            // destroy before the device goes away.
            for slot in self.async_ring.lock().unwrap().slots.iter_mut().flatten() {
                self.device.destroy_fence(slot.fence, None);
            }
            self.device.destroy_command_pool(self.transient_pool, None);
            // The allocator must release its blocks before the device goes away.
            drop(self.allocator.get_mut().expect("allocator lock").take());
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Load the Vulkan loader: an explicit `DREAM_VULKAN_LIBRARY` override
/// first, then the system library, then the well-known macOS install
/// paths (MoltenVK ships `libvulkan.dylib`, which the default search
/// rules do not always find). Returns the entry plus a note naming where
/// it came from for diagnostics.
fn load_vulkan_entry() -> (Entry, String) {
    let mut tried = Vec::new();
    // An explicit override wins over every default: a host pointing at a
    // specific loader gets exactly that one or a clear error.
    if let Ok(override_path) = std::env::var("DREAM_VULKAN_LIBRARY") {
        let path = std::path::PathBuf::from(&override_path);
        tried.push(override_path.clone());
        if let Ok(entry) = unsafe { Entry::load_from(&path) } {
            return (entry, override_path);
        }
    }
    // The system loader first, then the well-known macOS install paths
    // (MoltenVK ships `libvulkan.dylib`, which the default search rules do
    // not always find).
    tried.push("system loader".to_string());
    if let Ok(entry) = unsafe { Entry::load() } {
        return (entry, "system loader".to_string());
    }
    #[cfg(target_os = "macos")]
    {
        for path in [
            "/opt/homebrew/lib/libvulkan.dylib",
            "/usr/local/lib/libvulkan.dylib",
        ] {
            tried.push(path.to_string());
            if let Ok(entry) = unsafe { Entry::load_from(std::path::Path::new(path)) } {
                return (entry, path.to_string());
            }
        }
    }
    panic!("no Vulkan loader found (tried: {})", tried.join(", "));
}

unsafe fn pick_physical_device(
    instance: &Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> (vk::PhysicalDevice, u32, bool) {
    let devices = unsafe { instance.enumerate_physical_devices() }.expect("enumerate GPUs");
    let mut rejected = Vec::new();
    let mut fallback = None;
    for physical_device in devices {
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "unknown".to_string());
        // Vulkan 1.3 is a hard requirement: dynamic rendering is issued
        // through the core entry points everywhere in the engine, so a 1.2
        // adapter would need a separate dispatch path we deliberately do
        // not pretend to support.
        if props.api_version < vk::make_api_version(0, 1, 3, 0) {
            rejected.push(format!("{name}: Vulkan < 1.3"));
            continue;
        }
        // dynamicRendering must actually be supported, not just implied by
        // the version number.
        let mut features_13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features_13);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
        if features_13.dynamic_rendering != vk::TRUE {
            rejected.push(format!("{name}: no dynamicRendering feature"));
            continue;
        }

        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let mut chosen = None;
        for (index, family) in families.iter().enumerate() {
            // The thumbnail pass is compute: the queue must do graphics,
            // compute AND present.
            let capable = family
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);
            let present = unsafe {
                surface_loader
                    .get_physical_device_surface_support(physical_device, index as u32, surface)
                    .unwrap_or(false)
            };
            if capable && present {
                chosen = Some(index as u32);
                break;
            }
        }
        let Some(family) = chosen else {
            rejected.push(format!("{name}: no graphics+compute+present queue"));
            continue;
        };
        // The scene pass needs per-attachment blend state: colour blends for
        // LoD crossfades while the velocity attachment must stay unblended.
        let features = unsafe { instance.get_physical_device_features(physical_device) };
        if features.independent_blend != vk::TRUE {
            rejected.push(format!("{name}: no independentBlend"));
            continue;
        }
        // MoltenVK requires VK_KHR_portability_subset at device creation;
        // enable it when (and only when) the device lists it.
        let portability_subset = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        e.extension_name_as_c_str()
                            .map(|n| n == c"VK_KHR_portability_subset")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        };

        let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let candidate = (physical_device, family, portability_subset);
        if discrete {
            for why in &rejected {
                eprintln!("GPU candidate rejected: {why}");
            }
            return candidate;
        }
        fallback.get_or_insert(candidate);
    }
    let reasons = if rejected.is_empty() {
        "none enumerated".to_string()
    } else {
        rejected.join("; ")
    };
    fallback.unwrap_or_else(|| {
        panic!("no suitable Vulkan 1.3 device (graphics+compute+present, dynamicRendering, independentBlend). Rejected: {reasons}")
    })
}

// --- Handles ---------------------------------------------------------------

/// A game- or engine-owned buffer. Frees its Vulkan memory on drop.
pub struct GpuBuffer {
    ctx: Arc<GpuContext>,
    pub buffer: vk::Buffer,
    allocation: Option<Allocation>,
    pub size: u64,
}

/// An engine-owned image with a view. Frees its Vulkan memory on drop.
pub struct GpuImage {
    ctx: Arc<GpuContext>,
    pub image: vk::Image,
    pub view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl GpuImage {
    /// Bytes the allocator gave this image's backing allocation.
    pub fn allocation_size(&self) -> u64 {
        self.allocation.as_ref().map(|a| a.size()).unwrap_or(0)
    }
}

/// Find a memory type index for a resource. Retained for future explicit
/// allocations; gpu-allocator currently picks types on our behalf.
#[allow(dead_code)]
pub fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        let supported = type_filter & (1 << i) != 0;
        let has = props.memory_types[i as usize]
            .property_flags
            .contains(flags);
        supported && has
    })
}

/// Convenience for building a null-terminated name pointer list.
#[allow(dead_code)]
pub fn ext_names(names: &[&CStr]) -> Vec<*const c_char> {
    names.iter().map(|n| n.as_ptr()).collect()
}

pub(crate) fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

/// Whether an allocation failure justifies growing another descriptor pool
/// chunk. Only pool-local exhaustion does; device loss or real OOM must fail
/// loudly instead of looping forever.
pub(crate) fn pool_exhaustion_is_growable(err: vk::Result) -> bool {
    matches!(
        err,
        vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL
    )
}

#[cfg(test)]
mod tests {
    use ash::vk;

    #[test]
    fn only_pool_exhaustion_is_growable() {
        for err in [
            vk::Result::ERROR_OUT_OF_POOL_MEMORY,
            vk::Result::ERROR_FRAGMENTED_POOL,
        ] {
            assert!(super::pool_exhaustion_is_growable(err), "{err:?}");
        }
        for err in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_DEVICE_LOST,
            vk::Result::ERROR_UNKNOWN,
        ] {
            assert!(!super::pool_exhaustion_is_growable(err), "{err:?}");
        }
    }
}
