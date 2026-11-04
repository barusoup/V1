//! Asynchronous streaming uploads into a [`GpuTexture`].
//!
//! [`GpuTexture::from_rgba8`] is the right tool for data uploaded once: it
//! records a copy, submits it, and waits. A video frame changes every tick,
//! and waiting there would stall the CPU on the previous frame's rendering.
//! [`TextureUploader`] keeps a small ring of host-visible staging buffers and
//! submits each copy behind the frames already queued, returning immediately.
//! The caller draws the texture in the frame it submits next; the queue's
//! in-order execution means the copy is visible to that draw.
//!
//! The resource cost is one staging buffer per ring slot, not per texture and
//! not per frame. Two slots are enough for one upload per rendered frame; a
//! slot is only waited on if the GPU has fallen two uploads behind.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use dream::gfx::vulkan::GpuContext;
//! # use dream::texture::{GpuTexture, TextureOptions, TextureUploader};
//! # fn example(ctx: Arc<GpuContext>, texture: &GpuTexture) {
//! let mut uploader = TextureUploader::new(&ctx);
//! let pixels = vec![0u8; 1920 * 1080 * 4];
//! uploader.upload(texture, &pixels);
//! // Record this frame's draw with `texture`; the copy is already ahead of it.
//! # }
//! ```

use super::gpu::GpuTexture;
use crate::gfx::vulkan::{GpuBuffer, GpuContext, image_barrier};
use ash::vk;
use std::sync::Arc;

/// In-flight uploads the uploader can keep ahead of the renderer.
const UPLOAD_SLOTS: usize = 2;

struct UploadSlot {
    staging: Option<GpuBuffer>,
    capacity: u64,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Serial of the copy submitted from this slot. `None` once retired.
    serial: Option<u64>,
}

/// Reusable staging memory and submission ring for dynamic RGBA8 textures.
///
/// One uploader can feed any number of same-context textures. It is not
/// thread-safe and should live beside the renderer that submits frames.
pub struct TextureUploader {
    ctx: Arc<GpuContext>,
    pool: vk::CommandPool,
    slots: Vec<UploadSlot>,
    next: usize,
}

impl TextureUploader {
    /// Create an uploader on `ctx`. Allocates command buffers and fences, but
    /// no pixel staging memory until the first upload.
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let pool = unsafe {
            ctx.device().create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.queue_family())
                    .flags(
                        vk::CommandPoolCreateFlags::TRANSIENT
                            | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ),
                None,
            )
        }
        .expect("create texture upload command pool");
        let cmds = unsafe {
            ctx.device().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(UPLOAD_SLOTS as u32),
            )
        }
        .expect("allocate texture upload command buffers");
        let slots = cmds
            .into_iter()
            .map(|cmd| UploadSlot {
                staging: None,
                capacity: 0,
                cmd,
                fence: unsafe {
                    ctx.device()
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                }
                .expect("create texture upload fence"),
                serial: None,
            })
            .collect();
        Self {
            ctx: Arc::clone(ctx),
            pool,
            slots,
            next: 0,
        }
    }

    /// Queue `pixels` (width * height * 4 RGBA8 bytes) into `texture`.
    ///
    /// The upload is submitted, not waited on. It is safe to draw `texture`
    /// immediately after this call: the draw is submitted later on the same
    /// queue. The texture must have exactly one mip level and an RGBA8
    /// format. Uploading to a texture that a previous frame still samples is
    /// also ordered correctly, because the copy is newer in queue order.
    pub fn upload(&mut self, texture: &GpuTexture, pixels: &[u8]) {
        assert!(
            Arc::ptr_eq(&self.ctx, &texture.ctx),
            "texture uploader and texture belong to different GpuContexts"
        );
        assert_eq!(
            texture.mip_levels, 1,
            "dynamic uploads need a single-mip texture"
        );
        assert!(
            matches!(
                texture.vk_format,
                vk::Format::R8G8B8A8_SRGB | vk::Format::R8G8B8A8_UNORM
            ),
            "dynamic uploads need an RGBA8 texture"
        );
        assert_eq!(
            pixels.len(),
            texture.width as usize * texture.height as usize * 4,
            "pixel buffer must match the texture's dimensions"
        );

        let index = self.next;
        self.next = (self.next + 1) % self.slots.len();
        self.wait_slot(index);

        let need = pixels.len() as u64;
        {
            let slot = &mut self.slots[index];
            if slot.capacity < need {
                // The wait above guarantees no submitted copy still reads the
                // old buffer before it is dropped.
                slot.staging = Some(
                    self.ctx
                        .create_transfer_buffer("dynamic texture staging", need),
                );
                slot.capacity = need;
            }
            slot.staging
                .as_mut()
                .expect("staging allocated above")
                .write(0, pixels);
        }

        let device = self.ctx.device();
        let slot = &self.slots[index];
        let staging = slot.staging.as_ref().expect("staging allocated");
        let image = texture.image.image;
        let extent = vk::Extent3D {
            width: texture.width,
            height: texture.height,
            depth: 1,
        };
        let range = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        unsafe {
            device
                .reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty())
                .expect("reset texture upload command buffer");
            device
                .reset_fences(&[slot.fence])
                .expect("reset texture upload fence");
            device
                .begin_command_buffer(
                    slot.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin texture upload");
            // Previous frames may still be sampling this image; the barrier
            // waits for those reads before the transfer overwrites it.
            image_barrier(
                device,
                slot.cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::ImageAspectFlags::COLOR,
            );
            let region = vk::BufferImageCopy::default()
                .image_subresource(range)
                .image_extent(extent);
            device.cmd_copy_buffer_to_image(
                slot.cmd,
                staging.buffer,
                image,
                vk::ImageLayout::GENERAL,
                &[region],
            );
            // Make the copy visible to every shader stage of later draws.
            image_barrier(
                device,
                slot.cmd,
                image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::ImageAspectFlags::COLOR,
            );
            device
                .end_command_buffer(slot.cmd)
                .expect("end texture upload");
            let cmds = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let serial = self.ctx.submit_tracked(&submit, slot.fence);
            // The slot is borrowed immutably above; finish by recording its
            // serial through a short mutable borrow.
            let slot = &mut self.slots[index];
            slot.serial = Some(serial);
        }
    }

    /// The context this uploader creates staging memory on.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Uploads currently queued on the GPU. Zero means every staging buffer
    /// is free.
    pub fn pending(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.serial.is_some())
            .count()
    }

    /// Wait for every queued upload. Dropping the uploader does this too.
    pub fn wait_idle(&mut self) {
        for index in 0..self.slots.len() {
            self.wait_slot(index);
        }
    }

    fn wait_slot(&mut self, index: usize) {
        let slot = &mut self.slots[index];
        let Some(serial) = slot.serial.take() else {
            return;
        };
        unsafe {
            self.ctx
                .device()
                .wait_for_fences(&[slot.fence], true, u64::MAX)
                .expect("wait for texture upload");
        }
        self.ctx.confirm_completed(serial);
    }
}

impl Drop for TextureUploader {
    fn drop(&mut self) {
        for index in 0..self.slots.len() {
            self.wait_slot(index);
        }
        unsafe {
            for slot in &self.slots {
                self.ctx.device().destroy_fence(slot.fence, None);
            }
            self.ctx.device().destroy_command_pool(self.pool, None);
        }
    }
}
