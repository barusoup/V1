use super::gpu::GpuTexture;
use crate::gfx::metal::{GpuBuffer, GpuContext};
use crate::gfx::renderer::SlotSync;
use block2::RcBlock;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLOrigin, MTLSize,
};
use std::ptr::NonNull;
use std::sync::Arc;

const UPLOAD_SLOTS: usize = 2;

struct UploadSlot {
    staging: Option<GpuBuffer>,
    capacity: u64,
    sync: Arc<SlotSync>,
    pending: bool,
}

pub struct TextureUploader {
    ctx: Arc<GpuContext>,
    slots: Vec<UploadSlot>,
    next: usize,
}

impl TextureUploader {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let slots = (0..UPLOAD_SLOTS)
            .map(|_| UploadSlot {
                staging: None,
                capacity: 0,
                sync: SlotSync::new(),
                pending: false,
            })
            .collect();
        Self {
            ctx: Arc::clone(ctx),
            slots,
            next: 0,
        }
    }

    pub fn upload(&mut self, texture: &GpuTexture, pixels: &[u8]) {
        let (w, h) = texture.size();
        let bytes = (w as u64) * (h as u64) * 4;
        assert_eq!(
            pixels.len() as u64,
            bytes,
            "TextureUploader::upload: expected {bytes} bytes for {w}x{h} RGBA8, got {}",
            pixels.len()
        );
        let idx = self.next;
        self.next = (idx + 1) % self.slots.len();
        let slot = &mut self.slots[idx];

        if slot.pending {
            let _ = slot.sync.wait(None);
            slot.pending = false;
        }
        if slot.capacity < bytes {
            slot.staging = Some(self.ctx.create_host_buffer("texture upload staging", bytes));
            slot.capacity = bytes;
        }
        slot.staging
            .as_mut()
            .expect("staging allocated")
            .write_bytes(0, pixels);

        let cmd = self
            .ctx
            .queue()
            .commandBuffer()
            .expect("create upload command buffer");
        let blit = cmd.blitCommandEncoder().expect("blit encoder");
        let row_bytes = (w as usize) * 4;
        let staging_buf = &slot.staging.as_ref().unwrap().buffer;
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                staging_buf,
                0,
                row_bytes,
                row_bytes * (h as usize),
                MTLSize {
                    width: w as usize,
                    height: h as usize,
                    depth: 1,
                },
                texture.raw_texture(),
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        if texture.mip_levels() > 1 {
            blit.generateMipmapsForTexture(texture.raw_texture());
        }
        blit.endEncoding();

        let serial = self.ctx.next_serial();
        let sync = Arc::clone(&slot.sync);
        sync.mark_pending(serial);
        let handler = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            sync.mark_done(0.0);
        });
        unsafe {
            cmd.addCompletedHandler(RcBlock::as_ptr(&handler));
        }
        cmd.commit();
        slot.pending = true;
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    pub fn pending(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.pending && !s.sync.state.lock().unwrap().done)
            .count()
    }

    pub fn wait_idle(&mut self) {
        for slot in &mut self.slots {
            if slot.pending {
                let _ = slot.sync.wait(None);
                slot.pending = false;
            }
        }
    }
}

impl Drop for TextureUploader {
    fn drop(&mut self) {
        self.wait_idle();
    }
}
