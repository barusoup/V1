use super::{AsyncSlot, GpuContext};
use ash::vk;
use std::sync::atomic::Ordering;

impl GpuContext {
    /// --- Submission serials ------------------------------------------------
    ///
    /// Every `queue_submit` on this queue goes through `submit_tracked`,
    /// which assigns a serial in actual submission order. The *owner* of
    /// the fence -- renderer frame slot, async ring slot, or one-shot --
    /// confirms completion after observing it signalled; `completed_serial`
    /// is a high-water mark, sound because this queue is in-order.
    pub fn submitted_serial(&self) -> u64 {
        self.submitted_serial.load(Ordering::Relaxed)
    }

    /// Highest submission serial whose owner observed its fence signalled.
    pub fn completed_serial(&self) -> u64 {
        self.completed_serial.load(Ordering::Relaxed)
    }
    pub fn blocking_submissions(&self) -> u64 {
        self.blocking_submissions.load(Ordering::Relaxed)
    }

    /// Submit `submit` with `fence` under the queue lock; returns the
    /// assigned serial. The caller owns `fence` and must call
    /// `confirm_completed(serial)` only after the fence is known signalled
    /// (wait or `get_fence_status` -- never before).
    /// A serial is assigned ONLY on successful submit, so the watermark and
    /// pending serials never reference a failed submission.
    pub fn try_submit_tracked(
        &self,
        submit: &vk::SubmitInfo,
        fence: vk::Fence,
    ) -> Result<u64, vk::Result> {
        let _queue = self.queue_lock.lock().unwrap();
        unsafe {
            self.device
                .queue_submit(self.queue, &[*submit], fence)
                .map(|_| self.submitted_serial.fetch_add(1, Ordering::Relaxed) + 1)
        }
    }

    /// Convenience wrapper: panics on submit failure (used by one-shot and
    /// async work where a dead queue cannot be reported upstream).
    pub fn submit_tracked(&self, submit: &vk::SubmitInfo, fence: vk::Fence) -> u64 {
        self.try_submit_tracked(submit, fence)
            .expect("queue submit")
    }

    /// Mark `serial` complete after its fence was observed signalled.
    /// In-order queue: earlier serials are necessarily complete too.
    pub fn confirm_completed(&self, serial: u64) {
        self.completed_serial.fetch_max(serial, Ordering::Relaxed);
    }

    /// True when every submission up to and including `serial` completed.
    pub fn submission_complete(&self, serial: u64) -> bool {
        self.completed_serial() >= serial
    }

    /// Reusable ring slot for one-shot work that must not block: record into
    /// a persistent command buffer and submit it, returning the tracking
    /// serial, or `None` when the ring is full (caller retries next poll).
    /// Slot reuse confirms the slot's own previous serial first.
    pub fn submit_async(&self, record: impl FnOnce(vk::CommandBuffer)) -> Option<u64> {
        let mut ring = self.async_ring.lock().unwrap();
        let mut index = None;
        for (i, slot) in ring.slots.iter_mut().enumerate() {
            match slot {
                None => {
                    index = Some(i);
                    break;
                }
                Some(slot) => {
                    let free = match slot.serial {
                        None => true,
                        Some(serial) if self.submission_complete(serial) => true,
                        Some(serial) => unsafe {
                            // Owner-side confirmation: only this slot's own
                            // fence may prove its serial completed.
                            if self.device.get_fence_status(slot.fence).unwrap_or(false) {
                                self.confirm_completed(serial);
                                true
                            } else {
                                false
                            }
                        },
                    };
                    if free {
                        index = Some(i);
                        break;
                    }
                }
            }
        }
        let index = index?;
        let _transient = self.transient_lock.lock().unwrap();
        let slot = ring.slots[index].get_or_insert_with(|| unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.transient_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc_info).unwrap()[0];
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            AsyncSlot {
                cmd,
                fence,
                serial: None,
            }
        });
        unsafe {
            self.device
                .reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.device.reset_fences(&[slot.fence]).unwrap();
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(slot.cmd, &begin).unwrap();
            record(slot.cmd);
            self.device.end_command_buffer(slot.cmd).unwrap();
            let cmds = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let serial = self.submit_tracked(&submit, slot.fence);
            slot.serial = Some(serial);
            Some(serial)
        }
    }

    /// Run a short command buffer and wait for it to finish (uploads,
    /// barriers, diagnostic readbacks). Serial-tracked like everything else.
    pub fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
        let _transient = self.transient_lock.lock().unwrap();
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.transient_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc_info).unwrap()[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin).unwrap();
            record(cmd);
            self.device.end_command_buffer(cmd).unwrap();

            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let serial = self.submit_tracked(&submit, fence);
            self.blocking_submissions.fetch_add(1, Ordering::Relaxed);
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();
            // Owner confirms: this fence signals exactly `serial`.
            self.confirm_completed(serial);
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.transient_pool, &cmds);
        }
    }
}
