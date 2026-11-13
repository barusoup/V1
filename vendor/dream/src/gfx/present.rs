/// How finished frames reach the screen.
///
/// - [`Self::Fifo`] is vsync: frames are queued and shown in order, the
///   renderer never draws a frame that will not be shown, the loop blocks
///   on the display and the GPU idles between refreshes. Lowest power. This
///   is the default and is always supported.
/// - [`Self::Mailbox`] is latest-frame-wins: lower latency, but frames that
///   are replaced before they are shown were still rendered, so an app that
///   never waits on the display must pace itself or it pins the GPU.
/// - [`Self::Immediate`] presents without waiting for vsync and may tear.
///
/// A mode the surface does not offer falls back to [`Self::Fifo`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PresentMode {
    /// Vertical sync (`FIFO`). The default.
    #[default]
    Fifo,
    /// Latest-frame-wins (`MAILBOX`), falling back to FIFO.
    Mailbox,
    /// No vsync (`IMMEDIATE`), falling back to FIFO. May tear.
    Immediate,
}
