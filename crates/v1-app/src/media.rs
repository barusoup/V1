// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Windows System Media Transport Controls (SMTC).
//!
//! This is what makes earbuds, keyboard media keys, and the Windows media flyout drive the app.
//! Bluetooth transport buttons do not arrive as ordinary key messages: Windows only delivers them
//! to the media app that owns the active SMTC session, so an app that never registers one simply
//! sees nothing.
//!
//! The session is tied to the window. Transport buttons arrive on a COM thread and are forwarded to
//! the UI thread through the event loop proxy; the UI then applies them like a press on the bar.
//! Display metadata and the timeline are pushed back so the flyout shows the track and seeks land
//! in the right place. Everything here is best effort: if the OS refuses SMTC, the app still runs.

/// A transport action requested by the OS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Command {
    Play,
    Pause,
    Toggle,
    Next,
    Previous,
    Stop,
    /// Seek to this position, in seconds.
    Seek(f64),
}

/// What the SMTC session should show. Borrowed from the track view, so building it every frame
/// copies nothing.
#[derive(Debug, Clone, Copy)]
pub struct MediaInfo<'a> {
    pub title: &'a str,
    pub artist: &'a str,
    pub album: Option<&'a str>,
    /// Playback position and length, in seconds.
    pub position: f64,
    pub duration: f64,
    pub playing: bool,
}

#[cfg(windows)]
pub use imp::MediaControls;

#[cfg(not(windows))]
pub struct MediaControls;

#[cfg(not(windows))]
impl MediaControls {
    pub fn new<F>(_window: &winit::window::Window, _on_command: F) -> Option<Self>
    where
        F: Fn(Command) + Send + 'static,
    {
        None
    }

    pub fn update(&mut self, _info: Option<&MediaInfo<'_>>) {}
}

#[cfg(windows)]
mod imp {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use windows::core::{Result as WinResult, HSTRING};
    use windows::Foundation::{EventRegistrationToken, TimeSpan, TypedEventHandler};
    use windows::Media::{
        MediaPlaybackStatus, MediaPlaybackType, PlaybackPositionChangeRequestedEventArgs,
        SystemMediaTransportControls, SystemMediaTransportControlsButton,
        SystemMediaTransportControlsButtonPressedEventArgs, SystemMediaTransportControlsTimelineProperties,
    };
    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::WinRT::{
        ISystemMediaTransportControlsInterop, RoGetActivationFactory, RoInitialize,
        RO_INIT_SINGLETHREADED,
    };
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    use super::{Command, MediaInfo};

    /// How often the timeline is refreshed while playing. The flyout interpolates between updates,
    /// so once a second keeps the scrubber right without crossing into COM on every frame.
    const TIMELINE_EVERY: Duration = Duration::from_secs(1);

    pub struct MediaControls {
        controls: SystemMediaTransportControls,
        button_token: EventRegistrationToken,
        seek_token: Option<EventRegistrationToken>,
        /// The title/artist/album currently pushed to the flyout.
        shown: Option<(String, String, Option<String>)>,
        status: Option<MediaPlaybackStatus>,
        last_timeline: Option<Instant>,
        cleared: bool,
    }

    impl MediaControls {
        /// Attach an SMTC session to `window`. `on_command` runs when the OS sends a transport
        /// button; it may be called from a COM thread, so it must be thread-safe.
        pub fn new<F>(window: &winit::window::Window, on_command: F) -> Option<Self>
        where
            F: Fn(Command) + Send + Sync + 'static,
        {
            let hwnd = window_hwnd(window)?;
            // Winit has already made this thread a single-threaded COM apartment. Matching the
            // apartment keeps RoInitialize from changing it under winit; an error (including
            // "already initialized") is harmless.
            unsafe {
                let _ = RoInitialize(RO_INIT_SINGLETHREADED);
            }
            let class = HSTRING::from("Windows.Media.SystemMediaTransportControls");
            let interop: ISystemMediaTransportControlsInterop =
                unsafe { RoGetActivationFactory(&class) }.ok()?;
            let controls: SystemMediaTransportControls =
                unsafe { interop.GetForWindow(hwnd) }.ok()?;

            let _ = controls.SetIsEnabled(true);
            let _ = controls.SetIsPlayEnabled(true);
            let _ = controls.SetIsPauseEnabled(true);
            let _ = controls.SetIsStopEnabled(true);
            let _ = controls.SetIsNextEnabled(true);
            let _ = controls.SetIsPreviousEnabled(true);
            let _ = controls.SetIsFastForwardEnabled(false);
            let _ = controls.SetIsRewindEnabled(false);
            // Do not take over channel or record buttons.
            let _ = controls.SetIsChannelUpEnabled(false);
            let _ = controls.SetIsChannelDownEnabled(false);
            let _ = controls.SetIsRecordEnabled(false);

            let callback: Arc<dyn Fn(Command) + Send + Sync> = Arc::new(on_command);

            let button_callback = Arc::clone(&callback);
            let handler = TypedEventHandler::<
                SystemMediaTransportControls,
                SystemMediaTransportControlsButtonPressedEventArgs,
            >::new(move |_sender, args| {
                if let Some(args) = args.as_ref() {
                    if let WinResult::Ok(button) = args.Button() {
                        if let Some(command) = command_for(button) {
                            button_callback(command);
                        }
                    }
                }
                Ok(())
            });
            let button_token = controls.ButtonPressed(&handler).ok()?;

            // Seek requests from the media flyout, if the interface is available on this Windows.
            let seek_callback = Arc::clone(&callback);
            let seek_handler = TypedEventHandler::<
                SystemMediaTransportControls,
                PlaybackPositionChangeRequestedEventArgs,
            >::new(move |_sender, args| {
                if let Some(args) = args.as_ref() {
                    if let Ok(position) = args.RequestedPlaybackPosition() {
                        seek_callback(Command::Seek(ticks_to_seconds(position)));
                    }
                }
                Ok(())
            });
            let seek_token = controls.PlaybackPositionChangeRequested(&seek_handler).ok();

            Some(Self {
                controls,
                button_token,
                seek_token,
                shown: None,
                status: None,
                last_timeline: None,
                cleared: false,
            })
        }

        /// Push the current track and position to the media flyout. Cheap to call every frame: only
        /// a track change, a play/pause change, and the once-a-second timeline actually cross COM.
        pub fn update(&mut self, info: Option<&MediaInfo<'_>>) {
            let Some(info) = info else {
                self.clear();
                return;
            };
            self.cleared = false;

            let key = (
                info.title.to_owned(),
                info.artist.to_owned(),
                info.album.map(str::to_owned),
            );
            if self.shown.as_ref() != Some(&key) {
                let mut updated = false;
                if let Ok(updater) = self.controls.DisplayUpdater() {
                    let _ = updater.SetType(MediaPlaybackType::Music);
                    if let Ok(music) = updater.MusicProperties() {
                        let _ = music.SetTitle(&HSTRING::from(info.title));
                        let _ = music.SetArtist(&HSTRING::from(info.artist));
                        if let Some(album) = info.album.filter(|album| !album.is_empty()) {
                            let _ = music.SetAlbumTitle(&HSTRING::from(album));
                        }
                    }
                    updated = updater.Update().is_ok();
                }
                if updated {
                    self.shown = Some(key);
                    // Force a timeline push with the new duration.
                    self.last_timeline = None;
                }
            }

            let status = if info.playing {
                MediaPlaybackStatus::Playing
            } else {
                MediaPlaybackStatus::Paused
            };
            if self.status != Some(status)
                && self.controls.SetPlaybackStatus(status).is_ok()
            {
                self.status = Some(status);
            }

            if self
                .last_timeline
                .is_none_or(|at| at.elapsed() >= TIMELINE_EVERY)
            {
                self.set_timeline(info);
                self.last_timeline = Some(Instant::now());
            }
        }

        /// Say that nothing is playing, so the flyout stops showing the last song.
        fn clear(&mut self) {
            if self.cleared {
                return;
            }
            let _ = self.controls.SetPlaybackStatus(MediaPlaybackStatus::Stopped);
            let mut cleared = true;
            if let Ok(updater) = self.controls.DisplayUpdater() {
                let _ = updater.ClearAll();
                cleared = updater.Update().is_ok();
            }
            if cleared {
                self.shown = None;
                self.status = None;
                self.last_timeline = None;
                self.cleared = true;
            }
        }

        fn set_timeline(&self, info: &MediaInfo<'_>) {
            let Ok(timeline) = SystemMediaTransportControlsTimelineProperties::new() else {
                return;
            };
            let duration = seconds_to_ticks(info.duration);
            let position = seconds_to_ticks(info.position).clamp(0, duration);
            let _ = timeline.SetStartTime(TimeSpan { Duration: 0 });
            let _ = timeline.SetEndTime(TimeSpan { Duration: duration });
            let _ = timeline.SetMinSeekTime(TimeSpan { Duration: 0 });
            let _ = timeline.SetMaxSeekTime(TimeSpan { Duration: duration });
            let _ = timeline.SetPosition(TimeSpan { Duration: position });
            let _ = self.controls.UpdateTimelineProperties(&timeline);
        }
    }

    impl Drop for MediaControls {
        fn drop(&mut self) {
            if let Some(token) = self.seek_token.take() {
                let _ = self
                    .controls
                    .RemovePlaybackPositionChangeRequested(token);
            }
            let _ = self.controls.RemoveButtonPressed(self.button_token);
            let _ = self.controls.SetIsEnabled(false);
        }
    }

    /// The window's native handle, as the interop interface wants it.
    fn window_hwnd(window: &winit::window::Window) -> Option<HWND> {
        let handle = window.window_handle().ok()?;
        match handle.as_raw() {
            RawWindowHandle::Win32(win32) => {
                Some(HWND(win32.hwnd.get() as *mut core::ffi::c_void))
            }
            _ => None,
        }
    }

    fn command_for(button: SystemMediaTransportControlsButton) -> Option<Command> {
        // The constants are plain i32 wrappers, so compare by value.
        if button == SystemMediaTransportControlsButton::Play {
            Some(Command::Play)
        } else if button == SystemMediaTransportControlsButton::Pause {
            Some(Command::Pause)
        } else if button == SystemMediaTransportControlsButton::Stop {
            Some(Command::Stop)
        } else if button == SystemMediaTransportControlsButton::Next {
            Some(Command::Next)
        } else if button == SystemMediaTransportControlsButton::Previous {
            Some(Command::Previous)
        } else {
            None
        }
    }

    /// 100-nanosecond ticks to seconds.
    fn ticks_to_seconds(time: TimeSpan) -> f64 {
        time.Duration as f64 / 10_000_000.0
    }

    fn seconds_to_ticks(seconds: f64) -> i64 {
        if !seconds.is_finite() || seconds <= 0.0 {
            return 0;
        }
        (seconds * 10_000_000.0)
            .round()
            .clamp(0.0, i64::MAX as f64) as i64
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn transport_buttons_map_to_the_app_s_commands() {
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Play),
                Some(Command::Play)
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Pause),
                Some(Command::Pause)
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Next),
                Some(Command::Next)
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Previous),
                Some(Command::Previous)
            );
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::Stop),
                Some(Command::Stop)
            );
            // Record, fast-forward, rewind and channel buttons are not ours.
            assert_eq!(command_for(SystemMediaTransportControlsButton::Record), None);
            assert_eq!(
                command_for(SystemMediaTransportControlsButton::FastForward),
                None
            );
        }

        #[test]
        fn timeline_positions_convert_between_seconds_and_ticks() {
            assert_eq!(seconds_to_ticks(0.0), 0);
            assert_eq!(seconds_to_ticks(-3.0), 0);
            assert_eq!(seconds_to_ticks(f64::NAN), 0);
            assert_eq!(seconds_to_ticks(1.5), 15_000_000);
            assert!((ticks_to_seconds(TimeSpan { Duration: 15_000_000 }) - 1.5).abs() < 1e-9);
        }
    }
}
