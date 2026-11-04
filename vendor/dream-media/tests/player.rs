#![allow(dead_code)]

mod common;

use common::{FixtureDir, long_gop_clip, rotated_clip, sample_audio, sample_clip, tools};
use dream_media::{Player, PlayerOptions, Status};
use std::time::{Duration, Instant};

struct Run {
    frames: Vec<(u32, u32, i64)>,
    elapsed: Duration,
}

fn drive(player: &mut Player, limit: Duration, mut done: impl FnMut(&Player) -> bool) -> Run {
    let started = Instant::now();
    let mut frames = Vec::new();
    loop {
        let now = Instant::now();
        if let Some(frame) = player.poll(now) {
            frames.push((frame.width(), frame.height(), frame.pts_ms()));
        }
        if done(player) {
            return Run {
                frames,
                elapsed: started.elapsed(),
            };
        }
        assert!(
            started.elapsed() < limit,
            "timed out in state {:?}",
            player.status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_display_rotation_swaps_the_decoded_size() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("rotation");
    let clip = rotated_clip(&tools, &dir);
    let info = dream_media::probe_with(&tools, &clip).unwrap();
    assert_eq!(info.size(), Some((240, 320)));

    let mut player = Player::open_with(
        &tools,
        &clip,
        PlayerOptions {
            max_size: (1920, 1080),
            audio: false,
            ..Default::default()
        },
    )
    .unwrap();
    let run = drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });
    assert!(!run.frames.is_empty());
    assert!(
        run.frames
            .iter()
            .all(|frame| frame.0 == 240 && frame.1 == 320)
    );
}

#[test]
fn changing_the_decode_bounds_restarts_at_the_same_position() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("resize");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((640, 480))).unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing && player.position_ms(Instant::now()) > 100
    });
    let now = Instant::now();
    player.set_max_size((160, 120), now);
    let run = drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing && player.position_ms(Instant::now()) >= 100
    });
    assert!(
        run.frames
            .iter()
            .all(|frame| frame.0 == 160 && frame.1 == 120),
        "resized frames: {:?}",
        run.frames
    );
}

#[test]
fn looping_starts_the_next_pass_without_ending() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("loop");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(
        &tools,
        &clip,
        PlayerOptions {
            max_size: (320, 240),
            audio: false,
            looping: true,
            ..Default::default()
        },
    )
    .unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });
    let started = Instant::now();
    while started.elapsed() < Duration::from_millis(2600) {
        let _ = player.poll(Instant::now());
        std::thread::sleep(Duration::from_millis(4));
    }
    assert_eq!(player.status(), Status::Playing);
    assert!(
        player.position_ms(Instant::now()) < 1200,
        "second pass should have restarted, position {}",
        player.position_ms(Instant::now())
    );
}

fn video_options(max: (u32, u32)) -> PlayerOptions {
    PlayerOptions {
        max_size: max,
        audio: false,
        ..Default::default()
    }
}

#[test]
fn a_clip_plays_to_its_end_at_its_size() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("play");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((640, 480))).unwrap();
    assert_eq!(player.info().size(), Some((320, 240)));
    let duration = player.duration_ms().expect("probe reports a duration");
    assert!((1900..=2100).contains(&duration), "duration was {duration}");

    let run = drive(&mut player, Duration::from_secs(20), |player| {
        player.status() == Status::Ended
    });
    assert!(
        run.frames
            .iter()
            .all(|frame| frame.0 == 320 && frame.1 == 240)
    );
    assert!(
        (35..=55).contains(&run.frames.len()),
        "a 2 s clip at 25 fps showed {} frames",
        run.frames.len()
    );
    assert!(
        run.elapsed >= Duration::from_millis(1800),
        "ended too early: {:?}",
        run.elapsed
    );
}

#[test]
fn a_clip_is_downscaled_to_the_bounds_it_is_given() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("downscale");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((160, 120))).unwrap();
    let run = drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });
    assert!(!run.frames.is_empty());
    assert!(
        run.frames
            .iter()
            .all(|frame| frame.0 == 160 && frame.1 == 120)
    );
    let _ = player.poll(Instant::now());
}

#[test]
fn a_seek_while_paused_shows_a_frame_at_the_position() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("paused-seek");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });

    let now = Instant::now();
    player.pause(now);
    player.seek(1000);
    let run = drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Paused && player.position_ms(Instant::now()) >= 1000
    });
    assert!(
        run.frames.iter().any(|frame| frame.2 >= 900),
        "no frame at the sought position: {:?}",
        run.frames
    );
    assert_eq!(player.position_ms(Instant::now()), 1000);
}

#[test]
fn a_seek_in_a_long_gop_transport_stream_still_shows_a_frame() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("long-gop");
    let clip = long_gop_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });

    let now = Instant::now();
    player.pause(now);
    player.seek(1500);
    let run = drive(&mut player, Duration::from_secs(60), |player| {
        player.status() == Status::Paused && player.position_ms(Instant::now()) >= 1500
    });
    assert!(
        run.frames.iter().any(|frame| frame.2 >= 1400),
        "the seek showed no frame: {:?}",
        run.frames
    );
}

#[test]
fn pausing_holds_the_position() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("pause");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing && player.position_ms(Instant::now()) > 200
    });

    let now = Instant::now();
    player.pause(now);
    let held = player.position_ms(now);
    let later = now + Duration::from_millis(300);
    assert_eq!(player.position_ms(later), held);
    assert_eq!(player.status(), Status::Paused);
    assert!(held > 0);
}

#[test]
fn play_after_the_end_starts_from_zero() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("replay");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    drive(&mut player, Duration::from_secs(20), |player| {
        player.status() == Status::Ended
    });

    player.play(Instant::now());
    let run = drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });
    assert!(!run.frames.is_empty());
    assert!(player.position_ms(Instant::now()) < 500);
}

#[test]
fn rubbish_is_rejected_or_fails_within_the_deadline() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("garbage");
    let path = dir.path("broken.mp4");
    std::fs::write(&path, b"this is not a video, only text").unwrap();
    match Player::open_with(&tools, &path, video_options((640, 480))) {
        Err(_) => {}
        Ok(mut player) => {
            drive(&mut player, Duration::from_secs(20), |player| {
                matches!(player.status(), Status::Failed(_))
            });
        }
    }
}

#[test]
fn dropping_a_playing_player_returns_promptly() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("drop");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });
    let started = Instant::now();
    drop(player);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "drop took {:?}",
        started.elapsed()
    );
}

#[test]
fn music_plays_to_its_end_with_no_picture() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("audio");
    let track = sample_audio(&tools, &dir);
    let mut player = match Player::open_with(
        &tools,
        &track,
        PlayerOptions {
            audio: true,
            ..Default::default()
        },
    ) {
        Ok(player) => player,
        // Headless CI and locked-down machines may have no output device.
        Err(_) => return,
    };
    assert!(player.info().is_audio_only());
    let run = drive(&mut player, Duration::from_secs(30), |player| {
        player.status() == Status::Ended
    });
    assert!(run.frames.is_empty());
    assert!(
        player
            .duration_ms()
            .is_some_and(|ms| (1400..=1600).contains(&ms))
    );
}

#[test]
fn video_with_audio_also_reaches_its_end() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("video-audio");
    let clip = sample_clip(&tools, &dir);
    let mut player = match Player::open_with(
        &tools,
        &clip,
        PlayerOptions {
            max_size: (320, 240),
            audio: true,
            ..Default::default()
        },
    ) {
        Ok(player) => player,
        Err(_) => return,
    };
    let run = drive(&mut player, Duration::from_secs(30), |player| {
        player.status() == Status::Ended
    });
    assert!(!run.frames.is_empty());
}

#[test]
fn the_first_frame_wake_is_bounded() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("wake");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });
    let now = Instant::now();
    if let Some(wake) = player.next_wake(now) {
        assert!(wake <= now + Duration::from_millis(120));
    }
    let _ = player.poll(now);
}

#[test]
fn seeking_while_playing_moves_the_picture_and_the_clock() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("seek-playing");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing
    });

    player.seek(1200);
    let started = Instant::now();
    let mut forward = None;
    while started.elapsed() < Duration::from_secs(15) {
        if let Some(frame) = player.poll(Instant::now())
            && frame.pts_ms() >= 1100
        {
            forward = Some(frame.pts_ms());
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(forward.is_some(), "no frame near the forward seek target");
    let position = player.position_ms(Instant::now());
    assert!(
        (1200..1700).contains(&position),
        "clock after forward seek: {position}"
    );

    player.seek(200);
    let started = Instant::now();
    let mut backward = None;
    while started.elapsed() < Duration::from_secs(15) {
        if let Some(frame) = player.poll(Instant::now())
            && frame.pts_ms() <= 600
        {
            backward = Some(frame.pts_ms());
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(backward.is_some(), "no frame near the backward seek target");
    assert!(
        player.position_ms(Instant::now()) < 1100,
        "clock did not move back: {}",
        player.position_ms(Instant::now())
    );
}

#[test]
fn changing_the_decode_bounds_while_paused_keeps_the_picture() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("resize-paused");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    drive(&mut player, Duration::from_secs(15), |player| {
        player.status() == Status::Playing && player.position_ms(Instant::now()) > 200
    });

    let now = Instant::now();
    player.pause(now);
    let held = player.position_ms(now);
    player.set_max_size((160, 120), now);

    let started = Instant::now();
    let mut resized = None;
    while started.elapsed() < Duration::from_secs(15) {
        if let Some(frame) = player.poll(Instant::now()) {
            resized = Some((frame.size(), frame.pts_ms()));
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let ((width, height), pts) = resized.expect("a frame at the new size");
    assert_eq!((width, height), (160, 120));
    assert!(
        pts + 50 >= held,
        "resized frame at {pts} ms jumped behind {held} ms"
    );
    assert_eq!(player.status(), Status::Paused);
}

#[test]
fn two_players_play_independently_and_one_can_be_dropped() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("two-players");
    let clip = sample_clip(&tools, &dir);
    let mut first = Player::open_with(&tools, &clip, video_options((160, 120))).unwrap();
    let mut second = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();

    let started = Instant::now();
    let (mut first_frames, mut second_frames) = (0, 0);
    while started.elapsed() < Duration::from_secs(15) && (first_frames == 0 || second_frames == 0) {
        if let Some(frame) = first.poll(Instant::now()) {
            assert_eq!(frame.size(), (160, 120));
            first_frames += 1;
        }
        if let Some(frame) = second.poll(Instant::now()) {
            assert_eq!(frame.size(), (320, 240));
            second_frames += 1;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        first_frames > 0 && second_frames > 0,
        "first {first_frames}, second {second_frames}"
    );

    let before = second.position_ms(Instant::now());
    drop(first);
    let mut later_frames = 0;
    let started = Instant::now();
    while started.elapsed() < Duration::from_millis(700) {
        if second.poll(Instant::now()).is_some() {
            later_frames += 1;
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    assert!(
        later_frames > 0 || second.position_ms(Instant::now()) > before,
        "the surviving player stopped: {:?}",
        second.status()
    );
}

#[test]
fn frame_timestamps_advance_monotonically() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("monotonic");
    let clip = sample_clip(&tools, &dir);
    let mut player = Player::open_with(&tools, &clip, video_options((320, 240))).unwrap();
    let run = drive(&mut player, Duration::from_secs(20), |player| {
        player.status() == Status::Ended
    });
    assert!(run.frames.len() >= 30, "only {} frames", run.frames.len());
    assert!(
        run.frames.windows(2).all(|pair| pair[0].2 <= pair[1].2),
        "timestamps went backwards: {:?}",
        run.frames
    );
}

#[test]
fn an_audio_track_can_seek_while_paused() {
    let Some(tools) = tools() else {
        return;
    };
    let dir = FixtureDir::new("audio-seek");
    let track = sample_audio(&tools, &dir);
    let mut player = match Player::open_with(
        &tools,
        &track,
        PlayerOptions {
            audio: true,
            ..Default::default()
        },
    ) {
        Ok(player) => player,
        Err(_) => return,
    };

    player.pause(Instant::now());
    player.seek(700);
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(15) {
        let _ = player.poll(Instant::now());
        if player.status() == Status::Paused && player.position_ms(Instant::now()) >= 700 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(player.status(), Status::Paused);
    let position = player.position_ms(Instant::now());
    assert!(
        (700..900).contains(&position),
        "paused audio position {position}"
    );

    player.play(Instant::now());
    drive(&mut player, Duration::from_secs(30), |player| {
        player.status() == Status::Ended
    });
}

#[test]
fn missing_tools_are_an_error_not_a_panic() {
    let dir = FixtureDir::new("missing-tools");
    let clip = dir.path("anything.mp4");
    std::fs::write(&clip, b"not a video").unwrap();
    let tools = dream_media::Tools::from_paths(dir.path("no-ffmpeg"), dir.path("no-ffprobe"));
    let error = match Player::open_with(&tools, &clip, video_options((320, 240))) {
        Ok(_) => panic!("missing tools should not open a player"),
        Err(error) => error,
    };
    assert!(!error.to_string().is_empty());
}
