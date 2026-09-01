//! NDL `DirectMedia` **v2** (webOS 5+): `NDL_DirectMediaLoad` plus
//! `NDL_DirectVideoPlay(buffer, size, pts)`, a render-buffer query, a flush and HDR mastering
//! metadata. The path every currently-working TV takes.
//!
//! Never calls `NDL_DirectVideoSetArea` — stutters above 1080p, and v2 sizes its own
//! punch-through plane (v1 can't; see [`super::v1`]).
use std::ffi::c_uint;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

use crate::core::media::{AudioFormat, AudioPlane, AudioSink, MediaClock, NotReady, Samples, VideoSink, VideoSinkCaps};

use super::{arm_load, ensure_init, ensure_not_poisoned, ffi, settle_before_retry, wait_load_completed};
use super::{lock_ffi, mark_frame_fed_logged, NdlCodec, LOAD_COMPLETED};

/// How long past the `NDL_DirectMediaLoad` CALL [`NdlVideo::ensure_loaded`] holds frames while
/// `LOADCOMPLETED` is missing.
const FEED_ANYWAY_AFTER: Duration = Duration::from_millis(1_000);

/// One empty Opus frame — `mariotaku/ss4s`'s `opus_empty_frame_211`. Its TOC declares STEREO,
/// matching the load; the generic `0xF8 0xFF 0xFE` declares mono. (A CX took both.)
const OPUS_SILENCE: [u8; 3] = [0xec, 0xff, 0xfe];

/// Packet duration of the prime's stamps (ms), matching the real audio plane's 48 kHz / 5 ms.
const PRIME_PACKET_MS: i64 = 5;

/// How far ahead of wall-clock the prime's stamps may run, in packets.
const PRIME_LEAD: i64 = 8;

/// Lead the real stream's stamps carry over the player clock. NDL needs a fed audio plane to pace
/// the picture; this is therefore a video-pacing depth, not merely an audio-latency setting.
const PLANE_LEAD_MS: i64 = PRIME_LEAD * PRIME_PACKET_MS;

/// Gap between clock-plane top-ups.
const PRIME_RETRY: Duration = Duration::from_millis(20);

/// The video feed can carry the software clock itself while pictures are flowing. Let its lead
/// decay by this much before another top-up, roughly preserving the old 20 ms metronome cadence
/// while removing cross-thread FFI-lock competition from the hot 4K120 path.
const VIDEO_CLOCK_ASSIST_MARGIN_MS: i64 = 20;

/// If video has not successfully fed for this long, the clock thread takes ownership back so a
/// network/capture drought cannot starve NDL's pacing plane.
const VIDEO_CLOCK_WATCHDOG_MS: i64 = 50;

/// Wait for the next fixed metronome deadline without letting work time accumulate into cadence.
fn sleep_to_next_clock_tick(next_tick: &mut Instant) {
    *next_tick += PRIME_RETRY;
    let mut now = Instant::now();
    while *next_tick <= now {
        *next_tick += PRIME_RETRY;
        now = Instant::now();
    }
    std::thread::sleep(*next_tick - now);
}

/// How long the clock plane waits for real offloaded audio before filling silence itself.
const REAL_FEED_GRACE_MS: i64 = 300;

/// The audio plane every V2 load asks for: Opus, stereo, 48 kHz.
fn plane_config() -> ffi::AudioUnion {
    ffi::AudioOpusInfo {
        kind: 3,
        unknown1: 0,
        channels: 2,
        unknown2: 0,
        sample_rate: 48.0,
        stream_header: std::ptr::null(),
        _padding: [0; 4],
    }
    .to_union()
}

/// One loaded NDL v2 video decode session. Dropping unloads it (not `NDL_DirectMediaQuit`).
pub struct NdlVideo {
    fns: &'static ffi::V2,
    /// PTS in ms since load (NDL's local clock, not wall-clock or host capture clock).
    load_instant: Instant,
    /// When `NDL_DirectMediaLoad` was issued.
    load_requested: Instant,
    /// Whether this load got an audio plane.
    audio: bool,
    /// Highest audio stamp fed so far (ms).
    last_audio_pts_ms: AtomicI64,
    /// Player-clock ms at the last real packet fed by [`Self::play_audio`].
    last_real_feed_ms: AtomicI64,
    /// While true, the software-audio route piggybacks silent clock top-ups on the video feed.
    /// The clock thread remains a watchdog but stays off the global NDL FFI lock while pictures
    /// are flowing, eliminating the hottest lock hand-off at 4K120.
    video_drives_clock: AtomicBool,
    /// Constant software-clock base captured when the metronome starts. The historical clock path
    /// intentionally keeps this offset from the load prime; the video-assisted path must target
    /// the exact same timeline rather than silently changing A/V sync.
    video_clock_base_ms: AtomicI64,
    /// Player-clock ms at the last successful `NDL_DirectVideoPlay`.
    last_video_feed_ms: AtomicI64,
    /// Whether this load has reported `LOADCOMPLETED`.
    load_confirmed: AtomicBool,
    /// HDR mastering metadata that arrived before the video plane had taken a frame.
    pending_hdr: Mutex<Option<ffi::HdrInfo>>,
    /// Last metadata actually handed to `NDL_DirectVideoSetHDRInfo`.
    applied_hdr: Mutex<Option<ffi::HdrInfo>>,
}

impl NdlVideo {
    /// Load NDL video stream. Calls `NDL_DirectMediaInit` on first use.
    pub fn load(app_id: &str, width: i32, height: i32, codec: NdlCodec, audio: bool) -> Result<Self> {
        ensure_not_poisoned()?;
        let fns = ffi::v2()?;
        ensure_init(app_id, true)?;
        let video = ffi::VideoInfo {
            width,
            height,
            kind: codec.ndl_type(),
            unknown1: 0,
        };
        if audio {
            let unloads_before = super::unload_count();
            match Self::try_load(fns, video, true) {
                Ok(loaded) if loaded.load_confirmed.load(Ordering::Relaxed) => return Ok(loaded),
                Ok(_) => tracing::warn!("NDL audio-enabled load failed (no LOADCOMPLETED) — retrying video-only"),
                Err(e) => tracing::warn!("NDL audio-enabled load failed ({e:#}) — retrying video-only"),
            }
            fns.unload();
            settle_before_retry(unloads_before);
        }
        Self::try_load(fns, video, false)
    }

    /// One `NDL_DirectMediaLoad` attempt, waited out to `LOADCOMPLETED`.
    fn try_load(fns: &'static ffi::V2, video: ffi::VideoInfo, audio: bool) -> Result<Self> {
        let mut info = ffi::DataInfo {
            video,
            audio: if audio { plane_config() } else { ffi::AudioUnion::SILENT },
        };
        arm_load();
        let load_requested = Instant::now();
        fns.load(&mut info, Some(super::on_load_state))?;
        let (primed_pts_ms, confirmed) = if audio {
            Self::prime_audio(fns)
        } else {
            (0, wait_load_completed())
        };
        Ok(Self {
            fns,
            load_instant: Instant::now(),
            load_requested,
            audio,
            last_audio_pts_ms: AtomicI64::new(primed_pts_ms),
            last_real_feed_ms: AtomicI64::new(0),
            video_drives_clock: AtomicBool::new(false),
            video_clock_base_ms: AtomicI64::new(0),
            last_video_feed_ms: AtomicI64::new(0),
            load_confirmed: AtomicBool::new(confirmed),
            pending_hdr: Mutex::new(None),
            applied_hdr: Mutex::new(None),
        })
    }

    /// Feed silent Opus packets until the audio-enabled load reports `LOADCOMPLETED`.
    fn prime_audio(fns: &'static ffi::V2) -> (i64, bool) {
        let silence = &OPUS_SILENCE[..];
        let start = Instant::now();
        let mut pts_ms = 0;
        while !LOAD_COMPLETED.fired() {
            if start.elapsed() >= super::LOAD_COMPLETE_TIMEOUT {
                tracing::warn!(
                    "NDL load: no LOADCOMPLETED within {:?} of priming {pts_ms}ms of silence",
                    super::LOAD_COMPLETE_TIMEOUT
                );
                return (pts_ms, false);
            }
            let target_ms = start.elapsed().as_millis() as i64 + PRIME_LEAD * PRIME_PACKET_MS;
            {
                let _ffi = lock_ffi();
                while pts_ms < target_ms {
                    if let Err(e) = fns.audio_play(silence, pts_ms) {
                        tracing::warn!("NDL audio prime rejected at {pts_ms}ms: {e:#}");
                        return (pts_ms, LOAD_COMPLETED.fired());
                    }
                    pts_ms += PRIME_PACKET_MS;
                }
            }
            super::poll_until(PRIME_RETRY, || LOAD_COMPLETED.fired());
        }
        tracing::info!(
            "NDL audio prime: LOADCOMPLETED after {:?} ({pts_ms}ms of silence)",
            start.elapsed()
        );
        (pts_ms, true)
    }

    pub fn has_audio_plane(&self) -> bool {
        self.audio
    }

    /// Feed one Opus packet to the audio plane, stamped on the player clock.
    pub fn play_audio(&self, packet: &[u8]) -> Result<()> {
        if !self.load_confirmed.load(Ordering::Relaxed) {
            return Ok(());
        }
        let now_ms = (self.elapsed_ns() / 1_000_000) as i64;
        let target_ms = now_ms + PLANE_LEAD_MS;
        {
            let _ffi = lock_ffi();
            let pts_ms = self
                .last_audio_pts_ms
                .fetch_max(target_ms, Ordering::Relaxed)
                .max(target_ms);
            self.fns.audio_play(packet, pts_ms)?;
        }
        self.last_real_feed_ms.store(now_ms, Ordering::Relaxed);
        Ok(())
    }

    /// Feed silence while the caller already owns the process-wide NDL FFI lock.
    fn feed_silence_locked(&self, from_ms: i64, target_ms: i64) -> Result<i64> {
        let silence = &OPUS_SILENCE[..];
        let mut pts_ms = from_ms.max(self.last_audio_pts_ms.load(Ordering::Relaxed));
        while pts_ms < target_ms {
            pts_ms += PRIME_PACKET_MS;
            if let Err(e) = self.fns.audio_play(silence, pts_ms) {
                self.last_audio_pts_ms.fetch_max(pts_ms, Ordering::Relaxed);
                return Err(e);
            }
        }
        self.last_audio_pts_ms.fetch_max(pts_ms, Ordering::Relaxed);
        Ok(pts_ms)
    }

    /// Feed silence up to a fixed lead over the player clock, returning the last stamp fed.
    /// The target is sampled after acquiring the NDL lock, so lock wait cannot consume the lead.
    fn burst_silence(&self, from_ms: i64, base_ms: i64) -> Result<i64> {
        let _ffi = lock_ffi();
        let now_ms = (self.elapsed_ns() / 1_000_000) as i64;
        self.feed_silence_locked(from_ms, base_ms + now_ms + PLANE_LEAD_MS)
    }

    /// Keep the audio plane fed until `stop`. On the software route, active video submissions
    /// carry the metronome inside the same NDL critical section; this thread becomes a watchdog.
    pub fn run_clock_plane(&self, stop: &std::sync::atomic::AtomicBool, yields_to_real: bool) {
        let base_ms = if yields_to_real {
            0
        } else {
            self.last_audio_pts_ms.load(Ordering::Relaxed)
        };
        if !yields_to_real {
            self.video_clock_base_ms.store(base_ms, Ordering::Relaxed);
            self.video_drives_clock.store(true, Ordering::Release);
            tracing::info!("NDL clock plane: video-assisted software metronome enabled");
        }

        let mut pts_ms = self.last_audio_pts_ms.load(Ordering::Relaxed);
        let mut filling = false;
        let mut next_tick = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            let now_ms = (self.elapsed_ns() / 1_000_000) as i64;

            if yields_to_real && now_ms - self.last_real_feed_ms.load(Ordering::Relaxed) < REAL_FEED_GRACE_MS {
                if filling {
                    tracing::info!("NDL clock plane: host audio resumed at {now_ms}ms — yielding");
                    filling = false;
                }
                sleep_to_next_clock_tick(&mut next_tick);
                continue;
            }

            if !yields_to_real && self.video_drives_clock.load(Ordering::Acquire) {
                let last_video_ms = self.last_video_feed_ms.load(Ordering::Relaxed);
                if last_video_ms > 0 && now_ms - last_video_ms < VIDEO_CLOCK_WATCHDOG_MS {
                    sleep_to_next_clock_tick(&mut next_tick);
                    continue;
                }
            }

            if yields_to_real && !filling {
                tracing::warn!(
                    "NDL clock plane: no host audio for {REAL_FEED_GRACE_MS}ms — filling silence \
                     to keep the picture paced (host capture is likely dead)"
                );
                filling = true;
            }

            match self.burst_silence(pts_ms, base_ms) {
                Ok(fed_to) => pts_ms = fed_to,
                Err(e) => {
                    if !yields_to_real {
                        self.video_drives_clock.store(false, Ordering::Release);
                    }
                    tracing::warn!("NDL clock plane stopping at {pts_ms}ms: {e:#}");
                    return;
                }
            }
            sleep_to_next_clock_tick(&mut next_tick);
        }
        if !yields_to_real {
            self.video_drives_clock.store(false, Ordering::Release);
        }
        tracing::info!("NDL clock plane ending at {pts_ms}ms");
    }

    /// How far the audio plane's stamps currently run ahead of the player clock, in ms.
    pub fn audio_plane_lead_ms(&self) -> i64 {
        self.last_audio_pts_ms.load(Ordering::Relaxed) - (self.elapsed_ns() / 1_000_000) as i64
    }

    /// Nanoseconds since `load()` (NDL PTS domain).
    pub(crate) fn elapsed_ns(&self) -> u64 {
        self.load_instant.elapsed().as_nanos() as u64
    }

    fn ensure_loaded(&self) -> Result<()> {
        if self.load_confirmed.load(Ordering::Relaxed) {
            return Ok(());
        }
        let elapsed = self.load_requested.elapsed();
        if LOAD_COMPLETED.fired() {
            tracing::info!("NDL LOADCOMPLETED landed {elapsed:?} after load");
        } else if elapsed >= FEED_ANYWAY_AFTER {
            tracing::warn!("NDL: still no LOADCOMPLETED {elapsed:?} after the load — feeding anyway");
        } else {
            return Err(NotReady.into());
        }
        self.load_confirmed.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn pending_hdr(&self) -> MutexGuard<'_, Option<ffi::HdrInfo>> {
        self.pending_hdr.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn replay_pending_hdr(&self) {
        let mut pending = self.pending_hdr();
        if let Some(info) = pending.take() {
            tracing::info!("NDL: applying HDR metadata held until the first accepted frame");
            if let Err(e) = self.apply_hdr_info(info) {
                tracing::warn!("NDL: applying held HDR metadata failed: {e:#}");
            }
        }
    }

    fn apply_hdr_info(&self, info: ffi::HdrInfo) -> Result<()> {
        let mut applied = self.applied_hdr.lock().unwrap_or_else(PoisonError::into_inner);
        if *applied == Some(info) {
            return Ok(());
        }
        self.set_hdr_info(info)?;
        *applied = Some(info);
        Ok(())
    }

    fn set_hdr_info(&self, info: ffi::HdrInfo) -> Result<()> {
        let _ffi = lock_ffi();
        self.fns.set_hdr_info(info)
    }

    /// Feed one access unit at `pts_ns` (ns since `load()`). On the software-audio route the video
    /// thread also services the silent NDL clock before submitting the picture, while it already
    /// owns the same FFI lock. This serializes the calls exactly as before but removes the competing
    /// clock thread from the 120 Hz hot path.
    pub fn play(&self, au: &[u8], pts_ns: u64) -> Result<()> {
        self.ensure_loaded()?;
        let pts_ms = (pts_ns / 1_000_000) as i64;
        let first_frame = {
            let _ffi = lock_ffi();

            if self.video_drives_clock.load(Ordering::Acquire) {
                let now_ms = (self.elapsed_ns() / 1_000_000) as i64;
                let target_lead_ms = self.video_clock_base_ms.load(Ordering::Relaxed) + PLANE_LEAD_MS;
                let current_lead_ms = self.last_audio_pts_ms.load(Ordering::Relaxed) - now_ms;
                if current_lead_ms <= target_lead_ms - VIDEO_CLOCK_ASSIST_MARGIN_MS {
                    let from_ms = self.last_audio_pts_ms.load(Ordering::Relaxed);
                    if let Err(e) = self.feed_silence_locked(from_ms, now_ms + target_lead_ms) {
                        // Clock-plane failure must never masquerade as a video decode failure. Drop
                        // the assist and let the watchdog thread make the final attempt instead.
                        self.video_drives_clock.store(false, Ordering::Release);
                        tracing::warn!("NDL video-assisted clock top-up failed: {e:#}");
                    }
                }
            }

            self.fns.video_play(au, pts_ms)?;
            self.last_video_feed_ms
                .store((self.elapsed_ns() / 1_000_000) as i64, Ordering::Relaxed);
            mark_frame_fed_logged("NDL", self.load_instant)
        };
        if first_frame {
            self.replay_pending_hdr();
        }
        Ok(())
    }

    /// Apply HDR mastering metadata.
    pub fn set_color_info(
        &self,
        meta: Option<&punktfunk_core::quic::HdrMeta>,
        color: punktfunk_core::quic::ColorInfo,
    ) -> Result<()> {
        let Some(m) = meta else {
            return Ok(());
        };
        let [g, b, r] = m.display_primaries;
        let info = ffi::HdrInfo {
            display_primaries_x0: c_uint::from(g[0]),
            display_primaries_y0: c_uint::from(g[1]),
            display_primaries_x1: c_uint::from(b[0]),
            display_primaries_y1: c_uint::from(b[1]),
            display_primaries_x2: c_uint::from(r[0]),
            display_primaries_y2: c_uint::from(r[1]),
            white_point_x: c_uint::from(m.white_point[0]),
            white_point_y: c_uint::from(m.white_point[1]),
            max_display_mastering_luminance: m.max_display_mastering_luminance as c_uint,
            min_display_mastering_luminance: m.min_display_mastering_luminance as c_uint,
            max_content_light_level: c_uint::from(m.max_cll),
            max_pic_average_light_level: c_uint::from(m.max_fall),
            transfer_characteristics: c_uint::from(color.transfer),
            color_primaries: c_uint::from(color.primaries),
            matrix_coeffs: c_uint::from(color.matrix),
            reserved: [0; 32],
        };
        let mut pending = self.pending_hdr();
        if !super::presenting() {
            *pending = Some(info);
            return Ok(());
        }
        *pending = None;
        self.apply_hdr_info(info)
    }

    /// Buffered-but-undisplayed frames in NDL.
    pub fn render_buffer_length(&self) -> Option<i32> {
        let _ffi = lock_ffi();
        self.fns.render_buffer_length()
    }

    pub fn flush(&self) -> Result<()> {
        if !self.load_confirmed.load(Ordering::Relaxed) && !LOAD_COMPLETED.fired() {
            return Ok(());
        }
        let _ffi = lock_ffi();
        self.fns.flush_render_buffer()
    }
}

impl Drop for NdlVideo {
    fn drop(&mut self) {
        arm_load();
        self.fns.unload();
    }
}

impl MediaClock for NdlVideo {
    fn now_ns(&self) -> u64 {
        self.elapsed_ns()
    }
}

impl AudioSink for NdlVideo {
    fn name(&self) -> &'static str {
        "NDL Opus plane"
    }

    fn format(&self) -> AudioFormat {
        AudioFormat::Opus { channels: 2 }
    }

    fn feed(&self, samples: Samples<'_>, _host_pts_ns: u64) -> Result<()> {
        let Samples::Opus(packet) = samples else {
            bail!("NDL audio plane takes Opus packets only");
        };
        self.play_audio(packet)
    }

    fn depth_ms(&self) -> Option<i64> {
        Some(self.audio_plane_lead_ms())
    }
}

impl AudioPlane for NdlVideo {
    fn lead_ms(&self) -> i64 {
        self.audio_plane_lead_ms()
    }

    fn run_keepalive(&self, stop: &std::sync::atomic::AtomicBool, yields_to_real: bool) {
        self.run_clock_plane(stop, yields_to_real);
    }
}

impl VideoSink for std::sync::Arc<NdlVideo> {
    fn name(&self) -> &'static str {
        "NDL v2"
    }

    fn caps(&self) -> VideoSinkCaps {
        VideoSinkCaps {
            pts: true,
            partial_au: true,
            flush: true,
            render_queue: true,
        }
    }

    fn feed(&self, au: &[u8], pts_ns: u64) -> Result<()> {
        self.play(au, pts_ns)
    }

    fn flush(&self) -> Result<()> {
        NdlVideo::flush(self)
    }

    fn queue_depth(&self) -> Option<u32> {
        self.render_buffer_length().and_then(|d| u32::try_from(d).ok())
    }

    fn set_color(
        &self,
        meta: Option<&punktfunk_core::quic::HdrMeta>,
        color: punktfunk_core::quic::ColorInfo,
    ) -> Result<()> {
        self.set_color_info(meta, color)
    }

    fn clock(&self) -> Option<&dyn MediaClock> {
        Some(self.as_ref())
    }

    fn audio_plane(&self) -> Option<std::sync::Arc<dyn AudioPlane>> {
        self.has_audio_plane()
            .then(|| Self::clone(self) as std::sync::Arc<dyn AudioPlane>)
    }

    fn is_dead(&self) -> bool {
        super::fatal()
    }
}
