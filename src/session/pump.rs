//! The threads that drain the transport into the pipeline: access units into the video stage,
//! packets into the audio stage. Everything wire-shaped lives here and nothing else — what a
//! delivery MEANS is the stages' business.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use punktfunk_core::client::{AudioPacket, NativeClient};
use punktfunk_core::packet::{FLAG_SOF, USER_FLAG_RECOVERY_ANCHOR};
use punktfunk_core::reanchor::DROP_CREDIT_WINDOW;
use punktfunk_core::PunktfunkError;

use crate::platform::webos::device::boost_current_thread;
use crate::services::join::{join_with_timeout, SHUTDOWN_JOIN_TIMEOUT};
use crate::session::audio::AudioStage;
use crate::session::priority::{boost_hot_threads, spawn_vendor_decode_thread_renicer};
use crate::session::stage::{SinkResult, VideoStage, WireFrame};
use crate::session::StreamStats;

/// Longest a `next_frame` call parks before the loop re-checks `stop`.
const FRAME_WAIT: Duration = Duration::from_millis(500);
/// Cadence of the pump's liveness check: refreshes the overlay's backlog figure, and is the
/// only place a "nothing is arriving" line can come from.
const HEARTBEAT: Duration = Duration::from_secs(2);
/// How often the heartbeat's detail line reaches the log. The line is a trend ("still
/// draining, still not holding"), and one every couple of seconds buried the rest of the log
/// saying nothing new.
const VIDEO_LOG_INTERVAL: Duration = Duration::from_secs(15);

/// A stamp that fires once per `interval` and re-arms itself.
struct Tick {
    interval: Duration,
    last: Instant,
}

impl Tick {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            last: Instant::now(),
        }
    }

    fn due(&mut self) -> bool {
        let ready = self.last.elapsed() >= self.interval;
        if ready {
            self.last = Instant::now();
        }
        ready
    }
}

/// Current client realtime in nanoseconds, in the same domain `clock_offset_shared()` maps to the
/// host capture clock. Kept here instead of reusing the stage's submit-time helper because this is
/// deliberately sampled BEFORE any NDL work: the whole point is to measure how old a frame already
/// is when it leaves punktfunk-core.
fn realtime_ns() -> Option<i128> {
    let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
    i128::try_from(since_epoch.as_nanos()).ok()
}

/// One log-window's source-vs-delivery cadence. Measured on AU HEADS only: slice-progressive
/// pieces repeat both the frame index and host PTS, and counting their tails would make a 120 Hz
/// source look arbitrarily faster merely because its pictures happened to span more FEC blocks.
///
/// The useful quantity is `delivery_delta - source_delta`: genuine irregular game cadence appears
/// in BOTH deltas and therefore cancels, while transport/FEC/scheduler bunching appears only in the
/// local delivery delta. That lets a 4K120 trace distinguish "host produced ~120, TV received in
/// bursts" from "TV received clean ~120 but handed NDL late stamps" without a second machine.
///
/// `pre_ndl_age` adds the missing standing-latency dimension: client realtime is translated into
/// the host clock with core's live clock offset and compared directly with the host capture PTS.
/// If that age grows while NDL backlog stays small, the debt is already in transport/FEC/queues;
/// if it stays low while `late_stamp` rises, the fault is on the NDL/presentation side.
#[derive(Default)]
struct CadenceWindow {
    last_host_pts_ns: Option<u64>,
    last_arrival: Option<Instant>,
    aus: u64,
    source_intervals: u64,
    source_ns: u64,
    arrival_intervals: u64,
    arrival_ns: u64,
    delivery_delta_samples: u64,
    delivery_delta_abs_ns: u64,
    delivery_delta_max_ns: u64,
    max_arrival_gap_ns: u64,
    repeated_pts: u64,
    backwards_pts: u64,
    pre_ndl_age_samples: u64,
    pre_ndl_age_sum_ns: i128,
    pre_ndl_age_min_ns: Option<i64>,
    pre_ndl_age_max_ns: Option<i64>,
}

#[derive(Clone, Copy, Default)]
struct CadenceSnapshot {
    aus: u64,
    source_intervals: u64,
    source_ns: u64,
    arrival_intervals: u64,
    arrival_ns: u64,
    delivery_delta_samples: u64,
    delivery_delta_abs_ns: u64,
    delivery_delta_max_ns: u64,
    max_arrival_gap_ns: u64,
    repeated_pts: u64,
    backwards_pts: u64,
    pre_ndl_age_samples: u64,
    pre_ndl_age_sum_ns: i128,
    pre_ndl_age_min_ns: Option<i64>,
    pre_ndl_age_max_ns: Option<i64>,
}

impl CadenceWindow {
    fn observe(&mut self, frame: &punktfunk_core::session::Frame, clock_offset_ns: i64) {
        if frame.part.is_some_and(|part| !part.first) {
            return;
        }
        let now = Instant::now();
        self.aus = self.aus.saturating_add(1);

        let arrival_delta = self.last_arrival.map(|previous| {
            u64::try_from(now.duration_since(previous).as_nanos()).unwrap_or(u64::MAX)
        });
        if let Some(ns) = arrival_delta {
            self.arrival_intervals = self.arrival_intervals.saturating_add(1);
            self.arrival_ns = self.arrival_ns.saturating_add(ns);
            self.max_arrival_gap_ns = self.max_arrival_gap_ns.max(ns);
        }
        self.last_arrival = Some(now);

        let source_delta = match self.last_host_pts_ns {
            Some(previous) if frame.pts_ns > previous => {
                let ns = frame.pts_ns - previous;
                self.source_intervals = self.source_intervals.saturating_add(1);
                self.source_ns = self.source_ns.saturating_add(ns);
                Some(ns)
            }
            Some(previous) if frame.pts_ns == previous => {
                self.repeated_pts = self.repeated_pts.saturating_add(1);
                None
            }
            Some(_) => {
                self.backwards_pts = self.backwards_pts.saturating_add(1);
                None
            }
            None => None,
        };
        self.last_host_pts_ns = Some(frame.pts_ns);

        if let (Some(arrival_ns), Some(source_ns)) = (arrival_delta, source_delta) {
            let distortion = arrival_ns.abs_diff(source_ns);
            self.delivery_delta_samples = self.delivery_delta_samples.saturating_add(1);
            self.delivery_delta_abs_ns = self.delivery_delta_abs_ns.saturating_add(distortion);
            self.delivery_delta_max_ns = self.delivery_delta_max_ns.max(distortion);
        }

        // A zero offset is core's pre-sync/default value. Skip it rather than publish a giant
        // epoch-vs-PTS number that looks like real standing latency during session bring-up.
        if clock_offset_ns != 0 {
            if let Some(client_realtime_ns) = realtime_ns() {
                let age_ns = client_realtime_ns
                    .checked_add(i128::from(clock_offset_ns))
                    .and_then(|host_now_ns| host_now_ns.checked_sub(i128::from(frame.pts_ns)));
                if let Some(age_ns) = age_ns.and_then(|ns| i64::try_from(ns).ok()) {
                    self.pre_ndl_age_samples = self.pre_ndl_age_samples.saturating_add(1);
                    self.pre_ndl_age_sum_ns = self.pre_ndl_age_sum_ns.saturating_add(i128::from(age_ns));
                    self.pre_ndl_age_min_ns = Some(self.pre_ndl_age_min_ns.map_or(age_ns, |old| old.min(age_ns)));
                    self.pre_ndl_age_max_ns = Some(self.pre_ndl_age_max_ns.map_or(age_ns, |old| old.max(age_ns)));
                }
            }
        }
    }

    /// Reset only the WINDOW counters. Keep the two previous stamps so the first AU after a log
    /// still contributes an interval instead of creating a blind spot every 15 seconds.
    fn take(&mut self) -> CadenceSnapshot {
        let out = CadenceSnapshot {
            aus: self.aus,
            source_intervals: self.source_intervals,
            source_ns: self.source_ns,
            arrival_intervals: self.arrival_intervals,
            arrival_ns: self.arrival_ns,
            delivery_delta_samples: self.delivery_delta_samples,
            delivery_delta_abs_ns: self.delivery_delta_abs_ns,
            delivery_delta_max_ns: self.delivery_delta_max_ns,
            max_arrival_gap_ns: self.max_arrival_gap_ns,
            repeated_pts: self.repeated_pts,
            backwards_pts: self.backwards_pts,
            pre_ndl_age_samples: self.pre_ndl_age_samples,
            pre_ndl_age_sum_ns: self.pre_ndl_age_sum_ns,
            pre_ndl_age_min_ns: self.pre_ndl_age_min_ns,
            pre_ndl_age_max_ns: self.pre_ndl_age_max_ns,
        };
        self.aus = 0;
        self.source_intervals = 0;
        self.source_ns = 0;
        self.arrival_intervals = 0;
        self.arrival_ns = 0;
        self.delivery_delta_samples = 0;
        self.delivery_delta_abs_ns = 0;
        self.delivery_delta_max_ns = 0;
        self.max_arrival_gap_ns = 0;
        self.repeated_pts = 0;
        self.backwards_pts = 0;
        self.pre_ndl_age_samples = 0;
        self.pre_ndl_age_sum_ns = 0;
        self.pre_ndl_age_min_ns = None;
        self.pre_ndl_age_max_ns = None;
        out
    }
}

impl CadenceSnapshot {
    fn source_fps(self) -> f64 {
        rate_hz(self.source_intervals, self.source_ns)
    }

    fn arrival_fps(self) -> f64 {
        rate_hz(self.arrival_intervals, self.arrival_ns)
    }

    fn mean_delivery_delta_ms(self) -> f64 {
        mean_ms(self.delivery_delta_abs_ns, self.delivery_delta_samples)
    }

    fn mean_pre_ndl_age_ms(self) -> f64 {
        if self.pre_ndl_age_samples == 0 {
            return 0.0;
        }
        self.pre_ndl_age_sum_ns as f64 / self.pre_ndl_age_samples as f64 / 1_000_000.0
    }

    fn min_pre_ndl_age_ms(self) -> f64 {
        self.pre_ndl_age_min_ns.map_or(0.0, |ns| ns as f64 / 1_000_000.0)
    }

    fn max_pre_ndl_age_ms(self) -> f64 {
        self.pre_ndl_age_max_ns.map_or(0.0, |ns| ns as f64 / 1_000_000.0)
    }
}

fn rate_hz(intervals: u64, total_ns: u64) -> f64 {
    if intervals == 0 || total_ns == 0 {
        return 0.0;
    }
    intervals as f64 * 1_000_000_000.0 / total_ns as f64
}

fn mean_ms(total_ns: u64, samples: u64) -> f64 {
    if samples == 0 {
        return 0.0;
    }
    total_ns as f64 / samples as f64 / 1_000_000.0
}

/// Drives the video thread: transport → [`VideoStage`], plus the counters and the loss/HDR
/// side-channels that ride the same loop.
struct VideoPump {
    client: Arc<NativeClient>,
    stage: VideoStage,
    stats: Arc<StreamStats>,
    /// Live host-minus-client clock skew from core. Kept as the shared atomic rather than
    /// re-fetching/cloning it for every frame; core re-syncs this cell in place mid-session.
    clock_offset: Arc<AtomicI64>,
    /// Whether the host's per-content HDR metadata is worth draining. False on every session
    /// where nothing would apply it: an SDR or non-HEVC stream.
    is_hdr: bool,
    /// Core's cumulative drop count as of the last frame, to edge-detect new drops.
    last_dropped_seen: u64,
    /// Frame-index gaps pre-cover the reassembler's delayed drop accounting.
    drop_credit: u64,
    drop_credit_expiry: Option<Instant>,
    /// `punktfunk-core` registers its internal UDP/FEC data-plane worker asynchronously. Its
    /// `hot_thread_ids()` contract says the set is complete only after the first frame, so the
    /// one-shot fleet renice is deferred until a returned frame proves that worker exists.
    hot_threads_boosted: bool,
    cadence: CadenceWindow,
    /// `PacingHealth::late_stamps` is cumulative; keep the last logged value so each cadence line
    /// reports the late-stamp RATE for exactly the same 15-second window as source/arrival timing.
    last_pacing_late_logged: u64,
    heartbeat: Tick,
    video_log: Tick,
}

impl VideoPump {
    fn new(client: Arc<NativeClient>, stage: VideoStage, stats: Arc<StreamStats>, is_hdr: bool) -> Self {
        let last_dropped_seen = client.frames_dropped();
        let clock_offset = client.clock_offset_shared();
        Self {
            client,
            stage,
            stats,
            clock_offset,
            is_hdr,
            last_dropped_seen,
            drop_credit: 0,
            drop_credit_expiry: None,
            hot_threads_boosted: false,
            cadence: CadenceWindow::default(),
            last_pacing_late_logged: 0,
            heartbeat: Tick::new(HEARTBEAT),
            video_log: Tick::new(VIDEO_LOG_INTERVAL),
        }
    }

    fn run(&mut self, stop: &AtomicBool) {
        while !stop.load(Ordering::Relaxed) {
            match self.client.next_frame(FRAME_WAIT) {
                Ok(frame) => self.on_frame(&frame),
                Err(PunktfunkError::NoFrame) => {
                    if self.heartbeat.due() {
                        // INFO for the same reason as the main heartbeat — and this arm is the
                        // one that says "nothing is arriving at all", which is a different fault
                        // from "arriving but not presenting".
                        tracing::info!("video: {} frames (idle)", self.frames());
                    }
                }
                // A teardown the user asked for reaches both pumps as `Closed`, so it is not an
                // error in either — the audio pump already logged it at INFO.
                Err(PunktfunkError::Closed) => {
                    tracing::info!("video pump ending: session closed");
                    break;
                }
                Err(e) => {
                    tracing::error!("video pump: {e:#}");
                    break;
                }
            }
            self.forward_hdr_meta();
        }
    }

    /// Frames taken off the transport this session — the counter the overlay reads.
    fn frames(&self) -> u64 {
        self.stats.frames.load(Ordering::Relaxed)
    }

    fn on_frame(&mut self, frame: &punktfunk_core::session::Frame) {
        // A returned frame is the synchronization point core documents for its hot-thread
        // registry: by now the internal UDP receive/FEC worker has registered itself. Boosting
        // before `next_frame` could miss exactly that throughput-critical thread and leave it at
        // nice 0 while video/decode ran at -10 — worst at 4K120 where one late FEC burst can spend
        // a whole 8.33 ms frame budget. Do this once, before any first-frame stage work.
        if !self.hot_threads_boosted {
            boost_hot_threads(&self.client);
            self.hot_threads_boosted = true;
        }
        self.cadence
            .observe(frame, self.clock_offset.load(Ordering::Relaxed));
        self.stats.bytes.fetch_add(frame.data.len() as u64, Ordering::Relaxed);
        self.heartbeat();

        // Everything wire-shaped, and nothing else: whether this delivery is decodable at all,
        // and how one AU's pieces fit together, is the stage's bookkeeping.
        let wire = WireFrame {
            data: &frame.data,
            pts_ns: frame.pts_ns,
            index: frame.frame_index,
            part: frame.part,
            reanchor: frame.flags & u32::from(FLAG_SOF) != 0 || frame.flags & USER_FLAG_RECOVERY_ANCHOR != 0,
            // Parts repeat the AU flags. Count a recovery boundary once.
            recovery_mark: frame.flags & punktfunk_core::packet::USER_FLAG_RECOVERY_POINT != 0
                && frame.part.is_none_or(|part| part.first),
            loss: self.note_loss(frame),
        };
        match self.stage.submit(&wire) {
            SinkResult::Presented { decode_us } => {
                if let Some(us) = decode_us {
                    self.client.report_decode_us(us);
                }
            }
            SinkResult::Held => {}
            SinkResult::NeedKeyframe => {
                if let Err(e) = self.client.request_keyframe() {
                    tracing::warn!("request_keyframe: {e:#}");
                }
            }
            // Nothing above this loop can revive the decoder — the load is gone and the plane
            // threads have exited with it — so the session ends and the runtime returns to the
            // menu, where the next launch builds a fresh pipeline.
            SinkResult::Dead => {
                // Once, not per frame: the stream loop needs a poll or two to notice, and the
                // stage answers `Dead` to every delivery in the meantime.
                if !self.stats.decoder_dead.swap(true, Ordering::Relaxed) {
                    tracing::error!("decoder failed for good — ending the session");
                }
            }
        }
    }

    /// Refreshes the overlay's backlog figure, and on a slower cadence logs the pump's state.
    fn heartbeat(&mut self) {
        if !self.heartbeat.due() {
            return;
        }
        let backlog = self.stage.poll_backlog_depth();
        self.stats
            .render_backlog
            .store(backlog.unwrap_or(-1), Ordering::Relaxed);
        let pacing = self.stage.pacing_health();
        self.stats.pacing_jitter_us.store(
            u32::try_from(pacing.jitter_ns.max(0) / 1_000).unwrap_or(u32::MAX),
            Ordering::Relaxed,
        );
        self.stats.pacing_late.store(pacing.late_stamps, Ordering::Relaxed);
        let plane_lead = self.stage.audio_plane_lead_ms();
        if let Some(ms) = plane_lead {
            self.stats
                .audio_plane_lead_ms
                .store(i32::try_from(ms).unwrap_or(i32::MAX), Ordering::Relaxed);
        }
        // `backlog` separates "the decoder is behind" from "frames are arriving late" —
        // indistinguishable before this, since play() decodes and presents in one opaque call.
        // Logged on its own slower cadence: the overlay wants a fresh depth, the log does not.
        //
        // DEBUG, so it costs a telemetry listener or `TELEMETRY_LEVEL=debug` to see — the
        // on-device file sink is INFO-only (`logger::resolved_level`).
        if self.video_log.due() {
            let cadence = self.cadence.take();
            let late_window = pacing.late_stamps.saturating_sub(self.last_pacing_late_logged);
            self.last_pacing_late_logged = pacing.late_stamps;
            let late_pct = if cadence.aus == 0 {
                0.0
            } else {
                late_window as f64 * 100.0 / cadence.aus as f64
            };
            // `late_stamp` is the judder, counted: frames NDL was handed too late to pace. The
            // rest describes the loop that produced them (see `session::timeline::PacingHealth`).
            tracing::debug!(
                "pacing: late_stamp={} (+{} / {:.1}%) jitter={:.1}ms cushion={:.1}ms reanchors={}",
                pacing.late_stamps,
                late_window,
                late_pct,
                pacing.jitter_ns as f64 / 1e6,
                pacing.cushion_ns as f64 / 1e6,
                pacing.reanchors,
            );
            // This line is the 4K120 fork in the road. Source ~120 + arrival well below/bursty
            // points before NDL (transport/FEC/scheduling); source ~120 + arrival ~120 + late
            // stamps points at the NDL/presentation side. Because `delivery_delta` subtracts each
            // source interval from the matching arrival interval, real variable game cadence is
            // not misdiagnosed as network jitter. `pre_ndl_age` says whether that timing mismatch
            // accumulated into standing latency before this thread even entered the stage.
            tracing::debug!(
                "cadence: aus={} source={:.2}fps arrival={:.2}fps delivery_delta_mean={:.2}ms \
                 delivery_delta_max={:.2}ms arrival_gap_max={:.2}ms repeated_pts={} backwards_pts={} \
                 pre_ndl_age_mean={:.2}ms min={:.2}ms max={:.2}ms samples={}",
                cadence.aus,
                cadence.source_fps(),
                cadence.arrival_fps(),
                cadence.mean_delivery_delta_ms(),
                cadence.delivery_delta_max_ns as f64 / 1e6,
                cadence.max_arrival_gap_ns as f64 / 1e6,
                cadence.repeated_pts,
                cadence.backwards_pts,
                cadence.mean_pre_ndl_age_ms(),
                cadence.min_pre_ndl_age_ms(),
                cadence.max_pre_ndl_age_ms(),
                cadence.pre_ndl_age_samples,
            );
            tracing::debug!(
                "video: {} frames, parts={}, holding={}, dropped={}, backlog={}, plane_lead={}",
                self.frames(),
                // Against `frames`: 0 means slice-progressive delivery never fired on this mode
                // (core emits early parts only for an AU spanning more than one FEC block), so the
                // whole lever is inert here and its copy cost is not being paid either.
                self.stage.parts_fed(),
                self.stage.holding(),
                self.client.frames_dropped(),
                backlog.map_or_else(|| "n/a".to_string(), |b| b.to_string()),
                // The audio plane's depth is a video figure: NDL paces the picture on it, and a
                // lead sagging towards zero is what a stutter report should be read against.
                plane_lead.map_or_else(|| "n/a".to_string(), |ms| format!("{ms}ms")),
            );
        }
    }

    /// Whether loss reaches this frame — a sequence gap, or a frame the transport gave up on.
    fn note_loss(&mut self, frame: &punktfunk_core::session::Frame) -> bool {
        // From core v0.28 this returns the gap WIDTH (0 = contiguous) where it used to return a
        // bare "was there a gap" bool; `> 0` is the same predicate. Keep the width for the log
        // line — how many frames the hole swallowed is the number worth having when reading a
        // freeze report, not merely that one existed.
        // Slice-progressive pieces repeat their AU index. Observe it once, on the first piece.
        let au_first = frame.part.is_none_or(|part| part.first);
        let gap_width = if au_first {
            self.client.note_frame_index(frame.frame_index)
        } else {
            0
        };
        let dropped_now = self.client.frames_dropped();
        let dropped_delta = dropped_now.saturating_sub(self.last_dropped_seen);
        self.last_dropped_seen = dropped_now;
        let now = Instant::now();
        if self.drop_credit_expiry.is_some_and(|expiry| now >= expiry) {
            self.drop_credit = 0;
            self.drop_credit_expiry = None;
        }
        if gap_width > 0 {
            self.drop_credit = self.drop_credit.saturating_add(u64::from(gap_width));
            self.drop_credit_expiry = Some(now + DROP_CREDIT_WINDOW);
        }
        let credited = dropped_delta.min(self.drop_credit);
        self.drop_credit -= credited;
        if self.drop_credit == 0 {
            self.drop_credit_expiry = None;
        }
        let dropped = dropped_delta > credited;
        let lost = gap_width > 0 || dropped;
        if lost && !self.stage.holding() {
            // Logged alongside the freeze the sink reports next: a sequence hole and a frame the
            // transport itself gave up on point at different faults.
            tracing::warn!("loss: gap={gap_width} dropped={dropped} (frame {})", frame.frame_index);
        }
        lost
    }

    /// Hands the decoder any per-content HDR mastering metadata the host has sent.
    fn forward_hdr_meta(&mut self) {
        if !self.is_hdr {
            return;
        }
        // Collapse startup/keyframe repeats to the newest value. Applying an older queued value
        // first can delay a genuine mastering change by several frames.
        let mut latest = None;
        while let Ok(meta) = self.client.next_hdr_meta(Duration::ZERO) {
            latest = Some(meta);
        }
        let Some(meta) = latest else {
            return;
        };
        tracing::info!(
            "HDR metadata received: primaries={:?} white={:?} max_dml={} min_dml={} max_cll={} max_fall={}",
            meta.display_primaries,
            meta.white_point,
            meta.max_display_mastering_luminance,
            meta.min_display_mastering_luminance,
            meta.max_cll,
            meta.max_fall,
        );
        if let Err(e) = self.stage.set_color_info(Some(&meta), self.client.color) {
            tracing::warn!("NDL set_color_info: {e:#}");
        }
    }
}

/// The video thread's body: boost the threads that carry the stream, then pump until `stop`.
// The thread body owns everything it is handed — the `Arc`s die with it, which is what keeps
// the client and the stats alive for exactly as long as the pump runs.
#[allow(clippy::needless_pass_by_value)]
pub(super) fn video_pump(
    client: Arc<NativeClient>,
    stage: VideoStage,
    stop: Arc<AtomicBool>,
    stats: Arc<StreamStats>,
    is_hdr: bool,
) {
    // Register this thread immediately and boost it directly. The rest of core's hot-thread set is
    // deliberately boosted only after the first frame in `VideoPump::on_frame`, when core says its
    // asynchronously spawned UDP/FEC worker has registered too.
    client.register_hot_thread();
    boost_current_thread();
    spawn_vendor_decode_thread_renicer();
    VideoPump::new(client, stage, stats, is_hdr).run(&stop);
}

/// How long an audio drain parks on an empty plane before re-checking `stop`.
const AUDIO_WAIT: Duration = Duration::from_millis(100);

/// The shared body of both audio threads: pull packets, hand each to `play`, exit on `stop` or
/// a closed plane.
///
/// A thread of its own on either path, not a drain bolted onto another loop. Bolted onto the
/// video pump (where the offloaded path first lived) audio only drained after a `next_frame`
/// call that blocks up to [`FRAME_WAIT`], so a video drought — an encoder stall on the host, a
/// loss hold — chopped audio into ≤500 ms stalls *with packets already waiting*, and in normal
/// flow packets drained in per-video-frame clumps that all took the same drain-time PTS. Bolted
/// onto the main loop (where the software path lived, forced by `sdl2::audio::AudioQueue` being
/// `!Send`) it sat behind the UI's software rasterizer on a 2-3 core panel, and `docs/NOTES.md`
/// already named the 500 ms stats-overlay raster as an underrun source because of it. Core's
/// `next_audio` docs ask for exactly this thread ("packets arrive every 5 ms"), and its pull
/// methods are one-thread-per-plane safe by contract.
fn audio_drain(client: &NativeClient, stop: &AtomicBool, what: &str, mut play: impl FnMut(&AudioPacket)) {
    // Same boost the video pump requests for itself — 5 ms packets are the most
    // latency-sensitive cadence in the session.
    boost_current_thread();
    while !stop.load(Ordering::Relaxed) {
        match client.next_audio(AUDIO_WAIT) {
            Ok(packet) => play(&packet),
            Err(PunktfunkError::NoFrame) => {}
            Err(e) => {
                tracing::info!("{what} ending: {e:#}");
                break;
            }
        }
    }
}

/// The one audio pump: every route, every format.
///
/// Which sink it feeds is the route (`core::model::AudioRoutePref`), and what the sink takes is
/// the sink's own business ([`AudioStage`]) — this loop is blind to both.
///
/// Teardown safety on the plane routes: the stage holds an `Arc` of the plane, which is the same
/// handle as the video load, so the process-global NDL unload in `NdlVideo::drop` cannot run until
/// this thread has exited — a feed can never race the unload, whichever thread
/// `Connected::shutdown` happens to join first.
pub(super) fn audio_pump(client: &NativeClient, stage: &mut AudioStage, stop: &AtomicBool) {
    let what = stage.sink_name();
    let mut packets: u32 = 0;
    audio_drain(client, stop, what, |packet| {
        if let Err(e) = stage.play(packet.seq, packet.pts_ns, &packet.data) {
            tracing::warn!("audio error (seq {}): {e:#}", packet.seq);
            return;
        }
        packets = packets.wrapping_add(1);
        // ~15s, matching the video heartbeat (packets are 5ms each).
        if packets % 3_000 == 0 {
            tracing::debug!(
                "audio: {what}, depth={}, peak={:.4}",
                stage
                    .depth_ms()
                    .map_or_else(|| "n/a".to_string(), |ms| format!("{ms}ms")),
                stage.peak().unwrap_or(0.0),
            );
        }
    });
}

/// Spawns the audio thread for a session whose sink lives outside `connect` (the SDL device, which
/// belongs to whichever thread initialised SDL).
pub fn spawn_audio_feed(
    client: Arc<NativeClient>,
    mut stage: AudioStage,
    stop: Arc<AtomicBool>,
) -> Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("punktfunk-webos-audio".into())
        .spawn(move || audio_pump(&client, &mut stage, &stop))
        .context("spawn audio thread")
}

/// Joins the audio thread, bounded by the same timeout every other teardown join uses — a thread
/// wedged in an Opus decode must not hold the whole app on the way back to the menu. This is the
/// SDL route only, so a wedge here needs no `ndl::poison()`.
pub fn join_audio_feed(handle: std::thread::JoinHandle<()>) -> bool {
    join_with_timeout(handle, SHUTDOWN_JOIN_TIMEOUT, "audio-feed", || ())
}
