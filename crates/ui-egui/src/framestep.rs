//! Frame-stepped capture (`--framestep`): video of the window made frame by frame, each frame a
//! fixed step of a virtual clock, drawn offscreen at a fixed size and handed to a writer (ffmpeg,
//! a raw file, PNGs), however long a frame takes to draw or the design to compute.
//!
//! - Time: the app's clock (`ui_kit::clock`) freezes at the first `ui.step`; then egui's frame
//!   time, [`crate::now_ms`] and the drawing's `Instant`s move only by the steps. Frames between
//!   steps (the OS asking for one, a step waiting for work to finish) see a delta of 0.
//! - Size: the window's UI is laid out as `size / (2 × zoom)` points at 2 pixels a point (a 4K
//!   frame is a 1920 × 1080 window on a Retina screen, at the app's own zoom) and drawn into a
//!   texture of `size` pixels (`egui_wgpu::framestep`), whatever the window's size, or whether it
//!   is shown.
//! - Steps (`ui.step {frames, dt}`): before each frame the app must be idle on the wall clock (a
//!   file loading or saving, meshing, an operation check, a control job, a preview, tessellation
//!   or display refinement being computed); then the clock moves by `dt`, queued input
//!   (`ui.click`, `ui.key`…) is fed to that frame, and the frame is drawn and captured.
//! - Writer (`ui.record`): frames go to ffmpeg's stdin, or to a raw file; `ui.step {png_dir}`
//!   writes PNGs too.

use crate::SolveApp;
use serde_json::{Value, json};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::time::{Duration, Instant};

/// The capture size when `--framestep` names none: 4K UHD.
pub const DEFAULT_SIZE: (u32, u32) = (3840, 2160);
/// Pixels a point before the app's zoom: a Retina screen's.
pub const NATIVE_PIXELS_PER_POINT: f32 = 2.0;
/// A step waits at most this long (wall clock) for the app to be idle, then goes ahead and says so.
const GATE_TIMEOUT: Duration = Duration::from_secs(20);
/// The longest step asked for in one call (frames).
const MAX_FRAMES: u32 = 100_000;

/// `"3840x2160"` → (3840, 2160); even sizes from 16 to 8192 pixels.
pub fn parse_size(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.trim().split_once(['x', 'X'])?;
    let (w, h) = (w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?);
    let ok = |v: u32| (16..=8192).contains(&v) && v % 2 == 0;
    (ok(w) && ok(h)).then_some((w, h))
}

/// Milliseconds since the Unix epoch on the wall clock, for waits that are about the machine, not
/// the picture: `ui.screenshot`'s frame delay and deadline still pass while the clock is frozen
/// (the frame it takes shows the state of the last step).
pub(crate) fn wall_ms() -> f64 {
    use web_time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0)
}

/// Turn frame-stepped capture on for this run (at start, before the first frame).
/// SOLIDCRAFT_FRAMESTEP_INFLIGHT=1..3 sets how many frames may be read back while the next ones
/// run (default 2; 1 reads each back before the next frame).
pub fn enable(size: (u32, u32)) {
    crate::ui_kit::clock::enable();
    egui_wgpu::framestep::set_size(Some(size));
    egui_wgpu::framestep::set_sink(Some(Box::new(sink)));
    if let Some(n) = std::env::var("SOLIDCRAFT_FRAMESTEP_INFLIGHT").ok().and_then(|v| v.trim().parse().ok()) {
        egui_wgpu::framestep::set_in_flight(n);
    }
}

/// Whether this run captures frame by frame.
pub fn enabled() -> bool {
    egui_wgpu::framestep::size().is_some()
}

/// A `ui.step` under way.
struct Step {
    reply: Sender<Value>,
    frames: u32,
    dt: f64,
    /// Frames armed so far (each one captured before the next is armed).
    issued: u32,
    /// `egui_wgpu::framestep::captured()` when the step began.
    base: u64,
    began: Instant,
    /// Wall time spent waiting for the app to be idle, and since when it is waiting now.
    gate_wait: Duration,
    gate_since: Option<Instant>,
    /// What was waited for (each reason once), and frames that went ahead after [`GATE_TIMEOUT`].
    waited_for: Vec<&'static str>,
    timeouts: u32,
    png_dir: Option<PathBuf>,
    /// The readout to go back to once the step is done (PNG steps read back pixels).
    restore_readout: Option<egui_wgpu::framestep::Readout>,
}

#[derive(Default)]
struct Stepper {
    step: Option<Step>,
    /// The app's zoom, as of the last frame.
    zoom: Option<f32>,
}

static STEPPER: Mutex<Stepper> = Mutex::new(Stepper { step: None, zoom: None });

fn stepper() -> std::sync::MutexGuard<'static, Stepper> {
    STEPPER.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// What the app is still computing, on the wall clock: a captured frame waits until it is done,
/// so it shows the same thing however fast the machine is.
pub(crate) fn busy(app: &SolveApp) -> Vec<&'static str> {
    let mut busy = Vec::new();
    let mut add = |on: bool, what: &'static str| {
        if on {
            busy.push(what);
        }
    };
    add(!app.file_load.idle(), "file");
    add(app.saving.is_some(), "save");
    add(app.verifying.is_some(), "verify");
    add(crate::open_meshing::active(app), "meshing");
    add(app.op_check.busy(), "op_check");
    add(app.control_job.is_some(), "control_job");
    // A feature's OK evaluated off the window thread, a history recompute (a rename, the marker,
    // undo) and the commands queued behind it, a delete's check: each ends on the wall clock.
    add(app.applying.is_some(), "apply");
    add(app.deferred.is_some() || !app.deferred_queue.is_empty(), "deferred");
    add(app.delete_check.is_some(), "delete_check");
    add(app.preview.busy, "preview");
    add(app.viewport.tess_ready.is_some(), "tessellation");
    add(app.viewport.lod.lock().is_ok_and(|l| l.busy()), "refinement");
    add(app.viewport.mesh_lods.busy(), "mesh_levels");
    busy
}

/// The frame's input, before the app reads it: the capture size and the clock's time; and
/// whether this frame is a step (captured, its time moved on). Queued input is fed only to steps
/// while the clock is frozen.
pub(crate) fn begin_frame(app: &SolveApp, raw: &mut egui::RawInput) -> bool {
    let Some((w, h)) = egui_wgpu::framestep::size() else { return false };
    let mut s = stepper();
    let ppp = NATIVE_PIXELS_PER_POINT * s.zoom.unwrap_or(1.0);
    raw.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32 / ppp, h as f32 / ppp)));
    raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(NATIVE_PIXELS_PER_POINT);
    let step = s.step.as_mut().is_some_and(|step| {
        let t = Instant::now();
        let take = next_frame(step, || busy(app), t);
        egui_wgpu::framestep::profile_add("step_check", t.elapsed());
        take
    });
    if let Some(t) = crate::ui_kit::clock::secs() {
        raw.time = Some(t);
    }
    if crate::ui_kit::clock::frozen() {
        raw.predicted_dt = if step { s.step.as_ref().map_or(0.0, |st| st.dt as f32) } else { 0.0 };
    }
    step
}

/// Whether `step` takes this frame: the frame before it was captured, frames are left, and the app
/// is idle (or has kept a frame waiting past [`GATE_TIMEOUT`]). A frame taken moves the clock
/// and is armed for capture.
fn next_frame(step: &mut Step, busy: impl FnOnce() -> Vec<&'static str>, now: Instant) -> bool {
    let done = egui_wgpu::framestep::captured().saturating_sub(step.base);
    let in_flight = u64::from(step.issued).saturating_sub(done);
    if in_flight >= u64::from(egui_wgpu::framestep::in_flight()) || step.issued >= step.frames || egui_wgpu::framestep::armed() {
        return false;
    }
    let busy = busy();
    if !busy.is_empty() {
        for b in busy {
            if !step.waited_for.contains(&b) {
                step.waited_for.push(b);
            }
        }
        let since = *step.gate_since.get_or_insert(now);
        if now.saturating_duration_since(since) < GATE_TIMEOUT {
            return false;
        }
        step.timeouts += 1;
    }
    if let Some(since) = step.gate_since.take() {
        step.gate_wait += now.saturating_duration_since(since);
    }
    crate::ui_kit::clock::advance(step.dt);
    if let Some(dir) = &step.png_dir {
        // The painter numbers armed frames in order: this one is the next.
        let seq = egui_wgpu::framestep::armed_total() + u64::from(egui_wgpu::framestep::armed());
        NEXT_PNG.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push_back((seq, dir.join(format!("frame_{:06}.png", step.issued))));
    }
    egui_wgpu::framestep::arm();
    step.issued += 1;
    true
}

/// After the frame's logic: keep frames coming while a step runs; answer it once its last frame
/// is captured; keep the window painting while it is hidden.
pub(crate) fn after_logic(ctx: &egui::Context) {
    if !enabled() {
        return;
    }
    let mut s = stepper();
    s.zoom = Some(ctx.zoom_factor());
    ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
    let Some(step) = s.step.as_ref() else { return };
    ctx.request_repaint();
    let done = egui_wgpu::framestep::captured().saturating_sub(step.base);
    if step.issued >= step.frames && done >= u64::from(step.frames) {
        let step = s.step.take().expect("checked above");
        if let Some(r) = step.restore_readout {
            egui_wgpu::framestep::set_readout(r);
        }
        let _ = step.reply.send(json!({"ok": true, "result": step_report(&step)}));
    }
}

/// The time each part of making frames took since the last report (ms: mean, max, count).
fn profile_report() -> Value {
    let mut out = serde_json::Map::new();
    for (name, st) in egui_wgpu::framestep::profile_take() {
        let ms = |d: Duration| (d.as_secs_f64() * 1e5).round() / 100.0;
        let mean = if st.n > 0 { st.total / st.n as u32 } else { Duration::ZERO };
        out.insert(name.to_string(), json!({"mean_ms": ms(mean), "max_ms": ms(st.max), "n": st.n}));
    }
    Value::Object(out)
}

fn step_report(step: &Step) -> Value {
    json!({
        "profile": profile_report(),
        "frames": step.issued,
        "dt": step.dt,
        "clock": crate::ui_kit::clock::secs(),
        "captured_total": egui_wgpu::framestep::captured(),
        "wall_ms": step.began.elapsed().as_secs_f64() * 1000.0,
        "gate_wait_ms": step.gate_wait.as_secs_f64() * 1000.0,
        "waited_for": step.waited_for,
        "gate_timeouts": step.timeouts,
    })
}

/// `ui.step {frames, dt?, png_dir?, release?}`: answered once `frames` frames are captured, each
/// `dt` seconds (default 1/30) on from the one before. `frames: 0` only freezes the clock;
/// `release: true` lets it follow the wall clock again.
pub(crate) fn control_step(ctx: &egui::Context, params: &Value, reply: Sender<Value>) {
    let fail = |e: &str| {
        let _ = reply.send(json!({"ok": false, "error": e}));
    };
    if !enabled() {
        return fail("frame stepping is off: start SolidCraft with --framestep (or SOLIDCRAFT_FRAMESTEP=3840x2160)");
    }
    if let Some(k) = params.as_object().and_then(|o| o.keys().find(|k| !["frames", "dt", "png_dir", "release"].contains(&k.as_str()))) {
        return fail(&format!("unknown ui.step field `{k}` (it takes frames, dt, png_dir, release)"));
    }
    let mut s = stepper();
    if s.step.is_some() {
        return fail("a ui.step is still running");
    }
    if params.get("release").and_then(Value::as_bool) == Some(true) {
        crate::ui_kit::clock::freeze(false);
        let _ = reply.send(json!({"ok": true, "result": {"frozen": false, "clock": crate::ui_kit::clock::secs()}}));
        return;
    }
    let frames = match params.get("frames").map(Value::as_u64) {
        None => 1,
        Some(Some(n)) if n <= u64::from(MAX_FRAMES) => n as u32,
        Some(_) => return fail(&format!("`frames` must be a whole number up to {MAX_FRAMES}")),
    };
    let dt = match params.get("dt").map(Value::as_f64) {
        None => 1.0 / 30.0,
        Some(Some(dt)) if dt.is_finite() && dt > 0.0 && dt <= 10.0 => dt,
        Some(_) => return fail("`dt` must be seconds, above 0 and at most 10"),
    };
    let png_dir = match params.get("png_dir") {
        None | Some(Value::Null) => None,
        Some(Value::String(d)) => {
            let d = PathBuf::from(d);
            if let Err(e) = std::fs::create_dir_all(&d) {
                return fail(&format!("png_dir: {e}"));
            }
            Some(d)
        }
        Some(_) => return fail("`png_dir` must be a folder path"),
    };
    crate::ui_kit::clock::freeze(true);
    let restore_readout = png_dir.as_ref().map(|_| {
        let r = egui_wgpu::framestep::readout();
        egui_wgpu::framestep::set_readout(egui_wgpu::framestep::Readout::Texture);
        r
    });
    // The report's profile covers this step only.
    let _ = egui_wgpu::framestep::profile_take();
    let step = Step {
        reply,
        frames,
        dt,
        issued: 0,
        base: egui_wgpu::framestep::captured(),
        began: Instant::now(),
        gate_wait: Duration::ZERO,
        gate_since: None,
        waited_for: Vec::new(),
        timeouts: 0,
        png_dir,
        restore_readout,
    };
    if frames == 0 {
        if let Some(r) = step.restore_readout {
            egui_wgpu::framestep::set_readout(r);
        }
        let _ = step.reply.send(json!({"ok": true, "result": step_report(&step)}));
        return;
    }
    s.step = Some(step);
    ctx.request_repaint();
}

// ---------------------------------------------------------------- the writer

/// A captured frame on its way to the writer thread.
struct Shot {
    width: u32,
    height: u32,
    pixels: Pixels,
    bytes: Vec<u8>,
    png: Option<PathBuf>,
}

use egui_wgpu::framestep::Pixels;

enum Msg {
    Frame(Shot),
    Start {
        out: PathBuf,
        ffmpeg: Option<String>,
        args: Option<Vec<String>>,
        fps: f64,
        nv12: bool,
        done: Sender<Result<(), String>>,
    },
    Stop(Sender<Value>),
    /// Frames written so far and how many differ from the one before (once the frames queued
    /// before it are written).
    Status(Sender<Value>),
}

/// Where captured frames' PNGs go, by the frames' numbers (set as each frame is armed).
static NEXT_PNG: Mutex<std::collections::VecDeque<(u64, PathBuf)>> = Mutex::new(std::collections::VecDeque::new());

struct Writer {
    tx: SyncSender<Msg>,
    /// Buffers back from the writer, reused for the next frames.
    pool: Receiver<Vec<u8>>,
}

static WRITER: Mutex<Option<Writer>> = Mutex::new(None);

fn writer() -> std::sync::MutexGuard<'static, Option<Writer>> {
    let mut w = WRITER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if w.is_none() {
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        let (pool_tx, pool) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name("framestep-writer".into()).spawn(move || write_loop(rx, pool_tx));
        if spawned.is_ok() {
            *w = Some(Writer { tx, pool });
        }
    }
    w
}

/// The painter's sink: a copy of the frame to the writer thread (waits while two are queued).
fn sink(frame: egui_wgpu::framestep::Frame<'_>) {
    let png = {
        let mut q = NEXT_PNG.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while q.front().is_some_and(|(seq, _)| *seq < frame.seq) {
            q.pop_front();
        }
        if q.front().is_some_and(|(seq, _)| *seq == frame.seq) { q.pop_front().map(|(_, p)| p) } else { None }
    };
    let (tx, reused) = {
        let guard = writer();
        let Some(w) = guard.as_ref() else { return };
        (w.tx.clone(), w.pool.try_recv().ok())
    };
    let t = Instant::now();
    let mut bytes = reused.unwrap_or_default();
    bytes.clear();
    bytes.extend_from_slice(frame.bytes);
    egui_wgpu::framestep::profile_add("sink_copy", t.elapsed());
    let t = Instant::now();
    let _ = tx.send(Msg::Frame(Shot { width: frame.width, height: frame.height, pixels: frame.pixels, bytes, png }));
    egui_wgpu::framestep::profile_add("sink_queue_wait", t.elapsed());
}

/// A cheap fingerprint of a frame (tells repeated frames apart from new ones).
fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let words = bytes.chunks_exact(8);
    let rest = words.remainder();
    for w in words {
        h = (h ^ u64::from_le_bytes(w.try_into().unwrap_or([0; 8]))).wrapping_mul(0x0100_0000_01b3).rotate_left(17);
    }
    for &b in rest {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// An open recording: ffmpeg reading frames on its stdin, or a raw file.
struct Recording {
    out: PathBuf,
    child: Option<std::process::Child>,
    pipe: Box<dyn std::io::Write + Send>,
    /// The encoder arguments (None: the defaults for the layout).
    args: Option<Vec<String>>,
    fps: f64,
    /// Frames go out as NV12 (converted on the CPU when one comes back as pixels).
    nv12: bool,
    size: Option<(u32, u32, Pixels)>,
    /// Reused for frames converted on the CPU.
    converted: Vec<u8>,
    ffmpeg: Option<String>,
    frames: u64,
    unique: u64,
    last: Option<u64>,
    began: Instant,
    error: Option<String>,
}

impl Recording {
    /// Starts ffmpeg on the first frame, when the frame's size and byte order are known.
    fn write(&mut self, shot: &Shot) {
        if self.error.is_some() {
            return;
        }
        let mut bytes = &shot.bytes[..];
        let mut pixels = shot.pixels;
        if self.nv12 && pixels != Pixels::Nv12 {
            let t = Instant::now();
            egui_wgpu::framestep::to_nv12(&shot.bytes, shot.width, shot.height, pixels == Pixels::Bgra, &mut self.converted);
            egui_wgpu::framestep::profile_add("cpu_nv12", t.elapsed());
            bytes = &self.converted;
            pixels = Pixels::Nv12;
        }
        let key = (shot.width, shot.height, pixels);
        if self.size.is_none() {
            if let Some(ffmpeg) = &self.ffmpeg {
                let mut cmd = std::process::Command::new(ffmpeg);
                cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt"])
                    .arg(pixels.ffmpeg_name())
                    .args(["-s", &format!("{}x{}", shot.width, shot.height), "-r", &format!("{}", self.fps)])
                    .args(input_tags(pixels))
                    .args(["-i", "-"])
                    .args(self.args.clone().unwrap_or_else(|| default_encoder_args(pixels)))
                    .arg(&self.out)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::piped());
                match cmd.spawn() {
                    Ok(mut child) => {
                        self.pipe = Box::new(child.stdin.take().expect("piped stdin"));
                        self.child = Some(child);
                    }
                    Err(e) => {
                        self.error = Some(format!("ffmpeg did not start: {e}"));
                        return;
                    }
                }
            }
            self.size = Some(key);
        } else if self.size != Some(key) {
            self.error = Some("the frame size or layout changed during the recording".into());
            return;
        }
        let t = Instant::now();
        if let Err(e) = self.pipe.write_all(bytes) {
            self.error = Some(format!("writing a frame: {e}"));
            return;
        }
        egui_wgpu::framestep::profile_add("pipe_write", t.elapsed());
        let t = Instant::now();
        let f = fingerprint(bytes);
        egui_wgpu::framestep::profile_add("fingerprint", t.elapsed());
        if self.last != Some(f) {
            self.unique += 1;
        }
        self.last = Some(f);
        self.frames += 1;
    }

    fn finish(mut self) -> Value {
        let _ = self.pipe.flush();
        drop(std::mem::replace(&mut self.pipe, Box::new(std::io::sink())));
        let mut ffmpeg = Value::Null;
        if let Some(child) = self.child.take() {
            match child.wait_with_output() {
                Ok(out) => {
                    ffmpeg =
                        json!({"status": out.status.code(), "stderr": String::from_utf8_lossy(&out.stderr).chars().take(2000).collect::<String>()});
                    if !out.status.success() && self.error.is_none() {
                        self.error = Some("ffmpeg failed".into());
                    }
                }
                Err(e) => self.error = Some(format!("ffmpeg: {e}")),
            }
        }
        json!({
            "out": self.out,
            "frames": self.frames,
            "unique_frames": self.unique,
            "size": self.size.map(|(w, h, _)| [w, h]),
            "pix_fmt": self.size.map(|(_, _, p)| p.ffmpeg_name()),
            "wall_ms": self.began.elapsed().as_secs_f64() * 1000.0,
            "ffmpeg": ffmpeg,
            "error": self.error,
        })
    }
}

fn write_loop(rx: Receiver<Msg>, pool: Sender<Vec<u8>>) {
    let mut rec: Option<Recording> = None;
    for msg in rx {
        match msg {
            Msg::Frame(shot) => {
                if let Some(r) = rec.as_mut() {
                    r.write(&shot);
                }
                if let Some(path) = &shot.png {
                    write_png(path, &shot);
                }
                let _ = pool.send(shot.bytes);
            }
            Msg::Start { out, ffmpeg, args, fps, nv12, done } => {
                if let Some(old) = rec.take() {
                    old.finish();
                }
                let pipe: Result<Box<dyn std::io::Write + Send>, String> = if ffmpeg.is_some() {
                    Ok(Box::new(std::io::sink()))
                } else {
                    std::fs::File::create(&out)
                        .map(|f| Box::new(std::io::BufWriter::with_capacity(1 << 20, f)) as Box<dyn std::io::Write + Send>)
                        .map_err(|e| format!("{}: {e}", out.display()))
                };
                let _ = done.send(pipe.map(|pipe| {
                    rec = Some(Recording {
                        out,
                        child: None,
                        pipe,
                        args,
                        fps,
                        nv12,
                        size: None,
                        converted: Vec::new(),
                        ffmpeg,
                        frames: 0,
                        unique: 0,
                        last: None,
                        began: Instant::now(),
                        error: None,
                    });
                }));
            }
            Msg::Stop(done) => {
                let _ = done.send(rec.take().map_or(Value::Null, Recording::finish));
            }
            Msg::Status(done) => {
                let _ = done.send(rec.as_ref().map_or(Value::Null, |r| json!({"frames": r.frames, "unique_frames": r.unique, "error": r.error})));
            }
        }
    }
}

fn write_png(path: &std::path::Path, shot: &Shot) {
    if shot.pixels == Pixels::Nv12 {
        log::warn!("framestep: {}: the frame came back as NV12, not pixels", path.display());
        return;
    }
    let mut rgba = shot.bytes.clone();
    if shot.pixels == Pixels::Bgra {
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    match solidcraft_viewport::png(shot.width as usize, shot.height as usize, rgba) {
        Some(png) => {
            if let Err(e) = std::fs::write(path, png) {
                log::warn!("framestep: {}: {e}", path.display());
            }
        }
        None => log::warn!("framestep: PNG encoding failed"),
    }
}

/// What the input frames are, for ffmpeg: NV12 frames are BT.709 limited range already. Without
/// the tags ffmpeg 7.1 takes them for unknown colours and converts every frame (through RGB, on the
/// CPU: 370 ms of CPU time a 4K frame) to the output's tags.
fn input_tags(pixels: Pixels) -> &'static [&'static str] {
    if pixels == Pixels::Nv12 { &["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"] } else { &[] }
}

/// The encoder arguments when `ui.record` gives none, for frames in `pixels`' layout.
pub fn default_encoder_args(pixels: Pixels) -> Vec<String> {
    let args: &[&str] = if pixels == Pixels::Nv12 { &NV12_ENCODER_ARGS } else { &DEFAULT_ENCODER_ARGS };
    args.iter().map(|s| (*s).to_string()).collect()
}

/// The encoder arguments for NV12 frames (already BT.709 limited range, made on the GPU):
/// HEVC on the Mac's media engine, 40 Mbit/s, tagged so players show the window's colours;
/// `prio_speed`: 9 ms a 4K frame instead of 15 (the same size and PSNR at this bit rate).
pub const NV12_ENCODER_ARGS: [&str; 16] = [
    "-c:v",
    "hevc_videotoolbox",
    "-b:v",
    "40M",
    "-prio_speed",
    "1",
    "-tag:v",
    "hvc1",
    "-colorspace",
    "bt709",
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-color_range",
    "tv",
];

/// The encoder arguments for 4-byte frames when `ui.record` gives none: HEVC on the Mac's media
/// engine, 40 Mbit/s, BT.709 limited range (tagged), so players show the colours the window had.
pub const DEFAULT_ENCODER_ARGS: [&str; 16] = [
    "-vf",
    "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
    "-c:v",
    "hevc_videotoolbox",
    "-b:v",
    "40M",
    "-tag:v",
    "hvc1",
    "-colorspace",
    "bt709",
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-color_range",
    "tv",
];

/// `ui.record {out, fps?, ffmpeg?, args?, pix_fmt?}` starts writing captured frames: into ffmpeg
/// (`ffmpeg`: its path, or `true` for the one on PATH; `args`: the encoder arguments after the
/// input, [`default_encoder_args`] without) writing `out`, or without `ffmpeg` raw (`out` gets the
/// bytes). `pix_fmt`: `nv12` (BT.709 limited range, converted on the GPU: the default with
/// ffmpeg) or `pixels` (the window's 4-byte pixels, `bgra` or `rgba` as the reply of `stop` says:
/// the default for a raw file). `ui.record {status: true}`: frames written so far and how many
/// differ from the frame before. `ui.record {stop: true}` ends it: the same, and ffmpeg's status.
pub(crate) fn control_record(p: &Value) -> Result<Value, String> {
    if !enabled() {
        return Err("frame stepping is off: start SolidCraft with --framestep".into());
    }
    if let Some(k) =
        p.as_object().and_then(|o| o.keys().find(|k| !["out", "fps", "ffmpeg", "args", "pix_fmt", "stop", "status"].contains(&k.as_str())))
    {
        return Err(format!("unknown ui.record field `{k}` (it takes out, fps, ffmpeg, args, pix_fmt, stop, status)"));
    }
    let tx = writer().as_ref().map(|w| w.tx.clone()).ok_or("the frame writer did not start")?;
    if p.get("status").and_then(Value::as_bool) == Some(true) {
        let (done, rx) = std::sync::mpsc::channel();
        tx.send(Msg::Status(done)).map_err(|e| e.to_string())?;
        return rx.recv_timeout(Duration::from_secs(50)).map_err(|e| format!("the writer did not answer: {e}"));
    }
    if p.get("stop").and_then(Value::as_bool) == Some(true) {
        let (done, rx) = std::sync::mpsc::channel();
        tx.send(Msg::Stop(done)).map_err(|e| e.to_string())?;
        egui_wgpu::framestep::set_readout(egui_wgpu::framestep::Readout::Texture);
        return rx.recv_timeout(Duration::from_secs(50)).map_err(|e| format!("the writer did not finish: {e}"));
    }
    let out = PathBuf::from(p.get("out").and_then(Value::as_str).ok_or("`out` (a file path) is needed")?);
    let fps = p.get("fps").and_then(Value::as_f64).unwrap_or(30.0);
    if !(fps.is_finite() && fps > 0.0 && fps <= 240.0) {
        return Err("`fps` must be above 0 and at most 240".into());
    }
    let ffmpeg = match p.get("ffmpeg") {
        None | Some(Value::Null | Value::Bool(false)) => None,
        Some(Value::Bool(true)) => Some("ffmpeg".to_string()),
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err("`ffmpeg` must be a path or true".into()),
    };
    let args = match p.get("args") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => Some(a.iter().map(|v| v.as_str().map(str::to_string).ok_or("`args` must be strings")).collect::<Result<_, _>>()?),
        Some(_) => return Err("`args` must be a list of strings".into()),
    };
    // NV12 (made on the GPU) for an encoder, unless asked otherwise; the window's pixels raw.
    let (w, h) = egui_wgpu::framestep::size().unwrap_or(DEFAULT_SIZE);
    let nv12 = match p.get("pix_fmt").map(Value::as_str) {
        None | Some(Some("auto")) => ffmpeg.is_some() && egui_wgpu::framestep::nv12_fits(w, h),
        Some(Some("nv12")) if egui_wgpu::framestep::nv12_fits(w, h) => true,
        Some(Some("nv12")) => return Err("`pix_fmt` nv12 needs a width divisible by 4".into()),
        Some(Some("bgra" | "rgba" | "pixels")) => false,
        Some(_) => return Err("`pix_fmt` must be \"nv12\", \"pixels\" or \"auto\"".into()),
    };
    let (done, rx) = std::sync::mpsc::channel();
    tx.send(Msg::Start { out: out.clone(), ffmpeg, args, fps, nv12, done }).map_err(|e| e.to_string())?;
    rx.recv_timeout(Duration::from_secs(10)).map_err(|e| e.to_string())??;
    egui_wgpu::framestep::set_readout(if nv12 { egui_wgpu::framestep::Readout::Nv12 } else { egui_wgpu::framestep::Readout::Texture });
    Ok(json!({"recording": out, "pix_fmt": if nv12 { "nv12" } else { "pixels" }}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse() {
        assert_eq!(parse_size("3840x2160"), Some((3840, 2160)));
        assert_eq!(parse_size(" 1920X1080 "), Some((1920, 1080)));
        assert_eq!(parse_size("3841x2160"), None);
        assert_eq!(parse_size("0x0"), None);
        assert_eq!(parse_size("4k"), None);
    }

    #[test]
    fn fingerprints_tell_frames_apart() {
        let a = vec![7u8; 4096 + 3];
        let mut b = a.clone();
        b[4097] = 8;
        assert_eq!(fingerprint(&a), fingerprint(&a.clone()));
        assert_ne!(fingerprint(&a), fingerprint(&b));
    }

    fn step(frames: u32) -> (Step, Receiver<Value>) {
        let (reply, rx) = std::sync::mpsc::channel();
        let s = Step {
            reply,
            frames,
            dt: 1.0 / 30.0,
            issued: 0,
            base: egui_wgpu::framestep::captured(),
            began: Instant::now(),
            gate_wait: Duration::ZERO,
            gate_since: None,
            waited_for: Vec::new(),
            timeouts: 0,
            png_dir: None,
            restore_readout: None,
        };
        (s, rx)
    }

    /// The step logic against the painter's counter, with the painter played by `take_and_deliver`:
    /// one frame armed at a time, none while the app is busy, the clock moved only by steps, and a
    /// frame that stays busy past the timeout goes ahead and is counted. One test owns the
    /// process-wide clock and capture state.
    #[test]
    fn steps_wait_for_capture_and_idle_and_move_the_clock_exactly() {
        crate::ui_kit::clock::disable();
        crate::ui_kit::clock::enable();
        crate::ui_kit::clock::freeze(true);
        let t0 = crate::ui_kit::clock::secs().unwrap();
        let (mut s, _rx) = step(3);
        let now = Instant::now();
        // Busy: no frame, no time.
        assert!(!next_frame(&mut s, || vec!["meshing"], now));
        assert_eq!(crate::ui_kit::clock::secs().unwrap(), t0);
        assert_eq!(s.waited_for, ["meshing"]);
        // Idle: frame 1 is armed and the clock moves one step.
        assert!(next_frame(&mut s, Vec::new, now + Duration::from_millis(5)));
        assert!(egui_wgpu::framestep::armed());
        assert!((crate::ui_kit::clock::secs().unwrap() - t0 - 1.0 / 30.0).abs() < 1e-12);
        assert!(s.gate_wait >= Duration::from_millis(5));
        // Not captured yet: frames in between see no step (OS repaints).
        for _ in 0..5 {
            assert!(!next_frame(&mut s, Vec::new, now));
        }
        // The painter captures it.
        deliver_armed();
        assert!(next_frame(&mut s, Vec::new, now));
        deliver_armed();
        // Busy past the timeout: the frame goes ahead and is counted.
        assert!(!next_frame(&mut s, || vec!["preview"], now));
        assert!(next_frame(&mut s, || vec!["preview"], now + GATE_TIMEOUT + Duration::from_millis(1)));
        assert_eq!(s.timeouts, 1);
        deliver_armed();
        // All three done: nothing more is armed, and the clock moved exactly three steps.
        assert!(!next_frame(&mut s, Vec::new, now));
        assert_eq!(s.issued, 3);
        assert_eq!(egui_wgpu::framestep::captured() - s.base, 3);
        assert!((crate::ui_kit::clock::secs().unwrap() - t0 - 0.1).abs() < 1e-12);
        // Frames in flight: with two allowed, the next frame is armed while one is still being
        // read back, a third only once the first is handed on; frames arrive in order.
        egui_wgpu::framestep::set_in_flight(2);
        let (mut s, _rx) = step(4);
        assert!(next_frame(&mut s, Vec::new, now));
        assert!(egui_wgpu::framestep::test_take());
        assert!(next_frame(&mut s, Vec::new, now), "one in flight: the next frame runs");
        assert!(egui_wgpu::framestep::test_take());
        assert!(!next_frame(&mut s, Vec::new, now), "two in flight: wait");
        egui_wgpu::framestep::test_deliver();
        assert!(next_frame(&mut s, Vec::new, now));
        assert!(egui_wgpu::framestep::test_take());
        egui_wgpu::framestep::test_deliver();
        egui_wgpu::framestep::test_deliver();
        assert!(next_frame(&mut s, Vec::new, now));
        deliver_armed();
        assert!(!next_frame(&mut s, Vec::new, now));
        assert_eq!(egui_wgpu::framestep::captured() - s.base, 4);
        // One allowed: each frame is handed on before the next is armed.
        egui_wgpu::framestep::set_in_flight(1);
        let (mut s, _rx) = step(2);
        assert!(next_frame(&mut s, Vec::new, now));
        assert!(egui_wgpu::framestep::test_take());
        assert!(!next_frame(&mut s, Vec::new, now));
        egui_wgpu::framestep::test_deliver();
        assert!(next_frame(&mut s, Vec::new, now));
        deliver_armed();
        egui_wgpu::framestep::set_in_flight(egui_wgpu::framestep::DEFAULT_IN_FLIGHT);
        crate::ui_kit::clock::disable();
    }

    /// The painter's side of an armed frame, without a GPU: take the flag, count a frame.
    fn deliver_armed() {
        assert!(egui_wgpu::framestep::armed());
        egui_wgpu::framestep::test_capture();
    }

    /// A recording that takes NV12 converts frames that come back as pixels (a PNG step's) on the
    /// CPU, and counts frames and repeats; a raw file gets exactly the NV12 bytes.
    #[test]
    fn recordings_take_nv12_and_convert_pixels() {
        let dir = std::env::temp_dir().join(format!("framestep-rec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("raw.nv12");
        let mut rec = Recording {
            out: out.clone(),
            child: None,
            pipe: Box::new(std::fs::File::create(&out).unwrap()),
            args: None,
            fps: 30.0,
            nv12: true,
            size: None,
            converted: Vec::new(),
            ffmpeg: None,
            frames: 0,
            unique: 0,
            last: None,
            began: Instant::now(),
            error: None,
        };
        let white = Shot { width: 4, height: 2, pixels: Pixels::Bgra, bytes: vec![255; 32], png: None };
        let mut nv12 = Vec::new();
        egui_wgpu::framestep::to_nv12(&white.bytes, 4, 2, true, &mut nv12);
        rec.write(&white);
        rec.write(&Shot { width: 4, height: 2, pixels: Pixels::Nv12, bytes: nv12.clone(), png: None });
        rec.write(&Shot { width: 4, height: 2, pixels: Pixels::Nv12, bytes: vec![16; 12], png: None });
        let report = rec.finish();
        assert_eq!((report["frames"].as_u64(), report["unique_frames"].as_u64()), (Some(3), Some(2)), "{report}");
        assert_eq!(report["pix_fmt"], "nv12");
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(bytes, [nv12.clone(), nv12, vec![16; 12]].concat());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn encoder_arguments_follow_the_layout() {
        let nv12 = default_encoder_args(Pixels::Nv12);
        assert!(!nv12.iter().any(|a| a == "-vf"), "NV12 is already converted: {nv12:?}");
        assert!(nv12.windows(2).any(|w| w == ["-c:v", "hevc_videotoolbox"]));
        assert!(default_encoder_args(Pixels::Bgra).iter().any(|a| a == "-vf"));
        // NV12 input is tagged, so ffmpeg converts nothing.
        assert!(input_tags(Pixels::Nv12).windows(2).any(|w| w == ["-color_range", "tv"]));
        assert!(input_tags(Pixels::Bgra).is_empty());
    }
}
