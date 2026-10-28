//! Built-in frame recorder (`ui.record.start` / `ui.record.stop`).
//!
//! While recording, the app's clock is virtual: frame `n` of the recording sees the input time
//! `t0 + n / fps`, whatever the wall clock says, so every glide, animation and agent-cursor move
//! advances exactly one frame's worth per recorded frame and a slow frame (a recompute) never
//! shows as a jump. The window paints each frame offscreen (`egui_wgpu::record`) and the frames
//! are written as `f00000.png`, `f00001.png`… by a few writer threads behind a bounded queue: a
//! full queue blocks the frame, so memory stays flat and no frame is dropped.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;

use serde_json::{Value, json};

/// Frames waiting for a writer, at most.
const QUEUE: usize = 4;
/// Threads encoding PNGs.
const WRITERS: usize = 3;

/// The virtual clock of a recording: the input time of the next frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clock {
    /// Input time of the recording's first frame.
    pub t0: f64,
    pub fps: f64,
}

impl Clock {
    /// The input time of the frame after `frames` recorded frames.
    pub fn time(&self, frames: u64) -> f64 {
        self.t0 + frames as f64 / self.fps
    }
}

/// A recording in progress.
pub struct Recording {
    pub dir: PathBuf,
    pub clock: Clock,
    /// Frames handed to the writers so far (painted frames).
    frames: Arc<AtomicU64>,
    written: Arc<AtomicU64>,
    error: Arc<Mutex<Option<String>>>,
    writers: Vec<JoinHandle<()>>,
}

/// The app's clock correction: the input time eframe gives plus `offset`, so time never steps
/// back once a recording (whose virtual time ran ahead of or behind the wall clock) ends.
#[derive(Default)]
pub struct TimeShift {
    pub offset: f64,
    /// The last input time handed to the app.
    pub last: f64,
    /// The next frame resumes from `last` (a recording just ended).
    pub resync: bool,
}

impl Recording {
    /// The frames painted so far.
    pub fn frames(&self) -> u64 {
        self.frames.load(Ordering::Relaxed)
    }
}

fn write_png(dir: &Path, index: u64, frame: &egui_wgpu::record::Frame) -> Result<(), String> {
    use image::ImageEncoder;
    let path = dir.join(format!("f{index:05}.png"));
    let file = std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        std::io::BufWriter::new(file),
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::Adaptive,
    );
    encoder.write_image(&frame.rgba, frame.width, frame.height, image::ExtendedColorType::Rgba8).map_err(|e| format!("{}: {e}", path.display()))
}

fn writer(rx: Arc<Mutex<Receiver<(u64, egui_wgpu::record::Frame)>>>, dir: PathBuf, written: Arc<AtomicU64>, error: Arc<Mutex<Option<String>>>) {
    loop {
        let next = rx.lock().unwrap_or_else(PoisonError::into_inner).recv();
        let Ok((index, frame)) = next else { return };
        match write_png(&dir, index, &frame) {
            Ok(()) => {
                written.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                log::error!("record: {e}");
                error.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert(e);
            }
        }
    }
}

/// Start recording into `dir` at `fps`, the first frame being this one (input time `t0`).
pub fn start(dir: PathBuf, fps: f64, pixels_per_point: Option<f32>, t0: f64) -> Result<Recording, String> {
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (tx, rx): (SyncSender<(u64, egui_wgpu::record::Frame)>, _) = sync_channel(QUEUE);
    let rx = Arc::new(Mutex::new(rx));
    let frames = Arc::new(AtomicU64::new(0));
    let written = Arc::new(AtomicU64::new(0));
    let error = Arc::new(Mutex::new(None));
    let mut writers = Vec::new();
    for i in 0..WRITERS {
        let (rx, dir, written, error) = (rx.clone(), dir.clone(), written.clone(), error.clone());
        let spawned = std::thread::Builder::new().name(format!("record-writer-{i}")).spawn(move || writer(rx, dir, written, error));
        match spawned {
            Ok(h) => writers.push(h),
            Err(e) => log::error!("record: starting a writer: {e}"),
        }
    }
    if writers.is_empty() {
        return Err("could not start the frame writers".into());
    }
    let counter = frames.clone();
    let sink = move |frame: egui_wgpu::record::Frame| {
        let index = counter.fetch_add(1, Ordering::Relaxed);
        // Blocks while the queue is full: the frame waits for a writer, nothing is dropped.
        if tx.send((index, frame)).is_err() {
            log::error!("record: the frame writers stopped");
        }
    };
    egui_wgpu::record::start(egui_wgpu::record::Target { pixels_per_point, sink: Box::new(sink) });
    Ok(Recording { dir, clock: Clock { t0, fps }, frames, written, error, writers })
}

/// Stop: the window stops recording, the queued frames are written, the writers end.
pub fn stop(rec: Recording) -> Value {
    // Dropping the target drops the queue's sender: the writers finish what is queued and end.
    drop(egui_wgpu::record::stop());
    for w in rec.writers {
        let _ = w.join();
    }
    let frames = rec.frames.load(Ordering::Relaxed);
    let error = rec.error.lock().unwrap_or_else(PoisonError::into_inner).take();
    json!({
        "frames": frames,
        "written": rec.written.load(Ordering::Relaxed),
        "fps": rec.clock.fps,
        "seconds": frames as f64 / rec.clock.fps,
        "dir": rec.dir.to_string_lossy(),
        "error": error,
    })
}

/// The frame's input time: the recording's virtual clock while recording, else eframe's time
/// shifted to carry on from where a recording left it.
pub fn input_time(rec: Option<&Recording>, shift: &mut TimeShift, raw: &mut egui::RawInput) {
    if let Some(rec) = rec {
        let t = rec.clock.time(rec.frames());
        raw.time = Some(t);
        raw.predicted_dt = (1.0 / rec.clock.fps) as f32;
        shift.last = t;
        shift.resync = true;
        return;
    }
    let Some(wall) = raw.time else { return };
    if shift.resync {
        shift.resync = false;
        shift.offset = shift.last + 1.0 / 60.0 - wall;
    }
    if shift.offset != 0.0 {
        raw.time = Some(wall + shift.offset);
    }
    shift.last = wall + shift.offset;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each recorded frame advances the virtual clock by exactly 1/fps.
    #[test]
    fn virtual_clock_advances_one_frame_per_frame() {
        let clock = Clock { t0: 12.5, fps: 60.0 };
        for n in 0..600u64 {
            let dt = clock.time(n + 1) - clock.time(n);
            assert!((dt - 1.0 / 60.0).abs() < 1e-9, "frame {n}: {dt}");
        }
        assert!((clock.time(240) - 16.5).abs() < 1e-9);
    }

    /// Time never steps back when a recording ends, and is untouched when none ran.
    #[test]
    fn time_carries_on_after_a_recording() {
        let mut shift = TimeShift::default();
        let mut raw = egui::RawInput { time: Some(3.0), ..Default::default() };
        input_time(None, &mut shift, &mut raw);
        assert_eq!(raw.time, Some(3.0));
        // A recording ran ahead to 20 s of virtual time while the wall clock reached 5 s.
        shift.last = 20.0;
        shift.resync = true;
        let mut raw = egui::RawInput { time: Some(5.0), ..Default::default() };
        input_time(None, &mut shift, &mut raw);
        let t = raw.time.unwrap_or_default();
        assert!(t > 20.0 && t < 20.1);
        let mut raw = egui::RawInput { time: Some(5.5), ..Default::default() };
        input_time(None, &mut shift, &mut raw);
        assert!((raw.time.unwrap_or_default() - (t + 0.5)).abs() < 1e-9);
    }
}
