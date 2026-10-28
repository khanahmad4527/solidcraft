//! Timed control-channel motion: `ui.camera`, `ui.move` and `ui.text` with `seconds` (or
//! `interval`) are played out by the window itself, one step per rendered frame, so a screen
//! recording sees even motion at the display's rate. The request is answered when the motion
//! has finished, so a client sequences steps by waiting for each reply.

use std::sync::mpsc::Sender;

use serde_json::{Value, json};
use solidcraft_engine::render::Camera;

use crate::SolveApp;

/// The longest glide or typing run one request may ask for.
pub const MAX_SECONDS: f64 = 600.0;

/// What a glide moves.
pub enum Motion {
    /// The orbit camera, its numbers interpolated as given (a yaw past 2π turns past a full turn).
    Camera { from: Camera, to: Camera, last: Camera },
    /// The pointer, delivered as real pointer moves each frame (widgets hover as it passes).
    Pointer { from: egui::Pos2, to: egui::Pos2 },
    /// Typed text, one character every `interval` seconds.
    Text { chars: Vec<char>, interval: f64, sent: usize },
}

impl Motion {
    fn kind(&self) -> u8 {
        match self {
            Motion::Camera { .. } => 0,
            Motion::Pointer { .. } => 1,
            Motion::Text { .. } => 2,
        }
    }
}

/// One timed motion and the request waiting on it.
pub struct Glide {
    pub motion: Motion,
    /// When it started (`SolveApp::now`).
    pub start: f64,
    pub seconds: f64,
    /// The reply, attached by the window once the request's handler returned.
    pub reply: Option<Sender<Value>>,
}

/// `seconds` (or another duration key) as a request gives it: absent, or finite and within
/// 0..=[`MAX_SECONDS`].
pub fn duration(p: &Value, key: &str) -> Result<Option<f64>, String> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(s) if s.is_finite() && (0.0..=MAX_SECONDS).contains(&s) => Ok(Some(s)),
            _ => Err(format!("`{key}` must be a number of seconds from 0 to {MAX_SECONDS}")),
        },
    }
}

/// Smoothstep easing of `t` in 0..=1.
pub fn ease(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The camera `t` of the way from `a` to `b` (eased by the caller), each number interpolated
/// as given.
pub fn camera_at(a: &Camera, b: &Camera, t: f64) -> Camera {
    let mix = |x: f64, y: f64| x + (y - x) * t;
    Camera {
        target: a.target.lerp(b.target, t),
        yaw: mix(a.yaw, b.yaw),
        pitch: mix(a.pitch, b.pitch),
        roll: mix(a.roll, b.roll),
        distance: mix(a.distance, b.distance),
        fov: b.fov,
        up: b.up,
    }
}

/// Start a glide; one of the same kind already running ends where it is and is answered
/// (`interrupted`).
pub fn start(app: &mut SolveApp, motion: Motion, seconds: f64) {
    let kind = motion.kind();
    let (old, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut app.glides).into_iter().partition(|g| g.motion.kind() == kind);
    app.glides = keep;
    for g in old {
        answer(g, true);
    }
    app.glides.push(Glide { motion, start: app.now, seconds, reply: None });
}

/// Attach the reply to the glides a request just started.
pub fn attach(app: &mut SolveApp, reply: &Sender<Value>) {
    for g in app.glides.iter_mut().filter(|g| g.reply.is_none()) {
        g.reply = Some(reply.clone());
    }
}

/// Whether any glide is still playing.
pub fn active(app: &SolveApp) -> bool {
    !app.glides.is_empty()
}

fn answer(g: Glide, interrupted: bool) {
    if let Some(reply) = g.reply {
        let _ = reply.send(json!({"ok": true, "result": {"seconds": g.seconds, "interrupted": interrupted}}));
    }
}

/// Advance every glide to this frame (`app.now`), answering those that finished.
pub fn step(app: &mut SolveApp, ctx: &egui::Context) {
    if app.glides.is_empty() {
        return;
    }
    let now = app.now;
    let mut finished = Vec::new();
    let mut glides = std::mem::take(&mut app.glides);
    for (i, g) in glides.iter_mut().enumerate() {
        let t = if g.seconds > 0.0 { ((now - g.start) / g.seconds).clamp(0.0, 1.0) } else { 1.0 };
        let done = t >= 1.0;
        match &mut g.motion {
            Motion::Camera { from, to, last } => {
                // Something else moved the camera (the user navigated, or another request):
                // the glide gives way.
                if app.cam != *last {
                    finished.push((i, true));
                    continue;
                }
                let cam = if done { *to } else { camera_at(from, to, ease(t)) };
                if cam.is_valid() {
                    app.cam = cam;
                    *last = cam;
                }
            }
            Motion::Pointer { from, to } => {
                let pos = if done { *to } else { *from + (*to - *from) * ease(t) as f32 };
                // The agent cursor follows each move as it reaches the app.
                app.synthetic.push(egui::Event::PointerMoved(pos));
            }
            Motion::Text { chars, interval, sent } => {
                let due = if *interval > 0.0 { ((now - g.start) / *interval).floor() as usize + 1 } else { chars.len() };
                let due = due.min(chars.len());
                if due > *sent {
                    let text: String = chars.get(*sent..due).map(|c| c.iter().collect()).unwrap_or_default();
                    app.synthetic.push(egui::Event::Text(text));
                    *sent = due;
                }
                if *sent >= chars.len() {
                    finished.push((i, false));
                }
                continue;
            }
        }
        if done {
            finished.push((i, false));
        }
    }
    for (i, interrupted) in finished.into_iter().rev() {
        if i < glides.len() {
            answer(glides.remove(i), interrupted);
        }
    }
    // Glides started while stepping (none today) are kept too.
    glides.append(&mut app.glides);
    app.glides = glides;
    ctx.request_repaint();
}

/// The pointer where the window last saw it, or where queued synthetic moves leave it.
pub fn pointer_now(app: &SolveApp, ctx: &egui::Context) -> Option<egui::Pos2> {
    app.synthetic
        .iter()
        .rev()
        .find_map(|e| match e {
            egui::Event::PointerMoved(p) => Some(*p),
            _ => None,
        })
        .or_else(|| ctx.input(|i| i.pointer.latest_pos()))
}
