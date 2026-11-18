//! The app's clock: the wall clock, or with frame-stepped capture (`--framestep`) a virtual one.
//!
//! The virtual clock follows the wall clock until it is frozen (the first `ui.step`); from then on
//! it moves only when [`advance`] moves it, a frame's step at a time, so every animation, timer
//! and toast drawn in a captured frame depends only on the steps before it, never on how long a
//! frame took to draw. Frames between steps (the OS asking for a repaint, a step waiting for the
//! app to be idle) see the same time: a delta of 0.
//!
//! Everything that decides what a frame draws reads its time here: [`secs`] (egui's frame time),
//! [`now_ms`] (milliseconds since the Unix epoch) and [`instant`].

use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Virtual {
    /// The wall clock when the virtual clock began.
    start: Instant,
    /// Milliseconds since the Unix epoch at `start`.
    start_epoch_ms: f64,
    /// Frozen at these seconds since `start`; `None`: following the wall clock, `offset` ahead.
    frozen: Option<f64>,
    /// Seconds the free-running clock is ahead of the wall clock (time skipped while frozen
    /// comes back as an offset, so the clock never goes backwards).
    offset: f64,
}

static CLOCK: Mutex<Option<Virtual>> = Mutex::new(None);

fn lock() -> std::sync::MutexGuard<'static, Option<Virtual>> {
    CLOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn wall_epoch_ms() -> f64 {
    use web_time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0)
}

impl Virtual {
    fn secs(&self, now: Instant) -> f64 {
        self.frozen.unwrap_or_else(|| now.saturating_duration_since(self.start).as_secs_f64() + self.offset)
    }
}

/// Use the virtual clock from now on (it starts at 0 s, following the wall clock).
pub fn enable() {
    let mut c = lock();
    if c.is_none() {
        *c = Some(Virtual { start: Instant::now(), start_epoch_ms: wall_epoch_ms(), frozen: None, offset: 0.0 });
    }
}

/// Whether the virtual clock is in use.
pub fn enabled() -> bool {
    lock().is_some()
}

/// Whether the virtual clock is frozen: it moves only by [`advance`].
pub fn frozen() -> bool {
    lock().is_some_and(|c| c.frozen.is_some())
}

/// Stop the virtual clock where it is (`true`), or let it follow the wall clock again from where
/// it stands (`false`).
pub fn freeze(on: bool) {
    let now = Instant::now();
    if let Some(c) = lock().as_mut() {
        let secs = c.secs(now);
        if on {
            c.frozen = Some(secs);
        } else if c.frozen.take().is_some() {
            c.offset = secs - now.saturating_duration_since(c.start).as_secs_f64();
        }
    }
}

/// Move a frozen virtual clock on by `dt` seconds (non-finite or negative: not at all).
pub fn advance(dt: f64) {
    if !(dt.is_finite() && dt > 0.0) {
        return;
    }
    if let Some(c) = lock().as_mut()
        && let Some(t) = c.frozen.as_mut()
    {
        *t += dt;
    }
}

/// Seconds since the clock began: the virtual clock's, or `None` without one (egui's own then).
pub fn secs() -> Option<f64> {
    lock().map(|c| c.secs(Instant::now()))
}

/// Milliseconds since the Unix epoch, by the app's clock.
pub fn now_ms() -> f64 {
    match *lock() {
        Some(c) => c.start_epoch_ms + c.secs(Instant::now()) * 1000.0,
        None => wall_epoch_ms(),
    }
}

/// `Instant::now()`, by the app's clock.
pub fn instant() -> Instant {
    let now = Instant::now();
    match *lock() {
        Some(c) => c.start + Duration::from_secs_f64(c.secs(now).max(0.0)),
        None => now,
    }
}

/// Back to the wall clock (tests).
#[cfg(any(test, feature = "test-support"))]
pub fn disable() {
    *lock() = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test owns the process-wide clock (tests run in parallel).
    #[test]
    fn frozen_clock_moves_only_by_steps_and_never_back() {
        disable();
        assert!(!enabled());
        let wall = now_ms();
        assert!((now_ms() - wall).abs() < 1000.0);
        assert!(secs().is_none());

        enable();
        assert!(enabled() && !frozen());
        let a = secs().unwrap();
        freeze(true);
        let t0 = secs().unwrap();
        assert!(t0 >= a);
        let (ms0, i0) = (now_ms(), instant());
        std::thread::sleep(Duration::from_millis(30));
        // Frames between steps: no time passes.
        assert_eq!(secs().unwrap(), t0);
        assert_eq!(now_ms(), ms0);
        assert_eq!(instant(), i0);
        // 30 steps of 1/30 s: exactly one second, whatever the wall clock did.
        for _ in 0..30 {
            advance(1.0 / 30.0);
        }
        advance(-1.0);
        advance(f64::NAN);
        let t1 = secs().unwrap();
        assert!((t1 - t0 - 1.0).abs() < 1e-9, "{t1} {t0}");
        assert!((now_ms() - ms0 - 1000.0).abs() < 1e-6);
        assert_eq!(instant().duration_since(i0).as_millis(), 1000);
        // Released: it follows the wall clock again from where it stood, never going back.
        freeze(false);
        let t2 = secs().unwrap();
        assert!(t2 >= t1 && t2 - t1 < 0.5, "{t2} {t1}");
        std::thread::sleep(Duration::from_millis(20));
        assert!(secs().unwrap() > t2);
        // A free clock is not moved by steps.
        let t3 = secs().unwrap();
        advance(10.0);
        assert!(secs().unwrap() < t3 + 1.0);
        disable();
    }
}
