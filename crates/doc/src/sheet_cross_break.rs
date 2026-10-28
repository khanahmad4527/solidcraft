//! Cross Break (#8345): two shallow creases pressed corner to corner across one flat panel of a
//! sheet, so that the panel bows a little and stops drumming. Like a perforation field it is a
//! marking: the two lines are kept in flat coordinates on the sheet model and go to the view, the
//! flat pattern and the DXF; the B-rep, the outline and the volume do not change.
//!
//! The creases are the diagonals of the smallest rectangle round the panel's flat outline,
//! clipped to the panel (a corner relief or a notch shortens them instead of leaving them on
//! air), each then stopping `inset` short of the panel's edge at both ends.

use serde::{Deserialize, Serialize};
use solidcraft_geom::{Loop2, Vec2, Vec3};

use super::{FlatWork, SheetBody, flip_x};
use crate::{DocError, Result, apply_point, mat_inverse};

/// The pick is on no flat face of a panel of the sheet.
pub const CROSS_BREAK_OFF_PANEL: &str =
    "a cross break needs a point on the top or bottom face of a flat panel of the sheet (not on a bend, a rim or off the sheet)";
/// The panel already carries one.
pub const CROSS_BREAK_TWICE: &str = "this panel already has a cross break";

/// The most outline points the rectangle round a panel is fitted to.
const MAX_OUTLINE_POINTS: usize = 20_000;
/// A crease shorter than this (mm) is no crease.
const MIN_CREASE: f64 = 1e-3;

/// A cross break on one panel of a sheet (`sheet.cross_break`), flat coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetCrossBreak {
    pub feature: u64,
    /// The panel: None the base, else a flange's index.
    pub panel: Option<usize>,
    /// The two creases, each from one end to the other.
    pub lines: [(Vec2, Vec2); 2],
    /// How far the middle of the panel bows (mm), for the drawing and the press.
    pub depth: f64,
    /// The panel bows towards its top face (the side up bends turn to).
    pub up: bool,
}

impl SheetCrossBreak {
    /// The creases mirrored across the flat's y axis: a mirrored copy's.
    pub(super) fn flip_flat(&mut self) {
        for (a, b) in &mut self.lines {
            (*a, *b) = (flip_x(*b), flip_x(*a));
        }
    }
}

impl SheetBody {
    /// Mark a cross break on the panel whose top or bottom face `pick` (sheet coordinates) lies
    /// on, for feature `feature`: two creases corner to corner, each `inset` short of the panel's
    /// edge, bowing `depth` towards the top face when `up`. Returns the panel (None the base).
    /// Refused off a panel face, on a panel that already has one, and when the inset leaves
    /// nothing of a crease.
    pub fn add_cross_break(&mut self, feature: u64, pick: Vec3, inset: f64, depth: f64, up: bool) -> Result<Option<usize>> {
        if !(inset.is_finite() && inset >= 0.0) {
            return Err(DocError::Invalid(format!("a cross break's inset must be 0 or more (got {inset})")));
        }
        if !(depth.is_finite() && depth > 0.0) {
            return Err(DocError::Invalid(format!("a cross break's depth must be more than 0 (got {depth})")));
        }
        if !(pick.x.is_finite() && pick.y.is_finite() && pick.z.is_finite()) {
            return Err(DocError::Invalid(CROSS_BREAK_OFF_PANEL.into()));
        }
        let (panel, outline) = self.panel_under(pick)?;
        if self.cross_breaks.iter().any(|c| c.panel == panel) {
            return Err(DocError::Invalid(CROSS_BREAK_TWICE.into()));
        }
        let pts = outline.ccw().polyline(1e-2).into_vec();
        if pts.len() < 3 || pts.len() > MAX_OUTLINE_POINTS {
            return Err(DocError::Invalid(format!("a cross break needs a panel outline of 3 to {MAX_OUTLINE_POINTS} points")));
        }
        let [c0, c1, c2, c3] = enclosing_rectangle(&pts).ok_or_else(|| DocError::Invalid("the panel has no area for a cross break".into()))?;
        let mut lines = [(c0, c2), (c3, c1)];
        for (a, b) in &mut lines {
            let (lo, hi) =
                inside_span(&pts, *a, *b).ok_or_else(|| DocError::Invalid("the panel's middle is not material: no room for a cross break".into()))?;
            let len = a.dist(*b);
            let (s0, s1) = (lo + inset / len, hi - inset / len);
            if (s1 - s0) * len < MIN_CREASE {
                return Err(DocError::Invalid(format!("an inset of {inset} mm leaves nothing of the cross break on this panel")));
            }
            (*a, *b) = (a.lerp(*b, s0), a.lerp(*b, s1));
        }
        self.cross_breaks.push(SheetCrossBreak { feature, panel, lines, depth, up });
        Ok(panel)
    }

    /// The panel whose top or bottom face a point lies on (within a quarter of the thickness),
    /// with its flat outline: the nearest face when two are that close.
    fn panel_under(&self, pick: Vec3) -> Result<(Option<usize>, Loop2)> {
        let tol = self.t * 0.25;
        let mut work = FlatWork::new();
        let mut best: Option<(f64, Option<usize>, Loop2)> = None;
        for (pi, region) in self.material_panels(None, &mut work)? {
            let Some(inverse) = mat_inverse(&self.world(pi)) else { continue };
            let q = apply_point(&inverse, pick);
            let off = (q.z - self.t).abs().min(q.z.abs());
            if off > tol || !region.outer.contains(Vec2::new(q.x, q.y)) {
                continue;
            }
            if best.as_ref().is_none_or(|(d, ..)| off < *d) {
                best = Some((off, pi, region.outer));
            }
        }
        best.map(|(_, pi, outline)| (pi, outline)).ok_or_else(|| DocError::Invalid(CROSS_BREAK_OFF_PANEL.into()))
    }

    /// The persistent name of a cross break's panel (`base`, a flange's tag, or `flat` while the
    /// sheet is unfolded), for the flat pattern's listing.
    pub fn cross_break_panel(&self, c: &SheetCrossBreak) -> String {
        self.panel_name(c.panel)
    }

    /// The creases of every cross break as world segments, a hair above the top face and below
    /// the bottom one so that they read on either side of the panel: what the view draws.
    pub fn cross_break_segments(&self) -> Vec<(Vec3, Vec3)> {
        let lift = (self.t * 0.01).clamp(1e-3, 0.05);
        let mut out = Vec::new();
        for c in &self.cross_breaks {
            let world = self.world(c.panel);
            for (a, b) in c.lines {
                for z in [self.t + lift, -lift] {
                    let (p, q) = (apply_point(&world, Vec3::new(a.x, a.y, z)), apply_point(&world, Vec3::new(b.x, b.y, z)));
                    if [p, q].iter().all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()) {
                        out.push((p, q));
                    }
                }
            }
        }
        out
    }
}

/// The corners (in order round it) of the smallest rectangle round the points, whose sides run
/// along one side of their convex hull. None for points with no area.
fn enclosing_rectangle(pts: &[Vec2]) -> Option<[Vec2; 4]> {
    let hull = convex_hull(pts);
    if hull.len() < 3 {
        return None;
    }
    let mut best: Option<(f64, [Vec2; 4])> = None;
    for k in 0..hull.len() {
        let (Some(&a), Some(&b)) = (hull.get(k), hull.get((k + 1) % hull.len())) else { continue };
        let Some(u) = (b - a).normalized() else { continue };
        let v = u.perp();
        let (mut u0, mut u1, mut v0, mut v1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for p in &hull {
            let (s, t) = (p.dot(u), p.dot(v));
            (u0, u1, v0, v1) = (u0.min(s), u1.max(s), v0.min(t), v1.max(t));
        }
        let area = (u1 - u0) * (v1 - v0);
        // A nearly equal rectangle keeps the earlier one, so a square-cornered panel keeps its
        // own sides however its points happen to be ordered.
        if best.as_ref().is_none_or(|(b, _)| area < *b - 1e-9 * b.abs().max(1.0)) {
            let at = |s: f64, t: f64| u * s + v * t;
            best = Some((area, [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)]));
        }
    }
    best.filter(|(area, _)| *area > 1e-9).map(|(_, r)| r)
}

/// The convex hull of the points, counter-clockwise (monotone chain).
fn convex_hull(pts: &[Vec2]) -> Vec<Vec2> {
    let mut p: Vec<Vec2> = pts.iter().copied().filter(|q| q.is_finite()).collect();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    p.dedup_by(|a, b| a.dist(*b) < 1e-12);
    if p.len() < 3 {
        return p;
    }
    let turn = |o: Vec2, a: Vec2, b: Vec2| (a - o).cross(b - o);
    let mut hull: Vec<Vec2> = Vec::with_capacity(p.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let points: Box<dyn Iterator<Item = &Vec2>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in points {
            while hull.len() >= start + 2 {
                let (Some(&o), Some(&a)) = (hull.get(hull.len() - 2), hull.last()) else { break };
                if turn(o, a, q) > 1e-12 {
                    break;
                }
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    hull
}

/// Along the line from `a` to `b` (0 at `a`, 1 at `b`), the stretch inside the closed polygon
/// `pts` that holds the middle: between the outline crossings nearest the middle on either
/// side. None when the middle is outside the polygon.
fn inside_span(pts: &[Vec2], a: Vec2, b: Vec2) -> Option<(f64, f64)> {
    let mid = a.lerp(b, 0.5);
    if !Loop2::polygon(pts).contains(mid) {
        return None;
    }
    let d = b - a;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let n = pts.len();
    for k in 0..n {
        let (Some(&p), Some(&q)) = (pts.get(k), pts.get((k + 1) % n)) else { continue };
        let e = q - p;
        let den = d.cross(e);
        if den.abs() < 1e-12 {
            continue;
        }
        let s = (p - a).cross(e) / den;
        let t = (p - a).cross(d) / den;
        if !(-1e-9..=1.0 + 1e-9).contains(&t) {
            continue;
        }
        if s < 0.5 {
            lo = lo.max(s);
        } else {
            hi = hi.min(s);
        }
    }
    (hi > lo).then_some((lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> Vec2 {
        Vec2::new(x, y)
    }

    #[test]
    fn the_rectangle_round_a_turned_square_is_the_square() {
        let c = std::f64::consts::FRAC_1_SQRT_2 * 10.0;
        let r = enclosing_rectangle(&[v(0.0, 0.0), v(c, c), v(0.0, 2.0 * c), v(-c, c)]).unwrap();
        let side = r[0].dist(r[1]);
        assert!((side - 10.0).abs() < 1e-9 && (r[1].dist(r[2]) - 10.0).abs() < 1e-9, "{r:?}");
    }

    #[test]
    fn a_crease_stops_at_a_notched_corner() {
        // A 100 x 50 panel with a 10 x 10 notch at (0, 0): the diagonal from there starts where
        // it meets the notch's side, a tenth of the way along.
        let pts = [v(10.0, 0.0), v(100.0, 0.0), v(100.0, 50.0), v(0.0, 50.0), v(0.0, 10.0), v(10.0, 10.0)];
        let (lo, hi) = inside_span(&pts, v(0.0, 0.0), v(100.0, 50.0)).unwrap();
        assert!((lo - 0.1).abs() < 1e-9 && (hi - 1.0).abs() < 1e-9, "{lo} {hi}");
        // The middle off the material: no span.
        let ring = [v(0.0, 0.0), v(100.0, 0.0), v(100.0, 10.0), v(10.0, 10.0), v(10.0, 50.0), v(0.0, 50.0)];
        assert!(inside_span(&ring, v(0.0, 0.0), v(100.0, 50.0)).is_none());
    }
}
