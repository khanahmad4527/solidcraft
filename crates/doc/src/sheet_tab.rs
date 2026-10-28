//! Tab (#8325): a closed profile drawn on a flat face of a sheet (either side, or in a plane
//! level with one) adds its area to that panel through the sheet's thickness, joined into the
//! panel's outline. The material stays one body and one panel: it unfolds with its panel, its
//! flat pattern and DXF carry it, and later flanges, hems and folds use its edges like any other
//! edge of the sheet.

use solidcraft_geom::{Plane, Region2, Seg2, Vec2, Vec3};
use solidcraft_kernel as kernel;

use super::{FlatWork, SheetBody, fitted_region, flat_polyline, flat_regions_overlap, material_boolean, material_union, planar_region};
use crate::{DocError, Result, apply_point, apply_vector, mat_inverse};

/// The tab's sketch is not on a flat face of the sheet.
pub const TAB_OFF_SHEET: &str = "draw the tab on a flat face of a sheet metal body (either side), or in a plane level with one";
/// The tab's profile does not meet the face it is drawn on.
pub const TAB_APART: &str = "the tab must touch or overlap the outline of the face it is drawn on: draw it from an edge of that face out";
/// The tab reaches into a bend.
pub const TAB_IN_BEND: &str = "the tab reaches into a bend: keep it on the flat of its panel, clear of the bend zones";
/// The tab lands on other material of the sheet.
pub const TAB_OVERLAPS: &str = "the tab would overlap another panel of the sheet once folded or in the flat pattern";
/// The tab lies inside the face already.
pub const TAB_ADDS_NOTHING: &str = "the tab adds no material: it lies inside the face already; draw it out past an edge of the face";

/// How far off a face's plane a sketch plane may lie and still be on it.
const LEVEL_TOL: f64 = 1e-6;

/// How long a stretch two outlines must share for a tab that only abuts its panel to join it.
const SHARED_EDGE: f64 = 1e-6;

impl SheetBody {
    /// The panels (None = the base) with a flat face, top or bottom, in `plane` (world): the
    /// panels a tab drawn there could join (while the sheet is unfolded, where they lie flat).
    pub fn level_panels(&self, plane: &Plane) -> Vec<Option<usize>> {
        let n = plane.normal();
        self.panels()
            .into_iter()
            .map(|(pi, _)| pi)
            .filter(|pi| {
                let w = self.world(*pi);
                let z = apply_vector(&w, Vec3::Z);
                if z.cross(n).len() > 1e-6 {
                    return false;
                }
                let o = apply_point(&w, Vec3::ZERO);
                let tol = LEVEL_TOL * (1.0 + o.len().max(plane.origin.len()) * 1e-3);
                let h = (plane.origin - o).dot(z);
                h.abs() <= tol || (h - self.t).abs() <= tol
            })
            .collect()
    }

    /// Add a tab: `regions` (closed profiles in `plane`, world) join the panel with a face in
    /// that plane through the full thickness (see the module docs). `lines` are the sketch's
    /// lines (an id and ends in `plane`), which name the base's new edges. Refused, leaving the
    /// sheet as it was, when the sheet is unfolded, the plane holds no face of it, the profile
    /// doesn't meet the face's outline or adds nothing to it, or it would reach into a bend or
    /// overlap other material of the sheet, folded or flat. Returns the panel it joined.
    pub fn add_tab(&mut self, plane: &Plane, regions: &[Region2], lines: &[(String, Vec2, Vec2)]) -> Result<Option<usize>> {
        if self.flat {
            return Err(DocError::Invalid("refold the sheet before adding a tab".into()));
        }
        if regions.is_empty() {
            return Err(DocError::Invalid("the tab needs a closed profile".into()));
        }
        let level = self.level_panels(plane);
        if level.is_empty() {
            return Err(DocError::Invalid(TAB_OFF_SHEET.into()));
        }
        let mut work = FlatWork::new();
        let panels = self.panels();
        // The first level panel the profile meets.
        let mut found = None;
        for pi in level {
            let Some((_, region)) = panels.iter().find(|(p, _)| *p == pi) else { continue };
            let mapped = regions.iter().map(|r| self.region_to_flat(pi, plane, r).and_then(|r| planar_region(&r))).collect::<Result<Vec<_>>>()?;
            let tab = material_union(mapped, &mut work)?;
            let panel = planar_region(region)?;
            if !meets(&panel, &tab, &mut work)? {
                continue;
            }
            let merged = material_union(std::iter::once(panel.clone()).chain(tab.iter().cloned()).collect(), &mut work)?;
            if let [one] = merged.as_slice() {
                found = Some((pi, panel, tab, one.clone()));
                break;
            }
        }
        let Some((pi, panel, tab, merged)) = found else {
            return Err(DocError::Invalid(TAB_APART.into()));
        };
        let added = merged.area().abs() - panel.area().abs();
        if added <= 1e-9 * panel.area().abs().max(1.0) {
            return Err(DocError::Invalid(TAB_ADDS_NOTHING.into()));
        }
        // Flat: clear of every bend zone and of every other panel.
        for f in &self.flanges {
            let zone = self.zone(f);
            for t in &tab {
                if flat_regions_overlap(t, &zone, &mut work)? {
                    return Err(DocError::Invalid(TAB_IN_BEND.into()));
                }
            }
        }
        for (pj, other) in &panels {
            if *pj == pi {
                continue;
            }
            for t in &tab {
                if flat_regions_overlap(t, other, &mut work)? {
                    return Err(DocError::Invalid(TAB_OVERLAPS.into()));
                }
            }
        }
        if !self.flanges.is_empty() {
            self.tab_clear_when_folded(pi, &tab, &mut work)?;
        }
        self.join_tab(pi, merged, plane, lines)?;
        Ok(pi)
    }

    /// Refused when the tab, folded with its panel `pi`, would run into other material of the
    /// folded sheet (another panel or a bend). The middle half of the tab's thickness is
    /// intersected with the folded solid: whatever it meets beyond its own panel's material is
    /// a collision.
    fn tab_clear_when_folded(&self, pi: Option<usize>, tab: &[Region2], work: &mut FlatWork) -> Result<()> {
        let mut local = self.clone();
        local.frame = Plane::XY;
        let solid = local.solid()?;
        let w = local.world(pi);
        let slab_plane = Plane::new(apply_point(&w, Vec3::ZERO), apply_vector(&w, Vec3::X), apply_vector(&w, Vec3::Y))
            .ok_or_else(|| DocError::Invalid("invalid sheet panel frame".into()))?;
        let fitted = tab.iter().map(fitted_region).collect::<Result<Vec<_>>>()?;
        let (z0, z1) = (self.t * 0.25, self.t * 0.75);
        let mut met = 0.0;
        for slab in kernel::extrude(&slab_plane, &fitted, z0, z1)? {
            if let Some(common) = kernel::boolean(&solid, &slab, kernel::BoolOp::Intersect)? {
                met += kernel::measure(&common)?.volume;
            }
        }
        let own: Vec<Region2> = local.material_panels(None, work)?.into_iter().filter(|(p, _)| *p == pi).map(|(_, r)| r).collect();
        let shared: f64 = material_boolean(&fitted, &own, kernel::BoolOp::Intersect, work)?.iter().map(|r| r.area().abs()).sum();
        let whole: f64 = fitted.iter().map(|r| r.area().abs()).sum();
        let h = z1 - z0;
        if met > shared * h + 1e-3 * whole * h + 1e-9 {
            return Err(DocError::Invalid(TAB_OVERLAPS.into()));
        }
        Ok(())
    }

    /// Make `merged` (the panel with the tab) panel `pi`'s outline. Holes the tab closes become
    /// cut-outs; the sketch's lines name the base's new edges.
    fn join_tab(&mut self, pi: Option<usize>, merged: Region2, plane: &Plane, lines: &[(String, Vec2, Vec2)]) -> Result<()> {
        let outer = merged.outer.ccw();
        self.holes.extend(merged.holes.iter().map(|h| Region2 { outer: h.ccw(), holes: Vec::new() }));
        match pi {
            None => {
                self.base = Region2 { outer, holes: Vec::new() };
                // The sketch's lines in flat coordinates (the base's world is its frame).
                if let Some(inv) = mat_inverse(&self.world(None)) {
                    let flat = |p: Vec2| {
                        let q = apply_point(&inv, plane.to_world(p));
                        Vec2::new(q.x, q.y)
                    };
                    for (id, a, b) in lines {
                        self.base_curves.push((id.clone(), flat(*a), flat(*b)));
                    }
                }
            }
            Some(i) => {
                let lines_only = outer.segs.iter().all(|s| matches!(s, Seg2::Line { .. }));
                let shape = flat_polyline(&outer)?.into_vec();
                let f = self.flanges.get_mut(i).ok_or_else(|| DocError::Invalid("tab panel".into()))?;
                f.shape = Some(shape);
                f.exact = if lines_only { None } else { Some(outer) };
            }
        }
        Ok(())
    }
}

/// Whether the tab meets the panel: overlaps it, or shares a stretch of outline with it (a tab
/// drawn from an edge out). Touching at a corner only is not meeting.
fn meets(panel: &Region2, tab: &[Region2], work: &mut FlatWork) -> Result<bool> {
    if !material_boolean(std::slice::from_ref(panel), tab, kernel::BoolOp::Intersect, work)?.is_empty() {
        return Ok(true);
    }
    let edges = |r: &Region2| -> Result<Vec<(Vec2, Vec2)>> {
        let mut out = Vec::new();
        for lp in std::iter::once(&r.outer).chain(&r.holes) {
            let pts = flat_polyline(lp)?;
            out.extend(pts.edges().map(|(a, b)| (*a, *b)));
        }
        Ok(out)
    };
    let mine = edges(panel)?;
    for t in tab {
        let theirs = edges(t)?;
        work.charge(mine.len().saturating_mul(theirs.len()))?;
        for &(a, b) in &mine {
            for &(c, d) in &theirs {
                if shared_length(a, b, c, d) > SHARED_EDGE {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// How long segments a→b and c→d run along each other (on one line).
fn shared_length(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> f64 {
    let Some(u) = (b - a).normalized() else { return 0.0 };
    let off = |p: Vec2| (p - a).cross(u).abs();
    let tol = 1e-7 * (1.0 + a.len().max(b.len()));
    if off(c) > tol || off(d) > tol {
        return 0.0;
    }
    let (s0, s1) = (0.0_f64, (b - a).dot(u));
    let (t0, t1) = ((c - a).dot(u), (d - a).dot(u));
    (s1.min(t0.max(t1)) - s0.max(t0.min(t1))).max(0.0)
}
