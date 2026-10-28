//! The contract of `sheet.cross_break` (in `cmd/sheet.rs`). It has a module of its own so that
//! the other sheet metal contracts, which come separately, need not share a file with it.

use std::borrow::Cow;

use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use super::check::whole;
use super::types::{FeatureCommon, LengthExpr, Point3, feature_common};

/// What `direction` must be: the handler's message and the contract's.
pub const CROSS_BREAK_DIRECTION: &str = "`direction` must be up or down";

/// Mark a cross break on a flat panel of a sheet metal body: two diagonal creases pressed corner
/// to corner across the panel so that it bows a little and stiffens. They are the diagonals of
/// the smallest rectangle round the panel's flat outline, clipped to the panel, each `inset`
/// short of its edge. A marking: the creases are drawn on the model, listed in the flat pattern
/// and written to the DXF (layers CROSS_BREAK_UP and CROSS_BREAK_DOWN); the B-rep, the outline
/// and the volume stay as they are. One per panel. Makes no body.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct CrossBreakParams {
    /// A point on the top or bottom face of the flat panel to mark (not on a bend or a rim).
    pub face: Point3,
    /// How far each crease stops short of the panel's edge, measured along the crease. Default 0.
    #[serde(default)]
    pub inset: Option<LengthExpr>,
    /// How far the middle of the panel bows: recorded for the drawing and the press, not
    /// modelled. Default 1 mm.
    #[serde(default)]
    pub depth: Option<LengthExpr>,
    /// Which face the panel bows towards: `up` (its top face, the side up bends turn to) or
    /// `down`. Default down.
    #[serde(default)]
    pub direction: Option<CrossBreakDirection>,
    /// The sheet metal body. Default: the sheet with a panel face under `face`.
    #[serde(default)]
    pub body: Option<String>,
    #[serde(flatten, deserialize_with = "feature_common")]
    pub common: FeatureCommon,
}

/// `up` or `down`, as the handler reads `direction`, with its message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrossBreakDirection(pub bool);

impl<'de> Deserialize<'de> for CrossBreakDirection {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match Value::deserialize(d)?.as_str() {
            Some("up") => Ok(CrossBreakDirection(true)),
            Some("down") => Ok(CrossBreakDirection(false)),
            _ => Err(D::Error::custom(whole(CROSS_BREAK_DIRECTION))),
        }
    }
}

impl JsonSchema for CrossBreakDirection {
    fn schema_name() -> Cow<'static, str> {
        "CrossBreakDirection".into()
    }
    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({"description": "Which face the panel bows towards: up (its top face) or down (the default).", "type": "string", "enum": ["up", "down"]})
    }
}

#[cfg(test)]
#[path = "sheet_cross_break_tests.rs"]
mod tests;
