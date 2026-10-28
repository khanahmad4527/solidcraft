//! Contracts and result types for the commands in `cmd/sheet.rs`.

use schemars::JsonSchema;
use serde::Deserialize;

use super::types::{FeatureCommon, IdOrName, ProfilesParam, feature_common, present};

/// Add a tab to a sheet metal body: the closed profiles of a sketch drawn on a flat face of the
/// sheet (either side, or in a plane level with one) join that panel through the sheet's
/// thickness, part of its outline. The body stays one sheet: the tab unfolds with its panel (the
/// flat pattern and DXF carry it), and later flanges, hems and folds can use its edges. The
/// thickness is the body's rule's. Refused when the profile doesn't touch or overlap the face's
/// outline, adds nothing to it, reaches into a bend zone, or would overlap another panel once
/// folded or in the flat pattern, and while the sheet is unfolded. Makes no body.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct TabParams {
    /// The tab's sketch, by id or name, drawn on a flat face of the sheet. Default: the active
    /// sketch, else the last one.
    #[serde(default, deserialize_with = "present")]
    #[schemars(with = "IdOrName")]
    pub sketch: Option<IdOrName>,
    /// Which of the sketch's closed profiles make the tab. Default: all of them.
    #[serde(default, deserialize_with = "present")]
    #[schemars(with = "ProfilesParam")]
    pub profiles: Option<ProfilesParam>,
    /// Each index or point pick in `profiles` is exactly the region it names. Without it, a pick
    /// on a drawn profile that reference geometry (the face's projected outline) cuts into
    /// regions takes the whole drawn profile. The dialogs send `true`.
    #[serde(default)]
    pub exact_regions: Option<bool>,
    /// The sheet metal body to add the tab to. Default: the first sheet with a face in the
    /// sketch's plane that the profile meets.
    #[serde(default)]
    pub body: Option<String>,
    #[serde(flatten, deserialize_with = "feature_common")]
    pub common: FeatureCommon,
}

#[cfg(test)]
#[path = "sheet_tests.rs"]
mod tests;
