use serde_json::{Value, json};

use super::CrossBreakParams;
use crate::Session;
use crate::schema::features::FeatureResult;
use crate::schema::tests::{assert_keys, assert_strict_keeps_todays_errors, conforms};
use crate::schema::{ParamCheck, input_schema_for, output_schema_for};

#[test]
fn schemas_and_the_key_check_agree() {
    assert_keys(
        input_schema_for::<CrossBreakParams>(),
        &["body", "body_name", "body_names", "depth", "direction", "face", "inset", "name", "participants"],
    );
}

#[test]
fn strict_keeps_todays_errors() {
    assert_strict_keeps_todays_errors(&[
        ("sheet.cross_break", json!({"face": [0, 0, 0], "direction": "sideways"})),
        ("sheet.cross_break", json!({"face": [0, 0, 0], "direction": 3})),
        ("sheet.cross_break", json!({"face": [0, 0, 0], "inset": "nowhere + 1"})),
    ]);
}

/// A 100 x 60 sheet on XY.
fn sheet(s: &mut Session) {
    for (id, p) in [
        ("sketch.create", json!({"plane": "XY", "name": "Base"})),
        ("sketch.rectangle.two_point", json!({"p0": [0, 0], "p1": [100, 60]})),
        ("sketch.finish", json!({})),
        ("sheet.flange", json!({"sketch": "Base"})),
    ] {
        if let Err(e) = s.execute(id, &p) {
            panic!("{id}: {e}");
        }
    }
}

/// Every form the command takes answers with a `FeatureResult` and, in Warn, no warnings.
#[test]
fn cross_break_replies_with_feature_result() {
    let schema = output_schema_for::<FeatureResult>();
    let schema = schema.as_value();
    for p in [
        json!({"face": [50, 30, 0]}),
        json!({"face": {"x": 50, "y": 30, "z": 0}, "inset": 5, "depth": "1.5 mm", "direction": "up"}),
        json!({"face": [50, 30, 0], "direction": "down", "body": "Body1", "name": "Lid stiffener"}),
        json!({"face": [50, 30, 0], "inset": null, "depth": null, "direction": null}),
    ] {
        let mut s = Session { param_check: ParamCheck::Warn, ..Session::default() };
        sheet(&mut s);
        let r = s.execute("sheet.cross_break", &p).unwrap_or_else(|e| panic!("{p}: {e}"));
        assert!(conforms(schema, schema, &r), "{r} doesn't match {schema}");
        assert!(r.get("param_warnings").is_none(), "{p}: {r}");
        assert_eq!(r["created_bodies"], json!([]), "{p}: no new body");
    }
    let mut s = Session { param_check: ParamCheck::Strict, ..Session::default() };
    sheet(&mut s);
    let e = s.execute("sheet.cross_break", &json!({"face": [50, 30, 0], "angle": 5})).unwrap_err().to_string();
    assert!(e.contains("angle"), "{e}");
    assert!(matches!(s.execute("sheet.cross_break", &json!({"face": [50, 30, 0]})), Ok(Value::Object(_))));
}
