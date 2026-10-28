use serde_json::{Value, json};

use super::TabParams;
use crate::Session;
use crate::schema::features::FeatureResult;
use crate::schema::tests::{assert_keys, assert_strict_keeps_todays_errors, conforms};
use crate::schema::{ParamCheck, input_schema_for, output_schema_for};

#[test]
fn schemas_and_the_key_check_agree() {
    assert_keys(input_schema_for::<TabParams>(), &["body", "body_name", "body_names", "exact_regions", "name", "participants", "profiles", "sketch"]);
}

#[test]
fn strict_keeps_todays_errors() {
    assert_strict_keeps_todays_errors(&[
        ("sheet.tab", json!({"sketch": true})),
        ("sheet.tab", json!({"sketch": "Nowhere"})),
        ("sheet.tab", json!({"profiles": [{"point": [1]}]})),
        ("sheet.tab", json!({"profiles": [["l1"], 7]})),
    ]);
}

/// A 100 x 60 sheet and a 40 x 40 profile on its top face, over its long edge by 10 mm.
fn sheet_with_tab_sketch(s: &mut Session) {
    for (id, p) in [
        ("sketch.create", json!({"plane": "XY", "name": "Base"})),
        ("sketch.rectangle.two_point", json!({"p0": [0, 0], "p1": [100, 60]})),
        ("sketch.finish", json!({})),
        ("sheet.flange", json!({"sketch": "Base"})),
        ("sketch.create", json!({"plane": {"face": [50, 30, 2.5]}, "name": "Ear"})),
        ("sketch.rectangle.two_point", json!({"p0": [30, 50], "p1": [70, 90]})),
        ("sketch.finish", json!({})),
    ] {
        if let Err(e) = s.execute(id, &p) {
            panic!("{id}: {e}");
        }
    }
}

/// Every form the command takes answers with a `FeatureResult` and, in Warn, no warnings.
#[test]
fn tab_replies_with_feature_result() {
    let schema = output_schema_for::<FeatureResult>();
    let schema = schema.as_value();
    for p in [
        json!({}),
        json!({"sketch": "Ear"}),
        json!({"sketch": "Ear", "profiles": "all", "body": "Body1", "name": "Ear tab"}),
        json!({"sketch": "Ear", "profiles": [0], "exact_regions": true}),
    ] {
        let mut s = Session { param_check: ParamCheck::Warn, ..Session::default() };
        sheet_with_tab_sketch(&mut s);
        let r = s.execute("sheet.tab", &p).unwrap_or_else(|e| panic!("{p}: {e}"));
        assert!(conforms(schema, schema, &r), "{r} doesn't match {schema}");
        assert!(r.get("param_warnings").is_none(), "{p}: {r}");
        assert_eq!(r["created_bodies"], json!([]), "{p}: no new body");
        assert_eq!(r["changed_bodies"], json!(["Body1"]), "{p}");
    }
    let mut s = Session { param_check: ParamCheck::Strict, ..Session::default() };
    sheet_with_tab_sketch(&mut s);
    let e = s.execute("sheet.tab", &json!({"sketch": "Ear", "height": 5})).unwrap_err().to_string();
    assert!(e.contains("height"), "{e}");
    assert!(matches!(s.execute("sheet.tab", &json!({"sketch": "Ear"})), Ok(Value::Object(_))));
}
