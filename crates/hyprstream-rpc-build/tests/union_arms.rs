//! Fixture parse test for tagged-union arm extraction (#273).
//!
//! Compiles a tiny Cap'n Proto schema containing a union with a Void arm, a
//! scalar arm, and a `group` arm, then asserts the parsed `StructDef.union_arms`
//! captures the right `ArmPayload` shape for each — including the group's leaf
//! `FieldDef`s.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;

use hyprstream_rpc_build::schema::parse_from_cgr_path;
use hyprstream_rpc_build::schema::types::{ArmPayload, StructDef};

const TINY_SCHEMA: &str = r#"
@0xc0ffeec0ffeec0ff;

struct Knob {
  union {
    off @0 :Void;
    level @1 :UInt32;
    ranged :group {
      lo @2 :UInt32;
      hi @3 :UInt32;
      wrap @4 :Bool;
    }
  }
}
"#;

/// Compile `TINY_SCHEMA` to a CGR file and parse it via the CGR reader.
fn parse_tiny() -> Vec<StructDef> {
    let tmp = std::env::temp_dir().join(format!("hyprstream_union_arms_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    let capnp_path = tmp.join("tiny.capnp");
    std::fs::write(&capnp_path, TINY_SCHEMA).expect("write tiny.capnp");

    let cgr_path = tmp.join("tiny.cgr");
    capnpc::CompilerCommand::new()
        .src_prefix(&tmp)
        .file(&capnp_path)
        .raw_code_generator_request_path(&cgr_path)
        .run()
        .expect("compile tiny.capnp to CGR");

    let parsed = parse_from_cgr_path(Path::new(&cgr_path), "tiny").expect("parse CGR");

    let _ = std::fs::remove_dir_all(&tmp);
    parsed.structs
}

#[test]
fn union_arms_capture_void_scalar_and_group() {
    let structs = parse_tiny();

    let knob = structs
        .iter()
        .find(|s| s.name == "Knob")
        .expect("Knob struct present");

    // The synthetic group node (`Knob.ranged`) must NOT appear as a standalone struct.
    assert!(
        !structs.iter().any(|s| s.name.contains('.')),
        "synthetic group nodes should be excluded from structs: {:?}",
        structs.iter().map(|s| &s.name).collect::<Vec<_>>()
    );

    assert_eq!(knob.discriminant_count, 3, "three union arms");
    assert_eq!(knob.union_arms.len(), 3, "three union arms extracted");

    // Arms are sorted by discriminant: off (0), level (1), ranged (2).
    let off = &knob.union_arms[0];
    assert_eq!(off.name, "off");
    assert_eq!(off.discriminant_value, 0);
    assert!(matches!(off.payload, ArmPayload::Void), "off is Void");

    let level = &knob.union_arms[1];
    assert_eq!(level.name, "level");
    assert_eq!(level.discriminant_value, 1);
    match &level.payload {
        ArmPayload::Type(t) => assert_eq!(t, "UInt32"),
        other => panic!("level should be Type(UInt32), got {other:?}"),
    }

    let ranged = &knob.union_arms[2];
    assert_eq!(ranged.name, "ranged");
    assert_eq!(ranged.discriminant_value, 2);
    match &ranged.payload {
        ArmPayload::Group(leaves) => {
            assert_eq!(leaves.len(), 3, "group has three leaf fields");
            assert_eq!(leaves[0].name, "lo");
            assert_eq!(leaves[0].type_name, "UInt32");
            assert_eq!(leaves[1].name, "hi");
            assert_eq!(leaves[1].type_name, "UInt32");
            assert_eq!(leaves[2].name, "wrap");
            assert_eq!(leaves[2].type_name, "Bool");
            // Leaf fields are non-union members.
            for leaf in leaves {
                assert_eq!(leaf.discriminant_value, 0xFFFF);
            }
        }
        other => panic!("ranged should be Group, got {other:?}"),
    }
}
