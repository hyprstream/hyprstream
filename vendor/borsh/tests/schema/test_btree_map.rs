use crate::common_macro::schema_imports::*;

use alloc::collections::{BTreeMap, BTreeSet};

#[test]
fn b_tree_map() {
    let actual_name = BTreeMap::<u64, String>::declaration();
    let mut actual_defs = schema_map!();
    BTreeMap::<u64, String>::add_definitions_recursively(&mut actual_defs);
    assert_eq!("BTreeMap<u64, String>", actual_name);
    assert_eq!(
        schema_map! {
            "BTreeMap<u64, String>" => Definition::Sequence {
                length_width: Definition::DEFAULT_LENGTH_WIDTH,
                length_range: Definition::DEFAULT_LENGTH_RANGE,
                elements: "(u64, String)".to_string(),
            } ,
            "(u64, String)" => Definition::Tuple { elements: vec![ "u64".to_string(), "String".to_string()]},
            "u64" => Definition::Primitive(8),
            "String" => Definition::Sequence {
                length_width: Definition::DEFAULT_LENGTH_WIDTH,
                length_range: Definition::DEFAULT_LENGTH_RANGE,
                elements: "u8".to_string()
            },
            "u8" => Definition::Primitive(1)
        },
        actual_defs
    );
}

#[test]
fn b_tree_set() {
    let actual_name = BTreeSet::<String>::declaration();
    let mut actual_defs = schema_map!();
    BTreeSet::<String>::add_definitions_recursively(&mut actual_defs);
    assert_eq!("BTreeSet<String>", actual_name);
    assert_eq!(
        schema_map! {
            "BTreeSet<String>" => Definition::Sequence {
                length_width: Definition::DEFAULT_LENGTH_WIDTH,
                length_range: Definition::DEFAULT_LENGTH_RANGE,
                elements: "String".to_string(),
            },
            "String" => Definition::Sequence {
                length_width: Definition::DEFAULT_LENGTH_WIDTH,
                length_range: Definition::DEFAULT_LENGTH_RANGE,
                elements: "u8".to_string()
            },
            "u8" => Definition::Primitive(1)
        },
        actual_defs
    );
}
