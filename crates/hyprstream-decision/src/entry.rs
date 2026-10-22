//! The `EntryType` of the jev-1 profile: `string | object | array | null`, plus numbers
//! and booleans inside structured entries (a superset-permissive extension — structured
//! criteria routinely carry numeric thresholds).
//!
//! [`Entry`] is deliberately *not* `serde_json::Value`: the workspace's `serde_json` is
//! built without `preserve_order`, and question-spec order is canonical (D6 — argmax ties
//! and the frozen serialization both depend on authoring order). Mappings here are
//! insertion-ordered vectors of pairs; duplicate keys are preserved at parse time so the
//! validation walk can report them as positioned errors instead of silently dropping one.

use std::fmt;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;

/// An authored value: the payload of `state`, `instructions`, and criteria entries.
#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    /// Absent/undescribed. On choice options and score levels this means "undescribed"
    /// (D3); on `instructions` it is equivalent to omitting the key (D4).
    Null,
    /// Boolean inside a structured entry.
    Bool(bool),
    /// Number inside a structured entry (finite only — non-finite values are rejected at
    /// parse time).
    Number(f64),
    /// Plain text — the common case for rubric text.
    Str(String),
    /// Ordered array.
    Seq(Vec<Entry>),
    /// Insertion-ordered mapping. May contain duplicate keys as parsed; the authoring
    /// validation rejects duplicates at spec-bearing levels with a positioned error.
    Map(Vec<(String, Entry)>),
}

impl Entry {
    /// The canonical text form used by the frozen serialization: strings verbatim;
    /// structured entries as compact JSON (insertion order preserved); `Null` renders
    /// empty and is treated as absent by the serializer.
    pub fn canonical_text(&self) -> String {
        let mut out = String::new();
        self.write_canonical(&mut out);
        out
    }

    fn write_canonical(&self, out: &mut String) {
        match self {
            Self::Null => {}
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(value) => {
                // f64 Display is shortest-roundtrip and never emits inf/NaN spellings;
                // non-finite values are rejected before an Entry exists.
                out.push_str(&value.to_string());
            }
            Self::Str(text) => out.push_str(text),
            Self::Seq(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write_json(out);
                }
                out.push(']');
            }
            Self::Map(pairs) => {
                out.push('{');
                for (index, (key, value)) in pairs.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_json_string(key, out);
                    out.push(':');
                    value.write_json(out);
                }
                out.push('}');
            }
        }
    }

    /// Compact-JSON form for structured positions inside arrays/objects (strings are
    /// quoted there, and `null` must render as `null` — unlike the top-level canonical
    /// text rule, where Null renders empty).
    fn write_json(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Str(text) => write_json_string(text, out),
            other => other.write_canonical(out),
        }
    }
}

impl fmt::Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical_text())
    }
}

fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Map keys during deserialization. YAML 1.1 parses unquoted `true:`/`false:` keys as
/// booleans and numeric keys as numbers; the jev-1 noul criteria literally use the keys
/// `"true"`/`"false"`, so scalar keys are stringified rather than rejected. JSON object
/// keys are always strings, so this only bites on the YAML path.
#[derive(Debug, Clone, PartialEq)]
enum MapKey {
    Name(String),
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
}

impl MapKey {
    fn into_name(self) -> Result<String, String> {
        match self {
            Self::Name(name) => Ok(name),
            Self::Bool(value) => Ok(value.to_string()),
            Self::Int(value) => Ok(value.to_string()),
            Self::UInt(value) => Ok(value.to_string()),
            Self::Float(value) if value.is_finite() => Ok(value.to_string()),
            Self::Float(value) => Err(format!("non-finite number used as a mapping key: {value}")),
        }
    }
}

impl<'de> Deserialize<'de> for MapKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct KeyVisitor;
        impl Visitor<'_> for KeyVisitor {
            type Value = MapKey;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a mapping key (string, boolean, or number)")
            }

            fn visit_str<E>(self, value: &str) -> Result<MapKey, E> {
                Ok(MapKey::Name(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> Result<MapKey, E> {
                Ok(MapKey::Name(value))
            }

            fn visit_bool<E>(self, value: bool) -> Result<MapKey, E> {
                Ok(MapKey::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<MapKey, E> {
                Ok(MapKey::Int(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<MapKey, E> {
                Ok(MapKey::UInt(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<MapKey, E> {
                Ok(MapKey::Float(value))
            }
        }
        deserializer.deserialize_any(KeyVisitor)
    }
}

impl<'de> Deserialize<'de> for Entry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct EntryVisitor;
        impl<'de> Visitor<'de> for EntryVisitor {
            type Value = Entry;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("any YAML/JSON value (string, object, array, number, boolean, or null)")
            }

            fn visit_unit<E>(self) -> Result<Entry, E> {
                Ok(Entry::Null)
            }

            fn visit_none<E>(self) -> Result<Entry, E> {
                Ok(Entry::Null)
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Entry, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                Entry::deserialize(deserializer)
            }

            fn visit_bool<E>(self, value: bool) -> Result<Entry, E> {
                Ok(Entry::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Entry, E> {
                Ok(Entry::Number(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Entry, E> {
                Ok(Entry::Number(value as f64))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Entry, E>
            where
                E: de::Error,
            {
                if value.is_finite() {
                    Ok(Entry::Number(value))
                } else {
                    Err(de::Error::custom(format!(
                        "non-finite numbers are not representable: {value}"
                    )))
                }
            }

            fn visit_str<E>(self, value: &str) -> Result<Entry, E> {
                Ok(Entry::Str(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> Result<Entry, E> {
                Ok(Entry::Str(value))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Entry, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Entry::Seq(items))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Entry, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut pairs = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((key, value)) = map.next_entry::<MapKey, Entry>()? {
                    let name = key.into_name().map_err(de::Error::custom)?;
                    // Duplicates are preserved, not collapsed: the authoring validation
                    // reports them with an exact document path.
                    pairs.push((name, value));
                }
                Ok(Entry::Map(pairs))
            }
        }
        deserializer.deserialize_any(EntryVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_text_of_structured_entry_is_compact_json() {
        let entry = Entry::Map(vec![
            (
                "b".to_owned(),
                Entry::Seq(vec![Entry::Number(1.0), Entry::Str("x".into())]),
            ),
            ("a".to_owned(), Entry::Bool(true)),
        ]);
        assert_eq!(entry.canonical_text(), r#"{"b":[1,"x"],"a":true}"#);
    }

    #[test]
    fn strings_render_verbatim_at_top_level() {
        assert_eq!(
            Entry::Str("plain text".into()).canonical_text(),
            "plain text"
        );
        assert_eq!(Entry::Null.canonical_text(), "");
    }

    #[test]
    fn nested_nulls_render_as_json_null_not_empty() {
        // A null inside a structured position must stay valid JSON: `{"a":null}` and
        // `[null]`, never `{"a":}` or `[]` (which corrupts the canonical serialization
        // and round-trips through parse_json).
        let entry = Entry::Map(vec![(
            "a".to_owned(),
            Entry::Seq(vec![Entry::Null, Entry::Bool(true)]),
        )]);
        assert_eq!(entry.canonical_text(), r#"{"a":[null,true]}"#);
        let reparsed: Entry = serde_json::from_str(&entry.canonical_text())
            .unwrap_or_else(|error| panic!("valid JSON: {error}"));
        assert_eq!(reparsed, entry);
    }
}
