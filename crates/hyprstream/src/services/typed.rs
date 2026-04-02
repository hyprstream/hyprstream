//! Typed response carrying JSON (human-readable) and optionally capnp (machine-readable).
//!
//! Used by VFS ctl files that want to return structured data — callers get JSON
//! by default (Display impl) while internal consumers can access the capnp bytes
//! for zero-copy deserialization.

/// Typed response carrying JSON (human-readable) and optionally capnp (machine-readable).
#[derive(Clone, Debug)]
pub struct TypedValue {
    /// Schema identifier, e.g. "solver.SubmitResponse".
    pub schema: String,
    /// JSON serialization (always present).
    pub json: String,
    /// Cap'n Proto bytes (present when created from capnp).
    pub capnp: Option<Vec<u8>>,
}

impl TypedValue {
    /// Create a TypedValue from a JSON string with a schema identifier.
    pub fn from_json(schema: impl Into<String>, json: impl Into<String>) -> Self {
        Self {
            schema: schema.into(),
            json: json.into(),
            capnp: None,
        }
    }

    /// Create a TypedValue with both JSON and capnp representations.
    pub fn from_capnp(schema: impl Into<String>, json: impl Into<String>, capnp: Vec<u8>) -> Self {
        Self {
            schema: schema.into(),
            json: json.into(),
            capnp: Some(capnp),
        }
    }
}

impl std::fmt::Display for TypedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.json)
    }
}

impl std::str::FromStr for TypedValue {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Validate it's JSON, wrap as TypedValue with empty schema.
        let _: serde_json::Value = serde_json::from_str(s)?;
        Ok(TypedValue {
            schema: String::new(),
            json: s.to_owned(),
            capnp: None,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn display_returns_json() {
        let tv = TypedValue::from_json("test.Foo", r#"{"bar":42}"#);
        assert_eq!(tv.to_string(), r#"{"bar":42}"#);
    }

    #[test]
    fn from_str_roundtrip() {
        let json = r#"{"key":"value"}"#;
        let tv: TypedValue = json.parse().unwrap();
        assert_eq!(tv.json, json);
        assert!(tv.schema.is_empty());
        assert!(tv.capnp.is_none());
    }

    #[test]
    fn from_str_invalid_json() {
        let result = "not json".parse::<TypedValue>();
        assert!(result.is_err());
    }

    #[test]
    fn from_capnp_has_both() {
        let tv = TypedValue::from_capnp("test.Bar", r#"{"x":1}"#, vec![1, 2, 3]);
        assert_eq!(tv.schema, "test.Bar");
        assert_eq!(tv.json, r#"{"x":1}"#);
        assert_eq!(tv.capnp, Some(vec![1, 2, 3]));
    }
}
