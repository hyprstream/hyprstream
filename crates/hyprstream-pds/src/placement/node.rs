//! `ai.hyprstream.placement.node` — a schedulable node advertised by an account.
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.placement.node",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["repo", "labels", "resources", "groups", "createdAt"],
//!     "properties": {
//!       "repo":      { "type": "string", "format": "at-uri" },
//!       "labels":    { "type": "array", "items": { "type": "object",
//!                        "required": ["key", "value"], "properties": {
//!                          "key":   { "type": "string" },
//!                          "value": { "type": "string" } } } },
//!       "resources": { "type": "array", "items": { "type": "object",
//!                        "required": ["name", "capacity"], "properties": {
//!                          "name":     { "type": "string" },
//!                          "capacity": { "type": "string" } } } },
//!       "groups":    { "type": "array", "items": { "type": "string", "format": "at-uri" } },
//!       "createdAt": { "type": "string", "format": "datetime" }
//!     }
//! }}}
//! ```
//!
//! `groups` are at-uris of `ai.hyprstream.placement.group` records the node has
//! consented to be placed within. `resources[].capacity` carries a Kubernetes
//! quantity string (e.g. `"16"`, `"100m"`, `"8Gi"`).

use anyhow::{bail, ensure, Result};

use super::{map_get_str, validate_at_uri, validate_datetime};
use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// The NSID of this record type.
pub const COLLECTION_NSID: &str = "ai.hyprstream.placement.node";

/// A `{key, value}` label pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Label {
    pub key: String,
    pub value: String,
}

/// A `{name, capacity}` resource advertisement. `capacity` is a k8s-quantity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Resource {
    pub name: String,
    pub capacity: String,
}

/// An `ai.hyprstream.placement.node` record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeRecord {
    /// at-uri of the owning repo. `format: "at-uri"`.
    pub repo: String,
    /// Free-form `{key, value}` labels.
    pub labels: Vec<Label>,
    /// Advertised `{name, capacity}` resources (capacity is a k8s-quantity).
    pub resources: Vec<Resource>,
    /// at-uris of consented placement groups. Each is `format: "at-uri"`.
    pub groups: Vec<String>,
    /// ISO-8601 UTC datetime. `format: "datetime"`.
    pub created_at: String,
}

impl NodeRecord {
    /// Validate and construct a record.
    pub fn new(
        repo: impl Into<String>,
        labels: Vec<Label>,
        resources: Vec<Resource>,
        groups: Vec<String>,
        created_at: impl Into<String>,
    ) -> Result<Self> {
        let repo = repo.into();
        let created_at = created_at.into();
        validate_at_uri(&repo)?;
        for l in &labels {
            validate_label(l)?;
        }
        for r in &resources {
            validate_resource(r)?;
        }
        for g in &groups {
            validate_at_uri(g)?;
        }
        validate_datetime(&created_at)?;
        Ok(NodeRecord {
            repo,
            labels,
            resources,
            groups,
            created_at,
        })
    }

    /// Encode to canonical DAG-CBOR bytes (deterministic).
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    /// The CID of this record (CIDv1 dag-cbor over its canonical bytes).
    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.to_dag_cbor())
    }

    /// Build the typed [`DagCbor`] value form.
    pub fn to_value(&self) -> DagCbor {
        let labels = DagCbor::list(self.labels.iter().map(|l| {
            DagCbor::str_map([
                ("key", DagCbor::Text(l.key.clone())),
                ("value", DagCbor::Text(l.value.clone())),
            ])
        }));
        let resources = DagCbor::list(self.resources.iter().map(|r| {
            DagCbor::str_map([
                ("name", DagCbor::Text(r.name.clone())),
                ("capacity", DagCbor::Text(r.capacity.clone())),
            ])
        }));
        let groups = DagCbor::list(self.groups.iter().map(|g| DagCbor::Text(g.clone())));
        DagCbor::str_map([
            ("repo", DagCbor::Text(self.repo.clone())),
            ("labels", labels),
            ("resources", resources),
            ("groups", groups),
            ("createdAt", DagCbor::Text(self.created_at.clone())),
        ])
    }

    /// Decode canonical DAG-CBOR bytes into a record, validating all fields.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let map = value.as_map()?;
        let repo = map_get_str(value, "repo", COLLECTION_NSID)?.to_owned();
        let labels = value
            .get("labels")
            .ok_or_else(|| anyhow::anyhow!("{COLLECTION_NSID}: missing required field \"labels\""))?
            .as_list()?
            .iter()
            .map(parse_label)
            .collect::<Result<Vec<_>>>()?;
        let resources = value
            .get("resources")
            .ok_or_else(|| {
                anyhow::anyhow!("{COLLECTION_NSID}: missing required field \"resources\"")
            })?
            .as_list()?
            .iter()
            .map(parse_resource)
            .collect::<Result<Vec<_>>>()?;
        let groups = value
            .get("groups")
            .ok_or_else(|| anyhow::anyhow!("{COLLECTION_NSID}: missing required field \"groups\""))?
            .as_list()?
            .iter()
            .map(|g| Ok(g.as_str()?.to_owned()))
            .collect::<Result<Vec<_>>>()?;
        let created_at = map_get_str(value, "createdAt", COLLECTION_NSID)?.to_owned();
        // Reject unknown extra fields (the lexicon is frozen at 5 fields).
        for (k, _v) in map {
            match k.as_str()? {
                "repo" | "labels" | "resources" | "groups" | "createdAt" => {}
                other => {
                    bail!("{COLLECTION_NSID}: unknown field {other:?} (lexicon is 5 fields)")
                }
            }
        }
        Self::new(repo, labels, resources, groups, created_at)
    }
}

fn parse_label(value: &DagCbor) -> Result<Label> {
    let map = value.as_map()?;
    let key = map_get_str(value, "key", COLLECTION_NSID)?.to_owned();
    let val = map_get_str(value, "value", COLLECTION_NSID)?.to_owned();
    for (k, _v) in map {
        match k.as_str()? {
            "key" | "value" => {}
            other => {
                bail!("{COLLECTION_NSID}: unknown label field {other:?} (lexicon is 2 fields)")
            }
        }
    }
    Ok(Label { key, value: val })
}

fn parse_resource(value: &DagCbor) -> Result<Resource> {
    let map = value.as_map()?;
    let name = map_get_str(value, "name", COLLECTION_NSID)?.to_owned();
    let capacity = map_get_str(value, "capacity", COLLECTION_NSID)?.to_owned();
    for (k, _v) in map {
        match k.as_str()? {
            "name" | "capacity" => {}
            other => {
                bail!("{COLLECTION_NSID}: unknown resource field {other:?} (lexicon is 2 fields)")
            }
        }
    }
    Ok(Resource { name, capacity })
}

// ── nested-object validators ────────────────────────────────────────────────

fn validate_label(l: &Label) -> Result<()> {
    ensure!(!l.key.is_empty(), "label key must not be empty");
    Ok(())
}

fn validate_resource(r: &Resource) -> Result<()> {
    ensure!(!r.name.is_empty(), "resource name must not be empty");
    validate_k8s_quantity(&r.capacity)?;
    Ok(())
}

/// A Kubernetes quantity string (e.g. `"16"`, `"100m"`, `"8Gi"`, `"2.5"`). We
/// do not fully parse the suffix grammar here; we reject the obviously-invalid
/// forms (empty, whitespace, or a non-numeric leading character).
fn validate_k8s_quantity(s: &str) -> Result<()> {
    ensure!(!s.is_empty(), "k8s quantity must not be empty: {s:?}");
    ensure!(
        !s.chars().any(char::is_whitespace),
        "k8s quantity must not contain whitespace: {s:?}"
    );
    let first = s.chars().next().unwrap_or(' ');
    ensure!(
        first.is_ascii_digit() || first == '.' || first == '+' || first == '-',
        "k8s quantity must start with a digit/sign/dot: {s:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;

    fn sample() -> NodeRecord {
        NodeRecord::new(
            "at://did:web:alice.example.com",
            vec![Label {
                key: "zone".into(),
                value: "us-east".into(),
            }],
            vec![Resource {
                name: "nvidia.com/gpu".into(),
                capacity: "8".into(),
            }],
            vec!["at://did:web:alice.example.com/ai.hyprstream.placement.group/3kxy".into()],
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid sample")
    }

    #[test]
    fn record_round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = NodeRecord::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
        assert_eq!(r.to_dag_cbor(), back.to_dag_cbor());
    }

    #[test]
    fn record_fields_canonical_order() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let v = DagCbor::decode(&bytes).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        // canonical (lexicographic byte) key order.
        assert_eq!(
            keys,
            vec!["createdAt", "groups", "labels", "repo", "resources"]
        );
    }

    #[test]
    fn record_rejects_extra_fields() {
        let mut extra = sample().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("extra".into()), DagCbor::Unsigned(1)));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(NodeRecord::from_value(&extra).is_err());
    }

    #[test]
    fn record_rejects_unknown_nested_label_field() {
        let r = sample();
        let mut v = r.to_value();
        if let DagCbor::Map(ref mut pairs) = v {
            for (k, val) in pairs.iter_mut() {
                if k.as_str().ok() == Some("labels") {
                    if let DagCbor::List(items) = val {
                        if let Some(DagCbor::Map(obj)) = items.first_mut() {
                            obj.push((DagCbor::Text("zzz".into()), DagCbor::Text("x".into())));
                        }
                    }
                }
            }
        }
        assert!(NodeRecord::from_value(&v).is_err());
    }

    #[test]
    fn record_validates_formats() {
        // Bad repo at-uri.
        assert!(
            NodeRecord::new("nope", vec![], vec![], vec![], "2026-06-23T12:34:56.789Z").is_err()
        );
        // Bad group at-uri.
        assert!(NodeRecord::new(
            "at://did:web:x",
            vec![],
            vec![],
            vec!["not-an-at-uri".into()],
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Bad datetime.
        assert!(NodeRecord::new("at://did:web:x", vec![], vec![], vec![], "2026-06-23").is_err());
        // Empty label key.
        assert!(NodeRecord::new(
            "at://did:web:x",
            vec![Label {
                key: "".into(),
                value: "v".into()
            }],
            vec![],
            vec![],
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Bad k8s quantity.
        assert!(NodeRecord::new(
            "at://did:web:x",
            vec![],
            vec![Resource {
                name: "cpu".into(),
                capacity: "lots".into()
            }],
            vec![],
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
    }
}
