//! `ai.hyprstream.placement.*` — the P1 placement-directory lexicons.
//!
//! Four DAG-CBOR record types form the consent-scoped placement directory that
//! sits alongside [`crate::record::ModelRecord`] (`ai.hyprstream.model`):
//!
//! - [`node::NodeRecord`] (`ai.hyprstream.placement.node`) — a schedulable node
//!   advertised by an account: its labels, capacities, and group consents.
//! - [`workload::WorkloadRecord`] (`ai.hyprstream.placement.workload`) — a
//!   workload→node placement decision.
//! - [`group::GroupRecord`] (`ai.hyprstream.placement.group`) — an atproto *list*
//!   record naming a consent group.
//! - [`group_item::GroupItemRecord`] (`ai.hyprstream.placement.groupItem`) — an
//!   atproto *listitem* record adding a node (subject) to a group.
//!
//! # Encoding
//!
//! Like `ai.hyprstream.model`, every record is **DAG-CBOR** with canonical
//! map-key order — **pure lexicographic byte order** (RFC 7049 §4.2.1 "core
//! determinism", not length-first). The encoder applies that ordering, so
//! callers construct fields in any order. Each record has a frozen
//! field set: decoding rejects unknown fields. Optional fields are *omitted*
//! from the encoded map when absent (never encoded as `null`).
//!
//! All format-bearing fields are validated at construction time, mirroring
//! `record.rs`: `at-uri` fields must start with `at://`, `did` fields must start
//! with `did:` (both `did:web` and `did:key` are accepted), and `datetime`
//! fields must match the atproto ISO-8601 shape (UTC, milliseconds, trailing
//! `Z`).

use anyhow::{ensure, Result};

use crate::dag_cbor::DagCbor;

pub mod group;
pub mod group_item;
pub mod node;
pub mod workload;

pub use group::GroupRecord;
pub use group_item::GroupItemRecord;
pub use node::NodeRecord;
pub use workload::WorkloadRecord;

// ── shared field accessors ──────────────────────────────────────────────────

/// Read a required string field, attributing errors to `nsid`.
pub(crate) fn map_get_str<'a>(value: &'a DagCbor, key: &str, nsid: &str) -> Result<&'a str> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{nsid}: missing required field {key:?}"))?
        .as_str()
}

/// Read an optional string field. Returns `None` when the field is omitted; an
/// error if it is present but not a text string.
pub(crate) fn map_get_opt_str<'a>(value: &'a DagCbor, key: &str) -> Result<Option<&'a str>> {
    match value.get(key) {
        None => Ok(None),
        Some(v) => Ok(Some(v.as_str()?)),
    }
}

// ── field validators (lexicon formats) ──────────────────────────────────────

/// `format: "at-uri"` — must start with `at://` and carry a non-empty,
/// whitespace-free authority. (DID grammar is enforced by the resolver.)
pub(crate) fn validate_at_uri(s: &str) -> Result<()> {
    ensure!(
        s.starts_with("at://"),
        "at-uri must start with \"at://\": {s:?}"
    );
    let rest = &s[5..];
    ensure!(!rest.is_empty(), "at-uri must have an authority: {s:?}");
    ensure!(
        !rest.chars().any(char::is_whitespace),
        "at-uri must not contain whitespace: {s:?}"
    );
    Ok(())
}

/// A DID string — must start with `did:` and carry a non-empty, whitespace-free
/// method-specific identifier. Both `did:web:…` and `did:key:…` are accepted.
pub(crate) fn validate_did(s: &str) -> Result<()> {
    ensure!(s.starts_with("did:"), "did must start with \"did:\": {s:?}");
    let rest = &s[4..];
    ensure!(!rest.is_empty(), "did must have a method: {s:?}");
    ensure!(
        !rest.chars().any(char::is_whitespace),
        "did must not contain whitespace: {s:?}"
    );
    // method:identifier — both segments non-empty.
    let mut parts = rest.splitn(2, ':');
    let method = parts.next().unwrap_or("");
    let id = parts.next().unwrap_or("");
    ensure!(
        !method.is_empty() && !id.is_empty(),
        "did must be \"did:<method>:<id>\": {s:?}"
    );
    Ok(())
}

/// `format: "datetime"` — atproto ISO-8601 UTC, millisecond precision, `Z`.
pub(crate) fn validate_datetime(s: &str) -> Result<()> {
    ensure!(s.ends_with('Z'), "datetime must end with 'Z' (UTC): {s:?}");
    let pre = &s[..s.len() - 1];
    ensure!(pre.len() >= 20, "datetime too short: {s:?}");
    let bytes = pre.as_bytes();
    ensure!(
        bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes[10] == b'T'
            && bytes[13] == b':'
            && bytes[16] == b':',
        "datetime must be ISO-8601 (YYYY-MM-DDTHH:MM:SS): {s:?}"
    );
    ensure!(
        bytes.get(19) == Some(&b'.'),
        "datetime must have millisecond precision (.mmm): {s:?}"
    );
    Ok(())
}

/// A required free-form string that must not be empty (`field` names it for the
/// error message).
pub(crate) fn validate_nonempty(s: &str, field: &str) -> Result<()> {
    ensure!(!s.is_empty(), "{field} must not be empty");
    Ok(())
}
