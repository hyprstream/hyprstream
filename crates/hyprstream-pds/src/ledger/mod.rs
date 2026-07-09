//! `ai.hyprstream.ledger.*` — the cellular-ledger inventory lexicons (#924).
//!
//! Three DAG-CBOR record types form the **holder-controlled inventory**: the
//! signed entitlements, usage receipts, and ledger heads that live in a holder's
//! own atproto repo (its signed PDS commit is the only signature these records
//! need — see [`crate::commit`]).
//!
//! - [`allocation::AllocationRecord`] (`ai.hyprstream.ledger.allocation`) — an
//!   issuer-signed grant: the held entitlement. References the UCAN grant CID,
//!   carries the issuer-named [`allocation::Unit`], amount, epoch, issuer DID,
//!   holder (subject), and the grant [`allocation::GrantClass`].
//! - [`receipt::ReceiptRecord`] (`ai.hyprstream.ledger.receipt`) — a dual-signed
//!   usage record binding **both** principals (spender + host, the #681
//!   two-principal vocabulary), the referenced allocation CID, the transfer id,
//!   and the quantum spent.
//! - [`checkpoint::CheckpointRecord`] (`ai.hyprstream.ledger.checkpoint`) — a
//!   signed ledger head (seq, head hash, state roots) that **references** — never
//!   embeds — an external anchor.
//!
//! # Pseudonymity constraints (the #928 charter — review-blockers)
//!
//! These lexicons are the P0 target of the pseudonymity charter. The rules are
//! enforced **structurally** — by the shape of the records, not by a runtime
//! check that could be bypassed:
//!
//! 1. **No legal-identity field, anywhere.** Every record has a frozen field set
//!    and decoding rejects unknown fields, so a `legalName`/`kyc`/… field is not
//!    representable: it fails to decode. Audit binds pseudonyms only.
//! 2. **Pairwise per-cell subject identifiers.** The `holder`/`spender`/`host`
//!    fields are validated by [`validate_subject_id`], which accepts a `did:key`
//!    / `did:at9p` *or* an opaque pairwise identifier — it never *requires* a
//!    DID, let alone a `did:web`. A cell-scoped pairwise ID drops straight in.
//! 3. **The unit names its issuer.** [`allocation::Unit`] bakes the issuer DID
//!    into the unit itself — there is no bare unit string. Credits are issuer
//!    *liabilities* (D8-1), never bearer tokens; [`allocation::AllocationRecord`]
//!    additionally requires the top-level `issuer` to equal `unit.issuer`.
//! 4. **Grant class couples anonymity to spend mode.**
//!    [`allocation::GrantClass::Prepaid`] is bearer-like, issuable to a bare
//!    `did:key`, and lease-mode-only; [`allocation::GrantClass::Underwritten`]
//!    is the only detect-mode-eligible class (see
//!    [`allocation::GrantClass::detect_mode_eligible`]).
//!
//! # Encoding
//!
//! Like the other hyprstream lexicons, every record is **DAG-CBOR** with
//! canonical (pure-lexicographic) map-key order applied by the encoder, so
//! callers construct fields in any order. Each record has a frozen field set:
//! decoding rejects unknown fields. Optional fields are *omitted* from the
//! encoded map when absent (never encoded as `null`). CID references are carried
//! as `format: "cid"` **strings** (not DAG-CBOR links): the referenced blocks
//! (UCAN grants, allocations, external ledger state) live outside this repo's
//! MST, so a tag-42 link would be a false claim of local containment — the same
//! reasoning [`crate::record::ModelRecord::current_oid`] documents.

use anyhow::{bail, ensure, Result};

use crate::dag_cbor::DagCbor;

pub mod allocation;
pub mod checkpoint;
pub mod receipt;

pub use allocation::{AllocationRecord, GrantClass, Unit};
pub use checkpoint::{CheckpointRecord, StateRoot};
pub use receipt::ReceiptRecord;

// ── shared field accessors ──────────────────────────────────────────────────

/// Read a required string field, attributing errors to `nsid`.
pub(crate) fn map_get_str<'a>(value: &'a DagCbor, key: &str, nsid: &str) -> Result<&'a str> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{nsid}: missing required field {key:?}"))?
        .as_str()
}

/// Read a required unsigned-integer field, attributing errors to `nsid`.
pub(crate) fn map_get_uint(value: &DagCbor, key: &str, nsid: &str) -> Result<u64> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{nsid}: missing required field {key:?}"))?
        .as_unsigned()
}

/// Read an optional string field. Returns `None` when omitted; an error if it is
/// present but not a text string.
pub(crate) fn map_get_opt_str<'a>(value: &'a DagCbor, key: &str) -> Result<Option<&'a str>> {
    match value.get(key) {
        None => Ok(None),
        Some(v) => Ok(Some(v.as_str()?)),
    }
}

// ── field validators (lexicon formats) ──────────────────────────────────────

/// A DID string — must start with `did:` and carry a non-empty, whitespace-free
/// `method:identifier`. Both `did:web`, `did:key`, and `did:at9p` are accepted.
/// Used for fields that name a *real, signing* identity (an issuer), never for a
/// pseudonymous subject (see [`validate_subject_id`]).
pub(crate) fn validate_did(s: &str) -> Result<()> {
    ensure!(s.starts_with("did:"), "did must start with \"did:\": {s:?}");
    let rest = &s[4..];
    ensure!(!rest.is_empty(), "did must have a method: {s:?}");
    ensure!(
        !rest.chars().any(char::is_whitespace),
        "did must not contain whitespace: {s:?}"
    );
    let mut parts = rest.splitn(2, ':');
    let method = parts.next().unwrap_or("");
    let id = parts.next().unwrap_or("");
    ensure!(
        !method.is_empty() && !id.is_empty(),
        "did must be \"did:<method>:<id>\": {s:?}"
    );
    Ok(())
}

/// A **pseudonymous subject identifier** (`holder`/`spender`/`host`).
///
/// The #928 charter (rule 3) requires these fields to carry either a
/// pseudonymous DID (`did:key`/`did:at9p`) *or* an opaque **pairwise per-cell
/// identifier** — an issuer-linkable, cell-unlinkable string the subject
/// presents differently to each cell. We therefore accept any non-empty,
/// whitespace-free token and deliberately **do not** require a DID shape: forcing
/// a DID here would forbid the pairwise-ID presentation the charter mandates and
/// would push subjects toward a stable global identity. (A `did:web` — an opt-in
/// linkage to a domain — is likewise never required for participation.)
pub(crate) fn validate_subject_id(s: &str, field: &str) -> Result<()> {
    ensure!(!s.is_empty(), "{field} must not be empty");
    ensure!(
        !s.chars().any(char::is_whitespace),
        "{field} must not contain whitespace: {s:?}"
    );
    Ok(())
}

/// `format: "cid"` — a CIDv1 (`b…`/`z…`) or CIDv0 (`Qm…`) string. Mirrors the
/// `currentOid` validation in [`crate::record`]: we carry external content
/// addresses as strings and reject the obviously-bogus forms.
pub(crate) fn validate_cid_string(s: &str, field: &str) -> Result<()> {
    ensure!(!s.is_empty(), "{field} cid string must be non-empty");
    let ok = (s.starts_with('b') && s.len() > 8)
        || (s.starts_with('z') && s.len() > 8)
        || (s.starts_with("Qm") && s.len() > 8);
    ensure!(
        ok,
        "{field} must be a CIDv1 (b/z…) or CIDv0 (Qm…) string: {s:?}"
    );
    Ok(())
}

/// A required free-form string that must not be empty or contain whitespace
/// (`field` names it for the error message). Used for opaque locators (transfer
/// ids, anchor references) that are neither DIDs nor CIDs.
pub(crate) fn validate_token(s: &str, field: &str) -> Result<()> {
    ensure!(!s.is_empty(), "{field} must not be empty");
    ensure!(
        !s.chars().any(char::is_whitespace),
        "{field} must not contain whitespace: {s:?}"
    );
    Ok(())
}

/// Reject any map key not in `allowed`, attributing the error to `nsid`.
pub(crate) fn reject_unknown_fields(value: &DagCbor, allowed: &[&str], nsid: &str) -> Result<()> {
    let map = value.as_map()?;
    for (k, _v) in map {
        let key = k.as_str()?;
        if !allowed.contains(&key) {
            bail!("{nsid}: unknown field {key:?} (frozen lexicon fields: {allowed:?})");
        }
    }
    Ok(())
}
