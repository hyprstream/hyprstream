//! R7 v1 private AGPL Flight decide-substrate: bounded prepared registry,
//! immutable pin resolution seam, and the request-local secret wrapper.
//!
//! doc(hidden) throughout: this is the Flight-side substrate consumed by the
//! embedding binary's injected callbacks — it is NOT a public backbone API and
//! carries no Apache-core trait. The G1 `authorize_verified` boundary remains
//! REQUIRED; every operator method re-derives the validated owner key from the
//! verified context and fails closed without a tenant.
//!
//! S1 scope: registry + resolver only. The execute loop is DEFERRED until a
//! reviewed `ExecutionControl` protocol (deadline/cancel/terminal permits)
//! exists; `DecisionSubstrateError::ExecutionFailed` is reserved so the error
//! surface is already frozen. `DecisionPins::quantization`/
//! `arrow_schema_identity` and the raw ordered question template exist now
//! because the approved execute-time response matching requires them.
//!
//! Panicking is correct in the in-file unit fixtures: unwrap/expect are
//! allowed module-wide, mirroring the repo's other test modules.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::VerifiedFlightContext;
use hyprstream_decision::arrow::DecisionSchema;
use hyprstream_decision::spec::QuestionSet;

use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

/// Owned substrate error. Transport mapping lives in exactly one place.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionSubstrateError {
    /// Malformed document, limits violated, empty/wildcard tenant, inconsistent
    /// resolution output.
    InvalidArgument(String),
    /// No verified owner context / expired verified context.
    Unauthenticated(String),
    /// Unknown, closed, or cross-owner handle (deliberately indistinguishable).
    NotFound,
    /// Any statement/row/aggregate-byte cap.
    ResourceExhausted(String),
    /// Resolver-reported configured-alias / expected-artifact-commit mismatch.
    PinMismatch(String),
    /// Resolver-reported failure. Fail closed; metadata absence fails here.
    ResolutionFailed(String),
    /// RESERVED for the deferred execute loop (error surface frozen now).
    ExecutionFailed(String),
}

impl DecisionSubstrateError {
    /// Single transport mapping for the Flight SQL routes.
    pub fn into_status(self) -> tonic::Status {
        use tonic::Status;
        match self {
            Self::InvalidArgument(m) => Status::invalid_argument(m),
            Self::Unauthenticated(m) => Status::unauthenticated(m),
            Self::NotFound => Status::not_found(
                "unknown prepared statement handle (stale or drifted; re-prepare)",
            ),
            Self::ResourceExhausted(m) => Status::resource_exhausted(m),
            Self::PinMismatch(m) | Self::ResolutionFailed(m) => Status::failed_precondition(m),
            Self::ExecutionFailed(m) => Status::internal(m),
        }
    }
}

impl From<DecisionSubstrateError> for tonic::Status {
    fn from(value: DecisionSubstrateError) -> Self {
        value.into_status()
    }
}

// ---------------------------------------------------------------------------
// owner key (validated creator identity; no R6 type dependency in S1)
// ---------------------------------------------------------------------------

const MAX_OWNER_LEN: usize = 128;

fn valid_owner_part(value: &str) -> bool {
    !value.is_empty()
        && value != "*"
        && value.len() <= MAX_OWNER_LEN
        && !value.chars().any(char::is_control)
        && value.trim() == value
}

/// Validated creator identity: BOTH coordinates are copied from the G1
/// verified context and enforced on every registry operation, so two
/// principals in the same tenant cannot touch each other's statements.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OwnerKey {
    tenant: String,
    subject: String,
}

impl OwnerKey {
    /// Derive from the verified context. Fails closed without a tenant and
    /// rejects empty/wildcard/control/over-long values.
    pub fn from_verified(ctx: &VerifiedFlightContext) -> Result<Self, DecisionSubstrateError> {
        let tenant = ctx.tenant.as_deref().ok_or_else(|| {
            DecisionSubstrateError::Unauthenticated(
                "verified caller context carries no tenant".to_owned(),
            )
        })?;
        if !valid_owner_part(tenant) {
            return Err(DecisionSubstrateError::Unauthenticated(
                "verified tenant is not a bounded domain".to_owned(),
            ));
        }
        let subject = ctx.subject.clone();
        if !valid_owner_part(&subject) {
            return Err(DecisionSubstrateError::Unauthenticated(
                "verified caller subject is not a bounded identity".to_owned(),
            ));
        }
        Ok(Self {
            tenant: tenant.to_owned(),
            subject,
        })
    }

    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    pub fn subject(&self) -> &str {
        &self.subject
    }
}

// ---------------------------------------------------------------------------
// request-local secret wrapper
// ---------------------------------------------------------------------------

/// The original caller's bearer, request-locally only: it is never stored in
/// the registry, never serialized, never logged, and drops with the request.
/// Consumed exclusively by the injected resolve callback to present the
/// original end-user credential to the Model service.
#[doc(hidden)]
pub struct OriginalBearer(String);

impl OriginalBearer {
    /// Extract from the request authorization header value. Rejects missing
    /// values, non-Bearer schemes, and empty tokens.
    pub fn from_authorization_header(value: Option<&str>) -> Result<Self, DecisionSubstrateError> {
        let value = value.ok_or_else(|| {
            DecisionSubstrateError::Unauthenticated("original bearer required".to_owned())
        })?;
        let token = value.strip_prefix("Bearer ").ok_or_else(|| {
            DecisionSubstrateError::Unauthenticated("original bearer required".to_owned())
        })?;
        if token.trim().is_empty() {
            return Err(DecisionSubstrateError::Unauthenticated(
                "original bearer required".to_owned(),
            ));
        }
        Ok(Self(token.to_owned()))
    }

    /// Expose the secret to the injected callback only.
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for OriginalBearer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OriginalBearer(REDACTED)")
    }
}

// ---------------------------------------------------------------------------
// pins + raw template (immutable observed identity)
// ---------------------------------------------------------------------------

const MAX_PIN_LEN: usize = 256;

fn non_bounded(value: &str, what: &str) -> Result<(), DecisionSubstrateError> {
    if value.is_empty() || value.len() > MAX_PIN_LEN || value.chars().any(char::is_control) {
        return Err(DecisionSubstrateError::InvalidArgument(format!(
            "{what} must be non-empty, bounded, and free of control characters"
        )));
    }
    Ok(())
}

fn is_full_oid(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Complete immutable observed identity of a prepared decide statement —
/// everything the execute-time response match needs, captured once at CREATE.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionPins {
    /// Fixed CONFIGURED authorization alias (routing/policy coordinate; equal
    /// to the trusted staging pin's model_ref). Never user-supplied.
    pub authorization_model_ref: String,
    /// Independently observed immutable artifact commit (full 40-hex OID).
    pub expected_artifact_commit: String,
    /// The decide document's real schema version.
    pub schema_version: String,
    /// Resolved model id/version from Model metadata.
    pub model_id: String,
    /// Clef encoding contract id.
    pub encoding_version: String,
    /// Quantization profile pin (e.g. "fp8-e4m3-dynamic").
    pub quantization: String,
    /// Server-admitted compute profile pin (e.g. "cpu-f32").
    pub admitted_compute_profile: String,
    pub head_weights_sha256: String,
    pub head_config_sha256: String,
    pub tokenizer_sha256: String,
    pub backbone_config_sha256: String,
    /// Arrow identity from metadata (matched against the resolved
    /// fingerprint at CREATE; re-matched at execute against responses).
    pub arrow_schema_identity: String,
    /// Calibration version; `None` is raw/uncalibrated and is preserved.
    pub calibration_version: Option<String>,
}

impl DecisionPins {
    /// Structural validation: exact OID/digest shapes, bounded non-empty
    /// pins, no wildcard, calibration Option preserved (never defaulted).
    pub fn validate(&self) -> Result<(), DecisionSubstrateError> {
        non_bounded(&self.authorization_model_ref, "authorization_model_ref")?;
        if !is_full_oid(&self.expected_artifact_commit) {
            return Err(DecisionSubstrateError::InvalidArgument(
                "expected_artifact_commit must be a full 40-hex commit OID".to_owned(),
            ));
        }
        non_bounded(&self.schema_version, "schema_version")?;
        non_bounded(&self.model_id, "model_id")?;
        non_bounded(&self.encoding_version, "encoding_version")?;
        non_bounded(&self.quantization, "quantization")?;
        non_bounded(&self.admitted_compute_profile, "admitted_compute_profile")?;
        for (name, digest) in [
            ("head_weights_sha256", &self.head_weights_sha256),
            ("head_config_sha256", &self.head_config_sha256),
            ("tokenizer_sha256", &self.tokenizer_sha256),
            ("backbone_config_sha256", &self.backbone_config_sha256),
            ("arrow_schema_identity", &self.arrow_schema_identity),
        ] {
            non_bounded(digest, name)?;
        }
        if !is_sha256_hex(&self.head_weights_sha256)
            || !is_sha256_hex(&self.head_config_sha256)
            || !is_sha256_hex(&self.tokenizer_sha256)
            || !is_sha256_hex(&self.backbone_config_sha256)
        {
            return Err(DecisionSubstrateError::InvalidArgument(
                "artifact digests must be full 64-hex sha256 values".to_owned(),
            ));
        }
        if let Some(calib) = &self.calibration_version {
            non_bounded(calib, "calibration_version")?;
        }
        Ok(())
    }
}

/// FULL ordered raw question template: the exact ordered `questions` JSON from
/// the decide document (instructions/criteria and raw numeric spelling
/// preserved). The parsed [`QuestionSet`] is a view; this template is the
/// reconstructible native input.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawQuestionsTemplate {
    pub raw_json: Arc<str>,
    pub question_ids: Arc<[String]>,
}

/// Resolver output at CREATE. Everything Arc-shared so registry snapshots are
/// handle-only clones (no unaccounted byte copies).
#[doc(hidden)]
#[derive(Clone)]
pub struct ResolvedDecisionPins {
    pub pins: Arc<DecisionPins>,
    pub question_set: Arc<QuestionSet>,
    pub raw: Arc<RawQuestionsTemplate>,
    /// `DecisionSchema::fingerprint()` of the parsed set. The operator
    /// recomputes this independently and rejects mismatches before insert.
    pub arrow_fingerprint: Arc<str>,
}

const MAX_LIVE_PREPARED_PAYLOAD_BYTES: usize = 128 * 1024 * 1024;

fn add_retained_bytes(total: &mut usize, bytes: usize) -> Result<(), DecisionSubstrateError> {
    *total = total.checked_add(bytes).ok_or_else(|| {
        DecisionSubstrateError::ResourceExhausted("prepared payload size overflow".to_owned())
    })?;
    Ok(())
}

fn retained_capacity_bytes<T>(capacity: usize) -> Result<usize, DecisionSubstrateError> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| {
            DecisionSubstrateError::ResourceExhausted("prepared payload size overflow".to_owned())
        })
}

fn entry_retained_bytes(
    entry: &hyprstream_decision::entry::Entry,
) -> Result<usize, DecisionSubstrateError> {
    use hyprstream_decision::entry::Entry;
    let mut total = std::mem::size_of::<Entry>();
    match entry {
        Entry::Str(value) => add_retained_bytes(&mut total, value.capacity())?,
        Entry::Seq(values) => {
            add_retained_bytes(
                &mut total,
                retained_capacity_bytes::<Entry>(values.capacity())?,
            )?;
            for value in values {
                add_retained_bytes(&mut total, entry_retained_bytes(value)?)?;
            }
        }
        Entry::Map(values) => {
            add_retained_bytes(
                &mut total,
                retained_capacity_bytes::<(String, Entry)>(values.capacity())?,
            )?;
            for (key, value) in values {
                add_retained_bytes(&mut total, key.capacity())?;
                add_retained_bytes(&mut total, entry_retained_bytes(value)?)?;
            }
        }
        Entry::Null | Entry::Bool(_) | Entry::Number(_) => {}
    }
    Ok(total)
}

fn question_set_retained_bytes(set: &QuestionSet) -> Result<usize, DecisionSubstrateError> {
    use hyprstream_decision::spec::{QuestionBody, QuestionSet};
    let mut total = std::mem::size_of::<QuestionSet>();
    add_retained_bytes(
        &mut total,
        retained_capacity_bytes::<hyprstream_decision::spec::QuestionSpec>(
            set.questions.capacity(),
        )?,
    )?;
    if let Some(state) = &set.state {
        add_retained_bytes(&mut total, entry_retained_bytes(state)?)?;
    }
    for question in &set.questions {
        add_retained_bytes(&mut total, question.id.capacity())?;
        if let Some(instructions) = &question.instructions {
            add_retained_bytes(&mut total, entry_retained_bytes(instructions)?)?;
        }
        match &question.body {
            QuestionBody::Noul { criteria } => {
                if let Some(criteria) = criteria {
                    add_retained_bytes(&mut total, std::mem::size_of_val(criteria))?;
                    if let Some(value) = &criteria.on_true {
                        add_retained_bytes(&mut total, entry_retained_bytes(value)?)?;
                    }
                    if let Some(value) = &criteria.on_false {
                        add_retained_bytes(&mut total, entry_retained_bytes(value)?)?;
                    }
                }
            }
            QuestionBody::Choice { options } => {
                add_retained_bytes(
                    &mut total,
                    retained_capacity_bytes::<hyprstream_decision::spec::ChoiceOption>(
                        options.capacity(),
                    )?,
                )?;
                for option in options {
                    add_retained_bytes(&mut total, option.name.capacity())?;
                    if let Some(rubric) = &option.rubric {
                        add_retained_bytes(&mut total, entry_retained_bytes(rubric)?)?;
                    }
                }
            }
            QuestionBody::Score { levels } => {
                add_retained_bytes(
                    &mut total,
                    retained_capacity_bytes::<Option<hyprstream_decision::entry::Entry>>(
                        levels.capacity(),
                    )?,
                )?;
                for level in levels.iter().flatten() {
                    add_retained_bytes(&mut total, entry_retained_bytes(level)?)?;
                }
            }
        }
    }
    Ok(total)
}

fn prepared_payload_retained_bytes(
    payload: &ResolvedDecisionPins,
    owner: &OwnerKey,
) -> Result<usize, DecisionSubstrateError> {
    let mut total = std::mem::size_of::<DecisionPins>()
        + std::mem::size_of::<RawQuestionsTemplate>()
        + std::mem::size_of::<String>()
        + std::mem::size_of::<QuestionSet>()
        + std::mem::size_of::<OwnerKey>();
    add_retained_bytes(&mut total, owner.tenant.capacity())?;
    add_retained_bytes(&mut total, owner.subject.capacity())?;
    for pin in [
        &payload.pins.authorization_model_ref,
        &payload.pins.expected_artifact_commit,
        &payload.pins.schema_version,
        &payload.pins.model_id,
        &payload.pins.encoding_version,
        &payload.pins.quantization,
        &payload.pins.admitted_compute_profile,
        &payload.pins.head_weights_sha256,
        &payload.pins.head_config_sha256,
        &payload.pins.tokenizer_sha256,
        &payload.pins.backbone_config_sha256,
        &payload.pins.arrow_schema_identity,
    ] {
        add_retained_bytes(&mut total, pin.capacity())?;
    }
    if let Some(calibration) = &payload.pins.calibration_version {
        add_retained_bytes(&mut total, calibration.capacity())?;
    }
    // CREATE stores a second owned copy for the public prepared-statement
    // snapshot API, in addition to the schema-version pin above.
    add_retained_bytes(&mut total, payload.pins.schema_version.len())?;
    add_retained_bytes(&mut total, payload.raw.raw_json.len())?;
    add_retained_bytes(
        &mut total,
        retained_capacity_bytes::<String>(payload.raw.question_ids.len())?,
    )?;
    for id in payload.raw.question_ids.iter() {
        add_retained_bytes(&mut total, id.capacity())?;
    }
    add_retained_bytes(&mut total, payload.arrow_fingerprint.len())?;
    add_retained_bytes(
        &mut total,
        question_set_retained_bytes(&payload.question_set)?,
    )?;
    // Include Arc/control-block and small metadata allocations conservatively.
    add_retained_bytes(&mut total, 8 * std::mem::size_of::<usize>())?;
    Ok(total)
}

fn binding_metadata_bytes(rows: usize) -> Result<usize, DecisionSubstrateError> {
    let row_bytes = rows
        .checked_mul(std::mem::size_of::<String>())
        .ok_or_else(|| {
            DecisionSubstrateError::ResourceExhausted("binding size overflow".to_owned())
        })?;
    row_bytes
        .checked_add(std::mem::size_of::<ChargedBinding>())
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<usize>()))
        .ok_or_else(|| {
            DecisionSubstrateError::ResourceExhausted("binding size overflow".to_owned())
        })
}

struct LiveAllocationLease {
    bytes: usize,
    live: Arc<AtomicU64>,
}

impl LiveAllocationLease {
    fn reserve(live: &Arc<AtomicU64>, bytes: usize) -> Result<Self, DecisionSubstrateError> {
        let bytes_u64 = u64::try_from(bytes).map_err(|_| {
            DecisionSubstrateError::ResourceExhausted("prepared payload size overflow".to_owned())
        })?;
        loop {
            let current = live.load(Ordering::Relaxed);
            let next = current.checked_add(bytes_u64).ok_or_else(|| {
                DecisionSubstrateError::ResourceExhausted(
                    "prepared payload accounting overflow".to_owned(),
                )
            })?;
            if next > MAX_LIVE_PREPARED_PAYLOAD_BYTES as u64 {
                return Err(DecisionSubstrateError::ResourceExhausted(
                    "global live prepared-payload byte cap".to_owned(),
                ));
            }
            if live
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(Self {
                    bytes,
                    live: Arc::clone(live),
                });
            }
        }
    }
}

impl Drop for LiveAllocationLease {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes as u64, Ordering::AcqRel);
    }
}

// ---------------------------------------------------------------------------
// limits + accounting (statement counts and aggregate bound bytes, 3 levels)
// ---------------------------------------------------------------------------

/// Hard cap on a single prepare document (bytes).
pub const MAX_PREPARE_DOCUMENT_BYTES: usize = 4 * 1024 * 1024;

/// Lossless view of the top-level prepare document. `RawValue` retains the
/// caller's exact ordered questions JSON, while serde's derived struct rejects
/// duplicate `schema_version`, `questions`, `model`, or `state` fields.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CallerPrepareDocument<'a> {
    schema_version: &'a str,
    #[serde(borrow)]
    questions: &'a serde_json::value::RawValue,
    #[serde(default, borrow)]
    calib_version: Option<&'a serde_json::value::RawValue>,
    #[serde(default, borrow)]
    model: Option<&'a serde_json::value::RawValue>,
    #[serde(default, borrow)]
    state: Option<&'a serde_json::value::RawValue>,
}

/// Bounded ceiling on concurrent prepares so `Semaphore::new` can never
/// panic on an oversized value.
pub const MAX_CONCURRENT_PREPARES_CEILING: usize = 4096;

/// Bounded resource limits. REQUIRED levels (tenant/global statements and
/// aggregate bytes, rows, per-statement bytes, prepare concurrency): `0`
/// means DISABLED (every use denies) — never unlimited. OPTIONAL per-owner
/// levels: `0` means unlimited for that stricter level. Cross-level sanity
/// is validated fallibly (`validate()`); constructors are fallible.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct PreparedDecideLimits {
    pub max_statements_per_owner: usize,
    pub max_statements_per_tenant: usize,
    pub max_statements_global: usize,
    pub max_bound_rows: usize,
    pub max_state_bytes_per_statement: usize,
    pub max_bound_bytes_per_owner: usize,
    pub max_bound_bytes_per_tenant: usize,
    pub max_bound_bytes_global: usize,
    /// Max concurrent CREATE-time resolver calls (bounded prepare queue).
    pub max_concurrent_prepares: usize,
}

impl PreparedDecideLimits {
    /// Fallible construction with REQUIRED-level and cross-level checks.
    pub fn new(
        max_statements_per_owner: usize,
        max_statements_per_tenant: usize,
        max_statements_global: usize,
        max_bound_rows: usize,
        max_state_bytes_per_statement: usize,
        max_bound_bytes_per_owner: usize,
        max_bound_bytes_per_tenant: usize,
        max_bound_bytes_global: usize,
        max_concurrent_prepares: usize,
    ) -> Result<Self, DecisionSubstrateError> {
        let limits = Self {
            max_statements_per_owner,
            max_statements_per_tenant,
            max_statements_global,
            max_bound_rows,
            max_state_bytes_per_statement,
            max_bound_bytes_per_owner,
            max_bound_bytes_per_tenant,
            max_bound_bytes_global,
            max_concurrent_prepares,
        };
        limits.validate()?;
        Ok(limits)
    }

    /// REQUIRED levels: zero = disabled/deny (never unlimited). OPTIONAL
    /// per-owner levels: zero = unlimited. Cross-level: the stricter level
    /// may never exceed the looser one. Prepare concurrency has a fixed
    /// bounded ceiling so the semaphore constructor cannot panic.
    pub fn validate(&self) -> Result<(), DecisionSubstrateError> {
        for (name, value) in ["max_statements_per_tenant", "max_statements_global"]
            .iter()
            .zip([self.max_statements_per_tenant, self.max_statements_global])
        {
            if value == 0 {
                return Err(DecisionSubstrateError::InvalidArgument(format!(
                    "{name} = 0 would disable the prepared registry (zero is deny, not unlimited)"
                )));
            }
        }
        for (name, value) in [
            ("max_bound_rows", self.max_bound_rows),
            (
                "max_state_bytes_per_statement",
                self.max_state_bytes_per_statement,
            ),
            (
                "max_bound_bytes_per_tenant",
                self.max_bound_bytes_per_tenant,
            ),
            ("max_bound_bytes_global", self.max_bound_bytes_global),
        ] {
            if value == 0 {
                return Err(DecisionSubstrateError::InvalidArgument(format!(
                    "{name} = 0 would disable binding (zero is deny, not unlimited)"
                )));
            }
        }
        if self.max_concurrent_prepares == 0 {
            return Err(DecisionSubstrateError::InvalidArgument(
                "max_concurrent_prepares = 0 would disable CREATE (zero is deny, not unlimited)"
                    .to_owned(),
            ));
        }
        if self.max_concurrent_prepares > MAX_CONCURRENT_PREPARES_CEILING {
            return Err(DecisionSubstrateError::InvalidArgument(format!(
                "max_concurrent_prepares {} exceeds the bounded ceiling {}",
                self.max_concurrent_prepares, MAX_CONCURRENT_PREPARES_CEILING
            )));
        }
        Ok(())
    }
}

impl Default for PreparedDecideLimits {
    fn default() -> Self {
        Self {
            max_statements_per_owner: 8,
            max_statements_per_tenant: 32,
            max_statements_global: 256,
            max_bound_rows: 256,
            max_state_bytes_per_statement: 1024 * 1024,
            max_bound_bytes_per_owner: 8 * 1024 * 1024,
            max_bound_bytes_per_tenant: 32 * 1024 * 1024,
            max_bound_bytes_global: 128 * 1024 * 1024,
            max_concurrent_prepares: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StatementGeneration(pub u64);

/// Shared counted cell: drop decrements exactly `bytes`/1 statement.
struct CountedCell {
    value: Arc<AtomicU64>,
}

impl CountedCell {
    fn new(value: Arc<AtomicU64>) -> Self {
        Self { value }
    }
    fn add(&self, bytes: u64) {
        self.value.fetch_add(bytes, Ordering::Relaxed);
    }
    fn sub(&self, bytes: u64) {
        self.value.fetch_sub(bytes, Ordering::Relaxed);
    }
    fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// A live aggregate-byte reservation. The charge follows this handle: while
/// ANY `Arc` to the binding snapshot that owns it exists (entry or an
/// in-flight snapshot), the bytes stay charged. Drop releases without taking
/// the registry lock (atomic decrements only — no destructor deadlock).
#[doc(hidden)]
pub struct ByteLease {
    bytes: usize,
    global: CountedCell,
    tenant: CountedCell,
    owner: CountedCell,
}

impl ByteLease {
    fn reserve(
        accounting: &ByteAccounting,
        owner: &OwnerKey,
        bytes: usize,
    ) -> Result<Self, DecisionSubstrateError> {
        let global = CountedCell::new(Arc::clone(&accounting.bytes_global));
        let tenant = CountedCell::new(accounting.tenant_bytes_cell(owner.tenant())?);
        let owner_cell = CountedCell::new(accounting.owner_bytes_cell(owner)?);
        if accounting.limits.max_bound_bytes_global > 0
            && global.get() + bytes as u64 > accounting.limits.max_bound_bytes_global as u64
        {
            return Err(DecisionSubstrateError::ResourceExhausted(
                "global aggregate bound-byte cap".to_owned(),
            ));
        }
        if accounting.limits.max_bound_bytes_per_tenant > 0
            && tenant.get() + bytes as u64 > accounting.limits.max_bound_bytes_per_tenant as u64
        {
            return Err(DecisionSubstrateError::ResourceExhausted(
                "tenant aggregate bound-byte cap".to_owned(),
            ));
        }
        if accounting.limits.max_bound_bytes_per_owner > 0
            && owner_cell.get() + bytes as u64 > accounting.limits.max_bound_bytes_per_owner as u64
        {
            return Err(DecisionSubstrateError::ResourceExhausted(
                "owner aggregate bound-byte cap".to_owned(),
            ));
        }
        global.add(bytes as u64);
        tenant.add(bytes as u64);
        owner_cell.add(bytes as u64);
        Ok(Self {
            bytes,
            global,
            tenant,
            owner: owner_cell,
        })
    }
}

impl Drop for ByteLease {
    fn drop(&mut self) {
        // Atomic decrements only: never touches the registry lock.
        let bytes = self.bytes as u64;
        self.global.sub(bytes);
        self.tenant.sub(bytes);
        self.owner.sub(bytes);
    }
}

/// Bounded identity-cell caches. Cells whose counter is zero AND whose Arc
/// is only held by the map (strong_count == 1) are prunable: pruning never
/// frees a live lease (a live lease keeps the counter Arc + value alive) and
/// never loses charged bytes (a charged cell is non-zero). When a map is
/// still over its cap after pruning, new identities are REJECTED
/// (ResourceExhausted) rather than evicting charged cells.
/// Identity-cell-map cap: tied to the CONFIGURED global statement ceiling
/// (floor 16). Held closed zero-byte snapshots keep their cells alive past
/// this cap, so new identities are rejected until released.
fn identity_cell_cap(max_statements_global: usize) -> usize {
    max_statements_global.max(16)
}

struct ByteAccounting {
    limits: PreparedDecideLimits,
    bytes_global: Arc<AtomicU64>,
    bytes_tenants: Mutex<BTreeMap<String, Arc<AtomicU64>>>,
    bytes_owners: Mutex<BTreeMap<OwnerKey, Arc<AtomicU64>>>,
}

impl ByteAccounting {
    /// Prune SAFE cells only: zero charge AND no live lease/snapshot holder
    /// (Arc::strong_count == 1 means only the map still references it).
    /// Charged or held cells are never freed.
    fn prune_if_full<K: Ord + Clone>(
        map: &mut BTreeMap<K, Arc<AtomicU64>>,
        cap: usize,
        what: &str,
    ) -> Result<(), DecisionSubstrateError> {
        if map.len() >= cap {
            map.retain(|_, cell| cell.load(Ordering::Relaxed) != 0 || Arc::strong_count(cell) > 1);
            if map.len() >= cap {
                return Err(DecisionSubstrateError::ResourceExhausted(format!(
                    "{what} identity cache full (all entries carry live state)"
                )));
            }
        }
        Ok(())
    }

    fn tenant_bytes_cell(&self, tenant: &str) -> Result<Arc<AtomicU64>, DecisionSubstrateError> {
        let mut map = self.bytes_tenants.lock();
        if let Some(existing) = map.get(tenant) {
            return Ok(Arc::clone(existing));
        }
        ByteAccounting::prune_if_full(
            &mut map,
            identity_cell_cap(self.limits.max_statements_global),
            "tenant",
        )?;
        Ok(Arc::clone(
            map.entry(tenant.to_owned())
                .or_insert_with(|| Arc::new(AtomicU64::new(0))),
        ))
    }

    fn owner_bytes_cell(&self, owner: &OwnerKey) -> Result<Arc<AtomicU64>, DecisionSubstrateError> {
        let mut map = self.bytes_owners.lock();
        if let Some(existing) = map.get(owner) {
            return Ok(Arc::clone(existing));
        }
        ByteAccounting::prune_if_full(
            &mut map,
            identity_cell_cap(self.limits.max_statements_global),
            "owner",
        )?;
        Ok(Arc::clone(
            map.entry(owner.clone())
                .or_insert_with(|| Arc::new(AtomicU64::new(0))),
        ))
    }
}

// ---------------------------------------------------------------------------
// close state (preserved for the later ExecutionControl protocol)
// ---------------------------------------------------------------------------

#[doc(hidden)]
#[derive(Default)]
pub struct CloseState {
    closed: AtomicBool,
}

impl CloseState {
    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// prepared statement + registry
// ---------------------------------------------------------------------------

/// One prepared decide statement, owned by exactly one [`OwnerKey`].
#[doc(hidden)]
#[derive(Clone)]
pub struct PreparedStatement {
    /// RANDOM OPAQUE per CREATE — unrelated to document bytes.
    pub handle: [u8; 32],
    payload: Arc<ChargedPreparedPayload>,
    pub binding: BindingSnapshot,
    pub close_state: Arc<CloseState>,
    pub created_at_unix_ms: i64,
    pub generation: StatementGeneration,
}

/// Immutable model/template payload whose accounting lease lives exactly as
/// long as every retained prepared-statement snapshot.
struct ChargedPreparedPayload {
    owner: OwnerKey,
    pins: DecisionPins,
    question_set: QuestionSet,
    raw_json: String,
    question_ids: Vec<String>,
    arrow_fingerprint: String,
    schema_version: String,
    _lease: LiveAllocationLease,
}

impl PreparedStatement {
    pub fn owner(&self) -> &OwnerKey {
        &self.payload.owner
    }

    pub fn pins(&self) -> &DecisionPins {
        &self.payload.pins
    }

    pub fn question_set(&self) -> &QuestionSet {
        &self.payload.question_set
    }

    pub fn raw_template(&self) -> (&str, &[String]) {
        (&self.payload.raw_json, &self.payload.question_ids)
    }

    pub fn arrow_fingerprint(&self) -> &str {
        &self.payload.arrow_fingerprint
    }

    pub fn schema_version(&self) -> &str {
        &self.payload.schema_version
    }
}

/// Immutable shared binding. `lease` keeps the aggregate bytes charged until
/// the LAST holder (entry or in-flight snapshot) drops.
#[doc(hidden)]
#[derive(Clone)]
pub struct BindingSnapshot {
    inner: Arc<ChargedBinding>,
}

struct ChargedBinding {
    states: Box<[String]>,
    bound_bytes: usize,
    generation: StatementGeneration,
    _lease: ByteLease,
    _metadata_lease: LiveAllocationLease,
    close_state: Arc<CloseState>,
}

impl BindingSnapshot {
    /// Close-invalidation token: true once the owning statement was closed.
    /// The deferred ExecutionControl protocol consults this before/while any
    /// future execute loop runs.
    pub fn is_closed(&self) -> bool {
        self.inner.close_state.is_closed()
    }

    pub fn states(&self) -> &[String] {
        &self.inner.states
    }

    pub fn bound_bytes(&self) -> usize {
        self.inner.bound_bytes
    }

    pub fn generation(&self) -> StatementGeneration {
        self.inner.generation
    }

    /// Charged aggregate bytes this snapshot's lease currently accounts for.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn charged_bytes(&self) -> usize {
        self.inner._lease.bytes
    }
}

#[doc(hidden)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PreparedDecideCounts {
    pub statements_global: u64,
    pub bound_bytes_global: u64,
    pub live_payload_bytes_global: u64,
    pub live_binding_metadata_bytes_global: u64,
}

#[doc(hidden)]
pub struct PreparedDecideRegistry {
    limits: PreparedDecideLimits,
    accounting: Arc<ByteAccounting>,
    stmt_global: Arc<AtomicU64>,
    live_payload_bytes: Arc<AtomicU64>,
    live_binding_metadata_bytes: Arc<AtomicU64>,
    stmt_tenants: Mutex<BTreeMap<String, Arc<AtomicU64>>>,
    stmt_owners: Mutex<BTreeMap<OwnerKey, Arc<AtomicU64>>>,
    entries: Mutex<BTreeMap<(OwnerKey, [u8; 32]), Arc<PreparedStatement>>>,
}

fn now_unix_ms() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
    )
    .unwrap_or(i64::MAX)
}

impl PreparedDecideRegistry {
    pub fn new(limits: PreparedDecideLimits) -> Result<Self, DecisionSubstrateError> {
        limits.validate()?;
        let accounting = Arc::new(ByteAccounting {
            limits: limits.clone(),
            bytes_global: Arc::new(AtomicU64::new(0)),
            bytes_tenants: Mutex::new(BTreeMap::new()),
            bytes_owners: Mutex::new(BTreeMap::new()),
        });
        Ok(Self {
            limits,
            accounting,
            stmt_global: Arc::new(AtomicU64::new(0)),
            live_payload_bytes: Arc::new(AtomicU64::new(0)),
            live_binding_metadata_bytes: Arc::new(AtomicU64::new(0)),
            stmt_tenants: Mutex::new(BTreeMap::new()),
            stmt_owners: Mutex::new(BTreeMap::new()),
            entries: Mutex::new(BTreeMap::new()),
        })
    }

    fn stmt_tenant_cell(&self, tenant: &str) -> Result<Arc<AtomicU64>, DecisionSubstrateError> {
        let mut map = self.stmt_tenants.lock();
        if let Some(existing) = map.get(tenant) {
            return Ok(Arc::clone(existing));
        }
        ByteAccounting::prune_if_full(
            &mut map,
            identity_cell_cap(self.limits.max_statements_global),
            "tenant",
        )?;
        Ok(Arc::clone(
            map.entry(tenant.to_owned())
                .or_insert_with(|| Arc::new(AtomicU64::new(0))),
        ))
    }

    fn stmt_owner_cell(&self, owner: &OwnerKey) -> Result<Arc<AtomicU64>, DecisionSubstrateError> {
        let mut map = self.stmt_owners.lock();
        if let Some(existing) = map.get(owner) {
            return Ok(Arc::clone(existing));
        }
        ByteAccounting::prune_if_full(
            &mut map,
            identity_cell_cap(self.limits.max_statements_global),
            "owner",
        )?;
        Ok(Arc::clone(
            map.entry(owner.clone())
                .or_insert_with(|| Arc::new(AtomicU64::new(0))),
        ))
    }

    /// Insert under `owner`, enforcing per-owner/tenant/global statement caps.
    pub(crate) fn insert(
        &self,
        owner: OwnerKey,
        resolved: ResolvedDecisionPins,
    ) -> Result<([u8; 32], StatementGeneration), DecisionSubstrateError> {
        let mut entries = self.entries.lock();
        let (stmt_tenant, stmt_owner) = (
            self.stmt_tenant_cell(owner.tenant())?,
            self.stmt_owner_cell(&owner)?,
        );
        let global = self.stmt_global.load(Ordering::Relaxed);
        if self.limits.max_statements_global > 0
            && global >= self.limits.max_statements_global as u64
        {
            return Err(DecisionSubstrateError::ResourceExhausted(
                "global prepared-statement cap".to_owned(),
            ));
        }
        if self.limits.max_statements_per_tenant > 0
            && stmt_tenant.load(Ordering::Relaxed) >= self.limits.max_statements_per_tenant as u64
        {
            return Err(DecisionSubstrateError::ResourceExhausted(
                "tenant prepared-statement cap".to_owned(),
            ));
        }
        if self.limits.max_statements_per_owner > 0
            && stmt_owner.load(Ordering::Relaxed) >= self.limits.max_statements_per_owner as u64
        {
            return Err(DecisionSubstrateError::ResourceExhausted(
                "owner prepared-statement cap".to_owned(),
            ));
        }
        if Arc::strong_count(&resolved.pins) != 1
            || Arc::strong_count(&resolved.question_set) != 1
            || Arc::strong_count(&resolved.raw) != 1
            || Arc::strong_count(&resolved.raw.raw_json) != 1
            || Arc::strong_count(&resolved.raw.question_ids) != 1
            || Arc::strong_count(&resolved.arrow_fingerprint) != 1
        {
            return Err(DecisionSubstrateError::InvalidArgument(
                "resolver must transfer exclusive ownership of prepared payload data".to_owned(),
            ));
        }
        let retained_bytes = prepared_payload_retained_bytes(&resolved, &owner)?;
        let lease = LiveAllocationLease::reserve(&self.live_payload_bytes, retained_bytes)?;
        // Owner identity is part of the charged payload so Arc::make_mut on a
        // retained statement snapshot cannot duplicate unaccounted strings.
        let payload_owner = owner.clone();
        let pins = Arc::try_unwrap(resolved.pins).map_err(|_| {
            DecisionSubstrateError::InvalidArgument(
                "resolver retained a prepared-payload alias".to_owned(),
            )
        })?;
        let question_set = Arc::try_unwrap(resolved.question_set).map_err(|_| {
            DecisionSubstrateError::InvalidArgument(
                "resolver retained a question-set alias".to_owned(),
            )
        })?;
        let raw = Arc::try_unwrap(resolved.raw).map_err(|_| {
            DecisionSubstrateError::InvalidArgument(
                "resolver retained a raw-template alias".to_owned(),
            )
        })?;
        let raw_json = raw.raw_json.to_string();
        let question_ids = raw.question_ids.iter().cloned().collect();
        let schema_version = pins.schema_version.clone();
        let payload = Arc::new(ChargedPreparedPayload {
            owner: payload_owner,
            pins,
            question_set,
            raw_json,
            question_ids,
            arrow_fingerprint: resolved.arrow_fingerprint.to_string(),
            schema_version,
            _lease: lease,
        });
        // Random opaque handle, collision-checked under the lock.
        let mut handle = [0u8; 32];
        loop {
            let a = *uuid::Uuid::new_v4().as_bytes();
            let b = *uuid::Uuid::new_v4().as_bytes();
            handle[..16].copy_from_slice(&a);
            handle[16..].copy_from_slice(&b);
            if !entries.contains_key(&(owner.clone(), handle)) {
                break;
            }
        }
        // ONE shared close state for the entry and EVERY binding snapshot
        // (including the initial one): close() must invalidate pre-rebind
        // snapshots too.
        let close_state = Arc::new(CloseState::default());
        let metadata_lease = LiveAllocationLease::reserve(
            &self.live_binding_metadata_bytes,
            binding_metadata_bytes(0)?,
        )?;
        // Zero-byte initial binding: a lease is unnecessary at zero bytes, but
        // the snapshot shape stays uniform for rebind/execute.
        let binding = BindingSnapshot {
            inner: Arc::new(ChargedBinding {
                states: Box::new([]),
                bound_bytes: 0,
                generation: StatementGeneration(1),
                _lease: ByteLease {
                    bytes: 0,
                    global: CountedCell::new(Arc::clone(&self.accounting.bytes_global)),
                    tenant: CountedCell::new(self.accounting.tenant_bytes_cell(owner.tenant())?),
                    owner: CountedCell::new(self.accounting.owner_bytes_cell(&owner)?),
                },
                _metadata_lease: metadata_lease,
                close_state: Arc::clone(&close_state),
            }),
        };
        let created_at_unix_ms = now_unix_ms();
        let generation = StatementGeneration(1);
        entries.insert(
            (owner.clone(), handle),
            Arc::new(PreparedStatement {
                handle,
                payload,
                binding,
                close_state,
                created_at_unix_ms,
                generation,
            }),
        );
        stmt_tenant.fetch_add(1, Ordering::Relaxed);
        stmt_owner.fetch_add(1, Ordering::Relaxed);
        self.stmt_global.fetch_add(1, Ordering::Relaxed);
        Ok((handle, generation))
    }

    /// Owner-checked lookup of the immutable shared statement snapshot.
    pub fn snapshot(
        &self,
        owner: &OwnerKey,
        handle: &[u8; 32],
    ) -> Result<Arc<PreparedStatement>, DecisionSubstrateError> {
        let entries = self.entries.lock();
        entries
            .get(&(owner.clone(), {
                let mut h = [0u8; 32];
                h.copy_from_slice(handle);
                h
            }))
            .map(Arc::clone)
            .ok_or(DecisionSubstrateError::NotFound)
    }

    /// Owner-checked close: marks the close-invalidation token, removes the
    /// entry (releasing bound bytes when the last shared snapshot drops) and
    /// decrements the statement counters.
    pub fn close(&self, owner: &OwnerKey, handle: &[u8; 32]) -> Result<(), DecisionSubstrateError> {
        let mut entries = self.entries.lock();
        let mut key_handle = [0u8; 32];
        key_handle.copy_from_slice(handle);
        let entry = entries
            .remove(&(owner.clone(), key_handle))
            .ok_or(DecisionSubstrateError::NotFound)?;
        entry.close_state.closed.store(true, Ordering::Relaxed);
        // Statement-count cells for the closed entry are pruned on their
        // next admission pass (zero + unreferenced); no await under lock.
        let tenant_cell = self.stmt_tenant_cell(owner.tenant())?;
        let owner_cell = self.stmt_owner_cell(owner)?;
        tenant_cell.fetch_sub(1, Ordering::Relaxed);
        owner_cell.fetch_sub(1, Ordering::Relaxed);
        self.stmt_global.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Owner-checked REBIND: validates per-statement rows/bytes, reserves the
    /// NEW aggregate lease (charged against tenant-wide AND global aggregates
    /// including retained old snapshots), then atomically replaces the shared
    /// binding and bumps the generation. The old lease stays charged until
    /// its last snapshot drops. Never appends; the binding is replaced.
    pub fn rebind(
        &self,
        owner: &OwnerKey,
        handle: &[u8; 32],
        states: Vec<String>,
    ) -> Result<StatementGeneration, DecisionSubstrateError> {
        let mut entries = self.entries.lock();
        let mut key_handle = [0u8; 32];
        key_handle.copy_from_slice(handle);
        let entry = entries
            .get(&(owner.clone(), key_handle))
            .ok_or(DecisionSubstrateError::NotFound)?
            .clone();
        if entry.close_state.is_closed() {
            return Err(DecisionSubstrateError::NotFound);
        }
        if states.len() > self.limits.max_bound_rows {
            return Err(DecisionSubstrateError::ResourceExhausted(format!(
                "bound rows {} exceed cap {}",
                states.len(),
                self.limits.max_bound_rows
            )));
        }
        // Preflight caller-owned strings before making any copy. This check is
        // intentionally before normalization so oversized input cannot force
        // an unbounded second allocation while the registry lock is held.
        let bytes: usize = states
            .iter()
            .map(String::len)
            .try_fold(0usize, usize::checked_add)
            .ok_or_else(|| {
                DecisionSubstrateError::ResourceExhausted("bound state bytes overflow".to_owned())
            })?;
        if bytes > self.limits.max_state_bytes_per_statement {
            return Err(DecisionSubstrateError::ResourceExhausted(format!(
                "bound state bytes {bytes} exceed per-statement cap {}",
                self.limits.max_state_bytes_per_statement
            )));
        }
        let metadata_lease = LiveAllocationLease::reserve(
            &self.live_binding_metadata_bytes,
            binding_metadata_bytes(states.len())?,
        )?;
        // Reserve the NEW lease first (charged against the live counters,
        // including retained old snapshots — conservative by design).
        let lease = ByteLease::reserve(&self.accounting, owner, bytes)?;
        // Normalize into EXACT-capacity strings after admission: a caller
        // handoff with huge spare capacity must not be charged at its inflated
        // capacity (same spelling => same content charge).
        let states: Vec<String> = states
            .into_iter()
            .map(|state| String::from(state.as_str()))
            .collect();
        let generation = StatementGeneration(entry.generation.0 + 1);
        let binding = BindingSnapshot {
            inner: Arc::new(ChargedBinding {
                states: states.into_boxed_slice(),
                bound_bytes: bytes,
                generation,
                _lease: lease,
                _metadata_lease: metadata_lease,
                close_state: Arc::clone(&entry.close_state),
            }),
        };
        let prepared = Arc::make_mut(
            entries
                .get_mut(&(owner.clone(), key_handle))
                .expect("entry present"),
        );
        prepared.binding = binding;
        prepared.generation = generation;
        Ok(generation)
    }

    /// Owner-checked current binding snapshot (zero byte copies).
    pub fn binding(
        &self,
        owner: &OwnerKey,
        handle: &[u8; 32],
    ) -> Result<BindingSnapshot, DecisionSubstrateError> {
        let entries = self.entries.lock();
        let mut key_handle = [0u8; 32];
        key_handle.copy_from_slice(handle);
        let entry = entries
            .get(&(owner.clone(), key_handle))
            .ok_or(DecisionSubstrateError::NotFound)?;
        Ok(BindingSnapshot::clone(&entry.binding))
    }

    /// Telemetry: (global statements, global bound bytes).
    pub fn open_counts(&self) -> PreparedDecideCounts {
        PreparedDecideCounts {
            statements_global: self.stmt_global.load(Ordering::Relaxed),
            bound_bytes_global: self.accounting.bytes_global.load(Ordering::Relaxed),
            live_payload_bytes_global: self.live_payload_bytes.load(Ordering::Relaxed),
            live_binding_metadata_bytes_global: self
                .live_binding_metadata_bytes
                .load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// injected resolver (S1: CREATE-time immutable pin resolution)
// ---------------------------------------------------------------------------

/// CREATE-time immutable pin resolution seam. The REAL implementation (S3,
/// embedding binary) calls the Model decisionMetadata RPC with
/// `bearer.expose_secret()` as the original end-user bearer, validates the
/// exact (configured alias, expected full OID) tuple against the trusted
/// staging pin, and parses the decide document. Absence of the metadata
/// implementation fails closed; a fake implementation is test-only.
#[doc(hidden)]
#[async_trait::async_trait]
pub trait DecideResolve: Send + Sync {
    async fn resolve_pins(
        &self,
        owner: &OwnerKey,
        bearer: &OriginalBearer,
        expected_artifact_commit: &str,
        decide_document: &str,
    ) -> Result<ResolvedDecisionPins, DecisionSubstrateError>;
}

// ---------------------------------------------------------------------------
// operator
// ---------------------------------------------------------------------------

fn check_expiry(ctx: &VerifiedFlightContext) -> Result<(), DecisionSubstrateError> {
    // Native v1 boundary: the expiry must be PRESENT and in the future. The
    // G1 legacy Option is retained upstream, but this boundary fails closed
    // on absence rather than treating it as unlimited.
    let exp = ctx.exp.ok_or_else(|| {
        DecisionSubstrateError::Unauthenticated(
            "verified caller context carries no expiry".to_owned(),
        )
    })?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(i64::MAX);
    if exp <= now {
        return Err(DecisionSubstrateError::Unauthenticated(
            "verified caller context expired".to_owned(),
        ));
    }
    Ok(())
}

fn canonical_commit(value: &str) -> Result<(), DecisionSubstrateError> {
    if !is_full_oid(value) {
        return Err(DecisionSubstrateError::InvalidArgument(
            "expected_artifact_commit must be a full 40-hex commit OID".to_owned(),
        ));
    }
    Ok(())
}

/// Exact CREATE-time resolution consistency. Structural pins plus
/// cross-field agreement: the resolved pins must carry EXACTLY the prepared
/// expected artifact commit; the metadata Arrow identity AND the resolved
/// fingerprint must equal an independently recomputed fingerprint; the raw
/// ordered question template must parse (full IR, state excluded) and agree
/// with the resolved QuestionSet, ids, types, and criteria presence.
fn ensure_resolution_consistent(
    resolved: &ResolvedDecisionPins,
    expected_artifact_commit: &str,
    caller_schema_version: &str,
    caller_calibration_version: Option<&str>,
    caller_raw_questions: &str,
    caller_set: &hyprstream_decision::spec::QuestionSet,
) -> Result<(), DecisionSubstrateError> {
    resolved.pins.validate()?;
    if resolved.pins.schema_version != caller_schema_version {
        return Err(DecisionSubstrateError::InvalidArgument(
            "resolved pins disagree with the prepared schema version".to_owned(),
        ));
    }
    if resolved.pins.calibration_version.as_deref() != caller_calibration_version {
        return Err(DecisionSubstrateError::InvalidArgument(
            "resolved pins disagree with the prepared calibration version".to_owned(),
        ));
    }
    if resolved.pins.expected_artifact_commit != expected_artifact_commit {
        return Err(DecisionSubstrateError::InvalidArgument(
            "resolved pins disagree with the prepared expected artifact commit".to_owned(),
        ));
    }
    if resolved.raw.raw_json.is_empty() || resolved.raw.question_ids.is_empty() {
        return Err(DecisionSubstrateError::InvalidArgument(
            "resolved raw question template must be non-empty".to_owned(),
        ));
    }
    if resolved.raw.raw_json.as_ref() != caller_raw_questions {
        return Err(DecisionSubstrateError::InvalidArgument(
            "resolved raw question template does not match the caller's original questions JSON"
                .to_owned(),
        ));
    }
    if resolved.arrow_fingerprint.is_empty() {
        return Err(DecisionSubstrateError::InvalidArgument(
            "resolved Arrow fingerprint must be non-empty".to_owned(),
        ));
    }
    let schema = DecisionSchema::from_question_set(&resolved.question_set).map_err(|error| {
        DecisionSubstrateError::InvalidArgument(format!(
            "question set is not usable on the columnar surface: {error}"
        ))
    })?;
    let recomputed = schema.fingerprint();
    if resolved.arrow_fingerprint.as_ref() != recomputed {
        return Err(DecisionSubstrateError::InvalidArgument(
            "resolved Arrow fingerprint does not match the parsed question set".to_owned(),
        ));
    }
    if resolved.pins.arrow_schema_identity != recomputed {
        return Err(DecisionSubstrateError::InvalidArgument(
            "metadata Arrow identity does not match the parsed question set".to_owned(),
        ));
    }
    // Raw ordered ids must EXACTLY match the ordered QuestionSet ids.
    // The resolved set (and the raw template) must reproduce the CALLER's
    // prepared document questions exactly — ordered, state excluded. A
    // resolver returning any other internally coherent template is rejected.
    if resolved.question_set.questions != caller_set.questions {
        return Err(DecisionSubstrateError::InvalidArgument(
            "resolved question set does not match the caller's prepare document".to_owned(),
        ));
    }
    let caller_ids: Vec<&str> = caller_set.questions.iter().map(|q| q.id.as_str()).collect();
    if resolved.raw.question_ids.as_ref() != caller_ids.as_slice() {
        return Err(DecisionSubstrateError::InvalidArgument(
            "raw question ids do not exactly match the caller's ordered questions".to_owned(),
        ));
    }
    // Strict raw-template check: parse the raw questions (wrapped) into the
    // full IR and compare with the resolved set, state excluded; then verify
    // the raw JSON structurally (keys, declared types, criteria presence).
    let wrapped = format!(r#"{{"questions":{}}}"#, resolved.raw.raw_json);
    let parsed = hyprstream_decision::author::parse_json(&wrapped).map_err(|error| {
        DecisionSubstrateError::InvalidArgument(format!(
            "raw question template does not parse: {error}"
        ))
    })?;
    if parsed.questions != resolved.question_set.questions {
        return Err(DecisionSubstrateError::InvalidArgument(
            "raw question template does not reproduce the resolved question set".to_owned(),
        ));
    }
    let raw_value: serde_json::Value = serde_json::from_str(resolved.raw.raw_json.as_ref())
        .map_err(|error| {
            DecisionSubstrateError::InvalidArgument(format!(
                "raw question template is not valid JSON: {error}"
            ))
        })?;
    let raw_object = raw_value.as_object().ok_or_else(|| {
        DecisionSubstrateError::InvalidArgument(
            "raw question template must be a JSON object".to_owned(),
        )
    })?;
    if raw_object.len() != resolved.question_set.questions.len() {
        return Err(DecisionSubstrateError::InvalidArgument(
            "raw question template key count differs from the resolved question set".to_owned(),
        ));
    }
    for question in &resolved.question_set.questions {
        let entry = raw_object.get(&question.id).ok_or_else(|| {
            DecisionSubstrateError::InvalidArgument(format!(
                "raw template is missing question {}",
                question.id
            ))
        })?;
        let declared_type = entry.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
            DecisionSubstrateError::InvalidArgument(format!(
                "raw template question {} is missing its type",
                question.id
            ))
        })?;
        let expected_type = match question.kind {
            hyprstream_decision::spec::QuestionKind::Noul => "noul",
            hyprstream_decision::spec::QuestionKind::Choice => "choice",
            hyprstream_decision::spec::QuestionKind::Score => "score",
            hyprstream_decision::spec::QuestionKind::Span => "span",
            hyprstream_decision::spec::QuestionKind::Derived => "derived",
        };
        if declared_type != expected_type {
            return Err(DecisionSubstrateError::InvalidArgument(format!(
                "raw template question {} declares {declared_type}, expected {expected_type}",
                question.id
            )));
        }
        if question.kind != hyprstream_decision::spec::QuestionKind::Noul
            && entry.get("criteria").is_none()
        {
            return Err(DecisionSubstrateError::InvalidArgument(format!(
                "raw template question {} is missing criteria",
                question.id
            )));
        }
    }
    Ok(())
}

/// Bounded prepared-decide operator: G1 context in, owned registry state out.
/// The execute loop is deferred (see module docs); close invalidation tokens
/// are already carried by every snapshot.
#[doc(hidden)]
pub struct DecideOperator {
    resolve: Arc<dyn DecideResolve>,
    registry: PreparedDecideRegistry,
    prepare_permits: Arc<tokio::sync::Semaphore>,
}

impl DecideOperator {
    pub fn new(
        resolve: Arc<dyn DecideResolve>,
        limits: PreparedDecideLimits,
    ) -> Result<Self, DecisionSubstrateError> {
        limits.validate()?;
        let prepare_permits = Arc::new(tokio::sync::Semaphore::new(limits.max_concurrent_prepares));
        Ok(Self {
            resolve,
            registry: PreparedDecideRegistry::new(limits)?,
            prepare_permits,
        })
    }

    pub fn registry(&self) -> &PreparedDecideRegistry {
        &self.registry
    }

    /// CREATE: owner derivation + expiry on entry, mandatory canonical
    /// expected commit, bounded document, bounded try-acquire prepare
    /// permit, resolver await, expiry + exact-consistency re-check, capped
    /// insert. Absent/non-canonical commit and oversized documents are
    /// rejected BEFORE any await.
    pub async fn create_prepared(
        &self,
        caller: &VerifiedFlightContext,
        bearer: &OriginalBearer,
        expected_artifact_commit: &str,
        decide_document: &str,
    ) -> Result<([u8; 32], StatementGeneration), DecisionSubstrateError> {
        let owner = OwnerKey::from_verified(caller)?;
        check_expiry(caller)?;
        canonical_commit(expected_artifact_commit)?;
        if decide_document.len() > MAX_PREPARE_DOCUMENT_BYTES {
            return Err(DecisionSubstrateError::InvalidArgument(format!(
                "decide document exceeds the bounded prepare size {}",
                MAX_PREPARE_DOCUMENT_BYTES
            )));
        }
        let permit = self
            .prepare_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| {
                DecisionSubstrateError::ResourceExhausted(
                    "bounded prepare queue exhausted".to_owned(),
                )
            })?;
        // Independent parse of the ACTUAL caller prepare document. This is
        // the anchor every resolved/raw template is compared against — a
        // self-consistent template that was not derived from this document
        // is rejected. `schema_version` is extracted here (the IR parser
        // accepts only `state`/`questions` at the top level).
        let document: CallerPrepareDocument<'_> =
            serde_json::from_str(decide_document).map_err(|error| {
                DecisionSubstrateError::InvalidArgument(format!(
                    "prepare document is not valid JSON: {error}"
                ))
            })?;
        if document.schema_version.is_empty() {
            return Err(DecisionSubstrateError::InvalidArgument(
                "prepare document requires a non-empty schema_version".to_owned(),
            ));
        }
        // These fields are accepted by the outer contract; model pin
        // validation remains the resolver's responsibility.
        let _model = document.model;
        let _state = document.state;
        let caller_raw_questions = document.questions.get();
        let wrapped = format!(r#"{{"questions":{caller_raw_questions}}}"#);
        let caller_set = hyprstream_decision::author::parse_json(&wrapped).map_err(|error| {
            DecisionSubstrateError::InvalidArgument(format!(
                "prepare document does not parse as a question-spec document: {error}"
            ))
        })?;
        let caller_schema_version = document.schema_version;
        let caller_calibration_version: Option<String> = document
            .calib_version
            .map(|raw| serde_json::from_str(raw.get()))
            .transpose()
            .map_err(|error| {
                DecisionSubstrateError::InvalidArgument(format!(
                    "prepare document calib_version must be a string or null: {error}"
                ))
            })?
            .flatten();
        let resolved = self
            .resolve
            .resolve_pins(&owner, bearer, expected_artifact_commit, decide_document)
            .await?;
        drop(permit);
        // Expiry re-check AFTER the await: a context that expired mid-flight
        // must not publish registry state.
        check_expiry(caller)?;
        // Exact consistency before insert (structural AND cross-field):
        // resolved pins/set/raw must all agree with the caller's document.
        ensure_resolution_consistent(
            &resolved,
            expected_artifact_commit,
            caller_schema_version,
            caller_calibration_version.as_deref(),
            caller_raw_questions,
            &caller_set,
        )?;
        self.registry.insert(owner, resolved)
    }

    /// CLOSE: owner-checked removal; releases aggregate bytes when the last
    /// shared snapshot drops; close-invalidation token set first.
    pub fn close_prepared(
        &self,
        caller: &VerifiedFlightContext,
        handle: &[u8; 32],
    ) -> Result<(), DecisionSubstrateError> {
        let owner = OwnerKey::from_verified(caller)?;
        check_expiry(caller)?;
        self.registry.close(&owner, handle)
    }

    /// REBIND: owner-checked replace of the active binding (never append)
    /// with an atomic generation bump; aggregate caps include retained old
    /// snapshots by design.
    pub fn rebind(
        &self,
        caller: &VerifiedFlightContext,
        handle: &[u8; 32],
        states: Vec<String>,
    ) -> Result<StatementGeneration, DecisionSubstrateError> {
        let owner = OwnerKey::from_verified(caller)?;
        check_expiry(caller)?;
        self.registry.rebind(&owner, handle, states)
    }

    /// SNAPSHOT: owner-checked immutable statement view (Arc-shared).
    pub fn snapshot(
        &self,
        caller: &VerifiedFlightContext,
        handle: &[u8; 32],
    ) -> Result<Arc<PreparedStatement>, DecisionSubstrateError> {
        let owner = OwnerKey::from_verified(caller)?;
        check_expiry(caller)?;
        self.registry.snapshot(&owner, handle)
    }
}

// ---------------------------------------------------------------------------
// unit fixtures + causal tests (substrate-internal; no network, no models)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_decision::entry::Entry;
    use hyprstream_decision::spec::{QuestionBody, QuestionKind, QuestionSpec};
    use std::time::Duration;

    const COMMIT: &str = "7b43c2d799e7b5e9fef96b6137840088d75af0d6";
    const TENANT: &str = "tenant.test";

    fn registry_with(limits: PreparedDecideLimits) -> PreparedDecideRegistry {
        PreparedDecideRegistry::new(limits).expect("valid test limits")
    }

    fn ctx(tenant: Option<&str>, subject: &str, exp: Option<i64>) -> VerifiedFlightContext {
        VerifiedFlightContext {
            tenant: tenant.map(ToOwned::to_owned),
            subject: subject.to_owned(),
            exp,
        }
    }

    fn now() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(i64::MAX)
    }

    fn owner(tenant: &str, subject: &str) -> OwnerKey {
        OwnerKey::from_verified(&ctx(Some(tenant), subject, Some(now() + 600)))
            .expect("valid verified context")
    }

    fn question_set() -> QuestionSet {
        QuestionSet {
            state: None,
            questions: vec![QuestionSpec {
                id: "q1".to_owned(),
                kind: QuestionKind::Noul,
                instructions: Some(Entry::Str("The service is down.".to_owned())),
                body: QuestionBody::Noul { criteria: None },
            }],
        }
    }

    fn pins(commit: &str) -> DecisionPins {
        DecisionPins {
            authorization_model_ref: "clef-flash-fp8:main".to_owned(),
            expected_artifact_commit: commit.to_owned(),
            schema_version: "jev-1".to_owned(),
            model_id: "clef-flash@7b43c2d7".to_owned(),
            encoding_version: "clef-native-v1".to_owned(),
            quantization: "fp8-e4m3-dynamic".to_owned(),
            admitted_compute_profile: "cpu-f32".to_owned(),
            head_weights_sha256: "b".repeat(64),
            head_config_sha256: "c".repeat(64),
            tokenizer_sha256: "d".repeat(64),
            backbone_config_sha256: "e".repeat(64),
            arrow_schema_identity: "f".repeat(64), // overwritten by the fake resolver
            calibration_version: None,
        }
    }

    fn resolved(commit: &str) -> ResolvedDecisionPins {
        let question_set = Arc::new(question_set());
        let fingerprint = Arc::<str>::from(
            DecisionSchema::from_question_set(&question_set)
                .expect("columnar surface")
                .fingerprint()
                .as_str(),
        );
        let mut p = pins(commit);
        p.arrow_schema_identity = fingerprint.to_string();
        ResolvedDecisionPins {
            pins: Arc::new(p),
            question_set,
            raw: Arc::new(RawQuestionsTemplate {
                raw_json: Arc::from(
                    r#"{"q1":{"type":"noul","instructions":"The service is down."}}"#,
                ),
                question_ids: Arc::from([String::from("q1")]),
            }),
            arrow_fingerprint: fingerprint,
        }
    }

    // -- owner key ----------------------------------------------------------

    #[test]
    fn owner_key_rejects_missing_tenant() {
        assert!(matches!(
            OwnerKey::from_verified(&ctx(None, "alice", Some(i64::MAX))),
            Err(DecisionSubstrateError::Unauthenticated(_))
        ));
    }

    #[test]
    fn owner_key_rejects_wildcard_empty_and_blank() {
        for tenant in ["*", "", " spaced ", "a\nb"] {
            assert!(
                OwnerKey::from_verified(&ctx(Some(tenant), "alice", Some(i64::MAX))).is_err(),
                "tenant {tenant:?} must fail closed"
            );
        }
    }

    // -- original bearer ----------------------------------------------------

    #[test]
    fn original_bearer_redacts_and_validates() {
        let debug = format!(
            "{:?}",
            OriginalBearer::from_authorization_header(Some("Bearer sekrit")).unwrap()
        );
        assert!(!debug.contains("sekrit"), "debug must redact the secret");
        assert!(OriginalBearer::from_authorization_header(None).is_err());
        assert!(OriginalBearer::from_authorization_header(Some("Basic abc")).is_err());
        assert!(OriginalBearer::from_authorization_header(Some("Bearer ")).is_err());
    }

    // -- pins ---------------------------------------------------------------

    #[test]
    fn pins_validate_shapes() {
        assert!(pins(COMMIT).validate().is_ok());
        let mut bad = pins(COMMIT);
        bad.expected_artifact_commit = "7b43".to_owned();
        assert!(bad.validate().is_err(), "partial OID must fail");
        let mut bad = pins(COMMIT);
        bad.head_weights_sha256 = "zz".repeat(64);
        assert!(bad.validate().is_err(), "non-hex digest must fail");
        let mut ok = pins(COMMIT);
        ok.calibration_version = Some("cal-v2".to_owned());
        assert!(ok.validate().is_ok(), "calibration Some must be preserved");
    }

    // -- registry -----------------------------------------------------------

    fn registry() -> PreparedDecideRegistry {
        registry_with(PreparedDecideLimits::default())
    }

    fn insert_for(registry: &PreparedDecideRegistry, owner: &OwnerKey) -> [u8; 32] {
        let r = resolved(COMMIT);
        registry.insert(owner.clone(), r).expect("insert").0
    }

    #[test]
    fn handles_are_random_opaque_and_owner_isolated() {
        let registry = registry();
        let alice = owner(TENANT, "alice");
        let h1 = insert_for(&registry, &alice);
        let h2 = insert_for(&registry, &alice);
        assert_ne!(
            h1, h2,
            "handles must be random per CREATE, not document-derived"
        );

        // Same tenant, DIFFERENT principal: isolation is by owner, not tenant.
        let bob = owner(TENANT, "bob");
        assert!(matches!(
            registry.snapshot(&bob, &h1),
            Err(DecisionSubstrateError::NotFound)
        ));
        assert!(matches!(
            registry.close(&bob, &h1),
            Err(DecisionSubstrateError::NotFound)
        ));
        assert!(registry.snapshot(&alice, &h1).is_ok());
    }

    #[test]
    fn statement_caps_are_owner_tenant_and_global() {
        let registry = registry_with(PreparedDecideLimits {
            max_statements_per_owner: 1,
            ..PreparedDecideLimits::default()
        });
        let alice = owner(TENANT, "alice");
        let _ = insert_for(&registry, &alice);
        assert!(matches!(
            registry.insert(alice, resolved(COMMIT)),
            Err(DecisionSubstrateError::ResourceExhausted(m)) if m.contains("owner")
        ));

        let registry = registry_with(PreparedDecideLimits {
            max_statements_per_tenant: 1,
            ..PreparedDecideLimits::default()
        });
        let _ = insert_for(&registry, &owner(TENANT, "alice"));
        let bob = owner(TENANT, "bob");
        assert!(matches!(
            registry.insert(bob, resolved(COMMIT)),
            Err(DecisionSubstrateError::ResourceExhausted(m)) if m.contains("tenant")
        ));

        let registry = registry_with(PreparedDecideLimits {
            max_statements_global: 1,
            ..PreparedDecideLimits::default()
        });
        let _ = insert_for(&registry, &owner("t1", "alice"));
        assert!(matches!(
            registry.insert(owner("t2", "alice"), resolved(COMMIT)),
            Err(DecisionSubstrateError::ResourceExhausted(m)) if m.contains("global")
        ));
    }

    #[tokio::test]
    async fn rebind_replaces_bumps_generation_and_enforces_aggregate_caps() {
        let registry = registry_with(PreparedDecideLimits {
            max_bound_rows: 8,
            max_state_bytes_per_statement: 64,
            max_bound_bytes_per_tenant: 25,
            ..PreparedDecideLimits::default()
        });
        let alice = owner(TENANT, "alice");
        let handle = insert_for(&registry, &alice);

        let gen1 = registry
            .rebind(&alice, &handle, vec!["x".repeat(10)])
            .unwrap();
        assert_eq!(
            gen1,
            StatementGeneration(2),
            "create is generation 1; first rebind is 2"
        );
        let snap = registry.binding(&alice, &handle).unwrap();
        assert_eq!(snap.states().len(), 1);
        assert_eq!(snap.bound_bytes(), 10);

        // Hold the OLD binding across a rebind: the new lease is charged on
        // top of the retained one (conservative aggregate accounting).
        let held = registry.binding(&alice, &handle).unwrap();
        let gen2 = registry
            .rebind(&alice, &handle, vec!["y".repeat(10)])
            .unwrap();
        assert_eq!(gen2, StatementGeneration(3));
        assert_eq!(registry.open_counts().bound_bytes_global, 20);
        assert_eq!(held.bound_bytes(), 10, "old snapshot is immutable");

        // While the old snapshot is held, a rebind that would push the
        // tenant aggregate past the cap fails WITHOUT disturbing the
        // published binding.
        let result = registry.rebind(&alice, &handle, vec!["z".repeat(10)]);
        assert!(matches!(
            result,
            Err(DecisionSubstrateError::ResourceExhausted(_))
        ));
        assert_eq!(
            registry.binding(&alice, &handle).unwrap().generation(),
            gen2
        );
        assert_eq!(registry.open_counts().bound_bytes_global, 20);

        // Release the old snapshot: its bytes leave the aggregate, and the
        // same rebind now fits.
        drop(held);
        drop(snap); // BOTH holders must drop before lease1's bytes release.
        assert_eq!(registry.open_counts().bound_bytes_global, 10);
        let gen3 = registry
            .rebind(&alice, &handle, vec!["z".repeat(10)])
            .unwrap();
        assert_eq!(gen3, StatementGeneration(4));
        // B2 had no test-side holder, so the entry swap released lease2 at
        // rebind3: charged = lease3 only (10).
        assert_eq!(registry.open_counts().bound_bytes_global, 10);
    }

    #[tokio::test]
    async fn close_releases_bytes_marks_token_and_orphans_the_snapshot() {
        let registry = registry();
        let alice = owner(TENANT, "alice");
        let handle = insert_for(&registry, &alice);
        registry
            .rebind(&alice, &handle, vec!["state".to_owned()])
            .unwrap();
        let held = registry.binding(&alice, &handle).unwrap();
        assert!(!held.is_closed());

        registry.close(&alice, &handle).unwrap();
        assert!(matches!(
            registry.snapshot(&alice, &handle),
            Err(DecisionSubstrateError::NotFound)
        ));
        assert_eq!(registry.open_counts().statements_global, 0);
        // The held snapshot survives (Arc) with its close token SET for the
        // deferred ExecutionControl protocol, and its bytes stay charged
        // until it drops.
        assert!(held.is_closed());
        assert_eq!(held.charged_bytes(), 5);
        assert_eq!(registry.open_counts().bound_bytes_global, 5);
        drop(held);
        assert_eq!(registry.open_counts().bound_bytes_global, 0);
    }

    #[test]
    fn live_payload_and_binding_metadata_follow_retained_snapshot_lifetimes() {
        let registry = registry();
        let alice = owner(TENANT, "alice");
        let handle = insert_for(&registry, &alice);
        let held_statement = registry.snapshot(&alice, &handle).unwrap();
        let held_binding = registry.binding(&alice, &handle).unwrap();
        let admitted = registry.open_counts();
        assert!(admitted.live_payload_bytes_global > 0);
        assert!(admitted.live_binding_metadata_bytes_global > 0);

        registry.close(&alice, &handle).unwrap();
        assert_eq!(registry.open_counts().statements_global, 0);
        assert_eq!(
            registry.open_counts().live_payload_bytes_global,
            admitted.live_payload_bytes_global,
            "prepared payload stays charged while a statement snapshot retains it"
        );
        assert_eq!(
            registry.open_counts().live_binding_metadata_bytes_global,
            admitted.live_binding_metadata_bytes_global,
            "binding storage stays charged while a binding snapshot retains it"
        );

        drop(held_statement);
        assert_eq!(registry.open_counts().live_payload_bytes_global, 0);
        assert!(registry.open_counts().live_binding_metadata_bytes_global > 0);
        drop(held_binding);
        assert_eq!(registry.open_counts().live_binding_metadata_bytes_global, 0);
    }

    #[tokio::test]
    async fn rebind_rows_and_per_statement_bytes_are_capped() {
        let registry = registry_with(PreparedDecideLimits {
            max_bound_rows: 2,
            max_state_bytes_per_statement: 8,
            ..PreparedDecideLimits::default()
        });
        let alice = owner(TENANT, "alice");
        let handle = insert_for(&registry, &alice);
        assert!(matches!(
            registry.rebind(
                &alice,
                &handle,
                vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
            ),
            Err(DecisionSubstrateError::ResourceExhausted(_))
        ));
        assert!(matches!(
            registry.rebind(&alice, &handle, vec!["x".repeat(9)]),
            Err(DecisionSubstrateError::ResourceExhausted(_))
        ));
        let binding = registry.binding(&alice, &handle).unwrap();
        assert!(binding.states().is_empty());
    }

    // -- operator + injected resolver ---------------------------------------

    #[derive(Clone, Copy)]
    enum FakeBehavior {
        Allow,
        FingerprintMismatch,
    }

    struct FakeResolve {
        calls: AtomicU64,
        delay: Option<Duration>,
        behavior: FakeBehavior,
    }

    impl FakeResolve {
        fn allow() -> std::sync::Arc<Self> {
            std::sync::Arc::new(Self {
                calls: AtomicU64::new(0),
                delay: None,
                behavior: FakeBehavior::Allow,
            })
        }
        fn calls(&self) -> u64 {
            self.calls.load(Ordering::Relaxed)
        }
    }

    #[async_trait::async_trait]
    impl DecideResolve for FakeResolve {
        async fn resolve_pins(
            &self,
            _owner: &OwnerKey,
            _bearer: &OriginalBearer,
            expected_artifact_commit: &str,
            _decide_document: &str,
        ) -> Result<ResolvedDecisionPins, DecisionSubstrateError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }
            match self.behavior {
                FakeBehavior::Allow => {
                    if expected_artifact_commit != COMMIT {
                        return Err(DecisionSubstrateError::PinMismatch(format!(
                            "expected {expected_artifact_commit} does not match the configured staging pin"
                        )));
                    }
                    Ok(resolved(COMMIT))
                }
                FakeBehavior::FingerprintMismatch => {
                    let mut r = resolved(COMMIT);
                    r.arrow_fingerprint = Arc::from("mismatched-fingerprint");
                    Ok(r)
                }
            }
        }
    }

    fn operator_with(
        fake: std::sync::Arc<FakeResolve>,
        limits: PreparedDecideLimits,
    ) -> DecideOperator {
        DecideOperator::new(fake, limits).expect("valid test limits")
    }

    #[tokio::test]
    async fn create_prepared_allows_after_real_resolution_path() {
        let fake = FakeResolve::allow();
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();

        let document = r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}},"calib_version":null}"#;
        let (handle, generation) = operator
            .create_prepared(&caller, &bearer, COMMIT, document)
            .await
            .expect("allow path must publish the prepared statement");
        assert_eq!(generation, StatementGeneration(1));
        assert_eq!(
            counter.calls(),
            1,
            "resolver must be consulted exactly once"
        );
        let prepared = operator
            .snapshot(&caller, &handle)
            .expect("snapshot after allow");
        assert_eq!(prepared.pins().expected_artifact_commit, COMMIT);
        assert_eq!(prepared.owner().subject, "alice");
        assert!(!prepared.binding.is_closed());
    }

    #[tokio::test]
    async fn create_prepared_rejects_calibration_pin_mismatch() {
        let fake = FakeResolve::allow();
        let counter = Arc::clone(&fake);
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        let document = r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}},"calib_version":"cal-v2"}"#;

        assert!(matches!(
            operator
                .create_prepared(&caller, &bearer, COMMIT, document)
                .await,
            Err(DecisionSubstrateError::InvalidArgument(message))
                if message.contains("calibration version")
        ));
        assert_eq!(counter.calls(), 1);
        assert_eq!(operator.registry().open_counts().statements_global, 0);
    }

    #[tokio::test]
    async fn create_prepared_rejects_duplicate_document_keys_before_resolution() {
        let fake = FakeResolve::allow();
        let counter = Arc::clone(&fake);
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        let documents = [
            r#"{"schema_version":"jev-1","schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#,
            r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"discarded","instructions":"The service is down."}}}"#,
            r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul"},"q1":{"type":"noul"}}}"#,
        ];
        for document in documents {
            assert!(matches!(
                operator
                    .create_prepared(&caller, &bearer, COMMIT, document)
                    .await,
                Err(DecisionSubstrateError::InvalidArgument(_))
            ));
        }
        assert_eq!(
            counter.calls(),
            0,
            "invalid authored JSON must fail before resolution"
        );
        assert_eq!(operator.registry().open_counts().statements_global, 0);
    }

    #[tokio::test]
    async fn create_prepared_rejects_bad_commit_before_await() {
        let fake = FakeResolve::allow();
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        assert!(matches!(
            operator
                .create_prepared(&caller, &bearer, "7b43", "{}")
                .await,
            Err(DecisionSubstrateError::InvalidArgument(_))
        ));
        assert_eq!(
            counter.calls(),
            0,
            "non-canonical commit must be rejected before any await"
        );
    }

    #[tokio::test]
    async fn create_prepared_rejects_oversized_document_before_await() {
        let fake = FakeResolve::allow();
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        let document = " ".repeat(MAX_PREPARE_DOCUMENT_BYTES + 1);
        assert!(matches!(
            operator
                .create_prepared(&caller, &bearer, COMMIT, &document)
                .await,
            Err(DecisionSubstrateError::InvalidArgument(_))
        ));
        assert_eq!(counter.calls(), 0);
    }

    #[tokio::test]
    async fn zero_prepare_concurrency_disables_create_at_construction() {
        // REQUIRED level zero = disabled/deny: construction itself fails
        // closed before any resolver call (no semaphore is ever created).
        let fake = FakeResolve::allow();
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let result = DecideOperator::new(
            fake,
            PreparedDecideLimits {
                max_concurrent_prepares: 0,
                ..PreparedDecideLimits::default()
            },
        );
        assert!(matches!(
            result,
            Err(DecisionSubstrateError::InvalidArgument(m))
                if m.contains("would disable CREATE")
        ));
        assert_eq!(counter.calls(), 0);
    }

    #[tokio::test]
    async fn bounded_prepare_queue_rejects_when_exhausted() {
        // max_concurrent_prepares = 1 with a SLOW resolver: a concurrent
        // second CREATE is rejected by the bounded try-acquire instead of
        // queueing unboundedly.
        let fake = std::sync::Arc::new(FakeResolve {
            calls: AtomicU64::new(0),
            delay: Some(Duration::from_millis(400)),
            behavior: FakeBehavior::Allow,
        });
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = Arc::new(operator_with(
            fake,
            PreparedDecideLimits {
                max_concurrent_prepares: 1,
                ..PreparedDecideLimits::default()
            },
        ));
        let caller_a = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer_a = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        let op_a = Arc::clone(&operator);
        let first = tokio::spawn(async move {
            op_a.create_prepared(&caller_a, &bearer_a, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#).await
        });

        // Give the first CREATE time to acquire the single permit and enter
        // the resolver await; the concurrent second CREATE must then be
        // rejected by the bounded queue without reaching the resolver.
        tokio::time::sleep(Duration::from_millis(150)).await;
        let caller_b = ctx(Some(TENANT), "bob", Some(now() + 600));
        let bearer_b = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        let second = tokio::time::timeout(
            Duration::from_secs(10),
            operator.create_prepared(&caller_b, &bearer_b, COMMIT, "{}"),
        )
        .await
        .expect("bounded create must not wedge");
        assert!(matches!(
            second,
            Err(DecisionSubstrateError::ResourceExhausted(m)) if m.contains("bounded prepare queue")
        ));
        assert_eq!(
            counter.calls(),
            1,
            "rejected create must not reach the resolver"
        );

        let first = first
            .await
            .expect("first task joins")
            .expect("first create allowed");
        assert_eq!(first.1, StatementGeneration(1));
    }

    #[tokio::test]
    async fn create_prepared_rejects_inconsistent_fingerprint_and_inserts_nothing() {
        let fake = std::sync::Arc::new(FakeResolve {
            calls: AtomicU64::new(0),
            delay: None,
            behavior: FakeBehavior::FingerprintMismatch,
        });
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#).await,
            Err(DecisionSubstrateError::InvalidArgument(_))
        ));
        assert_eq!(
            operator.registry().open_counts().statements_global,
            0,
            "inconsistent resolution must insert nothing"
        );
        assert_eq!(counter.calls(), 1);
    }

    #[tokio::test]
    async fn create_prepared_rechecks_expiry_after_await() {
        let fake = std::sync::Arc::new(FakeResolve {
            calls: AtomicU64::new(0),
            delay: Some(Duration::from_millis(1200)),
            behavior: FakeBehavior::Allow,
        });
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() + 1));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        let result = operator.create_prepared(&caller, &bearer, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#).await;
        assert!(
            matches!(result, Err(DecisionSubstrateError::Unauthenticated(m)) if m.contains("expired")),
            "context expiring mid-resolve must deny after the await"
        );
        assert_eq!(
            operator.registry().open_counts().statements_global,
            0,
            "expired mid-flight must insert nothing"
        );
    }

    #[tokio::test]
    async fn expired_context_denies_at_entry() {
        let fake = FakeResolve::allow();
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() - 1));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#).await,
            Err(DecisionSubstrateError::Unauthenticated(m)) if m.contains("expired")
        ));
        assert_eq!(counter.calls(), 0);
    }

    #[tokio::test]
    async fn same_tenant_different_principal_is_isolated_through_the_operator() {
        let fake = FakeResolve::allow();
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let alice = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        let (handle, _) = operator
            .create_prepared(&alice, &bearer, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#)
            .await
            .expect("allow");
        let bob = ctx(Some(TENANT), "bob", Some(now() + 600));
        assert!(
            matches!(
                operator.snapshot(&bob, &handle),
                Err(DecisionSubstrateError::NotFound)
            ),
            "same-tenant different principal must not observe another owner's statement"
        );
    }

    // -- F1: expiry required (present AND future) ----------------------------

    #[tokio::test]
    async fn missing_expiry_denies_with_zero_resolver_calls() {
        let fake = FakeResolve::allow();
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", None);
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#).await,
            Err(DecisionSubstrateError::Unauthenticated(m)) if m.contains("no expiry")
        ));
        assert_eq!(
            counter.calls(),
            0,
            "absent expiry fails closed before the resolver"
        );
    }

    // -- F2: fallible limits validation --------------------------------------

    #[test]
    fn required_zero_limits_are_disabled_not_unlimited() {
        for limits in [
            PreparedDecideLimits {
                max_statements_per_tenant: 0,
                ..PreparedDecideLimits::default()
            },
            PreparedDecideLimits {
                max_statements_global: 0,
                ..PreparedDecideLimits::default()
            },
            PreparedDecideLimits {
                max_bound_bytes_per_tenant: 0,
                ..PreparedDecideLimits::default()
            },
            PreparedDecideLimits {
                max_bound_bytes_global: 0,
                ..PreparedDecideLimits::default()
            },
            PreparedDecideLimits {
                max_concurrent_prepares: 0,
                ..PreparedDecideLimits::default()
            },
        ] {
            assert!(
                limits.validate().is_err(),
                "required level 0 must validate as disabled/deny"
            );
            assert!(PreparedDecideRegistry::new(limits.clone()).is_err());
        }
    }

    #[test]
    fn oversized_prepare_concurrency_rejects_before_semaphore() {
        let limits = PreparedDecideLimits {
            max_concurrent_prepares: MAX_CONCURRENT_PREPARES_CEILING + 1,
            ..PreparedDecideLimits::default()
        };
        assert!(
            limits.validate().is_err(),
            "ceiling violation must fail validation"
        );
    }

    #[test]
    fn optional_per_owner_zero_stays_legal() {
        // OPTIONAL per-owner zero = unlimited at that stricter level, and it
        // must not trip the REQUIRED-level zero=deny validation.
        let limits = PreparedDecideLimits {
            max_statements_per_owner: 0,
            max_bound_bytes_per_owner: 0,
            ..PreparedDecideLimits::default()
        };
        assert!(limits.validate().is_ok());
        assert!(PreparedDecideRegistry::new(limits).is_ok());
    }

    // -- F3: exact-capacity normalization ------------------------------------

    #[tokio::test]
    async fn same_spelling_huge_capacity_is_charged_exactly() {
        let registry = registry_with(PreparedDecideLimits {
            max_bound_bytes_per_tenant: 5,
            ..PreparedDecideLimits::default()
        });
        let alice = owner(TENANT, "alice");
        let handle = insert_for(&registry, &alice);

        // 1 byte of content carrying ~1 MiB of spare capacity: the accounted
        // charge must be the exact content size, not the caller's capacity.
        let mut bloated = String::with_capacity(1024 * 1024);
        bloated.push('x');
        registry.rebind(&alice, &handle, vec![bloated]).unwrap();
        let binding = registry.binding(&alice, &handle).unwrap();
        assert_eq!(
            binding.bound_bytes(),
            1,
            "normalization must trim spare capacity"
        );
        assert_eq!(registry.open_counts().bound_bytes_global, 1);

        // The stored state keeps the same spelling (1 char).
        assert_eq!(binding.states().first().map(String::len), Some(1));
    }

    // -- consistency: valid-shape but disagreeing resolutions ---------------

    fn resolved_with(
        commit: &str,
        mutate: impl FnOnce(&mut ResolvedDecisionPins),
    ) -> ResolvedDecisionPins {
        let mut r = resolved(commit);
        mutate(&mut r);
        r
    }

    /// Resolver that IGNORES the expected commit argument and always yields
    /// the COMMIT-based resolution — isolating the operator's exact-commit
    /// comparison from the fake's own guard.
    struct FixedResolver(std::sync::Arc<ResolvedDecisionPins>);
    #[async_trait::async_trait]
    impl DecideResolve for FixedResolver {
        async fn resolve_pins(
            &self,
            _owner: &OwnerKey,
            _bearer: &OriginalBearer,
            _expected_artifact_commit: &str,
            _decide_document: &str,
        ) -> Result<ResolvedDecisionPins, DecisionSubstrateError> {
            Ok(ResolvedDecisionPins::clone(&self.0))
        }
    }

    #[tokio::test]
    async fn valid_shape_wrong_commit_inserts_nothing() {
        let fixed = std::sync::Arc::new(resolved(COMMIT));
        let counter = FixedResolver(std::sync::Arc::clone(&fixed));
        let operator = DecideOperator::new(
            std::sync::Arc::new(counter),
            PreparedDecideLimits::default(),
        )
        .expect("valid limits");
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        // Valid caller document; the ONLY inconsistency is the supplied
        // expected commit (f*40) vs the resolved pins (7b43...).
        let other_commit = "f".repeat(40);
        let document = r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#;
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, &other_commit, document).await,
            Err(DecisionSubstrateError::InvalidArgument(m))
                if m.contains("disagree with the prepared expected artifact commit")
        ));
        assert_eq!(
            operator.registry().open_counts().statements_global,
            0,
            "zero insert"
        );
    }

    #[tokio::test]
    async fn valid_shape_wrong_arrow_identity_inserts_nothing() {
        let fake = std::sync::Arc::new(FakeResolve {
            calls: AtomicU64::new(0),
            delay: None,
            behavior: FakeBehavior::FingerprintMismatch,
        });
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        // The fake sets resolved.arrow_fingerprint AND pins
        // arrow_schema_identity to a mismatched value with otherwise valid
        // shapes; the independent recompute must catch both.
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#).await,
            Err(DecisionSubstrateError::InvalidArgument(m))
                if m.contains("Arrow") || m.contains("Arrow identity")
        ));
        assert_eq!(counter.calls(), 1);
        assert_eq!(operator.registry().open_counts().statements_global, 0);
    }

    #[tokio::test]
    async fn valid_shape_wrong_raw_ids_insert_nothing() {
        // Resolution whose raw ordered ids (q9) disagree with its own parsed
        // set (q1): the operator must reject on the exact-ids check.
        let tampered = resolved_with(COMMIT, |r| {
            r.raw = Arc::new(RawQuestionsTemplate {
                raw_json: Arc::from(r#"{"q9":{"type":"noul"}}"#),
                question_ids: Arc::from([String::from("q9")]),
            });
        });
        struct Tampered(std::sync::Arc<ResolvedDecisionPins>);
        #[async_trait::async_trait]
        impl DecideResolve for Tampered {
            async fn resolve_pins(
                &self,
                _owner: &OwnerKey,
                _bearer: &OriginalBearer,
                _expected: &str,
                _doc: &str,
            ) -> Result<ResolvedDecisionPins, DecisionSubstrateError> {
                Ok(ResolvedDecisionPins::clone(&self.0))
            }
        }
        let operator = DecideOperator::new(
            std::sync::Arc::new(Tampered(std::sync::Arc::new(tampered))),
            PreparedDecideLimits::default(),
        )
        .expect("valid limits");
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        let document = r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#;
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, COMMIT, document).await,
            Err(DecisionSubstrateError::InvalidArgument(m))
                if m.contains("original questions JSON")
        ));
        assert_eq!(
            operator.registry().open_counts().statements_global,
            0,
            "zero insert"
        );
    }

    #[tokio::test]
    async fn raw_instructions_mismatch_inserts_nothing() {
        // Raw template drops the instructions the resolved QuestionSet
        // carries: the strict IR reproduction check must catch it.
        let tampered = resolved_with(COMMIT, |r| {
            r.raw = Arc::new(RawQuestionsTemplate {
                raw_json: Arc::from(r#"{"q1":{"type":"noul"}}"#),
                question_ids: Arc::from([String::from("q1")]),
            });
        });
        struct RawEcho(std::sync::Arc<ResolvedDecisionPins>);
        #[async_trait::async_trait]
        impl DecideResolve for RawEcho {
            async fn resolve_pins(
                &self,
                _owner: &OwnerKey,
                _bearer: &OriginalBearer,
                _expected: &str,
                _doc: &str,
            ) -> Result<ResolvedDecisionPins, DecisionSubstrateError> {
                Ok(ResolvedDecisionPins::clone(&self.0))
            }
        }
        let operator = DecideOperator::new(
            std::sync::Arc::new(RawEcho(std::sync::Arc::new(tampered))),
            PreparedDecideLimits::default(),
        )
        .expect("valid limits");
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#).await,
            Err(DecisionSubstrateError::InvalidArgument(m))
                if m.contains("original questions JSON")
        ));
        assert_eq!(operator.registry().open_counts().statements_global, 0);
    }

    // -- identity-cell cardinality: held closed zero snapshots ---------------

    #[tokio::test]
    async fn held_closed_zero_snapshots_cap_new_identities_then_release() {
        // Zero-byte initial-binding leases keep their identity cells alive
        // past close (held snapshots): the map fills to the identity cap and
        // a NEW identity is rejected until the held snapshots release.
        let registry = registry_with(PreparedDecideLimits {
            max_statements_global: 8,
            ..PreparedDecideLimits::default()
        });
        let mut held: Vec<Arc<PreparedStatement>> = Vec::new();
        for i in 0..16 {
            let owner = owner(TENANT, &format!("principal-{i}"));
            let handle = insert_for(&registry, &owner);
            held.push(registry.snapshot(&owner, &handle).unwrap());
            registry.close(&owner, &handle).unwrap();
        }
        assert_eq!(registry.open_counts().statements_global, 0);

        // Identity map is full of HELD closed zero snapshots: a new
        // identity must be rejected.
        let fresh = owner(TENANT, "principal-fresh");
        assert!(matches!(
            registry.insert(fresh.clone(), resolved(COMMIT)),
            Err(DecisionSubstrateError::ResourceExhausted(m)) if m.contains("identity cache full")
        ));

        // Release every held snapshot: cells prune and admission works.
        held.clear();
        assert!(registry.insert(fresh, resolved(COMMIT)).is_ok());
    }

    #[tokio::test]
    async fn full_identity_map_existing_owner_succeeds_new_identity_rejects() {
        // identity_cell_cap ties to max_statements_global (floor 16) -> 16
        // with max_statements_global = 32... in THIS test the ceiling is
        // derived from the actual limits below.
        let limits = PreparedDecideLimits {
            max_statements_global: 32,
            ..PreparedDecideLimits::default()
        };
        let cap = identity_cell_cap(limits.max_statements_global);
        let registry = registry_with(limits.clone());

        // Fill cap-1 byte identity cells with closed-but-held snapshots.
        let mut held: Vec<Arc<PreparedStatement>> = Vec::new();
        let mut held_bytes: usize = 0;
        for i in 0..(cap - 1) {
            let owner = owner(TENANT, &format!("principal-{i}"));
            let handle = insert_for(&registry, &owner);
            let state = format!("s{i}");
            registry
                .rebind(&owner, &handle, vec![state.clone()])
                .unwrap();
            held_bytes += state.len();
            held.push(registry.snapshot(&owner, &handle).unwrap());
            registry.close(&owner, &handle).unwrap();
        }

        // The OPEN keeper fills the LAST identity cell.
        let keeper = owner(TENANT, "keeper");
        let keeper_handle = insert_for(&registry, &keeper);
        let keeper_state = "keeper-state".to_owned();
        registry
            .rebind(&keeper, &keeper_handle, vec![keeper_state.clone()])
            .unwrap();
        held_bytes += keeper_state.len();
        assert_eq!(
            registry.open_counts().bound_bytes_global as usize,
            held_bytes,
            "fill must account every held byte exactly"
        );

        // Causal positive control at the FULL map: the keeper (existing
        // identity, equal-size rebind) causes no cell growth and succeeds.
        let generation = registry
            .rebind(&keeper, &keeper_handle, vec![format!("{keeper_state}-2")])
            .expect("existing identity must not be rejected by the full map");
        assert_eq!(generation.0, 3);
        held_bytes += 2; // "-2" suffix adds two bytes

        // Hold a keeper snapshot BEFORE the rejection: the keeper cell stays
        // live through the assertion below.
        let keeper_snapshot = registry.binding(&keeper, &keeper_handle).unwrap();
        let charged_at_full = registry.open_counts().bound_bytes_global;
        assert_eq!(charged_at_full as usize, held_bytes);

        // A genuinely NEW identity is rejected while every cell is held;
        // the charged bytes are unchanged by the rejected admission.
        let fresh = owner(TENANT, "principal-fresh");
        assert!(matches!(
            registry.insert(fresh.clone(), resolved(COMMIT)),
            Err(DecisionSubstrateError::ResourceExhausted(m))
                if m.contains("identity cache full")
        ));
        assert_eq!(
            registry.open_counts().bound_bytes_global,
            charged_at_full,
            "a rejected admission must not change the charged bytes"
        );

        // Explicit terminal: close the keeper, drop its snapshot and the
        // held snapshots — their cells become prunable and the fresh
        // identity is admitted. The keeper binding is released by the close
        // (no keeper snapshot is retained past this point).
        registry
            .close(&keeper, &keeper_handle)
            .expect("keeper close at terminal");
        drop(keeper_snapshot);
        drop(held);

        assert!(registry.insert(fresh, resolved(COMMIT)).is_ok());
        assert_eq!(registry.open_counts().statements_global, 1);
    }

    // -- resolver returns a COHERENT alternative template --------------------

    /// Resolver that ignores the caller document and returns its OWN
    /// internally coherent alternative (different instructions; matching raw
    /// JSON; matching fingerprint). All consistency INTERNALS hold; only the
    /// independent caller-document comparison can reject it.
    struct AlternativeTemplateResolver {
        calls: std::sync::atomic::AtomicU64,
    }

    #[async_trait::async_trait]
    impl DecideResolve for AlternativeTemplateResolver {
        async fn resolve_pins(
            &self,
            _owner: &OwnerKey,
            _bearer: &OriginalBearer,
            _expected_artifact_commit: &str,
            _decide_document: &str,
        ) -> Result<ResolvedDecisionPins, DecisionSubstrateError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let question_set = Arc::new(QuestionSet {
                state: None,
                questions: vec![QuestionSpec {
                    id: "q1".to_owned(),
                    kind: QuestionKind::Noul,
                    instructions: Some(Entry::Str(
                        "ALTERNATIVE phrasing from a foreign template.".to_owned(),
                    )),
                    body: QuestionBody::Noul { criteria: None },
                }],
            });
            let fingerprint = Arc::<str>::from(
                DecisionSchema::from_question_set(&question_set)
                    .expect("columnar surface")
                    .fingerprint()
                    .as_str(),
            );
            let pins = DecisionPins {
                authorization_model_ref: "clef-flash-fp8:main".to_owned(),
                expected_artifact_commit: COMMIT.to_owned(),
                schema_version: "jev-1".to_owned(),
                model_id: "clef-flash@7b43c2d7".to_owned(),
                encoding_version: "clef-native-v1".to_owned(),
                quantization: "fp8-e4m3-dynamic".to_owned(),
                admitted_compute_profile: "cpu-f32".to_owned(),
                head_weights_sha256: "b".repeat(64),
                head_config_sha256: "c".repeat(64),
                tokenizer_sha256: "d".repeat(64),
                backbone_config_sha256: "e".repeat(64),
                arrow_schema_identity: fingerprint.to_string(),
                calibration_version: None,
            };
            Ok(ResolvedDecisionPins {
                pins: Arc::new(pins),
                question_set,
                raw: Arc::new(RawQuestionsTemplate {
                    raw_json: Arc::from(
                        r#"{"q1":{"type":"noul","instructions":"ALTERNATIVE phrasing from a foreign template."}}"#,
                    ),
                    question_ids: Arc::from([String::from("q1")]),
                }),
                arrow_fingerprint: fingerprint,
            })
        }
    }

    #[tokio::test]
    async fn coherent_alternative_template_from_resolver_inserts_nothing() {
        // Caller document (UNCHANGED, coherent fixture with instructions):
        let caller_document = r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#;
        let fake = std::sync::Arc::new(AlternativeTemplateResolver {
            calls: std::sync::atomic::AtomicU64::new(0),
        });
        let counter = std::sync::Arc::clone(&fake);
        let operator =
            DecideOperator::new(fake, PreparedDecideLimits::default()).expect("valid limits");
        let caller = ctx(Some(TENANT), "alice", Some(now() + 600));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        // The alternative template is internally coherent (its own raw/IR/
        // fingerprint all agree), so ONLY the independent caller-document
        // comparison can reject it — and it must.
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, COMMIT, caller_document).await,
            Err(DecisionSubstrateError::InvalidArgument(m))
                if m.contains("original questions JSON")
        ));
        assert_eq!(
            counter.calls.load(Ordering::Relaxed),
            1,
            "resolver consulted once"
        );
        assert_eq!(
            operator.registry().open_counts().statements_global,
            0,
            "a foreign coherent template must insert nothing"
        );
        assert_eq!(
            operator.registry().open_counts().bound_bytes_global,
            0,
            "a rejected foreign template must leak no bound-byte state"
        );
    }

    #[tokio::test]
    async fn authorize_gate_remains_required_by_construction() {
        // A context with no tenant cannot even form an owner key: the G1
        // authorize + substrate owner derivation stay the only entry path.
        let fake = FakeResolve::allow();
        let counter = std::sync::Arc::clone(&fake);
        let _ = &counter;
        let operator = operator_with(fake, PreparedDecideLimits::default());
        let caller = ctx(None, "alice", Some(i64::MAX));
        let bearer = OriginalBearer::from_authorization_header(Some("Bearer user-jwt")).unwrap();
        assert!(matches!(
            operator.create_prepared(&caller, &bearer, COMMIT, r#"{"schema_version":"jev-1","questions":{"q1":{"type":"noul","instructions":"The service is down."}}}"#).await,
            Err(DecisionSubstrateError::Unauthenticated(_))
        ));
        assert_eq!(counter.calls(), 0);
    }
}
