//! Typed MoQ/event object identity + declared track-policy resolver (v16 §10,
//! WS-I / #1510).
//!
//! The MoQ/event plane previously passed its track/prefix string into the
//! *RPC/VFS* resolver's service-domain coordinate
//! (`pep.rs` → [`super::dispatch_pep::RpcObjectLabelResolver`]), so a track
//! name was reinterpreted as a VFS path. v16 §10 removes that shared string
//! coordinate: every plane owns a typed object reference and a plane-specific
//! resolver, and "strings are parsed exactly once at the plane boundary and
//! cannot be reinterpreted by another resolver".
//!
//! This module is that end state for MoQ/event:
//!
//! - [`MoqEventObjectRef`] is the exact decoded typed identity. A boundary
//!   string parses into it exactly once ([`MoqEventObjectRef::parse`]) or the
//!   check denies — there is no string path fallback and no rewriting of
//!   unknown input after the fact.
//! - The service coordinate inside the identity is validated by the **one**
//!   canonical service-domain rule
//!   ([`validate_service_domain`][crate::envelope::validate_service_domain],
//!   1..=[`MAX_SERVICE_DOMAIN_BYTES`] bytes) — not a second, plane-local
//!   identity rule.
//! - [`MoqEventLabelResolver`] is the plane's own resolver trait
//!   (a `MoqEventObjectRef` in, a [`SecurityLabel`] out). It is deliberately
//!   *not* the RPC resolver: dispatch, VFS/9P, CAS, and MoQ/event coordinates
//!   cannot cross resolver types.
//! - [`MoqEventPolicyTable`] is the declared track-policy metadata — the
//!   authoritative label source for this plane. Lookup is an exact typed match
//!   against declared rows; unknown, unlisted, or newer-vocabulary inputs
//!   deny. There is no floor allowlist and no bootstrap exception: a table
//!   that declares nothing labels nothing.
//!
//! ## Dependency on the generated dispatch inventory (WS-D)
//!
//! The end-state source of declared rows is the generated method-policy
//! inventory (WS-D / #1505), which emits one reviewed row per annotated leaf
//! across the service schemas. Until that inventory lands, the table is the
//! **interface seam**: it is built from explicitly declared
//! [`MoqEventPolicyRow`]s and rejects (not ignores) malformed input at
//! construction. WS-D's generated code converts its inventory into rows here
//! through one reviewed conversion — this module must not duplicate the
//! annotation table.
//!
//! This module compiles for wasm32 (pure types + the shared canonicalization
//! rule); the PEP that consumes it remains native-only, matching
//! [`super::pep`].

use std::collections::BTreeMap;

use super::label::SecurityLabel;
use crate::envelope::{validate_service_domain, MAX_SERVICE_DOMAIN_BYTES};

/// Maximum number of segments in a track/prefix identity. A bound on parse
/// work per admission check; not the service-domain rule (which is
/// byte-length based and applies only to the service segment).
pub const MAX_TRACK_SEGMENTS: usize = 8;

/// Maximum length in bytes of one non-service identity segment. Distinct from
/// [`MAX_SERVICE_DOMAIN_BYTES`], which bounds only the canonical service
/// domain through the shared rule.
pub const MAX_TRACK_SEGMENT_BYTES: usize = 128;

/// The track-policy vocabulary revision this resolver understands. A policy
/// table stamped with a *newer* revision (emitted by a newer generator whose
/// vocabulary this build does not know) is rejected rather than partially
/// interpreted — the "newer identities deny" rule.
pub const SUPPORTED_TRACK_POLICY_REVISION: u32 = 1;

/// Drift guard: the plane grammar is specified against the shared canonical
/// service-domain cap (Gate-2 value 7 froze `aud` at 128 bytes through the
/// same constant). If the shared rule's bound changes, this module's bounds
/// must be re-reviewed together.
const _: () = assert!(
    MAX_SERVICE_DOMAIN_BYTES == 128,
    "the shared canonical service-domain cap changed; re-review the MoQ/event plane grammar bounds"
);

/// The MoQ data plane a coordinate belongs to. Closed and exhaustive: adding
/// a plane is a resolver-contract change, not a string reinterpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MoqEventPlane {
    /// Event-bus topic prefixes (`{source}.{entity}.{event}` dot grammar;
    /// the authz coordinate is the `{source}` prefix).
    Event,
    /// MoQ stream tracks / broadcast paths (`{tenant}/{service}/{topic}/…`
    /// and `local/{service}/…` slash grammars).
    Stream,
}

impl MoqEventPlane {
    /// Wire-stable discriminant for decoded ingress.
    pub const fn discriminant(self) -> u16 {
        match self {
            MoqEventPlane::Event => 0,
            MoqEventPlane::Stream => 1,
        }
    }

    /// Decode a plane discriminant. Unknown values return `None` — the
    /// boundary denies rather than falling back to a default plane.
    pub const fn from_discriminant(value: u16) -> Option<Self> {
        match value {
            0 => Some(MoqEventPlane::Event),
            1 => Some(MoqEventPlane::Stream),
            _ => None,
        }
    }
}

/// The exact decoded track/prefix identity for one MoQ/event object.
///
/// Constructed only by [`Self::parse`] at the plane boundary. The service
/// coordinate is canonical (shared 1..=128-byte rule); the remaining segments
/// are preserved verbatim so the identity is exact, and the plane makes it
/// non-reinterpretable by any other plane's resolver.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MoqEventObjectRef {
    plane: MoqEventPlane,
    service_domain: String,
    segments: Box<[String]>,
}

impl MoqEventObjectRef {
    /// Parse the plane's coordinate grammar exactly once.
    ///
    /// Returns `None` for every deviation — wrong segment count, empty or
    /// oversized segments, path-traversal segments, a `/` inside an event
    /// coordinate (which would alias the broadcast-path namespace), or a
    /// noncanonical service segment. Unknown input denies; it is never
    /// rewritten.
    pub fn parse(plane: MoqEventPlane, coordinate: &str) -> Option<Self> {
        match plane {
            MoqEventPlane::Event => Self::parse_dot_coordinate(coordinate),
            MoqEventPlane::Stream => Self::parse_slash_coordinate(coordinate),
        }
    }

    /// Parse an event-plane topic prefix: dot-separated, 1..=
    /// [`MAX_TRACK_SEGMENTS`] segments, segment 0 is the canonical service
    /// domain (the event `source`). Segments never contain `/`: the event
    /// namespace is flat per source, and a `/` would alias a broadcast path.
    fn parse_dot_coordinate(coordinate: &str) -> Option<Self> {
        let mut segments = Vec::new();
        for segment in coordinate.split('.') {
            check_segment(segment)?;
            // The event namespace is flat per source: a `/` in an event
            // segment would alias the slash-separated broadcast-path
            // namespace, so it denies here rather than parse ambiguously.
            if segment.contains('/') {
                return None;
            }
            segments.push(segment.to_owned());
        }
        if segments.len() > MAX_TRACK_SEGMENTS {
            return None;
        }
        let service_domain = canonical_domain(&segments[0])?;
        Some(Self {
            plane: MoqEventPlane::Event,
            service_domain,
            segments: segments.into(),
        })
    }

    /// Parse a stream-plane track/broadcast name: slash-separated,
    /// 2..=[`MAX_TRACK_SEGMENTS`] segments, segment 1 is the canonical
    /// service domain (`{tenant}/{service}/…` and `local/{service}/…` both
    /// carry it at index 1). A lone segment has no service coordinate and
    /// denies.
    fn parse_slash_coordinate(coordinate: &str) -> Option<Self> {
        let mut segments = Vec::new();
        for segment in coordinate.split('/') {
            check_segment(segment)?;
            segments.push(segment.to_owned());
        }
        if segments.len() < 2 || segments.len() > MAX_TRACK_SEGMENTS {
            return None;
        }
        let service_domain = canonical_domain(&segments[1])?;
        Some(Self {
            plane: MoqEventPlane::Stream,
            service_domain,
            segments: segments.into(),
        })
    }

    /// The plane this identity belongs to.
    pub fn plane(&self) -> MoqEventPlane {
        self.plane
    }

    /// The canonical service domain of the declaring service.
    pub fn service_domain(&self) -> &str {
        &self.service_domain
    }

    /// The exact (verbatim, post-validation) identity segments.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// The raw coordinate this ref would audit as (plane-tagged; the audit
    /// trail records what was decoded, never a reinterpretation).
    pub fn audit_coordinate(&self) -> String {
        let separator = match self.plane {
            MoqEventPlane::Event => '.',
            MoqEventPlane::Stream => '/',
        };
        let mut out = String::new();
        for segment in &self.segments[..] {
            if !out.is_empty() {
                out.push(separator);
            }
            out.push_str(segment);
        }
        out
    }
}

/// Validate one non-service identity segment: non-empty, bounded, no NUL, no
/// path-traversal names. (The service segment additionally passes the shared
/// canonical service-domain rule.)
fn check_segment(segment: &str) -> Option<()> {
    if segment.is_empty()
        || segment.len() > MAX_TRACK_SEGMENT_BYTES
        || segment == "."
        || segment == ".."
        || segment.as_bytes().contains(&0)
    {
        return None;
    }
    Some(())
}

/// Apply the one canonical service-domain rule. A segment that is not
/// already canonical denies — it is never rewritten.
fn canonical_domain(segment: &str) -> Option<String> {
    validate_service_domain(segment).ok()?;
    Some(segment.to_owned())
}

/// The MoQ/event plane's own object-label resolver (v16 §10).
///
/// Takes the typed [`MoqEventObjectRef`] only — never a bare string, never
/// another plane's coordinate. `None` ⇒ unlabeled ⇒ deny (D2/D3).
pub trait MoqEventLabelResolver: Send + Sync {
    fn resolve(&self, object: &MoqEventObjectRef) -> Option<SecurityLabel>;
}

/// Fail-closed resolver: every object is unlabeled. The structural default
/// when no declared track-policy table exists; combined with an installed
/// PEP it denies everything regardless of clearance.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllMoqEventResolver;

impl MoqEventLabelResolver for DenyAllMoqEventResolver {
    fn resolve(&self, _object: &MoqEventObjectRef) -> Option<SecurityLabel> {
        None
    }
}

/// Why a declared track-policy table could not be built. Construction
/// failure is fail-closed: the caller keeps no partial table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackPolicyError {
    /// The table's vocabulary revision is newer than
    /// [`SUPPORTED_TRACK_POLICY_REVISION`]; this build refuses to partially
    /// interpret it.
    UnsupportedRevision { table: u32, supported: u32 },
    /// A row declares a service domain that fails the canonical rule.
    NoncanonicalServiceDomain(String),
    /// Two rows declare the same plane + service domain with different
    /// labels. Ambiguous declarations are rejected, never merged.
    DuplicateDeclaration {
        plane: MoqEventPlane,
        service_domain: String,
    },
}

impl std::fmt::Display for TrackPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackPolicyError::UnsupportedRevision { table, supported } => write!(
                f,
                "track-policy revision {table} is newer than supported revision {supported}"
            ),
            TrackPolicyError::NoncanonicalServiceDomain(domain) => write!(
                f,
                "track-policy row declares noncanonical service domain {domain:?}"
            ),
            TrackPolicyError::DuplicateDeclaration {
                plane,
                service_domain,
            } => write!(
                f,
                "duplicate track-policy declaration for plane {:?} service {service_domain:?}",
                plane
            ),
        }
    }
}

impl std::error::Error for TrackPolicyError {}

/// One declared track-policy row: the declaring service's label for its
/// objects on one plane. The generated dispatch inventory (WS-D) is the
/// end-state producer; rows are otherwise declared explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoqEventPolicyRow {
    /// The plane this declaration covers.
    pub plane: MoqEventPlane,
    /// The canonical service domain of the declaring service.
    pub service_domain: String,
    /// The label for the declared service's objects on that plane.
    pub label: SecurityLabel,
}

impl MoqEventPolicyRow {
    /// Build a row, rejecting a noncanonical service domain at construction
    /// (the shared rule — there is no plane-local rewrite).
    pub fn new(
        plane: MoqEventPlane,
        service_domain: impl Into<String>,
        label: SecurityLabel,
    ) -> Result<Self, TrackPolicyError> {
        let service_domain = service_domain.into();
        if validate_service_domain(&service_domain).is_err() {
            return Err(TrackPolicyError::NoncanonicalServiceDomain(service_domain));
        }
        Ok(Self {
            plane,
            service_domain,
            label,
        })
    }
}

/// Declared track-policy metadata: the closed, validated set of rows the
/// resolver trusts. This is the reviewed seam onto the generated inventory.
///
/// Lookup is an exact typed match on `(plane, canonical service domain)`.
/// Anything not declared — unknown service, other plane, empty table — is
/// unlabeled and denies. There is no ancestor walk, no floor, and no
/// bootstrap exception.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoqEventPolicyTable {
    revision: u32,
    rows: BTreeMap<(MoqEventPlane, String), SecurityLabel>,
}

impl MoqEventPolicyTable {
    /// Build the table from declared rows.
    ///
    /// Fails (returning no table) on a revision newer than supported, a
    /// noncanonical service domain, or two rows declaring the same plane +
    /// service with different labels. Older or equal revisions are accepted:
    /// a newer *server* understanding an older table is forward-compatible;
    /// the reverse is not, and denies.
    pub fn build(
        revision: u32,
        rows: impl IntoIterator<Item = MoqEventPolicyRow>,
    ) -> Result<Self, TrackPolicyError> {
        if revision > SUPPORTED_TRACK_POLICY_REVISION {
            return Err(TrackPolicyError::UnsupportedRevision {
                table: revision,
                supported: SUPPORTED_TRACK_POLICY_REVISION,
            });
        }
        let mut map = BTreeMap::new();
        for row in rows {
            if validate_service_domain(&row.service_domain).is_err() {
                return Err(TrackPolicyError::NoncanonicalServiceDomain(
                    row.service_domain,
                ));
            }
            let key = (row.plane, row.service_domain);
            if let Some(existing) = map.get(&key) {
                if *existing != row.label {
                    return Err(TrackPolicyError::DuplicateDeclaration {
                        plane: row.plane,
                        service_domain: key.1,
                    });
                }
                continue;
            }
            map.insert(key, row.label);
        }
        Ok(Self {
            revision,
            rows: map,
        })
    }

    /// The table that declares nothing. Every lookup is unlisted and denies;
    /// this is the honest pre-inventory state, not a bypass.
    pub fn empty() -> Self {
        Self {
            revision: SUPPORTED_TRACK_POLICY_REVISION,
            rows: BTreeMap::new(),
        }
    }

    /// The vocabulary revision this table was declared at.
    pub fn revision(&self) -> u32 {
        self.revision
    }

    /// Number of declared rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no rows are declared.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Exact typed lookup: the declared label for this object's plane +
    /// canonical service domain, or `None` when unlisted.
    pub fn resolve(&self, object: &MoqEventObjectRef) -> Option<SecurityLabel> {
        self.rows
            .get(&(object.plane, object.service_domain.clone()))
            .copied()
    }
}

/// The production resolver: exact-match lookup against a validated declared
/// track-policy table.
#[derive(Debug, Clone)]
pub struct DeclaredTrackPolicyResolver {
    table: MoqEventPolicyTable,
}

impl DeclaredTrackPolicyResolver {
    /// Wrap a validated table.
    pub fn new(table: MoqEventPolicyTable) -> Self {
        Self { table }
    }

    /// The declared table this resolver enforces.
    pub fn table(&self) -> &MoqEventPolicyTable {
        &self.table
    }
}

impl MoqEventLabelResolver for DeclaredTrackPolicyResolver {
    fn resolve(&self, object: &MoqEventObjectRef) -> Option<SecurityLabel> {
        self.table.resolve(object)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::auth::mac::{Assurance, CompartmentSet, Level};

    fn public_label() -> SecurityLabel {
        SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn secret_label() -> SecurityLabel {
        SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY)
    }

    // ── plane discriminants ───────────────────────────────────────────

    #[test]
    fn plane_discriminants_decode_exactly_the_known_set() {
        assert_eq!(
            MoqEventPlane::from_discriminant(MoqEventPlane::Event.discriminant()),
            Some(MoqEventPlane::Event)
        );
        assert_eq!(
            MoqEventPlane::from_discriminant(MoqEventPlane::Stream.discriminant()),
            Some(MoqEventPlane::Stream)
        );
        for unknown in [2u16, 3, 7, u16::MAX] {
            assert_eq!(MoqEventPlane::from_discriminant(unknown), None);
        }
    }

    // ── event-prefix identity ─────────────────────────────────────────

    #[test]
    fn event_prefix_parses_source_as_canonical_service_domain() {
        let object =
            MoqEventObjectRef::parse(MoqEventPlane::Event, "worker").expect("bare source parses");
        assert_eq!(object.plane(), MoqEventPlane::Event);
        assert_eq!(object.service_domain(), "worker");
        assert_eq!(object.segments(), ["worker"].map(String::from).as_slice());

        let object = MoqEventObjectRef::parse(MoqEventPlane::Event, "registry.repo789.push")
            .expect("topic prefix parses");
        assert_eq!(object.service_domain(), "registry");
        assert_eq!(
            object.segments(),
            ["registry", "repo789", "push"].map(String::from).as_slice()
        );
    }

    #[test]
    fn event_prefix_rejects_noncanonical_service_domains() {
        // The shared canonical rule: lowercase-initial, charset, 1..=128
        // bytes. None of these are rewritten — they deny.
        for bad in [
            "", "Worker",  // uppercase
            "_worker", // starts with underscore, not letter/digit
            "wor ker", // space
            "wo:rker", // internal map-key separator
            "wor/ker", // broadcast-path alias
        ] {
            assert!(
                MoqEventObjectRef::parse(MoqEventPlane::Event, bad).is_none(),
                "{bad:?} must not parse as an event prefix"
            );
        }
    }

    #[test]
    fn event_prefix_rejects_oversized_and_unbounded_input() {
        let long_segment = "w".repeat(MAX_TRACK_SEGMENT_BYTES + 1);
        assert!(MoqEventObjectRef::parse(MoqEventPlane::Event, &long_segment).is_none());

        let too_many = ["worker"; MAX_TRACK_SEGMENTS + 1].join(".");
        assert!(MoqEventObjectRef::parse(MoqEventPlane::Event, &too_many).is_none());

        let oversized_domain = "w".repeat(MAX_SERVICE_DOMAIN_BYTES + 1);
        assert!(
            MoqEventObjectRef::parse(MoqEventPlane::Event, &oversized_domain).is_none(),
            "service domain beyond the shared 128-byte rule denies"
        );
    }

    #[test]
    fn event_prefix_rejects_empty_segments() {
        for bad in [".worker", "worker.", "worker..push", "wor..ker"] {
            assert!(MoqEventObjectRef::parse(MoqEventPlane::Event, bad).is_none());
        }
    }

    // ── stream-track identity ─────────────────────────────────────────

    #[test]
    fn stream_track_parses_service_at_second_segment() {
        let object = MoqEventObjectRef::parse(MoqEventPlane::Stream, "alice/streams/run-1/i0")
            .expect("tenant track parses");
        assert_eq!(object.plane(), MoqEventPlane::Stream);
        assert_eq!(object.service_domain(), "streams");
        assert_eq!(
            object.segments(),
            ["alice", "streams", "run-1", "i0"]
                .map(String::from)
                .as_slice()
        );

        let object = MoqEventObjectRef::parse(MoqEventPlane::Stream, "local/streams/deadbeef")
            .expect("broadcast path parses");
        assert_eq!(object.service_domain(), "streams");

        let object = MoqEventObjectRef::parse(MoqEventPlane::Stream, "local/events/worker")
            .expect("event broadcast path parses under the stream grammar");
        assert_eq!(object.service_domain(), "events");
    }

    #[test]
    fn stream_track_rejects_single_segment_and_bad_grammar() {
        // A lone segment has no service coordinate.
        assert!(MoqEventObjectRef::parse(MoqEventPlane::Stream, "alice").is_none());
        // Empty segments and traversal names deny.
        for bad in [
            "/streams/x",
            "alice//run",
            "alice/streams/run/",
            "./streams/x",
            "alice/../streams/run",
        ] {
            assert!(
                MoqEventObjectRef::parse(MoqEventPlane::Stream, bad).is_none(),
                "{bad:?} must not parse"
            );
        }
        // Noncanonical service segment denies.
        assert!(MoqEventObjectRef::parse(MoqEventPlane::Stream, "alice/Streams/x").is_none());
    }

    #[test]
    fn identities_are_not_reinterpretable_across_grammars() {
        // An event coordinate with a '/' cannot smuggle a broadcast path.
        assert!(MoqEventObjectRef::parse(MoqEventPlane::Event, "local/events/worker").is_none());
        // A stream coordinate and an event coordinate for similar-looking
        // strings remain distinct typed identities (different plane, domain).
        let stream = MoqEventObjectRef::parse(MoqEventPlane::Stream, "local/events/worker")
            .expect("stream grammar applies");
        let event = MoqEventObjectRef::parse(MoqEventPlane::Event, "events.worker")
            .expect("event grammar applies");
        assert_eq!(stream.service_domain(), "events");
        assert_eq!(event.service_domain(), "events");
        assert_ne!(stream, event, "plane is part of identity");
    }

    #[test]
    fn audit_coordinate_round_trips_the_decoded_segments() {
        let object = MoqEventObjectRef::parse(MoqEventPlane::Stream, "alice/streams/run-1/i0")
            .expect("parses");
        assert_eq!(object.audit_coordinate(), "alice/streams/run-1/i0");
        let object =
            MoqEventObjectRef::parse(MoqEventPlane::Event, "registry.repo789.push").unwrap();
        assert_eq!(object.audit_coordinate(), "registry.repo789.push");
    }

    // ── declared policy table ─────────────────────────────────────────

    fn declared_table() -> MoqEventPolicyTable {
        MoqEventPolicyTable::build(
            SUPPORTED_TRACK_POLICY_REVISION,
            [
                MoqEventPolicyRow::new(MoqEventPlane::Event, "worker", secret_label()).unwrap(),
                MoqEventPolicyRow::new(MoqEventPlane::Event, "registry", public_label()).unwrap(),
                MoqEventPolicyRow::new(MoqEventPlane::Stream, "streams", public_label()).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn table_resolves_exactly_the_declared_plane_and_service() {
        let table = declared_table();
        let worker =
            MoqEventObjectRef::parse(MoqEventPlane::Event, "worker.sandbox1.started").unwrap();
        assert_eq!(table.resolve(&worker), Some(secret_label()));

        // The same service on the *other* plane is a different object:
        // `worker` is declared for events only, so a stream track whose
        // service segment is `worker` is unlisted.
        let worker_as_stream =
            MoqEventObjectRef::parse(MoqEventPlane::Stream, "tenant/worker/topic1/i0").unwrap();
        assert_eq!(worker_as_stream.service_domain(), "worker");
        assert_eq!(table.resolve(&worker_as_stream), None);

        let streams = MoqEventObjectRef::parse(MoqEventPlane::Stream, "a/streams/b/c").unwrap();
        assert_eq!(table.resolve(&streams), Some(public_label()));
    }

    #[test]
    fn unlisted_services_and_empty_table_deny() {
        let table = declared_table();
        let unknown = MoqEventObjectRef::parse(MoqEventPlane::Event, "inference.session.x")
            .expect("parses, but is unlisted");
        assert_eq!(table.resolve(&unknown), None);

        let empty = MoqEventPolicyTable::empty();
        assert!(empty.is_empty());
        let declared = MoqEventObjectRef::parse(MoqEventPlane::Event, "worker").unwrap();
        assert_eq!(
            empty.resolve(&declared),
            None,
            "empty table denies everything"
        );
    }

    #[test]
    fn newer_table_revision_is_rejected_not_partially_interpreted() {
        let err = MoqEventPolicyTable::build(
            SUPPORTED_TRACK_POLICY_REVISION + 1,
            [MoqEventPolicyRow::new(MoqEventPlane::Event, "worker", public_label()).unwrap()],
        )
        .unwrap_err();
        assert_eq!(
            err,
            TrackPolicyError::UnsupportedRevision {
                table: SUPPORTED_TRACK_POLICY_REVISION + 1,
                supported: SUPPORTED_TRACK_POLICY_REVISION,
            }
        );
        // An older/equal revision is accepted (forward compatibility runs one
        // way: newer server, older table).
        MoqEventPolicyTable::build(
            SUPPORTED_TRACK_POLICY_REVISION,
            [MoqEventPolicyRow::new(MoqEventPlane::Event, "worker", public_label()).unwrap()],
        )
        .unwrap();
    }

    #[test]
    fn table_construction_is_fail_closed() {
        // Noncanonical row domain rejects.
        let err =
            MoqEventPolicyRow::new(MoqEventPlane::Event, "Worker", public_label()).unwrap_err();
        assert_eq!(
            err,
            TrackPolicyError::NoncanonicalServiceDomain("Worker".to_owned())
        );

        // Conflicting duplicate declarations reject — never merged.
        let err = MoqEventPolicyTable::build(
            SUPPORTED_TRACK_POLICY_REVISION,
            [
                MoqEventPolicyRow::new(MoqEventPlane::Event, "worker", public_label()).unwrap(),
                MoqEventPolicyRow::new(MoqEventPlane::Event, "worker", secret_label()).unwrap(),
            ],
        )
        .unwrap_err();
        assert_eq!(
            err,
            TrackPolicyError::DuplicateDeclaration {
                plane: MoqEventPlane::Event,
                service_domain: "worker".to_owned(),
            }
        );

        // Identical duplicate rows are idempotent, not ambiguous.
        MoqEventPolicyTable::build(
            SUPPORTED_TRACK_POLICY_REVISION,
            [
                MoqEventPolicyRow::new(MoqEventPlane::Event, "worker", public_label()).unwrap(),
                MoqEventPolicyRow::new(MoqEventPlane::Event, "worker", public_label()).unwrap(),
            ],
        )
        .unwrap();
    }

    #[test]
    fn deny_all_resolver_labels_nothing() {
        let resolver = DenyAllMoqEventResolver;
        let object = MoqEventObjectRef::parse(MoqEventPlane::Event, "worker").unwrap();
        assert_eq!(resolver.resolve(&object), None);
    }
}
