//! `mesh.hyprstream.io/v1alpha1` — [`TenantBinding`].
//!
//! [`TenantBinding`] is the **confused-deputy fix** from the epic #778 threat
//! table: an explicit, admin-created mapping between a Kubernetes namespace and
//! a hyprstream tenant. Creating one is gated by the `federation:register`
//! scope. The reconcilers (K5b) treat a binding as the *only* authority that
//! lets a resource in namespace `N` reference tenant `T`; any cross-binding
//! reference a `TenantBinding` does not cover is refused at admission.
//!
//! It is deliberately **cluster-scoped**: a tenant confined to its own
//! namespace must not be able to author the very binding that would grant it a
//! tenant identity. Only a cluster admin (holding `federation:register`) may
//! create these.
//!
//! # CRD → grant compilation (#929)
//!
//! A `TenantBinding` is an **admin-authorable surface**, never itself the
//! authority: when [`TenantBindingSpec::entitlement`] is set, the operator
//! *compiles* it into an issuer-signed UCAN grant (the operator signs as issuer)
//! and an [`hyprstream_pds::ledger::AllocationRecord`] that lands in the
//! tenant's inventory (PDS collection `ai.hyprstream.ledger.allocation`). The
//! enforcers (#781/#787/#790/#793/#794/#527) verify the *presented grant*, not
//! the CRD — the CRD only carries authorable intent. The compiled CIDs + epoch
//! are recorded in [`TenantBindingStatus`] as observed truth.
//!
//! Non-goal (per #921): a Kubernetes `ResourceQuota` as a central counter. The
//! grant is a held, offline-verifiable capability; accounting lives in the cell
//! ledger, never in a cluster-wide quota object.

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Desired state of a [`TenantBinding`]: namespace ↔ hyprstream tenant.
///
/// # Validation
/// - `namespace` must be a valid DNS-1123 label.
/// - `tenant` must be non-empty.
///
/// # Security
/// Creating/updating a binding requires the `federation:register` scope. This
/// object is the trust anchor for cross-tenant reference checks; it is
/// cluster-scoped so it cannot be forged from inside a tenant namespace.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "mesh.hyprstream.io",
    version = "v1alpha1",
    kind = "TenantBinding",
    plural = "tenantbindings",
    singular = "tenantbinding",
    shortname = "hstb",
    status = "TenantBindingStatus",
    category = "hyprstream",
    doc = "Explicit, admin-created namespace ↔ hyprstream-tenant binding (the confused-deputy guard, gated by federation:register). Cluster-scoped so a tenant cannot forge its own binding. When spec.entitlement is set, the operator compiles it into an issuer-signed grant that lands in the tenant's inventory (#929).",
    printcolumn = r#"{"name":"Namespace","type":"string","jsonPath":".spec.namespace"}"#,
    printcolumn = r#"{"name":"Tenant","type":"string","jsonPath":".spec.tenant"}"#,
    printcolumn = r#"{"name":"Bound","type":"boolean","jsonPath":".status.bound"}"#,
    printcolumn = r#"{"name":"Grant","type":"string","jsonPath":".status.grantCid"}"#,
    // `namespace` is a CEL reserved word, so the apiserver requires the field
    // to be referenced with the `__namespace__` escape (same rule as the Go
    // apiserver's `apiserver/schema/cel/model` escaping).
    validation = "self.spec.__namespace__.size() <= 63 && self.spec.__namespace__.matches('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')",
    validation = "self.spec.tenant != ''",
    validation = "has(self.spec.entitlement) ? self.spec.entitlement.amount >= 0 : true",
    validation = "has(self.spec.entitlement) ? self.spec.entitlement.unit.size() > 0 : true",
    cel
)]
#[serde(rename_all = "camelCase")]
pub struct TenantBindingSpec {
    /// The Kubernetes namespace this binding authorizes.
    pub namespace: String,

    /// The hyprstream tenant identifier (e.g. an atproto DID) the namespace is
    /// bound to. When [`Self::entitlement`] is set, this DID is the **holder**
    /// the compiled grant is issued to and the PDS whose inventory receives the
    /// allocation record.
    pub tenant: String,

    /// The resource entitlement an admin authors here and the operator compiles
    /// into an issuer-signed grant (#929).
    ///
    /// `None` (the default) leaves the binding as the pure namespace↔tenant
    /// mapping (the confused-deputy guard only — no grant is compiled, nothing
    /// lands in the tenant's inventory). `Some` makes the operator sign a UCAN
    /// grant as issuer and publish an `ai.hyprstream.ledger.allocation` record
    /// into the tenant's inventory on reconcile.
    ///
    /// This field is the **authoring surface**; it is never itself the
    /// authority. The enforcers verify the compiled grant, not the CRD.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entitlement: Option<TenantEntitlement>,
}

/// The authorable entitlement carried by a [`TenantBindingSpec`].
///
/// This is the k8s-side, dependency-free mirror of the fields the compiler
/// ([`crate::grant`]) lowers into an `ai.hyprstream.ledger.allocation` record.
/// It deliberately re-uses no `hyprstream-pds` / `hyprstream-rpc` types so the
/// CRD schema keeps compiling with no features and no cluster; the compiler
/// validates and maps these into the issuer-signed grant artifacts.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TenantEntitlement {
    /// The issuer-scoped credit unit code (e.g. `"compute-second"`,
    /// `"gpu-hour"`). The unit's issuer is the **operator's issuer DID** (the
    /// key the operator signs the grant with), bound by the compiler — credits
    /// are the unit-issuer's liability (D8-1), so the CRD never names the
    /// issuer: it is always the binding's operator.
    pub unit: String,

    /// Granted amount, in the unit's smallest quantum (`u64`).
    pub amount: u64,

    /// Grant class — the anonymity↔recourse coupling (#928 rule 4):
    /// `prepaid` (bearer-like, lease/prevention-only) or `underwritten`
    /// (issuer-relationship, the only detect-mode-eligible class).
    #[serde(default = "default_grant_class")]
    pub class: TenantGrantClass,

    /// Optional grant expiry (unix seconds). `None` = epoch-bound only (revoked
    /// by bumping the allocation epoch, not by wall-clock expiry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiration: Option<u64>,
}

fn default_grant_class() -> TenantGrantClass {
    TenantGrantClass::Underwritten
}

/// The k8s-side mirror of [`hyprstream_pds::ledger::GrantClass`].
///
/// Serialized as the lexicon string (`"prepaid"` / `"underwritten"`) so the
/// value is identical on the wire whether read from a CRD or an allocation
/// record; the compiler maps it to the PDS type.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum TenantGrantClass {
    /// Bearer-like: paid up front, issuable to a bare `did:key` with no
    /// identity; restricted to lease/prevention spend mode.
    Prepaid,
    /// Postpaid: the issuer has a real bilateral relationship with the holder
    /// and recovers from cheaters itself. The default, and the only
    /// detect-mode-eligible class.
    #[default]
    Underwritten,
}

impl TenantGrantClass {
    /// The lexicon string form (matches `hyprstream_pds::ledger::GrantClass::as_str`).
    pub fn as_str(self) -> &'static str {
        match self {
            TenantGrantClass::Prepaid => "prepaid",
            TenantGrantClass::Underwritten => "underwritten",
        }
    }
}

/// Observed truth for a [`TenantBinding`]. Written only by the operator.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TenantBindingStatus {
    /// Whether the operator has accepted and activated this binding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound: Option<bool>,

    /// Coarse lifecycle phase (`Pending`, `Bound`, `Rejected`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    /// Human-readable detail, e.g. why a binding was rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,

    /// `metadata.generation` last reconciled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,

    /// The CIDv1 of the compiled UCAN grant (the issuer-signed capability the
    /// operator mints from [`TenantBindingSpec::entitlement`]). Enforcers
    /// present and verify *this* CID; the CRD is only the authoring surface.
    /// `None` until the operator has compiled the entitlement (#929).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant_cid: Option<String>,

    /// The CIDv1 of the `ai.hyprstream.ledger.allocation` record landed in the
    /// tenant's inventory. This is the durable inventory line the holder
    /// controls; the grant field inside it references [`Self::grant_cid`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocation_cid: Option<String>,

    /// The issuance epoch of the compiled grant. Revocation = bumping this
    /// counter (stop renewing + reissue at a new epoch); enforcers honor the
    /// epoch, never an unbounded bearer token (#921).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch: Option<u64>,
}
