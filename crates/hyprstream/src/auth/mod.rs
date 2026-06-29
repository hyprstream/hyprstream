//! Authorization module for RBAC/ABAC access control
//!
//! Uses Casbin for policy-based access control with policies
//! stored in the `.registry/policies/` directory.
//!
//! Also provides JWT token authentication with Ed25519 signatures.

pub mod device_challenge;
pub mod identity_store;
pub mod key_rotation;
pub mod service_jwt;
pub mod federation;
pub mod federation_admission;
pub mod mesh_trust;
pub mod id_token_verify;
pub mod jwt;
mod policy_manager;
pub mod policy_migration;
pub mod policy_templates;
pub mod user_store;
pub mod rocksdb_store;
#[cfg(feature = "valkey")]
pub mod valkey;

pub use federation::FederationKeyResolver;
pub use jwt::{Claims, JwtError};
pub use key_rotation::{SigningKeyStore, Es256SigningKeyStore, Es256KeySlot, RotationStores};
pub use key_rotation::{MlDsaSigningKeyStore, MlDsaKeySlot};
pub use policy_manager::{PolicyManager, PolicyError, write_policy_file, global_policy_manager, set_global_policy_manager};
pub use policy_migration::migrate_policy_csv;
pub use policy_templates::{PolicyTemplate, ServicePolicyRule, SERVICE_BASE_POLICIES, get_template, get_templates};
pub use user_store::{DeviceRecord, DeviceStore, UserFilter, UserProfile, UserStore, PubkeyEntry, pubkey_fingerprint, decode_pubkey_base64};
pub use rocksdb_store::RocksDbUserStore;
#[cfg(feature = "valkey")]
pub use valkey::ValkeyUserStore;

/// Operation types that can be controlled via policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//
// VARIANT ORDER IS LOAD-BEARING (S3, #569): the first 14 variants are kept 1:1
// (in order) with the `ScopeAction` enum ordinals in
// `crates/hyprstream-rpc/schema/annotations.capnp`. The semantic blocks are:
//   Block A read-class:        Query @0, Subscribe @1
//   Block B write/authority:   Write @2, Create @3, Publish @4, Infer @5,
//                              Train @6, Context @7, Serve @8, Spawn @9
//   Block C admin authority:   Manage @10
//   Block D mesh authority:    MeshInvoke @11, MeshStage @12, MeshDelta @13
// `MeshStatus` is appended AFTER the 14 ScopeAction-mirrored variants: it is NOT a
// distinct ScopeAction (it shares the `query`/`query.status` wire action), so it has
// no ordinal in the schema enum and is excluded from the exact-inverse round-trip.
pub enum Operation {
    // ── Block A: read-class (ScopeAction @0..@1) ──────────────────────────
    /// Query data (q)
    Query,
    /// Subscribe to a stream/notification (b)
    Subscribe,
    // ── Block B: write/authority-class (ScopeAction @2..@9) ───────────────
    /// Write data (w)
    Write,
    /// Create a resource (r)
    Create,
    /// Publish/broadcast to subscribers (l)
    Publish,
    /// Run model inference (i)
    Infer,
    /// Train/fine-tune model (t)
    Train,
    /// Context-augmented generation (c)
    Context,
    /// Serve via API (s)
    Serve,
    /// Spawn a process or task (p)
    Spawn,
    // ── Block C: admin authority (ScopeAction @10) ────────────────────────
    /// Admin operations (m)
    Manage,
    // ─────────────────────────────────────────────────────────────────────
    // Block D: Inference-mesh (#319) — host↔host pipeline RPC abilities.
    // (ScopeAction @11..@13)
    //
    // These gate calls between inference hosts on the cluster mesh (e.g. the
    // router→host activation hand-off and host→aggregator delta submission).
    // They are split read-vs-authority so an operator can grant a *group* of
    // hosts the read ability cheaply while keeping the authority abilities
    // strictly per-host least-privilege (see policy_templates::mesh-host).
    //
    // The dot-strings are designed to map 1:1 onto a future UCAN `cmd`:
    //   Operation::MeshInvoke   ⇄ cmd=/mesh/rpc        (umbrella invoke right)
    //   Operation::MeshStage    ⇄ cmd=/mesh/infer/stage
    //   Operation::MeshDelta    ⇄ cmd=/mesh/delta/submit
    //   Operation::MeshStatus   ⇄ cmd=/mesh/query/status
    // (resource caveats — tenant/job/layers — live in the `mesh://` path).
    /// Umbrella mesh invoke right — `inference:peer-call` / `mesh:rpc` (authority).
    MeshInvoke,
    /// Submit an inference activation/stage to a peer host (authority).
    MeshStage,
    /// Submit a TTT delta to a peer/aggregator host (authority).
    MeshDelta,
    /// Read a peer host's mesh job/pipeline status (read-only).
    MeshStatus,
}

impl Operation {
    /// Get the short code for this operation
    pub fn code(&self) -> char {
        match self {
            Operation::Infer => 'i',
            Operation::Train => 't',
            Operation::Query => 'q',
            Operation::Write => 'w',
            Operation::Serve => 's',
            Operation::Manage => 'm',
            Operation::Context => 'c',
            Operation::Subscribe => 'b',
            Operation::Publish => 'l',
            Operation::Spawn => 'p',
            Operation::Create => 'r',
            // Mesh ops (#319) — distinct codes; not part of the model-capability
            // bitmask (`all()`), so these never collide in `check_all`.
            Operation::MeshInvoke => 'e',
            Operation::MeshStage => 'g',
            Operation::MeshDelta => 'd',
            Operation::MeshStatus => 'u',
        }
    }

    /// The unified scope-action vocabulary (S3, epic #547).
    ///
    /// Returns this operation's `ScopeAction` enumerant name — the SAME token used
    /// by the `$scope`/`$capability` capnp annotation and stored in
    /// [`crate::auth::Scope::action`]. There is one action vocabulary, not two:
    /// the schema annotation, the runtime `Scope`, and this enum all agree.
    ///
    /// This is the canonical bridge between the compile-time TE baseline (schema
    /// annotation) and runtime enforcement. `MeshStatus` shares `query` because the
    /// mesh read ability IS the canonical status read (#319), matching `as_str()`.
    pub fn as_capability(&self) -> &'static str {
        // Arms follow the ScopeAction ordinal blocks (annotations.capnp):
        // A read, B write/authority, C admin, D mesh-authority.
        match self {
            // Block A (@0..@1)
            Operation::Query | Operation::MeshStatus => "query",
            Operation::Subscribe => "subscribe",
            // Block B (@2..@9)
            Operation::Write => "write",
            Operation::Create => "create",
            Operation::Publish => "publish",
            Operation::Infer => "infer",
            Operation::Train => "train",
            Operation::Context => "context",
            Operation::Serve => "serve",
            Operation::Spawn => "spawn",
            // Block C (@10)
            Operation::Manage => "manage",
            // Block D (@11..@13)
            Operation::MeshInvoke => "meshInvoke",
            Operation::MeshStage => "meshStage",
            Operation::MeshDelta => "meshDelta",
        }
    }

    /// Parse an [`Operation`] from a unified scope-action token (a `ScopeAction`
    /// enumerant name — what the `$scope`/`$capability` annotation emits and what
    /// [`crate::auth::Scope::action`] carries). The inverse of [`Self::as_capability`].
    ///
    /// Returns `None` for an unknown token; callers MUST fail closed (no default-allow).
    pub fn from_capability(action: &str) -> Option<Self> {
        // Exact inverse of `as_capability`, arms ordered by ScopeAction ordinal blocks.
        // 1:1 with `ScopeAction` (#547/#569): `subscribe`/`publish` are distinct enforced
        // abilities — NOT collapsed into `context`/`serve`. Collapsing them made
        // `from_capability` non-invertible, so a granted `subscribe:notification:*` /
        // `publish:notification:*` scope was enforced as `context`/`serve` (granted ≠
        // enforced).
        Some(match action {
            // Block A (@0..@1)
            "query" => Operation::Query,
            "subscribe" => Operation::Subscribe,
            // Block B (@2..@9)
            "write" => Operation::Write,
            "create" => Operation::Create,
            "publish" => Operation::Publish,
            "infer" => Operation::Infer,
            "train" => Operation::Train,
            "context" => Operation::Context,
            "serve" => Operation::Serve,
            "spawn" => Operation::Spawn,
            // Block C (@10)
            "manage" => Operation::Manage,
            // Block D (@11..@13)
            "meshInvoke" => Operation::MeshInvoke,
            "meshStage" => Operation::MeshStage,
            "meshDelta" => Operation::MeshDelta,
            _ => return None,
        })
    }

    /// Get the dot-namespaced operation name for policy matching.
    ///
    /// Returns the canonical dot-namespaced string for this operation variant.
    ///
    /// Forwarded verbatim to the Casbin enforcer, which uses `keyMatch` to accept
    /// both this format and legacy flat strings stored in policy files.
    ///
    /// **Note on `Manage`:** This enum variant is a coarse gate over the `ttt.*`
    /// namespace. `as_str()` returns `"ttt.writeback"` as a representative string,
    /// but sub-actions (`ttt.evict`, `ttt.zero`) are all gated by the same `Manage`
    /// variant. When callers need to enforce at sub-action granularity (e.g. in an
    /// RPC handler), they should pass the specific dot-namespaced string directly to
    /// `check_with_domain` rather than going through this method.
    ///
    /// `Operation::Manage.as_str()` (i.e. `"ttt.writeback"`) is intentionally used
    /// only for gates that specifically require the *highest* privilege in the `ttt.*`
    /// namespace, such as issuing tokens on behalf of another subject. Handlers that
    /// only need any `ttt.*` permission (e.g. `ttt.evict`) must pass the specific
    /// action string to avoid incorrectly denying callers who hold a lesser `ttt.*`
    /// permission but not `ttt.writeback`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Operation::Infer   => "infer.generate",
            Operation::Train   => "ttt.train",
            // `Query` (model status) and `MeshStatus` (mesh read ability) are
            // the SAME wire action `query.status` by design — the mesh read
            // right IS the canonical status read (#319). `from_dot_str` resolves
            // the string back to `Query` (its canonical owner).
            Operation::Query | Operation::MeshStatus => "query.status",
            Operation::Write   => "persist.save",
            Operation::Serve   => "serve.api",
            Operation::Manage  => "ttt.writeback",
            Operation::Context => "context.augment",
            Operation::Subscribe => "subscribe",
            Operation::Publish => "publish",
            Operation::Spawn   => "spawn",
            Operation::Create  => "create",
            // Mesh (#319). `mesh.rpc` is the umbrella invoke right (alias
            // `inference:peer-call` accepted on parse). The authority actions
            // `infer.stage` / `delta.submit` are deliberately distinct strings
            // from the model-namespace `infer.generate` / `persist.save` so a
            // model grant can never satisfy a mesh-authority gate. (The read
            // ability `MeshStatus` shares the `query.status` arm above.)
            Operation::MeshInvoke => "mesh.rpc",
            Operation::MeshStage  => "infer.stage",
            Operation::MeshDelta  => "delta.submit",
        }
    }

    /// Parse an operation from a dot-namespaced or legacy flat string.
    ///
    /// Returns `None` for unrecognized strings.  Handles both the new
    /// dot-namespaced format (`"infer.generate"`) and the old flat format
    /// (`"infer"`) for migration compatibility.
    pub fn from_dot_str(s: &str) -> Option<Self> {
        match s {
            // New dot-namespaced and legacy flat strings merged per variant
            "infer.generate" | "infer" => Some(Self::Infer),
            "ttt.train" | "train" => Some(Self::Train),
            "ttt.writeback" | "ttt.evict" | "ttt.zero" | "manage" => Some(Self::Manage),
            "query.status" | "query.delta" | "query" => Some(Self::Query),
            "persist.save" | "persist.export" | "persist.snapshot" | "write" => Some(Self::Write),
            "context.augment" | "context" => Some(Self::Context),
            "serve.api" | "serve" => Some(Self::Serve),
            "subscribe" => Some(Self::Subscribe),
            "publish" => Some(Self::Publish),
            "spawn" => Some(Self::Spawn),
            "create" => Some(Self::Create),
            // Mesh (#319). `query.status` (the mesh read ability) is owned by
            // `Query` above, so it is intentionally NOT repeated here.
            "mesh.rpc" | "mesh:rpc" | "inference:peer-call" => Some(Self::MeshInvoke),
            "infer.stage" => Some(Self::MeshStage),
            "delta.submit" => Some(Self::MeshDelta),
            _ => None,
        }
    }

    /// Parse from short code
    pub fn from_code(c: char) -> Option<Self> {
        match c {
            'i' => Some(Operation::Infer),
            't' => Some(Operation::Train),
            'q' => Some(Operation::Query),
            'w' => Some(Operation::Write),
            's' => Some(Operation::Serve),
            'm' => Some(Operation::Manage),
            'c' => Some(Operation::Context),
            'b' => Some(Operation::Subscribe),
            'l' => Some(Operation::Publish),
            'p' => Some(Operation::Spawn),
            'r' => Some(Operation::Create),
            // Mesh (#319)
            'e' => Some(Operation::MeshInvoke),
            'g' => Some(Operation::MeshStage),
            'd' => Some(Operation::MeshDelta),
            'u' => Some(Operation::MeshStatus),
            _ => None,
        }
    }

    /// All operations
    pub fn all() -> &'static [Operation] {
        &[
            Operation::Infer,
            Operation::Train,
            Operation::Query,
            Operation::Write,
            Operation::Serve,
            Operation::Manage,
            Operation::Context,
        ]
    }
}

impl std::str::FromStr for Operation {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_dot_str(s)
            .ok_or_else(|| anyhow::anyhow!("Unknown operation: {}", s))
    }
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Convert capabilities to an access string based on policy
///
/// Returns a comma-separated list of permitted operations (e.g., "infer,train,serve").
/// Only includes operations that both the resource supports AND the user is permitted.
pub fn capabilities_to_access_string(
    policy_manager: &PolicyManager,
    user: &str,
    resource: &str,
    capabilities: &crate::archetypes::capabilities::CapabilitySet,
) -> String {
    use crate::archetypes::capabilities::{Context, Infer, Manage, Query, Serve, Train, Write};

    let mut permitted = Vec::new();

    // Only show access for capabilities the resource actually has AND user is permitted
    if capabilities.has::<Infer>() && policy_manager.check_sync(user, resource, Operation::Infer) {
        permitted.push("infer");
    }
    if capabilities.has::<Train>() && policy_manager.check_sync(user, resource, Operation::Train) {
        permitted.push("train");
    }
    if capabilities.has::<Query>() && policy_manager.check_sync(user, resource, Operation::Query) {
        permitted.push("query");
    }
    if capabilities.has::<Write>() && policy_manager.check_sync(user, resource, Operation::Write) {
        permitted.push("write");
    }
    if capabilities.has::<Serve>() && policy_manager.check_sync(user, resource, Operation::Serve) {
        permitted.push("serve");
    }
    if capabilities.has::<Manage>()
        && policy_manager.check_sync(user, resource, Operation::Manage)
    {
        permitted.push("manage");
    }
    if capabilities.has::<Context>()
        && policy_manager.check_sync(user, resource, Operation::Context)
    {
        permitted.push("context");
    }

    permitted.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_operation_codes() {
        assert_eq!(Operation::Infer.code(), 'i');
        assert_eq!(Operation::Train.code(), 't');
        assert_eq!(Operation::Query.code(), 'q');
        assert_eq!(Operation::Write.code(), 'w');
        assert_eq!(Operation::Serve.code(), 's');
        assert_eq!(Operation::Manage.code(), 'm');
        assert_eq!(Operation::Context.code(), 'c');
    }

    #[test]
    fn test_operation_from_code() {
        assert_eq!(Operation::from_code('i'), Some(Operation::Infer));
        assert_eq!(Operation::from_code('t'), Some(Operation::Train));
        assert_eq!(Operation::from_code('c'), Some(Operation::Context));
        assert_eq!(Operation::from_code('x'), None);
    }

    #[test]
    fn test_operation_dot_namespaced_strings() {
        assert_eq!(Operation::Infer.as_str(), "infer.generate");
        assert_eq!(Operation::Train.as_str(), "ttt.train");
        assert_eq!(Operation::Query.as_str(), "query.status");
        assert_eq!(Operation::Write.as_str(), "persist.save");
        assert_eq!(Operation::Manage.as_str(), "ttt.writeback");
        assert_eq!(Operation::Context.as_str(), "context.augment");
        assert_eq!(Operation::Serve.as_str(), "serve.api");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_operation_from_str() {
        // Legacy flat strings still parse correctly
        assert!(matches!(Operation::from_str("infer"), Ok(Operation::Infer)));
        assert!(matches!(Operation::from_str("train"), Ok(Operation::Train)));
        assert!(matches!(Operation::from_str("query"), Ok(Operation::Query)));
        assert!(matches!(Operation::from_str("write"), Ok(Operation::Write)));
        assert!(matches!(Operation::from_str("serve"), Ok(Operation::Serve)));
        assert!(matches!(Operation::from_str("manage"), Ok(Operation::Manage)));
        assert!(matches!(Operation::from_str("context"), Ok(Operation::Context)));
        // Dot-namespaced strings (emitted by Display) must also round-trip
        assert!(matches!("infer.generate".parse::<Operation>(), Ok(Operation::Infer)));
        assert!(matches!("ttt.train".parse::<Operation>(), Ok(Operation::Train)));
        assert!(matches!("ttt.writeback".parse::<Operation>(), Ok(Operation::Manage)));
        assert!(matches!("query.status".parse::<Operation>(), Ok(Operation::Query)));
        assert!(matches!("persist.save".parse::<Operation>(), Ok(Operation::Write)));
        assert!(matches!("serve.api".parse::<Operation>(), Ok(Operation::Serve)));
        assert!(matches!("context.augment".parse::<Operation>(), Ok(Operation::Context)));
        // Verify format!("{}", op).parse() round-trip for every variant
        for op in Operation::all() {
            let displayed = format!("{}", op);
            let parsed: Operation = displayed.parse().expect("round-trip parse failed");
            assert_eq!(&parsed, op);
        }
        assert!(Operation::from_str("foo").is_err());
    }

    #[test]
    fn test_mesh_operation_vocab() {
        // Authority + umbrella mesh actions round-trip via their distinct strings.
        assert_eq!(Operation::MeshInvoke.as_str(), "mesh.rpc");
        assert_eq!(Operation::MeshStage.as_str(), "infer.stage");
        assert_eq!(Operation::MeshDelta.as_str(), "delta.submit");
        assert_eq!(Operation::from_dot_str("mesh.rpc"), Some(Operation::MeshInvoke));
        assert_eq!(Operation::from_dot_str("mesh:rpc"), Some(Operation::MeshInvoke));
        assert_eq!(Operation::from_dot_str("inference:peer-call"), Some(Operation::MeshInvoke));
        assert_eq!(Operation::from_dot_str("infer.stage"), Some(Operation::MeshStage));
        assert_eq!(Operation::from_dot_str("delta.submit"), Some(Operation::MeshDelta));
        // The mesh-authority strings are DISTINCT from the model-namespace
        // strings, so a model grant can never satisfy a mesh-authority gate.
        assert_ne!(Operation::MeshStage.as_str(), Operation::Infer.as_str());
        assert_ne!(Operation::MeshDelta.as_str(), Operation::Write.as_str());
        // The mesh read ability reuses the canonical `query.status` wire action.
        assert_eq!(Operation::MeshStatus.as_str(), "query.status");
        assert_eq!(Operation::from_dot_str("query.status"), Some(Operation::Query));
        // Codes are distinct from the model-capability codes.
        for op in [Operation::MeshInvoke, Operation::MeshStage, Operation::MeshDelta, Operation::MeshStatus] {
            assert_eq!(Operation::from_code(op.code()), Some(op));
        }
        // Mesh ops are NOT part of the model-capability set.
        for op in &[Operation::MeshInvoke, Operation::MeshStage, Operation::MeshDelta, Operation::MeshStatus] {
            assert!(!Operation::all().contains(op));
        }
    }

    #[test]
    fn test_capability_token_is_exact_inverse() {
        // `as_capability`/`from_capability` are the canonical scope-action bridge and
        // MUST be exact inverses (1:1 with `ScopeAction`). Regression guard for the
        // grant/enforcement divergence (#569 peer review): `subscribe`/`publish` were
        // collapsed into `context`/`serve`, so a granted `subscribe`/`publish` scope was
        // enforced as a different ability (granted ≠ enforced).
        let ops = [
            Operation::Query,
            Operation::Write,
            Operation::Manage,
            Operation::Infer,
            Operation::Train,
            Operation::Serve,
            Operation::Context,
            Operation::Subscribe,
            Operation::Publish,
            Operation::Spawn,
            Operation::Create,
            Operation::MeshInvoke,
            Operation::MeshStage,
            Operation::MeshDelta,
        ];
        for op in ops {
            assert_eq!(
                Operation::from_capability(op.as_capability()),
                Some(op),
                "capability token round-trip is not 1:1 for {op:?}",
            );
        }
        // `subscribe`/`publish` are distinct enforced abilities — NOT context/serve.
        assert_eq!(Operation::from_capability("subscribe"), Some(Operation::Subscribe));
        assert_eq!(Operation::from_capability("publish"), Some(Operation::Publish));
        assert_eq!(Operation::Subscribe.as_capability(), "subscribe");
        assert_eq!(Operation::Publish.as_capability(), "publish");
        // Unknown tokens fail closed.
        assert_eq!(Operation::from_capability("nope"), None);
    }

    #[test]
    fn test_operation_all() {
        let all = Operation::all();
        assert_eq!(all.len(), 7);
        assert!(all.contains(&Operation::Context));
    }

    #[test]
    fn test_operation_from_dot_str() {
        // New dot-namespaced strings
        assert_eq!(Operation::from_dot_str("infer.generate"), Some(Operation::Infer));
        assert_eq!(Operation::from_dot_str("ttt.train"), Some(Operation::Train));
        assert_eq!(Operation::from_dot_str("ttt.writeback"), Some(Operation::Manage));
        assert_eq!(Operation::from_dot_str("ttt.evict"), Some(Operation::Manage));
        assert_eq!(Operation::from_dot_str("ttt.zero"), Some(Operation::Manage));
        assert_eq!(Operation::from_dot_str("query.status"), Some(Operation::Query));
        assert_eq!(Operation::from_dot_str("query.delta"), Some(Operation::Query));
        assert_eq!(Operation::from_dot_str("persist.save"), Some(Operation::Write));
        assert_eq!(Operation::from_dot_str("persist.export"), Some(Operation::Write));
        assert_eq!(Operation::from_dot_str("persist.snapshot"), Some(Operation::Write));
        assert_eq!(Operation::from_dot_str("context.augment"), Some(Operation::Context));
        assert_eq!(Operation::from_dot_str("serve.api"), Some(Operation::Serve));
        // Legacy flat strings
        assert_eq!(Operation::from_dot_str("infer"), Some(Operation::Infer));
        assert_eq!(Operation::from_dot_str("train"), Some(Operation::Train));
        assert_eq!(Operation::from_dot_str("query"), Some(Operation::Query));
        assert_eq!(Operation::from_dot_str("write"), Some(Operation::Write));
        assert_eq!(Operation::from_dot_str("serve"), Some(Operation::Serve));
        assert_eq!(Operation::from_dot_str("manage"), Some(Operation::Manage));
        assert_eq!(Operation::from_dot_str("context"), Some(Operation::Context));
        // Unknown
        assert_eq!(Operation::from_dot_str("unknown"), None);
        assert_eq!(Operation::from_dot_str(""), None);
    }
}
