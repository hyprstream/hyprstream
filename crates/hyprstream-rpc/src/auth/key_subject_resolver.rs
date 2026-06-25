//! Global signer-key → authorization-subject resolver (#446).
//!
//! # Why this exists
//!
//! The shared dispatch core ([`crate::service::dispatch::process_request`])
//! verifies every socket-served request — including same-uid local IPC over
//! Unix domain sockets — under [`crate::envelope::EnvelopeVerification::AnySigner`]
//! (`iroh_rpc.rs::run_bridge_dispatch_loop` hardcodes `AnySigner`, used by both
//! the iroh network plane and `uds_server`). `AnySigner` cryptographically
//! verifies the COSE composite against the envelope's `cnf` (the signer's
//! Ed25519 pubkey) but produces `key_derived_subject = anonymous` — so a
//! signed IPC request from one service to another loses its identity and is
//! denied any policy-gated write (e.g. `discovery:Announce`).
//!
//! The authoritative key→subject binding lives in the trust store
//! (`hyprstream-service::TrustStore`, populated fail-closed under #441), but
//! that crate sits *above* `hyprstream-rpc` in the dependency graph and cannot
//! be referenced from the dispatch core. This module is the inversion-of-control
//! seam: `hyprstream-service` installs a [`KeySubjectResolver`] at startup, and
//! the default [`crate::service::RequestService::resolve_key_subject`] consults
//! it. That makes *every* service (DiscoveryService included — it lives in
//! `hyprstream-discovery`, which also cannot see the trust store) resolve a
//! verified signer key to its authoritative `service:<name>` subject without
//! each service crate having to depend on the trust store.
//!
//! # Fail-closed invariant
//!
//! Resolution NEVER fabricates an identity: an unregistered signer key resolves
//! to `None` → `anonymous`. A genuinely anonymous caller (no registered key)
//! therefore stays denied for policy-gated writes. This only honors the
//! cryptographically-verified, *registered* identity of a legitimate signer —
//! it never loosens a grant to `anonymous`.

use std::sync::Arc;

use parking_lot::RwLock;

use crate::envelope::Subject;

/// Resolves a verified Ed25519 signer pubkey to its authorization subject.
///
/// Implemented by the trust store layer (`hyprstream-service`) and installed
/// via [`set_global`]. The input is the envelope `cnf` — the signer's Ed25519
/// public key, already cryptographically verified by the COSE signature check
/// before resolution is attempted.
pub trait KeySubjectResolver: Send + Sync {
    /// Resolve a verified signer pubkey to its authoritative subject.
    ///
    /// Returns `Some(subject)` only for a *registered* key (e.g.
    /// `service:discovery`, or a user's bare `sub`). Returns `None` for an
    /// unregistered/unknown key — the caller then stays `anonymous`
    /// (fail-closed). Implementations MUST NOT derive a fallback identity.
    fn resolve_subject(&self, signer_pubkey: &[u8; 32]) -> Option<Subject>;
}

static GLOBAL_KEY_SUBJECT_RESOLVER: RwLock<Option<Arc<dyn KeySubjectResolver>>> = RwLock::new(None);

/// Install the global key→subject resolver.
///
/// Called once during startup by the trust-store layer. Idempotent —
/// re-installing replaces the previous resolver.
pub fn set_global(resolver: Arc<dyn KeySubjectResolver>) {
    *GLOBAL_KEY_SUBJECT_RESOLVER.write() = Some(resolver);
}

/// Resolve a signer key to a subject via the installed global resolver.
///
/// Returns `None` if no resolver is installed (e.g. WASM client, tests that
/// don't bootstrap the trust store) or the key is unregistered — both yield
/// `anonymous`, preserving the fail-closed default.
pub fn resolve_subject(signer_pubkey: &[u8; 32]) -> Option<Subject> {
    GLOBAL_KEY_SUBJECT_RESOLVER
        .read()
        .as_ref()
        .and_then(|r| r.resolve_subject(signer_pubkey))
}
