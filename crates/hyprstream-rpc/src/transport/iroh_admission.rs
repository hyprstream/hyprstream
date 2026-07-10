//! #137 federation admission at the live iroh accept path (#282).
//!
//! The transport-agnostic two-stage gate
//! ([`crate::admission::FederationAdmissionGate`]) needs three inputs per inbound
//! peer: an RFC 6454 **origin**, the peer's authenticated **channel key**
//! ([`crate::admission::PeerChannelKey`]), and the peer's **DID**. On the iroh
//! accept path the channel key is no longer a documented seam (#137 lacked it on
//! the quinn path): an accepted iroh `Connection` is authenticated by the remote
//! endpoint's Ed25519 public key, surfaced as `Connection::remote_id()`. This
//! module turns that `remote_id()` into the gate inputs and defines the
//! object-safe hook the iroh `ProtocolHandler`s call.
//!
//! # What `remote_id()` resolves (and the origin seam)
//!
//! - **Channel key** — `remote_id().as_bytes()` is the peer's Ed25519 public key,
//!   exactly the `PeerChannelKey` the gate's key-binding stage matches. No mTLS
//!   client cert, no RFC 7250 plumbing (#200) needed: iroh's QUIC TLS already
//!   bound the channel to this key.
//! - **DID** — for a raw inbound iroh peer the self-certifying `did:key`
//!   ([`crate::did_web::ed25519_to_did_key`]) of `remote_id()` is the peer's
//!   identity: the key *is* the DID (Tiles interop, #281). The gate's key-binding
//!   stage then trivially admits (self-certifying), so the load-bearing decision
//!   is **stage 1 (origin)**.
//! - **Origin** — this is the residual seam. An RFC 6454 origin for an inbound
//!   iroh peer is *not* carried on the iroh channel itself; it arrives in the
//!   app-layer signed envelope / JWT (`iss`). At the raw accept loop we therefore
//!   only have the `did:key` (no http origin). The default policy-bound impl in
//!   the `hyprstream` crate treats the `did:key` string as the admission subject
//!   (the same shape `did:key` peers register under, #281), and the per-request
//!   app-layer path still re-verifies the envelope signer == `remote_id()`.
//!
//! # Fail-closed
//!
//! When an admission hook is installed and it rejects (or errors), the accept
//! handler MUST drop the connection. When no hook is installed the path is
//! open (pre-#282 behaviour) so existing single-tenant / loopback deployments
//! keep working until an operator opts into federation gating.
//!
//! # D3 — admission never consults pkarr (#895)
//!
//! This gate's only identity input is `Connection::remote_id()` — the
//! channel-bound Ed25519 pubkey iroh's QUIC TLS already verified — plus, for a
//! `did:at9p` peer, the GATE-verified capsule keys (D2 / #894). It **never**
//! reads a pkarr record: pkarr is a liveness/reach hint on the mainline DHT
//! and derives zero authority (see [`crate::transport::iroh_substrate`]'s "D3"
//! note). The reach a pkarr record advertises may be what lets us *dial* a
//! peer, but it plays no part in the *admit/deny* decision — that is the
//! liveness-only invariant D3 pins.

use std::sync::Arc;

/// Object-safe per-connection admission decision for an inbound iroh peer.
///
/// Implemented in the `hyprstream` crate over the #137
/// [`crate::admission::FederationAdmissionGate`] (which owns the `PolicyService`
/// origin stage + the `did:web`/`did:key` key-binding stage). Defined here —
/// where the iroh `ProtocolHandler`s live — so the handlers can invoke it
/// without a `hyprstream-rpc` → `hyprstream` dependency cycle.
///
/// **Fail-closed contract:** return `Ok(())` only when the peer is affirmatively
/// admitted; return `Err(_)` on denial **or** any inability to reach the
/// decision. The caller drops the connection on `Err`.
#[async_trait::async_trait]
pub trait IrohPeerAdmission: Send + Sync {
    /// Decide whether the peer identified by its authenticated Ed25519
    /// `node_id` (the iroh `remote_id()` bytes) may be admitted.
    async fn admit_peer(&self, node_id: &[u8; 32]) -> anyhow::Result<()>;
}

/// A shared, optional admission hook installed on an iroh accept handler.
pub type SharedIrohAdmission = Arc<dyn IrohPeerAdmission>;

/// Run an optional admission hook for an accepted iroh connection, returning
/// `true` if the connection should proceed (admitted, or no hook installed) and
/// `false` if it must be dropped (rejected / fail-closed).
///
/// Centralised so both the RPC (`hyprstream-rpc/1`) and streaming (`moql`)
/// accept paths apply identical semantics.
pub async fn check_admission(hook: Option<&SharedIrohAdmission>, node_id: &[u8; 32]) -> bool {
    match hook {
        None => true,
        Some(gate) => match gate.admit_peer(node_id).await {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    node_id = %short_node_id(node_id),
                    "iroh accept: federation admission rejected — dropping connection: {e}"
                );
                false
            }
        },
    }
}

/// Short hex fingerprint of a node_id for logs (never the full key).
fn short_node_id(node_id: &[u8; 32]) -> String {
    format!("{:02x}{:02x}{:02x}{:02x}…", node_id[0], node_id[1], node_id[2], node_id[3])
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Allow;
    #[async_trait::async_trait]
    impl IrohPeerAdmission for Allow {
        async fn admit_peer(&self, _node_id: &[u8; 32]) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct Deny;
    #[async_trait::async_trait]
    impl IrohPeerAdmission for Deny {
        async fn admit_peer(&self, _node_id: &[u8; 32]) -> anyhow::Result<()> {
            anyhow::bail!("nope")
        }
    }

    struct CountCalls(Arc<AtomicUsize>);
    #[async_trait::async_trait]
    impl IrohPeerAdmission for CountCalls {
        async fn admit_peer(&self, _node_id: &[u8; 32]) -> anyhow::Result<()> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn no_hook_admits_open() {
        assert!(check_admission(None, &[0u8; 32]).await);
    }

    #[tokio::test]
    async fn allow_hook_admits() {
        let hook: SharedIrohAdmission = Arc::new(Allow);
        assert!(check_admission(Some(&hook), &[7u8; 32]).await);
    }

    #[tokio::test]
    async fn deny_hook_rejects_fail_closed() {
        let hook: SharedIrohAdmission = Arc::new(Deny);
        assert!(!check_admission(Some(&hook), &[7u8; 32]).await);
    }

    #[tokio::test]
    async fn hook_is_actually_invoked_with_node_id() {
        let calls = Arc::new(AtomicUsize::new(0));
        let hook: SharedIrohAdmission = Arc::new(CountCalls(Arc::clone(&calls)));
        assert!(check_admission(Some(&hook), &[1u8; 32]).await);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
