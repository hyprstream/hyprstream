//! The `dial()` factory: the single place transport choice is made.
//!
//! Today transport selection leaks into generated client code, which hardcodes
//! `ZmqConnection::new(...)`. Per the A1 addressing spike, that decision belongs
//! in exactly one place: [`dial`] takes a resolved [`TransportConfig`] and
//! returns a ready [`Arc<dyn RpcClient>`], erasing the concrete transport behind
//! the object-safe client trait (`Transport` itself is not object-safe — it has
//! `Sub`/`Pub` associated types — so erasure happens at the `RpcClient` layer).
//!
//! # Inproc dial table
//!
//! `inproc://` names resolve through a process-local registry mapping a name to
//! a co-located service's [`IrohRequestProcessor`]. The registry — not the
//! [`TransportConfig`] — holds the live handle: a `TransportConfig` is
//! `Clone + Eq` and wire-publishable (DiscoveryService), so an `Arc<dyn
//! IrohRequestProcessor>` cannot live inside it. Naming (resolver →
//! `TransportConfig`) and handles (registry → processor) stay separate; `dial()`
//! is where they meet.
//!
//! # Construction is synchronous; transports connect lazily
//!
//! `dial()` is sync and does no I/O — the inproc arm only looks up an existing
//! processor. Networked transports (quinn/iroh/moq) connect lazily on first
//! `send()` (cached like `ZmqConnection`), so dialing never blocks and the
//! `inventory`-registered sync factory pattern is preserved. Those arms land in
//! a follow-up increment of #151(a); ZMQ ipc/systemd endpoints stay on the
//! existing codegen path during the transition.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, Weak};

use anyhow::{anyhow, bail, Result};
use parking_lot::RwLock;

use crate::crypto::VerifyingKey;
use crate::rpc_client::{RpcClient, RpcClientImpl};
use crate::transport::in_memory::InMemoryTransport;
use crate::transport::rpc_session::IrohRequestProcessor;
use crate::transport::{EndpointType, TransportConfig};
use crate::transport_traits::Signer;

/// Process-local map of inproc endpoint name → co-located request processor.
///
/// Entries are `Weak`: the registry is a *lookup index*, not the owner. The
/// service spawn site retains the strong `Arc` for the service's lifetime, so
/// dropping the service (e.g. on shutdown) automatically tears down its bridge
/// thread AND leaves a dead `Weak` here that self-evicts on the next lookup —
/// no leak-by-forgotten-unregister, and no strong ref pinning a bridge thread
/// past shutdown.
type InprocRegistry = RwLock<HashMap<String, Weak<dyn IrohRequestProcessor>>>;

static INPROC_REGISTRY: OnceLock<InprocRegistry> = OnceLock::new();

fn registry() -> &'static InprocRegistry {
    INPROC_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a co-located service's request processor under an `inproc://` name.
///
/// Called at service spawn. The `name` is the endpoint without the scheme
/// (e.g. `hyprstream/registry` for `inproc://hyprstream/registry`). The caller
/// MUST retain `processor` (the strong `Arc`) for as long as the service should
/// be dialable — the registry only holds a `Weak`.
///
/// Overwriting a name whose service is still live is almost always a bug
/// (name-squatting / double-spawn); it is logged loudly. Existing dialed
/// clients keep the processor they captured; only future dials see the new one.
pub fn register_inproc(name: impl Into<String>, processor: &Arc<dyn IrohRequestProcessor>) {
    let name = name.into();
    let mut map = registry().write();
    if map.get(&name).is_some_and(|w| w.strong_count() > 0) {
        tracing::warn!(
            endpoint = %name,
            "register_inproc: overwriting a still-live in-process service registration"
        );
    }
    map.insert(name, Arc::downgrade(processor));
}

/// Explicitly drop a name's registration (best-effort; dead entries also
/// self-evict on lookup once the service's strong `Arc` is gone).
pub fn unregister_inproc(name: &str) {
    registry().write().remove(name);
}

/// Look up a co-located service's processor by inproc name, upgrading the
/// `Weak`. A stale (dead-service) entry is pruned in passing.
pub fn lookup_inproc(name: &str) -> Option<Arc<dyn IrohRequestProcessor>> {
    let mut map = registry().write();
    match map.get(name).and_then(Weak::upgrade) {
        Some(arc) => Some(arc),
        None => {
            // Present-but-dead → evict; absent → no-op.
            map.remove(name);
            None
        }
    }
}

/// Dial a resolved [`TransportConfig`], returning a ready RPC client.
///
/// `server_verifying_key` is the destination's response-verification key.
/// `None` does NOT disable signature verification — the response is still
/// cryptographically verified against the key embedded in its envelope; `None`
/// only declines to pin *which* identity that key must be. Passing `None` is
/// only sound when the transport itself authenticates the peer (e.g. pinned
/// TLS / QUIC cert).
///
/// **For `inproc://` there is no transport-level peer authentication** (it is a
/// function call into a registry-resolved processor), so `None` is *discouraged*
/// on the inproc path: without it, a name-squatting registration could be dialed
/// without detection. Callers SHOULD pass the resolved service verifying key for
/// inproc. (The codegen wire-up will thread the resolver-supplied key through;
/// see #151(a) follow-up.)
pub fn dial<S>(
    target: &TransportConfig,
    signer: S,
    server_verifying_key: Option<VerifyingKey>,
) -> Result<Arc<dyn RpcClient>>
where
    S: Signer + 'static,
{
    // Matched exhaustively on purpose: this is the one place transport choice is
    // made, so a newly-added EndpointType variant MUST be a compile error here
    // rather than silently falling through to a runtime bail.
    match &target.endpoint {
        EndpointType::Inproc { endpoint } => {
            let processor = lookup_inproc(endpoint).ok_or_else(|| {
                anyhow!("no in-process service registered for inproc endpoint '{endpoint}'")
            })?;
            let transport = InMemoryTransport::new(processor);
            Ok(Arc::new(RpcClientImpl::new(signer, transport, server_verifying_key))
                as Arc<dyn RpcClient>)
        }
        EndpointType::Quic { addr, server_name, auth } => {
            // SECURITY (#185): QUIC channel auth (WebPKI / cert-hash pin) binds the
            // *channel*, not the peer's DID identity. Identity is established at the
            // application layer — the response signature is verified against the
            // envelope-embedded key even when `server_verifying_key` is `None`,
            // just not *pinned* to an expected identity. For a networked peer,
            // prefer passing the resolver-known key; note (debug) when we can't.
            // (iroh's arm needs no such note — it is identity-bound at the
            // transport, RFC 7250.)
            if server_verifying_key.is_none() {
                tracing::debug!(
                    %addr,
                    "dial: networked QUIC peer with no expected verifying key — \
                     response identity unpinned (channel-level auth only, #185)"
                );
            }
            let transport = crate::transport::lazy_quinn::LazyQuinnTransport::new(
                *addr,
                server_name.clone(),
                auth.clone(),
            );
            Ok(Arc::new(RpcClientImpl::new(signer, transport, server_verifying_key))
                as Arc<dyn RpcClient>)
        }
        EndpointType::Iroh { direct_addrs, relay_url, .. }
            if direct_addrs.is_empty() && relay_url.is_none() =>
        {
            // Fail fast rather than hand iroh an EndpointId with no reachability,
            // which would fall through to discovery and time out (~10-30s). The
            // resolver is expected to supply at least one direct addr or a relay.
            bail!(
                "dial(): iroh endpoint has neither direct addrs nor a relay URL — \
                 not dialable; the resolver must supply reachability"
            )
        }
        EndpointType::Iroh { node_id, direct_addrs, relay_url } => {
            // iroh binds the connection to the peer's EndpointId (its pubkey),
            // so the transport authenticates the peer *identity* — a `None`
            // server_verifying_key is sound (response sig still verified, and
            // the channel is identity-bound, unlike QUIC's cert pin).
            let transport = crate::transport::lazy_iroh::LazyIrohTransport::new(
                *node_id,
                direct_addrs.clone(),
                relay_url.clone(),
            );
            Ok(Arc::new(RpcClientImpl::new(signer, transport, server_verifying_key))
                as Arc<dyn RpcClient>)
        }
        endpoint @ (EndpointType::Ipc { .. } | EndpointType::SystemdFd { .. }) => bail!(
            "dial(): endpoint {endpoint:?} is not served by the dial factory — \
             ZMQ ipc/systemd endpoints stay on the codegen path during the transition; \
             iroh/moq lazy transports land in a later #151(a) increment"
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::SigningKey;
    use crate::transport::QuicServerAuth;
    use crate::signer::LocalSigner;
    use crate::transport::rpc_session::from_fn;
    use bytes::Bytes;
    use rand::rngs::OsRng;

    fn test_signer() -> LocalSigner {
        LocalSigner::new(SigningKey::generate(&mut OsRng))
    }

    fn echo() -> Arc<dyn IrohRequestProcessor> {
        Arc::new(from_fn(|r: Bytes| async move { Ok(r) }))
    }

    #[test]
    fn register_lookup_unregister() {
        let name = "test/dial/register_lookup_unregister";
        assert!(lookup_inproc(name).is_none());
        let proc = echo();
        register_inproc(name, &proc);
        assert!(lookup_inproc(name).is_some());
        unregister_inproc(name);
        assert!(lookup_inproc(name).is_none());
    }

    #[test]
    fn lookup_self_evicts_when_service_dropped() {
        let name = "test/dial/self_evict";
        let proc = echo();
        register_inproc(name, &proc);
        assert!(lookup_inproc(name).is_some());
        // Service shuts down: drop the only strong ref. The Weak in the
        // registry is now dead and must not resolve (and is pruned).
        drop(proc);
        assert!(lookup_inproc(name).is_none(), "dead-service entry must not resolve");
    }

    #[test]
    fn dial_inproc_resolves_registered_processor() {
        let name = "test/dial/dial_inproc_resolves";
        let proc = echo();
        register_inproc(name, &proc);

        let cfg = TransportConfig::inproc(name);
        let client = dial(&cfg, test_signer(), None);
        assert!(client.is_ok(), "dialing a registered inproc endpoint must succeed");

        unregister_inproc(name);
    }

    #[test]
    fn dial_inproc_unregistered_errors() {
        let cfg = TransportConfig::inproc("test/dial/never_registered");
        let err = dial(&cfg, test_signer(), None);
        assert!(err.is_err(), "dialing an unregistered inproc endpoint must error");
    }

    #[test]
    fn dial_unsupported_endpoint_errors() {
        let cfg = TransportConfig::ipc("/tmp/hyprstream-test.sock");
        let err = dial(&cfg, test_signer(), None);
        assert!(err.is_err(), "ipc dial is not yet served by the factory");
    }

    #[test]
    fn dial_quic_webpki_builds_client() {
        // Plain `quic()` = WebPKI/CA validation — a valid auth policy, so dial()
        // builds a (lazy, not-yet-connected) client.
        let cfg = TransportConfig::quic("127.0.0.1:9999".parse().unwrap(), "localhost");
        let client = dial(&cfg, test_signer(), None);
        assert!(client.is_ok(), "WebPKI QUIC dial must build a client without connecting");
    }

    #[test]
    fn dial_quic_pinned_builds_client() {
        // Cert-hash-pinned QUIC: builds a lazy client — sync, no I/O.
        let cfg = TransportConfig::quic_pinned("127.0.0.1:9999".parse().unwrap(), "localhost", [7u8; 32]);
        let client = dial(&cfg, test_signer(), None);
        assert!(client.is_ok(), "pinned QUIC dial must build a client without connecting");
    }

    #[test]
    fn dial_quic_web_pki_pinned_builds_client() {
        // WebPKI + pin (defence in depth): builds a lazy client.
        let cfg = TransportConfig {
            endpoint: EndpointType::Quic {
                addr: "127.0.0.1:9999".parse().unwrap(),
                server_name: "localhost".to_owned(),
                auth: QuicServerAuth::web_pki_pinned(vec![[9u8; 32]]).unwrap(),
            },
            curve: None,
            bind_mode: crate::transport::BindMode::Connect,
        };
        let client = dial(&cfg, test_signer(), None);
        assert!(client.is_ok(), "WebPKI+pin QUIC dial must build a client");
    }

    #[test]
    fn quic_server_auth_rejects_no_requirement() {
        assert!(QuicServerAuth::pinned(vec![]).is_err(), "empty pin set is no auth");
        assert!(QuicServerAuth::web_pki_pinned(vec![]).is_err(), "empty pin set is no auth");
        assert!(QuicServerAuth::web_pki().require_web_pki());
    }
}
