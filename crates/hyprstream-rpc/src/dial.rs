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
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, bail, Result};
use parking_lot::RwLock;

use crate::crypto::VerifyingKey;
use crate::rpc_client::{RpcClient, RpcClientImpl};
use crate::transport::in_memory::InMemoryTransport;
use crate::transport::rpc_session::IrohRequestProcessor;
use crate::transport::{EndpointType, TransportConfig};
use crate::transport_traits::Signer;

/// Process-local map of inproc endpoint name → co-located request processor.
type InprocRegistry = RwLock<HashMap<String, Arc<dyn IrohRequestProcessor>>>;

static INPROC_REGISTRY: OnceLock<InprocRegistry> = OnceLock::new();

fn registry() -> &'static InprocRegistry {
    INPROC_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a co-located service's request processor under an `inproc://` name.
///
/// Called at service spawn. The `name` is the endpoint without the scheme
/// (e.g. `hyprstream/registry` for `inproc://hyprstream/registry`). Returns the
/// previous processor if one was registered under the same name.
pub fn register_inproc(
    name: impl Into<String>,
    processor: Arc<dyn IrohRequestProcessor>,
) -> Option<Arc<dyn IrohRequestProcessor>> {
    registry().write().insert(name.into(), processor)
}

/// Remove a co-located service's processor (called at service shutdown).
pub fn unregister_inproc(name: &str) -> Option<Arc<dyn IrohRequestProcessor>> {
    registry().write().remove(name)
}

/// Look up a co-located service's processor by inproc name.
pub fn lookup_inproc(name: &str) -> Option<Arc<dyn IrohRequestProcessor>> {
    registry().read().get(name).cloned()
}

/// Dial a resolved [`TransportConfig`], returning a ready RPC client.
///
/// `server_verifying_key` is the destination's response-verification key
/// (`None` skips response signature verification — only sound when the
/// transport itself authenticates the peer, e.g. pinned TLS). The inproc path
/// is verified end-to-end via the in-process processor's signed responses, so
/// callers may pass the service's verifying key here.
pub fn dial<S>(
    target: &TransportConfig,
    signer: S,
    server_verifying_key: Option<VerifyingKey>,
) -> Result<Arc<dyn RpcClient>>
where
    S: Signer + 'static,
{
    match &target.endpoint {
        EndpointType::Inproc { endpoint } => {
            let processor = lookup_inproc(endpoint).ok_or_else(|| {
                anyhow!("no in-process service registered for inproc endpoint '{endpoint}'")
            })?;
            let transport = InMemoryTransport::new(processor);
            Ok(Arc::new(RpcClientImpl::new(signer, transport, server_verifying_key))
                as Arc<dyn RpcClient>)
        }
        other => bail!(
            "dial(): endpoint {other:?} is not yet served by the dial factory \
             — lazy quinn/iroh/moq transports land in a later #151(a) increment; \
             ZMQ ipc/systemd endpoints stay on the codegen path during the transition"
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::SigningKey;
    use crate::signer::LocalSigner;
    use crate::transport::rpc_session::from_fn;
    use bytes::Bytes;
    use rand::rngs::OsRng;

    fn test_signer() -> LocalSigner {
        LocalSigner::new(SigningKey::generate(&mut OsRng))
    }

    #[test]
    fn register_lookup_unregister() {
        let name = "test/dial/register_lookup_unregister";
        assert!(lookup_inproc(name).is_none());
        let proc = Arc::new(from_fn(|r: Bytes| async move { Ok(r) }));
        assert!(register_inproc(name, proc).is_none());
        assert!(lookup_inproc(name).is_some());
        assert!(unregister_inproc(name).is_some());
        assert!(lookup_inproc(name).is_none());
    }

    #[test]
    fn dial_inproc_resolves_registered_processor() {
        let name = "test/dial/dial_inproc_resolves";
        let proc = Arc::new(from_fn(|r: Bytes| async move { Ok(r) }));
        register_inproc(name, proc);

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
}
