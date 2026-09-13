//! Explicit authorization fixtures shared by RPC integration tests.

use std::sync::Arc;

use hyprstream_rpc::auth::mac::{install_mac_dispatch_pep, MacDecision, MacDispatchPep};
use hyprstream_rpc::service::EnvelopeContext;

/// Install an explicit permit PEP for tests whose subject is transport or
/// envelope mechanics rather than MAC policy.
///
/// Production's uninstalled state remains deny-at-rest. Each integration-test
/// process must opt into this fixture before it expects handler dispatch.
pub fn install_explicit_dispatch_pep() {
    struct ExplicitFixturePep;

    impl MacDispatchPep for ExplicitFixturePep {
        fn check(
            &self,
            _ctx: &EnvelopeContext,
            _service_domain: &str,
            _method: Option<&[u16]>,
        ) -> MacDecision {
            MacDecision::Permit
        }
    }

    install_mac_dispatch_pep(Arc::new(ExplicitFixturePep));
}
