//! Explicit authorization fixtures shared by Hyprstream integration tests.

use std::sync::Arc;

use hyprstream_rpc::auth::mac::{install_mac_dispatch_pep, MacDecision, MacDispatchPep};
use hyprstream_rpc::service::EnvelopeContext;

/// Explicitly opt a non-MAC integration fixture into dispatch.
///
/// The production uninstalled state is deny-at-rest; these tests install a
/// permit PEP only because their assertions concern another subsystem.
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
