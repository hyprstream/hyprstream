//! Schema-owned RPC labels for verified user and service identities.
//!
//! The same validated leaf inventory drives the signature policy and this
//! mandatory label gate. No service-name, parent-path, or transitional-label
//! fallback is allowed. Scope, tenant and mutation checks remain downstream.

use anyhow::{ensure, Result};
use hyprstream_rpc::auth::mac::{MacDecision, MacDenyReason, MacDispatchPep};
use hyprstream_rpc::proof::policy::{
    collect_generated_rows, validate_generated_rows, AuthenticationRequirement,
    GeneratedMethodPolicyRow,
};
use hyprstream_rpc::service::EnvelopeContext;
use std::collections::BTreeMap;

pub(super) struct GeneratedDispatchPep {
    rows: BTreeMap<&'static str, BTreeMap<&'static [u16], GeneratedMethodPolicyRow>>,
}

impl GeneratedDispatchPep {
    pub(super) fn production() -> Result<Self> {
        Self::from_rows(collect_generated_rows()?)
    }

    pub(super) fn is_declared(&self, service: &str, method: Option<&[u16]>) -> bool {
        method.is_some_and(|path| {
            self.rows
                .get(service)
                .is_some_and(|rows| rows.contains_key(path))
        })
    }

    fn from_rows(rows: Vec<GeneratedMethodPolicyRow>) -> Result<Self> {
        ensure!(
            !rows.is_empty(),
            "no generated RPC dispatch labels are linked"
        );
        validate_generated_rows(&rows)?;
        let mut inventory: BTreeMap<_, BTreeMap<_, _>> = BTreeMap::new();
        for row in rows {
            inventory
                .entry(row.service)
                .or_default()
                .insert(row.leaf_path, row);
        }
        Ok(Self { rows: inventory })
    }
}

impl MacDispatchPep for GeneratedDispatchPep {
    fn check(
        &self,
        ctx: &EnvelopeContext,
        service_domain: &str,
        method: Option<&[u16]>,
    ) -> MacDecision {
        hyprstream_rpc::auth::mac::remember_verified_subject(ctx);
        let Some(row) = method.and_then(|path| self.rows.get(service_domain)?.get(path)) else {
            return MacDecision::Deny(MacDenyReason::UnlabeledObject);
        };
        // Hybrid enrollment authenticates a workload; it is not permission to
        // mint credentials or administer sessions for other principals. Keep the issuing-service
        // boundary explicit even if a retained policy has the old service:*
        // IssueToken grant. Human callers still require downstream policy.
        if service_domain == "policy"
            && matches!(row.symbolic_path,
                "issueToken" | "registerSession" | "revokeSession"
                    | "revokeCredential" | "exchangeDelegated" | "exchangeWit")
        {
            let subject = ctx.subject();
            if subject.name().is_some_and(|name| name.starts_with("service:"))
                && (subject.is_federated()
                    || !matches!(subject.name(), Some("service:oauth" | "service:policy")))
            {
                return MacDecision::Deny(MacDenyReason::NoClearance);
            }
        }
        if row.authentication == AuthenticationRequirement::UnauthenticatedAllowed {
            // Validation requires exactly SYSTEM_LOW, an explicit public reason
            // and an unauthenticated-capable signature policy. The dispatch
            // pipeline still verifies the declared signature policy first.
            return MacDecision::Permit;
        }
        let Some(selected) = hyprstream_rpc::auth::mac::global_mac_activation_control()
            .select_context(ctx.security_context())
        else {
            return MacDecision::Deny(MacDenyReason::NoClearance);
        };
        if selected.can_access(&row.target_label) {
            MacDecision::Permit
        } else {
            MacDecision::Deny(MacDenyReason::FloorDeny)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_inventory_is_rejected_before_installation() -> Result<()> {
        assert!(GeneratedDispatchPep::from_rows(Vec::new()).is_err());
        let rows = collect_generated_rows()?;
        let row = rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing generated inventory"))?
            .clone();
        assert!(GeneratedDispatchPep::from_rows(vec![row.clone(), row.clone()]).is_err());
        let mut empty_path = row;
        empty_path.leaf_path = &[];
        assert!(GeneratedDispatchPep::from_rows(vec![empty_path]).is_err());
        let mut public = rows
            .into_iter()
            .find(|row| row.authentication == AuthenticationRequirement::UnauthenticatedAllowed)
            .ok_or_else(|| anyhow::anyhow!("missing public schema declaration"))?;
        public.public_reason = None;
        assert!(GeneratedDispatchPep::from_rows(vec![public]).is_err());
        Ok(())
    }

    #[test]
    fn public_declarations_do_not_create_anonymous_fallbacks() -> Result<()> {
        let pep = GeneratedDispatchPep::production()?;
        let ctx = EnvelopeContext::from_callback_service(0, "untrusted");
        for rows in pep.rows.values() {
            for row in rows.values() {
                let expected =
                    if row.authentication == AuthenticationRequirement::UnauthenticatedAllowed {
                        MacDecision::Permit
                    } else {
                        MacDecision::Deny(MacDenyReason::NoClearance)
                    };
                assert_eq!(
                    pep.check(&ctx, row.service, Some(row.leaf_path)),
                    expected,
                    "{}:{}",
                    row.service,
                    row.symbolic_path
                );
            }
        }
        for path in [
            None,
            Some(&[][..]),
            Some(&[u16::MAX][..]),
            Some(&[0, 0][..]),
        ] {
            assert_eq!(
                pep.check(&ctx, "policy", path),
                MacDecision::Deny(MacDenyReason::UnlabeledObject)
            );
        }
        assert_eq!(
            pep.check(&ctx, "/registry", Some(&[0])),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );
        Ok(())
    }
}
