//! OAuth-side adapter for the Policy prepare/commit RPC.
//!
//! `FederateIssuer` owns the browser challenge and verifies both signatures;
//! this adapter relays only its typed verified source/possession evidence to
//! Policy. The OAuth runner installs it in PostgreSQL builds; Policy remains
//! fail-closed unless its complete authority and session-store runtime is ready.
#![allow(dead_code)]

use anyhow::{ensure, Result};
use async_trait::async_trait;
use hyprstream_rpc_std::policy_client::{
    CommitFederateSession, FederateCommittedSession, FederateVerifiedSource, PolicyClient,
    PrepareFederateSession, PreparedFederateSession,
};
use hyprstream_session_store::{Session, Source as StoreSource};

use super::{
    federate_host::{FederateAdmission, PreparedDecision},
    federate_source::{self, AuthorityBinding, Source, VerifiedPossession},
};

/// Only the authenticated Policy client constructed for this OAuth service is
/// passed here by `OAuthService::run`.
pub(crate) struct PolicyFederateAdmission {
    client: PolicyClient,
}

impl PolicyFederateAdmission {
    pub(crate) fn new(client: PolicyClient) -> Self {
        Self { client }
    }
}

fn source_wire(source: StoreSource) -> FederateVerifiedSource {
    FederateVerifiedSource {
        issuer: source.issuer,
        subject: source.subject,
        jti: source.jti,
        nonce: source.nonce,
        token_hash: source.token_hash.to_vec(),
        issued_at: source.issued_at,
        expires_at: source.expires_at,
        atproto_did: source.atproto_did,
    }
}

fn prepared_binding(
    response: PreparedFederateSession,
    requested: &str,
    verified_did: &str,
) -> Result<PreparedDecision> {
    federate_source::scopes(requested)?;
    let requested_vec: Vec<String> = requested.split(' ').map(str::to_owned).collect();
    ensure!(
        response.policy_handle.len() == 32
            && response.subject == verified_did
            && response.requested == requested_vec
            && !response.granted.is_empty()
            && response
                .granted
                .iter()
                .all(|grant| requested_vec.contains(grant)),
        "Policy prepared binding or DID principal mismatch"
    );
    let granted = response.granted.join(" ");
    federate_source::scopes(&granted)?;
    Ok(PreparedDecision {
        binding: AuthorityBinding {
            account: response.account_id,
            subject: response.subject,
            tenant: response.tenant,
            requested: requested.to_owned(),
            granted,
            revision: response.revision,
        },
        policy_handle: response.policy_handle,
    })
}

fn committed_session(
    receipt: FederateCommittedSession,
    expected_subject: &str,
    ed: [u8; 32],
    pq: &[u8],
) -> Result<Session> {
    ensure!(
        receipt.subject == expected_subject,
        "Policy committed principal does not match verified ATProto DID"
    );
    ensure!(
        receipt.ed_public.as_slice() == ed && receipt.pq_public == pq && receipt.proof_epoch > 0,
        "Policy committed signer mismatch"
    );
    Ok(Session {
        host: receipt.host,
        sid: receipt.sid,
        account_id: receipt.account_id,
        subject: receipt.subject,
        tenant: receipt.tenant,
        client_id: receipt.client,
        resource: receipt.resource,
        scopes: receipt.scopes,
        grant_revision: receipt.grant_revision,
        ed_public: receipt.ed_public.as_slice().try_into()?,
        pq_public: receipt.pq_public,
        generation: receipt.generation.as_slice().try_into()?,
        collision_inventory_id: receipt.collision_inventory_id.as_slice().try_into()?,
        proof_epoch: receipt.proof_epoch,
        expires_at: receipt.expires_at,
    })
}

#[async_trait]
impl FederateAdmission for PolicyFederateAdmission {
    async fn prepare(
        &self,
        source: &Source,
        requested: &str,
        challenge_id: [u8; 32],
        created_at: u64,
    ) -> Result<PreparedDecision> {
        federate_source::scopes(requested)?;
        let (ed, pq) = source.proof_public_keys();
        ensure!(pq.len() == 1952, "invalid committed signer suite");
        let store_source = source.store_source();
        let verified_did = store_source.atproto_did.clone();
        let data = PrepareFederateSession {
            source: source_wire(store_source),
            requested: requested.split(' ').map(str::to_owned).collect(),
            challenge_id: challenge_id.to_vec(),
            challenge_created_at: i64::try_from(created_at)?,
            ed_public: ed.to_vec(),
            pq_public: pq.to_vec(),
        };
        prepared_binding(
            self.client.prepare_federate_session(&data).await?,
            requested,
            &verified_did,
        )
    }

    async fn commit(
        &self,
        evidence: VerifiedPossession,
        policy_handle: Vec<u8>,
    ) -> Result<Session> {
        ensure!(policy_handle.len() == 32, "invalid Policy handle");
        let (source, ed, pq, binding, challenge_id, created_at, expires_at) =
            evidence.into_admission_parts();
        ensure!(
            binding.subject == source.atproto_did,
            "Policy challenge subject does not match verified ATProto DID"
        );
        let data = CommitFederateSession {
            source: source_wire(source),
            policy_handle,
            challenge_id: challenge_id.to_vec(),
            challenge_created_at: i64::try_from(created_at)?,
            challenge_expires_at: i64::try_from(expires_at)?,
            ed_public: ed.to_vec(),
            pq_public: pq.clone(),
        };
        let receipt = self.client.commit_federate_session(&data).await?;
        committed_session(receipt, &binding.subject, ed, &pq)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn prepared() -> PreparedFederateSession {
        PreparedFederateSession {
            account_id: "a".into(),
            subject: "did:plc:abcdefghijklmnopqrstuvwx".into(),
            tenant: "tenant".into(),
            requested: vec!["query:registry:test".into()],
            granted: vec!["query:registry:test".into()],
            revision: "revision".into(),
            policy_handle: vec![1; 32],
        }
    }

    #[test]
    fn prepared_binding_rejects_changed_scope_and_handle() {
        let requested = "query:registry:test";
        assert_eq!(
            prepared_binding(prepared(), requested, "did:plc:abcdefghijklmnopqrstuvwx",)
                .unwrap()
                .binding
                .granted,
            requested
        );
        let mut changed = prepared();
        changed.subject = "alice".into();
        assert!(prepared_binding(changed, requested, "did:plc:abcdefghijklmnopqrstuvwx",).is_err());
        let mut changed = prepared();
        changed.granted = vec!["infer:model:other".into()];
        assert!(prepared_binding(changed, requested, "did:plc:abcdefghijklmnopqrstuvwx",).is_err());
        let mut changed = prepared();
        changed.requested = vec!["query:registry:other".into()];
        assert!(prepared_binding(changed, requested, "did:plc:abcdefghijklmnopqrstuvwx",).is_err());
        let mut changed = prepared();
        changed.policy_handle.clear();
        assert!(prepared_binding(changed, requested, "did:plc:abcdefghijklmnopqrstuvwx",).is_err());
    }

    #[test]
    fn committed_receipt_rejects_signer_substitution() {
        let receipt = FederateCommittedSession {
            host: federate_source::HOST.into(),
            sid: "sid".into(),
            account_id: "a".into(),
            subject: "did:plc:abcdefghijklmnopqrstuvwx".into(),
            tenant: "tenant".into(),
            client: federate_source::CLIENT.into(),
            resource: federate_source::HOST.into(),
            scopes: vec!["query:registry:test".into()],
            grant_revision: "revision".into(),
            ed_public: vec![1; 32],
            pq_public: vec![2; 1952],
            generation: vec![3; 32],
            collision_inventory_id: vec![4; 32],
            proof_epoch: 1,
            expires_at: chrono::Utc::now().timestamp() + 60,
        };
        let did = "did:plc:abcdefghijklmnopqrstuvwx";
        assert!(committed_session(receipt.clone(), did, [1; 32], &[2; 1952]).is_ok());
        assert!(committed_session(receipt.clone(), did, [5; 32], &[2; 1952]).is_err());
        assert!(committed_session(receipt.clone(), did, [1; 32], &[6; 1952]).is_err());
        assert!(committed_session(
            receipt,
            "did:plc:bcdefghijklmnopqrstuvwxy2",
            [1; 32],
            &[2; 1952]
        )
        .is_err());
    }
}
