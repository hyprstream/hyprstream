//! Initial atproto repository state for a hosted PDS account.
//!
//! A hosted account starts with a real, signed repo head even before it has
//! application records. The empty MST root and ES256 commit are immutable
//! blocks; the commit CID is bound into operation zero by
//! [`HostedAccountMint::prepare_pds_genesis`](crate::HostedAccountMint::prepare_pds_genesis).

use anyhow::{ensure, Result};
use p256::ecdsa::{SigningKey, VerifyingKey};

use crate::cid::Cid;
use crate::commit::{Commit, UnsignedCommit};
use crate::mst::Node;
use crate::tid::Tid;

/// A verified empty-MST root and signed initial repo commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostedRepoGenesis {
    mst_root: Cid,
    mst_blocks: Vec<(Cid, Vec<u8>)>,
    /// Native commit retained for the Hyprstream repository and DID-log
    /// binding. Its signature covers the native DAG-CBOR ordering.
    commit: Commit,
    commit_bytes: Vec<u8>,
    /// Canonical AT Protocol representation of the same empty repository
    /// state. This is an explicit dual-format bridge; it is never substituted
    /// for the native commit or its DID-log binding.
    public_commit: Commit,
    public_commit_bytes: Vec<u8>,
}

impl HostedRepoGenesis {
    pub(crate) fn seal(did: &str, signing_key: &SigningKey, rev: Tid) -> Result<Self> {
        let tree = Node::empty();
        let mst_root = tree.root_cid();
        let mst_blocks = tree
            .all_blocks()
            .into_iter()
            .map(|(cid, block)| (cid, block.encode()))
            .collect::<Vec<_>>();
        ensure!(
            mst_blocks.iter().any(|(cid, _)| *cid == mst_root),
            "initial hosted repo omits its MST root block"
        );

        let unsigned = UnsignedCommit::new(did, mst_root, rev, None);
        let commit = Commit::sign(&unsigned, signing_key);
        commit.verify(signing_key.verifying_key())?;
        ensure!(
            commit.did == did && commit.data == mst_root && commit.prev.is_none(),
            "initial hosted repo commit does not describe the requested empty repo"
        );
        let commit_bytes = commit.to_dag_cbor();
        let public_commit = Commit::sign_atproto(&unsigned, signing_key)?;
        public_commit.verify_atproto(signing_key.verifying_key())?;
        let public_commit_bytes = public_commit.to_atproto_dag_cbor()?;

        Ok(Self {
            mst_root,
            mst_blocks,
            commit,
            commit_bytes,
            public_commit,
            public_commit_bytes,
        })
    }

    /// CID of the canonical empty MST root block.
    #[must_use]
    pub fn mst_root(&self) -> Cid {
        self.mst_root
    }

    /// Every MST block needed to materialize the initial repository.
    #[must_use]
    pub fn mst_blocks(&self) -> &[(Cid, Vec<u8>)] {
        &self.mst_blocks
    }

    /// Signed ES256 initial commit.
    #[must_use]
    pub fn commit(&self) -> &Commit {
        &self.commit
    }

    /// Canonical DAG-CBOR bytes of the signed initial commit.
    #[must_use]
    pub fn commit_bytes(&self) -> &[u8] {
        &self.commit_bytes
    }

    /// CID of the signed initial commit bound into the genesis DID operation.
    #[must_use]
    pub fn commit_cid(&self) -> Cid {
        self.commit.cid()
    }

    /// Canonical AT Protocol commit for the same empty repository state.
    #[must_use]
    pub fn public_commit(&self) -> &Commit {
        &self.public_commit
    }

    /// Canonical AT Protocol bytes for the dual-format public genesis.
    #[must_use]
    pub fn public_commit_bytes(&self) -> &[u8] {
        &self.public_commit_bytes
    }

    /// CID of the canonical public genesis. The native CID remains the DID
    /// operation's bound head; callers must not conflate the two.
    pub fn public_commit_cid(&self) -> Result<Cid> {
        self.public_commit.cid_atproto()
    }

    /// Re-verify the stored block linkage and commit signature.
    pub fn verify(&self, verifying_key: &VerifyingKey) -> Result<()> {
        ensure!(
            self.mst_blocks
                .iter()
                .any(|(cid, bytes)| *cid == self.mst_root && Cid::from_dag_cbor(bytes) == *cid),
            "initial hosted repo MST root block is missing or corrupt"
        );
        ensure!(
            self.commit.data == self.mst_root && self.commit.prev.is_none(),
            "initial hosted repo commit linkage is invalid"
        );
        ensure!(
            self.commit.to_dag_cbor() == self.commit_bytes,
            "initial hosted repo commit bytes are not canonical"
        );
        self.commit.verify(verifying_key)?;
        ensure!(
            self.public_commit.did == self.commit.did
                && self.public_commit.data == self.commit.data
                && self.public_commit.rev == self.commit.rev
                && self.public_commit.prev == self.commit.prev
                && self.public_commit.to_atproto_dag_cbor()? == self.public_commit_bytes,
            "dual-format public genesis does not match native repository state"
        );
        self.public_commit.verify_atproto(verifying_key)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use p256::ecdsa::SigningKey;

    use super::*;

    #[test]
    fn empty_repo_genesis_is_signed_and_contains_its_root_block() {
        let key = SigningKey::from_slice(&[19; 32]).unwrap();
        let repo = HostedRepoGenesis::seal(
            "did:web:alice.accounts.example",
            &key,
            Tid::from_micros(7, 1),
        )
        .unwrap();

        repo.verify(key.verifying_key()).unwrap();
        assert_eq!(repo.commit().did, "did:web:alice.accounts.example");
        assert_eq!(repo.commit().data, repo.mst_root());
        assert_eq!(repo.mst_blocks().len(), 1);
        assert_eq!(repo.commit_cid(), Cid::from_dag_cbor(repo.commit_bytes()));
        assert_eq!(
            repo.public_commit_cid().unwrap(),
            Cid::from_dag_cbor(repo.public_commit_bytes())
        );
        repo.verify(key.verifying_key()).unwrap();
    }
}
