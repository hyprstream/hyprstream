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
    commit: Commit,
    commit_bytes: Vec<u8>,
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

        Ok(Self {
            mst_root,
            mst_blocks,
            commit,
            commit_bytes,
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
        self.commit.verify(verifying_key)
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
    }
}
