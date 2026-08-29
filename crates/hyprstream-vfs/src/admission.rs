//! Trusted effective-namespace composition for sandbox admission.
//!
//! This module constructs the *actual* forkable [`Namespace`] and its
//! canonical commitment together. It deliberately accepts explicit mount
//! capability handles plus immutable policy-issued assertions; a path, qid,
//! or [`Subject`](hyprstream_rpc::Subject) can neither manufacture an identity
//! nor grant access by being named here. The public constructors provide type
//! discipline, not cryptographic verification: the surrounding admission
//! boundary remains responsible for authenticating those assertions.
//! Composition starts from [`Namespace::new`], so this bounded seam selects
//! capabilities but does not itself install a [`NamespacePep`](crate::NamespacePep)
//! or bind object labels. Authenticated policy assertion, PEP installation, and
//! label binding remain production follow-up work.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{BindFlag, MountTarget, Namespace, NamespaceError, NamespaceMountTopology};

/// Immutable policy commitment associated with one effective namespace.
///
/// This is a trusted policy assertion rather than a capability or independently
/// authenticated proof. It is committed alongside mount topology so a digest
/// never means topology alone.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NamespacePolicyCommitment(Arc<str>);

impl NamespacePolicyCommitment {
    /// Accept a stable, nonempty policy assertion at the trusted boundary.
    pub fn from_trusted_policy(
        value: impl Into<Arc<str>>,
    ) -> Result<Self, NamespaceAdmissionError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(NamespaceAdmissionError::MissingPolicyCommitment);
        }
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable identity/commitment assertion for a mount capability selected by
/// trusted policy. It is not inferred from a path, qid, caller subject, or a
/// trait object's address, but this type does not authenticate it by itself.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MountIdentity(Arc<str>);

impl MountIdentity {
    /// Accept a stable, nonempty mount identity assertion at the trusted
    /// boundary.
    pub fn from_trusted_policy(
        value: impl Into<Arc<str>>,
    ) -> Result<Self, NamespaceAdmissionError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(NamespaceAdmissionError::MissingMountIdentity);
        }
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// One explicitly identified mount operation in a namespace admission.
#[derive(Clone)]
pub struct AdmittedMount {
    prefix: String,
    target: MountTarget,
    bind: BindFlag,
    identity: MountIdentity,
}

impl AdmittedMount {
    /// Pair a mount capability with the immutable policy identity assertion
    /// that selected it. `None` fails closed; callers must never substitute a
    /// path/qid or Subject-derived string for this assertion.
    pub fn new(
        prefix: impl Into<String>,
        target: MountTarget,
        bind: BindFlag,
        identity: Option<MountIdentity>,
    ) -> Result<Self, NamespaceAdmissionError> {
        let identity = identity.ok_or(NamespaceAdmissionError::MissingMountIdentity)?;
        Ok(Self {
            prefix: prefix.into(),
            target,
            bind,
            identity,
        })
    }
}

/// Failure constructing an admitted effective namespace.
#[derive(Debug)]
pub enum NamespaceAdmissionError {
    MissingPolicyCommitment,
    MissingMountIdentity,
    EmptyMountTable,
    Namespace(NamespaceError),
}

impl std::fmt::Display for NamespaceAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingPolicyCommitment => {
                f.write_str("effective namespace requires a policy commitment")
            }
            Self::MissingMountIdentity => {
                f.write_str("effective namespace requires every mount's policy identity")
            }
            Self::EmptyMountTable => {
                f.write_str("effective namespace requires at least one identified mount")
            }
            Self::Namespace(error) => write!(f, "namespace composition failed: {error}"),
        }
    }
}

impl std::error::Error for NamespaceAdmissionError {}

impl From<NamespaceError> for NamespaceAdmissionError {
    fn from(value: NamespaceError) -> Self {
        Self::Namespace(value)
    }
}

/// An admitted, frozen namespace and the deterministic commitment derived from
/// its final effective mount table.
///
/// The namespace stays private and can only be supplied to a sandbox as a
/// [`fork`](Self::fork). Callers cannot replace individual mounts after the
/// digest has been derived.
pub struct AdmittedNamespace {
    namespace: Namespace,
    canonical_manifest: Vec<u8>,
    digest: [u8; 32],
}

/// Identity-only mirror of the namespace's final mount table. It lets the
/// admission manifest commit to actual effective union behavior without ever
/// exposing mount capabilities or preserving overwritten construction history.
struct IdentifiedMountEntry {
    targets: Vec<MountIdentity>,
    upper_target_index: Option<usize>,
}

impl AdmittedNamespace {
    /// Construct the actual namespace and its canonical manifest in one
    /// operation. The manifest serializes the final effective table sorted by
    /// normalized prefix: independent binds therefore have one commitment
    /// regardless of construction order, while union order, upper semantics,
    /// and the identities that actually remain installed stay significant.
    pub fn compose(
        policy: NamespacePolicyCommitment,
        mounts: impl IntoIterator<Item = AdmittedMount>,
    ) -> Result<Self, NamespaceAdmissionError> {
        let mounts: Vec<_> = mounts.into_iter().collect();
        if mounts.is_empty() {
            return Err(NamespaceAdmissionError::EmptyMountTable);
        }

        let mut namespace = Namespace::new();
        let mut identified = BTreeMap::<String, IdentifiedMountEntry>::new();

        for mount in mounts {
            let prefix = canonical_prefix(&mount.prefix);
            apply_identified_bind(&mut identified, &prefix, mount.bind, mount.identity.clone());
            namespace.bind_mount(&prefix, mount.target, mount.bind)?;
        }

        let mut canonical = b"hyprstream-effective-namespace-v1\0".to_vec();
        append_field(&mut canonical, b"policy", policy.as_str().as_bytes());
        for (prefix, entry) in identified {
            append_field(&mut canonical, b"prefix", prefix.as_bytes());
            append_field(
                &mut canonical,
                b"target-count",
                &(entry.targets.len() as u64).to_be_bytes(),
            );
            for (index, identity) in entry.targets.iter().enumerate() {
                append_field(
                    &mut canonical,
                    b"target-index",
                    &(index as u64).to_be_bytes(),
                );
                append_field(&mut canonical, b"identity", identity.as_str().as_bytes());
            }
            append_field(
                &mut canonical,
                b"upper-target-index",
                &entry
                    .upper_target_index
                    .map_or(u64::MAX, |index| index as u64)
                    .to_be_bytes(),
            );
        }

        let digest = *blake3::hash(&canonical).as_bytes();
        Ok(Self {
            namespace,
            canonical_manifest: canonical,
            digest,
        })
    }

    /// Fork the exact admitted namespace for one sandbox delivery.
    #[must_use]
    pub fn fork(&self) -> Namespace {
        self.namespace.fork()
    }

    /// Derived effective-namespace digest, never caller description bytes.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Deterministic canonical manifest used to derive [`Self::digest`].
    #[must_use]
    pub fn canonical_manifest(&self) -> &[u8] {
        &self.canonical_manifest
    }

    /// Inspect the frozen topology without exposing raw mount capabilities.
    #[must_use]
    pub fn topology(&self) -> Vec<NamespaceMountTopology> {
        self.namespace.mount_topology()
    }
}

fn canonical_prefix(prefix: &str) -> String {
    let mut components = Vec::new();
    for component in prefix.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop();
            }
            component => components.push(component),
        }
    }
    if components.is_empty() {
        "/".to_owned()
    } else {
        format!("/{}", components.join("/"))
    }
}

fn apply_identified_bind(
    entries: &mut BTreeMap<String, IdentifiedMountEntry>,
    prefix: &str,
    bind: BindFlag,
    identity: MountIdentity,
) {
    match bind {
        BindFlag::Replace => {
            entries.insert(
                prefix.to_owned(),
                IdentifiedMountEntry {
                    targets: vec![identity],
                    upper_target_index: None,
                },
            );
        }
        BindFlag::Before => {
            let entry = entries
                .entry(prefix.to_owned())
                .or_insert_with(|| IdentifiedMountEntry {
                    targets: Vec::new(),
                    upper_target_index: None,
                });
            entry.targets.insert(0, identity);
            if let Some(index) = entry.upper_target_index.as_mut() {
                *index += 1;
            }
        }
        BindFlag::After => {
            let entry = entries
                .entry(prefix.to_owned())
                .or_insert_with(|| IdentifiedMountEntry {
                    targets: Vec::new(),
                    upper_target_index: None,
                });
            entry.targets.push(identity);
        }
        BindFlag::Upper => {
            let entry = entries
                .entry(prefix.to_owned())
                .or_insert_with(|| IdentifiedMountEntry {
                    targets: Vec::new(),
                    upper_target_index: None,
                });
            entry.targets.insert(0, identity);
            entry.upper_target_index = Some(0);
        }
    }
}

fn append_field(output: &mut Vec<u8>, name: &[u8], value: &[u8]) {
    output.extend_from_slice(&(name.len() as u64).to_be_bytes());
    output.extend_from_slice(name);
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::{SyntheticMount, SyntheticNode};

    fn target() -> MountTarget {
        Arc::new(SyntheticMount::new(SyntheticNode::dir()))
    }

    fn identity(value: &str) -> MountIdentity {
        MountIdentity::from_trusted_policy(value).unwrap()
    }

    fn policy() -> NamespacePolicyCommitment {
        NamespacePolicyCommitment::from_trusted_policy("policy-v1").unwrap()
    }

    #[test]
    fn missing_or_unstable_mount_identity_fails_closed() {
        let policy = NamespacePolicyCommitment::from_trusted_policy("policy-v1").unwrap();
        assert!(NamespacePolicyCommitment::from_trusted_policy(" ").is_err());
        assert!(MountIdentity::from_trusted_policy(" ").is_err());
        assert!(matches!(
            AdmittedMount::new("/work", target(), BindFlag::Replace, None),
            Err(NamespaceAdmissionError::MissingMountIdentity)
        ));
        assert!(matches!(
            AdmittedNamespace::compose(policy, Vec::<AdmittedMount>::new()),
            Err(NamespaceAdmissionError::EmptyMountTable)
        ));
    }

    #[test]
    fn equivalent_normalized_topology_reconstructs_deterministically() {
        let first = AdmittedNamespace::compose(
            policy(),
            [
                AdmittedMount::new(
                    "/work/../work/",
                    target(),
                    BindFlag::Replace,
                    Some(identity("lower-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Upper,
                    Some(identity("upper-v1")),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let reconstructed = AdmittedNamespace::compose(
            policy(),
            [
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Replace,
                    Some(identity("lower-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "work/.",
                    target(),
                    BindFlag::Upper,
                    Some(identity("upper-v1")),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            first.topology(),
            vec![NamespaceMountTopology {
                prefix: "/work".into(),
                target_count: 2,
                upper_target_index: Some(0),
            }]
        );
        assert_eq!(
            first.canonical_manifest(),
            reconstructed.canonical_manifest()
        );
        assert_eq!(first.digest(), reconstructed.digest());
    }

    #[test]
    fn manifest_binds_final_union_order_and_upper_without_history() {
        let after = AdmittedNamespace::compose(
            policy(),
            [
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Replace,
                    Some(identity("lower-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::After,
                    Some(identity("second-v1")),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let before = AdmittedNamespace::compose(
            policy(),
            [
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Replace,
                    Some(identity("lower-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Before,
                    Some(identity("second-v1")),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let upper = AdmittedNamespace::compose(
            policy(),
            [
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Replace,
                    Some(identity("lower-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Upper,
                    Some(identity("second-v1")),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert_ne!(after.digest(), before.digest());
        assert_ne!(before.digest(), upper.digest());
        assert_eq!(after.topology()[0].upper_target_index, None);
        assert_eq!(upper.topology()[0].upper_target_index, Some(0));

        let changed_identity = AdmittedNamespace::compose(
            policy(),
            [
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Replace,
                    Some(identity("lower-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::After,
                    Some(identity("second-v2")),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert_ne!(after.digest(), changed_identity.digest());

        // An overwritten bind is absent from the final manifest: replace it
        // with the same final target and the digest reconstructs exactly.
        let overwritten = AdmittedNamespace::compose(
            policy(),
            [
                AdmittedMount::new(
                    "/obsolete",
                    target(),
                    BindFlag::Replace,
                    Some(identity("obsolete-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Replace,
                    Some(identity("lower-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "/obsolete",
                    target(),
                    BindFlag::Replace,
                    Some(identity("obsolete-v2")),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let final_only = AdmittedNamespace::compose(
            policy(),
            [
                AdmittedMount::new(
                    "/work",
                    target(),
                    BindFlag::Replace,
                    Some(identity("lower-v1")),
                )
                .unwrap(),
                AdmittedMount::new(
                    "/obsolete",
                    target(),
                    BindFlag::Replace,
                    Some(identity("obsolete-v2")),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(overwritten.digest(), final_only.digest());
    }
}
