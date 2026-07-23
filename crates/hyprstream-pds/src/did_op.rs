//! The permanent genesis operation for a hyprstream-hosted `did:web` account.
//!
//! This module deliberately implements only operation zero. Successor-chain
//! validation, fork selection, and historical key-validity queries belong to
//! C1 (#1167). The bytes emitted here are nevertheless the first record in that
//! future log, so their grammar is a one-way door for every minted account.
//!
//! # Rotation custody is structural
//!
//! [`GenesisRotationKeys`] is not a `Vec`. It has exactly three ordered slots:
//!
//! 0. a required [`UserRotationKey`];
//! 1. an enrolled recovery key or an explicit
//!    [`RecoveryKeyEnrollment::Declined`];
//! 2. an enrolled host key or an explicit [`HostKeyEnrollment::Absent`].
//!
//! A [`HostRotationKey`] therefore cannot be passed as priority zero:
//!
//! ```compile_fail
//! use hyprstream_pds::{
//!     GenesisRotationKeys, HostKeyEnrollment, HostRotationKey,
//!     RecoveryKeyEnrollment,
//! };
//! # fn example(host: HostRotationKey) {
//! let _ = GenesisRotationKeys::new(
//!     host,
//!     RecoveryKeyEnrollment::Declined,
//!     HostKeyEnrollment::Absent,
//! );
//! # }
//! ```
//!
//! The wire form preserves those slots as a fixed three-element
//! `rotation_keys` array. A declined/absent slot is encoded as a role-bearing
//! map whose `key` is `null`, so recovery absence is a signed decision rather
//! than an omitted field or decoder default.
//!
//! # Hybrid pin
//!
//! Rotation is not an atproto interoperability surface. Every rotation key and
//! the genesis signature therefore consists of Ed25519 + ML-DSA-65, with both
//! components required. There is no classical constructor or verification
//! mode.

use std::collections::BTreeSet;

use anyhow::{bail, ensure, Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use hyprstream_crypto::cose_sign::{
    assemble_composite_nested, sign_composite, split_composite, verify_composite,
};
use hyprstream_crypto::pq::{
    ml_dsa_sk_to_vk_bytes, ml_dsa_vk_bytes, ml_dsa_vk_from_bytes, MlDsaSigningKey,
};

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// Version sealed into operation number zero.
pub const DID_OP_VERSION: u16 = 1;
/// Domain-separation context bound into both signature components.
pub const DID_OP_SIGNATURE_CONTEXT: &str = "hyprstream-did-op/1";

const COMPOSITE_ALGORITHM: &str = "id-MLDSA65-Ed25519";
const ED25519_PUBLIC_KEY_LEN: usize = 32;
const ED25519_SIGNATURE_LEN: usize = 64;
const ML_DSA65_PUBLIC_KEY_LEN: usize = 1952;
const ML_DSA65_SIGNATURE_LEN: usize = 3309;
const GENESIS_ROTATION_SLOT_COUNT: usize = 3;

/// A pinned-Hybrid public key authorized to control a hosted account.
///
/// The fields are private so malformed component lengths cannot be represented.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HybridRotationKey {
    ed25519: Vec<u8>,
    mldsa65: Vec<u8>,
}

impl HybridRotationKey {
    /// Construct a rotation key from its mandatory classical and PQ components.
    pub fn new(ed25519: impl Into<Vec<u8>>, mldsa65: impl Into<Vec<u8>>) -> Result<Self> {
        let key = Self {
            ed25519: ed25519.into(),
            mldsa65: mldsa65.into(),
        };
        key.validate()?;
        Ok(key)
    }

    /// The Ed25519 verifying-key bytes.
    pub fn ed25519(&self) -> &[u8] {
        &self.ed25519
    }

    /// The ML-DSA-65 verifying-key bytes.
    pub fn mldsa65(&self) -> &[u8] {
        &self.mldsa65
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.ed25519.len() == ED25519_PUBLIC_KEY_LEN,
            "rotation Ed25519 key must be {ED25519_PUBLIC_KEY_LEN} bytes"
        );
        ensure!(
            self.mldsa65.len() == ML_DSA65_PUBLIC_KEY_LEN,
            "rotation ML-DSA-65 key must be {ML_DSA65_PUBLIC_KEY_LEN} bytes"
        );
        let ed: [u8; ED25519_PUBLIC_KEY_LEN] = self
            .ed25519
            .as_slice()
            .try_into()
            .context("rotation Ed25519 key has an invalid length")?;
        VerifyingKey::from_bytes(&ed).context("rotation Ed25519 key is not a valid point")?;
        ml_dsa_vk_from_bytes(&self.mldsa65).context("rotation ML-DSA-65 key is invalid")?;
        Ok(())
    }

    fn identity(&self) -> Vec<u8> {
        let mut identity = Vec::with_capacity(self.ed25519.len() + self.mldsa65.len());
        identity.extend_from_slice(&self.ed25519);
        identity.extend_from_slice(&self.mldsa65);
        identity
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("ed25519", DagCbor::Bytes(self.ed25519.clone())),
            ("mldsa65", DagCbor::Bytes(self.mldsa65.clone())),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(value, &["ed25519", "mldsa65"], "rotation key")?;
        Self::new(
            required(value, "ed25519", "rotation key")?
                .as_bytes()?
                .to_vec(),
            required(value, "mldsa65", "rotation key")?
                .as_bytes()?
                .to_vec(),
        )
    }
}

macro_rules! custody_key {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct $name(HybridRotationKey);

        impl $name {
            /// Bind valid pinned-Hybrid public material to this custody role.
            pub fn new(key: HybridRotationKey) -> Self {
                Self(key)
            }

            /// The role-bound public key.
            pub fn key(&self) -> &HybridRotationKey {
                &self.0
            }
        }
    };
}

custody_key!(
    UserRotationKey,
    "The required user-held key occupying genesis priority zero."
);
custody_key!(
    RecoveryRotationKey,
    "A user-held recovery key kept on a device distinct from the primary key."
);
custody_key!(
    HostRotationKey,
    "An optional host-held rotation key, representable only at priority two."
);

/// The deliberate outcome of the recovery-key enrollment offer.
///
/// There is intentionally no `Default`: the mint caller must record either a
/// concrete recovery key or an explicit decline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecoveryKeyEnrollment {
    Enrolled(RecoveryRotationKey),
    Declined,
}

/// Whether the deployment is enrolled as the lowest-priority rotation actor.
///
/// There is intentionally no `Default`; absence is sealed explicitly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostKeyEnrollment {
    Enrolled(HostRotationKey),
    Absent,
}

/// A reference to one ordered genesis rotation slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotationSlotRef<'a> {
    pub priority: u8,
    pub role: &'static str,
    pub key: Option<&'a HybridRotationKey>,
}

/// The only representable genesis rotation-key ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenesisRotationKeys {
    user: UserRotationKey,
    recovery: RecoveryKeyEnrollment,
    host: HostKeyEnrollment,
}

impl GenesisRotationKeys {
    /// Establish the permanent priority order.
    ///
    /// Equal key material in two custody slots is rejected: a "recovery" key
    /// that is the primary key is not recovery, and a host key equal to either
    /// user-held key would blur the custody boundary.
    pub fn new(
        user: UserRotationKey,
        recovery: RecoveryKeyEnrollment,
        host: HostKeyEnrollment,
    ) -> Result<Self> {
        let keys = Self {
            user,
            recovery,
            host,
        };
        keys.validate()?;
        Ok(keys)
    }

    pub fn user(&self) -> &UserRotationKey {
        &self.user
    }

    pub fn recovery(&self) -> &RecoveryKeyEnrollment {
        &self.recovery
    }

    pub fn host(&self) -> &HostKeyEnrollment {
        &self.host
    }

    /// Return all three priority slots, including deliberate empty slots.
    pub fn ordered_slots(&self) -> [RotationSlotRef<'_>; GENESIS_ROTATION_SLOT_COUNT] {
        let recovery = match &self.recovery {
            RecoveryKeyEnrollment::Enrolled(key) => Some(key.key()),
            RecoveryKeyEnrollment::Declined => None,
        };
        let host = match &self.host {
            HostKeyEnrollment::Enrolled(key) => Some(key.key()),
            HostKeyEnrollment::Absent => None,
        };
        [
            RotationSlotRef {
                priority: 0,
                role: "user",
                key: Some(self.user.key()),
            },
            RotationSlotRef {
                priority: 1,
                role: "recovery",
                key: recovery,
            },
            RotationSlotRef {
                priority: 2,
                role: "host",
                key: host,
            },
        ]
    }

    fn validate(&self) -> Result<()> {
        self.user.key().validate()?;
        let mut identities = BTreeSet::new();
        ensure!(
            identities.insert(self.user.key().identity()),
            "duplicate primary user rotation key"
        );
        if let RecoveryKeyEnrollment::Enrolled(key) = &self.recovery {
            key.key().validate()?;
            ensure!(
                identities.insert(key.key().identity()),
                "recovery rotation key must differ from the primary user key"
            );
        }
        if let HostKeyEnrollment::Enrolled(key) = &self.host {
            key.key().validate()?;
            ensure!(
                identities.insert(key.key().identity()),
                "host rotation key must differ from every user-held key"
            );
        }
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::list(self.ordered_slots().into_iter().map(|slot| {
            DagCbor::str_map([
                (
                    "key",
                    slot.key.map_or(DagCbor::Null, HybridRotationKey::to_value),
                ),
                ("role", DagCbor::Text(slot.role.to_owned())),
            ])
        }))
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        let slots = value.as_list()?;
        ensure!(
            slots.len() == GENESIS_ROTATION_SLOT_COUNT,
            "genesis rotation_keys must contain exactly {GENESIS_ROTATION_SLOT_COUNT} explicit slots"
        );

        let user = parse_slot(&slots[0], "user")?
            .map(UserRotationKey::new)
            .ok_or_else(|| anyhow::anyhow!("priority-zero user rotation key is required"))?;
        let recovery = match parse_slot(&slots[1], "recovery")? {
            Some(key) => RecoveryKeyEnrollment::Enrolled(RecoveryRotationKey::new(key)),
            None => RecoveryKeyEnrollment::Declined,
        };
        let host = match parse_slot(&slots[2], "host")? {
            Some(key) => HostKeyEnrollment::Enrolled(HostRotationKey::new(key)),
            None => HostKeyEnrollment::Absent,
        };
        Self::new(user, recovery, host)
    }
}

fn parse_slot(value: &DagCbor, expected_role: &str) -> Result<Option<HybridRotationKey>> {
    reject_unknown(value, &["key", "role"], "rotation slot")?;
    let role = required(value, "role", "rotation slot")?.as_str()?;
    ensure!(
        role == expected_role,
        "genesis priority slot for {expected_role:?} cannot contain role {role:?}"
    );
    let key = required(value, "key", "rotation slot")?;
    if key.is_null() {
        Ok(None)
    } else {
        HybridRotationKey::from_value(key).map(Some)
    }
}

/// The repo state deliberately recorded at genesis.
///
/// `EmptyRepo` encodes as `null`; it means the account has no repo head yet,
/// not "skip head validation".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenesisRepoHead {
    EmptyRepo,
    Existing(Cid),
}

impl GenesisRepoHead {
    fn to_value(self) -> DagCbor {
        match self {
            Self::EmptyRepo => DagCbor::Null,
            Self::Existing(cid) => DagCbor::Link(cid),
        }
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        if value.is_null() {
            Ok(Self::EmptyRepo)
        } else {
            Ok(Self::Existing(*value.as_link()?))
        }
    }
}

/// Operation zero without its signature.
///
/// All fields are private so operation zero cannot accidentally acquire a
/// predecessor, a non-zero sequence, or omit its version.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsignedGenesisDidOp {
    did: String,
    doc_cid: Cid,
    rotation_keys: GenesisRotationKeys,
    head_at_op: GenesisRepoHead,
}

impl UnsignedGenesisDidOp {
    pub fn new(
        did: impl Into<String>,
        doc_cid: Cid,
        rotation_keys: GenesisRotationKeys,
        head_at_op: GenesisRepoHead,
    ) -> Result<Self> {
        let op = Self {
            did: did.into(),
            doc_cid,
            rotation_keys,
            head_at_op,
        };
        op.validate()?;
        Ok(op)
    }

    pub fn version(&self) -> u16 {
        DID_OP_VERSION
    }

    pub fn sequence(&self) -> u64 {
        0
    }

    pub fn did(&self) -> &str {
        &self.did
    }

    pub fn doc_cid(&self) -> Cid {
        self.doc_cid
    }

    pub fn rotation_keys(&self) -> &GenesisRotationKeys {
        &self.rotation_keys
    }

    pub fn head_at_op(&self) -> GenesisRepoHead {
        self.head_at_op
    }

    pub fn to_dag_cbor(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(self.to_value().encode())
    }

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        let op = Self::from_value(&value)?;
        ensure!(
            op.to_dag_cbor()? == bytes,
            "genesis DidOp is not canonical DAG-CBOR"
        );
        Ok(op)
    }

    fn validate(&self) -> Result<()> {
        validate_host_form_did_web(&self.did)?;
        self.rotation_keys.validate()
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("did", DagCbor::Text(self.did.clone())),
            ("doc_cid", DagCbor::Link(self.doc_cid)),
            ("head_at_op", self.head_at_op.to_value()),
            ("prev", DagCbor::Null),
            ("rotation_keys", self.rotation_keys.to_value()),
            ("seq", DagCbor::Unsigned(0)),
            ("version", DagCbor::Unsigned(u64::from(DID_OP_VERSION))),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(
            value,
            &[
                "version",
                "prev",
                "seq",
                "did",
                "doc_cid",
                "rotation_keys",
                "head_at_op",
            ],
            "unsigned genesis DidOp",
        )?;
        let version = required(value, "version", "unsigned genesis DidOp")?.as_unsigned()?;
        ensure!(
            version == u64::from(DID_OP_VERSION),
            "unsupported genesis DidOp version {version}"
        );
        ensure!(
            required(value, "prev", "unsigned genesis DidOp")?.is_null(),
            "genesis DidOp prev must be null"
        );
        ensure!(
            required(value, "seq", "unsigned genesis DidOp")?.as_unsigned()? == 0,
            "genesis DidOp seq must be zero"
        );
        Self::new(
            required(value, "did", "unsigned genesis DidOp")?
                .as_str()?
                .to_owned(),
            *required(value, "doc_cid", "unsigned genesis DidOp")?.as_link()?,
            GenesisRotationKeys::from_value(required(
                value,
                "rotation_keys",
                "unsigned genesis DidOp",
            )?)?,
            GenesisRepoHead::from_value(required(value, "head_at_op", "unsigned genesis DidOp")?)?,
        )
    }
}

/// The decomposed, context-bound composite signature stored in a `DidOp`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DidOpSignature {
    protected: Vec<u8>,
    ed25519: Vec<u8>,
    mldsa65: Vec<u8>,
}

impl DidOpSignature {
    /// Construct the canonical v1 signature wrapper from raw component
    /// signatures returned by a user-side composite signer.
    pub fn from_components(
        ed25519: impl Into<Vec<u8>>,
        mldsa65: impl Into<Vec<u8>>,
    ) -> Result<Self> {
        Self::new(signature_protected_header(), ed25519, mldsa65)
    }

    /// Construct from all wire components.
    ///
    /// Prefer [`Self::from_components`] for v1 signing. This form exists for
    /// strict wire decoding and rejects any non-canonical protected header.
    pub fn new(
        protected: impl Into<Vec<u8>>,
        ed25519: impl Into<Vec<u8>>,
        mldsa65: impl Into<Vec<u8>>,
    ) -> Result<Self> {
        let signature = Self {
            protected: protected.into(),
            ed25519: ed25519.into(),
            mldsa65: mldsa65.into(),
        };
        signature.validate()?;
        Ok(signature)
    }

    pub fn protected(&self) -> &[u8] {
        &self.protected
    }

    pub fn ed25519(&self) -> &[u8] {
        &self.ed25519
    }

    pub fn mldsa65(&self) -> &[u8] {
        &self.mldsa65
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.protected == signature_protected_header(),
            "DidOp signature protected header is not the canonical v1 binding"
        );
        ensure!(
            self.ed25519.len() == ED25519_SIGNATURE_LEN,
            "DidOp Ed25519 signature must be {ED25519_SIGNATURE_LEN} bytes"
        );
        ensure!(
            self.mldsa65.len() == ML_DSA65_SIGNATURE_LEN,
            "DidOp ML-DSA-65 signature must be {ML_DSA65_SIGNATURE_LEN} bytes"
        );
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            (
                "context",
                DagCbor::Text(DID_OP_SIGNATURE_CONTEXT.to_owned()),
            ),
            ("ed25519", DagCbor::Bytes(self.ed25519.clone())),
            ("mldsa65", DagCbor::Bytes(self.mldsa65.clone())),
            ("protected", DagCbor::Bytes(self.protected.clone())),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(
            value,
            &["context", "protected", "ed25519", "mldsa65"],
            "DidOp signature",
        )?;
        ensure!(
            required(value, "context", "DidOp signature")?.as_str()? == DID_OP_SIGNATURE_CONTEXT,
            "DidOp signature context mismatch"
        );
        Self::new(
            required(value, "protected", "DidOp signature")?
                .as_bytes()?
                .to_vec(),
            required(value, "ed25519", "DidOp signature")?
                .as_bytes()?
                .to_vec(),
            required(value, "mldsa65", "DidOp signature")?
                .as_bytes()?
                .to_vec(),
        )
    }
}

/// A sealed, self-signed genesis operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenesisDidOp {
    unsigned: UnsignedGenesisDidOp,
    signature: DidOpSignature,
}

impl GenesisDidOp {
    /// Verify a user-produced signature and seal operation zero.
    ///
    /// Only the required priority-zero user key is accepted. Recovery and host
    /// keys authorize future operations according to the ordered fork rules;
    /// neither can self-sign genesis.
    pub fn seal(unsigned: UnsignedGenesisDidOp, signature: DidOpSignature) -> Result<Self> {
        unsigned.validate()?;
        signature.validate()?;
        verify_signature(
            &unsigned.to_dag_cbor()?,
            &signature,
            unsigned.rotation_keys.user.key(),
        )?;
        Ok(Self {
            unsigned,
            signature,
        })
    }

    pub fn unsigned(&self) -> &UnsignedGenesisDidOp {
        &self.unsigned
    }

    pub fn signature(&self) -> &DidOpSignature {
        &self.signature
    }

    pub fn to_dag_cbor(&self) -> Result<Vec<u8>> {
        self.unsigned.validate()?;
        self.signature.validate()?;
        Ok(self.to_value().encode())
    }

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        reject_unknown(
            &value,
            &[
                "version",
                "prev",
                "seq",
                "did",
                "doc_cid",
                "rotation_keys",
                "head_at_op",
                "sig",
            ],
            "genesis DidOp",
        )?;
        let unsigned_value = DagCbor::str_map([
            ("did", required(&value, "did", "genesis DidOp")?.clone()),
            (
                "doc_cid",
                required(&value, "doc_cid", "genesis DidOp")?.clone(),
            ),
            (
                "head_at_op",
                required(&value, "head_at_op", "genesis DidOp")?.clone(),
            ),
            ("prev", required(&value, "prev", "genesis DidOp")?.clone()),
            (
                "rotation_keys",
                required(&value, "rotation_keys", "genesis DidOp")?.clone(),
            ),
            ("seq", required(&value, "seq", "genesis DidOp")?.clone()),
            (
                "version",
                required(&value, "version", "genesis DidOp")?.clone(),
            ),
        ]);
        let unsigned = UnsignedGenesisDidOp::from_value(&unsigned_value)?;
        let signature = DidOpSignature::from_value(required(&value, "sig", "genesis DidOp")?)?;
        let op = Self::seal(unsigned, signature)?;
        ensure!(
            op.to_dag_cbor()? == bytes,
            "genesis DidOp is not canonical DAG-CBOR"
        );
        Ok(op)
    }

    pub fn cid(&self) -> Result<Cid> {
        Ok(Cid::from_dag_cbor(&self.to_dag_cbor()?))
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("did", DagCbor::Text(self.unsigned.did.clone())),
            ("doc_cid", DagCbor::Link(self.unsigned.doc_cid)),
            ("head_at_op", self.unsigned.head_at_op.to_value()),
            ("prev", DagCbor::Null),
            ("rotation_keys", self.unsigned.rotation_keys.to_value()),
            ("seq", DagCbor::Unsigned(0)),
            ("sig", self.signature.to_value()),
            ("version", DagCbor::Unsigned(u64::from(DID_OP_VERSION))),
        ])
    }
}

/// Sign operation zero with the user-held priority-zero key.
///
/// This helper is suitable for a user-side signer. The host mint path consumes
/// the returned signature but never needs either private key.
pub fn sign_genesis(
    unsigned: &UnsignedGenesisDidOp,
    ed25519: &SigningKey,
    mldsa65: &MlDsaSigningKey,
) -> Result<DidOpSignature> {
    unsigned.validate()?;
    let user_key = unsigned.rotation_keys.user.key();
    ensure!(
        user_key.ed25519 == ed25519.verifying_key().to_bytes(),
        "genesis Ed25519 signer does not match priority-zero user key"
    );
    ensure!(
        user_key.mldsa65 == ml_dsa_sk_to_vk_bytes(mldsa65),
        "genesis ML-DSA-65 signer does not match priority-zero user key"
    );

    let protected = signature_protected_header();
    let composite = sign_composite(ed25519, Some(mldsa65), &unsigned.to_dag_cbor()?, &protected)
        .context("genesis DidOp composite signing failed")?;
    let (ed_signature, pq_signature) = split_composite(&composite)?;
    let pq_signature = pq_signature
        .ok_or_else(|| anyhow::anyhow!("genesis DidOp signing produced no ML-DSA-65 component"))?;
    DidOpSignature::new(protected, ed_signature, pq_signature)
}

fn verify_signature(
    payload: &[u8],
    signature: &DidOpSignature,
    key: &HybridRotationKey,
) -> Result<()> {
    signature.validate()?;
    key.validate()?;
    let ed_bytes: [u8; ED25519_PUBLIC_KEY_LEN] = key
        .ed25519
        .as_slice()
        .try_into()
        .context("rotation Ed25519 key has an invalid length")?;
    let ed_vk = VerifyingKey::from_bytes(&ed_bytes).context("rotation Ed25519 key is invalid")?;
    let pq_vk = ml_dsa_vk_from_bytes(&key.mldsa65).context("rotation ML-DSA-65 key is invalid")?;
    let composite = assemble_composite_nested(
        (ed_vk.to_bytes().to_vec(), signature.ed25519.clone()),
        Some((ml_dsa_vk_bytes(&pq_vk), signature.mldsa65.clone())),
    )
    .context("failed to reassemble genesis DidOp composite signature")?;
    let verified = verify_composite(
        &composite,
        &ed_vk,
        Some(&pq_vk),
        payload,
        &signature_protected_header(),
        true,
    )
    .context("genesis DidOp priority-zero signature verification failed")?;
    ensure!(
        verified.eddsa && verified.ml_dsa,
        "genesis DidOp did not verify both pinned-Hybrid components"
    );
    Ok(())
}

fn signature_protected_header() -> Vec<u8> {
    DagCbor::str_map([
        ("alg", DagCbor::Text(COMPOSITE_ALGORITHM.to_owned())),
        (
            "context",
            DagCbor::Text(DID_OP_SIGNATURE_CONTEXT.to_owned()),
        ),
    ])
    .encode()
}

/// Validate the host-only `did:web` form used for accounts.
///
/// The future A2/A3 mint seam supplies an allocated label and deployment zone;
/// this check independently refuses path-form, port-bearing, uppercase, or
/// non-LDH DIDs before their value is permanently signed.
pub(crate) fn validate_host_form_did_web(did: &str) -> Result<()> {
    let host = did
        .strip_prefix("did:web:")
        .ok_or_else(|| anyhow::anyhow!("hosted account DID must use did:web"))?;
    ensure!(!host.is_empty(), "hosted account did:web host is empty");
    ensure!(
        !host.contains(':') && !host.contains('/') && !host.contains('%'),
        "hosted account DID must be host-form did:web without path, port, or percent encoding"
    );
    ensure!(
        host == host.to_ascii_lowercase(),
        "hosted account DID must already be lowercase"
    );
    let labels = host.split('.').collect::<Vec<_>>();
    ensure!(
        labels.len() >= 2,
        "hosted account DID must contain a deployment-qualified DNS host"
    );
    ensure!(
        host.len() <= 253,
        "hosted account DID host exceeds DNS limit"
    );
    for label in labels {
        ensure!(
            !label.is_empty() && label.len() <= 63,
            "hosted account DID contains an empty or oversized DNS label"
        );
        ensure!(
            label
                .bytes()
                .all(|byte| { byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' }),
            "hosted account DID labels must be lowercase LDH"
        );
        ensure!(
            !label.starts_with('-') && !label.ends_with('-'),
            "hosted account DID labels must not start or end with a hyphen"
        );
    }
    Ok(())
}

fn required<'a>(value: &'a DagCbor, key: &str, what: &str) -> Result<&'a DagCbor> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{what} missing required field {key:?}"))
}

fn reject_unknown(value: &DagCbor, allowed: &[&str], what: &str) -> Result<()> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    for (key, _) in value.as_map()? {
        let key = key.as_str()?;
        if !allowed.contains(key) {
            bail!("{what} contains unknown field {key:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use hyprstream_crypto::pq::{
        ml_dsa_generate_keypair, ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_bytes,
        MlDsaSigningKey,
    };
    use rand::rngs::OsRng;
    use sha2::{Digest as _, Sha256};

    struct Signer {
        ed: SigningKey,
        pq: MlDsaSigningKey,
        public: HybridRotationKey,
    }

    fn signer() -> Signer {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let public =
            HybridRotationKey::new(ed.verifying_key().to_bytes(), ml_dsa_vk_bytes(&pq_vk)).unwrap();
        Signer { ed, pq, public }
    }

    fn unsigned(
        user: &Signer,
        recovery: RecoveryKeyEnrollment,
        host: HostKeyEnrollment,
    ) -> UnsignedGenesisDidOp {
        UnsignedGenesisDidOp::new(
            "did:web:alice.acct.example.com",
            Cid::from_raw(b"sealed did document"),
            GenesisRotationKeys::new(UserRotationKey::new(user.public.clone()), recovery, host)
                .unwrap(),
            GenesisRepoHead::EmptyRepo,
        )
        .unwrap()
    }

    #[test]
    fn host_role_at_priority_zero_is_rejected_on_decode() {
        let user = signer();
        let rotations = GenesisRotationKeys::new(
            UserRotationKey::new(user.public),
            RecoveryKeyEnrollment::Declined,
            HostKeyEnrollment::Absent,
        )
        .unwrap();
        let mut slots = rotations.to_value().as_list().unwrap().to_vec();
        slots[0] = DagCbor::str_map([
            ("key", slots[0].get("key").expect("slot key").clone()),
            ("role", DagCbor::Text("host".to_owned())),
        ]);
        let error = GenesisRotationKeys::from_value(&DagCbor::List(slots)).unwrap_err();
        assert!(error.to_string().contains("cannot contain role"));
    }

    #[test]
    fn missing_priority_zero_user_key_is_rejected() {
        let rotations = DagCbor::list([
            DagCbor::str_map([
                ("key", DagCbor::Null),
                ("role", DagCbor::Text("user".to_owned())),
            ]),
            DagCbor::str_map([
                ("key", DagCbor::Null),
                ("role", DagCbor::Text("recovery".to_owned())),
            ]),
            DagCbor::str_map([
                ("key", DagCbor::Null),
                ("role", DagCbor::Text("host".to_owned())),
            ]),
        ]);
        let error = GenesisRotationKeys::from_value(&rotations).unwrap_err();
        assert!(error.to_string().contains("priority-zero user"));
    }

    #[test]
    fn declined_recovery_is_explicit_in_signed_bytes() {
        let user = signer();
        let unsigned = unsigned(
            &user,
            RecoveryKeyEnrollment::Declined,
            HostKeyEnrollment::Absent,
        );
        let value = DagCbor::decode(&unsigned.to_dag_cbor().unwrap()).unwrap();
        let slots = value.get("rotation_keys").unwrap().as_list().unwrap();
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[1].get("role").unwrap().as_str().unwrap(), "recovery");
        assert!(slots[1].get("key").unwrap().is_null());

        let roundtrip =
            UnsignedGenesisDidOp::from_dag_cbor(&unsigned.to_dag_cbor().unwrap()).unwrap();
        assert_eq!(
            roundtrip.rotation_keys().recovery(),
            &RecoveryKeyEnrollment::Declined
        );
    }

    #[test]
    fn sealed_ordering_roundtrips_byte_exactly() {
        let user = signer();
        let recovery = signer();
        let host = signer();
        let unsigned = unsigned(
            &user,
            RecoveryKeyEnrollment::Enrolled(RecoveryRotationKey::new(recovery.public.clone())),
            HostKeyEnrollment::Enrolled(HostRotationKey::new(host.public.clone())),
        );
        let signature = sign_genesis(&unsigned, &user.ed, &user.pq).unwrap();
        let sealed = GenesisDidOp::seal(unsigned, signature).unwrap();
        let bytes = sealed.to_dag_cbor().unwrap();
        let decoded = GenesisDidOp::from_dag_cbor(&bytes).unwrap();
        assert_eq!(decoded.to_dag_cbor().unwrap(), bytes);

        let slots = decoded.unsigned().rotation_keys().ordered_slots();
        assert_eq!(
            slots.map(|slot| (slot.priority, slot.role)),
            [(0, "user"), (1, "recovery"), (2, "host")]
        );
        assert_eq!(slots[0].key, Some(&user.public));
        assert_eq!(slots[1].key, Some(&recovery.public));
        assert_eq!(slots[2].key, Some(&host.public));
    }

    #[test]
    fn operation_zero_contains_version_from_first_signed_bytes() {
        let user = signer();
        let unsigned = unsigned(
            &user,
            RecoveryKeyEnrollment::Declined,
            HostKeyEnrollment::Absent,
        );
        let signature = sign_genesis(&unsigned, &user.ed, &user.pq).unwrap();
        let bytes = GenesisDidOp::seal(unsigned, signature)
            .unwrap()
            .to_dag_cbor()
            .unwrap();
        let value = DagCbor::decode(&bytes).unwrap();
        assert_eq!(
            value.get("version").unwrap().as_unsigned().unwrap(),
            u64::from(DID_OP_VERSION)
        );
        assert_eq!(value.get("seq").unwrap().as_unsigned().unwrap(), 0);
        assert!(value.get("prev").unwrap().is_null());
    }

    #[test]
    fn tampering_a_signed_field_fails_closed() {
        let user = signer();
        let unsigned = unsigned(
            &user,
            RecoveryKeyEnrollment::Declined,
            HostKeyEnrollment::Absent,
        );
        let signature = sign_genesis(&unsigned, &user.ed, &user.pq).unwrap();
        let tampered = UnsignedGenesisDidOp::new(
            unsigned.did().to_owned(),
            Cid::from_raw(b"different document"),
            unsigned.rotation_keys().clone(),
            unsigned.head_at_op(),
        )
        .unwrap();
        assert!(GenesisDidOp::seal(tampered, signature).is_err());
    }

    #[test]
    fn duplicate_custody_keys_are_rejected() {
        let user = signer();
        let error = GenesisRotationKeys::new(
            UserRotationKey::new(user.public.clone()),
            RecoveryKeyEnrollment::Enrolled(RecoveryRotationKey::new(user.public.clone())),
            HostKeyEnrollment::Absent,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("recovery rotation key must differ"));
    }

    #[test]
    fn path_form_did_is_never_sealable() {
        let user = signer();
        let rotations = GenesisRotationKeys::new(
            UserRotationKey::new(user.public),
            RecoveryKeyEnrollment::Declined,
            HostKeyEnrollment::Absent,
        )
        .unwrap();
        assert!(UnsignedGenesisDidOp::new(
            "did:web:example.com:users:alice",
            Cid::from_raw(b"doc"),
            rotations,
            GenesisRepoHead::EmptyRepo,
        )
        .is_err());
    }

    #[test]
    fn genesis_v1_unsigned_wire_digest_is_frozen() {
        let ed = SigningKey::from_bytes(&[7_u8; 32]);
        let pq = ml_dsa_sk_from_seed(&[8_u8; 32]);
        let user =
            HybridRotationKey::new(ed.verifying_key().to_bytes(), ml_dsa_sk_to_vk_bytes(&pq))
                .unwrap();
        let op = UnsignedGenesisDidOp::new(
            "did:web:alice.acct.example.com",
            Cid::from_raw(b"canonical did document fixture"),
            GenesisRotationKeys::new(
                UserRotationKey::new(user),
                RecoveryKeyEnrollment::Declined,
                HostKeyEnrollment::Absent,
            )
            .unwrap(),
            GenesisRepoHead::EmptyRepo,
        )
        .unwrap();
        let digest: [u8; 32] = Sha256::digest(op.to_dag_cbor().unwrap()).into();
        assert_eq!(
            digest,
            [
                203, 83, 224, 124, 141, 189, 39, 233, 164, 155, 250, 243, 220, 177, 180, 209, 123,
                118, 237, 64, 150, 190, 19, 13, 216, 118, 245, 117, 206, 73, 156, 47,
            ]
        );
    }
}
