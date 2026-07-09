//! `did:at9p` capsule, update-record, and shard-manifest schema.
//!
//! The structures in this module are deliberately just the durable record
//! grammar for #882. They encode through the existing [`crate::dag_cbor`]
//! deterministic codec and every byte-oriented decoder enforces the at9p R3
//! gate: `decode(bytes).encode() == bytes`, otherwise the input is rejected.
//! Signature verification, rotation-chain validation, witness checks, and MAC
//! label interpretation are later gates (#883/#884), not schema work.
//!
//! Codec placement decision: the schema lives in `hyprstream-pds`, beside the
//! single DAG-CBOR implementation it reuses. The resolver/discovery crates
//! already depend on `hyprstream-pds`, so this avoids moving the codec or
//! creating a second CBOR layer while keeping the schema available to both PDS
//! and resolver code.

use std::collections::BTreeSet;

use anyhow::{bail, ensure, Context, Result};
use hyprstream_rpc::cid::{encode_cid, Codec, HashAlgo};

use crate::dag_cbor::DagCbor;

pub const CAPSULE_VERSION: &str = "at9p-capsule/1";
pub const CAPSULE_SIGNATURE_CONTEXT: &str = "at9p-capsule/1";
pub const UPDATE_SIGNATURE_CONTEXT: &str = "at9p-update/1";
pub const MANIFEST_SIGNATURE_CONTEXT: &str = "at9p-manifest/1";

pub const ED25519_PUBLIC_KEY_LEN: usize = 32;
pub const ED25519_SIGNATURE_LEN: usize = 64;
pub const ML_DSA65_PUBLIC_KEY_LEN: usize = 1952;
pub const ML_DSA65_SIGNATURE_LEN: usize = 3309;
pub const H512_LEN: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HybridKeyPair {
    pub ed25519_pub: Vec<u8>,
    pub mldsa65_pub: Vec<u8>,
}

impl HybridKeyPair {
    pub fn new(ed25519_pub: impl Into<Vec<u8>>, mldsa65_pub: impl Into<Vec<u8>>) -> Result<Self> {
        let key = Self {
            ed25519_pub: ed25519_pub.into(),
            mldsa65_pub: mldsa65_pub.into(),
        };
        key.validate()?;
        Ok(key)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.ed25519_pub.len() == ED25519_PUBLIC_KEY_LEN,
            "ed25519 public key must be {ED25519_PUBLIC_KEY_LEN} bytes"
        );
        ensure!(
            self.mldsa65_pub.len() == ML_DSA65_PUBLIC_KEY_LEN,
            "ML-DSA-65 public key must be {ML_DSA65_PUBLIC_KEY_LEN} bytes"
        );
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("ed25519Pub", DagCbor::Bytes(self.ed25519_pub.clone())),
            ("mldsa65Pub", DagCbor::Bytes(self.mldsa65_pub.clone())),
        ])
    }

    /// KERI pre-rotation commitment digest for this keypair: `BLAKE3-512` over
    /// the canonical DAG-CBOR of the hybrid keypair (both the Ed25519 and
    /// ML-DSA-65 public keys). This is the value a capsule author places in
    /// `next_key_commitments` to pre-commit the *next* key set, and the value
    /// the B1 successor-check ([`crate::at9p_chain`]) recomputes from an
    /// update-record's revealed signing key to prove it was pre-committed.
    ///
    /// Hashing both components together binds them: an attacker cannot swap one
    /// half of the hybrid pair without changing the commitment.
    pub fn commitment_digest(&self) -> [u8; H512_LEN] {
        h512(&self.to_value().encode())
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(value, &["ed25519Pub", "mldsa65Pub"], "at9p hybrid keypair")?;
        Self::new(
            required_bytes(value, "ed25519Pub", "at9p hybrid keypair")?.to_vec(),
            required_bytes(value, "mldsa65Pub", "at9p hybrid keypair")?.to_vec(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceType {
    NinePExport,
    AtprotoPds,
}

impl ServiceType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NinePExport => "NinePExport",
            Self::AtprotoPds => "AtprotoPDS",
        }
    }

    fn parse(s: &str) -> Result<Self> {
        match s {
            "NinePExport" => Ok(Self::NinePExport),
            "AtprotoPDS" => Ok(Self::AtprotoPds),
            other => bail!("unknown at9p service type {other:?}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Transport {
    Iroh,
    Quic,
    Moq,
    Https,
}

impl Transport {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Iroh => "iroh",
            Self::Quic => "quic",
            Self::Moq => "moq",
            Self::Https => "https",
        }
    }

    fn parse(s: &str) -> Result<Self> {
        match s {
            "iroh" => Ok(Self::Iroh),
            "quic" => Ok(Self::Quic),
            "moq" => Ok(Self::Moq),
            "https" => Ok(Self::Https),
            other => bail!("unknown at9p endpoint transport {other:?}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServiceEndpoint {
    pub transport: Transport,
    pub address: String,
    pub node_id: Option<String>,
    pub relay: Option<String>,
    pub export: Option<String>,
}

impl ServiceEndpoint {
    pub fn new(transport: Transport, address: impl Into<String>) -> Result<Self> {
        let endpoint = Self {
            transport,
            address: address.into(),
            node_id: None,
            relay: None,
            export: None,
        };
        endpoint.validate()?;
        Ok(endpoint)
    }

    pub fn validate(&self) -> Result<()> {
        validate_nonempty_no_ws(&self.address, "service endpoint address")?;
        validate_optional_no_ws(self.node_id.as_deref(), "service endpoint nodeId")?;
        validate_optional_no_ws(self.relay.as_deref(), "service endpoint relay")?;
        validate_optional_no_ws(self.export.as_deref(), "service endpoint export")?;
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        let mut fields = vec![
            ("address", DagCbor::Text(self.address.clone())),
            (
                "transport",
                DagCbor::Text(self.transport.as_str().to_owned()),
            ),
        ];
        if let Some(node_id) = &self.node_id {
            fields.push(("nodeId", DagCbor::Text(node_id.clone())));
        }
        if let Some(relay) = &self.relay {
            fields.push(("relay", DagCbor::Text(relay.clone())));
        }
        if let Some(export) = &self.export {
            fields.push(("export", DagCbor::Text(export.clone())));
        }
        DagCbor::str_map(fields)
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(
            value,
            &["address", "transport", "nodeId", "relay", "export"],
            "at9p service endpoint",
        )?;
        let endpoint = Self {
            transport: Transport::parse(required_str(
                value,
                "transport",
                "at9p service endpoint",
            )?)?,
            address: required_str(value, "address", "at9p service endpoint")?.to_owned(),
            node_id: optional_str(value, "nodeId")?.map(str::to_owned),
            relay: optional_str(value, "relay")?.map(str::to_owned),
            export: optional_str(value, "export")?.map(str::to_owned),
        };
        endpoint.validate()?;
        Ok(endpoint)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServiceEntry {
    pub id: String,
    pub service_type: ServiceType,
    pub endpoint: ServiceEndpoint,
}

impl ServiceEntry {
    pub fn new(
        id: impl Into<String>,
        service_type: ServiceType,
        endpoint: ServiceEndpoint,
    ) -> Result<Self> {
        let entry = Self {
            id: id.into(),
            service_type,
            endpoint,
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<()> {
        validate_service_id(&self.id)?;
        self.endpoint.validate()
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("endpoint", self.endpoint.to_value()),
            ("id", DagCbor::Text(self.id.clone())),
            ("type", DagCbor::Text(self.service_type.as_str().to_owned())),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(value, &["endpoint", "id", "type"], "at9p service entry")?;
        Self::new(
            required_str(value, "id", "at9p service entry")?.to_owned(),
            ServiceType::parse(required_str(value, "type", "at9p service entry")?)?,
            ServiceEndpoint::from_value(required(value, "endpoint", "at9p service entry")?)?,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Delegation {
    pub id: String,
    pub target: String,
    pub capabilities: Vec<String>,
}

impl Delegation {
    pub fn new(
        id: impl Into<String>,
        target: impl Into<String>,
        capabilities: Vec<String>,
    ) -> Result<Self> {
        let delegation = Self {
            id: id.into(),
            target: target.into(),
            capabilities,
        };
        delegation.validate()?;
        Ok(delegation)
    }

    pub fn validate(&self) -> Result<()> {
        validate_nonempty_no_ws(&self.id, "delegation id")?;
        validate_did_or_endpoint(&self.target, "delegation target")?;
        ensure!(
            !self.capabilities.is_empty(),
            "delegation capabilities must not be empty"
        );
        validate_unique_texts(&self.capabilities, "delegation capabilities")?;
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            (
                "capabilities",
                DagCbor::list(self.capabilities.iter().cloned().map(DagCbor::Text)),
            ),
            ("id", DagCbor::Text(self.id.clone())),
            ("target", DagCbor::Text(self.target.clone())),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(value, &["capabilities", "id", "target"], "at9p delegation")?;
        let capabilities = required(value, "capabilities", "at9p delegation")?
            .as_list()?
            .iter()
            .map(|item| Ok(item.as_str()?.to_owned()))
            .collect::<Result<Vec<_>>>()?;
        Self::new(
            required_str(value, "id", "at9p delegation")?.to_owned(),
            required_str(value, "target", "at9p delegation")?.to_owned(),
            capabilities,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoseCompositeSignature {
    pub context: String,
    pub protected: Vec<u8>,
    pub ed25519_signature: Vec<u8>,
    pub mldsa65_signature: Vec<u8>,
}

impl CoseCompositeSignature {
    pub fn new(
        context: impl Into<String>,
        protected: impl Into<Vec<u8>>,
        ed25519_signature: impl Into<Vec<u8>>,
        mldsa65_signature: impl Into<Vec<u8>>,
    ) -> Result<Self> {
        let sig = Self {
            context: context.into(),
            protected: protected.into(),
            ed25519_signature: ed25519_signature.into(),
            mldsa65_signature: mldsa65_signature.into(),
        };
        sig.validate()?;
        Ok(sig)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.context.as_str(),
                CAPSULE_SIGNATURE_CONTEXT | UPDATE_SIGNATURE_CONTEXT | MANIFEST_SIGNATURE_CONTEXT
            ),
            "unsupported at9p COSE context {:?}",
            self.context
        );
        ensure!(
            !self.protected.is_empty(),
            "COSE protected header is required"
        );
        ensure!(
            self.ed25519_signature.len() == ED25519_SIGNATURE_LEN,
            "Ed25519 signature must be {ED25519_SIGNATURE_LEN} bytes"
        );
        ensure!(
            self.mldsa65_signature.len() == ML_DSA65_SIGNATURE_LEN,
            "ML-DSA-65 signature must be {ML_DSA65_SIGNATURE_LEN} bytes"
        );
        Ok(())
    }

    fn ensure_context(&self, expected: &str) -> Result<()> {
        ensure!(
            self.context == expected,
            "at9p signature context mismatch: expected {expected:?}, got {:?}",
            self.context
        );
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("context", DagCbor::Text(self.context.clone())),
            ("ed25519", DagCbor::Bytes(self.ed25519_signature.clone())),
            ("mldsa65", DagCbor::Bytes(self.mldsa65_signature.clone())),
            ("protected", DagCbor::Bytes(self.protected.clone())),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(
            value,
            &["context", "protected", "ed25519", "mldsa65"],
            "at9p composite signature",
        )?;
        Self::new(
            required_str(value, "context", "at9p composite signature")?.to_owned(),
            required_bytes(value, "protected", "at9p composite signature")?.to_vec(),
            required_bytes(value, "ed25519", "at9p composite signature")?.to_vec(),
            required_bytes(value, "mldsa65", "at9p composite signature")?.to_vec(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapsuleBody {
    pub version: String,
    pub subject_keys: Vec<HybridKeyPair>,
    pub next_key_commitments: Vec<[u8; H512_LEN]>,
    pub services: Vec<ServiceEntry>,
    pub label_hints: Option<Vec<String>>,
    pub delegations: Option<Vec<Delegation>>,
    pub witnesses: Option<Vec<String>>,
}

impl CapsuleBody {
    pub fn new(subject_keys: Vec<HybridKeyPair>, services: Vec<ServiceEntry>) -> Result<Self> {
        let body = Self {
            version: CAPSULE_VERSION.to_owned(),
            subject_keys,
            next_key_commitments: Vec::new(),
            services,
            label_hints: None,
            delegations: None,
            witnesses: None,
        };
        body.validate()?;
        Ok(body)
    }

    /// Canonical DAG-CBOR bytes of the capsule body.
    ///
    /// This is the signed payload for an at9p capsule: the composite signature
    /// (see [`crate::at9p_sign`]) covers exactly these bytes, never the
    /// enclosing `signatures` field.
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == CAPSULE_VERSION,
            "unsupported at9p capsule version {:?}",
            self.version
        );
        ensure!(
            !self.subject_keys.is_empty(),
            "capsule subjectKeys must not be empty"
        );
        for key in &self.subject_keys {
            key.validate()?;
        }
        ensure!(
            !self.services.is_empty(),
            "capsule services must not be empty"
        );
        validate_unique_services(&self.services)?;
        for service in &self.services {
            service.validate()?;
        }
        if let Some(label_hints) = &self.label_hints {
            ensure!(
                !label_hints.is_empty(),
                "labelHints must not be empty when present"
            );
            validate_unique_texts(label_hints, "labelHints")?;
        }
        if let Some(delegations) = &self.delegations {
            ensure!(
                !delegations.is_empty(),
                "delegations must not be empty when present"
            );
            for delegation in delegations {
                delegation.validate()?;
            }
        }
        if let Some(witnesses) = &self.witnesses {
            ensure!(
                !witnesses.is_empty(),
                "witnesses must not be empty when present"
            );
            validate_unique_texts(witnesses, "witnesses")?;
            for witness in witnesses {
                validate_did_or_endpoint(witness, "witness")?;
            }
        }
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        let service_map = DagCbor::str_map(
            self.services
                .iter()
                .map(|service| (service.id.clone(), service.to_value())),
        );
        let mut fields = vec![
            (
                "nextKeyCommitments",
                DagCbor::list(
                    self.next_key_commitments
                        .iter()
                        .map(|digest| DagCbor::Bytes(digest.to_vec())),
                ),
            ),
            ("services", service_map),
            (
                "subjectKeys",
                DagCbor::list(self.subject_keys.iter().map(HybridKeyPair::to_value)),
            ),
            ("version", DagCbor::Text(self.version.clone())),
        ];
        if let Some(label_hints) = &self.label_hints {
            fields.push((
                "labelHints",
                DagCbor::list(label_hints.iter().cloned().map(DagCbor::Text)),
            ));
        }
        if let Some(delegations) = &self.delegations {
            fields.push((
                "delegations",
                DagCbor::list(delegations.iter().map(Delegation::to_value)),
            ));
        }
        if let Some(witnesses) = &self.witnesses {
            fields.push((
                "witnesses",
                DagCbor::list(witnesses.iter().cloned().map(DagCbor::Text)),
            ));
        }
        DagCbor::str_map(fields)
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(
            value,
            &[
                "version",
                "subjectKeys",
                "nextKeyCommitments",
                "services",
                "labelHints",
                "delegations",
                "witnesses",
            ],
            "at9p capsule body",
        )?;
        let subject_keys = required(value, "subjectKeys", "at9p capsule body")?
            .as_list()?
            .iter()
            .map(HybridKeyPair::from_value)
            .collect::<Result<Vec<_>>>()?;
        let next_key_commitments = required(value, "nextKeyCommitments", "at9p capsule body")?
            .as_list()?
            .iter()
            .map(|digest| h512_from_slice(digest.as_bytes()?))
            .collect::<Result<Vec<_>>>()?;
        let services = parse_services(required(value, "services", "at9p capsule body")?)?;
        let label_hints = optional_text_list(value, "labelHints")?;
        let delegations = match value.get("delegations") {
            Some(v) => Some(
                v.as_list()?
                    .iter()
                    .map(Delegation::from_value)
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => None,
        };
        let witnesses = optional_text_list(value, "witnesses")?;
        let body = Self {
            version: required_str(value, "version", "at9p capsule body")?.to_owned(),
            subject_keys,
            next_key_commitments,
            services,
            label_hints,
            delegations,
            witnesses,
        };
        body.validate()?;
        Ok(body)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Capsule {
    pub body: CapsuleBody,
    pub signatures: CoseCompositeSignature,
}

impl Capsule {
    pub fn new(body: CapsuleBody, signatures: CoseCompositeSignature) -> Result<Self> {
        signatures.ensure_context(CAPSULE_SIGNATURE_CONTEXT)?;
        body.validate()?;
        Ok(Self { body, signatures })
    }

    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&decode_canonical(bytes, "at9p capsule")?)
    }

    pub fn cid512(&self) -> Result<String> {
        at9p_capsule_cid512(&self.to_dag_cbor())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("body", self.body.to_value()),
            ("signatures", self.signatures.to_value()),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(value, &["body", "signatures"], "at9p capsule")?;
        Self::new(
            CapsuleBody::from_value(required(value, "body", "at9p capsule")?)?,
            CoseCompositeSignature::from_value(required(value, "signatures", "at9p capsule")?)?,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UpdateRecord {
    pub subject_cid512: String,
    pub epoch: u64,
    pub prev_record_digest: [u8; H512_LEN],
    pub new_capsule_body: CapsuleBody,
    pub expires_at: String,
    pub signatures: CoseCompositeSignature,
}

impl UpdateRecord {
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&decode_canonical(bytes, "at9p update record")?)
    }

    /// Canonical DAG-CBOR bytes of the update record *without* its `signatures`
    /// field — the payload the composite signature covers (see
    /// [`crate::at9p_sign`]).
    pub fn signable_bytes(&self) -> Vec<u8> {
        self.signable_value().encode()
    }

    fn signable_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("epoch", DagCbor::Unsigned(self.epoch)),
            ("expiresAt", DagCbor::Text(self.expires_at.clone())),
            ("newCapsuleBody", self.new_capsule_body.to_value()),
            (
                "prevRecordDigest",
                DagCbor::Bytes(self.prev_record_digest.to_vec()),
            ),
            ("subjectCid512", DagCbor::Text(self.subject_cid512.clone())),
        ])
    }

    fn validate(&self) -> Result<()> {
        validate_cid512(&self.subject_cid512)?;
        validate_datetime(&self.expires_at)?;
        self.new_capsule_body.validate()?;
        self.signatures.ensure_context(UPDATE_SIGNATURE_CONTEXT)?;
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("epoch", DagCbor::Unsigned(self.epoch)),
            ("expiresAt", DagCbor::Text(self.expires_at.clone())),
            ("newCapsuleBody", self.new_capsule_body.to_value()),
            (
                "prevRecordDigest",
                DagCbor::Bytes(self.prev_record_digest.to_vec()),
            ),
            ("signatures", self.signatures.to_value()),
            ("subjectCid512", DagCbor::Text(self.subject_cid512.clone())),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(
            value,
            &[
                "subjectCid512",
                "epoch",
                "prevRecordDigest",
                "newCapsuleBody",
                "expiresAt",
                "signatures",
            ],
            "at9p update record",
        )?;
        let record = Self {
            subject_cid512: required_str(value, "subjectCid512", "at9p update record")?.to_owned(),
            epoch: required(value, "epoch", "at9p update record")?.as_unsigned()?,
            prev_record_digest: h512_from_slice(required_bytes(
                value,
                "prevRecordDigest",
                "at9p update record",
            )?)?,
            new_capsule_body: CapsuleBody::from_value(required(
                value,
                "newCapsuleBody",
                "at9p update record",
            )?)?,
            expires_at: required_str(value, "expiresAt", "at9p update record")?.to_owned(),
            signatures: CoseCompositeSignature::from_value(required(
                value,
                "signatures",
                "at9p update record",
            )?)?,
        };
        record.validate()?;
        Ok(record)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardEntry {
    pub index: u64,
    pub shard_len: u64,
    pub shard_digest: [u8; H512_LEN],
}

impl ShardEntry {
    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("index", DagCbor::Unsigned(self.index)),
            ("shardDigest", DagCbor::Bytes(self.shard_digest.to_vec())),
            ("shardLen", DagCbor::Unsigned(self.shard_len)),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(
            value,
            &["index", "shardLen", "shardDigest"],
            "at9p shard entry",
        )?;
        Ok(Self {
            index: required(value, "index", "at9p shard entry")?.as_unsigned()?,
            shard_len: required(value, "shardLen", "at9p shard entry")?.as_unsigned()?,
            shard_digest: h512_from_slice(required_bytes(
                value,
                "shardDigest",
                "at9p shard entry",
            )?)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    pub subject_cid512: String,
    pub codec: String,
    pub total_len: u64,
    pub shards: Vec<ShardEntry>,
    pub signatures: CoseCompositeSignature,
}

impl Manifest {
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&decode_canonical(bytes, "at9p manifest")?)
    }

    fn validate(&self) -> Result<()> {
        validate_cid512(&self.subject_cid512)?;
        validate_nonempty_no_ws(&self.codec, "manifest codec")?;
        ensure!(!self.shards.is_empty(), "manifest shards must not be empty");
        let mut expected = 0u64;
        let mut summed_len = 0u64;
        for shard in &self.shards {
            ensure!(
                shard.index == expected,
                "manifest shard index must be contiguous: expected {expected}, got {}",
                shard.index
            );
            ensure!(shard.shard_len > 0, "manifest shardLen must be nonzero");
            summed_len = summed_len
                .checked_add(shard.shard_len)
                .context("manifest shard lengths overflow")?;
            expected = expected
                .checked_add(1)
                .context("manifest shard index overflow")?;
        }
        ensure!(
            summed_len == self.total_len,
            "manifest totalLen {total_len} does not equal shard sum {summed_len}",
            total_len = self.total_len
        );
        self.signatures.ensure_context(MANIFEST_SIGNATURE_CONTEXT)?;
        Ok(())
    }

    fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("codec", DagCbor::Text(self.codec.clone())),
            (
                "shards",
                DagCbor::list(self.shards.iter().map(ShardEntry::to_value)),
            ),
            ("signatures", self.signatures.to_value()),
            ("subjectCid512", DagCbor::Text(self.subject_cid512.clone())),
            ("totalLen", DagCbor::Unsigned(self.total_len)),
        ])
    }

    fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown(
            value,
            &["subjectCid512", "codec", "totalLen", "shards", "signatures"],
            "at9p manifest",
        )?;
        let manifest = Self {
            subject_cid512: required_str(value, "subjectCid512", "at9p manifest")?.to_owned(),
            codec: required_str(value, "codec", "at9p manifest")?.to_owned(),
            total_len: required(value, "totalLen", "at9p manifest")?.as_unsigned()?,
            shards: required(value, "shards", "at9p manifest")?
                .as_list()?
                .iter()
                .map(ShardEntry::from_value)
                .collect::<Result<Vec<_>>>()?,
            signatures: CoseCompositeSignature::from_value(required(
                value,
                "signatures",
                "at9p manifest",
            )?)?,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Shard {
    pub subject_cid512: String,
    pub index: u64,
    pub shard_bytes: Vec<u8>,
}

impl Shard {
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        DagCbor::str_map([
            ("index", DagCbor::Unsigned(self.index)),
            ("shardBytes", DagCbor::Bytes(self.shard_bytes.clone())),
            ("subjectCid512", DagCbor::Text(self.subject_cid512.clone())),
        ])
        .encode()
    }

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = decode_canonical(bytes, "at9p shard")?;
        reject_unknown(
            &value,
            &["subjectCid512", "index", "shardBytes"],
            "at9p shard",
        )?;
        let shard = Self {
            subject_cid512: required_str(&value, "subjectCid512", "at9p shard")?.to_owned(),
            index: required(&value, "index", "at9p shard")?.as_unsigned()?,
            shard_bytes: required_bytes(&value, "shardBytes", "at9p shard")?.to_vec(),
        };
        validate_cid512(&shard.subject_cid512)?;
        Ok(shard)
    }
}

/// BLAKE3 extended to a 512-bit (`H512_LEN`-byte) output — the at9p digest
/// primitive (`H512` in design #879 §5). Used for capsule CIDs, update-record
/// linkage digests (`prev-record-digest`), and key commitments.
pub fn h512(bytes: &[u8]) -> [u8; H512_LEN] {
    let mut digest = [0u8; H512_LEN];
    let mut hasher = blake3::Hasher::new();
    hasher.update(bytes);
    hasher.finalize_xof().fill(&mut digest);
    digest
}

pub fn at9p_capsule_cid512(canonical_capsule_bytes: &[u8]) -> Result<String> {
    let digest = h512(canonical_capsule_bytes);
    encode_cid(Codec::At9pCapsule, HashAlgo::Blake3, &digest)
}

fn decode_canonical(bytes: &[u8], record_type: &str) -> Result<DagCbor> {
    let value = DagCbor::decode(bytes).with_context(|| format!("{record_type}: decode failed"))?;
    ensure!(
        value.encode() == bytes,
        "{record_type}: non-canonical DAG-CBOR encoding"
    );
    Ok(value)
}

fn required<'a>(value: &'a DagCbor, key: &str, nsid: &str) -> Result<&'a DagCbor> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{nsid}: missing required field {key:?}"))
}

fn required_str<'a>(value: &'a DagCbor, key: &str, nsid: &str) -> Result<&'a str> {
    required(value, key, nsid)?.as_str()
}

fn required_bytes<'a>(value: &'a DagCbor, key: &str, nsid: &str) -> Result<&'a [u8]> {
    required(value, key, nsid)?.as_bytes()
}

fn optional_str<'a>(value: &'a DagCbor, key: &str) -> Result<Option<&'a str>> {
    match value.get(key) {
        Some(v) => Ok(Some(v.as_str()?)),
        None => Ok(None),
    }
}

fn optional_text_list(value: &DagCbor, key: &str) -> Result<Option<Vec<String>>> {
    value
        .get(key)
        .map(|v| {
            v.as_list()?
                .iter()
                .map(|item| Ok(item.as_str()?.to_owned()))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()
}

fn reject_unknown(value: &DagCbor, allowed: &[&str], nsid: &str) -> Result<()> {
    for (k, _) in value.as_map()? {
        let key = k.as_str()?;
        ensure!(allowed.contains(&key), "{nsid}: unknown field {key:?}");
    }
    Ok(())
}

fn parse_services(value: &DagCbor) -> Result<Vec<ServiceEntry>> {
    let mut services = Vec::new();
    for (key, entry) in value.as_map()? {
        let id = key.as_str()?;
        validate_service_id(id)?;
        let service = ServiceEntry::from_value(entry)?;
        ensure!(
            service.id == id,
            "service map key {id:?} does not match entry id {:?}",
            service.id
        );
        services.push(service);
    }
    Ok(services)
}

fn h512_from_slice(bytes: &[u8]) -> Result<[u8; H512_LEN]> {
    ensure!(
        bytes.len() == H512_LEN,
        "H512 digest must be {H512_LEN} bytes, got {}",
        bytes.len()
    );
    let mut digest = [0u8; H512_LEN];
    digest.copy_from_slice(bytes);
    Ok(digest)
}

fn validate_service_id(s: &str) -> Result<()> {
    ensure!(s.starts_with('#'), "service id must start with '#': {s:?}");
    ensure!(s.len() > 1, "service id must include a token: {s:?}");
    ensure!(
        s.as_bytes()[1..]
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.')),
        "service id token contains invalid characters: {s:?}"
    );
    Ok(())
}

fn validate_nonempty_no_ws(s: &str, field: &str) -> Result<()> {
    ensure!(!s.is_empty(), "{field} must not be empty");
    ensure!(
        !s.chars().any(char::is_whitespace),
        "{field} must not contain whitespace"
    );
    Ok(())
}

fn validate_optional_no_ws(value: Option<&str>, field: &str) -> Result<()> {
    if let Some(value) = value {
        validate_nonempty_no_ws(value, field)?;
    }
    Ok(())
}

fn validate_did_or_endpoint(s: &str, field: &str) -> Result<()> {
    validate_nonempty_no_ws(s, field)?;
    ensure!(
        s.starts_with("did:") || s.starts_with("https://") || s.starts_with("iroh:"),
        "{field} must be a did: identifier or endpoint URI"
    );
    Ok(())
}

fn validate_datetime(s: &str) -> Result<()> {
    ensure!(s.ends_with('Z'), "datetime must end with 'Z': {s:?}");
    ensure!(s.len() >= 20, "datetime too short: {s:?}");
    ensure!(
        s.as_bytes().get(4) == Some(&b'-')
            && s.as_bytes().get(7) == Some(&b'-')
            && s.as_bytes().get(10) == Some(&b'T')
            && s.as_bytes().get(13) == Some(&b':')
            && s.as_bytes().get(16) == Some(&b':'),
        "datetime must be ISO-8601 UTC: {s:?}"
    );
    Ok(())
}

fn validate_cid512(s: &str) -> Result<()> {
    let cid = hyprstream_rpc::cid::decode_cid(s)?;
    ensure!(
        cid.codec == Codec::At9pCapsule,
        "cid512 must use at9p-capsule multicodec"
    );
    ensure!(
        cid.multihash.algo == HashAlgo::Blake3 && cid.multihash.digest.len() == H512_LEN,
        "cid512 must use BLAKE3-512 multihash"
    );
    Ok(())
}

fn validate_unique_services(services: &[ServiceEntry]) -> Result<()> {
    let mut ids = BTreeSet::new();
    for service in services {
        ensure!(
            ids.insert(service.id.as_str()),
            "duplicate service id {:?}",
            service.id
        );
    }
    Ok(())
}

fn validate_unique_texts(values: &[String], field: &str) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        validate_nonempty_no_ws(value, field)?;
        ensure!(
            seen.insert(value.as_str()),
            "duplicate {field} entry {value:?}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]

    use proptest::prelude::*;

    use super::*;

    fn bytes(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    fn digest(seed: u8) -> [u8; H512_LEN] {
        let mut out = [0u8; H512_LEN];
        for (i, b) in out.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
        out
    }

    fn signature(context: &str, seed: u8) -> CoseCompositeSignature {
        CoseCompositeSignature::new(
            context,
            bytes(8, seed),
            bytes(ED25519_SIGNATURE_LEN, seed.wrapping_add(1)),
            bytes(ML_DSA65_SIGNATURE_LEN, seed.wrapping_add(2)),
        )
        .unwrap()
    }

    fn key(seed: u8) -> HybridKeyPair {
        HybridKeyPair::new(
            bytes(ED25519_PUBLIC_KEY_LEN, seed),
            bytes(ML_DSA65_PUBLIC_KEY_LEN, seed.wrapping_add(16)),
        )
        .unwrap()
    }

    fn service(id: String, service_type: ServiceType, seed: u8) -> ServiceEntry {
        let mut endpoint =
            ServiceEndpoint::new(Transport::Iroh, format!("iroh://node{seed}")).unwrap();
        endpoint.node_id = Some(format!("node{seed}"));
        endpoint.relay = Some(format!("https://relay{seed}.example"));
        endpoint.export = Some(format!("/exports/{seed}"));
        ServiceEntry::new(id, service_type, endpoint).unwrap()
    }

    fn sample_capsule(seed: u8) -> Capsule {
        let mut body = CapsuleBody::new(
            vec![key(seed)],
            vec![
                service("#ns".to_owned(), ServiceType::NinePExport, seed),
                service(
                    "#pds".to_owned(),
                    ServiceType::AtprotoPds,
                    seed.wrapping_add(1),
                ),
            ],
        )
        .unwrap();
        body.next_key_commitments = vec![digest(seed.wrapping_add(3))];
        body.label_hints = Some(vec![format!("tenant{seed}")]);
        body.witnesses = Some(vec![format!("did:at9p:witness{seed}")]);
        Capsule::new(body, signature(CAPSULE_SIGNATURE_CONTEXT, seed)).unwrap()
    }

    fn arb_capsule() -> impl Strategy<Value = Capsule> {
        (0u8..=200).prop_map(sample_capsule)
    }

    proptest! {
        #[test]
        fn capsule_round_trip_is_byte_stable(capsule in arb_capsule()) {
            let bytes = capsule.to_dag_cbor();
            let decoded = Capsule::from_dag_cbor(&bytes).unwrap();
            prop_assert_eq!(&decoded, &capsule);
            prop_assert_eq!(decoded.to_dag_cbor(), bytes);
            prop_assert!(decoded.cid512().unwrap().starts_with('b'));
        }

        #[test]
        fn update_record_round_trip_is_byte_stable(capsule in arb_capsule(), epoch in 0u64..1000) {
            let subject_cid512 = capsule.cid512().unwrap();
            let record = UpdateRecord {
                subject_cid512,
                epoch,
                prev_record_digest: digest(42),
                new_capsule_body: capsule.body,
                expires_at: "2026-07-08T12:00:00Z".to_owned(),
                signatures: signature(UPDATE_SIGNATURE_CONTEXT, 7),
            };
            let bytes = record.to_dag_cbor();
            let decoded = UpdateRecord::from_dag_cbor(&bytes).unwrap();
            prop_assert_eq!(&decoded, &record);
            prop_assert_eq!(decoded.to_dag_cbor(), bytes);
        }

        #[test]
        fn manifest_round_trip_is_byte_stable(capsule in arb_capsule(), shard_len in 1u64..4096) {
            let manifest = Manifest {
                subject_cid512: capsule.cid512().unwrap(),
                codec: "dag-cbor".to_owned(),
                total_len: shard_len,
                shards: vec![ShardEntry {
                    index: 0,
                    shard_len,
                    shard_digest: digest(9),
                }],
                signatures: signature(MANIFEST_SIGNATURE_CONTEXT, 10),
            };
            let bytes = manifest.to_dag_cbor();
            let decoded = Manifest::from_dag_cbor(&bytes).unwrap();
            prop_assert_eq!(&decoded, &manifest);
            prop_assert_eq!(decoded.to_dag_cbor(), bytes);
        }
    }

    #[test]
    fn capsule_rejects_unsorted_map_keys() {
        let good = sample_capsule(1).to_dag_cbor();
        let decoded = DagCbor::decode(&good).unwrap();
        let DagCbor::Map(mut pairs) = decoded else {
            panic!("capsule must encode as map");
        };
        pairs.reverse();
        let bad = DagCbor::Map(pairs).encode();
        assert!(Capsule::from_dag_cbor(&bad).is_err());
    }

    #[test]
    fn capsule_rejects_duplicate_map_keys() {
        let bad = DagCbor::Map(vec![
            (
                DagCbor::Text("body".to_owned()),
                sample_capsule(1).body.to_value(),
            ),
            (
                DagCbor::Text("body".to_owned()),
                sample_capsule(2).body.to_value(),
            ),
            (
                DagCbor::Text("signatures".to_owned()),
                signature(CAPSULE_SIGNATURE_CONTEXT, 1).to_value(),
            ),
        ])
        .encode();
        assert!(Capsule::from_dag_cbor(&bad).is_err());
    }

    #[test]
    fn update_rejects_non_minimal_ints() {
        let capsule = sample_capsule(5);
        let record = UpdateRecord {
            subject_cid512: capsule.cid512().unwrap(),
            epoch: 23,
            prev_record_digest: digest(44),
            new_capsule_body: capsule.body,
            expires_at: "2026-07-08T12:00:00Z".to_owned(),
            signatures: signature(UPDATE_SIGNATURE_CONTEXT, 8),
        };
        let mut bad = record.to_dag_cbor();
        let pos = bad
            .windows(5)
            .position(|w| w == b"epoch")
            .expect("epoch key present")
            + 5;
        assert_eq!(bad[pos], 23);
        bad.splice(pos..pos + 1, [0x18, 23]);
        assert!(UpdateRecord::from_dag_cbor(&bad).is_err());
    }

    #[test]
    fn capsule_cid512_uses_rpc_at9p_blake3_512_cid_api() {
        let capsule = sample_capsule(3);
        let cid = capsule.cid512().unwrap();
        let decoded = hyprstream_rpc::cid::decode_cid(&cid).unwrap();
        assert_eq!(decoded.codec, Codec::At9pCapsule);
        assert_eq!(decoded.multihash.algo, HashAlgo::Blake3);
        assert_eq!(decoded.multihash.digest.len(), H512_LEN);
    }
}
