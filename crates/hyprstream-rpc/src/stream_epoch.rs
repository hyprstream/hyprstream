//! Identified point-to-point stream epoch cryptography (#554).
//!
//! This module is deliberately carrier-neutral.  A stock MOQT relay forwards the
//! resulting encrypted Object bytes without learning this transcript or any key.
//! Network key release remains gated by #726; the types here make the authority
//! input explicit without implementing the restricted-anonymous work in
//! #1060-#1062.

use anyhow::{ensure, Result};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use crate::crypto::backend::{derive_key, keyed_mac};
use crate::crypto::hybrid_kem::SuiteId;

/// The only stream KEM suite accepted by this profile.
pub const IDENTIFIED_STREAM_SUITE: SuiteId = SuiteId::HyKemX25519MlKem768;
/// Protocol/domain version included in every canonical transcript.
pub const IDENTIFIED_STREAM_VERSION: u16 = 1;

const SESSION_ROOT_DOMAIN: &str = "hyprstream identified-stream session-root v1";
const BINDING_HASH_DOMAIN: &str = "hyprstream identified-stream binding-hash v1";
const ROUTE_TOPIC_DOMAIN: &str = "hyprstream identified-stream route-topic v1";
const CONTROL_TOPIC_DOMAIN: &str = "hyprstream identified-stream control-topic v1";
const EPOCH_SECRET_DOMAIN: &str = "hyprstream identified-stream epoch-ratchet v1";
const DATA_MAC_DOMAIN: &str = "hyprstream identified-stream data-mac v1";
const DATA_AEAD_DOMAIN: &str = "hyprstream identified-stream data-aead v1";
const CONTROL_MAC_DOMAIN: &str = "hyprstream identified-stream control-mac v1";
const REKEY_AUTH_DOMAIN: &str = "hyprstream identified-stream rekey-auth v1";
const NONCE_DOMAIN: &str = "hyprstream identified-stream nonce-domain v1";
const EPOCH_NAMESPACE_DOMAIN: &str = "hyprstream identified-stream epoch-namespace v1";

const MAX_TEXT: usize = 1024;
const MAX_CAPABILITY: usize = 4096;

/// Carrier profile selected by signed resolution/admission policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamCarrierProfile {
    /// Hyprstream owns the endpoint and requires its hybrid transport policy.
    OwnedHybridTransport = 1,
    /// A standards-compatible relay is an explicitly untrusted classical carrier.
    StandardPublicRelay = 2,
}

/// The selected route's role in the stream transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamRouteRole {
    Direct = 1,
    Relay = 2,
}

/// Direction is a key and nonce domain, never merely descriptive metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamDirection {
    ProducerToConsumer = 1,
    ConsumerToProducer = 2,
}

/// Accepted-current identity state frozen into the stream authorization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamAcceptedState {
    pub identity_did: String,
    pub digest: [u8; 64],
    pub epoch: u64,
}

impl StreamAcceptedState {
    fn validate(&self, label: &str) -> Result<()> {
        validate_did(&self.identity_did, label)?;
        ensure!(
            self.digest.iter().any(|byte| *byte != 0),
            "{label} accepted-state digest is all zero"
        );
        Ok(())
    }
}

/// Complete identified-session binding consumed by HyKEM and every epoch KDF.
///
/// `consumer_state.identity_did` is the identified principal.  Anonymous key
/// release intentionally has no constructor in this slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdentifiedStreamBinding {
    carrier_profile: StreamCarrierProfile,
    route_role: StreamRouteRole,
    producer_endpoint_did: String,
    consumer_endpoint_did: String,
    producer_state: StreamAcceptedState,
    consumer_state: StreamAcceptedState,
    service: String,
    capability: String,
    track: String,
    producer_kem_key_id: String,
    consumer_kem_key_id: String,
    max_blocks_per_epoch: u64,
    suite: SuiteId,
}

impl IdentifiedStreamBinding {
    /// Build the pinned identified profile.  Suite negotiation is intentionally
    /// absent: callers cannot request a classical-only or unknown suite.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        carrier_profile: StreamCarrierProfile,
        route_role: StreamRouteRole,
        producer_endpoint_did: impl Into<String>,
        consumer_endpoint_did: impl Into<String>,
        producer_state: StreamAcceptedState,
        consumer_state: StreamAcceptedState,
        service: impl Into<String>,
        capability: impl Into<String>,
        track: impl Into<String>,
        producer_kem_key_id: impl Into<String>,
        consumer_kem_key_id: impl Into<String>,
        max_blocks_per_epoch: u64,
    ) -> Result<Self> {
        let binding = Self {
            carrier_profile,
            route_role,
            producer_endpoint_did: producer_endpoint_did.into(),
            consumer_endpoint_did: consumer_endpoint_did.into(),
            producer_state,
            consumer_state,
            service: service.into(),
            capability: capability.into(),
            track: track.into(),
            producer_kem_key_id: producer_kem_key_id.into(),
            consumer_kem_key_id: consumer_kem_key_id.into(),
            max_blocks_per_epoch,
            suite: IDENTIFIED_STREAM_SUITE,
        };
        binding.validate()?;
        Ok(binding)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.suite == IDENTIFIED_STREAM_SUITE,
            "identified stream requires the pinned X25519+ML-KEM-768 suite"
        );
        validate_did(&self.producer_endpoint_did, "producer endpoint")?;
        validate_did(&self.consumer_endpoint_did, "consumer endpoint")?;
        self.producer_state.validate("producer")?;
        self.consumer_state.validate("consumer")?;
        validate_text(&self.service, "service", MAX_TEXT)?;
        validate_text(&self.capability, "capability", MAX_CAPABILITY)?;
        validate_text(&self.track, "track", MAX_TEXT)?;
        validate_text(&self.producer_kem_key_id, "producer KEM key id", MAX_TEXT)?;
        validate_text(&self.consumer_kem_key_id, "consumer KEM key id", MAX_TEXT)?;
        ensure!(
            self.producer_state.identity_did == self.producer_endpoint_did,
            "producer endpoint DID does not match accepted state"
        );
        ensure!(
            self.consumer_state.identity_did == self.consumer_endpoint_did,
            "consumer endpoint DID does not match accepted state"
        );
        ensure!(
            self.max_blocks_per_epoch > 0,
            "max_blocks_per_epoch must be non-zero"
        );
        ensure!(
            self.max_blocks_per_epoch <= (u64::MAX >> 16),
            "max_blocks_per_epoch exceeds the deterministic nonce counter domain"
        );
        Ok(())
    }

    /// The policy-pinned suite; this is descriptive, not negotiated.
    pub fn suite(&self) -> SuiteId {
        self.suite
    }

    /// Admission-pinned upper bound before an authenticated epoch transition is required.
    pub fn max_blocks_per_epoch(&self) -> u64 {
        self.max_blocks_per_epoch
    }

    /// Canonical length-framed session transcript.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(512);
        out.extend_from_slice(&IDENTIFIED_STREAM_VERSION.to_be_bytes());
        out.extend_from_slice(&self.suite.as_u16().to_be_bytes());
        out.push(self.carrier_profile as u8);
        out.push(self.route_role as u8);
        put_text(&mut out, &self.producer_endpoint_did);
        put_text(&mut out, &self.consumer_endpoint_did);
        put_state(&mut out, &self.producer_state);
        put_state(&mut out, &self.consumer_state);
        put_text(&mut out, &self.service);
        put_text(&mut out, &self.capability);
        put_text(&mut out, &self.track);
        put_text(&mut out, &self.producer_kem_key_id);
        put_text(&mut out, &self.consumer_kem_key_id);
        out.extend_from_slice(&self.max_blocks_per_epoch.to_be_bytes());
        out
    }

    pub fn binding_hash(&self) -> [u8; 32] {
        derive_key(BINDING_HASH_DOMAIN, &self.canonical_bytes())
    }
}

/// Generic authority seam shared by identified and future anonymous releases.
///
/// This module provides no anonymous-capability implementation and derives no
/// assurance from this enum.  #1060-#1062 must supply a reviewed concrete type
/// and gate before `AnonymousCapability` can authorize production key release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamKeyReleasePrincipal<I, A> {
    Identified(I),
    AnonymousCapability(A),
}

/// Authorization/key-release policy seam.  Crypto code consumes an authorized
/// binding; network implementations must call a gate at the #726 release PEP.
pub trait StreamKeyReleaseGate<P> {
    type Grant;
    fn authorize_release(
        &self,
        principal: &P,
        binding: &IdentifiedStreamBinding,
    ) -> Result<Self::Grant>;
}

/// Directional keys for one track epoch.
#[derive(Clone)]
pub struct StreamEpochKeys {
    pub epoch: u64,
    pub direction: StreamDirection,
    pub mac_key: Zeroizing<[u8; 32]>,
    pub enc_key: Zeroizing<[u8; 32]>,
    pub control_mac_key: Zeroizing<[u8; 32]>,
    rekey_auth_key: Zeroizing<[u8; 32]>,
    nonce_domain: [u8; 4],
    epoch_namespace: [u8; 32],
}

impl StreamEpochKeys {
    /// Deterministic AES-GCM nonce: a transcript-derived 32-bit domain followed
    /// by the authenticated per-epoch sequence number.  The key and prefix both
    /// differ across direction/track/epoch.
    pub fn nonce(&self, sequence_number: u64) -> [u8; 12] {
        let mut nonce = [0u8; 12];
        nonce[..4].copy_from_slice(&self.nonce_domain);
        nonce[4..].copy_from_slice(&sequence_number.to_be_bytes());
        nonce
    }

    pub fn epoch_namespace(&self) -> &[u8; 32] {
        &self.epoch_namespace
    }
}

/// Both direction domains for one committed epoch.
#[derive(Clone)]
pub struct StreamEpochKeySet {
    pub epoch: u64,
    pub producer_to_consumer: StreamEpochKeys,
    pub consumer_to_producer: StreamEpochKeys,
}

impl StreamEpochKeySet {
    pub fn for_direction(&self, direction: StreamDirection) -> &StreamEpochKeys {
        match direction {
            StreamDirection::ProducerToConsumer => &self.producer_to_consumer,
            StreamDirection::ConsumerToProducer => &self.consumer_to_producer,
        }
    }
}

/// Authenticated epoch transition.  It contains no traffic key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamEpochCommit {
    pub binding_hash: [u8; 32],
    pub previous_epoch: u64,
    pub epoch: u64,
    pub producer_namespace: [u8; 32],
    pub consumer_namespace: [u8; 32],
    pub auth_tag: [u8; 32],
}

impl StreamEpochCommit {
    fn authenticated_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(112);
        out.extend_from_slice(&self.binding_hash);
        out.extend_from_slice(&self.previous_epoch.to_be_bytes());
        out.extend_from_slice(&self.epoch.to_be_bytes());
        out.extend_from_slice(&self.producer_namespace);
        out.extend_from_slice(&self.consumer_namespace);
        out
    }
}

/// Stateful, one-way, explicit stream epoch ratchet.
///
/// `prepare_next` is non-mutating.  `commit_prepared`/`accept_commit` make the
/// transition atomic: an invalid or interrupted pending update leaves the prior
/// committed epoch live and exposes no pending traffic keys.
#[derive(Clone)]
pub struct StreamEpochRatchet {
    binding: IdentifiedStreamBinding,
    binding_hash: [u8; 32],
    route_topic: String,
    control_topic: String,
    epoch: u64,
    secret: Zeroizing<[u8; 32]>,
}

impl StreamEpochRatchet {
    pub(crate) fn from_hybrid_secret(
        hybrid_secret: &[u8; 32],
        binding: IdentifiedStreamBinding,
    ) -> Result<Self> {
        binding.validate()?;
        let binding_bytes = binding.canonical_bytes();
        let mut root_input = Zeroizing::new(Vec::with_capacity(32 + binding_bytes.len()));
        root_input.extend_from_slice(hybrid_secret);
        root_input.extend_from_slice(&binding_bytes);
        let secret = Zeroizing::new(derive_key(SESSION_ROOT_DOMAIN, &root_input));
        let binding_hash = binding.binding_hash();
        let route_topic = hex::encode(derive_label(ROUTE_TOPIC_DOMAIN, &secret, &binding_hash));
        let control_topic = hex::encode(derive_label(CONTROL_TOPIC_DOMAIN, &secret, &binding_hash));
        Ok(Self {
            binding,
            binding_hash,
            route_topic,
            control_topic,
            epoch: 0,
            secret,
        })
    }

    pub fn binding(&self) -> &IdentifiedStreamBinding {
        &self.binding
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn route_topic(&self) -> &str {
        &self.route_topic
    }

    pub fn control_topic(&self) -> &str {
        &self.control_topic
    }

    pub fn current_keys(&self) -> StreamEpochKeySet {
        derive_epoch_key_set(&self.secret, &self.binding, self.epoch)
    }

    pub fn prepare_next(&self) -> Result<StreamEpochCommit> {
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("stream epoch exhausted"))?;
        let next_secret = next_epoch_secret(&self.secret, &self.binding_hash, next_epoch);
        let next_keys = derive_epoch_key_set(&next_secret, &self.binding, next_epoch);
        let mut commit = StreamEpochCommit {
            binding_hash: self.binding_hash,
            previous_epoch: self.epoch,
            epoch: next_epoch,
            producer_namespace: *next_keys.producer_to_consumer.epoch_namespace(),
            consumer_namespace: *next_keys.consumer_to_producer.epoch_namespace(),
            auth_tag: [0u8; 32],
        };
        let current = self.current_keys();
        commit.auth_tag = keyed_mac(
            &current.producer_to_consumer.rekey_auth_key,
            &commit.authenticated_bytes(),
        );
        Ok(commit)
    }

    pub fn commit_prepared(&mut self, commit: &StreamEpochCommit) -> Result<StreamEpochKeySet> {
        self.accept_commit(commit)
    }

    pub fn accept_commit(&mut self, commit: &StreamEpochCommit) -> Result<StreamEpochKeySet> {
        ensure!(
            commit.binding_hash == self.binding_hash,
            "stream rekey binding mismatch"
        );
        ensure!(
            commit.previous_epoch == self.epoch,
            "stream rekey is stale, replayed, or skips the committed epoch"
        );
        ensure!(
            commit.epoch
                == self
                    .epoch
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("stream epoch exhausted"))?,
            "stream rekey target is not the next epoch"
        );
        let current = self.current_keys();
        let expected = keyed_mac(
            &current.producer_to_consumer.rekey_auth_key,
            &commit.authenticated_bytes(),
        );
        ensure!(
            bool::from(expected.ct_eq(&commit.auth_tag)),
            "stream rekey authentication failed"
        );
        let next_secret = next_epoch_secret(&self.secret, &self.binding_hash, commit.epoch);
        let next_keys = derive_epoch_key_set(&next_secret, &self.binding, commit.epoch);
        ensure!(
            commit.producer_namespace == *next_keys.producer_to_consumer.epoch_namespace()
                && commit.consumer_namespace == *next_keys.consumer_to_producer.epoch_namespace(),
            "stream rekey namespace commitment mismatch"
        );

        // Atomic commit: all validation above is non-mutating.
        self.secret = next_secret;
        self.epoch = commit.epoch;
        Ok(next_keys)
    }
}

fn derive_epoch_key_set(
    epoch_secret: &[u8; 32],
    binding: &IdentifiedStreamBinding,
    epoch: u64,
) -> StreamEpochKeySet {
    StreamEpochKeySet {
        epoch,
        producer_to_consumer: derive_direction_keys(
            epoch_secret,
            binding,
            epoch,
            StreamDirection::ProducerToConsumer,
        ),
        consumer_to_producer: derive_direction_keys(
            epoch_secret,
            binding,
            epoch,
            StreamDirection::ConsumerToProducer,
        ),
    }
}

fn derive_direction_keys(
    epoch_secret: &[u8; 32],
    binding: &IdentifiedStreamBinding,
    epoch: u64,
    direction: StreamDirection,
) -> StreamEpochKeys {
    let mut transcript = binding.canonical_bytes();
    transcript.push(direction as u8);
    transcript.extend_from_slice(&epoch.to_be_bytes());
    let derive = |domain: &'static str| {
        let mut input = Zeroizing::new(Vec::with_capacity(32 + transcript.len()));
        input.extend_from_slice(epoch_secret);
        input.extend_from_slice(&transcript);
        derive_key(domain, &input)
    };
    let nonce = derive(NONCE_DOMAIN);
    let mut nonce_domain = [0u8; 4];
    nonce_domain.copy_from_slice(&nonce[..4]);
    StreamEpochKeys {
        epoch,
        direction,
        mac_key: Zeroizing::new(derive(DATA_MAC_DOMAIN)),
        enc_key: Zeroizing::new(derive(DATA_AEAD_DOMAIN)),
        control_mac_key: Zeroizing::new(derive(CONTROL_MAC_DOMAIN)),
        rekey_auth_key: Zeroizing::new(derive(REKEY_AUTH_DOMAIN)),
        nonce_domain,
        epoch_namespace: derive(EPOCH_NAMESPACE_DOMAIN),
    }
}

fn next_epoch_secret(
    current: &[u8; 32],
    binding_hash: &[u8; 32],
    epoch: u64,
) -> Zeroizing<[u8; 32]> {
    let mut input = Zeroizing::new([0u8; 72]);
    input[..32].copy_from_slice(current);
    input[32..64].copy_from_slice(binding_hash);
    input[64..].copy_from_slice(&epoch.to_be_bytes());
    Zeroizing::new(derive_key(EPOCH_SECRET_DOMAIN, &*input))
}

fn derive_label(domain: &'static str, secret: &[u8; 32], binding_hash: &[u8; 32]) -> [u8; 32] {
    let mut input = Zeroizing::new([0u8; 64]);
    input[..32].copy_from_slice(secret);
    input[32..].copy_from_slice(binding_hash);
    derive_key(domain, &*input)
}

fn validate_did(value: &str, label: &str) -> Result<()> {
    validate_text(value, label, MAX_TEXT)?;
    ensure!(value.starts_with("did:"), "{label} is not a DID");
    Ok(())
}

fn validate_text(value: &str, label: &str, limit: usize) -> Result<()> {
    ensure!(!value.is_empty(), "{label} is empty");
    ensure!(value.len() <= limit, "{label} exceeds {limit} bytes");
    ensure!(!value.bytes().any(|byte| byte == 0), "{label} contains NUL");
    Ok(())
}

fn put_text(out: &mut Vec<u8>, value: &str) {
    // Bindings are validated and frozen at construction, so this conversion is
    // infallible in practice. Encoding u64 avoids a panic path in the canonicalizer.
    let len = u64::try_from(value.len()).unwrap_or(u64::MAX);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_state(out: &mut Vec<u8>, state: &StreamAcceptedState) {
    put_text(out, &state.identity_did);
    out.extend_from_slice(&state.digest);
    out.extend_from_slice(&state.epoch.to_be_bytes());
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn binding(profile: StreamCarrierProfile) -> IdentifiedStreamBinding {
        IdentifiedStreamBinding::new(
            profile,
            StreamRouteRole::Relay,
            "did:at9p:producer",
            "did:at9p:consumer",
            StreamAcceptedState {
                identity_did: "did:at9p:producer".into(),
                digest: [0x11; 64],
                epoch: 7,
            },
            StreamAcceptedState {
                identity_did: "did:at9p:consumer".into(),
                digest: [0x22; 64],
                epoch: 9,
            },
            "inference.generate",
            "infer:model:qwen",
            "local/streams/tokens",
            "did:at9p:producer#mesh-kem",
            "urn:hyprstream:ephemeral-kem:123",
            1024,
        )
        .unwrap()
    }

    fn ratchet(profile: StreamCarrierProfile) -> StreamEpochRatchet {
        StreamEpochRatchet::from_hybrid_secret(&[0x44; 32], binding(profile)).unwrap()
    }

    #[test]
    fn transcript_mutations_separate_every_security_axis() {
        let base = binding(StreamCarrierProfile::StandardPublicRelay);
        let base_hash = base.binding_hash();
        let mut mutations = Vec::new();
        let mut v = base.clone();
        v.carrier_profile = StreamCarrierProfile::OwnedHybridTransport;
        mutations.push(v);
        let mut v = base.clone();
        v.route_role = StreamRouteRole::Direct;
        mutations.push(v);
        let mut v = base.clone();
        v.producer_endpoint_did.push_str(":other");
        v.producer_state.identity_did = v.producer_endpoint_did.clone();
        mutations.push(v);
        let mut v = base.clone();
        v.consumer_endpoint_did.push_str(":other");
        v.consumer_state.identity_did = v.consumer_endpoint_did.clone();
        mutations.push(v);
        let mut v = base.clone();
        v.producer_state.digest[0] ^= 1;
        mutations.push(v);
        let mut v = base.clone();
        v.consumer_state.epoch += 1;
        mutations.push(v);
        let mut v = base.clone();
        v.capability.push_str(":other");
        mutations.push(v);
        let mut v = base.clone();
        v.service.push_str(":other");
        mutations.push(v);
        let mut v = base.clone();
        v.track.push_str("-other");
        mutations.push(v);
        let mut v = base.clone();
        v.producer_kem_key_id.push_str("-other");
        mutations.push(v);
        let mut v = base.clone();
        v.consumer_kem_key_id.push_str("-other");
        mutations.push(v);
        let mut v = base.clone();
        v.max_blocks_per_epoch += 1;
        mutations.push(v);
        for mutation in mutations {
            assert_ne!(mutation.binding_hash(), base_hash);
        }
    }

    #[test]
    fn direction_track_epoch_keys_and_nonces_are_disjoint() {
        let r = ratchet(StreamCarrierProfile::StandardPublicRelay);
        let k0 = r.current_keys();
        assert_ne!(
            *k0.producer_to_consumer.enc_key,
            *k0.consumer_to_producer.enc_key
        );
        assert_ne!(
            k0.producer_to_consumer.nonce(4),
            k0.consumer_to_producer.nonce(4)
        );
        assert_ne!(
            k0.producer_to_consumer.nonce(4),
            k0.producer_to_consumer.nonce(5)
        );

        let mut other_track = binding(StreamCarrierProfile::StandardPublicRelay);
        other_track.track.push_str("-other");
        let other = StreamEpochRatchet::from_hybrid_secret(&[0x44; 32], other_track).unwrap();
        assert_ne!(
            *k0.producer_to_consumer.enc_key,
            *other.current_keys().producer_to_consumer.enc_key
        );
    }

    #[test]
    fn authenticated_rekey_is_atomic_replay_safe_and_one_way() {
        let mut sender = ratchet(StreamCarrierProfile::StandardPublicRelay);
        let mut receiver = sender.clone();
        let old = sender.current_keys();
        let commit = sender.prepare_next().unwrap();

        let mut tampered = commit.clone();
        tampered.producer_namespace[0] ^= 1;
        assert!(receiver.accept_commit(&tampered).is_err());
        assert_eq!(
            receiver.epoch(),
            0,
            "invalid pending update mutated committed state"
        );

        let sender_next = sender.commit_prepared(&commit).unwrap();
        let receiver_next = receiver.accept_commit(&commit).unwrap();
        assert_eq!(sender_next.epoch, 1);
        assert_eq!(
            *sender_next.producer_to_consumer.enc_key,
            *receiver_next.producer_to_consumer.enc_key
        );
        assert_ne!(
            *old.producer_to_consumer.enc_key,
            *sender_next.producer_to_consumer.enc_key
        );
        assert!(
            receiver.accept_commit(&commit).is_err(),
            "replayed commit accepted"
        );
    }

    #[test]
    fn profile_substitution_breaks_key_agreement() {
        let owned = ratchet(StreamCarrierProfile::OwnedHybridTransport);
        let relay = ratchet(StreamCarrierProfile::StandardPublicRelay);
        assert_ne!(owned.binding_hash, relay.binding_hash);
        assert_ne!(
            *owned.current_keys().producer_to_consumer.enc_key,
            *relay.current_keys().producer_to_consumer.enc_key
        );
    }

    #[test]
    fn identified_binding_rejects_anonymous_or_stale_shape() {
        let mut b = binding(StreamCarrierProfile::StandardPublicRelay);
        b.consumer_endpoint_did = "anonymous".into();
        assert!(b.validate().is_err());
        b.consumer_endpoint_did = b.consumer_state.identity_did.clone();
        b.consumer_state.digest = [0; 64];
        assert!(b.validate().is_err());
    }

    #[test]
    fn pinned_hykem_handshake_requires_both_components_and_binds_context() {
        use crate::crypto::hybrid_kem::{generate_recipient, HybridKemMaterial};
        use crate::crypto::key_exchange::{
            client_identified_stream_epoch, server_identified_stream_epoch,
        };

        let keypair = generate_recipient(IDENTIFIED_STREAM_SUITE).unwrap();
        let binding = binding(StreamCarrierProfile::StandardPublicRelay);
        let (material, server) =
            server_identified_stream_epoch(&keypair.public().encode(), binding.clone()).unwrap();
        let client =
            client_identified_stream_epoch(&keypair, &material.encode(), binding.clone()).unwrap();
        assert_eq!(
            *server.current_keys().producer_to_consumer.enc_key,
            *client.current_keys().producer_to_consumer.enc_key
        );

        // A classical-only material list is structurally rejected before key use.
        let mut classical_only = material.clone();
        classical_only.shares.truncate(1);
        assert!(client_identified_stream_epoch(
            &keypair,
            &classical_only.encode(),
            binding.clone()
        )
        .is_err());

        // Mutating either component changes or invalidates the agreed key.  ML-KEM
        // uses implicit rejection, so key mismatch is the expected failure signal.
        for index in 0..material.shares.len() {
            let mut crossed: HybridKemMaterial = material.clone();
            crossed.shares[index].bytes[0] ^= 1;
            if let Ok(crossed_client) =
                client_identified_stream_epoch(&keypair, &crossed.encode(), binding.clone())
            {
                assert_ne!(
                    *server.current_keys().producer_to_consumer.enc_key,
                    *crossed_client.current_keys().producer_to_consumer.enc_key
                );
            }
        }

        let mut substituted = binding;
        substituted.track.push_str("-substituted");
        let wrong_context =
            client_identified_stream_epoch(&keypair, &material.encode(), substituted).unwrap();
        assert_ne!(
            *server.current_keys().producer_to_consumer.enc_key,
            *wrong_context.current_keys().producer_to_consumer.enc_key
        );
    }

    #[test]
    fn client_kem_public_survives_authenticated_envelope_codec() {
        use crate::crypto::hybrid_kem::generate_recipient;
        use crate::envelope::RequestEnvelope;
        use crate::{FromCapnp, ToCapnp};

        let recipient = generate_recipient(IDENTIFIED_STREAM_SUITE)
            .unwrap()
            .public();
        let envelope = RequestEnvelope::new(b"start identified stream".to_vec())
            .with_client_kem_public(recipient.clone())
            .unwrap();
        let mut message = capnp::message::Builder::new_default();
        envelope
            .write_to(&mut message.init_root::<crate::common_capnp::request_envelope::Builder>());
        let decoded = RequestEnvelope::read_from(
            message
                .into_reader()
                .get_root::<crate::common_capnp::request_envelope::Reader>()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(decoded.client_kem_public, Some(recipient));
    }

    #[test]
    fn stock_relay_payload_is_opaque_and_forwarded_byte_identically() {
        use crate::crypto::event_crypto::{decrypt_event_full, encrypt_event_with_nonce};
        use crate::stream_consumer::stream_aead_aad;

        let ratchet = ratchet(StreamCarrierProfile::StandardPublicRelay);
        let keys = ratchet.current_keys();
        let traffic = &keys.producer_to_consumer;
        let plaintext = b"private inference token";
        let nonce = traffic.nonce(0);
        let aad = stream_aead_aad(ratchet.route_topic(), 0, 0, 0);
        let (tag, ciphertext, sent_nonce, commitment) =
            encrypt_event_with_nonce(&traffic.enc_key, nonce, &aad, plaintext).unwrap();

        // These are ordinary opaque Object payload bytes.  A stock relay copies
        // them without a HyKEM/epoch parser or any traffic key.
        let mut object = Vec::new();
        object.extend_from_slice(&sent_nonce);
        object.extend_from_slice(&commitment);
        object.extend_from_slice(&tag);
        object.extend_from_slice(&ciphertext);
        assert!(!object
            .windows(plaintext.len())
            .any(|window| window == plaintext));
        let relayed = object.clone();
        assert_eq!(relayed, object);

        let opened =
            decrypt_event_full(&traffic.enc_key, &sent_nonce, &tag, &ciphertext, &aad).unwrap();
        assert_eq!(opened, plaintext);
    }
}
