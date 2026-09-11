//! Session-scoped PQ binding overlay: trust-on-first-use continuity for
//! browser and other dynamically-identified clients, with out-of-band
//! promotion.
//!
//! # Why this exists
//!
//! [`crate::envelope::KeyedPqTrustStore`] is admin-anchored and immutable after
//! install: an operator enrolls each peer's ML-DSA-65 key out of band. A
//! browser's Ed25519 identity is user/device-generated, so no operator can
//! pre-enroll it, and under the mandatory Hybrid suite an unanchored identity
//! cannot have its signature checked at all — the request is dropped.
//!
//! This overlay is the additive answer. It is consulted by the envelope verify
//! path *after* the admin-anchored store misses, and it **never mutates that
//! store**: its entries are established out of band by contract, and the
//! federation perimeter deliberately leaves it untouched. Consult-then-fall-
//! back, never write-through.
//!
//! # The load-bearing property
//!
//! A [`PqProvenance::TofuBound`] entry makes a signature **verifiable** without
//! making the signer **more trusted**. Those are two different questions and
//! this module answers them through two different methods that share no return
//! type:
//!
//! - [`SessionPqOverlay::verifying_key_for`] — verifiability. Returns a key and
//!   nothing else, so the verify path structurally cannot learn provenance from
//!   it and therefore cannot launder it into an assurance decision.
//! - [`SessionPqOverlay::provenance_for`] — trust. The MAC assurance seam reads
//!   this, and [`PqProvenance::key_material`] refuses to map `TofuBound` to
//!   anything above `Classical`.
//!
//! # What a composite has to prove before anything is recorded
//!
//! The nested composite binds inner→outer — the outer ML-DSA-65 layer signs
//! `payload ‖ inner_signature` — and **not** outer→inner. So a composite that
//! verifies proves possession of the ML-DSA-65 key plus possession of one
//! Ed25519 *signature*, which is not the same as possession of the Ed25519
//! *private key*: anyone holding a copy of a valid inner layer can drop the
//! outer, re-sign with an ML-DSA key of their own, and produce a composite that
//! verifies under the captured identity.
//!
//! That is survivable while the PQ key is always resolved from a trust store,
//! and fatal the moment it is recorded. So a binding is established only from a
//! composite whose inner EdDSA layer *commits* to the key above it
//! ([`crate::crypto::cose_sign::PQ_BINDING_HEADER_LABEL`]) — a commitment that
//! lives inside the Ed25519 signature. An uncommitted composite still verifies;
//! it just cannot establish a binding.
//!
//! # Rebinding is loud by construction
//!
//! TOFU's security value is change detection, not initial trust. A silent
//! last-write-wins rebind voids every guarantee here, so the API offers no way
//! to perform one:
//!
//! - [`SessionPqOverlay::observe_first_contact`] inserts into a **vacant** slot
//!   only. An occupied slot naming a different key yields
//!   [`FirstContactOutcome::RebindRefused`], and the [`RebindEvent`] is
//!   published to the event sink *before* that value is returned, so a caller
//!   that discards the return value has still surfaced the event.
//! - The only way to change an established binding is
//!   [`SessionPqOverlay::apply_rebind`], which takes a [`RebindApproval`]. A
//!   `RebindApproval` can only be built by consuming a `RebindEvent`, which
//!   only the overlay can mint, and only by refusing a rebind. Legitimate key
//!   rotation therefore travels the same visible path as an attack.
//!
//! # Promotion and revocation
//!
//! Promotion to [`PqProvenance::OobVerified`] requires presenting the
//! [`composite_fingerprint`] — a hash over `ed25519 ‖ ml_dsa_65`, covering the
//! *pair*, because fingerprinting the PQ half alone confirms possession of a
//! key without binding it to the identity it claims to belong to. The value
//! must have been obtained through a channel with different compromise
//! assumptions from the one being bootstrapped; this module verifies the value
//! matches, and cannot verify where the operator got it.
//!
//! `TofuBound` entries expire with the session. An `OobVerified` promotion
//! outlives one, so it has an explicit revocation path
//! ([`SessionPqOverlay::revoke_promotion`],
//! [`SessionPqOverlay::revoke_binding`]) — an irreversible promotion would be
//! worse than no promotion.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::auth::mac::{Assurance, VerifiedKeyMaterial};
use crate::crypto::pq::{ml_dsa_vk_bytes, MlDsaVerifyingKey};

/// Domain separation tag for the composite fingerprint.
const FINGERPRINT_DOMAIN: &[u8] = b"hyprstream/pq-composite-fingerprint/v1";

/// Default lifetime of a first-contact binding. It is session continuity, not
/// enrollment: it lapses so that a device that stops appearing stops being
/// remembered.
pub const DEFAULT_TOFU_TTL_MS: i64 = 12 * 60 * 60 * 1000;

/// Default ceiling on retained bindings.
///
/// Recording costs an attacker only a self-minted Ed25519 + ML-DSA-65 keypair —
/// no enrollment, no credential, no prior relationship — and the verify path
/// runs before authorization, so an unbounded map is a memory-exhaustion
/// surface. At the ceiling the least-recently-seen first-contact entry is
/// evicted rather than the new one refused; see
/// [`SessionPqOverlay::observe_first_contact`] for why refusing is the worse
/// failure. Promoted bindings are never evicted.
pub const DEFAULT_MAX_ENTRIES: usize = 4096;

/// How a PQ verifying key came to be associated with an Ed25519 identity.
///
/// The absence of an entry is the third state of the model — `Classical`, no PQ
/// key bound — and is deliberately not a variant here: it is not something the
/// overlay records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PqProvenance {
    /// Recorded at first contact, over a channel authenticated only
    /// classically. Proves the client held both private keys at that moment
    /// and nothing more. Sufficient to check a signature; insufficient to
    /// believe the signer any harder than a classical one.
    TofuBound,
    /// Confirmed through a channel with different compromise assumptions than
    /// the one being bootstrapped — an operator-published deployment
    /// fingerprint, or an already-authenticated session on another device.
    OobVerified,
}

impl PqProvenance {
    /// The verified key material this provenance supports.
    ///
    /// This is the single point where a binding is allowed to influence
    /// assurance, and the only mapping that reaches
    /// [`VerifiedKeyMaterial::PqHybrid`] is [`PqProvenance::OobVerified`].
    #[must_use]
    pub fn key_material(self) -> VerifiedKeyMaterial {
        match self {
            // Capped. A first-contact record is continuity, not an anchor.
            PqProvenance::TofuBound => VerifiedKeyMaterial::Classical,
            PqProvenance::OobVerified => VerifiedKeyMaterial::PqHybrid,
        }
    }

    /// The highest MAC assurance this provenance can support.
    #[must_use]
    pub fn assurance_ceiling(self) -> Assurance {
        self.key_material().assurance()
    }

    /// Stable label for logs and audit records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            PqProvenance::TofuBound => "tofu-bound",
            PqProvenance::OobVerified => "oob-verified",
        }
    }
}

/// A recorded binding.
#[derive(Debug, Clone)]
struct OverlayEntry {
    ml_dsa_vk: Vec<u8>,
    provenance: PqProvenance,
    /// Wall-clock ms after which the entry is treated as absent. `None` for a
    /// promoted binding, which outlives the session that created it and is
    /// withdrawn by revocation rather than by lapse.
    expires_at_ms: Option<i64>,
    /// Monotonically increasing stamp of the last contact that touched this
    /// entry. Orders eviction when the table is full.
    last_seen_seq: u64,
}

impl OverlayEntry {
    fn is_live(&self, now_ms: i64) -> bool {
        match self.expires_at_ms {
            Some(deadline) => now_ms < deadline,
            None => true,
        }
    }
}

/// A surfaced attempt to bind an already-bound identity to a different PQ key.
///
/// Minted only by [`SessionPqOverlay::observe_first_contact`] refusing a
/// rebind. It is the sole input to [`RebindApproval`], which makes "change an
/// established binding" unreachable without an event having been raised first.
#[derive(Debug, Clone)]
#[must_use = "a refused rebind is a security event; record it or approve it deliberately"]
pub struct RebindEvent {
    identity: [u8; 32],
    established_vk: Vec<u8>,
    established_provenance: PqProvenance,
    presented_vk: Vec<u8>,
}

impl RebindEvent {
    /// The Ed25519 identity whose binding was challenged.
    #[must_use]
    pub fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    /// Provenance of the binding already on file.
    #[must_use]
    pub fn established_provenance(&self) -> PqProvenance {
        self.established_provenance
    }

    /// Fingerprint of the pairing already on file.
    #[must_use]
    pub fn established_fingerprint(&self) -> [u8; 32] {
        fingerprint_bytes(&self.identity, &self.established_vk)
    }

    /// Fingerprint of the pairing that was presented and refused.
    #[must_use]
    pub fn presented_fingerprint(&self) -> [u8; 32] {
        fingerprint_bytes(&self.identity, &self.presented_vk)
    }

    /// Turn this surfaced event into an approval to replace the binding.
    ///
    /// Consumes the event: an approval cannot exist unless a rebind was
    /// refused first. `justification` is carried into the applied-rebind
    /// notification so the record says who decided and on what basis.
    #[must_use]
    pub fn approve(self, provenance: PqProvenance, justification: impl Into<String>) -> RebindApproval {
        RebindApproval {
            event: self,
            provenance,
            justification: justification.into(),
        }
    }
}

/// An operator decision to replace an established binding, carrying the event
/// it answers.
#[derive(Debug, Clone)]
pub struct RebindApproval {
    event: RebindEvent,
    provenance: PqProvenance,
    justification: String,
}

impl RebindApproval {
    /// The event this approval answers.
    pub fn event(&self) -> &RebindEvent {
        &self.event
    }

    /// The provenance the replacement binding will carry.
    #[must_use]
    pub fn provenance(&self) -> PqProvenance {
        self.provenance
    }
}

/// Outcome of offering a first-contact observation.
#[derive(Debug, Clone)]
#[must_use]
pub enum FirstContactOutcome {
    /// No live binding existed; the presented key is now recorded `TofuBound`.
    Recorded,
    /// A live binding exists and names the same key. Nothing changed.
    AlreadyBound(PqProvenance),
    /// A live binding exists and names a *different* key. **Not applied.**
    RebindRefused(RebindEvent),
    /// The overlay is at capacity and no expired entry could be reclaimed. The
    /// binding was not recorded; the request that offered it fails closed.
    CapacityExhausted,
}

/// Why a promotion was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionError {
    /// No live binding exists for the identity — there is nothing to promote.
    /// Promotion never creates a binding: the out-of-band channel confirms a
    /// pairing the node already observed, it does not introduce one.
    NotBound,
    /// The out-of-band fingerprint does not match the recorded pairing.
    FingerprintMismatch,
}

impl std::fmt::Display for PromotionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromotionError::NotBound => f.write_str("no live PQ binding for this identity"),
            PromotionError::FingerprintMismatch => {
                f.write_str("out-of-band fingerprint does not match the recorded pairing")
            }
        }
    }
}

impl std::error::Error for PromotionError {}

/// Sink for binding lifecycle events.
///
/// Every method has a default no-op body except the two that report a change of
/// trust state; implementors are expected to route those to the audit journal.
pub trait PqBindingEventSink: Send + Sync {
    /// An identity presented a PQ key different from its established binding.
    /// The binding was left alone.
    fn on_rebind_refused(&self, event: &RebindEvent);

    /// An established binding was replaced under an explicit approval.
    fn on_rebind_applied(&self, approval: &RebindApproval);

    /// A first-contact binding was recorded.
    fn on_first_contact(&self, _identity: &[u8; 32], _fingerprint: &[u8; 32]) {}

    /// A binding was promoted to `OobVerified`.
    fn on_promoted(&self, _identity: &[u8; 32], _fingerprint: &[u8; 32], _channel: &str) {}

    /// A promotion or a whole binding was withdrawn.
    fn on_revoked(&self, _identity: &[u8; 32], _from: PqProvenance, _to: Option<PqProvenance>) {}

    /// A first-contact binding was evicted to make room. The identity loses
    /// change detection until it is seen again.
    fn on_evicted(&self, _identity: &[u8; 32]) {}
}

/// Sink that records events to `tracing` at warning level.
pub struct TracingPqBindingEventSink;

impl PqBindingEventSink for TracingPqBindingEventSink {
    fn on_rebind_refused(&self, event: &RebindEvent) {
        tracing::warn!(
            identity = %hex::encode(event.identity()),
            established_provenance = event.established_provenance().as_str(),
            established_fingerprint = %hex::encode(event.established_fingerprint()),
            presented_fingerprint = %hex::encode(event.presented_fingerprint()),
            "PQ rebinding refused: identity presented a different ML-DSA-65 key than its \
             established binding; the binding was NOT changed"
        );
    }

    fn on_rebind_applied(&self, approval: &RebindApproval) {
        tracing::warn!(
            identity = %hex::encode(approval.event().identity()),
            from_fingerprint = %hex::encode(approval.event().established_fingerprint()),
            to_fingerprint = %hex::encode(approval.event().presented_fingerprint()),
            provenance = approval.provenance().as_str(),
            justification = %approval.justification,
            "PQ binding replaced under explicit approval"
        );
    }

    fn on_first_contact(&self, identity: &[u8; 32], fingerprint: &[u8; 32]) {
        tracing::info!(
            identity = %hex::encode(identity),
            fingerprint = %hex::encode(fingerprint),
            "PQ binding recorded at first contact (verifiable, assurance capped at Classical)"
        );
    }

    fn on_promoted(&self, identity: &[u8; 32], fingerprint: &[u8; 32], channel: &str) {
        tracing::warn!(
            identity = %hex::encode(identity),
            fingerprint = %hex::encode(fingerprint),
            channel,
            "PQ binding promoted to out-of-band verified"
        );
    }

    fn on_revoked(&self, identity: &[u8; 32], from: PqProvenance, to: Option<PqProvenance>) {
        tracing::warn!(
            identity = %hex::encode(identity),
            from = from.as_str(),
            to = to.map_or("none", PqProvenance::as_str),
            "PQ binding withdrawn"
        );
    }

    fn on_evicted(&self, identity: &[u8; 32]) {
        tracing::warn!(
            identity = %hex::encode(identity),
            "PQ binding evicted under capacity pressure; change detection for this identity \
             is reset until it is seen again"
        );
    }
}

/// The fingerprint compared over the out-of-band channel.
///
/// Covers the **pair** — `domain ‖ ed25519 ‖ ml_dsa_65` — not the PQ key alone.
/// A fingerprint over only the PQ half would confirm that someone holds that
/// key while saying nothing about which identity it belongs to, which is
/// exactly the substitution the out-of-band comparison exists to catch.
#[must_use]
pub fn composite_fingerprint(ed25519_pubkey: &[u8; 32], ml_dsa_vk: &MlDsaVerifyingKey) -> [u8; 32] {
    fingerprint_bytes(ed25519_pubkey, &ml_dsa_vk_bytes(ml_dsa_vk))
}

fn fingerprint_bytes(ed25519_pubkey: &[u8; 32], ml_dsa_vk_bytes: &[u8]) -> [u8; 32] {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(FINGERPRINT_DOMAIN);
    hasher.update(ed25519_pubkey);
    hasher.update(ml_dsa_vk_bytes);
    hasher.finalize().into()
}

/// Render a fingerprint the way it is shown to a human performing the
/// comparison: uppercase hex in space-separated quads, so a mismatch in any
/// position is easy to spot when read aloud or compared by eye.
#[must_use]
pub fn format_fingerprint(fingerprint: &[u8; 32]) -> String {
    let hex = hex::encode_upper(fingerprint);
    hex.as_bytes()
        .chunks(4)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Additive, session-scoped PQ binding overlay.
pub struct SessionPqOverlay {
    entries: RwLock<HashMap<[u8; 32], OverlayEntry>>,
    sink: Arc<dyn PqBindingEventSink>,
    tofu_ttl_ms: i64,
    max_entries: usize,
    /// Recency counter for eviction ordering. Wall-clock is unsuitable: many
    /// first contacts can land inside one millisecond.
    contact_seq: std::sync::atomic::AtomicU64,
}

impl Default for SessionPqOverlay {
    fn default() -> Self {
        Self::new(Arc::new(TracingPqBindingEventSink))
    }
}

impl SessionPqOverlay {
    /// Build an overlay with the default TTL and capacity.
    #[must_use]
    pub fn new(sink: Arc<dyn PqBindingEventSink>) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            sink,
            tofu_ttl_ms: DEFAULT_TOFU_TTL_MS,
            max_entries: DEFAULT_MAX_ENTRIES,
            contact_seq: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Override the first-contact lifetime.
    #[must_use]
    pub fn with_tofu_ttl_ms(mut self, ttl_ms: i64) -> Self {
        self.tofu_ttl_ms = ttl_ms;
        self
    }

    /// Override the retained-binding ceiling.
    #[must_use]
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// The ML-DSA-65 key to check this identity's signature against.
    ///
    /// Returns the key and nothing else. There is deliberately no provenance in
    /// the return type: the verify path has no business knowing how much the
    /// binding is believed, and cannot accidentally propagate it.
    #[must_use]
    pub fn verifying_key_for(&self, ed25519_pubkey: &[u8; 32]) -> Option<MlDsaVerifyingKey> {
        let now = crate::envelope::current_timestamp();
        let entries = self.entries.read();
        let entry = entries.get(ed25519_pubkey)?;
        if !entry.is_live(now) {
            return None;
        }
        crate::crypto::pq::ml_dsa_vk_from_bytes(&entry.ml_dsa_vk).ok()
    }

    /// How much this identity's binding is believed, if it has a live one.
    ///
    /// This is the trust question, answered separately from
    /// [`Self::verifying_key_for`].
    #[must_use]
    pub fn provenance_for(&self, ed25519_pubkey: &[u8; 32]) -> Option<PqProvenance> {
        let now = crate::envelope::current_timestamp();
        let entries = self.entries.read();
        let entry = entries.get(ed25519_pubkey)?;
        entry.is_live(now).then_some(entry.provenance)
    }

    /// The fingerprint of this identity's live binding, for display in an
    /// out-of-band comparison ceremony.
    #[must_use]
    pub fn fingerprint_for(&self, ed25519_pubkey: &[u8; 32]) -> Option<[u8; 32]> {
        let now = crate::envelope::current_timestamp();
        let entries = self.entries.read();
        let entry = entries.get(ed25519_pubkey)?;
        entry
            .is_live(now)
            .then(|| fingerprint_bytes(ed25519_pubkey, &entry.ml_dsa_vk))
    }

    /// Offer a PQ key observed at first contact.
    ///
    /// **Call only after the composite has verified against `presented_vk` AND
    /// reported the inner layer's commitment to it** (`CompositeVerified::
    /// pq_bound`). Verification alone is not enough: the nesting binds
    /// inner→outer only, so a composite without that commitment proves
    /// possession of the ML-DSA-65 key plus possession of one replayable
    /// Ed25519 signature — not of the Ed25519 private key. Recording on those
    /// weaker grounds lets anyone holding a captured envelope bind their own PQ
    /// key to the sender's identity.
    ///
    /// This never overwrites a **live** entry. A live entry naming a different
    /// key produces [`FirstContactOutcome::RebindRefused`], and the event
    /// reaches the sink before this returns.
    ///
    /// # Change detection is bounded by the TTL
    ///
    /// A binding that has lapsed is, by the session-scoped model, forgotten:
    /// the identity is unbound again and the next contact is genuinely a first
    /// one, recorded without a refusal even if it names a different key. So the
    /// window in which a key substitution is caught is exactly the TTL, and an
    /// attacker who can wait out a device's idle period sees an unbound
    /// identity rather than a challenged one. Raising the TTL widens detection
    /// at the cost of remembering devices longer; a promotion
    /// ([`Self::promote_out_of_band`]) removes the deadline entirely, which is
    /// the durable answer for an identity worth pinning.
    pub fn observe_first_contact(
        &self,
        ed25519_pubkey: [u8; 32],
        presented_vk: &MlDsaVerifyingKey,
    ) -> FirstContactOutcome {
        let now = crate::envelope::current_timestamp();
        let presented_bytes = ml_dsa_vk_bytes(presented_vk);
        let seq = self
            .contact_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut evicted: Option<[u8; 32]> = None;
        let mut return_capacity_exhausted = false;

        let outcome = {
            let mut entries = self.entries.write();
            match entries.get_mut(&ed25519_pubkey) {
                Some(entry) if entry.is_live(now) => {
                    if entry.ml_dsa_vk == presented_bytes {
                        entry.last_seen_seq = seq;
                        FirstContactOutcome::AlreadyBound(entry.provenance)
                    } else {
                        // Refuse. The established binding stays exactly as it
                        // is; the only path that replaces it is `apply_rebind`,
                        // which needs the event this constructs.
                        FirstContactOutcome::RebindRefused(RebindEvent {
                            identity: ed25519_pubkey,
                            established_vk: entry.ml_dsa_vk.clone(),
                            established_provenance: entry.provenance,
                            presented_vk: presented_bytes,
                        })
                    }
                }
                _ => {
                    // Vacant or lapsed. Reclaim lapsed entries first — a quiet
                    // period must not be permanently fatal.
                    if entries.len() >= self.max_entries {
                        entries.retain(|_, e| e.is_live(now));
                    }
                    // Still full: evict the least-recently-seen *first-contact*
                    // entry rather than refusing the new one.
                    //
                    // Refusing was the obvious fail-closed choice and it is the
                    // wrong one here. Recording costs an attacker only a
                    // self-minted keypair — no enrollment, no credential — and
                    // this runs before authorization, so filling the table is
                    // seconds of work. Under refusal that locks out every NEW
                    // dynamic client until entries lapse, i.e. an unauthenticated
                    // party gets to switch the feature off. Under eviction the
                    // worst an attacker achieves is resetting other identities'
                    // change detection, which is the same exposure a lapse
                    // already carries, and legitimate clients keep connecting.
                    //
                    // Promoted bindings are never evicted: they are an operator
                    // statement, not session continuity.
                    if entries.len() >= self.max_entries && !entries.contains_key(&ed25519_pubkey) {
                        let victim = entries
                            .iter()
                            .filter(|(_, e)| e.provenance == PqProvenance::TofuBound)
                            .min_by_key(|(_, e)| e.last_seen_seq)
                            .map(|(id, _)| *id);
                        match victim {
                            Some(id) => {
                                entries.remove(&id);
                                evicted = Some(id);
                            }
                            // Every slot holds a promotion. Refuse rather than
                            // discard operator-established trust.
                            None => return_capacity_exhausted = true,
                        }
                    }
                    if return_capacity_exhausted {
                        FirstContactOutcome::CapacityExhausted
                    } else {
                        entries.insert(
                            ed25519_pubkey,
                            OverlayEntry {
                                ml_dsa_vk: presented_bytes,
                                provenance: PqProvenance::TofuBound,
                                expires_at_ms: Some(now.saturating_add(self.tofu_ttl_ms)),
                                last_seen_seq: seq,
                            },
                        );
                        FirstContactOutcome::Recorded
                    }
                }
            }
        };

        match &outcome {
            FirstContactOutcome::RebindRefused(event) => self.sink.on_rebind_refused(event),
            FirstContactOutcome::Recorded => {
                let fp = fingerprint_bytes(&ed25519_pubkey, &ml_dsa_vk_bytes(presented_vk));
                self.sink.on_first_contact(&ed25519_pubkey, &fp);
            }
            FirstContactOutcome::CapacityExhausted => {
                tracing::warn!(
                    identity = %hex::encode(ed25519_pubkey),
                    "PQ overlay at capacity with every slot promoted; first-contact binding \
                     refused (request fails closed)"
                );
            }
            FirstContactOutcome::AlreadyBound(_) => {}
        }
        if let Some(id) = evicted {
            // An evicted identity loses change detection: its next contact is a
            // fresh first contact. Say so, rather than letting the table quietly
            // forget.
            tracing::warn!(
                evicted_identity = %hex::encode(id),
                admitted_identity = %hex::encode(ed25519_pubkey),
                "PQ overlay at capacity; evicted the least-recently-seen first-contact binding"
            );
            self.sink.on_evicted(&id);
        }

        outcome
    }

    /// Publish a rebinding event **without any possibility of recording**.
    ///
    /// The verify path reaches a rebinding attempt before the presented key has
    /// been checked against anything — the whole point is that the request is
    /// being rejected. [`Self::observe_first_contact`] must not be reused there:
    /// its contract is "the composite already verified against this key", and a
    /// binding that lapses between the lookup and the call would send it down
    /// the recording branch with an unproven key. This entry point cannot
    /// record under any interleaving.
    pub fn surface_rebind(&self, ed25519_pubkey: [u8; 32], presented_vk: &MlDsaVerifyingKey) {
        let now = crate::envelope::current_timestamp();
        let event = {
            let entries = self.entries.read();
            entries
                .get(&ed25519_pubkey)
                .filter(|e| e.is_live(now))
                .map(|entry| RebindEvent {
                    identity: ed25519_pubkey,
                    established_vk: entry.ml_dsa_vk.clone(),
                    established_provenance: entry.provenance,
                    presented_vk: ml_dsa_vk_bytes(presented_vk),
                })
        };
        if let Some(event) = event {
            self.sink.on_rebind_refused(&event);
        }
    }

    /// Replace an established binding, under an approval that answers a
    /// surfaced [`RebindEvent`].
    ///
    /// This is the only mutation that can change an existing binding, and it is
    /// unreachable without the overlay having refused a rebind first. Returns
    /// `false` when the identity's binding no longer matches the one the event
    /// described — a concurrent change means the approval answered a stale
    /// observation and must be re-taken.
    pub fn apply_rebind(&self, approval: RebindApproval) -> bool {
        let now = crate::envelope::current_timestamp();
        let applied = {
            let mut entries = self.entries.write();
            match entries.get(&approval.event.identity) {
                Some(entry) if entry.ml_dsa_vk == approval.event.established_vk => {
                    let expires_at_ms = match approval.provenance {
                        PqProvenance::TofuBound => Some(now.saturating_add(self.tofu_ttl_ms)),
                        PqProvenance::OobVerified => None,
                    };
                    entries.insert(
                        approval.event.identity,
                        OverlayEntry {
                            ml_dsa_vk: approval.event.presented_vk.clone(),
                            provenance: approval.provenance,
                            expires_at_ms,
                            last_seen_seq: self
                                .contact_seq
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                        },
                    );
                    true
                }
                _ => false,
            }
        };
        if applied {
            self.sink.on_rebind_applied(&approval);
        }
        applied
    }

    /// Promote a live binding to [`PqProvenance::OobVerified`].
    ///
    /// `oob_fingerprint` is the value the operator obtained through the
    /// independent channel; it must equal [`composite_fingerprint`] over the
    /// recorded pairing. `channel` names where it came from and is carried into
    /// the notification.
    ///
    /// Promotion never creates a binding — an identity the node has never
    /// observed cannot be promoted, because there would be no observed pairing
    /// for the fingerprint to confirm.
    ///
    /// # Errors
    ///
    /// [`PromotionError::NotBound`] when no live binding exists;
    /// [`PromotionError::FingerprintMismatch`] when the value does not match.
    pub fn promote_out_of_band(
        &self,
        ed25519_pubkey: &[u8; 32],
        oob_fingerprint: &[u8; 32],
        channel: &str,
    ) -> std::result::Result<(), PromotionError> {
        use subtle::ConstantTimeEq as _;
        let now = crate::envelope::current_timestamp();
        {
            let mut entries = self.entries.write();
            let entry = entries
                .get_mut(ed25519_pubkey)
                .filter(|e| e.is_live(now))
                .ok_or(PromotionError::NotBound)?;
            let recorded = fingerprint_bytes(ed25519_pubkey, &entry.ml_dsa_vk);
            if !bool::from(recorded.ct_eq(oob_fingerprint)) {
                return Err(PromotionError::FingerprintMismatch);
            }
            entry.provenance = PqProvenance::OobVerified;
            // A promotion is a statement about the pairing, not about the
            // session that happened to observe it, so it stops lapsing.
            entry.expires_at_ms = None;
        }
        self.sink
            .on_promoted(ed25519_pubkey, oob_fingerprint, channel);
        Ok(())
    }

    /// Withdraw a promotion, returning the binding to `TofuBound` continuity
    /// (and to lapsing) without breaking the session using it.
    ///
    /// Returns `false` when the identity has no live `OobVerified` binding.
    pub fn revoke_promotion(&self, ed25519_pubkey: &[u8; 32]) -> bool {
        let now = crate::envelope::current_timestamp();
        let revoked = {
            let mut entries = self.entries.write();
            match entries.get_mut(ed25519_pubkey) {
                Some(entry)
                    if entry.is_live(now) && entry.provenance == PqProvenance::OobVerified =>
                {
                    entry.provenance = PqProvenance::TofuBound;
                    entry.expires_at_ms = Some(now.saturating_add(self.tofu_ttl_ms));
                    true
                }
                _ => false,
            }
        };
        if revoked {
            self.sink.on_revoked(
                ed25519_pubkey,
                PqProvenance::OobVerified,
                Some(PqProvenance::TofuBound),
            );
        }
        revoked
    }

    /// Withdraw the binding entirely. The identity returns to the `Classical`
    /// state — no PQ key bound — and its next contact is a fresh first contact.
    ///
    /// Returns `false` when there was nothing to withdraw.
    pub fn revoke_binding(&self, ed25519_pubkey: &[u8; 32]) -> bool {
        let removed = self.entries.write().remove(ed25519_pubkey);
        match removed {
            Some(entry) => {
                self.sink.on_revoked(ed25519_pubkey, entry.provenance, None);
                true
            }
            None => false,
        }
    }

    /// Number of live bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        let now = crate::envelope::current_timestamp();
        self.entries
            .read()
            .values()
            .filter(|v| v.is_live(now))
            .count()
    }

    /// Whether the overlay holds no live bindings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Process-global overlay.
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
static SESSION_PQ_OVERLAY: std::sync::OnceLock<Arc<SessionPqOverlay>> = std::sync::OnceLock::new();

/// Install the process-global overlay. First write wins.
///
/// **Not installed by default.** With no overlay installed the verify path
/// behaves exactly as it did before this existed: an identity with no
/// admin-anchored ML-DSA-65 key is rejected under the mandatory Hybrid suite.
/// Trust-on-first-use is a deployment decision, so it is opted into explicitly
/// rather than acquired by linking this crate.
///
/// # Errors
///
/// Returns `Err` when an overlay was already installed.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_session_pq_overlay(overlay: Arc<SessionPqOverlay>) -> anyhow::Result<()> {
    SESSION_PQ_OVERLAY
        .set(overlay)
        .map_err(|_| anyhow::anyhow!("session PQ overlay already installed"))
}

/// The installed process-global overlay, if any.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn global_session_pq_overlay() -> Option<Arc<SessionPqOverlay>> {
    SESSION_PQ_OVERLAY.get().cloned()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn pq_key(seed: u8) -> MlDsaVerifyingKey {
        let sk = crate::crypto::pq::ml_dsa_sk_from_seed(&[seed; 32]);
        crate::crypto::pq::ml_dsa_vk_from_bytes(&crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&sk))
            .expect("derived verifying key round-trips")
    }

    fn ed_identity(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    #[derive(Default)]
    struct CountingSink {
        refused: std::sync::atomic::AtomicUsize,
        applied: std::sync::atomic::AtomicUsize,
        first_contact: std::sync::atomic::AtomicUsize,
        promoted: std::sync::atomic::AtomicUsize,
        revoked: std::sync::atomic::AtomicUsize,
    }

    impl CountingSink {
        fn count(v: &std::sync::atomic::AtomicUsize) -> usize {
            v.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn bump(v: &std::sync::atomic::AtomicUsize) {
            v.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl PqBindingEventSink for CountingSink {
        fn on_rebind_refused(&self, _event: &RebindEvent) {
            Self::bump(&self.refused);
        }
        fn on_rebind_applied(&self, _approval: &RebindApproval) {
            Self::bump(&self.applied);
        }
        fn on_first_contact(&self, _identity: &[u8; 32], _fingerprint: &[u8; 32]) {
            Self::bump(&self.first_contact);
        }
        fn on_promoted(&self, _identity: &[u8; 32], _fingerprint: &[u8; 32], _channel: &str) {
            Self::bump(&self.promoted);
        }
        fn on_revoked(&self, _i: &[u8; 32], _f: PqProvenance, _t: Option<PqProvenance>) {
            Self::bump(&self.revoked);
        }
    }

    fn overlay_with_sink() -> (SessionPqOverlay, Arc<CountingSink>) {
        let sink = Arc::new(CountingSink::default());
        (SessionPqOverlay::new(sink.clone()), sink)
    }

    #[test]
    fn first_contact_records_a_tofu_binding() {
        let (overlay, sink) = overlay_with_sink();
        let id = ed_identity(1);
        let key = pq_key(11);

        assert!(matches!(
            overlay.observe_first_contact(id, &key),
            FirstContactOutcome::Recorded
        ));
        assert_eq!(CountingSink::count(&sink.first_contact), 1);
        assert_eq!(
            ml_dsa_vk_bytes(&overlay.verifying_key_for(&id).unwrap()),
            ml_dsa_vk_bytes(&key)
        );
        assert_eq!(overlay.provenance_for(&id), Some(PqProvenance::TofuBound));
    }

    #[test]
    fn repeat_contact_with_the_same_key_changes_nothing() {
        let (overlay, sink) = overlay_with_sink();
        let id = ed_identity(2);
        let key = pq_key(12);

        let _ = overlay.observe_first_contact(id, &key);
        let outcome = overlay.observe_first_contact(id, &key);
        assert!(matches!(
            outcome,
            FirstContactOutcome::AlreadyBound(PqProvenance::TofuBound)
        ));
        assert_eq!(CountingSink::count(&sink.refused), 0);
        assert_eq!(CountingSink::count(&sink.first_contact), 1);
    }

    #[test]
    fn a_different_key_is_refused_and_surfaced_not_silently_written() {
        let (overlay, sink) = overlay_with_sink();
        let id = ed_identity(3);
        let original = pq_key(13);
        let substitute = pq_key(14);

        let _ = overlay.observe_first_contact(id, &original);
        let outcome = overlay.observe_first_contact(id, &substitute);

        let FirstContactOutcome::RebindRefused(event) = outcome else {
            panic!("a different key must be refused, not accepted");
        };
        // The event reached the sink before the caller saw the outcome.
        assert_eq!(CountingSink::count(&sink.refused), 1);
        assert_eq!(event.established_provenance(), PqProvenance::TofuBound);
        assert_ne!(event.established_fingerprint(), event.presented_fingerprint());
        // Last write did NOT win.
        assert_eq!(
            ml_dsa_vk_bytes(&overlay.verifying_key_for(&id).unwrap()),
            ml_dsa_vk_bytes(&original)
        );
    }

    #[test]
    fn replacing_a_binding_requires_an_approval_built_from_the_refusal() {
        let (overlay, sink) = overlay_with_sink();
        let id = ed_identity(4);
        let original = pq_key(15);
        let rotated = pq_key(16);

        let _ = overlay.observe_first_contact(id, &original);
        let FirstContactOutcome::RebindRefused(event) =
            overlay.observe_first_contact(id, &rotated)
        else {
            panic!("rotation must surface as a refused rebind");
        };

        assert!(overlay.apply_rebind(event.approve(PqProvenance::TofuBound, "operator rotation")));
        assert_eq!(CountingSink::count(&sink.applied), 1);
        assert_eq!(
            ml_dsa_vk_bytes(&overlay.verifying_key_for(&id).unwrap()),
            ml_dsa_vk_bytes(&rotated)
        );
        // Rotation through the visible path does not confer trust on its own.
        assert_eq!(overlay.provenance_for(&id), Some(PqProvenance::TofuBound));
    }

    #[test]
    fn a_stale_approval_does_not_apply() {
        let (overlay, _sink) = overlay_with_sink();
        let id = ed_identity(5);
        let original = pq_key(17);
        let rotated = pq_key(18);
        let other = pq_key(19);

        let _ = overlay.observe_first_contact(id, &original);
        let FirstContactOutcome::RebindRefused(event) =
            overlay.observe_first_contact(id, &rotated)
        else {
            panic!("expected a refusal");
        };
        // The binding moves underneath the pending approval.
        overlay.revoke_binding(&id);
        let _ = overlay.observe_first_contact(id, &other);

        assert!(!overlay.apply_rebind(event.approve(PqProvenance::OobVerified, "stale")));
        assert_eq!(
            ml_dsa_vk_bytes(&overlay.verifying_key_for(&id).unwrap()),
            ml_dsa_vk_bytes(&other)
        );
    }

    #[test]
    fn tofu_never_reaches_pq_hybrid_key_material() {
        assert_eq!(
            PqProvenance::TofuBound.key_material(),
            VerifiedKeyMaterial::Classical
        );
        assert_eq!(PqProvenance::TofuBound.assurance_ceiling(), Assurance::Classical);
        assert!(PqProvenance::TofuBound.assurance_ceiling() < Assurance::PqHybrid);
    }

    #[test]
    fn only_oob_verified_reaches_pq_hybrid() {
        // Exhaustive over the provenance domain: exactly one variant may raise.
        for p in [PqProvenance::TofuBound, PqProvenance::OobVerified] {
            let raises = p.key_material() == VerifiedKeyMaterial::PqHybrid;
            assert_eq!(raises, p == PqProvenance::OobVerified);
        }
    }

    #[test]
    fn promotion_requires_the_composite_fingerprint() {
        let (overlay, sink) = overlay_with_sink();
        let id = ed_identity(6);
        let key = pq_key(20);
        let _ = overlay.observe_first_contact(id, &key);

        // A fingerprint over the PQ half bound to a *different* identity must
        // not promote: the value covers the pair, not the key alone.
        let wrong = composite_fingerprint(&ed_identity(7), &key);
        assert_eq!(
            overlay.promote_out_of_band(&id, &wrong, "printed"),
            Err(PromotionError::FingerprintMismatch)
        );
        assert_eq!(overlay.provenance_for(&id), Some(PqProvenance::TofuBound));

        let right = composite_fingerprint(&id, &key);
        assert!(overlay.promote_out_of_band(&id, &right, "printed").is_ok());
        assert_eq!(overlay.provenance_for(&id), Some(PqProvenance::OobVerified));
        assert_eq!(CountingSink::count(&sink.promoted), 1);
    }

    #[test]
    fn promotion_cannot_invent_a_binding() {
        let (overlay, _sink) = overlay_with_sink();
        let id = ed_identity(8);
        let key = pq_key(21);
        assert_eq!(
            overlay.promote_out_of_band(&id, &composite_fingerprint(&id, &key), "printed"),
            Err(PromotionError::NotBound)
        );
        assert!(overlay.verifying_key_for(&id).is_none());
    }

    #[test]
    fn a_promotion_can_be_revoked() {
        let (overlay, sink) = overlay_with_sink();
        let id = ed_identity(9);
        let key = pq_key(22);
        let _ = overlay.observe_first_contact(id, &key);
        overlay
            .promote_out_of_band(&id, &composite_fingerprint(&id, &key), "printed")
            .unwrap();

        assert!(overlay.revoke_promotion(&id));
        assert_eq!(CountingSink::count(&sink.revoked), 1);
        // Trust is withdrawn; verifiability and the live session are not.
        assert_eq!(overlay.provenance_for(&id), Some(PqProvenance::TofuBound));
        assert!(overlay.verifying_key_for(&id).is_some());
        // Idempotent: nothing left to revoke.
        assert!(!overlay.revoke_promotion(&id));
    }

    #[test]
    fn revoking_a_binding_returns_the_identity_to_unbound() {
        let (overlay, _sink) = overlay_with_sink();
        let id = ed_identity(10);
        let key = pq_key(23);
        let _ = overlay.observe_first_contact(id, &key);
        assert!(overlay.revoke_binding(&id));
        assert!(overlay.verifying_key_for(&id).is_none());
        assert!(overlay.provenance_for(&id).is_none());
        assert!(!overlay.revoke_binding(&id));
    }

    #[test]
    fn a_lapsed_first_contact_binding_stops_verifying() {
        let (overlay, _sink) = overlay_with_sink();
        let overlay = overlay.with_tofu_ttl_ms(0);
        let id = ed_identity(11);
        let key = pq_key(24);
        let _ = overlay.observe_first_contact(id, &key);
        assert!(overlay.verifying_key_for(&id).is_none());
        assert!(overlay.is_empty());
    }

    #[test]
    fn a_promoted_binding_does_not_lapse() {
        let (overlay, _sink) = overlay_with_sink();
        let overlay = overlay.with_tofu_ttl_ms(60_000);
        let id = ed_identity(12);
        let key = pq_key(25);
        let _ = overlay.observe_first_contact(id, &key);
        overlay
            .promote_out_of_band(&id, &composite_fingerprint(&id, &key), "printed")
            .unwrap();
        // Shrinking the TTL cannot retroactively expire a promotion: it carries
        // no deadline at all.
        assert_eq!(overlay.provenance_for(&id), Some(PqProvenance::OobVerified));
    }

    #[test]
    fn a_full_table_evicts_the_stalest_entry_rather_than_locking_new_clients_out() {
        let (overlay, _sink) = overlay_with_sink();
        let overlay = overlay.with_max_entries(2);
        let stale = ed_identity(13);
        let recent = ed_identity(14);
        let arriving = ed_identity(15);

        let _ = overlay.observe_first_contact(stale, &pq_key(26));
        let _ = overlay.observe_first_contact(recent, &pq_key(27));
        // Touch `recent` so `stale` is unambiguously the least-recently-seen.
        let _ = overlay.observe_first_contact(recent, &pq_key(27));

        assert!(matches!(
            overlay.observe_first_contact(arriving, &pq_key(28)),
            FirstContactOutcome::Recorded
        ));
        assert!(
            overlay.verifying_key_for(&arriving).is_some(),
            "a new client must not be locked out by a full table"
        );
        assert!(overlay.verifying_key_for(&stale).is_none());
        assert!(overlay.verifying_key_for(&recent).is_some());
    }

    #[test]
    fn a_promoted_binding_is_never_evicted() {
        let (overlay, _sink) = overlay_with_sink();
        let overlay = overlay.with_max_entries(1);
        let promoted = ed_identity(16);
        let key = pq_key(29);
        let _ = overlay.observe_first_contact(promoted, &key);
        overlay
            .promote_out_of_band(&promoted, &composite_fingerprint(&promoted, &key), "printed")
            .unwrap();

        // Every slot holds operator-established trust, so the arriving identity
        // is refused rather than that trust discarded.
        assert!(matches!(
            overlay.observe_first_contact(ed_identity(17), &pq_key(30)),
            FirstContactOutcome::CapacityExhausted
        ));
        assert_eq!(
            overlay.provenance_for(&promoted),
            Some(PqProvenance::OobVerified)
        );
    }

    #[test]
    fn surfacing_a_rebind_can_never_record() {
        let (overlay, sink) = overlay_with_sink();
        let id = ed_identity(18);
        let original = pq_key(31);
        let other = pq_key(32);

        // No binding at all: surfacing must not create one.
        overlay.surface_rebind(id, &other);
        assert!(overlay.verifying_key_for(&id).is_none());
        assert_eq!(CountingSink::count(&sink.refused), 0);

        let _ = overlay.observe_first_contact(id, &original);
        overlay.surface_rebind(id, &other);
        assert_eq!(CountingSink::count(&sink.refused), 1);
        assert_eq!(
            ml_dsa_vk_bytes(&overlay.verifying_key_for(&id).unwrap()),
            ml_dsa_vk_bytes(&original)
        );
    }

    #[test]
    fn the_fingerprint_covers_both_halves() {
        let a = ed_identity(15);
        let b = ed_identity(16);
        let k1 = pq_key(28);
        let k2 = pq_key(29);
        assert_ne!(composite_fingerprint(&a, &k1), composite_fingerprint(&b, &k1));
        assert_ne!(composite_fingerprint(&a, &k1), composite_fingerprint(&a, &k2));
    }

    #[test]
    fn fingerprint_display_is_grouped_for_reading_aloud() {
        let fp = composite_fingerprint(&ed_identity(17), &pq_key(30));
        let rendered = format_fingerprint(&fp);
        assert_eq!(rendered.split(' ').count(), 16);
        assert!(rendered
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_lowercase() || c == ' '));
    }
}
