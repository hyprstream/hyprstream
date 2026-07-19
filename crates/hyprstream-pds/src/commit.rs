//! Signed atproto repository commits.
//!
//! A commit is the **signed mutable pointer**: it binds an MST root CID (the
//! immutable record-store snapshot) to a DID at a revision, signed by the DID's
//! `#atproto` P-256 key. This is what federation verifies to trust a host's
//! claim about an account's state — the consensus-free versioning primitive.
//!
//! # Structure (atproto v3)
//!
//! ```text
//! Commit {
//!   did:     String,        // account DID (did:web / did:plc)
//!   version: u64 = 3,       // protocol version
//!   data:    Cid (link),    // MST root
//!   rev:     String,        // TID revision
//!   prev:    Option<Cid>,   // previous commit (None for genesis)
//!   sig:     Vec<u8>,       // ES256 signature over the unsigned commit
//! }
//! ```
//!
//! # Signing
//!
//! `sig` is ES256 (P-256 ECDSA over SHA-256) of the DAG-CBOR encoding of the
//! **unsigned commit** — the commit object with the `sig` field *omitted* (not
//! present-and-empty). The verifier re-encodes the unsigned form and checks the
//! signature against the DID's published `#atproto` P-256 verifying key.

use anyhow::{anyhow, bail, ensure, Result};
use p256::ecdsa::{signature::Signer, Signature, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;
use crate::tid::Tid;

/// The commit version number (atproto v3).
pub const COMMIT_VERSION: u64 = 3;

/// An unsigned commit — the form that gets DAG-CBOR-encoded and signed.
///
/// Field order matches atproto (`did`, `version`, `data`, `rev`, `prev`); the
/// encoder re-sorts canonically (**pure lexicographic byte order**, RFC 7049
/// §4.2.1 "core determinism" — not length-first) so the order here is only for
/// readability. `version` is always [`COMMIT_VERSION`] = 3.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsignedCommit {
    pub did: String,
    pub version: u64,
    pub data: Cid,
    pub rev: Tid,
    pub prev: Option<Cid>,
}

impl UnsignedCommit {
    pub fn new(did: impl Into<String>, data: Cid, rev: Tid, prev: Option<Cid>) -> Self {
        UnsignedCommit {
            did: did.into(),
            version: COMMIT_VERSION,
            data,
            rev,
            prev,
        }
    }

    /// DAG-CBOR encode the unsigned commit (no `sig` field). This is what gets signed.
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("did", DagCbor::Text(self.did.clone())),
            ("version", DagCbor::Unsigned(self.version)),
            ("data", DagCbor::Link(self.data)),
            ("rev", DagCbor::Text(self.rev.encode())),
            (
                "prev",
                match &self.prev {
                    Some(c) => DagCbor::Link(*c),
                    None => DagCbor::Null,
                },
            ),
        ])
    }
}

/// A signed commit — the unsigned commit plus its ES256 `sig`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Commit {
    pub did: String,
    pub version: u64,
    pub data: Cid,
    pub rev: Tid,
    pub prev: Option<Cid>,
    pub sig: Vec<u8>,
}

impl Commit {
    /// Sign an unsigned commit with a P-256 (`#atproto`) signing key,
    /// producing a [`Commit`].
    ///
    /// The signature is ES256: ECDSA over SHA-256 of the unsigned commit's
    /// DAG-CBOR bytes. The signature is the fixed-width (64-byte) R‖V‖S
    /// encoding — `Signature::to_vec` in `p256` 0.13 gives this form, which is
    /// the JOSE/COSE/atproto-preferred raw concatenation (not DER).
    pub fn sign(unsigned: &UnsignedCommit, key: &SigningKey) -> Self {
        let unsigned_bytes = unsigned.to_dag_cbor();
        // ES256 = ECDSA with SHA-256. p256's `Signer<Vec<u8>>` impl hashes
        // internally and emits the raw 64-byte R‖V‖S form.
        let sig: Signature = key.sign(&unsigned_bytes);
        Commit {
            did: unsigned.did.clone(),
            version: unsigned.version,
            data: unsigned.data,
            rev: unsigned.rev,
            prev: unsigned.prev,
            sig: sig.to_vec(),
        }
    }

    /// DAG-CBOR encode the (signed) commit. The `sig` field is a byte string.
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("did", DagCbor::Text(self.did.clone())),
            ("version", DagCbor::Unsigned(self.version)),
            ("data", DagCbor::Link(self.data)),
            ("rev", DagCbor::Text(self.rev.encode())),
            (
                "prev",
                match &self.prev {
                    Some(c) => DagCbor::Link(*c),
                    None => DagCbor::Null,
                },
            ),
            ("sig", DagCbor::Bytes(self.sig.clone())),
        ])
    }

    /// Classify this commit's account DID as an accepted repo authority for our
    /// PDS (`did:web`/`did:plc`/`did:at9p`), or `Err` if the method is not one we
    /// host a repo for.
    ///
    /// This is the method-level acceptance gate (#908); it is independent of and
    /// additional to the signature check in [`Commit::verify`]. A caller ingesting
    /// a commit should accept the authority *and* verify the signature.
    pub fn repo_authority(&self) -> Result<crate::repo_authority::RepoAuthority> {
        crate::repo_authority::accept_repo_authority(&self.did)
    }

    /// The [`UnsignedCommit`] form of this commit (drops `sig`).
    pub fn unsigned(&self) -> UnsignedCommit {
        UnsignedCommit {
            did: self.did.clone(),
            version: self.version,
            data: self.data,
            rev: self.rev,
            prev: self.prev,
        }
    }

    /// Decode a signed commit from DAG-CBOR bytes, validating structure.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let did = value
            .get("did")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'did'"))?
            .as_str()?
            .to_owned();
        let version = value
            .get("version")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'version'"))?
            .as_unsigned()?;
        ensure!(
            version == COMMIT_VERSION,
            "unsupported commit version {version} (expected {COMMIT_VERSION})"
        );
        let data = *value
            .get("data")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'data'"))?
            .as_link()?;
        let rev_str = value
            .get("rev")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'rev'"))?
            .as_str()?;
        let rev = Tid::parse(rev_str)?;
        let prev_val = value
            .get("prev")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'prev'"))?;
        let prev = if prev_val.is_null() {
            None
        } else {
            Some(*prev_val.as_link()?)
        };
        let sig = value
            .get("sig")
            .ok_or_else(|| anyhow::anyhow!("commit missing 'sig'"))?
            .as_bytes()?
            .to_vec();
        Ok(Commit {
            did,
            version,
            data,
            rev,
            prev,
            sig,
        })
    }

    /// The CID of this (signed) commit block.
    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.to_dag_cbor())
    }

    /// Verify the commit's signature against a `#atproto` P-256 verifying key.
    ///
    /// Re-encodes the unsigned commit, hashes with SHA-256, and verifies the
    /// ES256 signature. Returns `Ok(())` on success.
    ///
    /// This is the core of the D5 untrusted-host posture: a host can serve any
    /// commit it likes, but only commits signed by the account's `#atproto`
    /// key are accepted by verifiers.
    ///
    /// **Key-rotation caveat (#918):** this single-key form assumes the DID
    /// document still advertises the exact key that signed the commit. The
    /// ES256 `#atproto` store rotates on the order of days, and
    /// `oauth::did_document` publishes only the *active* slot, so a historical
    /// commit signed by a now-rotated-out key fails here even though it was
    /// legitimately signed. Callers that must verify commits across a rotation
    /// boundary should use [`Commit::verify_against_keys`] with the full set
    /// of currently-published `#atproto` slots (active + drain).
    pub fn verify(&self, vk: &VerifyingKey) -> Result<()> {
        use p256::ecdsa::signature::Verifier;
        let unsigned = self.unsigned();
        let unsigned_bytes = unsigned.to_dag_cbor();
        // ES256 verification: parse the signature, then verify over the bytes.
        // p256 0.13's `Signature::from_slice` accepts the 64-byte raw R‖V‖S form.
        let signature = Signature::from_slice(&self.sig)
            .map_err(|e| anyhow::anyhow!("invalid ES256 signature bytes: {e}"))?;
        vk.verify(&unsigned_bytes, &signature)
            .map_err(|e| anyhow::anyhow!("ES256 signature verification failed: {e}"))
    }

    /// Rotation-tolerant commit verification — the trust-chain fix for #918.
    ///
    /// Accepts the commit if its ES256 signature verifies under **any one** of
    /// the `#atproto` slots in `keys` whose `[nbf, exp]` validity window covers
    /// `now`. `keys` is the bounded, typed [`RotationKeySet`] constructed from
    /// the resolved DID document (or the authoritative rotation store that
    /// produced it) — never an arbitrary caller-supplied list of bare keys.
    ///
    /// # Why a typed bounded set, not bare keys
    ///
    /// A bare-`VerifyingKey` iterator would let a caller pass a cached or stale
    /// drain key and accept commits forged by a long-compromised
    /// rotated-out private key forever, or a lead key before its `nbf`. The
    /// validity window is therefore carried *on the slot* and enforced **inside
    /// this call**: only slots whose window covers `now` are even tried, and a
    /// slot whose `exp` has passed (or whose `nbf` is still in the future) is
    /// rejected by construction. The set is also capped at
    /// [`RotationKeySet::MAX_SLOTS`] so a malicious/buggy DID document cannot
    /// flood the verifier. See [`RotationKeySet`] for the construction contract.
    ///
    /// # Fail-closed contract
    ///
    /// - `keys` has no slot whose window covers `now` (empty set, all expired,
    ///   or all pre-`nbf`) → `Err`. The verifier refuses to accept a signature
    ///   it cannot bind to at least one currently-live authoritative slot.
    /// - At least one live slot verifies → `Ok(())`.
    /// - Live slot(s) but none verify → `Err` (the last attempted slot's
    ///   signature error is surfaced; the commit is untrusted).
    pub fn verify_against_keys(&self, keys: &RotationKeySet, now: i64) -> Result<()> {
        let mut count = 0usize;
        let mut last_err: Option<anyhow::Error> = None;
        for vk in keys.live_slots(now) {
            count += 1;
            match self.verify(vk) {
                Ok(()) => return Ok(()),
                Err(e) => last_err = Some(e),
            }
        }
        match last_err {
            Some(e) => Err(e.context(format!(
                "ES256 signature verified under none of the {count} live #atproto slot(s) at now={now}"
            ))),
            // No slot was live at `now` (empty set, all expired, or all
            // pre-nbf): fail closed — never accept a signature we cannot bind
            // to at least one currently-live authoritative slot.
            None => bail!(
                "no #atproto slot in the published set is live at now={now} \
                 — refusing to verify without a currently-valid authoritative slot"
            ),
        }
    }

    /// Compute the SHA-256 digest of the unsigned commit's DAG-CBOR bytes.
    ///
    /// Exposed for callers that compose their own signing/verification flow
    /// (e.g. using the DID's key material from `hyprstream::auth::key_rotation`
    /// directly). Standard ES256 signers hash internally, so most callers want
    /// [`Commit::sign`] / [`Commit::verify`] instead.
    pub fn unsigned_digest(unsigned: &UnsignedCommit) -> [u8; 32] {
        Sha256::digest(unsigned.to_dag_cbor()).into()
    }
}

// ============================================================================
// RotationKeySet — the bounded, typed #atproto slot set (#918)
// ============================================================================

/// A single `#atproto` verification slot: a P-256 verifying key plus the unix-
/// second `[nbf, exp]` validity window it is authoritative for.
///
/// **Opaque by construction** (#918 review r2): the fields are private and the
/// only builder is the crate-internal [`RotationKeySet::from_did_document`].
/// A public caller cannot mint an attacker-chosen `(key, window)` pair — the
/// authority and bounds come solely from a resolved, freshness-attested DID
/// document.
#[derive(Clone, Debug)]
pub struct RotationKeySlot {
    vk: VerifyingKey,
    nbf: i64,
    exp: i64,
}

impl RotationKeySlot {
    /// A slot authoritative for the whole of time (the default for the active
    /// `#atproto` key, whose window the DID document does not bound).
    pub(crate) const UNBOUNDED_NBF: i64 = i64::MIN;
    pub(crate) const UNBOUNDED_EXP: i64 = i64::MAX;

    /// Build a slot with an explicit `[nbf, exp]` window (unix seconds,
    /// inclusive on both ends). Crate-internal: only `from_did_document`
    /// constructs slots.
    pub(crate) fn new(vk: VerifyingKey, nbf: i64, exp: i64) -> Self {
        Self { vk, nbf, exp }
    }

    /// Build an (active-slot-style) unbounded slot. Crate-internal.
    pub(crate) fn unbounded(vk: VerifyingKey) -> Self {
        Self {
            vk,
            nbf: Self::UNBOUNDED_NBF,
            exp: Self::UNBOUNDED_EXP,
        }
    }

    /// Whether this slot's window covers `now` (inclusive on both ends).
    pub fn is_live_at(&self, now: i64) -> bool {
        now >= self.nbf && now <= self.exp
    }

    /// The slot's verifying key.
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.vk
    }

    /// The slot's `nbf` bound (unix seconds).
    pub fn nbf(&self) -> i64 {
        self.nbf
    }

    /// The slot's `exp` bound (unix seconds).
    pub fn exp(&self) -> i64 {
        self.exp
    }
}

/// Absolute ceiling on the number of slots a [`RotationKeySet`] will carry.
///
/// A resolved DID document advertising more `#atproto`-family verification
/// methods than this is treated as malformed (the producer publishes at most
/// two: `#atproto` + `#atproto-drain`); the cap exists so a malicious or
/// buggy document cannot flood the verifier.
pub const ROTATION_MAX_SLOTS: usize = 8;

/// The bounded, typed set of `#atproto` verification slots a verifier accepts
/// commits under at a given instant — the rotation-survivable trust root for
/// #918.
///
/// **Sealed API (#918 review r2):** the ONLY public constructor is
/// [`RotationKeySet::from_did_document`], which is authority-validated and
/// freshness-bound (it cross-checks the document's active key against the
/// authoritative active key the resolver supplies, so a stale pre-rotation
/// document cannot smuggle a rotated-out key in as an unbounded slot). The
/// raw [`RotationKeySet::testing_from_slots`] constructor is gated
/// `#[cfg(test)]`. A public caller cannot construct a set from bare keys.
///
/// [`Commit::verify_against_keys`] enforces each slot's `[nbf, exp]` window at
/// verify time so a cached/stale drain key cannot accept commits forged after
/// its expiry, nor a lead key before its `nbf`.
///
/// # Design: drain-slot publication (chosen), not re-sign-on-rotation
///
/// Issue #918 lists two candidate fixes:
///
/// 1. **Drain-slot publication** *(chosen)* — publish the drain (and lead)
///    ES256 slots as additional `#atproto`-style verification methods in the
///    DID document, and have the verifier accept any currently-published slot
///    whose window is live. This keeps verification fail-closed, preserves
///    commits byte-for-byte (history is not rewritten), and bounds the
///    accepted key set by the ES256 store's `drain_secs` window.
/// 2. **Re-sign on rotation** *(rejected)* — when the active ES256 key
///    rotates, walk every stored commit and re-sign it with the new active
///    key. This rewrites persisted history invisibly (a commit CID recorded at
///    *T* no longer matches the served commit at *T+1*) and couples the
///    rotation task to the PDS store. The signed mutable pointer stays
///    write-once at publish time; the drain window expires old slots loudly.
#[derive(Clone, Debug)]
pub struct RotationKeySet {
    slots: Vec<RotationKeySlot>,
}

impl RotationKeySet {
    /// Maximum number of slots a set will carry (see [`ROTATION_MAX_SLOTS`]).
    pub const MAX_SLOTS: usize = ROTATION_MAX_SLOTS;

    /// Construct the key set from a resolved DID document, validated against
    /// the authoritative active key the resolver supplies (#918 review r2).
    ///
    /// `doc` is the DID document resolved for `expected_did`. `authoritative_active`
    /// is the key the resolver knows is the **current** `#atproto` active key
    /// (from the live rotation store, or a freshness-attested fetch). The two
    /// are cross-checked: the document's `#atproto` method MUST verify against
    /// `authoritative_active`. A stale pre-rotation document — whose `#atproto`
    /// is the now-rotated-out key — therefore fails closed rather than being
    /// reconstructed as an unbounded slot for the rotated-out key.
    ///
    /// Authority validation (all fail closed):
    /// - the document's `id` MUST equal `expected_did`;
    /// - exactly one `#atproto` and at most one `#atproto-drain` method
    ///   (duplicates are rejected, not silently first-matched);
    /// - each method is `type: "Multikey"` and `controller: expected_did`;
    /// - the `#atproto` key equals `authoritative_active` (the freshness gate);
    /// - `#atproto-drain` carries explicit integer `nbf`/`exp`.
    ///
    /// The active `#atproto` slot is unbounded unless the document advertises
    /// explicit `nbf`/`exp` on it (then those are honored); `#atproto-drain`
    /// MUST carry `nbf`/`exp` — an unbounded drain key is exactly the
    /// stale-key-forever failure this type prevents.
    pub fn from_did_document(
        doc: &serde_json::Value,
        expected_did: &str,
        authoritative_active: &VerifyingKey,
    ) -> Result<Self> {
        // Authority: the document must be for the DID being resolved.
        let doc_id = doc
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow!("DID document has no `id`"))?;
        ensure!(
            doc_id == expected_did,
            "DID document id {doc_id:?} does not match the DID being resolved {expected_did:?}"
        );
        let vms = doc
            .get("verificationMethod")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow!("DID document has no verificationMethod array"))?;

        // Match methods by FULL id (`{did}#atproto` / `{did}#atproto-drain`),
        // not bare suffix — a `#atproto` from a different controller/DID must
        // not be picked up. Collect exactly one active and at most one drain;
        // duplicates are a malformed/attacker document and fail closed.
        let atproto_id = format!("{expected_did}#atproto");
        let drain_id = format!("{expected_did}#atproto-drain");
        let mut atproto_vm: Option<&serde_json::Value> = None;
        let mut drain_vm: Option<&serde_json::Value> = None;
        for vm in vms {
            let Some(id) = vm.get("id").and_then(serde_json::Value::as_str) else {
                continue;
            };
            if id == atproto_id {
                if atproto_vm.is_some() {
                    bail!("DID document has more than one `{atproto_id}` method");
                }
                atproto_vm = Some(vm);
            } else if id == drain_id {
                if drain_vm.is_some() {
                    bail!("DID document has more than one `{drain_id}` method");
                }
                drain_vm = Some(vm);
            }
            // Other VMs are ignored (mesh keys, etc.) — they are not #atproto.
        }

        let atproto = atproto_vm.ok_or_else(|| {
            anyhow!("DID document has no `{atproto_id}` verification method")
        })?;
        validate_vm_authority(atproto, &atproto_id, expected_did)?;
        let atproto_vk = decode_p256_multibase_vm(atproto, "#atproto")?;
        // Freshness gate: the document's active key MUST equal the authoritative
        // current active key. A stale pre-rotation document (active = the
        // rotated-out key) fails here — it cannot be smuggled in as an
        // unbounded slot.
        ensure!(
            atproto_vk == *authoritative_active,
            "DID document's #atproto key does not match the authoritative current active key \
             — refusing a stale/pre-rotation document"
        );
        // The active `#atproto` slot defaults to unbounded (atproto does not
        // publish its window), BUT if the DID document advertises explicit
        // `nbf`/`exp` on it, honor them — the verifier enforces whatever the
        // authoritative document publishes.
        let atproto_slot = match (
            atproto.get("nbf").and_then(serde_json::Value::as_i64),
            atproto.get("exp").and_then(serde_json::Value::as_i64),
        ) {
            (Some(nbf), Some(exp)) => {
                ensure!(
                    exp >= nbf,
                    "#atproto slot has inverted window: nbf={nbf} exp={exp}"
                );
                RotationKeySlot::new(atproto_vk, nbf, exp)
            }
            (None, None) => RotationKeySlot::unbounded(atproto_vk),
            _ => bail!(
                "#atproto slot must carry both `nbf` and `exp`, or neither (partial bounds rejected)"
            ),
        };
        let mut slots: Vec<RotationKeySlot> = vec![atproto_slot];

        // The optional #atproto-drain slot, bounded by explicit nbf/exp.
        if let Some(drain) = drain_vm {
            validate_vm_authority(drain, &drain_id, expected_did)?;
            let vk = decode_p256_multibase_vm(drain, "#atproto-drain")?;
            let nbf = drain
                .get("nbf")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| anyhow!("#atproto-drain slot must carry integer `nbf` bounds"))?;
            let exp = drain
                .get("exp")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| anyhow!("#atproto-drain slot must carry integer `exp` bounds"))?;
            ensure!(
                exp >= nbf,
                "#atproto-drain slot has inverted window: nbf={nbf} exp={exp}"
            );
            slots.push(RotationKeySlot::new(vk, nbf, exp));
        }
        Self::from_slots_inner(slots)
    }

    /// Test-only constructor from explicit slots (`#[cfg(test)]`-gated so it
    /// is absent from non-test builds). Production code MUST construct the set
    /// via [`RotationKeySet::from_did_document`], the only authority-validated
    /// path; this raw constructor exists purely for focused unit tests.
    #[cfg(test)]
    pub fn testing_from_slots(slots: Vec<RotationKeySlot>) -> Result<Self> {
        Self::from_slots_inner(slots)
    }

    fn from_slots_inner(slots: Vec<RotationKeySlot>) -> Result<Self> {
        ensure!(
            !slots.is_empty(),
            "a RotationKeySet must carry at least one #atproto slot"
        );
        ensure!(
            slots.len() <= Self::MAX_SLOTS,
            "rotation key set has {} slots, exceeding the cap of {}",
            slots.len(),
            Self::MAX_SLOTS
        );
        Ok(Self { slots })
    }

    /// The active `#atproto` slot (the first slot, always present). Exposed so
    /// a resolver/freshness layer can read the document-advertised active key
    /// and cross-check it against the authoritative rotation store.
    pub fn active_slot(&self) -> &RotationKeySlot {
        // The first slot is the active `#atproto` by construction.
        &self.slots[0]
    }

    /// Number of slots in the set (regardless of whether they are live now).
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the set carries zero slots. (Construction forbids empty, but
    /// callers may want a defensive check after resolution.)
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Iterate the verifying keys whose `[nbf, exp]` window covers `now`.
    ///
    /// This is the live-at-`now` filter [`Commit::verify_against_keys`] applies
    /// before attempting any signature, so an expired drain key or a pre-`nbf`
    /// lead key is never consulted.
    pub fn live_slots(&self, now: i64) -> impl Iterator<Item = &VerifyingKey> {
        self.slots.iter().filter_map(move |s| {
            if s.is_live_at(now) {
                Some(&s.vk)
            } else {
                None
            }
        })
    }
}

/// Decode a `publicKeyMultibase` (multibase `z` + base58btc + multicodec
/// `p256-pub` `0x80 0x24` + 33-byte compressed SEC1) into a P-256 verifying
/// key, with the multicodec prefix checked.
///
/// `label` is the verification-method id suffix (`#atproto` / `#atproto-drain`)
/// used in error messages; the caller already matched it.
fn decode_p256_multibase_vm(vm: &serde_json::Value, label: &str) -> Result<VerifyingKey> {
    const P256_PUB_MULTICODEC: [u8; 2] = [0x80, 0x24];
    let mb = vm
        .get("publicKeyMultibase")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("{label} verification method has no publicKeyMultibase string"))?;
    ensure!(
        mb.starts_with('z'),
        "{label} publicKeyMultibase must be multibase 'z' (base58btc)"
    );
    let payload = bs58::decode(&mb[1..])
        .into_vec()
        .map_err(|e| anyhow!("{label} publicKeyMultibase is not valid base58btc: {e}"))?;
    ensure!(
        payload.len() == 2 + 33,
        "{label} publicKeyMultibase payload is {} bytes, expected 35 (multicodec + compressed SEC1)",
        payload.len()
    );
    ensure!(
        payload[0] == P256_PUB_MULTICODEC[0] && payload[1] == P256_PUB_MULTICODEC[1],
        "{label} publicKeyMultibase has wrong multicodec prefix (not p256-pub 0x1200)"
    );
    VerifyingKey::from_sec1_bytes(&payload[2..])
        .map_err(|e| anyhow!("{label} publicKeyMultibase SEC1 point invalid: {e}"))
}

/// Validate a `#atproto`-family verification method's authority: it MUST be
/// `type: "Multikey"` and `controller: expected_did`. The id match (full
/// `{did}#fragment`) is established by the caller; this checks the rest of the
/// method so a method with the right id but a foreign controller/type is
/// rejected (a document carrying an attacker VM with a spoofed id still fails
/// closed on controller/type).
fn validate_vm_authority(
    vm: &serde_json::Value,
    full_id: &str,
    expected_did: &str,
) -> Result<()> {
    let vm_type = vm
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("`{full_id}` method has no `type`"))?;
    ensure!(
        vm_type == "Multikey",
        "`{full_id}` method type is {vm_type:?}, expected \"Multikey\""
    );
    let controller = vm
        .get("controller")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("`{full_id}` method has no `controller`"))?;
    ensure!(
        controller == expected_did,
        "`{full_id}` method controller is {controller:?}, expected {expected_did:?}"
    );
    Ok(())
}

/// Encode a P-256 verifying key as the multibase `z` + base58btc + multicodec
/// `p256-pub` (`0x80 0x24`) + compressed SEC1 string the DID-document producer
/// publishes — the symmetric partner of `decode_p256_multibase_vm`.
///
/// Exposed so a resolver that has a single authoritative `#atproto` key (the
/// legacy single-key seam) can build a minimal DID-document fragment and
/// construct a [`RotationKeySet`] via the ONLY public constructor
/// ([`RotationKeySet::from_did_document`]), keeping the API sealed: there is
/// no public bare-key constructor.
pub fn p256_public_key_multibase(vk: &VerifyingKey) -> String {
    use p256::EncodedPoint;
    let point: EncodedPoint = vk.to_encoded_point(true);
    let mut payload = vec![0x80, 0x24];
    payload.extend_from_slice(point.as_bytes());
    format!("z{}", bs58::encode(payload).into_string())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;
    use crate::record::{self, ModelRecord};
    use crate::tid::Tid;
    use std::collections::BTreeMap;

    fn make_signed_commit() -> (Commit, VerifyingKey) {
        // Build a small MST, sign a commit over its root.
        let mut recs = BTreeMap::new();
        for i in 0..3 {
            let tid = Tid::from_micros(1_700_000_000_000_000 + i, i as u16);
            let rec = ModelRecord::new(
                "at://did:web:alice.example.com",
                format!("bafyreiexampleoid{i:020}"),
                "2026-06-23T12:34:56.789Z",
            )
            .expect("record");
            recs.insert(tid, rec.cid());
        }
        let tree = crate::mst::Node::from_records(record::COLLECTION_NSID, &recs);
        let root = tree.root_cid();

        let signing_key = SigningKey::random(&mut rand::rngs::OsRng);
        let verifying_key = VerifyingKey::from(&signing_key);

        let unsigned = UnsignedCommit::new("did:web:alice.example.com", root, Tid::now(), None);
        let commit = Commit::sign(&unsigned, &signing_key);
        (commit, verifying_key)
    }

    #[test]
    fn commit_sign_and_verify_round_trip() {
        let (commit, vk) = make_signed_commit();
        commit.verify(&vk).expect("signature must verify");
    }

    #[test]
    fn commit_detects_wrong_key() {
        let (commit, _vk) = make_signed_commit();
        // A different key must fail.
        let other = SigningKey::random(&mut rand::rngs::OsRng);
        let other_vk = VerifyingKey::from(other);
        assert!(
            commit.verify(&other_vk).is_err(),
            "signature must not verify under a different key"
        );
    }

    #[test]
    fn commit_detects_tampered_data() {
        let (mut commit, vk) = make_signed_commit();
        // Tamper with the MST root pointer — signature must fail.
        commit.data = Cid::from_dag_cbor(b"tampered");
        assert!(
            commit.verify(&vk).is_err(),
            "tampered data must fail signature verify"
        );
    }

    #[test]
    fn commit_dag_cbor_round_trip() {
        let (commit, _vk) = make_signed_commit();
        let bytes = commit.to_dag_cbor();
        let back = Commit::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(commit, back);
    }

    #[test]
    fn commit_unsigned_has_no_sig_field() {
        // The unsigned encoding MUST NOT carry a sig field (atproto signs the
        // object without sig, not sig=empty).
        let unsigned = UnsignedCommit::new(
            "did:web:x",
            Cid::from_dag_cbor(b"r"),
            Tid::from_raw(1),
            None,
        );
        let v = unsigned.to_value();
        assert!(
            v.get("sig").is_none(),
            "unsigned commit must not have a sig field"
        );
    }

    // ── #918: rotation-survivable verification ───────────────────────────────

    /// A commit signed by a since-rotated-out key must still verify against the
    /// published slot set (active + drain) while the drain window is live —
    /// the drain-slot design.
    #[test]
    fn commit_verify_survives_key_rotation_within_drain_window() {
        let (commit, original_vk) = make_signed_commit();
        // The active key rotates: a brand-new key replaces it.
        let new_signing = SigningKey::random(&mut rand::rngs::OsRng);
        let new_vk = VerifyingKey::from(&new_signing);
        let now = 10_000;

        // Single-key verify against the *new* (current DID document) key fails —
        // this is the #918 regression: the historical commit looks untrusted.
        assert!(
            commit.verify(&new_vk).is_err(),
            "historical commit must not verify under the rotated-in key alone"
        );

        // Drain window covers `now`: active=new (unbounded), drain=old [0, 20000].
        let set = RotationKeySet::testing_from_slots(vec![
            RotationKeySlot::unbounded(new_vk),
            RotationKeySlot::new(original_vk, 0, 20_000),
        ])
        .unwrap();
        commit
            .verify_against_keys(&set, now)
            .expect("commit signed by a live drain slot must verify");
    }

    /// Same commit AFTER the drain window has expired must fail: the expired
    /// drain key is filtered out by `live_slots(now)` and only the rotated-in
    /// active key remains live, under which the historical commit does not
    /// verify.
    #[test]
    fn commit_verify_fails_after_drain_expiry() {
        let (commit, original_vk) = make_signed_commit();
        let new_vk = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        // Drain window [0, 20000]; `now` is past expiry.
        let set = RotationKeySet::testing_from_slots(vec![
            RotationKeySlot::unbounded(new_vk),
            RotationKeySlot::new(original_vk, 0, 20_000),
        ])
        .unwrap();
        let err = commit
            .verify_against_keys(&set, 30_000)
            .expect_err("expired drain key must not be consulted");
        let msg = format!("{err}");
        assert!(
            msg.contains("none of the") || msg.contains("live"),
            "expiry error should name the live-slot filter, got: {msg}"
        );
    }

    /// A drain slot whose `nbf` is still in the future must not be consulted
    /// either (the lead-key-before-nbf case).
    #[test]
    fn commit_verify_fails_before_nbf() {
        let (commit, original_vk) = make_signed_commit();
        let set = RotationKeySet::testing_from_slots(vec![
            // The only live slot is the pre-nbf drain key; active is a random
            // (non-signing) key whose window is also pre-nbf, so no slot is
            // live at `now`.
            RotationKeySlot::new(original_vk, 100_000, 200_000),
        ])
        .unwrap();
        let err = commit
            .verify_against_keys(&set, 50_000)
            .expect_err("pre-nbf slot must not be consulted");
        let msg = format!("{err}");
        assert!(
            msg.contains("no #atproto slot"),
            "pre-nbf error should explain no live slot, got: {msg}"
        );
    }

    /// A genuinely bad signature must still fail under rotation-tolerant verify.
    #[test]
    fn commit_verify_against_keys_rejects_bad_signature() {
        let (commit, _original_vk) = make_signed_commit();
        // Two unrelated keys the commit was never signed by, both live.
        let a = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        let b = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        let set = RotationKeySet::testing_from_slots(vec![
            RotationKeySlot::unbounded(a),
            RotationKeySlot::unbounded(b),
        ])
        .unwrap();
        assert!(
            commit.verify_against_keys(&set, 0).is_err(),
            "a signature under no published slot must fail, not silently fall back"
        );
    }

    /// Construction with zero slots is refused — the set can never be empty.
    #[test]
    fn rotation_key_set_rejects_empty() {
        let err = RotationKeySet::testing_from_slots(vec![])
            .expect_err("empty slot set must fail construction");
        let msg = format!("{err}");
        assert!(
            msg.contains("at least one #atproto slot"),
            "empty-set error should explain fail-closed reason, got: {msg}"
        );
    }

    /// Rotation-tolerant verify still detects a tampered commit across a slot set.
    #[test]
    fn commit_verify_against_keys_detects_tampered_data() {
        let (mut commit, original_vk) = make_signed_commit();
        let new_vk = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        commit.data = Cid::from_dag_cbor(b"tampered");
        let set = RotationKeySet::testing_from_slots(vec![
            RotationKeySlot::unbounded(new_vk),
            RotationKeySlot::unbounded(original_vk),
        ])
        .unwrap();
        assert!(
            commit.verify_against_keys(&set, 0).is_err(),
            "tampered commit must fail under every published slot"
        );
    }

    /// Helper: encode a P-256 verifying key as the multibase `z` + base58btc +
    /// multicodec `p256-pub` (`0x80 0x24`) + compressed SEC1 string the DID
    /// document producer emits — mirrors `did_document::p256_to_multibase`.
    fn p256_to_multibase(vk: &VerifyingKey) -> String {
        use p256::EncodedPoint;
        let point: EncodedPoint = vk.to_encoded_point(true);
        let mut payload = vec![0x80, 0x24];
        payload.extend_from_slice(point.as_bytes());
        format!("z{}", bs58::encode(payload).into_string())
    }

    /// `RotationKeySet::from_did_document` parses the producer's DID-document
    /// shape (active `#atproto` + bounded `#atproto-drain`) and the resulting
    /// set verifies a drained-key commit within the window. This is the
    /// parser-level round-trip; the producer/consumer round-trip lives in the
    /// `hyprstream` / `hyprstream-discovery` integration tests.
    #[test]
    fn rotation_key_set_from_did_document_round_trip() {
        let (commit, drain_vk) = make_signed_commit();
        let active_vk = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        let did = "did:web:alice.example.com";
        let doc = serde_json::json!({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": did,
            "verificationMethod": [
                {
                    "id": format!("{did}#atproto"),
                    "type": "Multikey",
                    "controller": did,
                    "publicKeyMultibase": p256_to_multibase(&active_vk),
                },
                {
                    "id": format!("{did}#atproto-drain"),
                    "type": "Multikey",
                    "controller": did,
                    "publicKeyMultibase": p256_to_multibase(&drain_vk),
                    "nbf": 0,
                    "exp": 20_000,
                },
            ],
        });
        let set = RotationKeySet::from_did_document(&doc, did, &active_vk).expect("parse");
        // Within the drain window: drained-key commit verifies.
        commit
            .verify_against_keys(&set, 10_000)
            .expect("drained-key commit verifies within the drain window");
        // After expiry: drain slot filtered out, only active (non-signing) key
        // live → historical commit fails.
        assert!(
            commit.verify_against_keys(&set, 30_000).is_err(),
            "after drain expiry the historical commit must fail"
        );
    }

    /// A `#atproto-drain` slot without explicit `nbf`/`exp` is rejected — an
    /// unbounded drain key is exactly the stale-key-forever failure this type
    /// exists to prevent.
    #[test]
    fn rotation_key_set_rejects_unbounded_drain() {
        let (commit, drain_vk) = make_signed_commit();
        let active_vk = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        let did = "did:web:alice.example.com";
        let doc = serde_json::json!({
            "id": did,
            "verificationMethod": [
                {
                    "id": format!("{did}#atproto"),
                    "type": "Multikey",
                    "controller": did,
                    "publicKeyMultibase": p256_to_multibase(&active_vk),
                },
                {
                    "id": format!("{did}#atproto-drain"),
                    "type": "Multikey",
                    "controller": did,
                    "publicKeyMultibase": p256_to_multibase(&drain_vk),
                    // No nbf/exp — must be rejected.
                },
            ],
        });
        let err = RotationKeySet::from_did_document(&doc, did, &active_vk)
            .expect_err("unbounded drain must be rejected");
        let _ = commit;
        let msg = format!("{err}");
        assert!(
            msg.contains("`nbf`") || msg.contains("`exp`"),
            "missing-bounds error should name nbf/exp, got: {msg}"
        );
    }

    /// #918 review r2 (stale-document downgrade): a pre-rotation document whose
    /// `#atproto` is the now-rotated-out key MUST fail closed — the resolver
    /// supplies the authoritative CURRENT active key (K'), and the stale
    /// document's `#atproto` (K) does not match, so `from_did_document`
    /// refuses instead of reconstructing K as an unbounded slot.
    #[test]
    fn from_did_document_rejects_stale_pre_rotation_document() {
        let (_commit_signed_by_old, old_vk) = make_signed_commit();
        // The active key has since rotated to a brand-new key K'.
        let current_active = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
        let did = "did:web:alice.example.com";
        // Stale document: still advertises the OLD key as `#atproto`, no drain.
        let stale_doc = serde_json::json!({
            "id": did,
            "verificationMethod": [{
                "id": format!("{did}#atproto"),
                "type": "Multikey",
                "controller": did,
                "publicKeyMultibase": p256_to_multibase(&old_vk),
            }],
        });
        let err = RotationKeySet::from_did_document(&stale_doc, did, &current_active)
            .expect_err("a stale pre-rotation document must fail closed");
        let msg = format!("{err}");
        assert!(
            msg.contains("authoritative current active key")
                || msg.contains("stale/pre-rotation"),
            "stale-document error should name the freshness gate, got: {msg}"
        );
    }

    /// #918 review r2 (authority validation): a document whose `id` is for a
    /// different DID must fail closed (the set is DID-bound, not suffix-bound).
    #[test]
    fn from_did_document_rejects_wrong_did() {
        let (_c, vk) = make_signed_commit();
        let did = "did:web:alice.example.com";
        let other_did = "did:web:mallory.example.com";
        let doc = serde_json::json!({
            "id": other_did,
            "verificationMethod": [{
                "id": format!("{other_did}#atproto"),
                "type": "Multikey",
                "controller": other_did,
                "publicKeyMultibase": p256_to_multibase(&vk),
            }],
        });
        assert!(
            RotationKeySet::from_did_document(&doc, did, &vk).is_err(),
            "a document for a different DID must fail closed"
        );
    }

    /// #918 review r2 (duplicate methods): two `#atproto` methods must fail
    /// closed rather than silently first-matching.
    #[test]
    fn from_did_document_rejects_duplicate_atproto() {
        let (_c, vk) = make_signed_commit();
        let did = "did:web:alice.example.com";
        let doc = serde_json::json!({
            "id": did,
            "verificationMethod": [
                { "id": format!("{did}#atproto"), "type": "Multikey",
                  "controller": did, "publicKeyMultibase": p256_to_multibase(&vk) },
                { "id": format!("{did}#atproto"), "type": "Multikey",
                  "controller": did, "publicKeyMultibase": p256_to_multibase(&vk) },
            ],
        });
        let err = RotationKeySet::from_did_document(&doc, did, &vk)
            .expect_err("duplicate #atproto must fail closed");
        assert!(
            format!("{err}").contains("more than one"),
            "duplicate-method error should be explicit, got: {err}"
        );
    }
}
