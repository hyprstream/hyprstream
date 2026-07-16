//! B1: `did:at9p` update-record chain validation — the `successor-check` (#885).
//!
//! This module answers one question, purely and fail-closed: **given a
//! predecessor state (the genesis capsule or a previously-validated
//! update-record) and a candidate update-record, is the candidate an authorized
//! successor?** It decides *which key is authorized to sign the rotation* and
//! then enforces the four `successor-check` gates from design #879 §5.5/§7.1.
//!
//! # KERI-style pre-rotation (the trust model)
//!
//! `did:at9p` rotates keys the way KERI does: a capsule/record does not name the
//! key that will be *allowed* to rotate it next — it commits to that key's
//! **digest** (`next_key_commitments`, the pre-rotation commitment). A rotation
//! is an [`UpdateRecord`] that *reveals* the now-current keys in its
//! `new_capsule_body.subject_keys` and is signed by them; the successor-check
//! proves those revealed keys are exactly the ones the predecessor pre-committed
//! to. Because rotation authority lives in keys that were only ever published as
//! a hash until used, theft of a *current* signing key does not grant the
//! ability to rotate the identity (design §7.1, A11).
//!
//! # The four gates ([`validate_successor`])
//!
//! 1. **Signer authorization** — the update-record's signing key
//!    (`new_capsule_body.subject_keys[0]`, revealed by the rotation) must have
//!    its [`HybridKeyPair::commitment_digest`] present in the predecessor's
//!    `next_key_commitments`. This is the key-selection decision B1 owns: it
//!    picks *which* key [`verify_update_record`] must verify against, rather than
//!    trusting a caller-supplied key.
//! 2. **Linkage** — `prev_record_digest` equals the predecessor's record digest
//!    (`H512` of its canonical bytes), chaining the segment back toward genesis.
//! 3. **Monotonic epoch** — strictly increasing epoch.
//! 4. **Freshness** — `now < expires_at`, with `now` supplied by the caller
//!    (this module does no clock I/O — it is a pure function so tests and the
//!    duplicity layer can drive time deterministically).
//!
//! Any violation fails closed with a typed [`SuccessorError`] naming the failed
//! gate. Success returns the candidate's [`ChainState`], ready to serve as the
//! predecessor for the next link.
//!
//! # What this is NOT: duplicity detection (B2, #886)
//!
//! The successor-check validates a candidate *against a single predecessor*. It
//! deliberately does **not** implement duplicity/fork detection — a second,
//! divergent *but individually valid* successor at an epoch the client has
//! already accepted (design §7.2). That watermark comparison is B2's job, and
//! this API is shaped to be wrapped by it: [`ChainState`] carries exactly the
//! persistent `(epoch, record_digest)` pair B2 stores as its high-watermark, so
//! B2 can compare a freshly [`validate_successor`]-accepted state against the
//! persisted watermark and raise duplicity (hard fail + alarm) when a valid
//! record diverges from the recorded digest at `epoch <= watermark`.

use crate::at9p::{h512, Capsule, HybridKeyPair, UpdateRecord, H512_LEN};
use crate::at9p_sign::verify_update_record;

/// A validated point in an identity's key-rotation chain: either the genesis
/// capsule ([`ChainState::genesis`]) or a successfully validated update-record
/// (the return value of [`validate_successor`]).
///
/// It is simultaneously (a) the predecessor input to the next successor-check
/// and (b) the persistent duplicity **watermark** the B2 layer (#886) stores:
/// the `(epoch, record_digest)` pair plus the commitments needed to admit the
/// next link.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainState {
    /// The genesis cid512 that names this identity. Constant across the whole
    /// chain; every update-record must carry the same `subject_cid512`.
    pub subject_cid512: String,
    /// Epoch of this state. Genesis is epoch 0 by construction; the first
    /// update-record must therefore carry epoch >= 1.
    pub epoch: u64,
    /// `H512` over the canonical bytes of the record this state represents — the
    /// value the *next* update-record must echo in `prev_record_digest`, and the
    /// digest half of the B2 watermark.
    pub record_digest: [u8; H512_LEN],
    /// The keys this state pre-commits to for the *next* rotation. Empty means a
    /// declared-immutable identity (§7.1) — no update-record is ever a valid
    /// successor.
    pub next_key_commitments: Vec<[u8; H512_LEN]>,
}

impl ChainState {
    /// Build the predecessor state for a genesis capsule.
    ///
    /// The caller is responsible for having verified the capsule's own
    /// self-signature and its `H512(bytes) == cid512` gate first
    /// (`verify_capsule`, GATE 1/2); this only projects the fields the
    /// successor-check needs. The genesis record digest is `H512` of the
    /// capsule's canonical bytes — the same digest embedded in its cid512 — so
    /// the first update-record's `prev_record_digest` chains to it directly.
    pub fn genesis(capsule: &Capsule) -> anyhow::Result<Self> {
        Ok(Self {
            subject_cid512: capsule.cid512()?,
            epoch: 0,
            record_digest: h512(&capsule.to_dag_cbor()?),
            next_key_commitments: capsule.body.next_key_commitments.clone(),
        })
    }

    /// Project the predecessor state of an **already-validated** update-record —
    /// e.g. reconstructing the watermark from a persisted record on startup.
    /// For validating a fresh candidate, prefer [`validate_successor`], which
    /// returns this state only after all gates pass.
    pub fn from_validated_update(record: &UpdateRecord) -> Self {
        Self {
            subject_cid512: record.subject_cid512.clone(),
            epoch: record.epoch,
            // Already validated by contract; use the unchecked serializer so
            // this infallible projector stays infallible (a record that
            // reached here came from `from_dag_cbor` or the signing path).
            record_digest: h512(&record.encode_value()),
            next_key_commitments: record.new_capsule_body.next_key_commitments.clone(),
        }
    }
}

/// A typed successor-check failure. Every variant names the gate that failed;
/// callers (and the B2 duplicity layer) can match on the specific reason.
#[derive(Debug)]
#[non_exhaustive]
pub enum SuccessorError {
    /// The candidate names a different identity than the predecessor
    /// (`subject_cid512` mismatch) — it is not part of this chain at all.
    SubjectMismatch { expected: String, found: String },
    /// The predecessor pre-committed to no next keys: a declared-immutable
    /// identity (§7.1). Its chain is frozen at its current epoch and no
    /// update-record is ever a valid successor.
    ///
    /// The richer immutable-identity semantics are owned by B3 (#887); this is
    /// only the fail-closed boundary the successor-check enforces.
    ImmutableIdentity,
    /// Epoch did not strictly increase over the predecessor.
    NonMonotonicEpoch { predecessor: u64, candidate: u64 },
    /// `prev_record_digest` does not match the predecessor's record digest — the
    /// segment does not chain back to the accepted history.
    BrokenLinkage,
    /// The revealed signing key's commitment digest is absent from the
    /// predecessor's `next_key_commitments`: the key was never pre-rotated, so
    /// it is not authorized to rotate this identity.
    SignerNotCommitted,
    /// The pinned-Hybrid composite signature did not verify under the authorizing
    /// (pre-committed) key.
    SignatureInvalid(anyhow::Error),
    /// `now >= expires_at`: the record is stale (freshness gate, §7.2). `now` is
    /// caller-supplied.
    Expired { now: String, expires_at: String },
    /// The candidate record or a supplied timestamp was structurally malformed;
    /// fail-closed rather than guess.
    Malformed(anyhow::Error),
}

impl std::fmt::Display for SuccessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SubjectMismatch { expected, found } => write!(
                f,
                "successor-check failed [subject]: update names {found:?}, predecessor is {expected:?}"
            ),
            Self::ImmutableIdentity => write!(
                f,
                "successor-check failed [signer]: predecessor is a declared-immutable identity (empty next_key_commitments); no successor is valid"
            ),
            Self::NonMonotonicEpoch {
                predecessor,
                candidate,
            } => write!(
                f,
                "successor-check failed [epoch]: candidate epoch {candidate} does not strictly exceed predecessor epoch {predecessor}"
            ),
            Self::BrokenLinkage => write!(
                f,
                "successor-check failed [linkage]: prev_record_digest does not match the predecessor's record digest"
            ),
            Self::SignerNotCommitted => write!(
                f,
                "successor-check failed [signer]: signing key was not pre-committed in the predecessor's next_key_commitments"
            ),
            Self::SignatureInvalid(err) => {
                write!(f, "successor-check failed [signature]: {err}")
            }
            Self::Expired { now, expires_at } => write!(
                f,
                "successor-check failed [freshness]: now {now:?} is not before expires_at {expires_at:?}"
            ),
            Self::Malformed(err) => write!(f, "successor-check failed [malformed]: {err}"),
        }
    }
}

impl std::error::Error for SuccessorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SignatureInvalid(err) | Self::Malformed(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

/// Validate that `candidate` is an authorized successor of `predecessor` as of
/// `now`, enforcing every `successor-check` gate (design #879 §5.5) fail-closed.
///
/// `now` is an ISO-8601 UTC (`Z`-terminated) instant supplied by the caller;
/// this function performs no clock I/O so it stays pure and deterministic.
///
/// On success returns the candidate's [`ChainState`] — feed it back as the
/// `predecessor` for the next link, and (for the B2 duplicity layer, #886)
/// persist its `(epoch, record_digest)` as the identity's high-watermark.
pub fn validate_successor(
    predecessor: &ChainState,
    candidate: &UpdateRecord,
    now: &str,
) -> Result<ChainState, SuccessorError> {
    // Structural schema gates first: a malformed candidate is rejected before any
    // trust decision (fail-closed). `new_capsule_body.validate()` also guarantees
    // `subject_keys` is non-empty, which the signer selection below relies on.
    candidate
        .new_capsule_body
        .validate()
        .map_err(SuccessorError::Malformed)?;

    // Gate 0 (identity binding): the update must be for the same DID. The
    // subject_cid512 names the genesis capsule and is invariant across the chain.
    if candidate.subject_cid512 != predecessor.subject_cid512 {
        return Err(SuccessorError::SubjectMismatch {
            expected: predecessor.subject_cid512.clone(),
            found: candidate.subject_cid512.clone(),
        });
    }

    // A declared-immutable identity (empty commitments) has no valid successor.
    if predecessor.next_key_commitments.is_empty() {
        return Err(SuccessorError::ImmutableIdentity);
    }

    // Gate 3 (monotonic epoch): strictly increasing. Checked before the crypto so
    // a stale/replayed record is rejected cheaply. Also pins a valid record to its
    // position in the chain — a record valid at a later epoch does not validate
    // against an earlier predecessor.
    if candidate.epoch <= predecessor.epoch {
        return Err(SuccessorError::NonMonotonicEpoch {
            predecessor: predecessor.epoch,
            candidate: candidate.epoch,
        });
    }

    // Gate 2 (linkage): chain the segment back to the predecessor's record.
    if candidate.prev_record_digest != predecessor.record_digest {
        return Err(SuccessorError::BrokenLinkage);
    }

    // Gate 1 (signer authorization): the rotation reveals its now-current keys in
    // new_capsule_body.subject_keys; the primary key is the authorizing signer.
    // B1 selects THIS key for verification rather than trusting any caller input.
    let signer: &HybridKeyPair =
        candidate
            .new_capsule_body
            .subject_keys
            .first()
            .ok_or_else(|| {
                SuccessorError::Malformed(anyhow::anyhow!(
                    "update capsule body has no subject keys"
                ))
            })?;
    let signer_digest = signer.commitment_digest();
    if !predecessor
        .next_key_commitments
        .iter()
        .any(|commitment| commitment == &signer_digest)
    {
        return Err(SuccessorError::SignerNotCommitted);
    }

    // Gate 1 (signature): the pre-committed key must actually have signed the
    // record. Pinned-Hybrid (EdDSA + ML-DSA-65) verification (#939); done last as
    // the most expensive check.
    verify_update_record(candidate, signer).map_err(SuccessorError::SignatureInvalid)?;

    // Gate 4 (freshness): now < expires_at.
    if !datetime_before(now, &candidate.expires_at).map_err(SuccessorError::Malformed)? {
        return Err(SuccessorError::Expired {
            now: now.to_owned(),
            expires_at: candidate.expires_at.clone(),
        });
    }

    Ok(ChainState::from_validated_update(candidate))
}

/// Total ordering of two ISO-8601 UTC (`Z`-terminated) instants: returns
/// `Ok(true)` iff `a` is strictly before `b`. Pure integer parse — no clock, no
/// external date crate. Malformed input is an error (fail-closed), never a
/// silent `false`.
pub(crate) fn datetime_before(a: &str, b: &str) -> anyhow::Result<bool> {
    Ok(datetime_nanos(a)? < datetime_nanos(b)?)
}

/// Parse a `YYYY-MM-DDTHH:MM:SS[.fraction]Z` UTC instant into nanoseconds since
/// the Unix epoch, as an `i128` sort key. Fractional seconds are optional and
/// compared to nanosecond precision.
fn datetime_nanos(s: &str) -> anyhow::Result<i128> {
    let b = s.as_bytes();
    anyhow::ensure!(
        s.len() >= 20 && s.ends_with('Z'),
        "datetime must be ISO-8601 UTC ending in 'Z': {s:?}"
    );
    anyhow::ensure!(
        b.get(4) == Some(&b'-')
            && b.get(7) == Some(&b'-')
            && b.get(10) == Some(&b'T')
            && b.get(13) == Some(&b':')
            && b.get(16) == Some(&b':'),
        "datetime has malformed field separators: {s:?}"
    );

    let year: i64 = parse_int_field(s, 0..4, "year")?;
    let month: i64 = parse_int_field(s, 5..7, "month")?;
    let day: i64 = parse_int_field(s, 8..10, "day")?;
    let hour: i64 = parse_int_field(s, 11..13, "hour")?;
    let minute: i64 = parse_int_field(s, 14..16, "minute")?;
    let second: i64 = parse_int_field(s, 17..19, "second")?;

    anyhow::ensure!((1..=12).contains(&month), "month out of range: {s:?}");
    anyhow::ensure!((1..=31).contains(&day), "day out of range: {s:?}");
    anyhow::ensure!((0..=23).contains(&hour), "hour out of range: {s:?}");
    anyhow::ensure!((0..=59).contains(&minute), "minute out of range: {s:?}");
    // Allow a leap second (60) so real timestamps are not rejected.
    anyhow::ensure!((0..=60).contains(&second), "second out of range: {s:?}");

    // Fractional seconds live between index 19 and the trailing 'Z'.
    let tail = &s[19..s.len() - 1];
    let nanos_frac: i128 = if tail.is_empty() {
        0
    } else {
        anyhow::ensure!(
            tail.starts_with('.') && tail.len() > 1,
            "datetime fractional part must be a non-empty '.digits': {s:?}"
        );
        let mut digits = tail[1..].to_owned();
        anyhow::ensure!(
            digits.bytes().all(|d| d.is_ascii_digit()),
            "datetime fractional part must be digits: {s:?}"
        );
        // Normalize to exactly 9 fractional digits (nanoseconds).
        digits.truncate(9);
        while digits.len() < 9 {
            digits.push('0');
        }
        digits
            .parse::<i128>()
            .map_err(|e| anyhow::anyhow!("bad fractional seconds in {s:?}: {e}"))?
    };

    let days = days_from_civil(year, month, day);
    let secs = days
        .checked_mul(86_400)
        .and_then(|d| d.checked_add(hour * 3_600 + minute * 60 + second))
        .ok_or_else(|| anyhow::anyhow!("datetime overflow: {s:?}"))?;
    Ok(i128::from(secs) * 1_000_000_000 + nanos_frac)
}

fn parse_int_field(s: &str, range: std::ops::Range<usize>, field: &str) -> anyhow::Result<i64> {
    let slice = s
        .get(range)
        .ok_or_else(|| anyhow::anyhow!("datetime missing {field} field: {s:?}"))?;
    anyhow::ensure!(
        slice.bytes().all(|d| d.is_ascii_digit()),
        "datetime {field} field is not numeric: {s:?}"
    );
    slice
        .parse::<i64>()
        .map_err(|e| anyhow::anyhow!("datetime {field} parse error in {s:?}: {e}"))
}

/// Days from the Unix epoch (1970-01-01) for a proleptic-Gregorian date, via
/// Howard Hinnant's `days_from_civil` algorithm. Valid for all `month` in
/// `1..=12` and `day` in `1..=31`.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::at9p::{
        CapsuleBody, ServiceEndpoint, ServiceEntry, ServiceType, Transport, H512_LEN,
    };
    use crate::at9p_sign::{sign_capsule, sign_update_record};
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use rand::rngs::OsRng;

    const FUTURE: &str = "2099-01-01T00:00:00Z";

    /// A hybrid signer plus its public keypair (the pre-rotation commitment
    /// preimage).
    struct Signer {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        keypair: HybridKeyPair,
    }

    fn signer() -> Signer {
        let ed_sk = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let keypair = HybridKeyPair::new(
            ed_sk.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_vk),
        )
        .unwrap();
        Signer {
            ed_sk,
            pq_sk,
            keypair,
        }
    }

    fn service_entries() -> Vec<ServiceEntry> {
        let endpoint = ServiceEndpoint::new(Transport::Iroh, "iroh://node0").unwrap();
        vec![ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap()]
    }

    /// Build a capsule body whose primary subject key is `subject` and which
    /// pre-commits to each key in `next`.
    fn body_committing_to(subject: &Signer, next: &[&Signer]) -> CapsuleBody {
        let mut body = CapsuleBody::new(vec![subject.keypair.clone()], service_entries()).unwrap();
        body.next_key_commitments = next.iter().map(|s| s.keypair.commitment_digest()).collect();
        body
    }

    /// Genesis: subject key `g`, pre-committing to `n1`.
    fn genesis(g: &Signer, n1: &Signer) -> (Capsule, ChainState) {
        let body = body_committing_to(g, &[n1]);
        let capsule = sign_capsule(body, &g.ed_sk, &g.pq_sk).unwrap();
        let state = ChainState::genesis(&capsule).unwrap();
        (capsule, state)
    }

    /// Build an update-record signed by `signer_key`, revealing `signer_key` as
    /// the new primary subject key and pre-committing to `next`.
    fn update(
        subject_cid512: &str,
        epoch: u64,
        prev_digest: [u8; H512_LEN],
        signer_key: &Signer,
        next: &[&Signer],
        expires_at: &str,
    ) -> UpdateRecord {
        let body = body_committing_to(signer_key, next);
        sign_update_record(
            subject_cid512.to_owned(),
            epoch,
            prev_digest,
            body,
            expires_at.to_owned(),
            &signer_key.ed_sk,
            &signer_key.pq_sk,
        )
        .unwrap()
    }

    #[test]
    fn valid_successor_accepted() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, gen_state) = genesis(&g, &n1);
        let rec = update(
            &gen_state.subject_cid512,
            1,
            gen_state.record_digest,
            &n1,
            &[&n2],
            FUTURE,
        );
        let next_state = validate_successor(&gen_state, &rec, "2026-07-09T00:00:00Z")
            .expect("valid successor must be accepted");
        assert_eq!(next_state.epoch, 1);
        assert_eq!(next_state.record_digest, h512(&rec.to_dag_cbor().unwrap()));
        assert_eq!(next_state.subject_cid512, gen_state.subject_cid512);
    }

    #[test]
    fn multi_step_chain_extends() {
        let (g, n1, n2, n3) = (signer(), signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &n1);
        let r1 = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[&n2], FUTURE);
        let s1 = validate_successor(&s0, &r1, "2026-07-09T00:00:00Z").unwrap();
        let r2 = update(&s1.subject_cid512, 2, s1.record_digest, &n2, &[&n3], FUTURE);
        let s2 = validate_successor(&s1, &r2, "2026-07-09T00:00:00Z")
            .expect("second link must validate against the first");
        assert_eq!(s2.epoch, 2);
    }

    #[test]
    fn signer_not_committed_rejected() {
        let (g, n1, evil) = (signer(), signer(), signer());
        let (_cap, gen_state) = genesis(&g, &n1);
        // `evil` was never pre-committed; it signs and reveals itself.
        let rec = update(
            &gen_state.subject_cid512,
            1,
            gen_state.record_digest,
            &evil,
            &[&n1],
            FUTURE,
        );
        let err = validate_successor(&gen_state, &rec, "2026-07-09T00:00:00Z").unwrap_err();
        assert!(
            matches!(err, SuccessorError::SignerNotCommitted),
            "expected SignerNotCommitted, got {err:?}"
        );
    }

    #[test]
    fn broken_linkage_rejected() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, gen_state) = genesis(&g, &n1);
        // Sign over a WRONG prev digest so the signature is valid but linkage fails.
        let rec = update(
            &gen_state.subject_cid512,
            1,
            [0xAAu8; H512_LEN],
            &n1,
            &[&n2],
            FUTURE,
        );
        let err = validate_successor(&gen_state, &rec, "2026-07-09T00:00:00Z").unwrap_err();
        assert!(
            matches!(err, SuccessorError::BrokenLinkage),
            "expected BrokenLinkage, got {err:?}"
        );
    }

    #[test]
    fn non_increasing_epoch_rejected() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, gen_state) = genesis(&g, &n1);
        // Epoch 0 == genesis epoch → not strictly increasing.
        let rec = update(
            &gen_state.subject_cid512,
            0,
            gen_state.record_digest,
            &n1,
            &[&n2],
            FUTURE,
        );
        let err = validate_successor(&gen_state, &rec, "2026-07-09T00:00:00Z").unwrap_err();
        assert!(
            matches!(err, SuccessorError::NonMonotonicEpoch { .. }),
            "expected NonMonotonicEpoch, got {err:?}"
        );
    }

    #[test]
    fn expired_record_rejected() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, gen_state) = genesis(&g, &n1);
        let rec = update(
            &gen_state.subject_cid512,
            1,
            gen_state.record_digest,
            &n1,
            &[&n2],
            "2020-01-01T00:00:00Z",
        );
        // now is AFTER expires_at.
        let err = validate_successor(&gen_state, &rec, "2026-07-09T00:00:00Z").unwrap_err();
        assert!(
            matches!(err, SuccessorError::Expired { .. }),
            "expected Expired, got {err:?}"
        );
    }

    /// A record that is individually valid at its own position (epoch 2, chaining
    /// through r1) is rejected when checked against the WRONG predecessor
    /// (genesis): the pre-rotation binding ties a valid record to one epoch/link.
    #[test]
    fn valid_record_rejected_at_wrong_epoch() {
        let (g, n1, n2, n3) = (signer(), signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &n1);
        let r1 = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[&n2], FUTURE);
        let s1 = validate_successor(&s0, &r1, "2026-07-09T00:00:00Z").unwrap();
        // r2 is a perfectly valid successor of s1...
        let r2 = update(&s1.subject_cid512, 2, s1.record_digest, &n2, &[&n3], FUTURE);
        validate_successor(&s1, &r2, "2026-07-09T00:00:00Z").expect("r2 valid against s1");
        // ...but not against genesis: its epoch 2 chains through r1, not genesis,
        // and its signer (n2) was never committed by genesis. First failing gate
        // here is linkage.
        let err = validate_successor(&s0, &r2, "2026-07-09T00:00:00Z").unwrap_err();
        assert!(
            matches!(err, SuccessorError::BrokenLinkage),
            "expected BrokenLinkage against wrong predecessor, got {err:?}"
        );
    }

    /// B3 (#887) boundary: an identity that pre-committed to no next keys is
    /// declared-immutable — even an otherwise well-formed, correctly-signed
    /// update is rejected.
    #[test]
    fn immutable_identity_has_no_successor() {
        let (g, n1) = (signer(), signer());
        // Genesis with EMPTY next_key_commitments.
        let body = CapsuleBody::new(vec![g.keypair.clone()], service_entries()).unwrap();
        assert!(body.next_key_commitments.is_empty());
        let capsule = sign_capsule(body, &g.ed_sk, &g.pq_sk).unwrap();
        let gen_state = ChainState::genesis(&capsule).unwrap();
        let rec = update(
            &gen_state.subject_cid512,
            1,
            gen_state.record_digest,
            &n1,
            &[&n1],
            FUTURE,
        );
        let err = validate_successor(&gen_state, &rec, "2026-07-09T00:00:00Z").unwrap_err();
        assert!(
            matches!(err, SuccessorError::ImmutableIdentity),
            "expected ImmutableIdentity, got {err:?}"
        );
    }

    #[test]
    fn subject_mismatch_rejected() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, gen_state) = genesis(&g, &n1);
        // A totally different identity's genesis.
        let (other_g, other_n1) = (signer(), signer());
        let (_oc, other_state) = genesis(&other_g, &other_n1);
        let rec = update(
            &other_state.subject_cid512,
            1,
            gen_state.record_digest,
            &n1,
            &[&n2],
            FUTURE,
        );
        let err = validate_successor(&gen_state, &rec, "2026-07-09T00:00:00Z").unwrap_err();
        assert!(
            matches!(err, SuccessorError::SubjectMismatch { .. }),
            "expected SubjectMismatch, got {err:?}"
        );
    }

    #[test]
    fn datetime_before_orders_correctly() {
        assert!(datetime_before("2026-07-09T00:00:00Z", "2026-07-09T00:00:01Z").unwrap());
        assert!(!datetime_before("2026-07-09T00:00:01Z", "2026-07-09T00:00:00Z").unwrap());
        // Equal instant is NOT strictly before (freshness requires now < expires).
        assert!(!datetime_before("2026-07-09T00:00:00Z", "2026-07-09T00:00:00Z").unwrap());
        // Across a year/month/day boundary.
        assert!(datetime_before("2025-12-31T23:59:59Z", "2026-01-01T00:00:00Z").unwrap());
        // Fractional seconds honored.
        assert!(datetime_before("2026-07-09T00:00:00.100Z", "2026-07-09T00:00:00.200Z").unwrap());
        // Malformed input fails closed (Err, never a silent false).
        assert!(datetime_before("not-a-date", FUTURE).is_err());
    }

    #[test]
    fn tampered_signature_rejected() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, gen_state) = genesis(&g, &n1);
        let mut rec = update(
            &gen_state.subject_cid512,
            1,
            gen_state.record_digest,
            &n1,
            &[&n2],
            FUTURE,
        );
        // Flip a byte in the Ed25519 signature: still schema-valid, crypto-invalid.
        rec.signatures.ed25519_signature[0] ^= 0x01;
        let err = validate_successor(&gen_state, &rec, "2026-07-09T00:00:00Z").unwrap_err();
        assert!(
            matches!(err, SuccessorError::SignatureInvalid(_)),
            "expected SignatureInvalid, got {err:?}"
        );
    }
}
