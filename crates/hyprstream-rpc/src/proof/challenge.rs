//! Server challenge protocol for unattributed proofs.
//!
//! For unattributed proofs the server challenge (CWT `Nonce`, RFC 9711) is
//! REQUIRED, not optional. The server maintains a rotating opaque challenge
//! with a short reviewed window. Clients MAY reuse the current challenge
//! across requests while `request_id` remains unique per request. A missing,
//! expired, or unknown challenge in an unattributed proof denies before any
//! replay-store insertion.
//!
//! Every `DispatchDenied` response carries the current challenge uniformly,
//! regardless of the internal denial cause. A client without a challenge
//! sends its request, receives `DispatchDenied` bearing a usable challenge,
//! and performs at most one immediate bounded retry with a fresh
//! `request_id`.

use super::{MAX_CHALLENGE_BYTES, MIN_CHALLENGE_BYTES};

/// A rotating server challenge with its acceptance deadline.
#[derive(Debug, Clone)]
pub struct ServerChallenge {
    /// The opaque challenge value (16..64 bytes).
    pub value: Vec<u8>,
    /// The nominal window end (Unix seconds).
    pub window_end: u64,
    /// The acceptance deadline: `window_end` plus one bounded overlap for
    /// retry and clock skew. A proof citing this value verifies iff
    /// `verifier_now < challenge_accept_until` everywhere in the replay
    /// admission domain.
    pub accept_until: u64,
}

impl ServerChallenge {
    /// Create a new challenge with the given value and deadline.
    pub fn new(value: Vec<u8>, window_end: u64, accept_until: u64) -> Self {
        assert!(
            value.len() >= MIN_CHALLENGE_BYTES && value.len() <= MAX_CHALLENGE_BYTES,
            "challenge value must be {MIN_CHALLENGE_BYTES}..{MAX_CHALLENGE_BYTES} bytes"
        );
        Self {
            value,
            window_end,
            accept_until,
        }
    }

    /// Check whether a presented challenge value is current at the given time.
    ///
    /// Returns `true` if the value matches and `now` precedes `accept_until`.
    pub fn is_valid_at(&self, presented: &[u8], now: u64) -> bool {
        presented == self.value && now < self.accept_until
    }

    /// The replay-record expiry for an unattributed proof using this challenge.
    ///
    /// `replay_record_expiry = min(proof.exp, challenge_accept_until)`
    pub fn replay_record_expiry(&self, proof_exp: u64) -> u64 {
        proof_exp.min(self.accept_until)
    }
}

/// A rotating challenge manager that produces and validates challenges.
///
/// In production this rotates on a reviewed schedule (e.g. every 30 seconds
/// with a 5-second overlap). Tests can use fixed values.
pub struct ChallengeManager {
    /// All still-acceptable challenges, including the current one and any
    /// rotated-out values whose `accept_until` has not passed. A proof
    /// citing any of these values validates if `now < accept_until`.
    challenges: parking_lot::RwLock<Vec<ServerChallenge>>,
    /// The overlap duration in seconds (how long after rotation a previous
    /// challenge is still accepted).
    overlap_seconds: u64,
}

impl ChallengeManager {
    /// Create with an initial challenge and a rotation overlap.
    pub fn new(initial: ServerChallenge, overlap_seconds: u64) -> Self {
        Self {
            challenges: parking_lot::RwLock::new(vec![initial]),
            overlap_seconds,
        }
    }

    /// Get the current (newest still-valid) challenge for attaching to
    /// `DispatchDenied`. Returns `None` if rotation has stalled and all
    /// challenges have expired — the caller must refuse service rather than
    /// advertise an unusable challenge.
    pub fn current(&self, now: u64) -> Option<ServerChallenge> {
        let chals = self.challenges.read();
        chals.iter().rev().find(|c| now < c.accept_until).cloned()
    }

    /// Validate a presented challenge against all still-acceptable values.
    /// Returns the matched challenge's `accept_until` if valid, or `None`.
    /// This atomically returns the deadline so the caller can compute
    /// `min(proof.exp, accept_until)` without a TOCTOU race against
    /// rotation.
    pub fn validate(&self, presented: &[u8], now: u64) -> Option<u64> {
        let chals = self.challenges.read();
        chals
            .iter()
            .find(|c| c.is_valid_at(presented, now))
            .map(|c| c.accept_until)
    }

    /// Rotate to a new challenge value. The previous challenge(s) remain
    /// accepted until their own `accept_until` deadline, so a retry that
    /// crosses rotation is not denied early.
    pub fn rotate(&self, value: Vec<u8>, window_seconds: u64, now: u64) {
        let window_end = now + window_seconds;
        let accept_until = window_end + self.overlap_seconds;
        let new = ServerChallenge::new(value, window_end, accept_until);
        let mut chals = self.challenges.write();
        // Prune expired challenges (accept_until has passed).
        chals.retain(|c| c.accept_until > now);
        // Push the new challenge.
        chals.push(new);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn challenge_valid_within_window() {
        let ch = ServerChallenge::new(vec![0xab; 16], 1000, 1010);
        assert!(ch.is_valid_at(&[0xab; 16], 1005));
    }

    #[test]
    fn challenge_invalid_after_deadline() {
        let ch = ServerChallenge::new(vec![0xab; 16], 1000, 1010);
        assert!(!ch.is_valid_at(&[0xab; 16], 1010));
        assert!(!ch.is_valid_at(&[0xab; 16], 1020));
    }

    #[test]
    fn challenge_invalid_wrong_value() {
        let ch = ServerChallenge::new(vec![0xab; 16], 1000, 1010);
        assert!(!ch.is_valid_at(&[0xcd; 16], 1005));
    }

    #[test]
    fn replay_record_expiry_uses_min() {
        let ch = ServerChallenge::new(vec![0xab; 16], 1000, 1010);
        // proof.exp = 1050, challenge_accept_until = 1010 → min = 1010
        assert_eq!(ch.replay_record_expiry(1050), 1010);
        // proof.exp = 1005, challenge_accept_until = 1010 → min = 1005
        assert_eq!(ch.replay_record_expiry(1005), 1005);
    }

    /// A proof citing a rotated-out but still-acceptable challenge must be
    /// retained against **its own** `accept_until`, not the current
    /// challenge's. Validating and then separately reading `current()` would
    /// return the wrong (later) deadline and let a rotated-out proof outlive
    /// its acceptance window.
    #[test]
    fn validate_returns_the_matched_challenge_deadline_not_the_current_one() {
        let mgr = ChallengeManager::new(ServerChallenge::new(vec![0x01; 16], 1000, 1005), 5);
        mgr.rotate(vec![0x02; 16], 30, 1000); // new: window_end 1030, accept_until 1035

        let old = mgr
            .validate(&[0x01; 16], 1002)
            .expect("rotated-out challenge is still acceptable inside its overlap");
        let new = mgr
            .validate(&[0x02; 16], 1002)
            .expect("current challenge is acceptable");

        assert_eq!(old, 1005, "matched deadline must be the old challenge's");
        assert_eq!(new, 1035);
        assert_ne!(
            old,
            mgr.current(1002).expect("a current challenge exists").accept_until,
            "the matched deadline must not be the current challenge's deadline"
        );
        // The retention window follows the matched deadline.
        assert_eq!(1_000_000u64.min(old), 1005);
    }

    /// Validation is a single atomic read: a value whose own deadline has
    /// passed denies even while a newer challenge is live.
    #[test]
    fn validate_denies_expired_value_while_a_newer_one_is_live() {
        let mgr = ChallengeManager::new(ServerChallenge::new(vec![0x01; 16], 1000, 1005), 5);
        mgr.rotate(vec![0x02; 16], 30, 1000);
        assert_eq!(mgr.validate(&[0x01; 16], 1006), None);
        assert_eq!(mgr.validate(&[0x02; 16], 1006), Some(1035));
    }

    /// An unknown value denies regardless of timing.
    #[test]
    fn validate_denies_unknown_value() {
        let mgr = ChallengeManager::new(ServerChallenge::new(vec![0x01; 16], 1000, 1005), 5);
        assert_eq!(mgr.validate(&[0xff; 16], 1002), None);
    }

    /// Rotation prunes values whose acceptance deadline has passed, so a
    /// stockpiled proof citing a long-dead challenge can never re-validate.
    #[test]
    fn rotation_prunes_expired_challenges() {
        let mgr = ChallengeManager::new(ServerChallenge::new(vec![0x01; 16], 1000, 1005), 5);
        mgr.rotate(vec![0x02; 16], 30, 1010);
        assert_eq!(mgr.validate(&[0x01; 16], 1002), None);
        assert_eq!(mgr.challenges.read().len(), 1);
    }

    /// When every challenge has expired the manager advertises nothing: the
    /// caller must refuse service rather than publish an unusable challenge.
    #[test]
    fn current_is_none_when_rotation_has_stalled() {
        let mgr = ChallengeManager::new(ServerChallenge::new(vec![0x01; 16], 1000, 1005), 5);
        assert!(mgr.current(1006).is_none());
    }
}
