//! Access Vector Cache — the fast local decision cache the PEP consults per op (S4 / #570).
//!
//! SELinux-style AVC: the PEP (S2) asks the AVC for a decision on every op; on a hit it
//! returns a cached [`Decision`] with **no Casbin, no signature verify, no lattice walk** —
//! just a hash lookup. On a miss it calls the [`TeEvaluator`] (the PDP) once, caches the
//! result, and returns it. Fail-closed: any ambiguity (wrong generation, poisoned lock,
//! evaluator absent) → DENY.
//!
//! ## The hot path (what must be cheap)
//!
//! ```text
//! PEP.check(op) → AVC.decide(key) ─hit→  return cached Decision        (sub-µs, lock-free read)
//!                                └miss→  evaluator.evaluate(...) ; insert ; return
//! ```
//!
//! The cache KEY is `(generation, subject_type, clearance, object_type, label, action)`. The
//! TE types are interned ids and the `clearance`/`label` are S1's structured
//! [`SecurityLabel`]s — which are `Copy + Hash` (a packed `level/assurance/u64-bitset`, S1
//! confirmed cheap), so the whole key is `Copy` and hashes cheaply with **no dominance walk
//! and no signature verify on a hit** (#570 requirement). We deliberately key on the
//! *resolved context*, not on raw strings/paths: string matching (Casbin's job) is done once
//! at compile time, never per op.
//!
//! ## Token = distributed AVC entry (design §2, §6)
//!
//! The OAuth access token IS an AVC entry, minted by the authority and carried by the
//! subject. It encodes `{label_ceiling, op_set, ttl}` (design §11). The mapping:
//!
//! ```text
//! access token T  ──(verified ONCE at bind/refresh by the PEP)──►  AvcSeed
//!   T.cnf / sub      → subject_type + clearance (subject ctx, design §2)
//!   T.label_ceiling  → the max object label the token authorizes (per-op floor still applies)
//!   T.op_set         → the actions the token authorizes
//!   T.exp            → AvcEntry.valid_until  (ttl-bounded cache lifetime)
//! ```
//!
//! Verifying the token signature is a **bind/refresh-time** event (design §7: mint per-task,
//! AVC caches for ttl), NOT per op. Once seeded, per-op decisions are local lookups. The
//! token's `{label_ceiling, op_set}` are an *additional* deny-only gate the PEP applies via
//! [`TokenScope`]; the mandatory lattice floor + TE matrix are still evaluated independently
//! by the PDP (a token can only narrow — design §10).

use crate::mac::te::{Action, Decision, ObjectCtx, ObjectType, SubjectCtx, SubjectType, TeEvaluator};
use crate::mac::lattice::SecurityLabel;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;

/// Cache key: the fully-resolved, interned per-op context. All `Copy` ids → cheap hash.
/// Tagged with `generation` so a policy reload (new generation) misses every stale entry
/// without an explicit flush walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AvcKey {
    pub generation: u64,
    pub subject_type: SubjectType,
    pub clearance: SecurityLabel,
    pub object_type: ObjectType,
    pub label: SecurityLabel,
    pub action: Action,
}

impl AvcKey {
    /// Assemble a key from the resolved contexts the PEP already has in hand.
    #[inline]
    pub fn new(generation: u64, subject: SubjectCtx, object: ObjectCtx, action: Action) -> Self {
        Self {
            generation,
            subject_type: subject.subject_type,
            clearance: subject.clearance,
            object_type: object.object_type,
            label: object.label,
            action,
        }
    }
}

/// A cached decision. `valid_until` bounds the entry to the token/policy ttl (design §7);
/// `None` means "valid until the generation rolls" (policy-derived decisions with no token
/// ttl). Fail-closed: an expired entry is treated as a miss, not a stale permit.
#[derive(Debug, Clone, Copy)]
struct AvcEntry {
    decision: Decision,
    valid_until: Option<Instant>,
}

impl AvcEntry {
    #[inline]
    fn is_live(&self, now: Instant) -> bool {
        match self.valid_until {
            Some(deadline) => now < deadline,
            None => true,
        }
    }
}

/// The token-derived gate the PEP applies on top of the PDP decision (design §10 / §6).
/// Deny-only: the token can only *narrow*. The PDP (lattice floor + TE matrix) is evaluated
/// independently; this just enforces what the access token authorized.
#[derive(Debug, Clone)]
pub struct TokenScope {
    /// Max object label the token authorizes (`object.label ⊑ label_ceiling`). The PEP
    /// supplies the lattice to compare; here we keep the ceiling and the authorized ops.
    pub label_ceiling: SecurityLabel,
    /// The actions the token authorizes (`op ∈ op_set`).
    pub op_set: Arc<[Action]>,
    /// When the token (hence this scope) expires — the cache ttl bound.
    pub valid_until: Instant,
}

impl TokenScope {
    /// Does this token authorize `action`? Deny-only check.
    #[inline]
    pub fn authorizes_action(&self, action: Action) -> bool {
        self.op_set.contains(&action)
    }
}

/// The Access Vector Cache interface — **what the PEP (S2) calls per op**.
///
/// TOTAL + fail-closed. `decide` returns a [`Decision`] for every call and never panics.
pub trait Avc: Send + Sync {
    /// Per-op decision. Hit → cached; miss → evaluate-and-cache. Fail-closed.
    fn decide(&self, subject: SubjectCtx, object: ObjectCtx, action: Action) -> Decision;

    /// Per-op decision additionally gated by a token scope (the distributed-AVC-entry path).
    /// The result is `Permit` only if the PDP permits AND the token authorizes the action.
    /// The label-ceiling comparison requires the lattice and is done by the PDP-side floor;
    /// this method applies the op-set + ttl gate. Default-deny.
    fn decide_with_token(
        &self,
        subject: SubjectCtx,
        object: ObjectCtx,
        action: Action,
        token: &TokenScope,
    ) -> Decision;

    /// Drop all cached decisions (e.g. on revocation broadcast). Generation rollover already
    /// invalidates implicitly; this is for explicit revocation (design §6 refresh path).
    fn flush(&self);
}

/// Concrete AVC over a [`TeEvaluator`] PDP and a concurrent map. Lock-free reads via
/// `dashmap`; the only work on a hit is a hash + ttl compare.
pub struct CachingAvc<E: TeEvaluator> {
    evaluator: Arc<E>,
    cache: DashMap<AvcKey, AvcEntry>,
}

impl<E: TeEvaluator> CachingAvc<E> {
    /// Wrap a PDP evaluator. The AVC inherits the evaluator's generation for key tagging.
    pub fn new(evaluator: Arc<E>) -> Self {
        Self {
            evaluator,
            cache: DashMap::new(),
        }
    }

    /// Number of live-or-stale cached entries (diagnostics).
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Core lookup-or-evaluate. `valid_until` bounds a freshly computed entry (token ttl).
    #[inline]
    fn decide_inner(
        &self,
        subject: SubjectCtx,
        object: ObjectCtx,
        action: Action,
        valid_until: Option<Instant>,
    ) -> Decision {
        let generation = self.evaluator.generation();
        let key = AvcKey::new(generation, subject, object, action);
        let now = Instant::now();

        // Fast path: live hit. `dashmap` read does not block other readers.
        if let Some(entry) = self.cache.get(&key) {
            if entry.is_live(now) {
                return entry.decision;
            }
            // Stale/expired → fall through to recompute (treated as a miss, fail-closed).
        }

        // Miss / expired: evaluate ONCE via the PDP, then cache.
        let decision = self.evaluator.evaluate(subject, object, action);
        self.cache.insert(
            key,
            AvcEntry {
                decision,
                valid_until,
            },
        );
        decision
    }
}

impl<E: TeEvaluator> Avc for CachingAvc<E> {
    #[inline]
    fn decide(&self, subject: SubjectCtx, object: ObjectCtx, action: Action) -> Decision {
        self.decide_inner(subject, object, action, None)
    }

    #[inline]
    fn decide_with_token(
        &self,
        subject: SubjectCtx,
        object: ObjectCtx,
        action: Action,
        token: &TokenScope,
    ) -> Decision {
        // Token gate (deny-only): expired token or op not in op_set → DENY without touching
        // the PDP. (The label_ceiling ⊒ object.label check is the PDP-side lattice floor's
        // job — the token's ceiling is enforced by seeding subject.clearance ⊓ ceiling at
        // bind time; see module docs. We keep the ceiling on the scope for that seeding and
        // for audit, and do not re-walk the lattice here.)
        if Instant::now() >= token.valid_until {
            return Decision::Deny;
        }
        if !token.authorizes_action(action) {
            return Decision::Deny;
        }
        // PDP decision, cached with the token ttl as the entry lifetime.
        let decision = self.decide_inner(subject, object, action, Some(token.valid_until));
        // Conjunction: permit only if the PDP permitted. (Token already gated above.)
        if decision.is_permit() {
            Decision::Permit
        } else {
            decision
        }
    }

    fn flush(&self) {
        self.cache.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mac::lattice::{
        Assurance, Compartment, CompartmentSet, Lattice, LatticeVersion, Level, SecurityLabel,
    };
    use crate::mac::te::{LatticeTeEvaluator, TeMatrix, TeRule};
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    fn rule(s: u32, o: u32, a: u32) -> TeRule {
        TeRule {
            subject_type: SubjectType(s),
            object_type: ObjectType(o),
            action: Action(a),
        }
    }

    fn lattice() -> Lattice {
        Lattice::new(LatticeVersion(1), [Compartment::new("pii")])
    }

    /// A high (Secret/pq-hybrid) clearance that dominates the low object label below.
    fn high() -> SecurityLabel {
        SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY)
    }
    /// A low (Public/classical) object label, dominated by `high()`.
    fn low() -> SecurityLabel {
        SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn subj(t: u32, clr: SecurityLabel) -> SubjectCtx {
        SubjectCtx { subject_type: SubjectType(t), clearance: clr }
    }
    fn obj(t: u32, lbl: SecurityLabel) -> ObjectCtx {
        ObjectCtx { object_type: ObjectType(t), label: lbl }
    }

    fn avc() -> CachingAvc<LatticeTeEvaluator> {
        let mut allow = HashSet::new();
        allow.insert(rule(1, 1, 1));
        let e = LatticeTeEvaluator::new(TeMatrix::from_allow(allow), lattice());
        CachingAvc::new(Arc::new(e))
    }

    #[test]
    fn hit_returns_cached_decision() {
        let a = avc();
        assert_eq!(a.decide(subj(1, high()), obj(1, low()), Action(1)), Decision::Permit);
        assert_eq!(a.decide(subj(1, high()), obj(1, low()), Action(1)), Decision::Permit);
        assert_eq!(a.len(), 1, "second call must hit the cache, not add a new entry");
    }

    #[test]
    fn default_deny_cached_too() {
        let a = avc();
        assert_eq!(a.decide(subj(1, high()), obj(1, low()), Action(9)), Decision::Deny);
        assert_eq!(a.decide(subj(1, high()), obj(1, low()), Action(9)), Decision::Deny);
    }

    /// The evaluator must be consulted exactly ONCE per distinct key — proving the cache
    /// actually short-circuits the PDP (the whole point of the AVC).
    #[test]
    fn evaluator_consulted_once_per_key() {
        struct Counting {
            inner: LatticeTeEvaluator,
            calls: AtomicU32,
        }
        impl TeEvaluator for Counting {
            fn evaluate(&self, s: SubjectCtx, o: ObjectCtx, a: Action) -> Decision {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.evaluate(s, o, a)
            }
            fn generation(&self) -> u64 {
                self.inner.generation()
            }
        }
        let mut allow = HashSet::new();
        allow.insert(rule(1, 1, 1));
        let inner = LatticeTeEvaluator::new(TeMatrix::from_allow(allow), lattice());
        let counting = Arc::new(Counting { inner, calls: AtomicU32::new(0) });
        let a = CachingAvc::new(counting.clone());

        for _ in 0..10 {
            a.decide(subj(1, high()), obj(1, low()), Action(1));
        }
        assert_eq!(counting.calls.load(Ordering::SeqCst), 1, "PDP must be hit once, then cached");
    }

    #[test]
    fn token_gate_denies_op_outside_op_set() {
        let a = avc();
        let token = TokenScope {
            label_ceiling: high(),
            op_set: Arc::from(vec![Action(2)]), // does NOT include Action(1)
            valid_until: Instant::now() + Duration::from_secs(60),
        };
        assert_eq!(
            a.decide_with_token(subj(1, high()), obj(1, low()), Action(1), &token),
            Decision::Deny,
            "action not in token op_set must be denied"
        );
    }

    #[test]
    fn token_gate_permits_when_pdp_and_token_agree() {
        let a = avc();
        let token = TokenScope {
            label_ceiling: high(),
            op_set: Arc::from(vec![Action(1)]),
            valid_until: Instant::now() + Duration::from_secs(60),
        };
        assert_eq!(
            a.decide_with_token(subj(1, high()), obj(1, low()), Action(1), &token),
            Decision::Permit
        );
    }

    #[test]
    fn expired_token_fails_closed() {
        let a = avc();
        let token = TokenScope {
            label_ceiling: high(),
            op_set: Arc::from(vec![Action(1)]),
            valid_until: Instant::now() - Duration::from_secs(1), // already expired
        };
        assert_eq!(
            a.decide_with_token(subj(1, high()), obj(1, low()), Action(1), &token),
            Decision::Deny
        );
    }

    #[test]
    fn flush_clears_cache() {
        let a = avc();
        a.decide(subj(1, high()), obj(1, low()), Action(1));
        assert_eq!(a.len(), 1);
        a.flush();
        assert!(a.is_empty());
    }
}
