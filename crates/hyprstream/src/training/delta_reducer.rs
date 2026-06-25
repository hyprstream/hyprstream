//! N-way delta aggregation (param-server reducer) — replaces DO-Merge for
//! multi-contributor merges.
//!
//! # Why this exists
//!
//! [`crate::training::merge`]'s DO-Merge (`do_merge`) and `additive_merge`
//! assert *shape equality* on the raw LoRA A/B factors. The moment two
//! contributors carry different per-layer ranks — the normal case once the rank
//! oracle (`ttn_profile`) assigns non-uniform per-layer ranks, or once
//! different workers run with different `layer_overrides` — those paths
//! hard-fail. They are also non-associative (path-dependent blend weights +
//! direction renormalization), so an N-way merge would depend on contributor
//! order.
//!
//! This module replaces the **N-way** aggregation with an order-independent,
//! rank-heterogeneity-tolerant reducer. The 1-vs-1 save-time DO-Merge path may
//! stay; this is the path used when *several* independent TTT contributors are
//! aggregated each round.
//!
//! # Algorithm (stack + truncated SVD in delta-space)
//!
//! For one layer key, each contributor `i` supplies factors `A_i [r_i, in]`,
//! `B_i [out, r_i]` and a fold-in weight `w_i` (provenance trust × norm-bound ×
//! per-key scaling). The aggregate effective weight is
//!
//! ```text
//!   ΔW = Σ_i w_i · (B_i · A_i)
//!      = B_stack · A_stack
//! ```
//!
//! where `B_stack = [ w_1·B_1 | w_2·B_2 | … ]` is `[out, Σr_i]` and
//! `A_stack = [ A_1 ; A_2 ; … ]` is `[Σr_i, in]`. This is exact: stacking the
//! factors *is* the weighted sum of the rank-1 updates, with no dense
//! `[out, in]` materialization and no requirement that the `r_i` agree.
//!
//! We then take **one** rank-`R` truncated SVD of `ΔW` per round, computed on
//! the tall-skinny stacked factors (`Σr` is the sum of contributor ranks —
//! small) without ever forming the dense `[out, in]`:
//!
//! 1. economy SVD of `A_stack [Σr, in]` → `A_stack = Uₐ·diag(Sₐ)·Vhₐ`
//!    (factors are `[Σr, Σr]`, `[Σr]`, `[Σr, in]` since `Σr ≪ in`);
//! 2. `C = B_stack · Uₐ · diag(Sₐ)`  →  `[out, Σr]` (small, no dense `out×in`);
//! 3. economy SVD of `C [out, Σr]` → `C = U_c·diag(S_c)·Vh_c`;
//! 4. `ΔW = U_c · diag(S_c) · (Vh_c · Vhₐ)` — an exact SVD of `ΔW`;
//! 5. truncate to `R`: `B_out = U_c[:, :R]·diag(S_c[:R])` `[out, R]`,
//!    `A_out = (Vh_c·Vhₐ)[:R, :]` `[R, in]`.
//!
//! The reduced factors are stored with `scaling = 1.0` (the weights and
//! per-contributor scaling are already folded in). Output rank is the fixed,
//! operational `R` — stable checkpoint shapes.
//!
//! # Order-independence
//!
//! `ΔW` is a plain sum, so permuting contributors permutes columns of
//! `B_stack` and rows of `A_stack` identically and leaves the product `ΔW`
//! unchanged; the SVD of `ΔW` is therefore identical up to fp rounding.
//!
//! # Provenance / norm gate (secure multitenant)
//!
//! Aggregation is over an *explicit* contributor set — the reducer never
//! discovers or mixes deltas across tenants. Each contributor carries a
//! verified host id (Phase-2 mesh trust, #354) and a `delta_norm_ratio`. The
//! [`ContributionGate`] rejects (does **not** silently drop or clamp-to-zero)
//! any contributor whose norm exceeds the bound, and weights survivors by
//! `trust × f(norm_ratio)`. A pluggable [`ValidationHook`] seam is provided for
//! the deferred held-out-eval host (default impl is a no-op).
//!
//! # Send/Sync
//!
//! tch `Tensor` is `!Send`. The reducer keeps **all** tensor work on a single
//! device/thread (`reduce()` takes `&[Contribution]` by reference and returns
//! freshly-built tensors; it never moves a `Tensor` across a thread boundary).
//! Contributions cross thread/host boundaries as serialized delta *bytes*
//! (safetensors) + plain-old-data provenance, deserialized on the reducer
//! thread — never as live `Tensor`s.
//!
//! # Optimizer state
//!
//! Per the spike, Muon momentum is **dropped** at aggregation: averaging
//! optimizer state across independent TTT runs is unsound. The reduced delta
//! starts with a fresh (empty) optimizer.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use tch::{Kind, Tensor};

/// Provenance for one contributor's delta.
///
/// The transport that *verifies* the signature (param-server / mesh) is out of
/// scope for the reducer (a separate gated follow-on). By the time a
/// [`Contribution`] reaches the reducer its `host_id` is already an
/// operator-resolved, verified identity (Phase-2 mesh trust, #354). This struct
/// is plain-old-data so it can cross a thread/host boundary without moving any
/// `Tensor`.
#[derive(Debug, Clone)]
pub struct Provenance {
    /// Verified host id (operator label resolved from the signing identity).
    pub host_id: String,
    /// Trust weight for this host in `[0, 1]`. `1.0` = full trust. A downweighted
    /// (e.g. newly-seen) host moves the merge less.
    pub trust: f64,
}

impl Provenance {
    /// Construct a fully-trusted contributor identity.
    pub fn trusted(host_id: impl Into<String>) -> Self {
        Self { host_id: host_id.into(), trust: 1.0 }
    }
}

/// One contributor in an N-way aggregation round.
///
/// Holds the contributor's LoRA factors *for a single layer key already* — the
/// reducer is invoked per key by [`reduce_state_dicts`]. Factors are borrowed
/// (shallow clones / views are fine); the reducer does not mutate them.
pub struct Contribution {
    /// `A_i [r_i, in]`.
    pub lora_a: Tensor,
    /// `B_i [out, r_i]`.
    pub lora_b: Tensor,
    /// Per-key LoRA scaling (`alpha / rank`) already associated with these
    /// factors. Folded into the stack weight so the reduced factors can use
    /// `scaling = 1.0`.
    pub scaling: f64,
    /// `||ΔW_i|| / ||W_base||` for this key (from
    /// [`crate::training::TenantDelta::delta_norm_ratio`]). Used by the gate.
    pub delta_norm_ratio: f64,
    /// Verified provenance.
    pub provenance: Provenance,
}

/// Gate + weighting policy applied to each contributor before it enters the
/// stack. Anomalous / over-norm contributors are **rejected** (error), never
/// silently dropped or clamped to zero — a quiet drop would let a poisoned host
/// shrink the merge invisibly.
#[derive(Debug, Clone)]
pub struct ContributionGate {
    /// Reject any contributor whose `delta_norm_ratio` exceeds this bound.
    /// Defends against norm-inflation poisoning.
    pub max_norm_ratio: f64,
    /// Reject any contributor whose resolved trust is below this floor.
    pub min_trust: f64,
}

impl Default for ContributionGate {
    fn default() -> Self {
        // 2.0 = the aggregate update may be at most ~2x the base-weight norm for
        // a single contributor; well above legitimate TTT drift, well below a
        // runaway. min_trust=0 admits any positively-trusted host; raise to
        // exclude provisional peers.
        Self { max_norm_ratio: 2.0, min_trust: 0.0 }
    }
}

impl ContributionGate {
    /// Validate one contributor and return its non-negative stack weight
    /// `trust × f(norm_ratio) × scaling`, or an error if it fails the gate.
    ///
    /// `f(norm_ratio) = 1 / (1 + norm_ratio)` softly downweights large-norm
    /// (more-likely-anomalous) deltas that still pass the hard bound, so a
    /// borderline contributor moves the merge less than a small clean one.
    fn weight_for(&self, c: &Contribution) -> Result<f64> {
        let trust = c.provenance.trust;
        if !(trust.is_finite()) || trust < 0.0 {
            return Err(anyhow!(
                "contributor '{}' has invalid trust {}",
                c.provenance.host_id,
                trust
            ));
        }
        if trust < self.min_trust {
            return Err(anyhow!(
                "contributor '{}' rejected: trust {} < min_trust {}",
                c.provenance.host_id,
                trust,
                self.min_trust
            ));
        }
        let ratio = c.delta_norm_ratio;
        if !ratio.is_finite() || ratio < 0.0 {
            return Err(anyhow!(
                "contributor '{}' has invalid delta_norm_ratio {}",
                c.provenance.host_id,
                ratio
            ));
        }
        if ratio > self.max_norm_ratio {
            return Err(anyhow!(
                "contributor '{}' rejected: delta_norm_ratio {:.4} exceeds bound {:.4} (possible norm-inflation poisoning)",
                c.provenance.host_id,
                ratio,
                self.max_norm_ratio
            ));
        }
        let norm_weight = 1.0 / (1.0 + ratio);
        Ok(trust * norm_weight * c.scaling)
    }
}

/// Pluggable post-reduce validation seam.
///
/// The real implementation (held-out-eval host) is a deferred, gated follow-on.
/// Defining the trait now lets the eval-host swap in with zero consumer churn.
pub trait ValidationHook: Send + Sync {
    /// Inspect the freshly-reduced factors for one key. Return `Ok(())` to
    /// accept or an error to reject the round. Default: accept everything.
    ///
    /// Tensors are passed by reference and must not be moved across threads.
    fn validate(&self, _key: &str, _reduced_a: &Tensor, _reduced_b: &Tensor) -> Result<()> {
        Ok(())
    }
}

/// No-op validation hook (the default until the held-out-eval host lands).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopValidation;

impl ValidationHook for NoopValidation {}

/// Order-independent N-way delta reducer.
///
/// Implementors merge an explicit contributor set into a fixed-rank delta. The
/// stack + truncated-SVD implementation is [`StackSvdReducer`]; the trait keeps
/// it swappable (held-out-eval variants, future native reducers).
pub trait DeltaReducer {
    /// Reduce contributors for a **single** layer key into fixed-rank factors
    /// `(A_out [R, in], B_out [out, R])`, with the result expressed at
    /// `scaling = 1.0` (weights + per-contributor scaling already folded in).
    ///
    /// All tensors stay on `contributions[*].lora_a.device()` (must be uniform).
    fn reduce_key(&self, key: &str, contributions: &[Contribution]) -> Result<(Tensor, Tensor)>;
}

/// Stack-in-delta-space + one truncated SVD per round to a fixed rank `R`.
pub struct StackSvdReducer {
    /// Fixed output rank — an operational contract (stable checkpoint shapes).
    pub rank: usize,
    /// Contributor gate / weighting.
    pub gate: ContributionGate,
    /// Pluggable validation hook (default no-op).
    pub validation: Box<dyn ValidationHook>,
}

impl StackSvdReducer {
    /// Construct with a fixed output rank and the default gate / no-op hook.
    pub fn new(rank: usize) -> Self {
        Self { rank, gate: ContributionGate::default(), validation: Box::new(NoopValidation) }
    }

    /// Override the contributor gate.
    pub fn with_gate(mut self, gate: ContributionGate) -> Self {
        self.gate = gate;
        self
    }

    /// Install a validation hook (e.g. the held-out-eval host once it exists).
    pub fn with_validation(mut self, validation: Box<dyn ValidationHook>) -> Self {
        self.validation = validation;
        self
    }
}

impl DeltaReducer for StackSvdReducer {
    fn reduce_key(&self, key: &str, contributions: &[Contribution]) -> Result<(Tensor, Tensor)> {
        let _guard = tch::no_grad_guard();

        if contributions.is_empty() {
            return Err(anyhow!("reduce_key('{}'): no contributors", key));
        }
        if self.rank == 0 {
            return Err(anyhow!("reduce_key('{}'): fixed rank R must be >= 1", key));
        }

        // Uniform device + matching in/out dims across contributors. The reducer
        // does all tensor work on this one device/thread (Send/Sync invariant).
        let device = contributions[0].lora_a.device();
        let out_dim = contributions[0].lora_b.size()[0];
        let in_dim = contributions[0].lora_a.size()[1];

        // Build the stacked factors, folding the gate weight into B (one side
        // only — weighting both A and B would square the weight).
        //   B_stack = [ w_1·B_1 | … ]  [out, Σr]
        //   A_stack = [   A_1   ; … ]  [Σr, in]
        let mut b_cols: Vec<Tensor> = Vec::with_capacity(contributions.len());
        let mut a_rows: Vec<Tensor> = Vec::with_capacity(contributions.len());

        for c in contributions {
            // Validate device + shape consistency.
            if c.lora_a.device() != device || c.lora_b.device() != device {
                return Err(anyhow!(
                    "reduce_key('{}'): contributor '{}' on a different device — reducer requires a single device",
                    key,
                    c.provenance.host_id
                ));
            }
            let a_size = c.lora_a.size();
            let b_size = c.lora_b.size();
            if a_size.len() != 2 || b_size.len() != 2 {
                return Err(anyhow!("reduce_key('{}'): factors must be 2D", key));
            }
            let (ri, in_i) = (a_size[0], a_size[1]);
            let (out_i, ri_b) = (b_size[0], b_size[1]);
            if in_i != in_dim || out_i != out_dim {
                return Err(anyhow!(
                    "reduce_key('{}'): contributor '{}' dims [out={},in={}] differ from [out={},in={}]",
                    key, c.provenance.host_id, out_i, in_i, out_dim, in_dim
                ));
            }
            if ri != ri_b {
                return Err(anyhow!(
                    "reduce_key('{}'): contributor '{}' rank mismatch A[0]={} vs B[1]={}",
                    key, c.provenance.host_id, ri, ri_b
                ));
            }

            // Gate + weight (rejects over-norm / untrusted; never silent-drops).
            let weight = self.gate.weight_for(c)?;

            // Compute the reducer in fp32 (deltas are fp32 delta-space).
            let a = c.lora_a.to_kind(Kind::Float);
            let b = c.lora_b.to_kind(Kind::Float) * weight;
            a_rows.push(a);
            b_cols.push(b);
        }

        // Σr small (sum of contributor ranks). Concat is the only unavoidable
        // materialization; both stacks are tall-skinny.
        let a_refs: Vec<&Tensor> = a_rows.iter().collect();
        let b_refs: Vec<&Tensor> = b_cols.iter().collect();
        let a_stack = Tensor::cat(&a_refs, 0); // [Σr, in]
        let b_stack = Tensor::cat(&b_refs, 1); // [out, Σr]
        let sigma_r = a_stack.size()[0];

        // If the whole stack is ~zero (all contributors empty/cancelled), return
        // zero factors at the fixed rank rather than running SVD on zeros.
        let stack_norm: f64 = a_stack.norm().double_value(&[]) * b_stack.norm().double_value(&[]);
        let r = self.rank as i64;
        if stack_norm < 1e-12 {
            let a_out = Tensor::zeros([r, in_dim], (Kind::Float, device));
            let b_out = Tensor::zeros([out_dim, r], (Kind::Float, device));
            self.validation.validate(key, &a_out, &b_out)?;
            return Ok((a_out, b_out));
        }

        // Exact rank-Σr SVD of ΔW = B_stack · A_stack, computed via two economy
        // SVDs on the tall-skinny stacks (never forms the dense [out,in]).
        //
        // 1) A_stack = Ua · diag(Sa) · Vha   (economy: some=true, compute_uv=true)
        let (ua, sa, va) = a_stack.svd(true, true); // Ua [Σr,k], Sa [k], Va [in,k]
        // tch's `svd` returns V (not Vh); Vha = Va^T.
        let k = sa.size()[0]; // = min(Σr, in) = Σr
        let _ = k;
        // 2) C = B_stack · Ua · diag(Sa)  →  [out, Σr]
        let c = b_stack.matmul(&ua) * sa.unsqueeze(0); // broadcast diag(Sa) over rows
        // 3) C = Uc · diag(Sc) · Vhc
        let (uc, sc, vc) = c.svd(true, true); // Uc [out,m], Sc [m], Vc [Σr,m]
        // 4) Vh_final = Vhc · Vha = Vc^T · Va^T = (Va · Vc)^T  →  [m, in]
        //    ΔW = Uc · diag(Sc) · Vh_final
        let vh_final = va.matmul(&vc).tr(); // [m, in]

        // 5) Truncate to fixed R (pad with zeros if Σr < R so shapes are stable).
        let avail = sc.size()[0].min(sigma_r);
        let take = avail.min(r);

        let uc_t = uc.narrow(1, 0, take); // [out, take]
        let sc_t = sc.narrow(0, 0, take); // [take]
        let vh_t = vh_final.narrow(0, 0, take); // [take, in]

        // B_out = Uc[:, :R] · diag(Sc[:R])   [out, take]
        let b_take = &uc_t * sc_t.unsqueeze(0);
        // A_out = Vh_final[:R, :]            [take, in]
        let a_take = vh_t.contiguous();
        let b_take = b_take.contiguous();

        let (a_out, b_out) = if take < r {
            // Pad the rank dimension with zeros to the fixed R (zero-energy
            // directions; keeps checkpoint shapes invariant).
            let a_pad = Tensor::zeros([r - take, in_dim], (Kind::Float, device));
            let b_pad = Tensor::zeros([out_dim, r - take], (Kind::Float, device));
            let a_out = Tensor::cat(&[&a_take, &a_pad], 0);
            let b_out = Tensor::cat(&[&b_take, &b_pad], 1);
            (a_out, b_out)
        } else {
            (a_take, b_take)
        };

        self.validation.validate(key, &a_out, &b_out)?;
        Ok((a_out, b_out))
    }
}

/// Aggregate N contributors' full state dicts into one reduced state dict.
///
/// Each contributor supplies a `(state_dict, provenance, norm_ratios)` where the
/// state dict uses the `extract_state_dict` key convention
/// (`"layer.module.lora_a"` / `"layer.module.lora_b"`). The reducer runs the
/// stack + truncated-SVD per layer key and emits reduced factors at the fixed
/// rank `R`. Effective-rank / Muon-momentum metadata are intentionally dropped
/// (fresh optimizer post-aggregation, per the spike).
///
/// Keys present in only some contributors are aggregated over the subset that
/// has them — this is still order-independent and tolerant of contributors that
/// trained different module sets.
pub fn reduce_state_dicts(
    reducer: &dyn DeltaReducer,
    contributors: &[ContributorState],
) -> Result<HashMap<String, Tensor>> {
    let _guard = tch::no_grad_guard();
    if contributors.is_empty() {
        return Err(anyhow!("reduce_state_dicts: no contributors"));
    }

    // Collect the union of layer keys across all contributors.
    let mut layer_keys: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for cs in contributors {
        for k in cs.state.keys() {
            if let Some(layer_key) = k.strip_suffix(".lora_a") {
                if seen.insert(layer_key.to_owned()) {
                    layer_keys.push(layer_key.to_owned());
                }
            }
        }
    }
    layer_keys.sort(); // deterministic output ordering

    let mut result: HashMap<String, Tensor> = HashMap::new();

    for layer_key in &layer_keys {
        let a_key = format!("{}.lora_a", layer_key);
        let b_key = format!("{}.lora_b", layer_key);

        let mut contribs: Vec<Contribution> = Vec::new();
        for cs in contributors {
            let (Some(a), Some(b)) = (cs.state.get(&a_key), cs.state.get(&b_key)) else {
                continue; // this contributor didn't train this key
            };
            let norm_ratio = cs
                .norm_ratios
                .get(layer_key)
                .copied()
                // module-name fallback ("layer.module" -> "module") mirrors
                // delta_norm_ratio's own keying tolerance.
                .or_else(|| {
                    layer_key
                        .split_once('.')
                        .and_then(|(_, m)| cs.norm_ratios.get(m).copied())
                })
                .unwrap_or(0.0);
            contribs.push(Contribution {
                lora_a: a.shallow_clone(),
                lora_b: b.shallow_clone(),
                scaling: cs.scaling.get(layer_key).copied().unwrap_or(1.0),
                delta_norm_ratio: norm_ratio,
                provenance: cs.provenance.clone(),
            });
        }

        if contribs.is_empty() {
            continue;
        }

        let (a_out, b_out) = reducer.reduce_key(layer_key, &contribs)?;
        result.insert(a_key, a_out);
        result.insert(b_key, b_out);
    }

    if result.is_empty() {
        return Err(anyhow!("reduce_state_dicts: no aggregatable layer keys across contributors"));
    }

    Ok(result)
}

/// One contributor's full state for [`reduce_state_dicts`].
///
/// Plain-old-data + tensors that stay on one thread. The `state` tensors are
/// expected to already be on the reducer's device (deserialized from bytes on
/// this thread — see the module-level Send/Sync note).
pub struct ContributorState {
    /// `extract_state_dict`-keyed factors (`"layer.module.lora_a"` etc.).
    pub state: HashMap<String, Tensor>,
    /// Per-layer-key scaling (`alpha / rank`).
    pub scaling: HashMap<String, f64>,
    /// Per-layer-key (or per-module) `delta_norm_ratio`.
    pub norm_ratios: HashMap<String, f64>,
    /// Verified provenance for this contributor.
    pub provenance: Provenance,
}

/// Static rank↔quant coupling cap.
///
/// Coarse quantization (e.g. block-wise FP8) gives a delta limited headroom: a
/// rank the layer's stored dtype can't actually express is wasted (and adds
/// noise from untrained directions). This clamps an oracle-recommended rank by
/// the layer's stored element size. **Static only** — applied once at
/// `from_profile` time, never in the live `auto_adapt` path.
///
/// Returns the rank cap for a given stored element size in bytes:
/// - 1 byte (FP8 / int8): cap 8
/// - 2 bytes (FP16 / BF16): cap 16
/// - otherwise (FP32+, or `0` = unknown): no cap (returns `usize::MAX`)
pub fn rank_cap_for_dtype_bytes(elem_bytes: usize) -> usize {
    match elem_bytes {
        1 => 8,          // FP8 / int8: coarse, low usable rank
        2 => 16,         // FP16 / BF16
        _ => usize::MAX, // FP32 and wider (or 0 = unknown): full oracle rank
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use tch::Device;

    /// Build a contributor for a single key with the given rank and dims.
    /// `seed` perturbs the values so contributors differ.
    fn make_contrib(
        rank: i64,
        in_dim: i64,
        out_dim: i64,
        trust: f64,
        norm_ratio: f64,
        host: &str,
        seed: i64,
    ) -> Contribution {
        let a = Tensor::randn([rank, in_dim], (Kind::Float, Device::Cpu)) + seed as f64 * 0.01;
        let b = Tensor::randn([out_dim, rank], (Kind::Float, Device::Cpu)) + seed as f64 * 0.01;
        Contribution {
            lora_a: a,
            lora_b: b,
            scaling: 1.0,
            delta_norm_ratio: norm_ratio,
            provenance: Provenance { host_id: host.to_owned(), trust },
        }
    }

    /// Dense effective weight ΔW = Σ_i scaling_i·w_i·(B_i·A_i) for a contributor
    /// set, replicating the gate weighting — used to check the reduced factors
    /// reconstruct the same ΔW (within the truncation rank).
    fn dense_delta(gate: &ContributionGate, contribs: &[Contribution]) -> Tensor {
        let out_dim = contribs[0].lora_b.size()[0];
        let in_dim = contribs[0].lora_a.size()[1];
        let mut acc = Tensor::zeros([out_dim, in_dim], (Kind::Float, Device::Cpu));
        for c in contribs {
            let w = gate.weight_for(c).unwrap();
            acc += c.lora_b.matmul(&c.lora_a) * w;
        }
        acc
    }

    fn reconstruct(a: &Tensor, b: &Tensor) -> Tensor {
        b.matmul(a)
    }

    #[test]
    fn test_output_rank_is_fixed_r() {
        let reducer = StackSvdReducer::new(8);
        let contribs = vec![
            make_contrib(4, 32, 16, 1.0, 0.1, "h1", 1),
            make_contrib(16, 32, 16, 1.0, 0.1, "h2", 2),
        ];
        let (a, b) = reducer.reduce_key("0.q_proj", &contribs).unwrap();
        assert_eq!(a.size(), vec![8, 32]); // [R, in]
        assert_eq!(b.size(), vec![16, 8]); // [out, R]
    }

    #[test]
    fn test_order_independent() {
        // R large enough to capture the full combined rank (4+5+6=15 -> R=16),
        // so truncation is lossless and permutation must be exact to fp.
        let reducer = StackSvdReducer::new(16);
        let c1 = make_contrib(4, 24, 20, 1.0, 0.1, "h1", 1);
        let c2 = make_contrib(5, 24, 20, 1.0, 0.1, "h2", 2);
        let c3 = make_contrib(6, 24, 20, 1.0, 0.1, "h3", 3);

        // Clone the underlying tensors so both orderings see identical inputs.
        let clone = |c: &Contribution| Contribution {
            lora_a: c.lora_a.copy(),
            lora_b: c.lora_b.copy(),
            scaling: c.scaling,
            delta_norm_ratio: c.delta_norm_ratio,
            provenance: c.provenance.clone(),
        };

        let order_a = vec![clone(&c1), clone(&c2), clone(&c3)];
        let order_b = vec![clone(&c3), clone(&c1), clone(&c2)];

        let (a1, b1) = reducer.reduce_key("k", &order_a).unwrap();
        let (a2, b2) = reducer.reduce_key("k", &order_b).unwrap();

        // Compare the reconstructed ΔW (factorization is only unique up to
        // rotation; the product is the invariant).
        let w1 = reconstruct(&a1, &b1);
        let w2 = reconstruct(&a2, &b2);
        let diff: f64 = (&w1 - &w2).abs().max().double_value(&[]);
        assert!(diff < 1e-3, "permuting contributors must not change ΔW, max diff = {}", diff);
    }

    #[test]
    fn test_tolerates_rank_heterogeneous_contributors() {
        // The exact case DO-Merge hard-failed on: contributors with different
        // ranks. Must succeed and reconstruct the weighted sum.
        let reducer = StackSvdReducer::new(32); // R >= Σr so reconstruction is exact
        let contribs = vec![
            make_contrib(4, 20, 16, 1.0, 0.1, "h1", 1),
            make_contrib(8, 20, 16, 1.0, 0.1, "h2", 2),
            make_contrib(16, 20, 16, 1.0, 0.1, "h3", 3),
        ];
        let expected = dense_delta(&reducer.gate, &contribs);
        let (a, b) = reducer.reduce_key("0.q_proj", &contribs).unwrap();
        let got = reconstruct(&a, &b);
        let diff: f64 = (&got - &expected).abs().max().double_value(&[]);
        assert!(
            diff < 1e-3,
            "rank-heterogeneous reduction must reconstruct the weighted sum, max diff = {}",
            diff
        );
    }

    #[test]
    fn test_norm_bound_gate_rejects_over_norm() {
        let gate = ContributionGate { max_norm_ratio: 1.0, min_trust: 0.0 };
        let reducer = StackSvdReducer::new(8).with_gate(gate);
        let contribs = vec![
            make_contrib(4, 16, 16, 1.0, 0.2, "good", 1),
            make_contrib(4, 16, 16, 1.0, 5.0, "evil", 2), // norm_ratio 5.0 > 1.0
        ];
        let err = reducer.reduce_key("k", &contribs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("evil"), "error should name the rejected host: {}", msg);
        assert!(msg.contains("exceeds bound"), "error should explain the norm bound: {}", msg);
    }

    #[test]
    fn test_min_trust_gate_rejects_untrusted() {
        let gate = ContributionGate { max_norm_ratio: 10.0, min_trust: 0.5 };
        let reducer = StackSvdReducer::new(8).with_gate(gate);
        let contribs = vec![make_contrib(4, 16, 16, 0.2, 0.1, "provisional", 1)];
        let err = reducer.reduce_key("k", &contribs).unwrap_err();
        assert!(err.to_string().contains("min_trust"), "{}", err);
    }

    #[test]
    fn test_provenance_weighting_moves_merge_less() {
        // A down-weighted contributor should move the merge less than a
        // fully-trusted one. Compare the merge of [base, low-trust X] against
        // [base, full-trust X]: the latter must be farther from base-alone.
        let reducer = StackSvdReducer::new(16);
        let base = make_contrib(8, 24, 24, 1.0, 0.1, "base", 10);
        let x_full = make_contrib(8, 24, 24, 1.0, 0.1, "x", 99);
        let x_low = Contribution {
            lora_a: x_full.lora_a.copy(),
            lora_b: x_full.lora_b.copy(),
            scaling: 1.0,
            delta_norm_ratio: 0.1,
            provenance: Provenance { host_id: "x".to_owned(), trust: 0.05 },
        };
        let base_clone = |seed_marker: &Contribution| Contribution {
            lora_a: seed_marker.lora_a.copy(),
            lora_b: seed_marker.lora_b.copy(),
            scaling: 1.0,
            delta_norm_ratio: 0.1,
            provenance: seed_marker.provenance.clone(),
        };

        let (ba, bb) = reducer.reduce_key("k", &[base_clone(&base)]).unwrap();
        let w_base = reconstruct(&ba, &bb);

        let (fa, fb) = reducer
            .reduce_key("k", &[base_clone(&base), x_full])
            .unwrap();
        let w_full = reconstruct(&fa, &fb);

        let (la, lb) = reducer.reduce_key("k", &[base_clone(&base), x_low]).unwrap();
        let w_low = reconstruct(&la, &lb);

        let move_full: f64 = (&w_full - &w_base).abs().sum(Kind::Float).double_value(&[]);
        let move_low: f64 = (&w_low - &w_base).abs().sum(Kind::Float).double_value(&[]);
        assert!(
            move_low < move_full,
            "down-weighted contributor should move the merge less: low={} full={}",
            move_low,
            move_full
        );
    }

    #[test]
    fn test_reduce_state_dicts_end_to_end() {
        // Two contributors, overlapping + disjoint keys, heterogeneous ranks.
        let mk_state = |rank: i64, keys: &[&str]| -> HashMap<String, Tensor> {
            let mut s = HashMap::new();
            for k in keys {
                s.insert(format!("{}.lora_a", k), Tensor::randn([rank, 32], (Kind::Float, Device::Cpu)));
                s.insert(format!("{}.lora_b", k), Tensor::randn([16, rank], (Kind::Float, Device::Cpu)));
            }
            s
        };
        let c1 = ContributorState {
            state: mk_state(4, &["0.q_proj", "0.v_proj"]),
            scaling: HashMap::new(),
            norm_ratios: HashMap::new(),
            provenance: Provenance::trusted("h1"),
        };
        let c2 = ContributorState {
            state: mk_state(8, &["0.q_proj", "1.q_proj"]),
            scaling: HashMap::new(),
            norm_ratios: HashMap::new(),
            provenance: Provenance::trusted("h2"),
        };
        let reducer = StackSvdReducer::new(8);
        let out = reduce_state_dicts(&reducer, &[c1, c2]).unwrap();

        // Union of keys: 0.q_proj, 0.v_proj, 1.q_proj — all at fixed R=8.
        for key in ["0.q_proj", "0.v_proj", "1.q_proj"] {
            let a = &out[&format!("{}.lora_a", key)];
            let b = &out[&format!("{}.lora_b", key)];
            assert_eq!(a.size(), vec![8, 32], "{key} A");
            assert_eq!(b.size(), vec![16, 8], "{key} B");
        }
        // No optimizer/effective-rank metadata leaks into the reduced dict.
        assert!(out.keys().all(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")));
    }

    #[test]
    fn test_zero_contributors_errors() {
        let reducer = StackSvdReducer::new(8);
        assert!(reducer.reduce_key("k", &[]).is_err());
    }

    #[test]
    fn test_zero_stack_returns_zero_factors() {
        let reducer = StackSvdReducer::new(8);
        let a = Tensor::zeros([4, 16], (Kind::Float, Device::Cpu));
        let b = Tensor::zeros([16, 4], (Kind::Float, Device::Cpu));
        let contribs = vec![Contribution {
            lora_a: a,
            lora_b: b,
            scaling: 1.0,
            delta_norm_ratio: 0.0,
            provenance: Provenance::trusted("h1"),
        }];
        let (ra, rb) = reducer.reduce_key("k", &contribs).unwrap();
        assert_eq!(ra.size(), vec![8, 16]);
        assert_eq!(rb.size(), vec![16, 8]);
        let norm: f64 = reconstruct(&ra, &rb).norm().double_value(&[]);
        assert!(norm < 1e-6);
    }

    #[test]
    fn test_rank_cap_for_dtype_bytes() {
        assert_eq!(rank_cap_for_dtype_bytes(1), 8); // FP8/int8
        assert_eq!(rank_cap_for_dtype_bytes(2), 16); // FP16/BF16
        assert_eq!(rank_cap_for_dtype_bytes(4), usize::MAX); // FP32
        assert_eq!(rank_cap_for_dtype_bytes(0), usize::MAX); // unknown
    }
}
