//! Transport-independent inter-host pipeline planning and recovery (#325).
//!
//! The activation wire protocol belongs to #324. This module deliberately keeps
//! tensors and transport handles out of its types so a router can plan a chain
//! before any model weights are loaded and can recover a failed chain without
//! fabricating transport behaviour.
//!
//! Three invariants are enforced here:
//! - stages are contiguous and use the fewest LAN hosts whose usable VRAM fits;
//! - pipeline utilization is sized from `m / (m + S - 1)` (bubble fraction
//!   `(S - 1) / (m + S - 1)`);
//! - a failed stage invalidates the whole request epoch. Because KV is
//!   stage-local, the safe baseline is replay from the prompt on a replacement
//!   chain, never reuse a partially-populated downstream cache.
//!
//! Continuous decode itself is already present behind
//! `ModelOperations::forward_batched` / `TorchEngine::forward_batched_step`.
//! This module supplies the cross-stage depth calculation; the live streaming
//! scheduler remains tracked by #329 and is not duplicated here.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use hyprstream_rpc::identity::Did;
use hyprstream_rpc::resolver::{ResolutionEvidence, ServiceQuery, ServiceResolver};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::VerifyingKey;
use rand::RngCore;
use serde_json::{Map, Value};
use sha2::{Digest, Sha512};

const BASIS_POINTS: u64 = 10_000;

/// Weight footprint derived from a Hugging Face `config.json`.
///
/// Decoder layers are treated as equal-sized for stage assignment. The
/// estimate covers standard Llama/Qwen dense and MoE decoder layouts. When a
/// shard manifest provides `metadata.total_size`, its authoritative checkpoint
/// size scales the analytic regions while preserving their relative split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFootprint {
    num_layers: usize,
    layer_bytes: u64,
    embedding_bytes: u64,
    final_norm_bytes: u64,
    output_head_bytes: u64,
    tied_word_embeddings: bool,
}

impl ModelFootprint {
    /// Read `<model_dir>/config.json`, optionally refining the estimate from
    /// `<model_dir>/model.safetensors.index.json`'s `metadata.total_size`.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let manifest_path = model_dir.join("model.safetensors.index.json");
        let manifest_total = if manifest_path.exists() {
            let raw = std::fs::read_to_string(&manifest_path)
                .with_context(|| format!("failed to read {}", manifest_path.display()))?;
            let value: Value = serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
            let total = value
                .pointer("/metadata/total_size")
                .and_then(Value::as_u64)
                .filter(|total| *total > 0)
                .ok_or_else(|| {
                    anyhow!(
                        "{} has no positive metadata.total_size; refusing to discard its authoritative capacity bound",
                        manifest_path.display()
                    )
                })?;
            Some(total)
        } else {
            None
        };
        Self::from_config_path_with_total(&config_path, manifest_total)
    }

    /// Read a model `config.json` without consulting a shard manifest.
    pub fn from_config_path(config_path: &Path) -> Result<Self> {
        Self::from_config_path_with_total(config_path, None)
    }

    fn from_config_path_with_total(
        config_path: &Path,
        checkpoint_total_bytes: Option<u64>,
    ) -> Result<Self> {
        let raw = std::fs::read_to_string(config_path)
            .with_context(|| format!("failed to read required {}", config_path.display()))?;
        let root: Value = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse {}", config_path.display()))?;
        let root_obj = root
            .as_object()
            .ok_or_else(|| anyhow!("{} must contain a JSON object", config_path.display()))?;
        let text_obj = root_obj
            .get("text_config")
            .and_then(Value::as_object)
            .unwrap_or(root_obj);

        let num_layers = required_usize(text_obj, "num_hidden_layers", config_path)?;
        let hidden = required_u64(text_obj, "hidden_size", config_path)?;
        let intermediate = required_u64(text_obj, "intermediate_size", config_path)?;
        let vocab = required_u64(text_obj, "vocab_size", config_path)?;
        let attention_heads = required_u64(text_obj, "num_attention_heads", config_path)?;
        let kv_heads = text_obj
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .unwrap_or(attention_heads);
        if attention_heads == 0 || kv_heads == 0 {
            bail!("{} has a zero attention-head count", config_path.display());
        }
        let head_dim = text_obj
            .get("head_dim")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| hidden / attention_heads);
        if head_dim == 0 {
            bail!("{} resolves head_dim to zero", config_path.display());
        }

        let dtype = text_obj
            .get("torch_dtype")
            .or_else(|| root_obj.get("torch_dtype"))
            .and_then(Value::as_str)
            .unwrap_or("float32");
        let bits_per_weight = bits_per_weight(dtype)?;
        let tied_word_embeddings = text_obj
            .get("tie_word_embeddings")
            .or_else(|| root_obj.get("tie_word_embeddings"))
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let query_width = checked_mul(attention_heads, head_dim, "query width")?;
        let kv_width = checked_mul(kv_heads, head_dim, "KV width")?;
        let attention_params = checked_sum(&[
            checked_mul(hidden, query_width, "query projection")?,
            checked_mul(hidden, kv_width, "key projection")?,
            checked_mul(hidden, kv_width, "value projection")?,
            checked_mul(query_width, hidden, "output projection")?,
        ])?;
        let num_experts = text_obj
            .get("num_experts")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let mlp_params = if num_experts == 0 {
            checked_mul(
                checked_mul(hidden, intermediate, "MLP projection")?,
                3,
                "gated MLP projections",
            )?
        } else {
            // MoE capacity is determined by every resident expert, not only the
            // `num_experts_per_tok` experts selected for one token.
            let moe_intermediate = text_obj
                .get("moe_intermediate_size")
                .and_then(Value::as_u64)
                .filter(|value| *value > 0)
                .unwrap_or(intermediate);
            let expert_params = checked_mul(
                checked_mul(
                    checked_mul(hidden, moe_intermediate, "MoE expert projection")?,
                    3,
                    "MoE gated expert projections",
                )?,
                num_experts,
                "all MoE experts",
            )?;
            let router_params = checked_mul(hidden, num_experts, "MoE router")?;
            let shared_intermediate = text_obj
                .get("shared_expert_intermediate_size")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let shared_params = if shared_intermediate == 0 {
                0
            } else {
                checked_sum(&[
                    checked_mul(
                        checked_mul(hidden, shared_intermediate, "shared expert projection")?,
                        3,
                        "shared gated expert projections",
                    )?,
                    hidden, // optional shared-expert gate; conservative if absent
                ])?
            };
            checked_sum(&[expert_params, router_params, shared_params])?
        };
        let norm_params = checked_mul(hidden, 2, "per-layer norms")?;
        let layer_params = checked_sum(&[attention_params, mlp_params, norm_params])?;
        let embedding_params = checked_mul(vocab, hidden, "token embeddings")?;
        let output_head_params = if tied_word_embeddings {
            0
        } else {
            checked_mul(vocab, hidden, "output head")?
        };
        let unique_params = checked_sum(&[
            checked_mul(layer_params, num_layers as u64, "decoder layers")?,
            embedding_params,
            hidden,
            output_head_params,
        ])?;

        let analytic_bytes = params_to_bytes(unique_params, bits_per_weight)?;
        let target_bytes = checkpoint_total_bytes.unwrap_or(analytic_bytes);
        if target_bytes == 0 || analytic_bytes == 0 {
            bail!("{} describes an empty model", config_path.display());
        }

        let scale_region = |params: u64| -> Result<u64> {
            let bytes = params_to_bytes(params, bits_per_weight)?;
            scale_bytes(bytes, target_bytes, analytic_bytes)
        };
        let layer_bytes = scale_region(layer_params)?.max(1);
        let embedding_bytes = scale_region(embedding_params)?;
        let final_norm_bytes = scale_region(hidden)?;
        // A tied head is absent from the checkpoint, but a last stage on another
        // host must load a physical copy of the embedding matrix as its lm_head.
        let output_head_bytes = if tied_word_embeddings {
            embedding_bytes
        } else {
            scale_region(output_head_params)?
        };

        Ok(Self {
            num_layers,
            layer_bytes,
            embedding_bytes,
            final_norm_bytes,
            output_head_bytes,
            tied_word_embeddings,
        })
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    #[must_use]
    pub fn layer_bytes(&self) -> u64 {
        self.layer_bytes
    }

    /// Estimated weights for a single-host, unsplit model. A tied lm_head is not
    /// counted twice on this path.
    pub fn single_host_weight_bytes(&self) -> Result<u64> {
        let head = if self.tied_word_embeddings {
            0
        } else {
            self.output_head_bytes
        };
        checked_sum(&[
            checked_mul(
                self.layer_bytes,
                self.num_layers as u64,
                "all decoder layers",
            )?,
            self.embedding_bytes,
            self.final_norm_bytes,
            head,
        ])
    }

    fn first_stage_fixed_bytes(&self) -> u64 {
        self.embedding_bytes
    }

    fn last_stage_fixed_bytes(&self) -> Result<u64> {
        self.final_norm_bytes
            .checked_add(self.output_head_bytes)
            .ok_or_else(|| anyhow!("last-stage footprint overflow"))
    }
}

/// Cryptographic identity of a pipeline host.
///
/// The planner never accepts an arbitrary string as a host identity.  A host is
/// its `did:at9p` capsule identity; discovery binds its current reach and keys.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HostId(pub Did);

impl HostId {
    /// Construct a host identity only from the self-certifying host DID method.
    pub fn new(did: Did) -> Result<Self> {
        anyhow::ensure!(did.is_did_at9p(), "host identity must be a did:at9p DID");
        Ok(Self(did))
    }
}

/// Static, non-authoritative placement facts discovered for a host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostAttributes {
    pub available_vram_bytes: u64,
    /// A local placement-cost hint. It is never used as proof of reachability.
    pub affinity: Option<String>,
    /// A host-declared cost hint for future mover selection, never authority.
    pub declared_bandwidth_bytes_per_second: Option<u64>,
}

/// Available capacity for one pipeline host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCapacity {
    pub host_id: HostId,
    pub attributes: HostAttributes,
}

/// Planner safety reserves applied independently on every selected host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerOptions {
    /// Fraction of reported VRAM withheld for allocator overhead and activations.
    pub reserve_basis_points: u16,
    /// Additional absolute per-host reserve (for KV/runtime state).
    pub runtime_reserve_bytes: u64,
}

impl Default for PlannerOptions {
    fn default() -> Self {
        Self {
            reserve_basis_points: 1_000,
            runtime_reserve_bytes: 0,
        }
    }
}

/// One contiguous decoder stage assigned to one host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStageAssignment {
    pub stage_index: usize,
    pub host_id: HostId,
    pub layer_range: Range<usize>,
    pub estimated_weight_bytes: u64,
    pub usable_vram_bytes: u64,
}

/// Capacity-aware, pure-value plan. Reach is proved separately by [`admit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterhostPipelinePlan {
    pub stages: Vec<PipelineStageAssignment>,
}

impl InterhostPipelinePlan {
    #[must_use]
    pub fn boundary_count(&self) -> usize {
        self.stages.len().saturating_sub(1)
    }
}

/// Plans the smallest feasible set of hosts, then apportions contiguous layers
/// according to each host's residual usable VRAM.
pub struct InterhostPipelinePlanner;

impl InterhostPipelinePlanner {
    pub fn plan(
        model: &ModelFootprint,
        hosts: &[HostCapacity],
        options: PlannerOptions,
    ) -> Result<InterhostPipelinePlan> {
        if hosts.is_empty() {
            bail!("inter-host pipeline planning requires at least one host");
        }
        if options.reserve_basis_points as u64 >= BASIS_POINTS {
            bail!("reserve_basis_points must be less than {BASIS_POINTS}");
        }
        let mut seen = BTreeSet::new();
        for host in hosts {
            if !host.host_id.0.is_did_at9p() || !seen.insert(&host.host_id) {
                bail!("pipeline host identities must be unique did:at9p identifiers");
            }
        }

        let mut candidates = Vec::with_capacity(hosts.len());
        for host in hosts {
            let usable = usable_bytes(host.attributes.available_vram_bytes, options)?;
            if usable > 0 {
                candidates.push((host, usable));
            }
        }
        candidates.sort_by(|(left, left_bytes), (right, right_bytes)| {
            right_bytes
                .cmp(left_bytes)
                .then_with(|| left.host_id.cmp(&right.host_id))
        });

        for host_count in 1..=candidates.len().min(model.num_layers) {
            let selected = &candidates[..host_count];
            if let Some(stages) = try_plan_selected(model, selected)? {
                return Ok(InterhostPipelinePlan { stages });
            }
        }

        let total_usable = candidates.iter().try_fold(0_u64, |sum, (_, usable)| {
            sum.checked_add(*usable)
                .ok_or_else(|| anyhow!("aggregate host capacity overflow"))
        })?;
        bail!(
            "model does not fit the resolved host candidates: {} layers at ~{} bytes/layer, {} usable bytes total",
            model.num_layers,
            model.layer_bytes,
            total_usable
        )
    }
}

fn try_plan_selected(
    model: &ModelFootprint,
    selected: &[(&HostCapacity, u64)],
) -> Result<Option<Vec<PipelineStageAssignment>>> {
    if selected.len() == 1 {
        let required = model.single_host_weight_bytes()?;
        return Ok((required <= selected[0].1).then(|| {
            vec![PipelineStageAssignment {
                stage_index: 0,
                host_id: selected[0].0.host_id.clone(),
                layer_range: 0..model.num_layers,
                estimated_weight_bytes: required,
                usable_vram_bytes: selected[0].1,
            }]
        }));
    }

    let mut best: Option<(Vec<PipelineStageAssignment>, u64)> = None;
    for first in 0..selected.len() {
        for last in 0..selected.len() {
            if first == last {
                continue;
            }
            let mut order = vec![first];
            order.extend((0..selected.len()).filter(|idx| *idx != first && *idx != last));
            order.push(last);

            let mut max_layers = Vec::with_capacity(order.len());
            let mut feasible = true;
            for (stage, candidate_index) in order.iter().copied().enumerate() {
                let fixed = if stage == 0 {
                    model.first_stage_fixed_bytes()
                } else if stage + 1 == order.len() {
                    model.last_stage_fixed_bytes()?
                } else {
                    0
                };
                let usable = selected[candidate_index].1;
                let Some(layer_budget) = usable.checked_sub(fixed) else {
                    feasible = false;
                    break;
                };
                let capacity =
                    usize::try_from(layer_budget / model.layer_bytes).unwrap_or(usize::MAX);
                if capacity == 0 {
                    feasible = false;
                    break;
                }
                max_layers.push(capacity);
            }
            let aggregate_capacity = max_layers.iter().try_fold(0_usize, |sum, capacity| {
                sum.checked_add(*capacity)
                    .ok_or_else(|| anyhow!("aggregate layer capacity overflow"))
            })?;
            if !feasible || aggregate_capacity < model.num_layers {
                continue;
            }

            let counts = apportion_layers(&max_layers, model.num_layers)?;
            let mut cursor = 0;
            let mut stages = Vec::with_capacity(order.len());
            let mut peak_utilization_bps = 0;
            for (stage, candidate_index) in order.iter().copied().enumerate() {
                let fixed = if stage == 0 {
                    model.first_stage_fixed_bytes()
                } else if stage + 1 == order.len() {
                    model.last_stage_fixed_bytes()?
                } else {
                    0
                };
                let layer_cost = checked_mul(
                    model.layer_bytes,
                    counts[stage] as u64,
                    "assigned decoder layers",
                )?;
                let estimated_weight_bytes = fixed
                    .checked_add(layer_cost)
                    .ok_or_else(|| anyhow!("stage footprint overflow"))?;
                let usable = selected[candidate_index].1;
                let utilization_bps = (estimated_weight_bytes as u128)
                    .checked_mul(BASIS_POINTS as u128)
                    .ok_or_else(|| anyhow!("stage utilization overflow"))?
                    .div_ceil(usable as u128);
                let utilization_bps = u64::try_from(utilization_bps)
                    .map_err(|_| anyhow!("stage utilization overflow"))?;
                peak_utilization_bps = peak_utilization_bps.max(utilization_bps);
                stages.push(PipelineStageAssignment {
                    stage_index: stage,
                    host_id: selected[candidate_index].0.host_id.clone(),
                    layer_range: cursor..cursor + counts[stage],
                    estimated_weight_bytes,
                    usable_vram_bytes: usable,
                });
                cursor += counts[stage];
            }
            debug_assert_eq!(cursor, model.num_layers);

            if best
                .as_ref()
                .is_none_or(|(_, best_peak)| peak_utilization_bps < *best_peak)
            {
                best = Some((stages, peak_utilization_bps));
            }
        }
    }
    Ok(best.map(|(stages, _)| stages))
}

fn apportion_layers(capacities: &[usize], total_layers: usize) -> Result<Vec<usize>> {
    debug_assert!(capacities.len() <= total_layers);
    let mut assigned = vec![1; capacities.len()];
    for _ in capacities.len()..total_layers {
        let next = (0..capacities.len())
            .filter(|idx| assigned[*idx] < capacities[*idx])
            .min_by(|left, right| {
                let lhs = (assigned[*left] as u128) * (capacities[*right] as u128);
                let rhs = (assigned[*right] as u128) * (capacities[*left] as u128);
                lhs.cmp(&rhs).then_with(|| left.cmp(right))
            })
            .ok_or_else(|| anyhow!("aggregate layer capacity changed during apportionment"))?;
        assigned[next] += 1;
    }
    Ok(assigned)
}

fn usable_bytes(available: u64, options: PlannerOptions) -> Result<u64> {
    let after_fraction = (available as u128)
        .checked_mul((BASIS_POINTS - options.reserve_basis_points as u64) as u128)
        .ok_or_else(|| anyhow!("VRAM reserve calculation overflow"))?
        / BASIS_POINTS as u128;
    let after_fraction =
        u64::try_from(after_fraction).map_err(|_| anyhow!("VRAM reserve calculation overflow"))?;
    Ok(after_fraction.saturating_sub(options.runtime_reserve_bytes))
}

const REACH_ATTESTATION_AAD: &[u8] = b"hyprstream/reach-attestation/v0";

/// Verified, dialable discovery result for a pipeline host.
#[derive(Debug, Clone)]
pub struct HostDescriptor {
    pub host_id: HostId,
    pub response_verifying_key: VerifyingKey,
    pub response_ml_dsa65: Vec<u8>,
    pub transport: TransportConfig,
    pub evidence: ResolutionEvidence,
    pub expires_at_unix_ms: i64,
}

/// Resolves a host identity to the discovery engine's validated current snapshot.
#[async_trait]
pub trait HostResolver: Send + Sync {
    async fn resolve_host(
        &self,
        host: &HostId,
        required_capabilities: &BTreeSet<String>,
    ) -> Result<HostDescriptor>;
}

/// Production [`HostResolver`] backed by the rpc crate's identity-bound resolver.
///
/// Each host is queried through its own deployment-selected discovery record.
/// The resolved DID is checked against the requested [`HostId`] before a
/// descriptor is returned.
pub struct RpcHostResolver {
    host_services: BTreeMap<HostId, String>,
    resolver: Arc<dyn ServiceResolver>,
}

impl RpcHostResolver {
    pub fn new(
        host_services: BTreeMap<HostId, String>,
        resolver: Arc<dyn ServiceResolver>,
    ) -> Result<Self> {
        anyhow::ensure!(
            !host_services.is_empty(),
            "RPC host resolver requires at least one host-specific service record"
        );
        for service_name in host_services.values() {
            ServiceQuery::network(service_name.clone())?;
        }
        Ok(Self {
            host_services,
            resolver,
        })
    }
}

#[async_trait]
impl HostResolver for RpcHostResolver {
    async fn resolve_host(
        &self,
        host: &HostId,
        required_capabilities: &BTreeSet<String>,
    ) -> Result<HostDescriptor> {
        let service_name = self.host_services.get(host).ok_or_else(|| {
            anyhow!(
                "no discovery service record configured for pipeline host {}",
                host.0
            )
        })?;
        let query = ServiceQuery::new(
            service_name.clone(),
            required_capabilities.iter().cloned(),
            hyprstream_rpc::resolver::ResolverProfile::NetworkDiscovery,
            1,
        )?;
        let resolved = self.resolver.resolve_service(query).await?;
        self.resolver.ensure_current(&resolved).await?;
        resolved.ensure_fresh(unix_millis_now()?)?;
        let resolved_host = HostId::new(resolved.service_did().clone())?;
        anyhow::ensure!(
            &resolved_host == host,
            "resolved service DID does not match requested host identity"
        );
        Ok(HostDescriptor {
            host_id: resolved_host,
            response_verifying_key: resolved.response_verifying_key(),
            response_ml_dsa65: resolved.response_ml_dsa65().to_vec(),
            transport: resolved.transport().clone(),
            evidence: resolved.evidence().clone(),
            expires_at_unix_ms: resolved.expires_at_unix_ms(),
        })
    }
}

/// Route class carried by a signed reach proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReachPathClass {
    Direct,
    Relay,
}

/// Fresh hybrid-signed reach proof for one adjacent pipeline boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReachAttestation {
    pub from: HostId,
    pub to: HostId,
    pub epoch: u64,
    pub nonce: [u8; 32],
    pub path_class: ReachPathClass,
    pub established_via: String,
    pub rtt_estimate_micros: u64,
    pub expires_at_unix_ms: i64,
    signature: Vec<u8>,
}

/// Per-boundary values admission requires the target host to sign.
///
/// The epoch is the accepted discovery snapshot currently authorizing the
/// target, and the nonce is generated for this one admission attempt. Neither
/// value is optional, so an attestation cannot be replayed across a snapshot
/// advance or a later launch attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReachChallenge {
    pub accepted_state_epoch: u64,
    pub nonce: [u8; 32],
}

impl ReachAttestation {
    /// Sign a fresh attestation with the target host's hybrid identity keys.
    pub fn sign(
        from: HostId,
        to: HostId,
        epoch: u64,
        nonce: [u8; 32],
        path_class: ReachPathClass,
        established_via: String,
        rtt_estimate_micros: u64,
        expires_at_unix_ms: i64,
        signing_key: &hyprstream_rpc::SigningKey,
        pq_signing_key: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    ) -> Result<Self> {
        let mut attestation = Self {
            from,
            to,
            epoch,
            nonce,
            path_class,
            established_via,
            rtt_estimate_micros,
            expires_at_unix_ms,
            signature: Vec::new(),
        };
        anyhow::ensure!(
            !attestation.established_via.trim().is_empty(),
            "reach attestation has no established transport"
        );
        attestation.signature = hyprstream_crypto::cose_sign::sign_composite(
            signing_key,
            Some(pq_signing_key),
            &attestation.signing_bytes(),
            REACH_ATTESTATION_AAD,
        )?;
        Ok(attestation)
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for field in [
            self.from.0.as_str().as_bytes(),
            self.to.0.as_str().as_bytes(),
        ] {
            bytes.extend_from_slice(&(field.len() as u64).to_be_bytes());
            bytes.extend_from_slice(field);
        }
        bytes.extend_from_slice(&self.epoch.to_be_bytes());
        bytes.extend_from_slice(&self.nonce);
        bytes.push(match self.path_class {
            ReachPathClass::Direct => 0,
            ReachPathClass::Relay => 1,
        });
        bytes.extend_from_slice(&(self.established_via.len() as u64).to_be_bytes());
        bytes.extend_from_slice(self.established_via.as_bytes());
        bytes.extend_from_slice(&self.rtt_estimate_micros.to_be_bytes());
        bytes.extend_from_slice(&self.expires_at_unix_ms.to_be_bytes());
        bytes
    }

    fn verify_for(
        &self,
        from: &HostId,
        to: &HostDescriptor,
        challenge: ReachChallenge,
        now_unix_ms: i64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.from == *from,
            "reach attestation source does not match boundary"
        );
        anyhow::ensure!(
            self.to == to.host_id,
            "reach attestation target does not match boundary"
        );
        anyhow::ensure!(
            self.epoch == challenge.accepted_state_epoch,
            "reach attestation epoch does not match the accepted discovery snapshot"
        );
        anyhow::ensure!(
            self.nonce == challenge.nonce,
            "reach attestation nonce does not match the admission challenge"
        );
        anyhow::ensure!(
            !self.established_via.trim().is_empty(),
            "reach attestation has no established transport"
        );
        anyhow::ensure!(
            self.expires_at_unix_ms > now_unix_ms,
            "reach attestation expired"
        );
        anyhow::ensure!(
            self.expires_at_unix_ms <= to.expires_at_unix_ms,
            "reach attestation outlives its resolved host descriptor"
        );
        let pq_key = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&to.response_ml_dsa65)
            .context("host descriptor has invalid ML-DSA-65 key")?;
        hyprstream_crypto::cose_sign::verify_composite(
            &self.signature,
            &to.response_verifying_key,
            Some(&pq_key),
            &self.signing_bytes(),
            REACH_ATTESTATION_AAD,
            true,
        )
        .context("reach attestation signature is invalid")
        .map(|_| ())
    }
}

/// Network and MAC-PEP view used only by admission, never by the pure planner.
#[async_trait]
pub trait NetworkView: Send + Sync {
    /// Authorize this host-process boundary through the installed MAC PEP.
    async fn authorize_boundary(&self, from: &HostId, to: &HostId) -> Result<()>;

    /// Return a fresh, target-signed reach proof for this exact boundary and
    /// admission challenge.
    ///
    /// There is deliberately no `Option` result: absent evidence is an error and
    /// therefore refuses admission.
    async fn reachable(
        &self,
        from: &HostId,
        to: &HostId,
        challenge: ReachChallenge,
    ) -> Result<ReachAttestation>;
}

/// Evidence bound to an exact pure plan and expiring with its shortest proof.
#[derive(Debug, Clone)]
pub struct AdmissionRecord {
    pub plan_digest: [u8; 64],
    pub attestations: Vec<ReachAttestation>,
    pub expires_at_unix_ms: i64,
}

/// Resolve every planned host, authorize each adjacent boundary, and fail closed
/// unless each boundary supplies a fresh valid [`ReachAttestation`].
pub async fn admit(
    plan: &InterhostPipelinePlan,
    resolver: &dyn HostResolver,
    network: &dyn NetworkView,
) -> Result<AdmissionRecord> {
    let required_capabilities = BTreeSet::from(["hyprstream-moq/1".to_owned()]);
    let mut descriptors = BTreeMap::new();
    for stage in &plan.stages {
        let descriptor = resolver
            .resolve_host(&stage.host_id, &required_capabilities)
            .await
            .with_context(|| format!("failed to resolve pipeline host {}", stage.host_id.0))?;
        anyhow::ensure!(
            descriptor.host_id == stage.host_id,
            "resolver returned a descriptor for a different host"
        );
        descriptors.insert(stage.host_id.clone(), descriptor);
    }
    admit_resolved(plan, &descriptors, network).await
}

/// Production path: discovery validates each host identity before the pure
/// planner runs, then the exact resulting plan is admitted against those snapshots.
pub async fn plan_and_admit(
    model: &ModelFootprint,
    candidates: &[HostCapacity],
    options: PlannerOptions,
    resolver: &dyn HostResolver,
    network: &dyn NetworkView,
) -> Result<(InterhostPipelinePlan, AdmissionRecord)> {
    let required_capabilities = BTreeSet::from(["hyprstream-moq/1".to_owned()]);
    let mut descriptors = BTreeMap::new();
    for candidate in candidates {
        let descriptor = resolver
            .resolve_host(&candidate.host_id, &required_capabilities)
            .await
            .with_context(|| format!("failed to resolve pipeline host {}", candidate.host_id.0))?;
        anyhow::ensure!(
            descriptor.host_id == candidate.host_id,
            "resolver returned a different host"
        );
        descriptors.insert(candidate.host_id.clone(), descriptor);
    }
    let plan = InterhostPipelinePlanner::plan(model, candidates, options)?;
    let record = admit_resolved(&plan, &descriptors, network).await?;
    Ok((plan, record))
}

async fn admit_resolved(
    plan: &InterhostPipelinePlan,
    descriptors: &BTreeMap<HostId, HostDescriptor>,
    network: &dyn NetworkView,
) -> Result<AdmissionRecord> {
    anyhow::ensure!(
        !plan.stages.is_empty(),
        "cannot admit an empty pipeline plan"
    );
    let now = unix_millis_now()?;
    let mut attestations = Vec::with_capacity(plan.boundary_count());
    let mut expires_at_unix_ms = i64::MAX;
    for pair in plan.stages.windows(2) {
        let from = &pair[0].host_id;
        let to = &pair[1].host_id;
        network.authorize_boundary(from, to).await?;
        let descriptor = descriptors
            .get(to)
            .ok_or_else(|| anyhow!("planned target host was not resolved"))?;
        let mut nonce = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut nonce);
        let challenge = ReachChallenge {
            accepted_state_epoch: descriptor.evidence.accepted_state_epoch,
            nonce,
        };
        let attestation = network.reachable(from, to, challenge).await?;
        attestation.verify_for(from, descriptor, challenge, now)?;
        expires_at_unix_ms = expires_at_unix_ms.min(attestation.expires_at_unix_ms);
        attestations.push(attestation);
    }
    if attestations.is_empty() {
        expires_at_unix_ms = now;
    }
    Ok(AdmissionRecord {
        plan_digest: plan_digest(plan),
        attestations,
        expires_at_unix_ms,
    })
}

fn plan_digest(plan: &InterhostPipelinePlan) -> [u8; 64] {
    let mut digest = Sha512::new();
    for stage in &plan.stages {
        digest.update((stage.host_id.0.as_str().len() as u64).to_be_bytes());
        digest.update(stage.host_id.0.as_str().as_bytes());
        digest.update((stage.stage_index as u64).to_be_bytes());
        digest.update((stage.layer_range.start as u64).to_be_bytes());
        digest.update((stage.layer_range.end as u64).to_be_bytes());
        digest.update(stage.estimated_weight_bytes.to_be_bytes());
        digest.update(stage.usable_vram_bytes.to_be_bytes());
    }
    digest.finalize().into()
}

fn unix_millis_now() -> Result<i64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?;
    i64::try_from(elapsed.as_millis()).map_err(|_| anyhow!("Unix timestamp does not fit i64"))
}

/// Exact pipeline bubble/utilization sizing for `S` stages and `m`
/// microbatches in flight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MicrobatchSizing {
    pub stages: usize,
    pub microbatches: usize,
    pub bubble_fraction: f64,
    pub utilization: f64,
}

impl MicrobatchSizing {
    pub fn for_depth(stages: usize, microbatches: usize) -> Result<Self> {
        if stages == 0 || microbatches == 0 {
            bail!("pipeline stages and microbatch depth must both be at least one");
        }
        let denominator = microbatches
            .checked_add(stages - 1)
            .ok_or_else(|| anyhow!("microbatch sizing overflow"))? as f64;
        let bubble_fraction = (stages - 1) as f64 / denominator;
        Ok(Self {
            stages,
            microbatches,
            bubble_fraction,
            utilization: 1.0 - bubble_fraction,
        })
    }

    /// Minimum integer `m` that reaches `target_utilization_basis_points`.
    /// For example, four stages require 27 microbatches for at least 90%.
    pub fn required_depth(stages: usize, target_utilization_basis_points: u16) -> Result<Self> {
        if stages == 0 {
            bail!("pipeline stages must be at least one");
        }
        let target = target_utilization_basis_points as u64;
        if target == 0 || target > BASIS_POINTS {
            bail!("target utilization must be in 1..={BASIS_POINTS} basis points");
        }
        if stages == 1 {
            return Self::for_depth(1, 1);
        }
        if target == BASIS_POINTS {
            bail!("100% utilization requires unbounded depth for a multi-stage pipeline");
        }
        // m/(m+S-1) >= u  =>  m >= u(S-1)/(1-u)
        let numerator = target
            .checked_mul((stages - 1) as u64)
            .ok_or_else(|| anyhow!("microbatch sizing overflow"))?;
        let depth = numerator.div_ceil(BASIS_POINTS - target).max(1);
        let depth = usize::try_from(depth).map_err(|_| anyhow!("microbatch depth overflow"))?;
        Self::for_depth(stages, depth)
    }
}

/// Lifecycle of one independently queued pipeline job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineJobState {
    Running,
    Draining,
    AwaitingReplacement,
    Completed,
}

/// Recovery instruction emitted after a stage/host dies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartDirective {
    pub failed_stage: usize,
    pub old_epoch: u64,
    pub new_epoch: u64,
    pub aborted_microbatches: Vec<u64>,
    /// Always true in v1: stage-local KV on a replacement cannot be inferred
    /// from surviving stages, so the prompt must be replayed end-to-end.
    pub invalidate_all_kv: bool,
}

/// Result of acknowledging a microbatch completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionDisposition {
    Accepted,
    StaleEpochDiscarded,
}

/// Transport-independent per-job epoch and drain state.
///
/// Keeping this state per job avoids a failed/draining chain becoming a global
/// head-of-line blocker. Queue ownership and bounded backpressure are supplied
/// by #324's transport layer.
#[derive(Debug, Clone)]
pub struct PipelineJobRecovery {
    stage_count: usize,
    epoch: u64,
    state: PipelineJobState,
    in_flight: BTreeSet<u64>,
    completed: BTreeSet<u64>,
    pending_replay: BTreeSet<u64>,
    resume_draining: bool,
}

impl PipelineJobRecovery {
    pub fn new(stage_count: usize) -> Result<Self> {
        if stage_count == 0 {
            bail!("a pipeline job requires at least one stage");
        }
        Ok(Self {
            stage_count,
            epoch: 0,
            state: PipelineJobState::Running,
            in_flight: BTreeSet::new(),
            completed: BTreeSet::new(),
            pending_replay: BTreeSet::new(),
            resume_draining: false,
        })
    }

    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    #[must_use]
    pub fn state(&self) -> PipelineJobState {
        self.state
    }

    pub fn admit(&mut self, microbatch_id: u64) -> Result<u64> {
        if self.state != PipelineJobState::Running {
            bail!("pipeline job is not accepting work while {:?}", self.state);
        }
        if self.completed.contains(&microbatch_id) || !self.in_flight.insert(microbatch_id) {
            bail!(
                "microbatch {microbatch_id} was already admitted in epoch {}",
                self.epoch
            );
        }
        self.pending_replay.remove(&microbatch_id);
        Ok(self.epoch)
    }

    pub fn complete(&mut self, epoch: u64, microbatch_id: u64) -> Result<CompletionDisposition> {
        if epoch < self.epoch {
            return Ok(CompletionDisposition::StaleEpochDiscarded);
        }
        if epoch > self.epoch {
            bail!(
                "completion epoch {epoch} is ahead of current epoch {}",
                self.epoch
            );
        }
        if !self.in_flight.remove(&microbatch_id) {
            bail!("microbatch {microbatch_id} is not in flight in epoch {epoch}");
        }
        self.completed.insert(microbatch_id);
        Ok(CompletionDisposition::Accepted)
    }

    /// Stop new admission but allow the current epoch to finish (maintenance).
    pub fn begin_drain(&mut self) -> Result<()> {
        if self.state != PipelineJobState::Running {
            bail!("pipeline job cannot begin drain while {:?}", self.state);
        }
        self.state = PipelineJobState::Draining;
        Ok(())
    }

    #[must_use]
    pub fn drain_complete(&self) -> bool {
        self.state == PipelineJobState::Draining
            && self.in_flight.is_empty()
            && self.pending_replay.is_empty()
    }

    /// Abort the current epoch immediately after a stage failure.
    pub fn stage_failed(
        &mut self,
        expected_epoch: u64,
        failed_stage: usize,
    ) -> Result<RestartDirective> {
        anyhow::ensure!(
            expected_epoch == self.epoch,
            "stage failure epoch {expected_epoch} does not match current epoch {}",
            self.epoch
        );
        if failed_stage >= self.stage_count {
            bail!(
                "failed stage {failed_stage} is outside a {}-stage pipeline",
                self.stage_count
            );
        }
        if matches!(
            self.state,
            PipelineJobState::Completed | PipelineJobState::AwaitingReplacement
        ) {
            bail!("pipeline job cannot fail a stage while {:?}", self.state);
        }
        let resume_draining = self.state == PipelineJobState::Draining;
        let old_epoch = self.epoch;
        let new_epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow!("pipeline epoch exhausted"))?;
        let mut aborted = std::mem::take(&mut self.in_flight);
        aborted.append(&mut self.pending_replay);
        let aborted_microbatches = aborted.iter().copied().collect();
        self.pending_replay = aborted;
        self.epoch = new_epoch;
        self.completed.clear();
        self.resume_draining = resume_draining;
        self.state = PipelineJobState::AwaitingReplacement;
        Ok(RestartDirective {
            failed_stage,
            old_epoch,
            new_epoch,
            aborted_microbatches,
            invalidate_all_kv: true,
        })
    }

    /// Install a healthy replacement chain and resume at the already-advanced epoch.
    pub fn replacement_ready(&mut self, expected_epoch: u64, stage_count: usize) -> Result<()> {
        if self.state != PipelineJobState::AwaitingReplacement {
            bail!("replacement is only valid after a stage failure");
        }
        anyhow::ensure!(
            expected_epoch == self.epoch,
            "replacement epoch {expected_epoch} does not match current epoch {}",
            self.epoch
        );
        if stage_count == 0 {
            bail!("a replacement pipeline requires at least one stage");
        }
        self.stage_count = stage_count;
        self.state = if std::mem::take(&mut self.resume_draining) {
            PipelineJobState::Draining
        } else {
            PipelineJobState::Running
        };
        Ok(())
    }

    /// Re-admit work aborted by a failure without reopening normal admission.
    ///
    /// This is used after a replacement resumes a planned drain. Only IDs that
    /// were already in flight before the failure may be replayed.
    pub fn replay_aborted(&mut self, microbatch_id: u64) -> Result<u64> {
        if self.state != PipelineJobState::Draining {
            bail!("drain-only replay requires a replacement in Draining state");
        }
        if !self.pending_replay.contains(&microbatch_id) {
            bail!("microbatch {microbatch_id} is not pending replay");
        }
        if !self.in_flight.insert(microbatch_id) {
            bail!(
                "microbatch {microbatch_id} is already in flight in epoch {}",
                self.epoch
            );
        }
        self.pending_replay.remove(&microbatch_id);
        Ok(self.epoch)
    }

    pub fn finish(&mut self) -> Result<()> {
        if !self.in_flight.is_empty() {
            bail!("cannot finish a pipeline job with microbatches still in flight");
        }
        if !self.pending_replay.is_empty() {
            bail!("cannot finish a pipeline job with microbatches pending replay");
        }
        if self.state == PipelineJobState::AwaitingReplacement {
            bail!("cannot finish while awaiting a replacement chain");
        }
        self.state = PipelineJobState::Completed;
        Ok(())
    }
}

fn required_u64(object: &Map<String, Value>, field: &str, path: &Path) -> Result<u64> {
    let value = object.get(field).and_then(Value::as_u64).ok_or_else(|| {
        anyhow!(
            "{} is missing required positive integer field `{field}`",
            path.display()
        )
    })?;
    if value == 0 {
        bail!("{} has zero `{field}`", path.display());
    }
    Ok(value)
}

fn required_usize(object: &Map<String, Value>, field: &str, path: &Path) -> Result<usize> {
    let value = required_u64(object, field, path)?;
    usize::try_from(value).map_err(|_| anyhow!("{field} does not fit usize"))
}

fn bits_per_weight(dtype: &str) -> Result<u64> {
    match dtype.to_ascii_lowercase().as_str() {
        "float64" | "f64" => Ok(64),
        "float32" | "f32" => Ok(32),
        "float16" | "f16" | "half" | "bfloat16" | "bf16" => Ok(16),
        "float8" | "fp8" | "float8_e4m3fn" | "float8_e5m2" => Ok(8),
        other => bail!("unsupported model torch_dtype `{other}` for capacity planning"),
    }
}

fn params_to_bytes(params: u64, bits: u64) -> Result<u64> {
    params
        .checked_mul(bits)
        .ok_or_else(|| anyhow!("model footprint overflow"))
        .map(|value| value.div_ceil(8))
}

fn scale_bytes(bytes: u64, target: u64, source: u64) -> Result<u64> {
    let scaled = (bytes as u128)
        .checked_mul(target as u128)
        .ok_or_else(|| anyhow!("model footprint scale overflow"))?
        .div_ceil(source as u128);
    u64::try_from(scaled).map_err(|_| anyhow!("model footprint scale overflow"))
}

fn checked_mul(left: u64, right: u64, region: &str) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{region} footprint overflow"))
}

fn checked_sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| anyhow!("model footprint overflow"))
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn synthetic_model(layers: usize, layer_bytes: u64) -> ModelFootprint {
        ModelFootprint {
            num_layers: layers,
            layer_bytes,
            embedding_bytes: 0,
            final_norm_bytes: 0,
            output_head_bytes: 0,
            tied_word_embeddings: false,
        }
    }

    fn host(id: &str, capacity: u64) -> HostCapacity {
        HostCapacity {
            host_id: HostId::new(Did::new(format!("did:at9p:{id}"))).unwrap(),
            attributes: HostAttributes {
                available_vram_bytes: capacity,
                affinity: Some("rack-a".to_owned()),
                declared_bandwidth_bytes_per_second: None,
            },
        }
    }

    fn no_reserve() -> PlannerOptions {
        PlannerOptions {
            reserve_basis_points: 0,
            runtime_reserve_bytes: 0,
        }
    }

    #[test]
    fn asymmetric_capacity_gets_asymmetric_contiguous_ranges() {
        let model = synthetic_model(12, 10);
        let plan = InterhostPipelinePlanner::plan(
            &model,
            &[host("mi210", 85), host("rtx5090", 45)],
            no_reserve(),
        )
        .unwrap();

        assert_eq!(plan.boundary_count(), 1);
        assert_eq!(plan.stages[0].host_id.0.as_str(), "did:at9p:mi210");
        assert_eq!(plan.stages[0].layer_range, 0..8);
        assert_eq!(plan.stages[1].host_id.0.as_str(), "did:at9p:rtx5090");
        assert_eq!(plan.stages[1].layer_range, 8..12);
    }

    #[test]
    fn planner_minimizes_boundaries_before_balancing() {
        let model = synthetic_model(12, 10);
        let plan = InterhostPipelinePlanner::plan(
            &model,
            &[host("big", 200), host("small-a", 80), host("small-b", 80)],
            no_reserve(),
        )
        .unwrap();
        assert_eq!(plan.stages.len(), 1);
        assert_eq!(plan.stages[0].host_id.0.as_str(), "did:at9p:big");
        assert_eq!(plan.stages[0].layer_range, 0..12);
    }

    #[test]
    fn planner_uses_affinity_only_as_a_cost_hint_and_rejects_insufficient_capacity() {
        let model = synthetic_model(12, 10);
        let mut remote = host("remote", 200);
        remote.attributes.affinity = Some("wan".to_owned());
        assert!(InterhostPipelinePlanner::plan(
            &model,
            &[host("local", 100), remote],
            no_reserve()
        )
        .is_ok());
        assert!(InterhostPipelinePlanner::plan(
            &model,
            &[host("a", 50), host("b", 50)],
            no_reserve()
        )
        .is_err());
    }

    #[test]
    fn host_below_runtime_reserve_does_not_poison_other_candidates() {
        let model = synthetic_model(12, 10);
        let plan = InterhostPipelinePlanner::plan(
            &model,
            &[host("undersized", 50), host("fits", 250)],
            PlannerOptions {
                reserve_basis_points: 0,
                runtime_reserve_bytes: 100,
            },
        )
        .unwrap();

        assert_eq!(plan.stages.len(), 1);
        assert_eq!(plan.stages[0].host_id.0.as_str(), "did:at9p:fits");
        assert_eq!(plan.stages[0].usable_vram_bytes, 150);
    }

    #[test]
    fn config_json_drives_model_footprint_and_tied_head() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "text_config": {
                    "num_hidden_layers": 4,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "vocab_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "torch_dtype": "bfloat16",
                    "tie_word_embeddings": true
                }
            }"#,
        )
        .unwrap();
        let footprint = ModelFootprint::from_model_dir(dir.path()).unwrap();
        assert_eq!(footprint.num_layers(), 4);
        assert!(footprint.layer_bytes() > 0);
        assert_eq!(footprint.output_head_bytes, footprint.embedding_bytes);
        assert!(footprint.single_host_weight_bytes().unwrap() > 0);
    }

    #[test]
    fn manifest_total_scales_the_analytic_estimate() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "num_hidden_layers": 2,
                "hidden_size": 8,
                "intermediate_size": 16,
                "vocab_size": 32,
                "num_attention_heads": 2,
                "torch_dtype": "float32"
            }"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":100000},"weight_map":{}}"#,
        )
        .unwrap();
        let footprint = ModelFootprint::from_model_dir(dir.path()).unwrap();
        let estimated = footprint.single_host_weight_bytes().unwrap();
        assert!(estimated >= 100_000);
        assert!(estimated < 101_000);
    }

    #[test]
    fn manifest_without_authoritative_total_size_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "num_hidden_layers": 2,
                "hidden_size": 8,
                "intermediate_size": 16,
                "vocab_size": 32,
                "num_attention_heads": 2,
                "torch_dtype": "float32"
            }"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{}}"#,
        )
        .unwrap();

        assert!(ModelFootprint::from_model_dir(dir.path())
            .unwrap_err()
            .to_string()
            .contains("authoritative capacity bound"));
    }

    #[test]
    fn moe_footprint_counts_all_resident_experts() {
        let dense_dir = tempfile::tempdir().unwrap();
        let moe_dir = tempfile::tempdir().unwrap();
        let base = r#"
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "vocab_size": 32,
            "num_attention_heads": 2,
            "torch_dtype": "bfloat16"
        "#;
        std::fs::write(dense_dir.path().join("config.json"), format!("{{{base}}}")).unwrap();
        std::fs::write(
            moe_dir.path().join("config.json"),
            format!(
                "{{{base}, \"num_experts\": 8, \"num_experts_per_tok\": 2, \
                 \"moe_intermediate_size\": 16}}"
            ),
        )
        .unwrap();

        let dense = ModelFootprint::from_model_dir(dense_dir.path()).unwrap();
        let moe = ModelFootprint::from_model_dir(moe_dir.path()).unwrap();
        assert!(moe.layer_bytes() > dense.layer_bytes() * 4);
    }

    struct MockHostResolver {
        descriptors: BTreeMap<HostId, HostDescriptor>,
        refuse: bool,
    }

    #[async_trait]
    impl HostResolver for MockHostResolver {
        async fn resolve_host(
            &self,
            host: &HostId,
            _required_capabilities: &BTreeSet<String>,
        ) -> Result<HostDescriptor> {
            if self.refuse {
                bail!("host is unresolvable");
            }
            self.descriptors
                .get(host)
                .cloned()
                .ok_or_else(|| anyhow!("host is unresolvable"))
        }
    }

    struct RecordingServiceResolver {
        snapshots: BTreeMap<String, hyprstream_rpc::resolver::SelectedService>,
        requests: parking_lot::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl ServiceResolver for RecordingServiceResolver {
        async fn resolve_service(
            &self,
            query: ServiceQuery,
        ) -> anyhow::Result<hyprstream_rpc::resolver::SelectedService> {
            self.requests.lock().push(query.service_name.clone());
            self.snapshots
                .get(&query.service_name)
                .cloned()
                .ok_or_else(|| anyhow!("unexpected discovery query {}", query.service_name))
        }

        async fn ensure_current(
            &self,
            _resolved: &hyprstream_rpc::resolver::SelectedService,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn selected_host_service(
        host: &HostId,
        service_name: &str,
        tag: u8,
    ) -> hyprstream_rpc::resolver::SelectedService {
        use hyprstream_rpc::crypto::hybrid_kem::{generate_recipient, SuiteId};
        use hyprstream_rpc::resolver::{
            select_service_candidate, AcceptedStateEvidence, AnchoredKemRecipient, ResolverProfile,
            ServiceCandidate,
        };

        let signing_key = hyprstream_rpc::SigningKey::from_bytes(&[tag; 32]);
        let (_pq_signing_key, pq_verifying_key) =
            hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let pq = hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&pq_verifying_key);
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768)
            .unwrap()
            .public();
        let now = unix_millis_now().unwrap();
        let candidate = ServiceCandidate {
            service_name: service_name.to_owned(),
            service_did: host.0.clone(),
            response_verifying_key: signing_key.verifying_key().to_bytes(),
            response_ml_dsa65: pq.clone(),
            response_key_id: format!("{}#response", host.0),
            request_kem_recipient: Some(AnchoredKemRecipient {
                key_id: format!("{}#kem", host.0),
                recipient,
                not_after_unix_ms: now + 120_000,
            }),
            transport: TransportConfig::iroh([tag; 32], Vec::new(), None),
            capabilities: ["hyprstream-moq/1".to_owned()].into_iter().collect(),
            accepted_state: AcceptedStateEvidence {
                service_did: host.0.clone(),
                digest: [tag; 64],
                epoch: u64::from(tag),
                expires_at_unix_ms: now + 120_000,
                response_ed25519: signing_key.verifying_key().to_bytes(),
                response_ml_dsa65: pq,
            },
            source_signer: signing_key.verifying_key().to_bytes(),
            expires_at_unix_ms: now + 120_000,
        };
        let query = ServiceQuery::new(
            service_name,
            ["hyprstream-moq/1".to_owned()],
            ResolverProfile::NetworkDiscovery,
            1,
        )
        .unwrap();
        select_service_candidate(&query, vec![candidate], now).unwrap()
    }

    #[tokio::test]
    async fn rpc_host_resolver_resolves_each_host_through_its_own_service_record() {
        let alpha = host("alpha", 80).host_id;
        let beta = host("beta", 80).host_id;
        let alpha_service = "inference-alpha".to_owned();
        let beta_service = "inference-beta".to_owned();
        let resolver = Arc::new(RecordingServiceResolver {
            snapshots: BTreeMap::from([
                (
                    alpha_service.clone(),
                    selected_host_service(&alpha, &alpha_service, 1),
                ),
                (
                    beta_service.clone(),
                    selected_host_service(&beta, &beta_service, 2),
                ),
            ]),
            requests: parking_lot::Mutex::new(Vec::new()),
        });
        let host_resolver = RpcHostResolver::new(
            BTreeMap::from([
                (alpha.clone(), alpha_service.clone()),
                (beta.clone(), beta_service.clone()),
            ]),
            resolver.clone(),
        )
        .unwrap();
        let required = BTreeSet::from(["hyprstream-moq/1".to_owned()]);

        assert_eq!(
            host_resolver
                .resolve_host(&alpha, &required)
                .await
                .unwrap()
                .host_id,
            alpha
        );
        assert_eq!(
            host_resolver
                .resolve_host(&beta, &required)
                .await
                .unwrap()
                .host_id,
            beta
        );
        assert_eq!(*resolver.requests.lock(), vec![alpha_service, beta_service]);
    }

    enum TestReachResponse {
        Attested,
        Missing,
        Expired,
        StaleEpoch,
        ReplayedNonce,
    }

    struct MockNetworkView {
        signing_keys: BTreeMap<
            HostId,
            (
                hyprstream_rpc::SigningKey,
                hyprstream_rpc::crypto::pq::MlDsaSigningKey,
            ),
        >,
        response: TestReachResponse,
        deny_mac: bool,
    }

    #[async_trait]
    impl NetworkView for MockNetworkView {
        async fn authorize_boundary(&self, _from: &HostId, _to: &HostId) -> Result<()> {
            anyhow::ensure!(!self.deny_mac, "MAC PEP denied pipeline boundary");
            Ok(())
        }

        async fn reachable(
            &self,
            from: &HostId,
            to: &HostId,
            challenge: ReachChallenge,
        ) -> Result<ReachAttestation> {
            if matches!(self.response, TestReachResponse::Missing) {
                bail!("missing reach attestation");
            }
            let (signing_key, pq_signing_key) = self
                .signing_keys
                .get(to)
                .ok_or_else(|| anyhow!("no target attestation key"))?;
            let now = unix_millis_now()?;
            let expiry = if matches!(self.response, TestReachResponse::Expired) {
                now - 1
            } else {
                now + 60_000
            };
            let epoch = if matches!(self.response, TestReachResponse::StaleEpoch) {
                challenge.accepted_state_epoch - 1
            } else {
                challenge.accepted_state_epoch
            };
            let nonce = if matches!(self.response, TestReachResponse::ReplayedNonce) {
                [0xA5; 32]
            } else {
                challenge.nonce
            };
            ReachAttestation::sign(
                from.clone(),
                to.clone(),
                epoch,
                nonce,
                ReachPathClass::Direct,
                "iroh".to_owned(),
                100,
                expiry,
                signing_key,
                pq_signing_key,
            )
        }
    }

    fn admission_fixture(
        response: TestReachResponse,
        refuse: bool,
    ) -> (Vec<HostCapacity>, MockHostResolver, MockNetworkView) {
        let candidates = vec![host("alpha", 80), host("beta", 80)];
        let now = unix_millis_now().unwrap();
        let mut descriptors = BTreeMap::new();
        let mut signing_keys = BTreeMap::new();
        for (index, candidate) in candidates.iter().enumerate() {
            let tag = u8::try_from(index + 1).unwrap();
            let signing_key = hyprstream_rpc::SigningKey::from_bytes(&[tag; 32]);
            let (pq_signing_key, pq_verifying_key) =
                hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
            descriptors.insert(
                candidate.host_id.clone(),
                HostDescriptor {
                    host_id: candidate.host_id.clone(),
                    response_verifying_key: signing_key.verifying_key(),
                    response_ml_dsa65: hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
                        &pq_verifying_key,
                    ),
                    transport: TransportConfig::iroh([tag; 32], Vec::new(), None),
                    evidence: ResolutionEvidence {
                        accepted_state_digest: [tag; 64],
                        accepted_state_epoch: 1,
                        selected_fingerprint: format!("host-{tag}"),
                        ordered_decisions: Vec::new(),
                    },
                    expires_at_unix_ms: now + 120_000,
                },
            );
            signing_keys.insert(candidate.host_id.clone(), (signing_key, pq_signing_key));
        }
        (
            candidates,
            MockHostResolver {
                descriptors,
                refuse,
            },
            MockNetworkView {
                signing_keys,
                response,
                deny_mac: false,
            },
        )
    }

    #[tokio::test]
    async fn real_admission_path_invokes_planner_and_records_verified_reach() {
        let (candidates, resolver, network) = admission_fixture(TestReachResponse::Attested, false);
        let (plan, record) = plan_and_admit(
            &synthetic_model(12, 10),
            &candidates,
            no_reserve(),
            &resolver,
            &network,
        )
        .await
        .unwrap();

        assert_eq!(plan.boundary_count(), 1);
        assert_eq!(record.attestations.len(), 1);
        assert!(record.expires_at_unix_ms > unix_millis_now().unwrap());
    }

    #[tokio::test]
    async fn missing_or_invalid_reach_attestation_refuses_admission() {
        for response in [
            TestReachResponse::Missing,
            TestReachResponse::Expired,
            TestReachResponse::StaleEpoch,
            TestReachResponse::ReplayedNonce,
        ] {
            let (candidates, resolver, network) = admission_fixture(response, false);
            assert!(plan_and_admit(
                &synthetic_model(12, 10),
                &candidates,
                no_reserve(),
                &resolver,
                &network,
            )
            .await
            .is_err());
        }
    }

    #[tokio::test]
    async fn stale_epoch_and_replayed_nonce_refuse_admission() {
        for response in [
            TestReachResponse::StaleEpoch,
            TestReachResponse::ReplayedNonce,
        ] {
            let (candidates, resolver, network) = admission_fixture(response, false);
            let result = plan_and_admit(
                &synthetic_model(12, 10),
                &candidates,
                no_reserve(),
                &resolver,
                &network,
            )
            .await;
            assert!(
                result.is_err(),
                "stale reach evidence must not admit a pipeline"
            );
            let error = result.unwrap_err();
            assert!(
                error.to_string().contains("accepted discovery snapshot")
                    || error.to_string().contains("admission challenge"),
                "unexpected refusal: {error:#}"
            );
        }
    }

    #[tokio::test]
    async fn unresolvable_host_refuses_admission() {
        let (candidates, resolver, network) = admission_fixture(TestReachResponse::Attested, true);
        assert!(plan_and_admit(
            &synthetic_model(12, 10),
            &candidates,
            no_reserve(),
            &resolver,
            &network,
        )
        .await
        .is_err());
    }

    #[test]
    fn bubble_fraction_and_required_depth_are_exact() {
        let single = MicrobatchSizing::for_depth(4, 1).unwrap();
        assert!((single.bubble_fraction - 0.75).abs() < f64::EPSILON);
        assert!((single.utilization - 0.25).abs() < f64::EPSILON);

        let ninety = MicrobatchSizing::required_depth(4, 9_000).unwrap();
        assert_eq!(ninety.microbatches, 27);
        assert!(ninety.utilization >= 0.9);
        let previous = MicrobatchSizing::for_depth(4, 26).unwrap();
        assert!(previous.utilization < 0.9);
    }

    #[test]
    fn stage_failure_advances_epoch_and_invalidates_partial_kv() {
        let mut recovery = PipelineJobRecovery::new(4).unwrap();
        let old_epoch = recovery.admit(10).unwrap();
        recovery.admit(11).unwrap();
        let restart = recovery.stage_failed(old_epoch, 2).unwrap();

        assert_eq!(restart.old_epoch, old_epoch);
        assert_eq!(restart.new_epoch, old_epoch + 1);
        assert_eq!(restart.aborted_microbatches, vec![10, 11]);
        assert!(restart.invalidate_all_kv);
        assert_eq!(recovery.state(), PipelineJobState::AwaitingReplacement);
        assert_eq!(
            recovery.complete(old_epoch, 10).unwrap(),
            CompletionDisposition::StaleEpochDiscarded
        );

        recovery.replacement_ready(restart.new_epoch, 3).unwrap();
        assert_eq!(recovery.admit(10).unwrap(), restart.new_epoch);
    }

    #[test]
    fn completion_rejects_future_epoch_without_consuming_inflight_work() {
        let mut recovery = PipelineJobRecovery::new(2).unwrap();
        let epoch = recovery.admit(10).unwrap();

        let error = recovery.complete(epoch + 1, 10).unwrap_err();
        assert!(error.to_string().contains("ahead of current epoch"));
        assert_eq!(
            recovery.complete(epoch, 10).unwrap(),
            CompletionDisposition::Accepted
        );
    }

    #[test]
    fn same_epoch_microbatch_id_reuse_is_refused_to_prevent_aba_completion() {
        let mut recovery = PipelineJobRecovery::new(2).unwrap();
        let epoch = recovery.admit(10).unwrap();
        assert_eq!(
            recovery.complete(epoch, 10).unwrap(),
            CompletionDisposition::Accepted
        );
        assert!(recovery.admit(10).is_err());
    }

    #[test]
    fn delayed_recovery_control_messages_are_epoch_fenced() {
        let mut recovery = PipelineJobRecovery::new(2).unwrap();
        let epoch = recovery.epoch();
        let restart = recovery.stage_failed(epoch, 0).unwrap();
        assert!(recovery.replacement_ready(epoch, 2).is_err());
        assert!(recovery.stage_failed(epoch, 0).is_err());
        recovery.replacement_ready(restart.new_epoch, 2).unwrap();
    }

    #[test]
    fn maintenance_drain_stops_admission_but_finishes_inflight() {
        let mut recovery = PipelineJobRecovery::new(2).unwrap();
        let epoch = recovery.admit(7).unwrap();
        recovery.begin_drain().unwrap();
        assert!(recovery.admit(8).is_err());
        assert!(!recovery.drain_complete());
        assert_eq!(
            recovery.complete(epoch, 7).unwrap(),
            CompletionDisposition::Accepted
        );
        assert!(recovery.drain_complete());
        recovery.finish().unwrap();
        assert_eq!(recovery.state(), PipelineJobState::Completed);
    }

    #[test]
    fn stage_failure_preserves_drain_and_only_replays_aborted_work() {
        let mut recovery = PipelineJobRecovery::new(2).unwrap();
        let old_epoch = recovery.admit(7).unwrap();
        recovery.begin_drain().unwrap();

        let restart = recovery.stage_failed(old_epoch, 1).unwrap();
        assert_eq!(restart.aborted_microbatches, vec![7]);
        recovery.replacement_ready(restart.new_epoch, 2).unwrap();

        assert_eq!(recovery.state(), PipelineJobState::Draining);
        assert!(recovery.admit(8).is_err());
        assert!(recovery.replay_aborted(8).is_err());
        assert!(!recovery.drain_complete());
        assert!(recovery.finish().is_err());

        let replay_epoch = recovery.replay_aborted(7).unwrap();
        assert_eq!(replay_epoch, restart.new_epoch);
        assert_eq!(
            recovery.complete(old_epoch, 7).unwrap(),
            CompletionDisposition::StaleEpochDiscarded
        );
        assert_eq!(
            recovery.complete(replay_epoch, 7).unwrap(),
            CompletionDisposition::Accepted
        );
        assert!(recovery.drain_complete());
        recovery.finish().unwrap();
        assert_eq!(recovery.state(), PipelineJobState::Completed);
    }

    #[test]
    fn recovery_state_is_per_job_not_global_hol() {
        let mut failed = PipelineJobRecovery::new(2).unwrap();
        let mut healthy = PipelineJobRecovery::new(2).unwrap();
        failed.admit(1).unwrap();
        failed.stage_failed(0, 0).unwrap();
        assert_eq!(healthy.admit(1).unwrap(), 0);
        assert_eq!(healthy.state(), PipelineJobState::Running);
    }
}
