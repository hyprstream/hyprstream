//! Model factory for creating models with unified configuration management
//!
//! This replaces the chaotic multiple paths for model creation with a single,
//! clean factory pattern.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::path::{Path, PathBuf};
use tch::{Device, Kind as DType, Tensor};
use tracing::{debug, info, instrument, warn};

use super::architectures::{gemma::GemmaModel, llama::LlamaModel, ModelOperations};
use super::device_pool::DevicePool;
use super::KVQuantType;
use super::model_config::{ModelArchitecture, ModelConfig};
use super::torch_utils::{safe_to_device, estimate_tensor_size_mb};
use crate::services::WorktreeClient;

/// Strict-loader opt-in (#315): when set truthy, multi-shard models that lack a
/// `model.safetensors.index.json` manifest are rejected instead of silently
/// reconstructing the shard set from a filename glob (which is fragile across
/// model families and can silently drop/duplicate shards).
const STRICT_LOADER_ENV: &str = "HYPRSTREAM_STRICT_LOADER";

fn strict_loader_enabled() -> bool {
    std::env::var(STRICT_LOADER_ENV)
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Materialize every FP8 weight in `weights` as BF16 once, applying the
/// companion block-wise `<name>_scale_inv` scales during the conversion, and
/// drop the scale tensors. Gated by `HYPRSTREAM_FP8_DEQUANT_LOAD` /
/// [`crate::config::RuntimeConfig::fp8_dequant_load`].
///
/// This runs on the raw checkpoint tensor map (all FP8 weights are 2D there,
/// stored `[out, in]` with scales `[out/128, in/128]`), so a single generic
/// pass covers every FP8 site downstream — llama-family `LinearProjection`,
/// Qwen3.5 fused QKV/gate-up projections, MoE expert stacks, and lm_head all
/// observe plain BF16 weights with no scales, and their lazy per-matmul
/// dequant branches never trigger. Returns the number of tensors converted.
pub(crate) fn dequantize_fp8_weights_at_load(weights: &mut HashMap<String, Tensor>) -> usize {
    #[inline]
    fn is_fp8(kind: tch::Kind) -> bool {
        matches!(kind, tch::Kind::Float8e4m3fn | tch::Kind::Float8e5m2)
    }

    let fp8_keys: Vec<String> = weights
        .iter()
        .filter(|(name, tensor)| is_fp8(tensor.kind()) && !name.ends_with("_scale_inv"))
        .map(|(name, _)| name.clone())
        .collect();
    let mut converted = 0;

    for key in fp8_keys {
        let Some(weight) = weights.remove(&key) else { continue };
        let scale_key = format!("{key}_scale_inv");
        let scale = weights.remove(&scale_key);
        if let Some(s) = &scale {
            let ws = weight.size();
            let ss = s.size();
            if !(ws.len() == 2 && ss.len() == 2 && ws[0] % ss[0] == 0 && ws[1] % ss[1] == 0) {
                // Non-conforming scale shape: leave the weight FP8 and the scale
                // in the map untouched. Downstream lazy dequant keeps ownership
                // of the pair and fails loudly if the shapes really are unusable,
                // instead of silently running on raw FP8-code magnitudes.
                warn!(
                    "FP8 weight '{}' scale shape {:?} does not block-divide weight shape {:?}; \
                     leaving FP8 + scale untouched for the lazy dequant path",
                    key, ss, ws
                );
                weights.insert(scale_key, s.shallow_clone());
                weights.insert(key, weight);
                continue;
            }
        }
        // Block-wise dequantization via 4D broadcast multiply (same math as the
        // lazy path in `LinearProjection::apply`): scale [r/128, c/128] is
        // broadcast across each 128-element block. Elementwise per block, so
        // the stored [out, in] orientation needs no special handling.
        let w_bf16 = weight.to_kind(tch::Kind::BFloat16);
        let dequantized = match scale {
            Some(scale) => {
                let ws = w_bf16.size();
                let ss = scale.size();
                let block_r = ws[0] / ss[0];
                let block_c = ws[1] / ss[1];
                let w_4d = w_bf16.view([ss[0], block_r, ss[1], block_c]);
                let s_4d = scale.to_kind(tch::Kind::BFloat16).view([ss[0], 1, ss[1], 1]);
                (w_4d * s_4d).reshape([ws[0], ws[1]])
            }
            None => w_bf16,
        };
        weights.insert(key, dequantized);
        converted += 1;
    }
    converted
}

/// Run load-time FP8 dequantization when requested — but never for a
/// multi-device pipeline: every checkpoint tensor is loaded onto the pool's
/// *primary* device first, so materializing the whole model as BF16 there
/// (before `stage_from_weights_with_config` spreads layers across the pool)
/// would peak at full-BF16 on one device and could OOM a model whose BF16
/// weights only fit across the pool. In that case the flag is ignored with a
/// warning and the FP8 lazy-dequant path is kept.
fn maybe_dequantize_fp8_weights_at_load(
    weights: &mut HashMap<String, Tensor>,
    fp8_dequant_load: bool,
    device_pool: Option<&DevicePool>,
) {
    if !fp8_dequant_load {
        return;
    }
    if device_pool.is_some_and(|p| !p.is_single()) {
        warn!(
            "FP8 dequant-at-load ignored for multi-device pipeline: all weights land on the \
             primary device before layer distribution, so full-model BF16 materialization \
             there could OOM; keeping FP8 lazy per-matmul dequant"
        );
        return;
    }
    let converted = dequantize_fp8_weights_at_load(weights);
    if converted > 0 {
        info!("Dequantized {} FP8 weight tensor(s) to BF16 at load", converted);
    }
}

/// Glob-fallback shard discovery, shared by the sync/async loader paths.
///
/// Loud by design (#315): a multi-shard model reached via glob (no index.json) is
/// a silent-degrade path — it warns, and under [`STRICT_LOADER_ENV`] it is a hard
/// error. A single globbed shard is unambiguous and passes quietly.
fn glob_shard_files(model_path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut shard_files = Vec::new();
    for entry in std::fs::read_dir(model_path)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            if (name.starts_with("model-") || name.starts_with("model.safetensors-"))
                && name.ends_with(".safetensors")
            {
                shard_files.push(entry.path());
            }
        }
    }
    shard_files.sort();

    if shard_files.len() > 1 {
        if strict_loader_enabled() {
            return Err(anyhow!(
                "{} multi-shard safetensors files found in {} but no \
                 model.safetensors.index.json manifest; refusing to reconstruct the \
                 shard set from a filename glob under {STRICT_LOADER_ENV}. \
                 Provide the index.json manifest.",
                shard_files.len(),
                model_path.display()
            ));
        }
        warn!(
            "⚠️ No model.safetensors.index.json in {}; reconstructing {} shards from a \
             filename glob (fragile). Set {STRICT_LOADER_ENV}=1 to require the manifest.",
            model_path.display(),
            shard_files.len()
        );
    }
    Ok(shard_files)
}

/// Factory for creating models with proper configuration management
pub struct ModelFactory;

/// Authoritative request for one contiguous Llama-family pipeline stage.
///
/// Stage position is derived from the range and the checkpoint's authoritative
/// `num_hidden_layers`: a range starting at zero owns the embedding, while a
/// range ending at `num_hidden_layers` owns the final norm and LM head. Keeping
/// those roles implicit prevents a caller from requesting an inconsistent
/// range/role combination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelStageRequest {
    pub layer_range: Range<usize>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

/// Resolved content of a weight shard after transparent LFS/XET pointer
/// resolution — the exact bytes the loader builds tensors from.
///
/// Large ordinary shards are returned as an open [`File`] so the KV-compat
/// base-weight digest can *stream* them (the on-disk bytes ARE the content —
/// already smudged, or never a pointer). A resolved pointer, or any small file,
/// is returned owned. This is the single shared resolution surface used by both
/// the loader ([`ModelFactory::load_file_with_pointer_detection`]) and the
/// KV-compat digest ([`ModelFactory::resolve_weight_for_digest`]) so the two
/// never disagree about what bytes a shard resolves to (#1277).
pub(crate) enum ResolvedWeight {
    /// Resolved bytes held in memory (a resolved LFS/XET pointer, or a small
    /// ordinary file).
    Owned(Vec<u8>),
    /// An open ordinary file ≥1 KiB; read or stream it directly.
    File(std::fs::File),
}

impl ModelFactory {
    /// Detect the dtype of a model by examining its tensors
    pub async fn detect_model_dtype(model_path: &Path) -> Result<DType> {
        // Check for single file first
        let single_file = model_path.join("model.safetensors");
        let file_to_check = if single_file.exists() {
            single_file
        } else {
            // Look for first shard file
            let shard_files = Self::find_shard_files(model_path)?;
            if shard_files.is_empty() {
                return Err(anyhow!("No model weights found in {}", model_path.display()));
            }
            shard_files[0].clone()
        };

        // Load just the metadata to check dtype
        let file_content = std::fs::read(&file_to_check)?;
        let tensors = safetensors::SafeTensors::deserialize(&file_content)?;

        // Check the first few tensors to determine predominant dtype.
        // FP8 models store quantized weights as FP8 but use BF16 for compute
        // (activations, KV cache, non-quantized modules). We detect FP8 separately
        // and return BF16 as the compute dtype.
        let mut f16_count = 0;
        let mut bf16_count = 0;
        let mut f32_count = 0;
        let mut fp8_count = 0;

        for (_, tensor) in tensors.tensors().into_iter().take(10) {
            match tensor.dtype() {
                safetensors::Dtype::F16 => f16_count += 1,
                safetensors::Dtype::BF16 => bf16_count += 1,
                safetensors::Dtype::F8_E4M3
                | safetensors::Dtype::F8_E5M2 => fp8_count += 1,
                safetensors::Dtype::F32 => f32_count += 1,
                _ => {},
            }
        }

        // FP8 models: weights are FP8 but compute dtype is BF16
        if fp8_count > 0 && fp8_count >= bf16_count && fp8_count >= f16_count {
            info!("Detected FP8 quantized model (storage: FP8, compute dtype: BF16)");
            return Ok(tch::Kind::BFloat16);
        }

        // Return the most common dtype
        if f16_count > bf16_count && f16_count > f32_count {
            info!("Detected F16 model");
            Ok(tch::Kind::Half)
        } else if bf16_count >= f16_count && bf16_count >= f32_count {
            info!("Detected BF16 model");
            Ok(tch::Kind::BFloat16)
        } else if f32_count > 0 {
            info!("Detected F32 model");
            Ok(tch::Kind::Float)
        } else {
            info!("Could not detect model dtype, defaulting to BF16");
            Ok(tch::Kind::BFloat16)
        }
    }

    /// Create a model from a directory containing weights and optionally config.json
    /// This is the ONLY way models should be created to ensure consistency
    ///
    /// # Multi-device pipeline (#314 wiring, #310 epic)
    ///
    /// Pass `device_pool = Some(pool)` with a pool of >1 device to build the model
    /// as a single multi-device pipeline stage: decoder layers are spread across
    /// the pool's devices via [`LayerDeviceMap::even_split`], and the model's
    /// `forward_layers` performs the lone cross-device copy at each stage boundary.
    /// `device` must be the pool's primary (first) device — non-layer weights
    /// (embeddings, final norm, lm_head) live there. When `device_pool` is `None`
    /// or holds a single device, construction is unchanged (whole model on
    /// `device`). At runtime this depends on #405 (from_weights device-placement
    /// fix) to actually place per-layer tensors on non-primary devices.
    #[instrument(name = "model_factory.create", skip(device, dtype, device_pool), fields(model_path = %model_path.display()))]
    pub async fn create(
        model_path: &Path,
        device: &Device,
        dtype: DType,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
        device_pool: Option<&DevicePool>,
        fp8_dequant_load: bool,
    ) -> Result<Box<dyn ModelOperations>> {
        info!("Loading model: {}", model_path.display());
        if let Some(mc) = max_context {
            info!("Using max_context override: {} tokens", mc);
        }
        if kv_quant_type != KVQuantType::None {
            info!("Using KV cache quantization: {:?}", kv_quant_type);
        }
        if fp8_dequant_load {
            info!("FP8 dequant-at-load enabled: FP8 weights will be materialized as BF16 during load");
        }
        if let Some(pool) = device_pool {
            if !pool.is_single() {
                info!(
                    "🔀 Multi-device pipeline: spreading layers across {} device(s) {:?} \
                     (primary {:?}); depends on #405 for correct per-layer placement",
                    pool.len(),
                    pool.devices(),
                    pool.primary(),
                );
            }
        }

        // Check if we have sharded files that need incremental loading
        let shard_files = Self::find_shard_files(model_path)?;

        if !shard_files.is_empty() && shard_files.len() > 1 {
            // Use incremental loading for large sharded models
            info!(
                "📦 Using incremental loading for {} shards",
                shard_files.len()
            );
            Self::create_incremental(model_path, device, dtype, shard_files, max_context, kv_quant_type, device_pool, fp8_dequant_load).await
        } else {
            // Standard loading for single files or small models
            let weights = Self::load_weights(model_path, device, dtype).await?;
            let config = ModelConfig::load(model_path, &weights)?;
            let model = Self::create_model_from_config(config, weights, device, dtype, max_context, kv_quant_type, model_path, device_pool, fp8_dequant_load)?;
            info!("✅ ModelFactory: Model created successfully");
            Ok(model)
        }
    }

    /// Create one Llama-family pipeline stage from an authoritative HF index.
    ///
    /// Unlike [`Self::create`], stage mode never falls back to a single-file or
    /// glob-discovered whole-model load. It requires
    /// `model.safetensors.index.json`, resolves the exact tensor names for the
    /// requested global layer range and its first/last role, opens only shards
    /// containing those names, and constructs no tensor outside that set.
    ///
    /// Llama, Qwen, and Mistral checkpoints use the production-wired
    /// `LlamaModel` stage path. Qwen3.5 remains deliberately out of scope.
    #[instrument(
        name = "model_factory.create_stage",
        skip(device, dtype, device_pool),
        fields(
            model_path = %model_path.display(),
            layer_start = request.layer_range.start,
            layer_end = request.layer_range.end,
        )
    )]
    pub async fn create_stage(
        model_path: &Path,
        device: &Device,
        dtype: DType,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
        device_pool: Option<&DevicePool>,
        request: ModelStageRequest,
        fp8_dequant_load: bool,
    ) -> Result<Box<dyn ModelOperations>> {
        let config_path = model_path.join("config.json");
        if !config_path.is_file() {
            bail!(
                "stage loading requires authoritative config {}",
                config_path.display()
            );
        }
        let empty_weights = HashMap::new();
        let config = ModelConfig::load(model_path, &empty_weights)?;
        if !matches!(
            &config.architecture,
            ModelArchitecture::Llama | ModelArchitecture::Qwen | ModelArchitecture::Mistral
        ) {
            bail!(
                "stage loading is supported only for Llama-family checkpoints; got {:?}",
                config.architecture
            );
        }

        Self::validate_stage_range(&request.layer_range, config.num_hidden_layers)?;
        let plan = Self::stage_weight_plan(
            model_path,
            config.num_hidden_layers,
            request.layer_range.clone(),
        )?;
        let metadata = Self::stage_tensor_metadata(&plan).await?;
        Self::validate_stage_tensor_schema(&config, &request.layer_range, &metadata)?;
        let mut weights =
            Self::load_weights_for_stage_plan(plan, request.layer_range.clone(), device, dtype)
                .await?;
        maybe_dequantize_fp8_weights_at_load(&mut weights, fp8_dequant_load, device_pool);

        Self::create_llama_model(
            config,
            weights,
            device,
            dtype,
            max_context,
            kv_quant_type,
            device_pool,
            Some(request.layer_range),
        )
    }

    /// Find all shard files in a model directory.
    ///
    /// Prefers `model.safetensors.index.json` (authoritative HuggingFace shard manifest)
    /// over filename glob patterns, which are fragile across model families.
    ///
    /// This is THE loader's shard selector — also used by the KV-compat
    /// base-weight digest (#1277) so the fingerprint covers exactly the
    /// loader-selected set, never extra/unreferenced `.safetensors` files.
    pub(crate) fn find_shard_files(model_path: &Path) -> Result<Vec<std::path::PathBuf>> {
        // 1. Use index file if present (most reliable)
        let index_path = model_path.join("model.safetensors.index.json");
        if index_path.exists() {
            return Self::shard_files_from_index(model_path, &index_path);
        }

        // 2. Single unsharded file
        let single_file = model_path.join("model.safetensors");
        if single_file.exists() {
            return Ok(vec![single_file]);
        }

        // 3. Fallback: glob for known shard naming patterns (loud / strict-gated)
        glob_shard_files(model_path)
    }

    /// Shard selection for the KV-compat base-weight digest (#1277).
    ///
    /// Fingerprint authority is stricter than loader permissiveness: the
    /// loader may warn-and-continue on a manifest-less multi-shard glob
    /// (unless [`STRICT_LOADER_ENV`]), but the digest must NEVER mint an
    /// authoritative identity over that ambiguous set — a glob can silently
    /// drop/duplicate shards, so two snapshots could share a fingerprint
    /// while loading different weights. Such a set is rejected here
    /// regardless of `HYPRSTREAM_STRICT_LOADER`, and the caller declines KV
    /// reuse. A single globbed shard remains unambiguous and passes.
    pub(crate) fn find_shard_files_for_digest(
        model_path: &Path,
    ) -> Result<Vec<std::path::PathBuf>> {
        let shards = Self::find_shard_files(model_path)?;
        if shards.len() > 1 && !model_path.join("model.safetensors.index.json").exists() {
            return Err(anyhow!(
                "{} loader-selected .safetensors shards in {} but no \
                 model.safetensors.index.json manifest; the glob-selected set is \
                 ambiguous, so no authoritative base-weight digest can be minted \
                 (KV reuse is declined). Provide the index.json manifest.",
                shards.len(),
                model_path.display()
            ));
        }
        Ok(shards)
    }

    /// Parse shard filenames from `model.safetensors.index.json`.
    /// Returns unique shard paths in sorted order (guaranteed by the index).
    fn shard_files_from_index(
        model_path: &Path,
        index_path: &Path,
    ) -> Result<Vec<std::path::PathBuf>> {
        let content = std::fs::read_to_string(index_path)?;
        let index: serde_json::Value = serde_json::from_str(&content)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("model.safetensors.index.json missing weight_map"))?;

        let mut seen = std::collections::BTreeSet::new();
        for filename in weight_map.values() {
            if let Some(s) = filename.as_str() {
                seen.insert(s.to_owned());
            }
        }

        let mut paths: Vec<_> = seen
            .into_iter()
            .map(|name| model_path.join(&name))
            .collect();
        paths.sort();

        info!(
            "📋 Index file lists {} shard(s): {:?}",
            paths.len(),
            paths.iter().map(|p| p.file_name().unwrap_or_default()).collect::<Vec<_>>()
        );
        Ok(paths)
    }

    fn validate_stage_range(layer_range: &Range<usize>, num_layers: usize) -> Result<()> {
        if layer_range.start >= layer_range.end || layer_range.end > num_layers {
            bail!(
                "invalid stage layer range {:?} for checkpoint with {} layers",
                layer_range,
                num_layers
            );
        }
        Ok(())
    }

    fn llama_layer_index(name: &str) -> Option<usize> {
        let rest = name.strip_prefix("model.layers.")?;
        let (index, suffix) = rest.split_once('.')?;
        if suffix.is_empty() {
            return None;
        }
        index.parse().ok()
    }

    fn resolve_required_alias(
        weight_map: &BTreeMap<String, String>,
        aliases: &[&str],
        description: &str,
    ) -> Result<String> {
        aliases
            .iter()
            .find(|name| weight_map.contains_key(**name))
            .map(|name| (*name).to_owned())
            .ok_or_else(|| {
                anyhow!(
                    "stage weight_map is missing required {description}; expected one of {}",
                    aliases.join(", ")
                )
            })
    }

    /// Add the block-wise `<name>_scale_inv` companion of a top-level stage
    /// tensor when the manifest has one. Layer-prefixed FP8 scales are
    /// selected automatically by the layer-index filter, but top-level tensors
    /// (embedding, lm_head) enter the plan by exact name — without this the
    /// stage loads a scale-less FP8 tensor and `dequantize_fp8_weights_at_load`
    /// casts raw FP8 codes straight to BF16 (#1519 review). No-op for
    /// unquantized checkpoints, which have no such key.
    fn insert_companion_scale(
        weight_map: &BTreeMap<String, String>,
        required: &mut BTreeSet<String>,
        name: &str,
    ) {
        let scale = format!("{name}_scale_inv");
        if weight_map.contains_key(&scale) {
            required.insert(scale);
        }
    }

    /// Resolve one stage to exact manifest tensor names grouped by shard.
    ///
    /// The returned map is the complete I/O plan: callers must not open any
    /// other shard or enumerate any other tensor while constructing weights.
    fn stage_weight_plan(
        model_path: &Path,
        num_layers: usize,
        layer_range: Range<usize>,
    ) -> Result<BTreeMap<PathBuf, BTreeSet<String>>> {
        Self::validate_stage_range(&layer_range, num_layers)?;

        let index_path = model_path.join("model.safetensors.index.json");
        let content = std::fs::read_to_string(&index_path).with_context(|| {
            format!(
                "stage loading requires authoritative shard index {}",
                index_path.display()
            )
        })?;
        let index: SafetensorsIndex = serde_json::from_str(&content).with_context(|| {
            format!(
                "stage loading requires a valid `weight_map` object in {}",
                index_path.display()
            )
        })?;
        if index.weight_map.is_empty() {
            bail!(
                "stage loading requires a non-empty `weight_map` in {}",
                index_path.display()
            );
        }
        if let Some((name, _)) = index
            .weight_map
            .iter()
            .find(|(name, shard)| name.is_empty() || shard.is_empty())
        {
            bail!(
                "stage weight_map in {} contains an empty tensor name or shard for `{name}`",
                index_path.display()
            );
        }

        let manifest_layers: BTreeSet<_> = index
            .weight_map
            .keys()
            .filter_map(|name| Self::llama_layer_index(name))
            .collect();
        let expected_layers: BTreeSet<_> = (0..num_layers).collect();
        if manifest_layers != expected_layers {
            bail!(
                "stage weight_map decoder layers {:?} do not exactly match expected range 0..{}",
                manifest_layers,
                num_layers
            );
        }

        let mut required = BTreeSet::new();
        for layer in layer_range.clone() {
            let layer_names: Vec<_> = index
                .weight_map
                .keys()
                .filter(|name| Self::llama_layer_index(name) == Some(layer))
                .cloned()
                .collect();
            if layer_names.is_empty() {
                bail!("stage weight_map has no tensors for required decoder layer {layer}");
            }
            required.extend(layer_names);
        }

        let is_first = layer_range.start == 0;
        let is_last = layer_range.end == num_layers;
        if is_first {
            let embed = Self::resolve_required_alias(
                &index.weight_map,
                &["model.embed_tokens.weight", "embed_tokens.weight"],
                "embedding tensor",
            )?;
            Self::insert_companion_scale(&index.weight_map, &mut required, &embed);
            required.insert(embed);
        }
        if is_last {
            required.insert(Self::resolve_required_alias(
                &index.weight_map,
                &["model.norm.weight", "norm.weight"],
                "final norm tensor",
            )?);

            if let Some(lm_head) = ["lm_head.weight", "model.lm_head.weight"]
                .into_iter()
                .find(|name| index.weight_map.contains_key(*name))
            {
                Self::insert_companion_scale(&index.weight_map, &mut required, lm_head);
                required.insert(lm_head.to_owned());
            } else if !is_first {
                bail!("a non-first final stage requires an explicit lm_head tensor in weight_map");
            }
        }

        let mut plan: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
        for name in required {
            let shard = index.weight_map.get(&name).ok_or_else(|| {
                anyhow!("required stage tensor `{name}` disappeared from weight_map")
            })?;
            plan.entry(model_path.join(shard)).or_default().insert(name);
        }

        Ok(plan)
    }

    /// Read only selected safetensors headers before allocating any `tch` tensor.
    ///
    /// This preflight deliberately reuses the authoritative I/O plan. It opens no
    /// shard outside that plan and calls neither `TensorView::data()` nor a
    /// device-placement API, so a schema mismatch fails before model memory is
    /// materialized.
    async fn stage_tensor_metadata(
        plan: &BTreeMap<PathBuf, BTreeSet<String>>,
    ) -> Result<BTreeMap<String, Vec<usize>>> {
        use memmap2::Mmap;

        let mut metadata = BTreeMap::new();
        for (shard, names) in plan {
            match Self::resolve_weight_for_digest(shard).await? {
                ResolvedWeight::File(file) => {
                    let mmap = unsafe { Mmap::map(&file) }.with_context(|| {
                        format!("failed to mmap required stage shard {}", shard.display())
                    })?;
                    let tensors = safetensors::SafeTensors::deserialize(&mmap).with_context(|| {
                        format!("failed to parse required stage shard {}", shard.display())
                    })?;
                    Self::record_stage_tensor_metadata(tensors, names, shard, &mut metadata)?;
                }
                ResolvedWeight::Owned(bytes) => {
                    let tensors = safetensors::SafeTensors::deserialize(&bytes).with_context(|| {
                        format!("failed to parse required stage shard {}", shard.display())
                    })?;
                    Self::record_stage_tensor_metadata(tensors, names, shard, &mut metadata)?;
                }
            }
        }
        Ok(metadata)
    }

    fn record_stage_tensor_metadata(
        tensors: safetensors::SafeTensors<'_>,
        names: &BTreeSet<String>,
        shard: &Path,
        metadata: &mut BTreeMap<String, Vec<usize>>,
    ) -> Result<()> {
        for name in names {
            let view = tensors.tensor(name).with_context(|| {
                format!(
                    "authoritative weight_map assigns `{name}` to {}, but the tensor is absent",
                    shard.display()
                )
            })?;
            if metadata.insert(name.clone(), view.shape().to_vec()).is_some() {
                bail!("authoritative stage plan contains duplicate tensor `{name}`");
            }
        }
        Ok(())
    }

    fn stage_metadata_alias<'a>(
        metadata: &'a BTreeMap<String, Vec<usize>>,
        aliases: &[&str],
        description: &str,
    ) -> Result<(String, &'a [usize])> {
        aliases
            .iter()
            .find_map(|name| {
                metadata
                    .get(*name)
                    .map(|shape| ((*name).to_owned(), shape.as_slice()))
            })
            .ok_or_else(|| {
                anyhow!(
                    "selected stage metadata is missing required {description}; expected one of {}",
                    aliases.join(", ")
                )
            })
    }

    fn require_stage_shape(
        metadata: &BTreeMap<String, Vec<usize>>,
        name: &str,
        expected: &[usize],
    ) -> Result<()> {
        let actual = metadata.get(name).ok_or_else(|| {
            anyhow!("selected stage metadata is missing required tensor `{name}`")
        })?;
        if actual.as_slice() != expected {
            bail!(
                "stage tensor `{name}` shape mismatch: config requires {:?}, checkpoint has {:?}",
                expected,
                actual
            );
        }
        Ok(())
    }

    fn require_optional_stage_shape(
        metadata: &BTreeMap<String, Vec<usize>>,
        name: &str,
        expected: &[usize],
    ) -> Result<()> {
        if metadata.contains_key(name) {
            Self::require_stage_shape(metadata, name, expected)?;
        }
        Ok(())
    }

    /// Fail closed when authoritative config dimensions disagree with selected
    /// tensor headers. This is stricter than the legacy whole-model loader,
    /// which historically reconciles embedding dimensions after loading: stage
    /// mode must prove the checkpoint/config pair is coherent before allocating
    /// a requested subset.
    fn validate_stage_tensor_schema(
        config: &ModelConfig,
        layer_range: &Range<usize>,
        metadata: &BTreeMap<String, Vec<usize>>,
    ) -> Result<()> {
        Self::validate_stage_range(layer_range, config.num_hidden_layers)?;

        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let query_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| anyhow!("query projection width overflows usize"))?;
        let kv_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| anyhow!("key/value projection width overflows usize"))?;
        if hidden == 0 || intermediate == 0 || query_width == 0 || kv_width == 0 {
            bail!(
                "stage config contains an invalid zero dimension: hidden={hidden}, \
                 intermediate={intermediate}, query_width={query_width}, kv_width={kv_width}"
            );
        }

        let is_first = layer_range.start == 0;
        let is_last = layer_range.end == config.num_hidden_layers;
        if is_first {
            let (name, shape) = Self::stage_metadata_alias(
                metadata,
                &["model.embed_tokens.weight", "embed_tokens.weight"],
                "embedding tensor",
            )?;
            let expected = [config.vocab_size, hidden];
            if shape != expected {
                bail!(
                    "stage tensor `{name}` shape mismatch: config requires {:?}, checkpoint has {:?}",
                    expected,
                    shape
                );
            }
        }

        if is_last {
            let (norm_name, norm_shape) = Self::stage_metadata_alias(
                metadata,
                &["model.norm.weight", "norm.weight"],
                "final norm tensor",
            )?;
            let expected_norm = [hidden];
            if norm_shape != expected_norm {
                bail!(
                    "stage tensor `{norm_name}` shape mismatch: config requires {:?}, checkpoint has {:?}",
                    expected_norm,
                    norm_shape
                );
            }

            if let Some((head_name, head_shape)) = ["lm_head.weight", "model.lm_head.weight"]
                .into_iter()
                .find_map(|name| metadata.get(name).map(|shape| (name, shape.as_slice())))
            {
                let expected_head = [config.vocab_size, hidden];
                if head_shape != expected_head {
                    bail!(
                        "stage tensor `{head_name}` shape mismatch: config requires {:?}, \
                         checkpoint has {:?}",
                        expected_head,
                        head_shape
                    );
                }
            } else if !is_first {
                bail!("a non-first final stage requires an explicit lm_head tensor");
            }
            // A complete-range stage without an explicit head ties the already
            // validated embedding, so its [vocab, hidden] contract is identical.
        }

        for layer in layer_range.clone() {
            let prefix = format!("model.layers.{layer}");
            let q_proj = format!("{prefix}.self_attn.q_proj.weight");
            let c_attn = format!("{prefix}.self_attn.c_attn.weight");
            if metadata.contains_key(&q_proj) {
                let k_proj = format!("{prefix}.self_attn.k_proj.weight");
                let v_proj = format!("{prefix}.self_attn.v_proj.weight");
                Self::require_stage_shape(metadata, &q_proj, &[query_width, hidden])?;
                Self::require_stage_shape(metadata, &k_proj, &[kv_width, hidden])?;
                Self::require_stage_shape(metadata, &v_proj, &[kv_width, hidden])?;
                Self::require_optional_stage_shape(
                    metadata,
                    &format!("{prefix}.self_attn.q_proj.bias"),
                    &[query_width],
                )?;
                Self::require_optional_stage_shape(
                    metadata,
                    &format!("{prefix}.self_attn.k_proj.bias"),
                    &[kv_width],
                )?;
                Self::require_optional_stage_shape(
                    metadata,
                    &format!("{prefix}.self_attn.v_proj.bias"),
                    &[kv_width],
                )?;
            } else if metadata.contains_key(&c_attn) {
                if config.num_key_value_heads != config.num_attention_heads {
                    bail!(
                        "stage layer {layer} uses combined c_attn with grouped-query config \
                         (attention heads {}, key/value heads {}), which the dense Llama \
                         constructor cannot represent",
                        config.num_attention_heads,
                        config.num_key_value_heads
                    );
                }
                let combined_width = query_width
                    .checked_mul(3)
                    .ok_or_else(|| anyhow!("combined attention width overflows usize"))?;
                Self::require_stage_shape(metadata, &c_attn, &[combined_width, hidden])?;
                Self::require_optional_stage_shape(
                    metadata,
                    &format!("{prefix}.self_attn.c_attn.bias"),
                    &[combined_width],
                )?;
            } else {
                bail!(
                    "selected stage metadata is missing required attention projection for layer \
                     {layer}; expected `{q_proj}` or `{c_attn}`"
                );
            }

            let o_proj = format!("{prefix}.self_attn.o_proj.weight");
            let c_proj = format!("{prefix}.self_attn.c_proj.weight");
            let output_name = if metadata.contains_key(&o_proj) {
                o_proj.as_str()
            } else {
                c_proj.as_str()
            };
            Self::require_stage_shape(metadata, output_name, &[hidden, query_width])?;
            let output_bias = output_name
                .strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.bias"))
                .unwrap_or_else(|| format!("{output_name}.bias"));
            Self::require_optional_stage_shape(metadata, &output_bias, &[hidden])?;

            for (suffix, expected) in [
                ("mlp.gate_proj.weight", [intermediate, hidden]),
                ("mlp.up_proj.weight", [intermediate, hidden]),
                ("mlp.down_proj.weight", [hidden, intermediate]),
            ] {
                Self::require_stage_shape(metadata, &format!("{prefix}.{suffix}"), &expected)?;
            }
            for (suffix, expected) in [
                ("mlp.gate_proj.bias", [intermediate]),
                ("mlp.up_proj.bias", [intermediate]),
                ("mlp.down_proj.bias", [hidden]),
            ] {
                Self::require_optional_stage_shape(
                    metadata,
                    &format!("{prefix}.{suffix}"),
                    &expected,
                )?;
            }
            Self::require_stage_shape(
                metadata,
                &format!("{prefix}.input_layernorm.weight"),
                &[hidden],
            )?;
            Self::require_stage_shape(
                metadata,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?;

            let q_norm = format!("{prefix}.self_attn.q_norm.weight");
            let k_norm = format!("{prefix}.self_attn.k_norm.weight");
            if config.use_qk_norm
                || metadata.contains_key(&q_norm)
                || metadata.contains_key(&k_norm)
            {
                Self::require_stage_shape(metadata, &q_norm, &[config.head_dim])?;
                Self::require_stage_shape(metadata, &k_norm, &[config.head_dim])?;
            }
        }

        Ok(())
    }

    #[cfg(test)]
    async fn load_weights_for_stage(
        model_path: &Path,
        num_layers: usize,
        layer_range: Range<usize>,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        let plan = Self::stage_weight_plan(model_path, num_layers, layer_range.clone())?;
        Self::load_weights_for_stage_plan(plan, layer_range, device, dtype).await
    }

    async fn load_weights_for_stage_plan(
        plan: BTreeMap<PathBuf, BTreeSet<String>>,
        layer_range: Range<usize>,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        let planned_tensors: usize = plan.values().map(BTreeSet::len).sum();
        info!(
            "Loading stage {:?}: {} exact tensor(s) from {} shard(s)",
            layer_range,
            planned_tensors,
            plan.len()
        );

        let mut weights = HashMap::with_capacity(planned_tensors);
        for (shard, names) in plan {
            Self::load_safetensors_file_selected(&shard, &names, &mut weights, device, dtype)
                .await?;
        }

        if weights.len() != planned_tensors {
            bail!(
                "stage loader planned {planned_tensors} tensors but materialized {}",
                weights.len()
            );
        }
        let loaded_bytes: usize = weights
            .values()
            .map(|tensor| tensor.numel() * tensor.kind().elt_size_in_bytes())
            .sum();
        info!(
            "Loaded authoritative stage {:?}: {} tensor(s), {} byte(s)",
            layer_range,
            weights.len(),
            loaded_bytes
        );
        Ok(weights)
    }

    /// Create model using incremental loading for large sharded models
    #[instrument(name = "model_factory.create_incremental", skip(device, dtype, shard_files, device_pool), fields(shard_count = shard_files.len()))]
    async fn create_incremental(
        model_path: &Path,
        device: &Device,
        dtype: DType,
        shard_files: Vec<std::path::PathBuf>,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
        device_pool: Option<&DevicePool>,
        fp8_dequant_load: bool,
    ) -> Result<Box<dyn ModelOperations>> {
        // For now, we still need to load all weights, but we do it more efficiently
        // by processing shards sequentially and immediately transferring to GPU
        info!("Loading {} weight shards", shard_files.len());

        let mut all_weights = HashMap::new();

        for (idx, shard_file) in shard_files.iter().enumerate() {
            info!("Loading shard {}/{}", idx + 1, shard_files.len());

            // Load shard weights directly to GPU to minimize CPU memory usage
            Self::load_safetensors_file(shard_file, &mut all_weights, device, dtype).await?;

            // Note: In a true streaming implementation, we would:
            // 1. Load layer weights
            // 2. Create that layer on GPU
            // 3. Free CPU memory before loading next layer
            // But this requires refactoring model architectures
        }

        // Load config and create model
        let config = ModelConfig::load(model_path, &all_weights)?;
        let model = Self::create_model_from_config(config, all_weights, device, dtype, max_context, kv_quant_type, model_path, device_pool, fp8_dequant_load)?;

        info!("Model loaded");
        Ok(model)
    }

    /// Load weights from safetensors files
    async fn load_weights(
        model_path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        let mut all_weights = HashMap::new();

        let single_file = model_path.join("model.safetensors");
        if single_file.exists() {
            info!("Loading model.safetensors");
            Self::load_safetensors_file(&single_file, &mut all_weights, device, dtype).await?;
            return Ok(all_weights);
        }

        // Look for sharded safetensors files — prefer index file, then glob patterns
        let model_path_buf = model_path.to_path_buf();
        let mut shard_files =
            tokio::task::spawn_blocking(move || -> Result<Vec<std::path::PathBuf>> {
                // Use index if present
                let index_path = model_path_buf.join("model.safetensors.index.json");
                if index_path.exists() {
                    return Self::shard_files_from_index(&model_path_buf, &index_path);
                }
                // Fallback glob (loud / strict-gated)
                glob_shard_files(&model_path_buf)
            })
            .await??;

        if !shard_files.is_empty() {
            shard_files.sort();
            info!("Loading {} weight shards", shard_files.len());
            for shard_file in shard_files {
                Self::load_safetensors_file(&shard_file, &mut all_weights, device, dtype).await?;
            }
            return Ok(all_weights);
        }

        Err(anyhow!(
            "No safetensors files found in {}",
            model_path.display()
        ))
    }

    /// Load a single safetensors file
    #[instrument(name = "model_factory.load_safetensor_file", skip(weights, device, dtype), fields(file = %path.display()))]
    async fn load_safetensors_file(
        path: &Path,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        Self::load_safetensors_file_inner(path, None, weights, device, dtype).await
    }

    /// Load only `selected` tensor names from one safetensors shard.
    ///
    /// Selection happens before `TensorView::data()` and before any `tch::Tensor`
    /// construction, so an unselected tensor is never materialized even when it
    /// shares a physical shard with selected tensors.
    async fn load_safetensors_file_selected(
        path: &Path,
        selected: &BTreeSet<String>,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        Self::load_safetensors_file_inner(path, Some(selected), weights, device, dtype).await
    }

    async fn load_safetensors_file_inner(
        path: &Path,
        selected: Option<&BTreeSet<String>>,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        // Stage selection always uses mmap: only selected tensor pages are copied
        // into owned CPU/device storage, so unrelated bytes in a shared shard do
        // not become a whole-shard heap allocation. Whole-model loading retains
        // its existing environment-controlled behavior.
        //
        // #315 note: for whole-model mode this env var is independent of
        // `RuntimeConfig.mmap`, which today gates nothing here.
        let use_mmap = selected.is_some()
            || std::env::var("HYPRSTREAM_USE_MMAP")
                .map(|v| v.to_lowercase() == "true" || v == "1")
                .unwrap_or(false);
        debug!(
            "safetensors load path: {}",
            if use_mmap { "mmap" } else { "read" }
        );

        // Load file data in a blocking task to avoid blocking the async runtime
        let path_buf = path.to_path_buf();

        // Resolve before deciding whether mmap is safe. A stage selects only a
        // few tensors, but a pointer shard must be resolved before safetensors
        // sees it; mmap is allowed only for an already-resolved ordinary file.
        match Self::resolve_weight_for_digest(&path_buf).await? {
            ResolvedWeight::File(file) if use_mmap => {
                use memmap2::Mmap;

                let mmap = unsafe { Mmap::map(&file)? };
                let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
                Self::create_tensors_from_safetensors_selected(
                    tensors, selected, weights, device, dtype,
                )
            }
            ResolvedWeight::File(_) => {
                let tensor_data = Self::load_file_with_pointer_detection(&path_buf).await?;
                let tensors = safetensors::SafeTensors::deserialize(&tensor_data)?;
                Self::create_tensors_from_safetensors_selected(
                    tensors, selected, weights, device, dtype,
                )
            }
            ResolvedWeight::Owned(tensor_data) => {
                let tensors = safetensors::SafeTensors::deserialize(&tensor_data)?;
                Self::create_tensors_from_safetensors_selected(
                    tensors, selected, weights, device, dtype,
                )
            }
        }
    }

    /// Load file with automatic LFS/XET pointer detection
    ///
    /// Fast path for already-smudged files, fallback for un-smudged pointers.
    /// Resolution is shared with the KV-compat base-weight digest via
    /// [`Self::try_resolve_lfs_pointer`] so both see the identical bytes (#1277).
    async fn load_file_with_pointer_detection(path: &Path) -> Result<Vec<u8>> {
        let metadata = tokio::fs::metadata(path).await?;

        // Large files cannot be LFS pointers (which are < 1KB)
        if metadata.len() >= 1024 {
            return tokio::fs::read(path).await.map_err(Into::into);
        }

        let data = tokio::fs::read(path).await?;
        if let Some(resolved) = Self::try_resolve_lfs_pointer(path, &data).await? {
            return Ok(resolved);
        }
        Ok(data)
    }

    /// If `data` is an un-smudged LFS/XET pointer, resolve it to the actual
    /// object content via the git2db LFS fallback. Returns `Ok(None)` when
    /// `data` is ordinary content (no pointer header, or too large to be one).
    /// Any resolution failure propagates as an error — **fail-closed**: the
    /// caller (loader *and* the KV-compat digest) never silently hashes or
    /// deserializes a pointer stub in place of the resolved bytes.
    ///
    /// Single source of truth for pointer resolution: extracted from the old
    /// inline block in `load_file_with_pointer_detection`.
    async fn try_resolve_lfs_pointer(path: &Path, data: &[u8]) -> Result<Option<Vec<u8>>> {
        // Large buffers cannot be LFS pointers (which are < 1 KiB).
        if data.len() >= 1024 {
            return Ok(None);
        }
        let text = match std::str::from_utf8(data) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        if !text.starts_with("version https://git-lfs")
            && !text.starts_with("version https://hawser")
        {
            return Ok(None);
        }
        #[cfg(feature = "xet")]
        {
            debug!("Un-smudged LFS pointer, resolving via git2db::lfs fallback: {}", path.display());
            let config = git2db::XetConfig::default();
            let storage = git2db::LfsStorage::new(&config)
                .await
                .map_err(|e| anyhow!("Failed to create LfsStorage: {}", e))?;
            let bytes = storage
                .load_file(path)
                .await
                .map_err(|e| anyhow!("Failed to load LFS file: {}", e))?;
            Ok(Some(bytes))
        }
        #[cfg(not(feature = "xet"))]
        {
            let _ = path;
            anyhow::bail!(
                "Un-smudged LFS pointer at {} but XET feature disabled. \
                 Enable with --features xet or ensure files are smudged during checkout.",
                path.display()
            )
        }
    }

    /// Resolve a weight shard for the KV-compat base-weight digest (#1277).
    ///
    /// Returns the shard's resolved content: a large ordinary shard as an open
    /// [`File`] (to stream), and a resolved pointer or any small file as owned
    /// bytes. This walks the **same** resolution path as the loader
    /// ([`Self::load_file_with_pointer_detection`] → [`Self::try_resolve_lfs_pointer`]),
    /// so the digest hashes the actual tensor bytes — never a pointer stub — and
    /// fails closed (returns `Err`) on any metadata/read/resolve error.
    pub(crate) async fn resolve_weight_for_digest(path: &Path) -> Result<ResolvedWeight> {
        let metadata = tokio::fs::metadata(path).await?;
        // A large file cannot be an LFS pointer (pointers are < 1 KiB): its
        // on-disk bytes are the resolved content — stream it, don't buffer it.
        if metadata.len() >= 1024 {
            let file = std::fs::File::open(path)?;
            return Ok(ResolvedWeight::File(file));
        }
        let data = tokio::fs::read(path).await?;
        if let Some(resolved) = Self::try_resolve_lfs_pointer(path, &data).await? {
            return Ok(ResolvedWeight::Owned(resolved));
        }
        Ok(ResolvedWeight::Owned(data))
    }

    /// Create tensors from deserialized safetensors
    fn create_tensors_from_safetensors(
        tensors: safetensors::SafeTensors,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        Self::create_tensors_from_safetensors_selected(tensors, None, weights, device, dtype)
    }

    fn create_tensors_from_safetensors_selected(
        tensors: safetensors::SafeTensors,
        selected: Option<&BTreeSet<String>>,
        weights: &mut HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Result<()> {
        let mut total_size_mb = 0.0;
        let tensors_list = if let Some(selected) = selected {
            selected
                .iter()
                .map(|name| {
                    tensors
                        .tensor(name)
                        .map(|view| (name.clone(), view))
                        .with_context(|| {
                            format!(
                                "authoritative weight_map assigns `{name}` to this shard, \
                                 but the tensor is absent"
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            tensors.tensors()
        };
        let tensor_count = tensors_list.len();

        info!(
            "Loading {} tensors to device {:?}",
            tensor_count,
            device
        );

        for (idx, (name, tensor_view)) in tensors_list.into_iter().enumerate() {
            let shape: Vec<i64> = tensor_view.shape().iter().map(|&x| x as i64).collect();
            let data = tensor_view.data();

            // Map safetensors dtype to tch::Kind for size estimation
            let tensor_kind = match tensor_view.dtype() {
                safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                safetensors::Dtype::F16 => tch::Kind::Half,
                safetensors::Dtype::F64 => tch::Kind::Double,
                safetensors::Dtype::F8_E4M3 => tch::Kind::Float8e4m3fn,
                safetensors::Dtype::F8_E5M2 => tch::Kind::Float8e5m2,
                // F32 and other types default to Float
                _ => tch::Kind::Float,
            };

            // Calculate tensor size for progress reporting
            let tensor_size_mb = estimate_tensor_size_mb(&shape, tensor_kind);
            total_size_mb += tensor_size_mb;

            // Support both F16 and BF16 models
            let tensor = match tensor_view.dtype() {
                safetensors::Dtype::BF16 => {
                    // Verify target dtype matches
                    if dtype != tch::Kind::BFloat16 && dtype != tch::Kind::Half {
                        return Err(anyhow!(
                            "Model dtype BF16 but target dtype is {:?}",
                            dtype
                        ));
                    }

                    // Create tensor from borrowed data, then make an owned copy
                    // IMPORTANT: from_blob only borrows the data pointer, so we must
                    // copy the tensor to own the data before the source buffer is freed
                    let borrowed_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::BFloat16,
                            Device::Cpu,
                        )
                    };

                    // Make an owned copy to prevent use-after-free
                    let cpu_tensor = borrowed_tensor.copy();

                    // Convert dtype if needed
                    let cpu_tensor = if dtype == tch::Kind::Half {
                        cpu_tensor.to_kind(tch::Kind::Half)
                    } else {
                        cpu_tensor
                    };

                    // Transfer to target device if needed (with OOM handling)
                    if *device != Device::Cpu {
                        let gpu_tensor = cpu_tensor.to_device(*device);
                        drop(cpu_tensor); // Explicitly free CPU memory
                        gpu_tensor
                    } else {
                        cpu_tensor
                    }
                }
                safetensors::Dtype::F16 => {
                    // Verify target dtype matches
                    if dtype != tch::Kind::Half && dtype != tch::Kind::BFloat16 {
                        return Err(anyhow!(
                            "Model dtype F16 but target dtype is {:?}",
                            dtype
                        ));
                    }

                    // Create tensor from borrowed data
                    let borrowed_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::Half,
                            Device::Cpu,
                        )
                    };

                    // Make an owned copy to prevent use-after-free
                    let cpu_tensor = borrowed_tensor.copy();

                    // Convert dtype if needed
                    let cpu_tensor = if dtype == tch::Kind::BFloat16 {
                        cpu_tensor.to_kind(tch::Kind::BFloat16)
                    } else {
                        cpu_tensor
                    };

                    // Transfer to target device if needed (with OOM handling)
                    if *device != Device::Cpu {
                        let gpu_tensor = cpu_tensor.to_device(*device);
                        drop(cpu_tensor); // Explicitly free CPU memory
                        gpu_tensor
                    } else {
                        cpu_tensor
                    }
                }
                safetensors::Dtype::F32 => {
                    // Support F32 as well for completeness
                    let borrowed_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::Float,
                            Device::Cpu,
                        )
                    };

                    // Make an owned copy
                    let cpu_tensor = borrowed_tensor.copy();

                    // Convert to target dtype
                    let cpu_tensor = match dtype {
                        tch::Kind::Half => cpu_tensor.to_kind(tch::Kind::Half),
                        tch::Kind::BFloat16 => cpu_tensor.to_kind(tch::Kind::BFloat16),
                        tch::Kind::Float => cpu_tensor,
                        _ => {
                            return Err(anyhow!(
                                "Cannot convert F32 to target dtype {:?}",
                                dtype
                            ))
                        }
                    };

                    // Transfer to target device if needed (with OOM handling)
                    if *device != Device::Cpu {
                        // Log progress for large models
                        if idx % 100 == 0 || tensor_size_mb > 10.0 {
                            info!(
                                "Transferring tensor {}/{} to GPU: {} ({:.2} MB, total: {:.2} MB)",
                                idx + 1, tensor_count, name, tensor_size_mb, total_size_mb
                            );
                        }

                        // Use safe wrapper to catch OOM panics
                        match safe_to_device(&cpu_tensor, *device) {
                            Ok(gpu_tensor) => {
                                drop(cpu_tensor); // Explicitly free CPU memory
                                gpu_tensor
                            }
                            Err(e) => {
                                drop(cpu_tensor); // CRITICAL: Free CPU memory before returning error
                                return Err(anyhow!(
                                    "GPU OOM loading tensor '{}': {} | Progress: {}/{} ({:.1} MB) | Try: smaller model, reduce max_position_embeddings, or free GPU memory",
                                    name, e, idx + 1, tensor_count, total_size_mb
                                ));
                            }
                        }
                    } else {
                        cpu_tensor
                    }
                }
                // FP8 formats: keep as FP8 in VRAM. ROCm 7.1+ dispatches FP8→BF16 on-GPU.
                // Block-wise scale tensors (weight_scale_inv) are loaded alongside and stored
                // in LinearProjection.scale for correct dequantization at matmul time.
                safetensors::Dtype::F8_E4M3 => {
                    let borrowed_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::Float8e4m3fn,
                            Device::Cpu,
                        )
                    };
                    let cpu_tensor = borrowed_tensor.copy();
                    if *device != Device::Cpu {
                        cpu_tensor.to_device(*device)
                    } else {
                        cpu_tensor
                    }
                }
                safetensors::Dtype::F8_E5M2 => {
                    let borrowed_tensor = unsafe {
                        Tensor::from_blob(
                            data.as_ptr(),
                            &shape,
                            &[],
                            tch::Kind::Float8e5m2,
                            Device::Cpu,
                        )
                    };
                    let cpu_tensor = borrowed_tensor.copy();
                    if *device != Device::Cpu {
                        cpu_tensor.to_device(*device)
                    } else {
                        cpu_tensor
                    }
                }
                dtype => {
                    return Err(anyhow!(
                        "Tensor '{}' has unsupported dtype {:?}. Supported: F16, BF16, F32, F8_E4M3, F8_E5M2",
                        name, dtype
                    ));
                }
            };

            weights.insert(name.clone(), tensor);
        }

        info!(
            "✅ Successfully loaded {} tensors ({:.2} MB total) to device {:?}",
            tensor_count, total_size_mb, device
        );

        Ok(())
    }

    /// Create model instance from configuration
    fn create_model_from_config(
        config: ModelConfig,
        mut weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
        model_path: &Path,
        device_pool: Option<&DevicePool>,
        fp8_dequant_load: bool,
    ) -> Result<Box<dyn ModelOperations>> {
        // Run TTN analysis: Tier 1 (embedded) → Tier 2 (cached) → Tier 3 (weight entropy SVD).
        // Weights are available here, enabling Tier 3 for unknown models.
        // Result cached to .analysis/layer_profile.json; non-fatal if it fails.
        // Runs in sync/blocking context (called from spawn_blocking), safe for SVD.
        if let Err(e) = crate::runtime::ttn_profile::get_layer_profile(model_path, &config, Some(&weights)) {
            tracing::warn!("TTN profile analysis failed (non-fatal): {e}");
        }

        // FP8 dequant-once-at-load: materialize every FP8 weight as BF16 with
        // its block scale applied, before any architecture constructor consumes
        // the map. Downstream FP8 lazy-dequant branches then never trigger —
        // the per-matmul dequant disappears from the hot path.
        maybe_dequantize_fp8_weights_at_load(&mut weights, fp8_dequant_load, device_pool);

        // Multi-device is only wired for Llama-family architectures today (the
        // only family with a `stage_from_weights_with_config` that honors a
        // `LayerDeviceMap`). Other architectures log and fall back to the
        // single-device path on the primary device.
        let wants_multi = device_pool.is_some_and(|p| !p.is_single());

        match config.architecture {
            ModelArchitecture::Llama => {
                info!("Creating Llama model");
                Self::create_llama_model(
                    config,
                    weights,
                    device,
                    dtype,
                    max_context,
                    kv_quant_type,
                    device_pool,
                    None,
                )
            }
            ModelArchitecture::Qwen => {
                info!("Creating Qwen model");
                Self::create_qwen_model(config, weights, device, dtype, max_context, kv_quant_type, device_pool)
            }
            ModelArchitecture::Gemma => {
                info!("Creating Gemma model");
                Self::create_gemma_model(config, weights, device, dtype, max_context, kv_quant_type)
            }
            ModelArchitecture::Mistral => {
                info!("Creating Mistral model");
                // For now, Mistral uses Llama architecture
                Self::create_llama_model(
                    config,
                    weights,
                    device,
                    dtype,
                    max_context,
                    kv_quant_type,
                    device_pool,
                    None,
                )
            }
            ModelArchitecture::Janus => {
                info!("Creating Janus multimodal model");
                Self::create_janus_model(config, weights, device, dtype, max_context, kv_quant_type)
            }
            ModelArchitecture::Qwen3_5 => {
                info!("Creating Qwen3.5 hybrid SSM/attention model");
                if wants_multi {
                    warn!(
                        "Multi-device requested for Qwen3.5, which does not yet implement the \
                         layer-split path; building single-device on {:?}",
                        device
                    );
                }
                Self::create_qwen3_5_model(config, weights, device, dtype, max_context, kv_quant_type)
            }
            ModelArchitecture::Unknown(arch) => Err(anyhow!("Unknown architecture: {}", arch)),
        }
    }

    fn create_llama_model(
        config: ModelConfig,
        mut weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
        device_pool: Option<&DevicePool>,
        stage_range: Option<Range<usize>>,
    ) -> Result<Box<dyn ModelOperations>> {
        use super::architectures::llama::LlamaConfig;
        use super::device_pool::LayerDeviceMap;

        // Apply max_context override if specified
        let effective_max_pos = max_context.unwrap_or(config.max_position_embeddings);
        if max_context.is_some() {
            info!("Overriding max_position_embeddings: {} -> {}", config.max_position_embeddings, effective_max_pos);
        }

        // Convert unified config to LlamaConfig
        let llama_config = LlamaConfig {
            version: config.version as u8,
            num_attention_heads: config.num_attention_heads as u32,
            num_key_value_heads: config.num_key_value_heads as u32,
            hidden_size: config.hidden_size as u32,
            head_dim: config.head_dim as u32,
            intermediate_size: config.intermediate_size as u32,
            max_position_embeddings: effective_max_pos as u32,
            rms_norm_eps: config.rms_norm_eps,
            vocab_size: config.vocab_size as u32,
            original_vocab_size: config.vocab_size as u32,  // Will be updated if padding is applied
            num_hidden_layers: config.num_hidden_layers as u32,
            rope_theta: config.rope_theta,
            rope_scaling: None,
            hidden_activation: config.hidden_activation,
            query_pre_attn_scalar: config.query_pre_attn_scalar,
            use_qk_norm: config.use_qk_norm,
            scale_embeddings: config.scale_embeddings,
            layer_types: vec![],
            rope_local_base_freq: None,
            // Preserve the *real* architecture so tokenizer-config dispatch (and
            // any other architecture-keyed behavior) sees Qwen/Mistral rather
            // than the Llama stand-in. Without this, a Qwen3 checkpoint loaded
            // via create_qwen_model→create_llama_model reports `Llama`, the
            // QwenTokenizerConfig (which pads the tokenizer vocab to the model's
            // embedding size) is never applied, and the tokenizer/embedding vocab
            // mismatch causes a CUDA index-out-of-bounds on the embedding lookup (#143).
            model_architecture: match config.architecture {
                super::model_config::ModelArchitecture::Qwen => {
                    super::architectures::ModelArchitecture::Qwen {
                        version: config.version as u8,
                        is_moe: config.is_moe,
                        context_length: config.max_position_embeddings,
                    }
                }
                super::model_config::ModelArchitecture::Mistral => {
                    super::architectures::ModelArchitecture::Mistral
                }
                super::model_config::ModelArchitecture::Gemma => {
                    super::architectures::ModelArchitecture::Gemma
                }
                // Llama, Janus, Qwen3_5 (not routed through LlamaModel), and unknowns
                // keep the Llama identity with the parsed version.
                _ => super::architectures::ModelArchitecture::Llama {
                    version: config.version as u8,
                },
            },
        };

        info!(
            "[create_llama_model] Passing llama_config.max_position_embeddings = {} to from_weights_with_config",
            llama_config.max_position_embeddings
        );

        // Multi-device pipeline (#314 wiring): when a multi-device `DevicePool`
        // is present, spread all decoder layers `[0..num_hidden_layers)` across
        // the pool via an even (parameter-balanced) split and build the model as
        // a single stage owning the full range with that device map. The
        // architecture's `forward_layers` then performs the lone cross-device
        // copy at each stage boundary. Single-device pools and `None` take the
        // original whole-model path unchanged.
        let num_layers = llama_config.num_hidden_layers as usize;
        let model = if let Some(layer_range) = stage_range {
            let device_map = if let Some(pool) = device_pool.filter(|p| !p.is_single()) {
                LayerDeviceMap::even_split(pool, num_layers)?
            } else {
                LayerDeviceMap::single(*device, num_layers)?
            };
            info!(
                "Building authoritative Llama stage {:?} of {} global layer(s)",
                layer_range, num_layers
            );
            LlamaModel::stage_from_weights_with_config(
                &mut weights,
                llama_config,
                &device_map,
                layer_range,
                dtype,
                kv_quant_type,
            )?
        } else if let Some(pool) = device_pool.filter(|p| !p.is_single()) {
            let device_map = LayerDeviceMap::even_split(pool, num_layers)?;
            info!(
                "🔀 Building Llama model as a multi-device pipeline: {} layer(s) across \
                 {} device(s) ({} stage boundary copies per forward)",
                num_layers,
                pool.len(),
                pool.len() - 1,
            );
            LlamaModel::stage_from_weights_with_config(
                &mut weights,
                llama_config,
                &device_map,
                0..num_layers,
                dtype,
                kv_quant_type,
            )?
        } else {
            // Pass mutable reference to allow incremental tensor freeing during construction
            LlamaModel::from_weights_with_config(
                &mut weights,
                llama_config,
                device,
                dtype,
                kv_quant_type,
            )?
        };

        Ok(Box::new(model))
    }

    fn create_qwen_model(
        config: ModelConfig,
        weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
        device_pool: Option<&DevicePool>,
    ) -> Result<Box<dyn ModelOperations>> {
        // Qwen uses Llama architecture with specific configuration
        // The key difference is in the config values, not the architecture
        info!("   Using Llama architecture with Qwen configuration");
        info!("   rope_theta: {} (from config)", config.rope_theta);
        Self::create_llama_model(
            config,
            weights,
            device,
            dtype,
            max_context,
            kv_quant_type,
            device_pool,
            None,
        )
    }

    fn create_gemma_model(
        _config: ModelConfig,
        weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
        _max_context: Option<usize>,
        _kv_quant_type: KVQuantType,
    ) -> Result<Box<dyn ModelOperations>> {
        // Gemma has its own implementation
        // TODO: Pass max_context and kv_quant_type to GemmaModel when it supports them
        Ok(Box::new(GemmaModel::from_weights(&weights, device, dtype)?))
    }

    fn create_janus_model(
        config: ModelConfig,
        weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
        max_context: Option<usize>,
        _kv_quant_type: KVQuantType,
    ) -> Result<Box<dyn ModelOperations>> {
        use super::architectures::janus::{
            JanusModel, JanusConfig, VisionEncoderConfig, ProjectorConfig,
        };
        use super::architectures::VisionEncoderType;

        // Apply max_context override if specified
        let effective_max_pos = max_context.unwrap_or(config.max_position_embeddings);

        // For now, create a simplified Janus config
        // In practice, this would be derived from the model's config.json
        let janus_config = JanusConfig {
            // Use Llama config for the language model
            language_config: Box::new(super::architectures::llama::LlamaConfig {
                version: 3,
                num_attention_heads: config.num_attention_heads as u32,
                num_key_value_heads: config.num_key_value_heads as u32,
                hidden_size: config.hidden_size as u32,
                head_dim: config.head_dim as u32,
                intermediate_size: config.intermediate_size as u32,
                max_position_embeddings: effective_max_pos as u32,
                rms_norm_eps: config.rms_norm_eps,
                vocab_size: config.vocab_size as u32,
                original_vocab_size: config.vocab_size as u32,
                num_hidden_layers: config.num_hidden_layers as u32,
                rope_theta: config.rope_theta,
                rope_scaling: None,  // TODO: Convert from config.rope_scaling
                hidden_activation: config.hidden_activation.clone(),
                query_pre_attn_scalar: config.query_pre_attn_scalar,
                use_qk_norm: config.use_qk_norm,
                scale_embeddings: config.scale_embeddings,
                layer_types: vec!["global".to_owned(); config.num_hidden_layers],
                rope_local_base_freq: None,
                model_architecture: super::architectures::ModelArchitecture::Llama { version: 3 },
            }),
            vision_config: VisionEncoderConfig {
                encoder_type: VisionEncoderType::SigLIP {
                    hidden_size: 1024,  // From config: vision_config.hidden_size
                    image_size: 384,
                    patch_size: 16,
                    num_layers: 24,
                },
                hidden_size: 1024,
                image_size: 384,
                patch_size: 16,
                num_layers: 24,
                num_patches: (384 / 16) * (384 / 16),  // 576 patches
                num_attention_heads: Some(16),  // From config: vision_config.num_attention_heads
                intermediate_size: Some(4096),  // From config: vision_config.intermediate_size
            },
            aligner_config: ProjectorConfig {
                input_dim: 1024,  // Vision hidden size (matches vision_config.hidden_size)
                output_dim: config.hidden_size,  // Language model hidden size
                hidden_dim: Some(config.hidden_size),  // 2-layer MLP
            },
            generation_config: None,  // No image generation for now
            device: *device,
            dtype,
        };

        Ok(Box::new(JanusModel::from_weights(
            weights,
            janus_config,
            *device,
            dtype,
        )?))
    }

    fn create_qwen3_5_model(
        config: ModelConfig,
        mut weights: HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
    ) -> Result<Box<dyn ModelOperations>> {
        use super::architectures::qwen3_5::{Qwen3_5Model, Qwen3_5TextConfig};
        use super::architectures::qwen3_5_vision::Qwen3_5VisionConfig;

        let effective_max_pos = max_context.unwrap_or(config.max_position_embeddings);

        let text_cfg = Qwen3_5TextConfig::from_model_config(&config, effective_max_pos);

        // Build vision config if the checkpoint has vision weights
        let vision_cfg = if config.has_vision {
            // vision_config is stored in the ModelConfig's raw JSON during loading
            // Use the fields already parsed into ModelConfig
            Some(Qwen3_5VisionConfig {
                depth: 27,
                hidden_size: 1152,
                intermediate_size: 4304,
                num_heads: 16,
                head_dim: 72,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                out_hidden_size: config.vision_out_hidden_size,
                rms_norm_eps: 1e-6,
            })
        } else if weights.keys().any(|k| k.starts_with("visual.")) {
            // Fallback: detect from weights even if config didn't set has_vision
            Some(Qwen3_5VisionConfig {
                depth: 27,
                hidden_size: 1152,
                intermediate_size: 4304,
                num_heads: 16,
                head_dim: 72,
                patch_size: 16,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                out_hidden_size: if config.vision_out_hidden_size > 0 { config.vision_out_hidden_size } else { 3584 },
                rms_norm_eps: 1e-6,
            })
        } else {
            None
        };

        Ok(Box::new(Qwen3_5Model::from_weights(
            &mut weights,
            text_cfg,
            vision_cfg,
            device,
            dtype,
            kv_quant_type,
        )?))
    }

    // =========================================================================
    // FsOps-aware methods (worktree-scoped, path-contained access)
    // =========================================================================

    /// Create a model using FsOps for weight loading.
    ///
    /// Uses FsOps::read_file() instead of direct filesystem access.
    /// The `model_path` is still needed for ModelConfig and architecture detection
    /// (which parse config.json), but weight data is read through FsOps.
    #[instrument(name = "model_factory.create_with_fs", skip(device, dtype, fs, device_pool), fields(model_path = %model_path.display()))]
    pub async fn create_with_fs(
        model_path: &Path,
        device: &Device,
        dtype: DType,
        max_context: Option<usize>,
        kv_quant_type: KVQuantType,
        fs: &WorktreeClient,
        device_pool: Option<&DevicePool>,
        fp8_dequant_load: bool,
    ) -> Result<Box<dyn ModelOperations>> {
        info!("Loading model via FsOps: {}", model_path.display());

        let shard_names = Self::find_shard_names_fs(fs).await?;

        if shard_names.len() > 1 {
            info!("Loading {} weight shards via FsOps", shard_names.len());
        }

        let weights = Self::load_weights_fs(fs, &shard_names, device, dtype).await?;
        let config = ModelConfig::load(model_path, &weights)?;
        let model = Self::create_model_from_config(config, weights, device, dtype, max_context, kv_quant_type, model_path, device_pool, fp8_dequant_load)?;
        info!("Model created successfully via FsOps");
        Ok(model)
    }

    /// Detect model dtype using FsOps for file reading.
    pub async fn detect_model_dtype_fs(fs: &WorktreeClient) -> Result<DType> {
        let shard_names = Self::find_shard_names_fs(fs).await?;
        if shard_names.is_empty() {
            return Err(anyhow!("No model weights found"));
        }

        let file_content = fs.read_file_chunked(&shard_names[0]).await?;
        let tensors = safetensors::SafeTensors::deserialize(&file_content)?;

        let mut f16_count = 0;
        let mut bf16_count = 0;
        let mut f32_count = 0;

        for (_, tensor) in tensors.tensors().into_iter().take(10) {
            match tensor.dtype() {
                safetensors::Dtype::F16 => f16_count += 1,
                safetensors::Dtype::BF16
                | safetensors::Dtype::F8_E4M3
                | safetensors::Dtype::F8_E5M2 => bf16_count += 1,
                safetensors::Dtype::F32 => f32_count += 1,
                _ => {},
            }
        }

        if f16_count > bf16_count && f16_count > f32_count {
            info!("Detected F16 model (via FsOps)");
            Ok(tch::Kind::Half)
        } else if bf16_count >= f16_count && bf16_count >= f32_count {
            info!("Detected BF16 model (via FsOps)");
            Ok(tch::Kind::BFloat16)
        } else if f32_count > 0 {
            info!("Detected F32 model (via FsOps)");
            Ok(tch::Kind::Float)
        } else {
            info!("Could not detect model dtype via FsOps, defaulting to BF16");
            Ok(tch::Kind::BFloat16)
        }
    }

    /// Find shard file names via FsOps (returns relative paths).
    async fn find_shard_names_fs(fs: &WorktreeClient) -> Result<Vec<String>> {
        // Check for single file first
        if fs.stat_path("model.safetensors").await.map(|s| s.exists).unwrap_or(false) {
            return Ok(vec!["model.safetensors".to_owned()]);
        }

        // Look for sharded files
        let entries = fs.list_dir_path(".").await?;
        let mut shard_names: Vec<String> = entries
            .into_iter()
            .filter(|e| {
                e.name.starts_with("model-") && e.name.ends_with(".safetensors")
            })
            .map(|e| e.name)
            .collect();

        shard_names.sort();
        Ok(shard_names)
    }

    /// Load weights from safetensors files via FsOps.
    async fn load_weights_fs(
        fs: &WorktreeClient,
        shard_names: &[String],
        device: &Device,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        let mut all_weights = HashMap::new();

        if shard_names.is_empty() {
            return Err(anyhow!("No safetensors files found"));
        }

        for (idx, name) in shard_names.iter().enumerate() {
            if shard_names.len() > 1 {
                info!("Loading shard {}/{} via FsOps: {}", idx + 1, shard_names.len(), name);
            }

            let data = fs.read_file_chunked(name).await?;
            let tensors = safetensors::SafeTensors::deserialize(&data)?;
            Self::create_tensors_from_safetensors(tensors, &mut all_weights, device, dtype)?;
        }

        Ok(all_weights)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod stage_subset_tests {
    use super::*;
    use safetensors::tensor::{serialize, Dtype, TensorView};

    struct FixtureTensor {
        name: String,
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl FixtureTensor {
        fn f32(name: impl Into<String>, shape: &[usize], seed: f32) -> Self {
            let elements = shape.iter().product();
            let data = (0..elements)
                .flat_map(|index| (seed + index as f32 * 0.01).to_le_bytes())
                .collect();
            Self {
                name: name.into(),
                dtype: Dtype::F32,
                shape: shape.to_vec(),
                data,
            }
        }

        fn i64(name: impl Into<String>) -> Self {
            Self::i64_shape(name, &[1])
        }

        fn i64_shape(name: impl Into<String>, shape: &[usize]) -> Self {
            let elements = shape.iter().product();
            let data = (0..elements)
                .flat_map(|index| (index as i64 + 7).to_le_bytes())
                .collect();
            Self {
                name: name.into(),
                dtype: Dtype::I64,
                shape: shape.to_vec(),
                data,
            }
        }
    }

    fn write_shard(dir: &Path, name: &str, tensors: &[FixtureTensor]) {
        let views: Vec<_> = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).unwrap(),
                )
            })
            .collect();
        let bytes = serialize(views, &None).unwrap();
        std::fs::write(dir.join(name), bytes).unwrap();
    }

    fn write_index(dir: &Path, entries: &[(&str, &str)]) {
        let weight_map: BTreeMap<_, _> = entries
            .iter()
            .map(|(tensor, shard)| ((*tensor).to_owned(), (*shard).to_owned()))
            .collect();
        let index = serde_json::json!({ "weight_map": weight_map });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
    }

    fn loaded_bytes(weights: &HashMap<String, Tensor>) -> usize {
        weights
            .values()
            .map(|tensor| tensor.numel() * tensor.kind().elt_size_in_bytes())
            .sum()
    }

    fn tiny_dense_tensors(explicit_head: bool) -> Vec<FixtureTensor> {
        let mut tensors = vec![
            FixtureTensor::f32("model.embed_tokens.weight", &[8, 4], 0.01),
            FixtureTensor::f32("model.norm.weight", &[4], 1.0),
        ];
        if explicit_head {
            tensors.push(FixtureTensor::f32("lm_head.weight", &[8, 4], 0.02));
        }

        let prefix = "model.layers.0";
        for (suffix, shape, seed) in [
            ("self_attn.q_proj.weight", vec![4, 4], 0.03),
            ("self_attn.k_proj.weight", vec![4, 4], 0.04),
            ("self_attn.v_proj.weight", vec![4, 4], 0.05),
            ("self_attn.o_proj.weight", vec![4, 4], 0.06),
            ("mlp.gate_proj.weight", vec![8, 4], 0.07),
            ("mlp.up_proj.weight", vec![8, 4], 0.08),
            ("mlp.down_proj.weight", vec![4, 8], 0.09),
            ("input_layernorm.weight", vec![4], 1.0),
            ("post_attention_layernorm.weight", vec![4], 1.0),
        ] {
            tensors.push(FixtureTensor::f32(
                format!("{prefix}.{suffix}"),
                &shape,
                seed,
            ));
        }
        tensors
    }

    fn write_tiny_dense_checkpoint(dir: &Path, tensors: &[FixtureTensor]) {
        const SHARD: &str = "model-00001-of-00001.safetensors";
        write_shard(dir, SHARD, tensors);
        let entries: Vec<_> = tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), SHARD))
            .collect();
        write_index(dir, &entries);

        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "vocab_size": 8,
            "max_position_embeddings": 16,
            "rms_norm_eps": 0.00001
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    fn assert_pointer_resolution_error(error: anyhow::Error) {
        let message = error.to_string();
        let normalized = message.to_lowercase();
        assert!(
            normalized.contains("pointer")
                || normalized.contains("lfs")
                || normalized.contains("xet"),
            "stage path did not report pointer resolution: {message}"
        );
        assert!(
            !normalized.contains("failed to parse required stage shard")
                && !normalized.contains("invalid safetensors"),
            "stage path reached safetensors parsing before pointer resolution: {message}"
        );
    }

    #[tokio::test]
    async fn stage_schema_validates_all_dense_roles() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_dense_checkpoint(dir.path(), &tiny_dense_tensors(true));

        let config = ModelConfig::load(dir.path(), &HashMap::new()).unwrap();
        let plan = ModelFactory::stage_weight_plan(dir.path(), 1, 0..1).unwrap();
        let metadata = ModelFactory::stage_tensor_metadata(&plan).await.unwrap();
        ModelFactory::validate_stage_tensor_schema(&config, &(0..1), &metadata).unwrap();

        for (name, bad_shape) in [
            ("model.embed_tokens.weight", vec![7, 4]),
            ("model.embed_tokens.weight", vec![8, 5]),
            ("lm_head.weight", vec![7, 4]),
            ("model.norm.weight", vec![5]),
            ("model.layers.0.self_attn.q_proj.weight", vec![3, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 7]),
            ("model.layers.0.input_layernorm.weight", vec![5]),
        ] {
            let mut mismatched = metadata.clone();
            mismatched.insert(name.to_owned(), bad_shape);
            let error = ModelFactory::validate_stage_tensor_schema(&config, &(0..1), &mismatched)
                .unwrap_err();
            assert!(
                error.to_string().contains(name) && error.to_string().contains("shape mismatch"),
                "unexpected schema error for {name}: {error}"
            );
        }

        let mut missing_layer_weight = metadata.clone();
        missing_layer_weight.remove("model.layers.0.self_attn.k_proj.weight");
        let missing_error =
            ModelFactory::validate_stage_tensor_schema(&config, &(0..1), &missing_layer_weight)
                .unwrap_err();
        assert!(
            missing_error
                .to_string()
                .contains("model.layers.0.self_attn.k_proj.weight"),
            "unexpected missing-weight error: {missing_error}"
        );

        // With no explicit head, a complete-range stage ties the already
        // validated [vocab, hidden] embedding.
        let mut tied = metadata;
        tied.remove("lm_head.weight");
        ModelFactory::validate_stage_tensor_schema(&config, &(0..1), &tied).unwrap();
    }

    #[tokio::test]
    async fn stage_schema_mismatch_fails_before_tensor_materialization() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = tiny_dense_tensors(false);
        *tensors
            .iter_mut()
            .find(|tensor| tensor.name == "model.embed_tokens.weight")
            .unwrap() = FixtureTensor::f32("model.embed_tokens.weight", &[7, 4], 0.01);
        *tensors
            .iter_mut()
            .find(|tensor| tensor.name == "model.layers.0.self_attn.q_proj.weight")
            .unwrap() = FixtureTensor::i64_shape("model.layers.0.self_attn.q_proj.weight", &[4, 4]);
        write_tiny_dense_checkpoint(dir.path(), &tensors);

        let error = ModelFactory::create_stage(
            dir.path(),
            &Device::Cpu,
            DType::Float,
            None,
            KVQuantType::None,
            None,
            ModelStageRequest { layer_range: 0..1 },
            false,
        )
        .await
        .err()
        .expect("config/tensor mismatch must fail");
        assert!(
            error.to_string().contains("model.embed_tokens.weight")
                && error.to_string().contains("shape mismatch"),
            "config mismatch must fail before the selected unsupported-dtype sentinel is \
             materialized: {error}"
        );
        assert!(
            !error.to_string().contains("unsupported dtype"),
            "selected tensors were materialized before schema validation: {error}"
        );
    }

    #[tokio::test]
    async fn stage_schema_complete_range_matches_whole_model() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_dense_checkpoint(dir.path(), &tiny_dense_tensors(false));

        let whole = ModelFactory::create(
            dir.path(),
            &Device::Cpu,
            DType::Float,
            None,
            KVQuantType::None,
            None,
            false,
        )
        .await
        .unwrap();
        let stage = ModelFactory::create_stage(
            dir.path(),
            &Device::Cpu,
            DType::Float,
            None,
            KVQuantType::None,
            None,
            ModelStageRequest { layer_range: 0..1 },
            false,
        )
        .await
        .unwrap();

        let input = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);
        let whole_logits = whole.forward(&input, None).unwrap();
        let stage_logits = stage.forward(&input, None).unwrap();
        assert_eq!(whole_logits.size(), stage_logits.size());
        assert!(
            whole_logits.to_device(Device::Cpu).allclose(
                &stage_logits.to_device(Device::Cpu),
                1e-5,
                1e-5,
                false
            ),
            "valid complete-range stage forward diverged from ModelFactory::create"
        );
    }

    #[test]
    fn stage_weight_plan_includes_top_level_fp8_companion_scales() {
        // FP8 checkpoints carry block-wise `<name>_scale_inv` companions for
        // quantized tensors. Layer-prefixed scales enter the stage plan via the
        // layer-index filter; top-level embedding/lm_head scales must be added
        // explicitly, or the stage loads a scale-less FP8 tensor and eager
        // dequant casts raw FP8 codes to BF16 (#1519 Codex review).
        let dir = tempfile::tempdir().unwrap();
        let shard = "model-00001-of-00001.safetensors";
        write_index(
            dir.path(),
            &[
                ("model.embed_tokens.weight", shard),
                ("model.embed_tokens.weight_scale_inv", shard),
                ("model.norm.weight", shard),
                ("lm_head.weight", shard),
                ("lm_head.weight_scale_inv", shard),
                ("model.layers.0.self_attn.q_proj.weight", shard),
                ("model.layers.0.self_attn.q_proj.weight_scale_inv", shard),
            ],
        );

        let plan = ModelFactory::stage_weight_plan(dir.path(), 1, 0..1).unwrap();
        let names: Vec<&str> = plan.values().flatten().map(String::as_str).collect();
        for expected in [
            "model.embed_tokens.weight",
            "model.embed_tokens.weight_scale_inv",
            "model.norm.weight",
            "lm_head.weight",
            "lm_head.weight_scale_inv",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight_scale_inv",
        ] {
            assert!(names.contains(&expected), "stage plan missing {expected}");
        }
        // No phantom scale entries: the plan contains exactly these tensors.
        assert_eq!(names.len(), 7, "unexpected extra tensors in stage plan");
    }

    #[tokio::test]
    async fn stage_pointer_shard_fails_before_mmap_in_preflight_and_selected_load() {
        const SHARD: &str = "model-00001-of-00001.safetensors";

        let dir = tempfile::tempdir().unwrap();
        write_tiny_dense_checkpoint(dir.path(), &tiny_dense_tensors(true));
        std::fs::write(
            dir.path().join(SHARD),
            b"version https://git-lfs.github.com/spec/v1\n\
oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n\
size 4096\n",
        )
        .unwrap();

        let plan = ModelFactory::stage_weight_plan(dir.path(), 1, 0..1).unwrap();
        let preflight_error = ModelFactory::stage_tensor_metadata(&plan).await.unwrap_err();
        assert_pointer_resolution_error(preflight_error);

        let selected_error = ModelFactory::load_weights_for_stage_plan(
            plan,
            0..1,
            &Device::Cpu,
            DType::Float,
        )
        .await
        .unwrap_err();
        assert_pointer_resolution_error(selected_error);
    }

    #[tokio::test]
    async fn out_of_range_sentinel_in_needed_shard_is_never_materialized() {
        let dir = tempfile::tempdir().unwrap();
        let shard = "model-00001-of-00001.safetensors";
        write_shard(
            dir.path(),
            shard,
            &[
                FixtureTensor::f32("model.embed_tokens.weight", &[2, 2], 0.1),
                FixtureTensor::f32("model.layers.0.marker.weight", &[2, 2], 0.2),
                FixtureTensor::i64("model.layers.1.sentinel.weight"),
            ],
        );
        write_index(
            dir.path(),
            &[
                ("model.embed_tokens.weight", shard),
                ("model.layers.0.marker.weight", shard),
                ("model.layers.1.sentinel.weight", shard),
            ],
        );

        let stage =
            ModelFactory::load_weights_for_stage(dir.path(), 2, 0..1, &Device::Cpu, DType::Float)
                .await
                .unwrap();
        assert_eq!(stage.len(), 2);
        assert!(stage.contains_key("model.embed_tokens.weight"));
        assert!(stage.contains_key("model.layers.0.marker.weight"));
        assert!(!stage.contains_key("model.layers.1.sentinel.weight"));

        let full_error = ModelFactory::load_weights(dir.path(), &Device::Cpu, DType::Float)
            .await
            .unwrap_err();
        assert!(
            full_error.to_string().contains("unsupported dtype"),
            "the sentinel must prove that whole-shard tensor materialization fails: {full_error}"
        );
    }

    #[tokio::test]
    async fn stage_create_uses_real_range_and_opens_only_needed_shards() {
        const NEEDED: &str = "model-00001-of-00002.safetensors";
        const UNNEEDED: &str = "model-00002-of-00002.safetensors";

        let dir = tempfile::tempdir().unwrap();
        let mut tensors = vec![FixtureTensor::f32(
            "model.embed_tokens.weight",
            &[8, 4],
            0.1,
        )];
        let prefix = "model.layers.0";
        for (suffix, shape) in [
            ("self_attn.q_proj.weight", vec![4, 4]),
            ("self_attn.k_proj.weight", vec![4, 4]),
            ("self_attn.v_proj.weight", vec![4, 4]),
            ("self_attn.o_proj.weight", vec![4, 4]),
            ("mlp.gate_proj.weight", vec![8, 4]),
            ("mlp.up_proj.weight", vec![8, 4]),
            ("mlp.down_proj.weight", vec![4, 8]),
            ("input_layernorm.weight", vec![4]),
            ("post_attention_layernorm.weight", vec![4]),
        ] {
            tensors.push(FixtureTensor::f32(
                format!("{prefix}.{suffix}"),
                &shape,
                0.2,
            ));
        }
        write_shard(dir.path(), NEEDED, &tensors);

        let mut entries: Vec<(String, String)> = tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), NEEDED.to_owned()))
            .collect();
        entries.push((
            "model.layers.1.self_attn.q_proj.weight".to_owned(),
            UNNEEDED.to_owned(),
        ));
        let borrowed_entries: Vec<_> = entries
            .iter()
            .map(|(tensor, shard)| (tensor.as_str(), shard.as_str()))
            .collect();
        write_index(dir.path(), &borrowed_entries);

        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "vocab_size": 8,
            "max_position_embeddings": 16,
            "rms_norm_eps": 0.00001
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        // UNNEEDED intentionally does not exist. Successful construction proves
        // the real [0,1) request did not open it.
        let model = ModelFactory::create_stage(
            dir.path(),
            &Device::Cpu,
            DType::Float,
            None,
            KVQuantType::None,
            None,
            ModelStageRequest { layer_range: 0..1 },
            false,
        )
        .await
        .unwrap();
        let input = Tensor::from_slice(&[0_i64, 1]).reshape([1, 2]);
        let embedded = model.embed_tokens(&input).unwrap();
        assert_eq!(embedded.size(), [1, 2, 4]);
    }

    #[tokio::test]
    async fn partial_stage_loads_fewer_tensors_and_bytes_than_whole_model() {
        let dir = tempfile::tempdir().unwrap();
        let first = "model-00001-of-00002.safetensors";
        let last = "model-00002-of-00002.safetensors";
        write_shard(
            dir.path(),
            first,
            &[
                FixtureTensor::f32("model.embed_tokens.weight", &[8, 4], 0.1),
                FixtureTensor::f32("model.layers.0.marker.weight", &[4, 4], 0.2),
            ],
        );
        write_shard(
            dir.path(),
            last,
            &[
                FixtureTensor::f32("model.layers.1.marker.weight", &[16, 16], 0.3),
                FixtureTensor::f32("model.norm.weight", &[4], 0.4),
                FixtureTensor::f32("lm_head.weight", &[8, 4], 0.5),
            ],
        );
        write_index(
            dir.path(),
            &[
                ("model.embed_tokens.weight", first),
                ("model.layers.0.marker.weight", first),
                ("model.layers.1.marker.weight", last),
                ("model.norm.weight", last),
                ("lm_head.weight", last),
            ],
        );

        let partial =
            ModelFactory::load_weights_for_stage(dir.path(), 2, 1..2, &Device::Cpu, DType::Float)
                .await
                .unwrap();
        let whole = ModelFactory::load_weights(dir.path(), &Device::Cpu, DType::Float)
            .await
            .unwrap();

        let partial_names: BTreeSet<_> = partial.keys().map(String::as_str).collect();
        assert_eq!(
            partial_names,
            BTreeSet::from([
                "lm_head.weight",
                "model.layers.1.marker.weight",
                "model.norm.weight",
            ])
        );
        assert!(partial.len() < whole.len());
        assert!(loaded_bytes(&partial) < loaded_bytes(&whole));
    }

    #[tokio::test]
    async fn stage_mode_fails_closed_without_a_valid_index() {
        let dir = tempfile::tempdir().unwrap();
        write_shard(
            dir.path(),
            "model.safetensors",
            &[FixtureTensor::f32(
                "model.embed_tokens.weight",
                &[2, 2],
                0.1,
            )],
        );

        let missing =
            ModelFactory::load_weights_for_stage(dir.path(), 1, 0..1, &Device::Cpu, DType::Float)
                .await
                .unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("requires authoritative shard index"),
            "got: {missing}"
        );

        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            br#"{"weight_map":[]}"#,
        )
        .unwrap();
        let malformed =
            ModelFactory::load_weights_for_stage(dir.path(), 1, 0..1, &Device::Cpu, DType::Float)
                .await
                .unwrap_err();
        assert!(
            malformed.to_string().contains("valid `weight_map` object"),
            "got: {malformed}"
        );
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod fp8_dequant_tests {
    use super::*;
    use crate::runtime::architectures::llama::LinearProjection;
    use safetensors::tensor::{serialize, Dtype, TensorView};

    /// FP8 E4M3 byte for an exactly-representable power of two:
    /// value = (-1)^sign * 2^log2 (S EEEE MMM, exponent bias 7, mantissa 0).
    fn f8e4m3_pow2(negative: bool, log2: i32) -> u8 {
        let sign = if negative { 0x80u8 } else { 0 };
        sign | (((log2 + 7) as u8) << 3)
    }

    /// Build an FP8 E4M3 weight tensor with a deterministic power-of-two
    /// pattern (exactly representable in both E4M3 and BF16).
    fn fp8_weight(rows: i64, cols: i64) -> Tensor {
        let vals: Vec<f32> = (0..(rows * cols))
            .map(|i| {
                let mag = [0.5f32, 1.0, 2.0][(i % 3) as usize];
                if i % 2 == 1 { -mag } else { mag }
            })
            .collect();
        Tensor::from_slice(&vals)
            .reshape([rows, cols])
            .to_kind(DType::Float8e4m3fn)
    }

    /// Deterministic FP8 weight + companion BF16 block scale (single block).
    fn fp8_weight_with_scale(rows: i64, cols: i64, scale: f32) -> (Tensor, Tensor) {
        let weight = fp8_weight(rows, cols);
        let scale = Tensor::from_slice(&[scale])
            .reshape([1, 1])
            .to_kind(DType::BFloat16);
        (weight, scale)
    }

    #[test]
    fn dequant_at_load_materializes_bf16_and_drops_scales() {
        let (weight, scale) = fp8_weight_with_scale(4, 4, 0.5);
        let untouched = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_kind(DType::BFloat16);
        let mut weights = HashMap::from([
            ("w.weight".to_owned(), weight),
            ("w.weight_scale_inv".to_owned(), scale),
            ("norm.weight".to_owned(), untouched),
        ]);

        let converted = dequantize_fp8_weights_at_load(&mut weights);

        assert_eq!(converted, 1);
        let w = weights.get("w.weight").expect("weight kept");
        assert_eq!(w.kind(), DType::BFloat16, "FP8 weight must be materialized as BF16");
        assert!(
            !weights.contains_key("w.weight_scale_inv"),
            "block scale must be dropped after load-time dequant"
        );
        assert_eq!(
            weights.get("norm.weight").expect("norm kept").kind(),
            DType::BFloat16,
            "non-FP8 tensors untouched"
        );

        // Values must equal the eager block dequantization: pattern is
        // ±{0.5, 1, 2} with scale 0.5 → ±{0.25, 0.5, 1} (exact in BF16).
        let expected: Vec<f32> = (0..16)
            .map(|i| {
                let mag = [0.25f32, 0.5, 1.0][(i % 3) as usize];
                if i % 2 == 1 { -mag } else { mag }
            })
            .collect();
        let expected = Tensor::from_slice(&expected).reshape([4, 4]);
        let got = w.to_kind(DType::Float);
        assert!(
            got.allclose(&expected, 1e-6, 1e-6, false),
            "load-time dequant values diverged from eager block dequantization"
        );
    }

    #[test]
    fn dequant_at_load_forward_matches_lazy_path() {
        // Same stored checkpoint tensor map fed through the production
        // `LinearProjection::take` load path twice: once left FP8 (lazy
        // per-matmul dequant), once dequantized at load.
        let mut lazy_weights = HashMap::new();
        let (w, s) = fp8_weight_with_scale(4, 8, 0.5);
        lazy_weights.insert("p.weight".to_owned(), w);
        lazy_weights.insert("p.weight_scale_inv".to_owned(), s);
        let lazy_proj = LinearProjection::take(&mut lazy_weights, "p.weight").unwrap();
        assert_eq!(lazy_proj.weight.kind(), DType::Float8e4m3fn);
        assert!(lazy_proj.scale.is_some());

        let mut eager_weights = HashMap::new();
        let (w, s) = fp8_weight_with_scale(4, 8, 0.5);
        eager_weights.insert("p.weight".to_owned(), w);
        eager_weights.insert("p.weight_scale_inv".to_owned(), s);
        assert_eq!(dequantize_fp8_weights_at_load(&mut eager_weights), 1);
        let eager_proj = LinearProjection::take(&mut eager_weights, "p.weight").unwrap();
        assert_eq!(eager_proj.weight.kind(), DType::BFloat16);
        assert!(eager_proj.scale.is_none(), "scale must be consumed at load");

        let input_vals: Vec<f32> = (0..8).map(|i| 0.125 * (i as f32 + 1.0)).collect();
        let input = Tensor::from_slice(&input_vals)
            .reshape([1, 8])
            .to_kind(DType::BFloat16);
        let y_lazy = lazy_proj.apply(&input).to_kind(DType::Float);
        let y_eager = eager_proj.apply(&input).to_kind(DType::Float);
        assert_eq!(y_lazy.size(), y_eager.size());
        assert!(
            y_lazy.allclose(&y_eager, 1e-3, 1e-3, false),
            "load-time dequant forward diverged from lazy per-matmul dequant"
        );
    }

    #[test]
    fn dequant_at_load_noop_without_fp8() {
        let mut weights = HashMap::from([(
            "w.weight".to_owned(),
            Tensor::from_slice(&[1.0f32, 2.0]).to_kind(DType::BFloat16),
        )]);
        assert_eq!(dequantize_fp8_weights_at_load(&mut weights), 0);
        assert_eq!(weights["w.weight"].kind(), DType::BFloat16);
    }

    #[test]
    fn dequant_at_load_leaves_nonconforming_scale_untouched() {
        // Scale shape [3, 1] does not block-divide weight [4, 4]: the pair must
        // be left FP8 + scale for the lazy path (which fails loudly), not cast
        // to BF16 without scales, and must not count as converted.
        let (weight, _) = fp8_weight_with_scale(4, 4, 0.5);
        let bad_scale = Tensor::from_slice(&[0.5f32, 0.5, 0.5])
            .reshape([3, 1])
            .to_kind(DType::BFloat16);
        let mut weights = HashMap::from([
            ("w.weight".to_owned(), weight),
            ("w.weight_scale_inv".to_owned(), bad_scale),
        ]);

        assert_eq!(dequantize_fp8_weights_at_load(&mut weights), 0);
        assert_eq!(
            weights["w.weight"].kind(),
            DType::Float8e4m3fn,
            "non-conforming weight must stay FP8 for the lazy dequant path"
        );
        assert!(
            weights.contains_key("w.weight_scale_inv"),
            "non-conforming scale must stay in the map"
        );
    }

    /// Byte-level fixture for the end-to-end checkpoint test.
    struct RawTensor {
        name: String,
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl RawTensor {
        fn bf16(name: impl Into<String>, shape: &[usize], fill: f32) -> Self {
            let elements = shape.iter().product();
            let bits = ((fill.to_bits() >> 16) as u16).to_le_bytes();
            Self {
                name: name.into(),
                dtype: Dtype::BF16,
                shape: shape.to_vec(),
                data: (0..elements).flat_map(|_| bits).collect(),
            }
        }

        fn f8(name: impl Into<String>, shape: &[usize]) -> Self {
            let elements = shape.iter().product();
            Self {
                name: name.into(),
                dtype: Dtype::F8_E4M3,
                shape: shape.to_vec(),
                data: (0..elements)
                    .map(|i| f8e4m3_pow2(i % 2 == 1, ((i % 3) as i32) - 1))
                    .collect(),
            }
        }

        fn f8_scale(name: impl Into<String>, scale: f32) -> Self {
            Self::bf16(name, &[1, 1], scale)
        }
    }

    /// Write a tiny Llama checkpoint with every projection weight stored as
    /// FP8 E4M3 + a 1x1 BF16 block scale (norms/embeddings stay BF16, as in
    /// real FP8 checkpoints).
    fn write_tiny_fp8_checkpoint(dir: &Path) {
        const SHARD: &str = "model-00001-of-00001.safetensors";
        let mut tensors = vec![
            RawTensor::bf16("model.embed_tokens.weight", &[8, 4], 0.02),
            RawTensor::bf16("model.norm.weight", &[4], 1.0),
        ];
        let prefix = "model.layers.0";
        for (suffix, shape) in [
            ("self_attn.q_proj.weight", vec![4, 4]),
            ("self_attn.k_proj.weight", vec![4, 4]),
            ("self_attn.v_proj.weight", vec![4, 4]),
            ("self_attn.o_proj.weight", vec![4, 4]),
            ("mlp.gate_proj.weight", vec![8, 4]),
            ("mlp.up_proj.weight", vec![8, 4]),
            ("mlp.down_proj.weight", vec![4, 8]),
        ] {
            tensors.push(RawTensor::f8(format!("{prefix}.{suffix}"), &shape));
            tensors.push(RawTensor::f8_scale(
                format!("{prefix}.{suffix}_scale_inv"),
                0.5,
            ));
        }
        tensors.push(RawTensor::bf16(
            format!("{prefix}.input_layernorm.weight"),
            &[4],
            1.0,
        ));
        tensors.push(RawTensor::bf16(
            format!("{prefix}.post_attention_layernorm.weight"),
            &[4],
            1.0,
        ));

        let views: Vec<_> = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).unwrap(),
                )
            })
            .collect();
        let bytes = serialize(views, &None).unwrap();
        std::fs::write(dir.join(SHARD), bytes).unwrap();

        let weight_map: BTreeMap<_, _> = tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), SHARD.to_owned()))
            .collect();
        let index = serde_json::json!({ "weight_map": weight_map });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();

        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "vocab_size": 8,
            "max_position_embeddings": 16,
            "rms_norm_eps": 0.00001
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn fp8_dequant_load_checkpoint_forward_matches_lazy_path() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_fp8_checkpoint(dir.path());

        let eager = ModelFactory::create(
            dir.path(),
            &Device::Cpu,
            DType::BFloat16,
            None,
            KVQuantType::None,
            None,
            true,
        )
        .await
        .unwrap();
        let lazy = ModelFactory::create(
            dir.path(),
            &Device::Cpu,
            DType::BFloat16,
            None,
            KVQuantType::None,
            None,
            false,
        )
        .await
        .unwrap();

        let input = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);
        let eager_logits = eager.forward(&input, None).unwrap().to_kind(DType::Float);
        let lazy_logits = lazy.forward(&input, None).unwrap().to_kind(DType::Float);
        assert_eq!(eager_logits.size(), lazy_logits.size());
        assert!(
            eager_logits.allclose(&lazy_logits, 1e-3, 1e-3, false),
            "checkpoint loaded with fp8_dequant_load diverged from lazy-dequant load"
        );
    }
}
