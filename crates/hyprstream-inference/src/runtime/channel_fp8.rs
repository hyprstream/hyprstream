//! Compressed-tensors FP8_DYNAMIC admission and CPU floating emulation.
//!
//! Scales are multipliers on stored E4M3FN values, never inverse scales or
//! 128x128 blocks. This profile intentionally does not use GPU scaled-mm.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tch::{Device, Kind, Tensor};

/// Recoverable FP8 execution failure exposed by the native model forward path.
#[derive(Debug)]
pub struct Fp8ExecutionError(pub(crate) String);
impl std::fmt::Display for Fp8ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FP8 execution failed: {}", self.0)
    }
}
impl std::error::Error for Fp8ExecutionError {}

#[derive(Debug)]
pub(crate) struct ChannelFp8Profile {
    ignore: Vec<String>,
}

impl ChannelFp8Profile {
    pub(crate) fn read(path: &Path) -> Result<Option<Self>> {
        let path = path.join("config.json");
        if !path.exists() {
            return Ok(None);
        }
        let config: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        Self::parse(&config)
    }

    pub(crate) fn parse(config: &Value) -> Result<Option<Self>> {
        let Some(q) = config.get("quantization_config").filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        if q["quant_method"] != "compressed-tensors" {
            return Ok(None);
        }
        ensure!(
            q["version"] == "0.19.0",
            "unsupported compressed-tensors producer version"
        );
        ensure!(
            q["format"] == "float-quantized" && q["quantization_status"] == "compressed",
            "unsupported compressed-tensors format/status"
        );
        ensure!(
            q["kv_cache_scheme"].is_null(),
            "compressed-tensors KV quantization unsupported"
        );
        for field in ["sparsity_config", "transform_config"] {
            ensure!(
                q[field].is_null() || q[field].as_object().is_some_and(serde_json::Map::is_empty),
                "unsupported compressed-tensors {field}"
            );
        }
        let groups = q["config_groups"]
            .as_object()
            .context("missing compressed-tensors config_groups")?;
        ensure!(
            groups.len() == 1,
            "exactly one FP8_DYNAMIC group is supported"
        );
        let group = groups.values().next().context("empty config_groups")?;
        ensure!(
            group.as_object().is_some_and(|g| g.keys().all(|k| matches!(
                k.as_str(),
                "format" | "targets" | "weights" | "input_activations" | "output_activations"
            ))),
            "unsupported compressed-tensors group field"
        );
        ensure!(
            group["format"] == "float-quantized"
                && group["targets"] == serde_json::json!(["Linear"]),
            "unsupported compressed-tensors targets/format"
        );
        ensure!(
            group["output_activations"].is_null(),
            "output activation quantization unsupported"
        );
        for (field, strategy, dynamic) in [
            ("weights", "channel", false),
            ("input_activations", "token", true),
        ] {
            let a = &group[field];
            ensure!(
                a["num_bits"] == 8
                    && a["type"] == "float"
                    && a["strategy"] == strategy
                    && a["symmetric"] == true
                    && a["dynamic"] == dynamic,
                "unsupported compressed-tensors {field} scheme"
            );
            for unsupported in [
                "group_size",
                "block_structure",
                "actorder",
                "scale_dtype",
                "zp_dtype",
            ] {
                ensure!(
                    a[unsupported].is_null(),
                    "unsupported {field}.{unsupported}"
                );
            }
        }
        let ignore = q["ignore"]
            .as_array()
            .context("missing ignore list")?
            .iter()
            .map(|v| {
                let s = v.as_str().context("ignore entry must be string")?;
                // Explicit paths only: accepting an unknown regex would silently change admission.
                ensure!(
                    !s.starts_with("re:") && !s.contains('*'),
                    "regex/glob ignore unsupported"
                );
                Ok(s.to_owned())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Self { ignore }))
    }

    /// Bound header reads and reject missing/ambiguous scale geometry before
    /// allocating tensor payloads, even when companions live in another shard.
    #[cfg(test)]
    fn preflight(&self, shards: &[PathBuf]) -> Result<()> {
        let mut resolved = shards
            .iter()
            .map(|path| {
                Ok((
                    path.clone(),
                    super::model_factory::ResolvedWeight::File(std::fs::File::open(path)?),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        self.preflight_resolved(&mut resolved)
    }

    pub(crate) fn preflight_resolved(
        &self,
        shards: &mut [(PathBuf, super::model_factory::ResolvedWeight)],
    ) -> Result<()> {
        let mut entries = HashMap::new();
        for (_, resolved) in shards {
            let object: serde_json::Map<String, Value> = resolved.with_reader(|reader| {
                let mut prefix = [0u8; 8];
                reader.read_exact(&mut prefix)?;
                let len = u64::from_le_bytes(prefix);
                ensure!(
                    len > 0 && len <= 16 * 1024 * 1024,
                    "invalid/oversized FP8 safetensors header"
                );
                let mut header = vec![0u8; len as usize];
                reader.read_exact(&mut header)?;
                Ok(serde_json::from_slice(&header)?)
            })?;
            for (name, tensor) in object {
                if name == "__metadata__" {
                    continue;
                }
                ensure!(
                    entries.insert(name.clone(), tensor).is_none(),
                    "duplicate tensor across shards: {name}"
                );
            }
        }
        let mut quantized = 0;
        for (name, tensor) in &entries {
            ensure!(
                !name.ends_with("_zero_point")
                    && !name.ends_with("_global_scale")
                    && !name.ends_with(".input_scale"),
                "unsupported FP8 quantization companion {name}"
            );
            if name.ends_with(".weight_scale") {
                ensure!(
                    entries.contains_key(name.trim_end_matches("_scale")),
                    "orphan FP8 scale {name}"
                );
            }
            let dtype = tensor["dtype"]
                .as_str()
                .context("tensor header lacks dtype")?;
            if !dtype.starts_with("F8_") {
                continue;
            }
            ensure!(
                dtype == "F8_E4M3",
                "FP8_DYNAMIC requires F8_E4M3, got {dtype} for {name}"
            );
            ensure!(name.ends_with(".weight"), "unexpected FP8 tensor {name}");
            let module = name.trim_end_matches(".weight");
            ensure!(!self.ignored(module), "ignored module is FP8: {name}");
            let shape = tensor["shape"].as_array().context("missing FP8 shape")?;
            ensure!(
                shape.len() == 2 && shape.iter().all(|d| d.as_u64().is_some_and(|v| v > 0)),
                "invalid FP8 matrix shape {name}"
            );
            let scale = entries
                .get(&format!("{name}_scale"))
                .context(format!("missing channel scale {name}"))?;
            ensure!(
                scale["shape"] == serde_json::json!([shape[0], 1]),
                "wrong channel-scale axis/shape for {name}"
            );
            ensure!(
                matches!(scale["dtype"].as_str(), Some("F32" | "BF16" | "F16")),
                "invalid channel scale dtype {name}"
            );
            ensure!(
                !entries.contains_key(&format!("{name}_scale_inv")),
                "ambiguous channel/block scale {name}"
            );
            quantized += 1;
        }
        ensure!(quantized > 0, "FP8_DYNAMIC checkpoint has no FP8 matrices");
        Ok(())
    }

    fn ignored(&self, module: &str) -> bool {
        self.ignore
            .iter()
            .any(|s| module == s || module.starts_with(&format!("{s}.")))
    }

    pub(crate) fn validate(&self, weights: &HashMap<String, Tensor>) -> Result<()> {
        let mut count = 0;
        for (name, weight) in weights {
            ensure!(
                !name.ends_with("_zero_point")
                    && !name.ends_with("_global_scale")
                    && !name.ends_with(".input_scale"),
                "unsupported FP8 quantization companion {name}"
            );
            if name.ends_with(".weight_scale") {
                let key = name.strip_suffix("_scale").context("invalid scale name")?;
                ensure!(weights.contains_key(key), "orphan FP8 scale {name}");
            }
            if !name.ends_with(".weight") {
                continue;
            }
            let module = name.trim_end_matches(".weight");
            let ignored = self.ignored(module);
            let fp8 = matches!(
                weight.kind(),
                Kind::Float8e4m3fn | Kind::Float8e5m2 | Kind::Float8e4m3fnuz | Kind::Float8e5m2fnuz
            );
            let scale = weights.get(&format!("{name}_scale"));
            if fp8 {
                ensure!(!ignored, "ignored module has FP8 weights: {name}");
                ensure!(
                    weight.kind() == Kind::Float8e4m3fn,
                    "FP8_DYNAMIC requires E4M3FN: {name}"
                );
                validate_channel_scale(
                    weight,
                    scale.context(format!("missing channel scale for {name}"))?,
                )?;
                ensure!(
                    !weights.contains_key(&format!("{name}_scale_inv")),
                    "ambiguous FP8 scales: {name}"
                );
                count += 1;
            } else {
                ensure!(scale.is_none(), "channel scale on floating weight {name}");
                // Only known text Linear sites are enforced. Embeddings, RMSNorm,
                // convolution kernels, vision, and the preserved head are not Linear targets.
                let text_linear = weight.dim() == 2
                    && module.contains(".layers.")
                    && (module.contains(".mlp.")
                        || module.contains(".self_attn.")
                        || module.contains(".linear_attn.in_proj")
                        || module.ends_with(".linear_attn.out_proj"));
                ensure!(
                    !text_linear || ignored,
                    "targeted Linear is not FP8: {name}"
                );
            }
        }
        ensure!(
            count > 0,
            "FP8_DYNAMIC checkpoint contains no quantized weights"
        );
        Ok(())
    }
}

pub(crate) fn validate_channel_scale(weight: &Tensor, scale: &Tensor) -> Result<()> {
    ensure!(
        weight.kind() == Kind::Float8e4m3fn,
        "channel FP8 requires E4M3FN"
    );
    let shape = weight.size();
    ensure!(
        shape.len() == 2 && shape.iter().all(|d| *d > 0),
        "channel FP8 weight must be nonempty [out,in]"
    );
    ensure!(
        scale.size() == [shape[0], 1],
        "channel scale must be [out,1], got {:?}",
        scale.size()
    );
    ensure!(
        matches!(scale.kind(), Kind::Float | Kind::BFloat16 | Kind::Half),
        "channel scale must be floating"
    );
    ensure!(
        scale.isfinite().all().int64_value(&[]) != 0 && scale.gt(0).all().int64_value(&[]) != 0,
        "channel scales must be finite and positive"
    );
    Ok(())
}

/// Dynamic symmetric token quantize/dequantize in the compute dtype.
///
/// Producer pin: compressed-tensors 0.19.0, commit
/// `525a7a7b84ebccdbc0db07956338b6f8c851d58f`:
/// `quantization/utils/helpers.py::calculate_qparams` divides amax by448 then
/// replaces **only zero** scales with dtype epsilon (lines80-139). No small-amax
/// floor is applied. `lifecycle/forward_helpers.py::_quantize_dequantize`
/// (lines168-204) divides, clamps/rounds to E4M3FN, then multiplies by the scale.
/// Source: <https://github.com/vllm-project/compressed-tensors/blob/525a7a7b84ebccdbc0db07956338b6f8c851d58f/src/compressed_tensors/quantization/utils/helpers.py>.
/// CPU FP32 is an explicit compute profile, not a claim of BF16 rounding parity. ATen's
/// E4M3FN cast supplies round-to-nearest-even, including subnormals. The producer
/// replaces *zero* scales with dtype epsilon; it does not clamp small scales.
pub(crate) fn dynamic_token_qdq(input: &Tensor) -> Result<Tensor> {
    ensure!(
        input.dim() >= 2 && input.size().last().is_some_and(|d| *d > 0),
        "FP8 input must contain nonempty token rows"
    );
    ensure!(
        matches!(input.kind(), Kind::Float | Kind::BFloat16 | Kind::Half),
        "unsupported FP8 activation compute dtype"
    );
    ensure!(
        input.isfinite().all().int64_value(&[]) != 0,
        "nonfinite FP8 activations"
    );
    let eps = match input.kind() {
        Kind::BFloat16 => 0.0078125,
        Kind::Half => 0.0009765625,
        _ => f32::EPSILON as f64,
    };
    let scale = input.abs().amax(-1, true) / 448.0;
    let scale = scale.where_self(&scale.ne(0), &scale.full_like(eps));
    let q = (input / &scale)
        .clamp(-448.0, 448.0)
        .to_kind(Kind::Float8e4m3fn);
    Ok(q.to_kind(input.kind()) * scale)
}

/// FP8 bytes stay resident. Each decoded scratch tile is at most K*256 floats;
/// output chunks are concatenated once, and no decoded weight is cached.
pub(crate) fn channel_matmul(input: &Tensor, weight: &Tensor, scale: &Tensor) -> Result<Tensor> {
    ensure!(
        weight.device() == Device::Cpu && input.device() == Device::Cpu,
        "channel FP8 execution is CPU-only until CPU acceptance"
    );
    let x = dynamic_token_qdq(input)?;
    let n = weight.size()[1];
    ensure!(
        input.size().last() == weight.size().first(),
        "channel FP8 input width mismatch"
    );
    let mut outputs = Vec::new();
    for offset in (0..n).step_by(256) {
        let width = (n - offset).min(256);
        let decoded = weight.narrow(1, offset, width).to_kind(input.kind())
            * scale.narrow(1, offset, width).to_kind(input.kind());
        ensure!(
            decoded.isfinite().all().int64_value(&[]) != 0,
            "nonfinite decoded FP8 weight tile"
        );
        let output = x.matmul(&decoded);
        ensure!(
            output.isfinite().all().int64_value(&[]) != 0,
            "nonfinite FP8 matmul output"
        );
        outputs.push(output);
    }
    if outputs.is_empty() {
        bail!("empty FP8 projection");
    }
    Ok(Tensor::cat(&outputs, -1))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::runtime::architectures::llama::LinearProjection;

    fn profile() -> Value {
        serde_json::json!({"quantization_config":{
            "quant_method":"compressed-tensors", "version":"0.19.0",
            "format":"float-quantized", "quantization_status":"compressed", "ignore":["lm_head"],
            "config_groups":{"g":{"format":"float-quantized", "targets":["Linear"],
                "weights":{"num_bits":8,"type":"float","strategy":"channel","symmetric":true,"dynamic":false},
                "input_activations":{"num_bits":8,"type":"float","strategy":"token","symmetric":true,"dynamic":true}}}
        }})
    }
    // Independent scalar E4M3FN oracle: no libtorch cast/scaling/matmul.
    fn decode(code: u8) -> f32 {
        let sign = if code & 128 == 0 { 1.0 } else { -1.0 };
        let exponent = (code >> 3) & 15;
        let mantissa = code & 7;
        if exponent == 15 && mantissa == 7 {
            return f32::NAN;
        }
        sign * if exponent == 0 {
            mantissa as f32 / 512.0
        } else {
            (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
        }
    }
    fn round_fp8(value: f32) -> f32 {
        let magnitude = value.abs().min(448.0);
        let mut code = 0;
        let mut distance = f32::INFINITY;
        for candidate in 0u8..=126 {
            let d = (decode(candidate) - magnitude).abs();
            if d < distance || (d == distance && candidate & 1 == 0) {
                code = candidate;
                distance = d;
            }
        }
        decode(code).copysign(value)
    }
    fn scalar_qdq(row: &[f32]) -> Vec<f32> {
        let max = row.iter().fold(0f32, |a, b| a.max(b.abs()));
        let mut scale = max / 448.0;
        if scale == 0.0 {
            scale = f32::EPSILON;
        }
        row.iter().map(|v| round_fp8(v / scale) * scale).collect()
    }
    fn raw(bytes: &[u8], shape: &[i64]) -> Tensor {
        unsafe { Tensor::from_blob(bytes.as_ptr(), shape, &[], Kind::Float8e4m3fn, Device::Cpu) }
            .copy()
    }
    fn values(t: &Tensor) -> Vec<f32> {
        Vec::<f32>::try_from(&t.to_kind(Kind::Float).reshape([-1])).unwrap()
    }
    fn close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() <= tol, "{a} != {b}");
        }
    }

    #[test]
    fn fp8_byte_patterns_match_independent_decoder() {
        let bytes: Vec<u8> = (0..=255).collect();
        let actual = values(&raw(&bytes, &[256]));
        for (byte, actual) in bytes.into_iter().zip(actual) {
            let expected = decode(byte);
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual, expected, "byte {byte}");
            }
        }
    }
    #[test]
    fn dynamic_qdq_matches_scalar_edges_and_noncontiguous_tokens() {
        let rows = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [448.0, 1.0625, 1.1875, -448.0, 0.0009765625],
            [1e-30, -1e-30, 0.0, 1e-32, -1e-32],
            [1e30, 1.0, -1e29, 0.0, 1e-5],
        ];
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let contiguous = Tensor::from_slice(&flat).reshape([4, 5]);
        let noncontiguous = contiguous.transpose(0, 1).contiguous().transpose(0, 1);
        assert!(!noncontiguous.is_contiguous());
        let expected: Vec<f32> = rows.iter().flat_map(|r| scalar_qdq(r)).collect();
        let actual = values(&dynamic_token_qdq(&noncontiguous).unwrap());
        for (a, b) in actual.into_iter().zip(expected) {
            assert!((a - b).abs() <= b.abs() * 1e-6 + 1e-35, "{a} != {b}");
        }
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                dynamic_token_qdq(&Tensor::from_slice(&[nonfinite, 1.0]).reshape([1, 2])).is_err()
            );
        }
    }
    #[test]
    fn activation_qdq_respects_compute_dtype_zero_policy_and_round_even() {
        for kind in [Kind::Float, Kind::BFloat16, Kind::Half] {
            let x = Tensor::from_slice(&[448f32, 1.0625, 1.1875, 0.0009765625, 0.0])
                .reshape([1, 5])
                .to_kind(kind);
            let y = dynamic_token_qdq(&x).unwrap();
            assert_eq!(y.kind(), kind);
            close(&values(&y), &[448.0, 1.0, 1.25, 0.0, 0.0], 0.0);
            let zeros = Tensor::zeros([2, 3], (kind, Device::Cpu));
            close(&values(&dynamic_token_qdq(&zeros).unwrap()), &[0.0; 6], 0.0);
        }
        let subnormal =
            Tensor::from_slice(&[f32::from_bits(1), -f32::from_bits(1)]).reshape([1, 2]);
        close(
            &values(&dynamic_token_qdq(&subnormal).unwrap()),
            &[0.0, 0.0],
            0.0,
        );
    }

    #[test]
    fn channel_projection_partial_tiles_scale_axis_and_qdq_match_scalar_math() {
        let n = 259usize; // two tiles, with a partial tail
        let k = 3usize;
        let bytes: Vec<u8> = (0..n * k)
            .map(|i| [0x38, 0xb8, 0x40, 0x30, 0x3c][i % 5])
            .collect();
        let scales: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32 / 16.0).collect();
        let input = [448f32, 1.0625, -7.0, -14.0, 0.03125, 2.0];
        let mut weights = HashMap::from([
            ("p.weight".to_owned(), raw(&bytes, &[n as i64, k as i64])),
            (
                "p.weight_scale".to_owned(),
                Tensor::from_slice(&scales).reshape([n as i64, 1]),
            ),
        ]);
        let projection = LinearProjection::take(&mut weights, "p.weight").unwrap();
        assert_eq!(projection.weight.kind(), Kind::Float8e4m3fn);
        assert!(projection.scale.is_none());
        let result = values(&projection.apply(&Tensor::from_slice(&input).reshape([2, k as i64])));
        let mut expected = Vec::new();
        for row in input.chunks(k) {
            let x = scalar_qdq(row);
            for out in 0..n {
                expected.push(
                    (0..k)
                        .map(|col| x[col] * decode(bytes[out * k + col]) * scales[out])
                        .sum(),
                );
            }
        }
        close(&result, &expected, 2e-4);
        // The first row includes a tie that activation QDQ changes.
        assert_ne!(scalar_qdq(&input[..k]), input[..k]);
    }
    #[test]
    fn channel_scale_admission_rejects_axis_nonfinite_zero_and_wrong_format() {
        let w = raw(&[0x38; 6], &[2, 3]);
        for s in [
            Tensor::ones([1, 2], (Kind::Float, Device::Cpu)),
            Tensor::ones([2], (Kind::Float, Device::Cpu)),
            Tensor::zeros([2, 1], (Kind::Float, Device::Cpu)),
            Tensor::from_slice(&[f32::NAN, 1.0]).reshape([2, 1]),
            Tensor::from_slice(&[-1f32, 1.0]).reshape([2, 1]),
        ] {
            assert!(validate_channel_scale(&w, &s).is_err());
        }
        assert!(validate_channel_scale(
            &w.to_kind(Kind::Float8e4m3fnuz),
            &Tensor::ones([2, 1], (Kind::Float, Device::Cpu))
        )
        .is_err());
        let mut missing = HashMap::from([("p.weight".to_owned(), w)]);
        assert!(LinearProjection::take(&mut missing, "p.weight").is_err());
    }
    #[test]
    fn checked_projection_returns_typed_execution_error_without_unwind() {
        let mut weights = HashMap::from([
            ("p.weight".to_owned(), raw(&[0x38; 6], &[2, 3])),
            (
                "p.weight_scale".to_owned(),
                Tensor::ones([2, 1], (Kind::Float, Device::Cpu)),
            ),
        ]);
        let p = LinearProjection::take(&mut weights, "p.weight").unwrap();
        let nonfinite = Tensor::from_slice(&[1f32, f32::NAN, 2.0]).reshape([1, 3]);
        let error = p.try_apply(&nonfinite).unwrap_err();
        assert!(error.downcast_ref::<Fp8ExecutionError>().is_some());
        let wrong_width = Tensor::ones([1, 2], (Kind::Float, Device::Cpu));
        assert!(p
            .try_apply(&wrong_width)
            .unwrap_err()
            .downcast_ref::<Fp8ExecutionError>()
            .is_some());
        // A rejected request does not poison or mutate the projection.
        assert!(p
            .try_apply(&Tensor::ones([1, 3], (Kind::Float, Device::Cpu)))
            .is_ok());
    }

    #[test]
    fn strict_profile_rejects_unsupported_schemes_and_missing_companions() {
        for (path, value) in [
            ("version", Value::from("unknown")),
            ("quantization_status", Value::from("frozen")),
        ] {
            let mut c = profile();
            c["quantization_config"][path] = value;
            assert!(ChannelFp8Profile::parse(&c).is_err());
        }
        for (path, value) in [
            ("strategy", Value::from("block")),
            ("dynamic", Value::from(true)),
            ("scale_dtype", Value::from("float8_e8m0fnu")),
        ] {
            let mut c = profile();
            c["quantization_config"]["config_groups"]["g"]["weights"][path] = value;
            assert!(ChannelFp8Profile::parse(&c).is_err());
        }
        let p = ChannelFp8Profile::parse(&profile()).unwrap().unwrap();
        let weights = HashMap::from([("p.weight".to_owned(), raw(&[0x38; 6], &[2, 3]))]);
        assert!(p.validate(&weights).is_err());
    }
    #[test]
    fn header_preflight_checks_cross_shard_scale_before_payload_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let p = ChannelFp8Profile::parse(&profile()).unwrap().unwrap();
        let write = |filename: &str, entry: Value| {
            let header = serde_json::to_vec(&entry).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            let path = dir.path().join(filename);
            std::fs::write(&path, bytes).unwrap();
            path
        };
        let weight = write(
            "a.safetensors",
            serde_json::json!({"p.weight":{"dtype":"F8_E4M3","shape":[2,3],"data_offsets":[0,6]}}),
        );
        let scale = write(
            "b.safetensors",
            serde_json::json!({"p.weight_scale":{"dtype":"F32","shape":[2,1],"data_offsets":[0,8]}}),
        );
        assert!(p.preflight(std::slice::from_ref(&weight)).is_err());
        p.preflight(&[weight.clone(), scale]).unwrap();
        let wrong = write(
            "c.safetensors",
            serde_json::json!({"p.weight_scale":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}),
        );
        assert!(p.preflight(&[weight, wrong]).is_err());
    }
}
