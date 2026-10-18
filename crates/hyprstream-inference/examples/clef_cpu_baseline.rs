//! Offline native CPU backbone receipt generator. Never downloads or serves.
//! Original FP32 runs require >=64 GiB MemAvailable, so they cannot run on the
//! 30 GiB development VM. The coordinator supplies pinned SHA256 manifests.
use anyhow::{anyhow, bail, ensure, Context, Result};
use hyprstream_inference::runtime::{
    architectures::ModelArchitecture, model_factory::ModelFactory, KVQuantType,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    collections::{BTreeSet, HashSet},
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};
use tch::{Device, Kind, Tensor};

const ORIGINAL: (&str, &str) = (
    "Cloudflare/clef-flash",
    "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
);
const FP8: (&str, &str) = (
    "kurcontko/clef-flash-FP8-Dynamic",
    "7b43c2d799e7b5e9fef96b6137840088d75af0d6",
);
const HELP: &str = "clef_cpu_baseline --model-dir DIR --manifest SHA256_MANIFEST_JSON --manifest-sha256 HEX --fixtures OFFICIAL_RECORDS_JSON --output NEW_DIR --source-sha DECLARED_GIT_SHA [--max-tokens 512] [--deadline-seconds 3600]\nCPU Float32 only; four Torch threads, no model downloads. Original needs64GiB MemAvailable; FP8 needs21GiB. CLI source-sha is declared metadata; coordinator build receipt must bind source/tree to executable SHA256.";

struct Args {
    model_dir: PathBuf,
    manifest: PathBuf,
    manifest_sha256: String,
    fixtures: PathBuf,
    output: PathBuf,
    declared_source_sha: String,
    max_tokens: usize,
    deadline: Duration,
}
impl Args {
    fn parse(values: impl IntoIterator<Item = String>) -> Result<Option<Self>> {
        let mut it = values.into_iter();
        let mut fields = std::collections::HashMap::new();
        while let Some(flag) = it.next() {
            if flag == "--help" {
                return Ok(None);
            }
            ensure!(
                matches!(
                    flag.as_str(),
                    "--model-dir"
                        | "--manifest"
                        | "--manifest-sha256"
                        | "--fixtures"
                        | "--output"
                        | "--source-sha"
                        | "--max-tokens"
                        | "--deadline-seconds"
                ),
                "unknown flag"
            );
            let value = it.next().context("missing flag value")?;
            ensure!(fields.insert(flag, value).is_none(), "duplicate flag");
        }
        let mut required = |key: &str| {
            fields
                .remove(key)
                .with_context(|| format!("required flag {key}"))
        };
        let model_dir = PathBuf::from(required("--model-dir")?);
        let manifest = PathBuf::from(required("--manifest")?);
        let manifest_sha256 = required("--manifest-sha256")?;
        let fixtures = PathBuf::from(required("--fixtures")?);
        let output = PathBuf::from(required("--output")?);
        let declared_source_sha = required("--source-sha")?;
        ensure!(
            hex_digest(&manifest_sha256, 64),
            "manifest SHA256 must be64hex characters"
        );
        ensure!(
            hex_digest(&declared_source_sha, 40),
            "declared source SHA must be40hex characters"
        );
        let max_tokens = fields
            .remove("--max-tokens")
            .unwrap_or_else(|| "512".to_owned())
            .parse()?;
        ensure!(
            (1..=1024).contains(&max_tokens),
            "max-tokens must be1..1024"
        );
        let deadline_seconds: u64 = fields
            .remove("--deadline-seconds")
            .unwrap_or_else(|| "3600".to_owned())
            .parse()?;
        ensure!(
            (1..=14400).contains(&deadline_seconds),
            "deadline must be1..14400seconds"
        );
        Ok(Some(Self {
            model_dir,
            manifest,
            manifest_sha256,
            fixtures,
            output,
            declared_source_sha,
            max_tokens,
            deadline: Duration::from_secs(deadline_seconds),
        }))
    }
}
fn hex_digest(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hex::encode(hash.finalize()))
}
fn bounded_read(path: &Path, max: u64) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.len() <= max,
        "metadata file exceeds byte limit"
    );
    let mut bytes = Vec::new();
    file.take(max + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= max,
        "metadata file grew beyond byte limit"
    );
    Ok(bytes)
}
fn relative_path(path: &str) -> Result<&Path> {
    let p = Path::new(path);
    ensure!(
        !p.as_os_str().is_empty() && p.components().all(|c| matches!(c, Component::Normal(_))),
        "manifest path must be relative without traversal"
    );
    Ok(p)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactFile {
    path: String,
    bytes: u64,
    sha256: String,
    verified: bool,
    // Original attestation metadata does not affect runtime verification.
    #[serde(default)]
    published_algorithm: Option<String>,
    #[serde(default)]
    published_pin: Option<String>,
}
#[derive(Deserialize)]
struct Manifest {
    repo: String,
    revision: String,
    files: Vec<ArtifactFile>,
}
impl Manifest {
    fn min_memory_gib(&self) -> Result<u64> {
        match (self.repo.as_str(), self.revision.as_str()) {
            pair if pair == ORIGINAL => Ok(64),
            pair if pair == FP8 => Ok(21),
            _ => bail!("manifest is not an admitted pinned Clef checkpoint"),
        }
    }
    fn verify(&self, model_dir: &Path) -> Result<Vec<serde_json::Value>> {
        ensure!(
            !self.files.is_empty() && self.files.len() <= 64,
            "manifest file count must be1..64"
        );
        let root = model_dir.canonicalize()?;
        let mut paths = HashSet::new();
        let mut total = 0u64;
        let mut records = Vec::new();
        for file in &self.files {
            let _ = &file.published_algorithm;
            let _ = &file.published_pin;
            let relative = relative_path(&file.path)?;
            ensure!(paths.insert(file.path.as_str()), "duplicate artifact path");
            ensure!(
                file.verified && hex_digest(&file.sha256, 64),
                "unverified or missing artifactSHA256"
            );
            total = total
                .checked_add(file.bytes)
                .context("artifact size overflow")?;
            ensure!(
                total <= 64 * 1024 * 1024 * 1024,
                "artifact exceeds64GiB byte limit"
            );
            let path = root.join(relative).canonicalize()?;
            ensure!(
                path.starts_with(&root),
                "artifact symlink escapes model root"
            );
            ensure!(
                path.metadata()?.is_file() && path.metadata()?.len() == file.bytes,
                "artifact size mismatch for {}",
                file.path
            );
            let actual = sha256(&path)?;
            ensure!(
                actual.eq_ignore_ascii_case(&file.sha256),
                "artifact digest mismatch for {}",
                file.path
            );
            records.push(serde_json::json!({"path":file.path,"bytes":file.bytes,"sha256":actual}));
        }
        ensure!(
            paths.contains("config.json"),
            "manifest must bind config.json"
        );
        let index = model_dir.join("model.safetensors.index.json");
        if index.exists() {
            ensure!(
                paths.contains("model.safetensors.index.json"),
                "manifest must bind shard index"
            );
            let index: serde_json::Value =
                serde_json::from_slice(&bounded_read(&index, 4 * 1024 * 1024)?)?;
            for shard in index["weight_map"]
                .as_object()
                .context("missing shard weight_map")?
                .values()
            {
                let name = shard.as_str().context("shard name must be a string")?;
                relative_path(name)?;
                ensure!(
                    paths.contains(name),
                    "loader-selected shard is not hash-bound"
                );
            }
        } else {
            ensure!(
                paths.contains("model.safetensors"),
                "manifest must bind model.safetensors"
            );
        }
        let config: serde_json::Value =
            serde_json::from_slice(&bounded_read(&model_dir.join("config.json"), 1024 * 1024)?)?;
        ensure!(
            config["model_type"] == "qwen3_5",
            "baseline requires Qwen3.5 backbone"
        );
        if self.repo == ORIGINAL.0 {
            ensure!(
                config
                    .get("quantization_config")
                    .is_none_or(serde_json::Value::is_null),
                "original baseline cannot have a quantization profile"
            );
        } else {
            ensure!(
                config["quantization_config"]["quant_method"] == "compressed-tensors",
                "FP8 baseline requires compressed-tensors profile"
            );
        }
        Ok(records)
    }
}
#[derive(Deserialize)]
struct Corpus {
    corpus_version: String,
    records: Vec<Record>,
}
#[derive(Deserialize)]
struct Record {
    input_ids: Vec<i64>,
    questions: Vec<Question>,
}
#[derive(Deserialize)]
struct Question {
    option_spans: Vec<[usize; 2]>,
}
impl Corpus {
    fn validate(&self, max_tokens: usize) -> Result<Vec<i64>> {
        ensure!(
            (2..=32).contains(&self.records.len()),
            "fixture needs2..32records for request isolation"
        );
        ensure!(
            self.corpus_version.len() <= 128,
            "corpus version exceeds limit"
        );
        let mut selected = BTreeSet::new();
        let mut tokens = 0usize;
        for record in &self.records {
            ensure!(
                !record.input_ids.is_empty() && record.input_ids.len() <= max_tokens,
                "fixture token count outside bound"
            );
            ensure!(
                record.input_ids.iter().all(|id| *id >= 0 && *id < 248320),
                "fixture token ID outside pinned vocabulary"
            );
            tokens += record.input_ids.len();
            ensure!(tokens <= 8192, "aggregate fixture token limit8192");
            ensure!(
                !record.questions.is_empty() && record.questions.len() <= 64,
                "fixture question count outside bound"
            );
            for q in &record.questions {
                ensure!(
                    !q.option_spans.is_empty() && q.option_spans.len() <= 256,
                    "fixture option count outside bound"
                );
                for [a, b] in &q.option_spans {
                    ensure!(
                        a < b && *b <= record.input_ids.len(),
                        "invalid option-token span"
                    );
                    selected.extend(record.input_ids[*a..*b].iter().copied());
                }
            }
        }
        ensure!(
            !selected.is_empty() && selected.len() <= 8192,
            "selected LM row count outside bound"
        );
        Ok(selected.into_iter().collect())
    }
}
fn ids_sha256(ids: &[i64]) -> String {
    let mut h = Sha256::new();
    h.update(b"clef-native-input-ids/v1\0");
    h.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        h.update(id.to_le_bytes());
    }
    hex::encode(h.finalize())
}
fn memory_kib() -> Result<u64> {
    fs::read_to_string("/proc/meminfo")?
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .context("LinuxMemAvailable admission required")
}
fn admit_memory(available_kib: u64, min_gib: u64) -> Result<()> {
    ensure!(
        available_kib >= min_gib * 1024 * 1024,
        "nativeCPU baseline requires{min_gib}GiB MemAvailable; observed{available_kib}KiB"
    );
    Ok(())
}
fn finite(tensor: &Tensor) -> Result<()> {
    ensure!(
        tensor.device() == Device::Cpu
            && tensor.kind() == Kind::Float
            && tensor.isfinite().all().int64_value(&[]) != 0,
        "baseline must produce finite Float32 CPU tensors"
    );
    Ok(())
}
fn max_diff(a: &Tensor, b: &Tensor) -> Result<f64> {
    ensure!(a.size() == b.size(), "repeat hidden-state shape mismatch");
    finite(b)?;
    let diff = (a - b).abs().max().double_value(&[]);
    ensure!(
        a.allclose(b, 1e-6, 1e-6, false),
        "stateless request repeat/isolation mismatch"
    );
    Ok(diff)
}
#[derive(Serialize)]
struct RecordReceipt {
    index: usize,
    token_count: usize,
    input_ids_sha256: String,
    hidden_shape: Vec<i64>,
    elapsed_ms: u128,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        use std::io::Write;
        writeln!(std::io::stdout().lock(), "{HELP}")?;
        return Ok(());
    };
    let started = Instant::now();
    let check = || {
        ensure!(
            started.elapsed() < args.deadline,
            "CPU baseline cooperative deadline exceeded"
        );
        Ok(())
    };
    let manifest_bytes = bounded_read(&args.manifest, 4 * 1024 * 1024)?;
    let manifest_sha = hex::encode(Sha256::digest(&manifest_bytes));
    ensure!(
        manifest_sha.eq_ignore_ascii_case(&args.manifest_sha256),
        "manifest SHA256 differs from coordinator pin"
    );
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    let min_gib = manifest.min_memory_gib()?;
    admit_memory(memory_kib()?, min_gib)?; // before reading any model payload
    let fixture_bytes = bounded_read(&args.fixtures, 2 * 1024 * 1024)?;
    let fixture_sha = hex::encode(Sha256::digest(&fixture_bytes));
    let corpus: Corpus = serde_json::from_slice(&fixture_bytes)?;
    let selected = corpus.validate(args.max_tokens)?;
    ensure!(
        !args.output.exists(),
        "output must be a new directory; preserve previous evidence"
    );
    let verified_files = manifest.verify(&args.model_dir)?;
    check()?;
    let admission_available_kib = memory_kib()?;
    admit_memory(admission_available_kib, min_gib)?;
    // This process is an offline single-model CPU runner. Force mmap for original
    // and FP8 profiles without changing any production loader/default behavior.
    std::env::set_var("HYPRSTREAM_USE_MMAP", "1");
    tch::set_num_threads(4);
    tch::set_num_interop_threads(1);
    let load_start = Instant::now();
    let model = ModelFactory::create(
        &args.model_dir,
        &Device::Cpu,
        Kind::Float,
        Some(args.max_tokens),
        KVQuantType::None,
        None,
        false,
        false,
    )
    .await?;
    let load_ms = load_start.elapsed().as_millis();
    ensure!(
        matches!(model.architecture(), ModelArchitecture::Qwen3_5),
        "unexpected backbone architecture"
    );
    let _no_grad = tch::no_grad_guard();
    let rows = model.decision_output_embedding_rows(&selected)?;
    finite(&rows)?;
    ensure!(
        rows.size() == [selected.len() as i64, 4096],
        "selected LM rows have unexpected shape"
    );
    let mut tensors = vec![
        (
            "embedding_token_ids".to_owned(),
            Tensor::from_slice(&selected),
        ),
        ("embedding_rows".to_owned(), rows),
    ];
    let mut receipts = Vec::new();
    for (i, record) in corpus.records.iter().enumerate() {
        check()?;
        let input =
            Tensor::from_slice(&record.input_ids).reshape([1, record.input_ids.len() as i64]);
        let forward_start = Instant::now();
        let hidden = model.decision_hidden_states_checked(&input, &check)?;
        finite(&hidden)?;
        ensure!(
            hidden.size() == [record.input_ids.len() as i64, 4096],
            "unexpected all-token hidden shape"
        );
        receipts.push(RecordReceipt {
            index: i,
            token_count: record.input_ids.len(),
            input_ids_sha256: ids_sha256(&record.input_ids),
            hidden_shape: hidden.size(),
            elapsed_ms: forward_start.elapsed().as_millis(),
        });
        tensors.push((format!("r{i}.hidden"), hidden.contiguous()));
    }
    let a = &corpus.records[0].input_ids;
    let b = &corpus.records[1].input_ids;
    let input_a = Tensor::from_slice(a).reshape([1, a.len() as i64]);
    let input_b = Tensor::from_slice(b).reshape([1, b.len() as i64]);
    // Mutate persistent generation state with another fixture record, avoiding
    // vocabulary projection. Fresh decision states must remain independent.
    check()?;
    let embedded = model.embed_tokens(&input_b)?;
    // Contaminate actual persistent KV/GDN state, with checks between layers.
    let mut hidden = embedded;
    for layer in 0..model.num_layers() {
        check()?;
        hidden = model.forward_layers(&hidden, layer..layer + 1, 0, None)?;
    }
    let repeat = model.decision_hidden_states_checked(&input_a, &check)?;
    let first = &tensors[2].1;
    let isolation_max_abs = max_diff(first, &repeat)?;
    let checks = Cell::new(0usize);
    let cancel = || {
        check()?;
        let n = checks.get() + 1;
        checks.set(n);
        if n == 3 {
            Err(anyhow!("baseline cancellation witness"))
        } else {
            Ok(())
        }
    };
    let cancelled = model
        .decision_hidden_states_checked(&input_a, &cancel)
        .err()
        .context("cooperative cancellation did not fire")?;
    ensure!(
        checks.get() == 3
            && cancelled
                .to_string()
                .contains("baseline cancellation witness"),
        "unexpected cancellation result"
    );
    let recovered = model.decision_hidden_states_checked(&input_a, &check)?;
    let cancel_recovery_max_abs = max_diff(first, &recovered)?;
    check()?;
    fs::create_dir(&args.output)?;
    Tensor::write_safetensors(&tensors, args.output.join("tensors.safetensors"))?;
    let status = fs::read_to_string("/proc/self/status")?;
    let executable_sha256 = sha256(&std::env::current_exe()?)?;
    let receipt = serde_json::json!({
        "receipt_version":"clef-native-backbone-cpu/v1","result":"PASS","repo":manifest.repo,"revision":manifest.revision,
        "declared_source_sha":args.declared_source_sha,"source_attestation":"caller-declared; coordinator build receipt must bind source/tree to executableSHA256",
        "executable_sha256":executable_sha256,"manifest_sha256":manifest_sha,"fixture_sha256":fixture_sha,
        "corpus_version":corpus.corpus_version,"device":"cpu","compute_dtype":"float32","mmap":true,
        "torch_threads":4,"torch_interop_threads":1,"max_tokens":args.max_tokens,"aggregate_token_limit":8192,
        "memory_admission_min_gib":min_gib,"admission_available_kib":admission_available_kib,"load_ms":load_ms,"elapsed_ms":started.elapsed().as_millis(),
        "records":receipts,"selected_output_lm_rows":selected.len(),"selected_ids_sha256":ids_sha256(&selected),
        "isolation_max_abs":isolation_max_abs,"cancel_checks":checks.get(),"cancel_recovery_max_abs":cancel_recovery_max_abs,
        "process_memory":status.lines().filter(|line|["VmHWM:","VmRSS:","VmSwap:","Threads:"].iter().any(|key|line.starts_with(key))).collect::<Vec<_>>(),
        "verified_artifact_files":verified_files,"tensors_sha256":sha256(&args.output.join("tensors.safetensors"))?,
        "scope":"native final-backbone hidden states and original outputLM rows only; no head/calibration/serving/upstream parity claim",
        "reference_warning":"synthetic head fixture hidden states are not a backbone oracle"
    });
    fs::write(
        args.output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    fn corpus() -> Corpus {
        serde_json::from_str(r#"{"corpus_version":"fixture-v1","records":[{"input_ids":[3,1,2],"questions":[{"option_spans":[[0,2],[1,3]]}]},{"input_ids":[4,2],"questions":[{"option_spans":[[0,2]]}]}]}"#).unwrap()
    }
    #[test]
    fn selected_output_rows_preserve_sorted_unique_token_ids() {
        assert_eq!(corpus().validate(3).unwrap(), [1, 2, 3, 4]);
    }
    #[test]
    fn input_admission_rejects_bad_spans_bounds_and_vocabulary() {
        let mut c = corpus();
        c.records[0].questions[0].option_spans[0] = [0, 4];
        assert!(c.validate(3).is_err());
        let mut c = corpus();
        c.records[0].input_ids[0] = -1;
        assert!(c.validate(3).is_err());
        assert!(corpus().validate(2).is_err());
    }
    #[test]
    fn original_memory_gate_cannot_be_overridden_on_small_vm() {
        let m = Manifest {
            repo: ORIGINAL.0.to_owned(),
            revision: ORIGINAL.1.to_owned(),
            files: vec![],
        };
        assert_eq!(m.min_memory_gib().unwrap(), 64);
        assert!(admit_memory(30 * 1024 * 1024, m.min_memory_gib().unwrap()).is_err());
        assert!(admit_memory(64 * 1024 * 1024, m.min_memory_gib().unwrap()).is_ok());
    }
    #[test]
    fn unknown_checkpoint_and_manifest_traversal_are_rejected() {
        let m = Manifest {
            repo: ORIGINAL.0.to_owned(),
            revision: "main".to_owned(),
            files: vec![],
        };
        assert!(m.min_memory_gib().is_err());
        for p in ["../weights", "/weights", "model/../../weights", ""] {
            assert!(relative_path(p).is_err());
        }
        assert!(relative_path("model-00001-of-00004.safetensors").is_ok());
    }
    #[test]
    fn input_hash_binds_order_length_and_values() {
        assert_ne!(ids_sha256(&[1, 2]), ids_sha256(&[2, 1]));
        assert_ne!(ids_sha256(&[1]), ids_sha256(&[1, 0]));
    }
    #[test]
    fn cli_rejects_missing_and_unbounded_values() {
        assert!(Args::parse(vec!["--unknown".to_owned(), "x".to_owned()]).is_err());
        assert!(Args::parse(Vec::<String>::new()).is_err());
        assert!(Args::parse(vec!["--help".to_owned()]).unwrap().is_none());
    }
}
