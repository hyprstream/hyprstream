//! Protected startup selection for one native model. No RPC mutation path.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelOperation {
    Generation,
    Decisions,
}
impl ModelOperation {
    pub(crate) fn action(self) -> &'static str {
        match self {
            Self::Generation => "infer",
            Self::Decisions => "decide",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAdmissionConfig {
    pub model_ref: String,
    pub commit: String,
    pub tree: String,
    pub operation: ModelOperation,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub local_lfs_directory: PathBuf,
    pub owner_subjects: Vec<String>,
    #[serde(skip)]
    captured: OnceLock<Result<ValidatedModelAdmission, String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidatedModelAdmission {
    pub(crate) model_ref: String,
    pub(crate) repository_name: String,
    pub(crate) commit: git2::Oid,
    pub(crate) tree: git2::Oid,
    pub(crate) operation: ModelOperation,
    pub(crate) manifest_sha256: String,
    pub(crate) local_lfs_directory: PathBuf,
    pub(crate) owner_subjects: Vec<String>,
    pub(crate) manifest: ModelManifest,
}

// This is the existing complete image inventory format, not another profile.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub(crate) struct ModelManifest {
    schema: String,
    repository: String,
    pub(crate) commit: String,
    pub(crate) tree: String,
    profile: String,
    compute_profile: String,
    quantization: String,
    directory: String,
    pub(crate) expanded_bytes: u64,
    pub(crate) files: Vec<ManifestFile>,
    pub(crate) lfs_paths: Vec<String>,
    metadata_bundle: MetadataBundle,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifestFile {
    pub(crate) path: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetadataBundle {
    branch: String,
    bytes: u64,
    sha256: String,
}

fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn relative(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && Path::new(value)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}
fn regular(path: &Path) -> Result<std::fs::Metadata> {
    ensure!(
        path.is_absolute() && path.canonicalize()?.as_path() == path,
        "admission path must be absolute and canonical"
    );
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        metadata.file_type().is_file(),
        "admission input must be a regular non-symlink file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        ensure!(
            metadata.permissions().mode() & 0o022 == 0,
            "admission input must not be group/world writable"
        );
    }
    Ok(metadata)
}

impl ModelAdmissionConfig {
    /// Capture the independently pinned bytes once, before startup publishes
    /// the immutable HyprConfig. Clones retain this same validated selection.
    pub(crate) fn validated(&self) -> Result<&ValidatedModelAdmission> {
        self.captured
            .get_or_init(|| self.capture().map_err(|e| format!("{e:#}")))
            .as_ref()
            .map_err(|e| anyhow::anyhow!("invalid model_admission: {e}"))
    }
    fn capture(&self) -> Result<ValidatedModelAdmission> {
        let parsed = crate::storage::ModelRef::parse(&self.model_ref)?;
        ensure!(
            matches!(parsed.git_ref, crate::storage::GitRef::Branch(_))
                && parsed.to_string() == self.model_ref,
            "model_admission requires a canonical local branch ref"
        );
        ensure!(
            hex(&self.commit, 40) && hex(&self.tree, 40) && hex(&self.manifest_sha256, 64),
            "model_admission requires full lowercase commit/tree/SHA256 pins"
        );
        ensure!(
            !self.owner_subjects.is_empty() && self.owner_subjects.len() <= 16,
            "model_admission requires 1..16 exact owner subjects"
        );
        let mut owners = BTreeSet::new();
        for owner in &self.owner_subjects {
            ensure!(
                owner.starts_with("did:")
                    && owner.len() <= 256
                    && owner
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b":._%-".contains(&c))
                    && owners.insert(owner),
                "invalid or duplicate exact owner DID"
            );
        }
        ensure!(
            regular(&self.manifest_path)?.len() <= 64 * 1024,
            "model manifest exceeds bound"
        );
        use std::io::Read as _;
        let mut bytes = Vec::new();
        std::fs::File::open(&self.manifest_path)?
            .take(64 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= 64 * 1024
                && format!("{:x}", Sha256::digest(&bytes)) == self.manifest_sha256,
            "model manifest digest does not match protected config"
        );
        let manifest: ModelManifest = serde_json::from_slice(&bytes)?;
        ensure!(
            matches!(
                manifest.schema.as_str(),
                "hyprstream-clef-native-model/1" | "hyprstream-native-model/1"
            ),
            "unsupported deployed model manifest version"
        );
        ensure!(
            manifest.commit == self.commit && manifest.tree == self.tree,
            "manifest commit/tree differs from protected config"
        );
        ensure!(
            !manifest.repository.is_empty()
                && !manifest.profile.is_empty()
                && !manifest.compute_profile.is_empty()
                && !manifest.quantization.is_empty()
                && relative(&manifest.directory),
            "incomplete deployed model manifest"
        );
        ensure!(
            !manifest.files.is_empty() && manifest.files.len() <= 128,
            "model manifest file count exceeds bound"
        );
        let mut paths = BTreeSet::new();
        let mut total = 0u64;
        for file in &manifest.files {
            ensure!(
                relative(&file.path)
                    && paths.insert(file.path.as_str())
                    && hex(&file.sha256, 64)
                    && file.bytes > 0,
                "invalid or duplicate model inventory entry"
            );
            total = total
                .checked_add(file.bytes)
                .context("model manifest size overflow")?;
        }
        ensure!(
            total == manifest.expanded_bytes,
            "model manifest expanded size differs from file sum"
        );
        for required in ["config.json", "tokenizer.json", "model.safetensors"] {
            ensure!(
                paths.contains(required),
                "model manifest missing required {required}"
            );
        }
        if self.operation == ModelOperation::Decisions {
            for required in [
                "clef_head/joint_head.safetensors",
                "clef_head/joint_head_config.json",
            ] {
                ensure!(
                    paths.contains(required),
                    "decision model missing supported original joint head {required}"
                );
            }
        }
        ensure!(
            manifest.lfs_paths.len() <= manifest.files.len(),
            "invalid LFS inventory count"
        );
        let mut lfs = BTreeSet::new();
        for path in &manifest.lfs_paths {
            ensure!(
                paths.contains(path.as_str()) && lfs.insert(path),
                "unknown or duplicate LFS inventory path"
            );
        }
        ensure!(
            hex(&manifest.metadata_bundle.sha256, 64)
                && manifest.metadata_bundle.bytes > 0
                && !manifest.metadata_bundle.branch.is_empty(),
            "invalid metadata archive identity"
        );
        ensure!(
            self.local_lfs_directory.is_absolute(),
            "local LFS directory must be absolute"
        );
        Ok(ValidatedModelAdmission {
            model_ref: self.model_ref.clone(),
            repository_name: parsed.name().to_owned(),
            commit: git2::Oid::from_str(&self.commit)?,
            tree: git2::Oid::from_str(&self.tree)?,
            operation: self.operation,
            manifest_sha256: self.manifest_sha256.clone(),
            local_lfs_directory: self.local_lfs_directory.clone(),
            owner_subjects: self.owner_subjects.clone(),
            manifest,
        })
    }
}

impl ValidatedModelAdmission {
    pub(crate) fn owner_rules(&self) -> [(String, &'static str); 4] {
        [
            (format!("registry:{}", self.repository_name), "query"),
            ("model:Load".into(), "write"),
            ("model:Status".into(), "query"),
            (format!("model:{}", self.model_ref), self.operation.action()),
        ]
    }
    pub(crate) fn local_lfs(&self) -> Result<BTreeMap<String, PathBuf>> {
        let directory = &self.local_lfs_directory;
        ensure!(
            directory.canonicalize()?.as_path() == directory
                && std::fs::symlink_metadata(directory)?.file_type().is_dir(),
            "local LFS directory must be canonical and non-symlink"
        );
        let mut expected = BTreeMap::new();
        for path in &self.manifest.lfs_paths {
            let file = self
                .manifest
                .files
                .iter()
                .find(|file| &file.path == path)
                .context("LFS member missing from captured inventory")?;
            let object = directory.join(&file.sha256);
            ensure!(
                regular(&object)?.len() == file.bytes,
                "local LFS object size differs from captured inventory"
            );
            ensure!(
                expected.insert(file.sha256.clone(), object).is_none(),
                "duplicate LFS object SHA256"
            );
        }
        let mut observed = BTreeSet::new();
        for entry in std::fs::read_dir(directory)? {
            let name = entry?
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF8 local LFS entry"))?;
            ensure!(
                expected.contains_key(&name) && observed.insert(name),
                "unexpected local LFS object"
            );
        }
        ensure!(
            observed.len() == expected.len(),
            "local LFS inventory incomplete"
        );
        Ok(expected)
    }
}

impl ValidatedModelAdmission {
    /// Check the complete immutable Git inventory before payload acquisition.
    /// The existing projection still hashes every actual LFS payload itself.
    pub(crate) fn verify_tree(&self, checkout: &Path) -> Result<()> {
        let repository = git2::Repository::open(checkout)?;
        let tree = repository.find_commit(self.commit)?.tree()?;
        ensure!(
            tree.id() == self.tree,
            "configured commit resolves to a different tree"
        );
        let expected: BTreeMap<_, _> = self
            .manifest
            .files
            .iter()
            .map(|file| (file.path.as_str(), file))
            .collect();
        let lfs: BTreeSet<_> = self.manifest.lfs_paths.iter().map(String::as_str).collect();
        let mut observed = BTreeSet::new();
        let mut failure = None;
        let walked = tree.walk(git2::TreeWalkMode::PreOrder, |prefix, entry| {
            let check = || -> Result<Option<String>> {
                if entry.kind() == Some(git2::ObjectType::Tree) {
                    return Ok(None);
                }
                ensure!(
                    entry.kind() == Some(git2::ObjectType::Blob)
                        && matches!(entry.filemode(), 0o100644 | 0o100755),
                    "unsupported entry in configured model tree"
                );
                let path = format!(
                    "{prefix}{}",
                    entry.name().context("non-UTF8 model tree name")?
                );
                let file = expected
                    .get(path.as_str())
                    .context("unexpected file in configured model tree")?;
                let (size, _) = repository.odb()?.read_header(entry.id())?;
                ensure!(
                    size <= if lfs.contains(path.as_str()) {
                        4096
                    } else {
                        usize::try_from(file.bytes)?
                    },
                    "model metadata blob exceeds bound"
                );
                let blob = repository.find_blob(entry.id())?;
                if lfs.contains(path.as_str()) {
                    let pointer = git2db::LfsPointer::parse(std::str::from_utf8(blob.content())?)?;
                    ensure!(
                        pointer.hash_method() == "sha256"
                            && pointer.oid() == file.sha256
                            && pointer.size() == file.bytes,
                        "Git LFS pointer differs from protected inventory"
                    );
                } else {
                    ensure!(
                        blob.content().len() as u64 == file.bytes
                            && format!("{:x}", Sha256::digest(blob.content())) == file.sha256,
                        "Git metadata differs from protected inventory"
                    );
                }
                Ok(Some(path))
            };
            match check() {
                Ok(Some(path)) => {
                    observed.insert(path);
                    git2::TreeWalkResult::Ok
                }
                Ok(None) => git2::TreeWalkResult::Ok,
                Err(error) => {
                    failure = Some(error);
                    git2::TreeWalkResult::Abort
                }
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
        walked?;
        ensure!(
            observed.len() == expected.len(),
            "configured model tree inventory is incomplete"
        );
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn tiny_fixture(
    operation: ModelOperation,
    name: &str,
    owner: &str,
) -> Result<(tempfile::TempDir, super::HyprConfig)> {
    let root = tempfile::tempdir()?;
    let lfs = root.path().join("lfs");
    std::fs::create_dir(&lfs)?;
    let mut paths = vec!["config.json", "tokenizer.json", "model.safetensors"];
    if operation == ModelOperation::Decisions {
        paths.extend([
            "clef_head/joint_head.safetensors",
            "clef_head/joint_head_config.json",
        ]);
    }
    let files: Vec<_> = paths.iter().map(|path| serde_json::json!({"path":path,"bytes":2,"sha256":format!("{:x}",Sha256::digest(b"{}"))})).collect();
    let manifest = serde_json::json!({"schema":"hyprstream-native-model/1","repository":"fixture/model","commit":"a".repeat(40),"tree":"b".repeat(40),"profile":"fixture-native","computeProfile":"cpu","quantization":"float32","directory":"fixture","expandedBytes":files.len()*2,"files":files,"lfsPaths":[],"metadataBundle":{"branch":"fixture","bytes":1,"sha256":"c".repeat(64)}});
    let bytes = serde_json::to_vec(&manifest)?;
    let path = root.path().join("manifest.json");
    std::fs::write(&path, &bytes)?;
    let config = serde_json::from_value(
        serde_json::json!({"model_admission":{"model_ref":name,"commit":"a".repeat(40),"tree":"b".repeat(40),"operation":operation,"manifest_path":path,"manifest_sha256":format!("{:x}",Sha256::digest(&bytes)),"local_lfs_directory":lfs,"owner_subjects":[owner]}}),
    )?;
    Ok((root, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distinct_selected_names_and_closed_operations_need_no_source_enum() -> Result<()> {
        for name in ["fixture-one:main", "fixture-two:main"] {
            for operation in [ModelOperation::Generation, ModelOperation::Decisions] {
                let (_root, config) = tiny_fixture(operation, name, "did:plc:fixtureowner")?;
                let selected = config.selected_model_admission()?.context("selection")?;
                assert_eq!(selected.model_ref, name);
                assert_eq!(selected.owner_rules().len(), 4);
                assert_eq!(
                    selected.owner_rules()[3],
                    (format!("model:{name}"), operation.action())
                );
                assert!(selected.local_lfs()?.is_empty());
            }
        }
        Ok(())
    }
    #[test]
    fn bad_digest_duplicate_inventory_and_unsupported_operation_fail_closed() -> Result<()> {
        for kind in ["digest", "duplicate", "path", "size", "head"] {
            let (_root, mut config) = tiny_fixture(
                ModelOperation::Decisions,
                "fixture:main",
                "did:plc:fixtureowner",
            )?;
            let selected = config.model_admission.as_mut().context("record")?;
            let mut manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&selected.manifest_path)?)?;
            match kind {
                "digest" => selected.manifest_sha256 = "d".repeat(64),
                "duplicate" => {
                    let file = manifest["files"][0].clone();
                    manifest["files"]
                        .as_array_mut()
                        .context("files")?
                        .push(file);
                }
                "path" => manifest["files"][0]["path"] = "../config.json".into(),
                "size" => manifest["expandedBytes"] = 999.into(),
                "head" => {
                    manifest["files"].as_array_mut().context("files")?.pop();
                }
                _ => unreachable!(),
            }
            if kind != "digest" {
                let bytes = serde_json::to_vec(&manifest)?;
                std::fs::write(&selected.manifest_path, &bytes)?;
                selected.manifest_sha256 = format!("{:x}", Sha256::digest(&bytes));
            }
            assert!(
                config.selected_model_admission().is_err(),
                "accepted {kind}"
            );
        }
        assert!(serde_json::from_str::<ModelOperation>("\"python\"").is_err());
        assert!(serde_json::from_str::<MetadataBundle>(
            "{\"branch\":\"a\",\"branch\":\"b\",\"bytes\":1,\"sha256\":\"x\"}"
        )
        .is_err());
        Ok(())
    }
    #[test]
    fn extra_hydration_symlinks_and_owner_patterns_are_rejected() -> Result<()> {
        let (_root, config) = tiny_fixture(
            ModelOperation::Generation,
            "fixture:main",
            "did:plc:fixtureowner",
        )?;
        let selected = config.selected_model_admission()?.context("selection")?;
        std::fs::write(selected.local_lfs_directory.join("extra"), b"x")?;
        assert!(selected.local_lfs().is_err());
        let (_root, config) =
            tiny_fixture(ModelOperation::Generation, "fixture:main", "did:plc:*")?;
        assert!(config.selected_model_admission().is_err());
        Ok(())
    }
    #[test]
    fn actual_git_inventory_binds_complete_tree_and_rejects_substitution() -> Result<()> {
        let (root, mut config) = tiny_fixture(
            ModelOperation::Generation,
            "fixture:main",
            "did:plc:fixtureowner",
        )?;
        let directory = root.path().join("tree");
        std::fs::create_dir(&directory)?;
        let repository = git2::Repository::init(&directory)?;
        for path in ["config.json", "tokenizer.json", "model.safetensors"] {
            std::fs::write(directory.join(path), b"{}")?;
        }
        let mut index = repository.index()?;
        index.add_all(["*"], git2::IndexAddOption::DEFAULT, None)?;
        let tree_id = index.write_tree()?;
        let tree = repository.find_tree(tree_id)?;
        let signature = git2::Signature::now("fixture", "fixture@example.invalid")?;
        let commit =
            repository.commit(Some("HEAD"), &signature, &signature, "fixture", &tree, &[])?;
        let record = config.model_admission.as_mut().context("record")?;
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&record.manifest_path)?)?;
        record.commit = commit.to_string();
        record.tree = tree_id.to_string();
        manifest["commit"] = record.commit.clone().into();
        manifest["tree"] = record.tree.clone().into();
        let bytes = serde_json::to_vec(&manifest)?;
        std::fs::write(&record.manifest_path, &bytes)?;
        record.manifest_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let selected = config.selected_model_admission()?.context("selection")?;
        selected.verify_tree(&directory)?;
        let mut wrong = selected.clone();
        wrong.tree = git2::Oid::zero();
        assert!(wrong.verify_tree(&directory).is_err());
        let mut wrong = selected.clone();
        wrong.manifest.files[0].sha256 = "d".repeat(64);
        assert!(wrong.verify_tree(&directory).is_err());
        let mut wrong = selected.clone();
        wrong.manifest.files.pop();
        assert!(wrong.verify_tree(&directory).is_err());
        Ok(())
    }
    #[cfg(feature = "postgres")]
    #[test]
    fn config_qualifies_exact_scope_without_upgrading_arbitrary_pin() -> Result<()> {
        let (_root, config) = tiny_fixture(
            ModelOperation::Decisions,
            "third-model:main",
            "did:plc:fixtureowner",
        )?;
        let pin = crate::services::model::StagingModelPin::from_config(&config)?.context("pin")?;
        let profile = pin.federate_profile()?;
        assert_eq!(
            profile.scopes(),
            crate::services::model::FederateModelProfile::from_config(&config)?
                .context("profile")?
                .scopes()
        );
        assert!(profile.scopes().contains("decide:model:third-model:main"));
        assert!(!profile.scopes().contains("infer:model:third-model:main"));
        assert!(
            crate::services::model::StagingModelPin::new(pin.model_ref, pin.commit)?
                .federate_profile()
                .is_err()
        );
        assert!(crate::services::model::FederateModelProfile::from_config(
            &super::super::HyprConfig::default()
        )?
        .is_none());
        Ok(())
    }
    #[test]
    #[ignore = "requires a real protected deployment config and exact Git repository; metadata only, no inference"]
    fn deployed_config_matches_complete_real_git_inventory() -> Result<()> {
        let path = std::env::var_os("HYPRSTREAM_MODEL_ADMISSION_TEST_CONFIG")
            .context("explicit deployment config required")?;
        let repository = std::env::var_os("HYPRSTREAM_MODEL_ADMISSION_TEST_REPOSITORY")
            .context("explicit immutable repository required")?;
        let config = super::super::HyprConfig::from_file(Path::new(&path))?;
        config.validate()?;
        let selected = config
            .selected_model_admission()?
            .context("selected model required")?;
        selected.local_lfs()?;
        selected.verify_tree(Path::new(&repository))?;
        Ok(())
    }
    #[test]
    fn immutable_capture_survives_manifest_replacement() -> Result<()> {
        let (_root, config) = tiny_fixture(
            ModelOperation::Generation,
            "fixture:main",
            "did:plc:fixtureowner",
        )?;
        let before = config
            .selected_model_admission()?
            .context("selection")?
            .clone();
        std::fs::write(
            &config
                .model_admission
                .as_ref()
                .context("record")?
                .manifest_path,
            b"{}",
        )?;
        assert_eq!(
            config
                .clone()
                .selected_model_admission()?
                .context("clone selection")?,
            &before
        );
        Ok(())
    }
}
