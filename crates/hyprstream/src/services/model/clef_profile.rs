//! Explicit server-owned Clef CPU admission contract. No RPC-supplied policy.
//!
//! Startup validates bounded metadata and exact local object membership. The
//! existing owned projection hashes/copies every payload against captured LFS
//! SHA256 and size before publishing any loader input; this is not a claim that
//! the cheap startup checks hash twelve GiB of weights.
use super::{InferenceCompute, InferenceDeploymentProfile, InferenceIsolationProfile};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

pub(super) const PROFILE: &str = "clef-flash-fp8-cpu-v1";
pub(super) const MODEL_REF: &str = "clef-flash-fp8:main";
pub(super) const COMMIT: &str = "7b43c2d799e7b5e9fef96b6137840088d75af0d6";
pub(super) const TREE: &str = "6684c66f43cf413bccd48025380a69ebe2e428a9";
pub(super) const EXPANDED_BYTES: u64 = 12_174_008_714;
pub(super) const MANIFEST_SHA256: &str =
    "1053bacfc8d9017ec27ea43172571e81ff6b77735874bbf1fc38cbdf4d8c851c";
const MANIFEST: &str = include_str!("clef_profile_manifest.json");

#[derive(Clone, Copy)]
struct LocalObject {
    path: &'static str,
    sha256: &'static str,
    bytes: u64,
}
const LOCAL_OBJECTS: [LocalObject; 3] = [
    LocalObject {
        path: "clef_head/joint_head.safetensors",
        sha256: "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
        bytes: 243_538_016,
    },
    LocalObject {
        path: "model.safetensors",
        sha256: "7799d5c3a47b8f249d667a040e3fd538ccbf47bd958dc6f2aa52af8346aefa6b",
        bytes: 11_910_387_016,
    },
    LocalObject {
        path: "tokenizer.json",
        sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        bytes: 19_989_325,
    },
];
#[derive(Deserialize)]
struct Manifest {
    repo: String,
    revision: String,
    files: Vec<ManifestFile>,
    all_verified: bool,
}
#[derive(Deserialize)]
struct ManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
    verified: bool,
}

fn validate_manifest() -> Result<()> {
    ensure!(
        MANIFEST.len() <= 16 * 1024,
        "Clef contract manifest exceeds bound"
    );
    ensure!(
        format!("{:x}", Sha256::digest(MANIFEST.as_bytes())) == MANIFEST_SHA256,
        "Clef source-owned contract manifest digest changed"
    );
    let manifest: Manifest = serde_json::from_str(MANIFEST)?;
    ensure!(
        manifest.repo == "kurcontko/clef-flash-FP8-Dynamic"
            && manifest.revision == COMMIT
            && manifest.all_verified,
        "Clef contract provenance changed"
    );
    ensure!(
        manifest.files.len() == 21,
        "Clef complete contract must contain21files"
    );
    let mut paths = BTreeSet::new();
    let mut total = 0_u64;
    for file in &manifest.files {
        ensure!(
            !file.path.is_empty()
                && Path::new(&file.path)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_)))
                && paths.insert(&file.path)
                && file.verified
                && file.bytes > 0
                && file.sha256.len() == 64
                && file
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid or duplicate Clef source contract entry"
        );
        total = total
            .checked_add(file.bytes)
            .context("Clef contract byte sum overflow")?;
    }
    ensure!(
        total == EXPANDED_BYTES,
        "Clef contract expanded byte total changed"
    );
    for object in LOCAL_OBJECTS {
        ensure!(
            manifest.files.iter().any(|f| f.path == object.path
                && f.sha256 == object.sha256
                && f.bytes == object.bytes),
            "Clef fixed hydration object differs from source contract"
        );
    }
    for (path, sha) in [
        (
            "clef_head/joint_head_config.json",
            "77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0",
        ),
        (
            "config.json",
            "0978bfcc5d6da6d51fa25051d792258a61e99a493c0172d26457630564d80131",
        ),
    ] {
        ensure!(
            manifest
                .files
                .iter()
                .any(|f| f.path == path && f.sha256 == sha),
            "Clef required config pin differs"
        );
    }
    Ok(())
}

pub(super) fn local_lfs(directory: &Path) -> Result<BTreeMap<String, PathBuf>> {
    validate_manifest()?;
    validate_directory(directory, &LOCAL_OBJECTS)
}

// Private helper: unit tests supply tiny real files without any public pin,
// release, projection-budget, or payload hash-verification override.
fn validate_directory(
    directory: &Path,
    objects: &[LocalObject],
) -> Result<BTreeMap<String, PathBuf>> {
    ensure!(
        directory.is_absolute(),
        "Clef LFS directory must be absolute"
    );
    let canonical = directory
        .canonicalize()
        .context("resolve Clef LFS directory")?;
    ensure!(
        canonical.as_os_str() == directory.as_os_str()
            && std::fs::symlink_metadata(directory)?.file_type().is_dir(),
        "Clef LFS directory must be canonical and not a symlink"
    );
    ensure!(
        objects.len() == 3,
        "Clef hydration requires exactly three objects"
    );
    let expected: BTreeSet<_> = objects.iter().map(|o| o.sha256).collect();
    ensure!(expected.len() == 3, "duplicate Clef hydration object");
    let mut count = 0;
    for entry in std::fs::read_dir(directory)? {
        count += 1;
        ensure!(count <= 3, "extra Clef LFS directory entry");
        let entry = entry?;
        let name = entry.file_name();
        ensure!(
            name.to_str().is_some_and(|name| expected.contains(name)),
            "unexpected Clef LFS directory entry"
        );
    }
    ensure!(count == 3, "missing Clef LFS object");
    let mut map = BTreeMap::new();
    for object in objects {
        let file = directory.join(object.sha256);
        let metadata = std::fs::symlink_metadata(&file).context("stat Clef LFS object")?;
        ensure!(
            metadata.file_type().is_file() && metadata.len() == object.bytes,
            "Clef LFS object must be a regular non-symlink file of the fixed size"
        );
        ensure!(
            file.canonicalize()?.as_os_str() == file.as_os_str(),
            "Clef LFS object must be canonical"
        );
        map.insert(object.sha256.to_owned(), file);
    }
    Ok(map)
}

pub(super) fn validate_deployment(profile: &InferenceDeploymentProfile) -> Result<()> {
    ensure!(
        profile.compute == InferenceCompute::Cpu,
        "Clef CPU profile requires HYPRSTREAM_INFERENCE_COMPUTE=cpu"
    );
    ensure!(
        profile.isolation == InferenceIsolationProfile::InProcess,
        "Clef CPU profile currently requires HYPRSTREAM_INFERENCE_ISOLATION=in-process"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: [LocalObject; 3] = [
        LocalObject {
            path: "head",
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            bytes: 3,
        },
        LocalObject {
            path: "backbone",
            sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            bytes: 4,
        },
        LocalObject {
            path: "tokenizer",
            sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            bytes: 5,
        },
    ];
    fn tiny_directory() -> Result<tempfile::TempDir> {
        let dir = tempfile::tempdir()?;
        for object in TINY {
            std::fs::write(
                dir.path().join(object.sha256),
                vec![7; usize::try_from(object.bytes)?],
            )?;
        }
        Ok(dir)
    }
    #[test]
    fn reviewed_manifest_is_complete_and_exact() -> Result<()> {
        validate_manifest()?;
        assert_eq!(LOCAL_OBJECTS.len(), 3);
        assert_eq!(TREE, "6684c66f43cf413bccd48025380a69ebe2e428a9");
        Ok(())
    }
    #[test]
    fn exact_directory_maps_only_fixed_objects() -> Result<()> {
        let dir = tiny_directory()?;
        let map = validate_directory(dir.path(), &TINY)?;
        assert_eq!(map.len(), 3);
        for object in TINY {
            assert_eq!(
                map.get(object.sha256),
                Some(&dir.path().join(object.sha256))
            );
        }
        Ok(())
    }
    #[test]
    fn missing_extra_wrong_size_and_nonregular_files_fail_closed() -> Result<()> {
        for kind in ["missing", "extra", "size", "directory", "socket"] {
            let dir = tiny_directory()?;
            let file = dir.path().join(TINY[0].sha256);
            let _socket = if kind == "socket" {
                std::fs::remove_file(&file)?;
                Some(std::os::unix::net::UnixListener::bind(&file)?)
            } else {
                None
            };
            match kind {
                "missing" => std::fs::remove_file(&file)?,
                "extra" => std::fs::write(dir.path().join("unrecognized"), b"x")?,
                "size" => std::fs::write(&file, b"wrong")?,
                "directory" => {
                    std::fs::remove_file(&file)?;
                    std::fs::create_dir(&file)?;
                }
                _ => {}
            }
            assert!(validate_directory(dir.path(), &TINY).is_err(), "{kind}");
        }
        Ok(())
    }
    #[test]
    fn relative_noncanonical_and_symlink_paths_fail_closed() -> Result<()> {
        let dir = tiny_directory()?;
        assert!(validate_directory(Path::new("relative"), &TINY).is_err());
        let noncanonical = dir
            .path()
            .join("..")
            .join(dir.path().file_name().context("temp dir name")?);
        assert!(validate_directory(&noncanonical, &TINY).is_err());
        let parent = tempfile::tempdir()?;
        let link = parent.path().join("link");
        std::os::unix::fs::symlink(dir.path(), &link)?;
        assert!(validate_directory(&link, &TINY).is_err());
        let file = dir.path().join(TINY[0].sha256);
        let source = parent.path().join("source");
        std::fs::rename(&file, &source)?;
        std::os::unix::fs::symlink(source, &file)?;
        assert!(validate_directory(dir.path(), &TINY).is_err());
        Ok(())
    }
    #[test]
    fn deployment_must_be_explicit_cpu_and_current_qualified_isolation() -> Result<()> {
        let mut profile = InferenceDeploymentProfile::default();
        assert!(validate_deployment(&profile).is_err());
        profile.compute = InferenceCompute::Gpu;
        assert!(validate_deployment(&profile).is_err());
        profile.compute = InferenceCompute::Cpu;
        validate_deployment(&profile)?;
        profile.isolation = InferenceIsolationProfile::PerTenantSubprocess;
        assert!(validate_deployment(&profile).is_err());
        Ok(())
    }
    #[test]
    fn startup_selects_fixed_clef_policy_with_tiny_private_validator() -> Result<()> {
        use super::super::StagingModelPin;
        let tiny = LOCAL_OBJECTS.map(|object| LocalObject { bytes: 4, ..object });
        let dir = tempfile::tempdir()?;
        for object in tiny {
            std::fs::write(dir.path().join(object.sha256), b"tiny")?;
        }
        let expected_map = validate_directory(dir.path(), &tiny)?;
        for overrides in [(None, None), (Some(MODEL_REF), Some(COMMIT))] {
            let pin = StagingModelPin::select_startup_pin(
                PROFILE,
                overrides.0,
                overrides.1,
                dir.path().to_str(),
                |path| validate_directory(path, &tiny),
            )?
            .context("Clef profile must select a pin")?;
            assert_eq!(pin.model_ref, MODEL_REF);
            assert_eq!(pin.commit, git2::Oid::from_str(COMMIT)?);
            assert_eq!(
                pin.capture_policy()?,
                git2db::pinned_disk::ProjectionPolicy::from_trusted_contract(
                    EXPANDED_BYTES,
                    expected_map.clone()
                )?
            );
            let mut deployment = InferenceDeploymentProfile::default();
            assert!(pin.validate_deployment(&deployment).is_err());
            deployment.compute = InferenceCompute::Cpu;
            pin.validate_deployment(&deployment)?;
        }
        // The actual public startup path uses the production fixed-size
        // validator and MUST reject these tiny files, not use the unit seam.
        assert!(
            StagingModelPin::from_startup_settings(PROFILE, None, None, dir.path().to_str())
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn startup_bad_settings_fail_before_hydration_and_default_staging_stay_ordinary() -> Result<()>
    {
        use super::super::StagingModelPin;
        for (profile, name, commit, directory) in [
            (PROFILE, None, None, None),
            (PROFILE, Some("wrong:main"), Some(COMMIT), Some("/unused")),
            (
                PROFILE,
                Some(MODEL_REF),
                Some("0000000000000000000000000000000000000000"),
                Some("/unused"),
            ),
            (PROFILE, Some(MODEL_REF), None, Some("/unused")),
            (PROFILE, None, Some(COMMIT), Some("/unused")),
            ("unknown-profile", None, None, None),
            ("default", None, None, Some("/unused")),
            ("staging", None, None, Some("/unused")),
        ] {
            let called = std::cell::Cell::new(false);
            let result =
                StagingModelPin::select_startup_pin(profile, name, commit, directory, |_| {
                    called.set(true);
                    anyhow::bail!("unexpected hydration resolver")
                });
            assert!(result.is_err());
            assert!(!called.get(), "invalid settings reached hydration");
        }
        assert!(StagingModelPin::from_startup_settings("default", None, None, None)?.is_none());
        let ordinary = StagingModelPin::from_startup_settings("staging", None, None, None)?
            .context("ordinary staging pin")?;
        assert_eq!(
            ordinary.capture_policy()?,
            git2db::pinned_disk::ProjectionPolicy::default()
        );
        assert_eq!(
            ordinary.capture_policy()?,
            git2db::pinned_disk::ProjectionPolicy::from_trusted_contract(
                2 * 1024 * 1024 * 1024,
                BTreeMap::new()
            )?
        );
        Ok(())
    }
    #[test]
    fn from_env_reads_explicit_profile_without_parent_environment_mutation() -> Result<()> {
        use super::super::StagingModelPin;
        const CHILD: &str = "HYPRSTREAM_CLEF_PROFILE_ENV_TEST";
        if let Ok(case) = std::env::var(CHILD) {
            let result = StagingModelPin::from_env();
            match case.as_str() {
                "default" => assert!(result?.is_none()),
                "missing-directory" => assert!(result
                    .err()
                    .context("missing dir accepted")?
                    .to_string()
                    .contains("HYPRSTREAM_CLEF_LFS_DIRECTORY is required")),
                "stray-directory" => assert!(result
                    .err()
                    .context("stray dir accepted")?
                    .to_string()
                    .contains("requires the explicit Clef CPU admission profile")),
                _ => anyhow::bail!("unknown isolated env fixture case"),
            }
            return Ok(());
        }
        for case in ["default", "missing-directory", "stray-directory"] {
            let mut child = std::process::Command::new(std::env::current_exe()?);
            child.args(["--exact", "services::model::clef_profile::tests::from_env_reads_explicit_profile_without_parent_environment_mutation", "--test-threads=1"])
                .env(CHILD, case);
            for key in [
                "HYPRSTREAM_MODEL_ADMISSION_PROFILE",
                "HYPRSTREAM_STAGING_MODEL_REF",
                "HYPRSTREAM_STAGING_MODEL_OID",
                "HYPRSTREAM_CLEF_LFS_DIRECTORY",
            ] {
                child.env_remove(key);
            }
            if case == "missing-directory" {
                child.env("HYPRSTREAM_MODEL_ADMISSION_PROFILE", PROFILE);
            }
            if case == "stray-directory" {
                child.env("HYPRSTREAM_CLEF_LFS_DIRECTORY", "/unused");
            }
            let result = child.output()?;
            ensure!(
                result.status.success(),
                "isolated startup environment case failed: {case}"
            );
            ensure!(
                String::from_utf8_lossy(&result.stdout).contains("1 passed;"),
                "isolated startup environment selector did not execute its test: {case}"
            );
        }
        Ok(())
    }
}
