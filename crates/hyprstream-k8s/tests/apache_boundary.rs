//! License-boundary regression for the reusable Kubernetes substrate.
//!
//! Cargo's resolved metadata graph is the source of dependency truth. Running
//! with all features includes optional edges, and metadata retains normal,
//! build, dev, target, renamed, `[patch]`, and `[replace]` resolution. Local
//! packages are classified by their resolved SPDX metadata rather than copied
//! names. When #1417's omission-checked `.github/license-boundary.toml` is
//! present, its `agpl_services` partition augments manifest metadata; that is
//! the only policy integration seam.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use toml::Value;

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<MetadataPackage>,
    resolve: Option<MetadataResolve>,
}

#[derive(Debug, Deserialize)]
struct MetadataPackage {
    id: String,
    name: String,
    license: Option<String>,
    manifest_path: PathBuf,
    source: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MetadataResolve {
    nodes: Vec<MetadataNode>,
}

#[derive(Debug, Deserialize)]
struct MetadataNode {
    id: String,
    deps: Vec<MetadataDependency>,
}

#[derive(Debug, Deserialize)]
struct MetadataDependency {
    pkg: String,
}

fn load_toml(path: &Path) -> Result<Value, String> {
    fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?
        .parse()
        .map_err(|error| format!("parse {}: {error}", path.display()))
}

/// Detect AGPL identifiers in SPDX expressions, including `-only`,
/// `-or-later`, deprecated `+`, compound `AND`/`OR`, and `WITH` forms.
///
/// Cargo validates package license expressions while resolving metadata. The
/// boundary only needs to extract identifiers without assuming one literal.
fn contains_agpl_identifier(expression: &str) -> bool {
    expression
        .split(|character: char| {
            !(character.is_ascii_alphanumeric()
                || character == '-'
                || character == '.'
                || character == '+')
        })
        .any(|identifier| identifier.starts_with("AGPL-"))
}

fn policy_agpl_packages(workspace_root: &Path) -> Result<BTreeSet<String>, String> {
    let path = workspace_root.join(".github/license-boundary.toml");
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let policy = load_toml(&path)?;
    policy["license_gate"]["agpl_services"]
        .as_array()
        .ok_or_else(|| format!("{} must define license_gate.agpl_services", path.display()))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{} has a non-string AGPL package", path.display()))
        })
        .collect()
}

fn resolved_metadata(
    workspace_root: &Path,
    root_manifest_path: &Path,
) -> Result<CargoMetadata, String> {
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version=1",
            "--all-features",
            "--offline",
            "--manifest-path",
        ])
        .arg(root_manifest_path)
        .current_dir(workspace_root)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .map_err(|error| format!("run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("parse cargo metadata JSON: {error}"))
}

fn check_apache_boundary(workspace_root: &Path, root_manifest_path: &Path) -> Result<(), String> {
    let metadata = resolved_metadata(workspace_root, root_manifest_path)?;
    let policy_agpl = policy_agpl_packages(workspace_root)?;
    let root_manifest = root_manifest_path
        .canonicalize()
        .map_err(|error| format!("resolve {}: {error}", root_manifest_path.display()))?;

    let packages = metadata
        .packages
        .into_iter()
        .map(|package| (package.id.clone(), package))
        .collect::<BTreeMap<_, _>>();
    let root_id = packages
        .values()
        .find_map(|package| {
            package
                .manifest_path
                .canonicalize()
                .ok()
                .filter(|path| path == &root_manifest)
                .map(|_| package.id.clone())
        })
        .ok_or_else(|| {
            format!(
                "cargo metadata omitted root package {}",
                root_manifest.display()
            )
        })?;
    let resolve = metadata
        .resolve
        .ok_or_else(|| "cargo metadata omitted the resolved graph".to_owned())?;
    let nodes = resolve
        .nodes
        .into_iter()
        .map(|node| (node.id.clone(), node))
        .collect::<BTreeMap<_, _>>();

    let mut queue = VecDeque::from([(root_id, Vec::<String>::new())]);
    let mut visited = BTreeSet::new();
    while let Some((id, mut chain)) = queue.pop_front() {
        if !visited.insert(id.clone()) {
            continue;
        }
        let package = packages
            .get(&id)
            .ok_or_else(|| format!("resolved graph references unknown package {id}"))?;
        chain.push(package.name.clone());

        if package.source.is_none() {
            let license = package.license.as_deref().ok_or_else(|| {
                format!(
                    "{} has no resolved package license; Apache boundary fails closed",
                    package.manifest_path.display()
                )
            })?;
            if policy_agpl.contains(&package.name) || contains_agpl_identifier(license) {
                return Err(format!(
                    "Apache-2.0-to-AGPL dependency: {} ({license})",
                    chain.join(" -> ")
                ));
            }
        }

        let node = nodes
            .get(&id)
            .ok_or_else(|| format!("resolved graph has no node for {id}"))?;
        for dependency in &node.deps {
            if !packages.contains_key(&dependency.pkg) {
                return Err(format!(
                    "resolved graph references unknown dependency {}",
                    dependency.pkg
                ));
            }
            queue.push_back((dependency.pkg.clone(), chain.clone()));
        }
    }
    Ok(())
}

fn assert_transitive_patch_resolved(
    metadata: &CargoMetadata,
    external_name: &str,
    local_name: &str,
) {
    let external = metadata
        .packages
        .iter()
        .find(|package| package.name == external_name)
        .unwrap_or_else(|| panic!("fixture did not resolve external package {external_name}"));
    assert!(
        external.source.is_some(),
        "fixture package {external_name} must come from a registry"
    );
    let local = metadata
        .packages
        .iter()
        .find(|package| package.name == local_name)
        .unwrap_or_else(|| panic!("fixture did not resolve local package {local_name}"));
    assert!(
        local.source.is_none(),
        "fixture package {local_name} must resolve to the local patch"
    );
    let resolve = metadata
        .resolve
        .as_ref()
        .expect("fixture metadata must include a resolved graph");
    let external_node = resolve
        .nodes
        .iter()
        .find(|node| node.id == external.id)
        .unwrap_or_else(|| panic!("fixture graph omitted external package {external_name}"));
    assert!(
        external_node
            .deps
            .iter()
            .any(|dependency| dependency.pkg == local.id),
        "fixture graph must contain {external_name} -> {local_name}"
    );
}

#[test]
fn hyprstream_k8s_complete_local_closure_excludes_agpl_services() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonical workspace root");
    let root_manifest_path = workspace_root.join("crates/hyprstream-k8s/Cargo.toml");
    check_apache_boundary(&workspace_root, &root_manifest_path).unwrap();
}

fn write_fixture(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

fn fixture_workspace(workspace_license: &str, extra: &str) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    write_fixture(
        &directory.path().join("Cargo.toml"),
        &format!(
            r#"[workspace]
members = ["crates/root", "crates/service"]
resolver = "2"

[workspace.package]
license = "{workspace_license}"

[workspace.dependencies]
renamed = {{ package = "new-service", path = "crates/service" }}

{extra}
"#
        ),
    );
    directory
}

fn write_fixture_packages(
    root: &Path,
    dependency_section: &str,
    service_name: &str,
    service_version: &str,
    service_license: &str,
) -> PathBuf {
    let root_manifest = root.join("crates/root/Cargo.toml");
    write_fixture(
        &root_manifest,
        &format!(
            r#"[package]
name = "apache-root"
version = "0.1.0"
license = "Apache-2.0"

{dependency_section}
"#
        ),
    );
    write_fixture(
        &root.join("crates/service/Cargo.toml"),
        &format!(
            r#"[package]
name = "{service_name}"
version = "{service_version}"
{service_license}
"#
        ),
    );
    write_fixture(&root.join("crates/service/src/lib.rs"), "");
    write_fixture(&root.join("crates/root/src/lib.rs"), "");
    root_manifest
}

fn new_service_fixture(
    workspace_license: &str,
    dependency_section: &str,
    service_license: &str,
) -> (tempfile::TempDir, PathBuf) {
    let directory = fixture_workspace(workspace_license, "");
    let root_manifest = write_fixture_packages(
        directory.path(),
        dependency_section,
        "new-service",
        "0.1.0",
        service_license,
    );
    (directory, root_manifest)
}

#[test]
fn renamed_new_agpl_package_is_rejected() {
    let (directory, root_manifest) = new_service_fixture(
        "MIT",
        r#"[dependencies]
renamed = { package = "new-service", path = "../service" }"#,
        r#"license = "AGPL-3.0-only""#,
    );
    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(error.contains("apache-root -> new-service"), "{error}");
}

#[test]
fn workspace_inherited_dependency_and_license_are_rejected() {
    let (directory, root_manifest) = new_service_fixture(
        "AGPL-3.0-or-later",
        r#"[dependencies]
renamed.workspace = true"#,
        "license.workspace = true",
    );
    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(error.contains("AGPL-3.0-or-later"), "{error}");
}

#[test]
fn build_dev_and_target_edges_are_all_rejected() {
    for dependency_section in [
        r#"[build-dependencies]
renamed = { package = "new-service", path = "../service" }"#,
        r#"[dev-dependencies]
renamed = { package = "new-service", path = "../service" }"#,
        r#"[target.'cfg(unix)'.dependencies]
renamed = { package = "new-service", path = "../service" }"#,
        r#"[target.'cfg(unix)'.build-dependencies]
renamed = { package = "new-service", path = "../service" }"#,
        r#"[target.'cfg(unix)'.dev-dependencies]
renamed = { package = "new-service", path = "../service" }"#,
    ] {
        let (directory, root_manifest) =
            new_service_fixture("MIT", dependency_section, r#"license = "AGPL-3.0-only""#);
        let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
        assert!(
            error.contains("apache-root -> new-service"),
            "edge escaped boundary: {dependency_section}\n{error}"
        );
    }
}

#[test]
fn compound_and_versioned_agpl_expressions_are_rejected() {
    for expression in [
        "AGPL-3.0-or-later",
        "AGPL-3.0+",
        "Apache-2.0 AND AGPL-3.0-only",
        "(MIT OR Apache-2.0) AND AGPL-3.0-or-later",
    ] {
        let (directory, root_manifest) = new_service_fixture(
            "MIT",
            r#"[dependencies]
renamed = { package = "new-service", path = "../service" }"#,
            &format!(r#"license = "{expression}""#),
        );
        let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
        assert!(
            error.contains("apache-root -> new-service"),
            "AGPL expression escaped boundary: {expression}\n{error}"
        );
    }
}

#[test]
fn canonical_policy_augments_manifest_metadata_without_a_copied_name_list() {
    let (directory, root_manifest) = new_service_fixture(
        "MIT",
        r#"[dependencies]
renamed = { package = "new-service", path = "../service" }"#,
        r#"license = "MIT""#,
    );
    write_fixture(
        &directory.path().join(".github/license-boundary.toml"),
        r#"[license_gate]
apache_roots = ["apache-root"]
agpl_services = ["new-service"]
other_packages = []
"#,
    );
    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(error.contains("apache-root -> new-service"), "{error}");
}

fn patched_anyhow_fixture(dependency: &str) -> (tempfile::TempDir, PathBuf) {
    let directory = fixture_workspace(
        "MIT",
        r#"[patch.crates-io]
anyhow = { path = "crates/service" }"#,
    );
    let root_manifest = write_fixture_packages(
        directory.path(),
        dependency,
        "anyhow",
        "99.0.0",
        r#"license = "AGPL-3.0-only""#,
    );
    (directory, root_manifest)
}

#[test]
fn direct_registry_dependency_patched_to_local_agpl_is_rejected() {
    let (directory, root_manifest) = patched_anyhow_fixture(
        r#"[dependencies]
anyhow = "=99.0.0""#,
    );
    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(error.contains("apache-root -> anyhow"), "{error}");
}

#[test]
fn renamed_registry_dependency_patched_to_local_agpl_is_rejected() {
    let (directory, root_manifest) = patched_anyhow_fixture(
        r#"[dependencies]
patched = { package = "anyhow", version = "=99.0.0" }"#,
    );
    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(error.contains("apache-root -> anyhow"), "{error}");
}

#[test]
fn replaced_registry_dependency_resolves_to_local_agpl_and_is_rejected() {
    let directory = fixture_workspace(
        "MIT",
        r#"[replace]
"anyhow:1.0.102" = { path = "crates/service" }"#,
    );
    let root_manifest = write_fixture_packages(
        directory.path(),
        r#"[dependencies]
anyhow = "=1.0.102""#,
        "anyhow",
        "1.0.102",
        r#"license = "AGPL-3.0-only""#,
    );
    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(error.contains("apache-root -> anyhow"), "{error}");
}

#[test]
fn transitive_registry_dependency_patched_to_local_agpl_is_rejected() {
    let directory = fixture_workspace(
        "MIT",
        r#"[patch.crates-io]
itoa = { path = "crates/service" }"#,
    );
    let root_manifest = write_fixture_packages(
        directory.path(),
        r#"[dependencies]
serde_json = "=1.0.150""#,
        "itoa",
        "1.0.99",
        r#"license = "AGPL-3.0-only""#,
    );

    let metadata = resolved_metadata(directory.path(), &root_manifest)
        .expect("transitive patch fixture must resolve offline");
    assert_transitive_patch_resolved(&metadata, "serde_json", "itoa");

    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(
        error.contains("apache-root -> serde_json -> itoa"),
        "complete dependency chain missing from boundary failure: {error}"
    );
}
