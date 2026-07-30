//! License-boundary regression for the reusable Kubernetes substrate.
//!
//! The walk resolves every reachable local normal, optional, build, dev, and
//! target-specific path dependency without asking Cargo to select features.
//! Classification comes from package SPDX metadata rather than package names.
//! When #1417's omission-checked `.github/license-boundary.toml` is present,
//! its `agpl_services` partition augments manifest metadata; this is the only
//! integration seam, so no policy list is copied into this crate.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;

#[derive(Debug)]
struct PendingManifest {
    path: PathBuf,
    chain: Vec<String>,
}

fn load_manifest(path: &Path) -> Result<Value, String> {
    fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?
        .parse()
        .map_err(|error| format!("parse {}: {error}", path.display()))
}

fn package_name(manifest: &Value, path: &Path) -> Result<String, String> {
    manifest["package"]["name"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{} has no package.name", path.display()))
}

fn dependency_tables(manifest: &Value) -> Vec<&toml::map::Map<String, Value>> {
    const SECTIONS: &[&str] = &["dependencies", "build-dependencies", "dev-dependencies"];
    let mut tables = Vec::new();
    for section in SECTIONS {
        if let Some(table) = manifest.get(*section).and_then(Value::as_table) {
            tables.push(table);
        }
    }
    if let Some(targets) = manifest.get("target").and_then(Value::as_table) {
        for target in targets.values().filter_map(Value::as_table) {
            for section in SECTIONS {
                if let Some(table) = target.get(*section).and_then(Value::as_table) {
                    tables.push(table);
                }
            }
        }
    }
    tables
}

fn local_dependencies(
    manifest_path: &Path,
    manifest: &Value,
    workspace_manifest: &Value,
    workspace_root: &Path,
) -> Result<Vec<(PathBuf, Option<String>)>, String> {
    let workspace_dependencies = workspace_manifest["workspace"]["dependencies"].as_table();
    let mut dependencies = Vec::new();
    for table in dependency_tables(manifest) {
        for (alias, declared) in table {
            let (source, base) = if declared
                .as_table()
                .and_then(|value| value.get("workspace"))
                .and_then(Value::as_bool)
                == Some(true)
            {
                let inherited = workspace_dependencies
                    .and_then(|dependencies| dependencies.get(alias))
                    .ok_or_else(|| {
                        format!(
                            "{} inherits missing workspace dependency {alias}",
                            manifest_path.display()
                        )
                    })?;
                (inherited, workspace_root)
            } else {
                (
                    declared,
                    manifest_path
                        .parent()
                        .ok_or_else(|| format!("{} has no parent", manifest_path.display()))?,
                )
            };
            let Some(source) = source.as_table() else {
                continue;
            };
            let Some(local_path) = source.get("path").and_then(Value::as_str) else {
                continue;
            };
            dependencies.push((
                base.join(local_path).join("Cargo.toml"),
                source
                    .get("package")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            ));
        }
    }
    Ok(dependencies)
}

fn resolved_license<'a>(
    manifest: &'a Value,
    workspace_manifest: &'a Value,
    path: &Path,
) -> Result<&'a str, String> {
    let package = manifest
        .get("package")
        .and_then(Value::as_table)
        .ok_or_else(|| format!("{} has no package table", path.display()))?;
    if let Some(license) = package.get("license").and_then(Value::as_str) {
        return Ok(license);
    }
    if package
        .get("license")
        .and_then(Value::as_table)
        .and_then(|license| license.get("workspace"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        return workspace_manifest
            .get("workspace")
            .and_then(Value::as_table)
            .and_then(|workspace| workspace.get("package"))
            .and_then(Value::as_table)
            .and_then(|package| package.get("license"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!(
                    "{} inherits missing workspace.package.license",
                    path.display()
                )
            });
    }
    Err(format!(
        "{} has no resolved package license; Apache boundary fails closed",
        path.display()
    ))
}

/// Detect AGPL identifiers in SPDX expressions, including `-only`,
/// `-or-later`, deprecated `+`, compound `AND`/`OR`, and `WITH` forms.
///
/// Cargo validates the expression syntax. The boundary only needs to extract
/// license identifiers without assuming one exact expression shape.
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
    let policy = load_manifest(&path)?;
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

fn check_apache_boundary(workspace_root: &Path, root_manifest_path: &Path) -> Result<(), String> {
    let workspace_manifest_path = workspace_root.join("Cargo.toml");
    let workspace_manifest = load_manifest(&workspace_manifest_path)?;
    let policy_agpl = policy_agpl_packages(workspace_root)?;
    let mut queue = VecDeque::from([PendingManifest {
        path: root_manifest_path.to_owned(),
        chain: Vec::new(),
    }]);
    let mut visited = BTreeSet::new();

    while let Some(pending) = queue.pop_front() {
        let canonical = pending
            .path
            .canonicalize()
            .map_err(|error| format!("resolve {}: {error}", pending.path.display()))?;
        if !visited.insert(canonical.clone()) {
            continue;
        }
        let manifest = load_manifest(&canonical)?;
        let name = package_name(&manifest, &canonical)?;
        let mut chain = pending.chain;
        chain.push(name.clone());
        let license = resolved_license(&manifest, &workspace_manifest, &canonical)?;
        if policy_agpl.contains(&name) || contains_agpl_identifier(license) {
            return Err(format!(
                "Apache-2.0-to-AGPL dependency: {} ({license})",
                chain.join(" -> ")
            ));
        }

        for (dependency_path, declared_package) in
            local_dependencies(&canonical, &manifest, &workspace_manifest, workspace_root)?
        {
            let dependency_manifest = load_manifest(&dependency_path)?;
            let dependency_name = package_name(&dependency_manifest, &dependency_path)?;
            if let Some(declared_package) = declared_package {
                if declared_package != dependency_name {
                    return Err(format!(
                        "{} declares package {declared_package}, resolved {dependency_name}",
                        canonical.display()
                    ));
                }
            }
            queue.push_back(PendingManifest {
                path: dependency_path,
                chain: chain.clone(),
            });
        }
    }
    Ok(())
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

fn fixture_workspace(workspace_license: &str) -> tempfile::TempDir {
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
"#
        ),
    );
    directory
}

fn write_fixture_packages(root: &Path, dependency_section: &str, service_license: &str) -> PathBuf {
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
name = "new-service"
version = "0.1.0"
{service_license}
"#
        ),
    );
    root_manifest
}

#[test]
fn renamed_new_agpl_package_is_rejected() {
    let directory = fixture_workspace("MIT");
    let root_manifest = write_fixture_packages(
        directory.path(),
        r#"[dependencies]
renamed = { package = "new-service", path = "../service" }"#,
        r#"license = "AGPL-3.0-only""#,
    );
    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(error.contains("apache-root -> new-service"), "{error}");
}

#[test]
fn workspace_inherited_dependency_and_license_are_rejected() {
    let directory = fixture_workspace("AGPL-3.0-or-later");
    let root_manifest = write_fixture_packages(
        directory.path(),
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
        let directory = fixture_workspace("MIT");
        let root_manifest = write_fixture_packages(
            directory.path(),
            dependency_section,
            r#"license = "AGPL-3.0-only""#,
        );
        assert!(
            check_apache_boundary(directory.path(), &root_manifest).is_err(),
            "edge escaped boundary: {dependency_section}"
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
        let directory = fixture_workspace("MIT");
        let root_manifest = write_fixture_packages(
            directory.path(),
            r#"[dependencies]
renamed = { package = "new-service", path = "../service" }"#,
            &format!(r#"license = "{expression}""#),
        );
        assert!(
            check_apache_boundary(directory.path(), &root_manifest).is_err(),
            "AGPL expression escaped boundary: {expression}"
        );
    }
}

#[test]
fn canonical_policy_augments_manifest_metadata_without_a_copied_name_list() {
    let directory = fixture_workspace("MIT");
    let root_manifest = write_fixture_packages(
        directory.path(),
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
