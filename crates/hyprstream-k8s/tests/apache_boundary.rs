//! License-boundary regression for the reusable Kubernetes substrate.
//!
//! This intentionally resolves every local normal, optional, build, dev, and
//! target-specific path dependency without asking Cargo to select a feature
//! set. A forbidden service anywhere in that complete closure fails the test.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;

const AGPL_SERVICES: &[&str] = &[
    "hyprstream",
    "hyprstream-appview",
    "hyprstream-discovery",
    "hyprstream-k8s-pds",
    "hyprstream-ledger",
    "hyprstream-pds",
    "hyprstream-pds-service",
    "hyprstream-service",
];

#[derive(Debug)]
struct PendingManifest {
    path: PathBuf,
    chain: Vec<String>,
}

fn load_manifest(path: &Path) -> Value {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .parse()
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn package_name(manifest: &Value, path: &Path) -> String {
    manifest["package"]["name"]
        .as_str()
        .unwrap_or_else(|| panic!("{} has no package.name", path.display()))
        .to_owned()
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
) -> Vec<(PathBuf, Option<String>)> {
    let workspace_dependencies = workspace_manifest["workspace"]["dependencies"]
        .as_table()
        .expect("workspace.dependencies must be a table");
    let mut dependencies = Vec::new();
    for table in dependency_tables(manifest) {
        for (alias, declared) in table {
            let (source, base) = if declared
                .as_table()
                .and_then(|value| value.get("workspace"))
                .and_then(Value::as_bool)
                == Some(true)
            {
                (
                    workspace_dependencies.get(alias).unwrap_or_else(|| {
                        panic!(
                            "{} inherits missing workspace dependency {alias}",
                            manifest_path.display()
                        )
                    }),
                    workspace_root,
                )
            } else {
                (
                    declared,
                    manifest_path.parent().expect("manifest has parent"),
                )
            };
            let Some(source) = source.as_table() else {
                continue;
            };
            let Some(local_path) = source.get("path").and_then(Value::as_str) else {
                continue;
            };
            let dependency_manifest = base.join(local_path).join("Cargo.toml");
            let declared_package = source
                .get("package")
                .and_then(Value::as_str)
                .map(str::to_owned);
            dependencies.push((dependency_manifest, declared_package));
        }
    }
    dependencies
}

#[test]
fn hyprstream_k8s_complete_local_closure_excludes_agpl_services() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonical workspace root");
    let workspace_manifest_path = workspace_root.join("Cargo.toml");
    let workspace_manifest = load_manifest(&workspace_manifest_path);
    let root_manifest_path = workspace_root.join("crates/hyprstream-k8s/Cargo.toml");

    let mut queue = VecDeque::from([PendingManifest {
        path: root_manifest_path,
        chain: vec!["hyprstream-k8s".to_owned()],
    }]);
    let mut visited = BTreeSet::new();
    let mut closure = BTreeMap::new();

    while let Some(pending) = queue.pop_front() {
        let canonical = pending
            .path
            .canonicalize()
            .unwrap_or_else(|error| panic!("resolve {}: {error}", pending.path.display()));
        if !visited.insert(canonical.clone()) {
            continue;
        }
        let manifest = load_manifest(&canonical);
        let name = package_name(&manifest, &canonical);
        closure.insert(name.clone(), canonical.clone());

        let license = manifest["package"]["license"].as_str().unwrap_or_default();
        if AGPL_SERVICES.contains(&name.as_str())
            || license.split(" OR ").any(|id| id == "AGPL-3.0-only")
        {
            panic!(
                "Apache-2.0-to-AGPL dependency: {}",
                pending.chain.join(" -> ")
            );
        }

        for (dependency_path, declared_package) in
            local_dependencies(&canonical, &manifest, &workspace_manifest, &workspace_root)
        {
            let dependency_manifest = load_manifest(&dependency_path);
            let dependency_name = package_name(&dependency_manifest, &dependency_path);
            if let Some(declared_package) = declared_package {
                assert_eq!(
                    declared_package,
                    dependency_name,
                    "{} declares package {declared_package}, resolved {dependency_name}",
                    canonical.display()
                );
            }
            let mut chain = pending.chain.clone();
            chain.push(dependency_name);
            queue.push_back(PendingManifest {
                path: dependency_path,
                chain,
            });
        }
    }

    assert!(
        closure.contains_key("hyprstream-k8s"),
        "root package must be in its own closure"
    );
}
