//! Resolved license-boundary regression for every permissive local package.
//!
//! Cargo's committed resolved lock graph is the source of dependency truth.
//! It retains normal, build, dev, target, renamed, `[patch]`, and `[replace]`
//! resolution without requiring every registry source to be cached. A
//! `--no-deps --locked --offline` metadata call validates the local manifests
//! and classifies local packages by their SPDX metadata rather than copied
//! names. Full metadata resolution remains in the causal fixtures. When
//! #1417's omission-checked `.github/license-boundary.toml` is present, its
//! `agpl_packages` partition augments manifest metadata. Every MIT/Apache
//! package must be a strict reusable `permissive_root` with no resolved path
//! to a local AGPL package.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use semver::{Version, VersionReq};
use serde::Deserialize;
use toml::Value;

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<MetadataPackage>,
    resolve: Option<MetadataResolve>,
}

#[derive(Clone, Debug, Deserialize)]
struct MetadataPackage {
    id: String,
    name: String,
    version: String,
    license: Option<String>,
    manifest_path: PathBuf,
    source: Option<String>,
    dependencies: Vec<DeclaredDependency>,
}

#[derive(Clone, Debug, Deserialize)]
struct DeclaredDependency {
    name: String,
    req: String,
    source: Option<String>,
    path: Option<PathBuf>,
    rename: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MetadataResolve {
    nodes: Vec<MetadataNode>,
}

#[derive(Debug, Deserialize)]
struct MetadataNode {
    id: String,
    deps: Vec<ResolvedDependency>,
}

#[derive(Debug, Deserialize)]
struct ResolvedDependency {
    pkg: String,
}

#[derive(Debug, Deserialize)]
struct CargoLock {
    package: Vec<LockedPackage>,
}

#[derive(Debug, Deserialize)]
struct LockedPackage {
    name: String,
    version: String,
    source: Option<String>,
    #[serde(default)]
    dependencies: Vec<String>,
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

fn policy_packages(workspace_root: &Path, key: &str) -> Result<BTreeSet<String>, String> {
    let path = workspace_root.join(".github/license-boundary.toml");
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let policy = load_toml(&path)?;
    policy["license_gate"][key]
        .as_array()
        .ok_or_else(|| format!("{} must define license_gate.{key}", path.display()))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{} has a non-string package in {key}", path.display()))
        })
        .collect()
}

fn policy_agpl_packages(workspace_root: &Path) -> Result<BTreeSet<String>, String> {
    policy_packages(workspace_root, "agpl_packages")
}

fn policy_permissive_roots(workspace_root: &Path) -> Result<BTreeSet<String>, String> {
    policy_packages(workspace_root, "permissive_roots")
}

fn policy_permissive_packages(workspace_root: &Path) -> Result<BTreeSet<String>, String> {
    let mut permissive = policy_packages(workspace_root, "mit_packages")?;
    permissive.extend(policy_packages(workspace_root, "apache_packages")?);
    Ok(permissive)
}

fn policy_standalone_manifests(workspace_root: &Path) -> Result<Vec<PathBuf>, String> {
    let path = workspace_root.join(".github/license-boundary.toml");
    let policy = load_toml(&path)?;
    policy["license_gate"]["standalone_manifests"]
        .as_array()
        .ok_or_else(|| {
            format!(
                "{} must define license_gate.standalone_manifests",
                path.display()
            )
        })?
        .iter()
        .map(|value| {
            let relative = value.as_str().ok_or_else(|| {
                format!(
                    "{} has a non-string manifest in standalone_manifests",
                    path.display()
                )
            })?;
            let manifest = workspace_root.join(relative);
            if !manifest.is_file() {
                return Err(format!(
                    "{} lists missing standalone manifest {relative}",
                    path.display()
                ));
            }
            Ok(manifest)
        })
        .collect()
}

fn validate_permissive_root_partition(workspace_root: &Path) -> Result<BTreeSet<String>, String> {
    let permissive = policy_permissive_packages(workspace_root)?;
    let roots = policy_permissive_roots(workspace_root)?;
    let omissions = permissive.difference(&roots).cloned().collect::<Vec<_>>();
    let non_permissive = roots.difference(&permissive).cloned().collect::<Vec<_>>();
    if !omissions.is_empty() || !non_permissive.is_empty() {
        return Err(format!(
            "permissive root partition mismatch; omissions: {omissions:?}; \
             non-permissive roots: {non_permissive:?}"
        ));
    }
    Ok(roots)
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

fn local_metadata(
    workspace_root: &Path,
    root_manifest_path: &Path,
) -> Result<CargoMetadata, String> {
    let output = Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--format-version=1",
            "--all-features",
            "--locked",
            "--offline",
            "--no-deps",
            "--manifest-path",
        ])
        .arg(root_manifest_path)
        .current_dir(workspace_root)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .map_err(|error| format!("run local-only cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "local-only cargo metadata failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("parse local-only cargo metadata JSON: {error}"))
}

fn standalone_manifest_metadata(manifest_path: &Path) -> Result<MetadataPackage, String> {
    let manifest = load_toml(manifest_path)?;
    let package = manifest
        .get("package")
        .and_then(Value::as_table)
        .ok_or_else(|| format!("{} has no [package] table", manifest_path.display()))?;
    let string = |key: &str| {
        package
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                format!(
                    "{} package.{key} must be an explicit string",
                    manifest_path.display()
                )
            })
    };
    let name = string("name")?;
    let version = string("version")?;
    let license = Some(string("license")?);
    Ok(MetadataPackage {
        id: format!("path+{}#{name}@{version}", manifest_path.display()),
        name,
        version,
        license,
        manifest_path: manifest_path.to_path_buf(),
        source: None,
        dependencies: Vec::new(),
    })
}

fn load_cargo_lock(path: &Path) -> Result<CargoLock, String> {
    let contents =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    toml::from_str(&contents).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn locked_dependency_index(packages: &[LockedPackage], dependency: &str) -> Result<usize, String> {
    let name = dependency
        .split_whitespace()
        .next()
        .ok_or_else(|| "Cargo.lock contains an empty dependency reference".to_owned())?;
    let mut candidates = packages
        .iter()
        .enumerate()
        .filter(|(_, package)| package.name == name)
        .collect::<Vec<_>>();
    if dependency != name {
        let mut fields = dependency[name.len()..].trim().splitn(2, ' ');
        let version = fields
            .next()
            .ok_or_else(|| format!("invalid Cargo.lock dependency reference {dependency}"))?;
        candidates.retain(|(_, package)| package.version == version);
        if let Some(source) = fields.next() {
            let source = source
                .strip_prefix('(')
                .and_then(|value| value.strip_suffix(')'))
                .ok_or_else(|| format!("invalid Cargo.lock dependency source {dependency}"))?;
            candidates.retain(|(_, package)| {
                package.source.as_deref().is_some_and(|candidate| {
                    candidate == source
                        || candidate
                            .strip_prefix(source)
                            .is_some_and(|suffix| suffix.starts_with('#'))
                })
            });
        }
    }
    match candidates.as_slice() {
        [(index, _)] => Ok(*index),
        [] => Err(format!(
            "Cargo.lock dependency {dependency} has no resolved package"
        )),
        _ => Err(format!(
            "Cargo.lock dependency {dependency} is ambiguous across resolved packages"
        )),
    }
}

fn locked_root_index(packages: &[LockedPackage], root: &MetadataPackage) -> Result<usize, String> {
    let matches = packages
        .iter()
        .enumerate()
        .filter(|(_, package)| {
            package.name == root.name && package.version == root.version && package.source.is_none()
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(format!(
            "Cargo.lock omitted local root {} {}",
            root.name, root.version
        )),
        _ => Err(format!(
            "Cargo.lock has ambiguous local roots named {} {}",
            root.name, root.version
        )),
    }
}

fn source_identity_matches(resolved: &str, declared: &str) -> bool {
    resolved == declared
        || resolved
            .strip_prefix(declared)
            .is_some_and(|suffix| suffix.starts_with('#'))
        || declared
            .strip_prefix(resolved)
            .is_some_and(|suffix| suffix.starts_with('#'))
}

fn declared_dependency_matches_locked(
    dependency: &DeclaredDependency,
    package: &LockedPackage,
    local_packages: &[MetadataPackage],
) -> Result<bool, String> {
    if package.name != dependency.name {
        return Ok(false);
    }
    let requirement = VersionReq::parse(&dependency.req).map_err(|error| {
        format!(
            "parse declared requirement {} for {}: {error}",
            dependency.req, dependency.name
        )
    })?;
    let version = Version::parse(&package.version).map_err(|error| {
        format!(
            "parse locked version {} for {}: {error}",
            package.version, package.name
        )
    })?;
    if !requirement.matches(&version) {
        return Ok(false);
    }

    if let Some(path) = &dependency.path {
        if package.source.is_some() {
            return Ok(false);
        }
        let declared_path = path
            .canonicalize()
            .map_err(|error| format!("resolve declared dependency {}: {error}", path.display()))?;
        return Ok(local_packages.iter().any(|candidate| {
            candidate.name == package.name
                && candidate.version == package.version
                && candidate.source.is_none()
                && candidate
                    .manifest_path
                    .parent()
                    .and_then(|parent| parent.canonicalize().ok())
                    .is_some_and(|resolved_path| resolved_path == declared_path)
        }));
    }

    match (&dependency.source, &package.source) {
        // A crates.io declaration may resolve from a Git or local `[patch]` /
        // `[replace]`. The exact root lock edge, package name, and version
        // remain identity-bearing; `--locked` validates the substitution.
        (Some(declared), Some(resolved)) => {
            Ok(source_identity_matches(resolved, declared) || declared.starts_with("registry+"))
        }
        (Some(declared), None) if declared.starts_with("registry+") => {
            Ok(local_packages.iter().any(|candidate| {
                candidate.name == package.name
                    && candidate.version == package.version
                    && candidate.source.is_none()
            }))
        }
        (None, None) => Ok(true),
        _ => Ok(false),
    }
}

fn root_metadata<'a>(
    packages: &'a [MetadataPackage],
    root_manifest_path: &Path,
) -> Result<&'a MetadataPackage, String> {
    let root_manifest = root_manifest_path
        .canonicalize()
        .map_err(|error| format!("resolve {}: {error}", root_manifest_path.display()))?;
    packages
        .iter()
        .find(|package| {
            package
                .manifest_path
                .canonicalize()
                .is_ok_and(|path| path == root_manifest)
        })
        .ok_or_else(|| {
            format!(
                "cargo metadata omitted root package {}",
                root_manifest.display()
            )
        })
}

fn assert_root_dependencies_locked(
    metadata: &CargoMetadata,
    lock: &CargoLock,
    root_manifest_path: &Path,
) -> Result<(), String> {
    let root = root_metadata(&metadata.packages, root_manifest_path)?;
    let root_index = locked_root_index(&lock.package, root)?;
    let locked_root = &lock.package[root_index];
    let locked_dependency_indices = locked_root
        .dependencies
        .iter()
        .map(|dependency| locked_dependency_index(&lock.package, dependency))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for dependency in &root.dependencies {
        let matches = locked_dependency_indices
            .iter()
            .copied()
            .map(|index| {
                declared_dependency_matches_locked(
                    dependency,
                    &lock.package[index],
                    &metadata.packages,
                )
                .map(|matches| matches.then_some(index))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let alias = dependency.rename.as_deref().unwrap_or(&dependency.name);
        match matches.as_slice() {
            [_] => {}
            [] => {
                return Err(format!(
                    "Cargo.lock omitted exact resolved root dependency {alias} \
                     (package {} {}, source {:?}, path {:?})",
                    dependency.name, dependency.req, dependency.source, dependency.path
                ));
            }
            _ => {
                return Err(format!(
                    "declared root dependency {alias} ambiguously matches multiple lock identities"
                ));
            }
        }
    }
    Ok(())
}

fn check_locked_apache_boundary_with(
    policy_root: &Path,
    root_manifest_path: &Path,
    local_packages: &[MetadataPackage],
    lock: &CargoLock,
) -> Result<(), String> {
    let policy_agpl = policy_agpl_packages(policy_root)?;
    let root = root_metadata(local_packages, root_manifest_path)?;
    let root_index = locked_root_index(&lock.package, root)?;

    let mut queue = VecDeque::from([(root_index, Vec::<String>::new())]);
    let mut visited = BTreeSet::new();
    while let Some((index, mut chain)) = queue.pop_front() {
        if !visited.insert(index) {
            continue;
        }
        let package = &lock.package[index];
        chain.push(package.name.clone());

        if package.source.is_none() {
            let metadata_package = local_packages
                .iter()
                .find(|candidate| {
                    candidate.name == package.name
                        && candidate.version == package.version
                        && candidate.source.is_none()
                })
                .ok_or_else(|| {
                    format!(
                        "resolved local package {} {} has no local manifest metadata; permissive boundary fails closed",
                        package.name, package.version
                    )
                })?;
            let license = metadata_package.license.as_deref().ok_or_else(|| {
                format!(
                    "{} has no resolved package license; permissive boundary fails closed",
                    metadata_package.manifest_path.display()
                )
            })?;
            if policy_agpl.contains(&package.name) || contains_agpl_identifier(license) {
                return Err(format!(
                    "permissive-root-to-AGPL dependency: {} ({license})",
                    chain.join(" -> ")
                ));
            }
        }

        for dependency in &package.dependencies {
            let dependency_index = locked_dependency_index(&lock.package, dependency)?;
            queue.push_back((dependency_index, chain.clone()));
        }
    }
    Ok(())
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
                    "{} has no resolved package license; permissive boundary fails closed",
                    package.manifest_path.display()
                )
            })?;
            if policy_agpl.contains(&package.name) || contains_agpl_identifier(license) {
                return Err(format!(
                    "permissive-root-to-AGPL dependency: {} ({license})",
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
fn production_every_permissive_package_is_a_strict_root() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonical workspace root");
    let roots = validate_permissive_root_partition(&workspace_root).unwrap();
    assert!(!roots.is_empty(), "policy must select reusable roots");

    let mut workspace_metadata =
        local_metadata(&workspace_root, &workspace_root.join("Cargo.toml")).unwrap();
    let workspace_lock = load_cargo_lock(&workspace_root.join("Cargo.lock")).unwrap();
    let standalone_manifests = policy_standalone_manifests(&workspace_root).unwrap();
    for manifest_path in &standalone_manifests {
        let package = standalone_manifest_metadata(manifest_path).unwrap();
        assert!(
            workspace_metadata
                .packages
                .iter()
                .all(|candidate| candidate.name != package.name),
            "standalone manifest {} duplicates a workspace package",
            manifest_path.display()
        );
        workspace_metadata.packages.push(package);
    }

    let mut checked = BTreeSet::new();
    for package_name in &roots {
        let package = workspace_metadata
            .packages
            .iter()
            .find(|candidate| &candidate.name == package_name)
            .unwrap_or_else(|| panic!("policy package {package_name} has no local metadata"));
        let license = package
            .license
            .as_deref()
            .expect("package license metadata");
        assert!(
            matches!(license, "MIT" | "Apache-2.0"),
            "permissive package {package_name} has unauthorized license {license}"
        );

        let boundary = if locked_root_index(&workspace_lock.package, package).is_ok() {
            assert_root_dependencies_locked(
                &workspace_metadata,
                &workspace_lock,
                &package.manifest_path,
            )
            .unwrap();
            check_locked_apache_boundary_with(
                &workspace_root,
                &package.manifest_path,
                &workspace_metadata.packages,
                &workspace_lock,
            )
        } else {
            let manifest_path = &package.manifest_path;
            let lock_path = manifest_path
                .parent()
                .expect("standalone manifest parent")
                .join("Cargo.lock");
            let metadata = local_metadata(&workspace_root, manifest_path).unwrap();
            let lock = load_cargo_lock(&lock_path).unwrap();
            assert_root_dependencies_locked(&metadata, &lock, manifest_path).unwrap();
            check_locked_apache_boundary_with(
                &workspace_root,
                manifest_path,
                &metadata.packages,
                &lock,
            )
        };

        boundary.unwrap_or_else(|error| {
            panic!("reusable permissive root {package_name} is not strict: {error}")
        });
        checked.insert(package_name.clone());
    }

    assert_eq!(checked, roots, "every permissive package must be checked");
}

#[test]
fn production_lock_covers_every_declared_root_dependency_without_registry_sources() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonical workspace root");
    let root_manifest_path = workspace_root.join("crates/hyprstream-k8s/Cargo.toml");
    let metadata = local_metadata(&workspace_root, &workspace_root.join("Cargo.toml")).unwrap();
    let lock = load_cargo_lock(&workspace_root.join("Cargo.lock")).unwrap();
    assert_root_dependencies_locked(&metadata, &lock, &root_manifest_path).unwrap();
}

#[test]
fn root_lock_identity_check_rejects_omitted_renamed_agpl_version() {
    let directory = tempfile::tempdir().unwrap();
    write_fixture(
        &directory.path().join("Cargo.toml"),
        r#"[workspace]
members = ["root"]
exclude = ["service-v1", "service-v2"]
resolver = "2"
"#,
    );
    let root_manifest = directory.path().join("root/Cargo.toml");
    write_fixture(
        &root_manifest,
        r#"[package]
name = "apache-root"
version = "0.1.0"
license = "Apache-2.0"

[dependencies]
service_v1 = { package = "service", path = "../service-v1", optional = true }
service_v2 = { package = "service", path = "../service-v2", optional = true }

[features]
both = ["dep:service_v1", "dep:service_v2"]
"#,
    );
    write_fixture(
        &directory.path().join("service-v1/Cargo.toml"),
        r#"[package]
name = "service"
version = "1.0.0"
license = "MIT"
"#,
    );
    write_fixture(
        &directory.path().join("service-v2/Cargo.toml"),
        r#"[package]
name = "service"
version = "2.0.0"
license = "AGPL-3.0-only"
"#,
    );
    for package in ["root", "service-v1", "service-v2"] {
        write_fixture(&directory.path().join(package).join("src/lib.rs"), "");
    }

    let mut metadata = resolved_metadata(directory.path(), &root_manifest)
        .expect("duplicate-version alias fixture must resolve offline");
    let mut lock = load_cargo_lock(&directory.path().join("Cargo.lock"))
        .expect("duplicate-version alias fixture must write Cargo.lock");

    let root_manifest_canonical = root_manifest.canonicalize().unwrap();
    let root_metadata_index = metadata
        .packages
        .iter()
        .position(|package| {
            package
                .manifest_path
                .canonicalize()
                .is_ok_and(|path| path == root_manifest_canonical)
        })
        .expect("fixture root metadata");
    let mut duplicate_alias = metadata.packages[root_metadata_index]
        .dependencies
        .iter()
        .find(|dependency| dependency.rename.as_deref() == Some("service_v1"))
        .expect("service_v1 declaration")
        .clone();
    duplicate_alias.rename = Some("service_v1_duplicate".to_owned());
    metadata.packages[root_metadata_index]
        .dependencies
        .push(duplicate_alias);

    assert_root_dependencies_locked(&metadata, &lock, &root_manifest)
        .expect("multiple aliases may resolve to the same exact lock identity");
    let boundary_error = check_locked_apache_boundary_with(
        directory.path(),
        &root_manifest,
        &metadata.packages,
        &lock,
    )
    .unwrap_err();
    assert!(
        boundary_error.contains("AGPL-3.0-only"),
        "complete fixture lock must reach its AGPL identity: {boundary_error}"
    );

    let root_lock_index =
        locked_root_index(&lock.package, &metadata.packages[root_metadata_index]).unwrap();
    let agpl_edge_index = lock.package[root_lock_index]
        .dependencies
        .iter()
        .position(|dependency| {
            locked_dependency_index(&lock.package, dependency).is_ok_and(|index| {
                let package = &lock.package[index];
                package.name == "service" && package.version == "2.0.0" && package.source.is_none()
            })
        })
        .expect("root edge to AGPL service 2.0.0");
    lock.package[root_lock_index]
        .dependencies
        .remove(agpl_edge_index);

    assert!(
        lock.package[root_lock_index]
            .dependencies
            .iter()
            .any(|dependency| {
                locked_dependency_index(&lock.package, dependency).is_ok_and(|index| {
                    let package = &lock.package[index];
                    package.name == "service" && package.version == "1.0.0"
                })
            }),
        "MIT service 1.0.0 root edge must remain"
    );
    assert!(
        lock.package.iter().any(|package| {
            package.name == "service" && package.version == "2.0.0" && package.source.is_none()
        }),
        "AGPL package identity must remain in the lock after only its root edge is removed"
    );

    let error = assert_root_dependencies_locked(&metadata, &lock, &root_manifest).unwrap_err();
    assert!(
        error.contains("omitted exact resolved root dependency service_v2"),
        "identity consistency must reject the missing AGPL edge before traversal: {error}"
    );
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
agpl_packages = ["new-service"]
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
fn patched_local_agpl_app_omitted_from_permissive_roots_is_rejected() {
    let (directory, root_manifest) = patched_anyhow_fixture(
        r#"[dependencies]
anyhow = "=99.0.0""#,
    );
    write_fixture(
        &root_manifest,
        r#"[package]
name = "app"
version = "0.1.0"
license = "Apache-2.0"

[dependencies]
anyhow = "=99.0.0"
"#,
    );
    write_fixture(
        &directory.path().join(".github/license-boundary.toml"),
        r#"[license_gate]
mit_packages = []
agpl_packages = ["anyhow"]
apache_packages = ["app"]
permissive_roots = []
"#,
    );

    let resolved_error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(
        resolved_error.contains("app -> anyhow"),
        "fixture must prove the registry declaration resolves to local AGPL: {resolved_error}"
    );
    let role_error = validate_permissive_root_partition(directory.path()).unwrap_err();
    assert!(
        role_error.contains("omissions: [\"app\"]"),
        "an app omitted from permissive roots must fail closed: {role_error}"
    );
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
    let lock = load_cargo_lock(&directory.path().join("Cargo.lock"))
        .expect("transitive patch fixture must write a resolved Cargo.lock");
    let locked_error = check_locked_apache_boundary_with(
        directory.path(),
        &root_manifest,
        &metadata.packages,
        &lock,
    )
    .unwrap_err();
    assert!(
        locked_error.contains("apache-root -> serde_json -> itoa"),
        "locked graph omitted complete dependency chain: {locked_error}"
    );

    let error = check_apache_boundary(directory.path(), &root_manifest).unwrap_err();
    assert!(
        error.contains("apache-root -> serde_json -> itoa"),
        "complete dependency chain missing from boundary failure: {error}"
    );
}
