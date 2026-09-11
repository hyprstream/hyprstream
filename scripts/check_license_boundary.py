#!/usr/bin/env python3
"""Enforce the owner's exhaustive package licenses and one-way AGPL boundary."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tomllib
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASS_LICENSES = {
    "mit_packages": "MIT",
    "agpl_packages": "AGPL-3.0-only",
    "apache_packages": "Apache-2.0",
}


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def fail(message: str) -> None:
    raise SystemExit(f"license boundary: {message}")


def cargo_workspace_manifest_paths(root: Path) -> set[Path]:
    cargo = shlex.split(os.environ.get("LICENSE_BOUNDARY_CARGO_COMMAND", "cargo"))
    command = cargo + [
        "metadata",
        "--format-version=1",
        "--all-features",
        "--offline",
        "--no-deps",
        "--manifest-path",
        str(root / "Cargo.toml"),
    ]
    if (root / "Cargo.lock").is_file():
        command.insert(len(cargo) + 4, "--locked")
    else:
        command.remove("--no-deps")
    # The license-boundary checker is metadata-only.  Its fixture subprocesses
    # must not inherit a workflow-level compiler wrapper: the hosted deny job
    # deliberately does not install sccache, and Cargo still consults
    # RUSTC_WRAPPER while probing rustc for `cargo metadata`.
    environment = os.environ.copy()
    environment.pop("RUSTC_WRAPPER", None)
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if result.returncode:
        fail(f"cargo metadata failed: {result.stderr.strip()}")
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        fail(f"cargo metadata returned invalid JSON: {error}")

    workspace_members = set(metadata.get("workspace_members", []))
    packages = {
        package.get("id"): package
        for package in metadata.get("packages", [])
        if isinstance(package, dict)
    }
    if workspace_members - set(packages):
        fail("cargo metadata omitted workspace member package data")

    manifest_paths: set[Path] = set()
    for package_id in workspace_members:
        manifest_value = packages[package_id].get("manifest_path")
        if not isinstance(manifest_value, str):
            fail(f"cargo metadata package {package_id!r} has no manifest path")
        manifest_path = Path(manifest_value).resolve()
        if not manifest_path.is_relative_to(root):
            fail(f"workspace member is outside repository: {manifest_path}")
        manifest_paths.add(manifest_path)
    return manifest_paths


def configured_standalone_manifest_paths(root: Path, config: dict) -> set[Path]:
    values = config.get("standalone_manifests")
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        fail("missing or non-string classification 'standalone_manifests'")
    if len(values) != len(set(values)):
        fail(f"duplicate manifest in standalone_manifests: {values}")

    paths: set[Path] = set()
    for value in values:
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            fail(f"standalone manifest is outside repository: {value}")
        if path.name != "Cargo.toml" or not path.is_file():
            fail(f"standalone manifest does not exist: {value}")
        paths.add(path)

    root_manifest = load(root / "Cargo.toml")
    exclusions = root_manifest.get("workspace", {}).get("exclude", [])
    if not isinstance(exclusions, list) or not all(
        isinstance(value, str) for value in exclusions
    ):
        fail("workspace.exclude must be a list of paths")
    excluded_package_manifests: set[Path] = set()
    for pattern in exclusions:
        for excluded_path in root.glob(pattern):
            manifest_path = (
                excluded_path
                if excluded_path.name == "Cargo.toml"
                else excluded_path / "Cargo.toml"
            )
            if manifest_path.is_file():
                excluded_package_manifests.add(manifest_path.resolve())
    omitted = excluded_package_manifests - paths
    unsupported = paths - excluded_package_manifests
    if omitted or unsupported:
        fail(
            "standalone_manifests/workspace.exclude package mismatch; "
            f"omitted: {[str(path.relative_to(root)) for path in sorted(omitted)]}; "
            "not excluded: "
            f"{[str(path.relative_to(root)) for path in sorted(unsupported)]}"
        )
    return paths


def package_manifests(
    root: Path, config: dict
) -> dict[str, tuple[Path, dict]]:
    manifests: dict[str, tuple[Path, dict]] = {}
    workspace_paths = cargo_workspace_manifest_paths(root)
    standalone_paths = configured_standalone_manifest_paths(root, config)
    overlap = workspace_paths & standalone_paths
    if overlap:
        fail(
            "standalone_manifests contains Cargo workspace member(s): "
            f"{[str(path.relative_to(root)) for path in sorted(overlap)]}"
        )
    manifest_paths = workspace_paths | standalone_paths
    for path in sorted(manifest_paths):
        manifest = load(path)
        package = manifest.get("package")
        if not isinstance(package, dict) or "name" not in package:
            continue
        name = str(package["name"])
        if name in manifests:
            fail(f"duplicate package name {name!r}")
        manifests[name] = (path, manifest)
    return manifests


def effective_package_license(
    root: Path,
    manifest_path: Path,
    manifest: dict,
    root_manifest: dict,
) -> str:
    package = manifest["package"]
    license_value = package.get("license")
    if isinstance(license_value, str):
        if not license_value:
            fail(f"{manifest_path}: package license is blank")
        return license_value

    if (
        isinstance(license_value, dict)
        and license_value.get("workspace") is True
        and set(license_value) == {"workspace"}
    ):
        workspace_manifest = (
            manifest
            if manifest_path == root / "Cargo.toml" or "workspace" in manifest
            else root_manifest
        )
        workspace_license = (
            workspace_manifest.get("workspace", {})
            .get("package", {})
            .get("license")
        )
        if not isinstance(workspace_license, str) or not workspace_license:
            fail(
                f"{manifest_path}: package license inherits a missing or blank "
                "workspace.package.license"
            )
        return workspace_license

    fail(f"{manifest_path}: package must declare one exact license")


def package_identity(manifest_path: Path, manifest: dict) -> tuple[str, str]:
    package = manifest["package"]
    name = package.get("name")
    version = package.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        fail(f"{manifest_path}: package must declare a string name and version")
    return name, version


def locked_dependency_index(packages: list[dict], dependency: str) -> int:
    fields = dependency.split(maxsplit=2)
    name = fields[0]
    candidates = [
        index
        for index, package in enumerate(packages)
        if package.get("name") == name
    ]
    if len(fields) >= 2:
        candidates = [
            index
            for index in candidates
            if packages[index].get("version") == fields[1]
        ]
    if len(fields) == 3:
        source = fields[2]
        if not source.startswith("(") or not source.endswith(")"):
            fail(f"invalid Cargo.lock dependency source: {dependency}")
        source = source[1:-1].split("#", 1)[0]
        candidates = [
            index
            for index in candidates
            if str(packages[index].get("source", "")).split("#", 1)[0] == source
        ]
    if len(candidates) != 1:
        fail(
            f"Cargo.lock dependency {dependency!r} resolves to "
            f"{len(candidates)} package identities"
        )
    return candidates[0]


def resolved_lock_graphs(
    root: Path,
    manifests: dict[str, tuple[Path, dict]],
    standalone_manifests: set[Path],
) -> list[tuple[Path, list[dict]]]:
    lock_paths = {root / "Cargo.lock"}
    lock_paths.update(
        manifest_path.parent / "Cargo.lock" for manifest_path in standalone_manifests
    )
    known_local = {
        package_identity(manifest_path, manifest)
        for manifest_path, manifest in manifests.values()
    }
    graphs: list[tuple[Path, list[dict]]] = []
    for lock_path in sorted(lock_paths):
        if not lock_path.is_file():
            continue
        lock = load(lock_path)
        packages = lock.get("package")
        if not isinstance(packages, list) or not all(
            isinstance(package, dict) for package in packages
        ):
            fail(f"{lock_path}: Cargo.lock has no package graph")
        for package in packages:
            name = package.get("name")
            version = package.get("version")
            if not isinstance(name, str) or not isinstance(version, str):
                fail(f"{lock_path}: resolved package has invalid identity")
            if package.get("source") is None and (name, version) not in known_local:
                fail(
                    f"{lock_path}: resolved local package {name} {version} "
                    "is outside the owner package universe"
                )
            dependencies = package.get("dependencies", [])
            if not isinstance(dependencies, list) or not all(
                isinstance(dependency, str) for dependency in dependencies
            ):
                fail(f"{lock_path}: {name} has malformed resolved dependencies")
        graphs.append((lock_path, packages))
    if not graphs:
        fail("no Cargo.lock graph covers the owner package universe")
    return graphs


def resolved_path_to_agpl(
    packages: list[dict],
    start: tuple[str, str],
    agpl_names: set[str],
) -> list[str] | None:
    root_indices = [
        index
        for index, package in enumerate(packages)
        if package.get("name") == start[0]
        and package.get("version") == start[1]
        and package.get("source") is None
    ]
    if len(root_indices) > 1:
        fail(f"Cargo.lock has ambiguous local package identity {start[0]} {start[1]}")
    if not root_indices:
        return None

    queue = deque([(root_indices[0], [start[0]])])
    visited: set[int] = set()
    while queue:
        index, chain = queue.popleft()
        if index in visited:
            continue
        visited.add(index)
        package = packages[index]
        if package.get("source") is None and package["name"] in agpl_names:
            return chain
        for dependency in package.get("dependencies", []):
            dependency_index = locked_dependency_index(packages, dependency)
            queue.append(
                (dependency_index, chain + [str(packages[dependency_index]["name"])])
            )
    return None


def resolved_paths_for_permissive_packages(
    root: Path,
    manifests: dict[str, tuple[Path, dict]],
    permissive: set[str],
    agpl: set[str],
    standalone_manifests: set[Path],
) -> dict[str, list[str] | None]:
    graphs = resolved_lock_graphs(root, manifests, standalone_manifests)
    paths: dict[str, list[str] | None] = {}
    for package_name in sorted(permissive):
        manifest_path, manifest = manifests[package_name]
        identity = package_identity(manifest_path, manifest)
        covered = False
        found_path = None
        for _, packages in graphs:
            in_graph = any(
                package.get("name") == identity[0]
                and package.get("version") == identity[1]
                and package.get("source") is None
                for package in packages
            )
            if not in_graph:
                continue
            covered = True
            path = resolved_path_to_agpl(packages, identity, agpl)
            if path is not None:
                found_path = path
                break
        if not covered:
            resolved_locals = sorted(
                {
                    (str(lock_path), str(package.get("name")), str(package.get("version")))
                    for lock_path, packages in graphs
                    for package in packages
                    if package.get("source") is None
                }
            )
            fail(
                f"permissive package {package_name!r} has no committed "
                f"Cargo-resolved lock graph; resolved locals: {resolved_locals}"
            )
        paths[package_name] = found_path
    return paths


def classifications(config: dict, package_names: set[str]) -> dict[str, set[str]]:
    classes: dict[str, set[str]] = {}
    for class_name in CLASS_LICENSES:
        values = config.get(class_name)
        if not isinstance(values, list):
            fail(f"missing classification {class_name!r}")
        if not all(isinstance(value, str) for value in values):
            fail(f"non-string package in {class_name}: {values}")
        if len(values) != len(set(values)):
            fail(f"duplicate package in {class_name}: {values}")
        classes[class_name] = set(values)

    classified: set[str] = set()
    for class_name in CLASS_LICENSES:
        overlap = classified & classes[class_name]
        if overlap:
            fail(f"classification overlap: {sorted(overlap)}")
        classified |= classes[class_name]

    unknown = classified - package_names
    if unknown:
        fail(f"unknown package(s): {sorted(unknown)}")
    omitted = package_names - classified
    if omitted:
        fail(f"classification omission: {sorted(omitted)}")
    return classes


def named_package_set(config: dict, key: str, package_names: set[str]) -> set[str]:
    values = config.get(key)
    if not isinstance(values, list):
        fail(f"missing classification {key!r}")
    if not all(isinstance(value, str) for value in values):
        fail(f"non-string package in {key}: {values}")
    if len(values) != len(set(values)):
        fail(f"duplicate package in {key}: {values}")
    packages = set(values)
    unknown = packages - package_names
    if unknown:
        fail(f"unknown package(s) in {key}: {sorted(unknown)}")
    return packages


def main(root: Path = ROOT) -> None:
    root = root.resolve()
    root_manifest = load(root / "Cargo.toml")
    config = load(root / ".github" / "license-boundary.toml")["license_gate"]
    manifests = package_manifests(root, config)
    classes = classifications(config, set(manifests))
    for class_name, expected_license in CLASS_LICENSES.items():
        for package_name in sorted(classes[class_name]):
            manifest_path, manifest = manifests[package_name]
            actual_license = effective_package_license(
                root, manifest_path, manifest, root_manifest
            )
            if actual_license != expected_license:
                fail(
                    f"{package_name} is classified as {class_name} requiring "
                    f"{expected_license!r}, but its resolved manifest license is "
                    f"{actual_license!r}"
                )

    permissive = classes["mit_packages"] | classes["apache_packages"]
    agpl = classes["agpl_packages"]
    permissive_roots = named_package_set(
        config, "permissive_roots", set(manifests)
    )
    non_permissive_roots = permissive_roots - permissive
    if non_permissive_roots:
        fail(
            "permissive_roots contains non-MIT/Apache package(s): "
            f"{sorted(non_permissive_roots)}"
        )
    role_omissions = permissive - permissive_roots
    if role_omissions:
        fail(
            "permissive package root omission; every MIT/Apache package must "
            f"be guarded: {sorted(role_omissions)}"
        )

    paths_to_agpl = resolved_paths_for_permissive_packages(
        root,
        manifests,
        permissive,
        agpl,
        configured_standalone_manifest_paths(root, config),
    )
    for permissive_root in sorted(permissive_roots):
        path = paths_to_agpl[permissive_root]
        if path is not None:
            fail("permissive-root-to-AGPL dependency: " + " -> ".join(path))

    print(
        "license boundary OK: "
        f"{len(permissive_roots)} reusable MIT/Apache roots do not reach "
        f"{len(agpl)} AGPL packages; {len(manifests)} package licenses match policy"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.root)
