#!/usr/bin/env python3
"""Reject a configured Apache-2.0 root that reaches an AGPL-3.0-only service."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_SECTIONS = {"dependencies", "build-dependencies", "dev-dependencies"}
CLASS_NAMES = ("apache_roots", "agpl_services", "other_packages")


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def fail(message: str) -> None:
    raise SystemExit(f"license boundary: {message}")


def package_manifests(root: Path) -> dict[str, tuple[Path, dict]]:
    manifests: dict[str, tuple[Path, dict]] = {}
    for path in sorted((root / "crates").rglob("Cargo.toml")):
        manifest = load(path)
        package = manifest.get("package")
        if not isinstance(package, dict) or "name" not in package:
            continue
        name = str(package["name"])
        if name in manifests:
            fail(f"duplicate package name {name!r}")
        manifests[name] = (path, manifest)
    return manifests


def dependency_tables(manifest: dict) -> list[dict]:
    tables = [
        manifest[section]
        for section in DEPENDENCY_SECTIONS
        if isinstance(manifest.get(section), dict)
    ]
    targets = manifest.get("target", {})
    if isinstance(targets, dict):
        for target in targets.values():
            if not isinstance(target, dict):
                continue
            tables.extend(
                target[section]
                for section in DEPENDENCY_SECTIONS
                if isinstance(target.get(section), dict)
            )
    return tables


def local_dependency_name(
    alias: str,
    dependency: object,
    manifest_path: Path,
    workspace_dependencies: dict,
    workspace_manifest_path: Path,
) -> str | None:
    if not isinstance(dependency, dict):
        return None

    source = dependency
    source_manifest = manifest_path
    if dependency.get("workspace") is True:
        if alias not in workspace_dependencies:
            fail(
                f"{manifest_path}: workspace dependency {alias!r} "
                "is missing from [workspace.dependencies]"
            )
        inherited = workspace_dependencies[alias]
        if not isinstance(inherited, dict):
            return None
        source = inherited
        source_manifest = workspace_manifest_path

    path = source.get("path")
    if path is None:
        return None

    dependency_manifest_path = (source_manifest.parent / str(path) / "Cargo.toml").resolve()
    if not dependency_manifest_path.is_file():
        fail(
            f"{manifest_path}: local dependency {alias!r} "
            f"has no manifest at {dependency_manifest_path}"
        )
    dependency_manifest = load(dependency_manifest_path)
    package = dependency_manifest.get("package")
    if not isinstance(package, dict) or "name" not in package:
        fail(f"{dependency_manifest_path}: local dependency has no package name")
    package_name = str(package["name"])
    declared_name = source.get("package")
    if declared_name is not None and str(declared_name) != package_name:
        fail(
            f"{manifest_path}: local dependency {alias!r} declares package "
            f"{declared_name!r}, but {dependency_manifest_path} is {package_name!r}"
        )
    return package_name


def dependency_graph(
    root: Path,
    manifests: dict[str, tuple[Path, dict]],
    root_manifest: dict,
) -> dict[str, set[str]]:
    root_workspace_dependencies = root_manifest.get("workspace", {}).get("dependencies", {})
    if not isinstance(root_workspace_dependencies, dict):
        root_workspace_dependencies = {}

    graph: dict[str, set[str]] = {}
    for name, (manifest_path, manifest) in manifests.items():
        own_workspace = manifest.get("workspace")
        if isinstance(own_workspace, dict):
            workspace_dependencies = own_workspace.get("dependencies", {})
            workspace_manifest_path = manifest_path
        else:
            workspace_dependencies = root_workspace_dependencies
            workspace_manifest_path = root / "Cargo.toml"
        if not isinstance(workspace_dependencies, dict):
            workspace_dependencies = {}
        dependencies: set[str] = set()
        for table in dependency_tables(manifest):
            for alias, dependency in table.items():
                dependency_name = local_dependency_name(
                    str(alias),
                    dependency,
                    manifest_path,
                    workspace_dependencies,
                    workspace_manifest_path,
                )
                if dependency_name is not None:
                    dependencies.add(dependency_name)
        graph[name] = dependencies
    return graph


def classifications(config: dict, package_names: set[str]) -> dict[str, set[str]]:
    classes: dict[str, set[str]] = {}
    for class_name in CLASS_NAMES:
        values = config.get(class_name)
        if not isinstance(values, list):
            fail(f"missing classification {class_name!r}")
        if len(values) != len(set(values)):
            fail(f"duplicate package in {class_name}: {values}")
        classes[class_name] = {str(value) for value in values}

    classified: set[str] = set()
    for class_name in CLASS_NAMES:
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


def main(root: Path = ROOT) -> None:
    root = root.resolve()
    root_manifest = load(root / "Cargo.toml")
    config = load(root / ".github" / "license-boundary.toml")["license_gate"]
    manifests = package_manifests(root)
    classes = classifications(config, set(manifests))
    apache = classes["apache_roots"]
    services = classes["agpl_services"]

    graph = dependency_graph(root, manifests, root_manifest)
    for package, dependencies in sorted(graph.items()):
        unknown_dependencies = dependencies - set(manifests)
        if unknown_dependencies:
            fail(
                f"{package} has unknown local dependency package(s): "
                f"{sorted(unknown_dependencies)}"
            )

    for apache_root in sorted(apache):
        queue = [(apache_root, [apache_root])]
        visited = {apache_root}
        while queue:
            current, chain = queue.pop(0)
            for dependency in sorted(graph[current]):
                if dependency in services:
                    fail("Apache-2.0-to-AGPL dependency: " + " -> ".join(chain + [dependency]))
                if dependency not in visited:
                    visited.add(dependency)
                    queue.append((dependency, chain + [dependency]))

    print(
        "license boundary OK: "
        f"{len(apache)} Apache-2.0 roots do not reach {len(services)} AGPL services"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.root)
