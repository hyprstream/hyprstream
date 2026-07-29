#!/usr/bin/env python3
"""Reject a configured Apache-2.0 root that reaches an AGPL-3.0-only service."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".github" / "license-boundary.toml"
DEPENDENCY_SECTIONS = {"dependencies", "build-dependencies", "dev-dependencies"}


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def fail(message: str) -> None:
    raise SystemExit(f"license boundary: {message}")


def local_dependency_names(manifest: dict) -> set[str]:
    result: set[str] = set()

    def walk(value: object) -> None:
        if not isinstance(value, dict):
            return
        for key, section in value.items():
            if key in DEPENDENCY_SECTIONS and isinstance(section, dict):
                for name, dependency in section.items():
                    if isinstance(dependency, dict) and "path" in dependency:
                        result.add(str(dependency.get("package", name)))
            else:
                walk(section)

    walk(manifest)
    return result


def main() -> None:
    config = load(CONFIG_PATH)["license_gate"]
    apache = set(config["apache_roots"])
    services = set(config["agpl_services"])
    if apache & services:
        fail(f"classification overlap: {sorted(apache & services)}")

    manifests = {
        manifest["package"]["name"]: manifest
        for path in sorted((ROOT / "crates").glob("*/Cargo.toml"))
        if (manifest := load(path))
    }
    unknown = (apache | services) - set(manifests)
    if unknown:
        fail(f"unknown package(s): {sorted(unknown)}")

    graph = {
        name: local_dependency_names(manifest)
        for name, manifest in manifests.items()
    }
    for root in sorted(apache):
        queue = [(root, [root])]
        visited = {root}
        while queue:
            current, chain = queue.pop(0)
            for dependency in sorted(graph[current]):
                if dependency not in manifests:
                    fail(f"{current} has an unknown local dependency {dependency}")
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
    main()
