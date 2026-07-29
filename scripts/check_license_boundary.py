#!/usr/bin/env python3
"""Fail if a planned permissive crate reaches a planned AGPL service crate."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / ".github" / "crate-release.toml"


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def fail(message: str) -> None:
    raise SystemExit(f"license boundary: {message}")


DEPENDENCY_SECTIONS = {"dependencies", "build-dependencies", "dev-dependencies"}


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
    policy = load(POLICY_PATH)["license_boundary"]
    services = set(policy["agpl_services"])
    permissive = set(policy["permissive_crates"])
    if services & permissive:
        fail(f"classification overlap: {sorted(services & permissive)}")

    manifests = {
        load(path)["package"]["name"]: load(path)
        for path in sorted((ROOT / "crates").glob("*/Cargo.toml"))
    }
    unknown = (services | permissive) - set(manifests)
    if unknown:
        fail(f"unknown package(s): {sorted(unknown)}")

    graph = {
        name: local_dependency_names(manifest)
        for name, manifest in manifests.items()
    }
    for root in sorted(permissive):
        queue = [(root, [root])]
        visited = {root}
        while queue:
            current, chain = queue.pop(0)
            for dependency in sorted(graph[current]):
                if dependency not in manifests:
                    fail(f"{current} has an unknown local dependency {dependency}")
                if dependency in services:
                    fail("permissive-to-AGPL dependency: " + " -> ".join(chain + [dependency]))
                if dependency not in visited:
                    visited.add(dependency)
                    queue.append((dependency, chain + [dependency]))

    print(
        "license boundary OK: "
        f"{len(permissive)} permissive roots do not reach {len(services)} AGPL services"
    )


if __name__ == "__main__":
    main()
