#!/usr/bin/env python3
"""Fail if a planned Apache-2.0 crate reaches a planned AGPL service crate."""

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
    apache = set(policy["apache_crates"])
    runtime_targets = set(policy["apache_runtime_targets"])
    runtime_blocked = set(policy["topology_blocked_runtime_targets"])
    inference_runtime = policy["inference_runtime"]
    if policy["apache_license"] != "Apache-2.0":
        fail("Apache-2.0 must be the single permissive license")
    if inference_runtime["target_license"] != policy["apache_license"]:
        fail("inference runtime must use the selected Apache-2.0 license")
    if services & apache:
        fail(f"classification overlap: {sorted(services & apache)}")
    if not runtime_blocked <= runtime_targets:
        fail(
            "topology-blocked runtime target is not declared Apache-2.0: "
            f"{sorted(runtime_blocked - runtime_targets)}"
        )

    manifests = {
        load(path)["package"]["name"]: load(path)
        for path in sorted((ROOT / "crates").glob("*/Cargo.toml"))
    }
    unknown = (services | apache | runtime_targets) - set(manifests)
    if unknown:
        fail(f"unknown package(s): {sorted(unknown)}")
    if inference_runtime["source_crate"] not in services:
        fail("inference runtime source crate must remain an AGPL service until extraction")
    if not inference_runtime["extraction_required"]:
        fail("inference runtime must require extraction before the license split")

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
        f"{len(apache)} Apache-2.0 roots do not reach {len(services)} AGPL services; "
        f"{len(runtime_blocked)} runtime target(s) await topology extraction"
    )


if __name__ == "__main__":
    main()
