#!/usr/bin/env python3
"""Fail-closed validation for the crates.io release allowlist."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / ".github" / "crate-release.toml"


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def fail(message: str) -> None:
    print(f"crate release policy: {message}", file=sys.stderr)
    raise SystemExit(1)


def local_dependencies(manifest: dict) -> list[tuple[str, str, dict]]:
    result = []
    for section in ("dependencies", "build-dependencies", "dev-dependencies"):
        for name, value in manifest.get(section, {}).items():
            if isinstance(value, dict) and "path" in value:
                result.append((section, name, value))
    return result


def main() -> None:
    policy = load(POLICY_PATH)
    groups = {key: policy[key] for key in ("public_now", "public_later", "internal")}
    listed = [name for values in groups.values() for name in values]
    if len(listed) != len(set(listed)):
        fail("a package appears in more than one classification")

    manifests: dict[str, tuple[Path, dict]] = {}
    for path in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        manifest = load(path)
        name = manifest["package"]["name"]
        if name in manifests:
            fail(f"duplicate package name {name}")
        manifests[name] = (path, manifest)

    missing = sorted(set(manifests) - set(listed))
    stale = sorted(set(listed) - set(manifests))
    if missing or stale:
        fail(f"classification mismatch; unclassified={missing}, unknown={stale}")

    public = groups["public_now"]
    versions = set()
    position = {name: index for index, name in enumerate(public)}
    for name, (path, manifest) in manifests.items():
        publish = manifest["package"].get("publish")
        if name in position:
            if publish != [policy["registry"]]:
                fail(f"{path.relative_to(ROOT)} must set publish = [\"{policy['registry']}\"]")
            versions.add(manifest["package"]["version"])
            for section, dependency, value in local_dependencies(manifest):
                if dependency not in position:
                    fail(f"public crate {name} has non-public local {section} {dependency}")
                if position[dependency] >= position[name]:
                    fail(f"{dependency} must precede {name} in public_now")
                requirement = value.get("version")
                dependency_version = manifests[dependency][1]["package"]["version"]
                if requirement != f"={dependency_version}":
                    fail(
                        f"{name}'s {dependency} dependency must pin version "
                        f"=\"={dependency_version}\" alongside path"
                    )
            for section in ("dependencies", "build-dependencies"):
                for dependency, value in manifest.get(section, {}).items():
                    if isinstance(value, dict) and "git" in value:
                        fail(f"public crate {name} has git {section} {dependency}")
        elif publish is not False:
            fail(f"{path.relative_to(ROOT)} must set publish = false")

    if len(versions) != 1:
        fail(f"public_now must use one release-train version, got {sorted(versions)}")

    version = versions.pop()
    print(
        f"crate release policy OK: {len(public)} public now at {version}, "
        f"{len(groups['public_later'])} public later, {len(groups['internal'])} internal"
    )


if __name__ == "__main__":
    main()
