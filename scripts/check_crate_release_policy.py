#!/usr/bin/env python3
"""Fail-closed validation for the crates.io release allowlist."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / ".github" / "crate-release.toml"
SHIM_VERSION = "0.5.0"


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

    license_boundary = policy.get("license_boundary")
    if not isinstance(license_boundary, dict):
        fail("missing license_boundary ruling")
    license_policy_path = license_boundary.get("policy_path")
    if license_policy_path != ".github/license-boundary.toml":
        fail("license_boundary.policy_path must name .github/license-boundary.toml")
    license_policy = load(ROOT / license_policy_path).get("license_gate", {})
    for key in ("mit_packages", "agpl_packages"):
        if license_boundary.get(key) != license_policy.get(key):
            fail(f"license_boundary.{key} diverges from {license_policy_path}")
    if license_boundary.get("all_other_packages") != "Apache-2.0":
        fail("license_boundary.all_other_packages must remain Apache-2.0")
    license_classes = {
        key: set(license_policy.get(key, []))
        for key in ("mit_packages", "agpl_packages", "apache_packages")
    }
    if (
        license_classes["apache_packages"]
        != set(manifests)
        - license_classes["mit_packages"]
        - license_classes["agpl_packages"]
    ):
        fail("license policy does not classify every other crate as Apache-2.0")

    shim = policy["deprecation_shim"]
    source_service = shim["source_service"]
    if source_service not in manifests:
        fail(f"deprecation shim names unknown source service {source_service}")
    if shim["package_name"] != source_service:
        fail("deprecation shim must reserve the source service's crates.io name")
    if shim["license"] != "Apache-2.0":
        fail("deprecation shim must remain Apache-2.0")
    if shim["version"] != SHIM_VERSION:
        fail(f"deprecation shim version must remain {SHIM_VERSION}, not {shim['version']}")
    if shim["required_before_yank"] is not True:
        fail("deprecation shim must be required before yanking the stale package")
    if shim["release_order"] != ["publish-shim", "verify-shim", "yank-stale-versions"]:
        fail("deprecation shim release order must publish, verify, then yank stale versions")
    if manifests[source_service][1]["package"].get("publish") is not False:
        fail(f"real service {source_service} must remain non-publishable; use the separate shim")

    shim_relative_path = Path(shim["manifest_path"])
    if shim_relative_path.is_absolute() or ".." in shim_relative_path.parts:
        fail("deprecation shim manifest path must stay relative to the repository root")
    shim_path = ROOT / shim_relative_path
    if not shim_path.is_file():
        fail(f"deprecation shim manifest is missing: {shim['manifest_path']}")
    try:
        shim_path.relative_to(ROOT / "crates")
    except ValueError:
        pass
    else:
        fail("deprecation shim must stay outside crates/ to avoid the real-service name collision")
    root_workspace = load(ROOT / "Cargo.toml").get("workspace", {})
    if str(shim_path.parent.relative_to(ROOT)) not in root_workspace.get("exclude", []):
        fail("deprecation shim must be excluded from the normal workspace")

    shim_manifest = load(shim_path)
    shim_package = shim_manifest.get("package", {})
    if shim_package.get("name") != shim["package_name"]:
        fail("deprecation shim manifest must reserve the hyprstream package name")
    if shim_package.get("version") != shim["version"]:
        fail("deprecation shim manifest version must match deprecation_shim.version")
    if shim_package.get("version") != manifests[source_service][1]["package"]["version"]:
        fail("deprecation shim version must equal the real service version")
    if shim_package.get("license") != shim["license"]:
        fail("deprecation shim manifest must be Apache-2.0")
    if shim_package.get("publish") != [policy["registry"]]:
        fail(f"deprecation shim manifest must set publish = [\"{policy['registry']}\"]")
    for section in ("dependencies", "build-dependencies", "dev-dependencies"):
        if shim_manifest.get(section):
            fail(f"deprecation shim must not carry {section}")

    shim_lib = shim_path.parent / shim_manifest.get("lib", {}).get("path", "")
    if not shim_lib.is_file():
        fail("deprecation shim must include a documentation-only library target")
    for line in shim_lib.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            fail("deprecation shim library must not contain product implementation")

    readme_name = shim_package.get("readme")
    readme_path = shim_path.parent / readme_name if isinstance(readme_name, str) else None
    if readme_path is None or not readme_path.is_file():
        fail("deprecation shim must include its README")
    readme = readme_path.read_text(encoding="utf-8").lower()
    for required_text in ("deprecated", "metrics", "install", "client"):
        if required_text not in readme:
            fail(f"deprecation shim README must include current {required_text} guidance")

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
