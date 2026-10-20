#!/usr/bin/env python3
"""Program-boundary check: program leaf crates stay leaves.

Program-specific crates (e.g. `hyprstream-synthesis`, System One P1.3) are
workspace members so they build and gate with the platform toolchain, but
they are APPLICATIONS, not platform dependencies: no other workspace package
may declare a dependency on them. If a platform crate ever needs something
from a program crate, that is a signal the code belongs in a platform crate
instead — move it, do not link it.

Program leaf crates are listed in `PROGRAM_LEAF_CRATES` below. A crate that
grows into a shared platform contract graduates out of that set deliberately,
in a reviewed PR.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Program leaf crates: workspace members that no other workspace package may
# depend on.
PROGRAM_LEAF_CRATES = {"hyprstream-synthesis"}

_DEP_TABLES = (
    "dependencies",
    "dev-dependencies",
    "build-dependencies",
)


def manifest_dependencies(manifest: dict, workspace_dependencies: dict):
    """Yield resolved (canonical name, spec, inherited) across every table."""
    tables = [manifest[t] for t in _DEP_TABLES if isinstance(manifest.get(t), dict)]
    for target in (manifest.get("target") or {}).values():
        if isinstance(target, dict):
            tables.extend(target[t] for t in _DEP_TABLES if isinstance(target.get(t), dict))
    for table in tables:
        for name, spec in table.items():
            inherited = isinstance(spec, dict) and spec.get("workspace") is True
            if inherited:
                if name not in workspace_dependencies:
                    raise ValueError(f"unresolved inherited workspace dependency {name!r}")
                spec = workspace_dependencies[name]
            package = spec.get("package", name) if isinstance(spec, dict) else name
            if not isinstance(package, str) or not package:
                raise ValueError(f"invalid package identity for dependency {name!r}")
            yield package, spec, inherited


def manifest_dependency_names(
    manifest_path: Path, workspace_dependencies: dict | None = None
) -> set[str]:
    """Canonical package identities, including workspace-inherited aliases."""
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)
    return {name for name, _, _ in manifest_dependencies(manifest, workspace_dependencies or {})}


def automatic_member(path: Path, root: Path, excluded: set[Path]) -> bool:
    """Path dependencies belong only to their nearest owning workspace.

    Explicit package.workspace overrides upward search. Nested workspace roots
    (including virtual roots between a package and this root) stop discovery.
    """
    if not path.is_relative_to(root) or path in excluded:
        return False
    with (path / "Cargo.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    owner = (manifest.get("package") or {}).get("workspace")
    if owner is not None:
        return (path / owner).resolve() == root
    current = path
    while current != root:
        candidate = current / "Cargo.toml"
        if candidate.is_file():
            with candidate.open("rb") as handle:
                if "workspace" in tomllib.load(handle):
                    return False
        current = current.parent
    return True


def check(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    try:
        with (root / "Cargo.toml").open("rb") as handle:
            manifest = tomllib.load(handle)
        workspace = manifest.get("workspace") or {}
        dependencies = workspace.get("dependencies") or {}
        if not isinstance(dependencies, dict):
            raise ValueError("workspace.dependencies must be a table")
        excluded = {path.resolve() for pattern in workspace.get("exclude", []) for path in root.glob(pattern)}
        manifests = set()
        if "package" in manifest:
            manifests.add(root / "Cargo.toml")
        for pattern in workspace.get("members", []):
            paths = list(root.glob(pattern))
            if not paths:
                raise ValueError(f"workspace member pattern {pattern!r} has no matches")
            for path in paths:
                if path.resolve() not in excluded:
                    manifests.add(path.resolve() / "Cargo.toml")
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return [f"cannot resolve workspace manifests: {error}"]
    # Cargo automatically includes in-tree path dependencies recursively,
    # even when workspace.members lists only the first package in the chain.
    pending = sorted(manifests)
    visited = set()
    while pending:
        manifest_path = pending.pop()
        if manifest_path in visited:
            continue
        visited.add(manifest_path)
        try:
            with manifest_path.open("rb") as handle:
                manifest = tomllib.load(handle)
            resolved = list(manifest_dependencies(manifest, dependencies))
            for _, spec, inherited in resolved:
                if isinstance(spec, dict) and "path" in spec:
                    base = root if inherited else manifest_path.parent
                    path = (base / spec["path"]).resolve()
                    if automatic_member(path, root, excluded):
                        pending.append(path / "Cargo.toml")
            name = (manifest.get("package") or {}).get("name")
            if not name or name in PROGRAM_LEAF_CRATES:
                continue
            for dependency in sorted({name for name, _, _ in resolved}):
                if dependency in PROGRAM_LEAF_CRATES:
                    rel = manifest_path.relative_to(root)
                    errors.append(
                        f"{rel}: platform crate `{name}` must not depend on "
                        f"program leaf crate `{dependency}` (program crates are "
                        f"applications, not platform dependencies)"
                    )
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            errors.append(f"cannot resolve {manifest_path}: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (defaults to this script's checkout)",
    )
    arguments = parser.parse_args()
    errors = check(arguments.root)
    for error in errors:
        print(f"program boundary: {error}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
