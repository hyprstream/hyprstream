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


def manifest_dependency_names(manifest_path: Path) -> set[str]:
    """Every dependency name declared by a package manifest, across the
    normal, dev, build, and per-target dependency tables."""
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)

    names: set[str] = set()
    tables: list[dict] = []
    for table in _DEP_TABLES:
        if isinstance(manifest.get(table), dict):
            tables.append(manifest[table])
    for target in (manifest.get("target") or {}).values():
        if not isinstance(target, dict):
            continue
        for table in _DEP_TABLES:
            if isinstance(target.get(table), dict):
                tables.append(target[table])
    for table in tables:
        for name, spec in table.items():
            names.add(name)
            if isinstance(spec, dict) and "package" in spec:
                names.add(spec["package"])
    return names


def check(root: Path) -> list[str]:
    errors: list[str] = []
    for manifest_path in sorted((root / "crates").glob("*/Cargo.toml")):
        with manifest_path.open("rb") as handle:
            manifest = tomllib.load(handle)
        name = (manifest.get("package") or {}).get("name")
        if not name or name in PROGRAM_LEAF_CRATES:
            continue
        for dependency in sorted(manifest_dependency_names(manifest_path)):
            if dependency in PROGRAM_LEAF_CRATES:
                rel = manifest_path.relative_to(root)
                errors.append(
                    f"{rel}: platform crate `{name}` must not depend on "
                    f"program leaf crate `{dependency}` (program crates are "
                    f"applications, not platform dependencies)"
                )
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
