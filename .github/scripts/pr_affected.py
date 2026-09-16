#!/usr/bin/env python3
"""Affected-set PR check: map changed files to workspace crates, run
nextest + clippy on just that set. Full default-members fallback when
root-level inputs change; no-op for pure-docs PRs.

Runs inside the pinned builder container. Env: BASE_SHA, HEAD_SHA.
House rules apply: cargo commands must be able to fail (no || true,
no continue-on-error).
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys

print = functools.partial(print, flush=True)  # noqa: A001 — CI logs stream line-by-line

# Paths whose change invalidates the whole default-members build.
FULL_SET_TRIGGERS = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".config/",
    "workspace-hack/",
    ".github/",
    "Dockerfile",
    "appimage/",
)

CRATES_DIR = "crates/"

# Workspace inputs that live outside crates/: prefixes whose change must be
# mapped to the crates that embed them (include_str!/include_bytes! pull these
# into binaries and tests, so a JSON/markdown edit outside crates/ can still
# break a build). Found by grepping crates/ for includes that escape the crate.
EXTERNAL_INPUT_CRATES: dict[str, tuple[str, ...]] = {
    "lexicons/": ("hyprstream", "hyprstream-pds"),
    "docs/standards/v16/vectors/": ("hyprstream-rpc",),
    "docs/discovery-state.md": ("hyprstream",),
}

# Packages whose tests read baked guest-WASM artifacts through env vars. The
# two guest builds cost real compile time, so they run only when one of these
# packages is in the affected set or on the full-set fallback; once artifacts
# exist the tests' deny-on-missing CI guard is armed via CI=true, matching the
# merge gate (graviton-build-test.sh builds the same two guests unconditionally).
GUEST_ARTIFACT_PACKAGES = ("hyprstream-workers-python", "hyprstream-workers-wasmtime")
GUEST_BUILDS = (
    ("crates/hyprstream-workers-python-guest", "wasm32-unknown-unknown",
     "hyprstream_workers_python_guest.wasm", "HYPRSTREAM_PYGUEST_WASM"),
    ("crates/hyprstream-workers-wasmtime-fsguest", "wasm32-wasip1",
     "hyprstream-workers-wasmtime-fsguest.wasm", "HYPRSTREAM_FSGUEST_WASM"),
)


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def ensure_commit(sha: str) -> None:
    if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"]).returncode != 0:
        subprocess.run(["git", "fetch", "--no-tags", "origin", sha], check=True)


def run(cmd: list[str], *, cwd: str | None = None) -> int:
    print(f"+ {' '.join(cmd)}" + (f"  (in {cwd})" if cwd else ""))
    return subprocess.run(cmd, cwd=cwd).returncode


def changed_files(base: str, head: str) -> list[str]:
    ensure_commit(base)
    ensure_commit(head)
    merge_base = git("merge-base", base, head)
    # --no-renames: with rename detection a move out of crates/foo/ lists only
    # the destination, so the source crate would never be tested even though
    # deleting the file can break it. Splitting renames surfaces both sides.
    out = git("diff", "--no-renames", "--name-only", f"{merge_base}..{head}")
    return [line for line in out.splitlines() if line]


def workspace_members() -> dict[str, str]:
    """Map crate directory name -> package name for current members."""
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"cargo metadata failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    meta = json.loads(result.stdout)
    return {
        pkg["manifest_path"].split("/crates/")[-1].split("/")[0]: pkg["name"]
        for pkg in meta["packages"]
        if "/crates/" in pkg["manifest_path"] and pkg["manifest_path"].startswith(meta["workspace_root"])
    }


def seed_packages(files: list[str], members: dict[str, str]) -> set[str]:
    """Touched workspace packages: crates/<dir>/ changes plus the packages
    embedding inputs that live outside crates/ (include_str!/include_bytes!)."""
    touched_dirs = {
        f.split("/")[1]
        for f in files
        if f.startswith(CRATES_DIR) and len(f.split("/")) > 2
    }
    names = set(members.values())
    seeds = {members[d] for d in touched_dirs if d in members}
    seeds |= {
        pkg
        for f in files
        for prefix, pkgs in EXTERNAL_INPUT_CRATES.items()
        if f.startswith(prefix)
        for pkg in pkgs
        if pkg in names
    }
    return seeds


def prepare_guest_artifacts() -> int:
    """Build the two standalone guest-WASM workspaces and arm the tests'
    deny-on-missing guards. Mirrors graviton-build-test.sh: each guest is a
    standalone workspace, so cargo must run FROM its directory to pick up the
    guest-local .cargo/config.toml (the python guest pins getrandom's custom
    backend, #1013), and artifact paths resolve through the same (absolute)
    CARGO_TARGET_DIR the outer build uses."""
    target_dir = os.environ.get("CARGO_TARGET_DIR", "target")
    for crate_dir, wasm_target, artifact, env_var in GUEST_BUILDS:
        rc = run(["cargo", "build", "--release", "--target", wasm_target], cwd=crate_dir)
        if rc != 0:
            print(f"guest WASM build failed for {crate_dir} with exit {rc}", file=sys.stderr)
            return rc
        root = target_dir if os.path.isabs(target_dir) else os.path.join(crate_dir, target_dir)
        path = os.path.abspath(os.path.join(root, wasm_target, "release", artifact))
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            print(f"guest artifact is missing, unreadable, or empty: {path}", file=sys.stderr)
            return 1
        os.environ[env_var] = path
    os.environ["CI"] = "true"
    return 0


def main() -> int:
    base = os.environ.get("BASE_SHA")
    head = os.environ.get("HEAD_SHA")
    if not base or not head:
        print("BASE_SHA and HEAD_SHA are required", file=sys.stderr)
        return 1

    files = changed_files(base, head)
    full = any(f.startswith(trigger) or f == trigger for f in files for trigger in FULL_SET_TRIGGERS)
    members = workspace_members()
    seeds = seed_packages(files, members)

    if full:
        print("root-level input changed; running the full default-members set")
        pkgs: list[str] = []
    elif seeds:
        pkgs = sorted(seeds)
        print(f"affected crates: {' '.join(pkgs)}")
    else:
        ignored = [f for f in files if not f.startswith((".github/", CRATES_DIR))]
        print(f"no workspace crates affected ({len(ignored)} non-crate files changed); nothing to run")
        return 0

    if full or any(p in GUEST_ARTIFACT_PACKAGES for p in pkgs):
        print("::group::guest WASM builds (workers crates in the affected set)")
        rc = prepare_guest_artifacts()
        print("::endgroup::")
        if rc != 0:
            return rc

    args: list[str] = []
    for p in pkgs:
        args += ["-p", p]

    print(f"::group::cargo nextest run {' '.join(args)}")
    # Mirror the merge gate's invocation exactly (--cargo-profile ci-test,
    # --profile ci: the ci profile carries the flake quarantine, slow-timeout
    # caps, and test-group scheduling; ci-test compiles far lighter than dev).
    # --no-fail-fast: one load-sensitive deadline test must not cancel the
    # remaining thousands of results; the full signal is the point here.
    rc = subprocess.run(
        ["cargo", "nextest", "run", "--cargo-profile", "ci-test", "--profile", "ci",
         "--no-fail-fast", *args]
    ).returncode
    print("::endgroup::")
    if rc != 0:
        print(f"cargo nextest run failed with exit {rc}", file=sys.stderr)
        return rc

    print(f"::group::cargo clippy {' '.join(args)} --all-targets")
    rc = subprocess.run(
        ["cargo", "clippy", *args, "--all-targets", "--", "-D", "warnings"]
    ).returncode
    print("::endgroup::")
    if rc != 0:
        print(f"cargo clippy failed with exit {rc}", file=sys.stderr)
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
