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


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def ensure_commit(sha: str) -> None:
    if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"]).returncode != 0:
        subprocess.run(["git", "fetch", "--no-tags", "origin", sha], check=True)


def changed_files(base: str, head: str) -> list[str]:
    ensure_commit(base)
    ensure_commit(head)
    merge_base = git("merge-base", base, head)
    out = git("diff", "--name-only", f"{merge_base}..{head}")
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


def main() -> int:
    base = os.environ.get("BASE_SHA")
    head = os.environ.get("HEAD_SHA")
    if not base or not head:
        print("BASE_SHA and HEAD_SHA are required", file=sys.stderr)
        return 1

    files = changed_files(base, head)
    full = any(f.startswith(trigger) or f == trigger for f in files for trigger in FULL_SET_TRIGGERS)
    touched_dirs = {
        f.split("/")[1]
        for f in files
        if f.startswith(CRATES_DIR) and len(f.split("/")) > 2
    }

    members = workspace_members()
    affected = sorted(members.get(d) for d in touched_dirs if d in members)

    if full:
        print("root-level input changed; running the full default-members set")
        pkgs: list[str] = []
    elif affected:
        print(f"affected crates: {' '.join(affected)}")
        pkgs = [p for p in affected if p]
    else:
        ignored = [f for f in files if not f.startswith((".github/", CRATES_DIR))]
        print(f"no workspace crates affected ({len(ignored)} non-crate files changed); nothing to run")
        return 0

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
