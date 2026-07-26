#!/usr/bin/env python3
r"""Static regression for the Rust workflow trigger/job matrix (#1331).

WHY THIS EXISTS
---------------
The `Rust` workflow used to fire on every push to `main`, which (after a
merge-queue landing) duplicated the heavy self-hosted build and re-ran the
fast PR checks for an already-tested commit. #1331 removed the `push` trigger
and replaced it with `workflow_dispatch` for operator-requested full
validation of an arbitrary ref.

This gate freezes that decision so a future edit cannot silently reintroduce
the post-merge duplicate run. It also pins the corrected AppImage-trigger
comment: `appimage.yml` runs on schedule / tags / `workflow_dispatch` only —
never on a push to a branch.

METHODOLOGY
-----------
Pure Python-stdlib inspection — no third-party YAML parser, no network, no
runner, no build. A focused line-based parser extracts the top-level `on:`
triggers (and their nested children) and each job's `if:` expression; the
assertions then lock the matrix to the shape #1331 introduced. Failures exit
non-zero so the CI step can fail. Do NOT add `|| true`.

NON-VACUOUS
-----------
`--self-test` (run automatically by `main()` when the script is invoked with
no arguments after the live check) feeds known-bad mutations of the live
workflow through the same check function and asserts each one is rejected.
If a mutation slips through, the gate fails — proving the assertions fire.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUST_WF = ROOT / ".github" / "workflows" / "rust.yml"
APPIMAGE_WF = ROOT / ".github" / "workflows" / "appimage.yml"


# --------------------------------------------------------------------------
# Minimal GitHub-Actions-YAML structural parser (stdlib only).
#
# We do NOT attempt a general YAML parse. We extract exactly two things the
# gate needs: (1) the top-level `on:` trigger block as an ordered dict of
# {trigger_name: nested_subtext}, and (2) each job's `if:` expression under
# `jobs:`. Both are produced by reading indentation, which is sufficient for
# the constrained shapes GitHub Actions accepts for these fields.
# --------------------------------------------------------------------------


def _strip_key(token: str) -> str:
    """Strip surrounding YAML quotes and trailing comment from a key token."""
    token = token.split("#", 1)[0].strip()
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        token = token[1:-1]
    return token


def _on_blocks(text: str) -> dict[str, str]:
    """Return {trigger_name: nested_subtext} for each top-level trigger.

    `nested_subtext` is the raw block of lines indented under that trigger
    (empty string if the trigger has no children, e.g. `pull_request:` with
    nothing nested). Returns {} if the file has no top-level `on:`.

    Handles the two forms used in this repo:
      block:      on:\n  pull_request:\n  merge_group:\n
      inline/map: on: push               (single trigger, no nesting)
    """
    lines = text.splitlines()
    on_idx = None
    for i, line in enumerate(lines):
        if not line or line[0] in " \t" or line.lstrip().startswith("#"):
            continue
        if _strip_key(line.split(":", 1)[0]) == "on":
            on_idx = i
            break
    if on_idx is None:
        return {}

    header = lines[on_idx]
    after_colon = header.split(":", 1)[1] if ":" in header else ""
    inline = after_colon.split("#", 1)[0].strip()
    if inline:
        # inline form: `on: push` or `on: [push, pull_request]`
        if inline.startswith("[") and inline.endswith("]"):
            inner = inline[1:-1]
            return {_strip_key(t): "" for t in inner.split(",") if t.strip()}
        return {_strip_key(inline): ""}

    # block form: find the indent of the first non-blank/comment line after on:
    block_indent: int | None = None
    for line in lines[on_idx + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        block_indent = len(line) - len(line.lstrip())
        break
    if block_indent is None:
        return {}

    blocks: dict[str, str] = {}
    cur_name: str | None = None
    buf: list[str] = []
    for line in lines[on_idx + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            if cur_name is not None:
                buf.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent < block_indent:
            break  # dedent out of on:
        if indent == block_indent:
            if cur_name is not None:
                blocks[cur_name] = "\n".join(buf)
            stripped = line.strip()
            if stripped.startswith("- "):
                cur_name = _strip_key(stripped[2:])
            elif ":" in stripped:
                cur_name = _strip_key(stripped.split(":", 1)[0])
            else:
                cur_name = _strip_key(stripped)
            buf = []
        else:
            if cur_name is not None:
                buf.append(line)
    if cur_name is not None:
        blocks[cur_name] = "\n".join(buf)
    return blocks


def _jobs_if_map(text: str) -> dict[str, str | None]:
    """Return {job_name: if_expr_or_None} for top-level entries under `jobs:`.

    `if_expr_or_None` is the literal text after `if:` (comment stripped) for
    the first `if:` line found directly inside that job, or None if the job
    has no `if:`. Only direct children of `jobs:` are returned (2-space indent
    in this repo's files); nested `if:` lines deeper than the job body are
    ignored once the first `if:` has been recorded.
    """
    lines = text.splitlines()
    in_jobs = False
    jobs: dict[str, str | None] = {}
    cur_job: str | None = None
    cur_job_indent: int | None = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            cur_job = None
            cur_job_indent = None
            in_jobs = _strip_key(line.split(":", 1)[0]) == "jobs"
            continue
        if not in_jobs:
            continue
        if indent == 2 and line.rstrip(" \t").endswith(":"):
            cur_job = _strip_key(line.strip()[:-1])
            cur_job_indent = indent
            jobs[cur_job] = None
            continue
        if cur_job is None or cur_job_indent is None:
            continue
        if jobs.get(cur_job) is None and indent > cur_job_indent:
            m = re.match(r"\s*if:\s*(.+?)\s*(?:#.*)?$", line)
            if m:
                jobs[cur_job] = m.group(1).strip()
    return jobs


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_rust_text(text: str) -> None:
    triggers = list(_on_blocks(text).keys())
    for required in ("pull_request", "merge_group", "workflow_dispatch"):
        _assert(
            required in triggers,
            f"rust.yml: missing required trigger {required!r}; got {triggers}",
        )
    _assert(
        "push" not in triggers,
        "rust.yml: 'push' trigger must not be present (reintroduces the #1331 "
        f"duplicate post-merge run); got {triggers}",
    )

    jobs = _jobs_if_map(text)
    for name in ("clippy", "deny", "loopback-burndown", "wasm", "build"):
        _assert(name in jobs, f"rust.yml: required job {name!r} missing; got {sorted(jobs)}")

    for name in ("clippy", "wasm", "loopback-burndown"):
        cond = jobs.get(name)
        _assert(
            cond is not None and "github.event_name != 'merge_group'" in cond,
            f"rust.yml: job {name!r} must skip on merge_group (if={cond!r})",
        )

    build_if = jobs.get("build")
    _assert(
        build_if is not None and build_if.strip() == "github.event_name != 'pull_request'",
        f"rust.yml: 'build' if must be the single 'pull_request' skip (if={build_if!r})",
    )


def check_appimage_text(text: str) -> None:
    """Pin the corrected rust.yml comment: AppImage is nightly/tag/manual only."""
    blocks = _on_blocks(text)
    push = blocks.get("push", "")
    _assert(
        "branches" not in push,
        "appimage.yml: a 'push.branches' trigger is forbidden (the rust.yml "
        f"comment claims AppImage is nightly/tag/manual only); push block was:\n{push}",
    )


# --------------------------------------------------------------------------
# Negative mutation fixtures (keep the gate non-vacuous).
# --------------------------------------------------------------------------


def _mutations(rust_text: str) -> list[tuple[str, str]]:
    """Return (label, mutated_text) pairs that MUST each be rejected."""
    return [
        (
            "re-add push-to-main trigger",
            rust_text.replace(
                "on:\n  pull_request:",
                "on:\n  push:\n    branches: [ \"main\" ]\n  pull_request:",
                1,
            ),
        ),
        (
            "drop workflow_dispatch",
            rust_text.replace("  workflow_dispatch:\n", "", 1),
        ),
        (
            "make build fire on PR",
            rust_text.replace(
                "github.event_name != 'pull_request'",
                "github.event_name == 'pull_request'",
                1,
            ),
        ),
        (
            "make clippy run on merge_group",
            rust_text.replace(
                "    if: github.event_name != 'merge_group'\n    runs-on: [self-hosted, linux, arm64, graviton, hyprstream-merge-gate]\n    # The last successful run",
                "    if: github.event_name != 'workflow_dispatch'\n    runs-on: [self-hosted, linux, arm64, graviton, hyprstream-merge-gate]\n    # The last successful run",
                1,
            ),
        ),
        (
            "rename build job (removing the required gate)",
            rust_text.replace("  build:\n", "  build_renamed:\n", 1),
        ),
    ]


def self_test(rust_text: str) -> list[str]:
    """Run each negative mutation; return labels that WRONGLY passed check."""
    escaped: list[str] = []
    for label, mutated in _mutations(rust_text):
        if mutated == rust_text:
            escaped.append(f"{label} (mutation did not apply — fixture is stale)")
            continue
        try:
            check_rust_text(mutated)
        except AssertionError:
            continue  # correctly rejected
        escaped.append(label)
    return escaped


# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    rust_text = RUST_WF.read_text()
    appimage_text = APPIMAGE_WF.read_text()

    failures: list[str] = []
    for name, fn, arg in (
        ("check_rust_workflow", check_rust_text, rust_text),
        ("check_appimage_workflow", check_appimage_text, appimage_text),
    ):
        try:
            fn(arg)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")

    escaped = self_test(rust_text)
    for label in escaped:
        failures.append(f"self-test: mutation {label!r} was NOT rejected (gate is vacuous)")

    if failures:
        print("#1331 workflow-matrix regression FAILED:", file=sys.stderr)
        for fail in failures:
            print(f"  - {fail}", file=sys.stderr)
        return 1

    print("#1331 workflow-matrix regression: OK (incl. 5 negative mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
