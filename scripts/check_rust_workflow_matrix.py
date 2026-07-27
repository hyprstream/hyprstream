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
comment: `appimage.yml` runs on its one nightly schedule, `v*` tags, and manual
dispatch only — never on a push to a branch.

METHODOLOGY
-----------
Pure Python-stdlib inspection — no third-party YAML parser, no network, no
runner, no build. A focused fail-closed line parser extracts the top-level
`on:` triggers (including inline values), job blocks, and job-level `if:`
expressions. Assertions lock exact trigger sets, exact normalized conditions,
and the job identities that implement the #1331 event matrix. Unsupported YAML
shapes are rejected rather than guessed at. Failures exit non-zero so the CI
step can fail. Do NOT add `|| true`.

NON-VACUOUS
-----------
The self-test (run automatically after the live check) feeds known-bad
mutations of both workflows through the same check functions and asserts each
one is rejected. It includes semantic condition bypasses and alternate trigger
forms. If a mutation slips through, the gate fails — proving the assertions
fire.
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

    Handles the forms needed to fail closed around this repo:
      block:      on:\n  pull_request:\n  merge_group:\n
      inline/list: on: [pull_request, merge_group]

    A value on an individual trigger is retained in `nested_subtext`, so an
    inline map such as `push: {branches: [main]}` cannot disappear.
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
            inline_value = ""
            if stripped.startswith("- "):
                cur_name = _strip_key(stripped[2:])
            elif ":" in stripped:
                key, value = stripped.split(":", 1)
                cur_name = _strip_key(key)
                inline_value = value.split("#", 1)[0].strip()
            else:
                cur_name = _strip_key(stripped)
            buf = [inline_value] if inline_value else []
        else:
            if cur_name is not None:
                buf.append(line)
    if cur_name is not None:
        blocks[cur_name] = "\n".join(buf)
    return blocks


def _job_blocks(text: str) -> dict[str, str]:
    """Return raw text for each direct child of the top-level `jobs:` map."""
    lines = text.splitlines()
    in_jobs = False
    blocks: dict[str, list[str]] = {}
    cur_job: str | None = None
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#"):
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                cur_job = None
                in_jobs = _strip_key(line.split(":", 1)[0]) == "jobs"
                continue
            if in_jobs and indent == 2 and line.rstrip(" \t").endswith(":"):
                cur_job = _strip_key(line.strip()[:-1])
                blocks[cur_job] = []
                continue
        if in_jobs and cur_job is not None:
            blocks[cur_job].append(line)
    return {name: "\n".join(lines) for name, lines in blocks.items()}


def _jobs_if_map(text: str) -> dict[str, str | None]:
    """Return {job_name: if_expr_or_None} for top-level entries under `jobs:`.

    `if_expr_or_None` is the literal text after `if:` (comment stripped) for
    the first `if:` line found directly inside that job, or None if the job
    has no `if:`. Only direct children of `jobs:` are returned (2-space indent
    in this repo's files); nested `if:` lines deeper than the job body are
    ignored once the first `if:` has been recorded.
    """
    jobs: dict[str, str | None] = {}
    for name, block in _job_blocks(text).items():
        jobs[name] = None
        for line in block.splitlines():
            m = re.match(r" {4}if:\s*(.+?)\s*(?:#.*)?$", line)
            if m:
                jobs[name] = m.group(1).strip()
                break
    return jobs


def _normalized_condition(expr: str | None) -> str | None:
    """Normalize only harmless syntax around a GitHub Actions condition.

    Optional expression delimiters, one layer of YAML scalar quotes, quote
    style, and whitespace are normalized. Operators/parentheses/tokens are not
    rewritten: semantically broader or negated expressions therefore cannot
    retain an accepted condition as a substring and escape.
    """
    if expr is None:
        return None
    value = expr.strip()
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        value = value[1:-1].strip()
    if value.startswith("${{") and value.endswith("}}"):
        value = value[3:-2].strip()
    value = value.replace('"', "'")
    return re.sub(r"\s+", "", value)


def _content_lines(block: str) -> list[str]:
    """Return stripped non-comment lines from a constrained YAML sub-block."""
    return [
        line.split("#", 1)[0].strip()
        for line in block.splitlines()
        if line.split("#", 1)[0].strip()
    ]


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    return value


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_rust_text(text: str) -> None:
    trigger_blocks = _on_blocks(text)
    triggers = set(trigger_blocks)
    expected_triggers = {"pull_request", "merge_group", "workflow_dispatch"}
    _assert(
        triggers == expected_triggers,
        "rust.yml: triggers must be exactly pull_request, merge_group, and "
        f"workflow_dispatch; got {sorted(triggers)}",
    )
    for name, block in trigger_blocks.items():
        _assert(
            not _content_lines(block),
            f"rust.yml: trigger {name!r} must be unfiltered; got {block!r}",
        )

    jobs = _jobs_if_map(text)
    for name in ("clippy", "deny", "loopback-burndown", "wasm", "build"):
        _assert(name in jobs, f"rust.yml: required job {name!r} missing; got {sorted(jobs)}")

    skip_merge_group = "github.event_name!='merge_group'"
    for name in ("clippy", "wasm", "loopback-burndown"):
        cond = jobs.get(name)
        _assert(
            _normalized_condition(cond) == skip_merge_group,
            f"rust.yml: job {name!r} condition must be exactly the merge_group "
            f"skip (if={cond!r})",
        )

    _assert(
        jobs.get("deny") is None,
        f"rust.yml: 'deny' must run on every supported event (if={jobs.get('deny')!r})",
    )
    build_if = jobs.get("build")
    _assert(
        _normalized_condition(build_if) == "github.event_name!='pull_request'",
        f"rust.yml: 'build' condition must be exactly the pull_request skip "
        f"(if={build_if!r})",
    )


def check_appimage_text(text: str) -> None:
    """Pin AppImage to one nightly cron, v* tags, and manual dispatch only."""
    blocks = _on_blocks(text)
    expected_triggers = {"schedule", "workflow_dispatch", "push"}
    _assert(
        set(blocks) == expected_triggers,
        "appimage.yml: triggers must be exactly schedule, workflow_dispatch, "
        f"and push; got {sorted(blocks)}",
    )
    _assert(
        not _content_lines(blocks["workflow_dispatch"]),
        "appimage.yml: workflow_dispatch must be unfiltered",
    )

    schedule = _content_lines(blocks["schedule"])
    _assert(
        len(schedule) == 1 and schedule[0].startswith("- cron:"),
        f"appimage.yml: schedule must contain exactly one cron; got {schedule}",
    )
    cron = _unquote(schedule[0].split(":", 1)[1])
    _assert(
        cron == "30 3 * * *",
        f"appimage.yml: nightly cron must remain '30 3 * * *'; got {cron!r}",
    )

    push = _content_lines(blocks["push"])
    _assert(
        len(push) == 2 and push[0] == "tags:" and push[1].startswith("-"),
        "appimage.yml: push must use the fail-closed tags-only block form; "
        f"got {push}",
    )
    tag = _unquote(push[1][1:])
    _assert(
        tag == "v*",
        f"appimage.yml: push must be limited to the single 'v*' tag; got {tag!r}",
    )


# --------------------------------------------------------------------------
# Negative mutation fixtures (keep the gate non-vacuous).
# --------------------------------------------------------------------------


def _rust_mutations(rust_text: str) -> list[tuple[str, str]]:
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
            "add Rust schedule trigger",
            rust_text.replace(
                "  workflow_dispatch:\n",
                "  workflow_dispatch:\n  schedule:\n    - cron: '0 0 * * *'\n",
                1,
            ),
        ),
        (
            "drop workflow_dispatch trigger",
            rust_text.replace("  workflow_dispatch:\n", "", 1),
        ),
        (
            "broaden merge_group skip with always",
            rust_text.replace(
                "if: github.event_name != 'merge_group'",
                "if: github.event_name != 'merge_group' || always()",
                1,
            ),
        ),
        (
            "negate condition retaining expected substring",
            rust_text.replace(
                "if: github.event_name != 'merge_group'",
                "if: ${{ !(github.event_name != 'merge_group') }}",
                1,
            ),
        ),
        (
            "make clippy run on merge_group",
            rust_text.replace(
                "if: github.event_name != 'merge_group'",
                "if: github.event_name == 'merge_group'",
                1,
            ),
        ),
        (
            "make build run on pull_request",
            rust_text.replace(
                "if: github.event_name != 'pull_request'",
                "if: github.event_name == 'pull_request'",
                1,
            ),
        ),
        (
            "make deny skip pull_request",
            rust_text.replace(
                "  deny:\n    name: cargo-deny (bans + licenses)\n",
                "  deny:\n"
                "    name: cargo-deny (bans + licenses)\n"
                "    if: github.event_name != 'pull_request'\n",
                1,
            ),
        ),
        (
            "rename build job (removing the required gate)",
            rust_text.replace("  build:\n", "  build_renamed:\n", 1),
        ),
    ]


def _appimage_mutations(appimage_text: str) -> list[tuple[str, str]]:
    """Return AppImage trigger mutations that MUST each be rejected."""
    return [
        (
            "add AppImage pull_request trigger",
            appimage_text.replace("on:\n", "on:\n  pull_request:\n", 1),
        ),
        (
            "inline AppImage push branches map",
            appimage_text.replace(
                "  push:\n    tags:\n      - 'v*'\n",
                "  push: {branches: [main], tags: ['v*']}\n",
                1,
            ),
        ),
    ]


def self_test(rust_text: str, appimage_text: str) -> list[str]:
    """Run each negative mutation; return labels that WRONGLY passed check."""
    escaped: list[str] = []
    cases = [
        ("rust", check_rust_text, rust_text, _rust_mutations(rust_text)),
        (
            "appimage",
            check_appimage_text,
            appimage_text,
            _appimage_mutations(appimage_text),
        ),
    ]
    for source, check, original, mutations in cases:
        for label, mutated in mutations:
            if mutated == original:
                escaped.append(
                    f"{source}: {label} (mutation did not apply — fixture is stale)"
                )
                continue
            try:
                check(mutated)
            except AssertionError:
                continue  # correctly rejected
            escaped.append(f"{source}: {label}")
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

    escaped = self_test(rust_text, appimage_text)
    for label in escaped:
        failures.append(f"self-test: mutation {label!r} was NOT rejected (gate is vacuous)")

    if failures:
        print("#1331 workflow-matrix regression FAILED:", file=sys.stderr)
        for fail in failures:
            print(f"  - {fail}", file=sys.stderr)
        return 1

    mutation_count = len(_rust_mutations(rust_text)) + len(
        _appimage_mutations(appimage_text)
    )
    print(
        f"#1331 workflow-matrix regression: OK "
        f"(incl. {mutation_count} negative mutations)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
