#!/usr/bin/env python3
r"""Loopback-literal burn-down gate for #1152 (W4).

WHY THIS EXISTS
---------------
Hardcoded `127.0.0.1` / `localhost` / `::1` literals are the residue of the
single-host era. Multi-host operation is now mandatory (#1135), and loopback
literals in production code silently break remote reach (an issuer/audience
rewritten to `localhost` is unreachable from another host). We are NOT fixing
the existing occurrences in one pass — there are too many, and many require
design decisions. Instead we freeze the surface and make it shrink over time:
**the count can only go down.**

METHODOLOGY (stated, mechanical, reproducible — the first #1152 W4 attempt
shipped a wrong count (211) because "non-test" was undefined; both reviewers
flagged it. This section is the contract.)
  SCOPE    : every `.rs` file under `crates/*/src/` (production source trees).
  EXCLUDED : files whose path contains a `/tests/`, `/examples/`, or
             `/benches/` segment. We do **not** attempt to exclude
             `#[cfg(test)]` blocks *inside* src files — that needs a real
             parser and is not a quick CI grep (fable review, #1152). This is
             a **file-level** lint, so cfg(test) mentions count toward a file's
             total. The honest denominator is therefore "loopback mentions in
             production source files", not "loopback mentions in non-test
             code". That keeps the gate a plain text scan anyone can reproduce.
  COUNTS   : one (1) per **line** matching the pattern, however many matches
             are on that line. Line-based (not occurrence-based) keeps the
             burn-down monotone under line reflow — splitting one match across
             two lines cannot inflate the count.
  PATTERN  : `127\.0\.0\.1` | `localhost` (lowercase) | `::1`
             | `Ipv4Addr::LOCALHOST` | `Ipv6Addr::LOCALHOST`.
             The symbolic `*Addr::LOCALHOST` spellings are included because a
             bare string scan misses them (second reviewer, #1152).

WHY A PER-FILE COUNT BASELINE (NOT A FILE ALLOWLIST)
----------------------------------------------------
A file allowlist alone is **not** monotone: once a file is allowed, CI permits
any number of new occurrences in that file (second reviewer, #1152). This gate
records a per-file count in the baseline and fails when any file's count
*increases* OR when a previously-clean file introduces its first loopback
literal. The total therefore can only stay flat or decrease. Files that reach
zero simply drop out of the baseline on the next `--update`.

USAGE
-----
  scripts/check_loopback_burndown.py            # enforce (CI mode); exit 1 on regression
  scripts/check_loopback_burndown.py --update   # rewrite the baseline to the
                                                # current counts (maintainer only;
                                                # run after legitimately removing
                                                # loopback literals, then commit
                                                # the shrunk baseline)

Exit codes: 0 = clean (no growth), 1 = regression detected, 2 = usage error.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

PATTERN = re.compile(
    r"127\.0\.0\.1"
    r"|localhost"
    r"|::1"
    r"|Ipv4Addr::LOCALHOST"
    r"|Ipv6Addr::LOCALHOST"
)

BASELINE_NAME = "loopback-baseline.txt"


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)  # scripts/ -> repo root


def is_excluded(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(seg in ("tests", "examples", "benches") for seg in parts)


def scan(root: str) -> dict[str, int]:
    """Return {relpath: matching_line_count} for every src .rs file with > 0."""
    crates = os.path.join(root, "crates")
    totals: dict[str, int] = {}
    for dirpath, _dirs, files in os.walk(crates):
        norm = dirpath.replace("\\", "/")
        # Only walk source trees: path must contain a /src/ segment.
        if "/src/" not in norm + "/":
            continue
        if is_excluded(dirpath):
            continue
        for f in files:
            if not f.endswith(".rs"):
                continue
            full = os.path.join(dirpath, f)
            if is_excluded(full):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            n = sum(1 for ln in lines if PATTERN.search(ln))
            if n > 0:
                totals[os.path.relpath(full, root)] = n
    return dict(sorted(totals.items()))


def read_baseline(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                count_str, relpath = line.split("\t", 1)
            except ValueError:
                # Also tolerate "<count> <path>" (single space) for hand edits.
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise SystemExit(f"{path}: unparseable baseline line: {raw!r}") from None
                count_str, relpath = parts
            out[relpath.strip()] = int(count_str)
    return dict(sorted(out.items()))


def write_baseline(path: str, counts: dict[str, int]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "# Loopback-literal burn-down baseline for #1152 (W4).\n"
            "# Generated by `scripts/check_loopback_burndown.py --update`.\n"
            "# Each line: <matching-line-count>\\t<relpath>\n"
            "# CI fails if any count grows or a new file appears. Files that\n"
            "# reach zero drop out on the next --update.\n"
        )
        for relpath, n in counts.items():
            fh.write(f"{n}\t{relpath}\n")


def enforce(current: dict[str, int], baseline: dict[str, int]) -> int:
    grew: list[tuple[str, int, int]] = []
    new_files: list[tuple[str, int]] = []
    for relpath, n in current.items():
        if relpath not in baseline:
            new_files.append((relpath, n))
        elif n > baseline[relpath]:
            grew.append((relpath, baseline[relpath], n))

    cur_total = sum(current.values())
    base_total = sum(baseline.values())

    print(f"loopback burn-down: current {cur_total} matching lines across "
          f"{len(current)} files (baseline {base_total} / {len(baseline)} files)")

    if not grew and not new_files:
        if cur_total < base_total:
            print(f"  ↓ shrank by {base_total - cur_total} — re-run "
                  f"`scripts/check_loopback_burndown.py --update` and commit "
                  f"the smaller baseline to bank the win")
        return 0

    if new_files:
        print("\nNEW files introducing loopback literals (must be removed, or "
              "the literals justified and the file added to the baseline via "
              "`scripts/check_loopback_burndown.py --update`):")
        for relpath, n in sorted(new_files):
            print(f"  + {n}\t{relpath}")
    if grew:
        print("\nEXISTING files whose loopback count GREW (the burn-down is "
              "monotone-decreasing; reduce back to/below the baseline):")
        for relpath, old, new in sorted(grew):
            print(f"  ~ {old} -> {new}\t{relpath}")
    print("\nSee scripts/check_loopback_burndown.py for the methodology. #1152 W4.")
    return 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline to the current counts (maintainer; commit the result)",
    )
    args = ap.parse_args(argv)

    root = repo_root()
    baseline_path = os.path.join(os.path.dirname(__file__), BASELINE_NAME)
    current = scan(root)

    if args.update:
        write_baseline(baseline_path, current)
        print(f"baseline written: {sum(current.values())} matching lines across "
              f"{len(current)} files -> {baseline_path}")
        return 0

    baseline = read_baseline(baseline_path)
    return enforce(current, baseline)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
