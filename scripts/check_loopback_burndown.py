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
**the count can only go down, and every shrink must be banked.**

METHODOLOGY (stated, mechanical, reproducible — the first #1152 W4 attempt
shipped a wrong count (211) because "non-test" was undefined; both reviewers
flagged it. This section is the contract.)
  SCOPE    : every `.rs` file under `crates/<crate>/src/` (each crate's direct
             source tree, *not* nested src dirs like `crates/foo/vendor/src`).
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
  `::1` OVER-MATCH (known, accepted — documented, not fixed): the bare `::1`
             alternation also counts non-loopback IPv6 literals whose final
             hextet is `1` (e.g. `fc00::1`, `fd00::1`, `fe80::1` private-range
             test vectors in `services/oauth/registration.rs`). These are not
             loopback. Tightening the pattern would change the frozen baseline
             and force a re-`--update`, which is churn disproportionate to the
             risk: the over-count inflates the baseline symmetrically, so it
             cannot weaken the monotone ratchet. The stated pattern therefore
             honestly counts "lines matching the loopback spelling OR an
             `::<hex>1` literal", not strictly "loopback literals." Called out
             here so the contract matches the code (fable review F3, #1152).

WHY A PER-FILE COUNT BASELINE (NOT A FILE ALLOWLIST)
----------------------------------------------------
A file allowlist alone is **not** monotone: once a file is allowed, CI permits
any number of new occurrences in that file (second reviewer, #1152). This gate
records a per-file count in the baseline and fails when any file's count
*increases* OR when a previously-clean file introduces its first loopback
literal. The total therefore can only stay flat or decrease. Files that reach
zero simply drop out of the baseline on the next `--update`.

WHY THE GATE MUST FAIL WHEN IT SCANS NOTHING (#1145)
----------------------------------------------------
A gate whose purpose is to prevent a count from growing must never pass
vacuously. If the scan scope vanishes — `crates/` renamed or moved, a
partial/sparse checkout, a typo'd glob — a naive "no growth, no new files"
check reports the empty scope as success and the burn-down dies silently. This
is the #1145 "absent expected value disables the check" pattern, found inside
two fixes written to eliminate it; this gate is the third site and gets the
structural treatment: **the gate hard-fails when it cannot establish that the
scan covered a plausible source tree.** Concretely it fails closed if
`crates/` is missing, if walking it yields zero `.rs` source files, on any
`OSError` reading a source file (a dropped file must not read as shrinkage),
and `--update` will not bank an empty result that would silently retire the
gate. See F1 in the #1199 review record.

WHY SHRINKS MUST BE BANKED (F2)
-------------------------------
The frozen baseline is a per-file ceiling: any improvement that lands without
a baseline `--update` is regressable back up to that ceiling. To make the
"can only go down" invariant exact rather than a ratchet-with-slack, the gate
also fails when any baselined file's count has *decreased* but the smaller
baseline has not been committed — the fix is to run `--update` in the same PR
and commit the shrunk baseline (fable review F2, #1152). The cost is an
occasional baseline-file merge conflict; the win is that a shrink, once
landed, cannot be silently rolled back.

USAGE
-----
  scripts/check_loopback_burndown.py              # enforce (CI mode); exit 1 on
                                                   # regression OR unbanked shrink
                                                   # OR implausible scan scope
  scripts/check_loopback_burndown.py --update     # rewrite the baseline to the
                                                   # current counts (maintainer;
                                                   # run after legitimately
                                                   # removing loopback literals,
                                                   # then commit the shrunk
                                                   # baseline)
  scripts/check_loopback_burndown.py --update --force
                                                   # bank a result the gate
                                                   # considers suspicious (e.g.
                                                   # an empty baseline marking
                                                   # the burn-down complete, or a
                                                   # >50% drop). Never banks a
                                                   # zero-scope scan.

Exit codes: 0 = clean (no growth, no unbanked shrink, plausible scope),
            1 = regression / unbanked shrink / implausible scope,
            2 = usage error.
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

# A shrink this large in one step is suspicious enough that --update should
# require --force: it is far more often a vanished scan scope (partial
# checkout, renamed crate root) than a legitimate cleanup. Legitimate mass
# cleanups can pass --force consciously. Kept generous so normal incremental
# wins never trip it.
DRAMATIC_SHRINK_FRACTION = 0.5


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)  # scripts/ -> repo root


def is_excluded(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(seg in ("tests", "examples", "benches") for seg in parts)


def scan(root: str) -> tuple[dict[str, int], int]:
    """Return ({relpath: matching_line_count}, total_src_rs_files_scanned).

    Fails closed (#1145) rather than returning a vacuous empty result:
      * `crates/` missing  -> the scan root vanished (rename/move/sparse
        checkout). A returning-empty here would read as "burn-down complete."
      * zero `.rs` files walked under any `crates/*/src/` -> scope vanished.
      * any `OSError` reading a source file -> a dropped file must not read as
        shrinkage; surface it instead of silently skipping.
    """
    crates = os.path.join(root, "crates")
    if not os.path.isdir(crates):
        raise SystemExit(
            f"loopback burn-down: {crates} is not a directory — scan scope "
            f"vanished (partial/sparse checkout, renamed crate root?). The "
            f"gate must never pass vacuously (#1145); failing closed."
        )
    totals: dict[str, int] = {}
    src_files = 0
    for dirpath, _dirs, files in os.walk(crates):
        # Only each crate's DIRECT source tree: crates/<crate>/src/... — a
        # bare `/src/` substring match would also enter nested dirs like
        # `crates/foo/vendor/src`, inventing baseline entries (CodeRabbit).
        rel = os.path.relpath(dirpath, crates).replace("\\", "/")
        parts = [] if rel == "." else rel.split("/")
        if not (len(parts) >= 2 and parts[1] == "src"):
            continue
        if is_excluded(dirpath):
            continue
        for f in files:
            if not f.endswith(".rs"):
                continue
            full = os.path.join(dirpath, f)
            if is_excluded(full):
                continue
            src_files += 1
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError as exc:
                raise SystemExit(
                    f"loopback burn-down: cannot read source file {full}: "
                    f"{exc}. A file dropping out of the count reads as "
                    f"shrinkage (#1145); failing closed instead of skipping."
                ) from exc
            n = sum(1 for ln in lines if PATTERN.search(ln))
            if n > 0:
                totals[os.path.relpath(full, root)] = n
    return dict(sorted(totals.items())), src_files


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
            "# CI fails if any count grows, a new file appears, OR a shrink\n"
            "# is not banked via --update. Files that reach zero drop out on\n"
            "# the next --update. Do NOT hand-edit counts upward — the diff\n"
            "# is review-visible and that is the only defense.\n"
        )
        for relpath, n in counts.items():
            fh.write(f"{n}\t{relpath}\n")


def enforce(
    current: dict[str, int],
    baseline: dict[str, int],
    src_files: int,
) -> int:
    """F1/F2 policy: fail on growth, new files, OR unbanked shrink."""
    grew: list[tuple[str, int, int]] = []
    new_files: list[tuple[str, int]] = []
    shrank: list[tuple[str, int, int]] = []  # (relpath, baseline, current) incl. 0/gone
    for relpath, n in current.items():
        if relpath not in baseline:
            new_files.append((relpath, n))
        elif n > baseline[relpath]:
            grew.append((relpath, baseline[relpath], n))
        elif n < baseline[relpath]:
            shrank.append((relpath, baseline[relpath], n))
    # A baselined file absent from current was emptied or deleted — an
    # unbanked win (F2). The F1 scope-loss case is screened first in main()
    # via the src_files plausibility check, so reaching here with a missing
    # baselined file means it genuinely went away in this tree.
    for relpath, old in baseline.items():
        if relpath not in current:
            shrank.append((relpath, old, 0))

    cur_total = sum(current.values())
    base_total = sum(baseline.values())

    print(
        f"loopback burn-down: current {cur_total} matching lines across "
        f"{len(current)} files ({src_files} source files scanned; baseline "
        f"{base_total} / {len(baseline)} files)"
    )

    # Scope-loss hint: a large fraction of baselined files vanishing at once is
    # far more likely a partial/renamed checkout than a cleanup. Surface it
    # loudly so the maintainer reaches for `git status`, not `--force`.
    gone = [rp for rp, _old, new in shrank if new == 0]
    if baseline and len(gone) > len(baseline) * DRAMATIC_SHRINK_FRACTION:
        print(
            f"\nWARNING: {len(gone)} of {len(baseline)} baselined files are "
            f"missing from the scan — this looks like a vanished scope "
            f"(partial/sparse checkout, renamed crate root?), not a cleanup. "
            f"Verify the working tree before banking anything (#1145)."
        )

    failed = False
    if new_files:
        failed = True
        print(
            "\nNEW files introducing loopback literals (must be removed, or "
            "the literals justified and the file added to the baseline via "
            "`scripts/check_loopback_burndown.py --update`):"
        )
        for relpath, n in sorted(new_files):
            print(f"  + {n}\t{relpath}")
    if grew:
        failed = True
        print(
            "\nEXISTING files whose loopback count GREW (the burn-down is "
            "monotone-decreasing; reduce back to/below the baseline):"
        )
        for relpath, old, new in sorted(grew):
            print(f"  ~ {old} -> {new}\t{relpath}")
    if shrank:
        failed = True
        print(
            "\nSHRINK not banked — run "
            "`scripts/check_loopback_burndown.py --update` in this PR and "
            "commit the smaller baseline, or this win can be silently "
            "regressed later (F2):"
        )
        for relpath, old, new in sorted(shrank):
            tag = "(gone)" if new == 0 else f"{new}"
            print(f"  ~ {old} -> {tag}\t{relpath}")

    if failed:
        print("\nSee scripts/check_loopback_burndown.py for the methodology. #1152 W4.")
        return 1
    return 0


def update(
    baseline_path: str,
    current: dict[str, int],
    src_files: int,
    force: bool,
) -> int:
    """Bank the current counts. Refuses suspicious results (#1145)."""
    previous = read_baseline(baseline_path)
    prev_total = sum(previous.values())
    cur_total = sum(current.values())

    # F1: a zero-scope scan is never bankable, even with --force — there is
    # nothing to bank and the gate would retire itself.
    if src_files == 0:
        raise SystemExit(
            "loopback burn-down: --update refused: the scan walked 0 source "
            "files. The scope has vanished (#1145); fix the working tree "
            "before banking anything."
        )

    # F1: an empty baseline retiring a previously-non-empty gate is the
    # footgun — it is indistinguishable from scope loss at write time. Require
    # --force so the maintainer consciously confirms "the burn-down is done."
    if not current and previous:
        if not force:
            raise SystemExit(
                f"loopback burn-down: --update would write an EMPTY baseline "
                f"(current 0 matching lines) over a non-empty one "
                f"({prev_total}). That retires the gate. If the burn-down is "
                f"genuinely complete, re-run with --force to confirm (#1145)."
            )
        print(
            "loopback burn-down: --force banking an empty baseline — the "
            "burn-down is being retired. Re-introducing any loopback literal "
            "will fail CI from a clean slate."
        )

    # F1: a dramatic drop is much more often a vanished/partial scope than a
    # legitimate cleanup. Require --force; legitimate mass cleanups can pass it.
    if previous and cur_total < prev_total * DRAMATIC_SHRINK_FRACTION:
        if not force:
            raise SystemExit(
                f"loopback burn-down: --update refused: new total {cur_total} "
                f"is < {int(DRAMATIC_SHRINK_FRACTION * 100)}% of the previous "
                f"{prev_total} — this usually means the scan scope shrank, not "
                f"that the codebase was cleaned up (#1145). Verify, then "
                f"re-run with --force to bank."
            )
        print(
            f"loopback burn-down: --force banking a dramatic drop "
            f"({prev_total} -> {cur_total})."
        )

    write_baseline(baseline_path, current)
    print(
        f"baseline written: {cur_total} matching lines across {len(current)} "
        f"files ({src_files} source files scanned) -> {baseline_path}"
    )
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline to the current counts (maintainer; commit the result)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="with --update: bank a result the gate considers suspicious "
        "(empty baseline, or a >50%% drop). Never banks a zero-scope scan.",
    )
    args = ap.parse_args(argv)

    root = repo_root()
    baseline_path = os.path.join(os.path.dirname(__file__), BASELINE_NAME)
    current, src_files = scan(root)  # F1: fails closed on scope-loss / OSError

    if args.update:
        return update(baseline_path, current, src_files, args.force)

    # F1: refuse to pass vacuously. crates/ existing but yielding zero source
    # files is a vanished scope — enforce() would otherwise celebrate "no
    # growth, no new files" as success.
    if src_files == 0:
        raise SystemExit(
            "loopback burn-down: scan walked 0 .rs source files under "
            f"{os.path.join(root, 'crates')} — the scope has vanished "
            "(partial/sparse checkout, renamed crate root?). The gate must "
            "never pass vacuously (#1145); failing closed."
        )

    baseline = read_baseline(baseline_path)
    return enforce(current, baseline, src_files)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
