#!/usr/bin/env python3
"""Gate every workspace-hack dependency edge off the wasm32 target.

cargo-hakari's manage-deps emits `workspace-hack = { version = "0.1",
path = "..." }` into each member's [dependencies], which applies on every
target. The hack's dependency union is native-only (tokio/net -> mio, which
compile_error!s on wasm32; axum; ...), so a wasm build of any hack consumer
would compile mio for wasm32 and fail. The workspace's existing convention
for this class of problem is target-gating the dependency edge (see
crates/hyprstream-9p: "net pulls mio, which has no wasm32 backend").

This script rewrites each member manifest so the workspace-hack edge lives
in the `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` table:
inserted into the table when the manifest already has one, appended as a
new table otherwise. An empty `[dependencies]` header left behind is
harmless TOML and left in place.

`cargo hakari manage-deps` re-emits the ungated form whenever it runs (and
its final lockfile regeneration cannot parse partially-gated manifests), so
the CI gate sequence is:

    python3 scripts/hack_wasm_gate.py --ungate
    cargo hakari generate
    cargo hakari manage-deps --yes
    python3 scripts/hack_wasm_gate.py
    git diff --exit-code  (workspace-hack + member manifests)

i.e. regeneration equality is checked AFTER the wasm gate is re-applied.
Both modes are idempotent. Run with --check to fail (exit 1) without
writing if any manifest would change.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATED = "[target.'cfg(not(target_arch = \"wasm32\"))'.dependencies]"
UNGATED_TABLE = "[dependencies]"
HACK_LINE_RE = re.compile(r'^workspace-hack\s*=\s*\{[^\n]*\}$', re.M)
SECTION_RE = re.compile(r'^\[[^\]]+\]\s*$', re.M)


def strip_hack_lines(text: str) -> tuple[str, str | None]:
    """Remove every hack line; return (text, the first line seen or None)."""
    line = None
    out = []
    for l in text.splitlines(keepends=True):
        if HACK_LINE_RE.match(l.strip()) and l.strip() == l.strip():
            pass
        m = HACK_LINE_RE.match(l.rstrip("\n"))
        if m:
            if line is None:
                line = m.group(0)
            # drop the line and one preceding blank if it makes doubles
            if out and out[-1].strip() == "":
                out.pop()
            continue
        out.append(l)
    return "".join(out), line


def strip_own_gated_tables(text: str) -> str:
    """Remove gated tables whose body is exactly one hack line (our append)."""
    pattern = re.compile(
        r"\n*\[target\.'cfg\(not\(target_arch = \"wasm32\"\)'\)\.dependencies\]\n"
        r"workspace-hack\s*=\s*\{[^\n]*\}\n"
    )
    return pattern.sub("\n", text)


def gate_manifest(path: Path, write: bool = True) -> bool:
    """Normalize the workspace-hack edge into a not(wasm32) table.

    With write=False (check mode) the computed content is compared but not
    persisted: the function still returns True when the manifest would
    change, so --check can fail without dirtying the tree.
    """
    original = path.read_text()

    text, hack_line = strip_hack_lines(strip_own_gated_tables(original))
    if hack_line is None:
        return False  # no hack edge here

    # Collapse triple blank lines left by removals.
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text.endswith("\n"):
        text += "\n"

    gated_headers = [h for h in SECTION_RE.finditer(text) if h.group(0).strip() == GATED]
    if gated_headers:
        header_end = gated_headers[0].end()
        gated = text[:header_end] + "\n" + hack_line + text[header_end:]
    else:
        sep = "\n\n" if text.endswith("\n") else "\n\n\n"
        gated = text + sep + GATED + "\n" + hack_line + "\n"

    if gated != original:
        if write:
            path.write_text(gated)
        return True
    return False


def ungate_manifest(path: Path, write: bool = True) -> bool:
    """Restore hakari's canonical ungated edge in [dependencies].

    With write=False (check mode) the computed content is compared but not
    persisted: the function still returns True when the manifest would
    change, so --check can fail without dirtying the tree.
    """
    original = path.read_text()

    text, _ = strip_hack_lines(strip_own_gated_tables(original))
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text.endswith("\n"):
        text += "\n"

    # find the hack line as it appears in the ORIGINAL (to preserve path form)
    m = HACK_LINE_RE.search(original)
    if not m:
        return False
    line = m.group(0)

    dep_headers = [h for h in SECTION_RE.finditer(text) if h.group(0).strip() == UNGATED_TABLE]
    if dep_headers:
        header_end = dep_headers[0].end()
        ungated = text[:header_end] + "\n" + line + text[header_end:]
    else:
        # insert a [dependencies] table right after the [package] section's
        # first blank-line boundary — simplest: append before first other table
        first_table = SECTION_RE.search(text)
        if first_table:
            ungated = text[:first_table.start()] + UNGATED_TABLE + "\n" + line + "\n\n" + text[first_table.start():]
        else:
            ungated = text + "\n" + UNGATED_TABLE + "\n" + line + "\n"

    if ungated != original:
        if write:
            path.write_text(ungated)
        return True
    return False


def main() -> int:
    args = [a for a in sys.argv[1:]]
    check = "--check" in args
    ungate = "--ungate" in args
    verb = ungate_manifest if ungate else gate_manifest
    changed = []
    for manifest in sorted(ROOT.glob("crates/*/Cargo.toml")):
        if "workspace-hack" not in manifest.read_text():
            continue
        if verb(manifest, write=not check):
            changed.append(str(manifest.relative_to(ROOT)))
    if changed:
        for name in changed:
            print(f"{'ungated' if ungate else 'gated'}: {name}")
        if check:
            print(
                "hack_wasm_gate: --check failed; run `python3 scripts/hack_wasm_gate.py`",
                file=sys.stderr,
            )
            return 1
    else:
        print("hack_wasm_gate: nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
