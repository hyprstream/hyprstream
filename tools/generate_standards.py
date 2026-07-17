#!/usr/bin/env python3
"""Generate #1059 RFCXML artifacts with the exact pinned xml2rfc toolchain."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "docs/standards/rfc/draft-hyprstream-privacypass-pqhybrid-00.xml"
REQUIREMENTS = ROOT / "docs/standards/xml2rfc-requirements.txt"
PYTHON_VERSION = (ROOT / "docs/standards/.python-version").read_text(encoding="utf-8").strip()


def main() -> int:
    command = [
        "uvx", "--isolated", "--python", PYTHON_VERSION, "--from", "xml2rfc==3.34.0",
        "--with-requirements", str(REQUIREMENTS), "xml2rfc",
        "--no-network", "--text", "--html", "--path", str(XML.parent), str(XML),
    ]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
