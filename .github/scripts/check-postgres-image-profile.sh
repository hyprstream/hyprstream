#!/usr/bin/env bash
# Keep every independent artifact build on the reviewed staging Postgres shape.
set -euo pipefail

readonly FEATURES='otel,gittorrent,xet,credential-pds-postgres,pds-postgres,rocksdb'
readonly PROFILE_SOURCES=(
  Dockerfile
  .github/scripts/graviton-release-lanes.sh
  .github/workflows/graviton-build-validate.yml
  .github/scripts/postgres-image-qualification.sh
)

# Python is part of the pinned builder; ripgrep is not. Use one parser for
# exact profile presence and the feature-selection rejection in every source.
python3 - "${FEATURES}" "${PROFILE_SOURCES[@]}" <<'PYTHON'
import re
import sys
from pathlib import Path

features, *sources = sys.argv[1:]
for source in sources:
    text = Path(source).read_text()
    if features not in text:
        sys.exit(f"staging Postgres image profile missing from {source}")
    if "--no-default-features" not in text:
        sys.exit(f"{source} can build the artifact with default features")
    if re.search(r"--features(?:\s+|=)[^#\n]*credential-pds(?!-postgres)", text):
        sys.exit(f"{source} selects the PGlite credential-pds feature")
PYTHON

# Keep the qualification harness's redacted failure diagnostics executable;
# this uses a cargo stub and is not PostgreSQL qualification evidence.
bash .github/scripts/test-postgres-image-qualification.sh
