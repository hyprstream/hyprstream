#!/usr/bin/env bash
# Keep every independent artifact build on the reviewed staging Postgres shape.
set -euo pipefail

readonly FEATURES='otel,gittorrent,xet,credential-pds-postgres,pds-postgres,rocksdb,postgres-replay'
readonly PROFILE_SOURCES=(
  Dockerfile
  .github/scripts/graviton-release-lanes.sh
  .github/workflows/graviton-build-validate.yml
  .github/scripts/postgres-image-qualification.sh
  .github/workflows/docker-build.yml
)

# Python is part of the pinned builder; ripgrep is not. Use one parser for
# exact profile presence and the feature-selection rejection in every source.
python3 - "${FEATURES}" "${PROFILE_SOURCES[@]}" <<'PYTHON'
import re
import sys
from pathlib import Path

features, *sources = sys.argv[1:]
base_features = features.removesuffix(",postgres-replay")
for source in sources:
    text = Path(source).read_text()
    if source == "Dockerfile":
        if 'ARG HYPRSTREAM_FEATURES=otel,gittorrent,xet,credential-pds-postgres,pds-postgres,rocksdb' not in text \
                or '--features "${HYPRSTREAM_FEATURES}"' not in text:
            sys.exit("Dockerfile must consume the explicit per-artifact feature profile")
    elif source == ".github/workflows/docker-build.yml":
        amd64_arg = (
            "HYPRSTREAM_FEATURES=${{ matrix.variant == 'cpu' && '"
            + features
            + "' || '"
            + base_features
            + "' }}"
        )
        arm64_arg = "--build-arg HYPRSTREAM_FEATURES=" + features
        if amd64_arg not in text or arm64_arg not in text:
            sys.exit("both Docker CPU architecture children must use the shared replay-enabled profile")
    elif features not in text:
        sys.exit(f"staging Postgres image profile missing from {source}")
    if source != ".github/workflows/docker-build.yml" and "--no-default-features" not in text:
        sys.exit(f"{source} can build the artifact with default features")
    if re.search(r"--features(?:\s+|=)[^#\n]*credential-pds(?!-postgres)", text):
        sys.exit(f"{source} selects the PGlite credential-pds feature")

PYTHON

# Keep the qualification harness's redacted failure diagnostics executable;
# this uses a cargo stub and is not PostgreSQL qualification evidence.
bash .github/scripts/test-postgres-image-qualification.sh
