#!/usr/bin/env python3
"""Verify that staging-candidate inputs are real, exact-source CI gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """The supplied run cannot authorize candidate publication."""


def _require_run(
    run: dict[str, Any], *, source_sha: str, path: str, name: str
) -> None:
    if run.get("name") != name:
        raise ValidationError(f"expected workflow {name}")
    if run.get("path") != path:
        raise ValidationError(f"expected workflow path {path}")
    if run.get("event") != "workflow_dispatch":
        raise ValidationError("expected a manually dispatched real gate")
    if run.get("head_sha") != source_sha:
        raise ValidationError("run source does not match the candidate source")
    if run.get("conclusion") != "success":
        raise ValidationError("run did not succeed")


def _require_job(
    jobs: dict[str, Any], *, name: str, required_steps: set[str]
) -> None:
    for job in jobs.get("jobs", []):
        if job.get("name") != name or job.get("conclusion") != "success":
            continue
        completed = {
            step.get("name")
            for step in job.get("steps", [])
            if step.get("conclusion") == "success"
        }
        if required_steps <= completed:
            return
    missing = ", ".join(sorted(required_steps))
    raise ValidationError(f"missing successful real gate job steps: {missing}")


def validate(
    source_sha: str,
    oci_run: dict[str, Any],
    oci_jobs: dict[str, Any],
    preflight_run: dict[str, Any],
    preflight_jobs: dict[str, Any],
) -> None:
    _require_run(
        oci_run,
        source_sha=source_sha,
        path=".github/workflows/oci-runtime-validate.yml",
        name="OCI Runtime Image Validation",
    )
    _require_job(
        oci_jobs,
        name="build-runtime-and-smoke-test",
        required_steps={
            "Build runtime image (cpu-arm64)",
            "Qualify the OCI PostgreSQL profile against a live database",
            "Assert smoke container runs as uid 65532",
        },
    )
    _require_run(
        preflight_run,
        source_sha=source_sha,
        path=".github/workflows/merge-gate-preflight.yml",
        name="Merge-gate preflight",
    )
    _require_job(
        preflight_jobs,
        name="Build + test (arm64 preflight)",
        required_steps={
            "Assert immutable dispatch SHA",
            "Build + test (release, in rust-builder container)",
            "Assert browser conformance receipt",
        },
    )


def _json(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--oci-run", required=True)
    parser.add_argument("--oci-jobs", required=True)
    parser.add_argument("--preflight-run", required=True)
    parser.add_argument("--preflight-jobs", required=True)
    args = parser.parse_args()
    try:
        validate(
            args.source_sha,
            _json(args.oci_run),
            _json(args.oci_jobs),
            _json(args.preflight_run),
            _json(args.preflight_jobs),
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise SystemExit(f"candidate validation refused: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
