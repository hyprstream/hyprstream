#!/usr/bin/env python3
"""Causal fixtures for staging-candidate CI provenance verification."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
module_path = Path(__file__).with_name("verify-staging-candidate-validation.py")
spec = importlib.util.spec_from_file_location("candidate_validation", module_path)
assert spec and spec.loader
candidate_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate_validation)

SOURCE = "a" * 40


def run(name, path):
    return {
        "name": name,
        "path": path,
        "event": "workflow_dispatch",
        "head_sha": SOURCE,
        "conclusion": "success",
    }


def jobs(name, steps):
    return {
        "jobs": [
            {
                "name": name,
                "conclusion": "success",
                "steps": [{"name": step, "conclusion": "success"} for step in steps],
            }
        ]
    }


OCI_STEPS = {
    "Build runtime image (cpu-arm64)",
    "Qualify the OCI PostgreSQL profile against a live database",
    "Assert smoke container runs as uid 65532",
}
PREFLIGHT_STEPS = {
    "Assert immutable dispatch SHA",
    "Build + test (release, in rust-builder container)",
    "Assert browser conformance receipt",
}


class CandidateValidationTests(unittest.TestCase):
    def valid(self):
        return (
            run("OCI Runtime Image Validation", ".github/workflows/oci-runtime-validate.yml"),
            jobs("build-runtime-and-smoke-test", OCI_STEPS),
            run("Merge-gate preflight", ".github/workflows/merge-gate-preflight.yml"),
            jobs("Build + test (arm64 preflight)", PREFLIGHT_STEPS),
        )

    def test_accepts_real_exact_source_gates(self):
        candidate_validation.validate(SOURCE, *self.valid())

    def test_rejects_pr_placeholder(self):
        oci_run, oci_jobs, preflight_run, preflight_jobs = self.valid()
        oci_run["event"] = "pull_request"
        with self.assertRaises(candidate_validation.ValidationError):
            candidate_validation.validate(SOURCE, oci_run, oci_jobs, preflight_run, preflight_jobs)

    def test_rejects_wrong_workflow_path(self):
        oci_run, oci_jobs, preflight_run, preflight_jobs = self.valid()
        oci_run["path"] = ".github/workflows/rust.yml"
        with self.assertRaises(candidate_validation.ValidationError):
            candidate_validation.validate(SOURCE, oci_run, oci_jobs, preflight_run, preflight_jobs)

    def test_rejects_other_source_head(self):
        oci_run, oci_jobs, preflight_run, preflight_jobs = self.valid()
        preflight_run["head_sha"] = "b" * 40
        with self.assertRaises(candidate_validation.ValidationError):
            candidate_validation.validate(SOURCE, oci_run, oci_jobs, preflight_run, preflight_jobs)

    def test_rejects_missing_real_oci_step(self):
        oci_run, oci_jobs, preflight_run, preflight_jobs = self.valid()
        oci_jobs["jobs"][0]["steps"] = []
        with self.assertRaises(candidate_validation.ValidationError):
            candidate_validation.validate(SOURCE, oci_run, oci_jobs, preflight_run, preflight_jobs)

    def test_rejects_preflight_without_immutable_sha_assertion(self):
        oci_run, oci_jobs, preflight_run, preflight_jobs = self.valid()
        preflight_jobs["jobs"][0]["steps"] = [
            {"name": step, "conclusion": "success"}
            for step in PREFLIGHT_STEPS - {"Assert immutable dispatch SHA"}
        ]
        with self.assertRaises(candidate_validation.ValidationError):
            candidate_validation.validate(SOURCE, oci_run, oci_jobs, preflight_run, preflight_jobs)


if __name__ == "__main__":
    unittest.main()
