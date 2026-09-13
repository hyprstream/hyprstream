#!/usr/bin/env python3
"""Causal fixtures for the build-matrix contract gate.

Each test builds a minimal synthetic repository (manifest + one workflow),
runs the real checker functions in-process, and asserts the accept/reject
outcome. Mutations must be rejected for the stated reason — a fixture that
silently passes would mean the gate is vacuous.
"""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest

import sys

CHECKER_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHECKER_DIR))

import check_build_matrix as cbm  # noqa: E402


PIN_FILE = ".github/builder-image.env"
PIN_VALUE = (
    "ghcr.io/hyprstream/rust-builder-arm64@"
    "sha256:a9966f5e8b71319b63ffd2e684efe27f5144c6e419b5889b4c0465007c9da056"
)

BASE_MANIFEST: dict = {
    "schema_version": 1,
    "runner_classes": {
        "arm64-merge-gate": {
            "labels": ["self-hosted", "linux", "arm64", "hyprstream-merge-gate"],
            "architecture": "aarch64",
            "status": "live",
            "substrate": "synthetic arm64 pool",
        },
        "github-hosted": {
            "labels": ["ubuntu-latest"],
            "architecture": "x86_64",
            "status": "legacy-allowlisted",
            "substrate": "synthetic hosted pool",
        },
    },
    "rows": [
        {
            "id": "appimage-x86_64-cpu",
            "product": "appimage",
            "arch": "x86_64",
            "target_triple": "x86_64-unknown-linux-gnu",
            "backend": "cpu",
            "status": "active",
            "runner_class": "github-hosted",
            "builder_image": None,
            "rust_toolchain": "synthetic",
            "libtorch": {"source": "synthetic", "identity": "synthetic"},
            "accelerator": "none",
            "cache": {"kind": "s3-sccache", "namespace": "appimage/x86_64/cpu"},
            "artifacts": ["hyprstream-{version}-cpu-x86_64.AppImage"],
            "staging_artifacts": ["universal-staging-cpu"],
        },
        {
            "id": "appimage-aarch64-cpu",
            "product": "appimage",
            "arch": "aarch64",
            "target_triple": "aarch64-unknown-linux-gnu",
            "backend": "cpu",
            "status": "active",
            "runner_class": "arm64-merge-gate",
            "builder_image": f"pin:{PIN_FILE}",
            "rust_toolchain": "synthetic",
            "libtorch": {"source": "synthetic", "identity": "synthetic"},
            "accelerator": "none",
            "cache": {"kind": "none"},
            "artifacts": ["hyprstream-{version}-cpu-aarch64.AppImage"],
            "staging_artifacts": [],
        },
        {
            "id": "appimage-aarch64-cuda130",
            "product": "appimage",
            "arch": "aarch64",
            "target_triple": "aarch64-unknown-linux-gnu",
            "backend": "cuda130",
            "status": "gated",
            "runner_class": "arm64-merge-gate",
            "builder_image": None,
            "rust_toolchain": "synthetic",
            "libtorch": {"source": "synthetic", "identity": "synthetic"},
            "accelerator": "nvidia-cuda-13.0",
            "cache": {"kind": "none"},
            "artifacts": ["hyprstream-{version}-cuda130-aarch64.AppImage"],
            "staging_artifacts": [],
            "gate": {
                "blocker": "no ARM GPU in the fleet",
                "build_only_evidence": "https://example.invalid/run/1",
                "hardware_proof": None,
            },
        },
    ],
    "hosted_allowlist": {
        "wf.yml": {
            "hosted-job": {
                "target_class": "arm64-merge-gate",
                "reason": "pending migration",
            }
        }
    },
}

HOSTED_WF = """\
name: synthetic
on: [push]
jobs:
  hosted-job:
    runs-on: ubuntu-latest
    steps:
    - run: true
  selfhosted-job:
    runs-on: [self-hosted, linux, arm64, hyprstream-merge-gate]
    steps:
    - run: true
"""


class BuildMatrixFixtures(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        add = self.root / ".github"
        (add / "workflows").mkdir(parents=True)
        (add / "workflows" / "wf.yml").write_text(HOSTED_WF, encoding="utf-8")
        (self.root / PIN_FILE).write_text(PIN_VALUE + "\n", encoding="utf-8")
        self.manifest: dict = copy.deepcopy(BASE_MANIFEST)
        self.write_manifest()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_manifest(self) -> None:
        (self.root / ".github" / "build-matrix.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )

    def mutate(self, *path, value, write: bool = True) -> None:
        obj: dict = self.manifest
        for key in path[:-1]:
            obj = obj[key]
        if value is cbm:  # sentinel for deletion
            del obj[path[-1]]
        else:
            obj[path[-1]] = value
        if write:
            self.write_manifest()

    def expect_ok(self) -> None:
        remaining = cbm.run_checks(self.root)
        self.assertEqual(remaining, ["wf.yml:hosted-job"])

    def expect_fail(self, needle: str) -> None:
        with self.assertRaises(cbm.Fail) as ctx:
            cbm.run_checks(self.root)
        self.assertIn(needle, str(ctx.exception))

    # -- control -----------------------------------------------------------

    def test_clean_synthetic_repo_passes(self) -> None:
        self.expect_ok()

    # -- hosted-runner zero scan (N2.5) -------------------------------------

    def test_unallowlisted_hosted_job_is_rejected(self) -> None:
        self.manifest["hosted_allowlist"].pop("wf.yml")
        self.write_manifest()
        self.expect_fail("not in the build-matrix hosted_allowlist")

    def test_stale_allowlist_entry_is_rejected(self) -> None:
        # Migration completed (workflow moved to self-hosted) but the entry
        # was not deleted: the checklist must not be allowed to rot.
        wf = HOSTED_WF.replace("runs-on: ubuntu-latest", "runs-on: [self-hosted, linux, arm64, hyprstream-merge-gate]")
        (self.root / ".github" / "workflows" / "wf.yml").write_text(wf, encoding="utf-8")
        self.expect_fail("stale hosted_allowlist")

    def test_unknown_runs_on_labels_fail_closed(self) -> None:
        wf = HOSTED_WF.replace("runs-on: ubuntu-latest", "runs-on: ${{ matrix.os }}")
        (self.root / ".github" / "workflows" / "wf.yml").write_text(wf, encoding="utf-8")
        self.expect_fail("cannot classify")

    def test_job_without_runs_on_is_rejected(self) -> None:
        wf = HOSTED_WF + """\
  broken-job:
    steps:
    - run: true
"""
        (self.root / ".github" / "workflows" / "wf.yml").write_text(wf, encoding="utf-8")
        self.expect_fail("no single-line `runs-on:`")

    def test_hosted_target_must_not_be_github_hosted(self) -> None:
        self.mutate(
            "hosted_allowlist", "wf.yml", "hosted-job", "target_class",
            value="github-hosted",
        )
        self.expect_fail("only reviewed external services")

    def test_external_service_must_target_github_hosted(self) -> None:
        self.mutate(
            "hosted_allowlist", "wf.yml", "hosted-job",
            value={
                "target_class": "arm64-merge-gate",
                "external_service": True,
                "reason": "synthetic external",
            },
        )
        self.expect_fail("external_service entries must target")

    # -- ARM CUDA fail-closed gate (N3) --------------------------------------

    def test_arm_gpu_row_activated_without_proof_is_rejected(self) -> None:
        self.mutate("rows", 2, "status", value="active")
        self.expect_fail("hardware_proof")

    def test_arm_gpu_row_active_with_proof_is_accepted(self) -> None:
        self.mutate("rows", 2, "status", value="active", write=False)
        gate = self.manifest["rows"][2]["gate"]
        gate["hardware_proof"] = "https://example.invalid/preflight-green"
        self.write_manifest()
        self.expect_ok()

    def test_arm_gpu_row_without_gate_is_rejected(self) -> None:
        self.mutate("rows", 2, "gate", value=cbm)  # delete
        self.expect_fail("aarch64 GPU backend requires a `gate` object")

    def test_active_row_with_gate_object_is_rejected(self) -> None:
        # Consistency runs both ways: an active CPU row carrying a leftover
        # gate object hides a half-finished migration.
        row = copy.deepcopy(self.manifest["rows"][2]["gate"])
        self.mutate("rows", 1, "gate", value=row, write=False)
        self.manifest["rows"][1]["gate"]["hardware_proof"] = None
        self.write_manifest()
        self.expect_fail("active rows must not carry a `gate` object")

    # -- structural row rules --------------------------------------------------

    def test_duplicate_row_id_is_rejected(self) -> None:
        self.mutate("rows", 1, "id", value="appimage-x86_64-cpu")
        self.expect_fail("duplicate row id")

    def test_arch_mismatched_triple_is_rejected(self) -> None:
        self.mutate("rows", 1, "target_triple", value="x86_64-unknown-linux-gnu")
        self.expect_fail("target_triple must be 'aarch64-unknown-linux-gnu'")

    def test_row_arch_must_match_runner_class_arch(self) -> None:
        self.mutate("rows", 1, "runner_class", value="github-hosted")
        self.expect_fail("does not match runner class")

    def test_unpinned_builder_image_is_rejected(self) -> None:
        self.mutate(
            "rows", 1, "builder_image",
            value="ghcr.io/hyprstream/rust-builder-arm64:latest",
        )
        self.expect_fail("digest-pinned")

    def test_pin_ref_to_bad_pin_file_content_is_rejected(self) -> None:
        (self.root / ".github" / "builder-image.env").write_text(
            "ghcr.io/hyprstream/rust-builder-arm64:latest\n", encoding="utf-8"
        )
        self.expect_fail("exactly one digest-pinned image")

    def test_cache_namespace_missing_arch_token_is_rejected(self) -> None:
        self.mutate("rows", 0, "cache", value={"kind": "s3-sccache", "namespace": "appimage/cpu"})
        self.expect_fail("architecture token")

    def test_cache_namespace_missing_backend_token_is_rejected(self) -> None:
        self.mutate("rows", 0, "cache", value={"kind": "s3-sccache", "namespace": "appimage/x86_64"})
        self.expect_fail("backend token")

    def test_shared_cache_namespace_is_rejected(self) -> None:
        self.mutate(
            "rows", 1, "cache",
            value={"kind": "s3-sccache", "namespace": "appimage/x86_64/cpu"},
        )
        self.expect_fail("cache namespace")

    def test_artifact_template_missing_version_is_rejected(self) -> None:
        self.mutate("rows", 0, "artifacts", value=["hyprstream-cpu-x86_64.AppImage"])
        self.expect_fail("{version}")

    def test_shared_artifact_template_is_rejected(self) -> None:
        self.mutate(
            "rows", 1, "artifacts",
            value=["hyprstream-{version}-cpu-x86_64.AppImage"],
        )
        self.expect_fail("artifact template")

    def test_bad_schema_version_is_rejected(self) -> None:
        self.mutate("schema_version", value=2)
        self.expect_fail("unsupported schema_version")

    def test_missing_manifest_is_rejected(self) -> None:
        (self.root / ".github" / "build-matrix.json").unlink()
        self.expect_fail("manifest missing")


if __name__ == "__main__":
    unittest.main()
