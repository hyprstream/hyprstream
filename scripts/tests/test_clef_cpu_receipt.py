"""The provenance gate rejects stale binaries and obsolete authorization receipts."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("validate_clef_cpu_receipt", SCRIPTS / "validate_clef_cpu_receipt.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TREE = "abcdef0123456789abcdef0123456789abcdef01"


class ReceiptProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / VALIDATOR.SOURCE
        source.parent.mkdir(parents=True)
        source.write_text("candidate source\n")
        executable = self.root / "captured-executable"
        executable.write_bytes(b"candidate executable")
        identity = {"commit": COMMIT, "tree": TREE, "acceptance_source_sha256": VALIDATOR.sha(source)}
        self.build = {
            "schema": "clef-cpu-build/2", "source": identity, "source_after": copy.deepcopy(identity),
            "build_exit": 0, "command": ["cargo", "test", "--release", VALIDATOR.TEST],
            "compiler_artifact": {"profile": {"opt_level": "3", "test": True, "debug_assertions": False}},
            "environment": {"HYPRSTREAM_LIBTORCH_VARIANT": "cpu"},
            "executable": str(executable), "executable_sha256": VALIDATOR.sha(executable),
        }
        self.receipt = {
            "source_identity": {"git_sha": COMMIT[:7], "git_dirty": "false",
                                "acceptance_source_sha256": identity["acceptance_source_sha256"]},
            "result": "AUTHENTICATED_CPU_SERVICE_CASES_PASS", "t6_complete": True,
            "model_service_stop_complete": True,
            "configured_owner_subject": "did:plc:configuredowner",
            "configured_owner_authenticated": True, "unconfigured_owner_denied": True,
            "startup_profile": {"config_sha256": "a" * 64, "manifest_sha256": "b" * 64},
            "corpus_records": 5, "corpus_questions": 11,
            "orders": json.loads((SCRIPTS / "tests/fixtures/clef_cpu_order_summary.json").read_text())["orders"],
        }

        for index, order in enumerate(self.receipt["orders"]):
            order["subject"] = "bob" if index == 6 else "did:plc:configuredowner"
        self.receipt["source_identity"]["executable_sha256"] = self.build["executable_sha256"]
        self.run_result = {"configured_owner_subject": "did:plc:configuredowner", "candidate": COMMIT, "executable_sha256": self.build["executable_sha256"],
                           "exit": 0, "controller_exit": 0, "native_samples": 1,
                           "peak_native_swap_kib": 0, "peak_group_swap_kib": 0,
                           "config_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                           "config_unchanged_after": True, "manifest_unchanged_after": True}

    def run_validator(self):
        def git(command, **kwargs):
            return {("status", "--porcelain"): "", ("rev-parse", "HEAD"): COMMIT,
                    ("rev-parse", "HEAD^{tree}"): TREE}[tuple(command[3:])]
        with patch.object(VALIDATOR.subprocess, "check_output", side_effect=git):
            receipt_path = self.root / "receipt.json"
            receipt_path.write_text(json.dumps(self.receipt))
            digest = VALIDATOR.sha(receipt_path)
            result = {"service_receipt_sha256": digest, **self.run_result}
            return VALIDATOR.validate(self.receipt, self.build, self.root, COMMIT, result, digest)

    def test_matching_runtime_source_and_binary_bind_candidate(self):
        self.assertEqual(self.run_validator()["candidate"], COMMIT)

    def test_earlier_receipts_without_compiled_source_identity_are_rejected(self):
        del self.receipt["source_identity"]
        with self.assertRaisesRegex(ValueError, "source digest"):
            self.run_validator()

    def test_obsolete_infer_coordinate_is_rejected_even_with_matching_source(self):
        self.receipt["orders"][0]["coordinate"] = "inference:Decide/infer"
        with self.assertRaisesRegex(ValueError, "Decide/decide"):
            self.run_validator()

    def test_old_runtime_commit_or_dirty_build_is_rejected(self):
        for key, value in [("git_sha", "fffffff"), ("git_dirty", "true")]:
            with self.subTest(key=key):
                original = self.receipt["source_identity"][key]
                self.receipt["source_identity"][key] = value
                with self.assertRaisesRegex(ValueError, "build identity"):
                    self.run_validator()
                self.receipt["source_identity"][key] = original

    def test_real_receipt_order_count_is_ten_not_question_count(self):
        self.assertEqual(len(self.receipt["orders"]), 10)
        self.assertEqual(self.run_validator()["questions"], 11)
        self.receipt["orders"].append(copy.deepcopy(self.receipt["orders"][0]))
        with self.assertRaisesRegex(ValueError, "ten verified"):
            self.run_validator()

    def test_runtime_executable_mismatch_at_same_commit_is_rejected(self):
        self.receipt["source_identity"]["executable_sha256"] = "alternate-debug-or-features-build"
        with self.assertRaisesRegex(ValueError, "runtime executable"):
            self.run_validator()

    def test_failed_native_or_controller_exit_is_rejected(self):
        for field in ["exit", "controller_exit"]:
            with self.subTest(field=field):
                self.run_result[field] = 101
                with self.assertRaisesRegex(ValueError, "did not exit successfully"):
                    self.run_validator()
                self.run_result[field] = 0

    def test_run_receipt_or_executable_or_candidate_mismatch_is_rejected(self):
        for field in ["candidate", "executable_sha256", "service_receipt_sha256"]:
            with self.subTest(field=field):
                original = self.run_result.get(field)
                self.run_result[field] = "different run"
                with self.assertRaisesRegex(ValueError, "actual run result does not bind"):
                    self.run_validator()
                if original is None:
                    del self.run_result[field]
                else:
                    self.run_result[field] = original

    def test_unfinished_service_stop_is_rejected(self):
        self.receipt["model_service_stop_complete"] = False
        with self.assertRaisesRegex(ValueError, "completed model service stop"):
            self.run_validator()

    def test_missing_native_samples_or_observed_swapping_is_rejected(self):
        for field, value in [("native_samples", 0), ("peak_native_swap_kib", 1),
                             ("peak_group_swap_kib", 1)]:
            with self.subTest(field=field):
                original = self.run_result[field]
                self.run_result[field] = value
                with self.assertRaisesRegex(ValueError, "lacks native/group observations"):
                    self.run_validator()
                self.run_result[field] = original

    def test_selected_config_or_manifest_mismatch_is_rejected(self):
        for field in ["config_sha256", "manifest_sha256"]:
            with self.subTest(field=field):
                original = self.run_result[field]
                self.run_result[field] = "c" * 64
                with self.assertRaisesRegex(ValueError, "selected config/manifest does not match"):
                    self.run_validator()
                self.run_result[field] = original

    def test_selected_config_or_manifest_replacement_is_rejected(self):
        for field in ["config_unchanged_after", "manifest_unchanged_after"]:
            with self.subTest(field=field):
                self.run_result[field] = False
                with self.assertRaisesRegex(ValueError, "changed during execution"):
                    self.run_validator()
                self.run_result[field] = True

    def test_configured_owner_mismatch_or_unproven_other_did_denial_is_rejected(self):
        for field, value in [("configured_owner_subject", "did:plc:otherowner"),
                             ("configured_owner_authenticated", False), ("unconfigured_owner_denied", False)]:
            with self.subTest(field=field):
                original = self.receipt[field]
                self.receipt[field] = value
                with self.assertRaisesRegex(ValueError, "configured owner was not authenticated"):
                    self.run_validator()
                self.receipt[field] = original

    def test_old_literal_primary_subject_is_rejected(self):
        self.receipt["orders"][0]["subject"] = "alice"
        with self.assertRaisesRegex(ValueError, "work-order subjects do not bind"):
            self.run_validator()

    def test_changed_executable_is_rejected(self):
        Path(self.build["executable"]).write_bytes(b"overwritten shared target")
        with self.assertRaisesRegex(ValueError, "executable changed"):
            self.run_validator()


if __name__ == "__main__":
    unittest.main()
