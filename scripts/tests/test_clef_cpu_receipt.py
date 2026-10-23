"""The provenance gate rejects stale binaries and obsolete authorization receipts."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
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
        libtorch = self.root / "libtorch"
        library_dir = libtorch / "lib"
        library_dir.mkdir(parents=True)
        self.ldd_paths = {}
        for name in ("libtorch_cpu.so", "libc10.so", "libgomp.so.1"):
            library = library_dir / name
            library.write_bytes(f"captured {name}".encode())
            self.ldd_paths[name] = library
        identity = {"commit": COMMIT, "tree": TREE, "acceptance_source_sha256": VALIDATOR.sha(source)}
        self.build = {
            "schema": "clef-cpu-build/2", "source": identity, "source_after": copy.deepcopy(identity),
            "build_exit": 0, "command": ["cargo", "test", "--release", VALIDATOR.TEST],
            "compiler_artifact": {"profile": {"opt_level": "3", "test": True, "debug_assertions": False}},
            "environment": {"HYPRSTREAM_LIBTORCH_VARIANT": "cpu", "LIBTORCH": str(libtorch)},
            "executable": str(executable), "executable_sha256": VALIDATOR.sha(executable),
            "cpu_libraries_sha256": {name: VALIDATOR.sha(self.ldd_paths[name])
                                     for name in ("libtorch_cpu.so", "libc10.so")},
            "resolved_runtime_libraries": {
                name: {"path": str(path.resolve()), "sha256": VALIDATOR.sha(path)}
                for name, path in self.ldd_paths.items()},
        }
        self.receipt = {
            "source_identity": {"git_sha": COMMIT[:7], "git_dirty": "false",
                                "acceptance_source_sha256": identity["acceptance_source_sha256"]},
            "result": "AUTHENTICATED_CPU_SERVICE_CASES_PASS", "t6_complete": True,
            "model_service_stop_complete": True,
            "corpus_records": 5, "corpus_questions": 11,
            "orders": json.loads((SCRIPTS / "tests/fixtures/clef_cpu_order_summary.json").read_text())["orders"],
        }

        self.receipt["source_identity"]["executable_sha256"] = self.build["executable_sha256"]
        self.receipt["source_identity"]["runtime_libraries"] = {
            entry["path"]: {"sha256": entry["sha256"], "device": "08:01", "inode": index}
            for index, (name, entry) in enumerate(self.build["resolved_runtime_libraries"].items(), 1)
        }
        self.run_result = {"candidate": COMMIT, "executable_sha256": self.build["executable_sha256"],
                           "exit": 0, "controller_exit": 0, "native_samples": 1,
                           "peak_native_swap_kib": 0, "peak_group_swap_kib": 0}

    def run_validator(self):
        def git(command, **kwargs):
            return {("status", "--porcelain"): "", ("rev-parse", "HEAD"): COMMIT,
                    ("rev-parse", "HEAD^{tree}"): TREE}[tuple(command[3:])]
        def ldd(command, **kwargs):
            self.assertEqual(command, ["ldd", self.build["executable"]])
            output = "\n".join(f"{name} => {path} (0x0)"
                               for name, path in self.ldd_paths.items())
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        with (patch.object(VALIDATOR.subprocess, "check_output", side_effect=git),
              patch.object(VALIDATOR.subprocess, "run", side_effect=ldd)):
            receipt_path = self.root / "receipt.json"
            receipt_path.write_text(json.dumps(self.receipt))
            digest = VALIDATOR.sha(receipt_path)
            result = {"service_receipt_sha256": digest, **self.run_result}
            return VALIDATOR.validate(self.receipt, self.build, self.root, COMMIT, result, digest)

    def test_matching_runtime_source_and_binary_bind_candidate(self):
        self.assertEqual(self.run_validator()["candidate"], COMMIT)

    def test_resolved_library_symlink_uses_canonical_path(self):
        alias = self.root / "libtorch-cpu-alias.so"
        alias.symlink_to(self.ldd_paths["libtorch_cpu.so"])
        self.ldd_paths["libtorch_cpu.so"] = alias
        self.assertEqual(self.run_validator()["candidate"], COMMIT)

    def test_changed_or_missing_cpu_runtime_library_is_rejected(self):
        library = self.ldd_paths["libtorch_cpu.so"]
        library.write_bytes(b"different native code")
        with self.assertRaisesRegex(ValueError, "CPU runtime library changed"):
            self.run_validator()
        library.unlink()
        with self.assertRaisesRegex(ValueError, "CPU runtime library is unavailable"):
            self.run_validator()

    def test_changed_or_missing_resolved_runtime_library_is_rejected(self):
        library = self.ldd_paths["libgomp.so.1"]
        library.write_bytes(b"different OpenMP runtime")
        with self.assertRaisesRegex(ValueError, "runtime library path or contents changed"):
            self.run_validator()
        library.unlink()
        with self.assertRaisesRegex(ValueError, "runtime libraries cannot be resolved"):
            self.run_validator()

    def test_resolved_runtime_manifest_rejects_each_changed_library(self):
        for name in ("libtorch_cpu.so", "libc10.so", "libgomp.so.1"):
            with self.subTest(name=name):
                library = self.ldd_paths[name]
                original = library.read_bytes()
                cpu_hash = self.build["cpu_libraries_sha256"].get(name)
                try:
                    library.write_bytes(original + b" changed after build")
                    # Let the separate LIBTORCH-directory check pass so this
                    # assertion exercises the exact ldd-resolved build snapshot.
                    if cpu_hash is not None:
                        self.build["cpu_libraries_sha256"][name] = VALIDATOR.sha(library)
                    with self.assertRaisesRegex(ValueError, "runtime library path or contents changed"):
                        self.run_validator()
                finally:
                    library.write_bytes(original)
                    if cpu_hash is not None:
                        self.build["cpu_libraries_sha256"][name] = cpu_hash

    def test_loader_switch_to_different_library_is_rejected(self):
        replacement = self.root / "alternate-libgomp.so.1"
        replacement.write_bytes(self.ldd_paths["libgomp.so.1"].read_bytes())
        self.ldd_paths["libgomp.so.1"] = replacement
        with self.assertRaisesRegex(ValueError, "runtime library path or contents changed"):
            self.run_validator()

    def test_missing_library_manifest_fields_are_rejected(self):
        for field in ("cpu_libraries_sha256", "resolved_runtime_libraries"):
            with self.subTest(field=field):
                original = self.build.pop(field)
                with self.assertRaisesRegex(ValueError, "runtime library"):
                    self.run_validator()
                self.build[field] = original

    def test_runtime_loaded_library_inventory_is_required_and_matches_build(self):
        runtime = self.receipt["source_identity"].pop("runtime_libraries")
        with self.assertRaisesRegex(ValueError, "actual process runtime library identity is missing"):
            self.run_validator()
        self.receipt["source_identity"]["runtime_libraries"] = runtime

        for path in runtime:
            with self.subTest(path=path):
                original = runtime[path]["sha256"]
                runtime[path]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "actual process loaded runtime library differs"):
                    self.run_validator()
                runtime[path]["sha256"] = original

    def test_runtime_loaded_library_path_set_and_mapping_identity_are_checked(self):
        runtime = self.receipt["source_identity"]["runtime_libraries"]
        path = self.build["resolved_runtime_libraries"]["libgomp.so.1"]["path"]
        runtime.pop(path)
        alternate = str(self.root / "alternate-libgomp.so.1")
        runtime[alternate] = {"sha256": "0" * 64, "device": "08:01", "inode": 4}
        with self.assertRaisesRegex(ValueError, "runtime library set differs"):
            self.run_validator()
        runtime.pop(alternate)
        runtime[path] = {"sha256": self.build["resolved_runtime_libraries"]["libgomp.so.1"]["sha256"],
                         "device": "08:01", "inode": 3}

        extra_path = str(self.root / "libtorch_unexpected.so")
        extra = {extra_path: dict(next(iter(runtime.values())))}
        runtime.update(extra)
        with self.assertRaisesRegex(ValueError, "runtime library set differs"):
            self.run_validator()
        runtime.pop(extra_path)

        c10_path = self.build["resolved_runtime_libraries"]["libc10.so"]["path"]
        original_device = runtime[c10_path]["device"]
        runtime[c10_path]["device"] = "invalid"
        with self.assertRaisesRegex(ValueError, "mapping identity is malformed"):
            self.run_validator()
        runtime[c10_path]["device"] = original_device

        original_inode = runtime[c10_path]["inode"]
        runtime[c10_path]["inode"] = True
        with self.assertRaisesRegex(ValueError, "mapping identity is malformed"):
            self.run_validator()
        runtime[c10_path]["inode"] = original_inode

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

    def test_changed_executable_is_rejected(self):
        Path(self.build["executable"]).write_bytes(b"overwritten shared target")
        with self.assertRaisesRegex(ValueError, "executable changed"):
            self.run_validator()


if __name__ == "__main__":
    unittest.main()
