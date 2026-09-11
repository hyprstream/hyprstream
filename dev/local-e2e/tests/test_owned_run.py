from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import tempfile
import unittest


HARNESS_PATH = Path(__file__).resolve().parents[1] / "causal_harness.py"
SPEC = importlib.util.spec_from_file_location("causal_harness", HARNESS_PATH)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


class OwnedRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.task_root = Path(self.temp.name) / "task"
        self.task_root.mkdir(mode=0o700)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_unique_owned_roots_units_ports_and_cleanup(self) -> None:
        with HARNESS.OwnedRun(self.task_root) as first:
            first_root = first.run_root
            first_units = set(first.context["units"].values())
            first_ports = set(first.context["held_loopback_tcp_ports"].values())
            self.assertEqual(len(first_units), 8)
            self.assertEqual(len(first_ports), 8)
            for path in first.context["xdg"].values():
                info = Path(path).lstat()
                self.assertTrue(stat.S_ISDIR(info.st_mode))
                self.assertEqual(stat.S_IMODE(info.st_mode), 0o700)
            for port in first_ports:
                collision = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                with self.assertRaises(OSError):
                    collision.bind(("127.0.0.1", port))
                collision.close()
        assert first_root is not None
        self.assertFalse(first_root.exists())

        with HARNESS.OwnedRun(self.task_root) as second:
            self.assertTrue(first_units.isdisjoint(set(second.context["units"].values())))

    def test_secret_is_mode_0600_and_symlink_is_rejected(self) -> None:
        with HARNESS.OwnedRun(self.task_root) as run:
            secret = run.write_secret("cookie.jar", b"secret\n")
            self.assertEqual(stat.S_IMODE(secret.lstat().st_mode), 0o600)
            self.assertEqual(secret.read_bytes(), b"secret\n")
            outside = self.task_root / "outside"
            outside.write_text("canary\n", encoding="utf-8")
            assert run.run_root is not None
            link = run.run_root / "secrets" / "linked.jar"
            link.symlink_to(outside)
            with self.assertRaises(FileExistsError):
                run.write_secret("linked.jar", b"overwrite\n")
            self.assertEqual(outside.read_text(encoding="utf-8"), "canary\n")

    def test_context_file_has_no_secret_and_is_mode_0600(self) -> None:
        with HARNESS.OwnedRun(self.task_root) as run:
            context_path = Path(run.context["context_path"])
            self.assertEqual(stat.S_IMODE(context_path.lstat().st_mode), 0o600)
            parsed = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertNotIn("secret", json.dumps(parsed).lower())

    def _initialization_failure(self, mutation) -> None:
        class FailingRun(HARNESS.OwnedRun):
            def _write_owned_file(self, relative: str, content: bytes) -> Path:
                if relative == "context.json":
                    mutation(self)
                    raise RuntimeError("injected initialization failure")
                return super()._write_owned_file(relative, content)

        with self.assertRaises(HARNESS.HarnessError):
            FailingRun(self.task_root).__enter__()

    def test_initialization_failure_rejects_missing_and_changed_marker(self) -> None:
        for action in ("missing", "changed"):
            with self.subTest(action=action):
                leaked: list[Path] = []

                def mutate(run) -> None:
                    assert run.run_root is not None
                    leaked.append(run.run_root)
                    marker = run.run_root / "owner.marker"
                    if action == "missing":
                        marker.unlink()
                    else:
                        marker.write_text("wrong nonce\n", encoding="utf-8")

                self._initialization_failure(mutate)
                self.assertEqual(len(leaked), 1)
                self.assertTrue(leaked[0].is_dir())
                shutil.rmtree(leaked[0])

    def test_initialization_failure_rejects_directory_substitution(self) -> None:
        original = self.task_root / "preserved-original"
        replacement: list[Path] = []

        def mutate(run) -> None:
            assert run.run_root is not None
            run.run_root.rename(original)
            run.run_root.mkdir(mode=0o700)
            (run.run_root / "canary").write_text("preserve\n", encoding="utf-8")
            replacement.append(run.run_root)

        self._initialization_failure(mutate)
        self.assertTrue(original.is_dir())
        self.assertEqual(
            (replacement[0] / "canary").read_text(encoding="utf-8"), "preserve\n"
        )
        shutil.rmtree(original)
        shutil.rmtree(replacement[0])

    def test_initialization_failure_rejects_ancestor_symlink_substitution(self) -> None:
        original_task = self.task_root.with_name("task-preserved")
        decoy = self.task_root.with_name("task-decoy")
        original_runs: list[Path] = []

        def mutate(run) -> None:
            assert run.run_root is not None
            original_runs.append(run.run_root)
            self.task_root.rename(original_task)
            decoy.mkdir(mode=0o700)
            self.task_root.symlink_to(decoy, target_is_directory=True)

        self._initialization_failure(mutate)
        self.assertTrue(self.task_root.is_symlink())
        self.assertTrue(
            (original_task / original_runs[0].name / "owner.marker").is_file()
        )
        self.task_root.unlink()
        shutil.rmtree(original_task)
        shutil.rmtree(decoy)


if __name__ == "__main__":
    unittest.main()
