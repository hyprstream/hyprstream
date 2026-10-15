"""Exercise the guest update boundary without running Cargo or changing services."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "staging-live-loop.sh"


class GuestUpdateTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.quadlets = self.root / "quadlets"
        self.quadlets.mkdir()
        self.state = self.root / "state"
        self.cache = self.root / "cache"
        self.log = self.root / "commands"
        self.original = "[Container]\nImage=registry.example/hyprstream@sha256:original\nAutoUpdate=registry\n"
        self.quadlet = self.quadlets / "hyprstream-policy.container"
        self.quadlet.write_text(self.original)
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "HYPRSTREAM_DEV_QUADLET_DIR": str(self.quadlets),
            "HYPRSTREAM_DEV_STATE_DIR": str(self.state),
            "HYPRSTREAM_DEV_CACHE_DIR": str(self.cache),
            "TEST_COMMAND_LOG": str(self.log),
            "TEST_GRAPH_ROOT": str(self.root),
        }
        self.stub("id", 'if [ "$1" = -un ]; then echo rocky; else echo 1000; fi')
        self.stub("df", "printf 'Filesystem 1024-blocks Used Available Capacity Mounted\ntest 200000000 1 100000000 1%% /\n'")
        self.stub("podman", '''
printf 'podman %s\n' "$*" >> "$TEST_COMMAND_LOG"
if [ "$1" = info ]; then printf '%s\n' "$TEST_GRAPH_ROOT"; fi
''')
        self.stub("systemctl", '''
printf 'systemctl %s\n' "$*" >> "$TEST_COMMAND_LOG"
if [ "${TEST_FAIL_RESTART:-}" = true ] && [ "$2" = restart ]; then exit 1; fi
''')

    def stub(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/bin/sh\nset -eu\n{body}\n")
        path.chmod(0o755)

    def run_guest(self, *args, **environment):
        return subprocess.run(
            ["bash", str(SCRIPT), *args], env={**self.env, **environment},
            text=True, capture_output=True, check=False,
        )

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_build_uses_external_cache_and_bounded_jobs_without_restarting(self):
        self.assert_ok(self.run_guest("build"))
        commands = self.log.read_text()
        self.assertIn("--target runtime", commands)
        self.assertIn("CARGO_BUILD_JOBS=2", commands)
        self.assertIn(f"CARGO_TARGET_DIR={self.cache}/target", commands)
        self.assertIn(f"--volume {self.cache}:{self.cache}", commands)
        self.assertNotIn("systemctl", commands)
        self.assertEqual(self.quadlet.read_text(), self.original)

    def test_repeated_updates_preserve_release_rollback(self):
        self.assert_ok(self.run_guest("activate", "policy", "localhost/hyprstream-staging-dev:first"))
        self.assert_ok(self.run_guest("activate", "policy", "localhost/hyprstream-staging-dev:second"))
        self.assertIn("Image=localhost/hyprstream-staging-dev:second", self.quadlet.read_text())
        self.assertNotIn("AutoUpdate=", self.quadlet.read_text())
        backup = self.state / "hyprstream-policy.container.original"
        self.assertEqual(backup.read_text(), self.original)
        self.assert_ok(self.run_guest("rollback", "policy"))
        self.assertEqual(self.quadlet.read_text(), self.original)
        self.assertFalse(backup.exists())

    def test_failed_recovery_retains_rollback_configuration(self):
        result = self.run_guest("activate", "policy", "localhost/hyprstream-staging-dev:bad", TEST_FAIL_RESTART="true")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.quadlet.read_text(), self.original)
        self.assertEqual((self.state / "hyprstream-policy.container.original").read_text(), self.original)
        self.assertIn("retained rollback configuration", result.stderr)

    def test_rejects_excessive_jobs_before_build(self):
        result = self.run_guest("build", HYPRSTREAM_DEV_BUILD_JOBS="32")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())

    def test_rejects_unowned_service_or_nonlocal_image(self):
        for args in [("activate", "sshd", "localhost/hyprstream-staging-dev:first"),
                     ("activate", "policy", "attacker.example/image:latest")]:
            with self.subTest(args=args):
                self.assertEqual(self.run_guest(*args).returncode, 2)
        self.assertEqual(self.quadlet.read_text(), self.original)


if __name__ == "__main__":
    unittest.main()
