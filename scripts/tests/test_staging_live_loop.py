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
        self.stub("cp", '''
case "$*" in
  *.restore.*) [ "${TEST_FAIL_RESTORE:-}" != true ] || exit 1 ;;
  *.backup.*)
    if [ "${TEST_FAIL_BACKUP:-}" = true ]; then
      for cp_arg do cp_target=$cp_arg; done
      printf partial > "$cp_target"
      exit 1
    fi ;;
esac
exec /usr/bin/cp "$@"
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
        self.assertIn("HYPRSTREAM_BUILD_JOBS=2", commands)
        self.assertIn(f"CARGO_TARGET_DIR={self.cache}/target", commands)
        self.assertIn(f"--volume {self.cache}:{self.cache}:Z", commands)
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

    def test_shared_storage_requires_combined_build_headroom(self):
        self.stub("df", "printf 'Filesystem 1024-blocks Used Available Capacity Mounted\\ntest 200000000 1 41943040 1%% /\\n'")
        result = self.run_guest("build")
        self.assertEqual(result.returncode, 75)
        self.assertIn("60 GiB", result.stderr)
        self.assertNotIn("podman build", self.log.read_text())

    def test_distinct_filesystems_each_have_their_own_headroom(self):
        self.stub("df", "printf 'Filesystem 1024-blocks Used Available Capacity Mounted\\ntest 200000000 1 41943040 1%% /\\n'")
        self.stub("stat", '''
for stat_arg do stat_path=$stat_arg; done
case "$stat_path" in */cache) echo cache-filesystem ;; *) echo image-filesystem ;; esac
''')
        self.assert_ok(self.run_guest("build"))
        self.assertIn("podman build", self.log.read_text())

    def test_event_supports_the_legacy_quadlet_and_matching_unit(self):
        event = self.quadlets / "hyprstream.container"
        event.write_text(self.original)
        self.assert_ok(self.run_guest("activate", "event", "localhost/hyprstream-staging-dev:first"))
        self.assertIn("Image=localhost/hyprstream-staging-dev:first", event.read_text())
        self.assertIn("systemctl --user restart hyprstream.service", self.log.read_text())
        self.assertNotIn("hyprstream-event.service", self.log.read_text())
        self.assert_ok(self.run_guest("rollback", "event"))
        self.assertEqual(event.read_text(), self.original)
        commands = self.log.read_text()
        self.assertEqual(commands.count("systemctl --user restart hyprstream.service"), 2)
        self.assertNotIn("hyprstream-event.service", commands)

    def test_event_uses_the_current_quadlet_and_matching_unit(self):
        event = self.quadlets / "hyprstream-event.container"
        event.write_text(self.original)
        self.assert_ok(self.run_guest("activate", "event", "localhost/hyprstream-staging-dev:first"))
        self.assertIn("Image=localhost/hyprstream-staging-dev:first", event.read_text())
        self.assert_ok(self.run_guest("rollback", "event"))
        self.assertEqual(event.read_text(), self.original)
        commands = self.log.read_text()
        self.assertEqual(commands.count("systemctl --user restart hyprstream-event.service"), 2)
        self.assertNotIn("systemctl --user restart hyprstream.service", commands)
        self.assertEqual(self.quadlet.read_text(), self.original)

    def test_event_rejects_ambiguous_or_unsupported_layout_before_changes(self):
        current = self.quadlets / "hyprstream-event.container"
        legacy = self.quadlets / "hyprstream.container"
        unsupported = self.quadlets / "event.container"
        for layout in [(current, legacy), (unsupported,), ()]:
            with self.subTest(layout=[file.name for file in layout]):
                for file in layout:
                    file.write_text(self.original)
                for args in [("activate", "event", "localhost/hyprstream-staging-dev:first"),
                             ("rollback", "event")]:
                    result = self.run_guest(*args)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("Ambiguous" if len(layout) == 2 else "Missing Event Quadlet", result.stderr)
                    self.assertFalse(self.log.exists())
                    self.assertFalse(self.state.exists())
                    for file in layout:
                        self.assertEqual(file.read_text(), self.original)
                for file in layout:
                    file.unlink()

    def test_event_rejects_symlink_instead_of_falling_back(self):
        event = self.quadlets / "hyprstream-event.container"
        event.symlink_to(self.quadlet)
        result = self.run_guest("activate", "event", "localhost/hyprstream-staging-dev:first")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid Quadlet", result.stderr)
        self.assertEqual(self.quadlet.read_text(), self.original)
        self.assertFalse(self.log.exists())

    def test_normal_deployment_supersedes_previous_rollback_checkpoint(self):
        self.assert_ok(self.run_guest("activate", "policy", "localhost/hyprstream-staging-dev:first"))
        deployed = self.original.replace("sha256:original", "sha256:new-release")
        self.quadlet.write_text(deployed)
        self.assertEqual(self.run_guest("rollback", "policy").returncode, 2)
        self.assertEqual(self.quadlet.read_text(), deployed)
        self.assert_ok(self.run_guest("activate", "policy", "localhost/hyprstream-staging-dev:second"))
        self.assertEqual((self.state / "hyprstream-policy.container.original").read_text(), deployed)
        self.assert_ok(self.run_guest("rollback", "policy"))
        self.assertEqual(self.quadlet.read_text(), deployed)

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

    def test_restore_copy_failure_keeps_backup_and_does_not_retry_new_image(self):
        result = self.run_guest("activate", "policy", "localhost/hyprstream-staging-dev:bad",
                                TEST_FAIL_RESTART="true", TEST_FAIL_RESTORE="true")
        self.assertEqual(result.returncode, 1)
        self.assertEqual((self.state / "hyprstream-policy.container.original").read_text(), self.original)
        self.assertEqual(self.log.read_text().count("systemctl --user restart"), 1)

    def test_partial_initial_backup_never_becomes_a_rollback_checkpoint(self):
        result = self.run_guest("activate", "policy", "localhost/hyprstream-staging-dev:first",
                                TEST_FAIL_BACKUP="true")
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.state / "hyprstream-policy.container.original").exists())
        self.assertEqual(self.quadlet.read_text(), self.original)
        self.assert_ok(self.run_guest("activate", "policy", "localhost/hyprstream-staging-dev:second"))
        self.assertEqual((self.state / "hyprstream-policy.container.original").read_text(), self.original)

    def test_rejects_unowned_service_or_nonlocal_image(self):
        for args in [("activate", "sshd", "localhost/hyprstream-staging-dev:first"),
                     ("activate", "policy", "attacker.example/image:latest")]:
            with self.subTest(args=args):
                self.assertEqual(self.run_guest(*args).returncode, 2)
        self.assertEqual(self.quadlet.read_text(), self.original)


if __name__ == "__main__":
    unittest.main()
