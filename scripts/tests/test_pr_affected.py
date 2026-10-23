"""Regression tests for explicit affected-set package gates."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/pr_affected.py"
SPEC = importlib.util.spec_from_file_location("pr_affected", SCRIPT)
assert SPEC and SPEC.loader
PR_AFFECTED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PR_AFFECTED)


class ExplicitReleasePackageTests(unittest.TestCase):
    def run_affected(self, changed_files):
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            return type("Completed", (), {"returncode": 0})()

        with (
            patch.dict("os.environ", {"BASE_SHA": "base", "HEAD_SHA": "head"}),
            patch.object(PR_AFFECTED, "changed_files", return_value=changed_files),
            patch.object(
                PR_AFFECTED,
                "workspace_members",
                return_value={
                    name: name
                    for name in (*PR_AFFECTED.EXPLICIT_FULL_SET_PACKAGES, "hyprstream")
                },
            ),
            patch.object(PR_AFFECTED, "prepare_guest_artifacts", return_value=0),
            patch.object(PR_AFFECTED.subprocess, "run", side_effect=run),
        ):
            self.assertEqual(PR_AFFECTED.main(), 0)
        return commands

    def test_eval_only_change_runs_package_scoped_release_gates(self):
        commands = self.run_affected(["crates/hyprstream-eval/src/lib.rs"])

        self.assertEqual(
            commands,
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "--cargo-profile",
                    "release",
                    "--profile",
                    "ci",
                    "--no-fail-fast",
                    "-p",
                    "hyprstream-eval",
                ],
                [
                    "cargo",
                    "clippy",
                    "--release",
                    "-p",
                    "hyprstream-eval",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                ],
            ],
        )

    def test_root_trigger_keeps_full_set_and_adds_eval_release_gates(self):
        commands = self.run_affected(
            ["Cargo.toml", "crates/hyprstream-eval/src/lib.rs"]
        )

        self.assertIn(
            [
                "cargo",
                "nextest",
                "run",
                "--cargo-profile",
                "ci-test",
                "--profile",
                "ci",
                "--no-fail-fast",
            ],
            commands,
        )
        self.assertIn(
            [
                "cargo",
                "nextest",
                "run",
                "--cargo-profile",
                "release",
                "--profile",
                "ci",
                "--no-fail-fast",
                "-p",
                "hyprstream-eval",
            ],
            commands,
        )
        self.assertIn(
            [
                "cargo",
                "clippy",
                "--release",
                "-p",
                "hyprstream-eval",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
            commands,
        )

    def test_root_only_trigger_runs_eval_release_gates(self):
        commands = self.run_affected(["Cargo.toml"])

        self.assertIn(
            [
                "cargo",
                "nextest",
                "run",
                "--cargo-profile",
                "release",
                "--profile",
                "ci",
                "--no-fail-fast",
                "-p",
                "hyprstream-eval",
            ],
            commands,
        )

        self.assertIn(
            [
                "cargo",
                "clippy",
                "--release",
                "-p",
                "hyprstream-eval",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
            commands,
        )

    def test_root_fallback_tests_every_nondefault_system_one_package(self):
        for changes in [
            ["Cargo.lock"],
            ["Cargo.toml", "crates/hyprstream-decision-head/src/lib.rs"],
            [".github/scripts/pr_affected.py", "crates/hyprstream-decision/src/lib.rs"],
        ]:
            with self.subTest(changes=changes):
                commands = self.run_affected(changes)
                for package in ["hyprstream-decision", "hyprstream-calibration",
                                "hyprstream-decision-head", "hyprstream-decision-stub",
                                "hyprstream-eval"]:
                    self.assertIn(
                        ["cargo", "nextest", "run", "--cargo-profile", "release",
                         "--profile", "ci", "--no-fail-fast", "-p", package],
                        commands,
                    )
                    self.assertIn(
                        ["cargo", "clippy", "--release", "-p", package,
                         "--all-targets", "--", "-D", "warnings"],
                        commands,
                    )

    def test_narrow_decision_changes_do_not_expand_to_unrelated_packages(self):
        for package in ["hyprstream-decision", "hyprstream-decision-head"]:
            with self.subTest(package=package):
                commands = self.run_affected([f"crates/{package}/src/lib.rs"])
                self.assertEqual(len(commands), 2)
                self.assertTrue(all(command[command.index("-p") + 1] == package
                                    for command in commands))

    def test_native_change_keeps_native_and_nondefault_gates(self):
        commands = self.run_affected([
            "crates/hyprstream/src/services/model.rs",
            "crates/hyprstream-decision-head/src/lib.rs",
        ])
        packages = [command[command.index("-p") + 1] for command in commands]
        self.assertCountEqual(packages, ["hyprstream", "hyprstream",
                                        "hyprstream-decision-head", "hyprstream-decision-head"])



if __name__ == "__main__":
    unittest.main()
