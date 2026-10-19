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
                return_value={"hyprstream-eval": "hyprstream-eval"},
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


if __name__ == "__main__":
    unittest.main()
