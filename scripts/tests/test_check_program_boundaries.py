#!/usr/bin/env python3
"""Causal fixtures for the program-boundary rule: program leaf crates stay
leaves — no workspace package may depend on them."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[1] / "check_program_boundaries.py"


class ProgramBoundaryFixtures(unittest.TestCase):
    def run_checker(
        self,
        *,
        platform_deps: dict[str, list[str]] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        platform_deps = platform_deps or {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "crates" / "hyprstream-synthesis" / "src").mkdir(parents=True)
            (root / "crates" / "hyprstream-synthesis" / "Cargo.toml").write_text(
                "[package]\n"
                'name = "hyprstream-synthesis"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                'license = "Apache-2.0"\n',
                encoding="utf-8",
            )
            members = ["crates/hyprstream-synthesis"]
            for index, (crate, dependencies) in enumerate(platform_deps.items()):
                package = root / "crates" / crate
                (package / "src").mkdir(parents=True)
                lines = [
                    "[package]\n",
                    f'name = "{crate}"\n',
                    'version = "0.1.0"\n',
                    'edition = "2021"\n',
                    'license = "Apache-2.0"\n\n',
                    "[dependencies]\n",
                ]
                for number, dependency in enumerate(dependencies):
                    lines.append(
                        f'{dependency} = {{ path = "../dep{index}-{number}", version = "0.1" }}\n'
                    )
                    (root / "crates" / f"dep{index}-{number}" / "src").mkdir(parents=True)
                    (root / "crates" / f"dep{index}-{number}" / "Cargo.toml").write_text(
                        "[package]\n"
                        f'name = "{dependency}"\n'
                        'version = "0.1.0"\n'
                        'edition = "2021"\n'
                        'license = "Apache-2.0"\n',
                        encoding="utf-8",
                    )
                    members.append(f"crates/dep{index}-{number}")
                (package / "Cargo.toml").write_text("".join(lines))
                members.append(f"crates/{crate}")
            (root / "Cargo.toml").write_text(
                "[workspace]\n"
                f"members = {members!r}\n"
                'resolver = "2"\n',
                encoding="utf-8",
            )
            return subprocess.run(
                [sys.executable, str(CHECKER), "--root", str(root)],
                text=True,
                capture_output=True,
            )

    def test_platform_dependency_on_program_leaf_fails(self):
        result = self.run_checker(
            platform_deps={"hyprstream-platform-crate": ["hyprstream-synthesis"]}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hyprstream-platform-crate", result.stdout)
        self.assertIn("hyprstream-synthesis", result.stdout)

    def test_renamed_dependency_is_caught(self):
        # A renamed path dependency (`synth = { package = ... }`) resolves to
        # the same crate — the check must catch the renamed form too.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "crates" / "hyprstream-synthesis"
            leaf.mkdir(parents=True)
            (leaf / "Cargo.toml").write_text(
                '[package]\nname = "hyprstream-synthesis"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            platform = root / "crates" / "hyprstream-platform-crate"
            platform.mkdir(parents=True)
            (platform / "Cargo.toml").write_text(
                "[package]\n"
                'name = "hyprstream-platform-crate"\n'
                'version = "0.1.0"\n\n'
                "[dependencies]\n"
                'synth = { package = "hyprstream-synthesis", path = "../hyprstream-synthesis" }\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(CHECKER), "--root", str(root)],
                text=True,
                capture_output=True,
            )
            self.assertIn("hyprstream-platform-crate", result.stdout)

    def test_clean_workspace_passes(self):
        result = self.run_checker(
            platform_deps={
                "hyprstream-eval": ["hyprstream-decision"],
                "hyprstream-decision": [],
            }
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
