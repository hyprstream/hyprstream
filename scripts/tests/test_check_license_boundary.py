#!/usr/bin/env python3
"""Causal fixtures for the one-way Cargo package license boundary."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).resolve().parents[1] / "check_license_boundary.py"


class LicenseBoundaryFixtures(unittest.TestCase):
    def run_fixture(
        self,
        *,
        apache_manifest: str = "",
        middle_manifest: str = "",
        workspace_dependencies: str = "",
        apache_roots: tuple[str, ...] = ("apache",),
        agpl_services: tuple[str, ...] = ("service",),
        other_packages: tuple[str, ...] = ("middle",),
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".github").mkdir()
            for package in ("apache", "middle", "service"):
                (root / "crates" / package).mkdir(parents=True)
            (root / "vendor" / "ghost").mkdir(parents=True)

            (root / "Cargo.toml").write_text(
                "[workspace]\n"
                'members = ["crates/apache", "crates/middle"]\n'
                'exclude = ["crates/service"]\n'
                'resolver = "2"\n\n'
                "[workspace.dependencies]\n"
                f"{workspace_dependencies}",
                encoding="utf-8",
            )
            manifests = {
                "apache": apache_manifest,
                "middle": middle_manifest,
                "service": "",
            }
            for package, extra in manifests.items():
                (root / "crates" / package / "Cargo.toml").write_text(
                    "[package]\n"
                    f'name = "{package}"\n'
                    'version = "0.1.0"\n'
                    'edition = "2021"\n\n'
                    f"{extra}",
                    encoding="utf-8",
                )
            (root / "vendor" / "ghost" / "Cargo.toml").write_text(
                "[package]\n"
                'name = "ghost"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n',
                encoding="utf-8",
            )
            (root / ".github" / "license-boundary.toml").write_text(
                "[license_gate]\n"
                f"apache_roots = {list(apache_roots)!r}\n"
                f"agpl_services = {list(agpl_services)!r}\n"
                f"other_packages = {list(other_packages)!r}\n",
                encoding="utf-8",
            )
            return subprocess.run(
                [sys.executable, str(CHECKER), "--root", str(root)],
                text=True,
                capture_output=True,
                check=False,
            )

    def assert_boundary_failure(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Apache-2.0-to-AGPL dependency", result.stderr)

    def test_clean_graph_passes(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dependencies]\nmiddle = { path = "../middle" }\n'
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_direct_dependency_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dependencies]\nservice = { path = "../service" }\n'
        )
        self.assert_boundary_failure(result)

    def test_transitive_dependency_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dependencies]\nmiddle = { path = "../middle" }\n',
            middle_manifest='[dependencies]\nservice = { path = "../service" }\n',
        )
        self.assert_boundary_failure(result)
        self.assertIn("apache -> middle -> service", result.stderr)

    def test_renamed_dependency_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest=(
                "[dependencies]\n"
                'renamed-service = { package = "service", path = "../service" }\n'
            )
        )
        self.assert_boundary_failure(result)

    def test_target_dependency_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest=(
                "[target.'cfg(unix)'.dependencies]\n"
                'service = { path = "../service" }\n'
            )
        )
        self.assert_boundary_failure(result)

    def test_dev_dependency_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dev-dependencies]\nservice = { path = "../service" }\n'
        )
        self.assert_boundary_failure(result)

    def test_build_dependency_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest='[build-dependencies]\nservice = { path = "../service" }\n'
        )
        self.assert_boundary_failure(result)

    def test_workspace_inherited_renamed_dependency_fails(self) -> None:
        result = self.run_fixture(
            workspace_dependencies=(
                'service-alias = { package = "service", path = "crates/service" }\n'
            ),
            apache_manifest=(
                "[dependencies]\n"
                "service-alias = { workspace = true }\n"
            ),
        )
        self.assert_boundary_failure(result)

    def test_unknown_package_fails(self) -> None:
        result = self.run_fixture(other_packages=("middle", "ghost"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown package(s): ['ghost']", result.stderr)

    def test_unknown_local_dependency_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest=(
                "[dependencies]\n"
                'ghost = { path = "../../vendor/ghost" }\n'
            )
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown local dependency package(s): ['ghost']", result.stderr)

    def test_overlapping_classification_fails(self) -> None:
        result = self.run_fixture(other_packages=("middle", "service"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("classification overlap", result.stderr)

    def test_apache_classification_omission_fails(self) -> None:
        result = self.run_fixture(apache_roots=())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("classification omission: ['apache']", result.stderr)

    def test_service_classification_omission_fails(self) -> None:
        result = self.run_fixture(agpl_services=())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("classification omission: ['service']", result.stderr)


if __name__ == "__main__":
    unittest.main()
