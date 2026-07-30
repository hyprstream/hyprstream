#!/usr/bin/env python3
"""Causal fixtures for the exhaustive owner license map and AGPL boundary."""

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
        package_names: tuple[str, str, str] = ("apache", "middle", "service"),
        license_declarations: tuple[str, str, str] = (
            'license = "Apache-2.0"',
            'license = "Apache-2.0"',
            'license = "AGPL-3.0-only"',
        ),
        apache_manifest: str = "",
        middle_manifest: str = "",
        service_manifest: str = "",
        workspace_dependencies: str = "",
        workspace_license: str = "Apache-2.0",
        mit_packages: tuple[str, ...] = (),
        agpl_packages: tuple[str, ...] = ("service",),
        apache_packages: tuple[str, ...] = ("apache", "middle"),
        permissive_roots: tuple[str, ...] = ("apache",),
        agpl_aggregators: tuple[str, ...] = (),
        policy_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_name, middle_name, service_name = package_names
            (root / ".github").mkdir()
            for package in package_names:
                (root / "crates" / package).mkdir(parents=True)
            (root / "vendor" / "ghost").mkdir(parents=True)

            (root / "Cargo.toml").write_text(
                "[workspace]\n"
                f'members = ["crates/{first_name}", "crates/{middle_name}"]\n'
                f'exclude = ["crates/{service_name}"]\n'
                'resolver = "2"\n\n'
                "[workspace.package]\n"
                f'license = "{workspace_license}"\n\n'
                "[workspace.dependencies]\n"
                f"{workspace_dependencies}",
                encoding="utf-8",
            )
            extras = (apache_manifest, middle_manifest, service_manifest)
            for package, license_declaration, extra in zip(
                package_names, license_declarations, extras, strict=True
            ):
                (root / "crates" / package / "Cargo.toml").write_text(
                    "[package]\n"
                    f'name = "{package}"\n'
                    'version = "0.1.0"\n'
                    'edition = "2021"\n'
                    f"{license_declaration}\n\n"
                    f"{extra}",
                    encoding="utf-8",
                )
            (root / "vendor" / "ghost" / "Cargo.toml").write_text(
                "[package]\n"
                'name = "ghost"\n'
                'version = "0.1.0"\n'
                'edition = "2021"\n'
                'license = "Apache-2.0"\n',
                encoding="utf-8",
            )
            if policy_text is None:
                policy_text = (
                    "[license_gate]\n"
                    f"mit_packages = {list(mit_packages)!r}\n"
                    f"agpl_packages = {list(agpl_packages)!r}\n"
                    f"apache_packages = {list(apache_packages)!r}\n"
                    f"permissive_roots = {list(permissive_roots)!r}\n"
                    f"agpl_aggregators = {list(agpl_aggregators)!r}\n"
                )
            (root / ".github" / "license-boundary.toml").write_text(
                policy_text, encoding="utf-8"
            )
            return subprocess.run(
                [sys.executable, str(CHECKER), "--root", str(root)],
                text=True,
                capture_output=True,
                check=False,
            )

    def assert_boundary_failure(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("permissive-root-to-AGPL dependency", result.stderr)

    def test_clean_graph_passes(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dependencies]\nmiddle = { path = "../middle" }\n'
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_direct_apache_root_dependency_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dependencies]\nservice = { path = "../service" }\n'
        )
        self.assert_boundary_failure(result)

    def test_transitive_apache_root_dependency_fails_with_complete_path(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dependencies]\nmiddle = { path = "../middle" }\n',
            middle_manifest='[dependencies]\nservice = { path = "../service" }\n',
        )
        self.assert_boundary_failure(result)
        self.assertIn("apache -> middle -> service", result.stderr)

    def test_transitive_mit_root_dependency_fails_with_complete_path(self) -> None:
        result = self.run_fixture(
            package_names=("mit-root", "middle", "service"),
            license_declarations=(
                'license = "MIT"',
                'license = "Apache-2.0"',
                'license = "AGPL-3.0-only"',
            ),
            mit_packages=("mit-root",),
            apache_packages=("middle",),
            permissive_roots=("mit-root",),
            apache_manifest='[dependencies]\nmiddle = { path = "../middle" }\n',
            middle_manifest='[dependencies]\nservice = { path = "../service" }\n',
        )
        self.assert_boundary_failure(result)
        self.assertIn("mit-root -> middle -> service", result.stderr)

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

    def test_each_owner_named_package_in_wrong_class_fails(self) -> None:
        owner_licenses = {
            "bitsandbytes-sys": "MIT",
            "cas-serve": "MIT",
            "git-xet-filter": "MIT",
            "hyprstream-metrics": "AGPL-3.0-only",
            "hyprstream-flight": "AGPL-3.0-only",
            "hyprstream-vfs-server": "AGPL-3.0-only",
            "hyprstream-workers": "AGPL-3.0-only",
        }
        for package, license_name in owner_licenses.items():
            with self.subTest(package=package):
                result = self.run_fixture(
                    package_names=(package, "middle", "service"),
                    license_declarations=(
                        f'license = "{license_name}"',
                        'license = "Apache-2.0"',
                        'license = "AGPL-3.0-only"',
                    ),
                    mit_packages=(),
                    agpl_packages=("service",),
                    apache_packages=(package, "middle"),
                    permissive_roots=(),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("resolved manifest license", result.stderr)

    def test_former_provisional_agpl_package_is_now_apache(self) -> None:
        result = self.run_fixture(
            package_names=("hyprstream-k8s-pds", "middle", "service"),
            apache_packages=("hyprstream-k8s-pds", "middle"),
            permissive_roots=(),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_former_provisional_agpl_package_in_old_class_fails(self) -> None:
        result = self.run_fixture(
            package_names=("hyprstream-k8s-pds", "middle", "service"),
            apache_packages=("middle",),
            agpl_packages=("hyprstream-k8s-pds", "service"),
            permissive_roots=(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hyprstream-k8s-pds", result.stderr)
        self.assertIn("resolved manifest license", result.stderr)

    def test_blank_license_fails_closed(self) -> None:
        result = self.run_fixture(
            license_declarations=(
                'license = ""',
                'license = "Apache-2.0"',
                'license = "AGPL-3.0-only"',
            )
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package license is blank", result.stderr)

    def test_missing_license_fails_closed(self) -> None:
        result = self.run_fixture(
            license_declarations=(
                "",
                'license = "Apache-2.0"',
                'license = "AGPL-3.0-only"',
            )
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must declare one exact license", result.stderr)

    def test_wrong_workspace_inherited_license_fails(self) -> None:
        result = self.run_fixture(
            workspace_license="MIT",
            license_declarations=(
                "license.workspace = true",
                'license = "Apache-2.0"',
                'license = "AGPL-3.0-only"',
            ),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("resolved manifest license is 'MIT'", result.stderr)

    def test_exact_workspace_inherited_license_passes(self) -> None:
        result = self.run_fixture(
            license_declarations=(
                "license.workspace = true",
                'license = "Apache-2.0"',
                'license = "AGPL-3.0-only"',
            )
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_compound_license_expression_fails(self) -> None:
        result = self.run_fixture(
            license_declarations=(
                'license = "MIT OR Apache-2.0"',
                'license = "Apache-2.0"',
                'license = "AGPL-3.0-only"',
            )
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("'MIT OR Apache-2.0'", result.stderr)

    def test_duplicate_classification_fails(self) -> None:
        result = self.run_fixture(apache_packages=("apache", "apache", "middle"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate package in apache_packages", result.stderr)

    def test_unclassified_package_fails(self) -> None:
        result = self.run_fixture(apache_packages=("middle",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("classification omission: ['apache']", result.stderr)

    def test_overlapping_classification_fails(self) -> None:
        result = self.run_fixture(mit_packages=("apache",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("classification overlap: ['apache']", result.stderr)

    def test_unknown_package_fails(self) -> None:
        result = self.run_fixture(apache_packages=("apache", "middle", "ghost"))
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

    def test_malformed_policy_fails_closed(self) -> None:
        result = self.run_fixture(
            policy_text="[license_gate]\napache_packages = [",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TOMLDecodeError", result.stderr)

    def test_deleting_or_renaming_any_license_class_fails(self) -> None:
        classes = {
            "mit_packages": "[]",
            "agpl_packages": "['service']",
            "apache_packages": "['apache', 'middle']",
        }
        for missing_key in classes:
            for replacement in (None, f"{missing_key}_renamed"):
                with self.subTest(missing_key=missing_key, replacement=replacement):
                    lines = ["[license_gate]"]
                    for key, value in classes.items():
                        if key == missing_key:
                            if replacement is not None:
                                lines.append(f"{replacement} = {value}")
                        else:
                            lines.append(f"{key} = {value}")
                    lines.extend(
                        [
                            "permissive_roots = ['apache']",
                            "agpl_aggregators = []",
                        ]
                    )
                    result = self.run_fixture(policy_text="\n".join(lines) + "\n")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        f"missing classification '{missing_key}'", result.stderr
                    )

    def test_excluded_guest_package_is_discovered_and_checked(self) -> None:
        result = self.run_fixture(
            package_names=("apache", "middle", "hyprstream-workers-python-guest"),
            license_declarations=(
                'license = "Apache-2.0"',
                'license = "Apache-2.0"',
                'license = "Apache-2.0"',
            ),
            agpl_packages=(),
            apache_packages=(
                "apache",
                "middle",
                "hyprstream-workers-python-guest",
            ),
            permissive_roots=("apache", "hyprstream-workers-python-guest"),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("3 package licenses match policy", result.stdout)

    def test_apache_application_agpl_aggregation_is_explicit_and_allowed(self) -> None:
        result = self.run_fixture(
            package_names=("hyprstream", "middle", "hyprstream-flight"),
            apache_packages=("hyprstream", "middle"),
            agpl_packages=("hyprstream-flight",),
            permissive_roots=("middle",),
            agpl_aggregators=("hyprstream",),
            apache_manifest=(
                "[dependencies]\n"
                'hyprstream-flight = { path = "../hyprstream-flight" }\n'
            ),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 declared AGPL aggregator", result.stdout)

    def test_undeclared_agpl_aggregation_fails(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dependencies]\nservice = { path = "../service" }\n',
            permissive_roots=(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("undeclared AGPL aggregation path", result.stderr)

    def test_aggregator_cannot_be_a_permissive_root(self) -> None:
        result = self.run_fixture(
            apache_manifest='[dependencies]\nservice = { path = "../service" }\n',
            permissive_roots=("apache",),
            agpl_aggregators=("apache",),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("permissive root/AGPL aggregator overlap", result.stderr)

    def test_stale_agpl_aggregation_obligation_fails(self) -> None:
        result = self.run_fixture(
            permissive_roots=(),
            agpl_aggregators=("apache",),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not reach an AGPL package", result.stderr)


if __name__ == "__main__":
    unittest.main()
