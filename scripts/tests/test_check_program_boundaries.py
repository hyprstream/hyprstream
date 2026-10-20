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
            (root / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/*"]\n', encoding="utf-8")
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

    def test_inherited_aliases_all_dependency_tables_and_non_crates_member(self):
        for table in ("dependencies", "dev-dependencies", "build-dependencies",
                      "target.'cfg(unix)'.dependencies", "target.'cfg(unix)'.dev-dependencies",
                      "target.'cfg(unix)'.build-dependencies"):
            with self.subTest(table=table), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "Cargo.toml").write_text(
                    '[workspace]\nmembers = ["workspace-hack"]\n'
                    '[workspace.dependencies]\nsynth = { package = "hyprstream-synthesis", version = "0.1" }\n')
                member = root / "workspace-hack"
                member.mkdir()
                (member / "Cargo.toml").write_text(
                    '[package]\nname = "workspace-hack"\nversion = "0.1.0"\n'
                    f'[{table}]\nsynth.workspace = true\n')
                result = subprocess.run([sys.executable, str(CHECKER), "--root", str(root)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must not depend", result.stdout)
                self.assertIn("hyprstream-synthesis", result.stdout)

    def test_inherited_nonleaf_alias_and_missing_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["platform"]\n'
                '[workspace.dependencies]\nsynth = { package = "hyprstream-decision", version = "0.1" }\n')
            member = root / "platform"
            member.mkdir()
            path = member / "Cargo.toml"
            path.write_text('[package]\nname = "platform"\nversion = "0.1.0"\n[dependencies]\nsynth.workspace = true\n')
            run = lambda: subprocess.run([sys.executable, str(CHECKER), "--root", str(root)], capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            path.write_text(path.read_text().replace("synth.workspace", "missing.workspace"))
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unresolved inherited", result.stdout)
            (root / "Cargo.toml").write_text("[workspace\n")
            self.assertNotEqual(run().returncode, 0)

    def test_recursive_implicit_members_local_and_inherited_paths(self):
        # Neither helper nor leaf is listed in members. The inherited path
        # is root-relative, whereas package-local paths are manifest-relative.
        for inherited in (False, True):
            for table in ("dependencies", "dev-dependencies", "build-dependencies",
                          "target.'cfg(unix)'.dependencies"):
                with self.subTest(inherited=inherited, table=table), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    workspace = '[workspace]\nmembers = ["platform"]\n'
                    if inherited:
                        workspace += '[workspace.dependencies]\nhelper_alias = { package = "helper", path = "support/helper" }\n'
                    (root / "Cargo.toml").write_text(workspace)
                    for name, path in (("platform", "platform"), ("helper", "support/helper"),
                                       ("hyprstream-synthesis", "leaf")):
                        folder = root / path
                        folder.mkdir(parents=True)
                        (folder / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "0.1.0"\n')
                    with (root / "platform/Cargo.toml").open("a") as manifest:
                        manifest.write(f'[{table}]\n')
                        manifest.write('helper_alias.workspace = true\n' if inherited else
                                       'helper_alias = { package = "helper", path = "../support/helper" }\n')
                    helper = root / "support/helper/Cargo.toml"
                    helper.write_text(helper.read_text() + '[dependencies]\nsynth = { package = "hyprstream-synthesis", path = "../../leaf" }\n')
                    run = lambda: subprocess.run([sys.executable, str(CHECKER), "--root", str(root)], capture_output=True, text=True)
                    bad = run()
                    self.assertNotEqual(bad.returncode, 0)
                    self.assertIn('support/helper/Cargo.toml: platform crate `helper`', bad.stdout)
                    # Actual dependency target identity is authoritative.
                    helper.write_text(helper.read_text().replace('package = "hyprstream-synthesis"', 'package = "decision"'))
                    (root / "leaf/Cargo.toml").write_text('[package]\nname = "decision"\nversion = "0.1.0"\n')
                    self.assertEqual(run().returncode, 0)

    def test_implicit_member_excluded_or_owned_by_nested_workspace(self):
        for boundary in ("exclude", "nested", "nested-parent", "explicit-owner"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workspace = '[workspace]\nmembers = ["platform"]\n'
                if boundary == "exclude":
                    workspace += 'exclude = ["support/helper"]\n'
                (root / "Cargo.toml").write_text(workspace)
                for name, path in (("platform", "platform"), ("helper", "support/helper")):
                    folder = root / path
                    folder.mkdir(parents=True)
                    (folder / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "0.1.0"\n')
                with (root / "platform/Cargo.toml").open("a") as manifest:
                    manifest.write('[dependencies]\nhelper = { path = "../support/helper" }\n')
                helper = root / "support/helper/Cargo.toml"
                if boundary == "explicit-owner":
                    helper.write_text(helper.read_text() + 'workspace = ".."\n')
                helper.write_text(helper.read_text() + '[dependencies]\nsynth = { package = "hyprstream-synthesis", version = "0.1" }\n')
                if boundary == "nested":
                    helper.write_text(helper.read_text() + '[workspace]\n')
                elif boundary in ("nested-parent", "explicit-owner"):
                    (root / "support/Cargo.toml").write_text('[workspace]\nmembers = ["helper"]\n')
                result = subprocess.run([sys.executable, str(CHECKER), "--root", str(root)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_explicit_root_owner_and_path_cycle_are_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text('[workspace]\nmembers = ["platform"]\n')
            for name in ("platform", "helper"):
                (root / name).mkdir()
                (root / name / "Cargo.toml").write_text(f'[package]\nname = "{name}"\nversion = "0.1.0"\nworkspace = ".."\n')
            with (root / "platform/Cargo.toml").open("a") as manifest:
                manifest.write('[dependencies]\nhelper = { path = "../helper" }\n')
            with (root / "helper/Cargo.toml").open("a") as manifest:
                manifest.write('[dependencies]\nplatform = { path = "../platform" }\nsynth = { package = "hyprstream-synthesis", version = "0.1" }\n')
            result = subprocess.run([sys.executable, str(CHECKER), "--root", str(root)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('platform crate `helper`', result.stdout)


if __name__ == "__main__":
    unittest.main()
