"""Exercise actual CI commands without Cargo, including default-set mutations."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/synthesis-ci.sh"


def assert_selection(commands):
    """Reject commands that could silently fall back to default-members."""
    assert len(commands) == 3, commands
    lint, nextest, docs = commands
    for command in commands:
        assert "--locked" in command, command
        package = command.index("-p")
        assert command[package + 1] == "hyprstream-synthesis", command
    assert lint[0] == "clippy" and "--all-targets" in lint and "--release" in lint, lint
    assert lint[-3:] == ["--", "-D", "warnings"], lint
    assert nextest[:2] == ["nextest", "run"], nextest
    assert "--all-targets" in nextest, nextest
    assert nextest[nextest.index("--cargo-profile") + 1] == "release", nextest
    assert nextest[nextest.index("--profile") + 1] == "ci", nextest
    assert docs[0] == "test" and "--doc" in docs, docs
    assert "--release" in docs and "--profile" not in docs, docs


class SynthesisCi(unittest.TestCase):
    def capture(self, source=None, modes=("lint", "test")):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "commands.jsonl"
            cargo = root / "cargo"
            cargo.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,sys\n"
                "with open(os.environ['SYNTHESIS_TEST_LOG'],'a') as f: "
                "f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            )
            cargo.chmod(0o755)
            script = root / "lane.sh"
            script.write_text(SCRIPT.read_text() if source is None else source)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                       SYNTHESIS_TEST_LOG=str(log))
            for mode in modes:
                result = subprocess.run(["bash", str(script), mode], env=env,
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
            return [json.loads(line) for line in log.read_text().splitlines()]

    def test_actual_lane_selects_library_binary_integration_and_docs(self):
        assert_selection(self.capture())

    def test_root_manifest_fallback_cannot_omit_program(self):
        # Cargo.toml/lock changes make affected-set choose default-members;
        # these real lane commands must still name the excluded program.
        commands = self.capture()
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        default_members = {path.rsplit("/", 1)[-1]
                           for path in manifest["workspace"]["default-members"]}
        self.assertNotIn("hyprstream-synthesis", default_members)
        for command in commands:
            selected = ({command[command.index("-p") + 1]}
                        if "-p" in command else default_members)
            self.assertIn("hyprstream-synthesis", selected)
        mutated = SCRIPT.read_text().replace("-p hyprstream-synthesis", "")
        with self.assertRaises((AssertionError, ValueError)):
            assert_selection(self.capture(mutated))

    def test_wrong_profile_or_missing_targets_or_docs_is_rejected(self):
        for source in [
            SCRIPT.read_text().replace("--cargo-profile release", "--cargo-profile ci-test"),
            SCRIPT.read_text().replace("--cargo-profile release", "--cargo-profile dev"),
            SCRIPT.read_text().replace("cargo clippy --locked --release", "cargo clippy --locked"),
            SCRIPT.read_text().replace("cargo test --locked --release", "cargo test --locked"),
            SCRIPT.read_text().replace("--all-targets", ""),
            SCRIPT.read_text().replace("--doc", ""),
        ]:
            with self.subTest(source=source), self.assertRaises(AssertionError):
                assert_selection(self.capture(source))

    def test_live_blocking_workflow_and_required_test_script_call_lane(self):
        workflow = (ROOT / ".github/workflows/rust.yml").read_text()
        clippy = workflow.split("\n  clippy:", 1)[1].split("\n  deny:", 1)[0]
        self.assertRegex(clippy, r"(?m)^            bash \.github/scripts/synthesis-ci\.sh lint$")
        self.assertIn('python3 -m unittest discover -s scripts/tests -p "test_synthesis_ci.py"', workflow)
        required = (ROOT / ".github/scripts/graviton-build-test.sh").read_text()
        self.assertRegex(required, r'(?m)^run_phase "synthesis tests and doctests" bash \.github/scripts/synthesis-ci\.sh test$')
        # Existing five-package selection remains independent and intact.
        array = re.search(r"decision_packages=\((.*?)\n\)", required, re.S).group(1)
        self.assertEqual(set(re.findall(r"-p (\S+)", array)), {
            "hyprstream-decision", "hyprstream-calibration", "hyprstream-decision-head",
            "hyprstream-decision-stub", "hyprstream-eval",
        })


if __name__ == "__main__":
    unittest.main()
