"""Public archive package reproduction, stdlib only; no model/tensor execution."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RECIPE = HERE / 'derive-native-head-finite-load.py'
SOURCE = runpy.run_path(str(RECIPE))
BUNDLE = SOURCE['BUNDLE']


class PublicArchivedEvidenceTests(unittest.TestCase):
    def test_all_nine_original_input_pins_retrievable(self):
        total = 0
        for group, root in [('EVIDENCE_PINS', BUNDLE / 'public-receipts'),
                            ('SOURCE_PINS', BUNDLE / 'source-snapshot')]:
            for relative, digest in SOURCE[group].items():
                path = root / relative
                self.assertFalse(path.is_symlink())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                total += path.stat().st_size
        self.assertEqual(len(SOURCE['EVIDENCE_PINS']) + len(SOURCE['SOURCE_PINS']), 9)
        self.assertLess(total, 1024 * 1024)

    def test_fresh_isolated_checkout_default_command_reproduces_exact_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Only committed fixture/recipe bytes: no fleet-coord path or .git.
            fixture = root / 'crates/hyprstream-decision-head/fixtures'
            fixture.mkdir(parents=True)
            shutil.copyfile(RECIPE, fixture / RECIPE.name)
            destination = root / BUNDLE.relative_to(REPO)
            shutil.copytree(BUNDLE, destination)
            receipt = root / 'reproduced.json'
            result = subprocess.run([sys.executable, '-S', str(fixture / RECIPE.name),
                                     '--output', str(receipt)], cwd=root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(receipt.read_bytes(), (HERE / 'native-head-finite-load-receipt.json').read_bytes())
            again = subprocess.run([sys.executable, '-S', str(fixture / RECIPE.name),
                                    '--output', str(receipt)], cwd=root,
                                   capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)  # create-only output preserved
            self.assertEqual(receipt.read_bytes(), (HERE / 'native-head-finite-load-receipt.json').read_bytes())

    def test_wrong_public_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'clef-production-v4-audit.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'public input digest mismatch'):
                SOURCE['derive'](root, BUNDLE / 'source-snapshot')

    def test_wrong_historical_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'crates/hyprstream-decision-head/src/clef/head.rs'
            path.parent.mkdir(parents=True)
            path.write_text('changed source snapshot')
            with self.assertRaisesRegex(ValueError, 'public input digest mismatch'):
                SOURCE['derive'](BUNDLE / 'public-receipts', root)


if __name__ == '__main__':
    unittest.main()
