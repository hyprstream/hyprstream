"""Tiny comparison/admission regressions: stdlib only, no Torch or model."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("clef_compare_native", HERE / "compare_native.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def fixture():
    return {"records": [{"id": "record-z", "input_ids": [9, 2],
                         "questions": [
                             {"question_id": "z-noul", "question_type": "noul", "option_ids": ["true", "false"], "question_span": [0, 1], "option_spans": [[0, 1], [1, 2]]},
                             {"question_id": "a-choice", "question_type": "choice", "option_ids": ["z", "a", "m"], "question_span": [0, 1], "option_spans": [[0, 1]] * 3},
                             {"question_id": "score", "question_type": "score", "option_ids": ["0", "1", "2"], "question_span": [0, 1], "option_spans": [[0, 1]] * 3}],
                         "state_tokens_before_truncation": 3, "state_tokens_admitted": 1,
                         "state_truncated": True}]}


class ComparatorAdmissionTests(unittest.TestCase):
    def test_full_port_requires_native_records_with_any_tolerance_source(self):
        for include_hidden in (False, True):
            with self.subTest(include_hidden=include_hidden), self.assertRaisesRegex(ValueError, "requires --native-records"):
                compare.admit_comparison_mode("full", backbone_only=False, head_only=False,
                                               include_hidden=include_hidden, has_native_records=False)
        compare.admit_comparison_mode("full", backbone_only=False, head_only=False,
                                     include_hidden=True, has_native_records=True)

    def test_head_and_backbone_are_explicit_separate_scopes(self):
        compare.admit_comparison_mode("head", backbone_only=False, head_only=True,
                                     include_hidden=False, has_native_records=False)
        compare.admit_comparison_mode("full", backbone_only=True, head_only=False,
                                     include_hidden=True, has_native_records=False)
        for mode, hidden in (("full", False), ("head", True)):
            with self.subTest(mode=mode, hidden=hidden), self.assertRaisesRegex(ValueError, "head-only requires"):
                compare.admit_comparison_mode(mode, backbone_only=False, head_only=True,
                                               include_hidden=hidden, has_native_records=False)
        with self.assertRaisesRegex(ValueError, "backbone-only requires"):
            compare.admit_comparison_mode("head", backbone_only=True, head_only=False,
                                         include_hidden=False, has_native_records=False)

    def test_cli_missing_records_fails_before_reference_or_tensor_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, "-S", str(HERE / "compare_native.py"),
                                     "--reference", str(root / "missing-reference"),
                                     "--native-tensors", str(root / "missing.safetensors"),
                                     "--tolerance-file", str(HERE / "full-port-tolerances-v1.json"),
                                     "--report", str(root / "report.json")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --native-records", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse((root / "report.json").exists())

    def test_cli_full_profile_cannot_omit_records_even_with_head_only_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, "-S", str(HERE / "compare_native.py"),
                                     "--reference", str(root), "--head-only",
                                     "--native-tensors", str(root / "unused"),
                                     "--tolerance-file", str(HERE / "full-port-tolerances-v1.json"),
                                     "--report", str(root / "report.json")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("full-port tolerance profile requires --native-records", result.stderr)

    def test_cli_head_only_cannot_bypass_full_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "provenance.json").write_text(json.dumps({"mode": "full"}))
            result = subprocess.run([sys.executable, "-S", str(HERE / "compare_native.py"),
                                     "--reference", str(root), "--head-only", "--atol", "0", "--rtol", "0",
                                     "--native-tensors", str(root / "unused"), "--report", str(root / "report.json")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("head-only requires a head reference", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_exact_records_ignore_only_irrelevant_envelope(self):
        native = fixture()
        native["mode"] = "native"
        native["records"][0]["request"] = {"irrelevant": True}
        self.assertTrue(compare.encoded_records_match(fixture(), native))

    def test_each_encoding_or_label_change_fails_exact_match(self):
        changes = {
            "record id": lambda r: r["records"][0].update(id="other"),
            "tokens": lambda r: r["records"][0]["input_ids"].reverse(),
            "question id": lambda r: r["records"][0]["questions"][0].update(question_id="other"),
            "question order": lambda r: r["records"][0]["questions"].reverse(),
            "span": lambda r: r["records"][0]["questions"][0].update(question_span=[1, 2]),
            "option order": lambda r: r["records"][0]["questions"][0]["option_ids"].reverse(),
            "kind": lambda r: r["records"][0]["questions"][0].update(question_type="choice"),
            "before truncation": lambda r: r["records"][0].update(state_tokens_before_truncation=4),
            "admitted": lambda r: r["records"][0].update(state_tokens_admitted=2),
            "truncated": lambda r: r["records"][0].update(state_truncated=False),
            "missing field": lambda r: r["records"][0].pop("input_ids"),
            "missing record": lambda r: r["records"].clear(),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                native = fixture()
                change(native)
                self.assertFalse(compare.encoded_records_match(fixture(), native))


class WinnerReportingTests(unittest.TestCase):
    def test_near_tie_flip_is_reported_despite_numeric_tolerance(self):
        baseline, candidate = [0.50001, 0.49999], [0.49999, 0.50001]
        self.assertTrue(all(abs(x - y) <= 1e-4 + 1e-4 * abs(x) for x, y in zip(baseline, candidate)))
        result = compare.question_winners("r0.q0.probabilities", baseline, candidate, fixture())
        self.assertTrue(result["disagreement"])
        self.assertEqual(result["record_id"], "record-z")
        self.assertEqual(result["question_id"], "z-noul")
        self.assertEqual(result["canonical_to_model_permutation"], [1, 0])
        self.assertEqual(result["reference_winner"], {"model_index": 0, "option_id": "true", "canonical_index": 1})
        self.assertEqual(result["native_winner"], {"model_index": 1, "option_id": "false", "canonical_index": 0})

    def test_choice_score_and_exact_ties_map_to_canonical_ids(self):
        choice = compare.question_winners("r0.q1.logits", [1.0, 1.0, -1.0], [0.0, 2.0, -1.0], fixture())
        self.assertEqual(choice["reference_winner"]["option_id"], "z")  # first model index wins tie
        self.assertEqual(choice["native_winner"]["option_id"], "a")
        self.assertEqual(choice["canonical_to_model_permutation"], [1, 2, 0])
        score = compare.question_winners("r0.q2.probabilities", [0.2, 0.3, 0.5], [0.2, 0.3, 0.5], fixture())
        self.assertFalse(score["disagreement"])
        self.assertEqual(score["native_winner"]["canonical_index"], 2)

    def test_every_tensor_reported_and_questions_counted_once(self):
        comparisons = [compare.question_winners(f"r0.q0.{kind}", [0.6, 0.4], [0.4, 0.6], fixture())
                       for kind in ("logits", "probabilities")]
        comparisons.append(compare.question_winners("r0.q1.logits", [1, 2, 3], [1, 2, 3], fixture()))
        report = compare.summarize_winners(comparisons)
        self.assertEqual(report["tensor_comparison_count"], 3)
        self.assertEqual(report["tensor_disagreement_count"], 2)
        self.assertEqual(report["question_comparison_count"], 2)
        self.assertEqual(report["question_disagreement_count"], 1)
        self.assertEqual(set(report["questions"][0]["tensors"]), {"logits", "probabilities"})
        self.assertIn("report-only", report["policy"])

    def test_bad_domains_and_nonfinite_vectors_fail(self):
        for values in ([float("nan"), 0], [float("inf"), 0], [1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compare.question_winners("r0.q0.logits", values, [1, 0], fixture())
        invalid = copy.deepcopy(fixture())
        invalid["records"][0]["questions"][0]["option_ids"] = ["true", "true"]
        with self.assertRaisesRegex(ValueError, "invalid option domain"):
            compare.question_winners("r0.q0.logits", [1, 0], [1, 0], invalid)


if __name__ == "__main__":
    unittest.main()
