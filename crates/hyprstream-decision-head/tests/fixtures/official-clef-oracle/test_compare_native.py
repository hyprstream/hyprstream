"""Tiny comparison/admission regressions: stdlib only, no Torch or model."""
import copy
import importlib.util
import json
import contextlib
import hashlib
import io
import math
import types
from unittest import mock
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
    def test_tensor_inventory_is_derived_from_admitted_records_and_scope(self):
        records = fixture()
        required, permitted = compare.expected_tensor_inventory(records, require_backbone_inputs=False)
        self.assertIn("r0.q0.logits", required)
        self.assertIn("r0.q0.probabilities", required)
        self.assertIn("r0.normalized_hidden", required)
        self.assertNotIn("r0.hidden", required)
        self.assertNotIn("embedding_rows", required)
        self.assertIn("r0.hidden", permitted)
        self.assertIn("embedding_rows", permitted)
        compare.validate_tensor_inventory(records, required, require_backbone_inputs=False)
        with self.assertRaisesRegex(ValueError, "missing=.*r0.q1.probabilities"):
            compare.validate_tensor_inventory(records, required - {"r0.q1.probabilities"},
                                              require_backbone_inputs=False)
        with self.assertRaisesRegex(ValueError, "missing=.*embedding_rows"):
            compare.validate_tensor_inventory(records, required, require_backbone_inputs=True)
        with self.assertRaisesRegex(ValueError, "unexpected=.*untracked.tensor"):
            compare.validate_tensor_inventory(records, required | {"untracked.tensor"},
                                              require_backbone_inputs=False)
        backbone_keys = {"r0.hidden", "embedding_rows", "embedding_token_ids"}
        compare.validate_tensor_inventory(records, backbone_keys, require_backbone_inputs=True,
                                          backbone_only=True)

    def test_native_receipt_binds_profile_input_and_output_tensor_bytes(self):
        profile = json.loads((HERE / "full-port-tolerances-v1.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, native_path, receipt_path = (root / "input.safetensors",
                                                       root / "native.safetensors",
                                                       root / "export-receipt.json")
            input_path.write_bytes(b"native backbone input")
            native_path.write_bytes(b"native emitted tensors")
            receipt = {"schema_version": 1, "input_profile": "original-f32",
                       "native_input_tensor_path": str(input_path),
                       "native_input_tensors_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                       "native_tensors_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                       "source_revision": profile["source_revision"],
                       "compute_dtype": "float32", "device": "cpu"}
            receipt_path.write_text(json.dumps(receipt))
            admitted = compare.admit_native_receipt(receipt_path, native_path, input_path, profile)
            self.assertEqual(admitted["input_profile"], "original-f32")
            for field, value, message in (
                ("input_profile", "fp8-w8a8", "input_profile does not match"),
                ("native_tensors_sha256", "0" * 64, "tensor digest does not match"),
                ("native_input_tensors_sha256", "f" * 64, "input digest does not match"),
                ("source_revision", "0" * 40, "source_revision does not match"),
                ("compute_dtype", "bfloat16", "dtype/device do not match"),
                ("device", "cuda", "dtype/device do not match"),
            ):
                with self.subTest(field=field):
                    receipt_path.write_text(json.dumps(dict(receipt, **{field: value})))
                    with self.assertRaisesRegex(ValueError, message):
                        compare.admit_native_receipt(receipt_path, native_path, input_path, profile)
            receipt_path.unlink()
            with self.assertRaisesRegex(ValueError, "requires --native-receipt"):
                compare.admit_native_receipt(None, native_path, input_path, profile)

    def test_full_port_requires_native_records_with_any_tolerance_source(self):
        for include_hidden in (False, True):
            with self.subTest(include_hidden=include_hidden), self.assertRaisesRegex(ValueError, "requires --native-records"):
                compare.admit_comparison_mode("full", backbone_only=False, head_only=False,
                                               include_hidden=include_hidden, has_native_records=False)
        compare.admit_comparison_mode("full", backbone_only=False, head_only=False,
                                     include_hidden=True, has_native_records=True)

    def test_full_port_requires_hidden_even_with_records_present(self):
        with self.assertRaisesRegex(ValueError, "requires --include-hidden"):
            compare.admit_comparison_mode("full", backbone_only=False, head_only=False,
                                          include_hidden=False, has_native_records=True)
        compare.admit_comparison_mode("full", backbone_only=False, head_only=False,
                                     include_hidden=True, has_native_records=True)

    def test_cli_full_profile_without_hidden_fails_before_tensor_or_record_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "provenance.json").write_text(json.dumps({"mode": "full"}))
            result = subprocess.run([sys.executable, "-S", str(HERE / "compare_native.py"),
                "--reference", str(root), "--native-tensors", str(root / "unused"),
                "--native-records", str(root / "missing-records.json"),
                "--tolerance-file", str(HERE / "full-port-tolerances-v1.json"),
                "--report", str(root / "report.json")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --include-hidden", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse((root / "report.json").exists())

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
    def test_actual_committed_numeric_records_cover_all_three_types(self):
        records = json.loads((HERE / "records.json").read_text())
        seen = set()
        for ri, record in enumerate(records["records"]):
            for qi, question in enumerate(record["questions"]):
                tag = question["question_type"]
                self.assertIs(type(tag), int)
                expected_kind = ("noul", "choice", "score")[tag]
                ids = question["option_ids"]
                values = [float(i) for i in range(len(ids))]
                result = compare.question_winners(f"r{ri}.q{qi}.logits", values, values, records)
                self.assertEqual(result["question_type"], expected_kind)
                self.assertEqual(result["native_winner"]["option_id"], ids[-1])
                self.assertFalse(result["disagreement"])
                seen.add(tag)
        self.assertEqual(seen, {0, 1, 2})

    def test_numeric_tags_match_string_kind_and_reject_bool_float_or_unknown(self):
        records = fixture()
        for qi, tag in enumerate((0, 1, 2)):
            values = [float(i) for i in range(len(records["records"][0]["questions"][qi]["option_ids"]))]
            expected = compare.question_winners(f"r0.q{qi}.probabilities", values, values, records)
            numeric = copy.deepcopy(records)
            numeric["records"][0]["questions"][qi]["question_type"] = tag
            self.assertEqual(compare.question_winners(f"r0.q{qi}.probabilities", values, values, numeric), expected)
        for invalid in (True, False, 0.0, 1.0, -1, 3, "0", None):
            bad = fixture()
            bad["records"][0]["questions"][0]["question_type"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "unknown question type"):
                compare.question_winners("r0.q0.logits", [1, 0], [1, 0], bad)

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


class SingleValueTensor:
    """Stdlib numerical loader fixture; no Torch/safetensors or model execution."""
    shape = (1,)
    dtype = "float32"
    def __init__(self, value): self.value = value
    def double(self): return self
    def all(self): return bool(self.value)
    def abs(self): return SingleValueTensor(abs(self.value))
    def __sub__(self, other): return SingleValueTensor(self.value - other.value)
    def __add__(self, other): return SingleValueTensor(self.value + other)
    __radd__ = __add__
    def __mul__(self, other): return SingleValueTensor(self.value * (other.value if isinstance(other, SingleValueTensor) else other))
    __rmul__ = __mul__
    def __truediv__(self, other): return SingleValueTensor(self.value / other.value)
    def __le__(self, other): return SingleValueTensor(self.value <= other.value)
    def max(self): return self.value
    def argmax(self): return 0
    def square(self): return SingleValueTensor(self.value ** 2)
    def mean(self): return self
    def sqrt(self): return SingleValueTensor(math.sqrt(self.value))
    def clamp_min(self, minimum): return SingleValueTensor(max(self.value, minimum))
    def log(self): return SingleValueTensor(math.log(self.value))
    def sum(self): return self.value
    def __float__(self): return float(self.value)
    def tolist(self): return [self.value]


class FiniteReportAndProfileTests(unittest.TestCase):
    def run_small_report(self, reference_value, native_value):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            tensor_path, native_path = reference / "tensors.safetensors", root / "native.safetensors"
            # Deliberately synthetic loader bytes, hashed by the REAL CLI report.
            tensor_path.write_bytes(b"stdlib-reference-tensor-fixture")
            native_path.write_bytes(b"stdlib-native-tensor-fixture")
            records = {"records": [{"id":"record", "input_ids":[1], "questions":[
                {"question_id":"one", "question_type":2, "option_ids":["0"]}],
                "state_tokens_before_truncation":0, "state_tokens_admitted":0, "state_truncated":False}]}
            (reference / "records.json").write_text(json.dumps(records))
            provenance = {"mode":"head", "source_revision":"source-fixture", "files":{
                name:hashlib.sha256((reference/name).read_bytes()).hexdigest()
                for name in ("records.json", "tensors.safetensors")}}
            (reference / "provenance.json").write_text(json.dumps(provenance))
            safe = types.ModuleType("safetensors")
            loader = types.ModuleType("safetensors.torch")
            values = {"r0.normalized_hidden": 0.0, "r0.q0.logits": 1.0,
                      "r0.q0.probabilities": reference_value}
            loader.load_file = lambda path: {
                key: SingleValueTensor((reference_value if path == tensor_path else native_value)
                                       if key == "r0.q0.probabilities" else value)
                for key, value in values.items()
            }
            torch = types.ModuleType("torch")
            torch.int64 = "int64"
            torch.isfinite = lambda x: SingleValueTensor(math.isfinite(x.value))
            torch.unravel_index = lambda index, shape: [index]
            report = root / "report.json"
            argv = ["compare_native.py", "--reference", str(reference), "--native-tensors", str(native_path),
                    "--atol", "0", "--rtol", "0", "--report", str(report), "--head-only"]
            with mock.patch.dict(sys.modules, {"safetensors":safe, "safetensors.torch":loader, "torch":torch}), \
                 mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exited: compare.main()
            data = json.loads(report.read_text(), parse_constant=lambda value: self.fail("nonstandard JSON " + value))
            self.assertEqual(data["native_tensors_sha256"], hashlib.sha256(native_path.read_bytes()).hexdigest())
            self.assertEqual(data["reference_provenance_sha256"], hashlib.sha256((reference / "provenance.json").read_bytes()).hexdigest())
            return exited.exception.code, data

    def test_nonfinite_required_tensor_writes_hashed_failed_json_not_exception(self):
        for left, right in ((1.0, float("nan")), (1.0, float("inf")),
                            (float("-inf"), 1.0), (float("nan"), 1.0)):
            with self.subTest(left=left, right=right):
                code, report = self.run_small_report(left, right)
                self.assertEqual(code, 1)
                self.assertFalse(report["pass"])
                entry = report["tensors"]["r0.q0.probabilities"]
                self.assertFalse(entry["pass"])
                self.assertFalse(entry["finite"])
                for field in ("max_absolute_index", "max_absolute_error", "max_relative_error", "rmse", "kl_reference_native"):
                    self.assertIsNone(entry[field])
                self.assertNotIn("argmax", entry)

    def test_finite_equal_tensor_still_writes_passing_numeric_metrics(self):
        code, report = self.run_small_report(1.0, 1.0)
        self.assertEqual(code, 0)
        self.assertTrue(report["pass"])
        entry = report["tensors"]["r0.q0.probabilities"]
        self.assertTrue(entry["finite"])
        self.assertEqual(entry["max_absolute_error"], 0.0)
        self.assertEqual(entry["kl_reference_native"], 0.0)

    def test_full_original_profile_rejects_bf16_and_gpu_before_tensor_import(self):
        for dtype, device in (("bfloat16", "cpu"), ("float32", "cuda"), ("float32", "cuda:0")):
            with self.subTest(dtype=dtype, device=device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "provenance.json").write_text(json.dumps({"mode":"full", "compute_dtype":dtype, "device":device, "files":{}}))
                result = subprocess.run([sys.executable, "-S", str(HERE / "compare_native.py"),
                    "--reference", str(root), "--native-tensors", str(root/"unused"),
                    "--native-records", str(root/"unread-records"), "--include-hidden",
                    "--tolerance-file", str(HERE/"full-port-tolerances-v1.json"), "--report", str(root/"report")],
                    capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires a full Float32 CPU official oracle", result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)
                self.assertFalse((root/"report").exists())

    def test_original_cpu_profile_positive_and_separate_head_admission(self):
        compare.admit_original_f32_cpu_profile({"mode":"full", "compute_dtype":"float32", "device":"cpu"},
                                              backbone_only=False, tolerance_profile={"profile":"original"})
        compare.admit_original_f32_cpu_profile({"mode":"head", "compute_dtype":"bfloat16", "device":"cuda"},
                                              backbone_only=False, tolerance_profile=None)


if __name__ == "__main__":
    unittest.main()
