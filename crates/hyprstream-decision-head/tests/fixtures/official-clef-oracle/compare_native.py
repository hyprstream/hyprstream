#!/usr/bin/env python3
"""Compare native exported tensors with pinned official offline oracle outputs.
Explicit tolerances are REQUIRED. Does not run any model or loosen a gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ENCODED_FIELDS = ("id", "input_ids", "questions", "state_tokens_before_truncation", "state_tokens_admitted", "state_truncated")

def encoded_records_match(reference, native):
    """Compare every declared encoding/label field, without tensor imports."""
    left, right = reference["records"], native["records"]
    return len(left) == len(right) and all(
        all(field in r and field in n and r[field] == n[field] for field in ENCODED_FIELDS)
        for r, n in zip(left, right)
    )

def admit_comparison_mode(reference_mode, *, backbone_only, head_only, include_hidden, has_native_records):
    if reference_mode not in ("head", "full"):
        raise ValueError("reference mode must be head or full")
    if head_only and (reference_mode != "head" or include_hidden):
        raise ValueError("head-only requires a head reference and excludes --include-hidden")
    if backbone_only and reference_mode != "full":
        raise ValueError("backbone-only requires a full reference")
    if reference_mode == "full" and not backbone_only and not has_native_records:
        raise ValueError("full-port comparison requires --native-records for exact encoding and labels")
    if reference_mode == "full" and not backbone_only and not include_hidden:
        raise ValueError("full-port comparison requires --include-hidden for backbone and embedding tensors")

def admit_original_f32_cpu_profile(provenance, *, backbone_only, tolerance_profile):
    """The original-F32 CPU tolerance identity applies to full and backbone gates."""
    if backbone_only or tolerance_profile:
        if (provenance.get("mode") != "full" or provenance.get("compute_dtype") != "float32"
                or provenance.get("device") != "cpu"):
            raise ValueError("original-Float32 CPU gate requires a full Float32 CPU official oracle")

def expected_tensor_inventory(records, *, require_backbone_inputs, backbone_only=False):
    """Return required and permitted tensor names implied by admitted records."""
    items = records.get("records") if isinstance(records, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("oracle records must contain a non-empty records list")
    head_required, backbone_required, permitted = set(), set(), set()
    for record_index, record in enumerate(items):
        questions = record.get("questions") if isinstance(record, dict) else None
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"oracle record {record_index} must contain questions")
        normalized = f"r{record_index}.normalized_hidden"
        head_required.add(normalized)
        permitted.update((normalized, f"r{record_index}.hidden"))
        backbone_required.add(f"r{record_index}.hidden")
        for question_index, _question in enumerate(questions):
            logits = f"r{record_index}.q{question_index}.logits"
            probabilities = f"r{record_index}.q{question_index}.probabilities"
            head_required.update((logits, probabilities))
            permitted.update((logits, probabilities))
    permitted.update(("embedding_rows", "embedding_token_ids"))
    backbone_required.update(("embedding_rows", "embedding_token_ids"))
    required = backbone_required if backbone_only else head_required
    if require_backbone_inputs:
        required |= backbone_required
    return required, permitted

def validate_tensor_inventory(records, tensor_keys, *, require_backbone_inputs, backbone_only=False,
                              artifact="reference"):
    required, permitted = expected_tensor_inventory(records,
        require_backbone_inputs=require_backbone_inputs, backbone_only=backbone_only)
    actual = set(tensor_keys)
    missing, unexpected = sorted(required - actual), sorted(actual - permitted)
    if missing or unexpected:
        raise ValueError(f"{artifact} tensor inventory mismatch; missing={missing}, unexpected={unexpected}")
    return required

def admit_native_receipt(receipt_path, native_tensors_path, native_input_tensors_path, tolerance_profile):
    """Bind input artifact, declared profile, and exported tensor bytes together."""
    if tolerance_profile is not None and receipt_path is None:
        raise ValueError("predeclared full-port profile requires --native-receipt")
    if receipt_path is None:
        return None
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise ValueError("native export receipt must use schema_version 1")
    expected_profile = (tolerance_profile or {}).get("native_input_profile")
    if expected_profile is not None and receipt.get("input_profile") != expected_profile:
        raise ValueError("native export input_profile does not match tolerance profile")
    if receipt.get("native_tensors_sha256") != digest(native_tensors_path):
        raise ValueError("native export receipt tensor digest does not match --native-tensors")
    if tolerance_profile is not None and native_input_tensors_path is None:
        raise ValueError("predeclared full-port profile requires --native-input-tensors")
    input_digest = digest(native_input_tensors_path) if native_input_tensors_path else None
    if input_digest is not None and receipt.get("native_input_tensors_sha256") != input_digest:
        raise ValueError("native export receipt input digest does not match --native-input-tensors")
    if input_digest is not None and (not isinstance(receipt.get("native_input_tensor_path"), str)
                                     or not receipt["native_input_tensor_path"]):
        raise ValueError("native export receipt must record native_input_tensor_path")
    if tolerance_profile is not None:
        if not isinstance(expected_profile, str):
            raise ValueError("tolerance profile must declare native_input_profile")
        if receipt.get("source_revision") != tolerance_profile.get("source_revision"):
            raise ValueError("native export source_revision does not match tolerance profile")
        expected_dtype, expected_device = tolerance_profile.get("native_compute_dtype"), tolerance_profile.get("native_device")
        if not isinstance(expected_dtype, str) or not isinstance(expected_device, str):
            raise ValueError("tolerance profile must declare native_compute_dtype and native_device")
        if (receipt.get("compute_dtype") != expected_dtype or receipt.get("device") != expected_device):
            raise ValueError("native export compute dtype/device do not match tolerance profile")
    return receipt


def floating_comparison_entry(key, x, y, used_atol, used_rtol, torch):
    """Unavailable nonfinite metrics remain JSON null in a failed hashed report."""
    x, y = x.double(), y.double()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    entry = {"shape": list(x.shape), "absolute_tolerance": used_atol,
             "relative_tolerance": used_rtol, "max_absolute_index": None,
             "pass": False, "finite": finite, "max_absolute_error": None,
             "max_relative_error": None, "rmse": None}
    if key.endswith(".probabilities"):
        entry["kl_reference_native"] = None
    if not finite:
        return entry, x, y
    difference = (x - y).abs()
    entry["pass"] = bool((difference <= used_atol + used_rtol * x.abs()).all())
    entry["max_absolute_index"] = [int(v) for v in torch.unravel_index(difference.argmax(), difference.shape)]
    def metric(value):
        value = float(value)
        return value if math.isfinite(value) else None
    entry["max_absolute_error"] = metric(difference.max())
    entry["max_relative_error"] = metric((difference / x.abs().clamp_min(1e-30)).max())
    entry["rmse"] = metric(difference.square().mean().sqrt())
    if key.endswith(".probabilities"):
        entry["kl_reference_native"] = metric((x * (x.clamp_min(1e-30).log() - y.clamp_min(1e-30).log())).sum())
    return entry, x, y


def question_winners(key, reference, native, records):
    """Report first-index argmax in model and canonical label coordinates.

    Winner stability is reported, not silently added to the numerical gate.
    Inputs are finite vector lists from tensors that passed shape/dtype checks.
    """
    match = re.fullmatch(r"r(\d+)\.q(\d+)\.(logits|probabilities)", key)
    if match is None:
        raise ValueError(f"invalid question tensor key: {key}")
    record_index, question_index = int(match[1]), int(match[2])
    record = records["records"][record_index]
    question = record["questions"][question_index]
    ids = question["option_ids"]
    encoded_kind = question["question_type"]
    # Official records use exact integer tags; bool/float are not numeric tags.
    if type(encoded_kind) is int and encoded_kind in (0, 1, 2):
        kind = ("noul", "choice", "score")[encoded_kind]
    elif type(encoded_kind) is str and encoded_kind in ("noul", "choice", "score"):
        kind = encoded_kind
    else:
        raise ValueError(f"unknown question type: {encoded_kind}")
    if kind == "noul":
        canonical = ["false", "true"]
    elif kind == "choice":
        canonical = sorted(ids)
    elif kind == "score":
        canonical = [str(i) for i in range(len(ids))]
    else:
        raise ValueError(f"unknown question type: {kind}")
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(canonical):
        raise ValueError(f"invalid option domain: {key}")
    if len(reference) != len(ids) or len(native) != len(ids):
        raise ValueError(f"winner vector/domain cardinality mismatch: {key}")
    if not all(math.isfinite(v) for v in reference + native):
        raise ValueError(f"nonfinite winner vector: {key}")
    def winner(values):
        index = max(range(len(values)), key=values.__getitem__)
        return {"model_index": index, "option_id": ids[index], "canonical_index": canonical.index(ids[index])}
    left, right = winner(reference), winner(native)
    return {"record_index": record_index, "question_index": question_index,
            "record_id": record["id"], "question_id": question["question_id"],
            "question_type": kind, "model_option_ids": ids, "canonical_option_ids": canonical,
            "canonical_to_model_permutation": [ids.index(label) for label in canonical],
            "tensor_kind": match[3], "reference_winner": left, "native_winner": right,
            "disagreement": left["option_id"] != right["option_id"]}

def summarize_winners(comparisons):
    questions = {}
    for comparison in comparisons:
        coordinate = (comparison["record_index"], comparison["question_index"])
        if coordinate not in questions:
            questions[coordinate] = {key: value for key, value in comparison.items()
                                    if key not in ("tensor_kind", "reference_winner", "native_winner", "disagreement")}
            questions[coordinate].update(tensors={}, disagreement=False)
        question = questions[coordinate]
        question["tensors"][comparison["tensor_kind"]] = {
            key: comparison[key] for key in ("reference_winner", "native_winner", "disagreement")}
        question["disagreement"] |= comparison["disagreement"]
    return {"policy": "report-only; original full-port numerical tolerances unchanged",
            "tensor_comparison_count": len(comparisons),
            "tensor_disagreement_count": sum(c["disagreement"] for c in comparisons),
            "question_comparison_count": len(questions),
            "question_disagreement_count": sum(q["disagreement"] for q in questions.values()),
            "questions": [questions[key] for key in sorted(questions)]}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--native-tensors", required=True, type=Path)
    p.add_argument("--native-records", type=Path)
    p.add_argument("--native-receipt", type=Path, help="export receipt binding tensor bytes to input profile")
    p.add_argument("--native-input-tensors", type=Path, help="saved native backbone tensors named by export receipt")
    p.add_argument("--atol", type=float)
    p.add_argument("--rtol", type=float)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--probability-atol", type=float, help="separate pre-agreed absolute bound for probability tensors")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--backbone-only", action="store_true", help="only original-FP32 hidden and original embedding row comparison; no head claim")
    mode.add_argument("--head-only", action="store_true", help="explicit synthetic-head comparison; requires a head-mode reference and excludes hidden tensors")
    p.add_argument("--tolerance-file", type=Path, help="predeclared original-Float32 full-port tolerance profile")
    p.add_argument("--include-hidden", action="store_true", help="full-backbone comparison only; synthetic head inputs aren't a backbone oracle")
    a = p.parse_args()
    tolerance_profile = json.loads(a.tolerance_file.read_text()) if a.tolerance_file else None
    if tolerance_profile:
        if not tolerance_profile.get("declared_before_full_port_comparison"):
            raise SystemExit("tolerance profile must be declared before comparison")
    elif a.atol is None or a.rtol is None:
        raise SystemExit("explicit --atol/--rtol or predeclared --tolerance-file required")
    if (a.atol is not None and a.atol < 0) or (a.rtol is not None and a.rtol < 0) or (a.probability_atol is not None and a.probability_atol < 0):
        raise SystemExit("negative tolerance")
    if a.backbone_only and not tolerance_profile:
        raise SystemExit("backbone-only comparison requires predeclared --tolerance-file")
    if tolerance_profile and not a.backbone_only and not a.native_records:
        raise SystemExit("full-port tolerance profile requires --native-records")
    provenance = json.loads((a.reference / "provenance.json").read_text())
    try:
        admit_comparison_mode(provenance["mode"], backbone_only=a.backbone_only,
                              head_only=a.head_only, include_hidden=a.include_hidden,
                              has_native_records=a.native_records is not None)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for name, checksum in provenance["files"].items():
        if digest(a.reference / name) != checksum:
            raise SystemExit(f"reference file hash mismatch: {name}")
    try:
        admit_original_f32_cpu_profile(provenance, backbone_only=a.backbone_only,
                                       tolerance_profile=tolerance_profile)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    records = json.loads((a.reference / "records.json").read_text())
    try:
        native_receipt = admit_native_receipt(a.native_receipt, a.native_tensors,
                                              a.native_input_tensors, tolerance_profile)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    record_match = None
    if a.native_records:
        native_records = json.loads(a.native_records.read_text())
        record_match = encoded_records_match(records, native_records)
    # CLI/mode/record admission precedes even tensor-library imports. Tiny
    # regression tests can exercise the fail-closed boundary without Torch.
    from safetensors.torch import load_file
    import torch
    expected = load_file(a.reference / "tensors.safetensors")
    try:
        require_backbone_inputs = a.backbone_only or a.include_hidden
        validate_tensor_inventory(records, expected.keys(), require_backbone_inputs=require_backbone_inputs,
                                  backbone_only=a.backbone_only, artifact="reference")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    actual = load_file(a.native_tensors)
    try:
        validate_tensor_inventory(records, actual.keys(), require_backbone_inputs=require_backbone_inputs,
                                  backbone_only=a.backbone_only, artifact="native")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if a.backbone_only:
        required = [k for k in expected if k.endswith(".hidden")] + ["embedding_rows", "embedding_token_ids"]
    else:
        required = [k for k in expected if k.endswith((".logits", ".probabilities", ".normalized_hidden"))]
        if a.include_hidden:
            required += [k for k in expected if k.endswith(".hidden")]
            required += ["embedding_rows", "embedding_token_ids"]
    entries = {}
    passed = record_match is not False
    winners = []
    for key in sorted(required):
        if key not in actual:
            entries[key] = {"pass": False, "error": "missing native tensor"}
            passed = False
            continue
        x, y = expected[key], actual[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            entries[key] = {"pass": False, "error": "shape/dtype mismatch", "reference": list(x.shape), "native": list(y.shape), "reference_dtype": str(x.dtype), "native_dtype": str(y.dtype)}
            passed = False
            continue
        if x.dtype == torch.int64:
            same = torch.equal(x, y)
            entries[key] = {"pass": same, "exact_integer_match": same}
            passed = passed and same
            continue
        if key == "embedding_rows":
            used_atol, used_rtol = 0.0, 0.0  # unchanged original rows are exact, not a compute approximation
        elif tolerance_profile:
            category = "probabilities" if key.endswith(".probabilities") else "logits" if key.endswith(".logits") else "hidden"
            used_atol, used_rtol = tolerance_profile[category]["atol"], tolerance_profile[category]["rtol"]
        else:
            used_atol = a.probability_atol if key.endswith(".probabilities") and a.probability_atol is not None else a.atol
            used_rtol = a.rtol
        entry, x, y = floating_comparison_entry(key, x, y, used_atol, used_rtol, torch)
        if key.endswith((".logits", ".probabilities")) and entry["finite"]:
            winner = question_winners(key, x.tolist(), y.tolist(), records)
            entry["argmax"] = winner
            winners.append(winner)
        entries[key] = entry
        passed = passed and entry["pass"]
    report = {"pass": passed, "scope": "backbone-only-original-f32" if a.backbone_only else provenance["mode"], "tolerance_profile": tolerance_profile, "tolerance_profile_sha256": digest(a.tolerance_file) if a.tolerance_file else None, "source_revision": provenance["source_revision"], "reference_provenance_sha256": digest(a.reference / "provenance.json"), "native_tensors_sha256": digest(a.native_tensors), "native_receipt_sha256": digest(a.native_receipt) if a.native_receipt else None, "native_input_profile": native_receipt.get("input_profile") if native_receipt else None, "native_input_tensor_path": native_receipt.get("native_input_tensor_path") if native_receipt else None, "native_input_tensors_sha256": native_receipt.get("native_input_tensors_sha256") if native_receipt else None, "atol": a.atol, "rtol": a.rtol, "probability_atol": a.probability_atol, "exact_encoded_record_match": record_match, "tensors": entries, "limitation": "Head-only parity is not full backbone/FP8/calibration/CPU serving qualification."}
    report["argmax"] = summarize_winners(winners)
    a.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": passed, "tensor_count": len(entries), "exact_encoded_record_match": record_match, "report": str(a.report)}))
    raise SystemExit(0 if passed else 1)

if __name__ == "__main__":
    main()
