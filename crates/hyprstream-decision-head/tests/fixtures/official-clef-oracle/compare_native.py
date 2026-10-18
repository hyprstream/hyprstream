#!/usr/bin/env python3
"""Compare native exported tensors with pinned official offline oracle outputs.
Explicit tolerances are REQUIRED. Does not run any model or loosen a gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors.torch import load_file
import torch

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
    p.add_argument("--atol", type=float)
    p.add_argument("--rtol", type=float)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--probability-atol", type=float, help="separate pre-agreed absolute bound for probability tensors")
    p.add_argument("--backbone-only", action="store_true", help="only original-FP32 hidden and original embedding row comparison; no head claim")
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
    provenance = json.loads((a.reference / "provenance.json").read_text())
    for name, checksum in provenance["files"].items():
        if digest(a.reference / name) != checksum:
            raise SystemExit(f"reference file hash mismatch: {name}")
    if a.backbone_only and (provenance["mode"] != "full" or provenance["compute_dtype"] != "float32" or provenance["device"] != "cpu"):
        raise SystemExit("original-Float32 CPU backbone gate requires a full Float32 CPU official oracle")
    expected = load_file(a.reference / "tensors.safetensors")
    actual = load_file(a.native_tensors)
    if a.backbone_only:
        required = [k for k in expected if k.endswith(".hidden")] + ["embedding_rows", "embedding_token_ids"]
    else:
        required = [k for k in expected if k.endswith((".logits", ".probabilities", ".normalized_hidden"))]
        if a.include_hidden:
            required += [k for k in expected if k.endswith(".hidden")]
            required += ["embedding_rows", "embedding_token_ids"]
    entries = {}
    passed = True
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
        x, y = x.double(), y.double()
        difference = (x - y).abs()
        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        if key == "embedding_rows":
            used_atol, used_rtol = 0.0, 0.0  # unchanged original rows are exact, not a compute approximation
        elif tolerance_profile:
            category = "probabilities" if key.endswith(".probabilities") else "logits" if key.endswith(".logits") else "hidden"
            used_atol, used_rtol = tolerance_profile[category]["atol"], tolerance_profile[category]["rtol"]
        else:
            used_atol = a.probability_atol if key.endswith(".probabilities") and a.probability_atol is not None else a.atol
            used_rtol = a.rtol
        within = finite and bool((difference <= used_atol + used_rtol * x.abs()).all())
        entry = {"shape": list(x.shape), "absolute_tolerance": used_atol, "relative_tolerance": used_rtol, "max_absolute_index": [int(v) for v in torch.unravel_index(difference.argmax(), difference.shape)], "pass": within, "finite": finite, "max_absolute_error": float(difference.max()), "max_relative_error": float((difference / x.abs().clamp_min(1e-30)).max()), "rmse": float(difference.square().mean().sqrt())}
        if key.endswith(".probabilities"):
            entry["kl_reference_native"] = float((x * (x.clamp_min(1e-30).log() - y.clamp_min(1e-30).log())).sum())
        entries[key] = entry
        passed = passed and within
    record_match = None
    if a.native_records:
        ref = json.loads((a.reference / "records.json").read_text())
        native = json.loads(a.native_records.read_text())
        # Native emitter must preserve every encoded token/span/ID and accounting
        # field, independent of irrelevant corpus-mode/provenance envelope fields.
        fields = ["id", "input_ids", "questions", "state_tokens_before_truncation", "state_tokens_admitted", "state_truncated"]
        record_match = len(ref["records"]) == len(native["records"]) and all(all(r[k] == n[k] for k in fields) for r, n in zip(ref["records"], native["records"]))
        passed = passed and record_match
    report = {"pass": passed, "scope": "backbone-only-original-f32" if a.backbone_only else provenance["mode"], "tolerance_profile": tolerance_profile, "tolerance_profile_sha256": digest(a.tolerance_file) if a.tolerance_file else None, "source_revision": provenance["source_revision"], "reference_provenance_sha256": digest(a.reference / "provenance.json"), "native_tensors_sha256": digest(a.native_tensors), "atol": a.atol, "rtol": a.rtol, "probability_atol": a.probability_atol, "exact_encoded_record_match": record_match, "tensors": entries, "limitation": "Head-only parity is not full backbone/FP8/calibration/CPU serving qualification."}
    a.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": passed, "tensor_count": len(entries), "exact_encoded_record_match": record_match, "report": str(a.report)}))
    raise SystemExit(0 if passed else 1)

if __name__ == "__main__":
    main()
