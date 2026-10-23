#!/usr/bin/env python3
"""Offline official Clef oracle; never used by the native serving application.
Default head mode loads only the original joint head and selected LM rows.
Full mode is opt-in and must be scheduled separately by the coordinator.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

REVISION = "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

class SelectedRows:
    """Equivalent indexing over selected actual original lm_head.weight rows."""
    def __init__(self, token_ids, rows):
        self.index = {int(token): i for i, token in enumerate(token_ids)}
        self.rows = rows
    def __getitem__(self, token_ids):
        indices = torch.tensor([self.index[int(t)] for t in token_ids.tolist()], device=self.rows.device)
        return self.rows[indices]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).with_name("corpus.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["head", "full"], default="head")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--allow-full-model", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output must be new or empty; preserve previous oracle evidence")
    if args.mode == "full" and not args.allow_full_model:
        raise SystemExit("full reference requires --allow-full-model and coordinator resource admission")
    if args.mode == "head" and args.device != "cpu":
        raise SystemExit("head fixture profile is CPU; GPU is reserved for admitted full oracle runs")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("requested GPU unavailable; refusing CPU fallback")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dtype = getattr(torch, args.dtype)
    artifact = json.loads(args.artifact_manifest.read_text())
    assert artifact["revision"] == REVISION
    entries = {f["path"]: f for f in artifact["files"]}
    assert len(entries) == 17 and all(f["verified"] for f in entries.values())
    # Rehash every small/source/head dependency here; previously verified full
    # backbone shard identities are attested by the original manifest. Full mode
    # must use a separately hash-verified destination if artifacts were moved.
    checked = {}
    for name in ["joint_schema_model.py", "joint_head_config.json", "joint_head.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "model.safetensors.index.json"]:
        path = args.model_dir / name
        assert path.stat().st_size == entries[name]["bytes"], name
        actual = sha256(path)
        assert actual == entries[name]["sha256"], name
        checked[name] = actual
    source = args.model_dir / "joint_schema_model.py"
    spec = importlib.util.spec_from_file_location("official_pinned_clef", source)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    corpus = json.loads(args.corpus.read_text())
    encoded = []
    fixture_records = []
    selected = set()
    for item in corpus["records"]:
        request = dict(item["request"], id=item["id"])
        record = official.encode_record(tokenizer, request, max_length=item["max_length"], max_state_tokens=item.get("max_state_tokens"))
        encoded.append(record)
        for q in record.questions:
            for a, b in q.option_spans:
                selected.update(record.input_ids[a:b])
        original_state = len(official._tokens(tokenizer, official.render(request["state"])))
        # Admission count is source state tokenization truncated by its two caps.
        # Derive admitted count from full-vs-capped encoder length instead of
        # implementing schema packing or reproducing the native encoder.
        no_state = official.encode_record(tokenizer, request, max_length=item["max_length"], max_state_tokens=0)
        admitted_state = len(record.input_ids) - len(no_state.input_ids)
        fixture_records.append({"id": item["id"], "request": request, "max_length": item["max_length"], "max_state_tokens": item.get("max_state_tokens"), "input_ids": list(record.input_ids), "questions": [{"question_id": q.question_id, "question_type": q.question_type, "question_span": list(q.question_span), "option_spans": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)} for q in record.questions], "state_tokens_before_truncation": original_state, "state_tokens_admitted": admitted_state, "state_truncated": admitted_state < original_state})
    selected = sorted(selected)
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    embedding_key = "lm_head.weight"
    shard = index["weight_map"][embedding_key]
    assert (args.model_dir / shard).stat().st_size == entries[shard]["bytes"]
    with safe_open(args.model_dir / shard, framework="pt", device="cpu") as f:
        view = f.get_slice(embedding_key)
        embedding_shape = view.get_shape()
        rows = torch.cat([view[token:token + 1] for token in selected]).to(dtype=dtype)
    head_config = json.loads((args.model_dir / "joint_head_config.json").read_text())
    tensors = {"embedding_token_ids": torch.tensor(selected, dtype=torch.int64), "embedding_rows": rows.contiguous()}
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hooks = []
    traces = {}
    if args.mode == "head":
        # Construct the UNMODIFIED official class on meta to avoid a second
        # floating head allocation; load original BF16 tensors and convert each
        # key in place, then strict assign to this same class.
        with torch.device("meta"):
            head = official.JointSchemaHead(**head_config)
        weights = load_file(args.model_dir / "joint_head.safetensors", device="cpu")
        for key in list(weights):
            weights[key] = weights[key].to(dtype=dtype)
        head.load_state_dict(weights, strict=True, assign=True)
        del weights
        head.eval()
        hooks.append(head.hidden_norm.register_forward_hook(lambda _m, _a, output: traces.update(normalized_hidden=output.detach().cpu()[0])))
        for i, record in enumerate(encoded):
            length = len(record.input_ids)
            # Public deterministic SYNTHETIC hidden states. Saved tensor is the
            # shared input; native comparison must NOT regenerate these values.
            hidden = (torch.arange(length * head_config["hidden_size"], dtype=torch.float32).reshape(length, head_config["hidden_size"]) * 0.001 + i * 0.125).sin().to(dtype=dtype)
            batch = official.collate_records([record], tokenizer.pad_token_id, torch.device("cpu"))
            with torch.inference_mode():
                logits = head(hidden.unsqueeze(0), batch["input_ids"], batch["attention_mask"], [record], SelectedRows(selected, rows))[0]
            tensors[f"r{i}.hidden"] = hidden.contiguous()
            tensors[f"r{i}.normalized_hidden"] = traces["normalized_hidden"].contiguous()
            for j, values in enumerate(logits):
                tensors[f"r{i}.q{j}.logits"] = values.detach().cpu().contiguous()
                tensors[f"r{i}.q{j}.probabilities"] = values.float().softmax(-1).detach().cpu().contiguous()
    else:
        # Actual backbone/head call is the UNMODIFIED official release loader
        # and ClefModel.forward. Hooks observe tensors, never replace operations.
        from transformers import AutoProcessor
        model, processor = official.load_release_model(args.model_dir, device=args.device, dtype=dtype, local_files_only=True)
        hooks.append(model.head.register_forward_pre_hook(lambda _m, inputs: traces.update(hidden=inputs[0].detach().cpu()[0])))
        hooks.append(model.head.hidden_norm.register_forward_hook(lambda _m, _a, output: traces.update(normalized_hidden=output.detach().cpu()[0])))
        for i, record in enumerate(encoded):
            batch = official.collate_records([record], processor.tokenizer.pad_token_id, torch.device(args.device))
            with torch.inference_mode():
                logits = model(batch)[0]
            tensors[f"r{i}.hidden"] = traces["hidden"].contiguous()
            tensors[f"r{i}.normalized_hidden"] = traces["normalized_hidden"].contiguous()
            for j, values in enumerate(logits):
                tensors[f"r{i}.q{j}.logits"] = values.detach().cpu().contiguous()
                tensors[f"r{i}.q{j}.probabilities"] = values.float().softmax(-1).detach().cpu().contiguous()
    for hook in hooks:
        hook.remove()
    save_file(tensors, args.output / "tensors.safetensors", metadata={"mode": args.mode, "source_revision": REVISION, "dtype": args.dtype})
    canonical_write(args.output / "records.json", {"corpus_version": corpus["corpus_version"], "mode": args.mode, "records": fixture_records})
    provenance = {"source_repo": "Cloudflare/clef-flash", "source_revision": REVISION, "mode": args.mode, "input_kind": "synthetic_hidden_original_embedding_rows" if args.mode == "head" else "original_bf16_backbone_hidden", "device": args.device, "compute_dtype": args.dtype, "source_head_dtype": "bfloat16", "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), "torch": torch.__version__, "transformers": __import__("transformers").__version__, "tokenizers": __import__("tokenizers").__version__, "safetensors": __import__("safetensors").__version__, "python": sys.version, "platform": platform.platform(), "hostname": platform.node(), "cuda_runtime": torch.version.cuda, "hip_runtime": torch.version.hip, "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None, "elapsed_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "artifact_manifest_sha256": sha256(args.artifact_manifest), "corpus_sha256": sha256(args.corpus), "generator_sha256": sha256(__file__), "rehashed_files": checked, "embedding_shard": shard, "embedding_shard_attested_sha256": entries[shard]["sha256"], "embedding_shape": embedding_shape, "selected_embedding_rows": len(selected), "files": {name: sha256(args.output / name) for name in ["records.json", "tensors.safetensors"]}, "argv": sys.argv, "limitation": "Head mode proves official-head math on saved synthetic hidden input, not backbone correctness/calibration/full-model CPU acceptance."}
    canonical_write(args.output / "provenance.json", provenance)
    print(json.dumps({"output": str(args.output), "records": len(encoded), "selected_rows": len(selected), "mode": args.mode, "peak_rss_kib": provenance["peak_rss_kib"], "elapsed_seconds": provenance["elapsed_seconds"]}))

if __name__ == "__main__":
    main()
