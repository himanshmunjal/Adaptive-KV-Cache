"""LongBench evaluation for AdaptiveKVCache vs. the FP16 baseline.

Uses the official `THUDM/LongBench` dataset from the HF Hub and a
simplified token-level F1 metric (the same family of metric LongBench's own
QA tasks use; for publication-quality numbers, score outputs with the
official LongBench `metrics.py` instead -- this script focuses on being a
self-contained, dependency-light comparison harness).

Usage:
    python scripts/eval_longbench.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --task narrativeqa --n-samples 30
"""
import argparse
import csv
import json
import re
import string
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_kv import AdaptiveKVConfig  # noqa: E402
from adaptive_kv.generate import generate_with_adaptive_cache  # noqa: E402


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _longbench_parquet_files(task: str) -> list[str]:
    """Look up the parquet file path(s) for `task`, trying several locations
    in order. THUDM/LongBench is mid-migration off its old loading-script
    format: it was renamed to zai-org/LongBench, some tasks already have
    plain parquet files committed directly to `main`, others don't yet, and
    the auto-generated `refs/convert/parquet` mirror is missing entirely
    under the new name (404). We try, in order: the renamed repo's main
    branch, the original name's main branch, both repos' auto-convert ref,
    and finally a long-standing independent parquet mirror -- returning the
    first location that actually has files for this task.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError, RepositoryNotFoundError

    api = HfApi()
    candidates = [
        ("zai-org/LongBench", "main"),
        ("THUDM/LongBench", "main"),
        ("zai-org/LongBench", "refs/convert/parquet"),
        ("THUDM/LongBench", "refs/convert/parquet"),
        ("bzantium/LongBench", "refs/convert/parquet"),
    ]
    tried = []
    for repo_id, revision in candidates:
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        except (RevisionNotFoundError, RepositoryNotFoundError) as e:
            tried.append(f"{repo_id}@{revision}: {type(e).__name__}")
            continue
        task_files = [f for f in files if f.startswith(f"{task}/") and f.endswith(".parquet")
                      and "test" in f]
        if not task_files:
            # some layouts use "<task>_e" or no split subfolder; fall back to
            # any parquet file whose path contains the task name
            task_files = [f for f in files if task in f and f.endswith(".parquet")]
        if task_files:
            return [f"hf://datasets/{repo_id}@{revision}/{f}" for f in task_files]
        tried.append(f"{repo_id}@{revision}: no parquet files matching task '{task}'")

    raise ValueError(
        f"Could not find parquet files for task '{task}' in any known LongBench location.\n"
        + "\n".join(f"  - {t}" for t in tried)
        + "\nCheck https://huggingface.co/datasets/zai-org/LongBench/tree/main for the current "
          "file layout and task name, or pass a different --task."
    )


@torch.no_grad()
def generate_baseline(model, tok, prompt: str, max_new_tokens: int) -> str:
    from transformers import DynamicCache
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=7500).to(model.device)
    cache = DynamicCache()
    out_ids = model.generate(**inputs, past_key_values=cache, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="narrativeqa")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--importance-mode", default="key_diversity",
                     choices=["key_diversity", "attn", "attn_value"])
    ap.add_argument("--output-dir", default="results",
                     help="Directory to write per-run JSON + CSV results into (default: results/)")
    ap.add_argument("--run-name", default=None,
                     help="Basename for the output files (default: derived from model + timestamp)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    attn_impl = "eager" if args.importance_mode != "key_diversity" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                  attn_implementation=attn_impl,
                                                  device_map="auto")
    model.eval()
    print(f"Model loaded on device: {model.device}")
    cfg = AdaptiveKVConfig(importance_mode=args.importance_mode)

    ds = load_dataset(
        # THUDM/LongBench still ships a legacy Python loading script
        # (LongBench.py); `datasets>=4.5` refuses to execute loading scripts
        # at all (RuntimeError: "Dataset scripts are no longer supported"),
        # so `load_dataset("THUDM/LongBench", args.task, split="test")` fails
        # regardless of dataset version pinning here. Every HF dataset repo
        # gets an auto-generated parquet mirror under the `refs/convert/parquet`
        # ref, so we load from that -- but THUDM's exact file layout under
        # that ref isn't consistent across tasks (some are
        # "<task>/test/long_bench-test.parquet", others differ), so instead
        # of hardcoding a path pattern we ask the Hub what actually exists.
        "parquet",
        data_files=_longbench_parquet_files(args.task),
        split="train",
    )
    n = min(args.n_samples, len(ds))

    base_f1s, adapt_f1s, ratios = [], [], []
    per_sample = []
    for i in range(n):
        ex = ds[i]
        prompt = ex["context"] + "\n\nQuestion: " + ex["input"] + "\nAnswer:"
        golds = ex["answers"]

        base_out = generate_baseline(model, tok, prompt, args.max_new_tokens)
        _, stats = generate_with_adaptive_cache(
            model, tok, prompt, max_new_tokens=args.max_new_tokens,
            cache_config=cfg, return_stats=True,
            max_length=7500,
        )
        adapt_out = stats["generated_text"]

        base_f1 = max(f1_score(base_out, g) for g in golds)
        adapt_f1 = max(f1_score(adapt_out, g) for g in golds)
        base_f1s.append(base_f1)
        adapt_f1s.append(adapt_f1)
        ratios.append(stats["compression_ratio"])
        print(f"[{i + 1}/{n}] base_f1={base_f1:.3f} adaptive_f1={adapt_f1:.3f} "
              f"compression={stats['compression_ratio']:.3f}")
        per_sample.append({
            "sample": i, "base_f1": base_f1, "adaptive_f1": adapt_f1,
            "compression_ratio": stats["compression_ratio"],
        })

    mean_base_f1 = sum(base_f1s) / n
    mean_adapt_f1 = sum(adapt_f1s) / n
    mean_ratio = sum(ratios) / n
    print("\n=== Summary ===")
    print(f"Task: {args.task}  (n={n})")
    print(f"Baseline F1:  {mean_base_f1:.4f}")
    print(f"Adaptive F1:  {mean_adapt_f1:.4f}")
    print(f"Mean compression ratio: {mean_ratio:.3f} "
          f"({(1 - mean_ratio) * 100:.1f}% memory reduction)")

    # ---------------------------------------------------------------- #
    # Persist results
    # ---------------------------------------------------------------- #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.model.replace('/', '_')}_longbench_{args.task}_{timestamp}"

    payload = {
        "run_name": run_name,
        "timestamp_utc": timestamp,
        "args": vars(args),
        "per_sample": per_sample,
        "summary": {
            "baseline_f1": mean_base_f1,
            "adaptive_f1": mean_adapt_f1,
            "mean_compression_ratio": mean_ratio,
            "memory_reduction_pct": (1 - mean_ratio) * 100,
        },
    }

    json_path = out_dir / f"{run_name}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_dir / f"{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "base_f1", "adaptive_f1", "compression_ratio"])
        writer.writeheader()
        writer.writerows(per_sample)

    print(f"\nSaved results to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()