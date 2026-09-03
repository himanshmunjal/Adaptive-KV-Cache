"""Memory footprint and tokens/sec for AdaptiveKVCache vs. a full FP16
DynamicCache baseline, at increasing context lengths.

Usage:
    python scripts/eval_memory_throughput.py --model microsoft/Phi-3-mini-4k-instruct \
        --prompt-lengths 512 2048 8192 --max-new-tokens 128
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_kv import AdaptiveKVConfig  # noqa: E402
from adaptive_kv.generate import generate_with_adaptive_cache  # noqa: E402


def make_prompt(tok, n_tokens: int) -> str:
    # Repeat filler text and truncate to the exact token budget.
    filler = "The quick brown fox jumps over the lazy dog. " * 400
    ids = tok(filler, return_tensors="pt")["input_ids"][0][:n_tokens]
    return tok.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def baseline_run(model, tok, prompt: str, max_new_tokens: int):
    from transformers import DynamicCache
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    cache = DynamicCache()
    t0 = time.perf_counter()
    out = model(**inputs, past_key_values=cache, use_cache=True)
    prefill_s = time.perf_counter() - t0
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    t1 = time.perf_counter()
    n = 0
    for _ in range(max_new_tokens):
        out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        n += 1
    decode_s = time.perf_counter() - t1
    total = sum(l.keys.numel() + l.values.numel() for l in cache.layers if l.is_initialized)
    nbytes = total * 2  # fp16
    return {"tokens_per_second": n / decode_s, "cache_bytes": nbytes, "prefill_seconds": prefill_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--importance-mode", default="key_diversity",
                     choices=["key_diversity", "attn", "attn_value"])
    ap.add_argument("--output-dir", default="results",
                     help="Directory to write per-run JSON + CSV results into (default: results/)")
    ap.add_argument("--run-name", default=None,
                     help="Basename for the output files (default: derived from model + timestamp)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    attn_impl = "eager" if args.importance_mode != "key_diversity" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                  attn_implementation=attn_impl,
                                                  device_map="auto")
    model.eval()
    print(f"Model loaded on device: {model.device}")
    cfg = AdaptiveKVConfig(importance_mode=args.importance_mode)

    results = []
    print(f"{'prompt_len':>10} | {'baseline MB':>12} | {'adaptive MB':>12} | {'reduction':>9} "
          f"| {'baseline tok/s':>14} | {'adaptive tok/s':>14}")
    for L in args.prompt_lengths:
        prompt = make_prompt(tok, L)
        base = baseline_run(model, tok, prompt, args.max_new_tokens)
        _, adapt = generate_with_adaptive_cache(
            model, tok, prompt, max_new_tokens=args.max_new_tokens,
            cache_config=cfg, return_stats=True,
        )
        red = 1 - adapt["cache_bytes"] / base["cache_bytes"]
        print(f"{L:>10} | {base['cache_bytes'] / 1e6:12.1f} | {adapt['cache_bytes'] / 1e6:12.1f} "
              f"| {red * 100:8.1f}% | {base['tokens_per_second']:14.2f} "
              f"| {adapt['tokens_per_second']:14.2f}")
        results.append({
            "prompt_length": L,
            "baseline_cache_mb": base["cache_bytes"] / 1e6,
            "adaptive_cache_mb": adapt["cache_bytes"] / 1e6,
            "memory_reduction_pct": red * 100,
            "baseline_tokens_per_second": base["tokens_per_second"],
            "adaptive_tokens_per_second": adapt["tokens_per_second"],
        })

    # ---------------------------------------------------------------- #
    # Persist results
    # ---------------------------------------------------------------- #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.model.replace('/', '_')}_memthroughput_{timestamp}"

    payload = {"run_name": run_name, "timestamp_utc": timestamp, "args": vars(args), "results": results}

    json_path = out_dir / f"{run_name}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_dir / f"{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()