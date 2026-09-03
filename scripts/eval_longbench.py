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
import re
import string
import sys
from collections import Counter
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
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    attn_impl = "eager" if args.importance_mode != "key_diversity" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                  attn_implementation=attn_impl)
    model.eval()
    cfg = AdaptiveKVConfig(importance_mode=args.importance_mode)

    ds = load_dataset("THUDM/LongBench", args.task, split="test")
    n = min(args.n_samples, len(ds))

    base_f1s, adapt_f1s, ratios = [], [], []
    for i in range(n):
        ex = ds[i]
        prompt = ex["context"] + "\n\nQuestion: " + ex["input"] + "\nAnswer:"
        golds = ex["answers"]

        base_out = generate_baseline(model, tok, prompt, args.max_new_tokens)
        _, stats = generate_with_adaptive_cache(
            model, tok, prompt, max_new_tokens=args.max_new_tokens,
            cache_config=cfg, return_stats=True,
        )
        adapt_out = stats["generated_text"]

        base_f1 = max(f1_score(base_out, g) for g in golds)
        adapt_f1 = max(f1_score(adapt_out, g) for g in golds)
        base_f1s.append(base_f1)
        adapt_f1s.append(adapt_f1)
        ratios.append(stats["compression_ratio"])
        print(f"[{i + 1}/{n}] base_f1={base_f1:.3f} adaptive_f1={adapt_f1:.3f} "
              f"compression={stats['compression_ratio']:.3f}")

    print("\n=== Summary ===")
    print(f"Task: {args.task}  (n={n})")
    print(f"Baseline F1:  {sum(base_f1s) / n:.4f}")
    print(f"Adaptive F1:  {sum(adapt_f1s) / n:.4f}")
    print(f"Mean compression ratio: {sum(ratios) / n:.3f} "
          f"({(1 - sum(ratios) / n) * 100:.1f}% memory reduction)")


if __name__ == "__main__":
    main()
