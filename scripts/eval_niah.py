"""Needle-in-a-Haystack: insert a synthetic fact ("needle") at a controlled
depth inside a long filler context, ask the model to retrieve it, and check
whether the generated answer contains the needle's secret value. Sweeps
context length x insertion depth, comparing the FP16 baseline against
AdaptiveKVCache.

Usage:
    python scripts/eval_niah.py --model microsoft/Phi-3-mini-4k-instruct \
        --context-lengths 1000 4000 8000 --depths 0.1 0.5 0.9
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_kv import AdaptiveKVConfig  # noqa: E402
from adaptive_kv.generate import generate_with_adaptive_cache  # noqa: E402

FILLER = (
    "The grass was green. The sky was blue. The sun was shining. "
    "Birds were singing in the trees. It was a beautiful day for a walk in the park. "
)


def build_haystack(tok, n_tokens: int, depth: float, secret: int) -> tuple[str, str]:
    needle = f"The special magic number for this test is {secret}. Remember it well."
    filler_ids = tok(FILLER, return_tensors="pt")["input_ids"][0]
    n_filler_needed = n_tokens
    reps = n_filler_needed // filler_ids.shape[0] + 2
    long_filler = tok.decode(filler_ids.repeat(reps)[:n_tokens], skip_special_tokens=True)
    insert_char = int(len(long_filler) * depth)
    haystack = long_filler[:insert_char] + " " + needle + " " + long_filler[insert_char:]
    question = "\n\nWhat is the special magic number mentioned in the text above? Answer with just the number."
    return haystack + question, needle


@torch.no_grad()
def generate_baseline(model, tok, prompt: str, max_new_tokens: int = 16) -> str:
    from transformers import DynamicCache
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, past_key_values=DynamicCache(), max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[1000, 4000])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--trials-per-cell", type=int, default=3)
    ap.add_argument("--importance-mode", default="key_diversity",
                     choices=["key_diversity", "attn", "attn_value"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    attn_impl = "eager" if args.importance_mode != "key_diversity" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                  attn_implementation=attn_impl)
    model.eval()
    cfg = AdaptiveKVConfig(importance_mode=args.importance_mode)

    print(f"{'ctx_len':>8} {'depth':>6} {'baseline_acc':>12} {'adaptive_acc':>12}")
    for L in args.context_lengths:
        for d in args.depths:
            base_hits, adapt_hits = 0, 0
            for t in range(args.trials_per_cell):
                secret = random.randint(10000, 99999)
                prompt, needle = build_haystack(tok, L, d, secret)
                base_ans = generate_baseline(model, tok, prompt)
                _, stats = generate_with_adaptive_cache(
                    model, tok, prompt, max_new_tokens=16, cache_config=cfg, return_stats=True,
                )
                adapt_ans = stats["generated_text"]
                base_hits += str(secret) in base_ans
                adapt_hits += str(secret) in adapt_ans
            print(f"{L:>8} {d:>6.2f} {base_hits / args.trials_per_cell:>12.2f} "
                  f"{adapt_hits / args.trials_per_cell:>12.2f}")


if __name__ == "__main__":
    main()
