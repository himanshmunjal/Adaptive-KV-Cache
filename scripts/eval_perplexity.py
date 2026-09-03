"""Perplexity evaluation: FP16 baseline vs. AdaptiveKVCache, on WikiText-2.

Usage:
    python scripts/eval_perplexity.py --model meta-llama/Llama-3.2-1B \
        --seq-len 2048 --n-sequences 20

Requires network access to download the model and the `wikitext` dataset the
first time it's run.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_kv import AdaptiveKVCache, AdaptiveKVConfig  # noqa: E402


@torch.no_grad()
def ppl_baseline(model, ids: torch.Tensor):
    out = model(input_ids=ids, labels=ids)
    return out.loss.item()


@torch.no_grad()
def ppl_adaptive(model, ids: torch.Tensor, cache_config: AdaptiveKVConfig):
    """Auto-regressively re-encode with the adaptive cache and compute the
    same next-token cross-entropy the compressed cache would actually see
    during generation (each token's logits are conditioned on the
    already-compressed representation of everything before it).
    """
    cache = AdaptiveKVCache(model.config.num_hidden_layers, cache_config)
    needs_attn = cache_config.importance_mode != "key_diversity"
    total_loss, total_count = 0.0, 0
    chunk = 64
    pos = 0
    prev_logits = None
    while pos < ids.shape[1]:
        end = min(pos + chunk, ids.shape[1])
        step_ids = ids[:, pos:end]
        out = model(input_ids=step_ids, past_key_values=cache, use_cache=True,
                    output_attentions=needs_attn)
        if needs_attn:
            for i, attn in enumerate(out.attentions):
                cache.record_attention(i, attn[0])
        logits = out.logits
        targets = ids[:, pos + 1:end + 1] if end < ids.shape[1] else ids[:, pos + 1:end]
        pred = logits[:, : targets.shape[1], :]
        loss = torch.nn.functional.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), targets.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_count += targets.numel()
        pos = end
    return total_loss / total_count, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-sequences", type=int, default=10)
    ap.add_argument("--importance-mode", default="key_diversity",
                     choices=["key_diversity", "attn", "attn_value"])
    ap.add_argument("--recent-window", type=int, default=64)
    ap.add_argument("--fp16-frac", type=float, default=0.3)
    ap.add_argument("--int8-frac", type=float, default=0.3)
    ap.add_argument("--output-dir", default="results",
                     help="Directory to write per-run JSON + CSV results into (default: results/)")
    ap.add_argument("--run-name", default=None,
                     help="Basename for the output files (default: derived from model + timestamp)")
    ap.add_argument("--sanity-check", action="store_true",
                     help="Force fp16_frac=1.0, int8_frac=0.0 (no actual quantization, cache keeps "
                          "every token at FP16). If adaptive_nll still differs meaningfully from "
                          "base_nll under this setting, the discrepancy is coming from the chunked-"
                          "prefill eval harness itself, not from quantization -- run this BEFORE "
                          "trusting any compression-vs-quality numbers from this script.")
    args = ap.parse_args()
    if args.sanity_check:
        args.fp16_frac, args.int8_frac = 1.0, 0.0

    if args.sanity_check:
        print("=== SANITY CHECK MODE: fp16_frac=1.0, int8_frac=0.0 forced. "
              "adaptive_nll should track base_nll almost exactly (no quantization is happening). ===\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    attn_impl = "eager" if args.importance_mode != "key_diversity" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation=attn_impl,
    )
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids_all = tok(text, return_tensors="pt")["input_ids"][0]

    cfg = AdaptiveKVConfig(
        recent_window=args.recent_window, fp16_frac=args.fp16_frac,
        int8_frac=args.int8_frac, importance_mode=args.importance_mode,
    )

    base_losses, adapt_losses, ratios = [], [], []
    for i in range(args.n_sequences):
        start = i * args.seq_len
        chunk = ids_all[start:start + args.seq_len].unsqueeze(0)
        if chunk.shape[1] < args.seq_len:
            break
        base_losses.append(ppl_baseline(model, chunk))
        adapt_loss, cache = ppl_adaptive(model, chunk, cfg)
        adapt_losses.append(adapt_loss)
        ratios.append(cache.compression_ratio())
        print(f"[seq {i}] base_nll={base_losses[-1]:.4f}  adaptive_nll={adapt_losses[-1]:.4f}"
              f"  compression={ratios[-1]:.3f}")

    import math
    base_ppl = math.exp(sum(base_losses) / len(base_losses))
    adapt_ppl = math.exp(sum(adapt_losses) / len(adapt_losses))
    mean_ratio = sum(ratios) / len(ratios)
    rel_degradation = (adapt_ppl / base_ppl - 1) * 100
    print("\n=== Summary ===")
    print(f"Baseline (FP16, full cache) perplexity: {base_ppl:.3f}")
    print(f"Adaptive KV cache perplexity:            {adapt_ppl:.3f}")
    print(f"Relative degradation:                    {rel_degradation:.2f}%")
    print(f"Mean KV-cache compression ratio:          {mean_ratio:.3f}"
          f"  ({(1 - mean_ratio) * 100:.1f}% memory reduction)")

    if args.sanity_check and abs(rel_degradation) > 1.0:
        print(f"\n*** SANITY CHECK FAILED ***\n"
              f"No quantization was applied (fp16_frac=1.0) but adaptive_nll still differs from "
              f"base_nll by {rel_degradation:.2f}%. This means the discrepancy you're seeing between "
              f"baseline and adaptive perplexity is coming from the chunked-prefill eval harness "
              f"itself (ppl_adaptive's chunking / cache bookkeeping / target alignment), NOT from "
              f"the quantization. Any compression-vs-quality numbers produced without this flag are "
              f"not trustworthy until this gap is closed to ~0.")
    elif args.sanity_check:
        print(f"\nSanity check passed (|diff| = {abs(rel_degradation):.2f}% <= 1.0%): the harness "
              f"reproduces the baseline when no quantization is applied. Compression-vs-quality "
              f"numbers from a normal (non-sanity-check) run can be trusted to reflect the "
              f"quantization itself.")

    # ---------------------------------------------------------------- #
    # Persist results
    # ---------------------------------------------------------------- #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.model.replace('/', '_')}_{timestamp}"

    per_sequence = [
        {
            "sequence": i,
            "base_nll": base_losses[i],
            "adaptive_nll": adapt_losses[i],
            "compression_ratio": ratios[i],
        }
        for i in range(len(base_losses))
    ]

    payload = {
        "run_name": run_name,
        "timestamp_utc": timestamp,
        "args": vars(args),
        "per_sequence": per_sequence,
        "summary": {
            "baseline_perplexity": base_ppl,
            "adaptive_perplexity": adapt_ppl,
            "relative_degradation_pct": rel_degradation,
            "mean_compression_ratio": mean_ratio,
            "memory_reduction_pct": (1 - mean_ratio) * 100,
        },
    }

    json_path = out_dir / f"{run_name}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_dir / f"{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "base_nll", "adaptive_nll", "compression_ratio"])
        writer.writeheader()
        writer.writerows(per_sequence)

    print(f"\nSaved results to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
