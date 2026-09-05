"""Ablation: per-channel (chunked) K quantization vs. the original symmetric
per-token K/V scheme, at matched tier fractions / bit-widths.

This isolates the one change this project adds on top of the existing
continuous-promotion tiered cache (see README "Gap this fills"): SubKV/KIVI
report that the key cache has structured per-channel outliers while the
value cache is closer to per-token uniform, so per-channel K quantization
should reduce K reconstruction error (and downstream perplexity) at the same
bit-width/compression ratio, relative to quantizing K the same way as V.

Runs offline with a tiny randomly-initialized model by default (no network
needed, mirrors `tests/`); pass --model for a real HF checkpoint.

Usage:
    python scripts/eval_k_granularity_ablation.py
    python scripts/eval_k_granularity_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --seq-len 1024
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_kv import AdaptiveKVCache, AdaptiveKVConfig  # noqa: E402


def _tiny_model_and_tokenizer():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=1000, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(config)
    model.eval()

    class _Tok:
        eos_token_id = 2

        def __call__(self, ids_list):
            return torch.tensor([ids_list])

    return model, _Tok(), config.vocab_size


@torch.no_grad()
def run_once(model, ids: torch.Tensor, cfg: AdaptiveKVConfig, seq_len: int):
    """Feed `ids` through the model with the adaptive cache, chunked exactly
    like `eval_perplexity.py` does, then report next-token cross-entropy,
    K-reconstruction error against a true-FP16 reference cache, and the
    measured compression ratio.
    """
    from transformers import DynamicCache

    cache = AdaptiveKVCache(model.config.num_hidden_layers, cfg)
    ref_cache = DynamicCache()
    chunk = 32
    pos = 0
    total_loss, total_count = 0.0, 0
    while pos < seq_len:
        end = min(pos + chunk, seq_len)
        step_ids = ids[:, pos:end]
        out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
        model(input_ids=step_ids, past_key_values=ref_cache, use_cache=True)
        targets = ids[:, pos + 1:end + 1] if end < seq_len else ids[:, pos + 1:end]
        pred = out.logits[:, : targets.shape[1], :]
        loss = torch.nn.functional.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), targets.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_count += targets.numel()
        pos = end

    k_err = []
    for layer, ref_layer in zip(cache.layers, ref_cache.layers):
        approx_k, _ = layer._full_kv_dequant()
        ref_k = ref_layer.keys[0].to(approx_k.dtype)
        k_err.append((approx_k - ref_k).abs().mean().item())

    return {
        "nll": total_loss / total_count,
        "k_mae": sum(k_err) / len(k_err),
        "compression_ratio": cache.compression_ratio(),
        "tier_summary": cache.tier_summary(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id; omit to use a tiny synthetic model")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--recent-window", type=int, default=16)
    ap.add_argument("--fp16-frac", type=float, default=0.2)
    ap.add_argument("--int8-frac", type=float, default=0.3)
    ap.add_argument("--realloc-interval", type=int, default=16)
    ap.add_argument("--min-tokens-to-compress", type=int, default=32)
    ap.add_argument("--output-dir", default="results",
                     help="Directory to write per-run JSON + CSV results into (default: results/)")
    ap.add_argument("--run-name", default=None,
                     help="Basename for the output files (default: derived from model + timestamp)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                      device_map="auto")
        model.eval()
        # A single repeated sentence keeps every chunk's per-channel min/max
        # nearly identical to its per-token min/max (there's no real token-to-
        # token variation for per-channel grouping to exploit), which washes
        # out the very effect this ablation is meant to isolate and made the
        # channel-vs-token K-MAE gap look far smaller than it is on real text.
        # Natural text has the structured per-channel outliers SubKV/KIVI
        # report, so use a real passage instead.
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"][:200])
        ids = tok(text, return_tensors="pt")["input_ids"][:, : args.seq_len].to(model.device)
    else:
        model, tok, vocab_size = _tiny_model_and_tokenizer()
        model = model.to(device)
        torch.manual_seed(1)
        ids = torch.randint(3, vocab_size, (1, args.seq_len), device=device)
    print(f"Model loaded on device: {next(model.parameters()).device}")

    common = dict(
        recent_window=args.recent_window, sink_tokens=4,
        fp16_frac=args.fp16_frac, int8_frac=args.int8_frac,
        realloc_interval=args.realloc_interval,
        min_tokens_to_compress=args.min_tokens_to_compress,
        importance_mode="key_diversity",
    )

    results = {}
    for gran in ("token", "channel"):
        cfg = AdaptiveKVConfig(k_granularity=gran, **common)
        results[gran] = run_once(model, ids, cfg, args.seq_len)

    print(f"{'':>10} | {'K MAE':>10} | {'NLL':>8} | {'compression':>11} | tier_summary")
    for gran in ("token", "channel"):
        r = results[gran]
        print(f"{gran:>10} | {r['k_mae']:10.5f} | {r['nll']:8.4f} | "
              f"{r['compression_ratio']:11.3f} | {r['tier_summary']}")

    mae_reduction = 1 - results["channel"]["k_mae"] / results["token"]["k_mae"]
    print(f"\nPer-channel K quantization K-MAE change vs. per-token K: "
          f"{mae_reduction * 100:+.1f}% (positive = lower error)")

    # ---------------------------------------------------------------- #
    # Persist results
    # ---------------------------------------------------------------- #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_tag = args.model.replace("/", "_") if args.model else "tiny_synthetic"
    run_name = args.run_name or f"{model_tag}_k_granularity_ablation_{timestamp}"

    payload = {
        "run_name": run_name,
        "timestamp_utc": timestamp,
        "args": vars(args),
        "results": results,
        "k_mae_reduction_pct": mae_reduction * 100,
    }

    json_path = out_dir / f"{run_name}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_dir / f"{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["granularity", "k_mae", "nll", "compression_ratio", "tier_summary"])
        writer.writeheader()
        for gran in ("token", "channel"):
            r = results[gran]
            writer.writerow({
                "granularity": gran, "k_mae": r["k_mae"], "nll": r["nll"],
                "compression_ratio": r["compression_ratio"], "tier_summary": r["tier_summary"],
            })

    print(f"\nSaved results to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()