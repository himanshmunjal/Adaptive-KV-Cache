# Adaptive Token-Aware KV Cache Compression

Mixed-precision, dynamically-reallocated KV cache for efficient LLM inference.
Every token stays in the cache (nothing is evicted), but each token's Key/Value
pair is stored at **FP16**, **INT8**, or **INT4** depending on a running
importance score, re-evaluated every `realloc_interval` tokens so a token can
move *up* in precision again if it becomes relevant later.

## Gap this fills (see `Papers/` survey below)

| Paper | What it does | Limitation this project addresses |
|---|---|---|
| **H2O** (NeurIPS'23) | Evicts low-attention tokens, keeps "heavy hitters" + recent window at full precision | Binary keep/evict — a dropped token's information is gone forever, even if it becomes relevant again |
| **AQUA-KV** (2025) | Learns inter-layer/inter-role linear predictors to compress quantization residuals | Same bit-width for every token regardless of importance; importance-aware allocation is listed as future work |
| **VATP** (EMNLP'24) | Shows attention score alone is a poor importance proxy; combines it with value-vector L1 norm | Still used only for binary eviction, not precision allocation |
| **Dynamic KV Cache Quantization** (AAAI'26) | Per-channel (K) / per-token (V) quantization granularity with EMA-tracked scales, for multimodal LLMs | Adapts *granularity*, not *bit-width*, and not per-token by importance |
| **Spotlight Attention** (NeurIPS'25) | Non-linear hashing retrieval so tokens are never permanently evicted | Requires a trained hashing network; still binary in/out of the active working set per step |
| **KeyDiff** (NeurIPS'25) | Attention-free eviction using key cosine-dissimilarity (FlashAttention-compatible) | Still eviction (binary), not combined with mixed precision |
| **KVC-Q** (2026, KVC.docx) | Dynamic mixed-precision KV cache: recency + importance + *static* head-wise sensitivity profiling, integrated with PagedAttention | Very close in spirit to this project's "dynamic information fidelity" framing, but its head-aware allocation is an *offline-profiled, fixed* per-head bit budget, not a per-token score that lets a token move back up in precision as this project's `ImportanceTracker` does |
| **MPOQ** (mpoq.docx) | Data-free quantization via Matrix Product Operator decomposition, redistributing outliers into small local tensors that get high-precision treatment | Orthogonal, structural approach to the same outlier problem; a natural complement to (not replacement for) importance-based tiering — noted as future work below |
| **SubKV** (ABSTRACT.docx) | Per-**channel** quantization for K, per-**token** for V, plus dynamic-window and attention-sink-aware quantization, for sub-billion-parameter LLMs | This project already had sink-token and recent-window protection; it did **not** have channel-vs-token K/V asymmetry — now added, see below |

**Novel combination implemented here:** continuous 3-tier precision reallocation
(not binary keep/evict) driven by a hybrid, FlashAttention-compatible
importance signal (KeyDiff-style key-diversity by default; optional VATP-style
attention x value-norm score), with tokens able to be **promoted back** to
higher precision — directly addressing the "premature/irrevocable eviction"
failure mode that both H2O's own ablations and the Spotlight Attention paper
identify — **combined with** the channel-vs-token quantization-granularity
asymmetry the AAAI'26 paper and SubKV establish for K vs. V, which none of
the surveyed papers pair with importance-driven, promotable tiering. Concretely:
within the INT8/INT4 tiers, **Keys are quantized per-channel** (chunked over
`k_channel_group_int8`/`_int4` consecutive tokens, one scale/zero per
channel) while **Values stay per-token** — see `quant.py`'s `*_channel`
functions and `cache.py`'s `_rebuild_tier_k`. Set
`AdaptiveKVConfig(k_granularity="token")` to fall back to the original
symmetric scheme for an ablation baseline
(`scripts/eval_k_granularity_ablation.py`).

> Note: `Papers/2405.14203v1.pdf` ("GLaD," on predicting organic-photovoltaic
> power-conversion efficiency from molecular graphs) is unrelated to KV cache
> compression and was not used in this analysis — likely added to the folder
> by mistake.

## Package layout

```
adaptive_kv/
  quant.py        # per-token/per-head INT8/INT4 quantization (used for V, and for K
                   # when k_granularity="token"), plus per-channel/chunked INT8/INT4
                   # quantization (*_channel functions, used for K by default)
  importance.py   # ImportanceTracker: EMA scores from attention-mass, VATP-style
                   # attn*value-norm, or KeyDiff-style key-diversity
  cache.py         # AdaptiveKVCache / AdaptiveKVLayer: transformers.Cache-compatible,
                   # drop-in replacement for DynamicCache; K uses per-channel
                   # quantization, V per-token, within each INT8/INT4 tier
  generate.py       # explicit generation loop that feeds attention weights back
                   # into the cache each step (needed for the "attn"/"attn_value" modes)
scripts/
  eval_perplexity.py             # WikiText-2 perplexity, baseline vs. adaptive
  eval_memory_throughput.py      # KV-cache bytes and tokens/sec vs. context length
  eval_longbench.py              # THUDM/LongBench task accuracy (F1)
  eval_niah.py                    # Needle-in-a-Haystack retrieval accuracy sweep
  eval_k_granularity_ablation.py # channel-K vs. token-K: K-MAE, NLL, compression ratio
tests/                       # unit + integration tests (run with tiny random-init
                              # models, no network/model download required)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q        # validates the whole pipeline, no downloads needed
```

Minimal usage against a real model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from adaptive_kv import AdaptiveKVConfig
from adaptive_kv.generate import generate_with_adaptive_cache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="float16")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

text, stats = generate_with_adaptive_cache(
    model, tok, "Explain KV cache compression in two sentences.",
    max_new_tokens=64, return_stats=True,
)
print(text)
print(stats["tier_summary"], stats["compression_ratio"])
```

Then run the evaluation scripts against the target models (Llama-3.2-1B,
Qwen2.5-1.5B, Phi-3-mini):

```bash
python scripts/eval_perplexity.py        --model Qwen/Qwen2.5-1.5B-Instruct --seq-len 2048 --n-sequences 10
python scripts/eval_memory_throughput.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt-lengths 512 2048 8192
python scripts/eval_longbench.py         --model Qwen/Qwen2.5-1.5B-Instruct --task narrativeqa --n-samples 30
python scripts/eval_niah.py              --model Qwen/Qwen2.5-1.5B-Instruct --context-lengths 1000 4000 8000
```

`--importance-mode` selects the scoring signal: `key_diversity` (default,
attention-free, works with any attention backend), `attn` (H2O-style
accumulated attention mass, requires `attn_implementation="eager"`), or
`attn_value` (VATP-style attention x value-L1-norm, also requires eager).

Ablation for the K-channel/V-token quantization asymmetry (works offline, no
model download needed for the default synthetic-model mode):

```bash
python scripts/eval_k_granularity_ablation.py
python scripts/eval_k_granularity_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --seq-len 1024
```

It reports K-cache reconstruction MAE (against a true-FP16 `DynamicCache`
reference), next-token NLL, and compression ratio, for `k_granularity in
{"token", "channel"}` at matched tier fractions/bit-widths. **Caveat:** on
the tiny randomly-initialized model used by default, the two are
statistically indistinguishable (K-MAE within noise, see script output) —
expected, since SubKV/KIVI's per-channel-outlier finding is a property of
*trained* representations, not random weights. Confirming the K-MAE/NLL
improvement requires running it against a real pretrained checkpoint (same
disk-space caveat as the rest of the eval suite, below).

## Design notes / known simplifications

- **Batch size 1.** Tiering is computed once per sequence; extending to a
  per-example policy under batching is straightforward but unimplemented.
- **Global (not per-head) tiering.** Importance is averaged across attention
  heads before ranking, so all heads share one tier assignment per token —
  a common simplification (see H2O's own "global vs. local statistic"
  ablation) that keeps storage tensors regular and fast to slice.
- **Per-channel K / per-token V quantization** (new). Within the INT8/INT4
  tiers, Keys are quantized per-channel in fixed-size chunks of
  `k_channel_group_int8`/`_int4` consecutive tokens (`quant.py`'s
  `quantize_int8_channel`/`quantize_int4_channel`), while Values stay
  per-token (unchanged). This follows the SubKV/KIVI finding that K carries
  structured per-channel outliers while V is closer to per-token-uniform.
  Trade-off: because a chunk's scale/zero depend jointly on every token in
  it, the "reuse unchanged bytes across retiering passes" optimization below
  no longer applies to K (only to V) — every retiering pass fully
  requantizes the K tiers from full precision. Set
  `k_granularity="token"` to disable this and fall back to the original
  symmetric scheme (see `scripts/eval_k_granularity_ablation.py`).
- **Attention-sink protection.** The first `sink_tokens` positions (default
  4) are pinned to FP16 alongside the `recent_window`, independent of their
  importance score. StreamingLLM/AQUA-KV/KIVI all report that the first few
  tokens draw disproportionate attention while having a near-zero
  value-vector norm, which means the `attn_value` (VATP-style) scoring mode
  would otherwise systematically undervalue exactly the tokens most likely
  to matter.
- **Re-tiering reuses unchanged bytes.** When the tiering policy is
  re-evaluated, only tokens whose tier assignment actually *changes* are
  dequantized and requantized; tokens that stay in the same INT8/INT4 tier
  keep their already-quantized bytes verbatim (a vectorized gather, made
  possible because no token is ever evicted, so positions never have gaps).
  This eliminates the avoidable rounding drift a token would otherwise
  accumulate every time the tiering policy re-runs and re-confirms its
  current tier — see `tests/test_cache.py::test_retier_reuses_unchanged_quantized_bytes`.
  Tokens that genuinely cross a tier boundary still pick up ordinary
  quantization error, since that's inherent to storing them at a different
  bit-width, not an implementation shortcut. `realloc_interval` still trades
  reallocation responsiveness against the O(N log N) sort cost of each pass.
- **This is a research/simulation implementation** (fake-quantization with
  real reduced-dtype storage for accurate memory accounting), not a
  production fused CUDA kernel — `tokens/sec` numbers reflect Python-level
  quantize/dequantize overhead, not a lower bound on achievable throughput.
  A natural next step (noted in the AQUA-KV paper as future work for its own
  method) is a custom kernel, e.g. building on vLLM or llama.cpp.

## Evaluation checklist (from the project brief)

- [x] Memory usage — `scripts/eval_memory_throughput.py`, `cache.nbytes()` / `compression_ratio()`
- [x] Tokens/sec — `scripts/eval_memory_throughput.py`
- [x] Perplexity — `scripts/eval_perplexity.py` (WikiText-2)
- [x] K-channel vs. K-token ablation — `scripts/eval_k_granularity_ablation.py` (K-MAE, NLL, compression)
- [x] LongBench — `scripts/eval_longbench.py`
- [x] Needle-in-a-Haystack — `scripts/eval_niah.py`

Running these against the three target models requires downloading multi-GB
checkpoints; this was **not done in this environment because the local disk
had only ~4.6GB free (98% full)** — run the scripts above once you have space
or on a GPU machine. Everything else (algorithm, cache implementation,
quantization correctness, end-to-end generation, memory accounting) has been
validated with `pytest` against small randomly-initialized `transformers`
models, including a grouped-query-attention config, with no network access
required.
