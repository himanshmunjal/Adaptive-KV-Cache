# Adaptive Token-Aware KV Cache Compression

**Repo:** https://github.com/himanshmunjal/Adaptive-KV-Cache

Mixed-precision, dynamically-reallocated KV cache for efficient LLM inference.
Every token stays in the cache (nothing is evicted), but each token's Key/Value
pair is stored at **FP16**, **INT8**, or **INT4** depending on a running
importance score, re-evaluated every `realloc_interval` tokens so a token can
move *up* in precision again if it becomes relevant later.

**Headline result:** on Qwen2.5-1.5B-Instruct and Phi-3-mini-4k-instruct,
this cuts KV-cache memory by **~39-40%** at **<0.2% perplexity degradation**
(sanity-checked against a forced-FP16 run of the same harness — see
[`paper/`](paper/) for the full write-up, methodology, and every number below).

## Results

All numbers below are from real runs against the two models above (FP16
`DynamicCache` baseline, greedy decoding, default `AdaptiveKVConfig` —
see [Design notes](#design-notes--known-simplifications) for exact
hyperparameters). Full per-cell breakdowns, figures, and discussion are in
[`paper/paper.md`](paper/paper.md) / [`paper/ieee_paper.tex`](paper/ieee_paper.tex).

**Memory & perplexity** (the two headline targets: 20-40% memory reduction, <1% accuracy drop):

| Model | Baseline PPL | Adaptive PPL | Degradation | Memory reduction |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 9.371 | 9.387 | **0.16%** | **39.6%** |
| Phi-3-mini-4k-instruct | 6.119 | 6.121 | **0.02%** | **39.4%** |

**Downstream tasks** (NIAH: distractor-hardened retrieval; LongBench: narrativeqa F1;
both baseline/adaptive, chat-template + context-truncation bugs fixed — see
[Bug fixes](#bug-fixes-made-during-this-project) below):

| Model | NIAH @1000/4000/8000 (base) | NIAH @1000/4000/8000 (adapt) | LongBench F1 (base/adapt) |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.80/0.60/0.73 | 0.80/0.60/0.67 | 0.182/0.164 |
| Phi-3-mini-4k-instruct | 0.80/0.27/0.00† | 0.80/0.27/0.00† | 0.183/0.189 |

†Phi-3's 4000/8000-token cells exceed its native 4096-token context window once
the needle/distractors/question are added — a model context-window ceiling, not
a compression or task-difficulty effect. Baseline and adaptive are identical in
every Phi-3 NIAH cell either way.

**K-granularity ablation** (justifies the per-channel-K / per-token-V design):

| Model | Per-token K-MAE | Per-channel K-MAE | Improvement |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.62068 | 1.33974 | **+17.3%** |
| Phi-3-mini-4k-instruct | 1.27501 | 1.27298 | +0.2% (negligible — model-dependent, see paper) |

**Throughput** (research-prototype lower bound; fake-quantization dequantizes to FP16
before every matmul, so INT4/INT8 storage never becomes accelerated compute — see
[Design notes](#design-notes--known-simplifications)):

| Model | Prompt len | Baseline tok/s | Adaptive tok/s |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 2048 | 43.1 | 22.1 |
| Phi-3-mini-4k-instruct | 2048 | 22.4 | 17.0 |

## Bug fixes made during this project

Three evaluation-harness bugs were found and fixed while producing the results
above — worth knowing about since they each independently distort the numbers
if reintroduced:

1. **LongBench context truncation was deleting the question.** Truncating the
   *whole prompt* (context + appended question) from one end silently dropped
   the question itself whenever the context alone exceeded the token budget
   (true for most LongBench documents). Fixed in `scripts/eval_longbench.py`
   by truncating only the context (keeping first/last halves) before
   appending the question.
2. **NIAH had a ceiling effect.** A single needle with no other numeric
   content in the haystack reduces to "find the only sentence with a number,"
   solvable perfectly regardless of compression. Fixed by seeding 3
   phrasing-identical decoy needles at random depths in `scripts/eval_niah.py`.
3. **Neither eval script used the models' chat template.** Both models are
   instruction-tuned but were being fed raw-text continuation prompts instead
   of `tokenizer.apply_chat_template(...)` output — this alone roughly doubled
   absolute LongBench F1 and lifted NIAH out of a near-degenerate range for
   both baseline and adaptive once fixed.

A fourth fix improved (but did not change the correctness of) throughput:
`AdaptiveKVLayer` was fully re-dequantizing its INT8/INT4 tiers from scratch
on *every* decode step, even though those tiers only change once every
`realloc_interval` steps. Memoizing the dequantized tensors (invalidated only
when re-tiering actually runs, in `adaptive_kv/cache.py`) nearly doubled
adaptive decode throughput at an unchanged compression ratio.

## Full paper / report

See [`paper/`](paper/) for the complete write-up:

- [`paper/ieee_paper.tex`](paper/ieee_paper.tex) — full IEEE-format paper (methodology, math, algorithm listings, complexity analysis, all results, discussion, limitations, appendix)
- [`paper/paper.md`](paper/paper.md) — the same content in Markdown
- [`paper/single_page.tex`](paper/single_page.tex) / [`paper/single_page.md`](paper/single_page.md) — condensed one-page versions
- [`paper/figures/`](paper/figures/) — all diagrams and result plots (architecture, quantization scheme, tiering pipeline, memory/throughput, perplexity-vs-memory, NIAH heatmap, LongBench F1, K-granularity ablation, compression-quality Pareto curve)

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
(`scripts/eval_k_granularity_ablation.py`) — **empirically, this benefit is
model-dependent**: +17.3% K-MAE improvement on Qwen2.5-1.5B-Instruct, only
+0.2% on Phi-3-mini-4k-instruct (see [Results](#results) above and the paper's
discussion).

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
                   # quantization, V per-token, within each INT8/INT4 tier; memoizes
                   # INT8/INT4 dequantization between re-tiering passes
  generate.py       # explicit generation loop that feeds attention weights back
                   # into the cache each step (needed for the "attn"/"attn_value" modes)
scripts/
  eval_perplexity.py             # WikiText-2 perplexity, baseline vs. adaptive (+ --sanity-check
                                  # mode that isolates harness bugs from real quantization effects)
  eval_memory_throughput.py      # KV-cache bytes and tokens/sec vs. context length
  eval_longbench.py              # LongBench task accuracy (F1), chat-template + truncation-fixed
  eval_niah.py                    # Needle-in-a-Haystack retrieval accuracy sweep, distractor-hardened
  eval_k_granularity_ablation.py # channel-K vs. token-K: K-MAE, NLL, compression ratio
paper/
  ieee_paper.tex, paper.md        # full write-up (see Full paper / report above)
  single_page.tex, single_page.md # condensed one-page versions
  figures/                        # all diagrams and result plots
  make_diagrams.py, make_result_figures.py, finalize_paper.py,
  finalize_extra_experiments.py   # scripts that (re)generate the paper's figures/tables from results/*.json
results/                          # raw JSON + CSV output of every eval script run
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

Then run the evaluation scripts against the target models (results in the
table above used Qwen2.5-1.5B-Instruct and Phi-3-mini-4k-instruct):

```bash
python scripts/eval_perplexity.py        --model Qwen/Qwen2.5-1.5B-Instruct --seq-len 2048 --n-sequences 20
python scripts/eval_perplexity.py        --model Qwen/Qwen2.5-1.5B-Instruct --seq-len 2048 --n-sequences 5 --sanity-check
python scripts/eval_memory_throughput.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt-lengths 512 2048 8192
python scripts/eval_longbench.py         --model Qwen/Qwen2.5-1.5B-Instruct --task narrativeqa --n-samples 30
python scripts/eval_niah.py              --model Qwen/Qwen2.5-1.5B-Instruct --context-lengths 1000 4000 8000
```

`--importance-mode` selects the scoring signal: `key_diversity` (default,
attention-free, works with any attention backend), `attn` (H2O-style
accumulated attention mass, requires `attn_implementation="eager"`), or
`attn_value` (VATP-style attention x value-L1-norm, also requires eager). The
paper's importance-mode ablation (see [`paper/`](paper/)) shows all three stay
within the sub-1% perplexity budget at matched compression on both models.

Ablation for the K-channel/V-token quantization asymmetry (works offline, no
model download needed for the default synthetic-model mode):

```bash
python scripts/eval_k_granularity_ablation.py
python scripts/eval_k_granularity_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --seq-len 1024
```

It reports K-cache reconstruction MAE (against a true-FP16 `DynamicCache`
reference), next-token NLL, and compression ratio, for `k_granularity in
{"token", "channel"}` at matched tier fractions/bit-widths. When run with
`--model` against a real checkpoint, use natural-language probe text (the
script defaults to WikiText-2) rather than a repeated filler sentence — a
repeated sentence has almost no token-to-token variation for per-channel
grouping to exploit and washes out the effect being measured.

## Design notes / known simplifications

- **Batch size 1.** Tiering is computed once per sequence; extending to a
  per-example policy under batching is straightforward but unimplemented —
  see the paper's Limitations and Future Work section for what this would
  require (per-example re-tiering triggers, attention-mask-aware ranking).
- **Global (not per-head) tiering.** Importance is averaged across attention
  heads before ranking, so all heads share one tier assignment per token —
  a common simplification (see H2O's own "global vs. local statistic"
  ablation) that keeps storage tensors regular and fast to slice.
- **Per-channel K / per-token V quantization** (default). Within the INT8/INT4
  tiers, Keys are quantized per-channel in fixed-size chunks of
  `k_channel_group_int8`/`_int4` consecutive tokens (`quant.py`'s
  `quantize_int8_channel`/`quantize_int4_channel`), while Values stay
  per-token (unchanged). This follows the SubKV/KIVI finding that K carries
  structured per-channel outliers while V is closer to per-token-uniform —
  confirmed empirically for Qwen2.5-1.5B-Instruct (+17.3% K-MAE improvement)
  but **not** for Phi-3-mini-4k-instruct (+0.2%, negligible), so this default
  should be validated per model rather than assumed universal (see paper).
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
- **INT8/INT4 dequantization is memoized between re-tiering passes.** The
  cache used to fully re-dequantize its INT8/INT4 tiers from scratch on
  *every* decode step; since those tiers only actually change inside a
  re-tiering pass (every `realloc_interval` steps), the dequantized result is
  now cached and only recomputed when re-tiering invalidates it. This nearly
  doubled adaptive decode throughput with no change to memory or output —
  see `AdaptiveKVLayer._int8_dequant_cache`/`_int4_dequant_cache` in `cache.py`.
- **This is a research/simulation implementation** (fake-quantization with
  real reduced-dtype storage for accurate memory accounting), not a
  production fused CUDA kernel — `tokens/sec` numbers reflect Python-level
  quantize/dequantize overhead, not a lower bound on achievable throughput.
  Concretely: tensors are dequantized back to FP16 before every matmul, so no
  INT4/INT8 tensor-core acceleration is ever used — the compute cost of
  attention is unaffected by bit-width choice at all, which is why memory
  drops ~40% while decode throughput *drops* rather than improves. A natural
  next step (noted in the AQUA-KV paper as future work for its own method,
  and in this project's paper's Limitations section) is a fused kernel that
  performs the attention matmul directly against packed INT4/INT8 codes,
  e.g. building on vLLM or llama.cpp.

## Evaluation checklist

- [x] Memory usage — `scripts/eval_memory_throughput.py`, `cache.nbytes()` / `compression_ratio()`
- [x] Tokens/sec — `scripts/eval_memory_throughput.py`
- [x] Perplexity — `scripts/eval_perplexity.py` (WikiText-2), incl. `--sanity-check` harness validation
- [x] K-channel vs. K-token ablation — `scripts/eval_k_granularity_ablation.py` (K-MAE, NLL, compression)
- [x] LongBench — `scripts/eval_longbench.py`
- [x] Needle-in-a-Haystack — `scripts/eval_niah.py`
- [x] Importance-mode ablation (`attn` / `attn_value` / `key_diversity`) — see [`paper/`](paper/)
- [x] Compression-quality trade-off sweep — see [`paper/`](paper/)

All of the above have been run against Qwen2.5-1.5B-Instruct and
Phi-3-mini-4k-instruct on a GPU (results summarized above, full detail in
[`paper/`](paper/) and raw output in `results/*.json`/`*.csv`). The full suite
also passes `pytest` against small randomly-initialized `transformers` models,
including a grouped-query-attention config, with no network access required.
