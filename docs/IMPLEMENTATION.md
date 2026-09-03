# Adaptive KV Cache — Implementation Notes

A from-the-base walkthrough of how the mixed-precision, self-reallocating KV
cache is implemented: the data structures, the quantization math, the
importance scoring, the retiering algorithm, and how it plugs into a real
`transformers` model.

**Tier legend used throughout:** `FP16` = recent / heavy-hitter tokens,
`INT8` = medium importance, `INT4` = low importance.

---

## 1. Overview & the gap it fills

The KV cache stores one Key and one Value vector per token, per layer, per
attention head, and it grows for as long as generation runs — for long
contexts it routinely outgrows the model weights themselves. Every prior
method surveyed for this project compresses it in one of two ways:

- **Evict** tokens outright (H2O, KeyDiff, Spotlight Attention's retrieval) —
  a one-way door. Once a token is dropped, it's gone even if later context
  makes it relevant again.
- **Quantize everything to one fixed bit-width** (AQUA-KV, the AAAI'26
  multimodal quantizer) — this spends the same bits on a filler token as on
  the sentence that answers the eventual question.

Nothing in the survey does both at once: keep every token, but let *which
precision* it's stored at move continuously — up or down — as its importance
changes over the course of generation. That's what this implementation does.
Concretely, it's a custom `transformers.Cache` subclass that never deletes a
cached token; instead, every `realloc_interval` steps it re-ranks all cached
tokens by a running importance score and re-assigns each one to `FP16`,
`INT8`, or `INT4` storage.

---

## 2. Architecture — one layer's life cycle

Everything below happens once per transformer layer, per generation step.
The four modules map directly onto four files in `adaptive_kv/`.

```
 new K, V            append          ImportanceTracker        every realloc_interval:
 [H, T_new, D]  ───▶  → FP16 tier ───▶ key-diversity / attn /  ───▶  rank by score,
                       (always full        attn·‖v‖₁                protect recent_window
                        precision)
                                                                          │
                                                                          ▼
                                                            ┌─────────────────────────┐
                                                            │  FP16 store              │
                                                            │  INT8  (q, scale, zero)  │
                                                            │  INT4  (packed nibbles)  │
                                                            └─────────────────────────┘
                                                                          │
                                                                          ▼
                                                              dequantize (concat all 3 tiers)
                                                                          │
                                                                          ▼
                                                         full K, V ──▶ this step's attention math
```

One `AdaptiveKVLayer.update()` call — new tokens land in FP16, importance
updates every step, tiers reshuffle periodically, attention always sees the
full dequantized cache.

| Module | Role |
|---|---|
| `quant.py` | Per-token, per-head asymmetric INT8 / packed-INT4 quantize & dequantize, plus byte accounting. |
| `importance.py` | `ImportanceTracker` — EMA score per cached token, three scoring modes. |
| `cache.py` | `AdaptiveKVLayer` / `AdaptiveKVCache` — the tiered storage, retiering policy, and the `transformers.Cache` contract. |
| `generate.py` | Explicit prefill+decode loop that feeds attention weights back into the cache each step. |

---

## 3. Quantization layer

Both INT8 and INT4 use **asymmetric min-max quantization**, but with two
different granularities depending on which of K or V is being quantized:

- **Values** always use per-token, per-head grouping (reduction axis is
  `head_dim`), matching the granularity KIVI/AQUA-KV use for values.
- **Keys** use **per-channel, chunked** grouping by default (reduction axis
  is the token axis, computed within fixed-size chunks of `group`
  consecutive tokens — one scale/zero per channel per chunk). This follows
  the finding in SubKV (`Papers/ABSTRACT.docx`) and the AAAI'26 multimodal
  quantization paper that the key cache carries structured, per-channel
  outliers, unlike the more token-uniform value cache. Set
  `AdaptiveKVConfig(k_granularity="token")` to fall back to per-token K
  quantization (the ablation baseline, see README).

The per-token functions (`quantize_int8`/`quantize_int4`) are unchanged from
the original design; the per-channel ones
(`quantize_int8_channel`/`quantize_int4_channel`) share the same min-max
derivation, just reducing over a different axis — see below.

```
Quantize (shared derivation for INT8 and INT4)
scale = (max(x) − min(x)) / (qmax − qmin)
zero  = min(x) − qmin · scale
q     = clamp(round((x − zero) / scale), qmin, qmax)

Dequantize
x̂ = q · scale + zero
```

Because `zero` already folds in the `qmin` offset, `q` lands in
`[qmin, qmax]` with no further shift needed on either side — an early
version of this code added `qmin` twice (once in the quantizer, once in the
dequantizer) and silently doubled the error; the unit tests in
`tests/test_quant.py` catch exactly this class of bug by asserting the
round-trip error stays within one quantization step.

### INT4 packing

INT8 codes are stored one-per-byte in a signed `int8` tensor. INT4 codes are
shifted into the unsigned range `[0, 15]` and packed two-per-byte along the
head-dimension axis, halving that axis's storage:

```python
# adaptive_kv/quant.py
def _pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    # q: uint8 [..., group] with values in [0, 15], group even
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)   # [..., group // 2]
```

| Tier | Stored per token · head | Bytes / token · head (head_dim = D) |
|---|---|---|
| **FP16** | raw `[D]` fp16 vector | `2·D` |
| **INT8** | `[D]` int8 codes + scale + zero | `D + 4` |
| **INT4** | `[D/2]` packed bytes + scale + zero | `D/2 + 4` |

This is exactly what `quant.tier_nbytes()` computes, and it's what every
"compression ratio" number in the eval scripts is derived from — not an
estimate, the actual byte count of the tensors that back the cache.

---

## 4. Importance tracker

`ImportanceTracker` holds exactly one running score per cached token — a 1-D
tensor, `scores[i]`, aggregated across attention heads by mean. It supports
three ways of producing the raw per-step signal that feeds its EMA:

| Mode | Signal | Needs | Provenance |
|---|---|---|---|
| `key_diversity` (default) | 1 − cosine(key, mean key) | nothing extra — computed from `key_states` already in the cache | KeyDiff (2025) |
| `attn` | mean attention mass received this step | eager attention + `output_attentions=True` | H2O (2023) |
| `attn_value` | attention mass × normalized ‖value‖₁ | same as above | VATP (2024) |

`key_diversity` is the default because it's the only one that works under
any attention backend (SDPA, FlashAttention, eager) — it never needs the
attention matrix materialized, which is the exact problem KeyDiff's paper
identifies with attention-based scoring under FlashAttention/block-wise
processing.

```
EMA update (all three modes converge here)
scores ← decay · scores + (1 − decay) · raw_signal · N

key_diversity signal, per cached token i
k̄ = mean over tokens of mean-over-heads(key_i)
signal_i = max(0, 1 − cos(mean-over-heads(key_i), k̄))
```

Unlike H2O's plain cumulative sum (`decay = 1` recovers that exact
behavior), `decay < 1` lets a token's score *fade* as well as grow — which
is what allows a token demoted to INT4 to climb back to FP16 later if it
becomes relevant again, instead of being permanently written off the way
eviction-based methods are.

---

## 5. The cache itself — data structures

Each `AdaptiveKVLayer` is **one transformer layer's** cache. It keeps three
completely separate tensor groups — one per precision tier — instead of one
array with a per-element dtype tag, because PyTorch tensors are
single-dtype: mixed precision has to mean *separate tensors*, concatenated
only when a full view is needed.

| Field | Shape | Holds |
|---|---|---|
| `_fp16_k`, `_fp16_v` | `[H, n₀, D]` | Raw fp16 keys/values for the FP16 tier |
| `_int8_k`, `_int8_v` | `[H, n₁, D]` | int8 codes; paired with `_int8_k_scale/_zero`, shape `[H, n₁, 1]` |
| `_int4_k`, `_int4_v` | `[H, n₂, D/2]` | packed nibble codes; paired scale/zero as above |
| `pos_fp16 / pos_int8 / pos_int4` | `[nᵢ]` | global sequence position of each slot — the only thing that determines recency protection |
| `importance.scores` | `[n₀+n₁+n₂]` | running score, kept in the *same order* as the FP16→INT8→INT4 concatenation |
| `_global_len` | scalar | total tokens ever appended (monotonic sequence position counter) |

> **Note — order doesn't need to match arrival order.** Keys are cached
> *after* RoPE is applied, so each key vector already encodes its own
> position — attention over a cache is just a weighted sum over whatever's
> present, causally safe by construction since every cached token
> necessarily precedes the current one. That's what frees the retiering step
> to freely reshuffle tokens between tiers without tracking a global
> ordering.

Batch size is fixed at 1 throughout — `lazy_initialization` asserts on it.
The tiering decision in this implementation is *global* per token (shared
across heads, averaged), not per-head; both are simplifications made to keep
every tensor operation a plain slice/cat instead of a ragged, per-batch-item
or per-head bookkeeping structure. See [Limitations](#12-limitations--design-tradeoffs).

---

## 6. Retiering algorithm

Called from `update()` whenever `tokens_since_realloc ≥ realloc_interval`
*and* the cache already holds at least `min_tokens_to_compress` tokens.

1. **Dequantize everything (for scoring only).** Pull the full `[H, N, D]` K
   and V back to fp16 by concatenating and dequantizing all three tiers —
   ranking always happens on a clean, uniform-precision view, never on
   already-degraded numbers.
2. **Protect the recent window *and* the attention sinks.** Any token with
   `position ≥ global_len − recent_window` **or** `position < sink_tokens`
   is pinned to FP16 regardless of score. The recency guard is the same one
   H2O and StreamingLLM rely on (stops a token from being demoted the
   instant it's created, before its importance has had a chance to be
   observed); the sink-token guard is separate and protects the first few
   tokens specifically, because StreamingLLM/AQUA-KV/KIVI all find these
   "attention sink" tokens draw disproportionate attention while having a
   near-zero value-vector norm — which means the `attn_value` (VATP-style)
   scoring mode would otherwise *systematically undervalue exactly the
   tokens most likely to matter*.
3. **Rank the rest by score.** `argsort` the importance scores of every
   unprotected token, descending.
4. **Split by fraction.** The top `fp16_frac` of the ranked, unprotected
   tokens join the protected set at FP16; the next `int8_frac` go to INT8;
   everything left over goes to INT4.
5. **Rebuild only what changed.** For each of the INT8 and INT4 tiers,
   tokens whose new tier assignment is the *same* as their previous one keep
   their already-quantized bytes verbatim (a plain tensor gather); only
   tokens that actually change tier get dequantized-then-requantized. This
   is `_rebuild_tier()` in the code.

```python
# adaptive_kv/cache.py — AdaptiveKVLayer._retier
protected = (positions >= (self._global_len - self.cfg.recent_window)) | \
            (positions < self.cfg.sink_tokens)
order = torch.argsort(scores, descending=True)
unprotected_order = order[~protected[order]]
n_rest = unprotected_order.numel()
n_fp16_extra = int(round(self.cfg.fp16_frac * n_rest))
n_int8 = int(round(self.cfg.int8_frac * n_rest))

tier = torch.full((n,), INT4, dtype=torch.long, device=self.device)
tier[protected] = FP16
tier[unprotected_order[:n_fp16_extra]] = FP16
tier[unprotected_order[n_fp16_extra:n_fp16_extra + n_int8]] = INT8
# everything else stays INT4
```

```python
# adaptive_kv/cache.py — AdaptiveKVLayer._rebuild_tier (per INT8/INT4 tier)
target_mask = tier == tier_id
keep_mask = target_mask & (old_tier == tier_id)   # unchanged: reuse stored bytes as-is
fresh_mask = target_mask & ~keep_mask             # changed tier: (re)quantize from full precision
```

Because tokens are never evicted, `positions` always covers exactly
`0 .. global_len - 1` with no gaps, so "does this token's tier match its
previous tier" is a single vectorized lookup table (`old_tier_by_pos`,
`old_local_by_pos`), not a Python loop or eviction-aware bookkeeping. This
directly reduces the quantization drift discussed in
[Limitations](#12-limitations--design-tradeoffs) below: a token that stays
in INT8 across ten consecutive retiering passes now accumulates **zero**
extra rounding error from those passes, instead of ten dequantize→requantize
round-trips' worth. `tests/test_cache.py::test_retier_reuses_unchanged_quantized_bytes`
asserts this bit-for-bit. `realloc_interval` still trades reallocation
responsiveness against the O(N log N) sort cost of each retiering pass.

**Caveat for K (per-channel mode).** `_rebuild_tier` above (now
`_rebuild_tier_v`) is only used for V, and for K when `k_granularity="token"`.
When K uses the default `k_granularity="channel"`, `_rebuild_tier_k` always
requantizes the whole tier from full precision — a chunk's scale/zero are a
joint function of every token in it, so per-token byte-reuse doesn't apply.
It still reuses V's `keep-then-fresh` token *order* (not bytes) so K/V/positions
stay aligned. This is deterministic, though: if the set of tokens in a tier
and their order are unchanged between two retiering passes, K's requantized
bytes come out bit-identical anyway (same inputs -> same outputs).

---

## 7. Integration with `transformers.Cache`

Rather than writing a bespoke generation loop that reimplements attention,
the cache plugs into HuggingFace's own extension point: `CacheLayerMixin`.
`AdaptiveKVLayer` subclasses it and implements the four methods the
framework actually calls during a forward pass — the same contract
`DynamicCache`'s own `DynamicLayer` implements, which is what makes this a
true drop-in replacement.

| Method | Contract | What it does here |
|---|---|---|
| `lazy_initialization` | Called once, on the first real K/V seen | Infers `num_heads`/`head_dim`/dtype, allocates empty tier tensors |
| `update(key, value)` | Store new K/V, return what attention should see | Appends to FP16, updates importance, maybe retiers, returns the full dequantized cache |
| `get_seq_length()` | Total cached length, for mask/position bookkeeping | `n_fp16 + n_int8 + n_int4` |
| `get_mask_sizes(query_length)` | Sizes the causal mask | `(seq_length + query_length, 0)` — same as `DynamicLayer`, since all cached tokens legitimately precede the query |

`AdaptiveKVCache` itself is a thin `Cache` subclass — it just constructs one
`AdaptiveKVLayer` per transformer layer and forwards a handful of
memory-accounting methods (`nbytes()`, `compression_ratio()`,
`tier_summary()`) that sum across layers.

```python
# adaptive_kv/cache.py — usage
cache = AdaptiveKVCache(model.config.num_hidden_layers,
                         AdaptiveKVConfig(importance_mode="key_diversity"))
out = model(input_ids=ids, past_key_values=cache, use_cache=True)
# `cache` now works with model.generate(..., past_key_values=cache) too
```

---

## 8. Generation loop

The `key_diversity` mode needs nothing beyond what `update()` already does,
so it would run fine under plain `model.generate()`. The `attn` and
`attn_value` modes need the real softmax attention weights fed back in
after every forward pass — something HF's built-in generation loop has no
hook for — so `generate_with_adaptive_cache()` drives the loop explicitly
for all three modes, for consistency and so the eval scripts can time
prefill and decode separately.

```
prefill forward ──▶ record_attention ──▶ argmax next token ──▶ decode forward ─┐
(whole prompt,       (if mode ≠                (greedy or        (1 token,     │
 once)                key_diversity)             sampled)          uses cache) │
                                                       ▲                        │
                                                       └── loop until max_new_tokens or eos
```

Every call, `generate_with_adaptive_cache()` also records wall-clock time
for the prefill and decode phases separately and reads the cache's own byte
counters at the end, which is exactly the `stats` dict the memory/throughput
and LongBench eval scripts consume: `tokens_per_second`, `cache_bytes`,
`compression_ratio`, and `tier_summary` — a live
`{"fp16": …, "int8": …, "int4": …}` token count.

---

## 9. Memory accounting

Every "compression ratio" reported anywhere in this project is
`cache.nbytes() / cache.nbytes_fp16_equivalent()` — real byte counts of the
tensors actually allocated, summed across all layers, never an estimate:

```
nbytes()                 = Σ_layers  2·tier_nbytes(n_fp16,16) + 2·tier_nbytes(n_int8,8) + 2·tier_nbytes(n_int4,4)
nbytes_fp16_equivalent() = Σ_layers  2·tier_nbytes(n_fp16+n_int8+n_int4, 16)
compression_ratio()      = nbytes() / nbytes_fp16_equivalent()      # < 1.0 = smaller than fp16
```

The leading `2·` accounts for storing both K *and* V per tier. In the
integration test (`tests/test_memory_smoke.py`), a 300-token synthetic
sequence through a tiny 2-layer GQA model came out to **26,592 bytes** in
the adaptive cache against **38,400 bytes** in an equivalent `DynamicCache`
— a measured **30.8% reduction**, from the default config alone.

---

## 10. Testing & validation

Everything is validated against small, randomly-initialized real
`transformers` models (a 2-layer `LlamaForCausalLM` with grouped-query
attention enabled) rather than mocks — so the tests exercise the actual
`Cache` contract, not a stand-in for it. No network access or model download
is required to run the suite.

| File | Covers |
|---|---|
| `test_quant.py` | INT8/INT4 round-trip error bounds; nibble pack/unpack idempotency |
| `test_cache.py` | End-to-end generation in all three importance modes; recent-window tokens are never compressed; tier counts stay consistent with total sequence length |
| `test_memory_smoke.py` | Adaptive cache byte count vs. a real `DynamicCache` baseline on the same sequence |

`python -m pytest tests/ -q` → **9 passed** (added `test_sink_tokens_never_compressed`
and `test_retier_reuses_unchanged_quantized_bytes` alongside the original 7 to cover
the two optimizations in §6).

---

## 11. Evaluation scripts

Four CLI scripts under `scripts/`, all targeting Llama-3.2-1B / Qwen2.5-1.5B
/ Phi-3-mini, cover every metric in the project brief:

| Script | Metric | Method |
|---|---|---|
| `eval_perplexity.py` | Perplexity | WikiText-2, next-token cross-entropy re-encoded through the adaptive cache in 64-token chunks vs. an uncompressed baseline forward pass |
| `eval_memory_throughput.py` | Memory, tokens/sec | Real byte counts and measured decode-phase throughput vs. a `DynamicCache` baseline, swept over prompt length |
| `eval_longbench.py` | Task accuracy (F1) | `THUDM/LongBench` tasks (e.g. NarrativeQA), simplified token-F1 scoring |
| `eval_niah.py` | Needle-in-a-Haystack retrieval | Synthetic "magic number" fact inserted at swept context length × depth; checks it's reproduced correctly |

---

## 12. Limitations & design tradeoffs

- **Batch size 1.** Tiering is computed once per sequence, not per batch
  item — `lazy_initialization` asserts on it. Extending to a per-example
  policy under batching is a natural next step, not a structural blocker.
- **Global, not per-head, tiering.** Importance is head-averaged before
  ranking, so every head shares one token's tier assignment — this is the
  same tradeoff H2O's own "global vs. local statistic" ablation makes, kept
  here so tensor slices stay regular.
- **Retiering still re-quantizes tokens that change tier.** As of the
  optimization in §6, tokens whose tier assignment is *unchanged* between
  retiering passes now keep their stored bytes verbatim (no more drift for
  those). Tokens that actively oscillate between tiers across many
  reallocations still pick up ordinary quantization error each time they
  cross a tier boundary — that's inherent to representing them at a lower
  bit-width at all, not an implementation shortcut.
- **Simulation-grade throughput.** Quantize/dequantize run in plain PyTorch,
  not a fused kernel, so `tokens_per_second` reflects Python-level overhead,
  not a lower bound on what a production kernel (e.g. built on vLLM or
  llama.cpp) could achieve.

---

## 13. File map

```
adaptive_kv/
  __init__.py     # exports AdaptiveKVCache, AdaptiveKVConfig, ImportanceTracker
  quant.py        # INT8 / INT4 quantize, dequantize, byte accounting
  importance.py   # ImportanceTracker — 3 scoring modes
  cache.py        # AdaptiveKVLayer, AdaptiveKVCache, retiering policy
  generate.py     # explicit prefill+decode loop
scripts/
  eval_perplexity.py
  eval_memory_throughput.py
  eval_longbench.py
  eval_niah.py
tests/
  test_quant.py   test_cache.py   test_memory_smoke.py
```

*adaptive_kv — build notes · batch_size = 1 · default mode = key_diversity*
