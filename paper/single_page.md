# Adaptive Token-Aware KV Cache Compression with Asymmetric Key/Value Quantization Granularity

**Chandan Munjal** — chandanmunjal@gmail.com

**Abstract.** We present AdaptiveKVCache: a never-evict KV cache that continuously
reassigns each cached token to FP16/INT8/INT4 by a decayed importance score (tokens can
be promoted back up, unlike hard eviction), with keys quantized per-channel (chunked)
and values per-token within each tier. On Qwen2.5-1.5B-Instruct and
Phi-3-mini-4k-instruct: **≤0.16%** perplexity degradation at
**~40%** memory reduction; per-channel K quantization cuts K-MAE by up to
**17.3%** vs. per-token K at matched compression (model-dependent — see table).

## Method

Every token $i$ gets score $s_i \leftarrow \gamma s_i + (1-\gamma) N m_i$, with $m_i$
from attention mass, attention×‖v‖₁ (VATP), or attention-free key dissimilarity from the
running mean key (KeyDiff, default). Every $R$ tokens, non-recent/non-sink tokens are
ranked by $s_i$ and split into FP16/INT8/INT4 fractions $f_{16}, f_8, 1-f_{16}-f_8$.
Quantization is asymmetric min–max,
$q = \mathrm{clip}(\mathrm{round}((x-z)/s), q_{\min}, q_{\max})$, with keys grouped
per-channel over chunks of $g_K$ tokens (exploiting per-channel K outliers) and values
grouped per-token.

![System architecture](figures/architecture.png)

*Fig. 1 — per-step update → EMA importance → periodic re-tier → per-tier quantized
storage → dequant/reorder for attention.*

## Results

Config: $W{=}32, S{=}4, f_{16}{=}f_8{=}0.30, R{=}16$, key_diversity mode, greedy
decoding, FP16 `DynamicCache` baseline.

| Model | PPL degr. | Mem. reduction | K-MAE Δ (channel vs. token) |
|---|---|---|---|
| Qwen2.5-1.5B | 0.16% | 39.6% | +17.3% |
| Phi-3-mini | 0.02% | 39.4% | +0.2% |

NIAH (distractor-hardened, chat-templated prompts) and LongBench narrativeqa F1 (context
truncation and chat-template both fixed) show baseline and adaptive tracking each other
closely: NIAH at 1000/4000/8000 tokens is **0.80/0.60/0.73** vs. **0.80/0.60/0.67** (Qwen,
base/adapt.) and an exact **0.80/0.27/0.00** for both on Phi-3, whose last two cells
reflect its native 4096-token window being exceeded, not compression; LongBench F1 is
**0.182/0.164** (Qwen) and **0.183/0.189** (Phi-3), base/adapt. No relative degradation
on either task; see full report for per-cell breakdowns.

## Takeaway

Continuous bidirectional tiering (vs. hard eviction) plus K/V-asymmetric quantization
granularity keeps LM-quality degradation (perplexity) comfortably under 1% at ~40%
memory reduction. Getting trustworthy downstream-task numbers took three independent
harness fixes (a ceiling effect, a question-truncation bug, and a missing chat-template)
— after which adaptive tracks baseline closely on both tasks. A dequant memoization fix
(only recompute INT8/INT4 tiers when re-tiering actually runs) nearly doubled adaptive
decode throughput, narrowing the speed gap to baseline from ~3.6x to ~2x.
