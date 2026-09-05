"""Data-dependent result figures, built from results/*.json produced by the
scripts/eval_*.py suite. Run after the eval suite has produced fresh JSON files
for both models: python paper/make_result_figures.py
"""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "paper/figures"
RESULTS = "results"

MODELS = {
    "Qwen2.5-1.5B-Instruct": "Qwen_Qwen2.5-1.5B-Instruct",
    "Phi-3-mini-4k-instruct": "microsoft_Phi-3-mini-4k-instruct",
}

FP16_C, INT8_C, INT4_C = "#2b6cb0", "#d69e2e", "#c53030"
BASE_C, ADAPT_C = "#718096", "#38a169"


def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load(tag_prefix, suffix):
    path = latest(f"{RESULTS}/{tag_prefix}{suffix}*.json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


data = {}
for model, tag in MODELS.items():
    data[model] = {
        "ppl": load(tag, "_"),   # plain run: {TAG}_{timestamp}.json (no suffix tag)
        "memthroughput": load(tag, "_memthroughput_"),
        "niah": load(tag, "_niah_"),
        "longbench": load(tag, "_longbench_narrativeqa_"),
        "kgran": load(tag, "_k_granularity_ablation_"),
    }

# eval_perplexity.py writes "{model}_{timestamp}.json" with no extra suffix, which
# the generic "_" glob above over-matches (it also matches the suffixed files). Redo
# precisely: find files matching {tag}_{timestamp}.json only (no extra "_xxx_" segment).
for model, tag in MODELS.items():
    cands = sorted(glob.glob(f"{RESULTS}/{tag}_*.json"))
    plain = [c for c in cands if all(s not in c for s in
             ("_memthroughput_", "_niah_", "_longbench_", "_k_granularity_ablation_"))]
    data[model]["ppl"] = json.load(open(plain[-1])) if plain else None

missing = [f"{m}/{k}" for m, d in data.items() for k, v in d.items() if v is None]
if missing:
    print("WARNING: missing result files for:", missing, file=sys.stderr)

# --------------------------------------------------------------------- #
# Fig: perplexity degradation + memory reduction (grouped bars)
# --------------------------------------------------------------------- #
models = list(MODELS.keys())
ppl_degr = [data[m]["ppl"]["summary"]["relative_degradation_pct"] if data[m]["ppl"] else np.nan for m in models]
mem_red = [data[m]["ppl"]["summary"]["memory_reduction_pct"] if data[m]["ppl"] else np.nan for m in models]

fig, ax1 = plt.subplots(figsize=(6.2, 4))
x = np.arange(len(models))
w = 0.35
b1 = ax1.bar(x - w / 2, ppl_degr, w, color="#c53030", label="Perplexity degradation (%)")
ax1.axhline(1.0, color="#742a2a", linestyle="--", linewidth=1, label="1% budget")
ax1.set_ylabel("Perplexity degradation (%)", color="#c53030")
ax1.set_ylim(0, max(2.0, max([v for v in ppl_degr if v == v] + [1.5]) * 1.4))
ax2 = ax1.twinx()
b2 = ax2.bar(x + w / 2, mem_red, w, color="#2b6cb0", label="Memory reduction (%)")
ax2.set_ylabel("KV-cache memory reduction (%)", color="#2b6cb0")
ax2.set_ylim(0, 60)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
for i, v in enumerate(ppl_degr):
    if v == v:
        ax1.text(i - w / 2, v + 0.03, f"{v:.2f}%", ha="center", fontsize=9, color="#742a2a")
for i, v in enumerate(mem_red):
    if v == v:
        ax2.text(i + w / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color="#2c5282")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False, fontsize=8.5)
ax1.set_title("Perplexity degradation vs. memory reduction", pad=28)
plt.tight_layout()
plt.savefig(f"{OUT}/perplexity_vs_memory.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------- #
# Fig: memory + throughput vs. prompt length
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for model in models:
    mt = data[model]["memthroughput"]
    if mt is None:
        continue
    rows = mt["results"]
    L = [r["prompt_length"] for r in rows]
    base_mb = [r["baseline_cache_mb"] for r in rows]
    adapt_mb = [r["adaptive_cache_mb"] for r in rows]
    base_tps = [r["baseline_tokens_per_second"] for r in rows]
    adapt_tps = [r["adaptive_tokens_per_second"] for r in rows]
    axes[0].plot(L, base_mb, "o--", color=BASE_C, label=f"{model} FP16" if model == models[0] else None)
    axes[0].plot(L, adapt_mb, "o-", label=f"{model} adaptive")
    axes[1].plot(L, base_tps, "o--", color=BASE_C, label=f"{model} FP16" if model == models[0] else None)
    axes[1].plot(L, adapt_tps, "o-", label=f"{model} adaptive")
axes[0].set_xlabel("Prompt length (tokens)")
axes[0].set_ylabel("KV-cache size (MB)")
axes[0].set_title("Cache memory vs. context length")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("Prompt length (tokens)")
axes[1].set_ylabel("Decode tokens/second")
axes[1].set_title("Decode throughput vs. context length")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/memory_throughput.png", dpi=200, facecolor="white")
plt.close(fig)

# --------------------------------------------------------------------- #
# Fig: NIAH accuracy heatmap (context length x depth), baseline vs adaptive
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(len(models), 2, figsize=(9, 4.3 * len(models)), squeeze=False)
for mi, model in enumerate(models):
    niah = data[model]["niah"]
    if niah is None:
        continue
    rows = niah["results"]
    ctx_lens = sorted(set(r["context_length"] for r in rows))
    depths = sorted(set(r["depth"] for r in rows))
    for ci, key in enumerate(("baseline_acc", "adaptive_acc")):
        grid = np.zeros((len(depths), len(ctx_lens)))
        for r in rows:
            grid[depths.index(r["depth"]), ctx_lens.index(r["context_length"])] = r[key]
        ax = axes[mi][ci]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(ctx_lens))); ax.set_xticklabels(ctx_lens)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d:.1f}" for d in depths])
        ax.set_xlabel("context length"); ax.set_ylabel("depth")
        ax.set_title(f"{model}\n{'baseline' if ci == 0 else 'adaptive'}", fontsize=9.5)
        for (r, c), v in np.ndenumerate(grid):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)
fig.tight_layout(h_pad=4.0)
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="retrieval accuracy")
plt.savefig(f"{OUT}/niah_heatmap.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------- #
# Fig: LongBench F1 baseline vs adaptive, per model
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(5.5, 4))
base_f1 = [data[m]["longbench"]["summary"]["baseline_f1"] if data[m]["longbench"] else np.nan for m in models]
adapt_f1 = [data[m]["longbench"]["summary"]["adaptive_f1"] if data[m]["longbench"] else np.nan for m in models]
x = np.arange(len(models))
ax.bar(x - w / 2, base_f1, w, color=BASE_C, label="Baseline (FP16)")
ax.bar(x + w / 2, adapt_f1, w, color=ADAPT_C, label="AdaptiveKVCache")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Token-level F1")
ax.set_title("LongBench narrativeqa F1")
ax.legend(fontsize=9)
for i, v in enumerate(base_f1):
    if v == v:
        ax.text(i - w / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=8.5)
for i, v in enumerate(adapt_f1):
    if v == v:
        ax.text(i + w / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{OUT}/longbench_f1.png", dpi=200, facecolor="white")
plt.close(fig)

# --------------------------------------------------------------------- #
# Fig: K-granularity ablation, K-MAE (channel vs token)
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(5.5, 4))
tok_mae = [data[m]["kgran"]["results"]["token"]["k_mae"] if data[m]["kgran"] else np.nan for m in models]
chan_mae = [data[m]["kgran"]["results"]["channel"]["k_mae"] if data[m]["kgran"] else np.nan for m in models]
ax.bar(x - w / 2, tok_mae, w, color="#a0aec0", label="per-token K (ablation baseline)")
ax.bar(x + w / 2, chan_mae, w, color=FP16_C, label="per-channel, chunked K (ours)")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("K reconstruction MAE")
ax.set_title("K-granularity ablation: reconstruction error")
ax.legend(fontsize=8.5)
for i, v in enumerate(tok_mae):
    if v == v:
        ax.text(i - w / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
for i, v in enumerate(chan_mae):
    if v == v:
        ax.text(i + w / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/kgran_ablation.png", dpi=200, facecolor="white")
plt.close(fig)

print("Figures written to", OUT)
print(json.dumps({m: {k: (v is not None) for k, v in d.items()} for m, d in data.items()}, indent=2))
