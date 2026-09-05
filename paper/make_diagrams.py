"""Static (data-independent) architecture diagrams for the paper.
Run once with the .venv active: python paper/make_diagrams.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

OUT = "paper/figures"

FP16_C = "#2b6cb0"
INT8_C = "#d69e2e"
INT4_C = "#c53030"
BOX_EDGE = "#1a202c"
BG = "#f7fafc"


def box(ax, x, y, w, h, text, fc="#ffffff", ec=BOX_EDGE, fontsize=10, weight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, zorder=3)
    return b


def arrow(ax, x0, y0, x1, y1, style="-|>", color=BOX_EDGE, lw=1.4, connectionstyle=None):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, zorder=1,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)


# --------------------------------------------------------------------- #
# Figure 1: end-to-end system architecture / per-step data flow
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.2)
ax.axis("off")
fig.patch.set_facecolor("white")

box(ax, 0.3, 3.7, 1.7, 1.0, "New token\n$k_t, v_t$\n(FP16)", fc="#ebf8ff", fontsize=9.5)
box(ax, 2.5, 3.7, 1.9, 1.0, "Append to\nFP16 tier", fc="#ebf8ff", fontsize=9.5)
arrow(ax, 2.0, 4.2, 2.5, 4.2)

box(ax, 2.5, 2.1, 1.9, 1.0, "Importance\nTracker\n(EMA update)", fc="#f0fff4", fontsize=9.5)
arrow(ax, 3.45, 3.7, 3.45, 3.1)

box(ax, 5.0, 2.9, 2.1, 1.0, "every $R$ tokens:\nRe-tier ranking", fc="#fffaf0", fontsize=9.5)
arrow(ax, 4.4, 4.2, 5.0, 3.55, connectionstyle="arc3,rad=-0.15")
arrow(ax, 4.4, 2.6, 5.0, 3.15, connectionstyle="arc3,rad=0.15")

box(ax, 7.6, 4.2, 1.1, 0.7, "FP16", fc="#ebf8ff", ec=FP16_C, fontsize=10, weight="bold")
box(ax, 7.6, 3.2, 1.1, 0.7, "INT8", fc="#fefcbf", ec=INT8_C, fontsize=10, weight="bold")
box(ax, 7.6, 2.2, 1.1, 0.7, "INT4", fc="#fed7d7", ec=INT4_C, fontsize=10, weight="bold")
arrow(ax, 7.1, 3.5, 7.6, 4.5, connectionstyle="arc3,rad=-0.1")
arrow(ax, 7.1, 3.4, 7.6, 3.55)
arrow(ax, 7.1, 3.3, 7.6, 2.55, connectionstyle="arc3,rad=0.1")

ax.text(8.15, 5.0, "Per-tier storage", ha="center", fontsize=10, weight="bold")
ax.text(8.15, 1.9, "K: per-channel chunked quant.\nV: per-token quant.",
        ha="center", fontsize=8.3, style="italic", color="#4a5568")

box(ax, 9.2, 3.2, 1.5, 1.0, "Dequantize\n+ reorder by\nposition", fc="#faf5ff", fontsize=9)
arrow(ax, 8.7, 4.55, 9.2, 4.0, connectionstyle="arc3,rad=-0.2")
arrow(ax, 8.7, 3.55, 9.2, 3.7)
arrow(ax, 8.7, 2.55, 9.2, 3.4, connectionstyle="arc3,rad=0.2")

box(ax, 9.2, 1.3, 1.5, 1.2, "Attention\n$\\mathrm{softmax}(QK^\\top/\\sqrt{d})V$", fc="#edf2f7", fontsize=8.7)
arrow(ax, 9.95, 3.2, 9.95, 2.5)

ax.text(5.5, 0.5,
        "Recency window (last $W$ tokens) and sink tokens (first $S$ tokens) are excluded from re-tiering "
        "and always kept at FP16.",
        ha="center", fontsize=8.6, style="italic", color="#4a5568")

ax.set_title("AdaptiveKVCache: per-step update, periodic re-tiering, and attention read path",
              fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(f"{OUT}/architecture.png", dpi=200, facecolor="white")
plt.close(fig)

# --------------------------------------------------------------------- #
# Figure 2: asymmetric K/V quantization granularity
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.patch.set_facecolor("white")

n_tokens, head_dim, group = 12, 8, 4

# Left: K, per-channel chunked (reduction axis = tokens within a chunk)
ax = axes[0]
ax.set_xlim(-0.5, head_dim - 0.5)
ax.set_ylim(-0.5, n_tokens - 0.5)
ax.invert_yaxis()
for chunk_start in range(0, n_tokens, group):
    color = "#bee3f8" if (chunk_start // group) % 2 == 0 else "#e9f4ff"
    ax.add_patch(Rectangle((-0.5, chunk_start - 0.5), head_dim, min(group, n_tokens - chunk_start),
                            facecolor=color, edgecolor="none", zorder=0))
for c in range(head_dim):
    ax.add_patch(Rectangle((c - 0.5, -0.5), 1, n_tokens, fill=False, edgecolor="#4299e1", linewidth=1.6, zorder=2))
    ax.text(c, n_tokens + 0.3, f"$d_{{{c}}}$", ha="center", fontsize=7.5)
for chunk_start in range(0, n_tokens, group):
    ax.plot([-0.5, head_dim - 0.5], [chunk_start - 0.5, chunk_start - 0.5], color="#2b6cb0", lw=1.2, zorder=3)
    ax.text(head_dim - 0.1, chunk_start + group / 2 - 0.5,
            f"scale$_{{{chunk_start // group}}}$/zero$_{{{chunk_start // group}}}$\n(per channel)",
            fontsize=6.6, va="center", color="#2b6cb0")
ax.set_xticks([]); ax.set_yticks([])
ax.set_ylabel("token index (sequence position)")
ax.set_title(f"Keys: per-channel, chunked\n(one scale/zero per channel per {group}-token chunk)", fontsize=10)

# Right: V, per-token (reduction axis = channels, one scale/zero per token)
ax = axes[1]
ax.set_xlim(-0.5, head_dim - 0.5)
ax.set_ylim(-0.5, n_tokens - 0.5)
ax.invert_yaxis()
for t in range(n_tokens):
    color = "#feebc8" if t % 2 == 0 else "#fff5e6"
    ax.add_patch(Rectangle((-0.5, t - 0.5), head_dim, 1, facecolor=color, edgecolor="none", zorder=0))
for t in range(n_tokens):
    ax.plot([-0.5, head_dim - 0.5], [t - 0.5, t - 0.5], color="#d69e2e", lw=1.0, zorder=3)
    ax.text(head_dim - 0.1, t, f"scale$_{{{t}}}$/zero$_{{{t}}}$", fontsize=6.2, va="center", color="#b7791f")
for c in range(head_dim):
    ax.add_patch(Rectangle((c - 0.5, -0.5), 1, n_tokens, fill=False, edgecolor="#ecc94b", linewidth=0.6, zorder=2))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Values: per-token\n(one scale/zero per token, shared across channels)", fontsize=10)

fig.suptitle("Asymmetric quantization granularity within a tier (INT8 or INT4)", fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig(f"{OUT}/quant_scheme.png", dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------- #
# Figure 3: tiering timeline (sink / body / recent, with promotion arrows)
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(0, 40)
ax.set_ylim(0, 4)
ax.axis("off")
fig.patch.set_facecolor("white")

sink_n, recent_n, total = 4, 8, 40
tiers = ["FP16"] * sink_n
np.random.seed(3)
mid = total - sink_n - recent_n
mid_tiers = ["FP16"] * int(0.3 * mid) + ["INT8"] * int(0.3 * mid) + ["INT4"] * (mid - int(0.3 * mid) - int(0.3 * mid))
order = np.random.permutation(len(mid_tiers))
mid_tiers = [mid_tiers[i] for i in order]
tiers += mid_tiers + ["FP16"] * recent_n

colors = {"FP16": "#bee3f8", "INT8": "#fefcbf", "INT4": "#fed7d7"}
edges = {"FP16": FP16_C, "INT8": INT8_C, "INT4": INT4_C}
for i, t in enumerate(tiers):
    ax.add_patch(Rectangle((i, 1.2), 0.92, 1.0, facecolor=colors[t], edgecolor=edges[t], linewidth=1.2))

ax.add_patch(Rectangle((-0.3, 1.0), sink_n + 0.3, 1.4, fill=False, edgecolor="#2d3748", linewidth=1.6, linestyle="--"))
ax.text(sink_n / 2 - 0.15, 2.55, "sink tokens\n(always FP16)", ha="center", fontsize=8.6)

ax.add_patch(Rectangle((total - recent_n, 1.0), recent_n + 0.3, 1.4, fill=False, edgecolor="#2d3748",
                        linewidth=1.6, linestyle="--"))
ax.text(total - recent_n / 2, 2.55, "recency window\n(always FP16)", ha="center", fontsize=8.6)

ax.add_patch(Rectangle((sink_n, 1.0), mid, 1.4, fill=False, edgecolor="#805ad5", linewidth=1.4, linestyle=":"))
ax.text(sink_n + mid / 2, 3.0, "ranked by importance score $s_i$ -- re-tiered every $R$ tokens",
        ha="center", fontsize=9, color="#553c9a")

legend_handles = [mpatches.Patch(facecolor=colors[k], edgecolor=edges[k], label=k) for k in ("FP16", "INT8", "INT4")]
ax.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))

ax.set_title("Tier layout along the (never-evicted) sequence: recency + sink protection, "
              "continuous promotion/demotion elsewhere", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/tiering_pipeline.png", dpi=200, facecolor="white")
plt.close(fig)

print("wrote:",
      f"{OUT}/architecture.png,",
      f"{OUT}/quant_scheme.png,",
      f"{OUT}/tiering_pipeline.png")
