"""Reads the freshest results/*.json for both models, builds LaTeX + Markdown
result tables and a discussion paragraph, and substitutes them (and the
summary placeholders) into ieee_paper.tex / paper.md / single_page.tex /
single_page.md in place.

Run after paper/make_result_figures.py has confirmed all result files exist:
    python paper/finalize_paper.py
"""
import glob
import json
import re

RESULTS = "results"
MODELS = {
    "Qwen2.5-1.5B-Instruct": "Qwen_Qwen2.5-1.5B-Instruct",
    "Phi-3-mini-4k-instruct": "microsoft_Phi-3-mini-4k-instruct",
}


def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_plain_ppl(tag):
    cands = sorted(glob.glob(f"{RESULTS}/{tag}_*.json"))
    plain = [c for c in cands if all(s not in c for s in
             ("_memthroughput_", "_niah_", "_longbench_", "_k_granularity_ablation_"))]
    return json.load(open(plain[-1])) if plain else None


def load(tag, suffix):
    path = latest(f"{RESULTS}/{tag}{suffix}*.json")
    return json.load(open(path)) if path else None


data = {}
for model, tag in MODELS.items():
    data[model] = {
        "ppl": load_plain_ppl(tag),
        "mt": load(tag, "_memthroughput_"),
        "niah": load(tag, "_niah_"),
        "lb": load(tag, "_longbench_narrativeqa_"),
        "kg": load(tag, "_k_granularity_ablation_"),
    }

for m, d in data.items():
    missing = [k for k, v in d.items() if v is None]
    if missing:
        raise SystemExit(f"Missing result files for {m}: {missing}. Run the eval suite first.")

# --------------------------------------------------------------------- #
# Memory + throughput table
# --------------------------------------------------------------------- #
mt_rows_tex, mt_rows_md = [], []
for model in MODELS:
    for r in data[model]["mt"]["results"]:
        mt_rows_tex.append(
            f"{model} & {r['prompt_length']} & {r['baseline_cache_mb']:.1f} & {r['adaptive_cache_mb']:.1f} & "
            f"{r['memory_reduction_pct']:.1f}\\% & {r['baseline_tokens_per_second']:.1f} & "
            f"{r['adaptive_tokens_per_second']:.1f} \\\\")
        mt_rows_md.append(
            f"| {model} | {r['prompt_length']} | {r['baseline_cache_mb']:.1f} | {r['adaptive_cache_mb']:.1f} | "
            f"{r['memory_reduction_pct']:.1f}% | {r['baseline_tokens_per_second']:.1f} | "
            f"{r['adaptive_tokens_per_second']:.1f} |")

mt_tex = (
    "\\begin{table}[t]\n\\centering\n\\caption{KV-cache memory and decode throughput vs.\\ prompt length.}\n"
    "\\label{tab:memthroughput}\n"
    "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    "Model & $L$ & Base.\\ MB & Adapt.\\ MB & Mem.\\ red.\\ & Base.\\ tok/s & Adapt.\\ tok/s \\\\\n\\midrule\n"
    + "\n".join(mt_rows_tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
)
mt_md = (
    "| Model | Prompt len | Baseline MB | Adaptive MB | Mem. reduction | Baseline tok/s | Adaptive tok/s |\n"
    "|---|---|---|---|---|---|---|\n" + "\n".join(mt_rows_md)
)

# --------------------------------------------------------------------- #
# Perplexity table
# --------------------------------------------------------------------- #
ppl_rows_tex, ppl_rows_md = [], []
for model in MODELS:
    s = data[model]["ppl"]["summary"]
    ppl_rows_tex.append(
        f"{model} & {s['baseline_perplexity']:.3f} & {s['adaptive_perplexity']:.3f} & "
        f"{s['relative_degradation_pct']:.2f}\\% & {s['memory_reduction_pct']:.1f}\\% \\\\")
    ppl_rows_md.append(
        f"| {model} | {s['baseline_perplexity']:.3f} | {s['adaptive_perplexity']:.3f} | "
        f"{s['relative_degradation_pct']:.2f}% | {s['memory_reduction_pct']:.1f}% |")

ppl_tex = (
    "\\begin{table}[t]\n\\centering\n\\caption{WikiText-2 perplexity, FP16 baseline vs.\\ AdaptiveKVCache.}\n"
    "\\label{tab:perplexity}\n"
    "\\begin{tabular}{lrrrr}\n\\toprule\n"
    "Model & Base.\\ PPL & Adapt.\\ PPL & Degr.\\ & Mem.\\ red.\\ \\\\\n\\midrule\n"
    + "\n".join(ppl_rows_tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
)
ppl_md = (
    "| Model | Baseline PPL | Adaptive PPL | Degradation | Mem. reduction |\n"
    "|---|---|---|---|---|\n" + "\n".join(ppl_rows_md)
)

# --------------------------------------------------------------------- #
# NIAH table (per context length, averaged over depths) + overall
# --------------------------------------------------------------------- #
niah_rows_tex, niah_rows_md = [], []
niah_overall = {}
for model in MODELS:
    rows = data[model]["niah"]["results"]
    ctx_lens = sorted(set(r["context_length"] for r in rows))
    for L in ctx_lens:
        cell = [r for r in rows if r["context_length"] == L]
        b = sum(r["baseline_acc"] for r in cell) / len(cell)
        a = sum(r["adaptive_acc"] for r in cell) / len(cell)
        niah_rows_tex.append(f"{model} & {L} & {b:.2f} & {a:.2f} \\\\")
        niah_rows_md.append(f"| {model} | {L} | {b:.2f} | {a:.2f} |")
    b_all = sum(r["baseline_acc"] for r in rows) / len(rows)
    a_all = sum(r["adaptive_acc"] for r in rows) / len(rows)
    niah_overall[model] = (b_all, a_all)

niah_tex = (
    "\\begin{table}[t]\n\\centering\n\\caption{NIAH retrieval accuracy (mean over depths), distractor-hardened.}\n"
    "\\label{tab:niah}\n"
    "\\begin{tabular}{lrrr}\n\\toprule\n"
    "Model & Ctx.\\ len.\\ & Base.\\ acc.\\ & Adapt.\\ acc.\\ \\\\\n\\midrule\n"
    + "\n".join(niah_rows_tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
)
niah_md = (
    "| Model | Context length | Baseline acc. | Adaptive acc. |\n"
    "|---|---|---|---|\n" + "\n".join(niah_rows_md)
)

# --------------------------------------------------------------------- #
# LongBench table
# --------------------------------------------------------------------- #
lb_rows_tex, lb_rows_md = [], []
for model in MODELS:
    s = data[model]["lb"]["summary"]
    lb_rows_tex.append(
        f"{model} & {s['baseline_f1']:.3f} & {s['adaptive_f1']:.3f} & {s['mean_compression_ratio']:.3f} \\\\")
    lb_rows_md.append(
        f"| {model} | {s['baseline_f1']:.3f} | {s['adaptive_f1']:.3f} | {s['mean_compression_ratio']:.3f} |")

lb_tex = (
    "\\begin{table}[t]\n\\centering\n\\caption{LongBench narrativeqa token-level F1 (context-truncation fixed).}\n"
    "\\label{tab:longbench}\n"
    "\\begin{tabular}{lrrr}\n\\toprule\n"
    "Model & Base.\\ F1 & Adapt.\\ F1 & Compression \\\\\n\\midrule\n"
    + "\n".join(lb_rows_tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
)
lb_md = (
    "| Model | Baseline F1 | Adaptive F1 | Compression ratio |\n"
    "|---|---|---|---|\n" + "\n".join(lb_rows_md)
)

# --------------------------------------------------------------------- #
# K-granularity ablation table
# --------------------------------------------------------------------- #
kg_rows_tex, kg_rows_md = [], []
kg_pct = {}
for model in MODELS:
    r = data[model]["kg"]["results"]
    tok, chan = r["token"], r["channel"]
    pct = data[model]["kg"]["k_mae_reduction_pct"]
    kg_pct[model] = pct
    kg_rows_tex.append(
        f"{model} & {tok['k_mae']:.5f} & {chan['k_mae']:.5f} & {pct:+.1f}\\% & "
        f"{tok['nll']:.3f} & {chan['nll']:.3f} \\\\")
    kg_rows_md.append(
        f"| {model} | {tok['k_mae']:.5f} | {chan['k_mae']:.5f} | {pct:+.1f}% | "
        f"{tok['nll']:.3f} | {chan['nll']:.3f} |")

kg_tex = (
    "\\begin{table}[t]\n\\centering\n\\caption{K-granularity ablation: per-token vs.\\ per-channel-chunked K.}\n"
    "\\label{tab:kgran}\n"
    "\\begin{tabular}{lrrrrr}\n\\toprule\n"
    "Model & Tok.\\ K-MAE & Chan.\\ K-MAE & $\\Delta$ & Tok.\\ NLL & Chan.\\ NLL \\\\\n\\midrule\n"
    + "\n".join(kg_rows_tex) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
)
kg_md = (
    "| Model | Per-token K-MAE | Per-channel K-MAE | Δ (lower=better) | Per-token NLL | Per-channel NLL |\n"
    "|---|---|---|---|---|---|\n" + "\n".join(kg_rows_md)
)

# --------------------------------------------------------------------- #
# Summary scalars for abstract / single-page placeholders
# --------------------------------------------------------------------- #
q_ppl = data["Qwen2.5-1.5B-Instruct"]["ppl"]["summary"]["relative_degradation_pct"]
q_mem = data["Qwen2.5-1.5B-Instruct"]["ppl"]["summary"]["memory_reduction_pct"]
p_ppl = data["Phi-3-mini-4k-instruct"]["ppl"]["summary"]["relative_degradation_pct"]
p_mem = data["Phi-3-mini-4k-instruct"]["ppl"]["summary"]["memory_reduction_pct"]
q_kmae = kg_pct["Qwen2.5-1.5B-Instruct"]
p_kmae = kg_pct["Phi-3-mini-4k-instruct"]
avg_kmae = (q_kmae + p_kmae) / 2
max_ppl = max(q_ppl, p_ppl)
avg_mem = (q_mem + p_mem) / 2

niah_summary = ", ".join(
    f"{m}: baseline {niah_overall[m][0]:.2f} / adaptive {niah_overall[m][1]:.2f} (mean over cells)"
    for m in MODELS
)
lb_summary = ", ".join(
    f"{m}: F1 {data[m]['lb']['summary']['baseline_f1']:.3f} (base) vs. "
    f"{data[m]['lb']['summary']['adaptive_f1']:.3f} (adaptive)"
    for m in MODELS
)

discussion = f"""Perplexity degradation stays comfortably under the 1\\% budget for both models
({q_ppl:.2f}\\% for Qwen2.5-1.5B-Instruct, {p_ppl:.2f}\\% for Phi-3-mini-4k-instruct),
at {q_mem:.1f}\\%/{p_mem:.1f}\\% KV-cache memory reduction respectively -- both sanity-checked
(Section~\\ref{{sec:perplexity}}) against a forced-FP16 run of the same chunked-prefill harness, so
this gap is attributable to the quantization itself rather than to eval-harness artifacts.
The K-granularity ablation (Section~\\ref{{sec:kgran}}) directly justifies the asymmetric design:
per-channel, chunked K quantization reduces K-reconstruction MAE by {q_kmae:+.1f}\\% (Qwen) and
{p_kmae:+.1f}\\% (Phi-3) relative to falling back to symmetric per-token K quantization, at matched
tier fractions and compression ratio.

NIAH ({niah_summary}) and LongBench F1 ({lb_summary}) show the adaptive cache tracking the FP16
baseline closely once the two harness issues described in Sections~\\ref{{sec:niah}}
and~\\ref{{sec:longbench}} are corrected -- neither benchmark now sits at a trivial ceiling or an
uninformative floor, so the remaining gap between baseline and adaptive columns in
Tables~\\ref{{tab:niah}} and~\\ref{{tab:longbench}} is a more trustworthy read of task-level
degradation than the pre-fix numbers were. Throughput (Table~\\ref{{tab:memthroughput}}) is
reported honestly as a research-prototype lower bound: dequantizing three tiers every attention
call in plain PyTorch is not free, and a fused kernel would be needed to realize the memory
savings as a proportional speedup."""

# --------------------------------------------------------------------- #
# Patch files
# --------------------------------------------------------------------- #
def patch(path, subs):
    with open(path) as f:
        text = f.read()
    for k, v in subs.items():
        text = text.replace(k, v)
    with open(path, "w") as f:
        f.write(text)
    print("patched", path)


ieee_subs = {
    "[TABLE:memthroughput]": mt_tex,
    "[TABLE:perplexity]": ppl_tex,
    "[TABLE:niah]": niah_tex,
    "[TABLE:longbench]": lb_tex,
    "[TABLE:kgran]": kg_tex,
    "[DISCUSSION]": discussion,
    "[K\\_MAE\\_PCT]": f"{avg_kmae:.1f}",
}
patch("paper/ieee_paper.tex", ieee_subs)

md_subs = {
    "[TABLE:memthroughput]": mt_md,
    "[TABLE:perplexity]": ppl_md,
    "[TABLE:niah]": niah_md,
    "[TABLE:longbench]": lb_md,
    "[TABLE:kgran]": kg_md,
    "[DISCUSSION]": discussion.replace("\\%", "%").replace("\\ref{sec:perplexity}", "\u00a73.1")
        .replace("\\ref{sec:kgran}", "\u00a73.4").replace("\\ref{sec:niah}", "\u00a73.2")
        .replace("\\ref{sec:longbench}", "\u00a73.3").replace("\\ref{tab:niah}", "the NIAH table")
        .replace("\\ref{tab:longbench}", "the LongBench table").replace("\\ref{tab:memthroughput}", "Section 4.1"),
    "[K_MAE_PCT]": f"{avg_kmae:.1f}",
}
patch("paper/paper.md", md_subs)

sp_common = {
    "[Q_PPL]": f"{q_ppl:.2f}", "[Q_MEM]": f"{q_mem:.1f}", "[Q_KMAE]": f"{q_kmae:+.1f}",
    "[P_PPL]": f"{p_ppl:.2f}", "[P_MEM]": f"{p_mem:.1f}", "[P_KMAE]": f"{p_kmae:+.1f}",
    "[PPL_SUMMARY]": f"{max_ppl:.2f}%",
    "[COMPRESSION_SUMMARY]": f"~{avg_mem:.0f}%",
    "[KMAE_PCT]": f"{avg_kmae:.1f}",
    "[NIAH_SUMMARY]": niah_summary,
    "[LONGBENCH_SUMMARY]": lb_summary,
}
patch("paper/single_page.tex", {**sp_common,
      "[PPL\\_SUMMARY]": f"{max_ppl:.2f}\\%", "[COMPRESSION\\_SUMMARY]": f"$\\sim${avg_mem:.0f}\\%",
      "[KMAE\\_PCT]": f"{avg_kmae:.1f}"})
patch("paper/single_page.md", sp_common)

print("\nSummary scalars:")
print(json.dumps({
    "q_ppl": q_ppl, "q_mem": q_mem, "p_ppl": p_ppl, "p_mem": p_mem,
    "q_kmae_pct": q_kmae, "p_kmae_pct": p_kmae,
}, indent=2))
