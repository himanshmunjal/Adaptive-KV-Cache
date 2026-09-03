#!/usr/bin/env bash
# One-shot setup + eval run for a RunPod GPU pod (RTX 3090/4090, 24GB).
#
# Assumptions:
#   - A Network Volume is attached and mounted at /workspace (RunPod default).
#   - This project's files have already been copied onto the pod, e.g. into
#     /workspace/semester-project (see the upload note at the bottom of this
#     file for how to get them there).
#   - You're running this from inside that project directory.
#
# Usage (on the pod, inside a terminal / Jupyter terminal):
#   cd /workspace/semester-project
#   bash scripts/runpod_setup.sh
#
# Re-running is safe: venv creation and pip installs are idempotent, and the
# HF cache lives on the network volume so models aren't re-downloaded.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# --- 1. Persist the HF cache + venv on the network volume, not container disk. --- #
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
grep -qxF 'export HF_HOME=/workspace/hf_cache' ~/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
grep -qxF 'export TRANSFORMERS_CACHE=/workspace/hf_cache' ~/.bashrc || echo 'export TRANSFORMERS_CACHE=/workspace/hf_cache' >> ~/.bashrc

VENV_DIR=/workspace/venv
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- 2. Install deps. --- #
pip install --upgrade pip -q
pip install -r requirements.txt -q
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# --- 3. Hugging Face auth (needed for gated Llama-3.2-1B; Qwen2.5/Phi-3 are open). --- #
if [ -z "${HF_TOKEN:-}" ]; then
    echo
    echo "!! No HF_TOKEN set. If you need meta-llama/Llama-3.2-1B (gated), run:"
    echo "     huggingface-cli login"
    echo "   (accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B first)"
    echo "   Qwen2.5-1.5B-Instruct and Phi-3-mini-4k-instruct are open, no login needed."
    echo
else
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# --- 4. Sanity check: run the fast, no-download unit tests first. --- #
python -m pytest tests/ -q

# --- 5. Run the eval suite against all three target models. --- #
mkdir -p /workspace/results
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "meta-llama/Llama-3.2-1B"
    "microsoft/Phi-3-mini-4k-instruct"
)

for MODEL in "${MODELS[@]}"; do
    SAFE_NAME="$(echo "$MODEL" | tr '/' '_')"
    LOG="/workspace/results/${SAFE_NAME}.log"
    echo "=== $MODEL -> $LOG ===" | tee -a "$LOG"

    { python scripts/eval_memory_throughput.py --model "$MODEL" \
        --prompt-lengths 512 2048 8192 --max-new-tokens 128
      python scripts/eval_perplexity.py --model "$MODEL" \
        --seq-len 2048 --n-sequences 10
      python scripts/eval_k_granularity_ablation.py --model "$MODEL" --seq-len 1024
      python scripts/eval_niah.py --model "$MODEL" \
        --context-lengths 1000 4000 8000
      python scripts/eval_longbench.py --model "$MODEL" \
        --task narrativeqa --n-samples 30
    } >> "$LOG" 2>&1 || echo "!! $MODEL run hit an error -- check $LOG"

    echo "=== done: $MODEL ==="
done

echo
echo "All logs in /workspace/results/. Copy them back with, e.g.:"
echo "  runpodctl send /workspace/results   (or rsync/scp, see pod's Connect tab)"
