#!/usr/bin/env bash
# Stand-alone vLLM server launcher.
# Usage:  launch_vllm_server.sh [PORT] [QUANT]
#   PORT  – port d'écoute (défaut 8000)
#   QUANT – méthode de quantification (ex: int4_w4a16, fp8). Peut aussi être fournie via VLLM_QUANT.

set -euo pipefail

PORT=${1:-8000}
QUANT=${2:-${VLLM_QUANT:-""}}
HOST="0.0.0.0"
MODEL_PATH="results/step-1_SFT_fused"

# Réglages mémoire
GPU_UTIL=0.60          # xx % VRAM pour vLLM
MAX_CTX=32768           # context-length pour le KV-cache

CMD=(python3 -m trainers.vllm_server
  --model "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --gpu_memory_utilization "${GPU_UTIL}"
  --max_model_len "${MAX_CTX}"
  --enable_prefix_caching false)

# Ajouter la quantification si demandée
if [[ -n "${QUANT}" ]]; then
  CMD+=(--quantization "${QUANT}")
fi

echo "[launch_vllm_server] → ${CMD[*]}"
exec "${CMD[@]}" 