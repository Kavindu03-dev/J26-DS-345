#!/usr/bin/env bash
# Serve one model on port 8000. Usage: scripts/serve_vllm.sh [MODEL] [extra vllm args]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
shift || true

if [[ "$(uname -r)" == *icrosoft* ]]; then
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/c4-agent}"
fi

# Gated models need HF_TOKEN. Read only that key; never print it.
if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' .env | tr -d '\r' | tail -n 1)"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then export HF_TOKEN; else unset HF_TOKEN; fi

# --generation-config vllm: ignore the model's own sampling defaults (Qwen2.5 ships a
# repetition_penalty), so temperature 0 really is plain greedy decoding (ground rule 4).
exec uv run --no-sync vllm serve "$MODEL" \
  --host 127.0.0.1 --port "${VLLM_PORT:-8000}" \
  --seed 42 \
  --generation-config vllm \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}" \
  "$@"
