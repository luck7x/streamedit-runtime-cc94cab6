#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STREAMEDIT_CONDA_SH="${STREAMEDIT_CONDA_SH:-}"
STREAMEDIT_CONDA_ENV="${STREAMEDIT_CONDA_ENV:-}"
if [ -n "$STREAMEDIT_CONDA_ENV" ]; then
  : "${NVCC_PREPEND_FLAGS:=}" "${NVCC_APPEND_FLAGS:=}"
  export NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
  if [ -n "$STREAMEDIT_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$STREAMEDIT_CONDA_SH"
  fi
  while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
  conda activate "$STREAMEDIT_CONDA_ENV"
  hash -r
fi

cd "$HERE"

STREAMEDIT_CACHE_ROOT="${STREAMEDIT_CACHE_ROOT:-$HERE/deps/cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$STREAMEDIT_CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STREAMEDIT_CACHE_ROOT/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$STREAMEDIT_CACHE_ROOT/nv_compute}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HERE"

export STREAMEDIT_FP8_IMG="${STREAMEDIT_FP8_IMG:-1}"
export STREAMEDIT_FP8_TXT="${STREAMEDIT_FP8_TXT:-1}"
export STREAMEDIT_CUDA_GRAPH="${STREAMEDIT_CUDA_GRAPH:-1}"
export STREAMEDIT_SAGE_ATTN="${STREAMEDIT_SAGE_ATTN:-0}"
export STREAMEDIT_FP8_FAST_ACCUM="${STREAMEDIT_FP8_FAST_ACCUM:-0}"
export STREAMEDIT_LOW_VRAM="${STREAMEDIT_LOW_VRAM:-0}"

# Prompt enhancement (empty -> disabled, the raw user prompt is used as-is)
export PE_MODEL="${PE_MODEL:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

RECORD_DIR="${STREAMEDIT_RECORD_DIR:-$HERE/recordings}"

CKPT_ROOT="${STREAMEDIT_CKPT_ROOT:-$HERE/deps/checkpoints}"
DIT_CKPT="${STREAMEDIT_DIT_CKPT:-$CKPT_ROOT/StreamEdit/dit/streamedit_dit_0811.pth}"
VAE_CKPT="${STREAMEDIT_VAE_CKPT:-$CKPT_ROOT/StreamEdit/vae}"
TE_CKPT="${STREAMEDIT_TEXT_ENCODER_CKPT:-$CKPT_ROOT/MiMo-VL-7B-RL-2508}"
FACE_ONNX="${STREAMEDIT_FACE_ONNX:-$CKPT_ROOT/face_detection_yunet_2023mar.onnx}"
PERSON_ONNX="${STREAMEDIT_PERSON_ONNX:-$CKPT_ROOT/yolov8n.onnx}"

DEVICE="${STREAMEDIT_DEVICE:-cuda:0}"
HOST="${STREAMEDIT_HOST:-0.0.0.0}"
PORT="${STREAMEDIT_PORT:-8080}"

python xvideo/serving/serve_streamedit_streaming.py \
  --dit-ckpt          "$DIT_CKPT" \
  --vae-ckpt          "$VAE_CKPT" \
  --text-encoder-ckpt "$TE_CKPT" \
  --face-detector-onnx   "$FACE_ONNX" \
  --person-detector-onnx "$PERSON_ONNX" \
  --record-dir "$RECORD_DIR" \
  --device "$DEVICE" \
  --vae-encode-device "$DEVICE" \
  --vae-decode-device "$DEVICE" \
  --vae-pseudo-device "$DEVICE" \
  --postprocess-device "$DEVICE" \
  --width "${STREAMEDIT_WIDTH:-840}" --height "${STREAMEDIT_HEIGHT:-480}" \
  --fps "${STREAMEDIT_FPS:-24}" \
  --pe-timeout-s "${STREAMEDIT_PE_TIMEOUT_S:-60}" \
  --host "$HOST" --port "$PORT" \
  "$@"
