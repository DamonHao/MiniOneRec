#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/minionerec-uv-cache}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/local_mps/sft-dry-run}"

TRAIN_FILE="data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
EVAL_FILE="data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"
SID_INDEX_PATH="data/Amazon/index/Industrial_and_Scientific.index.json"
ITEM_META_PATH="data/Amazon/index/Industrial_and_Scientific.item.json"

for required_file in \
  "${TRAIN_FILE}" \
  "${EVAL_FILE}" \
  "${SID_INDEX_PATH}" \
  "${ITEM_META_PATH}"
do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

echo "Starting MiniOneRec SFT dry run"
echo "Base model: ${BASE_MODEL}"
echo "Output: ${OUTPUT_DIR}"

uv run python sft_mps.py \
  --base_model "${BASE_MODEL}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --category Industrial_and_Scientific \
  --sid_index_path "${SID_INDEX_PATH}" \
  --item_meta_path "${ITEM_META_PATH}" \
  --sample 8 \
  --num_epochs 1 \
  --cutoff_len 192 \
  --micro_batch_size 1 \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --freeze_LLM=True \
  --gradient_checkpointing=False \
  --device mps \
  --dtype float32
