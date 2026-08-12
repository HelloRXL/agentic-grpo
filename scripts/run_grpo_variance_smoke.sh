#!/bin/sh
set -eu

gpu=${1:-3}
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

CUDA_VISIBLE_DEVICES="$gpu" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src \
/data/raoxinlong/runtime/conda_envs/dov_mhstigma_ace/bin/python \
  -m airline_agent.grpo_train \
  --model outputs/models/sft-merged-v2 \
  --tasks data/tasks/train.jsonl \
  --task-id tau2-airline-23 \
  --task-id tau2-airline-33 \
  --user-prefix USER \
  --group-size 4 \
  --updates 1 \
  --max-steps 30 \
  --max-new-tokens 1024 \
  --temperature 0.8 \
  --output-dir outputs/grpo_variance_v1
