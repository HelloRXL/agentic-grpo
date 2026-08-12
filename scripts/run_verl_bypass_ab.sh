#!/bin/sh
# 在同一组空闲 GPU 上顺序运行 bypass=false/true，隔离显存优化的单一变量。
set -eu

gpu_list=${1:?用法: sh scripts/run_verl_bypass_ab.sh <GPU列表>}
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output_root=${VERL_AIRLINE_BYPASS_AB_OUTPUT_DIR:-"$project_dir/outputs/bypass_ab_$(date +%Y%m%d_%H%M%S)"}
max_steps=${VERL_AIRLINE_MAX_STEPS:-30}
python_bin=${VERL_PYTHON:-/data/raoxinlong/runtime/conda_envs/verl312/bin/python}
# 单卡下 actor 与 vLLM 共卡，32K KV cache 容易导致 vLLM 无法启动。这个专用
# 显存 smoke 固定为 8K prompt + 16K response = 24K 总上下文；正式训练仍由
# formal profile 维持 32K，且可通过环境变量覆盖以下三个预算。
max_model_length=${VERL_AIRLINE_MAX_MODEL_LENGTH:-24576}
response_length=${VERL_AIRLINE_RESPONSE_LENGTH:-16384}
ppo_max_token_length=${VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH:-24576}

run_one() {
  mode=$1
  output_dir="$output_root/$mode"
  VERL_AIRLINE_PROFILE=formal \
  VERL_AIRLINE_DATASET_MODE=smoke \
  VERL_AIRLINE_UPDATES=1 \
  VERL_AIRLINE_TOTAL_EPOCHS=1 \
  VERL_AIRLINE_MAX_STEPS="$max_steps" \
  VERL_AIRLINE_MAX_MODEL_LENGTH="$max_model_length" \
  VERL_AIRLINE_RESPONSE_LENGTH="$response_length" \
  VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH="$ppo_max_token_length" \
  VERL_AIRLINE_SAVE_FREQUENCY=0 \
  VERL_AIRLINE_TEST_FREQUENCY=0 \
  VERL_AIRLINE_BYPASS_MODE="$mode" \
  VERL_AIRLINE_OUTPUT_DIR="$output_dir" \
    sh "$project_dir/scripts/run_verl_airline_smoke.sh" "$gpu_list" 4
}

run_one false
run_one true

"$python_bin" "$project_dir/scripts/compare_bypass_memory.py" \
  --off "$output_root/false" \
  --on "$output_root/true" \
  --output "$output_root/comparison.json"
