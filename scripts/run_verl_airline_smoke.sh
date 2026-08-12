#!/bin/sh
set -eu

gpu_list=${1:-3}
profile=${VERL_AIRLINE_PROFILE:-current}
rollout_n=${2:-}
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Python 3.12 环境使用 vLLM 自带 FA2；项目 src/flash_attn 只补 veRL 的 bert_padding。
python_bin=${VERL_PYTHON:-/data/raoxinlong/runtime/conda_envs/verl312/bin/python}
# 多轮轨迹同时保存 Action、User 和 Tool token；正式 response budget 为 24576。
rollout_max_steps=${VERL_AIRLINE_MAX_STEPS:-}
# 32K 上下文至少需要约 3.5 GiB KV cache；0.1 在共享 A800 上可能只分到约 3.1 GiB。
# 保持 0.15，仍给 FSDP actor 留出显存；可通过环境变量进一步覆盖。
case "$profile" in
  current)
    default_rollout_n=2
    default_gpu_memory_utilization=0.15
    default_test_frequency=1
    default_save_frequency=1
    default_max_actor_checkpoints=1
    default_reward_mode=prm_lite_v1
    default_adv_estimator=grpo_lata
    export VERL_AIRLINE_TRAIN_BATCH_SIZE="${VERL_AIRLINE_TRAIN_BATCH_SIZE:-1}"
    export VERL_AIRLINE_PROMPT_LENGTH="${VERL_AIRLINE_PROMPT_LENGTH:-6144}"
    export VERL_AIRLINE_RESPONSE_LENGTH="${VERL_AIRLINE_RESPONSE_LENGTH:-24576}"
    export VERL_AIRLINE_MAX_MODEL_LENGTH="${VERL_AIRLINE_MAX_MODEL_LENGTH:-32768}"
    export VERL_AIRLINE_PPO_MINI_BATCH_SIZE="${VERL_AIRLINE_PPO_MINI_BATCH_SIZE:-1}"
    export VERL_AIRLINE_PPO_MICRO_BATCH_SIZE="${VERL_AIRLINE_PPO_MICRO_BATCH_SIZE:-1}"
    export VERL_AIRLINE_LOGPROB_MICRO_BATCH_SIZE="${VERL_AIRLINE_LOGPROB_MICRO_BATCH_SIZE:-1}"
    export VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH="${VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH:-32768}"
    export VERL_AIRLINE_PPO_EPOCHS="${VERL_AIRLINE_PPO_EPOCHS:-2}"
    export VERL_AIRLINE_LEARNING_RATE="${VERL_AIRLINE_LEARNING_RATE:-1.0e-6}"
    export VERL_AIRLINE_MAX_USER_TURNS="${VERL_AIRLINE_MAX_USER_TURNS:-40}"
    export VERL_AIRLINE_MAX_ASSISTANT_TURNS="${VERL_AIRLINE_MAX_ASSISTANT_TURNS:-40}"
    ;;
  reference_single)
    # 复用参考项目 Vanilla GRPO 的显式超参；TP=2 在单卡不可用，故固定为单卡
    # smoke。fused-kernel 单独做兼容性 A/B，避免把 FlashAttention 版本问题混入
    # bypass 显存对照。
    default_rollout_n=8
    default_gpu_memory_utilization=0.50
    # 参考 Vanilla 的 test_freq=100；单步显存 smoke 不额外跑验证 rollout。
    default_test_frequency=0
    default_save_frequency=1
    default_max_actor_checkpoints=1
    # 对齐参考 Vanilla GRPO；PRM-Lite 与 LATA 属于其后续消融实验。
    default_reward_mode=terminal_v4
    default_adv_estimator=grpo
    export VERL_AIRLINE_TRAIN_BATCH_SIZE="${VERL_AIRLINE_TRAIN_BATCH_SIZE:-4}"
    export VERL_AIRLINE_PROMPT_LENGTH="${VERL_AIRLINE_PROMPT_LENGTH:-8192}"
    export VERL_AIRLINE_RESPONSE_LENGTH="${VERL_AIRLINE_RESPONSE_LENGTH:-12288}"
    export VERL_AIRLINE_MAX_MODEL_LENGTH="${VERL_AIRLINE_MAX_MODEL_LENGTH:-24576}"
    export VERL_AIRLINE_PPO_MINI_BATCH_SIZE="${VERL_AIRLINE_PPO_MINI_BATCH_SIZE:-4}"
    export VERL_AIRLINE_PPO_MICRO_BATCH_SIZE="${VERL_AIRLINE_PPO_MICRO_BATCH_SIZE:-2}"
    export VERL_AIRLINE_LOGPROB_MICRO_BATCH_SIZE="${VERL_AIRLINE_LOGPROB_MICRO_BATCH_SIZE:-4}"
    export VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH="${VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH:-24576}"
    export VERL_AIRLINE_PPO_EPOCHS="${VERL_AIRLINE_PPO_EPOCHS:-1}"
    export VERL_AIRLINE_LEARNING_RATE="${VERL_AIRLINE_LEARNING_RATE:-5.0e-6}"
    export VERL_AIRLINE_MAX_USER_TURNS="${VERL_AIRLINE_MAX_USER_TURNS:-15}"
    export VERL_AIRLINE_MAX_ASSISTANT_TURNS="${VERL_AIRLINE_MAX_ASSISTANT_TURNS:-15}"
    ;;
  formal)
    # 正式实验配置：保留已验证的 32K 长轨迹预算，并采用购物项目同级的
    # batch=2、G=4、mini-batch=2/micro-batch=1。500 个 global step 对应
    # 500 次 policy 参数更新；20 个 epoch 只用于数据迭代规划，实际停止点
    # 由 total_training_steps 决定。
    default_rollout_n=4
    default_gpu_memory_utilization=0.20
    default_test_frequency=0
    default_save_frequency=50
    default_max_actor_checkpoints=10
    default_reward_mode=prm_lite_v1
    default_adv_estimator=grpo_lata
    export VERL_AIRLINE_TRAIN_BATCH_SIZE="${VERL_AIRLINE_TRAIN_BATCH_SIZE:-2}"
    export VERL_AIRLINE_PROMPT_LENGTH="${VERL_AIRLINE_PROMPT_LENGTH:-8192}"
    export VERL_AIRLINE_RESPONSE_LENGTH="${VERL_AIRLINE_RESPONSE_LENGTH:-24576}"
    export VERL_AIRLINE_MAX_MODEL_LENGTH="${VERL_AIRLINE_MAX_MODEL_LENGTH:-32768}"
    export VERL_AIRLINE_PPO_MINI_BATCH_SIZE="${VERL_AIRLINE_PPO_MINI_BATCH_SIZE:-2}"
    export VERL_AIRLINE_PPO_MICRO_BATCH_SIZE="${VERL_AIRLINE_PPO_MICRO_BATCH_SIZE:-1}"
    export VERL_AIRLINE_LOGPROB_MICRO_BATCH_SIZE="${VERL_AIRLINE_LOGPROB_MICRO_BATCH_SIZE:-1}"
    export VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH="${VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH:-32768}"
    export VERL_AIRLINE_PPO_EPOCHS="${VERL_AIRLINE_PPO_EPOCHS:-1}"
    export VERL_AIRLINE_LEARNING_RATE="${VERL_AIRLINE_LEARNING_RATE:-5.0e-6}"
    export VERL_AIRLINE_MAX_USER_TURNS="${VERL_AIRLINE_MAX_USER_TURNS:-30}"
    export VERL_AIRLINE_MAX_ASSISTANT_TURNS="${VERL_AIRLINE_MAX_ASSISTANT_TURNS:-30}"
    export VERL_AIRLINE_TOTAL_EPOCHS="${VERL_AIRLINE_TOTAL_EPOCHS:-20}"
    ;;
  *)
    printf '%s\n' "训练 profile 必须是 current、reference_single 或 formal" >&2
    exit 2
    ;;
esac

rollout_n=${rollout_n:-$default_rollout_n}
max_num_seqs=${VERL_AIRLINE_MAX_NUM_SEQS:-$rollout_n}
rollout_gpu_memory_utilization=${VERL_AIRLINE_GPU_MEMORY_UTILIZATION:-$default_gpu_memory_utilization}
dataset_mode=${VERL_AIRLINE_DATASET_MODE:-smoke}
training_updates=${VERL_AIRLINE_UPDATES:-1}
reward_mode=${VERL_AIRLINE_REWARD_MODE:-$default_reward_mode}
adv_estimator=${VERL_AIRLINE_ADV_ESTIMATOR:-$default_adv_estimator}
bypass_mode=${VERL_AIRLINE_BYPASS_MODE:-true}
test_frequency=${VERL_AIRLINE_TEST_FREQUENCY:-$default_test_frequency}
save_frequency=${VERL_AIRLINE_SAVE_FREQUENCY:-$default_save_frequency}
max_actor_checkpoints=${VERL_AIRLINE_MAX_ACTOR_CKPTS:-$default_max_actor_checkpoints}
validation_scope='tau2-airline-6,tau2-airline-45'

case "$dataset_mode" in
  smoke)
    rollout_max_steps=${rollout_max_steps:-4}
    ;;
  full)
    # 原始任务与 v8/v9 变体合计 45 条；正式 profile 的 batch=2 下，45 个
    # task 约构成一轮数据暴露。正式 profile 默认训练 500 个 global step；
    # 其余 profile 仍保持 45 step 的快速全量检查。
    if [ "$profile" = "formal" ]; then
      training_updates=${VERL_AIRLINE_UPDATES:-500}
    else
      training_updates=${VERL_AIRLINE_UPDATES:-45}
    fi
    rollout_max_steps=${rollout_max_steps:-30}
    # 冻结 test 在训练结束后单独评测；训练中反复验证既慢也会污染调参流程。
    test_frequency=0
    validation_scope='full_supported_test_deferred'
    ;;
  *)
    printf '%s\n' "数据模式必须是 smoke 或 full" >&2
    exit 2
    ;;
esac

case "$reward_mode" in
  terminal_v4|prm_lite_v1) ;;
  *)
    printf '%s\n' "奖励模式必须是 terminal_v4 或 prm_lite_v1" >&2
    exit 2
    ;;
esac
case "$adv_estimator" in
  grpo|grpo_lata) ;;
  *)
    printf '%s\n' "优势估计必须是 grpo 或 grpo_lata" >&2
    exit 2
    ;;
esac
case "$bypass_mode" in
  true|false) ;;
  *)
    printf '%s\n' "VERL_AIRLINE_BYPASS_MODE 必须是 true 或 false" >&2
    exit 2
    ;;
esac
case "$save_frequency" in
  ''|*[!0-9]*)
    printf '%s\n' "VERL_AIRLINE_SAVE_FREQUENCY 必须是非负整数" >&2
    exit 2
    ;;
esac
case "$max_actor_checkpoints" in
  ''|*[!0-9]*|0)
    printf '%s\n' "VERL_AIRLINE_MAX_ACTOR_CKPTS 必须是正整数" >&2
    exit 2
    ;;
esac
export VERL_AIRLINE_REWARD_MODE="$reward_mode"
export VERL_AIRLINE_ADV_ESTIMATOR="$adv_estimator"
export VERL_AIRLINE_UPDATES="$training_updates"
export VERL_AIRLINE_SAVE_FREQUENCY="$save_frequency"
export VERL_AIRLINE_MAX_ACTOR_CKPTS="$max_actor_checkpoints"
export VERL_AIRLINE_TEST_FREQUENCY="$test_frequency"

# CUDA_VISIBLE_DEVICES 中的每张物理卡对应一个 veRL worker。
case "$gpu_list" in
  ''|*[!0-9,]*)
    printf '%s\n' "GPU 列表格式错误，例如：3 或 3,5" >&2
    exit 2
    ;;
esac
gpu_count=$(printf '%s' "$gpu_list" | awk -F, '{print NF}')

case "$rollout_n" in
  ''|*[!0-9]*|0)
    printf '%s\n' "rollout 数量必须是正整数，例如：sh scripts/run_verl_airline_smoke.sh 3,5 2" >&2
    exit 2
    ;;
esac
case "$max_num_seqs" in
  ''|*[!0-9]*|0)
    printf '%s\n' "VERL_AIRLINE_MAX_NUM_SEQS 必须是正整数" >&2
    exit 2
    ;;
esac
if [ "$max_num_seqs" -gt "$rollout_n" ]; then
  printf '%s\n' "VERL_AIRLINE_MAX_NUM_SEQS 不能大于 rollout 数量" >&2
  exit 2
fi

# veRL 要求 rollout 组大小能被 GPU worker 数整除。
if [ $((rollout_n % gpu_count)) -ne 0 ]; then
  printf '%s\n' "rollout 数量必须能被 GPU 数量整除：GPU=$gpu_count, rollout=$rollout_n" >&2
  printf '%s\n' "例如三张卡请使用 rollout=3；若要使用 rollout=4，需要同时增大 data.train_batch_size。" >&2
  exit 2
fi

if ! "$python_bin" -c 'import verl' >/dev/null 2>&1; then
  printf '%s\n' "当前 Python 环境没有安装 veRL；请在独立 veRL 环境中设置 VERL_PYTHON。" >&2
  exit 2
fi

cd "$project_dir"
# Ray worker 会继承启动进程的环境变量；显式加载项目 .env，确保本轮 USER
# 配置来自当前文件，而不是复用终端此前 export 的旧模型名。
if [ -f "$project_dir/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$project_dir/.env"
  set +a
fi
export VERL_AIRLINE_ROOT="$project_dir"
if [ -z "${VERL_AIRLINE_CHAT_TEMPLATE:-}" ]; then
  VERL_AIRLINE_CHAT_TEMPLATE=$(cat "$project_dir/configs/qwen3_nonthinking_prefix_preserving.jinja")
fi
export VERL_AIRLINE_CHAT_TEMPLATE
template_hash=$(sha256sum "$project_dir/configs/qwen3_nonthinking_prefix_preserving.jinja")
template_hash=${template_hash%% *}
# Ray 同时需要临时对象空间和 Unix socket；/dev/shm 当前有充足空间，且路径足够短。
# 允许通过 RAY_TMPDIR 覆盖，方便在不同节点使用本地高速临时盘。
ray_tmpdir="${RAY_TMPDIR:-/dev/shm/airline-ray}"
mkdir -p "$ray_tmpdir"
export RAY_TMPDIR="$ray_tmpdir"
printf '%s\n' "Ray temporary directory: $RAY_TMPDIR"
export VERL_AIRLINE_TRAIN_FILE="${VERL_AIRLINE_TRAIN_FILE:-$project_dir/data/verl/airline_train.parquet}"
export VERL_AIRLINE_VAL_FILE="${VERL_AIRLINE_VAL_FILE:-$project_dir/data/verl/airline_val.parquet}"
base_model="${VERL_AIRLINE_BASE_MODEL:-$project_dir/outputs/models/sft-merged-v3}"
export VERL_AIRLINE_MODEL="${VERL_AIRLINE_MODEL:-$project_dir/outputs/models/sft-merged-v3-verl}"
export VERL_AIRLINE_OUTPUT_DIR="${VERL_AIRLINE_OUTPUT_DIR:-$project_dir/outputs/verl_airline_${dataset_mode}_n$rollout_n}"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export VERL_AIRLINE_OLD_LOGPROB_TRACE_DIR="$VERL_AIRLINE_OUTPUT_DIR/actor_old_logprob"

# veRL checkpoint 保存的是 actor、optimizer、dataloader 和 global step。默认自动
# 寻找同一输出目录下最新 global_step_*，因此作业被抢占后只会损失上一个 save_freq
# 之后的完整 update，不能恢复正在执行中的半个 backward。
resume_mode=${VERL_AIRLINE_RESUME_MODE:-auto}
resume_from_path=${VERL_AIRLINE_RESUME_FROM_PATH:-}
case "$resume_mode" in
  auto|disable|resume_path) ;;
  *)
    printf '%s\n' "VERL_AIRLINE_RESUME_MODE 必须是 auto、disable 或 resume_path" >&2
    exit 2
    ;;
esac

find_latest_checkpoint() {
  find "$VERL_AIRLINE_OUTPUT_DIR" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
    | awk -F_ 'NF == 3 && $3 ~ /^[0-9]+$/ {print $3}' \
    | sort -n \
    | tail -n 1
}

resume_checkpoint=''
case "$resume_mode" in
  auto)
    if [ -n "$resume_from_path" ]; then
      resume_checkpoint=$resume_from_path
    else
      latest_step=$(find_latest_checkpoint)
      if [ -n "$latest_step" ]; then
        resume_checkpoint="$VERL_AIRLINE_OUTPUT_DIR/global_step_$latest_step"
      fi
    fi
    ;;
  resume_path)
    if [ -z "$resume_from_path" ]; then
      printf '%s\n' "resume_path 模式必须设置 VERL_AIRLINE_RESUME_FROM_PATH" >&2
      exit 2
    fi
    resume_checkpoint=$resume_from_path
    ;;
  disable)
    ;;
esac

if [ -n "$resume_checkpoint" ]; then
  if [ ! -d "$resume_checkpoint" ] || [ "${resume_checkpoint#*global_step_}" = "$resume_checkpoint" ]; then
    printf '%s\n' "恢复目录必须是存在的 global_step_* checkpoint：$resume_checkpoint" >&2
    exit 2
  fi
  # FSDP checkpoint 按 world size 分片保存。例如双卡会产生
  # model_world_size_2_rank_0.pt 和 rank_1.pt，不能用单卡 worker 直接恢复。
  # 物理 GPU 编号可以变化，但本轮可见 GPU 数必须与保存时一致。
  checkpoint_state_file=$(find "$resume_checkpoint/actor" -maxdepth 1 -type f \
    -name 'model_world_size_*_rank_0.pt' -printf '%f\n' 2>/dev/null | head -n 1)
  checkpoint_world_size=$(printf '%s' "$checkpoint_state_file" \
    | sed -n 's/^model_world_size_\([0-9][0-9]*\)_rank_0\.pt$/\1/p')
  if [ -z "$checkpoint_world_size" ]; then
    printf '%s\n' "checkpoint 缺少 actor/model_world_size_*_rank_0.pt，不能安全恢复：$resume_checkpoint" >&2
    exit 2
  fi
  if [ "$checkpoint_world_size" -ne "$gpu_count" ]; then
    printf '%s\n' "checkpoint 使用 $checkpoint_world_size 张 GPU 保存，但本轮配置了 $gpu_count 张 GPU。FSDP 分片不能跨 world size 直接恢复。" >&2
    printf '%s\n' "请用相同数量的 GPU 重启，例如双卡 checkpoint 使用：sh scripts/run_verl_airline_smoke.sh 2,5 4" >&2
    exit 2
  fi
  export VERL_AIRLINE_RESUME_FROM_PATH="$resume_checkpoint"
  printf '%s\n' "检测到 $checkpoint_world_size 卡 checkpoint，将从此恢复：$resume_checkpoint"
else
  unset VERL_AIRLINE_RESUME_FROM_PATH || true
  printf '%s\n' "未检测到 checkpoint，将从 SFT 初始权重开始。"
fi

"$python_bin" scripts/prepare_verl_model.py \
  --source "$base_model" \
  --output "$VERL_AIRLINE_MODEL"

# parquet 的 extra_info 会保存 max_steps，因此每次运行都重建数据，避免复用旧预算。
train_data_args="--tasks data/tasks/train.jsonl --tasks data/tasks/variants/v8_train.jsonl --tasks data/tasks/variants/v9_train.jsonl --output $VERL_AIRLINE_TRAIN_FILE --max-steps $rollout_max_steps"
if [ "$dataset_mode" = "smoke" ]; then
  if [ "$profile" = "reference_single" ]; then
    # reference profile 的 train_batch_size=4，需要至少四条任务组成一个真实 batch。
    train_data_args="$train_data_args --task-id tau2-airline-0 --task-id tau2-airline-9 --task-id tau2-airline-23 --task-id tau2-airline-33"
  else
    train_data_args="$train_data_args --task-id tau2-airline-0 --task-id tau2-airline-9"
  fi
fi
# 这里仅拼接脚本自身已校验的路径和整数参数，不接收模型输出或用户输入。
# shellcheck disable=SC2086
"$python_bin" scripts/build_verl_airline_data.py $train_data_args
# smoke 的目标是验证多卡 AgentLoop、reward、GRPO loss 和 checkpoint 是否打通，
# 不应在每个 update 后遍历完整冻结 test 集。选两条短的只读/拒绝任务作为验证样本。
# full 模式仍导出完整 test parquet，但训练期间关闭验证；结束后走独立评测入口。
val_data_args="--tasks data/tasks/test.jsonl --output $VERL_AIRLINE_VAL_FILE --max-steps $rollout_max_steps"
if [ "$dataset_mode" = "smoke" ]; then
  val_data_args="$val_data_args --task-id tau2-airline-6 --task-id tau2-airline-45"
fi
# shellcheck disable=SC2086
"$python_bin" scripts/build_verl_airline_data.py $val_data_args

mkdir -p "$VERL_AIRLINE_OUTPUT_DIR"
# 保存本次运行的关键配置，避免只依赖终端里的 Hydra 覆盖参数。
cat > "$VERL_AIRLINE_OUTPUT_DIR/run_config.json" <<EOF
{
  "python": "$python_bin",
  "gpu": "$gpu_list",
  "profile": "$profile",
  "trainer_gpu_count": $gpu_count,
  "rollout_n": $rollout_n,
  "rollout_max_num_seqs": $max_num_seqs,
  "dataset_mode": "$dataset_mode",
  "validation_scope": "$validation_scope",
  "test_frequency": $test_frequency,
  "save_frequency": $save_frequency,
  "max_actor_checkpoints": $max_actor_checkpoints,
  "max_steps": $rollout_max_steps,
  "gpu_memory_utilization": $rollout_gpu_memory_utilization,
  "base_model": "$base_model",
  "policy_model": "$VERL_AIRLINE_MODEL",
  "chat_template_mode": "qwen3_nonthinking_prefix_preserving_v1",
  "chat_template_sha256": "$template_hash",
  "train_file": "$VERL_AIRLINE_TRAIN_FILE",
  "val_file": "$VERL_AIRLINE_VAL_FILE",
  "train_batch_size": $VERL_AIRLINE_TRAIN_BATCH_SIZE,
  "prompt_length": $VERL_AIRLINE_PROMPT_LENGTH,
  "response_length": $VERL_AIRLINE_RESPONSE_LENGTH,
  "max_action_tokens": 512,
  "max_model_len": $VERL_AIRLINE_MAX_MODEL_LENGTH,
  "ppo_max_token_len_per_gpu": $VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH,
  "log_prob_max_token_len_per_gpu": $VERL_AIRLINE_PPO_MAX_TOKEN_LENGTH,
  "lora_rank": 16,
  "lora_alpha": 32,
  "sft_lora_dropout": 0.05,
  "learning_rate": $VERL_AIRLINE_LEARNING_RATE,
  "reward_mode": "$reward_mode",
  "adv_estimator": "$adv_estimator",
  "bypass_mode": $bypass_mode,
  "resume_mode": "$resume_mode",
  "resume_checkpoint": "${resume_checkpoint:-}",
  "actor_old_logprob_trace_dir": "$VERL_AIRLINE_OLD_LOGPROB_TRACE_DIR",
  "lata_alpha": 1.05,
  "ppo_mini_batch_size": $VERL_AIRLINE_PPO_MINI_BATCH_SIZE,
  "ppo_micro_batch_size_per_gpu": $VERL_AIRLINE_PPO_MICRO_BATCH_SIZE,
  "log_prob_micro_batch_size_per_gpu": $VERL_AIRLINE_LOGPROB_MICRO_BATCH_SIZE,
  "ppo_epochs": $VERL_AIRLINE_PPO_EPOCHS,
  "total_epochs": ${VERL_AIRLINE_TOTAL_EPOCHS:-1},
  "total_training_steps": $training_updates
}
EOF

# 每秒采样一次 GPU 状态；nvidia-smi 不可用时保留空文件，不阻断训练。
gpu_samples="$VERL_AIRLINE_OUTPUT_DIR/gpu_samples.csv"
: > "$gpu_samples"
gpu_sampler_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while :; do
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' -v ts="$(date +%s)" '{print ts "," $0}' >> "$gpu_samples" || true
      sleep 1
    done
  ) &
  gpu_sampler_pid=$!
fi

started_at=$(date +%s)
status=0
set -- \
  "$python_bin" -m airline_agent.verl_train \
  --config-path="$project_dir/configs" \
  --config-name=verl_airline_grpo \
  trainer.n_gpus_per_node="$gpu_count" \
  trainer.nnodes=1 \
  actor_rollout_ref.rollout.n="$rollout_n" \
  actor_rollout_ref.rollout.max_num_seqs="$max_num_seqs" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_memory_utilization" \
  algorithm.rollout_correction.bypass_mode="$bypass_mode" \
  trainer.resume_mode="$resume_mode" \
  trainer.test_freq="$test_frequency" \
  trainer.save_freq="$save_frequency" \
  trainer.max_actor_ckpt_to_keep="$max_actor_checkpoints" \
  trainer.total_training_steps="$training_updates"
if [ -n "$resume_checkpoint" ]; then
  set -- "$@" "trainer.resume_from_path=$resume_checkpoint"
fi
CUDA_VISIBLE_DEVICES="$gpu_list" "$@" || status=$?
finished_at=$(date +%s)

if [ -n "$gpu_sampler_pid" ]; then
  kill "$gpu_sampler_pid" 2>/dev/null || true
  wait "$gpu_sampler_pid" 2>/dev/null || true
fi

# 将采样结果压缩成每张卡的峰值，便于比较不同 worker/rollout 配置。
"$python_bin" - "$gpu_samples" "$VERL_AIRLINE_OUTPUT_DIR/gpu_peak.json" <<'PY'
import csv
import json
import sys
from collections import defaultdict

samples = defaultdict(list)
try:
    with open(sys.argv[1], newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            timestamp, index, used, free, utilization = row[:5]
            try:
                samples[index.strip()].append({
                    "timestamp": int(timestamp),
                    "memory_used_mb": int(used.strip()),
                    "memory_free_mb": int(free.strip()),
                    "utilization_gpu_percent": int(utilization.strip()),
                })
            except ValueError:
                continue
except OSError:
    pass

peak = {}
for index, values in samples.items():
    memory_peak = max(v["memory_used_mb"] for v in values)
    peak[index] = {
        "memory_used_mb_start": values[0]["memory_used_mb"],
        "memory_used_mb_peak": memory_peak,
        "memory_used_mb_incremental_peak": memory_peak - values[0]["memory_used_mb"],
        "memory_free_mb_min": min(v["memory_free_mb"] for v in values),
        "utilization_gpu_percent_peak": max(v["utilization_gpu_percent"] for v in values),
        "samples": len(values),
    }

with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump({"available": bool(peak), "gpu_peak": peak}, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

cat > "$VERL_AIRLINE_OUTPUT_DIR/run_timing.json" <<EOF
{
  "gpu": "$gpu_list",
  "trainer_gpu_count": $gpu_count,
  "rollout_n": $rollout_n,
  "started_at_epoch": $started_at,
  "finished_at_epoch": $finished_at,
  "elapsed_seconds": $((finished_at - started_at)),
  "exit_code": $status
}
EOF

exit "$status"
