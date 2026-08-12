#!/bin/sh
set -eu

# 用法：sh scripts/wait_for_gpu_and_serve_qwen36_27b.sh [最低空闲GiB] [检查间隔秒] [端口]
#
# 仅在一张卡的空闲显存达到阈值后启动 vLLM。它不会终止其他人的进程，也无法避免
# “检测通过后、启动前被其他任务抢走显存”的竞态，因此启动前会再检查一次。
min_free_gib=${1:-70}
interval=${2:-10}
port=${3:-8002}

vllm_bin=${VLLM_QWEN_BIN:-/data/raoxinlong/runtime/conda_envs/vllm-qwen/bin/vllm}
model_path=${QWEN36_27B_MODEL:-/data/raoxinlong/model_cache/Qwen3.6-27B}
served_model_name=${QWEN36_27B_NAME:-qwen3.6-27b}
threshold_mib=$((min_free_gib * 1024))

case "$min_free_gib:$interval:$port" in
  *[!0-9:]*|*::*|0:*|*:0:*|*:*:0)
    printf '%s\n' '参数必须是正整数，例如：sh scripts/wait_for_gpu_and_serve_qwen36_27b.sh 70 15 8002' >&2
    exit 2
    ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' '未找到 nvidia-smi，无法检测 GPU 显存。' >&2
  exit 1
fi
if [ ! -x "$vllm_bin" ]; then
  printf '找不到 vLLM 可执行文件：%s\n' "$vllm_bin" >&2
  exit 1
fi
if [ ! -d "$model_path" ]; then
  printf '找不到模型目录：%s\n' "$model_path" >&2
  exit 1
fi

select_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
    awk -F, -v threshold="$threshold_mib" '
      {
        gpu = $1
        free = $2
        gsub(/[[:space:]]/, "", gpu)
        gsub(/[[:space:]]/, "", free)
        if (gpu ~ /^[0-9]+$/ && free ~ /^[0-9]+$/ && free >= threshold && free > best) {
          best = free
          selected = gpu
        }
      }
      END {
        if (selected != "") {
          print selected
        }
      }
    '
}

printf '等待空闲显存至少 %s GiB 的 GPU，每 %s 秒检查一次。\n' "$min_free_gib" "$interval"

while :; do
  selected_gpu=$(select_gpu || true)

  if [ -n "$selected_gpu" ]; then
    # 在 exec 前重查，降低检测与启动之间的显存竞争概率。
    free_mib=$(nvidia-smi -i "$selected_gpu" --query-gpu=memory.free \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
    case "$free_mib" in
      ''|*[!0-9]*) free_mib=0 ;;
    esac

    if [ "$free_mib" -ge "$threshold_mib" ]; then
      printf '使用 GPU %s：当前空闲 %s MiB，启动 %s。\n' \
        "$selected_gpu" "$free_mib" "$served_model_name"
      export CUDA_VISIBLE_DEVICES="$selected_gpu"
      exec "$vllm_bin" serve "$model_path" \
        --served-model-name "$served_model_name" \
        --host 127.0.0.1 \
        --port "$port" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --dtype bfloat16
    fi
  fi

  printf '[%s] 暂无空闲显存达到 %s GiB 的 GPU，继续等待。\n' \
    "$(date '+%F %T')" "$min_free_gib"
  sleep "$interval"
done
