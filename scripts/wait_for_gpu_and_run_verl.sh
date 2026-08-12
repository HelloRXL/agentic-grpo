#!/bin/sh
set -eu

# 用法：sh scripts/wait_for_gpu_and_run_verl.sh [rollout数量] [最低空闲GiB] [检查间隔秒]
rollout_n=${1:-2}
min_free_gib=${2:-25}
interval=${3:-30}

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
threshold_mib=$((min_free_gib * 1024))

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' '未找到 nvidia-smi，无法检测 GPU 显存。' >&2
  exit 1
fi

case "$rollout_n:$min_free_gib:$interval" in
  *[!0-9:]*|*::*|0:*|*:0:*|*:*:0)
    printf '%s\n' '参数必须是正整数，例如：sh scripts/wait_for_gpu_and_run_verl.sh 2 30 30' >&2
    exit 2
    ;;
esac

printf '等待空闲显存超过 %s GiB 的 GPU，检查间隔 %s 秒。\n' "$min_free_gib" "$interval"

while :; do
  gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)
  selected_gpu=''

  while IFS=, read -r gpu_id free_mib; do
    gpu_id=$(printf '%s' "$gpu_id" | tr -d '[:space:]')
    free_mib=$(printf '%s' "$free_mib" | tr -d '[:space:]')
    case "$gpu_id:$free_mib" in
      ''|*:*[!0-9]*) continue ;;
    esac
    if [ "$free_mib" -gt "$threshold_mib" ]; then
      selected_gpu=$gpu_id
      break
    fi
  done <<EOF
$gpu_info
EOF

  if [ -n "$selected_gpu" ]; then
    printf '找到 GPU %s（空闲显存至少 %s GiB），开始 veRL smoke。\n' "$selected_gpu" "$min_free_gib"
    exec sh "$project_dir/scripts/run_verl_airline_smoke.sh" "$selected_gpu" "$rollout_n"
  fi

  printf '[%s] 暂无满足条件的 GPU，继续等待。\n' "$(date '+%F %T')"
  sleep "$interval"
done
