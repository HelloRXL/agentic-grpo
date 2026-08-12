#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
task_index=${1:-0}

cd "$project_dir"
mkdir -p outputs
PYTHONPATH=src /data/raoxinlong/runtime/conda_envs/agentic_grpo/bin/python \
  -m airline_agent.real_run \
  --tasks data/tasks/test.jsonl \
  --index "$task_index" \
  --agent-prefix POLICY \
  --user-prefix USER \
  --max-steps 15 \
  --output "outputs/policy-${task_index}.json"
