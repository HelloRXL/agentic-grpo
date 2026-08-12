#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
split=${1:-all}

PYTHONPATH=src /data/raoxinlong/runtime/conda_envs/agentic_grpo/bin/python \
  -m airline_agent.baseline_run \
  --split "$split"
