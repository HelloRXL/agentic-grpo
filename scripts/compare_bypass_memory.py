#!/usr/bin/env python3
"""汇总同配置 bypass on/off 实验的 nvidia-smi 显存峰值。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMPARABLE_FIELDS = (
    "gpu",
    "profile",
    "trainer_gpu_count",
    "rollout_n",
    "dataset_mode",
    "max_steps",
    "gpu_memory_utilization",
    "base_model",
    "policy_model",
    "train_batch_size",
    "prompt_length",
    "response_length",
    "max_model_len",
    "ppo_max_token_len_per_gpu",
    "ppo_mini_batch_size",
    "ppo_micro_batch_size_per_gpu",
    "ppo_epochs",
)


def _read_json(directory: Path, filename: str) -> dict[str, Any]:
    path = directory / filename
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SystemExit(f"无法读取 {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} 不是合法 JSON: {error}") from error


def _selected_gpu_peak(run_config: dict[str, Any], peak: dict[str, Any]) -> dict[str, int]:
    if not peak.get("available"):
        raise SystemExit("gpu_peak.json 没有有效 nvidia-smi 样本，无法比较显存。")

    selected = str(run_config["gpu"]).split(",")
    samples = peak.get("gpu_peak", {})
    missing = [index for index in selected if index not in samples]
    if missing:
        raise SystemExit(f"gpu_peak.json 缺少目标 GPU {missing} 的采样结果。")

    return {
        "peak_used_mb": sum(samples[index]["memory_used_mb_peak"] for index in selected),
        "incremental_peak_mb": sum(samples[index]["memory_used_mb_incremental_peak"] for index in selected),
        "peak_per_gpu_mb": max(samples[index]["memory_used_mb_peak"] for index in selected),
        "incremental_peak_per_gpu_mb": max(
            samples[index]["memory_used_mb_incremental_peak"] for index in selected
        ),
    }


def _reduction(off: int, on: int) -> float | None:
    return round((off - on) / off * 100, 2) if off else None


def _old_logprob_peak(directory: Path, *, bypass_mode: bool) -> dict[str, Any]:
    """读取 actor 内部计量；bypass=true 时该重算路径按设计不会被执行。"""

    if bypass_mode:
        return {
            "executed": False,
            "reason": "bypass 直接复用 rollout log-prob，未执行 actor old-log-prob 重算",
        }

    samples: list[dict[str, Any]] = []
    for path in sorted((directory / "actor_old_logprob").glob("rank-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sample.get("succeeded"):
                samples.append(sample)
    if not samples:
        return {"executed": True, "available": False, "reason": "未找到成功的 actor old-log-prob 显存样本"}

    fields = (
        "allocated_peak_mb",
        "allocated_incremental_peak_mb",
        "reserved_peak_mb",
        "reserved_incremental_peak_mb",
    )
    return {
        "executed": True,
        "available": True,
        "samples": len(samples),
        "per_rank_peak_mb": {
            field: max(float(sample[field]) for sample in samples)
            for field in fields
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 veRL bypass on/off 的显存峰值")
    parser.add_argument("--off", type=Path, required=True, help="bypass=false 的运行目录")
    parser.add_argument("--on", type=Path, required=True, help="bypass=true 的运行目录")
    parser.add_argument("--output", type=Path, required=True, help="结果 JSON 文件")
    args = parser.parse_args()

    off_config = _read_json(args.off, "run_config.json")
    on_config = _read_json(args.on, "run_config.json")
    if off_config.get("bypass_mode") is not False or on_config.get("bypass_mode") is not True:
        raise SystemExit("--off 必须来自 bypass_mode=false，--on 必须来自 bypass_mode=true。")

    mismatches = {
        field: {"off": off_config.get(field), "on": on_config.get(field)}
        for field in COMPARABLE_FIELDS
        if off_config.get(field) != on_config.get(field)
    }
    if mismatches:
        raise SystemExit("两次运行除 bypass 外的配置不一致：\n" + json.dumps(mismatches, ensure_ascii=False, indent=2))

    off_peak = _selected_gpu_peak(off_config, _read_json(args.off, "gpu_peak.json"))
    on_peak = _selected_gpu_peak(on_config, _read_json(args.on, "gpu_peak.json"))
    old_logprob_off = _old_logprob_peak(args.off, bypass_mode=False)
    old_logprob_on = _old_logprob_peak(args.on, bypass_mode=True)
    result = {
        "comparison": "veRL rollout_correction.bypass_mode false -> true",
        "comparable_config": {field: off_config.get(field) for field in COMPARABLE_FIELDS},
        "bypass_off": off_peak,
        "bypass_on": on_peak,
        "reduction_percent": {
            "peak_used_mb_sum": _reduction(off_peak["peak_used_mb"], on_peak["peak_used_mb"]),
            "incremental_peak_mb_sum": _reduction(
                off_peak["incremental_peak_mb"], on_peak["incremental_peak_mb"]
            ),
            "peak_per_gpu_mb": _reduction(off_peak["peak_per_gpu_mb"], on_peak["peak_per_gpu_mb"]),
            "incremental_peak_per_gpu_mb": _reduction(
                off_peak["incremental_peak_per_gpu_mb"], on_peak["incremental_peak_per_gpu_mb"]
            ),
        },
        "actor_old_logprob_recompute": {
            "bypass_off": old_logprob_off,
            "bypass_on": old_logprob_on,
            "interpretation": (
                "这是 actor 重算 old log-prob 的局部 PyTorch 峰值。开启 bypass 后该路径被跳过；"
                "不要将跳过路径错误表述为一个仍在执行的 2GB forward。"
            ),
        },
        "interpretation": (
            "仅在两次运行使用同一批空闲 GPU，且没有其他进程改变显存占用时，"
            "该差值才能归因于 bypass。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
