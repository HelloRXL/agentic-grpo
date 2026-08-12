"""veRL actor old-log-prob 路径的显存计量，不修改训练计算。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.profiler import DistProfiler
from verl.workers.engine_workers import ActorRolloutRefWorker, _with_routing_replay_flag


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _write_memory_sample(sample: dict[str, object]) -> None:
    """每个 rank 写独立文件，避免 FSDP worker 并发追加同一个文件。"""

    directory = os.environ.get("VERL_AIRLINE_OLD_LOGPROB_TRACE_DIR")
    if not directory:
        return

    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"rank-{_rank()}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except OSError:
        # 显存观测不得覆盖原始训练异常或改变训练控制流。
        return


class AirlineActorRolloutWorker(ActorRolloutRefWorker):
    """在 bypass=false 的 actor old-log-prob forward 外围记录显存峰值。"""

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        # RayPPOTrainer 仅在 bypass=false 时调用该函数；bypass=true 会直接复用
        # rollout log-prob，因此没有“降到某个值”的重算路径，而是路径不存在。
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

        succeeded = False
        try:
            output = self.actor.infer_batch(data)
            succeeded = True
            return output.cpu() if output is not None else None
        finally:
            torch.cuda.synchronize(device)
            allocated_peak = torch.cuda.max_memory_allocated(device)
            reserved_peak = torch.cuda.max_memory_reserved(device)
            _write_memory_sample(
                {
                    "timestamp": time(),
                    "rank": _rank(),
                    "succeeded": succeeded,
                    "allocated_before_mb": round(allocated_before / 2**20, 2),
                    "allocated_peak_mb": round(allocated_peak / 2**20, 2),
                    "allocated_incremental_peak_mb": round(
                        max(0, allocated_peak - allocated_before) / 2**20, 2
                    ),
                    "reserved_before_mb": round(reserved_before / 2**20, 2),
                    "reserved_peak_mb": round(reserved_peak / 2**20, 2),
                    "reserved_incremental_peak_mb": round(
                        max(0, reserved_peak - reserved_before) / 2**20, 2
                    ),
                }
            )
