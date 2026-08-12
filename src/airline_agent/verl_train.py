"""加载项目算法扩展后启动 veRL 0.8 训练。"""

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class AirlineTaskRunner(TaskRunner):
    """确保自定义 advantage 在真正计算优势的 Ray 进程中完成注册。"""

    def add_actor_rollout_worker(self, config):
        """使用本地 worker 记录 bypass 省去的 actor 重算路径。"""

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role, need_reference_policy

        from .verl_memory import AirlineActorRolloutWorker

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        reference_is_fused = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        role = Role.ActorRolloutRef if need_reference_policy(config) and not reference_is_fused else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(AirlineActorRolloutWorker)
        self.mapping[role] = "global_pool"
        return AirlineActorRolloutWorker, RayWorkerGroup

    def run(self, config):
        from . import verl_lata  # noqa: F401

        return super().run(config)


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config) -> None:
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    runner = ray.remote(num_cpus=1)(AirlineTaskRunner)
    run_ppo(config, task_runner_class=runner)


if __name__ == "__main__":
    main()
