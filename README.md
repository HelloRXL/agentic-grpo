# Agentic GRPO Airline

个人开发的多轮航空客服 Agent 后训练项目。项目将航旅任务建模为“模型决策 - 工具执行 - 用户回复 - 环境状态变化”的闭环，用于探索 SFT、GRPO 与 veRL 在长轨迹工具调用任务中的工程实现。

## 任务与环境

任务参考 tau2 Airline 的航空客服场景，覆盖订单查询、航班检索、改签、取消、行李更新、预订创建和人工转接等业务。每个 episode 都从独立的航空数据库状态开始，Agent 通过结构化 JSON Action 调用工具，环境执行后返回 observation，并在终止时依据数据库状态和必要沟通要求进行校验。

运行时注册 11 个工具：用户与订单查询、机场代码映射、直飞/中转检索、航班状态查询、订单创建/取消/改签/行李更新，以及受业务规则约束的人工转接。

## 数据流程

训练数据来自任务转换、参考动作重放和多轮轨迹采集。转换器保留原任务请求、用户上下文、初始状态补丁、参考动作和沟通断言；重放审计验证工具可用性与目标状态。训练侧还支持交互策略变体和状态反事实变体，以覆盖信息不完整、多订单确认和不同初始业务状态等分支。

数据与模型权重不包含在本仓库中。请基于兼容的 tau2 Airline 数据准备本地任务文件和数据库路径。

## 代码结构

```text
src/airline_agent/
├── core/       # Action schema、工具注册、执行器、LLM client、rollout
├── domain/     # 航空数据库模型、环境、工具与状态补丁
├── agent/      # 多轮 AgentLoop、上下文和 User Simulator
├── tasks/      # 任务转换、重放审计与训练变体
└── verl_*.py   # veRL AgentLoop、LATA 与显存分析适配
scripts/        # 数据准备、SFT/GRPO/veRL 启动与模型合并脚本
tests/          # 核心业务、奖励、任务转换和 veRL 适配测试
```

## 训练链路

```text
Task Conversion -> Replay Audit -> Trajectory Collection -> SFT
                                                    -> Online GRPO / veRL
```

SFT 使用 assistant Action token 监督；在线训练通过 AgentLoop 采样完整多轮轨迹，奖励模块综合终止状态、数据库目标、沟通约束和过程质量。veRL 适配层保留 policy Action token trace，并支持异步 rollout、FSDP、GRPO-LATA advantage 与 checkpoint 恢复。

## 运行

项目依赖、模型路径、API 地址和本地数据路径均由运行环境配置。脚本入口位于 `scripts/`，核心命令模块包括：

```text
airline_agent.sft_train
airline_agent.grpo_train
airline_agent.verl_train
airline_agent.baseline_run
```

运行测试：

```bash
PYTHONPATH=src python -m pytest -q
```
