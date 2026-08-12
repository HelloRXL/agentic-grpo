"""用 Rich 实时展示一次 Agent rollout。"""

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


class LiveRolloutDisplay:
    """按 tau2 的终端风格展示每个事件，不改变 rollout 数据。"""

    def __init__(self, max_observation_chars: int = 2000) -> None:
        self._console = Console()
        self._max_observation_chars = max_observation_chars

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    def _panel(self, title: str, body: str, border_style: str) -> None:
        self._console.print(
            Panel(Text(body), title=title, border_style=border_style, expand=False)
        )

    def handle(self, event: str, payload: dict[str, Any]) -> None:
        """处理 AgentLoop 发出的实时展示事件。"""

        if event == "task_started":
            self._console.print(Rule("✈  Agentic Airline Rollout", style="bright_cyan"))
            self._panel(
                f"任务 {payload['task_id']}",
                str(payload.get("user_request") or "Agent 首轮问候，等待 User Simulator 回复"),
                "bright_cyan",
            )
            return

        step_index = payload.get("step_index", "?")
        if event == "agent_action":
            self._console.print(Rule(f"Step {step_index} · 🤖 Agent", style="bright_blue"))
            self._panel("Action", self._json(payload["action"]), "bright_blue")
            return

        if event == "parse_error":
            self._console.print(Rule(f"Step {step_index} · ⚠ Parse Error", style="yellow"))
            self._panel("模型原始输出", str(payload["raw_output"]), "yellow")
            self._panel("解析错误", str(payload["error"]), "red")
            return

        if event == "user_reply":
            self._panel("👤 User Simulator", str(payload["reply"]), "bright_green")
            return

        if event == "tool_result":
            result = payload["observation"]
            rendered = self._json(result)
            if len(rendered) > self._max_observation_chars:
                rendered = rendered[: self._max_observation_chars] + (
                    "\n… [terminal output truncated; full result is in JSON]"
                )
            success = result.get("success")
            self._panel(
                f"{'✅' if success else '❌'} Tool Observation",
                rendered,
                "bright_green" if success else "red",
            )
            return

        if event == "llm_error":
            self._panel("❌ LLM Error", str(payload["error"]), "red")
            return

        if event == "finished":
            self._panel("✅ Agent Finished", str(payload.get("answer") or ""), "bright_green")
            return

        if event == "max_steps":
            self._panel("⏹ Max Steps", "达到最大步数，rollout 结束。", "yellow")

    def show_summary(self, evaluation: dict[str, Any], output_path: str | None) -> None:
        """在 rollout 完成后显示 reward 和输出文件位置。"""

        environment = evaluation["environment_reward"]
        status = "✅ success" if environment["success"] else "❌ incomplete"
        body = (
            f"环境任务: {status}\n"
            f"Environment reward: {environment['reward']}\n"
            f"Judge pass: {'✅' if evaluation.get('judge_pass') else '❌'}\n"
            f"Official τ2 success: {'✅' if evaluation.get('full_task_success') else '❌'}\n"
            f"SFT accepted: {'✅' if evaluation.get('sft_accepted') else '❌'}\n"
            f"Strict action success: {'✅' if evaluation.get('strict_action_success') else '❌'}\n"
            f"环境诊断: {', '.join(environment['reasons']) or '无'}\n"
            f"完整任务失败原因: {', '.join(evaluation.get('full_task_failure_reasons', [])) or '无'}\n"
            f"动作诊断: {', '.join(evaluation.get('action_failures', [])) or '无'}"
        )
        if output_path:
            body += f"\n轨迹文件: {output_path}"
        self._console.print(Panel(Text(body), title="Evaluation", border_style="bright_magenta"))
        self._show_judge_verdict(evaluation.get("communication_verification"))

    def _show_judge_verdict(self, verdict: dict[str, Any] | None) -> None:
        """在终端显示 LLM Judge 的逐条断言结果。"""

        if verdict is None:
            return

        status_icons = {
            "satisfied": "PASS",
            "violated": "FAIL",
            "unknown": "UNKNOWN",
        }
        lines = [
            f"通信评估: {'PASS' if verdict.get('passed') else 'FAIL'}",
        ]
        for item in verdict.get("assertions", []):
            status = str(item.get("status", "unknown"))
            evidence = ", ".join(item.get("evidence_event_ids", [])) or "无"
            lines.extend(
                (
                    "",
                    f"[{status_icons.get(status, status.upper())}] 断言 {item.get('assertion_id')}",
                    f"证据: {evidence}",
                    f"理由: {item.get('rationale') or '无'}",
                )
            )
        if verdict.get("error"):
            lines.extend(("", f"Judge 错误: {verdict['error']}"))

        self._console.print(Rule("LLM Judge", style="bright_magenta"))
        self._panel("逐条断言判断", "\n".join(lines), "bright_magenta")
