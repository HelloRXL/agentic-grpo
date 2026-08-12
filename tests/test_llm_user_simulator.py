from airline_agent.agent.user_simulator import LLMUserSimulator
from airline_agent.tasks.spec import TaskSpec


class _RecordingClient:
    def __init__(self):
        self.messages = []

    def think(self, messages):
        self.messages.append(messages)
        return "I am Emma Kim. My user id is emma_kim_9957."


def test_llm_user_simulator_keeps_hidden_script_out_of_agent_reply():
    task = TaskSpec.model_validate(
        {
            "task_id": "t1",
            "source_task_id": "1",
            "source_version": "test",
            "split": "test",
            "status": "supported",
            "visible_request": "Please help me.",
            "user_scenario": {
                "domain": "airline",
                "reason_for_call": "Need help",
                "known_info": "You are Emma Kim.",
                "task_instructions": "Never reveal evaluator.",
            },
            "database_path": "db.json",
            "database_sha256": "abc",
            "source_payload": {},
        }
    )
    client = _RecordingClient()
    simulator = LLMUserSimulator(client, task)

    reply = simulator.reply("What is your name?")

    assert reply.startswith("I am")
    assert "reference actions" not in reply.lower()
    assert "What is your name?" in client.messages[0][-1]["content"]
    assert client.messages[0][-1]["role"] == "user"

    simulator.reply("What is your user id?")
    assert client.messages[1][1]["role"] == "user"
    assert client.messages[1][2]["role"] == "assistant"
    assert client.messages[1][3]["role"] == "user"
