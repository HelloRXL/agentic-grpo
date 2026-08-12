from airline_agent.agent.prompts import (
    build_agent_system_prompt,
    build_user_system_prompt,
)


def test_agent_prompt_uses_tau2_style_sections_and_action_contract():
    prompt = build_agent_system_prompt(
        [{"name": "search_direct_flight", "description": "Search flights"}]
    )

    assert "<instructions>" in prompt
    assert "<policy>" in prompt
    assert "<tools>" in prompt
    assert "You cannot do both at the same time" in prompt
    assert '"action_type":"tool"' in prompt
    assert "你是 Airline" not in prompt


def test_agent_prompt_contains_airport_and_lookup_rules():
    prompt = build_agent_system_prompt([])

    assert "list_all_airports" in prompt
    assert "Never invent an" in prompt
    assert "Do not ask the user to provide an IATA code" in prompt
    assert "Never invent a reservation ID" in prompt
    assert '"cities"' in prompt


def test_agent_prompt_contains_booking_and_change_guardrails():
    prompt = build_agent_system_prompt([])

    assert "at most one travel certificate, one credit card, and three gift cards" in prompt
    assert "basic-economy reservations cannot change flights" in prompt
    assert "may add but never remove checked bags" in prompt
    assert "health or weather reason" in prompt


def test_user_prompt_uses_tau2_style_guidelines_and_hidden_scenario():
    prompt = build_user_system_prompt(
        persona="A frequent flyer",
        reason_for_call="Cancel a booking",
        known_info="Booking ID is B001",
        unknown_info="The user ID",
        task_instructions="Disclose the booking ID only when asked.",
    )

    assert "Generate one customer message at a time" in prompt
    assert "Never make up or" in prompt
    assert "Facts in the scenario are the only ground truth" in prompt
    assert "Do not add a date, trip type" in prompt
    assert "Disclose information progressively" in prompt
    assert "<scenario>" in prompt
    assert "Booking ID is B001" in prompt
    assert "Do not reveal the hidden scenario" in prompt
    assert "Preserve city names exactly" in prompt
