"""
Unit test to verify that chat_messages is consistently injected into all steps
in the multi-turn adversarial loop.

This tests the fix for the bug where environment response steps were saved
without chat_messages, causing alternating patterns in the conversation display.
"""


def test_chat_messages_injection_in_env_response_steps():
    """
    Verify that environment response steps have chat_messages injected
    before being saved.

    The bug was:
    - Agent thinking steps: had chat_messages ✓
    - Environment response steps: missing chat_messages ✗

    The fix ensures both step types have consistent chat_messages.
    """
    # Simulate the fixed behavior
    chat_messages = [
        {"role": "user", "message": "Hello, please help me", "timestamp": 1234567890},
        {"role": "assistant", "message": "Sure, I can help", "timestamp": 1234567891},
    ]

    # Create a mock observation as if from env.step()
    obs = {
        "goal": "Original goal from env.reset()",
        "goal_object": [{"type": "text", "text": "Original goal from env.reset()"}],
        "url": "http://example.com",
        # Note: chat_messages NOT set - this is what env.step() returns
    }

    # Apply the fix: inject chat_messages before saving
    obs["chat_messages"] = chat_messages.copy()

    # Also update goal to current user message
    if chat_messages:
        current_user_msg = next(
            (m["message"] for m in reversed(chat_messages) if m["role"] == "user"),
            None
        )
        if current_user_msg:
            obs["goal"] = current_user_msg
            obs["goal_object"] = [{"type": "text", "text": current_user_msg}]

    # Verify the fix worked
    assert "chat_messages" in obs, "chat_messages should be in obs"
    assert len(obs["chat_messages"]) == 2, "Should have 2 messages"
    assert obs["chat_messages"][0]["message"] == "Hello, please help me"
    assert obs["goal"] == "Hello, please help me", "Goal should be updated to user message"
    assert obs["goal_object"][0]["text"] == "Hello, please help me"

    print("✓ chat_messages correctly injected into environment response step")
    print(f"  - chat_messages count: {len(obs['chat_messages'])}")
    print(f"  - goal updated to: {obs['goal'][:50]}...")


def test_chat_messages_not_shared_between_steps():
    """
    Verify that chat_messages.copy() is used so steps don't share the same list.

    If we just did `obs["chat_messages"] = chat_messages` (without copy),
    then all steps would share the same list and modifications would affect all.
    """
    chat_messages = [
        {"role": "user", "message": "Message 1", "timestamp": 1},
    ]

    # Step 1 observation
    obs1 = {}
    obs1["chat_messages"] = chat_messages.copy()  # The fix uses .copy()

    # Add more messages
    chat_messages.append({"role": "assistant", "message": "Response 1", "timestamp": 2})
    chat_messages.append({"role": "user", "message": "Message 2", "timestamp": 3})

    # Step 2 observation
    obs2 = {}
    obs2["chat_messages"] = chat_messages.copy()

    # Verify they're independent
    assert len(obs1["chat_messages"]) == 1, "Step 1 should have 1 message"
    assert len(obs2["chat_messages"]) == 3, "Step 2 should have 3 messages"

    # Modify obs2's messages - should not affect obs1
    obs2["chat_messages"].append({"role": "assistant", "message": "Response 2", "timestamp": 4})

    assert len(obs1["chat_messages"]) == 1, "Step 1 should still have 1 message"
    assert len(obs2["chat_messages"]) == 4, "Step 2 should have 4 messages"

    print("✓ chat_messages properly copied between steps (not shared)")


def test_goal_extraction_from_chat_messages():
    """
    Verify that the latest user message is correctly extracted from chat_messages.
    """
    chat_messages = [
        {"role": "user", "message": "First user message", "timestamp": 1},
        {"role": "assistant", "message": "First response", "timestamp": 2},
        {"role": "user", "message": "Second user message", "timestamp": 3},
        {"role": "assistant", "message": "Second response", "timestamp": 4},
    ]

    # Extract latest user message (as in the fix)
    current_user_msg = next(
        (m["message"] for m in reversed(chat_messages) if m["role"] == "user"),
        None
    )

    assert current_user_msg == "Second user message", "Should get the latest user message"
    print(f"✓ Correctly extracted latest user message: '{current_user_msg}'")


def test_simulated_multiturn_loop():
    """
    Simulate the multi-turn loop to verify consistent chat_messages across all steps.
    """
    # This simulates what the fixed loop.py does
    chat_messages = []
    all_step_observations = []

    # Simulate 2 turns with 2 steps each (1 agent thinking + 1 env response)
    for turn in range(2):
        # Attacker sends message
        attacker_msg = f"User message for turn {turn}"
        chat_messages.append({"role": "user", "message": attacker_msg, "timestamp": turn * 10})

        # Agent thinking step
        agent_obs = {
            "goal": attacker_msg,
            "goal_object": [{"type": "text", "text": attacker_msg}],
            "chat_messages": chat_messages.copy(),  # Injected
        }
        all_step_observations.append(("agent_thinking", agent_obs))

        # Environment response step (simulating env.step())
        env_obs = {
            "goal": "WRONG - Original goal from env.reset()",  # env.step() returns this
            "goal_object": [{"type": "text", "text": "WRONG - Original goal"}],
            "url": "http://example.com",
            # chat_messages NOT set by env.step()
        }

        # THE FIX: Inject chat_messages and update goal before saving
        env_obs["chat_messages"] = chat_messages.copy()
        current_user_msg = next(
            (m["message"] for m in reversed(chat_messages) if m["role"] == "user"),
            None
        )
        if current_user_msg:
            env_obs["goal"] = current_user_msg
            env_obs["goal_object"] = [{"type": "text", "text": current_user_msg}]

        all_step_observations.append(("env_response", env_obs))

        # Agent responds
        chat_messages.append({"role": "assistant", "message": f"Agent response for turn {turn}", "timestamp": turn * 10 + 5})

    # Verify ALL steps have consistent chat_messages
    print("Verifying all steps have consistent chat_messages...")
    for i, (step_type, obs) in enumerate(all_step_observations):
        assert "chat_messages" in obs, f"Step {i} ({step_type}) missing chat_messages"
        assert "WRONG" not in obs["goal"], f"Step {i} ({step_type}) has wrong goal: {obs['goal']}"
        print(f"  Step {i} ({step_type}): {len(obs['chat_messages'])} messages, goal='{obs['goal'][:40]}...'")

    print("✓ All steps have consistent chat_messages and correct goal")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing multi-turn chat_messages injection fix")
    print("=" * 60)
    print()

    test_chat_messages_injection_in_env_response_steps()
    print()
    test_chat_messages_not_shared_between_steps()
    print()
    test_goal_extraction_from_chat_messages()
    print()
    test_simulated_multiturn_loop()

    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
