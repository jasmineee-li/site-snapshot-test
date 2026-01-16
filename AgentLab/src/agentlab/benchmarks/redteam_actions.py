"""
Custom actions for the redteam benchmark.

These actions extend BrowserGym's action space for redteam-specific functionality.
"""

# Placeholder that gets injected at runtime by BrowserGym
# (same pattern as send_message_to_user in browsergym/core/action/functions.py)
send_message_to_user: callable = None

# Marker prefix used to detect done() calls in loop.py
DONE_MARKER = "__TASK_DONE__"


def done(reason: str = "Task completed"):
    """
    Signal that the task has been completed successfully.
    This will terminate the experiment.

    Examples:
        done()
        done("Successfully changed the flight to January 16")
    """
    send_message_to_user(f"{DONE_MARKER}: {reason}")
