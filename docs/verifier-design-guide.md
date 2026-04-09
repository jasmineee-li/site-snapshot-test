# Verifier Design Guide

Verifiers are programmatic checks that determine whether a browser-use agent completed a task correctly. Both function tasks and real tasks use the same verifier infrastructure. This document is the single reference for how verifiers work.

## Privileged Access

Verifiers have full backend access -- they can query application state, inspect the database/filesystem, and add internal APIs if needed. This privilege is what makes robust verification possible: the agent interacts only through the UI, but the verifier can inspect the ground truth underneath.

## Verifier Interface

All verifiers implement this signature:

```python
def verify(server_url: str) -> tuple[bool, str]:
    """
    Args:
        server_url: Base URL of the running server (e.g., 'http://localhost:8000')

    Returns:
        (passed, message) -- passed is True if the task was completed correctly,
        message is a human-readable explanation.
    """
```

Verifiers are generated artifacts and are treated as untrusted. They run only inside the sandboxed redteam execution backend, not on the host. Keep them limited to deterministic verification logic over local app state. Do not spawn subprocesses, read arbitrary host files, or make outbound network requests.

## Verification Architecture

The application's `AppState` is the single source of truth. On every mutation, the browser pushes the full state to the server via `PUT /api/state`. Verifiers read this state via `GET /api/state`. See `docs/app-environment-protocol.md` for full details.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/state` | `GET` | Read current application state as JSON |
| `/api/state` | `PUT` | Browser pushes state on every mutation |
| `/api/reset` | `POST` | Reset to seed state (called by eval harness, not verifiers) |
| `/api/events` | `GET` | SSE stream for reset notifications |

## Example

```python
import requests

def verify(server_url: str) -> tuple[bool, str]:
    resp = requests.get(f"{server_url}/api/state")
    if resp.status_code != 200:
        return False, "Could not retrieve application state."

    state = resp.json()

    match = [g for g in state['groups'] if g['name'] == 'Test Group']
    if not match:
        return False, "Group 'Test Group' not found."

    group = match[0]
    if group['visibility'] != 'public':
        return False, f"Expected visibility 'public', got '{group['visibility']}'."

    return True, "Group 'Test Group' exists with correct visibility."
```

## Design Principles

1. **Validate state, not UI.** Always read from `GET /api/state`. Never parse HTML or check visual properties.
2. **Be specific.** Check exact properties and relationships -- not just that an entity exists, but that its fields have the correct values.
3. **Validate side effects.** If an action has cascading consequences (e.g., deleting a group removes its members), verify those too.
4. **Locate entities by stable attributes.** Use names or other natural keys, not hardcoded IDs that break when seed data changes.
5. **Don't reset state.** The eval harness calls `POST /api/reset` before each task. Verifiers should assume the app started from seed data and only the agent's actions modified it.

## Handling Ambiguous Outcomes with LLM Judges

Some tasks produce outcomes that can't be verified with exact string matching -- for example, composing an email body, writing a description, or renaming something where multiple phrasings are acceptable. In these cases, use an LLM judge to evaluate the content:
- **Use LLM judging only at the atomic level.** Judge individual field values (e.g., "does this email body convey a meeting cancellation?"), not entire state trees.
- **Prefer deterministic checks wherever possible.** If a field has a finite set of valid values, use exact matching. Reserve LLM judging for genuinely open-ended content.
- **Combine both approaches.** Verify structural properties deterministically (correct recipient, correct label applied, item exists), then use an LLM judge only for the free-text fields.
