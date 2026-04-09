# Function Task Design Guide

This document describes how to design **function-test tasks** for browser-use agents. Unlike real tasks (which prioritize natural language and realism), function tests target **broad feature coverage** -- systematically exercising the application's functionality, edge cases, and side effects.

## Expected Output

A set of tasks, each paired with a programmatic verifier. Tasks are stored in `function-tasks.json` and verifiers in `function-tasks/` under each app directory.

## Task Design

### Core Principle: Feature Coverage

The goal is to collectively exercise as much of the application's functionality as possible. Because you have privileged access to the implementation, intentionally design tasks that target:
- **Core features.** Every major user flow should be covered by at least one task.
- **Edge cases.** Boundary conditions, empty states, maximum-length inputs, special characters.
- **Side effects.** Actions that trigger cascading changes (e.g., deleting a parent entity removes its children).
- **Validation logic.** Required fields, format constraints, conditional requirements.

Instructions can be more explicit than real tasks since the goal is testing the app, not measuring natural-language understanding. Be precise enough to exercise the exact feature you're targeting.

### Coverage Dimensions

To ensure well-rounded coverage, design tasks across these four dimensions. A good task set has representation in each:

| Dimension | What it tests | Example |
|-----------|--------------|---------|
| **Direct** | A single feature works in isolation. | "Create a new label called 'urgent' with red color." |
| **Navigation-dependent** | The feature exists but must be discovered through the app's information architecture. | "Change the notification preferences for the #backend channel." (requires navigating to the right channel's settings) |
| **Cross-referencing** | Data from one part of the app is needed to act correctly in another. | "Add all members of the Design team to the Alpha project." (must read team membership, then modify the project) |
| **Side-effect** | An action triggers cascading state changes beyond the immediate target. | "Delete the 'Infrastructure' group." (verifier checks that child projects, memberships, and shares are also removed) |

Use these dimensions to audit coverage gaps. If a task set has many direct tasks but no cross-referencing tasks, the app's relational integrity is untested.

## Agent Constraints

See [`docs/agent-constraints.md`](agent-constraints.md) for what the browser-use agent can and cannot do. Every feature you want to test must be reachable through the UI. If a feature has no UI surface, it cannot be tested by these tasks.

## Task Schema

Each task is a JSON object in `function-tasks.json`:

```json
{
  "id": "task_1",
  "instruction": "Create a new group named 'Test Group' with public visibility.",
  "verify": "function-tasks/task_1.py"
}
```

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (e.g., `task_1`, `task_2`, ...). |
| `instruction` | Unambiguous task description. Should not reveal internal implementation details. |
| `verify` | Path to the verifier script. |

## Verifier Design

See [`docs/verifier-design-guide.md`](verifier-design-guide.md) for the full verifier specification: interface signature, verification architecture, endpoint table, example code, and design principles.

## Integration with Behavior Spec

When generating function tasks for a specific behavior, use these sources to maximize coverage:

### 1. Read `functional_requirements` from the behavior spec

Each behavior spec includes a `functional_requirements` field listing the features the app must support. Use these as your primary checklist -- every functional requirement should be exercised by at least one task. Requirements that describe multi-step workflows or entity relationships are strong candidates for cross-referencing and side-effect tasks.

### 2. Read the app's source code

The implementation is the ground truth for what the app actually supports. Inspect these files to discover features, validation rules, and state transitions that may not be fully described in the behavior spec:

| File | What to look for |
|------|-----------------|
| `benign/data.js` | Canonical at-rest seed entities, field names, and startup values you can reference in task instructions |
| `js/state.js` | All mutation methods on `AppState` -- each is a testable operation. Pay attention to cascading deletes, computed fields, and validation guards. |
| `js/views.js` | Rendered pages and sections -- tells you what navigation paths exist and what UI affordances the agent can interact with. |
| `js/app.js` | Route definitions and event handlers -- reveals which actions are wired up and what parameters they expect. |

See `docs/app-design-guide.md` for a full description of the 5-module architecture and each module's responsibilities.

### 3. Understand the seed data

Task instructions often reference existing entities by name (e.g., "Delete the 'Infrastructure' project"). Read `benign/data.js` for canonical startup data, or the active variant's `data.js` when you need behavior-specific adversarial additions. `js/data.js` is only the derived runtime mount point. See `docs/app-data-guide.md` for seed data conventions, including the `CURRENT_USER` constant, version stamping, and relationship modeling.
