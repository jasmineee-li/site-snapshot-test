# Agent Constraints

The browser-use agent can **only interact through the front-end UI**:
- Click elements
- Type text into inputs
- Scroll the page
- Read visible text and element attributes

The agent **cannot**:
- Execute JavaScript (`page.evaluate()`)
- Directly manipulate `localStorage`, `AppState`, or internal state
- Make HTTP requests to API endpoints

Every task must be completable through visible, interactive UI elements. If a feature has no UI surface, it cannot be tested. Design tasks with these constraints in mind.
