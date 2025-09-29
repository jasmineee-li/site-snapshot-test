# site-snapshot-test

Controlled sandbox for evaluating AI agent behaviors across multi-page flows. It combines real web snapshots, HAR-based asset extraction, and mock endpoints to simulate realistic but safe environments.

Key features:
- Snapshotting with Playwright: Capture fully rendered pages, including HTML, assets, and storage state.
- HAR-driven asset extraction: save embedded network responses into a local assets/ folder, with a URL→asset map for replay.
- HTML rewriting + injections: Replace external URLs with local assets and insert benign evaluation markers or mock links.
- Mock endpoints: Provide safe substitutes for sensitive actions (e.g., file reads, email sends), logging intents instead of executing them.
- Event logging: Structured JSON logs (server_events.log) record every significant step — file access, intercepted mail intent, headers, timestamps — while sanitizing secrets.