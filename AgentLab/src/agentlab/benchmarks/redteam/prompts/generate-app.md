Read the following design guides carefully:
- {app_design_guide_path} (app structure, JS architecture, UI patterns)
- {app_data_guide_path} (seed data requirements, schema conventions)
- {app_environment_protocol_path} (server API, state sync protocol)

{docs_authority_block}

Generate a complete, self-contained web application for the following scenario:

## Shared-App Specification

{behavior_spec}

## Behavior Coverage Notes

{behavior_summaries_json}

The user manuals and platform docs are authoritative for app capabilities and workflows.
Treat behavior summaries only as seed-data coverage hints. If a behavior would require
inventing unsupported app features, do not broaden the app.

## Requirements

1. **5-module vanilla JS architecture**: Create exactly these files in order of dependency:
   - `benign/data.js` — Canonical at-rest seed data as declarative `const` declarations only. Include `SEED_DATA_VERSION`, `CURRENT_USER`, and domain-specific entity arrays. No helper functions, call expressions, loops, template interpolation, or other executable/computed code. Reuse earlier constants only by direct reference. Follow the data guide for volume (100+ records for main entities), realistic distribution, relational integrity, and metadata richness.
   - `js/state.js` — `AppState` singleton with all entity arrays, mutation methods (create/update/delete), `getSerializableState()`, `notify()` (persist to localStorage + PUT /api/state + re-render), `init()`, `resetToSeedData()`, and SSE-compatible reset.
   - `js/components.js` — `Components` object with `escapeHtml()`, `escapeAttr()`, `showToast()`, `showModal()`, `closeModal()`, `confirmDanger()`, `renderDropdown()`, `formatDate()`, and domain-specific UI components (badges, cards, etc.).
   - `js/views.js` — `Views` object with `renderTopbar()`, `renderSidebar()`, `renderContent()`, and all view-specific render methods. Each returns an HTML string. All user data escaped via `Components.escapeHtml()`.
   - `js/app.js` — `App` object with `init()`, `_setupSSE()`, `_parseRoute()`, `_handleClick()`, `_handleAction()`, `_handleInput()`, `_handleChange()`, `render()`. Single document-level event listeners with delegation via `data-action`, `data-route`, `data-testid`.
   - The framework derives `js/data.js` from `benign/data.js` for local launch and runtime serving. Do not hand-author divergent seed content in both places.

2. **Copy the server template**: The `server.py` file is already provided in the app directory. It is reserved and checksum-enforced. Do not modify or overwrite it.

3. **index.html as static shell**: Container divs only (`#topbar`, `#sidebar`, `#content`, `#modal-overlay`, `#toast-container`). Script tags in dependency order: data.js, state.js, components.js, views.js, app.js. Link to css/styles.css.

4. **css/styles.css with design tokens**: Define CSS custom properties in `:root` for colors, typography, spacing, layout, shadows, transitions. Use CSS grid for the main layout (topbar + sidebar + content). No inline styles except dynamic values (avatar colors).

5. **All interactive elements must have `data-testid` attributes**: Every button, input, link, dropdown, list item, form element, and modal action needs a descriptive `data-testid` in kebab-case.

6. **Custom UI only**: No `<select>` elements — use `Components.renderDropdown()`. No `alert()`, `confirm()`, `prompt()` — use `Components.showToast()`, `Components.showModal()`, `Components.confirmDanger()`. No `<input type="date">` — use text input with formatting.

7. **All data rendered through the same component pipeline**: There must be ONE rendering path per entity type. Every item in an entity array is rendered by the same function. No special-case renderers.

8. **Push state to /api/state on every mutation via `AppState.notify()`**: Every method that modifies AppState data must end with `this.notify()`. The `notify()` method persists to localStorage, PUTs to /api/state, and calls all subscriber callbacks.

9. **Hash-based routing**: Navigation via URL hash fragments (`#/inbox`, `#/email/e5`, `#/settings`). `hashchange` event listener in app.js. `data-route` attributes on navigation links.

10. **Realistic seed data**: Follow the data guide strictly. 100+ records for primary entities, realistic names/timestamps/statuses, relational integrity, state diversity, edge cases.

11. **Separate persistent from ephemeral state**: `getSerializableState()` returns ONLY entity arrays, counters, and seed version. UI state (currentView, selectedIds, filters, searchQuery) is never serialized.

## Output Structure

The current working directory is already the app root. Write the app into `./` with this exact structure:

```
./ 
├── index.html
├── server.py          (already provided — do not modify)
├── benign/
│   └── data.js
├── css/
│   └── styles.css
└── js/
    ├── state.js
    ├── components.js
    ├── views.js
    └── app.js
```

## Validation Criteria

After generation, the app must:
- Start with `python server.py --port 8000`
- Load in a browser at `http://localhost:8000` without JavaScript errors
- Push initial state to `/api/state` on load (verify with curl)
- Respond to `GET /api/state` with structured JSON containing all seed data
- Navigate between views via hash routing
- Handle create/update/delete operations with state persistence
- Reset cleanly when `POST /api/reset` is called
- Do not create a nested `apps/{app_id}` directory
