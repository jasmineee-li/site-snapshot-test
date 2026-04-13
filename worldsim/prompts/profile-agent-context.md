You are discovering agent context for the **{site_name}** site in a web benchmark.

Files are at `/workspace/benchmark`. Write output to `/workspace/output/`.

Produce `/workspace/output/AGENT_CONTEXT.json`.

## Procedure

1. Search the benchmark codebase for example agent prompts (check `examples/`,
   `prompts/`, `docs/`, `agents/` directories). If found, extract the full
   template and rename any benchmark-specific placeholders to the pipeline
   standard names: `{{INSTRUCTION}}` for the task objective, `{{START_URLS}}`
   for the start URL list.

2. Examine the evaluation harness. Determine whether evaluation parses agent
   text output (requires structured JSON) or checks browser state directly
   (DOM inspection, network events, URL matching).

3. If structured output is required, extract the expected JSON schema from the
   evaluator source code.

4. Check whether tasks embed per-task format specifications in their intent
   strings (e.g. via an `instantiation_dict` field that gets interpolated into
   the intent template).

5. Search for authentication credentials, storage state files, or auto-login
   mechanisms. Check source code, config files, Docker entrypoints, and
   example agent code.

6. Identify the site's platform name and role from README files, UI strings
   in the source, or Docker service names.

## JSON schema

```json
{{
  "response_format": {{
    "requires_structured_output": true,
    "output_schema": {{ "field": "type description" }},
    "per_task_format_field": "field_name_or_null",
    "description": "How the evaluator works and what the agent should return."
  }},
  "authentication": {{
    "pre_authenticated": false,
    "credentials": {{ "username": "...", "password": "..." }},
    "description": "How authentication works for this site."
  }},
  "agent_prompt_template": "Full template string with {{INSTRUCTION}} and {{START_URLS}} placeholders, or null.",
  "site_context": {{
    "platform_name": "Human-readable platform name",
    "description": "One-line description of what the site does."
  }}
}}
```

Field notes:

- `requires_structured_output`: true when the evaluator parses agent text (e.g. JSON response); false when evaluation checks browser state (DOM, network events).
- `output_schema`: the JSON structure the agent must return. Null when `requires_structured_output` is false.
- `per_task_format_field`: field name in the task's `instantiation_dict` that contains per-task formatting instructions baked into the intent string. Null if tasks do not embed format specs.
- `credentials`: login credentials if discoverable. Null if auth is handled externally or not needed.
- `agent_prompt_template`: must contain `{{INSTRUCTION}}` and `{{START_URLS}}` if present. Null if no template exists.

## Examples

<example benchmark="structured-output-with-template">
A benchmark where the evaluator parses agent JSON and vendor prompts exist:

```json
{{
  "response_format": {{
    "requires_structured_output": true,
    "output_schema": {{
      "task_type": "RETRIEVE | MUTATE | NAVIGATE",
      "status": "SUCCESS | NOT_FOUND_ERROR | PERMISSION_DENIED_ERROR | ACTION_NOT_ALLOWED_ERROR | DATA_VALIDATION_ERROR | UNKNOWN_ERROR",
      "retrieved_data": "array or null",
      "error_details": "string or null"
    }},
    "per_task_format_field": "retrieved_data_format_spec",
    "description": "Agent must return JSON with task_type, status, and retrieved_data. Some tasks embed format instructions in the intent via the instantiation_dict field retrieved_data_format_spec."
  }},
  "authentication": {{
    "pre_authenticated": false,
    "credentials": {{"username": "admin", "password": "admin1234"}},
    "description": "Admin panel with auto-login plugin. If session expires, re-authenticate with admin / admin1234."
  }},
  "agent_prompt_template": "You are an autonomous web agent operating in Merchant Admin Portal.\n\n## Authentication\nYou are already logged in as admin. To re-authenticate, use credentials: admin / admin1234.\n\n## Task\n- **Objective:** {{INSTRUCTION}}\n- **Start URLs:**\n{{START_URLS}}\n\n## Response Format\nReturn JSON with task_type, status, retrieved_data, error_details.",
  "site_context": {{
    "platform_name": "Merchant Admin Portal",
    "description": "An admin portal to manage an e-commerce business."
  }}
}}
```
</example>

<example benchmark="action-based-no-template">
A benchmark where evaluation checks browser DOM/network state:

```json
{{
  "response_format": {{
    "requires_structured_output": false,
    "output_schema": null,
    "per_task_format_field": null,
    "description": "Evaluation uses program_html and url_match checks against browser DOM state. Agent does not need to return structured text."
  }},
  "authentication": {{
    "pre_authenticated": true,
    "credentials": null,
    "description": "Browser session is pre-authenticated via storage_state JSON files loaded before task execution."
  }},
  "agent_prompt_template": null,
  "site_context": {{
    "platform_name": "Shopping Site",
    "description": "E-commerce storefront."
  }}
}}
```
</example>

<example benchmark="action-based-with-credentials">
A benchmark with action-based eval but discoverable credentials:

```json
{{
  "response_format": {{
    "requires_structured_output": false,
    "output_schema": null,
    "per_task_format_field": null,
    "description": "Evaluation uses DOM inspection, network event matching, and action sequence validation."
  }},
  "authentication": {{
    "pre_authenticated": false,
    "credentials": {{"username": "emma.lopez@gmail.com", "password": "Password.123"}},
    "description": "Agent must log in with provided credentials."
  }},
  "agent_prompt_template": null,
  "site_context": {{
    "platform_name": "E-commerce Store",
    "description": "Online shopping platform where customers browse products and make purchases."
  }}
}}
```
</example>
