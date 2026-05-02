You are profiling the **{site_name}** site's data model within a web benchmark.

Files are at `/workspace/benchmark`. Write output to `/workspace/output/`.
Neutral host-built indexes may also be present at:

- `/workspace/inputs/FILES_INDEX.json`
- `/workspace/inputs/ROUTES_INDEX.json`
- `/workspace/inputs/TASKS_INDEX.json`
- `/workspace/inputs/MANIFEST_SLICE.json`

Use these indexes only as navigation aids and citation scaffolding. They are
not authority for entity validity; the benchmark source remains primary.

Produce `/workspace/output/DATA_MODEL.json`.
Also produce `/workspace/output/DATA_MODEL_EVIDENCE.json`.

For every user-facing entity in this site: document its storage backend, fields
(name, type, whether mutable via the UI), relationships to other entities, and
sample values from seed data or source code.

For each entity and mutable field, preserve review evidence in
`DATA_MODEL_EVIDENCE.json`: cite the concrete source files, route literals,
schema definitions, fixtures, or task records that support the entity and field
claims. If the evidence is partial, record the limitation instead of filling the
gap by guesswork.

## JSON schema

```json
[
  {{
    "entity": "<entity_name>",
    "storage": "<storage_backend_and_table>",
    "fields": [
      {{"name": "<field_name>", "type": "<field_type>", "mutable_via_ui": false}},
      {{"name": "<field_name>", "type": "<field_type>", "mutable_via_ui": true}}
    ],
    "relationships": ["<relationship_description>"],
    "sample_values": [{{"<field>": "<value>"}}]
  }}
]
```

## Validation rules

- Every entity must have at least one field.
- Storage must reference a concrete backend (e.g. "MySQL catalog_product_entity", "PostgreSQL issues").

## Evidence sidecar schema

```json
{{
  "site_name": "{site_name}",
  "entities": [
    {{
      "entity": "<entity_name>",
      "storage_evidence": [
        {{"file": "<relative_path>", "line": 1, "claim": "<what this proves>"}}
      ],
      "field_evidence": [
        {{
          "field": "<field_name>",
          "file": "<relative_path>",
          "line": 1,
          "claim": "<what this proves>"
        }}
      ]
    }}
  ],
  "limitations": ["<uncertainty or missing evidence, if any>"]
}}
```
