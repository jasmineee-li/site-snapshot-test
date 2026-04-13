You are profiling the **{site_name}** site's data model within a web benchmark.

Files are at `/workspace/benchmark`. Write output to `/workspace/output/`.

Produce `/workspace/output/DATA_MODEL.json`.

For every user-facing entity in this site: document its storage backend, fields
(name, type, whether mutable via the UI), relationships to other entities, and
sample values from seed data or source code.

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
