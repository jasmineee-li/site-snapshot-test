# Secrets And Fixture Credentials

Use this when changing instance configs, auth setup, secret scanning, generated
benchmark artifacts, or docs that mention credentials.

## Classification

- Real secrets: live bearer/API/proxy tokens, OAuth tokens, Modal tokens, cloud
  credentials, and any credential that grants access outside a disposable local
  benchmark fixture.
- Fixture credentials: checked-in usernames/passwords that reproduce WebArena
  Verified or Postmill/GitLab benchmark state, such as `byteblaze` /
  `hello1234`, `postmill`, and `test1234`.
- Generated artifacts: runtime output under `logs/`, generated compose files,
  proxy port maps, and per-host instance files.
- Placeholders/examples: values like `<token from .proxy_token>` and documented
  environment variable names.

## Policy

- Do not blanket-remove fixture credentials. They are part of reproducible
  benchmark setup and test contracts.
- Do not commit real proxy/API tokens. `verification_proxy` should reference an
  external source such as `token_env` or `token_file` when the config is tracked.
- Some historical/generated instance configs may predate this rule and contain a
  literal `verification_proxy.token`. Treat those as remediation debt: do not
  copy the pattern into new configs, rotate/migrate before external sharing, and
  prefer token indirection when regenerating tracked host snapshots.
- Secret scanning should allow documented fixture credentials while still
  flagging high-entropy token fields and live bearer/API/proxy tokens.
- If a value is ambiguous, classify it in the PR or handoff before editing it.
