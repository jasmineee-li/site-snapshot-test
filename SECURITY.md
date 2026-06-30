# Security and Responsible Use

WARP is infrastructure for evaluating the safety of browser agents against
indirect prompt injection. It generates and runs adversarial content by design.

## Responsible use

- Run WARP only against host-environment instances you control, in isolated
  sandboxes. The pipeline connects to pre-running benchmark instances you
  supply; it is not meant to point at live systems.
- Do not target production services, real user accounts, money, or any system
  you are not authorized to test.
- Generated tasks embed adversarial payloads. Treat run artifacts (logs,
  transcripts, screenshots) as sensitive and keep them out of shared channels.

## Reporting a vulnerability

For a security issue in this code (as opposed to a finding about a model under
test), please open a private report rather than a public issue.

> Maintainer contact is withheld during double-blind review and will be added
> on acceptance.
