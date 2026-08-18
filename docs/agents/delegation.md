# Delegation

Rules for handing work to a subagent and for reading what comes back.

## Pass a claim with its provenance

Say where a fact came from whenever you did not measure it yourself, and ask
the agent to re-measure before relying on it. A prompt carries the authority of
an instruction, so a stale count or a wrong path travels into the work and into
the artifact that records it. A claim marked for re-measurement gets corrected;
a claim stated flatly is inherited.

## Ask for the counterfactual

Require the measurement that would come out differently if the claim were
false. A dependency bound earns its place when the same resolve is shown before
and after it. A configuration change earns its place when reverting one line
reproduces the error it suppresses. A test earns its place when regressing the
code it covers turns it red. Ask for that second run in the prompt, so the
report arrives with it already done.
