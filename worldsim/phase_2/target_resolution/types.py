"""Phase 2 target resolution types."""

from __future__ import annotations


class ResolverContractDriftError(RuntimeError):
    """Raised when :func:`derive_benign_target_resource` produces a
    record whose ``kind`` isn't addressable by any registered editor
    method — a sign the resolver and the editor-method contract registry
    have drifted apart. Fix: either add the missing kind to an existing
    ``@editor_method`` decorator, or stop emitting the kind here.
    """
