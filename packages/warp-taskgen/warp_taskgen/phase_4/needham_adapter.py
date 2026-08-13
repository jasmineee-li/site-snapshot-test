"""Compatibility shim for the shared Needham transcript builder.

The canonical Browser-Use -> Needham contract now lives in
``warp_taskgen.phase_4.needham_trace`` so VEA, Transcript Purpose, and artifact
persistence all consume the same code path. Keep this module for older tests and
imports that still reference ``needham_adapter.build_messages``.
"""

from __future__ import annotations

from warp_taskgen.phase_4.needham_trace import build_messages

__all__ = ["build_messages"]
