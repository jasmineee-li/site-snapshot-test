"""Decorator + binding primitives for the editor-method contract registry.

The contract that describes an editor method lives as a decorator on the
method itself (see ``@editor_method``). Editing a method body and editing
its contract happen in the same file, same visual frame, same diff hunk.

The decorator attaches a ``_editor_method_spec`` dict to the wrapped
function and returns the function unchanged. No wrapping, no import-time
registration, no side effects. The one-time walk that turns these
attributes into a registry happens in :func:`worldsim.editors._registry.register_editor`,
called once per editor class from :mod:`worldsim.editors.__init__`.

Bindings describe what each editor-method arg can hold:

* ``Token("{benign_issue_iid}")`` — value must start with one of these tokens.
* ``SelectorGroup("project", "{benign_project_id}", "{benign_project_path}")``
  — member of a selector group; at least one group member must be populated
  in the editor call, and its value must start with one of the declared
  tokens. Express OR-logic this way.
* ``FreeText()`` — any string value; no token constraint. Use for bodies,
  titles, and other natural-language payloads.

Arg names in ``bindings`` are the LLM-facing names (what the LLM emits in
``editor_calls[].args``), which may differ from the Python method's
parameter names (see the rename map in
``worldsim.seeding._editor_arg_name``). The coverage test in commit 2
bridges through that rename map.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

BindingKind = Literal["token", "selector", "free_text"]


@dataclass(frozen=True)
class BindingSpec:
    kind: BindingKind
    tokens: frozenset[str] = frozenset()
    selector_group: str | None = None
    required: bool = True


def Token(*tokens: str, required: bool = True) -> BindingSpec:
    """A bindable arg whose value must start with one of ``tokens``."""
    return BindingSpec(
        kind="token",
        tokens=frozenset(tokens),
        selector_group=None,
        required=required,
    )


def SelectorGroup(group: str, *tokens: str, required: bool = True) -> BindingSpec:
    """Member of a selector OR-group.

    The validator groups bindings by their ``selector_group`` string and
    requires at least one group member to be populated with a value that
    starts with one of its declared ``tokens``. Bindings with an empty
    ``tokens`` set still count as group members (any free-text value
    satisfies), but only if the group name is declared.
    """
    return BindingSpec(
        kind="selector",
        tokens=frozenset(tokens),
        selector_group=group,
        required=required,
    )


def FreeText(required: bool = True) -> BindingSpec:
    """A natural-language arg; the value is unconstrained by tokens."""
    return BindingSpec(
        kind="free_text",
        tokens=frozenset(),
        selector_group=None,
        required=required,
    )


def editor_method(
    *,
    kinds: frozenset[str] | set[str] | tuple[str, ...],
    http: tuple[str, str],
    bindings: Mapping[str, BindingSpec],
    surface_id_per_kind: Mapping[str, str] | None = None,
    required_editor_args: tuple[str, ...] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach contract metadata to an editor method. Zero side effects.

    ``kinds`` is the set of :data:`worldsim.phases.phase_2_target_resolver.ResourceKind`
    values this method can attach to. Pass ``frozenset()`` to declare the
    method exists but is never a valid Option A target (replaces the old
    ``_OPTION_A_DANGLING_METHODS`` frozenset).

    ``http`` is ``(verb, path_template)``. The path template uses Python
    ``str.format`` tokens (``{project_id}`` etc.) that resolve to method
    args at call time.

    ``bindings`` maps LLM-facing arg name → :class:`BindingSpec`.

    ``surface_id_per_kind`` maps each addressed kind to the profile
    ``surface_id`` the resolver's ``attach_surfaces_for_kind`` emits.

    ``required_editor_args`` is the ordered tuple of LLM-facing arg names
    the resolver's ``attach_surfaces_for_kind`` declares as required. When
    ``None``, derived from ``bindings`` (required entries, insertion order).
    """
    kinds_fz = frozenset(kinds)
    bindings_dict = dict(bindings)
    surfaces = dict(surface_id_per_kind or {})
    if required_editor_args is None:
        derived = tuple(name for name, spec in bindings_dict.items() if spec.required)
    else:
        derived = tuple(required_editor_args)

    spec_meta: dict[str, Any] = {
        "kinds": kinds_fz,
        "http": (str(http[0]), str(http[1])),
        "bindings": bindings_dict,
        "surface_id_per_kind": surfaces,
        "required_editor_args": derived,
    }

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn._editor_method_spec = spec_meta  # type: ignore[attr-defined]
        return fn

    return _decorate


__all__ = [
    "BindingKind",
    "BindingSpec",
    "FreeText",
    "SelectorGroup",
    "Token",
    "editor_method",
]
