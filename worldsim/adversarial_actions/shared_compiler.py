from __future__ import annotations

from worldsim.adversarial_actions.catalog import (
    ACTION_SIGNAL_BY_KIND as _CATALOG_ACTION_SIGNAL_BY_KIND,
)
from worldsim.adversarial_actions.catalog import (
    DEFAULT_ACTION_KINDS,
    EDITOR_METHODS_BY_MUTATION_ACTION,
    MUTATION_ACTION_BY_EDITOR_METHOD,
    MUTATION_ACTION_KINDS,
)

ACTION_KINDS: tuple[str, ...] = DEFAULT_ACTION_KINDS
ACTION_SIGNAL_BY_KIND: dict[str, str] = _CATALOG_ACTION_SIGNAL_BY_KIND
_BINARY_LABEL_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"blank", "populated"}),
    frozenset({"link", "no-link"}),
)
_MUTATION_ACTIONS = MUTATION_ACTION_KINDS
_EDITOR_ACTION_BY_METHOD = MUTATION_ACTION_BY_EDITOR_METHOD
_METHODS_BY_MUTATION_ACTION = EDITOR_METHODS_BY_MUTATION_ACTION
_MUTATION_SUCCESS_STATUSES = [200, 201, 202, 204, 302, 303]
