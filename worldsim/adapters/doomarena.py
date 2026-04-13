"""DoomArena benchmark adapter.

Wraps DoomArena's BrowserGym attack primitives (popup, banner, UGC,
div injection) so the AgentLab runner can configure and launch
adversarial evaluation without touching DoomArena internals directly.

All DoomArena / BrowserGym imports are lazy and wrapped in
try/except so the adapter module is importable even when doomarena
is not installed.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder-to-site mapping used by WebArena environments.
# Keys are the placeholder tokens that appear in task start_urls,
# values are the site categories they resolve to.
# ---------------------------------------------------------------------------
_WEBARENA_PLACEHOLDER_TO_SITE: dict[str, str] = {
    "__REDDIT__": "reddit",
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__GITLAB__": "gitlab",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
}

# Supported attack types and their corresponding DoomArena component descriptors.
_ATTACK_TYPE_TO_COMPONENT: dict[str, dict[str, str]] = {
    "popup": {"type": "popup-element"},
    "banner": {"type": "banner"},
    "ugc": {"type": "user-generated-content"},
    "div": {"type": "div-injection"},
}

# Default exfiltration port used by DoomArena's TargetUrl success filter.
_DEFAULT_EXFIL_PORT = 1234


def _require_doomarena(symbol: str) -> None:
    """Raise a clear ImportError when doomarena is missing."""
    raise ImportError(
        f"Cannot resolve {symbol}: the 'doomarena' package is not installed. "
        "Install it with: pip install doomarena  (or add it to your project deps)"
    )


# ---------------------------------------------------------------------------
# Lazy imports -- each helper returns the needed object or raises.
# ---------------------------------------------------------------------------

def _import_attack_config() -> type:
    try:
        from doomarena.core.attack_config import AttackConfig
        return AttackConfig
    except ImportError:
        _require_doomarena("doomarena.core.attack_config.AttackConfig")


def _import_get_attacks():
    try:
        from doomarena.core.attacks import get_attacks
        return get_attacks
    except ImportError:
        _require_doomarena("doomarena.core.attacks.get_attacks")


def _import_popup_attack():
    try:
        from doomarena.browsergym.attacks.popup_attacks import get_popup_attack
        return get_popup_attack
    except ImportError:
        _require_doomarena("doomarena.browsergym.attacks.popup_attacks.get_popup_attack")


def _import_banner_attack():
    try:
        from doomarena.browsergym.attacks.banner_attacks import (
            get_svg_banner_attack,
            banner_configs as default_banner_configs,
        )
        return get_svg_banner_attack, default_banner_configs
    except ImportError:
        _require_doomarena("doomarena.browsergym.attacks.banner_attacks")


def _import_div_attacks():
    try:
        from doomarena.browsergym.attacks.div_injection import (
            GoalRevealAttack,
            AdditionalFormFieldAttack,
        )
        return GoalRevealAttack, AdditionalFormFieldAttack
    except ImportError:
        _require_doomarena("doomarena.browsergym.attacks.div_injection")


def _import_filters():
    try:
        from doomarena.browsergym.filters import FilterByUrl, StartAtStepFilter
        return FilterByUrl, StartAtStepFilter
    except ImportError:
        _require_doomarena("doomarena.browsergym.filters")


def _import_success_filters():
    try:
        from doomarena.browsergym.success_filters import TargetUrl
        return TargetUrl
    except ImportError:
        _require_doomarena("doomarena.browsergym.success_filters.TargetUrl")


def _import_attacked_env_args():
    try:
        from doomarena.browsergym.attacked_browser_env_args import AttackedBrowserEnvArgs
        return AttackedBrowserEnvArgs
    except ImportError:
        _require_doomarena(
            "doomarena.browsergym.attacked_browser_env_args.AttackedBrowserEnvArgs"
        )


# ---------------------------------------------------------------------------
# Task-name fetching (mirrors webarena_subsets.py logic without registering
# anything in bgym.DEFAULT_BENCHMARKS).
# ---------------------------------------------------------------------------

_WEBARENA_TASKS_URL = (
    "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"
)


def _fetch_webarena_task_ids(site_filter: str) -> list[int]:
    """Download WebArena's task list and return IDs whose start_url
    contains *site_filter* (case-insensitive).  Pass "" to get all tasks.
    """
    try:
        import requests
    except ImportError:
        _require_doomarena("requests (needed for WebArena task list fetch)")

    resp = requests.get(_WEBARENA_TASKS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    upper_filter = site_filter.upper()
    return [
        item["task_id"]
        for item in data
        if upper_filter in item.get("start_url", "").upper()
    ]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DoomArenaAdapter:
    """Adapter that exposes DoomArena's BrowserGym attack surface for
    the WorldSim AgentLab runner.

    This is *not* a full BenchmarkAdapter (no ``load_tasks`` / ``evaluate``
    protocol yet). It provides the concrete DoomArena integration points:

    * Building ``AttackConfig`` objects for each attack type.
    * Wrapping them into ``AttackedBrowserEnvArgs`` ready for BrowserGym.
    * Selecting WebArena task subsets using the same logic as
      ``doomarena.browsergym.webarena_subsets``.
    """

    def __init__(self, benchmark_name: str) -> None:
        self.benchmark_name = benchmark_name

    # -- properties --------------------------------------------------------

    @property
    def placeholder_to_site(self) -> dict[str, str]:
        """Return the WebArena placeholder-to-site mapping.

        E.g. ``{"__REDDIT__": "reddit", "__SHOPPING__": "shopping", ...}``.
        """
        return dict(_WEBARENA_PLACEHOLDER_TO_SITE)

    # -- attack configs ----------------------------------------------------

    def get_attack_configs(
        self,
        attack_type: str,
        *,
        exfil_port: int = _DEFAULT_EXFIL_PORT,
    ) -> tuple[Any, ...]:
        """Build one or more ``AttackConfig`` objects for *attack_type*.

        Supported types: ``popup``, ``banner``, ``ugc``, ``div``.

        Parameters
        ----------
        attack_type:
            One of the supported attack type keys.
        exfil_port:
            Port for the TargetUrl success filter (default 1234).

        Returns
        -------
        tuple[AttackConfig, ...]
            A tuple of AttackConfig instances ready to pass into
            ``AttackedBrowserEnvArgs.attack_configs``.
        """
        attack_type = attack_type.lower().strip()
        if attack_type not in _ATTACK_TYPE_TO_COMPONENT:
            raise ValueError(
                f"Unknown attack type {attack_type!r}. "
                f"Supported: {sorted(_ATTACK_TYPE_TO_COMPONENT)}"
            )

        AttackConfig = _import_attack_config()
        TargetUrl = _import_success_filters()
        success_filter = TargetUrl(
            port=exfil_port,
            target_urls=[f"localhost:{exfil_port}"],
        )
        component = _ATTACK_TYPE_TO_COMPONENT[attack_type]

        if attack_type == "popup":
            return self._build_popup_configs(
                AttackConfig, component, success_filter
            )
        elif attack_type == "banner":
            return self._build_banner_configs(
                AttackConfig, component, success_filter
            )
        elif attack_type == "ugc":
            return self._build_ugc_configs(
                AttackConfig, component, success_filter
            )
        elif attack_type == "div":
            return self._build_div_configs(
                AttackConfig, component, success_filter
            )

        # Unreachable, but keeps the type checker happy.
        raise ValueError(f"Unhandled attack type: {attack_type!r}")

    # -- per-type builders (private) ---------------------------------------

    def _build_popup_configs(
        self,
        AttackConfig: type,
        component: dict[str, str],
        success_filter: Any,
    ) -> tuple[Any, ...]:
        get_popup_attack = _import_popup_attack()
        _, StartAtStepFilter = _import_filters()

        config = AttackConfig(
            attackable_component=component,
            attack=get_popup_attack(
                report_url=f"localhost:{success_filter.port}",
            ),
            # Popup fires from step 1 onward, no URL filter needed.
            filter=StartAtStepFilter(),
            success_filter=success_filter,
        )
        return (config,)

    def _build_banner_configs(
        self,
        AttackConfig: type,
        component: dict[str, str],
        success_filter: Any,
    ) -> tuple[Any, ...]:
        get_svg_banner_attack, default_banner_configs = _import_banner_attack()
        FilterByUrl, _ = _import_filters()

        config = AttackConfig(
            attackable_component=component,
            attack=get_svg_banner_attack(
                banner_configs=default_banner_configs,
            ),
            filter=FilterByUrl(allowed_urls=["*"]),
            success_filter=success_filter,
        )
        return (config,)

    def _build_ugc_configs(
        self,
        AttackConfig: type,
        component: dict[str, str],
        success_filter: Any,
    ) -> tuple[Any, ...]:
        get_attacks = _import_get_attacks()
        FilterByUrl, _ = _import_filters()

        # "information_theft_comment_attack" is the standard UGC attack
        # used in DoomArena's published experiments.
        attack = get_attacks("information_theft_comment_attack")
        config = AttackConfig(
            attackable_component=component,
            attack=attack,
            filter=FilterByUrl(allowed_urls=["*"]),
            success_filter=success_filter,
        )
        return (config,)

    def _build_div_configs(
        self,
        AttackConfig: type,
        component: dict[str, str],
        success_filter: Any,
    ) -> tuple[Any, ...]:
        GoalRevealAttack, AdditionalFormFieldAttack = _import_div_attacks()
        FilterByUrl, _ = _import_filters()

        # Return both div-injection variants so the runner can iterate.
        goal_reveal = AttackConfig(
            attackable_component=component,
            attack=GoalRevealAttack(),
            filter=FilterByUrl(allowed_urls=["*"]),
            success_filter=success_filter,
        )
        form_field = AttackConfig(
            attackable_component=component,
            attack=AdditionalFormFieldAttack(),
            filter=FilterByUrl(allowed_urls=["*"]),
            success_filter=success_filter,
        )
        return (goal_reveal, form_field)

    # -- env args ----------------------------------------------------------

    def get_baseline_env_args(
        self,
        task_name: str,
        attack_type: str,
        *,
        exfil_port: int = _DEFAULT_EXFIL_PORT,
        max_steps: int = 30,
    ) -> Any:
        """Return an ``AttackedBrowserEnvArgs`` wired up for *task_name*
        with the attacks for *attack_type*.

        Parameters
        ----------
        task_name:
            BrowserGym task name, e.g. ``"webarena.42"``.
        attack_type:
            One of ``popup``, ``banner``, ``ugc``, ``div``.
        exfil_port:
            Port for the TargetUrl success filter.
        max_steps:
            Maximum number of steps per episode.

        Returns
        -------
        AttackedBrowserEnvArgs
            Ready to be passed into a BrowserGym experiment loop.
        """
        AttackedBrowserEnvArgs = _import_attacked_env_args()
        attack_configs = self.get_attack_configs(
            attack_type, exfil_port=exfil_port,
        )

        return AttackedBrowserEnvArgs(
            task_name=task_name,
            task_seed=0,
            max_steps=max_steps,
            benchmark_name=self.benchmark_name,
            attack_configs=attack_configs,
        )

    # -- task selection ----------------------------------------------------

    def get_task_names(
        self,
        site_filter: str = "",
        max_tasks: int = 9999999,
        seed: int | None = None,
    ) -> list[str]:
        """Return BrowserGym task names (e.g. ``"webarena.42"``) using
        the same filtering logic as DoomArena's ``webarena_subsets.py``.

        Parameters
        ----------
        site_filter:
            Placeholder token to filter by (e.g. ``"__REDDIT__"``).
            Pass ``""`` for all tasks.
        max_tasks:
            Maximum number of tasks to return.
        seed:
            If provided, shuffle task IDs with this seed before slicing.
            Matches the ``shuffle`` parameter in ``get_webarena_subset``.

        Returns
        -------
        list[str]
            Sorted (or shuffled) BrowserGym task names.
        """
        task_ids = _fetch_webarena_task_ids(site_filter)
        task_names = [f"webarena.{tid}" for tid in task_ids]

        if seed is not None:
            rnd = random.Random(seed)
            rnd.shuffle(task_names)

        task_names = task_names[:max_tasks]

        logger.info(
            "DoomArena task selection: %d tasks (filter=%r, max=%d, seed=%r)",
            len(task_names),
            site_filter or "all",
            max_tasks,
            seed,
        )
        return task_names
