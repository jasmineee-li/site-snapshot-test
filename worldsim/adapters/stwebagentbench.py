"""STWebAgentBench benchmark adapter.

Maps STWebAgentBench task definitions (WebArena + SuiteCRM) into the
WorldSim task schema.  STWebAgentBench extends the standard WebArena
task format with a ``policies`` array on each task that encodes safety
constraints and their evaluation criteria.

All external imports are lazy so this module is importable even when
stwebagentbench is not installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder-to-site mapping.
# Includes the standard WebArena placeholders plus SuiteCRM.
# ---------------------------------------------------------------------------
_PLACEHOLDER_TO_SITE: dict[str, str] = {
    "__REDDIT__": "reddit",
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__GITLAB__": "gitlab",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
    "__SUITECRM__": "suitecrm",
}

# Reverse lookup: site name -> placeholder token.
_SITE_TO_PLACEHOLDER: dict[str, str] = {v: k for k, v in _PLACEHOLDER_TO_SITE.items()}

# ---------------------------------------------------------------------------
# Credentials per site.
# Sourced from stwebagentbench/browser_env/env_config.py ACCOUNTS dict.
# ---------------------------------------------------------------------------
_CREDENTIALS: dict[str, dict[str, str]] = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "suitecrm": {"username": "user", "password": "bitnami"},
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
    "wikipedia": {},
    "map": {},
}

# The 9 safety evaluator types defined in STWebAgentBench.
_SAFETY_EVAL_TYPES: frozenset[str] = frozenset({
    "is_ask_the_user",
    "is_url_match",
    "element_action_match",
    "is_sequence_match",
    "is_action_count",
    "is_program_html",
    "is_input_hallucination",
    "is_sensitive_data",
    "popup_error",
})

# Task-level eval types used for the benign reward function.
_TASK_EVAL_TYPES: frozenset[str] = frozenset({
    "string_match",
    "url_match",
    "program_html",
})

# Separator used in start_url for multi-tab tasks.
_MULTI_URL_SEPARATOR = " |AND| "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_site_from_start_url(start_url: str) -> str | None:
    """Return the site name for the first placeholder found in *start_url*."""
    for placeholder, site in _PLACEHOLDER_TO_SITE.items():
        if placeholder in start_url:
            return site
    return None


def _split_start_urls(start_url: str) -> list[str]:
    """Split a start_url string on the ``|AND|`` separator.

    Returns a list with one or more URL strings (still containing
    placeholders).
    """
    if _MULTI_URL_SEPARATOR in start_url:
        return [u.strip() for u in start_url.split(_MULTI_URL_SEPARATOR)]
    return [start_url]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class STWebAgentBenchAdapter:
    """Adapter for the STWebAgentBench benchmark.

    STWebAgentBench augments WebArena tasks with per-task safety
    policies.  Each policy carries its own ``eval`` block with one of
    9 evaluator types (``is_ask_the_user``, ``is_url_match``,
    ``element_action_match``, etc.).

    Sites covered: gitlab, shopping_admin, and suitecrm.
    SuiteCRM uses Angular hash routing (``/#/contacts``, ``/#/accounts``).
    """

    def __init__(self, benchmark_name: str) -> None:
        self.benchmark_name = benchmark_name

    # -- properties --------------------------------------------------------

    @property
    def placeholder_to_site(self) -> dict[str, str]:
        """WebArena placeholder-to-site mapping including SuiteCRM.

        E.g. ``{"__GITLAB__": "gitlab", ..., "__SUITECRM__": "suitecrm"}``.
        """
        return dict(_PLACEHOLDER_TO_SITE)

    @property
    def supported_sites(self) -> list[str]:
        """List of site names this benchmark may reference."""
        return sorted(_PLACEHOLDER_TO_SITE.values())

    # -- loading -----------------------------------------------------------

    def load_tasks(self, task_file: Path) -> list[dict]:
        """Load raw task definitions from the STWebAgentBench JSON file.

        Parameters
        ----------
        task_file:
            Path to ``test.raw.json`` (or equivalent).

        Returns
        -------
        list[dict]
            Raw task dicts as they appear in the JSON file.
        """
        task_file = Path(task_file)
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        data = json.loads(task_file.read_text())
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in {task_file}, got {type(data).__name__}"
            )

        logger.info(
            "STWebAgentBench: loaded %d tasks from %s",
            len(data),
            task_file,
        )
        return data

    # -- wrapping ----------------------------------------------------------

    def wrap_task(self, raw: dict) -> dict:
        """Map a raw STWebAgentBench task to the WorldSim task schema.

        The returned dict has the standard WorldSim fields plus a
        ``policies`` array preserved from the original task.

        Output schema::

            {
                "id": str,
                "site": str,
                "sites": list[str],
                "instruction": str,
                "start_urls": list[str],
                "data_seed": {"mechanism": "none"},
                "reward_function": {
                    "eval": <task eval block>,
                    "task_id": <task_id>,
                },
                "policies": [...],
                "intent_template_id": int | None,
                "require_login": bool,
                "require_reset": bool,
                "storage_state": str | None,
            }
        """
        task_id = raw.get("task_id")
        if task_id is None:
            task_id = raw.get("id", "unknown")

        # Sites list -- use the raw "sites" field if present.
        sites: list[str] = raw.get("sites", [])

        # Primary site: first entry in sites, or infer from start_url.
        if sites:
            site = sites[0]
        else:
            start_url = raw.get("start_url", "")
            detected = _detect_site_from_start_url(start_url)
            site = detected if detected else "unknown"
            sites = [site] if site != "unknown" else []

        # start_urls: split on |AND| separator for multi-tab tasks.
        raw_start_url = raw.get("start_url", "")
        start_urls = _split_start_urls(raw_start_url) if raw_start_url else []

        # Instruction text (STWebAgentBench uses "intent").
        instruction = raw.get("intent", raw.get("instruction", ""))

        # Policies array -- preserved as-is for downstream safety evaluation.
        policies = raw.get("policies", [])

        wrapped: dict[str, Any] = {
            "id": str(task_id),
            "site": site,
            "sites": sites,
            "instruction": instruction,
            "start_urls": start_urls,
            "data_seed": {
                "mechanism": "none",
            },
            "reward_function": {
                "eval": raw.get("eval", {}),
                "task_id": task_id,
            },
            "policies": policies,
            # Preserve metadata useful for downstream filtering / login.
            "intent_template_id": raw.get("intent_template_id"),
            "require_login": raw.get("require_login", False),
            "require_reset": raw.get("require_reset", False),
            "storage_state": raw.get("storage_state"),
        }

        # Optional metadata fields -- carry forward if present.
        if "task_metadata" in raw:
            wrapped["task_metadata"] = raw["task_metadata"]
        if "setup_scripts" in raw:
            wrapped["setup_scripts"] = raw["setup_scripts"]

        return wrapped

    # -- BrowserGym task name ----------------------------------------------

    def get_browsergym_task_name(self, task: dict) -> str:
        """Construct a BrowserGym-compatible task name.

        STWebAgentBench tasks do not have a standard BrowserGym
        registration.  We produce a name following the convention
        ``stwebagentbench.<task_id>`` which mirrors the
        ``webarena.<task_id>`` pattern.

        Parameters
        ----------
        task:
            A wrapped task dict (output of ``wrap_task``).

        Returns
        -------
        str
            Task name, e.g. ``"stwebagentbench.47"``.
        """
        task_id = task.get("id", task.get("task_id", "unknown"))
        return f"stwebagentbench.{task_id}"

    # -- credentials -------------------------------------------------------

    def get_credentials(self, site: str) -> dict:
        """Return login credentials for *site*.

        Parameters
        ----------
        site:
            Site name, e.g. ``"gitlab"``, ``"suitecrm"``.

        Returns
        -------
        dict
            ``{"username": ..., "password": ...}`` or ``{}`` for sites
            that do not require authentication.

        Raises
        ------
        ValueError
            If *site* is not recognized.
        """
        site = site.lower().strip()
        if site not in _CREDENTIALS:
            raise ValueError(
                f"Unknown site {site!r}. "
                f"Known sites: {sorted(_CREDENTIALS)}"
            )
        return dict(_CREDENTIALS[site])
