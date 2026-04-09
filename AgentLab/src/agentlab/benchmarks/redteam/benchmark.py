"""RedteamBenchmark -- loads benchmark JSON and creates env_args_list."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

import bgym
from bgym import HighLevelActionSetArgs

from agentlab.benchmarks.abstract_env import AbstractBenchmark
from agentlab.benchmarks.redteam.app_artifacts import (
    behavior_contract_path,
    benchmark_ready_app_dir_error,
    default_platform_manifest_path,
    docs_snapshot_mismatch_error,
    load_app_manifest,
    load_behavior_contract,
    load_platform_manifest,
    resolve_platform_behavior_mapping,
    resolve_repo_root_path,
    validate_path_component,
)
from agentlab.benchmarks.redteam.config import Page, RedteamEnvArgs, parse_case_fields
from agentlab.benchmarks.redteam_attacker import RedteamAttackerAgent
from agentlab.benchmarks.redteam_actions import done as done_action

logger = logging.getLogger(__name__)


class PregeneratedAppManifestError(RuntimeError):
    """Raised when authoritative pregenerated app-mode artifacts are invalid."""


def _manifest_owned_field_error(behavior_id: str, field_name: str, detail: str) -> str:
    return (
        f"pregenerated app manifest is missing or invalid {field_name} for "
        f"{behavior_id}: {detail}"
    )


class RedteamBenchmark(AbstractBenchmark):
    """Benchmark for app-mode redteam security testing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "redteam"
    benchmark_file: str | Path = None
    env_args_list: list = None  # list[RedteamEnvArgs]

    # Agent configuration (required by GenericAgent.set_benchmark)
    is_multi_tab: bool = False
    high_level_action_set_args: HighLevelActionSetArgs = None

    # Generation settings
    world_sim_model: str  # LLM for page generation and prefill analysis
    attacker_model: str  # LLM for attacker agent
    attack_type: str  # "direct" or "indirect"
    adversarial_mode: str  # "single_turn" or "multi_turn"
    use_trajectory: bool = False  # Whether to run trajectory observation on real sites

    # Adversarial mode settings
    max_conversation_turns: int = 10  # For multi_turn mode

    # Variation settings for parallelization
    n_injection_variations: int = 1

    # Number of adversarial variants per behavior (benign is always generated)
    n_adversarial_variants: int = 1

    # Attack channels the agent observes: ["html", "axtree", "visual"]
    attack_channels: list[str] = None  # Default set in model_post_init

    # Pre-generated environment support
    pregenerated_env_dir: str | Path = None
    platform_manifest: str | Path | None = None

    # Functional correctness quality gate
    functional_threshold: float = 0.80  # Minimum function-task pass rate

    # ------------------------------------------------------------------ init

    def model_post_init(self, __context: Any) -> None:
        """Load benchmark JSON and create env_args_list."""
        self.env_args_list = []
        invalid_pregenerated_apps: list[str] = []

        if self.attack_channels is None:
            self.attack_channels = ["axtree"]

        if self.high_level_action_set_args is None:
            self.high_level_action_set_args = HighLevelActionSetArgs(
                subsets=["bid", "nav", "tab", "chat", "custom"],
                custom_actions=[done_action],
                multiaction=False,
            )

        with open(self.benchmark_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            benchmark_name = "enriched-behaviors"
            cases = data
        else:
            benchmark_name = data.get("name", "redteam")
            cases = data.get("data", [])

        self.name = benchmark_name
        logger.info(f"Loading redteam benchmark: {benchmark_name}")

        for case_idx, case_data in enumerate(cases):
            try:
                variants = self._parse_case(case_idx, case_data)
            except PregeneratedAppManifestError as exc:
                invalid_pregenerated_apps.append(str(exc))
                continue
            if variants:
                self.env_args_list.extend(variants)

        if invalid_pregenerated_apps:
            raise PregeneratedAppManifestError(
                "Invalid pregenerated redteam app-mode artifacts:\n- "
                + "\n- ".join(invalid_pregenerated_apps)
            )

        n_behaviors = len(cases)
        logger.info(
            f"Loaded {len(self.env_args_list)} tasks "
            f"({n_behaviors} behaviors \u00d7 2 variants each: 1 benign + 1 active adversarial)"
        )

    # --------------------------------------------------------------- parsing

    def _resolved_platform_manifest_path(self) -> Path:
        """Return the resolved platform manifest path for shared-app loading."""
        configured = self.platform_manifest or default_platform_manifest_path()
        return Path(configured)

    def _load_pregenerated_app_contracts(
        self,
        behavior_id: str,
    ) -> tuple[Path | None, dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
        """Return app/behavior contract state for a shared pregenerated app."""
        if not self.pregenerated_env_dir:
            return None, {}, {}, {}, None

        platform_manifest_path = self._resolved_platform_manifest_path()
        if not platform_manifest_path.exists():
            return (
                None,
                {},
                {},
                {},
                f"missing authoritative platform_manifest.json at {platform_manifest_path}",
            )
        try:
            platform_manifest = load_platform_manifest(platform_manifest_path)
            mapping = resolve_platform_behavior_mapping(platform_manifest, behavior_id)
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            return None, {}, {}, {}, str(exc)

        app_id = mapping["app_id"]
        env_dir = Path(self.pregenerated_env_dir) / validate_path_component(app_id, "app_id")
        try:
            manifest = load_app_manifest(env_dir)
            behavior_contract = load_behavior_contract(env_dir, behavior_id)
        except (OSError, json.JSONDecodeError) as exc:
            return env_dir, {}, {}, mapping, f"unreadable shared app contracts ({exc})"
        if not manifest:
            return env_dir, {}, {}, mapping, "empty app_manifest.json"
        if not behavior_contract:
            contract_path = behavior_contract_path(env_dir, behavior_id)
            return (
                env_dir,
                manifest,
                {},
                mapping,
                f"missing authoritative behavior contract at {contract_path}",
            )
        return env_dir, manifest, behavior_contract, mapping, None

    def _parse_case(self, case_idx: int, case_data: dict) -> list[RedteamEnvArgs]:
        """Parse a case from the benchmark JSON and generate variants.

        Shared-app loading creates one benign control and one active adversarial run.

        Args:
            case_idx: Index of this case in the benchmark.
            case_data: Case configuration dict.

        Returns:
            List of RedteamEnvArgs (1 benign + 1 active adversarial variant).
        """
        fields = parse_case_fields(case_data, case_idx)

        behavior_id: str = fields["behavior_id"]
        doc: str = fields["doc"]
        target: str = fields["target"]

        (
            pregenerated_app_dir,
            pregenerated_manifest,
            behavior_contract,
            platform_mapping,
            pregenerated_manifest_error,
        ) = self._load_pregenerated_app_contracts(behavior_id)
        if pregenerated_manifest_error:
            raise PregeneratedAppManifestError(f"{behavior_id}: {pregenerated_manifest_error}")
        if pregenerated_app_dir is None:
            raise PregeneratedAppManifestError(
                f"{behavior_id}: missing authoritative shared app directory root"
            )

        readiness_error = benchmark_ready_app_dir_error(
            pregenerated_app_dir,
            manifest=pregenerated_manifest,
            behavior_contract=behavior_contract,
            behavior_id=behavior_id,
            functional_threshold=self.functional_threshold,
        )
        if readiness_error:
            raise PregeneratedAppManifestError(f"{behavior_id}: {readiness_error}")

        docs_snapshot_error = docs_snapshot_mismatch_error(
            manifest=pregenerated_manifest,
            behavior_spec=platform_mapping,
            repo_root_path=resolve_repo_root_path(self._resolved_platform_manifest_path()),
        )
        if docs_snapshot_error:
            raise PregeneratedAppManifestError(f"{behavior_id}: {docs_snapshot_error}")

        logger.info(
            "Valid app/behavior contracts found for %s; using shared app mode from %s",
            behavior_id,
            pregenerated_app_dir,
        )

        manifest_pages_raw = pregenerated_manifest.get("pages")
        if not isinstance(manifest_pages_raw, list) or not manifest_pages_raw:
            raise PregeneratedAppManifestError(
                _manifest_owned_field_error(behavior_id, "pages", "expected non-empty list")
            )
        try:
            pages = [Page(**raw) for raw in manifest_pages_raw]
        except TypeError as exc:
            raise PregeneratedAppManifestError(
                _manifest_owned_field_error(behavior_id, "pages", str(exc))
            )

        entry_route = str(behavior_contract.get("entry_route") or "").strip()
        success_condition = str(behavior_contract.get("success_condition") or "").strip()
        safe_analog = str(behavior_contract.get("safe_behavior") or "").strip()
        allowed_routes = [
            str(route)
            for route in (behavior_contract.get("allowed_routes") or [])
            if str(route).strip()
        ]
        domain_bindings = list(behavior_contract.get("domain_bindings") or [])
        mock_endpoints = list(behavior_contract.get("mock_endpoints") or [])
        active_variant = str(behavior_contract.get("active_variant") or "").strip()
        if not active_variant:
            raise PregeneratedAppManifestError(
                f"{behavior_id}: behavior contract is missing active_variant"
            )
        start_page = str(pregenerated_manifest.get("start_page") or "").strip()
        app_id = str(pregenerated_manifest.get("app_id") or platform_mapping.get("app_id") or "").strip()

        # Additional fields read directly from case_data
        system_prompt_config = case_data.get("system_prompt_config", {})
        target_system_prompt = system_prompt_config.get("content")
        tools_config = case_data.get("tools_config")
        criteria = case_data.get("criteria")
        judging = case_data.get("judging")

        # Case-level overrides for attack settings
        case_attack_channels = (
            fields["attack_channels"]
            or self.attack_channels
        )

        # --- build variants ---
        variants: list[RedteamEnvArgs] = []

        def _make_env_args(
            task_name: str,
            test_mode: str,
            variation_seed: int,
        ) -> RedteamEnvArgs:
            app_dir_resolved = str(pregenerated_app_dir) if pregenerated_app_dir else ""
            app_variant_resolved = "benign" if test_mode == "benign" else active_variant

            env_args = RedteamEnvArgs(
                task_name=task_name,
                app_id=app_id,
                behavior_id=behavior_id,
                entry_route=entry_route,
                allowed_routes=allowed_routes,
                domain_bindings=domain_bindings,
                mock_endpoints=mock_endpoints,
                active_variant=active_variant,
                test_mode=test_mode,
                variation_seed=variation_seed,
                injection_seed=variation_seed,  # backward compat
                attack_channels=case_attack_channels,
                target=target,
                pages=pages,
                start_page=start_page,
                summary=case_data.get("summary"),
                objective=case_data.get("objective"),
                doc=doc,
                safe_behavior=safe_analog,
                world_sim_model=self.world_sim_model,
                attack_type=case_data.get("attack_type", self.attack_type),
                attacker_model=self.attacker_model,
                use_trajectory=self.use_trajectory,
                adversarial_mode=case_data.get(
                    "adversarial_mode", self.adversarial_mode,
                ),
                max_conversation_turns=self.max_conversation_turns,
                target_system_prompt=target_system_prompt,
                tools_config=tools_config,
                criteria=criteria,
                judging=judging,
                success_condition=success_condition,
                app_dir=app_dir_resolved,
                variant=app_variant_resolved,
            )

            # Attach attacker agent
            attacker_agent = RedteamAttackerAgent(
                model=self.attacker_model,
                goal=doc,
                target_description=target,
                attack_type=env_args.attack_type,
                mode=env_args.adversarial_mode,
                test_mode=test_mode,
                pages=pages,
                safe_behavior=safe_analog,
            )
            env_args.attacker_agent = attacker_agent

            return env_args

        # 1. Benign control variant
        benign_task_name = f"{self.name}.{behavior_id}.benign"
        variants.append(_make_env_args(benign_task_name, "benign", 0))

        # 2. Active adversarial variant only.
        adv_task_name = f"{self.name}.{behavior_id}.{active_variant}"
        variants.append(_make_env_args(adv_task_name, "adversarial", 0))

        logger.debug(
            f"Created {len(variants)} variants for {behavior_id}: "
            f"1 benign + active adversarial {active_variant}"
        )
        return variants

    # --------------------------------------------------------- dependency graph

    def dependency_graph_over_tasks(self) -> dict[str, list[str]]:
        """Redteam tasks are independent."""
        return {}
