"""Page and RedteamEnvArgs dataclasses for app-mode redteam configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentlab.benchmarks.abstract_env import AbstractEnvArgs
from agentlab.benchmarks.redteam.behavior_ids import resolve_behavior_id
from agentlab.benchmarks.redteam.utils import strip_html


@dataclass
class Page:
    """Page definition from benchmark config.

    Each Page represents a website with multiple subdomains/routes.
    For example, Gmail might have subdomains: ["/mail/u/0/#inbox", "/mail/u/0/#settings"]
    """

    id: str  # Short identifier, e.g., "duckduckgo", "gmail" (no leading slash)
    base_site_url: str = None  # Root URL, e.g., "https://mail.google.com"
    subdomains: list[str] = field(
        default_factory=list
    )  # Paths to generate, e.g., ["/mail/u/0/#inbox"]
    details: dict[str, str] = field(default_factory=dict)  # subdomain -> description
    screenshots: list[str] = field(default_factory=list)  # Paths to screenshot files/folders
    existing_path: str = None  # Path to pre-generated HTML file (skips generation if provided)
    skip_modification: bool = False


@dataclass
class RedteamEnvArgs(AbstractEnvArgs):
    """Configuration for a single app-mode redteam task."""

    task_name: str
    target: str  # Attack goal description
    attacker_model: str
    pages: list[Page]
    attack_type: str  # "direct" or "indirect"
    adversarial_mode: str  # "single_turn" or "multi_turn"
    test_mode: str  # "benign" or "adversarial"
    start_page: str = None

    # Task description fields
    summary: str = None
    objective: str = None
    doc: str = None  # primary task description, replaces 'behavior'
    safe_behavior: str = None  # Safe analog - legitimate task for benign condition

    # Generation parameters
    injection_seed: int = 0  # Backward compat alias for variation_seed
    world_sim_model: str = "openai/gpt-4.1"
    use_trajectory: bool = False  # Whether to run trajectory observation on real sites

    # Attacker agent configuration
    attacker_system_prompt: str = None  # Auto-generated if None
    max_conversation_turns: int = 10  # Max turns for multi-turn mode

    target_system_prompt: str = (
        None  # System prompt for target model (from system_prompt_config.content)
    )
    tools_config: dict = None  # Tools configuration (messages, tools, custom_tool_schema)
    criteria: list = None  # Judging criteria
    judging: dict = None  # Judge configuration

    # Environment parameters (BrowserGym compatibility)
    headless: bool = True
    max_steps: int = 20
    task_seed: int = 0

    record_video: bool = False
    wait_for_user_message: bool = False
    viewport: dict = None
    slow_mo: int = None
    storage_state: str | Path | dict = None
    task_kwargs: dict = None

    # Success condition from simulation_config
    success_condition: str = None

    # Shared attacker agent instance (initialized in benchmark)
    attacker_agent: RedteamAttackerAgent = field(default=None, init=False)

    # Variation seed for reproducibility (alias: injection_seed for backward compat)
    variation_seed: int = 0

    # Attack channels the agent observes: ["html", "axtree", "visual"]
    attack_channels: list = field(default_factory=lambda: ["axtree"])

    # Behavior ID for grouping variants
    app_id: str = ""
    behavior_id: str = ""
    entry_route: str = ""
    allowed_routes: list[str] = field(default_factory=list)
    domain_bindings: list[dict] = field(default_factory=list)
    mock_endpoints: list[str] = field(default_factory=list)
    active_variant: str = ""

    # Parent directory grouping all variants (benign + adversarial)
    parent_exp_dir: str = ""
    # Variant name: "benign" or "adversarial_{behavior_id}_v0", etc.
    variant_name: str = ""
    # Stable run-group identifier for grouped redteam study layouts.
    run_group_id: str = ""
    # Whether this run uses the variant subdirectory structure
    is_variant_run: bool = False

    app_dir: str = ""  # Path to generated app directory (apps/{app_id}/)
    variant: str = ""  # Which variant data.js to use ("benign", "adversarial_{behavior_id}_v0", etc.)

    @property
    def variant_subdir(self) -> str:
        """Subdirectory name for this variant."""
        if self.test_mode == "benign":
            return "benign"
        if self.active_variant:
            return self.active_variant
        if self.variant:
            return self.variant
        raise ValueError(
            "Adversarial redteam runs require an explicit active_variant or variant name; "
            "legacy adversarial_v{seed} fallback is no longer supported."
        )

    @property
    def computed_task_name(self) -> str:
        """Generate task name from behavior_id, seed, and test_mode."""
        if self.behavior_id:
            return f"{self.behavior_id}.seed{self.variation_seed}.{self.test_mode}"
        return self.task_name

    @property
    def goal_instruction(self) -> str:
        """Authoritative user-visible task instruction for the runtime."""
        if isinstance(self.safe_behavior, str) and self.safe_behavior.strip():
            return self.safe_behavior
        return self.doc or ""

    @property
    def app_server_path(self) -> Path | None:
        """Path to the app's server.py, or None when no app directory is set."""
        if not self.app_dir:
            return None
        return Path(self.app_dir) / "server.py"

    def make_env(
        self,
        action_mapping,
        exp_dir: Path,
        exp_task_kwargs: dict = None,
        use_raw_page_output: bool = True,
    ) -> "RedteamEnv":
        """
        Create RedteamEnv instance.

        Args:
            action_mapping: Action mapping for the agent
            exp_dir: Experiment directory
            exp_task_kwargs: Additional task kwargs (unused for redteam)
            use_raw_page_output: Whether to use raw page output (unused for redteam)

        Returns:
            RedteamEnv instance
        """
        from agentlab.benchmarks.redteam.env import RedteamEnv

        return RedteamEnv(
            env_args=self,
            exp_dir=exp_dir,
            action_mapping=action_mapping,
        )


def parse_case_fields(case_data: dict, case_idx: int | None = None) -> dict:
    """Extract normalized benchmark fields from a behavior spec."""
    simulation_config = case_data.get("simulation_config", {})
    pages: list[Page] = []
    for page_data in simulation_config.get("pages", []):
        pages.append(
            Page(
                id=page_data.get("id", "").lstrip("/"),
                base_site_url=page_data.get("base_site_url", ""),
                subdomains=page_data.get("subdomains", ["/"]),
                details=page_data.get("details", {}),
                screenshots=page_data.get("screenshots", []),
                existing_path=page_data.get("existing_path"),
                skip_modification=page_data.get("skip_modification", False),
            )
        )

    start_page = simulation_config.get("start_page", "")
    if not start_page and pages:
        start_page = f"{pages[0].base_site_url}{pages[0].subdomains[0]}"

    return {
        "behavior_id": resolve_behavior_id(case_data, case_idx),
        "doc": strip_html(case_data.get("doc", "")),
        "target": case_data.get("description", ""),
        "pages": pages,
        "start_page": start_page,
        "success_condition": simulation_config.get("success_condition", ""),
        "attack_channels": case_data.get("attack_channels"),
        "n_adversarial_variants": case_data.get("n_adversarial_variants"),
        "raw": case_data,
    }
