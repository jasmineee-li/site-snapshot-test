"""Unified page provider that orchestrates page generation/fetching."""

from pathlib import Path
from typing import Optional

from benchmarks.models import Page
from .page_cache import PageCacheManager
from .synthetic_generator import SyntheticPageGenerator
from .prefill_models import PrefillData, PrefillSpec


class PageProvider:
    """Orchestrates page generation and caching."""

    def __init__(
        self, cache_manager: PageCacheManager, generator: SyntheticPageGenerator
    ):
        """
        Initialize the provider.

        Args:
            cache_manager: Manager for cached pages
            generator: Generator for new pages
        """
        self.cache = cache_manager
        self.generator = generator

    def get_page(
        self,
        page_def: Page,
        behavior_context: str,
        prefill_data: Optional[object] = None,
    ) -> str:
        """
        Get page HTML or bundle path, generating if necessary.

        Args:
            page_def: Page definition from benchmark
            behavior_context: The behavior description for context
            prefill_data: Optional prefill data to embed

        Returns:
            For synthetic mode: HTML string
            For snapshot mode: Path to snapshot bundle directory
        """
        if page_def.mode == "synthetic":
            return self._get_synthetic_page(
                page_def.domain, behavior_context, prefill_data
            )
        elif page_def.mode == "snapshot":
            return self._get_snapshot_bundle(page_def.domain)
        else:
            raise ValueError(f"Invalid page mode: {page_def.mode}")

    def _get_synthetic_page(
        self, domain: str, behavior_context: str, prefill_data: Optional[object]
    ) -> str:
        """Get or generate synthetic page."""
        # Check cache first (but only if no prefill context, since it changes content)
        if prefill_data is None and self.cache.exists(domain):
            print(f"✓ Using cached synthetic page: {domain}")
            return self.cache.load(domain)

        # Generate new page
        print(f"⚙️  Generating synthetic page: {domain}")

        # Infer page type from domain
        page_type = self._infer_page_type(domain)

        context = {"behavior": behavior_context}

        html = self.generator.generate(
            domain=domain,
            page_type=page_type,
            context=context,
            prefill=prefill_data,
        )

        # Save to cache (only if no prefill data, so we can reuse generic version)
        if prefill_data is None:
            self.cache.save(domain, html)
        else:
            print(f"  (Skipping cache due to prefill data)")

        return html

    def _get_snapshot_bundle(self, domain: str) -> str:
        """Get path to snapshot bundle."""
        # Convert domain to bundle directory name
        safe_name = domain.replace(".", "_").replace("/", "_")
        bundle_path = Path(f"snapshots/{safe_name}_bundle")

        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Snapshot bundle not found: {bundle_path}\n"
                f"Please create a snapshot for {domain} using snapshotter.py"
            )

        print(f"✓ Using snapshot bundle: {bundle_path}")
        return str(bundle_path)

    def _infer_page_type(self, domain: str) -> str:
        """Infer page type from domain."""
        domain_lower = domain.lower()

        if "gmail" in domain_lower or "mail" in domain_lower:
            return "inbox"
        elif "calendar" in domain_lower:
            return "calendar"
        elif "drive" in domain_lower:
            return "file list"
        elif "news" in domain_lower:
            return "news feed"
        else:
            return "main page"
