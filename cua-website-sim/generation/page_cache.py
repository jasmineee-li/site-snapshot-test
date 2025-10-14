"""Cache manager for generated synthetic pages."""

from pathlib import Path
from typing import Optional


class PageCacheManager:
    """Manages cached synthetic pages."""

    def __init__(self, cache_dir: str = "synthetic"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory where synthetic pages are stored
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_page_path(self, domain: str) -> Path:
        """
        Convert domain to file path.

        Args:
            domain: Domain name (e.g., "gmail.com")

        Returns:
            Path to the cache file (e.g., "synthetic/gmail_com.html")
        """
        # Replace dots and slashes with underscores for safe filename
        safe_name = domain.replace(".", "_").replace("/", "_")
        return self.cache_dir / f"{safe_name}.html"

    def exists(self, domain: str) -> bool:
        """
        Check if a synthetic page exists in cache.

        Args:
            domain: Domain name

        Returns:
            True if cached page exists
        """
        return self.get_page_path(domain).exists()

    def load(self, domain: str) -> str:
        """
        Load existing synthetic page from cache.

        Args:
            domain: Domain name

        Returns:
            HTML content

        Raises:
            FileNotFoundError: If page doesn't exist
        """
        path = self.get_page_path(domain)
        if not path.exists():
            raise FileNotFoundError(f"Cached page not found: {path}")
        return path.read_text(encoding="utf-8")

    def save(self, domain: str, html: str):
        """
        Save generated synthetic page to cache.

        Args:
            domain: Domain name
            html: HTML content to save
        """
        path = self.get_page_path(domain)
        path.write_text(html, encoding="utf-8")
        print(f"✓ Cached page saved: {path}")

    def clear(self, domain: Optional[str] = None):
        """
        Clear cache for a specific domain or all domains.

        Args:
            domain: Specific domain to clear, or None to clear all
        """
        if domain:
            path = self.get_page_path(domain)
            if path.exists():
                path.unlink()
                print(f"✓ Cache cleared for: {domain}")
        else:
            for path in self.cache_dir.glob("*.html"):
                path.unlink()
            print(f"✓ All cache cleared from: {self.cache_dir}")
