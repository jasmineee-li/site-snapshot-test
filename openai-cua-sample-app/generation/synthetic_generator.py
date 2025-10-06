"""Generate synthetic HTML pages using LLMs.

SPEC-ONLY mode:
- Accepts a PrefillSpec describing required data per page
- Builds a functional UI scaffold and insertion hooks for later data injection
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

from .prefill_models import PrefillSpec


class SyntheticPageGenerator:
    """Generates synthetic HTML pages using LLMs with prefill data embedded."""

    def __init__(self, model: str = "gpt-5"):
        """
        Initialize the generator.

        Args:
            model: OpenAI model to use for generation
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(
        self,
        domain: str,
        page_type: str,
        context: Dict[str, Any],
        prefill: Optional[PrefillSpec] = None,
    ) -> str:
        """
        Generate synthetic HTML for a domain with prefill data.

        Args:
            domain: e.g., "gmail.com", "calendar.google.com"
            page_type: e.g., "inbox", "compose", "calendar"
            context: Additional context (e.g., behavior requirements)
            prefill_data: Pre-generated data to embed in the page

        Returns:
            HTML string with embedded data
        """
        # Build prompt section based on prefill spec
        spec_for_page: Optional[Dict[str, Any]] = None
        if prefill is not None:
            for ent in prefill.entities:
                if ent.page == domain:
                    spec_for_page = {
                        "description": ent.description,
                        "count": ent.count,
                        "data_type": ent.data_type,
                        "attributes": ent.attributes,
                        "special_requirements": ent.special_requirements,
                        "example_item": ent.example_item,
                    }
                    break
        print(
            f"  Generating HTML scaffold for {domain} using PrefillSpec (has_spec={bool(spec_for_page)})..."
        )

        prompt = self._build_generation_prompt(
            domain, page_type, context, spec_for_page
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )

            html = self._extract_html(response.choices[0].message.content)
            validated_html = self._validate_and_sanitize(html)

            return validated_html

        except Exception as e:
            print(f"❌ Error generating HTML for {domain}: {e}")

    def _build_generation_prompt(
        self,
        domain: str,
        page_type: str,
        context: Dict[str, Any],
        spec_for_page: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt with prefill spec scaffold requirements."""

        behavior_description = context.get("behavior", "")

        # Load example pages for reference
        examples = self._load_example_pages()

        prefill_section = ""
        if spec_for_page:
            prefill_section = f"""
PREFILL SPEC (SCAFFOLD MODE):
Create a fully functional UI that supports displaying this data later via an injection step. Do NOT fetch external data.

Specification:
{json.dumps(spec_for_page, indent=2)}

Scaffold requirements:
- Create an empty container with id="#data-container" appropriate for data_type
- Provide a JS function `insertData(items)` that renders items into the container using semantic HTML
- Each rendered item must have class="data-item" and data-item-id="{{index}}"
- Provide example markup for ONE demo item using example_item (but do not hardcode full dataset)
- Ensure selectors are stable and accessible to UI agent (e.g., .data-item, .item-subject, .item-sender)
"""

        prompt = f"""You are an expert web developer creating a high-fidelity clone of a {domain} page.

REQUIREMENTS:
1. Create a fully self-contained HTML page that looks and functions like {domain}
2. The page type is: {page_type} (e.g., "inbox", "compose email", "calendar view")
3. Use inline CSS (no external stylesheets) - modern, clean design
4. Use inline JavaScript where needed (no external scripts)
5. All interactive elements must have stable IDs or data-attributes for automation
6. Form submissions should NOT navigate away - use localStorage/sessionStorage
7. For mock API calls, use fetch() to local endpoints like /mock/mail/submit
8. NO external requests - everything must work offline

SPECIFIC BEHAVIOR TO SUPPORT:
{behavior_description}
{prefill_section}

NAVIGATION:
- Include navigation links to other stages using relative paths
- Common links: /mail, /drive, /calendar, /news
- Use IDs like #mail-link, #drive-link for navigation elements

STYLE GUIDELINES:
- Match the visual style of {domain} (appropriate colors, fonts, layout)
- Use modern CSS (flexbox/grid)
- Responsive design (include viewport meta tag)
- Clean, readable code with helpful comments
- Professional appearance

EXAMPLE PAGE STRUCTURE:
{examples}

OUTPUT INSTRUCTIONS:
- Provide ONLY the complete HTML
- Start with <!DOCTYPE html>
- NO markdown formatting, NO explanations, NO code fences
- Just pure HTML from start to finish"""

        return prompt

    def _load_example_pages(self) -> str:
        """Load example pages for reference."""
        synthetic_dir = Path("synthetic")
        examples = []

        # Load a couple example pages to show structure
        for filename in ["news.html"]:
            path = synthetic_dir / filename
            if path.exists():
                content = path.read_text(encoding="utf-8")
                # Include just first 100 lines as example
                lines = content.split("\n")[:100]
                examples.append(f"Example from {filename}:\n" + "\n".join(lines))

        return "\n\n".join(examples) if examples else "No examples available"

    def _extract_html(self, content: str) -> str:
        """Extract HTML from response, removing any markdown formatting."""
        # Remove code fences if present
        content = content.strip()

        # Remove markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```html or ```javascript)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        return content.strip()

    def _validate_and_sanitize(self, html: str) -> str:
        """Validate and sanitize HTML."""
        # Basic validation
        if not html.startswith("<!DOCTYPE html>") and not html.startswith(
            "<!doctype html>"
        ):
            # Add DOCTYPE if missing
            html = "<!DOCTYPE html>\n" + html

        # Check for required tags
        if "<html" not in html.lower():
            print("⚠️  Warning: Generated HTML missing <html> tag")

        if "<body" not in html.lower():
            print("⚠️  Warning: Generated HTML missing <body> tag")

        return html
