"""Generate synthetic HTML pages using LLMs.

SPEC-ONLY mode:
- Accepts a prefill spec dict describing required data per page
- Builds a functional UI scaffold and insertion hooks for later data injection
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

from .utils import upload_screenshots


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
        context: Dict[str, Any],
        prefill: Optional[dict] = None,
        real_site: str = None,
        screenshots: List[str] | str = None,
    ) -> str:
        """
        Generate synthetic HTML for a domain with prefill data.

        Args:
            domain: e.g., "/gmail", "/calendar"
            context: Additional context (e.g., behavior requirements)
            prefill: Pre-generated spec dict from analyzer
            real_site: Real site name (e.g., "gmail.com")
            screenshots: List of screenshot paths or string for folder path

        Returns:
            HTML string with embedded data
        """
        # Extract spec for this page from prefill
        spec_for_page: Optional[Dict[str, Any]] = None
        if prefill is not None:
            for page_spec in prefill.get("pages", []):
                if page_spec.get("page") == domain:
                    spec_for_page = page_spec
                    break
        print(
            f"  Generating HTML scaffold for {domain} (has_spec={bool(spec_for_page)})..."
        )

        prompt = self._build_generation_prompt(
            domain, context, spec_for_page, real_site, screenshots
        )

        try:
            # Upload screenshots using shared utility
            file_ids = upload_screenshots(self.client, screenshots or [])
            
            input_content = [{"type": "input_text", "text": prompt}]
            for file_id in file_ids:
                input_content.append({"type": "input_image", "file_id": file_id})
            
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": input_content}],
            )
            print("SYNTHETIC GENERATOR RESPONSE: ", response.output_text)
            html = self._extract_html(response.output_text)
            validated_html = self._validate_and_sanitize(html)

            return validated_html

        except Exception as e:
            print(f"❌ Error generating HTML for {domain}: {e}")
            raise

    def _build_generation_prompt(
        self,
        domain: str,
        context: Dict[str, Any],
        spec_for_page: Optional[Dict[str, Any]],
        real_site: str = None,
        screenshots: List[str] | str = None,
    ) -> str:
        behavior_description = context.get("behavior")
        
        # Format specs section
        specs = ""
        if spec_for_page:
            specs = json.dumps(spec_for_page, indent=2)
        
        # Format screenshots section
        screenshots_info = ""
        if screenshots:
            screenshots_info = f"\n\Reference screenshot(s) are provided as images in this input. Use these to match the visual style, layout, colors, typography, and UI patterns of the real site."

        site_name = real_site if real_site else domain
        
        prompt = f"""You are an expert full-stack web developer tasked with creating a **high-fidelity, ultra-realistic clone** of the following website page: {site_name} (served at {domain}).

Use the information below to match both **visual appearance** and **basic interactivity** as closely as possible to the provided screenshots and specifications.

SPECIFICATIONS:
{specs}

---

### CORE OBJECTIVE
Produce a **single, self-contained HTML file** that accurately replicates the look and core functionality of the page.

Your HTML must:
- Load **instantly and fully offline**
- Look visually identical (layout, color, typography, spacing)
- Support local interactions (buttons, forms, tabs, modals)
- Expose **stable selectors** (IDs or `data-automation` attributes) for automation

---

### TECHNICAL REQUIREMENTS
1. ✅ **Single HTML file** — include all CSS and JS inline within `<style>` and `<script>` tags.  
   - No external files, fonts, or CDN links.  
   - Use base64 for any inline image assets if needed.

2. ✅ **Styling**
   - Use modern, clean CSS (Flexbox or Grid).
   - Include `<meta name="viewport" content="width=device-width, initial-scale=1">`.
   - No animations or transitions unless explicitly requested.
   - Use only system fonts (e.g. `font-family: -apple-system, Segoe UI, Roboto, sans-serif;`).

3. ✅ **Interactivity**
   - Implement local functionality (form submission, modals, etc.) using **inline JavaScript only**.
   - Simulate data persistence with `localStorage` or `sessionStorage`.
   - For mock APIs, use `fetch()` calls to local endpoints (e.g., `/mock/mail/submit`).
   - Prevent all network requests or page reloads.

4. ✅ **Navigation**
   - Include nav links for typical sections: `/mail`, `/drive`, `/calendar`, `/news`.
   - Each nav item must have both a unique ID (e.g., `#mail-link`) and `data-automation` attribute.
   - Links should not leave the page—simulate navigation locally if needed.

5. ✅ **Accessibility & Automation**
   - Provide descriptive `aria-label`s for icons and buttons.
   - All inputs, buttons, and nav elements must have stable IDs or `data-automation` tags.

6. ✅ **Code Style**
   - Clean, readable, well-indented HTML.
   - Include comments that describe logical sections and intended behaviors.
   - No explanatory text or markdown—output must start with `<!DOCTYPE html>`.

---

### VISUAL STYLE
- Match the color palette, spacing, and typography of {domain}.
- Use modern layout principles (Flexbox or Grid).
- Avoid unnecessary animation or shadowing unless seen in the screenshots.

---

### OUTPUT FORMAT
Return **only the complete HTML file**.

Do not include markdown formatting, explanations, or code fences.  
Begin output with `<!DOCTYPE html>` and end with `</html>`."""

        print("SYNTHETIC GENERATOR PROMPT: ", prompt)
        return prompt

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
