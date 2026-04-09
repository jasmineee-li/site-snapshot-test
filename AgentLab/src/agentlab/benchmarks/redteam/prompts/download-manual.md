Task: Download and Structure GUI User Manuals For Browser-Sim

You are preparing the committed manual corpus used for shared-app generation.

Input assumptions:
- Run from the Browser-Sim repo root or an isolated workspace rooted at the repo.
- The list of entry-point URLs is provided in `apps/user-manuals/{platform}/urls.txt`.
- The target output directory is `apps/user-manuals/{platform}/`.

Current rollout note:
- The active pilot is Amazon.
- The target app scope is intentionally narrow: home page, product discovery,
  product detail page (PDP), reviews on the PDP, and one simple in-page action
  on the PDP (recommended default: add to cart).
- Do not broaden the manual crawl toward checkout, returns, seller tooling,
  account settings, or order history unless explicitly requested.

1. Input Source
   - Read the entry-point URLs from `apps/user-manuals/{platform}/urls.txt`.
   - Use these URLs as starting points.
   - Crawl additional relevant pages only when they are clearly part of the
     same GUI user-manual surface.

2. Scope Limitation (Important)
   - Download only documentation related to the GUI (Graphical User Interface).
   - Skip documentation related to:
     - APIs
     - SDKs
     - developer guides
     - CLI tools
     - integration or backend configuration unless strictly GUI-related
   - Exclude marketing pages, investor pages, press pages, blog posts, and
     unrelated support material.

3. Pilot-Scope Limitation (Amazon)
   - Prioritize documentation relevant to:
     - browsing from the home page
     - navigating into products
     - product detail page structure and interactions
     - reviews and review visibility
     - add-to-cart or equivalent simple PDP action
   - Do not intentionally expand into:
     - checkout
     - payments
     - returns
     - account settings
     - order history
     - seller or marketplace management

3A. Fallback For Anti-Bot-Protected Sites
   - If the target site blocks direct automated fetches, do not keep escalating
     the scraper.
   - Instead, use a real browser session to load the help pages manually and
     save the relevant source pages locally under
     `apps/user-manuals/{platform}/raw/`.
   - When local raw captures are present, treat them as the source material and
     convert/clean them into the committed markdown corpus.
   - Preserve the original URL in each markdown file's `Source:` header even
     when the immediate conversion input is a local HTML or PDF capture.

4. Output Format
   - Convert all content into Markdown format.
   - Store it under `apps/user-manuals/{platform}/{feature-area}/`.

5. Link Handling
   - Convert internal links to relative links when practical so the corpus is
     usable locally.
   - External links may remain absolute.

6. Assets (Optional but Preferred)
   - Download images locally if reasonably feasible.
   - Update image references to use relative paths.
   - If image handling is too complex, skip it rather than degrading the text.

7. Storage Conventions
   - Every output file must begin with:

     # {Article Title}

     Source: {original-url}

     ---

   - Preserve a clear hierarchy by feature area when possible.

8. Quality Requirements
   - Remove scripts, nav bars, headers, footers, and UI clutter.
   - Keep only meaningful documentation content.
   - Maintain consistent formatting across all manuals.
   - Favor faithful functional documentation, not visual cloning.

9. Markdown Formatting Rules
   - One paragraph = one line (no hard wrapping).
   - One list item = one line.
   - Preserve structural blank lines between headings, paragraphs, lists, code
     blocks, and tables.
   - No Unicode Private Use Area (PUA) characters.
   - All code blocks must be fenced.
   - Do not merge structural blocks accidentally.

10. Post-Processing Validation
    - Scan for wrapped paragraphs and unwrap them.
    - Scan for PUA characters; target zero occurrences.
    - Spot-check 3-5 files per directory for:
      - proper `Source:` headers
      - correct heading/body separation
      - unbroken lists
      - correctly separated tables and paragraphs
      - closed code fences

11. Completion Standard
    - The result should be a committed manual snapshot that can be referenced by
      `docs_path` in `platform_manifest.json`.
    - The corpus should be narrow enough to support the current app boundary
      without silently broadening platform scope.
