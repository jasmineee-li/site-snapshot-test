"""Per-site core surface allowlist for Path A placement constraint.

A surface kind is in the core allowlist if and only if:
  1. A regular authenticated non-admin user can write content to it via the
     platform's public API (no placeholders, no admin fields, no platform chrome)
  2. The surface is naturally traversed by an agent doing a typical benign task
     on its host page (no buried sub-pages, no rarely-visited admin sections)

These two criteria together model a competent attacker maximizing encounter
probability under the WASP threat model. See docs/handoffs/wasp-aligned-
scoping-decision.md and the Path A handoff for the research-validity argument.

Out-of-scope kinds (label.description, wiki.body, snippet.description, profile.*,
project.description, group.description, milestone.description, comment_form_*)
must be filtered out at contract synthesis even if the editor module knows how
to write to them.
"""

from __future__ import annotations

CORE_SURFACES: dict[str, frozenset[str]] = {
    "gitlab": frozenset(
        {
            "issue.title",
            "issue.description",
            "mr.title",
            "mr.description",
            "note.body",
        }
    ),
    "reddit": frozenset(
        {
            "submission.title",
            "submission.body",
            "comment.body",
        }
    ),
}

RETIRED_ACTIVE_CARRIER_SURFACES: dict[str, frozenset[str]] = {
    "gitlab": frozenset({"issue.title", "mr.title"}),
    "reddit": frozenset({"submission.title"}),
}

# The editor registry predates Path A and uses attach-method IDs. Canonicalize
# them at the exposure-contract boundary so downstream datasets speak in the
# paper/methodology vocabulary while editor dispatch remains unchanged.
LEGACY_SURFACE_ALIASES: dict[str, dict[str, str]] = {
    "gitlab": {
        "issue_title": "issue.title",
        "issue_title_list": "issue.title",
        "issue_list_title": "issue.title",
        "issue_title_in_list": "issue.title",
        "issue_description": "issue.description",
        "issue_description_detail": "issue.description",
        "issue_detail_description": "issue.description",
        "mr_title": "mr.title",
        "mr_title_list": "mr.title",
        "mr_list_title": "mr.title",
        "mr_title_in_list": "mr.title",
        "mr_description": "mr.description",
        "mr_description_detail": "mr.description",
        "mr_detail_description": "mr.description",
        "issue_note": "note.body",
        "issue_note_body": "note.body",
        "note_on_issue": "note.body",
        "note_body_on_issue": "note.body",
        "mr_note": "note.body",
        "mr_note_body": "note.body",
        "note_on_mr": "note.body",
        "note_body_on_mr": "note.body",
    },
    "reddit": {
        "submission_title": "submission.title",
        "submission_title_forum_listing": "submission.title",
        "submission_title_listing": "submission.title",
        "submission_title_feed": "submission.title",
        "submission_title_detail": "submission.title",
        "submission_body": "submission.body",
        "submission_body_post_detail": "submission.body",
        "submission_body_detail": "submission.body",
        "comment_body": "comment.body",
        "comment_body_post_detail": "comment.body",
        "comment_body_thread": "comment.body",
        "comment_body_detail": "comment.body",
    },
}


def canonical_core_surface(site: str, surface_id: str | None) -> str | None:
    """Return the Path A canonical surface id for a raw editor surface id."""
    if not surface_id:
        return None
    site_key = site.strip().lower()
    surface = surface_id.strip()
    if not surface:
        return None
    return LEGACY_SURFACE_ALIASES.get(site_key, {}).get(surface, surface)


def is_core_surface(site: str, surface_id: str | None) -> bool:
    """Return True iff (site, surface_id) is in the per-site core allowlist."""
    canonical = canonical_core_surface(site, surface_id)
    if not canonical:
        return False
    allow = CORE_SURFACES.get(site.strip().lower())
    if allow is None:
        # Unknown site: fail closed. WASP scope is gitlab + reddit; anything
        # else should not reach Phase 2 placement at all.
        return False
    return canonical in allow


def retired_carrier_reason(site: str, surface_id: str | None) -> str | None:
    """Return the active-carrier retirement reason for a canonicalized surface."""
    canonical = canonical_core_surface(site, surface_id)
    if not canonical:
        return None
    retired = RETIRED_ACTIVE_CARRIER_SURFACES.get(site.strip().lower(), frozenset())
    if canonical in retired:
        return "retired_title_carrier_surface"
    return None


def is_active_carrier_surface(site: str, surface_id: str | None) -> bool:
    """Return True iff the surface is currently admissible as an IPI carrier.

    ``CORE_SURFACES`` answers whether a normal user can write a WASP-scope UGC
    field. Active carrier policy is stricter: retired title fields stay
    canonical for old artifacts and benign labels, but new Phase 1/2/4 mainline
    runs must not use them as injected payload carriers.
    """
    return is_core_surface(site, surface_id) and retired_carrier_reason(site, surface_id) is None
