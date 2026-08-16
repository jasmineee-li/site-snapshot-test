#!/usr/bin/env python3
"""Check the Classifieds listing before, during, and after the canary.

The three modes use independent HTTP sessions and write redacted JSON
evidence. The probe writes through the regular participant editor in
``write-read`` mode, reads as an anonymous user, and observes the golden-state
reset in ``absence`` mode. It never calls an admin or reset endpoint and never
queries the database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from warp_taskgen.classifieds_canary import validate_classifieds_loopback_origin
from warp_taskgen.editors.base import EditorError
from warp_taskgen.sites.classifieds_reply_html import (
    extract_listing_reply_id,
    normalize_reply_body,
    rendered_listing_reply_id_presence,
    rendered_listing_surface_present,
)
from warp_taskgen.sites.classifieds_writer import ClassifiedsAuthenticatedEditor


def _listing_url(site_url: str, listing_id: str) -> str:
    origin = validate_classifieds_loopback_origin(site_url)
    return f"{origin}/index.php?page=item&id={listing_id}"


def _get_public(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=30, allow_redirects=True)
    if response.status_code != 200:
        raise RuntimeError(f"Classifieds reader returned HTTP {response.status_code}")
    parsed = urlsplit(response.url)
    expected = urlsplit(url)
    if (parsed.scheme, parsed.netloc, parsed.path, parsed.query) != (
        expected.scheme,
        expected.netloc,
        expected.path,
        expected.query,
    ):
        raise RuntimeError(
            f"Classifieds reader redirected away from seeded listing: {response.url}"
        )
    return response.text


def _write_read(args: argparse.Namespace) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    evidence_path = Path(args.evidence)
    if evidence_path.is_file():
        loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            previous = loaded
    writer_session = requests.Session()
    editor = ClassifiedsAuthenticatedEditor(
        {
            "site_url": args.site_url,
            "auth": {
                "type": "storage_state",
                "storage_state": {"path": args.writer_storage_state},
            },
            "agent_auth": {"type": "none"},
        },
        writer_session,
    )
    if not editor.probe_authenticated():
        raise RuntimeError("writer storage state did not prove a regular participant session")
    try:
        result = editor.create_listing_reply(listing_id=args.listing_id, body=args.body)
    except EditorError as exc:
        raise RuntimeError(f"Classifieds writer failed: {exc.kind}: {exc}") from exc
    actor = str(result.get("actor_name") or "").strip()
    reply_id = str(result.get("reply_id") or "").strip()
    if not actor or not reply_id:
        raise RuntimeError("Classifieds editor returned incomplete identity evidence")

    reader_session = requests.Session()
    html = _get_public(reader_session, _listing_url(args.site_url, args.listing_id))
    observed_id = extract_listing_reply_id(
        html,
        actor=actor,
        body=args.body,
        listing_id=args.listing_id,
    )
    if observed_id != reply_id:
        raise RuntimeError(
            "fresh anonymous reader did not observe the exact created reply "
            f"(expected {reply_id!r}, got {observed_id!r})"
        )
    return {
        **(
            {"precondition": previous["precondition"]}
            if isinstance(previous.get("precondition"), dict)
            else {}
        ),
        "listing_id": args.listing_id,
        "reply_id": reply_id,
        "actor_name": actor,
        "body": args.body,
        "body_sha256": hashlib.sha256(normalize_reply_body(args.body).encode()).hexdigest(),
        "writer": {"regular_participant": True},
        "reader": {"independent": True, "anonymous": True, "exact_reply_id": True},
    }


def _absence(args: argparse.Namespace) -> dict[str, Any]:
    evidence_path = Path(args.evidence)
    previous = json.loads(evidence_path.read_text(encoding="utf-8"))
    actor = str(previous.get("actor_name") or "").strip()
    reply_id = str(previous.get("reply_id") or "").strip()
    if not actor or not reply_id:
        raise RuntimeError("probe evidence has no exact actor/reply identity")
    reader_session = requests.Session()
    html = _get_public(reader_session, _listing_url(args.site_url, args.listing_id))
    if not rendered_listing_surface_present(html, args.listing_id, origin=args.site_url):
        raise RuntimeError("golden-state reset did not preserve the exact listing surface")
    presence = rendered_listing_reply_id_presence(html, reply_id=reply_id)
    if presence is not False:
        state = "visible" if presence is True else "ambiguous"
        raise RuntimeError(f"golden-state reset left canary reply {state}: {reply_id}")
    return {
        **previous,
        "reset_observation": {
            "independent": True,
            "anonymous": True,
            "reply_absent": True,
            "checked_reply_id": reply_id,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the Classifieds listing before, during, or after the canary.",
        epilog=(
            "Inputs: a loopback site URL, exact listing ID, canary body, and evidence path. "
            "Output: redacted JSON evidence at --evidence. "
            "Safety: write-read uses the supplied regular-participant storage state and a "
            "separate anonymous reader session; the probe does not use admin, reset, or database access."
        ),
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    mode_help = {
        "precondition": "Confirm the exact listing exists and the canary body is absent.",
        "write-read": "Create one reply through the regular participant form and read it anonymously.",
        "absence": "Confirm the exact reply is absent after the golden-state reset.",
    }
    mode_epilog = {
        "precondition": (
            "Output: evidence JSON records the exact listing and an absent canary body. "
            "Safety: this mode uses an anonymous read and performs no mutation."
        ),
        "write-read": (
            "Output: evidence JSON records the exact reply ID, actor, and body digest. "
            "Safety: the write uses the regular participant session; readback uses a fresh anonymous session."
        ),
        "absence": (
            "Output: evidence JSON records the exact reply as absent after reset. "
            "Safety: this mode uses an anonymous read and performs no mutation."
        ),
    }
    for mode in ("precondition", "write-read", "absence"):
        command = sub.add_parser(
            mode,
            help=mode_help[mode],
            description=mode_help[mode],
            epilog=mode_epilog[mode],
        )
        command.add_argument(
            "--site-url",
            required=True,
            help="Configured loopback Classifieds origin used for the request.",
        )
        command.add_argument(
            "--listing-id",
            required=True,
            help="Exact seeded listing ID to inspect and, for write-read, update.",
        )
        command.add_argument(
            "--writer-storage-state",
            required=True,
            help="External participant storage-state file; used by write-read and required by this interface for every mode.",
        )
        command.add_argument(
            "--evidence",
            required=True,
            type=Path,
            help="JSON evidence path; write-read and precondition create it, absence reads and extends it.",
        )
        command.add_argument(
            "--body",
            required=True,
            help="Canary reply body used for the precondition check and exact readback.",
        )
    args = parser.parse_args(argv)
    if args.mode == "precondition":
        html = _get_public(requests.Session(), _listing_url(args.site_url, args.listing_id))
        if not rendered_listing_surface_present(html, args.listing_id, origin=args.site_url):
            raise RuntimeError("precondition failed: exact listing surface is unavailable")
        if normalize_reply_body(args.body) in normalize_reply_body(html):
            raise RuntimeError("precondition failed: the canary reply is already rendered")
        result: dict[str, Any] = {
            "listing_id": args.listing_id,
            "precondition": {"independent": True, "anonymous": True, "canary_absent": True},
        }
    elif args.mode == "write-read":
        result = _write_read(args)
    else:
        result = _absence(args)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"mode": args.mode, "evidence": str(args.evidence)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
