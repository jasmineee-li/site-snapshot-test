#!/usr/bin/env python3
"""Record immutable runtime identity for the two Classifieds canary images.

The command runs read-only Docker inspection and writes image IDs, platform
metadata, tags, and RepoDigests to a JSON evidence file. It does not pull,
start, stop, or modify containers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _inspect(image_ref: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .}}", image_ref],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"docker image inspect returned a non-object for {image_ref}")
    return {
        "ref": image_ref,
        "id": payload.get("Id"),
        "os": payload.get("Os"),
        "architecture": payload.get("Architecture"),
        "repo_digests": payload.get("RepoDigests", []),
        "repo_tags": payload.get("RepoTags", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record immutable runtime identity for the Classifieds canary images.",
        epilog=(
            "Inputs: local Docker image references for the web and database images. "
            "Output: image IDs, platform metadata, tags, and RepoDigests at --output. "
            "Safety: Docker inspection is read-only; this command does not pull or run images."
        ),
    )
    parser.add_argument(
        "--web-image-ref",
        required=True,
        help="Resolved Classifieds web image reference to inspect.",
    )
    parser.add_argument(
        "--db-image-ref",
        required=True,
        help="Resolved Classifieds database image reference to inspect.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="JSON image evidence output path."
    )
    args = parser.parse_args()
    payload = {
        "web": _inspect(args.web_image_ref),
        "db": _inspect(args.db_image_ref),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
