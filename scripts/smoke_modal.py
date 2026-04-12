"""Smoke test: verify run_claude_in_sandbox works end-to-end against Modal.

Creates a sandbox with one fake file, runs Claude Code with a trivial
prompt, and checks that the expected output file is produced.

Usage:
    uv run python scripts/smoke_modal.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Load .env before importing worldsim (which reads os.environ at import time)
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("worldsim").setLevel(logging.DEBUG)

import modal
modal.enable_output()

from worldsim.modal_sandbox import run_claude_in_sandbox

SMOKE_MODEL = "claude-haiku-4-5-20251001"


async def main() -> int:
    # Create a temp dir with one test file
    with tempfile.TemporaryDirectory() as tmp:
        test_file = Path(tmp) / "hello.txt"
        test_file.write_text("WorldSim v5 smoke test")

        print("Staging hello.txt into sandbox at /workspace/input/hello.txt ...")
        print("Running Claude Code with prompt: 'Read /workspace/input/hello.txt and write its contents to /workspace/output/result.txt'")
        print(f"Smoke model: {SMOKE_MODEL}")
        print("This may take 30-90 seconds on first run (Modal builds the base image) ...")
        print()

        outputs = await run_claude_in_sandbox(
            site_files={"/workspace/input/hello.txt": str(test_file)},
            prompt=(
                "Read the file at /workspace/input/hello.txt. "
                "Write its exact contents to /workspace/output/result.txt. "
                "Do nothing else."
            ),
            output_paths=["/workspace/output/result.txt"],
            timeout=300,
            model=SMOKE_MODEL,
        )

        result = outputs.get("/workspace/output/result.txt")
        if result is None:
            print("FAIL: /workspace/output/result.txt was not produced by Claude Code")
            print(f"Outputs: {outputs}")
            return 1

        result = result.strip()
        if "WorldSim v5 smoke test" in result:
            print(f"PASS: Claude Code read the file and wrote it back.")
            print(f"Content: {result!r}")
            return 0
        else:
            print(f"FAIL: unexpected content: {result!r}")
            print(f"Expected to contain: 'WorldSim v5 smoke test'")
            return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
