from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

GUIDE_PATH = Path(__file__).resolve().parents[1] / "agent_docs" / "classifieds-canary.md"
LOCAL_LINK = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def _guide_text() -> str:
    assert GUIDE_PATH.is_file(), f"missing Classifieds guide: {GUIDE_PATH}"
    return GUIDE_PATH.read_text(encoding="utf-8")


def _canonical_command(guide_text: str) -> list[str]:
    for line in guide_text.splitlines():
        if "scripts/run_classifieds_canary.py" not in line or "--host-config" not in line:
            continue
        command = shlex.split(line.strip())
        if "scripts/run_classifieds_canary.py" in command:
            return [token.replace("<run-name>", "guide-test") for token in command]
    raise AssertionError("guide does not document the canonical Classifieds canary command")


def test_classifieds_router_reaches_one_local_guide() -> None:
    package_root = GUIDE_PATH.parents[1]
    router = package_root / "CLAUDE.md"
    router_text = router.read_text(encoding="utf-8")

    assert "agent_docs/classifieds-canary.md" in router_text
    assert GUIDE_PATH.is_file()


def test_classifieds_guide_local_links_resolve() -> None:
    guide_text = _guide_text()
    local_targets = []
    for match in LOCAL_LINK.finditer(guide_text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local_targets.append(target.split("#", 1)[0])

    assert local_targets, "guide should point to local source-of-truth documents"
    for target in local_targets:
        assert (GUIDE_PATH.parent / target).resolve().exists(), target


def test_classifieds_canonical_command_matches_parser_without_host_access() -> None:
    package_root = GUIDE_PATH.parents[1]
    guide_text = _guide_text()
    documented = _canonical_command(guide_text)

    assert documented[:3] == ["uv", "run", "python"]
    script_index = documented.index("scripts/run_classifieds_canary.py")
    parser_args = documented[script_index + 1 :]
    assert "--host-config" in parser_args
    assert "--run-dir" in parser_args

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(package_root), env.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, *documented[3:], "--help"],
        cwd=package_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--host-config" in completed.stdout
    assert "--run-dir" in completed.stdout
