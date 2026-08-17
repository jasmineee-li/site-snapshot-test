from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = PACKAGE_ROOT / "agent_docs" / "site-onboarding.md"
LOCAL_LINK = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")
WORK_UNIT = re.compile(r"^### (\d+)\. ", re.MULTILINE)


def _guide_text() -> str:
    assert GUIDE_PATH.is_file(), f"missing Site onboarding guide: {GUIDE_PATH}"
    return GUIDE_PATH.read_text(encoding="utf-8")


def test_site_onboarding_router_reaches_one_local_guide() -> None:
    router_text = (PACKAGE_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "Adding or removing a Site" in router_text
    assert "agent_docs/site-onboarding.md" in router_text


def test_site_onboarding_guide_local_links_resolve() -> None:
    local_targets = []
    for match in LOCAL_LINK.finditer(_guide_text()):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        local_targets.append(target.split("#", 1)[0])

    assert local_targets
    for target in local_targets:
        assert (GUIDE_PATH.parent / target).resolve().exists(), target


def test_site_onboarding_work_units_have_observable_completion() -> None:
    guide_text = _guide_text()
    matches = list(WORK_UNIT.finditer(guide_text))

    assert [int(match.group(1)) for match in matches] == list(range(1, 10))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(guide_text)
        assert "**Done when:**" in guide_text[match.end() : end], match.group(0)


def test_documented_static_check_help_matches_the_installed_parser(tmp_path: Path) -> None:
    assert "uv run warp-taskgen site composition check --help" in _guide_text()
    env = {
        **os.environ,
        "PYTHONPATH": str(PACKAGE_ROOT),
        "PYTHON_DOTENV_DISABLED": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "warp_taskgen.main",
            "site",
            "composition",
            "check",
            "--help",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Active policy and live evidence" in completed.stdout
    assert "checked." in completed.stdout
    assert "--carrier" in completed.stdout
    assert "--action-kind" in completed.stdout
