from __future__ import annotations

from pathlib import Path

import yaml


def test_override_uses_explicit_advertise_and_bind_hosts() -> None:
    """Post-WASP scope (2026-04-21): only gitlab + reddit services. The
    map-volume `name:` overrides were dropped along with the map service."""
    repo_root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((repo_root / "scripts" / "webarena-compose-override.yml").read_text())

    services = compose["services"]
    assert sorted(services) == ["gitlab", "reddit"]

    gitlab_ports = services["gitlab"]["ports"]
    reddit_ports = services["reddit"]["ports"]
    assert gitlab_ports == [
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_GITLAB_PORT:-8023}:8023",
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_GITLAB_ENV_CTRL_PORT:-8024}:8877",
        "${WORLDSIM_DB_BIND_HOST:?set WORLDSIM_DB_BIND_HOST}:${WA_GITLAB_DB_PORT:-5433}:5432",
    ]
    assert reddit_ports == [
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_REDDIT_PORT:-9999}:80",
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_REDDIT_ENV_CTRL_PORT:-9998}:8877",
        "${WORLDSIM_DB_BIND_HOST:?set WORLDSIM_DB_BIND_HOST}:${WA_REDDIT_DB_PORT:-5434}:5432",
    ]

    # No top-level `volumes:` key after the map cleanup.
    assert "volumes" not in compose
