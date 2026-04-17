from __future__ import annotations

from pathlib import Path

import yaml


def test_override_uses_explicit_advertise_and_bind_hosts_and_maps_canonical_map_volumes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((repo_root / "scripts" / "webarena-compose-override.yml").read_text())

    shopping_ports = compose["services"]["shopping"]["ports"]
    gitlab_ports = compose["services"]["gitlab"]["ports"]
    map_ports = compose["services"]["map"]["ports"]
    assert shopping_ports == [
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_SHOPPING_PORT:-7770}:80",
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_SHOPPING_ENV_CTRL_PORT:-7771}:8877",
        "${WORLDSIM_DB_BIND_HOST:?set WORLDSIM_DB_BIND_HOST}:${WA_SHOPPING_DB_PORT:-3306}:3306",
    ]
    assert gitlab_ports == [
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_GITLAB_PORT:-8023}:8023",
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_GITLAB_ENV_CTRL_PORT:-8024}:8877",
        "${WORLDSIM_DB_BIND_HOST:?set WORLDSIM_DB_BIND_HOST}:${WA_GITLAB_DB_PORT:-5433}:5432",
    ]
    assert map_ports == [
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_MAP_PORT:-3030}:8080",
        "${WORLDSIM_BIND_HOST:?set WORLDSIM_BIND_HOST}:${WA_MAP_ENV_CTRL_PORT:-3031}:8877",
        "${WORLDSIM_DB_BIND_HOST:?set WORLDSIM_DB_BIND_HOST}:${WA_MAP_DB_PORT:-5435}:5432",
    ]
    assert compose["services"]["shopping"]["environment"] == [
        "WA_ENV_CTRL_EXTERNAL_SITE_URL=http://${WORLDSIM_ADVERTISE_HOST:?set WORLDSIM_ADVERTISE_HOST}:${WA_SHOPPING_PORT:-7770}"
    ]

    volumes = compose["volumes"]
    assert volumes["webarena-verified-map-tiles"]["name"] == "webarena-verified-map-tiles"
    assert volumes["webarena-verified-map-style"]["name"] == "webarena-verified-map-style"
    assert (
        volumes["webarena-verified-map-website-db"]["name"]
        == "webarena-verified-map-website-db"
    )
