from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bash(command: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def test_deploy_proxy_requires_explicit_topology_without_port_map(tmp_path: Path) -> None:
    scale_map = tmp_path / "scale.conf"
    scale_map.write_text("shopping_0:7770:17770\n")
    topology_file = tmp_path / "missing-topology"

    result = _bash(
        "source scripts/deploy_benchmark_proxy.sh; load_port_map",
        env={
            "PORT_MAP_FILE": "",
            "SCALE_PORT_MAP_FILE": str(scale_map),
            "TOPOLOGY_FILE": str(topology_file),
            "BENCHMARK_TOPOLOGY": "",
        },
    )

    assert result.returncode != 0
    assert "no topology configured" in result.stderr


def test_deploy_proxy_loads_scale_map_from_topology(tmp_path: Path) -> None:
    scale_map = tmp_path / "scale.conf"
    scale_map.write_text("shopping_0:7770:17770\n")

    result = _bash(
        (
            "source scripts/deploy_benchmark_proxy.sh; "
            "load_port_map; "
            'printf \'%s:%s:%s\\n\' "${SITE_NAMES[0]}" "${REAL_PORTS[0]}" "${PROXY_PORTS[0]}"'
        ),
        env={
            "PORT_MAP_FILE": "",
            "SCALE_PORT_MAP_FILE": str(scale_map),
            "BENCHMARK_TOPOLOGY": "scale",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "shopping_0:7770:17770" in result.stdout


def test_deploy_proxy_validates_verify_host_and_token() -> None:
    result = _bash(
        "source scripts/deploy_benchmark_proxy.sh; "
        "if validate_verify_host 'bad;host' || validate_proxy_token 'not-a-token'; "
        "then echo bad; else echo ok; fi"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_deploy_proxy_skips_invalid_port_map_entries(tmp_path: Path) -> None:
    port_map = tmp_path / "ports.conf"
    port_map.write_text("bad:abc:17770\nshopping:7770:17770\n")

    result = _bash(
        (
            "source scripts/deploy_benchmark_proxy.sh; "
            "load_port_map; "
            'printf \'%s:%s:%s\\n\' "${SITE_NAMES[0]}" "${REAL_PORTS[0]}" "${PROXY_PORTS[0]}"'
        ),
        env={
            "PORT_MAP_FILE": str(port_map),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "shopping:7770:17770" in result.stdout
    assert "invalid port values" in result.stderr


def test_benchmark_host_start_skips_proxy_verification_by_default() -> None:
    result = _bash(
        "source scripts/benchmark_host.sh; if should_verify_proxy_on_start; then echo yes; else echo no; fi"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no"


def test_benchmark_host_loads_legacy_map_from_topology(tmp_path: Path) -> None:
    topology_file = tmp_path / "topology"
    topology_file.write_text("legacy\n")

    result = _bash(
        (
            "source scripts/benchmark_host.sh; "
            "load_port_map_text; "
            "printf '%s\\n' \"$PORT_MAP_SOURCE\"; "
            "printf '%s\\n' \"$PORT_MAP_TEXT\""
        ),
        env={
            "PORT_MAP_FILE": "",
            "TOPOLOGY_FILE": str(topology_file),
            "BENCHMARK_TOPOLOGY": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "legacy WebArena map" in result.stdout
    assert "reddit:9999" in result.stdout


def test_benchmark_host_proxy_verification_requires_metadata(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("secret\n")

    result = _bash(
        (
            "source scripts/benchmark_host.sh; "
            "if proxy_verification_configured; then echo yes; else echo no; fi"
        ),
        env={
            "PROXY_TOKEN_FILE": str(token_file),
            "PROXY_METADATA_FILE": str(tmp_path / "missing-metadata"),
            "PROXY_SCHEME": "",
            "PROXY_VERIFY_HOST": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no"


def test_benchmark_host_proxy_verification_uses_metadata_file(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("a" * 64 + "\n")
    metadata_file = tmp_path / "proxy.env"
    metadata_file.write_text("PROXY_SCHEME=https\nPROXY_VERIFY_HOST=proxy.example.test\n")

    result = _bash(
        (
            "source scripts/benchmark_host.sh; "
            "if proxy_verification_configured; then echo yes; else echo no; fi"
        ),
        env={
            "PROXY_TOKEN_FILE": str(token_file),
            "PROXY_METADATA_FILE": str(metadata_file),
            "PROXY_SCHEME": "",
            "PROXY_VERIFY_HOST": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "yes"


def test_benchmark_host_proxy_verification_rejects_invalid_metadata_host(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("a" * 64 + "\n")
    metadata_file = tmp_path / "proxy.env"
    metadata_file.write_text("PROXY_SCHEME=https\nPROXY_VERIFY_HOST=bad;host\n")

    result = _bash(
        (
            "source scripts/benchmark_host.sh; "
            "if proxy_verification_configured; then echo yes; else echo no; fi"
        ),
        env={
            "PROXY_TOKEN_FILE": str(token_file),
            "PROXY_METADATA_FILE": str(metadata_file),
            "PROXY_SCHEME": "",
            "PROXY_VERIFY_HOST": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no"


def test_benchmark_host_loads_proxy_port_map_from_metadata(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("a" * 64 + "\n")
    proxy_map = tmp_path / "proxy_ports.conf"
    proxy_map.write_text("custom:7770:18770\n")
    metadata_file = tmp_path / "proxy.env"
    metadata_file.write_text(
        "PROXY_SCHEME=https\n"
        "PROXY_VERIFY_HOST=proxy.example.test\n"
        f"PROXY_PORT_MAP_FILE={proxy_map}\n"
    )

    result = _bash(
        (
            "source scripts/benchmark_host.sh; "
            "load_proxy_metadata; "
            "load_proxy_port_map_text; "
            'printf \'%s\\n%s\\n\' "$PORT_MAP_SOURCE" "$PORT_MAP_TEXT"'
        ),
        env={
            "PROXY_TOKEN_FILE": str(token_file),
            "PROXY_METADATA_FILE": str(metadata_file),
            "PROXY_SCHEME": "",
            "PROXY_VERIFY_HOST": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert str(proxy_map) in result.stdout
    assert "custom:7770:18770" in result.stdout


def test_benchmark_host_verify_proxy_on_start_requires_configuration() -> None:
    result = _bash(
        "source scripts/benchmark_host.sh; "
        "VERIFY_PROXY_ON_START=1; "
        "if should_verify_proxy_on_start && ! proxy_verification_configured; then exit 7; fi"
    )

    assert result.returncode == 7
