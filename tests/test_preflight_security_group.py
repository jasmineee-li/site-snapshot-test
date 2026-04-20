from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preflight_security_group.py"
_SPEC = importlib.util.spec_from_file_location("preflight_security_group", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_permission_covers_port = _MODULE._permission_covers_port
_required_ports = _MODULE._required_ports


def test_required_ports_collects_site_reset_and_db_ports(tmp_path: Path) -> None:
    instances_path = tmp_path / "instances.scale.json"
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "site_url": "http://203.0.113.10:7770",
                        "reset_endpoint": "http://203.0.113.10:7771/init",
                        "db_connection": "mysql://user:pass@203.0.113.10:3306/shop",
                    },
                    {
                        "site_url": "http://203.0.113.10:8888",
                        "reset_endpoint": "http://203.0.113.10:8889/init",
                        "db_connection": None,
                    },
                ]
            }
        )
    )

    assert _required_ports(instances_path) == {3306, 7770, 7771, 8888, 8889}


def test_required_ports_rejects_instance_without_explicit_port(tmp_path: Path) -> None:
    instances_path = tmp_path / "instances.scale.json"
    instances_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "site_url": "http://203.0.113.10",
                        "reset_endpoint": "http://203.0.113.10:7771/init",
                    }
                ]
            }
        )
    )

    try:
        _required_ports(instances_path)
    except ValueError as exc:
        assert "explicit port" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for site_url without explicit port")


def test_required_ports_rejects_non_object_instance(tmp_path: Path) -> None:
    instances_path = tmp_path / "instances.scale.json"
    instances_path.write_text(json.dumps({"instances": ["bad-entry"]}))

    try:
        _required_ports(instances_path)
    except ValueError as exc:
        assert "instances[0]" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-object instance entry")


def test_permission_covers_port_handles_ranges_and_all_protocols() -> None:
    assert _permission_covers_port({"IpProtocol": "tcp", "FromPort": 3306, "ToPort": 3309}, 3308)
    assert not _permission_covers_port(
        {"IpProtocol": "udp", "FromPort": 3306, "ToPort": 3309}, 3308
    )
    assert _permission_covers_port({"IpProtocol": "-1"}, 5433)
