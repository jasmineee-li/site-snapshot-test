from __future__ import annotations

from warp_taskgen.rewards.vendor_webarena import _build_webarena_environment_payload


def test_webarena_environment_payload_ignores_unsupported_sites():
    payload = _build_webarena_environment_payload(
        {
            "site_name": "shopping",
            "site_url": "http://shopping.test",
            "url_placeholders": {
                "__SHOPPING__": "http://shopping.test",
                "__WIKIPEDIA__": "http://wiki.test",
                "__GITLAB__": "http://gitlab.test",
            },
        }
    )

    assert payload == {"gitlab": ["http://gitlab.test"]}


def test_webarena_environment_payload_maps_postmill_to_reddit():
    payload = _build_webarena_environment_payload(
        {
            "site_name": "postmill",
            "site_url": "http://reddit.test",
        }
    )

    assert payload == {"reddit": ["http://reddit.test"]}
