from __future__ import annotations

from worldsim.phases.phase_0_recon import _apply_proxy_to_url


def test_apply_proxy_to_url_can_override_scheme() -> None:
    assert (
        _apply_proxy_to_url(
            "http://18.117.99.179:7770",
            10000,
            scheme="https",
        )
        == "https://18.117.99.179:17770"
    )
