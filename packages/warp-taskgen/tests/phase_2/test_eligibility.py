# ruff: noqa
# Auto-split from tests/test_phase_2_injections.py; shared helpers live in tests/phase_2/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401

def test_build_cell_targets_balances_across_available_cells():
    tasks = [
        {**_benign_task(), "id": "benign-1"},
        {**_benign_task(), "id": "benign-2"},
        {**_benign_task(), "id": "benign-3"},
    ]

    targets = phase_2_injections._build_cell_targets(_site_profile(), tasks[:2], tasks)

    assert sum(targets.values()) == 2
    assert len(targets) == len(phase_2_injections._FRAMINGS) * 2
