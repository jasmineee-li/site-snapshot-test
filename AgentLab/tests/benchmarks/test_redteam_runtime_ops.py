from __future__ import annotations

from pathlib import Path

from agentlab.benchmarks.redteam.runtime_ops import (
    ensure_authoritative_seed_data,
    ensure_local_seed_materialization,
    materialize_app_runtime,
)


def test_ensure_authoritative_seed_data_migrates_legacy_js_seed(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    (app_dir / "js").mkdir(parents=True)
    legacy = app_dir / "js" / "data.js"
    legacy.write_text("const MODE = 'legacy';\n", encoding="utf-8")

    canonical = ensure_authoritative_seed_data(app_dir)

    assert canonical == app_dir / "benign" / "data.js"
    assert canonical.read_text(encoding="utf-8") == "const MODE = 'legacy';\n"


def test_ensure_local_seed_materialization_derives_js_from_benign(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    (app_dir / "benign").mkdir(parents=True)
    (app_dir / "benign" / "data.js").write_text("const MODE = 'benign';\n", encoding="utf-8")

    target = ensure_local_seed_materialization(app_dir, variant_subdir="benign")

    assert target == app_dir / "js" / "data.js"
    assert target.read_text(encoding="utf-8") == "const MODE = 'benign';\n"


def test_materialize_app_runtime_uses_canonical_benign_seed(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    (app_dir / "benign").mkdir(parents=True)
    (app_dir / "js").mkdir(parents=True)
    (app_dir / "css").mkdir(parents=True)
    (app_dir / "server.py").write_text("print('server')\n", encoding="utf-8")
    (app_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (app_dir / "js" / "state.js").write_text("const AppState = {};\n", encoding="utf-8")
    (app_dir / "benign" / "data.js").write_text("const MODE = 'benign';\n", encoding="utf-8")

    runtime_dir = materialize_app_runtime(
        app_dir,
        variant_subdir="benign",
        runtime_dir=tmp_path / "runtime",
    )

    assert (runtime_dir / "js" / "data.js").read_text(encoding="utf-8") == "const MODE = 'benign';\n"


def test_materialize_app_runtime_rejects_symlinked_assets(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    (app_dir / "benign").mkdir(parents=True)
    (app_dir / "css").mkdir(parents=True)
    (app_dir / "js").mkdir(parents=True)
    outside_server = tmp_path / "outside_server.py"
    outside_server.write_text("print('outside')\n", encoding="utf-8")
    (app_dir / "server.py").symlink_to(outside_server)
    (app_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (app_dir / "js" / "state.js").write_text("const AppState = {};\n", encoding="utf-8")
    (app_dir / "benign" / "data.js").write_text("const MODE = 'benign';\n", encoding="utf-8")

    try:
        materialize_app_runtime(
            app_dir,
            variant_subdir="benign",
            runtime_dir=tmp_path / "runtime",
        )
    except RuntimeError as exc:
        assert "symlinked runtime asset" in str(exc)
    else:
        raise AssertionError("expected symlinked runtime asset to be rejected")


def test_materialize_app_runtime_rejects_variant_traversal(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    (app_dir / "benign").mkdir(parents=True)
    (app_dir / "js").mkdir(parents=True)
    (app_dir / "css").mkdir(parents=True)
    (app_dir / "server.py").write_text("print('server')\n", encoding="utf-8")
    (app_dir / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (app_dir / "js" / "state.js").write_text("const AppState = {};\n", encoding="utf-8")
    (app_dir / "benign" / "data.js").write_text("const MODE = 'benign';\n", encoding="utf-8")

    try:
        materialize_app_runtime(
            app_dir,
            variant_subdir="../outside",
            runtime_dir=tmp_path / "runtime",
        )
    except ValueError as exc:
        assert "variant_subdir" in str(exc)
    else:
        raise AssertionError("expected variant traversal to be rejected")


def test_ensure_local_seed_materialization_raises_when_requested_variant_is_missing(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    (app_dir / "benign").mkdir(parents=True)
    (app_dir / "js").mkdir(parents=True)
    (app_dir / "benign" / "data.js").write_text("const MODE = 'benign';\n", encoding="utf-8")
    (app_dir / "js" / "data.js").write_text("const MODE = 'stale';\n", encoding="utf-8")

    try:
        ensure_local_seed_materialization(app_dir, variant_subdir="adversarial_demo_v0")
    except FileNotFoundError as exc:
        assert "Variant data.js not found" in str(exc)
    else:
        raise AssertionError("expected missing adversarial variant to raise")
