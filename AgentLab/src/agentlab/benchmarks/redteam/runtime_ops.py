"""Shared app-runtime and derived seed-data materialization helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

from agentlab.benchmarks.redteam.app_artifacts import (
    ensure_within_dir,
    resolve_variant_data_path,
    validate_path_component,
)

APP_RUNTIME_PUBLIC_FILES = ("server.py", "index.html")
APP_RUNTIME_PUBLIC_DIRS = ("css", "js")


def _copy_if_changed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() == source.read_bytes():
        return
    shutil.copy2(source, target)


def _validated_source_path(app_root: Path, relative_path: str, *, field_name: str) -> Path:
    source = app_root / relative_path
    if source.is_symlink():
        raise RuntimeError(f"Refusing to materialize symlinked runtime asset: {relative_path}")
    resolved = ensure_within_dir(app_root, source, field_name)
    if not source.exists():
        raise FileNotFoundError(f"Required runtime asset missing: {source}")
    return resolved


def _copy_runtime_tree(source_dir: Path, target_dir: Path, *, app_root: Path) -> None:
    if source_dir.is_symlink():
        raise RuntimeError(
            f"Refusing to materialize symlinked runtime directory: {source_dir.relative_to(app_root).as_posix()}"
        )
    ensure_within_dir(app_root, source_dir, f"runtime directory {source_dir.name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_dir.iterdir()):
        if child.name == "__pycache__" or child.suffix == ".pyc":
            continue
        if child.is_symlink():
            raise RuntimeError(
                "Refusing to materialize symlinked runtime asset: "
                f"{child.relative_to(app_root).as_posix()}"
            )
        ensure_within_dir(app_root, child, f"runtime asset {child.name}")
        destination = target_dir / child.name
        if child.is_dir():
            _copy_runtime_tree(child, destination, app_root=app_root)
        elif child.is_file():
            shutil.copy2(child, destination)


def _validated_variant_data_path(app_dir: str | Path, variant_subdir: str) -> Path:
    app_root = Path(app_dir).resolve()
    if variant_subdir == "benign":
        return ensure_authoritative_seed_data(app_root)
    normalized_variant = validate_path_component(variant_subdir, "variant_subdir")
    variant_data = resolve_variant_data_path(
        app_root,
        normalized_variant,
        field_name="variant_subdir",
    )
    if variant_data.is_symlink():
        raise RuntimeError(
            f"Refusing to materialize symlinked variant asset: {normalized_variant}/data.js"
        )
    if not variant_data.exists():
        raise FileNotFoundError(f"Variant data.js not found: {variant_data}")
    return variant_data


def ensure_authoritative_seed_data(app_dir: str | Path) -> Path:
    """Return the canonical at-rest seed file, migrating legacy layouts if needed."""
    app_dir = Path(app_dir).resolve()
    benign_data = app_dir / "benign" / "data.js"
    legacy_data = app_dir / "js" / "data.js"

    if benign_data.exists():
        if benign_data.is_symlink():
            raise RuntimeError("Canonical benign seed data may not be symlinked")
        return benign_data
    if legacy_data.exists():
        if legacy_data.is_symlink():
            raise RuntimeError("Legacy benign seed data may not be symlinked")
        benign_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_data, benign_data)
        return benign_data
    raise FileNotFoundError(
        f"Canonical benign seed data not found: {benign_data} (legacy fallback {legacy_data})"
    )


def ensure_local_seed_materialization(
    app_dir: str | Path,
    *,
    variant_subdir: str = "benign",
) -> Path:
    """Ensure ``js/data.js`` exists as a derived copy of the selected variant."""
    app_dir = Path(app_dir).resolve()
    target_data = app_dir / "js" / "data.js"
    source_data = _validated_variant_data_path(app_dir, variant_subdir)

    _copy_if_changed(source_data, target_data)
    return target_data


def materialize_app_runtime(
    app_dir: str | Path,
    *,
    variant_subdir: str,
    runtime_dir: str | Path,
) -> Path:
    """Create an isolated runtime app with ``variant_subdir/data.js`` as ``js/data.js``."""
    app_dir = Path(app_dir).resolve()
    runtime_dir = Path(runtime_dir)
    variant_data = _validated_variant_data_path(app_dir, variant_subdir)

    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    runtime_dir.mkdir(parents=True, exist_ok=True)
    for filename in APP_RUNTIME_PUBLIC_FILES:
        source = app_dir / filename
        if source.exists():
            validated = _validated_source_path(
                app_dir,
                filename,
                field_name=f"runtime asset {filename}",
            )
            shutil.copy2(validated, runtime_dir / filename)

    for dirname in APP_RUNTIME_PUBLIC_DIRS:
        source_dir = app_dir / dirname
        if source_dir.exists():
            validated_dir = _validated_source_path(
                app_dir,
                dirname,
                field_name=f"runtime directory {dirname}",
            )
            _copy_runtime_tree(validated_dir, runtime_dir / dirname, app_root=app_dir)

    target_data = runtime_dir / "js" / "data.js"
    target_data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(variant_data, target_data)

    for filename in APP_RUNTIME_PUBLIC_FILES:
        if not (runtime_dir / filename).exists():
            raise FileNotFoundError(f"Required runtime asset missing: {app_dir / filename}")

    return runtime_dir
