from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from warp_taskgen import seeding
from warp_taskgen.editors.base import EditorError
from warp_taskgen.phase_4 import preflight as phase_4_preflight
from warp_taskgen.runtime_composition import classifieds_listing_reply_poc
from warp_taskgen.sites.classifieds_editor import ClassifiedsEditor
from warp_taskgen.sites.classifieds_writer import (
    ClassifiedsAuthenticatedEditor,
    classifieds_editor_factory,
)

ORIGIN = "http://127.0.0.1:18080"
LISTING_URL = f"{ORIGIN}/index.php?page=item&id=17"
BODY = "Visible details SIGNATURE-WRITER"


def _state(path: Path, *, domain: str = "127.0.0.1") -> Path:
    path.write_text(
        json.dumps(
            {
                "cookies": [
                    {
                        "name": "session_id",
                        "value": "fixture-cookie-value",
                        "domain": domain,
                        "path": "/",
                    }
                ],
                "origins": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _instance(state_path: Path) -> dict[str, object]:
    return {
        "site_name": "classifieds",
        "benchmark": "visualwebarena",
        "site_url": ORIGIN,
        "auth": {
            "type": "storage_state",
            "storage_state": {"path": str(state_path)},
        },
    }


def test_classifieds_writer_factory_loads_only_same_origin_cookies(tmp_path: Path) -> None:
    state_path = _state(tmp_path / "writer.json")
    session = requests.Session()

    editor = classifieds_editor_factory(_instance(state_path), session)

    assert isinstance(editor, ClassifiedsEditor)
    assert isinstance(editor, ClassifiedsAuthenticatedEditor)
    assert session.cookies.get_dict() == {"session_id": "fixture-cookie-value"}


@pytest.mark.parametrize("auth", [None, {"type": "none"}, {"type": "storage_state"}])
def test_classifieds_writer_factory_requires_explicit_storage_state_path(
    tmp_path: Path,
    auth: dict[str, object] | None,
) -> None:
    instance = {"site_url": ORIGIN, "auth": auth}
    session = requests.Session()

    with pytest.raises(EditorError, match="storage_state"):
        classifieds_editor_factory(instance, session)

    assert session.cookies.get_dict() == {}


def test_classifieds_writer_factory_rejects_missing_or_malformed_state(tmp_path: Path) -> None:
    missing = _instance(tmp_path / "missing.json")
    session = requests.Session()
    with pytest.raises(EditorError, match="storage_state"):
        classifieds_editor_factory(missing, session)
    assert session.cookies.get_dict() == {}

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("not-json", encoding="utf-8")
    malformed = _instance(malformed_path)
    with pytest.raises(EditorError, match="JSON"):
        classifieds_editor_factory(malformed, requests.Session())


def test_classifieds_writer_factory_rejects_foreign_cookie_state(tmp_path: Path) -> None:
    state_path = _state(tmp_path / "writer.json", domain="other.test")
    session = requests.Session()

    with pytest.raises(EditorError, match="same-origin"):
        classifieds_editor_factory(_instance(state_path), session)

    assert session.cookies.get_dict() == {}


def test_runtime_registration_preserves_probe_base_state_and_writer_auth(tmp_path: Path) -> None:
    state_path = _state(tmp_path / "writer.json")
    composition = classifieds_listing_reply_poc()
    registration = composition.seed_registry.get("visualwebarena", "classifieds")
    assert registration is not None
    factory = registration.editor_factory

    # Phase 4 calls this class-level seam before mutating the host.
    factory.probe_base_state({"site_name": "classifieds", "site_url": ORIGIN})
    result = phase_4_preflight._probe_seed_base_state(
        _instance(state_path),
        benchmark="visualwebarena",
        seed_registry=composition.seed_registry,
    )
    assert result.ok is True
    editor = registration.create(_instance(state_path), requests.Session())
    assert isinstance(editor, ClassifiedsAuthenticatedEditor)


@pytest.mark.parametrize(
    "site_url",
    [
        "https://classifieds.test:18080",
        "http://classifieds.test:18080",
        "http://localhost:18080",
        "http://127.0.0.1",
        "http://127.0.0.1:18080/path",
        "http://127.0.0.1:18080?public=1",
        "http://user:pass@127.0.0.1:18080",
    ],
)
def test_runtime_registration_rejects_public_or_non_exact_canary_origins(
    tmp_path: Path,
    site_url: str,
) -> None:
    state_path = _state(tmp_path / "writer.json")
    composition = classifieds_listing_reply_poc()
    registration = composition.seed_registry.get("visualwebarena", "classifieds")
    assert registration is not None

    with pytest.raises(EditorError, match="exact loopback"):
        registration.editor_factory.probe_base_state(
            {"site_name": "classifieds", "site_url": site_url}
        )
    with pytest.raises(EditorError, match="exact loopback"):
        registration.create(_instance(state_path) | {"site_url": site_url}, requests.Session())


def test_generic_apply_data_seed_uses_declared_writer_storage_state(
    tmp_path: Path, monkeypatch
) -> None:
    state_path = _state(tmp_path / "writer.json")
    composition = classifieds_listing_reply_poc()
    registration = composition.seed_registry.get("visualwebarena", "classifieds")
    assert registration is not None
    seed = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "visualwebarena",
                "site": "classifieds",
                "method": "create_listing_reply",
                "args": {"listing_id": "17", "body": BODY},
            }
        ],
    }
    instance = {
        **_instance(state_path),
        "seed_task": {
            "id": "classifieds-auth-seam",
            "site": "classifieds",
            "benchmark": "visualwebarena",
            "benign_target_resource": {"kind": "listing", "anchors": {"listing_id": "17"}},
        },
    }
    posted: dict[str, object] = {}
    cleaned: dict[str, object] = {}

    monkeypatch.setattr(
        ClassifiedsAuthenticatedEditor,
        "_fetch_form_state",
        lambda _self, *_args, **_kwargs: {
            "action": "/index.php",
            "fields": {
                "action": "add_comment",
                "page": "item",
                "id": "17",
                "replyId": "",
                "authorName": "fixture-writer",
                "authorEmail": "fixture-writer@example.test",
                "title": "",
                "body": "",
                "octoken": "csrf-17",
            },
        },
    )
    monkeypatch.setattr(
        ClassifiedsAuthenticatedEditor,
        "_submit_exact_form",
        lambda _self, action, fields: (
            posted.update(action=action, fields=dict(fields))
            or SimpleNamespace(status_code=302, headers={"Location": LISTING_URL}, text="")
        ),
    )
    monkeypatch.setattr(
        ClassifiedsAuthenticatedEditor,
        "_form_get",
        lambda _self, *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            url=LISTING_URL,
            text=(
                '<div class="comment"><h3><strong>Additional listing details</strong>'
                " <em>by fixture-writer:</em></h3>"
                f'<p>{BODY}</p><a class="comment-reply" data-id="88">Reply</a></div>'
            ),
        ),
    )
    monkeypatch.setattr(
        ClassifiedsAuthenticatedEditor,
        "_delete_listing_reply",
        lambda _self, **kwargs: cleaned.update(kwargs),
    )

    cleanup, metadata = seeding.apply_data_seed(
        seed,
        instance,
        seed_registry=composition.seed_registry,
    )

    assert posted["fields"]["body"] == BODY
    assert metadata["write_tokens"]["reply_id"] == "88"
    assert cleanup is not None
    cleanup.cleanup()
    assert cleaned["reply_id"] == "88"
