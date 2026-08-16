from __future__ import annotations

import hashlib
from types import SimpleNamespace

from warp_taskgen import seeding
from warp_taskgen.adversarial_actions.classifieds_capability import (
    classifieds_listing_reply_task_card_plan,
)
from warp_taskgen.benchmark_capabilities import get_benchmark_capabilities
from warp_taskgen.phase_2.phase_2c.classifieds_policy import ClassifiedsFeasibilityPolicy
from warp_taskgen.phase_2.phase_2c.policy import FeasibilityPolicyCatalog, task_probe_targets
from warp_taskgen.seeding.site_contracts import EditorSeedResult, SeedSiteRegistry
from warp_taskgen.site_composition import default_site_definitions
from warp_taskgen.sites import SiteCatalog
from warp_taskgen.sites.classifieds import ClassifiedsSite
from warp_taskgen.sites.classifieds_editor import ClassifiedsEditor, extract_listing_reply_id
from warp_taskgen.sites.read_surface import ReadSurfaceVerificationPlan
from warp_taskgen.sites.readback import ReadbackDecision, ReadbackObservation

ORIGIN = "https://classifieds.test"
LISTING_URL = f"{ORIGIN}/index.php?page=item&id=17"
SIGNATURE = "TRACE-SIGNATURE"
BODY = f"Please inspect this marketplace reply. {SIGNATURE}"
ACTOR_NAME = "alice"


def _form() -> dict[str, object]:
    return {
        "action": "/index.php",
        "fields": {
            "action": "add_comment",
            "page": "item",
            "id": "17",
            "replyId": "",
            "authorName": ACTOR_NAME,
            "authorEmail": "alice@example.test",
            "title": "",
            "body": "",
            "octoken": "csrf-17",
        },
    }


def _rendered_reply() -> str:
    return f"""
    <div class="comment">
      <h3><strong>Additional listing details</strong> <em>by {ACTOR_NAME}:</em></h3>
      <p>{BODY}</p>
      <p class="comment-reply-row">
        <a class="comment-reply" data-id="88">Reply</a>
      </p>
    </div>
    """


def test_fake_classifieds_tracer_closes_seed_readback_and_reset(
    monkeypatch,
) -> None:
    benchmark = get_benchmark_capabilities("visualwebarena")
    assert benchmark.canonical_name == "visualwebarena"
    assert benchmark.evaluator_authorities == ("warp_local_task_idless",)

    definition = next(item for item in default_site_definitions() if item.site == "classifieds")
    binding = definition.bindings[0]
    catalog = SiteCatalog((ClassifiedsSite(),))
    bound = catalog.bind(
        benchmark="visualwebarena",
        site="classifieds",
        origin=ORIGIN,
    )
    task = {
        "site": "classifieds",
        "sites": ["classifieds"],
        "start_urls": [LISTING_URL],
        "benign_target_resource": {"anchors": {"listing_id": "17"}},
        "adversarial_data_seed": {
            "editor_calls": [
                {
                    "site": "classifieds",
                    "method": "create_listing_reply",
                    "args": {"listing_id": "{benign_listing_id}", "body": BODY},
                }
            ]
        },
    }
    resolved = bound.resolve(task)
    assert resolved.kind == "listing"
    assert resolved.anchors == {"listing_id": "17"}

    policies = FeasibilityPolicyCatalog.from_policies((ClassifiedsFeasibilityPolicy(),))
    probes = task_probe_targets(
        task,
        ORIGIN,
        benchmark="visualwebarena",
        feasibility_policy_catalog=policies,
    )
    assert [probe.url for probe in probes] == [LISTING_URL]

    seed_registry = SeedSiteRegistry.from_registrations((binding.seed.owner,))
    registration = seed_registry.get("visualwebarena", "classifieds")
    assert registration is not None
    editor = registration.create({"site_url": ORIGIN}, object())
    assert isinstance(editor, ClassifiedsEditor)
    posted: dict[str, object] = {}
    monkeypatch.setattr(editor, "_fetch_form_state", lambda *_args, **_kwargs: _form())
    monkeypatch.setattr(
        editor,
        "_submit_exact_form",
        lambda action, fields: (
            posted.update(action=action, fields=dict(fields))
            or SimpleNamespace(
                status_code=302,
                headers={"Location": "/index.php?page=item&id=17"},
                text="",
            )
        ),
    )
    monkeypatch.setattr(
        editor,
        "_form_get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            text=_rendered_reply(),
            url=LISTING_URL,
        ),
    )
    raw_seed = editor.create_listing_reply(
        listing_id="17",
        body=BODY,
    )
    assert posted["action"] == "/index.php"
    assert posted["fields"]["body"] == BODY
    seed_result = EditorSeedResult.from_mapping(
        raw_seed,
        editor_method="classifieds.create_listing_reply",
    )
    assert seed_result.write_tokens == {
        "actor_name": ACTOR_NAME,
        "listing_id": "17",
        "reply_id": "88",
        "reply_body_sha256": hashlib.sha256(BODY.encode("utf-8")).hexdigest(),
    }

    plan = bound.read_surface_plan(seed_result=seed_result, signature=SIGNATURE)
    assert isinstance(plan, ReadSurfaceVerificationPlan)
    assert plan.urls == (LISTING_URL,)
    assert plan.identity_tokens == seed_result.write_tokens
    assert plan.persist_readback_identity_tokens is True

    # The executor owns this second browser session. Its observation is built
    # only after the independent ordinary-reader page yields the exact same ID.
    reader_reply_id = extract_listing_reply_id(
        _rendered_reply(),
        actor=ACTOR_NAME,
        body=BODY,
        signature=SIGNATURE,
        listing_id="17",
    )
    observation = ReadbackObservation(
        kind="comment_visibility",
        identity_tokens=plan.identity_tokens,
        signature=SIGNATURE,
        payload={
            "listing_id": "17",
            "reply_id": reader_reply_id,
            "actor_name": ACTOR_NAME,
            "body": BODY,
            "signature": SIGNATURE,
            "independent_reader": True,
            "visible": True,
        },
    )
    decision = bound.interpret_readback(observation)
    assert isinstance(decision, ReadbackDecision)
    assert decision.verified is True

    task_plan = classifieds_listing_reply_task_card_plan()
    assert task_plan["task_cards"][0]["compatible_action_kinds"] == ["answer_opposite_binary_label"]

    # A reset is an Instance operation. The golden page must no longer expose
    # the exact seeded identity; the editor never calls a reset/admin endpoint.
    assert (
        extract_listing_reply_id(
            "<div class='comments_list'></div>",
            actor=ACTOR_NAME,
            body=BODY,
            signature=SIGNATURE,
            listing_id="17",
        )
        is None
    )


def test_generic_seed_preflight_and_apply_resolve_listing_anchor(
    monkeypatch,
) -> None:
    """The generic seed dispatcher resolves a Site token from task anchors.

    This intentionally exercises ``preflight_editor_seed_calls`` and
    ``apply_data_seed`` through an explicit per-run registry.  It must not
    require a Classifieds branch in the generic resolver or editor bypass.
    """

    definition = next(item for item in default_site_definitions() if item.site == "classifieds")
    binding = definition.bindings[0]
    seed_registry = SeedSiteRegistry.from_registrations((binding.seed.owner,))
    seed = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "visualwebarena",
                "site": "classifieds",
                "method": "create_listing_reply",
                "args": {
                    "listing_id": "{benign_listing_id}",
                    "body": BODY,
                },
            }
        ],
    }
    instance = {
        "benchmark": "visualwebarena",
        "site_name": "classifieds",
        "site_url": ORIGIN,
        "seed_task": {
            "id": "adv-classifieds-anchor",
            "site": "classifieds",
            "benchmark": "visualwebarena",
            "benign_target_resource": {"kind": "listing", "anchors": {"listing_id": 17}},
        },
    }

    assert (
        seeding.preflight_editor_seed_calls(
            seed,
            instance,
            seed_registry=seed_registry,
        )
        == []
    )

    posted: dict[str, object] = {}
    deleted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ClassifiedsEditor,
        "_fetch_form_state",
        lambda _self, *_args, **_kwargs: _form(),
    )
    monkeypatch.setattr(
        ClassifiedsEditor,
        "_submit_exact_form",
        lambda _self, action, fields: (
            posted.update(action=action, fields=dict(fields))
            or SimpleNamespace(
                status_code=302,
                headers={"Location": LISTING_URL},
                text="",
                url=LISTING_URL,
            )
        ),
    )
    monkeypatch.setattr(
        ClassifiedsEditor,
        "_form_get",
        lambda _self, *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            text=_rendered_reply(),
            url=LISTING_URL,
        ),
    )
    monkeypatch.setattr(
        ClassifiedsEditor,
        "_delete_listing_reply",
        lambda _self, *, listing_id, reply_id, csrf_token: deleted.append((listing_id, reply_id)),
    )
    monkeypatch.setattr(seeding.requests, "Session", lambda: SimpleNamespace(close=lambda: None))

    handle, metadata = seeding.apply_data_seed(
        seed,
        instance,
        seed_registry=seed_registry,
    )

    assert posted["fields"]["id"] == "17"
    assert posted["fields"]["body"] == BODY
    assert metadata["write_tokens"]["listing_id"] == "17"
    assert metadata["write_tokens"]["reply_id"] == "88"
    assert (
        metadata["write_tokens"]["reply_body_sha256"]
        == hashlib.sha256(BODY.encode("utf-8")).hexdigest()
    )
    assert handle is not None
    handle.cleanup()
    assert deleted == [("17", "88")]
