from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from html import escape
from types import SimpleNamespace

from warp_taskgen.phase_1.rocket_chat_decisions import (
    generate_rocket_chat_conversation,
)
from warp_taskgen.phase_1.rocket_chat_task_envelope import compile_rocket_chat_benign_task
from warp_taskgen.rewards import run_reward_function
from warp_taskgen.runtime_composition import rocket_chat_conversation_decision_poc
from warp_taskgen.seeding.site_contracts import EditorSeedResult
from warp_taskgen.sites.catalog import SiteCatalog
from warp_taskgen.sites.readback import ReadbackObservation
from warp_taskgen.sites.rocketchat_readback import RocketChatThreadPanelReadbackAdapter
from warp_taskgen.sites.rocketchat_runtime import (
    RocketChatAuthSession,
    RocketChatCredentials,
    RocketChatHttpWriter,
    RocketChatRuntimeSite,
    RocketChatTransportError,
    preflight_rocket_chat_reader,
)


@dataclass
class FakeRocketChatTransport:
    username: str = "planner"
    rows: list[dict[str, object]] | None = None
    next_id: int = 0
    session_ids: list[str] | None = None
    physical_room_id: str = "physical-room-001"
    resolved_channels: list[str] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        self.rows = [] if self.rows is None else self.rows
        self.session_ids = [] if self.session_ids is None else self.session_ids
        self.resolved_channels = [] if self.resolved_channels is None else self.resolved_channels

    def login(self, credentials: RocketChatCredentials) -> RocketChatAuthSession:
        self.session_ids.append(f"session-{credentials.username}-{len(self.session_ids)}")
        return RocketChatAuthSession(
            user_id=f"uid-{credentials.username}",
            username=credentials.username,
            session_id=self.session_ids[-1],
            roles=("user",),
        )

    def channel_id(self, channel: str) -> str:
        self.resolved_channels.append(channel)
        return self.physical_room_id

    def send_message(self, *, room_id: str, body: str, thread_id: str | None = None):
        self.next_id += 1
        row: dict[str, object] = {
            "_id": f"message-{self.next_id}",
            "rid": room_id,
            "msg": body,
            "u": {"username": self.username},
        }
        if thread_id is not None:
            row["tmid"] = thread_id
        self.rows.append(row)
        return row

    def history(self, *, room_id: str):
        return tuple(row for row in self.rows if row.get("rid") == room_id)

    def thread_history(self, *, room_id: str, thread_id: str):
        return tuple(
            row for row in self.rows if row.get("rid") == room_id and row.get("tmid") == thread_id
        )

    def close(self) -> None:
        self.closed = True


class FailingAfterFirstWriteTransport(FakeRocketChatTransport):
    def send_message(self, *, room_id: str, body: str, thread_id: str | None = None):
        row = super().send_message(room_id=room_id, body=body, thread_id=thread_id)
        if len(self.rows) == 1:
            raise RocketChatTransportError("writer failed after first POST")
        return row


@dataclass
class FakeRocketChatResetter:
    calls: int = 0
    fail: bool = False

    def reset(self) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("reset owner failed")


class FakeRocketChatReadbackAdapter:
    """Injectable browser projection used only by exact readback tests."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def readback_visibility_selector(self, plan):
        correction_id = plan.identity_tokens["correction_message_id"]
        return f'[data-message-id="{correction_id}"] .message-body'

    def observe_readback_html(self, html: str, plan):
        del html
        return ReadbackObservation(
            kind="resource_signature",
            identity_tokens=plan.identity_tokens,
            payload={
                "room_id": plan.identity_tokens["room_id"],
                "room_name": plan.identity_tokens["room_name"],
                "thread_id": plan.identity_tokens["thread_id"],
                "reader_user_id": plan.identity_tokens["reader_user_id"],
                "reader_auth_context_id": plan.identity_tokens["reader_auth_context_id"],
                "independent_reader": True,
                "visible": True,
                "painted": True,
                "messages": self.rows,
            },
            signature=plan.signature,
        )


def _instance() -> dict[str, object]:
    return {
        "site_url": "http://rocketchat.test",
        "auth": {"credentials": {"username": "planner", "password": "writer-secret"}},
        "reader_auth": {
            "type": "http_headers",
            "credentials": {"username": "reviewer", "password": "reader-secret"},
            "headers": {"X-Auth-Token": "reader-token", "X-User-Id": "uid-reviewer"},
        },
    }


def _seed_result(receipt):
    tokens = {
        "attempt_id": receipt.attempt_id,
        "room_id": receipt.messages["plan"].room_id,
        "room_name": "project-alpha",
        "thread_id": receipt.messages["plan"].message_id,
        "writer_user": receipt.writer_context.username or receipt.writer_context.user_id,
        "reader_user_id": "uid-reviewer",
        "reader_auth_context_id": "reader-credentials-uid-reviewer",
    }
    for key, identity in receipt.messages.items():
        tokens[f"{key}_message_id"] = identity.message_id
        tokens[f"{key}_body_sha256"] = hashlib.sha256(identity.body.encode()).hexdigest()
    return EditorSeedResult.from_mapping(
        {
            "identity_tokens": tokens,
            "read_surface_urls": ["/channel/project-alpha"],
        },
        editor_method="rocketchat.seed_rocket_chat_conversation",
    )


def _thread_panel_html(receipt) -> str:
    rows: list[str] = []
    root_id = receipt.messages["plan"].message_id
    for key in ("plan", "update", "correction"):
        identity = receipt.messages[key]
        tmid = "" if key == "plan" else f' data-tmid="{root_id}"'
        rows.append(
            f'<li data-qa-id="UserMessage" data-id="{identity.message_id}"'
            f'{tmid} data-username="planner">'
            f'<div data-qa-type="message-body">{escape(identity.body)}</div></li>'
        )
    return (
        '<div class="rcx-thread-view"><section '
        'class="contextual-bar__content flex-tab threads">'
        '<div class="thread-list js-scroll-thread"><ul class="thread">'
        + "".join(rows)
        + "</ul></div></section></div>"
    )


def test_runtime_site_requires_all_message_ids_for_rest_identity_planning() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    site = SiteCatalog((RocketChatRuntimeSite(),)).bind(
        benchmark="theagentcompany", site="rocketchat", origin="http://rocketchat.test"
    )
    plan = site.read_surface_plan(
        seed_result=_seed_result(receipt), signature="Confirmed correction"
    )
    assert plan.verification_mode == "body_text"
    assert (
        plan.identity_tokens["correction_message_id"] == receipt.messages["correction"].message_id
    )
    broken_tokens = dict(_seed_result(receipt).write_tokens)
    broken_tokens.pop("update_message_id")
    broken = EditorSeedResult.from_mapping(
        {"identity_tokens": broken_tokens, "read_surface_urls": ["/channel/project-alpha"]},
        editor_method="rocketchat.seed_rocket_chat_conversation",
    )
    failure = site.read_surface_plan(seed_result=broken, signature="Confirmed correction")
    assert getattr(failure, "reason", "") == "missing_message_identity"


def test_runtime_site_uses_seed_resource_only_with_injected_painted_readback() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    rows = [
        {
            "logical_key": key,
            "message_id": identity.message_id,
            "room_id": identity.room_id,
            "thread_id": identity.thread_id,
            "author": identity.author,
            "body": identity.body,
        }
        for key, identity in receipt.messages.items()
    ]
    adapter = FakeRocketChatReadbackAdapter(rows)
    site = SiteCatalog((RocketChatRuntimeSite(readback_adapter=adapter),)).bind(
        benchmark="theagentcompany", site="rocketchat", origin="http://rocketchat.test"
    )
    plan = site.read_surface_plan(
        seed_result=_seed_result(receipt), signature="Confirmed correction"
    )

    assert plan.verification_mode == "seed_resource"
    assert plan.persist_readback_identity_tokens is True
    selector = site.readback_visibility_selector(plan)
    assert selector == (
        f'[data-message-id="{receipt.messages["correction"].message_id}"] .message-body'
    )
    observation = site.observe_readback_html("<main>exact</main>", plan)
    decision = site.interpret_readback(observation)
    assert decision.verified is True
    assert decision.reason == "exact_rocket_chat_conversation_painted"


def test_runtime_site_painted_readback_rejects_wrong_message_body_and_unpainted_dom() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    rows = [
        {
            "logical_key": key,
            "message_id": identity.message_id,
            "room_id": identity.room_id,
            "thread_id": identity.thread_id,
            "author": identity.author,
            "body": identity.body,
        }
        for key, identity in receipt.messages.items()
    ]
    adapter = FakeRocketChatReadbackAdapter(rows)
    site = SiteCatalog((RocketChatRuntimeSite(readback_adapter=adapter),)).bind(
        benchmark="theagentcompany", site="rocketchat", origin="http://rocketchat.test"
    )
    plan = site.read_surface_plan(
        seed_result=_seed_result(receipt), signature="Confirmed correction"
    )

    rows[1]["body"] = rows[0]["body"]  # stale same-text body under a new identity
    observed = site.observe_readback_html("<main>exact</main>", plan)
    decision = site.interpret_readback(observed)
    assert decision.verified is False
    assert decision.reason == "update_body_mismatch"

    rows[1]["body"] = receipt.messages["update"].body
    original_observer = adapter.observe_readback_html

    def unpainted(html, readback_plan):
        observation = original_observer(html, readback_plan)
        payload = dict(observation.payload)
        payload["painted"] = False
        return ReadbackObservation(
            kind=observation.kind,
            identity_tokens=observation.identity_tokens,
            payload=payload,
            signature=observation.signature,
        )

    adapter.observe_readback_html = unpainted
    observed = site.observe_readback_html("<main>dom only</main>", plan)
    decision = site.interpret_readback(observed)
    assert decision.verified is False
    assert decision.reason == "conversation_not_painted"


def test_measured_thread_panel_binds_exact_ids_order_and_body_without_fabricating_paint() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    site = SiteCatalog(
        (RocketChatRuntimeSite(readback_adapter=RocketChatThreadPanelReadbackAdapter()),)
    ).bind(benchmark="theagentcompany", site="rocketchat", origin="http://rocketchat.test")
    plan = site.read_surface_plan(
        seed_result=_seed_result(receipt), signature="Confirmed correction"
    )

    selector = site.readback_visibility_selector(plan)
    assert selector == (
        ".rcx-thread-view section.contextual-bar__content.flex-tab.threads "
        ".thread-list.js-scroll-thread ul.thread > "
        f"li[data-qa-id='UserMessage'][data-id='{receipt.messages['correction'].message_id}'] "
        "[data-qa-type='message-body']"
    )
    observed = site.observe_readback_html(_thread_panel_html(receipt), plan)
    assert isinstance(observed, ReadbackObservation)
    assert observed.payload["messages"][0]["message_id"] == receipt.messages["plan"].message_id
    assert observed.payload["messages"][1]["thread_id"] == receipt.messages["plan"].message_id
    assert "painted" not in observed.payload

    painted = replace(observed, payload={**observed.payload, "painted": True})
    decision = site.interpret_readback(painted)
    assert decision.verified is True


def test_measured_thread_panel_rejects_root_only_or_unscoped_rows() -> None:
    conversation = generate_rocket_chat_conversation()
    transport = FakeRocketChatTransport()
    receipt = RocketChatHttpWriter(_instance(), transport=transport).seed_conversation(conversation)
    adapter = RocketChatThreadPanelReadbackAdapter()
    site = SiteCatalog((RocketChatRuntimeSite(readback_adapter=adapter),)).bind(
        benchmark="theagentcompany", site="rocketchat", origin="http://rocketchat.test"
    )
    plan = site.read_surface_plan(
        seed_result=_seed_result(receipt), signature="Confirmed correction"
    )
    html = _thread_panel_html(receipt)
    root_only = html.replace(
        f' data-id="{receipt.messages["update"].message_id}"',
        ' data-id="unknown-update"',
    )
    observed = site.observe_readback_html(root_only, plan)
    assert getattr(observed, "reason", "") == "message_identity_mismatch"
    unscoped = html.replace('class="rcx-thread-view"', 'class="other-view"')
    observed = site.observe_readback_html(unscoped, plan)
    assert getattr(observed, "reason", "") == "message_count_mismatch"


def test_runtime_reader_preflight_never_reuses_writer_auth() -> None:
    result = preflight_rocket_chat_reader(
        {
            "site_url": "http://rocketchat.test",
            "auth": {"credentials": {"username": "planner", "password": "pw"}},
            "reader_auth": {
                "type": "credentials",
                "credentials": {"username": "reviewer", "password": "pw"},
            },
        }
    )
    assert result.ok is False
    assert result.reason == "reader_browser_auth_unavailable"


def test_reader_storage_state_binds_one_origin_and_physical_meteor_user(tmp_path) -> None:
    state_path = tmp_path / "reader-state.json"
    state_path.write_text(
        json.dumps(
            {
                "origins": [
                    {
                        "origin": "http://rocketchat.test",
                        "localStorage": [{"name": "Meteor.userId", "value": "uid-reviewer"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    instance = _instance()
    instance["reader_auth"] = {
        "type": "storage_state",
        "storage_state": str(state_path),
        "user_id": "uid-reviewer",
        "credentials": {"username": "reviewer", "password": "reader-secret"},
    }
    result = preflight_rocket_chat_reader(instance)
    assert result.ok is True
    assert result.reader_user_id == "uid-reviewer"
    assert result.browser_context_kwargs == {"storage_state": str(state_path)}

    mismatch = json.loads(state_path.read_text(encoding="utf-8"))
    mismatch["origins"][0]["localStorage"][0]["value"] = "uid-other"
    state_path.write_text(json.dumps(mismatch), encoding="utf-8")
    rejected = preflight_rocket_chat_reader(instance)
    assert rejected.ok is False
    assert rejected.reason == "reader_storage_identity_invalid"

    duplicate = json.loads(state_path.read_text(encoding="utf-8"))
    duplicate["origins"][0]["localStorage"][0]["value"] = "uid-reviewer"
    duplicate["origins"][0]["localStorage"].append(
        {"name": "Meteor.userId", "value": "uid-reviewer"}
    )
    state_path.write_text(json.dumps(duplicate), encoding="utf-8")
    rejected = preflight_rocket_chat_reader(instance)
    assert rejected.ok is False
    assert rejected.reason == "reader_storage_identity_invalid"


def test_phase2_admission_requires_reset_reader_and_painted_readback_contracts() -> None:
    conversation = generate_rocket_chat_conversation()
    task = compile_rocket_chat_benign_task(
        conversation,
        task_id="novel_rocketchat_1",
        instruction="Read the complete thread and return the current decision.",
    )
    instance = {
        **_instance(),
        "benchmark": "theagentcompany",
        "site_name": "rocketchat",
        "reset_endpoint": "http://reset.test:7771/init",
        "reader_auth": {
            "type": "http_headers",
            "credentials": {"username": "reviewer", "password": "reader-secret"},
            "headers": {"X-Auth-Token": "reader-token", "X-User-Id": "uid-reviewer"},
        },
    }

    unconfigured = rocket_chat_conversation_decision_poc(readback_adapter=None)
    blocked = unconfigured.phase_2_admission([task], [instance])
    assert blocked.admitted is False
    assert blocked.reason == "painted_readback_unconfigured"

    readback = FakeRocketChatReadbackAdapter([])
    configured = rocket_chat_conversation_decision_poc(readback_adapter=readback)
    admitted = configured.phase_2_admission([task], [instance])
    assert admitted.admitted is True
    assert "reset_endpoint" in admitted.checks

    without_reset = dict(instance)
    without_reset.pop("reset_endpoint")
    rejected = configured.phase_2_admission([task], [without_reset])
    assert rejected.admitted is False
    assert rejected.reason == "reset_endpoint_missing"

    drifted = {**task, "reward_function": {"eval": []}}
    rejected = configured.phase_2_admission([drifted], [instance])
    assert rejected.admitted is False
    assert rejected.reason == "task_contract_invalid"


def test_warp_local_rocket_chat_evaluator_dispatches_exact_decision() -> None:
    expected = {"owner": "Priya", "due_date": "2026-09-18"}
    passed, message = run_reward_function(
        {
            "benchmark": "theagentcompany",
            "eval": [{"evaluator": "RocketChatEvaluator", "expected": expected}],
        },
        {"benchmark": "theagentcompany"},
        SimpleNamespace(final_result=expected),
    )

    assert passed is True
    assert "[RocketChatEvaluator] PASS" in message


def test_rocket_chat_evaluator_rejects_native_task_id_authority() -> None:
    passed, message = run_reward_function(
        {
            "benchmark": "theagentcompany",
            "task_id": "native-task",
            "eval": [
                {
                    "evaluator": "RocketChatEvaluator",
                    "expected": {"owner": "Priya", "due_date": "2026-09-18"},
                }
            ],
        },
        {"benchmark": "theagentcompany"},
        SimpleNamespace(final_result={"owner": "Priya", "due_date": "2026-09-18"}),
    )

    assert passed is False
    assert "benchmark contract" in message
