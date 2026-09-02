"""Serialized no-model proof for the Rocket.Chat notification workflow.

Run this file alone against the pinned disposable TAC instance::

    PYTEST_ROCKETCHAT_NOTIFICATION_VERTICAL_SLICE=1 \
      bash scripts/run_integration_tests.sh \
      --instances <host-owned-tac-instances.json> --quiet -- \
      -k test_rocket_chat_notification_vertical_slice --junitxml=<run>/e2.xml

The explicit opt-in matters: the test resets its one selected Rocket.Chat
instance before and after mutation.  It must not share that reset scope with
another task.  This is a source/action/evaluator smoke; it runs no model.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from warp_taskgen.phase_1.rocket_chat_contracts import (
    RocketChatObservationFailure,
    RocketChatSeedReceipt,
)
from warp_taskgen.phase_1.rocket_chat_decisions import generate_rocket_chat_conversation
from warp_taskgen.phase_1.rocket_chat_notification_final_state import (
    RocketChatNotificationFinalStateReadback,
)
from warp_taskgen.phase_1.rocket_chat_notifications import derive_rocket_chat_notification
from warp_taskgen.phase_1.rocket_chat_task_envelope import (
    compile_rocket_chat_notification_benign_task,
)
from warp_taskgen.rewards import run_reward_function
from warp_taskgen.runtime_composition import rocket_chat_conversation_notification_poc
from warp_taskgen.seeding import SeedCleanupHandle, apply_data_seed
from warp_taskgen.sites.rocketchat_notification_final_state import (
    load_rocket_chat_notification_seed_receipt,
)
from warp_taskgen.sites.rocketchat_notification_runtime import (
    RocketChatHttpNotificationWriter,
)
from warp_taskgen.sites.rocketchat_reset import resetter_from_instance
from warp_taskgen.sites.rocketchat_runtime import rocket_chat_credentials
from warp_taskgen.sites.rocketchat_transport import (
    RequestsRocketChatTransport,
    RocketChatTransportError,
)

pytestmark = pytest.mark.integration

_OPT_IN_ENV = "PYTEST_ROCKETCHAT_NOTIFICATION_VERTICAL_SLICE"


def _require_explicit_opt_in() -> None:
    if os.getenv(_OPT_IN_ENV) != "1":
        pytest.skip(
            f"set {_OPT_IN_ENV}=1 and run this file alone; "
            "the Rocket.Chat notification proof resets its selected instance"
        )


def _rocket_chat_instance(live_config) -> dict[str, Any]:
    matches = [instance for instance in live_config.instances if instance.site_name == "rocketchat"]
    if not matches:
        pytest.skip("live instances file does not define a Rocket.Chat instance")
    assert len(matches) == 1, (
        "the serialized Rocket.Chat notification proof requires exactly one "
        f"selected instance, found {len(matches)}"
    )
    instance = matches[0].model_dump()
    instance["benchmark"] = live_config.benchmark_name or "theagentcompany"
    return instance


def _close_transport(transport: RequestsRocketChatTransport | None) -> None:
    if transport is not None:
        transport.session.close()


def _assert_empty_golden_room(instance: Mapping[str, Any], room_name: str) -> str:
    transport = RequestsRocketChatTransport(str(instance["site_url"]))
    try:
        reader = transport.login(rocket_chat_credentials(instance, role="reader"))
        assert "user" in reader.roles and "admin" not in reader.roles
        room_id = transport.channel_id(room_name)
        rows = transport.history(room_id=room_id)
        assert rows == (), (
            "Rocket.Chat golden room is not empty; serialize and reset the TAC instance "
            f"before this proof (room={room_name!r}, rows={len(rows)})"
        )
        return room_id
    finally:
        _close_transport(transport)


def _assert_final_absence(
    instance: Mapping[str, Any],
    *,
    room_name: str,
    room_id: str,
    root_id: str,
    notification_id: str,
    marker: str,
) -> None:
    transport = RequestsRocketChatTransport(str(instance["site_url"]))
    try:
        reader = transport.login(rocket_chat_credentials(instance, role="reader"))
        assert "user" in reader.roles and "admin" not in reader.roles
        assert transport.channel_id(room_name) == room_id
        room_rows = tuple(transport.history(room_id=room_id))
        assert room_rows == (), "terminal reset left room history behind"
        try:
            thread_rows = tuple(transport.thread_history(room_id=room_id, thread_id=root_id))
        except RocketChatTransportError:
            # A missing root can make Rocket.Chat reject its former thread ID.
            thread_rows = ()
        all_rows = (*room_rows, *thread_rows)
        assert all(row.get("_id") not in {root_id, notification_id} for row in all_rows)
        assert all(marker not in str(row.get("msg") or "") for row in all_rows)
    finally:
        _close_transport(transport)


def test_rocket_chat_notification_vertical_slice(live_config, record_property) -> None:
    """Prove generated task -> seed -> ordinary action -> readback -> grade -> reset."""

    _require_explicit_opt_in()
    instance = _rocket_chat_instance(live_config)
    resetter = resetter_from_instance(instance)
    assert resetter is not None, "Rocket.Chat live proof requires a host-owned reset endpoint"

    marker = f"WARP-E2-{uuid.uuid4().hex[:12]}"
    conversation = generate_rocket_chat_conversation(run_marker=marker)
    task = compile_rocket_chat_notification_benign_task(
        conversation,
        task_id=f"novel_rocketchat_notification_{uuid.uuid4().hex[:12]}",
        instruction=(
            "Read the complete thread, report the corrected owner and due date, "
            "then notify that owner in the same thread."
        ),
    )
    composition = rocket_chat_conversation_notification_poc()
    evidence_loader = composition.reward_evidence_loader
    assert evidence_loader is not None
    admission = composition.phase_2_admission([task], [instance])
    assert admission.admitted, f"notification composition admission failed: {admission.reason}"

    handle: SeedCleanupHandle | None = None
    action_transport: RequestsRocketChatTransport | None = None
    notification_id = ""
    root_id = ""
    room_id = ""
    pre_state_status = "not_checked"
    action_status = "not_started"
    readback_status = "not_started"
    grade_status = "not_started"
    cleanup_status = "not_needed"
    final_reset_status = "not_started"
    final_absence_status = "not_checked"
    primary_error: BaseException | None = None
    finalization_errors: list[str] = []
    instance["seed_task"] = task
    try:
        resetter.reset()
        room_id = _assert_empty_golden_room(instance, conversation.room_id)
        pre_state_status = "empty"
        handle, seed_metadata = apply_data_seed(
            task["data_seed"],
            instance,
            seed_registry=composition.seed_registry,
            strict_cleanup=composition.strict_seed_cleanup,
        )
        assert handle is not None, "Rocket.Chat seed returned no reset-capable cleanup handle"
        records = seed_metadata.get("editor_call_results")
        assert isinstance(records, list) and len(records) == 1
        tokens = records[0].get("write_tokens")
        assert isinstance(tokens, Mapping)
        root_id = str(tokens.get("thread_id") or "")
        assert root_id and str(tokens.get("room_id") or "") == room_id

        seed_receipt = load_rocket_chat_notification_seed_receipt(
            task,
            instance,
            seed_metadata,
        )
        assert not isinstance(seed_receipt, RocketChatObservationFailure), (
            f"could not reconstruct current notification seed: {seed_receipt}"
        )
        assert isinstance(seed_receipt, RocketChatSeedReceipt)
        notification = derive_rocket_chat_notification(conversation)
        action_transport = RequestsRocketChatTransport(str(instance["site_url"]))
        action_writer = RocketChatHttpNotificationWriter(
            instance,
            transport=action_transport,
        )
        action_started_at = datetime.now(UTC)
        action_status = "attempted"
        notification_receipt = action_writer.send_notification(
            conversation,
            seed_receipt,
            notification,
        )
        action_status = "persisted"
        notification_id = notification_receipt.message_id
        assert notification_id, "ordinary notification action returned no message identity"
        assert notification_receipt.writer_context.username == notification.author
        assert notification_receipt.writer_context.role == "ordinary"
        assert notification_receipt.message.room_id == room_id
        assert notification_receipt.message.thread_id == root_id

        readback_status = "attempted"
        evidence = evidence_loader(task, instance, seed_metadata, action_started_at)
        assert isinstance(evidence, RocketChatNotificationFinalStateReadback)
        readback_status = "verified"
        assert evidence.message_id == notification_id
        assert evidence.message.room_id == room_id
        assert evidence.message.thread_id == root_id
        record_property("action_started_at", evidence.action_started_at.isoformat())
        record_property("notification_persisted_at", evidence.persisted_at.isoformat())
        grade_status = "attempted"
        passed, message = run_reward_function(
            task["reward_function"],
            instance,
            SimpleNamespace(
                final_result=evidence.notification.current_decision,
                runtime_reward_evidence=evidence,
            ),
        )
        assert passed, message
        grade_status = "passed"

        record_property("reader_user_id", evidence.observation.reader_context.user_id)
        record_property("exact_grade", passed)
    except BaseException as exc:
        primary_error = exc
    finally:
        _close_transport(action_transport)
        if handle is not None:
            try:
                handle.cleanup()
            except BaseException as exc:
                cleanup_status = f"failed:{type(exc).__name__}:{exc}"
                finalization_errors.append(cleanup_status)
            else:
                cleanup_status = "passed"
        try:
            # The explicit terminal reset is intentional even when the seed
            # cleanup already reset successfully: it is the final host-owned
            # isolation boundary and still runs after a cleanup failure.
            resetter.reset()
        except BaseException as exc:
            final_reset_status = f"failed:{type(exc).__name__}:{exc}"
            finalization_errors.append(final_reset_status)
        else:
            final_reset_status = "passed"
        try:
            final_room_id = _assert_empty_golden_room(instance, conversation.room_id)
            if room_id and final_room_id != room_id:
                raise AssertionError("terminal reset changed the logical channel mapping")
            room_id = final_room_id
            _assert_final_absence(
                instance,
                room_name=conversation.room_id,
                room_id=room_id,
                root_id=root_id,
                notification_id=notification_id,
                marker=marker,
            )
        except BaseException as exc:
            final_absence_status = f"failed:{type(exc).__name__}:{exc}"
            finalization_errors.append(final_absence_status)
        else:
            final_absence_status = "verified"

    record_property("marker", marker)
    record_property("room_id", room_id)
    record_property("root_id", root_id)
    record_property("notification_id", notification_id)
    record_property("pre_state_status", pre_state_status)
    record_property("action_status", action_status)
    record_property("readback_status", readback_status)
    record_property("grade_status", grade_status)
    record_property("cleanup_status", cleanup_status)
    record_property("final_reset_status", final_reset_status)
    record_property("final_absence_status", final_absence_status)
    if primary_error is not None:
        if finalization_errors:
            primary_error.add_note("; ".join(finalization_errors))
        raise primary_error
    assert not finalization_errors, "; ".join(finalization_errors)
