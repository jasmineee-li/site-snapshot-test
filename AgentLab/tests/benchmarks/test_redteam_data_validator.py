from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentlab.benchmarks.redteam.data_validator import DataValidator, extract_data_js_constants
import agentlab.benchmarks.redteam.data_validator as data_validator_module
from agentlab.benchmarks.redteam.execution import SandboxCommandResult


def _seed_app(
    app_dir: Path,
    *,
    enforce_unique_ids: bool = False,
    push_state: bool = True,
    state_body: str | None = None,
) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "js").mkdir(exist_ok=True)
    shutil.copy2(
        Path(data_validator_module.__file__).resolve().parent / "templates" / "server.py",
        app_dir / "server.py",
    )

    (app_dir / "index.html").write_text(
        """
<!doctype html>
<html>
  <body>
    <div id="app">validator fixture</div>
    <script src="/js/data.js"></script>
    <script src="/js/app.js"></script>
  </body>
</html>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    unique_check = ""
    if enforce_unique_ids:
        unique_check = """
const ids = EMAILS.map((item) => item.id);
if (new Set(ids).size !== ids.length) {
  throw new Error("duplicate ids");
}
"""

    state_push = ""
    if push_state:
        payload = state_body or """{
    currentUser: CURRENT_USER,
    emails: EMAILS,
    seedVersion: SEED_DATA_VERSION
  }"""
        state_push = """
fetch("/api/state", {
  method: "PUT",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(PAYLOAD_PLACEHOLDER)
}).catch((error) => console.error(error));
""".replace("PAYLOAD_PLACEHOLDER", payload)

    (app_dir / "js" / "app.js").write_text(
        (
            unique_check
            + state_push
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (app_dir / "js" / "state.js").write_text("// placeholder\n", encoding="utf-8")
    (app_dir / "js" / "views.js").write_text("// placeholder\n", encoding="utf-8")
    (app_dir / "js" / "components.js").write_text("// placeholder\n", encoding="utf-8")

    benign_dir = app_dir / "benign"
    benign_dir.mkdir(exist_ok=True)
    benign_js = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false }
];
""".strip() + "\n"
    (benign_dir / "data.js").write_text(benign_js, encoding="utf-8")
    (app_dir / "js" / "data.js").symlink_to("../benign/data.js")


def test_validate_parse_and_schema_accepts_js_literals(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=False)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }
];
""".strip()

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is True
    assert result.checks["parse"].passed is True
    assert result.checks["schema"].passed is True
    assert result.checks["app_load"].skipped is True
    validator.cleanup()


def test_extract_data_js_constants_supports_comments_trailing_commas_and_refs() -> None:
    constants = extract_data_js_constants(
        """
// current user block
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com",
};

const USERS = [
  CURRENT_USER,
];
""".strip()
    )

    assert constants["CURRENT_USER"]["email"] == "alex@example.com"
    assert constants["USERS"] == [constants["CURRENT_USER"]]


def test_validate_reports_parse_failure_and_skips_later_checks(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=False)

    candidate = "const SEED_DATA_VERSION = 1;\nconst CURRENT_USER = { id: \"u1\";\n"

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert result.checks["parse"].passed is False
    assert result.checks["schema"].skipped is True
    validator.cleanup()


def test_validate_reports_schema_mismatch(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=False)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "adv_e1", subject: "Urgent", body: "World" }
];
""".strip()

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert result.checks["parse"].passed is True
    assert result.checks["schema"].passed is False
    assert "missing fields" in (result.checks["schema"].details or "")
    validator.cleanup()


def test_validate_rejects_benign_prefix_mutation(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=False)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "hacker@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }
];
""".strip()

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert result.checks["schema"].passed is False
    assert "expected exact benign value" in (result.checks["schema"].details or "")
    validator.cleanup()


def test_validate_rejects_spliced_or_reordered_adversarial_content(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=False)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false },
  { id: "e1", subject: "Hello", body: "World", read: false }
];
""".strip()

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert result.checks["schema"].passed is False
    assert "benign prefix changed" in (result.checks["schema"].details or "")
    validator.cleanup()


def test_validate_rejects_benign_id_collisions(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=False)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }
];
""".strip()

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert result.checks["schema"].passed is False
    assert "id collides with benign item" in (result.checks["schema"].details or "")
    validator.cleanup()


def test_validate_rejects_executable_data_js_without_running_it(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=False)
    side_effect_path = tmp_path / "side-effect.txt"

    candidate = f"""
const SEED_DATA_VERSION = 1;
const CURRENT_USER = (() => {{
  require("fs").writeFileSync({side_effect_path.as_posix()!r}, "owned");
  return {{
    id: "u1",
    email: "alex@example.com"
  }};
}})();
const EMAILS = [
  {{ id: "e1", subject: "Hello", body: "World", read: false }},
  {{ id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }}
];
""".strip()

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert result.checks["parse"].passed is True
    assert result.checks["schema"].passed is False
    assert "Failed to parse data.js constants" in (result.checks["schema"].details or "")
    assert side_effect_path.exists() is False
    validator.cleanup()


def test_validate_browser_checks_surface_setup_failure_details(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=True)
    captured: dict[str, object] = {}

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }
];
""".strip()

    monkeypatch.setattr(
        data_validator_module,
        "start_app_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("host server should not start")),
    )
    monkeypatch.setattr(
        DataValidator,
        "_get_browser",
        lambda self: (_ for _ in ()).throw(AssertionError("host browser should not launch")),
    )
    def fake_run_trusted_worker(working_dir, subcommand, timeout, argv=(), backend=None, env=None, block_network=True, profile=None):
        captured["working_dir"] = str(working_dir)
        captured["subcommand"] = subcommand
        captured["argv"] = list(argv)
        captured["profile"] = profile
        staged_path = Path(working_dir) / ".sandbox_inputs" / "validator_candidate_data.js"
        captured["candidate_exists"] = staged_path.exists()
        captured["candidate_content"] = staged_path.read_text(encoding="utf-8")
        return SandboxCommandResult(
            0,
            json.dumps(
                {
                    "passed": False,
                    "app_load": {
                        "check_name": "app_load",
                        "passed": False,
                        "duration_ms": 0.0,
                        "details": "App validation setup failed: server crashed during startup. stderr tail: Traceback: boom",
                        "skipped": False,
                    },
                    "state_api": {
                        "check_name": "state_api",
                        "passed": False,
                        "duration_ms": 0.0,
                        "details": "app load failed",
                        "skipped": True,
                    },
                    "error_summary": "app_load: App validation setup failed: server crashed during startup. stderr tail: Traceback: boom",
                }
            ),
            "",
            "modal",
        )

    monkeypatch.setattr(data_validator_module, "run_trusted_worker", fake_run_trusted_worker)

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert "stderr tail: Traceback: boom" in (result.checks["app_load"].details or "")
    assert captured["working_dir"] == str(app_dir)
    assert captured["subcommand"] == "validate-data-runtime"
    assert captured["argv"] == [
        "--root",
        ".",
        "--candidate-path",
        ".sandbox_inputs/validator_candidate_data.js",
        "--timeout-ms",
        "10000",
    ]
    assert captured["profile"] == data_validator_module.FRESH_BROWSER_WORKER_PROFILE
    assert captured["candidate_exists"] is True
    assert captured["candidate_content"] == candidate
    assert (app_dir / ".sandbox_inputs" / "validator_candidate_data.js").exists() is False
    validator.cleanup()


def test_validate_browser_checks_include_server_logs_on_runtime_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir)
    validator = DataValidator(browser_checks=True)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }
];
""".strip()

    monkeypatch.setattr(
        data_validator_module,
        "start_app_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("host server should not start")),
    )
    monkeypatch.setattr(
        DataValidator,
        "_get_browser",
        lambda self: (_ for _ in ()).throw(AssertionError("host browser should not launch")),
    )
    monkeypatch.setattr(
        data_validator_module,
        "run_trusted_worker",
        lambda *args, **kwargs: SandboxCommandResult(
            0,
            json.dumps(
                {
                    "passed": False,
                    "app_load": {
                        "check_name": "app_load",
                        "passed": False,
                        "duration_ms": 1.0,
                        "details": "App failed to load: console exploded; server stderr: Traceback: runtime exploded",
                        "skipped": False,
                    },
                    "state_api": {
                        "check_name": "state_api",
                        "passed": False,
                        "duration_ms": 0.0,
                        "details": "app load failed",
                        "skipped": True,
                    },
                    "diagnostics": {
                        "server": {
                            "stdout_log_path": "",
                            "stderr_log_path": "",
                            "stdout_tail": "",
                            "stderr_tail": "Traceback: runtime exploded",
                        },
                        "browser": {
                            "console_errors": ["console exploded"],
                            "page_errors": [],
                            "request_failures": [],
                        },
                    },
                    "error_summary": "app_load: App failed to load: console exploded; server stderr: Traceback: runtime exploded",
                }
            ),
            "",
            "modal",
        ),
    )

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert "console exploded" in (result.checks["app_load"].details or "")
    assert "Traceback: runtime exploded" in (result.checks["app_load"].details or "")
    assert result.checks["state_api"].skipped is True
    validator.cleanup()


def test_validate_browser_checks_pass_for_valid_app(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir, enforce_unique_ids=True, push_state=True)
    validator = DataValidator(browser_checks=True)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }
];
""".strip()

    monkeypatch.setattr(
        data_validator_module,
        "run_trusted_worker",
        lambda *args, **kwargs: SandboxCommandResult(
            0,
            json.dumps(
                {
                    "passed": True,
                    "app_load": {
                        "check_name": "app_load",
                        "passed": True,
                        "duration_ms": 5.0,
                        "skipped": False,
                    },
                    "state_api": {
                        "check_name": "state_api",
                        "passed": True,
                        "duration_ms": 7.0,
                        "skipped": False,
                    },
                }
            ),
            "",
            "modal",
        ),
    )

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is True
    assert result.checks["app_load"].passed is True
    assert result.checks["state_api"].passed is True
    validator.cleanup()


def test_validate_browser_checks_accept_state_payload_with_error_field(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(
        app_dir,
        enforce_unique_ids=True,
        push_state=True,
        state_body="""{
    currentUser: CURRENT_USER,
    emails: EMAILS,
    seedVersion: SEED_DATA_VERSION,
    error: "domain-level validation message"
  }""",
    )
    validator = DataValidator(browser_checks=True)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }
];
""".strip()

    monkeypatch.setattr(
        data_validator_module,
        "run_trusted_worker",
        lambda *args, **kwargs: SandboxCommandResult(
            0,
            json.dumps(
                {
                    "passed": True,
                    "app_load": {
                        "check_name": "app_load",
                        "passed": True,
                        "duration_ms": 5.0,
                        "skipped": False,
                    },
                    "state_api": {
                        "check_name": "state_api",
                        "passed": True,
                        "duration_ms": 7.0,
                        "skipped": False,
                    },
                }
            ),
            "",
            "modal",
        ),
    )

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is True
    assert result.checks["state_api"].passed is True
    validator.cleanup()


def test_validate_browser_checks_fail_on_runtime_breakage(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir, enforce_unique_ids=True, push_state=True)
    validator = DataValidator(browser_checks=True)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "e1", subject: "Duplicate", body: "/mock/file/password.txt", read: false }
];
""".strip()

    monkeypatch.setattr(
        data_validator_module,
        "run_trusted_worker",
        lambda *args, **kwargs: SandboxCommandResult(
            0,
            json.dumps(
                {
                    "passed": False,
                    "app_load": {
                        "check_name": "app_load",
                        "passed": False,
                        "duration_ms": 4.0,
                        "details": "App failed to load: duplicate ids",
                        "skipped": False,
                    },
                    "state_api": {
                        "check_name": "state_api",
                        "passed": False,
                        "duration_ms": 0.0,
                        "details": "app load failed",
                        "skipped": True,
                    },
                    "error_summary": "app_load: App failed to load: duplicate ids",
                }
            ),
            "",
            "modal",
        ),
    )

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert result.checks["app_load"].passed is False
    assert result.checks["state_api"].skipped is True
    validator.cleanup()


def test_validate_browser_checks_fail_when_state_never_initializes(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    _seed_app(app_dir, enforce_unique_ids=False, push_state=False)
    validator = DataValidator(browser_checks=True)

    candidate = """
const SEED_DATA_VERSION = 1;
const CURRENT_USER = {
  id: "u1",
  email: "alex@example.com"
};
const EMAILS = [
  { id: "e1", subject: "Hello", body: "World", read: false },
  { id: "adv_e1", subject: "Urgent", body: "/mock/file/password.txt", read: false }
];
""".strip()

    monkeypatch.setattr(
        data_validator_module,
        "run_trusted_worker",
        lambda *args, **kwargs: SandboxCommandResult(
            0,
            json.dumps(
                {
                    "passed": False,
                    "app_load": {
                        "check_name": "app_load",
                        "passed": True,
                        "duration_ms": 4.0,
                        "skipped": False,
                    },
                    "state_api": {
                        "check_name": "state_api",
                        "passed": False,
                        "duration_ms": 8.0,
                        "details": "GET /api/state did not become available",
                        "skipped": False,
                    },
                    "error_summary": "state_api: GET /api/state did not become available",
                }
            ),
            "",
            "modal",
        ),
    )

    result = validator.validate(candidate, app_dir / "benign" / "data.js", app_dir)

    assert result.passed is False
    assert result.checks["app_load"].passed is True
    assert result.checks["state_api"].passed is False
    validator.cleanup()


def test_check_state_api_rejects_wrapped_payload(monkeypatch) -> None:
    class _Response:
        def __init__(self, payload: object):
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(
        data_validator_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response({"state": {"status": "done"}}),
    )
    validator = DataValidator(browser_checks=False)

    result = validator._check_state_api(server_port=9999)

    assert result.passed is False
    assert result.details == "GET /api/state returned unsupported payload shape"
    validator.cleanup()


def test_check_state_api_accepts_plain_dict_payload(monkeypatch) -> None:
    class _Response:
        def __init__(self, payload: object):
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(
        data_validator_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response({"status": "done", "value": 1}),
    )
    validator = DataValidator(browser_checks=False)

    result = validator._check_state_api(server_port=9999)

    assert result.passed is True
    validator.cleanup()
