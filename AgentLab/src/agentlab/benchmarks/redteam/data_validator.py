"""Validation helpers for adversarial ``data.js`` hardening outputs."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.eval_harness import (
    get_server_process_diagnostics,
    start_app_server,
    stop_app_server,
)
from agentlab.benchmarks.redteam.app_artifacts import is_normalized_state_payload
from agentlab.benchmarks.redteam.execution import (
    FRESH_BROWSER_WORKER_PROFILE,
    run_trusted_worker,
)

logger = logging.getLogger(__name__)

_TOP_LEVEL_CONST_PATTERN = re.compile(r"(?m)^\s*const\s+([A-Za-z_$][\w$]*)\s*=")
_NUMBER_PATTERN = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_EXTRACTION_ERROR_KEY = "__browser_sim_extraction_error__"


class _DataJsLiteralError(ValueError):
    """Raised when ``data.js`` contains unsupported executable constructs."""


class _DataJsLiteralParser:
    """Parse the declarative ``data.js`` subset without evaluating code."""

    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.constants: dict[str, Any] = {}
        self.ordered_names: list[str] = []

    def parse(self) -> dict[str, Any]:
        while True:
            self._skip_ws_and_comments()
            if self.pos >= self.length:
                break

            self._expect_keyword("const")
            self._skip_ws_and_comments()
            name = self._parse_identifier()
            self._skip_ws_and_comments()
            self._expect_char("=")
            value = self._parse_value()
            self.constants[name] = deepcopy(value)
            self.ordered_names.append(name)
            self._skip_ws_and_comments()
            if self._peek() == ";":
                self.pos += 1

        return {name: self.constants[name] for name in self.ordered_names}

    def _skip_ws_and_comments(self) -> None:
        while self.pos < self.length:
            char = self.source[self.pos]
            if char.isspace():
                self.pos += 1
                continue
            if self.source.startswith("//", self.pos):
                self.pos += 2
                while self.pos < self.length and self.source[self.pos] not in "\r\n":
                    self.pos += 1
                continue
            if self.source.startswith("/*", self.pos):
                end = self.source.find("*/", self.pos + 2)
                if end == -1:
                    raise self._error("Unterminated block comment")
                self.pos = end + 2
                continue
            break

    def _parse_value(self) -> Any:
        self._skip_ws_and_comments()
        char = self._peek()
        if char in {"'", '"'}:
            return self._parse_string()
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char == "`":
            raise self._error("Template literals are not supported in data.js")
        if char == "-" or char.isdigit():
            return self._parse_number()
        if self._consume_keyword("true"):
            return True
        if self._consume_keyword("false"):
            return False
        if self._consume_keyword("null"):
            return None
        if self._is_identifier_start(char):
            name = self._parse_identifier()
            if name not in self.constants:
                raise self._error(
                    f"Unsupported identifier reference '{name}' in data.js"
                )
            return deepcopy(self.constants[name])
        raise self._error(f"Unsupported data.js value starting with {char!r}")

    def _parse_object(self) -> dict[str, Any]:
        self._expect_char("{")
        obj: dict[str, Any] = {}
        self._skip_ws_and_comments()
        if self._peek() == "}":
            self.pos += 1
            return obj

        while True:
            self._skip_ws_and_comments()
            key = self._parse_object_key()
            self._skip_ws_and_comments()
            self._expect_char(":")
            obj[key] = self._parse_value()
            self._skip_ws_and_comments()
            char = self._peek()
            if char == ",":
                self.pos += 1
                self._skip_ws_and_comments()
                if self._peek() == "}":
                    self.pos += 1
                    return obj
                continue
            if char == "}":
                self.pos += 1
                return obj
            raise self._error("Expected ',' or '}' in object literal")

    def _parse_object_key(self) -> str:
        char = self._peek()
        if char in {"'", '"'}:
            value = self._parse_string()
            if not isinstance(value, str):
                raise self._error("Object keys must be strings")
            return value
        if self._is_identifier_start(char):
            return self._parse_identifier()
        raise self._error("Unsupported object key in data.js")

    def _parse_array(self) -> list[Any]:
        self._expect_char("[")
        values: list[Any] = []
        self._skip_ws_and_comments()
        if self._peek() == "]":
            self.pos += 1
            return values

        while True:
            values.append(self._parse_value())
            self._skip_ws_and_comments()
            char = self._peek()
            if char == ",":
                self.pos += 1
                self._skip_ws_and_comments()
                if self._peek() == "]":
                    self.pos += 1
                    return values
                continue
            if char == "]":
                self.pos += 1
                return values
            raise self._error("Expected ',' or ']' in array literal")

    def _parse_string(self) -> str:
        quote = self._peek()
        self.pos += 1
        chars: list[str] = []
        while self.pos < self.length:
            char = self.source[self.pos]
            self.pos += 1
            if char == quote:
                return "".join(chars)
            if char != "\\":
                chars.append(char)
                continue
            if self.pos >= self.length:
                raise self._error("Unterminated escape sequence")
            escape = self.source[self.pos]
            self.pos += 1
            if escape in {"\\", "/", "'", '"'}:
                chars.append(escape)
            elif escape == "b":
                chars.append("\b")
            elif escape == "f":
                chars.append("\f")
            elif escape == "n":
                chars.append("\n")
            elif escape == "r":
                chars.append("\r")
            elif escape == "t":
                chars.append("\t")
            elif escape == "u":
                chars.append(self._parse_unicode_escape(4))
            elif escape == "x":
                chars.append(self._parse_unicode_escape(2))
            elif escape in {"\n", "\r"}:
                if escape == "\r" and self._peek() == "\n":
                    self.pos += 1
            else:
                chars.append(escape)
        raise self._error("Unterminated string literal")

    def _parse_unicode_escape(self, digits: int) -> str:
        end = self.pos + digits
        chunk = self.source[self.pos : end]
        if len(chunk) != digits or any(ch not in "0123456789abcdefABCDEF" for ch in chunk):
            raise self._error("Invalid escape sequence")
        self.pos = end
        return chr(int(chunk, 16))

    def _parse_number(self) -> int | float:
        match = _NUMBER_PATTERN.match(self.source, self.pos)
        if match is None:
            raise self._error("Invalid numeric literal")
        token = match.group(0)
        self.pos = match.end()
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token)

    def _parse_identifier(self) -> str:
        if not self._is_identifier_start(self._peek()):
            raise self._error("Expected identifier")
        start = self.pos
        self.pos += 1
        while self._is_identifier_part(self._peek()):
            self.pos += 1
        return self.source[start:self.pos]

    def _expect_keyword(self, keyword: str) -> None:
        if not self._consume_keyword(keyword):
            raise self._error(f"Expected '{keyword}' declaration")

    def _consume_keyword(self, keyword: str) -> bool:
        end = self.pos + len(keyword)
        if self.source[self.pos:end] != keyword:
            return False
        next_char = self.source[end] if end < self.length else ""
        if self._is_identifier_part(next_char):
            return False
        self.pos = end
        return True

    def _expect_char(self, expected: str) -> None:
        self._skip_ws_and_comments()
        if self._peek() != expected:
            raise self._error(f"Expected '{expected}'")
        self.pos += 1

    def _peek(self) -> str:
        if self.pos >= self.length:
            return ""
        return self.source[self.pos]

    @staticmethod
    def _is_identifier_start(char: str) -> bool:
        return char == "_" or char == "$" or char.isalpha()

    @classmethod
    def _is_identifier_part(cls, char: str) -> bool:
        return bool(char) and (cls._is_identifier_start(char) or char.isdigit())

    def _error(self, message: str) -> _DataJsLiteralError:
        line = self.source.count("\n", 0, self.pos) + 1
        line_start = self.source.rfind("\n", 0, self.pos)
        column = self.pos - line_start
        return _DataJsLiteralError(f"{message} at line {line}, column {column}")


def find_declared_constants(js_source: str) -> list[str]:
    """Return the ordered list of top-level ``const`` names declared in ``js_source``."""
    seen: set[str] = set()
    names: list[str] = []
    for match in _TOP_LEVEL_CONST_PATTERN.finditer(js_source):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def extract_data_js_constants(js_source: str, node_path: str = "node") -> dict[str, Any]:
    """Parse a declarative ``data.js`` source string and return its constants."""
    del node_path  # kept for backwards compatibility with existing call sites
    parser = _DataJsLiteralParser(js_source)
    return parser.parse()


def load_data_js_constants(path: str | Path, node_path: str = "node") -> dict[str, Any]:
    """Load a declarative ``data.js`` file from disk and parse its constants."""
    path = Path(path)
    return extract_data_js_constants(path.read_text(encoding="utf-8"), node_path=node_path)


def has_extraction_error(value: Any) -> bool:
    return isinstance(value, dict) and _EXTRACTION_ERROR_KEY in value


@dataclass
class CheckResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    duration_ms: float
    details: str | None = None
    skipped: bool = False

    @classmethod
    def skipped_result(cls, check_name: str, details: str | None = None) -> "CheckResult":
        return cls(check_name=check_name, passed=False, duration_ms=0.0, details=details, skipped=True)

    def to_dict(self) -> dict[str, Any]:
        if self.skipped:
            payload: dict[str, Any] = {"skipped": True}
            if self.details:
                payload["details"] = self.details
            return payload
        payload = {
            "check_name": self.check_name,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class DataValidationResult:
    """Result of validating a single adversarial ``data.js`` candidate."""

    passed: bool
    checks: dict[str, CheckResult]
    error_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "passed": self.passed,
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
        }
        if self.error_summary:
            payload["error_summary"] = self.error_summary
        return payload


class DataValidator:
    """Validate app-mode adversarial ``data.js`` content before accepting it."""

    def __init__(
        self,
        browser_checks: bool = True,
        node_path: str = "node",
        browser_timeout_ms: int = 10_000,
    ):
        self.browser_checks = browser_checks
        self.node_path = node_path
        self.browser_timeout_ms = browser_timeout_ms
        self._playwright = None
        self._browser = None

    def validate(
        self,
        adversarial_data_js: str,
        benign_data_js_path: Path,
        app_dir: Path,
    ) -> DataValidationResult:
        """Run ordered validation gates against a candidate adversarial ``data.js``."""
        checks: dict[str, CheckResult] = {}

        parse_result = self._check_parse(adversarial_data_js)
        checks["parse"] = parse_result
        if not parse_result.passed:
            self._mark_remaining_checks_skipped(checks, include_browser=self.browser_checks)
            return DataValidationResult(
                passed=False,
                checks=checks,
                error_summary=self._build_error_summary(checks),
            )

        schema_result = self._check_schema(adversarial_data_js, benign_data_js_path)
        checks["schema"] = schema_result
        if not schema_result.passed:
            self._mark_remaining_checks_skipped(checks, include_browser=self.browser_checks)
            return DataValidationResult(
                passed=False,
                checks=checks,
                error_summary=self._build_error_summary(checks),
            )

        if not self.browser_checks:
            checks["app_load"] = CheckResult.skipped_result("app_load", "browser checks disabled")
            checks["state_api"] = CheckResult.skipped_result("state_api", "browser checks disabled")
            return DataValidationResult(passed=True, checks=checks)

        payload = self._run_runtime_checks_with_worker(
            adversarial_data_js=adversarial_data_js,
            app_dir=Path(app_dir),
        )
        checks["app_load"] = self._check_result_from_payload("app_load", payload.get("app_load"))
        checks["state_api"] = self._check_result_from_payload("state_api", payload.get("state_api"))
        passed = bool(payload.get("passed"))
        error_summary = payload.get("error_summary")
        return DataValidationResult(
            passed=passed,
            checks=checks,
            error_summary=str(error_summary) if error_summary else None,
        )

    def cleanup(self):
        """Close any browser resources held by this validator."""
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def _run_runtime_checks_with_worker(
        self,
        *,
        adversarial_data_js: str,
        app_dir: Path,
    ) -> dict[str, Any]:
        inputs_dir = app_dir / ".sandbox_inputs"
        candidate_path = inputs_dir / "validator_candidate_data.js"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(adversarial_data_js, encoding="utf-8")
        try:
            result = run_trusted_worker(
                app_dir,
                "validate-data-runtime",
                timeout=max(60, int(self.browser_timeout_ms / 1000) + 10),
                argv=[
                    "--root",
                    ".",
                    "--candidate-path",
                    ".sandbox_inputs/validator_candidate_data.js",
                    "--timeout-ms",
                    str(self.browser_timeout_ms),
                ],
                block_network=True,
                profile=FRESH_BROWSER_WORKER_PROFILE,
            )
        finally:
            candidate_path.unlink(missing_ok=True)
            if inputs_dir.exists() and not any(inputs_dir.iterdir()):
                inputs_dir.rmdir()

        if result.returncode != 0:
            error_output = result.stderr or result.stdout or "Runtime validator helper failed."
            return {
                "passed": False,
                "app_load": {
                    "check_name": "app_load",
                    "passed": False,
                    "duration_ms": 0.0,
                    "details": f"App validation setup failed: {error_output}",
                    "skipped": False,
                },
                "state_api": {
                    "check_name": "state_api",
                    "passed": False,
                    "duration_ms": 0.0,
                    "details": "app load failed",
                    "skipped": True,
                },
                "error_summary": f"app_load: App validation setup failed: {error_output}",
            }
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            message = f"Runtime validator helper returned invalid JSON: {exc}"
            return {
                "passed": False,
                "app_load": {
                    "check_name": "app_load",
                    "passed": False,
                    "duration_ms": 0.0,
                    "details": f"App validation setup failed: {message}",
                    "skipped": False,
                },
                "state_api": {
                    "check_name": "state_api",
                    "passed": False,
                    "duration_ms": 0.0,
                    "details": "app load failed",
                    "skipped": True,
                },
                "error_summary": f"app_load: App validation setup failed: {message}",
            }

    def _check_result_from_payload(
        self,
        check_name: str,
        payload: Any,
    ) -> CheckResult:
        if not isinstance(payload, dict):
            return CheckResult(
                check_name=check_name,
                passed=False,
                duration_ms=0.0,
                details=f"{check_name} result missing from runtime helper payload",
            )
        details = payload.get("details")
        return CheckResult(
            check_name=str(payload.get("check_name", check_name)),
            passed=bool(payload.get("passed")),
            duration_ms=float(payload.get("duration_ms", 0.0)),
            details=str(details) if details else None,
            skipped=bool(payload.get("skipped", False)),
        )

    def _check_parse(self, data_js_content: str) -> CheckResult:
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            suffix=".js",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_file.write(data_js_content)
            tmp_path = temp_file.name

        try:
            node_binary = shutil.which(self.node_path) or self.node_path
            result = subprocess.run(
                [node_binary, "--check", tmp_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            if result.returncode != 0:
                return CheckResult(
                    check_name="parse",
                    passed=False,
                    duration_ms=duration_ms,
                    details=f"Syntax error: {(result.stderr or result.stdout).strip()}",
                )
            return CheckResult(check_name="parse", passed=True, duration_ms=duration_ms)
        except FileNotFoundError:
            duration_ms = (time.perf_counter() - start) * 1000
            return self._fallback_parse_check(data_js_content, duration_ms)
        finally:
            os.unlink(tmp_path)

    def _check_schema(
        self,
        adversarial_content: str,
        benign_data_js_path: Path,
    ) -> CheckResult:
        start = time.perf_counter()
        errors: list[str] = []
        try:
            benign_constants = load_data_js_constants(benign_data_js_path, node_path=self.node_path)
            adversarial_constants = extract_data_js_constants(adversarial_content, node_path=self.node_path)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            return CheckResult(
                check_name="schema",
                passed=False,
                duration_ms=duration_ms,
                details=f"Failed to parse data.js constants: {exc}",
            )

        benign_names = set(benign_constants)
        adversarial_names = set(adversarial_constants)
        missing = sorted(benign_names - adversarial_names)
        unexpected = sorted(adversarial_names - benign_names)
        if missing:
            errors.append(f"Missing constants: {', '.join(missing)}")
        if unexpected:
            errors.append(f"Unexpected constants: {', '.join(unexpected)}")

        for name in sorted(benign_names & adversarial_names):
            benign_value = benign_constants[name]
            adversarial_value = adversarial_constants[name]
            if isinstance(benign_value, list):
                self._check_array_invariants(
                    path=name,
                    benign_value=benign_value,
                    adversarial_value=adversarial_value,
                    errors=errors,
                )
            else:
                self._check_constant_invariants(
                    path=name,
                    benign_value=benign_value,
                    adversarial_value=adversarial_value,
                    errors=errors,
                )

        duration_ms = (time.perf_counter() - start) * 1000
        if errors:
            return CheckResult(
                check_name="schema",
                passed=False,
                duration_ms=duration_ms,
                details="; ".join(errors[:5]),
            )
        return CheckResult(check_name="schema", passed=True, duration_ms=duration_ms)

    def _check_app_load(self, server_port: int):
        start = time.perf_counter()
        url = f"http://127.0.0.1:{server_port}/"
        page = self._get_browser().new_page()
        errors: list[str] = []

        def _record_console(msg):
            if msg.type == "error":
                errors.append(msg.text)

        def _record_pageerror(exc):
            errors.append(str(exc))

        def _record_request_failed(request):
            if "favicon.ico" in request.url:
                return
            errors.append(f"Request failed: {request.url} ({request.failure})")

        page.on("console", _record_console)
        page.on("pageerror", _record_pageerror)
        page.on("requestfailed", _record_request_failed)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.browser_timeout_ms)
            page.wait_for_timeout(500)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            return page, CheckResult(
                check_name="app_load",
                passed=False,
                duration_ms=duration_ms,
                details=f"App failed to load: {exc}",
            )

        duration_ms = (time.perf_counter() - start) * 1000
        if errors:
            return page, CheckResult(
                check_name="app_load",
                passed=False,
                duration_ms=duration_ms,
                details="; ".join(errors[:3]),
            )

        return page, CheckResult(check_name="app_load", passed=True, duration_ms=duration_ms)

    def _check_state_api(self, server_port: int) -> CheckResult:
        start = time.perf_counter()
        deadline = time.monotonic() + (self.browser_timeout_ms / 1000)
        last_error: str | None = None

        while time.monotonic() < deadline:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{server_port}/api/state", method="GET")
                with urllib.request.urlopen(req, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not is_normalized_state_payload(payload):
                    last_error = "GET /api/state returned unsupported payload shape"
                elif payload == {}:
                    last_error = "GET /api/state returned empty state"
                else:
                    json.dumps(payload)
                    duration_ms = (time.perf_counter() - start) * 1000
                    return CheckResult(
                        check_name="state_api",
                        passed=True,
                        duration_ms=duration_ms,
                    )
            except urllib.error.HTTPError as exc:
                last_error = f"GET /api/state returned HTTP {exc.code}"
            except json.JSONDecodeError as exc:
                last_error = f"GET /api/state returned invalid JSON: {exc}"
                break
            except Exception as exc:
                last_error = f"GET /api/state failed: {exc}"
            time.sleep(0.1)

        duration_ms = (time.perf_counter() - start) * 1000
        return CheckResult(
            check_name="state_api",
            passed=False,
            duration_ms=duration_ms,
            details=last_error or "GET /api/state did not become available",
        )

    def _fallback_parse_check(self, data_js_content: str, duration_ms: float) -> CheckResult:
        stack: list[str] = []
        pairs = {"}": "{", "]": "[", ")": "("}
        in_string: str | None = None
        escaped = False

        for char in data_js_content:
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == in_string:
                    in_string = None
                continue

            if char in {"'", '"'}:
                in_string = char
            elif char in {"{", "[", "("}:
                stack.append(char)
            elif char in pairs:
                if not stack or stack.pop() != pairs[char]:
                    return CheckResult(
                        check_name="parse",
                        passed=False,
                        duration_ms=duration_ms,
                        details="Fallback parse failed: unbalanced delimiters",
                    )

        if in_string or stack:
            return CheckResult(
                check_name="parse",
                passed=False,
                duration_ms=duration_ms,
                details="Fallback parse failed: unterminated string or delimiter",
            )

        if "const" not in data_js_content:
            return CheckResult(
                check_name="parse",
                passed=False,
                duration_ms=duration_ms,
                details="Fallback parse failed: no const declarations found",
            )
        logger.warning("Node.js unavailable; parse validation ran in fallback mode")
        return CheckResult(check_name="parse", passed=True, duration_ms=duration_ms)

    def _check_constant_invariants(
        self,
        path: str,
        benign_value: Any,
        adversarial_value: Any,
        errors: list[str],
    ) -> None:
        if has_extraction_error(benign_value):
            errors.append(f"{path}: benign extraction error: {benign_value[_EXTRACTION_ERROR_KEY]}")
            return
        if has_extraction_error(adversarial_value):
            errors.append(f"{path}: adversarial extraction error: {adversarial_value[_EXTRACTION_ERROR_KEY]}")
            return

        if benign_value != adversarial_value:
            errors.append(f"{path}: shared constant changed; expected exact benign value")

    def _check_array_invariants(
        self,
        path: str,
        benign_value: Any,
        adversarial_value: Any,
        errors: list[str],
    ) -> None:
        if has_extraction_error(benign_value):
            errors.append(f"{path}: benign extraction error: {benign_value[_EXTRACTION_ERROR_KEY]}")
            return
        if has_extraction_error(adversarial_value):
            errors.append(f"{path}: adversarial extraction error: {adversarial_value[_EXTRACTION_ERROR_KEY]}")
            return
        if not self._types_compatible(benign_value, adversarial_value):
            errors.append(
                f"{path}: type mismatch (benign={self._type_name(benign_value)}, "
                f"adversarial={self._type_name(adversarial_value)})"
            )
            return

        if len(adversarial_value) < len(benign_value):
            errors.append(f"{path}: adversarial array removed benign items")
            return

        for index, benign_item in enumerate(benign_value):
            if benign_item != adversarial_value[index]:
                errors.append(f"{path}[{index}]: benign prefix changed or reordered")

        appended_items = adversarial_value[len(benign_value) :]
        if not appended_items:
            return

        if not benign_value:
            errors.append(f"{path}: cannot append adversarial items when benign array is empty")
            return

        schema_source = next((item for item in benign_value if item is not None), benign_value[0])
        benign_ids = {
            item["id"]
            for item in benign_value
            if isinstance(item, dict) and "id" in item
        }

        for index, item in enumerate(appended_items, start=len(benign_value)):
            self._check_appended_item_compatibility(
                path=f"{path}[{index}]",
                benign_schema=schema_source,
                adversarial_value=item,
                benign_ids=benign_ids,
                errors=errors,
            )

    def _check_appended_item_compatibility(
        self,
        path: str,
        benign_schema: Any,
        adversarial_value: Any,
        benign_ids: set[Any],
        errors: list[str],
    ) -> None:
        if has_extraction_error(adversarial_value):
            errors.append(f"{path}: adversarial extraction error: {adversarial_value[_EXTRACTION_ERROR_KEY]}")
            return

        if benign_schema is None:
            return

        if not self._types_compatible(benign_schema, adversarial_value):
            errors.append(
                f"{path}: type mismatch (benign={self._type_name(benign_schema)}, "
                f"adversarial={self._type_name(adversarial_value)})"
            )
            return

        if isinstance(benign_schema, dict):
            benign_keys = set(benign_schema)
            adversarial_keys = set(adversarial_value)
            missing = sorted(benign_keys - adversarial_keys)
            extra = sorted(adversarial_keys - benign_keys)
            if missing:
                errors.append(f"{path}: appended item schema mismatch; missing fields {missing}")
            if extra:
                errors.append(f"{path}: appended item schema mismatch; unexpected fields {extra}")
            if "id" in adversarial_value and adversarial_value["id"] in benign_ids:
                errors.append(f"{path}: appended item id collides with benign item {adversarial_value['id']}")
            for key in sorted(benign_keys & adversarial_keys):
                self._check_nested_schema(
                    path=f"{path}.{key}",
                    benign_schema=benign_schema[key],
                    adversarial_value=adversarial_value[key],
                    errors=errors,
                )
            return

        self._check_nested_schema(
            path=path,
            benign_schema=benign_schema,
            adversarial_value=adversarial_value,
            errors=errors,
        )

    def _check_nested_schema(
        self,
        path: str,
        benign_schema: Any,
        adversarial_value: Any,
        errors: list[str],
    ) -> None:
        if benign_schema is None:
            return

        if has_extraction_error(adversarial_value):
            errors.append(f"{path}: adversarial extraction error: {adversarial_value[_EXTRACTION_ERROR_KEY]}")
            return

        if not self._types_compatible(benign_schema, adversarial_value):
            errors.append(
                f"{path}: type mismatch (benign={self._type_name(benign_schema)}, "
                f"adversarial={self._type_name(adversarial_value)})"
            )
            return

        if isinstance(benign_schema, dict):
            benign_keys = set(benign_schema)
            adversarial_keys = set(adversarial_value)
            missing = sorted(benign_keys - adversarial_keys)
            extra = sorted(adversarial_keys - benign_keys)
            if missing:
                errors.append(f"{path}: missing fields {missing}")
            if extra:
                errors.append(f"{path}: unexpected fields {extra}")
            for key in sorted(benign_keys & adversarial_keys):
                self._check_nested_schema(
                    path=f"{path}.{key}",
                    benign_schema=benign_schema[key],
                    adversarial_value=adversarial_value[key],
                    errors=errors,
                )
            return

        if isinstance(benign_schema, list):
            if not benign_schema:
                if adversarial_value:
                    errors.append(f"{path}: expected empty list")
                return
            nested_schema = next((item for item in benign_schema if item is not None), benign_schema[0])
            for index, item in enumerate(adversarial_value):
                self._check_nested_schema(
                    path=f"{path}[{index}]",
                    benign_schema=nested_schema,
                    adversarial_value=item,
                    errors=errors,
                )

    def _types_compatible(self, benign_value: Any, adversarial_value: Any) -> bool:
        if benign_value is None:
            return True
        if isinstance(benign_value, bool):
            return isinstance(adversarial_value, bool)
        if isinstance(benign_value, (int, float)) and not isinstance(benign_value, bool):
            return isinstance(adversarial_value, (int, float)) and not isinstance(adversarial_value, bool)
        return type(benign_value) is type(adversarial_value)

    def _type_name(self, value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "number"
        if value is None:
            return "null"
        return type(value).__name__

    def _build_error_summary(self, checks: dict[str, CheckResult]) -> str | None:
        for name in ("parse", "schema", "app_load", "state_api"):
            check = checks.get(name)
            if check and not check.skipped and not check.passed:
                return f"{name}: {check.details}" if check.details else name
        return None

    def _mark_remaining_checks_skipped(self, checks: dict[str, CheckResult], include_browser: bool) -> None:
        if "schema" not in checks:
            checks["schema"] = CheckResult.skipped_result("schema", "previous check failed")
        if include_browser:
            if "app_load" not in checks:
                checks["app_load"] = CheckResult.skipped_result("app_load", "previous check failed")
            if "state_api" not in checks:
                checks["state_api"] = CheckResult.skipped_result("state_api", "previous check failed")

    def _attach_server_runtime_details(
        self,
        result: CheckResult,
        server_proc,
    ) -> CheckResult:
        if result.passed or result.skipped:
            return result
        return CheckResult(
            check_name=result.check_name,
            passed=result.passed,
            duration_ms=result.duration_ms,
            details=self._combine_runtime_details(result.details, server_proc),
            skipped=result.skipped,
        )

    def _combine_runtime_details(self, message: str | None, server_proc) -> str:
        parts = [message] if message else []
        if server_proc is not None:
            diagnostics = get_server_process_diagnostics(server_proc)
            if diagnostics.get("stderr_tail"):
                parts.append(f"server stderr: {diagnostics['stderr_tail']}")
            if diagnostics.get("stdout_tail"):
                parts.append(f"server stdout: {diagnostics['stdout_tail']}")
        return "; ".join(part for part in parts if part)

    def _get_browser(self):
        if self._browser is None:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
        return self._browser

    @contextmanager
    def _temporary_data_js(self, adversarial_content: str, app_dir: Path):
        data_js_path = app_dir / "js" / "data.js"
        if not data_js_path.exists() and not data_js_path.is_symlink():
            raise FileNotFoundError(f"Active data.js not found: {data_js_path}")

        original_link_target: str | None = None
        original_content: str | None = None
        if data_js_path.is_symlink():
            original_link_target = os.readlink(data_js_path)
        elif data_js_path.exists():
            original_content = data_js_path.read_text(encoding="utf-8")

        temp_path = app_dir / "js" / "__validation_data.js"
        temp_path.write_text(adversarial_content, encoding="utf-8")

        try:
            if data_js_path.exists() or data_js_path.is_symlink():
                data_js_path.unlink()
            try:
                data_js_path.symlink_to(temp_path.name)
            except OSError:
                data_js_path.write_text(adversarial_content, encoding="utf-8")
            yield temp_path
        finally:
            temp_path.unlink(missing_ok=True)
            if data_js_path.exists() or data_js_path.is_symlink():
                data_js_path.unlink()
            if original_link_target is not None:
                data_js_path.symlink_to(original_link_target)
            elif original_content is not None:
                data_js_path.write_text(original_content, encoding="utf-8")
