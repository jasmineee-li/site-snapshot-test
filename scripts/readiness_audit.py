#!/usr/bin/env python3
"""Report repository hygiene signals that affect autonomous agents.

The audit intentionally uses ``git ls-files``. Plain recursive discovery sees
local worktrees, caches, logs, benchmark clones, and scratch artifacts that are
not part of the repository contract.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CODE_SUFFIXES = {".py", ".ts", ".js"}
REVIEW_LOC = 500
SPLIT_LOC = 1200

GENERATED_PREFIXES = (
    "logs/",
    "pipeline_outputs/",
    "tmp/",
)
GENERATED_NAMES = {
    "instances.scale.json",
    "instances.scale.json.fragment",
    "compose.scale.yml",
    "compose.smoke.yml",
    "scripts/docker-compose.scale.yml",
    "scripts/proxy_ports.conf",
}

DEFERRED_TRACKED_GENERATED: set[str] = set()
TOKEN_SCAN_EXCLUDED_PREFIXES = ("tests/fixtures/",)

SECRET_VALUE_CHARS = r"A-Za-z0-9._~+/=-"
TOKEN_FIELD_RE = re.compile(
    rf'"(?P<key>[^"]*(?:token|api[_-]?key|secret|password|authorization)[^"]*)"\s*:\s*"(?P<value>[{SECRET_VALUE_CHARS}]{{24,}})"',
    re.IGNORECASE,
)
SECRET_ASSIGNMENT_RE = re.compile(
    rf"(?i)\b(?P<key>[\w.-]*(?:token|secret|api[_-]?key|authorization|password)[\w.-]*)"
    rf"\s*[:=]\s*['\"]?(?P<value>[{SECRET_VALUE_CHARS}]{{24,}})"
)
BEARER_VALUE_RE = re.compile(rf"(?i)\bBearer\s+([{SECRET_VALUE_CHARS}]{{24,}})")
QUOTED_HIGH_ENTROPY_RE = re.compile(rf"[`'\"]([{SECRET_VALUE_CHARS}]{{40,}})[`'\"]")
SECRET_CONTEXT_RE = re.compile(
    r"token|secret|api[_-]?key|authorization|bearer|password", re.IGNORECASE
)
STRICT_SECRET_CONTEXT_RE = re.compile(
    r"(?:current|proxy|bearer|api[_-]?key|secret).{0,80}token|"
    r"token.{0,80}(?:current|proxy|bearer|api[_-]?key|secret)|"
    r"authorization|bearer",
    re.IGNORECASE,
)
FIXTURE_CREDENTIAL_VALUES = {
    "admin123",
    "admin1234",
    "byteblaze",
    "hello1234",
    "MarvelsGrantMan136",
    "postmill",
    "test1234",
}


@dataclass(frozen=True)
class LargeFile:
    path: str
    loc: int


@dataclass(frozen=True)
class TokenFinding:
    path: str
    line: int
    kind: str


@dataclass(frozen=True)
class Audit:
    tracked_files: int
    code_files: int
    files_over_500_loc: list[LargeFile]
    files_over_1200_loc: list[LargeFile]
    tracked_generated: list[str]
    deferred_tracked_generated: list[str]
    token_findings: list[TokenFinding]


def _git_ls_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def _line_count(path: str) -> int:
    try:
        return len(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError:
        return 0


def _is_generated(path: str) -> bool:
    return path in GENERATED_NAMES or any(path.startswith(prefix) for prefix in GENERATED_PREFIXES)


def _token_findings(paths: list[str]) -> list[TokenFinding]:
    findings: list[TokenFinding] = []
    for path in paths:
        if any(path.startswith(prefix) for prefix in TOKEN_SCAN_EXCLUDED_PREFIXES):
            continue
        if not path.endswith((".json", ".yaml", ".yml", ".toml", ".md", ".py", ".sh")):
            continue
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines, start=1):
            token_match = TOKEN_FIELD_RE.search(line)
            if (
                token_match
                and _looks_like_secret_key(token_match.group("key"))
                and _looks_like_secret_value(token_match.group("value"))
            ):
                findings.append(TokenFinding(path, index, "token_field"))
                continue
            bearer_match = BEARER_VALUE_RE.search(line)
            assignment_match = SECRET_ASSIGNMENT_RE.search(line)
            if (bearer_match and _looks_like_secret_value(bearer_match.group(1))) or (
                assignment_match
                and _looks_like_secret_key(assignment_match.group("key"))
                and _looks_like_secret_value(assignment_match.group("value"))
            ):
                findings.append(TokenFinding(path, index, "secret_value"))
                continue
            if not STRICT_SECRET_CONTEXT_RE.search(line):
                continue
            quoted_match = QUOTED_HIGH_ENTROPY_RE.search(line)
            if quoted_match and _looks_like_secret_value(quoted_match.group(1)):
                findings.append(TokenFinding(path, index, "high_entropy"))
    return findings


def _looks_like_secret_value(value: str) -> bool:
    stripped = value.strip()
    if stripped in FIXTURE_CREDENTIAL_VALUES:
        return False
    if stripped.startswith("${") and stripped.endswith("}"):
        return False
    if stripped.startswith("$") and re.fullmatch(r"\$[A-Z][A-Z0-9_]*", stripped):
        return False
    if stripped.startswith("<") and stripped.endswith(">"):
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", stripped):
        return False
    if "/" in stripped and (
        stripped.startswith(("/", "docs/", "logs/", "tests/", "worldsim/"))
        or stripped.endswith((".json", ".md", ".py", ".txt", ".yaml", ".yml"))
    ):
        return False
    if re.fullmatch(r"_?[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", stripped):
        return False
    return any(char.isalpha() for char in stripped) and any(char.isdigit() for char in stripped)


def _looks_like_secret_key(key: str) -> bool:
    normalized = re.sub(r"[-.]", "_", key.strip().lower())
    if normalized.endswith(
        (
            "_endpoint",
            "_endpoints",
            "_env",
            "_file",
            "_files",
            "_generator",
            "_path",
            "_paths",
            "_source",
            "_sources",
            "_url",
            "_urls",
        )
    ):
        return False
    if normalized in {
        "required_token",
        "required_tokens",
        "token_endpoint",
        "token_source",
        "token_generator",
        "token_env",
        "token_file",
        "validation_endpoint",
    }:
        return False
    if normalized in {
        "access_token",
        "api_key",
        "authorization",
        "bearer_token",
        "client_secret",
        "id_token",
        "password",
        "refresh_token",
        "secret",
        "token",
    }:
        return True
    return normalized.endswith(("_api_key", "_password", "_secret", "_token"))


def build_audit() -> Audit:
    files = _git_ls_files()
    code = [path for path in files if Path(path).suffix in CODE_SUFFIXES]
    large = sorted(
        (LargeFile(path, _line_count(path)) for path in code),
        key=lambda item: item.loc,
        reverse=True,
    )
    generated = sorted(path for path in files if _is_generated(path))
    deferred = [path for path in generated if path in DEFERRED_TRACKED_GENERATED]
    blocking_generated = [path for path in generated if path not in DEFERRED_TRACKED_GENERATED]
    return Audit(
        tracked_files=len(files),
        code_files=len(code),
        files_over_500_loc=[item for item in large if item.loc > REVIEW_LOC],
        files_over_1200_loc=[item for item in large if item.loc > SPLIT_LOC],
        tracked_generated=blocking_generated,
        deferred_tracked_generated=deferred,
        token_findings=_token_findings(files),
    )


def _print_text(audit: Audit) -> None:
    print(f"tracked_files={audit.tracked_files}")
    print(f"code_files={audit.code_files}")
    print(f"files_over_500_loc={len(audit.files_over_500_loc)}")
    print(f"files_over_1200_loc={len(audit.files_over_1200_loc)}")
    print(f"tracked_generated={len(audit.tracked_generated)}")
    print(f"deferred_tracked_generated={len(audit.deferred_tracked_generated)}")
    print(f"token_findings={len(audit.token_findings)}")
    if audit.files_over_1200_loc:
        print("largest_files:")
        for item in audit.files_over_1200_loc[:10]:
            print(f"  {item.loc:5d} {item.path}")
    if audit.tracked_generated:
        print("tracked_generated_paths:")
        for path in audit.tracked_generated:
            print(f"  {path}")
    if audit.deferred_tracked_generated:
        print("deferred_tracked_generated_paths:")
        for path in audit.deferred_tracked_generated:
            print(f"  {path}")
    if audit.token_findings:
        print("token_findings:")
        for finding in audit.token_findings[:25]:
            print(f"  {finding.path}:{finding.line} {finding.kind}")
        if len(audit.token_findings) > 25:
            print(f"  ... {len(audit.token_findings) - 25} more")


def _json_default(value: Any) -> Any:
    if isinstance(value, LargeFile | TokenFinding):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--fail-on",
        action="append",
        choices=("tracked-generated", "tokens"),
        default=[],
        help="fail when the selected finding class is present",
    )
    args = parser.parse_args(argv)

    audit = build_audit()
    if args.json:
        print(json.dumps(asdict(audit), indent=2, sort_keys=True, default=_json_default))
    else:
        _print_text(audit)

    failed = False
    if "tracked-generated" in args.fail_on and audit.tracked_generated:
        failed = True
    if "tokens" in args.fail_on and audit.token_findings:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
