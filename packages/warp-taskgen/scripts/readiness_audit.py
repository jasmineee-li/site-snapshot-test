#!/usr/bin/env python3
"""Report repository hygiene signals that affect autonomous agents.

The audit intentionally uses ``git ls-files``. Plain recursive discovery sees
local worktrees, caches, logs, benchmark clones, and scratch artifacts that are
not part of the repository contract.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CODE_SUFFIXES = {".py", ".ts", ".js"}
REVIEW_LOC = 550
SPLIT_LOC = 1200

GENERATED_PREFIXES = (
    "logs/",
    "pipeline_outputs/",
    "tmp/",
)
GENERATED_NAMES = {
    "instances.scale.json",
    "instances.scale.json.fragment",
    "instances.smoke.json.fragment",
    "compose.scale.yml",
    "compose.smoke.yml",
    "scripts/docker-compose.scale.yml",
    "scripts/docker-compose.smoke.yml",
    "scripts/proxy_ports.conf",
}

DEFERRED_TRACKED_GENERATED: set[str] = set()

# True large-file exceptions. Deferred modularization work should not be added
# here; it should remain visible in the audit until it is split.
LARGE_FILE_EXEMPTION_PREFIXES: dict[str, str] = {
    "AgentLab/": "read-only upstream reference material; runtime imports are forbidden",
}
LARGE_FILE_EXEMPTIONS: dict[str, str] = {}

SECRET_VALUE_CHARS = r"A-Za-z0-9._~+/=-"
TOKEN_FIELD_RE = re.compile(
    rf'"(?P<key>[^"]*(?:token|api[_-]?key|secret|password|authorization)[^"]*)"\s*:\s*"(?P<value>[{SECRET_VALUE_CHARS}]{{24,}})"',
    re.IGNORECASE,
)
SECRET_ASSIGNMENT_RE = re.compile(
    rf"(?i)(?<![?&])\b(?P<key>[\w.-]*(?:token|secret|api[_-]?key|authorization|password)[\w.-]*)"
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
class LargeFileExemption:
    path: str
    loc: int
    reason: str


@dataclass(frozen=True)
class TokenFinding:
    path: str
    line: int
    kind: str


@dataclass(frozen=True)
class LegacyImportFinding:
    path: str
    line: int
    module: str


@dataclass(frozen=True)
class ActiveFacadeImportFinding:
    path: str
    line: int
    module: str


@dataclass(frozen=True)
class Audit:
    tracked_files: int
    code_files: int
    files_over_550_loc: list[LargeFile]
    files_over_1200_loc: list[LargeFile]
    large_file_exemptions: list[LargeFileExemption]
    tracked_generated: list[str]
    deferred_tracked_generated: list[str]
    token_findings: list[TokenFinding]
    legacy_phase_imports: list[LegacyImportFinding]
    active_facade_imports: list[ActiveFacadeImportFinding]


LEGACY_PHASE_IMPORT_MODULES = frozenset(
    {
        "worldsim.phases.phase_2_injections",
        "worldsim.phases.phase_2_output",
        "worldsim.phases.phase_2_target_resolver",
        "worldsim.phases.phase_2c_artifacts",
        "worldsim.phases.phase_2c_config",
        "worldsim.phases.phase_4_adversarial",
    }
)
# `worldsim.phases.phase_2_injections_api` is intentionally omitted: on
# `feat/worldsim-v5` it is the canonical Shape-C streaming L3 implementation,
# not a temporary compat wrapper. The PR #11 rename to
# `worldsim.phase_2.runner_api` was deferred to a later migration cycle, so
# this cutover narrows scope to the six wrappers that are pure shims.

LEGACY_PHASE_IMPORT_ALLOWED_PREFIXES = (
    "docs/",
    "tests/",
)

ACTIVE_COMPAT_FACADE_MODULES = frozenset(
    {
        "worldsim.main",
        "worldsim.phases.phase_1_generate_new_tasks_validation",
        "worldsim.phases.phase_2_exposure_contract",
        "worldsim.phases.phase_2_feasibility",
        "worldsim.phases.phase_2_text_fill",
    }
)
ACTIVE_COMPAT_FACADE_ALLOWED_PREFIXES = (
    "docs/",
    "tests/",
)


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


def _large_file_exemption_reason(path: str) -> str | None:
    if path in LARGE_FILE_EXEMPTIONS:
        return LARGE_FILE_EXEMPTIONS[path]
    for prefix, reason in LARGE_FILE_EXEMPTION_PREFIXES.items():
        if path.startswith(prefix):
            return reason
    return None


def _token_findings(paths: list[str]) -> list[TokenFinding]:
    findings: list[TokenFinding] = []
    for path in paths:
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


def _legacy_phase_import_findings(paths: list[str]) -> list[LegacyImportFinding]:
    return [
        LegacyImportFinding(finding.path, finding.line, finding.module)
        for finding in _module_import_findings(
            paths,
            modules=LEGACY_PHASE_IMPORT_MODULES,
            allowed_prefixes=LEGACY_PHASE_IMPORT_ALLOWED_PREFIXES,
        )
    ]


def _active_facade_import_findings(paths: list[str]) -> list[ActiveFacadeImportFinding]:
    return [
        ActiveFacadeImportFinding(finding.path, finding.line, finding.module)
        for finding in _module_import_findings(
            paths,
            modules=ACTIVE_COMPAT_FACADE_MODULES,
            allowed_prefixes=ACTIVE_COMPAT_FACADE_ALLOWED_PREFIXES,
        )
    ]


def _module_import_findings(
    paths: list[str],
    *,
    modules: frozenset[str],
    allowed_prefixes: tuple[str, ...],
) -> list[LegacyImportFinding]:
    findings: list[LegacyImportFinding] = []
    for path in paths:
        if not path.endswith(".py") or path.startswith(allowed_prefixes):
            continue
        try:
            source = Path(path).read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=path)
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_tracked_module(alias.name, modules):
                        findings.append(LegacyImportFinding(path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                if level > 0:
                    base = _resolve_relative_anchor(path, level)
                    if base is None:
                        continue
                    if node.module:
                        resolved = f"{base}.{node.module}"
                        if _is_tracked_module(resolved, modules):
                            findings.append(LegacyImportFinding(path, node.lineno, resolved))
                            continue
                        for alias in node.names:
                            child = f"{resolved}.{alias.name}"
                            if _is_tracked_module(child, modules):
                                findings.append(LegacyImportFinding(path, node.lineno, child))
                    else:
                        for alias in node.names:
                            child = f"{base}.{alias.name}"
                            if _is_tracked_module(child, modules):
                                findings.append(LegacyImportFinding(path, node.lineno, child))
                    continue
                module = node.module or ""
                if _is_tracked_module(module, modules):
                    findings.append(LegacyImportFinding(path, node.lineno, module))
                elif module in _parent_modules(modules):
                    for alias in node.names:
                        imported = f"{module}.{alias.name}"
                        if _is_tracked_module(imported, modules):
                            findings.append(LegacyImportFinding(path, node.lineno, imported))
    return findings


def _resolve_relative_anchor(path: str, level: int) -> str | None:
    """Return the absolute dotted package a `level`-level relative import resolves to.

    For module `worldsim/phases/foo.py`, level=1 resolves to `worldsim.phases`,
    level=2 resolves to `worldsim`. For `worldsim/phases/__init__.py` the answer
    is the same. Returns None if `level` walks above the package root.
    """
    parts = path.split("/")
    anchor_parts = parts[:-1]
    drop = level - 1
    if drop > len(anchor_parts):
        return None
    base_parts = anchor_parts[: len(anchor_parts) - drop] if drop else anchor_parts
    if not base_parts:
        return None
    return ".".join(base_parts)


def _is_legacy_phase_module(module: str) -> bool:
    return _is_tracked_module(module, LEGACY_PHASE_IMPORT_MODULES)


def _is_tracked_module(module: str, modules: frozenset[str]) -> bool:
    return any(
        module == tracked_module or module.startswith(f"{tracked_module}.")
        for tracked_module in modules
    )


def _parent_modules(modules: frozenset[str]) -> frozenset[str]:
    return frozenset(module.rsplit(".", 1)[0] for module in modules if "." in module)


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
    if re.match(r"(?i)^(?:gh[pousr]|github_pat|sk|xox[baprs])[_-]", stripped):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", stripped):
        return False
    if "/" in stripped and (
        stripped.startswith(("/", "docs/", "logs/", "tests/", "worldsim/"))
        or stripped.endswith((".json", ".md", ".py", ".txt", ".yaml", ".yml"))
    ):
        return False
    if re.fullmatch(r"_+[A-Za-z0-9_]+", stripped):
        return False
    if "_" in stripped and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
        return False
    if re.fullmatch(r"_?[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", stripped):
        return False
    return any(char.isalnum() for char in stripped)


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
    exempt_large = [
        LargeFileExemption(item.path, item.loc, reason)
        for item in large
        if item.loc > REVIEW_LOC and (reason := _large_file_exemption_reason(item.path)) is not None
    ]
    non_exempt_large = [item for item in large if _large_file_exemption_reason(item.path) is None]
    return Audit(
        tracked_files=len(files),
        code_files=len(code),
        files_over_550_loc=[item for item in non_exempt_large if item.loc > REVIEW_LOC],
        files_over_1200_loc=[item for item in non_exempt_large if item.loc > SPLIT_LOC],
        large_file_exemptions=exempt_large,
        tracked_generated=blocking_generated,
        deferred_tracked_generated=deferred,
        token_findings=_token_findings(files),
        legacy_phase_imports=_legacy_phase_import_findings(files),
        active_facade_imports=_active_facade_import_findings(files),
    )


def _print_text(audit: Audit) -> None:
    print(f"tracked_files={audit.tracked_files}")
    print(f"code_files={audit.code_files}")
    print(f"files_over_550_loc={len(audit.files_over_550_loc)}")
    print(f"files_over_1200_loc={len(audit.files_over_1200_loc)}")
    print(f"large_file_exemptions={len(audit.large_file_exemptions)}")
    print(f"tracked_generated={len(audit.tracked_generated)}")
    print(f"deferred_tracked_generated={len(audit.deferred_tracked_generated)}")
    print(f"token_findings={len(audit.token_findings)}")
    print(f"legacy_phase_imports={len(audit.legacy_phase_imports)}")
    print(f"active_facade_imports={len(audit.active_facade_imports)}")
    if audit.files_over_1200_loc:
        print("largest_files:")
        for item in audit.files_over_1200_loc[:10]:
            print(f"  {item.loc:5d} {item.path}")
    if audit.large_file_exemptions:
        print("large_file_exemptions:")
        for item in audit.large_file_exemptions:
            print(f"  {item.loc:5d} {item.path} # {item.reason}")
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
    if audit.legacy_phase_imports:
        print("legacy_phase_imports:")
        for finding in audit.legacy_phase_imports[:25]:
            print(f"  {finding.path}:{finding.line} {finding.module}")
        if len(audit.legacy_phase_imports) > 25:
            print(f"  ... {len(audit.legacy_phase_imports) - 25} more")
    if audit.active_facade_imports:
        print("active_facade_imports:")
        for finding in audit.active_facade_imports[:25]:
            print(f"  {finding.path}:{finding.line} {finding.module}")
        if len(audit.active_facade_imports) > 25:
            print(f"  ... {len(audit.active_facade_imports) - 25} more")


def _json_default(value: Any) -> Any:
    if isinstance(
        value,
        LargeFile | LargeFileExemption | TokenFinding | LegacyImportFinding | ActiveFacadeImportFinding,
    ):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--fail-on",
        action="append",
        choices=("legacy-imports", "tracked-generated", "tokens"),
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
    if "legacy-imports" in args.fail_on and audit.legacy_phase_imports:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
