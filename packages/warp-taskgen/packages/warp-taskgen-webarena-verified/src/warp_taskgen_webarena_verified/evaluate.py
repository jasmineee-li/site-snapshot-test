from __future__ import annotations

import json
import sys
from typing import Any

from webarena_verified.api import WebArenaVerified
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite


def evaluate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a WebArena task payload and return a JSON-serializable summary."""
    task_id = payload["task_id"]
    config = _build_config(payload.get("environments", {}))

    evaluator = WebArenaVerified(config=config)
    result = evaluator.evaluate_task(
        task_id=task_id,
        agent_response=payload.get("agent_response", {}),
        network_trace=payload.get("network_trace", []),
    )

    parts = []
    evaluators_results = []
    for evaluator_result in result.evaluators_results:
        status = (
            evaluator_result.status
            if isinstance(evaluator_result.status, str)
            else evaluator_result.status.value
        )
        part = f"[{evaluator_result.evaluator_name}] {status.upper()}"
        if evaluator_result.error_msg:
            part += f": {evaluator_result.error_msg}"
        parts.append(part)
        evaluators_results.append(
            {
                "evaluator_name": evaluator_result.evaluator_name,
                "status": status,
                "error_msg": evaluator_result.error_msg,
            }
        )

    return {
        "passed": result.score == 1.0,
        "score": result.score,
        "status": result.status if isinstance(result.status, str) else result.status.value,
        "message": "; ".join(parts) if parts else f"score={result.score}, status={result.status}",
        "evaluators_results": evaluators_results,
    }


def _build_config(environments_payload: dict[str, list[str]]) -> WebArenaVerifiedConfig:
    environments: dict[WebArenaSite, EnvironmentConfig] = {}
    for site_name, urls in environments_payload.items():
        if not urls:
            continue
        try:
            site = WebArenaSite(site_name)
        except ValueError:
            continue
        environments[site] = EnvironmentConfig(urls=urls)
    return WebArenaVerifiedConfig(environments=environments or None)


def main() -> None:
    payload = json.load(sys.stdin)
    response = evaluate_payload(payload)
    json.dump(response, sys.stdout)


if __name__ == "__main__":
    main()
