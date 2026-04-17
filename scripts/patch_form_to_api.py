#!/usr/bin/env python3
"""
Patch adversarial_tasks.json to convert form-based Magento and GitLab tasks
to API-based tasks for the 19 Phase 4 tasks.

Transformations:
1. Shopping review tasks: form -> api, /review/product/post/id/{N}/ -> /rest/V1/reviews
2. Shopping_admin createpost tasks: form -> api, /customer/account/createpost/ -> /rest/V1/customers
3. GitLab form tasks: form -> api, web paths -> /api/v4/... paths
4. Shopping_admin tasks: ensure delivery_channel.delivery_site = "shopping"
5. Reddit / Map tasks: no mechanism change, just verify
"""

import json
import re
import sys
from copy import deepcopy
from pathlib import Path

TASKS_FILE = Path("logs/phase_2/adversarial_tasks.json")

PHASE4_IDS = {
    "gitlab-shard-0_adv-156-authority-pt",
    "gitlab-shard-0_adv-44-moderation-appeal-pt",
    "gitlab-shard-0_adv-105-policy-mfs",
    "gitlab-shard-3_adv-417-error-recovery-markdown-2",
    "gitlab-shard-3_adv-418-seller-footnote-markdown",
    "gitlab-shard-8_adv_784_peer_tip_plaintext_1",
    "map-shard-1_AT-011",
    "reddit-shard-0_adv_reddit_014",
    "reddit-shard-0_adv_reddit_015",
    "shopping-shard-1_adv_167_ci_failure_det",
    "shopping-shard-1_adv_159_error_recovery_det",
    "shopping-shard-2_adv-238-moderation-appeal-1",
    "shopping-shard-2_adv-239-moderation-appeal-2",
    "shopping-shard-6_adv-435-error-recovery-3",
    "shopping_admin-shard-0_adv-007",
    "shopping_admin-shard-1_adv-112-authority-review-detail",
    "shopping_admin-shard-2_adv-183-policy-2",
    "shopping_admin-shard-2_adv-187-support_escalation-2",
    "shopping_admin-shard-2_adv-130-urgency-1",
}

# ── Shopping review tasks (form -> api) ──────────────────────────────────────

SHOPPING_REVIEW_IDS = {
    # shopping site
    "shopping-shard-1_adv_167_ci_failure_det",
    "shopping-shard-1_adv_159_error_recovery_det",
    "shopping-shard-2_adv-238-moderation-appeal-1",
    "shopping-shard-2_adv-239-moderation-appeal-2",
    "shopping-shard-6_adv-435-error-recovery-3",
    # shopping_admin site (review tasks, not createpost)
    "shopping_admin-shard-1_adv-112-authority-review-detail",
    "shopping_admin-shard-2_adv-183-policy-2",
    "shopping_admin-shard-2_adv-130-urgency-1",
}

# ── Shopping_admin createpost tasks (form -> api) ────────────────────────────

CREATEPOST_IDS = {
    "shopping_admin-shard-0_adv-007",
    "shopping_admin-shard-2_adv-187-support_escalation-2",
}

# ── GitLab form tasks (form -> api) ─────────────────────────────────────────

GITLAB_FORM_IDS = {
    "gitlab-shard-0_adv-156-authority-pt",  # MR
    "gitlab-shard-3_adv-417-error-recovery-markdown-2",  # notes
    "gitlab-shard-3_adv-418-seller-footnote-markdown",  # milestones
}

# ── GitLab tasks already API-based (no mechanism change needed) ──────────────

GITLAB_ALREADY_API = {
    "gitlab-shard-0_adv-44-moderation-appeal-pt",
    "gitlab-shard-0_adv-105-policy-mfs",
    "gitlab-shard-8_adv_784_peer_tip_plaintext_1",
}

# ── Reddit / Map tasks (no mechanism change needed) ──────────────────────────

NO_CHANGE_IDS = {
    "map-shard-1_AT-011",
    "reddit-shard-0_adv_reddit_014",
    "reddit-shard-0_adv_reddit_015",
}


def extract_product_id(path: str) -> int:
    """Extract product ID N from /review/product/post/id/N/"""
    m = re.search(r"/review/product/post/id/(\d+)/", path)
    if not m:
        raise ValueError(f"Cannot extract product ID from path: {path}")
    return int(m.group(1))


def convert_shopping_review(call: dict) -> dict:
    """Convert a shopping review form call to REST API call."""
    product_id = extract_product_id(call["path"])
    body_form = call["body_form"]

    title = body_form.get("title", "")
    detail = body_form.get("detail", "")
    nickname = body_form.get("nickname", "")

    new_call = {
        "method": "POST",
        "path": "/rest/V1/reviews",
        "body": {
            "review": {
                "title": title,
                "detail": detail,
                "nickname": nickname,
                "review_entity": "product",
                "entity_pk_value": product_id,
                "store_id": 1,
                "stores": [1],
            }
        },
    }
    # Preserve headers if present
    if "headers" in call:
        new_call["headers"] = call["headers"]
    return new_call


def convert_createpost(call: dict) -> dict:
    """Convert a customer createpost form call to REST API call."""
    body_form = call["body_form"]

    new_call = {
        "method": "POST",
        "path": "/rest/V1/customers",
        "body": {
            "customer": {
                "email": body_form.get("email", ""),
                "firstname": body_form.get("firstname", ""),
                "lastname": body_form.get("lastname", ""),
            },
            "password": body_form.get("password", ""),
        },
    }
    if "headers" in call:
        new_call["headers"] = call["headers"]
    return new_call


def convert_gitlab_mr(call: dict) -> dict:
    """Convert GitLab MR form call to API call."""
    body_form = call["body_form"]

    new_call = {
        "method": "POST",
        "path": "/api/v4/projects/189/merge_requests",
        "body": {
            "source_branch": body_form.get("merge_request[source_branch]", ""),
            "target_branch": body_form.get("merge_request[target_branch]", ""),
            "title": body_form.get("merge_request[title]", ""),
        },
    }
    if "headers" in call:
        new_call["headers"] = call["headers"]
    return new_call


def convert_gitlab_milestones(call: dict) -> dict:
    """Convert GitLab milestones form call to API call."""
    body_form = call["body_form"]

    new_call = {
        "method": "POST",
        "path": "/api/v4/projects/189/milestones",
        "body": {
            "title": body_form.get("milestone[title]", ""),
            "description": body_form.get("milestone[description]", ""),
        },
    }
    if "headers" in call:
        new_call["headers"] = call["headers"]
    return new_call


def convert_gitlab_notes(call: dict) -> dict:
    """Convert GitLab notes form call to API call."""
    body_form = call["body_form"]

    new_call = {
        "method": "POST",
        "path": "/api/v4/projects/189/issues/1/notes",
        "body": {
            "body": body_form.get("note[note]", ""),
        },
    }
    if "headers" in call:
        new_call["headers"] = call["headers"]
    return new_call


def patch_task(task: dict) -> dict:
    """Apply all transformations to a single task."""
    task = deepcopy(task)
    tid = task["id"]
    st = task["seed_template"]
    dc = task["delivery_channel"]

    # ── Shopping_admin tasks: ensure delivery_site = "shopping" ───────────
    if task.get("site") == "shopping_admin":
        dc["delivery_site"] = "shopping"

    # ── Shopping review tasks ────────────────────────────────────────────
    if tid in SHOPPING_REVIEW_IDS:
        st["mechanism"] = "api"
        dc["mechanism"] = "api"
        st["api_calls"] = [convert_shopping_review(c) for c in st["api_calls"]]

    # ── Createpost tasks ─────────────────────────────────────────────────
    elif tid in CREATEPOST_IDS:
        st["mechanism"] = "api"
        dc["mechanism"] = "api"
        st["api_calls"] = [convert_createpost(c) for c in st["api_calls"]]

    # ── GitLab form tasks ────────────────────────────────────────────────
    elif tid in GITLAB_FORM_IDS:
        st["mechanism"] = "api"
        dc["mechanism"] = "api"
        if tid == "gitlab-shard-0_adv-156-authority-pt":
            st["api_calls"] = [convert_gitlab_mr(c) for c in st["api_calls"]]
        elif tid == "gitlab-shard-3_adv-418-seller-footnote-markdown":
            st["api_calls"] = [convert_gitlab_milestones(c) for c in st["api_calls"]]
        elif tid == "gitlab-shard-3_adv-417-error-recovery-markdown-2":
            st["api_calls"] = [convert_gitlab_notes(c) for c in st["api_calls"]]

    # ── GitLab already API / Reddit / Map: no mechanism change ───────────
    elif tid in GITLAB_ALREADY_API or tid in NO_CHANGE_IDS:
        pass  # no changes needed beyond delivery_site (handled above)

    return task


def verify_no_body_form(task: dict) -> list[str]:
    """Check that no api_call still has body_form after patching."""
    issues = []
    for i, call in enumerate(task["seed_template"].get("api_calls", [])):
        if "body_form" in call:
            if task["id"] not in NO_CHANGE_IDS:
                issues.append(f"  api_call[{i}] still has body_form")
    return issues


def main():
    with open(TASKS_FILE) as f:
        data = json.load(f)

    found_ids = set()
    for i, task in enumerate(data):
        if task["id"] in PHASE4_IDS:
            found_ids.add(task["id"])
            data[i] = patch_task(task)

    # Verify all 19 found
    missing = PHASE4_IDS - found_ids
    if missing:
        print(f"ERROR: {len(missing)} tasks not found: {missing}", file=sys.stderr)
        sys.exit(1)

    # Write back
    with open(TASKS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Patched {len(found_ids)} tasks in {TASKS_FILE}")
    print()

    # ── Verification summary ─────────────────────────────────────────────
    with open(TASKS_FILE) as f:
        data = json.load(f)

    all_issues = []
    for task in data:
        if task["id"] in PHASE4_IDS:
            tid = task["id"]
            st = task["seed_template"]
            dc = task["delivery_channel"]
            calls = st.get("api_calls", [])

            print(f"{'─' * 60}")
            print(f"  ID:                  {tid}")
            print(f"  site:                {task.get('site')}")
            print(f"  seed_mechanism:      {st.get('mechanism')}")
            print(f"  delivery_mechanism:  {dc.get('mechanism')}")
            print(f"  delivery_site:       {dc.get('delivery_site')}")
            for j, c in enumerate(calls):
                body_type = "body" if "body" in c else ("body_form" if "body_form" in c else "NONE")
                print(f"  api_call[{j}].path:   {c.get('path')}")
                print(f"  api_call[{j}].type:   {body_type}")

            issues = verify_no_body_form(task)
            if issues:
                all_issues.extend([f"{tid}: {issue}" for issue in issues])

    print(f"{'─' * 60}")
    print()
    if all_issues:
        print("ISSUES FOUND:")
        for issue in all_issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("All 19 tasks verified OK. No remaining body_form on API tasks.")


if __name__ == "__main__":
    main()
