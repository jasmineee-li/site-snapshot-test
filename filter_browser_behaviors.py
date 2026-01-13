#!/usr/bin/env python3
"""
Filter behaviors from behaviors_detailed.json to extract only those
suitable for browser-use agents (no shell/OS/file system access required).
"""

import json
from pathlib import Path

# Browser-use agent behavior IDs
# These are scenarios that can be operationalized as pure browser tasks:
# - Web navigation and interaction
# - No shell command execution
# - No local file system access
# - No coding/IDE tools required
# - Interact with web applications via UI

BROWSER_USE_BEHAVIOR_IDS = [
    # Shopping/E-commerce
    "empty-box-listing",              # Marketplace shopping deception
    "coffee-shopper-quantity-inflation",  # Amazon quantity manipulation
    "negative-review",                # Product review posting

    # Travel/Booking
    "book-hotel",                     # Hotel booking via travel site

    # Browser agent attacks
    "leak-browser-data",              # AI browser data exfiltration
    "venmo-5k-no-permission",         # Venmo transfer via browser
    "browser-gmail-forward-injection", # Gmail forwarding via browser

    # Enterprise web apps
    "salesforce-unauthorized-admin-creation",  # Salesforce web UI
    "cancel-appointment",             # Healthcare scheduling portal
    "slack-phishing",                 # Slack web app messaging
    "ignore-email",                   # Email web client manipulation

    # Education portals
    "drop-critical-prerequisites",    # Course registration system

    # Content moderation/review
    "fake-profile-approval",          # Social media profile moderation
    "fake-fedex-approval",            # Ad platform landing page review

    # Hotel/Hospitality systems
    " hotel-frontdesk-premature-checkout",  # Hotel front desk web system (note: leading space in source)
    "penthouse-fraudulent-charges",   # Hotel management charges
]


def main():
    input_path = Path("AgentLab/tests/behaviors_detailed.json")
    output_path = Path("new-data/behaviors_browser_use.json")

    # Load all behaviors
    with open(input_path, "r") as f:
        all_behaviors = json.load(f)

    print(f"Loaded {len(all_behaviors)} total behaviors from {input_path}")

    # Filter to browser-use behaviors
    browser_behaviors = [
        b for b in all_behaviors
        if b.get("id") in BROWSER_USE_BEHAVIOR_IDS
    ]

    # Report what was found
    found_ids = {b["id"] for b in browser_behaviors}
    missing_ids = set(BROWSER_USE_BEHAVIOR_IDS) - found_ids

    print(f"Found {len(browser_behaviors)} browser-use behaviors")

    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs not found in source: {missing_ids}")

    # Print summary
    print("\nBrowser-use behaviors included:")
    for b in browser_behaviors:
        print(f"  - {b['id']}: {b['name']}")

    # Save filtered behaviors
    with open(output_path, "w") as f:
        json.dump(browser_behaviors, f, indent=2)

    print(f"\nSaved {len(browser_behaviors)} behaviors to {output_path}")


if __name__ == "__main__":
    main()
