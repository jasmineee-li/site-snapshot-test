#!/usr/bin/env python3
"""Test script to verify judge metadata fixes."""

import sys
import json
from pathlib import Path

sys.path.insert(0, 'AgentLab/src')

from agentlab.benchmarks.redteam_judge import JudgeRunner
from agentlab.experiments.loop import ExpResult

# Test with existing adversarial result
exp_dir = Path("results/2026-01-12_19-36-22_genericagent-anthropic-claude-opus-4-5-on-enriched-behaviors/2026-01-12_19-36-22_GenericAgent-anthropic_claude-opus-4.5_on_enriched-behaviors.salesforce-unauthorized-admin-creation.adversarial_v0_0")

print("=" * 70)
print("Testing Judge Metadata Fixes")
print("=" * 70)

# Load existing injections.json
injections_file = exp_dir / "injections.json"
print(f"\n1. Loading injection metadata from {injections_file}")
with open(injections_file) as f:
    injections = json.load(f)

print(f"   Test mode: {injections['test_mode']}")
print(f"   Injection locations: {injections['injection_locations']}")
print(f"   Placeholder count: {len(injections['placeholder_content'])}")

# Load old judge result for comparison
old_judge_file = exp_dir / "judge_result.json"
print(f"\n2. Loading old judge result from {old_judge_file}")
with open(old_judge_file) as f:
    old_result = json.load(f)

print(f"   Old adversarial_elements_found: {old_result['html_state_summary']['adversarial_elements_found']}")
print(f"   Old pages_analyzed: {len(old_result['html_state_summary']['pages_analyzed'])}")

# Test metadata loading in JudgeRunner
print(f"\n3. Testing JudgeRunner._load_injection_metadata()")
exp_result = ExpResult(exp_dir)
env_args = exp_result.exp_args.env_args

runner = JudgeRunner(env_args)
metadata = runner._load_injection_metadata(exp_dir)

print(f"   ✓ Loaded metadata successfully")
print(f"   Test mode: {metadata['test_mode']}")
print(f"   Injection locations: {metadata['injection_locations']}")
print(f"   Placeholder content keys: {list(metadata['placeholder_content'].keys())}")

# Re-run judge with new metadata-based approach
print(f"\n4. Re-running judge with metadata-based HTML state analysis")
episode_info = exp_result.episode_info

new_result = runner.evaluate(exp_dir, episode_info)

print(f"\n5. New judge result:")
print(f"   Overall success: {new_result['overall_success']}")
print(f"   Overall score: {new_result['overall_score']}")
print(f"   Adversarial elements found: {new_result['html_state_summary']['adversarial_elements_found']}")
print(f"   Pages analyzed: {len(new_result['html_state_summary']['pages_analyzed'])}")
print(f"   Metadata source: {new_result['html_state_summary'].get('metadata_source', 'N/A')}")

# Compare old vs new
print(f"\n6. Comparison:")
print(f"   Old adversarial_elements_found: {old_result['html_state_summary']['adversarial_elements_found']}")
print(f"   New adversarial_elements_found: {new_result['html_state_summary']['adversarial_elements_found']}")

if new_result['html_state_summary']['adversarial_elements_found'] > 0:
    print(f"   ✓ SUCCESS: Now detecting injections from metadata!")
    print(f"   Visited injected pages: {new_result['html_state_summary'].get('visited_injected_pages', [])}")
else:
    print(f"   ✗ FAIL: Still not detecting injections")

# Save new result for inspection
new_judge_file = exp_dir / "judge_result_new.json"
with open(new_judge_file, 'w') as f:
    json.dump(new_result, f, indent=2)
print(f"\n7. Saved new judge result to {new_judge_file}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
