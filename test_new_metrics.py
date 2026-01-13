#!/usr/bin/env python3
"""Test script for new redteam judging metrics."""

import sys
sys.path.insert(0, 'AgentLab/src')

from agentlab.analyze.inspect_results import load_result_df, global_report
from agentlab.benchmarks.redteam import analyze_results

print("=" * 60)
print("Testing New Redteam Judging Metrics")
print("=" * 60)

# Test 1: Load Results and Check Judge Columns
print("\n### Test 1: Load Results and Check Judge Columns ###\n")
df = load_result_df("results/2026-01-12_19-36-22_genericagent-anthropic-claude-opus-4-5-on-enriched-behaviors")
df_reset = df.reset_index(inplace=False)

judge_cols = [c for c in df_reset.columns if 'judge' in c.lower()]
print(f"Judge-related columns found: {judge_cols}")

expected = ['judge.overall_success', 'judge.overall_score', 'judge_result_full']
for col in expected:
    if col in df_reset.columns:
        print(f"✓ {col} found")
    else:
        print(f"✗ {col} missing")

# Test 2: Global Report with Auto-Detection
print("\n### Test 2: Global Report with Auto-Detection ###\n")
report = global_report(df)
print("Report columns:", list(report.columns))
print("\nReport:")
print(report)

# Test 3: Unified analyze_results Function
print("\n### Test 3: Unified analyze_results Function ###\n")
results_list = df_reset.to_dict('records')

print("\n--- Test 3a: Both benign and adversarial ---")
metrics = analyze_results(results_list, verbose=True)

print("\n--- Test 3b: Just adversarial ---")
adv_only = [r for r in results_list if r.get('env.test_mode') == 'adversarial']
print(f"Filtering to {len(adv_only)} adversarial results")
metrics_adv = analyze_results(adv_only, verbose=True)

print("\n--- Test 3c: Just benign ---")
benign_only = [r for r in results_list if r.get('env.test_mode') == 'benign']
print(f"Filtering to {len(benign_only)} benign results")
metrics_benign = analyze_results(benign_only, verbose=True)

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
