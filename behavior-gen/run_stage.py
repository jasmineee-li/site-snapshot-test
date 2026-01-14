#!/usr/bin/env python3
"""
Interactive stage runner for testing individual pipeline stages.

Usage:
    python run_stage.py stage2 --num 3           # Generate 3 trajectories
    python run_stage.py stage3 --workflow-file   # Analyze injection points
    python run_stage.py stage4 --injection-file  # Generate goal pairs
    python run_stage.py stage5 --goal-file       # Synthesize behaviors
    python run_stage.py stage6 --behavior-file   # Validate behavior
"""

import argparse
import json
import sys
from pathlib import Path

from pipeline import (
    load_websites,
    generate_trajectories,
    analyze_injection_points,
    generate_goal_pairs,
    synthesize_behaviors,
    validate_behavior,
    format_behavior_for_worldsim,
    Workflow,
    WorkflowStep,
    InjectionPoint,
    InjectionMechanism,
    AttackScenario,
    BehaviorPlan,
    SimulationConfig,
    PageConfig,
)


def save_json(data, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to: {filepath}")


def run_stage2(args):
    """Run Stage 2: Trajectory Generation."""
    print("=" * 60)
    print("Stage 2: Trajectory Generation")
    print("=" * 60)

    websites = load_websites(args.websites)
    print(f"Loaded {len(websites)} websites")

    workflows = generate_trajectories(
        websites=websites,
        num_trajectories=args.num,
        model=args.model
    )

    # Convert to serializable format
    output = {
        "workflows": [
            {
                "workflow_id": w.workflow_id,
                "user_goal": w.user_goal,
                "steps": [
                    {
                        "site": s.site,
                        "action_type": s.action_type,
                        "description": s.description,
                        "reads_external_content": s.reads_external_content
                    }
                    for s in w.steps
                ],
                "data_flow": w.data_flow,
                "auth_context": w.auth_context
            }
            for w in workflows
        ]
    }

    save_json(output, args.output)

    # Print summary
    print("\nGenerated Workflows:")
    for w in workflows:
        injection_steps = sum(1 for s in w.steps if s.reads_external_content)
        print(f"  - {w.workflow_id}: {w.user_goal[:60]}...")
        print(f"    Steps: {len(w.steps)}, Injection surfaces: {injection_steps}")


def run_stage3(args):
    """Run Stage 3: Injection Point Analysis."""
    print("=" * 60)
    print("Stage 3: Injection Point Analysis")
    print("=" * 60)

    # Load workflows
    with open(args.workflow_file, 'r') as f:
        data = json.load(f)

    workflows = []
    for w in data.get("workflows", []):
        steps = [
            WorkflowStep(
                site=s["site"],
                action_type=s["action_type"],
                description=s["description"],
                reads_external_content=s.get("reads_external_content", False)
            )
            for s in w.get("steps", [])
        ]
        workflows.append(Workflow(
            workflow_id=w["workflow_id"],
            user_goal=w["user_goal"],
            steps=steps,
            data_flow=w.get("data_flow", ""),
            auth_context=w.get("auth_context", [])
        ))

    print(f"Loaded {len(workflows)} workflows")

    # Analyze each workflow
    all_results = []
    for workflow in workflows:
        injection_points = analyze_injection_points(workflow, model=args.model)

        result = {
            "workflow_id": workflow.workflow_id,
            "injection_points": [
                {
                    "step_index": ip.step_index,
                    "site": ip.site,
                    "location_type": ip.location_type,
                    "injection_mechanisms": [
                        {
                            "mechanism": m.mechanism,
                            "description": m.description,
                            "attacker_access": m.attacker_access,
                            "human_blindspot": m.human_blindspot,
                            "plausibility_score": m.plausibility_score
                        }
                        for m in ip.injection_mechanisms
                    ],
                    "agent_context": ip.agent_context
                }
                for ip in injection_points
            ]
        }
        all_results.append(result)

        # Print summary
        print(f"\n{workflow.workflow_id}:")
        for ip in injection_points:
            best_score = max((m.plausibility_score for m in ip.injection_mechanisms), default=0)
            print(f"  - {ip.site} ({ip.location_type}): plausibility={best_score}")

    save_json({"analyses": all_results}, args.output)


def run_stage4(args):
    """Run Stage 4: Goal Pair Generation."""
    print("=" * 60)
    print("Stage 4: Adversarial-Benign Goal Pair Generation")
    print("=" * 60)

    # Load injection analysis
    with open(args.injection_file, 'r') as f:
        data = json.load(f)

    # Load original workflows for context
    with open(args.workflow_file, 'r') as f:
        workflow_data = json.load(f)

    workflow_map = {}
    for w in workflow_data.get("workflows", []):
        steps = [
            WorkflowStep(
                site=s["site"],
                action_type=s["action_type"],
                description=s["description"],
                reads_external_content=s.get("reads_external_content", False)
            )
            for s in w.get("steps", [])
        ]
        workflow_map[w["workflow_id"]] = Workflow(
            workflow_id=w["workflow_id"],
            user_goal=w["user_goal"],
            steps=steps,
            data_flow=w.get("data_flow", ""),
            auth_context=w.get("auth_context", [])
        )

    all_results = []
    for analysis in data.get("analyses", []):
        workflow_id = analysis["workflow_id"]
        workflow = workflow_map.get(workflow_id)

        if not workflow:
            print(f"Warning: Workflow {workflow_id} not found")
            continue

        for ip_data in analysis.get("injection_points", []):
            # Reconstruct InjectionPoint
            mechanisms = [
                InjectionMechanism(
                    mechanism=m["mechanism"],
                    description=m["description"],
                    attacker_access=m["attacker_access"],
                    human_blindspot=m["human_blindspot"],
                    plausibility_score=m.get("plausibility_score", 5)
                )
                for m in ip_data.get("injection_mechanisms", [])
            ]
            injection_point = InjectionPoint(
                step_index=ip_data["step_index"],
                site=ip_data["site"],
                location_type=ip_data["location_type"],
                injection_mechanisms=mechanisms,
                agent_context=ip_data.get("agent_context", {})
            )

            # Generate attack scenarios
            scenarios = generate_goal_pairs(workflow, injection_point, model=args.model)

            result = {
                "workflow_id": workflow_id,
                "injection_site": injection_point.site,
                "location_type": injection_point.location_type,
                "scenarios": [
                    {
                        "scenario_id": sc.scenario_id,
                        "user_task": sc.user_task,
                        "attacker_goal": sc.attacker_goal,
                        "harm_category": sc.harm_category,
                        "harm_severity": sc.harm_severity,
                        "harm_justification": sc.harm_justification,
                        "attack_vector": sc.attack_vector,
                        "model_rationalization": sc.model_rationalization,
                        "user_blindspot": sc.user_blindspot
                    }
                    for sc in scenarios
                ]
            }
            all_results.append(result)

            # Print summary
            print(f"\n{workflow_id} @ {injection_point.site}:")
            for sc in scenarios:
                print(f"  - [{sc.harm_category}] severity={sc.harm_severity}")
                print(f"    User task: {sc.user_task[:50]}...")
                print(f"    Attacker goal: {sc.attacker_goal[:50]}...")

    save_json({"scenarios": all_results}, args.output)


def run_stage5(args):
    """Run Stage 5: Behavior Synthesis."""
    print("=" * 60)
    print("Stage 5: Behavior Synthesis & Selection")
    print("=" * 60)

    # Load attack scenarios
    with open(args.scenario_file, 'r') as f:
        goal_data = json.load(f)

    # Load workflows
    with open(args.workflow_file, 'r') as f:
        workflow_data = json.load(f)

    # Load injection analysis
    with open(args.injection_file, 'r') as f:
        injection_data = json.load(f)

    # Build lookup maps
    workflow_map = {}
    for w in workflow_data.get("workflows", []):
        steps = [
            WorkflowStep(
                site=s["site"],
                action_type=s["action_type"],
                description=s["description"],
                reads_external_content=s.get("reads_external_content", False)
            )
            for s in w.get("steps", [])
        ]
        workflow_map[w["workflow_id"]] = Workflow(
            workflow_id=w["workflow_id"],
            user_goal=w["user_goal"],
            steps=steps,
            data_flow=w.get("data_flow", ""),
            auth_context=w.get("auth_context", [])
        )

    injection_map = {}
    for analysis in injection_data.get("analyses", []):
        wid = analysis["workflow_id"]
        for ip_data in analysis.get("injection_points", []):
            key = f"{wid}:{ip_data['site']}"
            mechanisms = [
                InjectionMechanism(
                    mechanism=m["mechanism"],
                    description=m["description"],
                    attacker_access=m["attacker_access"],
                    human_blindspot=m["human_blindspot"],
                    plausibility_score=m.get("plausibility_score", 5)
                )
                for m in ip_data.get("injection_mechanisms", [])
            ]
            injection_map[key] = InjectionPoint(
                step_index=ip_data["step_index"],
                site=ip_data["site"],
                location_type=ip_data["location_type"],
                injection_mechanisms=mechanisms,
                agent_context=ip_data.get("agent_context", {})
            )

    all_behaviors = []
    for sc_result in goal_data.get("scenarios", []):
        workflow_id = sc_result["workflow_id"]
        site = sc_result["injection_site"]
        workflow = workflow_map.get(workflow_id)
        injection_point = injection_map.get(f"{workflow_id}:{site}")

        if not workflow or not injection_point:
            print(f"Warning: Missing data for {workflow_id}:{site}")
            continue

        # Use highest severity scenario
        scenarios_data = sc_result.get("scenarios", [])
        if not scenarios_data:
            continue

        best_sc_data = max(scenarios_data, key=lambda s: s.get("harm_severity", 0))
        scenario = AttackScenario(
            scenario_id=best_sc_data["scenario_id"],
            user_task=best_sc_data["user_task"],
            attacker_goal=best_sc_data["attacker_goal"],
            harm_category=best_sc_data["harm_category"],
            harm_severity=best_sc_data.get("harm_severity", 5),
            harm_justification=best_sc_data.get("harm_justification", ""),
            attack_vector=best_sc_data.get("attack_vector", ""),
            model_rationalization=best_sc_data.get("model_rationalization", ""),
            user_blindspot=best_sc_data.get("user_blindspot", "")
        )

        # Synthesize behaviors
        plans, selected_idx = synthesize_behaviors(
            workflow, injection_point, scenario, model=args.model
        )

        if not plans:
            continue

        selected = plans[selected_idx]

        # Serialize simulation_config if present
        sim_config_dict = None
        if selected.simulation_config:
            sim_config_dict = {
                "start_page": selected.simulation_config.start_page,
                "environment_summary": selected.simulation_config.environment_summary,
                "success_condition": selected.simulation_config.success_condition,
                "pages": [
                    {
                        "id": p.id,
                        "base_site_url": p.base_site_url,
                        "subdomains": p.subdomains,
                        "details": p.details,
                        "mode": p.mode
                    }
                    for p in selected.simulation_config.pages
                ]
            }

        behavior_result = {
            "workflow_id": workflow_id,
            "injection_site": site,
            "scenario": {
                "user_task": scenario.user_task,
                "attacker_goal": scenario.attacker_goal,
                "harm_category": scenario.harm_category,
                "harm_severity": scenario.harm_severity
            },
            "selected_plan": {
                "behavior_id": selected.behavior_id,
                "name": selected.name,
                "description": selected.description,
                "threat_context": selected.threat_context,
                "user_scenario": selected.user_scenario,
                "user_task": selected.user_task,
                "attacker_goal": selected.attacker_goal,
                "injection_hint": selected.injection_hint,
                "success_criteria": selected.success_criteria,
                "simulation_config": sim_config_dict,
                "realism_assessment": selected.realism_assessment
            },
            "all_plans": [
                {
                    "behavior_id": p.behavior_id,
                    "name": p.name,
                    "description": p.description
                }
                for p in plans
            ],
            "selected_index": selected_idx
        }
        all_behaviors.append(behavior_result)

        print(f"\n{workflow_id}:")
        print(f"  Selected: {selected.name}")
        if sim_config_dict:
            print(f"  Start page: {sim_config_dict['start_page']}")

    save_json({"behaviors": all_behaviors}, args.output)


def run_stage6(args):
    """Run Stage 6: Quality Validation."""
    print("=" * 60)
    print("Stage 6: Quality Validation & Formatting")
    print("=" * 60)

    # Load synthesized behaviors
    with open(args.behavior_file, 'r') as f:
        behavior_data = json.load(f)

    validated_behaviors = []
    for b in behavior_data.get("behaviors", []):
        plan_data = b.get("selected_plan", {})

        # Reconstruct simulation_config if present
        sim_config = None
        sc_data = plan_data.get("simulation_config")
        if sc_data:
            pages = []
            for page_data in sc_data.get("pages", []):
                pages.append(PageConfig(
                    id=page_data.get("id", "unknown"),
                    base_site_url=page_data.get("base_site_url", ""),
                    subdomains=page_data.get("subdomains", []),
                    details=page_data.get("details", {}),
                    mode=page_data.get("mode", "synthetic")
                ))
            sim_config = SimulationConfig(
                start_page=sc_data.get("start_page", ""),
                environment_summary=sc_data.get("environment_summary", ""),
                success_condition=sc_data.get("success_condition", ""),
                pages=pages
            )

        plan = BehaviorPlan(
            behavior_id=plan_data.get("behavior_id", ""),
            name=plan_data.get("name", ""),
            description=plan_data.get("description", ""),
            threat_context=plan_data.get("threat_context", ""),
            user_scenario=plan_data.get("user_scenario", ""),
            user_task=plan_data.get("user_task", ""),
            attacker_goal=plan_data.get("attacker_goal", ""),
            injection_hint=plan_data.get("injection_hint", ""),
            success_criteria=plan_data.get("success_criteria", []),
            simulation_config=sim_config,
            realism_assessment=plan_data.get("realism_assessment", "")
        )

        # Validate
        validation = validate_behavior(plan, model=args.validation_model)

        result = {
            **b,
            "validation": validation
        }
        validated_behaviors.append(result)

        # Print summary
        passed = validation.get("pass", False)
        score = validation.get("overall_score", "N/A")
        status = "PASS" if passed else "FAIL"
        print(f"\n{plan.name}: [{status}] score={score}/10")

        if validation.get("issues"):
            for issue in validation["issues"][:3]:
                print(f"  ! {issue}")

    # Filter passing behaviors
    passing = [b for b in validated_behaviors if b.get("validation", {}).get("pass", False)]

    print(f"\n{'=' * 60}")
    print(f"Validation complete: {len(passing)}/{len(validated_behaviors)} passed")

    save_json({
        "all_behaviors": validated_behaviors,
        "passing_behaviors": passing
    }, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Run individual pipeline stages for testing"
    )
    subparsers = parser.add_subparsers(dest="stage", help="Stage to run")

    # Stage 2
    p2 = subparsers.add_parser("stage2", help="Trajectory Generation")
    p2.add_argument("--websites", "-w", default="websites.json")
    p2.add_argument("--num", "-n", type=int, default=3, help="Number of trajectories (default: 3 for testing)")
    p2.add_argument("--output", "-o", default="stage2_workflows.json")
    p2.add_argument("--model", "-m", default="openrouter/anthropic/claude-sonnet-4")

    # Stage 3
    p3 = subparsers.add_parser("stage3", help="Injection Point Analysis")
    p3.add_argument("--workflow-file", "-wf", default="stage2_workflows.json")
    p3.add_argument("--output", "-o", default="stage3_injections.json")
    p3.add_argument("--model", "-m", default="openrouter/anthropic/claude-sonnet-4")

    # Stage 4
    p4 = subparsers.add_parser("stage4", help="Attack Scenario Generation")
    p4.add_argument("--workflow-file", "-wf", default="stage2_workflows.json")
    p4.add_argument("--injection-file", "-if", default="stage3_injections.json")
    p4.add_argument("--output", "-o", default="stage4_scenarios.json")
    p4.add_argument("--model", "-m", default="openrouter/anthropic/claude-sonnet-4")

    # Stage 5
    p5 = subparsers.add_parser("stage5", help="Behavior Synthesis")
    p5.add_argument("--workflow-file", "-wf", default="stage2_workflows.json")
    p5.add_argument("--injection-file", "-if", default="stage3_injections.json")
    p5.add_argument("--scenario-file", "-sf", default="stage4_scenarios.json")
    p5.add_argument("--output", "-o", default="stage5_behaviors.json")
    p5.add_argument("--model", "-m", default="openrouter/anthropic/claude-sonnet-4")

    # Stage 6
    p6 = subparsers.add_parser("stage6", help="Quality Validation")
    p6.add_argument("--behavior-file", "-bf", default="stage5_behaviors.json")
    p6.add_argument("--output", "-o", default="stage6_validated.json")
    p6.add_argument("--validation-model", default="openrouter/anthropic/claude-sonnet-4")

    args = parser.parse_args()

    if args.stage == "stage2":
        run_stage2(args)
    elif args.stage == "stage3":
        run_stage3(args)
    elif args.stage == "stage4":
        run_stage4(args)
    elif args.stage == "stage5":
        run_stage5(args)
    elif args.stage == "stage6":
        run_stage6(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
