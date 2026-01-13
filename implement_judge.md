Plan for Integrating Judging into the Pipeline
Here's my proposed architecture:
1. Integration Points
redteam_test.py
    └── Study.run()
            └── loop.py: run_experiment()
                    └── After experiment completes:
                        └── JudgeRunner.evaluate(exp_result, behavior_config)
                            └── Returns JudgeResult
                    └── Save to exp_dir/judge_result.json
            └── After all experiments:
                └── Aggregate judge results into log.json
redteam_test.py    └── Study.run()            └── loop.py: run_experiment()                    └── After experiment completes:                        └── JudgeRunner.evaluate(exp_result, behavior_config)                            └── Returns JudgeResult                    └── Save to exp_dir/judge_result.json            └── After all experiments:                └── Aggregate judge results into log.json
2. Key Changes Required
A. In RedteamBenchmark or RedteamEnv:
Store the full behavior config (including judging section) in env_args
Currently judging config is not being passed through
B. In loop.py (experiment runner):
After env.step() loop completes, call judge_experiment(exp_result)
This runs all configured judges from env_args.judging.submit.panel
Save result to exp_dir/judge_result.json
C. Create JudgeRunner class in redteam_judge.py:
class JudgeRunner:
    def __init__(self, behavior_config: dict):
        self.panel = behavior_config['judging']['submit']['panel']
        self.criteria = behavior_config['criteria']
    
    def evaluate(self, exp_result) -> dict:
        results = []
        for judge_config in self.panel:
            if not judge_config.get('enabled', False):
                continue
            judge = self._get_judge(judge_config['judge_id'])
            result = judge.evaluate(
                exp_result,
                parameters=judge_config['parameters'],
                criteria_config=judge_config['criteria_config']
            )
            results.append(result)
        return self._aggregate(results)
class JudgeRunner:    def __init__(self, behavior_config: dict):        self.panel = behavior_config['judging']['submit']['panel']        self.criteria = behavior_config['criteria']        def evaluate(self, exp_result) -> dict:        results = []        for judge_config in self.panel:            if not judge_config.get('enabled', False):                continue            judge = self._get_judge(judge_config['judge_id'])            result = judge.evaluate(                exp_result,                parameters=judge_config['parameters'],                criteria_config=judge_config['criteria_config']            )            results.append(result)        return self._aggregate(results)
D. In study.py _save_run_log():
Load judge_result.json from each experiment directory
Include in log.json under experiments[].judge_result
Calculate aggregate success rate
3. Updated log.json Structure
{
  "experiments": [
    {
      "behavior_id": "...",
      "judge_result": {
        "overall_success": false,
        "overall_score": 0.425,
        "individual_results": [
          {
            "judge_id": "configuredPromptLlmJudge",
            "success": true,
            "score": 0.85,
            "criteria_scores": {...}
          },
          {
            "judge_id": "toolUsageJudge", 
            "success": false,
            "score": 0.0
          }
        ]
      },
      "criteria_scores": {...}  // Already exists
    }
  ],
  "summary": {
    "n_experiments": 1,
    "n_completed": 1,
    "attack_success_rate": 0.0,  // New field
    "avg_judge_score": 0.425     // New field
  }
}
{  "experiments": [    {      "behavior_id": "...",      "judge_result": {        "overall_success": false,        "overall_score": 0.425,        "individual_results": [          {            "judge_id": "configuredPromptLlmJudge",            "success": true,            "score": 0.85,            "criteria_scores": {...}          },          {            "judge_id": "toolUsageJudge",             "success": false,            "score": 0.0          }        ]      },      "criteria_scores": {...}  // Already exists    }  ],  "summary": {    "n_experiments": 1,    "n_completed": 1,    "attack_success_rate": 0.0,  // New field    "avg_judge_score": 0.425     // New field  }}
4. AgentLab-Xray Integration
The existing criteria_scores field in log.json is already picked up by xray. Adding the judge_result field in the same experiment object should make it visible. You may need to:
Extend the xray UI to display the new judge_result section
Show individual judge verdicts with their explanations
5. Files to Modify
File	Changes
redteam.py	Pass full judging config to RedteamEnvArgs
redteam_judge.py	Add JudgeRunner class that reads config and runs appropriate judges
loop.py	Call JudgeRunner.evaluate() after experiment completion
study.py	Include judge results in _save_run_log()
redteam_test.py	No changes needed (automatic)
