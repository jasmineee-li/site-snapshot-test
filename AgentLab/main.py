"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging
import os

# Optimize WebLinx registration - only register the splits we need
# This prevents each Ray worker from registering 26,410+ tasks (60+ seconds)
# Comment these out if you need tasks from other splits
os.environ["BROWSERGYM_WEBLINX_REGISTER_TRAIN"] = "true"   # cptbbef is in train split
os.environ["BROWSERGYM_WEBLINX_REGISTER_VALID"] = "false"  # Skip validation tasks
os.environ["BROWSERGYM_WEBLINX_REGISTER_TEST"] = "false"   # Skip test tasks
os.environ["BROWSERGYM_WEBLINX_REGISTER_TEST_OOD"] = "false"  # Skip OOD test tasks

from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
    AGENT_o3_MINI,
    AGENT_37_SONNET,
    AGENT_CLAUDE_SONNET_35,
    AGENT_GPT5_MINI,
)
from agentlab.experiments.study import Study
from bgym import DEFAULT_BENCHMARKS

logging.getLogger().setLevel(logging.INFO)

# choose your agent or provide a new agent
agent_args = [AGENT_GPT5_MINI]
# agent_args = [AGENT_4o]


# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
# benchmark = "workarena_l1"
# benchmark = "workarena_l2"
# benchmark = "workarena_l3"
# benchmark = "webarena"
# benchmark = "weblinx"  # full benchmark with 31,586 tasks

# Create a subset of WebLinx with just 10 tasks for testing
full_weblinx = DEFAULT_BENCHMARKS["weblinx"]()
task_names = [env_args.task_name for env_args in full_weblinx.env_args_list[:10]]
benchmark = full_weblinx.subset_from_list(task_names, benchmark_name_suffix="first_10")

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 4  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores


if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING)

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
