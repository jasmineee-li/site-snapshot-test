### Launch tutorial experiment
* Activate the python env `source .venv/bin/activate`
* Run experiments by editing and running `main.py` from the **AgentLab** directory (see `main.py` for configuration examples).
* This should launch your agent on miniwob tasks and save results to `$HOME/agentlab_results`.


### Visualize experiments in XRay
* run `agentlab-xray`
* select experiment directory
* Navigate around XRay to visualize the agent's behavior

### (optional) Load your results in a notebook
Optionally load the results in [inspect_results.ipynb](./inspect_results.ipynb).
This will help understand the content of results

### Change the experiment
go back to `main.py` and modify the benchmark configuration, e.g.
```python
benchmark = DEFAULT_BENCHMARKS["miniwob"]()  # 125 tasks
benchmark = benchmark.subset_from_glob(column="task_name", glob="*enter*") # filter only 7 tasks
```
This will change the benchmark from `"miniwob_tiny_test"` to all the `"*enter*"` tasks in the full miniwob.

Run the experiment again. This will launch on 7 tasks with 5 seeds each (35 experiments)

Finally refresh XRay to be able to load the new experiment.

### Exercise 1
What are the failures and why?