# GRPO for hidden multiobjective paths

Use the `graph_world` repository root and the GRPO Python 3.12 environment for the
commands below. See the [NF4 setup guide](docs/NF4_SETUP.md) for the CUDA
environment and [MOSP guide](docs/MOSP_GUIDE.md) for the task, model, and training
protocol.

| Experiment | Entry point | Guide |
| --- | --- | --- |
| MOSP training and paired evaluation | [run_mosp.sh](run_mosp.sh) | [Launcher and run artifacts](docs/RUN_MOSP.md) |
| Discovery-reward shaping | [experiment_shaping.sh](experiment_shaping/experiment_shaping.sh) | [Shaping](experiment_shaping/SHAPING.md) |
| MOO terminal rewards | [experiment_moo_rewards.sh](experiment_moo_rewards/experiment_moo_rewards.sh) | [Post-hoc scoring script](experiment_moo_rewards/combined_metrics_post_hoc_eval.py) |
| Before/after baseline comparison | [eval_baselines.py](eval_baselines.py) | [Protocol and presets](experiment_eval_baselines/EVAL_BASELINES.md) |
| Downstream benchmarks | [evaluate_osworld.py](experiment_downstream_tasks/evaluate_osworld.py), [evaluate_automationbench.py](experiment_downstream_tasks/evaluate_automationbench.py) | [OSWorld](experiment_downstream_tasks/osworld/README.md), [AutomationBench](experiment_downstream_tasks/automationbench/README.md) |

Preview the training and ablation launchers with `--dry-run`.

```bash
mosp_dir=grpo_mosp
"$mosp_dir/run_mosp.sh" --dry-run
"$mosp_dir/experiment_shaping/experiment_shaping.sh" --dry-run
"$mosp_dir/experiment_moo_rewards/experiment_moo_rewards.sh" --dry-run
python -m grpo_mosp.eval_baselines --help
```

Use the [smoke guide](smoke/README.md) to choose between CPU evaluator, CPU
training, and real-checkpoint GPU checks. The experiment-specific tests run
alongside the shared test suite.

```bash
python -m pytest -q grpo_mosp/{tests,experiment_shaping,experiment_moo_rewards}
```
