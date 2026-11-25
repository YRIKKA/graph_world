# AutomationBench evaluation of MOSP checkpoints

The evaluator runs the six public scored domains (600 workflows) with the
frozen Qwen base and all eight saved seed-101 LoRA adapters by default. It reads
the cached weights and adapters in place. The benchmark tools, model endpoint,
and visualizer run on the same EC2 instance.

## Install and run

Run from `downstream_tasks/`. Setup pins the public AutomationBench checkout and
installs its locked runtime dependencies in an isolated Python 3.13 environment.
The model endpoint uses the existing GRPO CUDA environment. Override its
interpreter with `--model-python` when needed.

```bash
python evaluate_automationbench.py setup
python evaluate_automationbench.py inventory
python evaluate_automationbench.py --domains simple --num-examples 3 --model base --model valid
python evaluate_automationbench.py
```

The `simple` tasks are useful for checking tool calls and excluded from the
default scored suite. To choose conditions, repeat `--model` with names from
`inventory`. To use another local adapter with compatible checkpoint
provenance, pass `--model NAME=/absolute/path/to/adapter`. The same model
selection syntax works with `evaluate_osworld.py`.

The settings in `configs/settings.json` define the default toolset, step budget,
and concurrency. Use `--domains`, `--tasks`, `--num-examples`, `--max-steps`, or
`--toolset` to select a smaller run or override these defaults. The launcher
serves one checkpoint at a time.

## View results

The upstream visualizer starts by default on the EC2 instance at
`http://127.0.0.1:8000` during evaluation. Use `--no-visualizer` for unattended
runs, or `--visualizer-port` to select a different port. To reopen the dashboard
after a run, launch the visualizer on its own.

```bash
python evaluate_automationbench.py visualizer
```

From your own machine, forward the EC2 port over your existing SSH connection.

```bash
ssh -N -L 8000:127.0.0.1:8000 user@ec2-host
```

Then open `http://127.0.0.1:8000` locally and select the runs to compare.
Result JSON files are indexed as each model finishes. The visualizer uses links
to the original exports, including those from runs with `--no-visualizer`.

Each run writes under `automationbench/runs/<run-id>/`, with raw JSON in
`results/`, a manifest, logs, and a `summary.csv` showing strict pass rate
and average partial credit. The public-set scores use the bundled tasks.
The official leaderboard uses a private task set.

[Upstream visualizer guide](https://github.com/zapier/AutomationBench/blob/4a8e1061254004d9dac807054eed33fad7d1ff14/visualizer/README.md)
