# MOSP smoke checks

Run these commands from the `graph_world` repository root using the relevant
Python 3.12 environment.

| Check | What it exercises | Dependencies |
| --- | --- | --- |
| CPU evaluator | Generates three graphs and verifies the full-information reference | [requirements-cpu.txt](../requirements-cpu.txt) |
| CPU GRPO training | Runs real LoRA and optimizer updates using a tiny randomly initialized Qwen2 model | PyTorch, Transformers, and PEFT from the [GRPO environment](../requirements.txt) |
| GPU NF4 integration | Checks frozen likelihoods, a GRPO update, and saved adapter reload with the pinned Qwen checkpoint | [NF4 setup](../docs/NF4_SETUP.md) |

```bash
python -m grpo_mosp.smoke
python -m grpo_mosp.smoke.smoke_train
python -m grpo_mosp.smoke.smoke_nf4
```

The CPU checks print JSON summaries. The GPU check reads [smoke.json](smoke.json)
and writes logs, caches, and `smoke_nf4.json` under a new timestamped run in
`full_runs/`. Use `--output` to specify a run directory.
