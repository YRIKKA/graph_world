# One launcher for GRPO training and before/after evaluation

`run_mosp.sh` orchestrates dataset generation, independent Qwen GRPO training,
validation of the resulting checkpoints, and the existing `eval_baselines.py`
comparison. It can be invoked from any working directory. New runs always create
`grpo_mosp/full_runs/run_YYYYMMDDHHMMSS/`, using an America/New_York
timestamp with whole-second precision. For example:
`full_runs/run_20261103123015/`. Each model has `<model>_trained/` and
`<model>_not_trained/` subdirectories, using the checkpoint basename.
`--runs-root /path/to/full_runs` selects a different parent named `full_runs`.
`--output` selects an exact correctly named run directory, normally for reuse.
Relative `--output` and `--runs-root` paths are resolved from the repository root,
including when the shell script is invoked from its own directory.

Run the examples from the `graph_world` repository root. Use a Python 3.12 environment with CUDA-enabled PyTorch for training/evaluation:

```bash
pip install -r grpo_mosp/requirements-eval.txt
grpo_mosp/run_mosp.sh --python /path/to/venv/bin/python --help
```

The shell launcher initializes Conda and activates `grpo`, then uses `python3.12` by default. `--python` or `MOSP_PYTHON` selects a
different interpreter. It does not install dependencies or alter Isaac Sim's
Python. Downloading the pinned Qwen weights happens when training/evaluation
loads them, using `cache/huggingface/` inside this run. Temporary files, compiler
caches, plotting caches and stage logs are also contained inside the same run.

## Run a training and comparison experiment

```bash
grpo_mosp/run_mosp.sh \
  --python /path/to/venv/bin/python \
  --runs-root grpo_mosp/full_runs \
  --mode all \
  --train-graphs 64 --validation-graphs 32 \
  --subgraphs 2 --nodes-per-subgraph 6 12 --outdegree 2 \
  --curriculum 1,2,3 --attempt-multiplier 8 \
  --training-runs 5 --steps 30 --group-size 8 \
  --eval-graphs 200 --eval-repetitions 5
```

This runs the stages sequentially on one GPU. The default checkpoint is
unsloth/Qwen3.8-27B-unsloth-bnb-4bit pinned to
`8aa5f05d26b7205477066e1449e0af13f762a299`; changing `--model` also requires an
explicit immutable `--revision` for that model. Training uses NF4 QLoRA with BF16 compute; see [NF4_SETUP.md](NF4_SETUP.md).

The defaults are five independent training seeds, 32 training graphs, 32
validation graphs, and 200 held-out graph families. Evaluation uses k=1,2,3,
B/k=1,2,4,8,16, five repetitions, and both independent exploration and
shared-transcript selection. This full configuration is a substantial experiment.
Use `--dry-run` to inspect all settings, training configs, the shared evaluation
config, and exact commands without generating data, downloading weights or
starting training.

The two models being compared always use **the exact same saved held-out graph
instances**, k values, budgets, repetition counts and evaluation controls. Each
model explores independently. The selection-only diagnostic supplies identical
externally collected transcripts and archives to every model. Checkpoint
identities and suite hashes are checked before completed results can be reused.

New run timestamps use whole seconds (`run_YYYYMMDDHHMMSS`). Existing runs
with microseconds remain readable and are not renamed. If a directory already
exists for the current second, creation fails with a timestamp-collision error;
retry in the next second or use that existing run explicitly for a later stage.

## Smaller GPU plumbing check

```bash
grpo_mosp/run_mosp.sh \
  --python /path/to/venv/bin/python \
  --runs-root grpo_mosp/full_runs \
  --training-runs 1 --steps 1 --group-size 2 \
  --train-graphs 4 --validation-graphs 2 --eval-graphs 10 \
  --curriculum 1 --attempts-per-graph 2 \
  --eval-budget-multipliers 8 --eval-repetitions 1 \
  --bootstrap-samples 50 --context-limit 4096
```

This still downloads/loads the actual 27B NF4 model. One optimizer group and one
training run test the plumbing; they do not establish a reliable training
procedure improvement. Groups with constant reward have no reward-driven GRPO
advantage. The full scientific design remains five runs and five repetitions.
Custom evaluation budget lists must include the prespecified primary B=8k.

## Separate stages and reuse

- `--mode prepare`: generate CPU-only training/validation data and all configs.
- `--mode train`: prepare data, then train every configured independent run.
- `--mode eval`: load a saved run's settings and compare its trained adapters with
  the exact pinned initialization; no training occurs.
- `--mode all`: prepare, train, select using validation, prepare the held-out suite,
  evaluate all checkpoints and produce tables/plots.

For example, first train with custom parameters:

```bash
grpo_mosp/run_mosp.sh --mode train \
  --runs-root /path/to/full_runs --attempts-per-graph 24 --steps 100
```

The launcher prints the newly created run directory. Evaluate that same directory
without repeating those settings (replace the example timestamp below):

```bash
grpo_mosp/run_mosp.sh --mode eval \
  --output /path/to/full_runs/run_20261103123015
```

The second command reads `<model>_trained/launch_config.json`. Data and per-model parameters are immutable
within that output directory; incompatible CLI overrides require a fresh output
directory. Complete matching training checkpoints can be reused by `all`/`train`.
Incomplete training or evaluation is not resumed silently. The underlying trainer
and harness have no optimizer/task resume CLI; use a fresh directory after a
partial run. `--no-plots` skips performance plots only; graph PNGs with visible
edge costs are always exported.

## Main controls and meanings

| Flags | Meaning |
|---|---|
| `--runs-root PATH` | Parent directory named `full_runs`; each new run gets its own model/timestamp child. |
| `--output PATH` | Exact `full_runs/run_<timestamp>` directory for selecting or adding models and later stages. |
| `--train-graphs N`, `--graph-replicates N` | N distinct fixed training graph instances, shared across training runs. Groups sample from this dataset; this is not the GRPO group size. |
| `--validation-graphs N` | Separate graphs for validation-only checkpoint selection. |
| `--eval-graphs N` | Held-out graph families, balanced over five domains and two difficulty strata; multiple of 10. |
| `--subgraphs N` or `--subgraphs MIN MAX` | Inclusive number of subgraphs per graph; minimum 2 for this benchmark's non-target sinks. |
| `--nodes-per-subgraph MIN MAX` | Inclusive node count per subgraph. Defaults to 6..12; two subgraphs yield 12..24 total nodes. |
| `--outdegree N` or `--outdegree MIN MAX` | Inclusive internal maximum-outdegree setting, sampled independently for each subgraph. Connectors can add one outgoing edge. Require 2 <= MIN <= MAX < minimum nodes per subgraph. |
| `--objectives D` | Number of additive objectives: water, time, energy, then risk/objective_N as needed. Default 3. |
| `--max-paths N` | Guard on exact source-target path enumeration before materialization. |
| `--max-candidates N` | Rejection-sampling cap per data split/evaluation domain. Narrow ranges may not contain enough accepted graphs in every stratum. |
| `--curriculum 1,2,3` | Requested-path k values over training stages; one value fixes k. |
| `--attempts-per-graph B` | Fixed number of traversal attempts **per complete training task on a fixed graph**; B must be at least every curriculum k. Alias: `--traversals-per-graph`. |
| `--attempt-multiplier Q` | Alternative to fixed B: training B=Qk; default 8. |
| `--group-size M` | Independent complete task rollouts on the same graph per GRPO group; minimum 2. Every rollout gets its own B attempts and private memory. |
| `--steps N`, `--epochs-per-group N` | Maximum GRPO rollout groups per training run (default 30), and optimizer updates reusing each fixed group (default 1). |
| `--early-stopping true\|false` | Validation plateau stopping and best-adapter export; default true. Also accepts `--no-early-stopping`. |
| `--training-runs N` | Independent trained checkpoints. Default seeds are 101,202,...; `--training-seeds 17,29` supplies an explicit list. |
| `--backend auto`, `--compute-dtype bfloat16` | Auto selects Unsloth NF4 for the default checkpoint. The resolved backend/dtype is identical in both evaluation arms. |
| `--lora-last-layers 8`, `--lora-alpha 16`, `--lora-mlp` | Restrict training to the last N language layers (0 means all); MLP adapters are off by default. |
| `--offload-embedding`, `--no-offload-embedding` | Host embedding offload, enabled by default and applied identically during training/evaluation. |
| `--max-gpu-memory-gib 38`, `--min-free-gpu-gib 24` | CUDA allocator ceiling and pre-load free-memory check; a ceiling of 0 disables the cap. These are controls, not a fit guarantee. |
| `--learning-rate`, `--beta`, `--clip`, `--max-grad-norm`, `--lora-rank` | Optimizer, reference-KL, clipping and adapter settings. |
| `--objective pareto` or `hypervolume` | Training terminal objective. Evaluation always reports Pareto slot score S and hypervolume separately. |
| `--history-format state_actions` | Complete current memory plus every move, with exact reconstruction of all earlier observations. `delta` uses reversible patches; `full` repeats tables. Shared by training and both evaluation arms. |
| `--temperature`, `--context-limit` | Sampling temperature and context limit. Evaluation uses the same values for initialization and post-trained models. Histories/memory are never silently truncated. |
| `--max-action-tokens`, `--max-task-tokens` | Evaluation generation limits, applied identically to both models; include forced JSON tokens. They do not impose extra caps on training rollouts. |
| `--eval-ks 1,2,3` | Evaluation k strata; defaults to distinct training curriculum values. |
| `--eval-budget-multipliers 1,2,4,8,16` | Both models receive identical evaluation B/k values; must include primary 8. These are independent of the training budget setting. |
| `--eval-repetitions N`, `--eval-relabelings N` | Repeated task sampling and identifier permutations; clustered by graph family during uncertainty estimation. |
| `--large-nodes-per-subgraph MIN MAX` | Size-shift bounds; defaults to training MAX+1 through MAX+4. |
| `--density-outdegree N` or `MIN MAX` | Density-shift bounds; defaults to base outdegree bounds+1. |
| `--difficulty-cutoff N` | Prespecified easy/hard split by exact path count. Must be below max-paths. |
| `--bootstrap-samples N` | Paired graph/bootstrap draws and crossed run/graph draws. |
| `--data-seed`, `--training-seed`, `--seed-stride`, `--suite-seed`, `--eval-seed`, `--bootstrap-seed` | Separate random-seed controls for data, training and evaluation. |

Edge costs retain the graph generator's default independent integer 0..5
sampling. No transition probabilities are assigned. Endpoints and Pareto truth
come from the graph constructor. The source/target remain fixed within every
training/evaluation task and throughout its traversal attempts.

The evaluation ID domain uses **the same configured graph distribution as
training**, including subgraph-count ranges, node ranges, outdegree ranges and
objective count. The four shifts change size, density, costs or objective count
relative to that base configuration. Training/validation/test exclusion detects
weighted rooted graph isomorphism, so relabeling a training graph cannot turn it
into a held-out test instance.

## Outputs and scientific use

`<model>_trained/launch_config.json` records that model's resolved controls. `data/` contains fixed disjoint
training/validation graphs, source hashes, rejection statistics and instance IDs.
`<model>_trained/configs/` contains its generated configurations, and
`<model>_trained/runs/` contains adapters, optimizer states, training trajectories,
validation evidence and training compute. Raw post-trained evaluation results and
per-checkpoint metric tables go in `<model>_trained/evaluation/`; the frozen
initialization's results go in `<model>_not_trained/evaluation/`. `suite/` holds
one shared held-out suite, while `comparisons/<model>/` contains paired
before/after reports, joint audit records and performance-versus-budget plots.

The convenience `all` workflow uses the declared, prespecified sample sizes and
compares each run's final checkpoint after validation scoring. For the separate
pilot/sample-size refinement workflow or validation selection over multiple
candidate checkpoints/hyperparameters, use the stages documented in
[EVAL_BASELINES.md](../experiment_eval_baselines/EVAL_BASELINES.md). This launcher does not substitute test
scores for validation or select favorable evaluation repetitions.

## Mandatory graph images and artifact layout

Every saved training, validation and held-out graph is exported with
`GraphGenerator.plot_graph` from `generate_graph_world.py`.
The renderer receives the saved topology and exact ordered edge-cost names,
with edge-cost display enabled and transition-probability display disabled.
Source and target have distinct node colors. All objective weights and their
legend are visible. Relabeled test presentations and cost-shifted instances each
get their own matching PNG. Images remain evaluator artifacts and are never
supplied to the hidden-graph policy.

```text
full_runs/run_<timestamp>/
  run_manifest.json
  data/
    train.jsonl
    validation.jsonl
    manifest.json
    images/train/<instance-id>.png
    images/validation/<instance-id>.png
  suite/
    suite.jsonl
    transcripts.jsonl
    images/<domain>/<graph-id>.png
  <model>_trained/
    launch_config.json
    configs/
    runs/seed-<seed>/adapter/
    runs/seed-<seed>/training_compute.json
    evaluation/grpo-<seed>/tasks.jsonl
    evaluation/grpo-<seed>/metrics.csv
    logs/
  <model>_not_trained/
    evaluation/initial/tasks.jsonl
    evaluation/initial/metrics.csv
  comparisons/<model>/
    suite.jsonl -> ../../suite/suite.jsonl
    tasks.jsonl
    comparison.csv
    comparison.md
    logs/
  figures/<model>/
    training/grpo/seed-<seed>/trajectory/       # Five metric curves, PNG and SVG
    evaluation/budget_curves/                  # Performance versus budget
    evaluation/<mode>/checkpoint_comparison/   # Five metrics, all checkpoints
  cache/
  tmp/
```

Training figures contain rollout-group means for reward, correct Pareto
fraction, complete success, valid-path fraction and duplicate rate. Their x axis
counts optimizer updates completed before the rollouts were collected; skipped
updates do not advance it. Each seed has separate curves, CSV data and a source
manifest. Evaluation bars include every checkpoint and saved marginal 95%
intervals, with separate domain panels for both experiment modes. `--no-plots`
disables metric figures. To export them from saved results without inference:

```bash
python -m grpo_mosp.figures \
  --run /path/to/full_runs/run_YYYYMMDDHHMMSS
```

Training graph coverage now defaults to `--shuffle-graphs true`: each pool is
sampled without replacement before its next coverage epoch. Set it to false for
replacement sampling. `--balance-graphs true` interleaves measured difficulty and
trade-off strata, while `--adaptive-graph-sampling true` prioritizes graphs whose
recent rollout groups provide reward variation. Both optional experiments
default to false and affect training only. See [SHAPING.md](../experiment_shaping/SHAPING.md) for the
exact sampling rules, audit artifacts and five independently selectable additive
reward bonuses, including full or restricted factorial sweeps.

Data/suite manifests record image paths/hashes. Reuse verifies that graph images
and saved data remain intact. The launcher captures each child process's output
both live on the terminal and in its stage log, plus the combined launcher log.
The run manifest records all redirected cache/temp locations. Existing older
`experiments/` or `runs/` artifacts are not moved or overwritten automatically.

## Multiple models in one run

Use the printed run directory as `--output` when adding another model. Supply the
new model's immutable revision and its training controls as needed:

```bash
grpo_mosp/run_mosp.sh --mode all \
  --output /path/to/full_runs/run_20261103123015 \
  --model Qwen/ANOTHER_CHECKPOINT --revision FULL_40_CHARACTER_COMMIT
```

The new model gets its own `_trained` and `_not_trained` directories and inherits
the existing run's graph/evaluation design. Existing models' settings and
checkpoints remain separate. The shared training/validation data must match,
and every model's comparison links to the exact same immutable `suite/` graph
instances and external transcripts. Incompatible data designs are rejected.
Model basenames must be distinct within a run.

When more than one model exists, include `--model` to select it for `eval`,
`train`, or `prepare`. The model's training settings live in its own
`_trained/launch_config.json`, and `run_manifest.json` lists all registered models.
Caches and temporary files remain shared within the timestamped run directory.

## Context limits and first-run failures

A context overflow counts input history plus the longest legal next response, not
just generated response tokens. Repeating an expanding memory table after every
move can exceed 32768 tokens even when the model has a larger native context.
`state_actions` is now the default; it preserves the complete observable state and
all prior moves without repeatedly serializing old tables. The encoder checks
that every earlier observation can be recovered exactly. Original raw messages
remain in saved rollouts. Both evaluation arms use the same format.

If any task still exceeds its token cap, training/evaluation retain it with reward
0 and record `context_limit`. Training logs its frequency and rejected token
count instead of aborting the whole campaign. Lower graph size/density/traversal
budgets or profile a larger `--context-limit` if that frequency persists.

Start a **fresh run** after changing protocol settings. An incomplete run has no
resume implementation, and immutable existing evaluation artifacts must not be
reused under new history rules. Omitting `--output` creates a fresh timestamped
run in the correct default `full_runs` directory.

## Console metrics, progress and log.txt

Every actual launcher invocation appends its console output to **`<run>/log.txt`**.
This master log includes stdout, stderr, warnings, progress bars and native-library
output from all training/evaluation child processes, as well as launcher failures.
Existing stage logs remain available. Help and `--dry-run` keep their no-write
behavior. Adding another model or running a later stage appends to the same run
log; model and seed/checkpoint labels identify each report.

Training prints a readable table per GRPO iteration: the iteration's and the
run's cumulative terminal reward, mean reward, Pareto fraction, complete success,
validity, duplicates, protocol/resource failures, hypervolume, discovery A,
selection gap L, generated tokens and edge traversals. Separate rows show GRPO
loss, zero-variance frequency, context size and elapsed time. The same metrics
are printed at run completion and across completed independent seeds for that
model. Across-run rates give each seed equal weight; reward/task/token/traversal
totals sum the actual records. Training rewards follow `--objective` and describe
rollouts collected while weights change.

Evaluation reports each frozen/adapted checkpoint by mode and traversal budget,
with separate primary-B=8k domain tables. Evaluation reward is the Pareto score S.
Final comparison tables print in-distribution and distribution-shift results
separately for each run and across runs, preserving the existing graph/stratum
weights and paired/crossed 95% bootstrap intervals. Raw reward totals are labeled
separately from weighted rates. Independent exploration and shared-transcript
selection are never blended. Full domain/k/difficulty/budget results remain in
the existing CSV/Markdown reports.

Progress bars count training iterations per seed, completed training runs per
model, validation/evaluation tasks per checkpoint, completed evaluated checkpoints,
and aggregation estimates. Bars use plain text lines so `log.txt` remains legible
without terminal escape codes. Failed work does not advance the completion count.

Additional files are:

- `<model>_trained/runs/seed-<seed>/log.txt` and `training_summary.json` per training run.
- `<model>_trained/training_summary.json` for completed runs across that model.
- Each frozen/adapted evaluation directory's `log.txt` and `console_summary.json`.
- `comparisons/<model>/log.txt` for the combined evaluation stages.

Workers already running retain their loaded reporting code. To print a fresh
snapshot of saved training metrics and progress without restarting or loading a
model, use:

```bash
conda activate grpo
python -m grpo_mosp.reporting \
  --output /path/to/full_runs/run_YYYYMMDDHHMMSS
```

This also appends the snapshot to the run's `log.txt`. It reads completed metric
records only, ignores a running writer's unfinished last line, and never modifies
weights, training records or configurations. Older records show **N/A** for
metrics they did not save. Completed-run aggregates exclude partial runs.

## Qwen reasoning template controls

All model-facing entry points (`run_mosp.sh`/`run_mosp.py`, `train.py`,
`eval_baselines.py`, `evaluate.py`, and `smoke_nf4.py`) accept:

- `--reasoning-effort {xhigh,medium,low}`: defaults to `xhigh`.
- `--preserve-thinking true|false`: defaults to `false`. Bare
  `--preserve-thinking` enables it; `--no-preserve-thinking` disables it.

The underscore spellings `--reasoning_effort` and `--preserve_thinking` also work.
CLI values override JSON fields with the corresponding underscore names; omitted
flags retain saved settings. Missing JSON fields use the defaults above. For example:

```bash
grpo_mosp/run_mosp.sh --dry-run \
  --reasoning-effort medium --preserve-thinking false
```

The launcher records both fields in every training and evaluation configuration.
Frozen and adapted policies receive the same values from the shared evaluation
configuration, including validation and shared-transcript diagnostics. Per-checkpoint
evaluation manifests record them as controls, and experiment hashes include them.
Existing immutable runs cannot be changed to different controls; create a new run.

These settings are passed to the pinned Qwen `apply_chat_template`, alongside the
existing `enable_thinking=False`. In this constrained-action pipeline there is no
free-form reasoning generation phase: the template ignores reasoning effort when
thinking is disabled. `preserve_thinking` controls template retention of earlier
assistant reasoning blocks; it does not create reasoning text or change graph
memory rules. This update does not enable free-form thinking or change the JSON
action grammar. Neither flag requires changes to the frozen NF4 checkpoint or
adapter loading provenance.


## Validation plateau stopping

Training defaults to `--steps 30 --epochs-per-group 1 --early-stopping true`.
The launcher automatically supplies its separate validation dataset. Direct
`train.py` invocations accept these same flags and `--validation-dataset PATH`.
Missing validation data or overlap with training graphs (including relabeled
copies) fails before model loading. Test graphs never enter the stopping rule.

The default monitor evaluates a fixed, deterministically selected subset of up to
8 validation graphs, every distinct evaluation k, one repetition, and B=8k. It
checks before training, every 5 completed groups, and at the maximum step count.
Scores are mean Pareto S with equal graph/k/repetition weights. Tasks use fixed
sampling seeds and fresh memory; validation restores model mode and all training
RNG states afterward. Validation uses the same constrained decoder, history,
reasoning settings, temperature, context limit and evaluation token limits.

After 3 checks without improvement, training stops once it has reached the final
curriculum stage. This guard gives increasing-k curricula a chance to train the
last stage. With a constant k the guard is immediately satisfied. Ties keep the
earlier checkpoint. The actual highest-scoring adapter is exported to `adapter/`
as improvements occur, together with its matching optimizer/RNG snapshot. The
initial adapter can win if training never improves validation. The best exported
adapter is retained even when all 30 steps complete; it is the adapter used by
subsequent baseline comparisons.

Controls (recorded in every training config):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--early-stopping-interval` | 5 | Groups between validation checks. |
| `--early-stopping-patience` | 3 | Consecutive checks without significant improvement. |
| `--early-stopping-min-delta` | 0 | Absolute S increase needed to reset patience. Smaller gains can still export the best adapter. |
| `--early-stopping-graphs` | 8 | Maximum fixed validation subset size. |
| `--early-stopping-repetitions` | 1 | Repetitions per validation graph and k. |
| `--early-stopping-seed` | 314159 | Fixed validation sampling seed shared across checks and training runs. |

`validation_metrics.jsonl` records each check and per-task metrics;
`early_stopping.json` records dataset/graph identities, selected iteration, best
score and stopping reason. Reports and training compute use the actual completed
iterations; validation tokens, traversals and time are recorded separately from
training rollouts, within training compute. Validation adds inference overhead;
use the interval/subset controls to adjust its cost, selecting settings on pilot
or validation data. Subsequent `select-validation` still evaluates the exported
adapter on the declared validation data before the frozen/adapted test comparison.

Use `--early-stopping false` to train all requested groups and export the final
adapter. Existing saved runs are not rewritten. `smoke/smoke.json` explicitly
uses one step and disables early stopping for its single-update plumbing check.


## Trained versus frozen evaluation headline

The `summarize` stage prints prominent `TRAINED vs FROZEN BASE` tables for
primary B=8k independent exploration and shared-transcript selection. It shows the frozen and GRPO-trained Pareto
scores, the trained-minus-frozen percentage-point change, its paired 95% interval,
and whether that interval is above, below, or includes zero. In-distribution,
shift macro averages, and each shift domain have separate rows. Trained scores
average independent training runs with the prespecified graph/stratum weights.
The detailed tables immediately before it retain each run and secondary metrics.
If supplied, the pre-GRPO checkpoint gets its own additional comparison headline.

Both the detailed tables and the headline appear live on the console and in the
run's root `log.txt`. The comparison-specific copy is
`comparisons/<model>/log.txt`; `comparisons/<model>/logs/summarize.log` captures the
summary stage. `comparison.csv` and `comparison.md` remain the exported scientific
tables. No inference or retraining is needed to print a completed evaluation
again: invoke `eval_baselines summarize` with that evaluation's exact saved config
and output directory (and optionally `--no-plots`). Direct module invocation
appends to the comparison output's `log.txt`; the root log additionally receives
it when invoked through `run_mosp.sh`.

## Component and total pipeline timing

Every real launcher invocation records start/end elapsed times for dependency
checks, training/validation data and weighted images, config export, each training
seed, validation selection, held-out suite/transcript preparation, held-out
checkpoint evaluation, and comparison aggregation/plots. A component-time table
and `TOTAL PIPELINE ELAPSED TIME` close the root `log.txt`; the total is its final
line for that invocation. Through `run_mosp.sh`, the total includes shell/Conda
startup. Reused stages report verification time with `reused` status. Failed and
interrupted invocations also report elapsed time and their status.

`pipeline_timings.jsonl` stores one append-only record per invocation, with model,
mode, timestamps, stage status and durations. Each invocation is measured
separately: a later `--mode eval` does not include a previous `--mode train` or idle
time between launches. `--mode all` measures the entire pipeline in one launch.
Dry runs/help remain free of filesystem writes.

Nested log timings include training model loading, per-iteration rollout and
optimization seconds, early-stopping validation, adapter exports, held-out graph
and external transcript generation, each validation candidate, each evaluated
checkpoint's model load/warmup and both evaluation modes, bootstrap aggregation,
and plot export. Parent-stage times include their children; they should not be
added together. Evaluation mode wall times include environment setup/scoring and
task-result writing, while existing per-task latency fields retain their original
inference scope.

The final frozen-versus-trained headline now appears for **both** independent
exploration and shared-transcript selection, each using its own paired scores and
95% confidence intervals. The latter is explicitly labeled as a selection-only
diagnostic with identical external transcripts and archives. Both tables are
printed to the console and the corresponding `log.txt` files.
