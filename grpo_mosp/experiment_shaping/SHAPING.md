# Shaping ablation and lightweight evaluation

## Combined seven-agent experiment

`experiment_shaping.py` and `experiment_shaping.sh` run exactly `none`, `edge`,
`path`, `hypervolume`, `cost_diversity`, `path_disjointness`, and `all` in one
shared experiment. `all` enables all five shaping signals. Each condition starts
from the same frozen base with fresh adapters. Training/validation graphs and
held-out graphs/transcripts are shared. The frozen base is evaluated once per
configured task cell, with its results reused for every comparison.

```bash
# Inspect the complete seven-agent plan without writing files or loading models.
jm_test/graph_world/grpo_mosp/experiment_shaping/experiment_shaping.sh --dry-run

# Train all seven and export one combined report against the frozen base.
jm_test/graph_world/grpo_mosp/experiment_shaping/experiment_shaping.sh

# Override the profile for a shorter screen.
jm_test/graph_world/grpo_mosp/experiment_shaping/experiment_shaping.sh --steps 12 --group-size 2

# Evaluate/reuse an existing seven-condition run with its saved settings.
jm_test/graph_world/grpo_mosp/experiment_shaping/experiment_shaping.sh --mode eval \
  --output /path/to/full_runs/run_YYYYMMDDHHMMSS
```

The default profile matches the latest screening setup: 32 groups per agent,
group size 4, one optimizer epoch per group, fixed training B=16, k curriculum
1,2, all three sampling switches true, eight training graphs, two validation
graphs, one training seed (101), no early stopping, ten held-out graphs, eval k=2
at B=16, one evaluation repetition, and both independent exploration and shared
transcript selection. Other defaults are difficulty cutoff 8, 200 bootstrap
draws, reasoning effort low, thinking retention false and 4096 task tokens.
The default graph curriculum is the persistent
`graph_curriculum.json`. Graph stages use the parameters in that file;
`--curriculum-graph` can select a different configuration. Existing flags override these settings;
later invocations load saved overrides rather than resetting the profile.

This produces 896 training rollouts and 160 held-out evaluation tasks across
eight checkpoints. It is a screening experiment, not replicated evidence of
training-procedure superiority. Adaptive sampling can give different agents
different within-pool graph orders, based on their own reward history.

All outputs share one `full_runs/run_YYYYMMDDHHMMSS/`. Console tables and `log.txt`
compare all seven conditions to the frozen base and also to unshaped GRPO.
`comparisons/<model>/shaping/comparison.csv`, `comparison.md` and `summary.json`
contain the combined results, separated by domain and evaluation mode. Training
curves and eight-checkpoint bar charts are under `figures/<model>/`.

The generic launcher now also accepts an arbitrary explicit list:

```bash
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh \
  --shaping-combinations none edge path hypervolume cost_diversity path_disjointness all
```

This list selector is exclusive with other shaping selectors; semantically
duplicate combinations are rejected. The experiment launcher fixes the seven
conditions above while allowing reward weights and training/evaluation controls
to be changed.

## Independent sampling experiments and five reward signals

The launcher supports three independently controlled training samplers:

```bash
# Replacement-sampling control, then coverage, balancing, and adaptation.
run=jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh
$run --shaping none --shuffle-graphs false
$run --shaping none --shuffle-graphs true
$run --shaping none --shuffle-graphs true --balance-graphs true
$run --shaping none --shuffle-graphs true --adaptive-graph-sampling true
$run --shaping none --shuffle-graphs true --balance-graphs true --adaptive-graph-sampling true
```

Each command starts a separate run with the same default data and training seeds.
Use a fresh output directory for each experiment. Shuffling defaults to true;
balancing and adaptation default to false. These flags also work with
`run_mosp.sh` and the direct trainer. They never change evaluation sampling.

With shuffling, every graph in a pool is visited once before that pool starts a
new coverage epoch. Curriculum pools keep separate coverage state, including
rehearsal visits. A short stage can end before its pool is exhausted: ensure
enough optimizer groups if complete coverage of every stage is required.

Balancing splits each pool at the median of two measured proxies. Difficulty is
shortest-hop distance times `(1 + mean excess outdegree)` times
`(1 + reachable dead-end fraction)`. Trade-off structure combines the number of
distinct true-frontier cost vectors and their spread relative to the fixed cost
bound. These are sampling-only training labels, never policy observations.
The four possible strata are interleaved with equal selection counts while
available. Ties or small pools can produce fewer strata. With strict coverage,
unequal strata retain unequal full-epoch totals; balancing improves the order
and partial-epoch coverage rather than oversampling rare graphs.

Adaptive sampling tracks an exponential moving average of zero-variance GRPO
groups per graph, with update coefficient 0.3. Its weight is
`0.25 + 0.75 * (1 - zero_variance_EMA)`; unseen graphs start at 1. Sampling uses
these weights within the eligible stratum. With shuffling this affects order;
without shuffling it affects frequency. The floor and coverage prevent permanent
exclusion. With strict coverage, graph-specific adaptation begins to affect
choices on later coverage epochs; it has no graph-specific history during the
first pass. Both combined-reward variance and unshaped terminal-score variance are
logged, including the terminal mean to distinguish all-failure/all-success cases.
No successful groups are filtered or retried. Adaptation uses combined rewards,
so shaping can change graph order across conditions; disable adaptation when
isolating shaping effects under identical graph sequences.

`metrics.jsonl` records the epoch, stratum, selection weight and adaptive feedback.
`graph_sampling.json` exports features and per-graph observations; selected-adapter
optimizer/RNG snapshots include sampler state. Fixed seeds and isolated graph
RNGs make scheduling reproducible independently of model token sampling.

All five reward signals can be selected independently:

| Signal | CLI toggle | Maximum task bonus flag | Default maximum |
|---|---|---|---:|
| First edge-cost discovery | `--shaping-edge` | `--edge-weight` | 0.01 |
| Distinct completed path | `--shaping-path` | `--path-weight` | 0.05 |
| Discovered hypervolume | `--shaping-hypervolume` | `--hypervolume-weight` | 0.05 |
| Cost-vector diversity | `--shaping-cost-diversity` | `--cost-diversity-weight` | 0.05 |
| Path disjointness | `--shaping-path-disjointness` | `--path-disjointness-weight` | 0.05 |

Individual toggles take `true` or `false`. Supplying them without `--shaping`
selects one custom agent, with unspecified signals off. Existing presets `none`,
`edge`, `path`, `both` and the default four-condition `ablation` remain available.
The three new signals default off; their weights only take effect when enabled.

```bash
# One agent, with each signal explicitly controlled.
$run --shaping-edge true --shaping-path true --shaping-hypervolume true \
  --shaping-cost-diversity false --shaping-path-disjointness true

# A named combination is equivalent.
$run --shaping 'edge+hypervolume+path_disjointness'

# All 32 subsets of five signals, including none; one agent per subset/seed.
$run --shaping factorial

# A smaller factorial experiment: eight subsets of these three signals.
$run --shaping factorial --shaping-signals hypervolume cost_diversity path_disjointness
```

Do not combine individual toggles with `ablation` or `factorial`; use
`--shaping-signals` to restrict factorial factors. A full factorial multiplies
training by 32 per seed. Use `--dry-run` to inspect the complete plan first. All
agents share the same held-out suite, and the frozen base is evaluated once.
Every checkpoint participates in comparison tables and figures.

The additive training return is `S + edge + path + hypervolume + cost_diversity
+ path_disjointness`, with disabled components zero. The new components are:

* **Hypervolume:** increments in the union of dominated boxes for discovered
  path-cost vectors, normalized by the fixed reference box. Objective j uses
  reference `(node_count - 1) * max_edge_cost + 1`. Coordinates are bounded to
  that box; normalization never uses the true Pareto-frontier volume.
* **Cost diversity:** each new path contributes its nearest previous vector's
  mean coordinatewise absolute distance, normalized by those fixed reference
  coordinates and by `max(1, k - 1)`. The first vector contributes zero.
* **Path disjointness:** each new path contributes its minimum directed-edge
  Jaccard distance to previously completed distinct paths, divided by
  `max(1, k - 1)`. The first path contributes zero. This rewards low overlap,
  rather than requiring strict edge disjointness.

The last two novelty sums are discovery-order dependent and saturate at their
configured task maximum. All components are task-deduplicated, nonnegative and
bounded. Equal-cost paths are retained: they can earn unique-path and
disjointness bonuses and receive full ordinary Pareto credit. They add no new
hypervolume or vector distance. Dominated paths can earn novelty bonuses, so
keep weights small and use unshaped validation S to assess benefit. No bonuses
are paid after failed/context-limited tasks or incomplete required submissions.
Every component and combined return is recorded separately in events, task
records, console tables, `log.txt` and training summaries. Validation, early
stopping and all held-out evaluation rewards remain unshaped S.

Run from the repository root. The shell wrapper activates the existing `grpo`
Conda environment through `run_mosp.sh`. No additional packages are required.

```bash
# Inspect all four conditions without writing files or loading weights.
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --dry-run

# Prepare shared data, train four agents, then compare with the frozen NF4 base.
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh

# Select just one condition: none, edge (option 1), path (option 2), or both.
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --shaping edge
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --shaping path
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --shaping both

# Increase replication after a separate pilot; seeds are paired across conditions.
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --training-seeds 101,202,303
```

`--shaping ablation` is the default: train `none`, `edge`, `path`, and `both`,
each from the exact pinned NF4 base and fresh LoRA initialization. Every condition
uses the same training/validation graphs, seed list and hyperparameters. Each
condition selects its own best adapter using unshaped validation S; consequently
early stopping may select different iteration counts. The frozen base is evaluated
once on the shared test suite, rather than reloaded for each comparison.

The default remains `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` at the original pinned
revision. All base launcher options for graph sizes, objectives, attempts per
graph, LoRA, memory controls, history, thinking settings and early stopping are
available. Training defaults remain 30 groups and one optimizer epoch per group.
This ablation fixes the terminal training objective to Pareto S.

## Path-count and graph curricula

The shaping launcher uses `--curriculum-k 1,2,3` for requested-path-count stages;
the former `--curriculum` spelling is no longer accepted by this launcher. A
single value, such as `--curriculum-k 3`, fixes k. Existing trainer JSON retains
the internal `curriculum` key for compatibility with the shared trainer. The
original non-shaping `run_mosp.sh` interface remains unchanged.

Graph complexity is optional:

```bash
# Omitted flag: fixed graph distribution, as before.
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --curriculum-k 1,2,3

# Bare flag: use grpo_mosp/graph_curriculum.json.
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --curriculum-graph --dry-run

# Explicit file: relative paths are anchored to the repository root.
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh \
  --curriculum-graph path/to/curriculum.json --curriculum-k 1,2,3
```

The supplied default `graph_curriculum.json` is valid JSON:

```json
{
  "subgraphs": [2, 3, 4],
  "nodes-per-subgraph": [[5, 7], [8, 10], [11, 13]],
  "outdegree": [2, 3, 4]
}
```

All three lists are required, nonempty and equal in length. Each node range is
an inclusive `[MIN, MAX]` pair, not Python tuple notation. Malformed JSON,
missing/unequal lists, invalid integer bounds and decreasing stage parameters
raise `ValueError` before data generation or GPU model loading. Subgraphs must be
at least two, and `2 <= outdegree < node MIN`. The stages must be ordered with
non-decreasing subgraph counts, node-range endpoints and internal outdegree caps.

When graph curriculum is enabled, these structural CLI settings are ignored:
`--subgraphs`, `--nodes-per-subgraph`, `--outdegree`, `--objectives`,
`--large-nodes-per-subgraph`, and `--density-outdegree`. The three JSON lists
define the stages; objectives retain the standard three default named costs.
Evaluation size/density shifts are derived from the final stage. Counts, seeds,
attempt budgets, `--max-paths`, `--max-candidates`, and `--difficulty-cutoff`
remain configurable as sampling/resource/evaluation controls.

The total `--steps` is split into nearly equal contiguous graph stages. With
30 steps and the supplied config, groups 1–10 use stage 1, 11–20 stage 2, and
21–30 stage 3. At later stages, 80% of groups sample the current pool and 20%
sample uniformly from earlier stage pools, to retain earlier skills. The first
group of every new stage always uses the current pool. k follows its own equal
stage schedule over the same training progress; the schedules need not have the
same number of stages. No future-stage training graphs are sampled early.

`--train-graphs` is the TOTAL number of training graphs, divided across pools as
evenly as possible: 16 graphs across three stages gives pools of 6, 5 and 5.
At least one graph and one training group must be available per stage. All graph
pools support the largest requested k so earlier stages can be rehearsed later.
Each graph, including its weighted image, is generated once and reused by all
shaping conditions. A dedicated per-seed RNG ensures paired conditions see the
same scheduled graph sequence, independent of model action sampling, when
adaptive graph sampling is disabled. Adaptive ordering uses each agent's own
reward history, while stage/rehearsal scheduling remains paired.

Validation uses a fixed set from the final-stage distribution throughout training.
Held-out in-distribution graphs also follow that final distribution; the original
four shifts are relative to it. Validation/test graphs exclude training graphs
from every stage and their relabeled copies. Training-stage sampling never reads
the validation or test scores.

Early-stopping patience resets on each graph-stage transition, while the global
best validation adapter is retained. Training cannot stop before both the graph
and k curricula reach their final stages. With short stages and patience 3, this
may mean running all 30 groups, deliberately giving the hard stage time to learn.
Realized graph difficulty is not guaranteed to increase: node counts and branching
are structural proxies, and exact-oracle rejection limits still apply.

Alternative designs, intentionally not implemented here, include mastery-gated
advancement using stage-specific validation scores; staggered changes to graph
complexity and k; learning-progress sampling that emphasizes stages where scores
are improving; and a traversal-budget curriculum that starts with more attempts
and gradually reduces them. These would require separately prespecified rules
and validation studies, rather than selecting schedules from held-out test scores.

The source path/hash, frozen parameters, stage memberships and schedule are saved
in configs/manifests. A readable config snapshot is stored in
`<model>_trained/shaping/configs/graph_curriculum.json`. Training pools and images
live in `data/graph_stages/stage-N/data/`; `data/train.jsonl` is their union for
provenance and exclusions. Later invocations load saved parameters when the flag
is omitted; the external source file need not remain present. Per-iteration logs
record the scheduled stage, actual sampled pool and whether rehearsal occurred.

## Bonus definitions

During each complete training task, track directed edges whose costs became known
and completed source-to-target node sequences. Counts persist across attempts and
reset between graph tasks. Equal-cost but different paths count separately.

Let E and P denote the numbers of distinct discovered edges and completed paths:

```
edge_bonus = edge_weight * min(E, edge_cap) / edge_cap
path_bonus = path_weight * min(P, path_cap) / path_cap
training_return = terminal_S + sum(all_enabled_shaping_bonuses)
```

Defaults:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--edge-weight` | 0.01 | Maximum total edge-information bonus per task. |
| `--path-weight` | 0.05 | Maximum total unique-path completion bonus per task. |
| `--edge-cap` | 0 | 0 resolves to the task's attempt budget B. |
| `--path-cap` | 0 | 0 resolves to the requested path count k. |

An event earns the change in the corresponding capped quantity. The maximum
edge-plus-path bonus is 0.06 with default weights; all five signals together
have a maximum bonus of 0.21. These are task totals, not per-move bonuses.
Changing `--beta` controls the existing GRPO KL penalty, not the discovery bonuses.

Bonuses are paid only after normal task completion. Protocol and context failures
receive no shaping bonus. When paths exist in the archive, the final submission
must contain `min(k, archive size)` distinct archived paths; otherwise all bonuses
are withheld. Empty slots are allowed for missing paths. When the archive is empty,
a normally completed task can still earn the edge bonus. A dominated but valid
completed/submitted path can earn shaping even when terminal S is zero. No oracle
Pareto membership is consulted by any shaping function.

Raw discovery events and paid bonuses are stored in each rollout's `shaping`
record. Failure can revoke earned events, so use the paid component totals when
analyzing training return. Observations and prompt rules remain unchanged and do
not expose bonus values or oracle labels. Validation and test evaluation receive
no shaping. Early stopping still maximizes validation S.

The implementation reuses outcome GRPO: intermediate events are summed into one
rollout return before group normalization, and all actions share its advantage.
It does not implement step-level process advantages. Small raw coefficients can
still have substantial influence when every rollout has the same terminal S.

## Lightweight evaluation

| Setting | Shaping launcher default |
| --- | --- |
| Training graphs / validation graphs | 16 / 8 |
| Training seeds per condition | 1 |
| Rollouts per training group | 4 |
| Held-out graph families | 20, across the original five domains |
| Evaluation k | 3 |
| Budget | B=8k only |
| Evaluation repetitions / relabelings | 1 / 1 |
| Modes | Independent exploration and shared-transcript selection |
| Bootstrap draws | 1000 |

With four conditions and the frozen base, defaults produce 200 evaluation tasks:
20 graphs × 1 k × 1 budget × 1 repetition × 2 modes × 5 checkpoints. This is a
pilot-sized comparison, not the original large confirmatory experiment.

Override the single k with `--eval-ks 1`, graph count with `--eval-graphs 50`, or
repetitions with `--eval-repetitions 2`. Graph count must remain a multiple of ten
to populate all domain/difficulty strata. `--selection-diagnostic false` disables
the shared-transcript mode. The lightweight evaluator deliberately rejects budget
sweeps, multiple k values and relabelings; use the original evaluator for those.

All checkpoints receive identical saved graph instances, prompts/observation
formats, grammar, temperature, context and token limits. Task sampling seeds are
also shared across checkpoints. Each model explores independently, with fresh
memory. Shared-transcript selection supplies identical fixed external archives.
Training/validation graphs and their relabeled copies are excluded from testing.
Every generated graph is exported with edge costs through the original plotter.

Tables report unshaped S for each condition against the frozen base and, when
present, against the no-shaping agent. CSV results also include each checkpoint
and all secondary metrics. Repetitions are averaged within graph families before
bootstrapping. Training seeds and graph identities are resampled as separate
factors, preserving seed pairing across conditions and model pairing across
graphs. Domain weighting matches the original evaluator: configured difficulty
weights within domains, equal weighting of the four domains for the shift macro.
One training seed provides no estimate of training-procedure variability. Choose
weights/caps on pilot or validation data, not the held-out comparison results.

## Separate stages and artifacts

```bash
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --mode prepare
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --mode train
# Supply the directory printed by the training launch:
jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --mode eval \
  --output /path/to/full_runs/run_YYYYMMDDHHMMSS
```

To use prepare followed by train, supply the same `--output` to the second call.
Each new invocation without `--output` creates a new experiment. Existing settings
are immutable; later stages load them without requiring repeated flags.

```
full_runs/run_YYYYMMDDHHMMSS/
  data/                                      # Shared training/validation graphs/images
  shaping_suite/                             # Shared held-out graphs/images/transcripts
  <model>_trained/shaping/
    launch_config.json
    configs/                                 # Per-condition/seed trainer inputs + eval config
    none|edge|path|both/runs/seed-N/           # Adapter, training logs, validation and compute
    none|edge|path|both/evaluation/seed-N/     # Unshaped per-task results
  <model>_not_trained/shaping/evaluation/initial/
  comparisons/<model>/shaping/                # CSV, Markdown, JSON, log.txt
  figures/<model>/training/<condition>/seed-N/ # Training curves, CSV and manifest
  figures/<model>/evaluation/                 # Checkpoint comparison bars, CSV and manifest
  log.txt
  pipeline_timings.jsonl
```

Progress, per-run and per-condition training summaries, both evaluation headline
tables and component/total timings are printed to the console and logs. Training
tables separate terminal reward, every bonus component and combined return.
Training compute (including validation) stays separate from inference costs.

Metric figures are exported automatically as PNG and SVG using headless pyplot.
For each condition and training seed, `trajectory/` contains reward, correct
Pareto fraction, complete success, valid-path fraction and duplicate-rate curves.
Each point is a rollout-group mean at the optimizer-update count of the policy
that collected it. Training reward includes any shaping bonuses; Pareto fraction
remains the unshaped terminal score. The selected adapter is marked on the full
trajectory. With a graph curriculum, `curriculum_stage_01/`, etc. contain the same
five plots for each scheduled stage; earlier-stage rehearsal points are marked.
Stages not reached before training ends are labeled explicitly.

Evaluation plots under `<mode>/checkpoint_comparison/` compare every evaluated
checkpoint, including the frozen base and each condition/seed. They show the
primary B=8k estimates with separate domain panels and saved marginal 95%
confidence intervals. Evaluation reward equals correct Pareto fraction S, so
those two requested charts have identical values. The other three charts show
complete success, valid-path fraction and duplicate rate. CSV files and manifests
alongside the figures record the plotted values and sources. `--no-plots`
disables these metric exports, while graph images are still exported.

To export figures from an existing run without loading models or rerunning tasks:

```bash
python -m jm_test.graph_world.grpo_mosp.figures \
  --run /path/to/full_runs/run_YYYYMMDDHHMMSS
```

This writes only under the run's `figures/`, including `figures/export.log`.
Optimizer counts inferred from older logs are labeled as inferred.

Standalone lightweight evaluator:

```bash
python -m jm_test.graph_world.grpo_mosp.eval_shaping all \
  --config /path/to/full_runs/run_YYYYMMDDHHMMSS/<model>_trained/shaping/configs/eval_shaping.json
# Or use prepare, run, summarize individually; summarize accepts --no-plots.
```

The evaluator never trains a model. `run_mosp_shaping.py` orchestrates the existing
`train.py`, which uses optional hooks in `rollout.py` and `shaping.py`. Original
`run_mosp.sh` runs remain unshaped unless shaping is explicitly supplied in a
trainer config. The existing environment, policy, GRPO loss and scientific scoring
definitions are reused.
