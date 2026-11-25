# GRPO for hidden multiobjective paths

For the optional four-condition discovery-reward experiment and lightweight
frozen-base comparison, see [SHAPING.md](../experiment_shaping/SHAPING.md) and run
`jm_test/graph_world/grpo_mosp/experiment_shaping/run_mosp_shaping.sh --dry-run`.
The shaping launcher also supports `--curriculum-k` and optional
`--curriculum-graph [JSON]`; its default graph-stage file is
[graph_curriculum.json](../graph_curriculum.json).

A self-contained, single-GPU implementation of outcome-supervised, multi-turn
GRPO. Run commands from the **IsaacSim repository root** with Python 3.12.
No Isaac Sim application or simulator imports are needed.

## Default policy and GPU

The default is **unsloth/Qwen3.8-27B-unsloth-bnb-4bit**, pinned to
`8aa5f05d26b7205477066e1449e0af13f762a299`. The experiment compares the frozen
prequantized **NF4 base** with **the identical NF4 base plus GRPO LoRA adapters**.
The base tensors stay frozen during training; both evaluation models are fully
frozen. There is no dense BF16 or FP8 baseline in this comparison.

The shared loader uses Unsloth `FastModel`, text-only loading, double-quantized
NF4, BF16 computation, embedding offload, no vLLM, and the pinned official chat
template with thinking disabled for constrained JSON actions. Defaults use rank-8
attention adapters on the last eight language layers, Unsloth gradient
checkpointing, sequential group collection and per-action gradient accumulation.
The KL reference disables the adapters on the same base; no second GPU copy is
loaded. Frozen and adapted checkpoints are evaluated sequentially as well.

AWS g6e.2xlarge provides one L40S and 64 GiB host RAM. Context length and other GPU
processes still determine fit; the checkpoint's native 262K context is not a
training memory guarantee. This decoder recomputes prefixes for inspectable
constrained likelihoods. See [NF4_SETUP.md](NF4_SETUP.md) for installation,
verification, loading identity checks, memory controls and limitations.

## Install and smoke checks

Use an isolated Python 3.12 CUDA environment, never Isaac Sim's embedded Python.
Install `requirements.txt` (also included by `requirements-eval.txt`); it pins the
modern Unsloth/Transformers stack required by this model. `requirements-legacy.txt`
retains the earlier Qwen2.5 stack for legacy experiments. CPU-only graph checks
need only `requirements-cpu.txt`.

```bash
python -m pytest -q jm_test/graph_world/grpo_mosp/{tests,experiment_shaping,experiment_moo_rewards}
python -m jm_test.graph_world.grpo_mosp.smoke.smoke_train
```

`smoke_train` uses a tiny random Qwen2 model and real LoRA/GRPO on CPU to test
optimization mathematics. It does not measure pretrained policy improvement.
The real-checkpoint smoke procedure is documented in [NF4_SETUP.md](NF4_SETUP.md).

## Dataset generation and training

Use the launcher to keep data, graph images, adapters, logs and caches together:

```bash
jm_test/graph_world/grpo_mosp/run_mosp.sh --mode prepare --dry-run
jm_test/graph_world/grpo_mosp/run_mosp.sh --mode train --python /path/to/isolated/python
# Then evaluate the printed run directory, without changing its controls:
jm_test/graph_world/grpo_mosp/run_mosp.sh --mode eval --python /path/to/isolated/python \
  --output /path/to/full_runs/run_YYYYMMDDHHMMSS
```

The standalone dataset utility uses `--count` for the accepted count **per split**. Train/validation/test use 12–24
nodes; `test_large` uses 26–32 nodes and `test_4d` adds a fourth objective, `risk`.
Evaluate validation, larger graphs and four-dimensional graphs with the same
command and their respective JSONL paths. Do not tune using test reports.
The default curriculum requests k=1, then 2, then 3, always with B=8k. M=8 is
the number of independent complete tasks on the **same graph** per GRPO group.
The small GPU smoke configuration uses M=2, one update and k=1.

The generator is the supplied `generate_graph_world.py`. The adapter calls
`GraphGenerator(edge_costs=True, mdp=False, seed=..., max_paths=4096)`, preserving
its automatically chosen `source`, `target`, and
`self.multiobjective_shortest_paths`. No transition probabilities are assigned.
It does not change edge weights or recompute endpoints afterward. `mosp.py` is
copied from the parent graph-world implementation so the MOSP package is standalone.
The constructor gained optional seed propagation and a path-count guard before
exhaustive enumeration. Existing unseeded callers retain their behavior.

**Requested defaults take precedence over the guide's sample distribution:**
water/time/energy, three integer components, each sampled independently from
**0 through 5**, rather than two components from 1 through 9. `edge_costs=True`
enables that default distribution; `edge_costs=False` would omit cost attributes
and make all objectives unit costs. The mathematical minimization and scoring
remain valid with zero-cost edges because graphs are finite DAGs. The evaluator
ignores the generator's derived `scalar_sum` entirely.

Two subgraphs provide branching, merging and possible reachable non-target sinks.
Candidate rejection reasons are: too many source-target paths (4096 cap), selected
target not a sink, fewer than three Pareto paths, no dominated path, or no reachable
non-target sink. `statistics.json` records rejected counts and accepted graph
seeds, nodes, edges, total paths and Pareto paths. Generating accepted graphs
therefore conditions the generator distribution on these properties.
Candidate seeds advance by five with a different residue per split, preventing
seed reuse even when rejection counts differ. Persisted instances are the
reproducible benchmark artifacts. Node IDs are randomly permuted opaque strings;
subgraph identities, themes and restrictions are never policy inputs.

## Mathematical task and observations

All additive named cost components are minimized. A vector strictly dominates
another if it is no worse in every component and better in at least one. Every
distinct source-target node sequence with nondominated costs counts, including
different paths with identical costs. There is no additional hop objective or
fixed weighted sum. The exact exhaustive oracle independently checks the
constructor's stored Pareto path set on every accepted graph.

A complete task has B forward attempts and one final submission. The hidden graph
and weights stay fixed. Initially the policy sees s,t,k,B,d,W,n, objective names,
rules, and source neighbors. Occupying a node reveals its outgoing neighbor IDs;
traversing an edge reveals that edge's cost. An explicit deterministic memory
table keeps **every** revealed adjacency, observed cost, dead end, current prefix,
accumulated vector, completed path and remaining attempt. Full interaction history
is also retained. Reaching the target archives the path. Reaching another sink
records a dead end. Either ends an attempt and resets to source if budget remains;
the reset adds no cost and is not a graph edge. There is no reverse backtracking.
A complete task uses at most B(n-1) moves. All nonterminal rewards are zero.

Movement uses exactly `{"move_to":"node_id"}`. After the last attempt the grammar
changes to `{"paths":[...]}` with exactly k slots. A slot is an archived path or
`[]`. Submission candidates are factored by slot, allowing duplicates and failed
slots without exponential Cartesian enumeration. The environment independently
validates actions and submissions. Unexpected malformed/illegal movement or a
malformed submission explicitly terminates with zero reward.

The finite-horizon POMDP latent state includes the graph, current node/prefix/cost,
attempt index, discovered facts, archive and phase. The observable history and
memory are policy inputs; the evaluator's full graph, undiscovered weights,
frontier size and correctness labels are excluded. Completed paths receive no
optimality feedback. Evaluation never updates model parameters.

Memory/history are **never silently truncated**. New launcher runs use
`--history-format state_actions`: the complete current memory plus every prior
move in order. Earlier observations are exactly reconstructible because the
fixed graph reveals adjacency on arrival and edge costs on traversal. The encoder
checks this inverse against every original observation before compacting; raw
messages and actual token contexts remain in rollout audit files. `delta` offers
reversible JSON patches; `full` repeats every complete table. The selected format
is identical for frozen and adapted evaluation models and both diagnostics.

A task that still exceeds `context_limit` terminates with reward/S=0 and an
explicit `context_limit` status; it remains in its GRPO group and evaluation
repetition. No task is filtered, retried, shortened, or selected using an oracle
score. Training logs `context_limit_failure_rate`, `max_context_tokens`, and
per-task `limit_details`. Inspect these rates when selecting graph size and
traversal budgets. Increasing the limit requires GPU profiling even within the
model's native positional support. Unexpected errors such as CUDA OOM still fail
the run explicitly.

## Terminal objective and GRPO

Default reward is `number of distinct submitted globally Pareto paths / k`.
A nonempty submitted path must be a true directed s–t path completed during this
task. Empty, duplicate, invalid and untraversed slots add no credit. Malformed
submissions or wrong slot counts score zero. The denominator is always k.
The global frontier is authoritative, even when a submitted/discovered path is
locally nondominated. All-k success is reported separately from mean slot reward.

Set `objective` to `hypervolume` only for an explicitly different terminal
experiment. Exact recursive box-union hypervolume uses reference coordinate
`(n-1)*W+1`, divided by the full true frontier's volume. It measures cost-space
coverage, counts equal vectors once, and can give positive credit to dominated
paths. It is always an evaluation metric and is never mixed into Pareto reward.

GRPO holds parameters fixed across the M complete rollouts, isolates their memory,
stores old log probabilities, and computes detached population-standard-deviation
advantages `(R-mean)/(std+1e-8)`. Each assistant output token receives the task's
outcome advantage. Rewards come only from final submission. Prompt, observation,
memory and chat-template boundary tokens have no policy loss positions.

Each decision stores its exact conditioning token IDs, generated IDs, allowed
trie children at every token, old log probabilities and cached reference log
probabilities. Sampling, current replay, and reference replay all normalize over
the same legal children at the configured temperature. Forced schema tokens have
probability one. No top-p/top-k truncation or extra stop-token policy is applied.
The final slot chunks are part of one assistant action; each conditions on the
previous chunks' actual generated tokens. Later turns use the official chat
template over the configured lossless history representation.

The clipped ratio objective uses per-rollout token averaging, then group averaging.
The conditional KL sample estimator is `exp(logp_ref-logp)- (logp_ref-logp)-1`
(the nonnegative k3 estimator). Under current-policy samples its expectation is
KL(current || reference); on frozen old-policy samples it is an approximation
during repeated optimizer epochs. Default is one epoch. Zero reward variance
produces exactly zero reward advantages; such groups receive **KL-only** updates
when beta>0; when beta=0, the entire optimizer step is skipped (including momentum).
AdamW weight decay is zero, so it adds no separate regularization objective. No artificial advantages
are introduced. Dropout stays disabled for collection and replay. Generated
lengths, zero-variance-group frequency and protocol failure frequency are logged.

Outputs include resolved config, metrics JSONL, complete trajectory/constraint
records, LoRA adapter/tokenizer, and optimizer/RNG state. There is currently no
resume CLI. Seeds and saved instances make runs repeatable within a fixed runtime;
bitwise reproducibility across CUDA versions/hardware is not promised.

## Baselines and metrics

`random` uses uniform random legal forward moves and exact Pareto filtering of
its discovered archive for final selection. `memory_pareto_filter` chooses among
least-visited observed outgoing edges and uses the same exact archive filtering.
Both use only revealed facts, the same B attempts, and random selection if the
local frontier has more than k paths. Local filtering may retain globally
dominated routes whose dominators remain undiscovered.

`privileged_selection` uses exactly the memory heuristic's exploration, then lets
the evaluator choose globally optimal paths from that archive. This is the direct
selection upper bound for that discovery history. `full_information` uses the
entire true Pareto set without a discovery budget and is the Pareto-count upper
bound. Choosing arbitrary k Pareto paths does not maximize hypervolume.

Reports contain per-task values and method averages: valid directed slot fraction,
distinct valid path fraction, Pareto fraction, all-k success, discovered Pareto
count, discovery fraction, discovery-minus-selection gap, dead-end rate, edge
traversals, raw/normalized hypervolume, distinct nondominated cost vectors, and
protocol failures. Zero-variance-group rate belongs to training metrics because
a single evaluation rollout has no GRPO reward group.

Budget-limited exploration cannot guarantee k Pareto discoveries. Different hidden
graphs may produce identical explored observations but have different global
frontiers. The privileged baselines help separate discovery and final-selection
limitations; reward scores do not remove this partial-observability limitation.

## Modules and acceptance checks

`dataset.py` generates graph-disjoint splits; `oracle.py` enumerates and filters;
`environment.py` implements transitions; `memory.py` serializes observables;
`scoring.py` scores terminal sets; `policy.py` implements Qwen and constraints;
`rollout.py` collects complete tasks; `grpo.py` optimizes them; `train.py` runs the
curriculum; `baselines.py` and `evaluate.py` produce comparison reports.
[Smoke checks](../smoke/README.md), `configs/` and `tests/` provide reproducible checks.
Tests cover strict dominance/ties/global membership, malformed/empty/duplicate/
untraversed submissions, reset/dead ends, memory isolation, hidden-data exclusion,
vector accumulation, nonterminal rewards, task bounds, seeded splits, decoder
supports and stored probabilities, context refusal, environment-token loss masks,
zero-variance advantages, and actual optimizer changes.

## Before/after evaluation

Use [`eval_baselines.py`](../eval_baselines.py) for controlled initialization-versus-GRPO
comparisons. [`EVAL_BASELINES.md`](../experiment_eval_baselines/EVAL_BASELINES.md) documents the 200-graph,
five-training-run, five-repetition design, validation-only checkpoint selection,
held-out shifts, paired/crossed bootstrap intervals, shared-transcript diagnostic
and budget plots. Configuration templates are `experiment_eval_baselines/configs/eval_baselines.json` and
`experiment_eval_baselines/configs/eval_baselines_pilot.json`.

## Shell launcher

[`experiment_shaping.sh`](../experiment_shaping/experiment_shaping.sh) runs the selected seven-agent
shaping experiment with one shared frozen reference and a combined comparison
report. Run it without arguments for the screening profile using `graph_curriculum.json`,
or add `--dry-run` to inspect all settings. Its Python entry point is
[`experiment_shaping.py`](../experiment_shaping/experiment_shaping.py); see [SHAPING.md](../experiment_shaping/SHAPING.md) for
the seven conditions, overrides, saved-run reuse and output locations.

[The MOO reward launcher](../experiment_moo_rewards/experiment_moo_rewards.sh)
runs eight terminal-reward conditions with one frozen reference. Use
`--dry-run` to inspect the plan and the
[post-hoc evaluator](../experiment_moo_rewards/combined_metrics_post_hoc_eval.py)
to rescore saved predictions.

[`run_mosp.sh`](../run_mosp.sh) provides `prepare`, `train`, `eval`, and `all` modes
with CLI controls for graph counts/ranges, GRPO settings, per-graph traversal
budgets and frozen-checkpoint comparisons. From `grpo_mosp/`, run `./run_mosp.sh --help`. See [RUN_MOSP.md](RUN_MOSP.md) for
complete examples and parameter meanings; `--dry-run` prints the resolved plan.

New launcher runs write all training/evaluation artifacts under
`full_runs/run_YYYYMMDDHHMMSS/`, including graph PNGs, logs, model caches
and temporary files. Every saved train/validation/test graph is rendered through
the parent graph-world `plot_graph` implementation with all edge costs displayed.
See [RUN_MOSP.md](RUN_MOSP.md) for the layout and how to select an existing run.

The timestamped run contains `<model>_trained/` for post-training artifacts and
results, and `<model>_not_trained/` for frozen-initialization results. Multiple
models can share one run using `--output <existing-run> --model <checkpoint>`.
Graph data and images stay shared in `data/` and `suite/`; model-specific paired
reports live in `comparisons/<model>/`. See [RUN_MOSP.md](RUN_MOSP.md).

Console output now includes per-iteration, per-run and per-model reward/metric
tables and progress bars. Every actual launcher run captures its complete console
output in `<run>/log.txt`, with additional per-training-run and per-checkpoint
logs. Training summaries describe changing-policy rollouts; held-out summaries
retain the prescribed weighting and paired/crossed confidence intervals. See
[RUN_MOSP.md](RUN_MOSP.md#console-metrics-progress-and-logtxt) for locations and
the read-only snapshot command for existing/running jobs.

Qwen template options are available throughout training and evaluation:
`--reasoning-effort {xhigh,medium,low}` (default `xhigh`) and
`--preserve-thinking true|false` (default `false`). Both comparison arms share the
same settings. Thinking generation remains disabled for constrained JSON actions;
reasoning effort takes effect only when thinking is enabled. See
[Qwen reasoning template controls](RUN_MOSP.md#qwen-reasoning-template-controls)
for aliases, configuration precedence, and retention behavior.


Training defaults to 30 rollout groups, one optimizer epoch per group, and
validation-based early stopping enabled. The monitor exports the best validation
adapter and stops on a plateau after reaching the final curriculum stage. Use
`--early-stopping false` to run all requested steps and export the final adapter.
See [validation plateau stopping](RUN_MOSP.md#validation-plateau-stopping) for
sampling, patience, compute costs and direct-training flags.
