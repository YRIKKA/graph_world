# Before and after GRPO: frozen Qwen evaluation

`eval_baselines.py` implements a comparison experiment; it does not assume that
post-training improved the policy. The estimated effect and its uncertainty are
the result. Run every command from the IsaacSim repository root in a separate
Python 3.12 environment. No pretrained checkpoint is downloaded by `prepare`,
`summarize`, or `pilot-plan`.

## Prespecified design

The confirmatory template is `configs/eval_baselines.json`:

- **200 independent graph families total**, 40 in each of five domains:
  in-distribution, larger size, greater density, shifted costs, and four objectives.
- Within each domain, 20 easy and 20 hard families. Difficulty is the fixed proxy
  `number of true source-target paths <= 32` (easy) or `> 32` (hard), not a
  model-performance statistic. The enumeration bound remains 4096 paths.
- Five independent training runs: seeds 101, 202, 303, 404, 505. Five evaluation
  repetitions per checkpoint, graph presentation, k and budget. Repetitions are
  never ranked or selected by oracle scores.
- k in {1,2,3}; B in {k,2k,4k,8k,16k}. **B=8k is primary**; other budgets are
  prespecified secondary comparisons. Each model explores independently.
- One presentation per family by default. Increase `relabelings` before freezing
  the suite to test multiple opaque identifier permutations of each family.
- Within a domain: equal weights for easy/hard strata, uniform weights across k,
  equal weights for graph families within a stratum. First average repetitions,
  then relabelings within families, then k. These fixed weights are retained in
  bootstrap draws; unequal stratum counts never change the estimand implicitly.
- In-distribution is reported separately. Each shifted domain has its own report;
  the additional shift macro equally weights the four shifts. There is no pooled
  ID-plus-shift headline score. Raw hypervolume is not averaged across dimensions;
  its normalized counterpart can be aggregated.

The ID generator is the training generator, with its default water/time/energy
costs in 0..5, 12..24 nodes and internal maximum outdegree 2. Difficulty balancing
changes the evaluation weighting relative to the natural training-generator
frequency; this is an explicitly balanced-difficulty estimand. Size uses 26..32
nodes. Density changes internal maximum outdegree to 3, preserving the same node
range and cost distribution. The cost shift replaces each component w by 2w+1
(range 1..11) and recomputes the global Pareto oracle; the additive offset can
change dominance across routes with different hop counts. Objectives adds risk,
a fourth independently sampled default-distribution component. These shifts
change one generator factor at a time; realized density, path counts, instances
and rejection statistics are saved.

The suite excludes all declared training datasets, each run's training dataset,
validation, pilot families and duplicate suite families. Exclusion uses directed,
rooted, weighted graph isomorphism: source/target roles, objective order and edge
costs must match. A WL hash is only a prefilter; exact isomorphism decides a
match, so merely relabeling training nodes cannot evade exclusion. Different
weights are a genuinely different graph instance. Add every graph-based SFT or
other prior training dataset to `excluded_datasets` as well.

## Checkpoints, provenance and controls

`initial_checkpoint` identifies the exact immutable initialization checkpoint,
not a newly created random adapter. Defaults pin Unsloth Qwen3.8-27B NF4 to
`8aa5f05d26b7205477066e1449e0af13f762a299`. Each `post_checkpoint` identifies a
base model and its learned adapter. Model/adapter Hub revisions must be full
40-character commits. Local checkpoints are identified by SHA-256 hashes of all
artifact files, independent of the local directory name. Copying the same
checkpoint into five directories cannot count as five independent runs.

Each training run must supply its saved training config with a distinct training
seed. The harness checks that its declared initialization equals the comparison
checkpoint. This pipeline records its actual base model/revision in
`initial_checkpoint`; it does not silently claim to load an SFT checkpoint.
For externally SFT-initialized GRPO training, set `pre_grpo_checkpoint` to the
checkpoint immediately before GRPO and record that same descriptor as
`initial_checkpoint` in the external run's training manifest. Both the original
initial checkpoint and the pre-GRPO checkpoint are then evaluated. The comparison
against **pre_grpo** isolates GRPO's added value; the comparison against the
original initialization includes the effects of SFT as well.

All evaluation models have `requires_grad=False`, `eval()` mode and inference-mode
forward passes. They share the initialization tokenizer and official chat template,
system prompt, observation serializer, complete persistent-memory rules, decoder
implementation, temperature, context limit, action-token limit and task-token
limit. No optimizer exists in the harness. Memory and token counters are reset
for every task and repetition. Checkpoint loading is sequential to fit one GPU.

The defaults are temperature 1, native context 32768, max_action_tokens 512 and
max_task_tokens 16384. Token caps count actual generated assistant tokens,
including forced JSON tokens. No token, memory fact or history is truncated.
A context/token limit failure is kept with S=0 and a separate `limit_failure`
flag, rather than dropping that graph or treating it as a malformed protocol
response. This intention-to-evaluate rule is fixed before testing and applies
equally to every checkpoint. Inspect limit rates when interpreting long-budget
results. A CUDA OOM, invalid checkpoint or unrelated runtime error stops the
experiment instead of being silently converted to a task outcome.

## Training runs and validation selection

Install training and plotting dependencies:

```bash
pip install -r jm_test/graph_world/grpo_mosp/requirements-eval.txt
```

Generate the training/validation data if it is not already fixed, and train
independent seeds using the supplied `configs/train_seed_101.json`,
`train_seed_202.json`, `train_seed_303.json`, `train_seed_404.json`, and
`train_seed_505.json`. For example:

```bash
python -m jm_test.graph_world.grpo_mosp.train --config jm_test/graph_world/grpo_mosp/configs/train_seed_101.json
```

There must be **five actual independently trained adapters** for the final
experiment. These configuration files are not fabricated trained checkpoints.
Existing runs can be used by editing the checkpoint/config/compute paths before
freezing the experiment. `training_compute.json` records wall time, GPU identity,
peak allocated VRAM, training tasks, generated tokens and traversals, separately
from downstream inference. Historical external runs may set `training_compute`
to null; their training compute will be explicitly unknown rather than estimated
from inference timing.

Use validation for selecting checkpoints or hyperparameters. `select-validation`
can score a list of prespecified candidate descriptors for each independent run.
An optional `--candidates candidates.json` maps run IDs to descriptor lists:

```json
{
  "grpo-101": [
    {
      "model": "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
      "revision": "8aa5f05d26b7205477066e1449e0af13f762a299",
      "loading": {"backend":"unsloth_nf4","quantization":"nf4","double_quant":true,
        "compute_dtype":"bfloat16","text_only":true,"enable_thinking":false,
        "fast_inference":false,"offload_embedding":true,"torch_compile":false},
      "adapter": "/path/to/full_runs/run_YYYYMMDDHHMMSS/Qwen3.8-27B-unsloth-bnb-4bit_trained/runs/seed-101/adapter"
    }
  ]
}
```

For candidates from different hyperparameter runs, an entry can instead be an
object with `checkpoint` (the descriptor), `training_config`, and
`training_compute` fields. Selection carries the winning provenance into the
selected configuration and excludes every candidate training dataset from
validation. This keeps hyperparameter selection on validation data as well.

Without a candidate file the command evaluates the one declared checkpoint per
run; it still records validation evidence. Selection maximizes mean validation
S at B=8k across fixed graphs/k/repetitions, breaking ties by checkpoint identity.
The command reads no held-out test suite. It writes a selected configuration and
per-run validation records containing candidate identities, every validation
score, data hash and the chosen identity. The evaluation runner verifies that
these records match the selected checkpoint and the unchanged validation data.
Validation must also exclude training graphs and their relabeled copies.

## Separate pilot, then confirmatory test

The pilot template uses 50 separate graph families and the first three independent
training runs, with five repetitions. Train those three runs, then:

```bash
mkdir -p jm_test/graph_world/grpo_mosp/experiments/pilot
python -m jm_test.graph_world.grpo_mosp.eval_baselines select-validation --config jm_test/graph_world/grpo_mosp/experiment_eval_baselines/configs/eval_baselines_pilot.json --selected-config jm_test/graph_world/grpo_mosp/experiments/pilot/eval_baselines_pilot.selected.json
python -m jm_test.graph_world.grpo_mosp.eval_baselines prepare --config jm_test/graph_world/grpo_mosp/experiments/pilot/eval_baselines_pilot.selected.json
python -m jm_test.graph_world.grpo_mosp.eval_baselines run --config jm_test/graph_world/grpo_mosp/experiments/pilot/eval_baselines_pilot.selected.json
python -m jm_test.graph_world.grpo_mosp.eval_baselines summarize --config jm_test/graph_world/grpo_mosp/experiments/pilot/eval_baselines_pilot.selected.json
python -m jm_test.graph_world.grpo_mosp.eval_baselines pilot-plan --config jm_test/graph_world/grpo_mosp/experiments/pilot/eval_baselines_pilot.selected.json
```

`pilot-plan` projects the primary ID paired-interval halfwidth for a grid of graph
counts and independent-training-run counts using separate crossed bootstrap
factors. Repetitions remain at the pilot's fixed count. The projection cannot
predict previously unseen run variability and is a precision-planning heuristic,
not a power guarantee. The graph counts in this planning grid refer to **ID graph
families**, not all five domains. Use the pilot to choose allocation before the
confirmatory evaluation, record the decision, and freeze a fresh configuration.
The command refuses to plan from confirmatory test outcomes. It never modifies
the confirmatory design automatically.

Finish the additional training runs, finalize any validation-only checkpoint
selection, and run the default 200-family confirmatory experiment:

```bash
mkdir -p jm_test/graph_world/grpo_mosp/experiments/confirmatory
python -m jm_test.graph_world.grpo_mosp.eval_baselines select-validation --config jm_test/graph_world/grpo_mosp/experiment_eval_baselines/configs/eval_baselines.json --selected-config jm_test/graph_world/grpo_mosp/experiments/confirmatory/eval_baselines.selected.json
python -m jm_test.graph_world.grpo_mosp.eval_baselines prepare --config jm_test/graph_world/grpo_mosp/experiments/confirmatory/eval_baselines.selected.json
python -m jm_test.graph_world.grpo_mosp.eval_baselines run --config jm_test/graph_world/grpo_mosp/experiments/confirmatory/eval_baselines.selected.json
python -m jm_test.graph_world.grpo_mosp.eval_baselines summarize --config jm_test/graph_world/grpo_mosp/experiments/confirmatory/eval_baselines.selected.json
```

The confirmatory preparation excludes the saved pilot suite, even when graph IDs
are permuted. The five runs may reuse the pilot's trained procedures/checkpoints;
pilot graphs are never reused. This initial full design evaluates **180,000
tasks**: 200 families x 3 k values x 5 budgets x 5 repetitions x 6 checkpoints x
2 diagnostic modes. It is a substantial experiment, not the CPU smoke test.
Additional relabelings or a pre-GRPO checkpoint increase that count.

`prepare` freezes graph data, external exploration and config/source hashes.
Changing datasets, protocol code or controls after that requires a fresh suite.
The runner requires all selected checkpoints before starting, keeps per-task
results durable, and refuses to overwrite results. An interrupted run remains
marked incomplete and cannot be summarized as if it were a complete comparison;
there is currently no resume CLI. Use a fresh experiment directory after an
interrupted run. `--output` overrides the configured experiment directory.

## Scores, diagnostics and uncertainty

S is the number of distinct submitted, archived, valid paths on the **global**
true Pareto frontier divided by k. All-k complete success requires k distinct
globally Pareto paths. Secondary metrics are valid directed slot fraction,
duplicate nonempty slots divided by k, protocol failures, raw and normalized
hypervolume, edge traversals, generated tokens and task latency. None of these
metrics determines which repetition is retained; every repetition is scored.

For archive D, A=min(k,|D intersect true frontier|)/k and L=A-S. These distinguish
exploration limitations from final-selection loss. A separate diagnostic has
both models select from the **same external uniform-random exploration**. It
records moves, archive and observation-history hashes once, then reconstructs
and verifies the complete observable transcript for every selector. No oracle
labels are supplied. That diagnostic times only selection; inherited exploration
traversals are reported separately as `external_edge_traversals`.

For each checkpoint comparison, the bootstrap resamples graph-family identities
within fixed domain/difficulty strata, using the same graph draw for before and
after. All repetitions and relabelings of a graph stay together. For the training
procedure, it additionally resamples independent training-run IDs as a separate
factor; a run draw is shared across all graph strata and a graph draw is shared
across all runs. The shared baseline is not copied into fictitious independent
baseline runs. The point estimate is mean run-level paired improvement under the
prespecified weights. The reported intervals are 2.5th/97.5th percentiles from
2000 bootstrap draws, **pointwise**, not simultaneous multiple-comparison bounds.
Five runs can still give imprecise estimates; report the intervals and domain
variation rather than claiming success from a positive point estimate alone.

A fixed neutral warmup precedes task timing for each loaded model. GPU timing is
synchronized. Per-task latency includes policy inference and environment handling
for independent exploration; external-transcript reconstruction is excluded from
selection-only latency. Model loading and warmup have separate records. Training
compute is never folded into those inference measurements.

## Artifacts and checks

The experiment directory contains:

- `suite.jsonl`: complete evaluator instances, immutable presentation IDs, family
  IDs, domains, difficulty, density and exact path counts.
- `suite_manifest.json`: controls/weights, exclusions, generation rejection counts,
  dataset and transcript hashes, and evaluation-source hashes.
- `transcripts.jsonl`: externally collected moves/archives and exact observable
  history hashes. Compact reconstruction avoids storing repeated memory tables.
- `evaluation_manifest.json`: content-addressed checkpoint identities, training
  seeds/configs, validation evidence, training compute and runtime versions.
- `tasks.jsonl`: every repetition's score, secondary metrics, status, submission,
  discovered archive, seed, model/graph IDs, token count, traversal count and
  latency. `save_histories=true` additionally stores full interaction histories.
- `completed.json`: completion count and result-file hash. Missing, duplicate or
  inconsistent task cells are rejected before aggregation.
- `comparison.csv`: all paired comparisons, metrics, k/difficulty strata, budgets,
  diagnostic modes, graph-cluster or crossed-run/graph 95% intervals.
- `comparison.md`: aggregate B=8k before/after table, keeping ID and shifts separate.
- `<run>/figures/<model>/evaluation/budget_curves/performance_vs_budget_*.svg`
  and `.png`: mean score and paired improvement with confidence bands, one row
  per domain and one figure per reference checkpoint.
- `<run>/figures/<model>/evaluation/<mode>/checkpoint_comparison/`: reward,
  correct Pareto fraction, complete success, valid-path fraction and duplicate
  rate bar charts for every checkpoint, including the frozen base. Domain panels
  use the primary B=8k weighted estimates and saved marginal 95% intervals.
  Evaluation reward equals S, so the reward and Pareto-fraction charts coincide.
- `inference_loading.json` and `pilot_plan.json`: separate load/warmup cost and
  optional pilot allocation projections.

Run the CPU acceptance tests with plotting dependencies installed:

```bash
python -m pytest -q jm_test/graph_world/grpo_mosp/tests/test_eval_baselines.py
```

Tests cover isomorphic training-copy exclusion, actual cost shifts, immutable
suites, checkpoint content identity, paired and crossed uncertainty, repetition/
relabel clustering, independent fresh task memory, shared-archive selection,
exact token/context failures, frozen models, provenance checks, completeness
rejection, and end-to-end CSV/Markdown/PNG/SVG/pilot generation with explicitly
synthetic tiny CPU policies. These tests do not claim Qwen improved; actual added
value requires the independent trained checkpoints and completed experiment.

The reporting choices follow the emphasis on interval estimates and uncertainty
in [Agarwal et al. (2021)](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
and reproducibility and variability across runs in
[Henderson et al. (2018)](https://arxiv.org/abs/1709.06560). The crossed run/graph
bootstrap is the explicit design used for this shared-suite experiment.

When invoked via `run_mosp.sh`, all these artifacts live beneath the launcher's
`full_runs/run_YYYYMMDDHHMMSS/` directory. Suite preparation additionally
exports `images/<domain>/<graph-id>.png` for every saved graph presentation,
using the parent graph-world `plot_graph` function with objective weights visible.
The suite manifest includes their hashes, and evaluator observations exclude them.

The timestamped run contains `<model>_trained/` for post-training artifacts and
results, and `<model>_not_trained/` for frozen-initialization results. Multiple
models can share one run using `--output <existing-run> --model <checkpoint>`.
Graph data and images stay shared in `data/` and `suite/`; model-specific paired
reports live in `comparisons/<model>/`. See [RUN_MOSP.md](../docs/RUN_MOSP.md).

## NF4 initialization versus GRPO adapters

The default initial checkpoint is the pinned Unsloth Qwen3.8-27B NF4 checkpoint.
Both evaluation arms use `model_backends.py`: the same NF4 tensors, BF16 compute,
text-only configuration, embedding-offload rule, official chat template and
`enable_thinking=False`. Only the GRPO arm loads adapters. Loading settings are
part of checkpoint identity, training provenance and evaluation configuration.
The adapter's `mosp_backend.json` must match the base revision and loading contract;
the harness rejects mismatches. Both models are fully frozen, evaluated
sequentially on the identical persisted held-out suite and shared selection
transcripts, with memory reset per task. See [NF4_SETUP.md](../docs/NF4_SETUP.md).

New launcher configurations record `history_format=state_actions` for training and
both evaluation arms. It supplies complete current memory and all prior moves,
verified by reconstructing every original observation. Raw messages remain
available for audit; no graph fact or action is dropped. Both the independent
exploration and shared-transcript diagnostics use this representation. Existing
configs without the field retain the legacy full-history behavior. Context-limit
failures remain in every task cell with S=0, as before; training now uses the same
resource-limit convention.
