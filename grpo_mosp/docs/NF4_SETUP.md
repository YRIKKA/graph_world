# Qwen3.8 NF4 QLoRA backend

The default base and tokenizer are pinned together to:

- Repository: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`
- Revision: `8aa5f05d26b7205477066e1449e0af13f762a299`
- Comparison: frozen NF4 versus the same NF4 checkpoint plus GRPO adapters.

[Unsloth's Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8/train) specifies
`FastModel` for RL and `fast_inference=False`. This checkpoint uses the
`qwen3_5` architecture and requires the modern Transformers/Unsloth stack.
Whole-model torch compilation is disabled in both arms: the compiled BF16
paths showed mode-dependent likelihood differences during verification. Unsloth
rewritten eager forwards still use its low-level FLA/Triton kernels. This choice
is part of checkpoint loading identity. We retain the custom multi-turn constrained GRPO objective rather than replacing
it with a text-completion trainer. No vLLM or separate reference model is used.

## Installation and launch

Use an isolated Python 3.12 environment. For a fully contained run, create the
runtime inside the run directory before invoking the launcher:

```bash
mosp_base=jm_test/graph_world/grpo_mosp
mosp_run="$mosp_base/full_runs/run_$(TZ=America/New_York date +%Y%m%d%H%M%S%6N)"
mkdir -p "$mosp_run/runtime"
python3.12 -m venv "$mosp_run/runtime/venv"
PIP_CACHE_DIR="$PWD/$mosp_run/cache/pip" \
  "$mosp_run/runtime/venv/bin/python" -m pip install -r "$mosp_base/requirements.txt"
"$mosp_base/run_mosp.sh" --python "$mosp_run/runtime/venv/bin/python" \
  --output "$mosp_run" --dry-run
```

Remove `--dry-run` to launch the full configured campaign. The launcher redirects
model downloads, compilation caches, temporary files, plots and stage logs into
the same run. Installations need disk space for CUDA wheels plus the 22.3 GB
checkpoint. Dependencies are isolated from Isaac Sim. The included requirements
pin the tested CUDA 12.8/PyTorch 2.10 and Transformers 5.17/Unsloth 2026.11 stack.

For a short real-model integration check, use `smoke_nf4` below before launching
a large campaign. Compilation makes the first run slower. For the ordinary
launcher, a small `--mode train --steps 1 --group-size 2 --curriculum 1
--attempts-per-graph 2 --context-limit 4096 --training-runs 1` checks actual
training plumbing on generated graph data; it is not a scientific comparison.

## Memory and objective controls

The base stays NF4; BF16 denotes arithmetic, not dense weight storage. The loader
verifies actual bitsandbytes NF4 modules and nested quantization states, refusing
a dense/FP8 fallback. Its immutable local snapshot prevents automatic redirection
to an upstream model or a moving tokenizer revision.

Default adapters have rank 8 and alpha 16, attention modules in the last eight
language layers, with no vision or MLP adapters. `--lora-last-layers 0` enables
all language layers. `--lora-mlp` adds MLP adapters. These changes increase memory
and alter the trainable hypothesis class; select them on validation data.

Embedding offload is enabled in both arms. Gradient checkpointing and per-action
backward calls bound activation retention. The group size counts sequential
complete tasks, not simultaneous GPU batches. The frozen reference is obtained
by disabling LoRA. Evaluation loads one checkpoint at a time and disables every
parameter gradient, including adapter gradients.

The default pre-load free-memory threshold is 24 GiB; the PyTorch allocator cap
is 38 GiB, reserving room for this instance's existing GPU workloads. The cap does
not cover every CUDA library allocation. Neither setting certifies that all
histories fit. Start with 4096-token smoke contexts and profile representative
long tasks before selecting larger limits. The launcher retains its existing
32768-token experimental context limit; native model positional support alone
is insufficient to promise fit. Context overflow is explicit; no history is
silently truncated. The recomputing decoder favors auditable likelihoods over
throughput. Training a full five-seed campaign on this GPU will be expensive.

## Checkpoint integrity and paired evaluation

Every adapter contains `mosp_backend.json`: pinned base identity, NF4 loading
contract, official-template hash, runtime package versions and trainable count.
`adapter_config.json` also names the public pinned base rather than a
machine-specific cache path. Training writes `backend.json` and compute records.
The evaluator checks base revision and loading contract before attaching an
adapter; changing compute dtype or embedding-offload settings requires a new
training/evaluation identity. Adapters are saved separately and never merged
into quantized base weights.

Both arms share the persisted held-out suite, k values, traversal budgets,
constrained grammar, sampling rules, token/context limits and memory rules.
Each explores independently; shared-transcript diagnostics use the identical
external archive. Existing confidence intervals, stratification, validation
selection and training-versus-inference cost records remain in effect.

Older Qwen2.5 configurations can use `--backend transformers` and their pinned
revision in a separate run. Their results do not represent this NF4 comparison.

## Reproducible real-checkpoint integration smoke

```bash
/path/to/isolated/python -m jm_test.graph_world.grpo_mosp.smoke.smoke_nf4
```

This creates a new timestamped run, exports the fixed diagnostic graph with edge
costs, checks frozen sampling/replay, verifies zero-initialized-adapter likelihoods
and finite nonzero adapter gradients, runs one genuine terminal-reward GRPO group,
saves adapters and reloads them frozen. Reload likelihoods must match. The report
is `smoke_nf4.json`. This diagnostic is separate from held-out evaluation; a
constant-reward group is valid but provides no reward-driven learning evidence.

## Instance verification

On this instance's L40S, the pinned runtime passed 57 automated checks and the
real-checkpoint smoke at a 4096-token limit. Measured peak PyTorch allocation was
22.33 GiB; 2,970,112 adapter parameters were trainable. Frozen-versus-zero-adapter
inference, collection-versus-replay, zero-adapter-versus-reference, and saved
adapter reload checks all showed zero maximum log-probability difference in this
short fixture. The two genuine task rewards were both 1, yielding a zero-advantage
GRPO group; finite nonzero adapter gradients were checked separately without an
optimizer step. This establishes runtime integration, not GRPO learning benefit
or a memory guarantee for long benchmark trajectories.

Local verification artifacts are in
`full_runs/run_20261103143115993146/smoke_nf4.json`, including per-stage reports,
the fixed graph PNG, saved adapter, resolved runtime requirements and logs. The
isolated interpreter is in that run's `runtime/venv/bin/python`. Training and
reload execute in separate processes so training offload/graph caches cannot
consume evaluation headroom.

The actual evaluator loader was also exercised sequentially with the frozen base
and two frozen adapter loads in one process. Each unload left approximately
0.008 GiB of PyTorch allocation, verifying that evaluation does not accumulate
model copies. Details are in `runtime/sequential-eval-report.json` in that run.

## Optimized convolution and bounded histories

`requirements.txt` includes `requirements-kernels.txt`, pinning the official
causal-conv1d 1.7.0 CUDA 12 / Torch 2.10 / CPython 3.12 Linux x86_64 wheel with its
SHA-256. This avoids an nvcc/source-build requirement. The wheel is installed in
`grpo` and its CUDA float32/BF16 forward and backward operations were checked.
The NF4 backend checks this dependency before loading weights. See the
[official kernel repository](https://github.com/Dao-AILab/causal-conv1d).

The earlier fallback warning reported a correct, slower PyTorch implementation;
it did not cause the context overflow. Unsloth messages about embedding and
gradient offload describe the intended memory-saving configuration.

New runs default to `--history-format state_actions`, preserving complete current
memory and every prior move while verifying exact reconstruction of all original
observations. `delta` and `full` remain available for explicit experiments. Context
exhaustion is a recorded zero-reward task outcome in training/evaluation; it never
triggers history truncation or task resampling. The configured 32768 limit remains
in effect. See [RUN_MOSP.md](RUN_MOSP.md) for controls and fresh-run requirements.

## Context-overflow fix verification

Artifacts are in `full_runs/run_20261103155333235495/runtime/fix-verification.json`.
The stable automated suite passed **76 tests**. A tokenizer-only stress audit of
the failed run's 64 saved graphs at k=1,2,3 checked 192 histories, verifying exact
reconstruction in every case. The largest raw prompt was 869356 tokens; the
largest `state_actions` prompt was 4997, with zero compact prompts over 32768.
These are diagnostic trajectories, not exhaustive guarantees or model evaluation
repetitions. The tokenizer's warning while counting oversized legacy histories
in that diagnostic log does not describe a model forward pass.

A genuine one-step NF4 GRPO regression used the original first-selected graph
(seed 1344), k=1, B=8, and two independent complete rollouts. It completed with
nonzero reward variance, zero context/protocol failures, and a maximum input plus
response length of 1190 tokens. Saved LoRA weights were finite and LoRA B tensors
changed from their zero initialization. Peak PyTorch allocation was 22.33 GiB.
This verifies execution; it is not a before/after performance experiment.

The optimized convolution dependency is installed together with its declared
`ninja==1.13.0` helper; `pip check` reports no broken requirements. Adapter saving
explicitly sets `save_embedding_layers=False`: vocabulary and base embeddings
are unchanged, so no remote vocabulary-config lookup is needed. An offline
adapter-only save check confirmed that warning is absent.
