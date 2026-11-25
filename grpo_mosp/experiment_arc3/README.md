# ARC-AGI-3 Qwen evaluation

Run from the `graph_world` repository root with the GRPO CUDA environment available.
The evaluator uses the same cached NF4 base and eight LoRA adapters selected by
the OSWorld evaluator. It loads one model at a time.

```bash
conda activate grpo
python -m grpo_mosp.experiment_arc3.evaluate_arc3 --list-games
python -m grpo_mosp.experiment_arc3.evaluate_arc3
```

## Duck comparison

`--harness none` uses the direct JSON action policy and is the default.
`--harness duck` uses Tufa's tool policy. Repeat the option to compare both on
the same selected games with the existing frozen NF4 checkpoint:

```bash
python -m grpo_mosp.experiment_arc3.duck.install
python -m grpo_mosp.experiment_arc3.evaluate_arc3 \
  --model base --harness none --harness duck --n-tasks 1 --seed 42
```

The installer fetches only the Python inference source pinned in `duck/install.py`.
Both conditions use one local model server per checkpoint and a separate ARC
scorecard per condition. The evaluator enforces the same action and recovery
limits for both, including each move inside a Duck tool batch. Runs selecting
Duck default to a shared 45-minute per-game limit for every condition. Change
it with `--game-timeout SECONDS`. `--allow-thinking` enables native Qwen
thinking in both conditions. Duck uses Tufa's thinking-preserving template
settings and replays reasoning across tool calls and retained conversation
history. It remains off unless requested in this evaluator. `--keep-notes`
applies to the direct policy. Duck manages its own history.

The Duck Python worker uses filesystem and syscall restrictions on x86-64
Linux with Landlock ABI 3 or newer and libseccomp. Its transcripts and runtime
state are written under `runs/<timestamp>/duck/<model>/<task>/`. The shared
CSVs identify the harness and report accepted moves, model calls, and generated
tokens. Paired runs also put optional frames and action traces in separate
harness directories. Omit `--n-tasks` for a full-catalog paired evaluation.

## Policy comparison with the official Tufa runner

`evaluate_arc3_duck.py` uses Tufa's inline runner, game sessions, resets, and
scoring. With `--harness none` (the default), the model sees the current grid
and the last 16 action records and returns one JSON action per reply.
`--harness duck` uses Tufa's Python tool policy and its native history
management, with the same Linux worker restrictions described above.
The default serves frozen Qwen with vLLM and BitsAndBytes.

```bash
conda activate grpo
python -m grpo_mosp.experiment_arc3.duck.install --runner
python -m grpo_mosp.experiment_arc3.evaluate_arc3_duck \
  --environments-dir /path/to/tufa/game-bundle
```

Compare both policies in one hosted ARC run:

```bash
python -m grpo_mosp.experiment_arc3.evaluate_arc3_duck \
  --online --game su15 --n-passes 1 --concurrent-jobs 1 \
  --harness none duck --deadline --reasoning-effort low medium
```

The runner evaluates every combination of the selected models, policies, and
reasoning efforts on the same games. Each condition starts fresh games and runs
its own preflight with `--deadline`. The combined table goes to `summary.csv`
and `log.txt`. Separate native reports are nested by model, policy, or effort
when the corresponding list has multiple entries.

Compare frozen Qwen, fine-tuned Qwen (the `valid` objective), and frozen Inkling
on the same game:

```bash
./grpo_mosp/experiment_arc3/run_evaluate_arc3_duck.sh \
  --online --game su15 --harness duck --reasoning-effort medium \
  --model qwen-frozen qwen-valid inkling-mini \
  --deadline --n-passes 1 --concurrent-jobs 1
```

The launcher runs the evaluator in `grpo` and loads one model at a time.
Serving uses vLLM. The optional `--backend vllm` flag is accepted.
The launcher selects vLLM's PyTorch sampler to avoid a CUDA toolkit mismatch
in the local Qwen environment. Each server writes `model_server.log` in its
model directory. The launcher gives each server a short, private socket
directory under `/tmp` and removes it on shutdown. Set `VLLM_RPC_BASE_PATH`
to use a short, writable directory that you manage.

- `qwen-frozen` (default) uses the NF4 checkpoint moved to `models/qwen-frozen`.
- `qwen-CONDITION` loads a saved LoRA adapter on that checkpoint. The eight
  condition names are in [models.yaml](../experiment_downstream_tasks/models.yaml).
- `inkling-mini` uses the official NVFP4 checkpoint in `models/inkling-mini`.
  This host uses a separate vLLM build with the
  [Ada support patch](https://github.com/vllm-project/vllm/pull/69165), pinned
  to commit `186cf6c33ffc0b1b008ab3b1b4943c8bb661cc82`.
  Override its Python with `--inkling-vllm-python`. The local default is
  `models/.vllm-inkling/bin/python`. Qwen uses `--vllm-python`, which defaults
  to `~/miniconda/envs/vllm/bin/python`.
- `ALIAS=/absolute/path` selects a compatible frozen snapshot or a Qwen
  adapter with matching checkpoint metadata. For reusable snapshot aliases, add
  `base_models` entries to [models.yaml](../experiment_downstream_tasks/models.yaml).
  Paths in that mapping can be relative to the file.

The manifest records the resolved model and adapter paths and their vLLM
executable. Both models use a shared context cap and send reasoning effort
through their respective native templates.

The defaults use Tufa's 25 versioned public games, 20 passes per game, and
32 workers. This evaluator uses a two-hour limit per level attempt and a 240-second
inference request timeout. The runner uses a 32,768-token context, enables thinking, and
samples with temperature 0.6, top-p 0.95, and top-k 20. Completions can
use the remaining context. `--reasoning-effort` accepts `low`, `medium`,
and `xhigh`, with `easy` accepted as an alias for `low`. It sets the model's
native reasoning effort for gameplay and the optional preflight.
The default is `medium`. Use `xhigh` to restore Qwen's implicit
effort from earlier runs. Effort is recorded in the run manifest.

Use the same offline game bundle as Tufa for matching environment
conditions. For the hosted ARC API, pass `--online`. This mode reads
`ARC_API_KEY` from the shell or `graph_world/.env`. Use `--env-file PATH`
if your key is stored elsewhere. For example, use `--env-file ../.env`
if you still keep it in the parent directory.
A smaller hosted pilot is
`python -m grpo_mosp.experiment_arc3.evaluate_arc3_duck --online --game ar25 --n-passes 1 --concurrent-jobs 1`.
The runner writes native reports and the pinned configuration manifest to
`runs/<timestamp>/`.

Select explicit games with `--game ar25 su15 wa30`. Before loading a model,
the runner checks that the selected games open. An unavailable game is
replaced by a seeded random draw from the unselected games of the
eligible difficulties. The runner warns and omits the game when no
replacement is available. It writes the checks to `game_availability.json`.
Each live game opens when its worker is ready to play.

With `--deadline`, each selected game gets its own preflight for each
condition. `--preflight-timeout SECONDS` sets the total time allowed for
that check, including multiple tool requests. It defaults to 600 seconds.
The check uses the saved initial board and validates an action without
submitting it. Results go to `preflight/<game-id>/preflight.json` and
`log.txt`. A failed preflight skips that game for the condition and the
runner continues with the remaining games.

In deadline mode the client allows at most 230 seconds for queueing and
generation together and closes the vLLM stream at the deadline.
Override the three budgets in seconds with `--generation-timeout`,
`--request-timeout`, and `--game-timeout`. Keep the server deadline shorter
than the client timeout when using `--deadline`. A transport failure gets one
retry. `--game-timeout` resets on a level advance or level retry.
`--bounded-recovery N` allows up to two retries per level by default.
Game over and level-time exhaustion use this allowance. Each retry
restores the current level and gives it a fresh time budget. A standalone
request deadline or invalid reply still ends the pass. `--max-output-tokens N` optionally
caps output for a local run and is recorded in the manifest. Omit it to
keep Tufa's output budget.

The console and `log.txt` show a formatted line for each finished game
pass and the same summary table as the other ARC evaluator. The run also
writes the aggregate statistics to `summary.csv`. The table uses the mean
Tufa RHAE across scored passes. Accepted moves exclude recovery resets.
At each level clear, `log.txt` records the game, pass, level, Tufa action
count, human baseline, and cumulative partial RHAE. The log also records
each level retry and the summary reports their total.
Gameplay timing begins when the live game opens, and the reported
variances are population variances.

Tufa's `summary.txt` reports mean and median scores, wins, total actions
and tokens, token throughput, wall-clock time, and per-game averages.
`diagnostics.html` includes score-versus-token and score-versus-time
curves, hypothetical per-level token caps, a per-game score-versus-token
scatter plot, and cumulative tokens over job time. The scoring tabs cover
official ARC, alternate level weights, and raw levels beaten. The report
links to per-pass gameplay replays in `movies/` and analysis transcripts.
`benchmark.json` stores native game and action records.

Use `--difficulty easy medium` to select both difficulty groups. Add
`--n-tasks 5 --seed 42` to sample five total across the eligible games,
stratified by difficulty. With only `--n-tasks`, sampling uses all
difficulty groups. If `--game` is supplied, sampling and difficulty
filtering apply to those games. Add `--list-games` to preview the selected
IDs before running the model.

## Game selection and budgets

The default is the complete authenticated public catalog returned by ARC.
Set `ARC_API_KEY` in the environment or use `--env-file` with an ARC key. The
default credential file is the repository-root `.env`. To evaluate a locally
installed ARC bundle that includes official human baselines, pass
`--environments-dir /path/to/bundle`. The offline game seed defaults to zero.

Use `--difficulty easy` to evaluate all easy games, or add `--n-tasks 5 --seed 42`
to randomly sample five easy games. The labels come from
`arc3_human_data_processed.csv` and include `easy`, `medium`, `hard`, and
`"extra hard"` (quote the two-word label on the command line).

With `--n-tasks` alone, sampling follows the catalog's difficulty proportions
and includes each available difficulty when the sample is large enough. Smaller
samples cover distinct difficulties, prioritizing the most common labels.
Games missing a label form a separate stratum. The seed defaults to zero, and
every model sees the same games. Add `--list-games` to preview the selection.
The run manifest records the selected versioned game IDs, difficulty labels,
and sampling seed.

The policy uses the rendered game frame and recent action history. Each level
allows five times its official human action baseline. A fresh level budget
starts on level advance or recovery. A recovery restarts the current level
and discards its pending actions. Only completed levels contribute to the
game budget. See [docs_action_limits.txt](docs_action_limits.txt)
for the action-limit policy, retry behavior, and output fields.
These settings apply equally to all models:

| Option | Behavior |
| --- | --- |
| `--keep-notes` | Off by default. Carries a model-written scratchpad across the game and supplies before/after frames, changed-cell examples, and coordinate labels for click actions. |
| `--allow-thinking` | Off by default. Enables Qwen thinking and shows previously tried actions for the current visual state. |
| `--bounded-recovery N` | Allows up to N current-level resets per game on the same scorecard after game over or action budget exhaustion. Defaults to 2. Use 0 to disable recovery. |

Each model call requests up to 131,072 output tokens in either thinking mode.
The local ARC server caps the combined tokenized prompt, including image
tokens, and generated output at 262,144 tokens. For longer prompts the server
reduces the output allowance to the remaining context. Generation can end
earlier on a stop token. The per-request `--timeout` is independently
configurable and defaults to 180 seconds.

The manifest records these settings and their budgets. The official toolkit
provides each game's score and aggregate RHAE. The task log and `tasks.csv`
record how many recovery resets each model used on each game. The summary
reports the mean and population variance of resets across games that started,
including games that used zero resets.

The console, `log.txt`, and CSVs report accepted gameplay moves per game and
the total for each model. The count includes moves ARC acknowledges even if
the board stays the same. Invalid replies, failed action requests, and resets
are excluded.

Each run creates `runs/YYYYMMDDHHMMSS/` in UTC and streams progress and the
final table to `log.txt`. It also writes `manifest.json`, `tasks.csv`,
`action_data.csv`, `summary.csv`, and one official scorecard JSON per model.
The action CSV records initialization, each gameplay decision or recovery reset,
and the final outcome for every model and game. It includes action names and
click coordinates, acceptance, level and attempt budgets, state transitions,
completed levels, errors, and timings. It is always written, including with
`--no-save-actions`. Per-game elapsed
time measures wall-clock game setup and play. Model startup and scorecard
retrieval are outside that measurement. The console and `log.txt` show per-game,
mean, and total time as `HH:MM:SS`, rounded to the nearest second. CSV durations
are recorded in seconds. The summary uses population variance in seconds
squared and reports timed and scored game coverage.

The trace options write artifacts under the run directory:

| Option | Default | Output |
| --- | --- | --- |
| `--save-actions` | On | `actions/<task>/<model>/actions.jsonl` records replies, parsed actions and click coordinates, and outcomes. Disable with `--no-save-actions`. |
| `--save-frames` | Off | `frames/<task>/<model>/action_NNNNN.npz` stores the original ARC color-index arrays under the `frames` key. |
| `--plot-frames` | Off | `frame_images/<task>/<model>/action_NNNNN_frame_NNN.png` renders every frame with PyPlot and the ARC palette. |

Index 0 contains the initial observation. Later frame files show the observation
after the corresponding accepted gameplay action or recovery reset. An action
can return multiple frames. The manifest maps versioned
game IDs to task directory names. Each option can be used independently.
Per-game timing includes the enabled trace exports.

[Official scoring methodology](https://docs.arcprize.org/methodology) · [ARC Toolkit example](https://docs.arcprize.org/toolkit/minimal)
