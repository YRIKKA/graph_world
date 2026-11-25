#!/usr/bin/env bash
# One entry point for dataset creation, independent GRPO runs, and paired eval.
# Qwen options are forwarded unchanged: --reasoning-effort xhigh|medium|low
# (default xhigh), --preserve-thinking true|false (default false).
# Training defaults: --steps 30 --epochs-per-group 1 --early-stopping true.
# Keep shell arguments as arrays: paths and model IDs are never evaluated as code.
set -euo pipefail
# Capture launch time before Conda activation; Python includes this startup cost.
export MOSP_PIPELINE_STARTED_AT="${EPOCHREALTIME}"
# Avoid import bytecode beside source files before the timestamped run exists.
export PYTHONDONTWRITEBYTECODE=1
# Keep model diagnostics in order when stage output is piped into a run log.
export PYTHONUNBUFFERED=1

# Initialize Conda in this noninteractive shell and activate the pipeline runtime.
source /home/ubuntu/miniconda/etc/profile.d/conda.sh
conda activate grpo

mosp_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
mosp_repo_root="$(cd -- "${mosp_script_dir}/.." && pwd -P)"
mosp_python="${MOSP_PYTHON:-python3.12}"
mosp_args=("$@")

# Select the interpreter before importing Python packages. The Python CLI also
# accepts this flag, so its full invocation can be recorded in the run manifest.
for ((mosp_i=0; mosp_i<${#mosp_args[@]}; mosp_i++)); do
    case "${mosp_args[mosp_i]}" in
        --python)
            if ((mosp_i + 1 >= ${#mosp_args[@]})); then
                echo 'run_mosp.sh: --python requires an executable' >&2
                exit 2
            fi
            mosp_python="${mosp_args[mosp_i+1]}"
            ;;
        --python=*) mosp_python="${mosp_args[mosp_i]#--python=}" ;;
    esac
done

export PYTHONPATH="${mosp_repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
# Resolve relative CLI paths consistently when invoked from any directory.
cd -- "${mosp_repo_root}"
# The shaping wrapper reuses Conda initialization and argument forwarding.
mosp_module="${MOSP_MODULE:-grpo_mosp.run_mosp}"
exec "${mosp_python}" -m "${mosp_module}" "${mosp_args[@]}"
