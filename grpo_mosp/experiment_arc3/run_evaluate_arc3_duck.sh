#!/usr/bin/env bash
set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${HOME}/miniconda/etc/profile.d/conda.sh"
conda activate grpo

export VLLM_USE_FLASHINFER_SAMPLER=0

cd -- "${script_dir}/../.."
exec python -m grpo_mosp.experiment_arc3.evaluate_arc3_duck "$@"
