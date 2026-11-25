#!/usr/bin/env bash
# Share Conda activation, timing, output capture and safe argument forwarding.
set -euo pipefail
export MOSP_MODULE=jm_test.graph_world.grpo_mosp.experiment_shaping.experiment_shaping
mosp_experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${mosp_experiment_dir}/../run_mosp.sh" "$@"
