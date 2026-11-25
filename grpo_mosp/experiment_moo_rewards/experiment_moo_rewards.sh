#!/usr/bin/env bash
# Share interpreter selection, Conda activation, timing and argument forwarding.
set -euo pipefail
export MOSP_MODULE=jm_test.graph_world.grpo_mosp.experiment_moo_rewards.experiment_moo_rewards
mosp_experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${mosp_experiment_dir}/../run_mosp.sh" "$@"
