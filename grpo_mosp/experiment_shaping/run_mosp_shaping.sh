#!/usr/bin/env bash
# Reuse the original launcher's Conda activation and safe argument forwarding.
set -euo pipefail
export MOSP_MODULE=jm_test.graph_world.grpo_mosp.run_mosp_shaping
mosp_shaping_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${mosp_shaping_dir}/../run_mosp.sh" "$@"
