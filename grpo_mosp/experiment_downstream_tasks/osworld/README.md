# OSWorld evaluation of MOSP checkpoints

The experiment uses OSWorld 2.1 and the eight saved seed-101 LoRA adapters
from `full_runs/moo_rewards_run_20261113140643`. The `base` condition uses the
same pinned NF4 checkpoint with the adapter absent. The checkpoint and adapters
are read in place. The local endpoint loads one model condition at a time.

The OSWorld harness uses screenshots, so the serving process loads the
multimodal version of the pinned checkpoint. It maps the text-only training
adapter onto the corresponding language layers and verifies the loaded tensors
against the original adapter file.

## Set up the benchmark

Run these commands from `grpo_mosp/experiment_downstream_tasks/`. Use the GRPO CUDA environment for
model inference and OSWorld's separate, locked Python 3.12 environment for its
runner. The config files with a `.yaml` extension use JSON syntax so the
launcher can read them with the standard library.

1. Accept access on Hugging Face for
   [task classes](https://huggingface.co/datasets/xlangai/osworld_v2_tasks)
   and [assets](https://huggingface.co/datasets/xlangai/osworld_v2_assets_gated).
   Authenticate with `hf auth login` or supply `HF_TOKEN` in the environment.
2. Make `uv` available on PATH. The launcher also recognizes an installation
   under `osworld/local/tools/bin/uv`.
3. Run setup and list the local model conditions.

```bash
python evaluate_osworld.py setup
python evaluate_osworld.py inventory
```

Setup installs the pinned public checkout and task classes under
`osworld/local/OSWorld-V2/`, and uses one asset directory under
`osworld/local/assets/osworld-v2.1/`. The gated assets are shared across all
conditions. The 299 GB archive of historical trajectories is excluded.

## Run the pilot and comparison

The existing GPU runner is in `us-east-2`, and OSWorld 2.1 pins its AWS
desktop to `ami-01017272139e01feb` in `us-east-1`. Keep the runner and cached
weights on the GPU host and give it private reachability to a benchmark subnet
in `us-east-1`. For example, your AWS administrator can peer the two VPCs if
their CIDRs do not overlap, add routes in both directions, and restrict the
benchmark security group to the GPU host's private IP or approved source CIDR.
Keep SSH and VNC off the public internet. The upstream AWS guide lists the
required desktop service ports. [Provider setup][aws] · [AWS VPC peering][peering]

Use the standard boto3 credential chain, ideally an approved instance role or
temporary profile. The GPU host's current `isaac_sim_ec2` role authenticated
successfully but received `UnauthorizedOperation` for EC2 image and network
inventory calls and for a security-group creation dry run. An AWS administrator
needs to grant access to read the benchmark resources and to launch, tag, and
terminate the benchmark instance. Network provisioning can be done separately
by the VPC administrator.

Set the `us-east-1` benchmark subnet and security group before running:

```bash
export AWS_SUBNET_ID=subnet-...
export AWS_SECURITY_GROUP_ID=sg-...
python evaluate_osworld.py check-aws
```

The preflight checks the AWS identity, image availability, and network IDs
without launching an instance. The launcher supplies the release's default
desktop password and gives each launched instance a run-specific name. Set
`OSWORLD_CLIENT_PASSWORD` only if the desktop password differs from the
release default. For the full index, configure `WEBSITE_HOST_SUFFIX`,
`GITLAB_URL`, and `GITLAB_PRIVATE_TOKEN` for the
[release-matched task services][osworld].

The pilot uses task 092 (Blender and LibreOffice Impress). Run it with the
`valid` adapter, then optionally repeat with the base model for comparison.
Use `--model-python` if you launch from a Python environment other than GRPO.

```bash
python evaluate_osworld.py run --model valid
python evaluate_osworld.py run --model base --model valid
python evaluate_osworld.py run --full
```

Omitting `--model` selects the base and all eight adapters in either mode.
Use repeated `--model` options to select individual conditions.

The full run uses every task ID from the pinned release, the base model, and
all eight adapters. Both modes use the same configured step budget, history,
and generation settings for every selected model. The runner disables video
and raw model-response dumps.

Each run creates an ignored `osworld/runs/<run-id>/` directory with the
resolved manifest, model and runner logs, upstream task results, and
`summary.csv`. The CSV includes every selected model/task pair and
distinguishes scored tasks from missing or invalid results. A zero from the
evaluator is retained as a valid score.

[osworld]: https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/docs/PUBLIC_EVALUATION_GUIDELINE_v2.1.md

[aws]: https://github.com/xlang-ai/OSWorld-V2/blob/osworld-v2.1/desktop_env/providers/aws/AWS_GUIDELINE.md
[peering]: https://docs.aws.amazon.com/vpc/latest/peering/vpc-peering-basics.html

## Compare models on a seeded task sample

From the experiment directory, use the external checkout and the desktop
task pool when running without mocked website or GitLab services.

```bash
python evaluate_osworld.py run \
  --checkout /home/ubuntu/OSWorld-V2 \
  --model base --model 'valid+hypervolume__weighted' \
  --num-tasks 10 --seed 42 \
  --task-pool osworld/configs/desktop_tasks.json
```

Repeat `--model` to change the models, and change `--num-tasks`, `--seed`, or
`--task-pool` to change the sample. Sampling uses the full pinned index when
`--task-pool` is omitted. Append `--plan-only` to inspect the exact selection
without launching EC2 or starting a model.

Each model receives the same saved `task_index.json`. The run manifest
records the seed, pool hash, task IDs, and checkpoint paths. The console
table is also written to `log.txt`, with native task scores in `summary.csv`
and a PyPlot comparison in `comparison.png`. The mean task score and
binary success rate (native score equal to one) use only task IDs scored
by every selected model. The table reports per-model score coverage.
