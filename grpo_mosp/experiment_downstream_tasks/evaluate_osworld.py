"""Run local Qwen checkpoints on the pinned OSWorld desktop benchmark."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .checkpoints import add_model_options, load_selection, model_paths, read_json, serve_checkpoint
else:
    from checkpoints import add_model_options, load_selection, model_paths, read_json, serve_checkpoint

BENCHMARK = Path(__file__).resolve().parent / "osworld"


def install_benchmark(checkout, assets, release):
    local_uv = BENCHMARK / "local/tools/bin/uv"
    uv = str(local_uv) if local_uv.is_file() else shutil.which("uv")
    if not uv:
        raise ValueError("Install uv and put it on PATH before running OSWorld setup")
    env = {**os.environ, "UV_CACHE_DIR": str(BENCHMARK / "local/uv-cache")}
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", release,
            "https://github.com/xlang-ai/OSWorld-V2.git", str(checkout),
        ], check=True)
    verify_release(checkout, release)
    subprocess.run([uv, "sync", "--frozen", "--python", "3.12"],
                   cwd=checkout, env=env, check=True)
    python = checkout / ".venv/bin/python"
    authenticated = subprocess.run([
        str(python), "-c", "from huggingface_hub import get_token; raise SystemExit(not get_token())",
    ], cwd=checkout, check=False)
    if authenticated.returncode:
        raise ValueError("Hugging Face authentication is needed for both gated OSWorld repositories. "
                         "Accept access and log in before retrying setup.")
    commands = [
        [uv, "run", "scripts/tools/download_osworld_v2_tasks.py",
         "--benchmark-release", release],
        [uv, "run", "scripts/tools/download_osworld_v2_assets.py",
         "--benchmark-release", release, "--target-dir", str(assets)],
    ]
    for command in commands:
        subprocess.run(command, cwd=checkout, env=env, check=True)


def verify_release(checkout, release):
    if not (checkout / "benchmark_releases" / f"{release}.json").is_file():
        raise ValueError(f"Missing OSWorld {release} checkout. Run the setup command first.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    tag = subprocess.check_output(
        ["git", "rev-parse", f"refs/tags/{release}^{{commit}}"], cwd=checkout, text=True,
    ).strip()
    if head != tag:
        raise ValueError(f"OSWorld checkout is not at {release}: {head}")
    return head


def aws_preflight(checkout, release, config):
    verify_release(checkout, release)
    python = checkout / ".venv/bin/python"
    if not python.is_file():
        raise ValueError("Install the OSWorld runner with the setup command first")
    manifest = read_json(checkout / "benchmark_releases" / f"{release}.json")
    geometry = f"{config['screen_width']}x{config['screen_height']}"
    image = manifest["provider_images"]["aws"]["ubuntu"].get(config["region"], {}).get(geometry)
    if image is None:
        raise ValueError(f"{release} has no pinned AWS Ubuntu image for {config['region']} at {geometry}")
    return subprocess.run([
        str(python), "-B", str(BENCHMARK / "check_aws.py"), config["region"], image["ami_id"],
    ], check=False).returncode


def dry_run_aws(checkout, release, config):
    verify_release(checkout, release)
    python = checkout / ".venv/bin/python"
    if not python.is_file():
        raise ValueError("The OSWorld checkout needs its Python environment")
    domain, task_id = task_pairs(BENCHMARK / "configs/pilot_tasks.json")[0]
    return subprocess.run([
        str(python), "-B", str(BENCHMARK / "dry_run_aws.py"), domain, task_id,
        str(config["screen_width"]), str(config["screen_height"]),
    ], cwd=checkout, timeout=45, check=False).returncode


def task_pairs(path):
    index = read_json(path)
    if not isinstance(index, dict):
        raise ValueError("Expected OSWorld task metadata grouped by domain")
    tasks = []
    for domain, ids in index.items():
        if not isinstance(ids, list):
            raise ValueError(f"Expected a list of task IDs for domain {domain}")
        if not domain.isidentifier() or any(
            not isinstance(task_id, str) or not task_id.isdecimal() for task_id in ids
        ):
            raise ValueError(f"Invalid OSWorld domain or task ID: {domain}")
        tasks.extend((domain, task_id) for task_id in ids)
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("The task selection must be nonempty and contain no duplicates")
    return tasks


def select_tasks(args, full_index):
    supported = set(task_pairs(full_index))
    if args.num_tasks is not None:
        index = args.task_pool or full_index
        pool = sorted(task_pairs(index))
        if set(pool) - supported:
            raise ValueError("Task pool contains IDs outside the pinned release")
        if args.num_tasks > len(pool):
            raise ValueError(f"Requested {args.num_tasks} tasks from a pool of {len(pool)}")
        tasks = sorted(random.Random(args.seed).sample(pool, args.num_tasks))
        selection = {
            "seed": args.seed, "num_tasks": args.num_tasks, "pool": str(index.resolve()),
            "pool_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        }
    else:
        index = full_index if args.full or args.task_id else BENCHMARK / "configs/pilot_tasks.json"
        tasks = task_pairs(index)
        if args.task_id:
            tasks = [(domain, task) for domain, task in tasks if task == args.task_id]
            if len(tasks) != 1:
                raise ValueError(f"Expected one match for OSWorld task {args.task_id!r}")
        if set(tasks) - supported:
            raise ValueError("Task selection contains IDs outside the pinned release")
        selection = {"num_tasks": len(tasks)}
    return index, tasks, selection


def write_task_index(path, tasks):
    grouped = {}
    for domain, task in tasks:
        grouped.setdefault(domain, []).append(task)
    path.write_text(json.dumps(grouped, indent=2) + "\n")


def log_message(run, message):
    print(message, flush=True)
    with (run / "log.txt").open("a") as output:
        output.write(message + "\n")


def write_summary(run, models, tasks):
    temporary = run / "summary.tmp"
    with temporary.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["model", "domain", "task_id", "score", "status"])
        for model in models:
            root = run / "results/pyautogui/screenshot" / model
            for domain, task_id in tasks:
                task = root / domain / task_id
                result = task / "result.txt"
                if result.is_file():
                    try:
                        score = float(result.read_text())
                        if not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError("Score outside [0, 1]")
                    except ValueError:
                        writer.writerow([model, domain, task_id, "", "invalid_result"])
                    else:
                        writer.writerow([model, domain, task_id, score, "scored"])
                else:
                    status = "unscored" if task.is_dir() else "not_run"
                    writer.writerow([model, domain, task_id, "", status])
    temporary.replace(run / "summary.csv")


def runner_command(checkout, config, task_index, result_dir, name, endpoint):
    return [
        str(checkout / ".venv/bin/python"), "scripts/python/run_multienv_qwen.py",
        "--model", name, "--base_url", endpoint, "--headless",
        "--provider_name", config["provider"], "--region", config["region"],
        "--test_all_meta_path", str(task_index), "--result_dir", str(result_dir),
        "--cache_dir", str(checkout / "cache"), "--no-stream", "--no-enable_thinking",
        "--no-enable_recording", "--no-endless_retry",
        "--max_api_retries", str(config["max_api_retries"]),
        "--num_envs", str(config["num_envs"]), "--max_steps", str(config["max_steps"]),
        "--max_tokens", str(config["max_tokens"]), "--image_max", str(config["image_max"]),
        "--history_n", str(config["history_n"]), "--temperature", str(config["temperature"]),
        "--screen_width", str(config["screen_width"]), "--screen_height", str(config["screen_height"]),
    ]


def evaluation_environment(checkout, use_checkout_settings):
    settings = {}
    if use_checkout_settings and (checkout / ".env").is_file():
        python = checkout / ".venv/bin/python"
        if not python.is_file():
            raise ValueError("The OSWorld checkout needs its Python environment")
        command = [
            str(python), "-B", "-c",
            "import json; from dotenv import dotenv_values; print(json.dumps(dotenv_values('.env')))",
        ]
        settings = {key: value for key, value in json.loads(
            subprocess.check_output(command, cwd=checkout, text=True)
        ).items() if value is not None}
    return {**settings, **os.environ}


def evaluate(args, config, model_config, models, snapshot, checkout, assets, release):
    commit = verify_release(checkout, release)
    full_index = checkout / "evaluation_examples/test_v2.json"
    task_index, tasks, selection = select_tasks(args, full_index)
    env = evaluation_environment(checkout, args.checkout is not None)
    if args.checkout and env.get("OSWORLD_FILE_BASE_URL"):
        assets = checkout / Path(env["OSWORLD_FILE_BASE_URL"]).expanduser()
    config = dict(config)
    if args.checkout and env.get("AWS_REGION"):
        config["region"] = env["AWS_REGION"]
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if not (checkout / ".venv/bin/python").is_file() or not assets.is_dir():
        raise ValueError("OSWorld needs its pinned environment and assets. Run setup first.")
    for _, task in tasks:
        if not (checkout / f"evaluation_examples/task_class/task_{task}.py").is_file():
            raise ValueError(f"Missing gated task class task_{task}.py. Run setup first.")
    required = ["AWS_SUBNET_ID", "AWS_SECURITY_GROUP_ID"]
    if args.full:
        required.extend(["WEBSITE_HOST_SUFFIX", "GITLAB_URL", "GITLAB_PRIVATE_TOKEN"])
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError(f"Missing evaluation environment variables: {', '.join(missing)}")

    release_metadata = read_json(checkout / "benchmark_releases" / f"{release}.json")
    manifest = {
        "settings": config, "osworld_checkout": str(checkout), "osworld_commit": commit,
        "task_selection": selection,
        "configured_aws_ami": env.get("OSWORLD_AMI_ID"),
        "task_revision": release_metadata["tasks"]["commit"],
        "asset_revision": release_metadata["assets"]["commit"],
        "task_index": str(task_index), "tasks": tasks,
        "model_config": model_config,
        "base_snapshot": str(snapshot), "model_python": args.model_python,
        "models": model_paths(models),
        "conditions": {},
    }
    if args.plan_only:
        print(json.dumps(manifest, indent=2))
        return

    run = BENCHMARK / "runs" / datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run.mkdir(parents=True)
    if args.task_id or args.num_tasks is not None:
        task_index = run / "task_index.json"
        write_task_index(task_index, tasks)
        manifest["task_index"] = str(task_index)
    instance_prefix = env.get("AWS_INSTANCE_NAME")
    instance_name = f"{instance_prefix or 'osworld'}-{run.name}"
    if not args.checkout and instance_prefix:
        instance_name = instance_prefix
    manifest["aws_instance_name"] = instance_name
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log_message(run, f"Run directory: {run}")
    log_message(run, "Selected tasks: " + ", ".join(f"{domain}/{task}" for domain, task in tasks))
    env.update(OSWORLD_FILE_BASE_URL=str(assets), OSWORLD_BENCHMARK_RELEASE=release,
               AWS_REGION=config["region"], OPENAI_API_KEY="local",
               AWS_INSTANCE_NAME=instance_name)
    desktop_password = env.get("OSWORLD_CLIENT_PASSWORD") or "osworld-public-evaluation"
    failed_conditions = []
    for name, adapter in models.items():
        log_message(run, f"Evaluating {name} on {len(tasks)} tasks")
        condition = manifest["conditions"][name] = {}
        try:
            with serve_checkpoint(args.model_python, snapshot, adapter, name,
                                  run / f"{name}.server.log", run / ".server-port") as endpoint:
                command = runner_command(checkout, config, task_index, run / "results", name, endpoint)
                command.extend(["--client_password", desktop_password])
                with (run / f"{name}.runner.log").open("w") as output:
                    completed = subprocess.run(
                        command, cwd=checkout, env=env, stdout=output, stderr=subprocess.STDOUT,
                    )
                condition["runner_exit_code"] = completed.returncode
                completed.check_returncode()
        except subprocess.CalledProcessError:
            condition["error"] = f"OSWorld runner failed. See {name}.runner.log"
            failed_conditions.append(name)
        except (OSError, RuntimeError, ValueError) as error:
            condition["error"] = str(error)
            failed_conditions.append(name)
        finally:
            write_summary(run, models, tasks)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run([
        str(checkout / ".venv/bin/python"), "-B", str(BENCHMARK / "report_results.py"), str(run),
    ], env={**env, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(run / "matplotlib-cache")}, check=True)
    with (run / "summary.csv").open(newline="") as output:
        unscored = sum(row["status"] != "scored" for row in csv.DictReader(output))
    if failed_conditions or unscored:
        raise RuntimeError(f"{len(failed_conditions)} conditions failed and {unscored} model/task pairs "
                           f"have no valid score. See {manifest_path} and {run / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "setup", "check-aws", "dry-run", "run"))
    add_model_options(parser)
    parser.add_argument("--checkout", type=Path,
                        help="Use an OSWorld checkout and its .env for run or dry-run.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--full", action="store_true", help="Use the full pinned OSWorld task index.")
    selection.add_argument("--task-id", help="Use exactly one task from the pinned index, e.g. 092.")
    selection.add_argument("--num-tasks", type=int, help="Sample this many task IDs without replacement.")
    parser.add_argument("--task-pool", type=Path, help="OSWorld task-index JSON to sample from.")
    parser.add_argument("--seed", type=int, default=42, help="Random task-sampling seed (default: 42).")
    parser.add_argument("--plan-only", action="store_true", help="Print the run plan without inference or launch.")
    parser.add_argument("--max-steps", type=int, help="Override the maximum agent steps per task.")
    args = parser.parse_args()
    if args.num_tasks is not None and args.num_tasks < 1:
        parser.error("--num-tasks must be positive")
    if args.task_pool and args.num_tasks is None:
        parser.error("--task-pool requires --num-tasks")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.checkout and args.command not in ("run", "dry-run"):
        parser.error("--checkout is only supported with run and dry-run")
    if (args.full or args.task_id or args.num_tasks is not None or args.plan_only
            or args.max_steps is not None) and args.command != "run":
        parser.error("Task selection, --plan-only, and --max-steps are only supported with run")
    config = read_json(BENCHMARK / "configs/evaluation.yaml")
    release = config["release"]
    checkout = args.checkout.expanduser().resolve() if args.checkout else BENCHMARK / "local/OSWorld-V2"
    assets = BENCHMARK / "local/assets" / release
    if args.command == "dry-run":
        sys.exit(dry_run_aws(checkout, release, config))
    if args.command == "setup":
        install_benchmark(checkout, assets, release)
        return
    if args.command == "check-aws":
        sys.exit(aws_preflight(checkout, release, config))
    model_config, snapshot, models = load_selection(args)
    if args.command == "inventory":
        print(json.dumps({"snapshot": str(snapshot), "models": model_paths(models)}, indent=2))
    else:
        evaluate(args, config, model_config, models, snapshot, checkout, assets, release)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
