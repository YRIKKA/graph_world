"""Evaluate the shared MOSP checkpoints on AutomationBench's public tasks."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .checkpoints import add_model_options, load_selection, model_paths, read_json, serve_checkpoint
else:
    from checkpoints import add_model_options, load_selection, model_paths, read_json, serve_checkpoint

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "automationbench"
LOCAL = BENCHMARK / "local"
CHECKOUT = LOCAL / "AutomationBench"
PYTHON = CHECKOUT / ".venv/bin/python"
REPOSITORY = "https://github.com/zapier/AutomationBench.git"


def verify_checkout(revision):
    if not (CHECKOUT / ".git").is_dir():
        raise ValueError("Run evaluate_automationbench.py setup to install the benchmark")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()
    if head != revision:
        raise ValueError(f"AutomationBench checkout is at {head}; expected {revision}")


def setup(revision):
    bundled_uv = ROOT / "osworld/local/tools/bin/uv"
    uv = shutil.which("uv") or (str(bundled_uv) if bundled_uv.is_file() else None)
    if not uv:
        raise ValueError("Install uv and put it on PATH before running setup")
    if not (CHECKOUT / ".git").is_dir():
        CHECKOUT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=CHECKOUT, check=True)
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
                          cwd=CHECKOUT, capture_output=True, text=True)
    if head.returncode:
        subprocess.run(["git", "fetch", "--depth", "1", REPOSITORY, revision],
                       cwd=CHECKOUT, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=CHECKOUT, check=True)
    verify_checkout(revision)
    env = {**os.environ, "UV_CACHE_DIR": str(LOCAL / "uv-cache"),
           "UV_PYTHON_INSTALL_DIR": str(LOCAL / "python")}
    subprocess.run([uv, "sync", "--locked", "--no-dev", "--python", "3.13"],
                   cwd=CHECKOUT, check=True, env=env)


def visualizer_command(port, ready=None):
    command = [str(PYTHON), "-u", str(BENCHMARK / "visualizer.py"), str(CHECKOUT), "--port", str(port)]
    if ready:
        command.extend(["--ready", str(ready)])
    return command


@contextmanager
def live_visualizer(run, port):
    ready = run / ".visualizer-ready"
    log = run / "visualizer.log"
    with log.open("w") as output, subprocess.Popen(
        visualizer_command(port, ready), stdout=output, stderr=subprocess.STDOUT,
    ) as process:
        try:
            deadline = time.monotonic() + 30
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if not ready.exists() or process.poll() is not None:
                raise RuntimeError(f"Visualizer failed to start on port {port}. See {log}")
            print(f"Visualizer during evaluation: http://127.0.0.1:{port}", flush=True)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            ready.unlink(missing_ok=True)


def evaluation_command(settings, name, endpoint, export):
    command = [
        str(PYTHON), "-m", "automationbench.scripts.eval",
        "--model", name, "--base-url", endpoint, "--api", "chat_completions",
        "--api-key-var", "AUTOMATIONBENCH_LOCAL_KEY", "--domains", settings["domains"],
        "--toolset", settings["toolset"], "--max-concurrent", str(settings["max_concurrent"]),
        "--max-steps", str(settings["max_steps"]), "--num-examples", str(settings["num_examples"]),
        "--export-json", str(export),
    ]
    if settings["tasks"]:
        command.extend(["--tasks", settings["tasks"]])
    return command


def write_report(run, manifest):
    manifest_path = run / "manifest.json"
    pending = run / "manifest.tmp"
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(manifest_path)
    temporary = run / "summary.tmp"
    with temporary.open("w", newline="") as output:
        columns = ["model", "status", "tasks", "pass_rate", "partial_credit", "aborted_tasks", "export_bytes"]
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for name, condition in manifest["conditions"].items():
            writer.writerow({"model": name, **condition})
    temporary.replace(run / "summary.csv")


def record_result(run, export):
    report = read_json(export)
    tasks, summary = report["tasks"], report["summary"]
    if not tasks:
        raise ValueError(f"No AutomationBench task results in {export}")
    links = CHECKOUT / "visualizer/runs/local" / run.name
    links.mkdir(parents=True, exist_ok=True)
    link = links / export.name
    if not link.is_symlink():
        link.symlink_to(export)
    aborted = len(summary.get("aborted_tasks", []))
    return {
        "status": "aborted_tasks" if aborted else "scored",
        "tasks": len(tasks), "pass_rate": summary["pass_rate"],
        "partial_credit": summary["avg_score"], "aborted_tasks": aborted,
        "export_bytes": export.stat().st_size,
    }


def evaluate(args, config, model_config, snapshot, models):
    settings = {
        "domains": args.domains or config["domains"], "toolset": args.toolset or config["toolset"],
        "max_steps": args.max_steps or config["max_steps"], "max_concurrent": config["max_concurrent"],
        "num_examples": args.num_examples, "tasks": args.tasks,
    }
    run = BENCHMARK / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    results = run / "results"
    results.mkdir(parents=True)
    manifest = {
        "automationbench_commit": config["revision"], "settings": settings,
        "model_config": model_config, "base_snapshot": str(snapshot),
        "model_python": args.model_python, "models": model_paths(models),
        "conditions": {name: {"status": "not_run"} for name in models},
    }
    write_report(run, manifest)
    print(f"Run directory: {run}", flush=True)
    env = {**os.environ, "AUTOMATIONBENCH_LOCAL_KEY": "local"}
    visualizer = live_visualizer(run, args.visualizer_port) if args.visualizer else nullcontext()
    with visualizer:
        for name, adapter in models.items():
            print(f"Evaluating {name}", flush=True)
            condition = manifest["conditions"][name]
            condition["status"] = "running"
            write_report(run, manifest)
            export = results / f"{name}.json"
            runner_log = run / f"{name}.runner.log"
            try:
                with serve_checkpoint(args.model_python, snapshot, adapter, name,
                                      run / f"{name}.server.log", run / ".server-port") as endpoint:
                    command = evaluation_command(settings, name, endpoint, export)
                    with runner_log.open("w") as output:
                        completed = subprocess.run(command, cwd=CHECKOUT, env=env,
                                                   stdout=output, stderr=subprocess.STDOUT)
                condition["runner_exit_code"] = completed.returncode
                if export.is_file():
                    condition.update(record_result(run, export))
                if completed.returncode or not export.is_file():
                    raise RuntimeError(f"AutomationBench did not finish successfully. See {runner_log}")
            except KeyboardInterrupt:
                condition["status"] = "interrupted"
                raise
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                condition.update(status="failed", error=str(error))
                print(f"{name}: {error}", file=sys.stderr)
            finally:
                write_report(run, manifest)
    if any(item["status"] != "scored" for item in manifest["conditions"].values()):
        raise RuntimeError(f"Incomplete AutomationBench run. See {run / 'manifest.json'}")
    print(f"Summary: {run / 'summary.csv'}")


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=["run", "inventory", "setup", "visualizer"])
    add_model_options(parser)
    parser.add_argument("--domains", help="Comma-separated domains. Default is the six public scored domains.")
    parser.add_argument("--tasks", help="Comma-separated AutomationBench task names within the selected domains.")
    parser.add_argument("--num-examples", type=positive_integer, default=-1, help="Limit examples per model run.")
    parser.add_argument("--max-steps", type=positive_integer, help="Maximum model turns per task.")
    parser.add_argument("--toolset", choices=["api", "zapier", "limited_zapier"])
    parser.add_argument("--visualizer", action=argparse.BooleanOptionalAction, default=True,
                        help="Serve the visualizer while evaluating (default: enabled).")
    parser.add_argument("--visualizer-port", type=positive_integer, default=8000)
    args = parser.parse_args()
    if args.visualizer_port > 65535:
        parser.error("--visualizer-port must be at most 65535")
    config = read_json(BENCHMARK / "configs/settings.json")
    if args.command == "setup":
        setup(config["revision"])
        return
    if args.command != "inventory":
        verify_checkout(config["revision"])
        if not PYTHON.is_file():
            raise ValueError("Run evaluate_automationbench.py setup to install the benchmark environment")
    if args.command == "visualizer":
        subprocess.run(visualizer_command(args.visualizer_port), cwd=CHECKOUT, check=True)
        return
    model_config, snapshot, models = load_selection(args)
    if args.command == "inventory":
        print(json.dumps({"snapshot": str(snapshot), "models": model_paths(models)}, indent=2))
    else:
        evaluate(args, config, model_config, snapshot, models)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
