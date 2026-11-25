"""Select and serve the shared downstream evaluation checkpoints."""

import json
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
GRPO = ROOT.parent
MODEL_CONFIG = ROOT / "models.yaml"


def add_model_options(parser):
    installed = Path.home() / "miniconda/envs/grpo/bin/python"
    parser.add_argument("--model", action="append", metavar="CONDITION",
                        help="Repeat to select conditions, or pass NAME=/path/to/adapter.")
    parser.add_argument("--model-python", default=str(installed) if installed.is_file() else sys.executable,
                        help="Python with the GRPO CUDA dependencies.")


def load_selection(args):
    config = read_json(MODEL_CONFIG)
    available = ["base", *config["conditions"]]
    selected = {}
    for spec in args.model or available:
        name, separator, local_path = spec.partition("=")
        if not separator:
            if name not in available:
                raise ValueError(f"Unknown condition {name!r}. Choose from {', '.join(available)}")
            path = None
        else:
            if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_+.-]{0,119}", name)
                    or name in available or not local_path):
                raise ValueError("A custom adapter needs a unique NAME=/path/to/adapter")
            path = Path(local_path).expanduser().resolve()
        if name in selected and path != selected[name]:
            raise ValueError(f"Conflicting paths for condition {name}")
        selected[name] = path

    snapshot, models = checkpoint_paths(config, selected)
    return config, snapshot, models


def model_paths(models):
    return {name: str(adapter) if adapter else None for name, adapter in models.items()}


def read_json(path):
    return json.loads(path.read_text())


def checkpoint_paths(config, selected):
    training = GRPO / config["training_run"]
    base, revision = config["base_model"], config["base_revision"]
    model_dir = base.split("/")[-1]
    cache = ROOT / config["base_cache"] if "base_cache" in config else training / "cache/huggingface/hub"
    snapshot = (cache / ("models--" + base.replace("/", "--")) / "snapshots" / revision).resolve()
    for name in ("config.json", "processor_config.json", "model.safetensors"):
        if not (snapshot / name).is_file():
            raise ValueError(f"Missing local base checkpoint file: {snapshot / name}")
    if not read_json(snapshot / "config.json").get("vision_config"):
        raise ValueError("The cached base checkpoint has no vision configuration")
    checkpoints = {}
    for name, path in selected.items():
        if name == "base":
            checkpoints[name] = None
            continue
        adapter = path or (training / f"{model_dir}_trained" / "moo_rewards" / name
                   / "runs" / f"seed-{config['seed']}" / "adapter")
        for filename in ("adapter_config.json", "adapter_model.safetensors", "mosp_backend.json"):
            if not (adapter / filename).is_file():
                raise ValueError(f"Missing adapter file: {adapter / filename}")
        checkpoint = read_json(adapter / "mosp_backend.json")["initial_checkpoint"]
        if (checkpoint["model"], checkpoint["revision"]) != (base, revision):
            raise ValueError(f"Adapter base differs from the configured checkpoint: {adapter}")
        checkpoints[name] = adapter
    return snapshot, checkpoints


@contextmanager
def serve_checkpoint(python, snapshot, adapter, name, log, ready, *, server_args=()):
    command = [
        python, "-B", str(ROOT / "serve_qwen.py"), "--snapshot", str(snapshot),
        "--model", name, "--ready-file", str(ready), *server_args,
    ]
    if adapter:
        command.extend(["--adapter", str(adapter)])
    with log.open("w") as output, subprocess.Popen(
        command, stdout=output, stderr=subprocess.STDOUT,
    ) as process:
        try:
            deadline = time.monotonic() + 300
            while process.poll() is None and time.monotonic() < deadline:
                if ready.is_file() and (port := ready.read_text().strip()).isdigit():
                    address = f"http://127.0.0.1:{port}"
                    with urlopen(f"{address}/health", timeout=5) as response:
                        if json.load(response)["model"] != name:
                            raise RuntimeError(f"Unexpected model at {address}")
                    yield f"{address}/v1"
                    return
                time.sleep(0.5)
            raise RuntimeError(f"Model server failed to start. See {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            ready.unlink(missing_ok=True)


