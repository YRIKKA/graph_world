"""Install the pinned Duck inference and runner source."""
import argparse
import sys
from subprocess import run
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

REVISION = "7652836056c59e044f093e3c13ed7438c814169e"
UPSTREAM = Path(__file__).resolve().parent / "upstream"
SOURCE_FILES = (
    "__init__.py",
    "agent/__init__.py",
    "agent/action_names.py",
    "agent/prompts.py",
    "agent/python_tool_sandbox.py",
    "agent/runtime_state.py",
    "agent/tool_agent.py",
    "agent/vision_context.py",
    "utils/__init__.py",
    "utils/grid_utils.py",
    "utils/openai_compat.py",
    "utils/segmentation.py",
)
RUNNER_FILES = {
    **{
        f"inference/{path}": f"ARC3-Inference/inference/{path}"
        for path in (
            "framework/__init__.py",
            "framework/kaggle.py",
            "framework/solver.py",
            "utils/viewer_artifacts.py",
        )
    },
    "configs/inference.json": "ARC3-Inference/configs/inference.json",
    **{
        f"taaf/{module}.py": f"tufa-arc-agi-framework/src/taaf/{module}.py"
        for module in (
            "__init__", "benchmark", "competition_arcade", "deploy", "deploy_inline",
            "deploy_kaggle", "deploy_slurm", "diagnostics", "game", "game_api",
            "game_examples", "kaggle_random", "solver", "solver_examples",
            "standard_benchmarks", "support",
        )
    },
}


def installed():
    marker = UPSTREAM / "REVISION"
    return (marker.is_file() and marker.read_text().strip() == REVISION
            and all((UPSTREAM / "inference" / path).is_file() for path in SOURCE_FILES))


def install():
    if installed():
        return
    UPSTREAM.mkdir(exist_ok=True)
    fetch_sources({
        f"inference/{path}": f"ARC3-Inference/inference/{path}" for path in SOURCE_FILES
    })
    (UPSTREAM / "REVISION").write_text(REVISION + "\n")


def fetch_sources(files):
    url = f"https://raw.githubusercontent.com/Tufalabs/duck-harness/{REVISION}"
    with TemporaryDirectory(prefix="source-", dir=UPSTREAM) as directory:
        staging = Path(directory)
        for path, source in files.items():
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(f"{url}/{source}", timeout=20) as response:
                target.write_bytes(response.read())
        for path in files:
            target = UPSTREAM / path
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / path).replace(target)


def runner_installed():
    marker = UPSTREAM / "RUNNER_REVISION"
    return (installed() and marker.is_file() and marker.read_text().strip() == REVISION
            and all((UPSTREAM / path).is_file() for path in RUNNER_FILES))


def install_runner():
    if runner_installed():
        return
    install()
    fetch_sources(RUNNER_FILES)
    run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target",
         str(UPSTREAM / "dependencies"), "imageio==2.37.2", "imageio-ffmpeg==0.6.0"],
        check=True,
    )
    (UPSTREAM / "RUNNER_REVISION").write_text(REVISION + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true", help="Also install Tufa's official runner.")
    args = parser.parse_args()
    (install_runner if args.runner else install)()
    print(f"Duck source {REVISION} is available at {UPSTREAM}")
