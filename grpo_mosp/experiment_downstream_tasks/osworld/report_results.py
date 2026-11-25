"""Report paired OSWorld V2 scores from a completed or partial run."""

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import mean

from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table


def read_scores(run, models):
    scores = {model: {} for model in models}
    with (run / "summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] == "scored":
                scores[row["model"]][(row["domain"], row["task_id"])] = float(row["score"])
    return scores


def paired_metrics(scores, tasks):
    paired = [task for task in tasks if all(task in values for values in scores.values())]
    # OSWorld's V2 monitor uses the native final score and a binary score of
    # one for each native score of exactly one.
    metrics = {
        model: (mean(values[task] for task in paired),
                mean(values[task] == 1.0 for task in paired))
        for model, values in scores.items()
    } if paired else {}
    return paired, metrics


def render_report(scores, tasks, paired, metrics):
    buffer = io.StringIO()
    console = Console(file=buffer, width=max(100, 44 + max(map(len, scores))),
                      force_terminal=False, color_system=None)
    table = Table(title=f"OSWorld comparison · {len(paired)}/{len(tasks)} paired tasks")
    table.add_column("Model", overflow="fold")
    table.add_column("Scored", justify="right")
    table.add_column("Mean task score", justify="right")
    table.add_column("Binary success", justify="right")
    for model, values in scores.items():
        fields = [f"{value:.1%}" for value in metrics[model]] if model in metrics else ["n/a", "n/a"]
        table.add_row(model, f"{len(values)}/{len(tasks)}", *fields)
    console.print(table)
    if len(paired) < len(tasks):
        console.print("Aggregate metrics use tasks scored by every selected model.")
    return buffer.getvalue()


def plot_comparison(path, scores, tasks, paired, metrics):
    width = min(22, max(10, 4 + 0.65 * len(tasks)))
    figure, (aggregate, per_task) = plt.subplots(2, 1, figsize=(width, 8),
                                                layout="constrained")
    chart_width = 0.8 / len(scores)
    for index, (model, values) in enumerate(scores.items()):
        offset = (index - (len(scores) - 1) / 2) * chart_width
        if model in metrics:
            aggregate.bar([offset, 1 + offset], metrics[model], chart_width, label=model)
        positions = [position + offset for position, task in enumerate(tasks) if task in values]
        per_task.bar(positions, [values[task] for task in tasks if task in values],
                     chart_width, label=model)
    aggregate.set_xticks([0, 1], ["Mean task score", "Binary success"])
    aggregate.set_title(f"Aggregate scores on {len(paired)}/{len(tasks)} paired tasks")
    if not paired:
        aggregate.text(0.5, 0.5, "No paired scores", ha="center", transform=aggregate.transAxes)
    per_task.set_xticks(range(len(tasks)), [task for _, task in tasks],
                       rotation=90 if len(tasks) > 30 else 0)
    per_task.set_title("Native per-task scores")
    per_task.set_xlabel("OSWorld task ID")
    for axis in (aggregate, per_task):
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.25)
    per_task.legend(fontsize="small")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run
    manifest = json.loads((run / "manifest.json").read_text())
    tasks = [tuple(task) for task in manifest["tasks"]]
    scores = read_scores(run, manifest["models"])
    paired, metrics = paired_metrics(scores, tasks)
    report = render_report(scores, tasks, paired, metrics)
    with (run / "log.txt").open("a") as log:
        log.write(report)
    print(report, end="", flush=True)
    plot_comparison(run / "comparison.png", scores, tasks, paired, metrics)
    print(f"Plot: {run / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
