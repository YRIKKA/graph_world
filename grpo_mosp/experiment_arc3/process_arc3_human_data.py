"""Compute solve rates and difficulty quartiles for the ARC-AGI-3 demo set."""

import json
import sys
from pathlib import Path

import pandas as pd


INPUT_CSV = Path(__file__).with_name("arc3_human_data.csv")
BASELINES = Path(__file__).with_name("metadata") / "arc3_human_action_baselines.json"
DIFFICULTY_LABELS = ["easy", "medium", "hard", "extra hard"]


def process_arc3_human_data(input_csv=INPUT_CSV):
    data = (
        pd.read_csv(input_csv)
        .drop(columns=["Top 10 Replays (score)"])
        .rename(columns={"Env.": "environment", "Plays": "plays", "Solves": "solves"})
    )
    data["solve_rate"] = data["solves"] / data["plays"]
    data["difficulty"] = pd.qcut(data["solve_rate"], q=4, labels=DIFFICULTY_LABELS)
    baselines = {
        game_id.split("-", 1)[0]: json.dumps(actions, separators=(",", ":"))
        for game_id, actions in json.loads(BASELINES.read_text(encoding="utf-8")).items()
    }
    data["human_actions"] = data["environment"].map(baselines)
    if data["human_actions"].isna().any():
        missing = data.loc[data["human_actions"].isna(), "environment"]
        raise ValueError(f"Missing ARC baseline data for {', '.join(missing)}")
    return data


if __name__ == "__main__":
    process_arc3_human_data().to_csv(sys.stdout, index=False)

