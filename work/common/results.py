"""Read experiment records without altering their stored scores."""

import json
import math
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT / "work"
PATTERN = re.compile(r"\(([\d.]+),(\d+),(\d+)\)")


def parse_config(name):
    match = PATTERN.search(str(name))
    if not match:
        raise ValueError(f"No (R,N,S) configuration in {name}")
    return round(float(match[1]), 1), int(match[2]), int(match[3])


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def load_truth(result_dir=None, model="rfscm", task="band_gap"):
    root = Path(result_dir) if result_dir else WORK / "result"
    truth = {}
    for path in sorted(root.glob(f"{model}_{task}_*/results.json")):
        config = parse_config(path.parent.name)
        value = read_json(path)["average_metrics"]["mae"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Non-numeric MAE in {path}")
        if not math.isfinite(value):
            raise ValueError(f"Non-finite MAE in {path}")
        if config in truth and truth[config] != float(value):
            raise ValueError(f"Conflicting records for {config}")
        truth[config] = float(value)
    return truth


def load_state(path):
    state = read_json(path)
    observations = state.get("params_obs", [])
    values = state.get("y_obs", [])
    if len(observations) != len(values):
        raise ValueError(f"Unaligned params_obs/y_obs in {path}")
    return [tuple(c) for c in observations], [float(y) for y in values]
