"""Search a recorded objective; model training requires explicit --train."""

import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import params
from common.results import WORK, read_json, write_json
from common.search import candidates, select_ei

MODELS = ("rfscm", "crabnet", "megnet", "schnet", "alignn")
RUNNERS = {model: f"{model}_run" for model in MODELS}


def metric(record, task):
    metrics = record["average_metrics"]
    if task == "metallicity":
        for key in ("roc_auc", "roc_auc_score", "auc"):
            if key in metrics:
                value = float(metrics[key])
                break
        else:
            # RF metallicity records store 1 - ROC-AUC under the "mae" key.
            value = 1.0 - float(metrics["mae"])
        if not 0 <= value <= 1:
            raise ValueError("ROC-AUC outside [0,1].")
    else:
        value = float(metrics["mae"])
    if not math.isfinite(value):
        raise ValueError("Non-finite evaluation score.")
    return value


def result_path(root, model, task, config):
    r, n, s = config
    for ratio in (str(r), str(int(r)) if r in (0, 1) else str(r)):
        path = root / f"{model}_{task}_({ratio},{n},{s})" / "results.json"
        if path.exists():
            return path
    return None


def evaluate(args, model, config):
    path = result_path(args.result_dir, model, args.task, config)
    if path is None and args.train:
        if args.result_dir.resolve() != (WORK / "result").resolve():
            raise ValueError("--train uses work/result; custom result roots are read-only.")
        python = args.model_python.get(model)
        if not python:
            raise ValueError(f"Supply --model-python {model}=/path/to/python.")
        r, n, s = config
        subprocess.run([
            sys.executable, "-c",
            f"from process_data import process_data; process_data({args.task!r},{r},{n},{s})",
        ], cwd=WORK, check=True)
        subprocess.run([
            python, "-u", "-c",
            f"from {RUNNERS[model]} import run; run({args.task!r},{r},{n},{s})",
        ], cwd=WORK, check=True)
        path = result_path(args.result_dir, model, args.task, config)
    if path is None:
        raise FileNotFoundError(f"No recorded result: {model}/{args.task}/{config}")
    return metric(read_json(path), args.task)


def run(args):
    if args.problem == 2 and len(args.models) != 2:
        raise ValueError("Problem 2 requires exactly two models.")
    if args.problem == 1 and len(args.models) != 1:
        raise ValueError("Problem 1 requires exactly one model.")
    if args.task == "metallicity" and "crabnet" in args.models:
        raise ValueError("The recorded CrabNet workload supports regression only.")
    grid = candidates(exclude_reference=args.problem == 3)
    if not 1 <= args.n_init <= args.budget <= len(grid):
        raise ValueError("Require 1 <= n_init <= budget <= grid size.")
    references = {}
    if args.problem == 3:
        for model in args.models:
            path = args.reference_dir / f"GT_{model}_{args.task}.json"
            references[model] = metric(read_json(path), args.task)
    rng = random.Random(args.seed)
    initial = rng.sample(grid, args.n_init)
    observations, scores, records = [], [], []
    while len(observations) < args.budget:
        config = initial[len(observations)] if len(observations) < args.n_init else (
            select_ei(grid, observations, scores, rng, trees=args.trees)
        )
        raw = {m: evaluate(args, m, config) for m in args.models}
        if args.problem == 1:
            objective = next(iter(raw.values()))
            score = objective if args.task == "metallicity" else -objective
        elif args.problem == 2:
            objective = abs(raw[args.models[0]] - raw[args.models[1]])
            score = objective
        else:
            objective = float(np.mean([abs(raw[m] - references[m]) for m in raw]))
            score = -objective
        observations.append(config)
        scores.append(score)
        records.append({"params": config, "metrics": raw, "objective": objective})
        best = int(np.argmax(scores))
        write_json(args.output, {
            "settings": {
                "problem": args.problem, "models": args.models, "task": args.task,
                "budget": args.budget, "n_init": args.n_init, "seed": args.seed,
                "trees": args.trees, "reference": references,
            },
            "params_obs": observations, "y_obs": scores,
            "objective_convention": "maximization",
            "records": records, "best": records[best],
        })
        print(f"k={len(observations)}/{args.budget} (R,N,S)={config} "
              f"objective={objective:.6g}", flush=True)
    return records[int(np.argmax(scores))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["rfscm"])
    parser.add_argument("--task", choices=params.tasks, default="band_gap")
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--n-init", type=int, default=4,
                        help="Initial samples; defaults to 4.")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-dir", type=Path, default=WORK / "result")
    parser.add_argument("--reference-dir", type=Path, default=WORK / "GT")
    parser.add_argument("--output", type=Path, default=WORK / "outputs/search.json")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model-python", action="append", default=[], metavar="MODEL=PATH")
    args = parser.parse_args()
    try:
        args.model_python = dict(entry.split("=", 1) for entry in args.model_python)
        if args.output.exists():
            raise FileExistsError("Output exists; choose a new --output to preserve it.")
        print(run(args))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
