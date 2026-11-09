"""Export primary search tables and shared-configuration model comparisons."""

import argparse
import csv
from pathlib import Path

import numpy as np

from common.results import WORK, load_state, load_truth, write_json


def primary_tables():
    rows = []
    for path in sorted(WORK.glob("optimization*.json")):
        if "BEST" in path.name:
            continue
        configs, values = load_state(path)
        if not values:
            continue
        problem = 2 if "dual" in path.name else (3 if "2nd" in path.name else 1)
        task = next((t for t in ("band_gap", "bulk_modulus", "formation_energy",
                                "metallicity", "refractive_index", "shear_modulus")
                     if t in path.name), None)
        if task is None:
            continue
        ys = np.array(values)
        if problem == 3 or task != "metallicity":
            ys = np.abs(ys)
        elif np.mean(ys) < 0.4999:
            ys = 1 - ys
        higher = problem == 2 or (problem == 1 and task == "metallicity")
        best = int(np.argmax(ys) if higher else np.argmin(ys))
        rows.append({
            "state": path.name, "problem": problem, "task": task,
            "best_observed": float(ys[best]), "best_step": best + 1,
            "recorded": len(ys), "r": configs[best][0],
            "n": configs[best][1], "s": configs[best][2],
        })
    return rows


def ranking(result_dir):
    baseline = load_truth(result_dir)
    frames, summary = [], []
    for model in ("alignn", "megnet", "schnet", "crabnet"):
        values = load_truth(result_dir, model)
        for path in sorted(WORK.glob(f"optimization_1st_state_{model}_band_gap*.json")):
            configs, scores = load_state(path)
            values.update(zip(configs, map(abs, scores)))
        shared = sorted(set(values) & set(baseline))
        deltas = [values[c] - baseline[c] for c in shared]
        if not deltas:
            raise ValueError(f"No shared configurations for {model}.")
        for config, delta in zip(shared, deltas):
            frames.append({
                "model": model, "r": config[0], "n": config[1], "s": config[2],
                "model_mae": values[config], "rfscm_mae": baseline[config],
                "delta_mae": delta,
            })
        summary.append({
            "model": model, "shared": len(shared),
            "wins": int(np.sum(np.array(deltas) < 0)),
            "losses": int(np.sum(np.array(deltas) > 0)),
            "ties": int(np.sum(np.array(deltas) == 0)),
            "mean_delta_mae": float(np.mean(deltas)),
            "median_delta_mae": float(np.median(deltas)),
        })
    return frames, summary


def ranking_plot(frames, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ("alignn", "megnet", "schnet", "crabnet")
    data = [[r["delta_mae"] for r in frames if r["model"] == model] for model in models]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(data, labels=["ALIGNN", "MEGNet", "SchNet", "CrabNet"],
               showmeans=True, meanprops={"marker": "D", "markerfacecolor": "gold",
                                         "markeredgecolor": "black"},
               medianprops={"color": "black"})
    ax.axhline(0, color="gray", linestyle="--", label="Equal MAE")
    ax.plot([], [], marker="D", color="gold", linestyle="", label="Mean")
    ax.plot([], [], color="black", label="Median")
    ax.set_ylabel("MAE difference relative to RF-SCM (eV)", fontsize=21)
    ax.tick_params(labelsize=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=16)
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=WORK / "result")
    parser.add_argument("--output-dir", type=Path, default=WORK / "outputs/summary")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "primary_tables.csv").exists():
        parser.error("Output exists; choose a new --output-dir.")
    frames, summary = ranking(args.result_dir)
    for name, rows in (("primary_tables", primary_tables()), ("ranking_pairs", frames)):
        with (args.output_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.output_dir / "ranking_summary.json", summary)
    ranking_plot(frames, args.output_dir / "ranking.png")
    print(summary)


if __name__ == "__main__":
    main()
