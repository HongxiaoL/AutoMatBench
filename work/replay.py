"""Reproduce BO/RS/LHS and restricted-pool searches without model training."""

import argparse
import csv
from pathlib import Path

import numpy as np

from common.results import WORK, load_truth, write_json
from common.search import candidates, replay


def campaign(truth, budgets, seed_count, min_neighbors=1, n_init=None):
    truth = {c: truth[c] for c in candidates(min_neighbors) if c in truth}
    if not truth:
        raise ValueError("Empty lookup table.")
    optimum = min(truth.values())
    sorted_values = sorted(truth.values())
    top1 = sorted_values[int(0.01 * len(sorted_values))]
    top5 = sorted_values[int(0.05 * len(sorted_values))]
    runs, summary = [], []
    for budget in budgets:
        for method, offset in (("bo", 1000000), ("rs", 2000000), ("lhs", 3000000)):
            method_runs = []
            for seed in range(seed_count):
                picks, ys = replay(truth, budget, offset + seed, method, n_init)
                best = np.minimum.accumulate(ys)
                record = {
                    "method": method, "budget": budget, "seed": offset + seed,
                    "n_init": max(1, round(budget * 0.3)) if n_init is None else n_init,
                    "params_obs": picks, "y_obs": ys, "best_so_far": best.tolist(),
                    "simple_regret": float(best[-1] - optimum),
                    "cumulative_regret": float(np.sum(np.array(ys) - optimum)),
                }
                runs.append(record)
                method_runs.append(record)
                print(f"{method.upper()} B={budget} seed={offset + seed} "
                      f"S>={min_neighbors} done", flush=True)
            bests = [r["best_so_far"][-1] for r in method_runs]
            regrets = [r["simple_regret"] for r in method_runs]
            summary.append({
                "method": method, "budget": budget, "n_configs": len(truth),
                "min_neighbors": min_neighbors, "seed_count": seed_count,
                "mean_best_mae": float(np.mean(bests)),
                "median_best_mae": float(np.median(bests)),
                "sr_median": float(np.median(regrets)),
                "sr_p10": float(np.percentile(regrets, 10)),
                "sr_p90": float(np.percentile(regrets, 90)),
                "cr_median": float(np.median([r["cumulative_regret"] for r in method_runs])),
                "top1_hit_rate": float(np.mean(np.array(bests) <= top1)),
                "top5_hit_rate": float(np.mean(np.array(bests) <= top5)),
            })
    return runs, summary


def plot(runs, truth, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budget = max(r["budget"] for r in runs)
    fig, ax = plt.subplots(figsize=(13, 7))
    styles = (("bo", "#c44e52", "-"), ("rs", "#4c72b0", "--"),
              ("lhs", "#55a868", "-."))
    for method, color, linestyle in styles:
        traces = np.array([r["best_so_far"] for r in runs
                           if r["budget"] == budget and r["method"] == method])
        if method == "bo":
            for trace in traces:
                ax.plot(np.arange(1, budget + 1), trace, color=color, alpha=0.12)
        ax.plot(np.arange(1, budget + 1), np.median(traces, axis=0),
                color=color, linestyle=linestyle, linewidth=2.5, label=method.upper())
    ax.axhline(min(truth.values()), color="goldenrod", linestyle=":",
               label="Global minimum")
    ax.set(xlabel="Evaluation step k", ylabel="Best-so-far MAE (eV)",
           xlim=(0.5, budget + 0.5), ylim=(0, None))
    ax.set_xticks(np.arange(1, budget + 1))
    ax.xaxis.label.set_size(21)
    ax.yaxis.label.set_size(21)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=WORK / "result")
    parser.add_argument("--budgets", type=int, nargs="+", default=[12, 24, 36])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--min-neighbors", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--n-init", type=int, default=None,
                        help="Defaults to round(0.3*B).")
    parser.add_argument("--output-dir", type=Path, default=WORK / "outputs/replay")
    args = parser.parse_args()
    truth = load_truth(args.result_dir)
    expected = len(candidates())
    if len(truth) != expected:
        parser.error(f"Full table required: found {len(truth)}, expected {expected}.")
    if args.seeds < 1:
        parser.error("--seeds must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "runs.json").exists():
        parser.error("Output already exists; choose a new --output-dir.")
    runs, summary = campaign(truth, args.budgets, args.seeds,
                             args.min_neighbors, args.n_init)
    write_json(args.output_dir / "runs.json", runs)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    pool = {c: y for c, y in truth.items() if c[2] >= args.min_neighbors}
    plot(runs, pool, args.output_dir / "trajectory.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
