"""Audit recorded held-out counts and source proportions without retraining."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from common.results import WORK, parse_config, read_json, write_json

REPRESENTATIVE = ((0.2, 24, 1), (0.2, 24, 2), (0.2, 24, 4),
                  (0.2, 24, 8), (0.2, 4, 4), (0.2, 48, 4))


def source_fraction(ids):
    mp = sum(str(mid).startswith(("mp-", "mvc-")) for mid in ids)
    jarvis = sum(str(mid).startswith("JVASP-") for mid in ids)
    if mp + jarvis != len(ids) or not ids:
        raise ValueError("Empty member list or unknown material source.")
    return jarvis / (mp + jarvis)


def fold_shift(train_ids, test_ids):
    if set(train_ids) & set(test_ids):
        raise ValueError("Train/test member overlap.")
    return abs(source_fraction(test_ids) - source_fraction(train_ids))


def audit_records(result_dir, split_dir):
    rows, representatives, shifts, coverage = [], [], [], []
    for path in sorted(result_dir.glob("rfscm_band_gap_*/results.json")):
        config = parse_config(path.parent.name)
        r, n, s = config
        record = read_json(path)
        folds = record["fold_results"]
        folder = split_dir / f"band_gap_({r},{n},{s})_folds"
        split_available = all((folder / f"{side}.json").exists()
                              for side in ("train", "val", "test"))
        if split_available:
            train, val, test = [read_json(folder / f"{side}.json")
                                for side in ("train", "val", "test")]
        else:
            train, val, test = {}, {}, {}
        maes = [float(f["mae"]) for f in folds.values()]
        counts = [len(f["targets"]) for f in folds.values()]
        row = {
            "r": r, "n": n, "s": s, "n_nominal": n * s,
            "n_split": sum(map(len, test.values())) if split_available else None,
            "n_eval": sum(counts), "n_folds_eval": len(folds),
            "MAE_recorded": record["average_metrics"]["mae"],
            "MAE_fold_mean": float(np.mean(maes)),
            "MAE_pooled": float(np.average(maes, weights=counts)),
        }
        rows.append(row)
        if config in REPRESENTATIVE:
            representatives.append({
                **row, "fold_sd": float(np.std(maes, ddof=1)),
            })
        d, d_fit = [], []
        reason = None
        for key, fold in folds.items():
            fold_id = key.removeprefix("fold_")
            if fold_id not in train or fold_id not in test or fold_id not in val:
                reason = "missing split membership"
                break
            if len(test[fold_id]) != len(fold["targets"]):
                # No prediction IDs were saved, so the evaluated subset is unknown.
                reason = "partial test evaluation; evaluated IDs unavailable"
                break
            d.append(fold_shift(train[fold_id], test[fold_id]))
            d_fit.append(fold_shift(train[fold_id] + val[fold_id], test[fold_id]))
        coverage.append({"config": list(config), "included": reason is None,
                         "reason": reason})
        if reason is None and d:
            shifts.append({
                "config_id": path.parent.name, "r": r, "n": n, "s": s,
                "MAE": row["MAE_recorded"], "D": float(np.mean(d)),
                "D_train_plus_val": float(np.mean(d_fit)),
                "n_folds": len(d),
            })
    return rows, representatives, shifts, coverage


def interaction_stats(frame, covariate="D"):
    from statsmodels.formula.api import ols

    base = "MAE ~ C(n) + C(r) + C(s) + C(n):C(s) + C(r):C(s)"
    full = base + " + C(n):C(r)"
    reduced = ols(base, frame).fit()
    complete = ols(full, frame).fit()
    reduced_d = ols(base + f" + {covariate}", frame).fit()
    complete_d = ols(full + f" + {covariate}", frame).fit()
    unadjusted = complete.rsquared - reduced.rsquared
    adjusted = complete_d.rsquared - reduced_d.rsquared
    return {
        "n_configs": len(frame), "covariate": covariate,
        "incremental_r2_unadjusted": unadjusted,
        "incremental_r2_adjusted": adjusted,
        "difference": adjusted - unadjusted,
        "covariate_only_r2": ols(f"MAE ~ {covariate}", frame).fit().rsquared,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=WORK / "result")
    parser.add_argument("--split-dir", type=Path, default=WORK / "data")
    parser.add_argument("--output-dir", type=Path, default=WORK / "outputs/audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "coverage.json").exists():
        parser.error("Output exists; choose a new --output-dir.")
    rows, representatives, shifts, coverage = audit_records(args.result_dir, args.split_dir)
    write_csv(args.output_dir / "heldout_counts.csv", rows)
    write_csv(args.output_dir / "representative_uncertainty.csv", representatives)
    write_csv(args.output_dir / "config_with_source_shift.csv", shifts)
    write_json(args.output_dir / "coverage.json", coverage)
    if shifts:
        frame = pd.DataFrame(shifts)
        stats = [interaction_stats(frame, variable)
                 for variable in ("D", "D_train_plus_val")]
        mixed = frame[~frame.r.isin([0, 1])]
        if len(mixed):
            stats.append(interaction_stats(mixed))
        write_json(args.output_dir / "source_shift_stats.json", stats)
        print(stats)
    print(f"Counts: {len(rows)} configurations; source analysis: {len(shifts)}. "
          f"Excluded: {len(coverage) - len(shifts)}.")


if __name__ == "__main__":
    main()
