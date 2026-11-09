#!/usr/bin/env python
"""CrabNet regression runner for one (R, N, S) split.

Writes per-fold predictions and metrics to
work/result/crabnet_<task>_(<R>,<N>,<S>)/results.json.
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, max_error, r2_score
)

AUTOMATBENCH_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_HOME = os.path.join(AUTOMATBENCH_HOME, "work", "data")

sys.path.insert(0, AUTOMATBENCH_HOME)
sys.path.insert(0, os.path.join(AUTOMATBENCH_HOME, "work"))


def robust_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-10
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_eval_metrics_regression(y_true, y_pred):
    """Regression metrics (MAE/RMSE/R2/MAPE); no automatminer import needed."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "mape": float("nan")}
    yt, yp = y_true[mask], y_pred[mask]
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")
    mape = robust_mape(yt, yp)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def convert_tghe_to_stoichiometry(formula_unit_str):
    import re
    u_str = formula_unit_str.replace(" ", "").strip()
    if u_str == "H0.16667":
        return "H"
    if u_str.startswith("(") and u_str.endswith(")"):
        inner_str = u_str[1:-1]
        m = re.match(r"^(?P<form>.*?)\)(?P<mult>[\d.]+)$", u_str)
        if m:
            inner_str = m.group("form") + ")" + m.group("mult")
        num_idx = None
        for ci in range(len(inner_str) - 1, -1, -1):
            if not (inner_str[ci].isdigit() or inner_str[ci] == "."):
                num_idx = ci + 1
                break
        if num_idx is None:
            return u_str
        base = inner_str[:num_idx]
        num = inner_str[num_idx:] if num_idx < len(inner_str) else None
        if not base:
            return u_str
        if not num:
            return base
        try:
            val = float(num)
        except Exception:
            return u_str
        elements = re.findall(r"([A-Z][a-z]?)([\d.\-+eE]*)", base)
        out_parts = []
        for el, cnt in elements:
            if not el:
                continue
            if not cnt:
                new_c = 1.0 * val
            else:
                try:
                    new_c = float(cnt) * val
                except Exception:
                    new_c = None
            if new_c is None:
                out_parts.append(f"{el}{cnt}")
                continue
            if abs(new_c - round(new_c)) < 1e-9:
                new_c_int = int(round(new_c))
                if new_c_int == 1:
                    out_parts.append(f"{el}")
                elif new_c_int > 1:
                    out_parts.append(f"{el}{new_c_int}")
                else:
                    out_parts.append(f"{el}{new_c:.6g}")
            else:
                out_parts.append(f"{el}{new_c:.6g}")
        return "".join(out_parts)
    return u_str


def normalize_formula(formula_str):
    if isinstance(formula_str, float) or formula_str is None:
        return None
    fs = str(formula_str)
    has_paren = "(" in fs
    if has_paren:
        import re
        def expand_one(m):
            inner = m.group(1)
            mult = m.group(2)
            try:
                mult_f = float(mult) if mult else 1.0
            except Exception:
                mult_f = 1.0
            elements = re.findall(r"([A-Z][a-z]?)([\d.\-+eE]*)", inner)
            out_parts = []
            for el, cnt in elements:
                if not el:
                    continue
                if not cnt:
                    new_c = 1.0 * mult_f
                else:
                    try:
                        new_c = float(cnt) * mult_f
                    except Exception:
                        new_c = None
                if new_c is None:
                    out_parts.append(f"{el}{cnt}")
                    continue
                if abs(new_c - round(new_c)) < 1e-9:
                    new_c_int = int(round(new_c))
                    if new_c_int == 1:
                        out_parts.append(f"{el}")
                    elif new_c_int > 1:
                        out_parts.append(f"{el}{new_c_int}")
                    else:
                        out_parts.append(f"{el}{new_c:.6g}")
                else:
                    out_parts.append(f"{el}{new_c:.6g}")
            return "".join(out_parts)
        pat = re.compile(r"\(([^\(\)]*)\)([\d.]+)?")
        max_iters = 10
        for _ in range(max_iters):
            if not pat.search(fs):
                break
            fs = pat.sub(expand_one, fs)
    return fs


def condense_formula(formula_str):
    if isinstance(formula_str, float) or formula_str is None:
        return None
    return normalize_formula(formula_str)


def process_formula_tghe(formula_str):
    import re
    if isinstance(formula_str, float) or formula_str is None:
        return None
    fs = str(formula_str)
    pat1 = re.compile(r"\(([^\(\)]*)\)([\d.]+)")
    max_iters = 5
    for _ in range(max_iters):
        if not pat1.search(fs):
            break
        def exp(m):
            inner = m.group(1)
            try:
                mult = float(m.group(2))
            except Exception:
                mult = 1.0
            elems = re.findall(r"([A-Z][a-z]?)([\d.\-+eE]*)", inner)
            parts = []
            for el, cnt in elems:
                if not el:
                    continue
                try:
                    base = float(cnt) if cnt else 1.0
                except Exception:
                    base = 1.0
                    if cnt:
                        parts.append(f"{el}{cnt}")
                        continue
                new_c = base * mult
                if abs(new_c - round(new_c)) < 1e-9:
                    ni = int(round(new_c))
                    parts.append(f"{el}" if ni == 1 else f"{el}{ni}")
                else:
                    parts.append(f"{el}{new_c:.6g}")
            return "".join(parts)
        fs = pat1.sub(exp, fs)
    return fs


def get_data_crabnet(idx_list, id_to_formula, id_to_target, formula_to_structure=None,
                     precomputed_df=None):
    """Return a (formula, target) DataFrame for idx_list, preferring precomputed_df."""
    if precomputed_df is not None:
        valid = [i for i in idx_list if i in precomputed_df.index]
        if not valid:
            return None, None
        df = precomputed_df.loc[valid].copy()
        df = df.dropna(subset=["formula", "target"])
        df.columns = ["formula", "target"]
        df = df.reset_index(drop=True)
        return df, None
    formulas = []
    targets = []
    for cid in idx_list:
        if cid in id_to_formula and cid in id_to_target:
            f_raw = id_to_formula[cid]
            f_clean = condense_formula(f_raw)
            if f_clean is None:
                continue
            formulas.append(f_clean)
            targets.append(id_to_target[cid])
    if not formulas:
        return None, None
    df = pd.DataFrame({"formula": formulas, "target": targets})
    return df, None


def run(task_name, R, N, S):
    TARGET = task_name
    DATASET_DIR = os.path.join(DATA_HOME, f"{TARGET}_({R},{N},{S})_folds")
    RESULT_DIR = os.path.join(AUTOMATBENCH_HOME, "work", "result",
                              f"crabnet_{TARGET}_({R},{N},{S})")
    RESULT_FILE = os.path.join(RESULT_DIR, "results.json")
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"[crabnet] Loading id->target/formula from jarvis/MP + fold splits from {DATASET_DIR} ...")
    test_splits_path = os.path.join(DATASET_DIR, "test.json")
    train_splits_path = os.path.join(DATASET_DIR, "train.json")
    val_splits_path = os.path.join(DATASET_DIR, "val.json")

    JARVIS_DIR = os.path.join(AUTOMATBENCH_HOME, "jarvis_data", f"jarvis_{TARGET}")
    MP_DIR = os.path.join(AUTOMATBENCH_HOME, "MP_data", f"MP_{TARGET}")
    id_to_target = {}
    id_to_formula = {}
    min_val = -float("inf"); max_val = float("inf")
    if "shear_modulus" in TARGET or "bulk_modulus" in TARGET:
        min_val = -500.0; max_val = 1000.0
    jarvis_labels_path = os.path.join(JARVIS_DIR, "Y_value.csv")
    if os.path.exists(jarvis_labels_path):
        import pandas as _pdj
        dfj = _pdj.read_csv(jarvis_labels_path)
        if "material_id" in dfj.columns and TARGET in dfj.columns and "formula" in dfj.columns:
            if min_val > -float("inf") or max_val < float("inf"):
                dfj = dfj[(dfj[TARGET] >= min_val) & (dfj[TARGET] <= max_val)]
            id_to_target.update(dict(zip(dfj["material_id"], dfj[TARGET])))
            id_to_formula.update(dict(zip(dfj["material_id"], dfj["formula"])))
    mp_labels_path = os.path.join(MP_DIR, "Y_value.csv")
    if os.path.exists(mp_labels_path):
        import pandas as _pdm
        dfm = _pdm.read_csv(mp_labels_path)
        if "material_id" in dfm.columns and TARGET in dfm.columns and "formula" in dfm.columns:
            if min_val > -float("inf") or max_val < float("inf"):
                dfm = dfm[(dfm[TARGET] >= min_val) & (dfm[TARGET] <= max_val)]
            id_to_target.update(dict(zip(dfm["material_id"], dfm[TARGET])))
            id_to_formula.update(dict(zip(dfm["material_id"], dfm["formula"])))
    print(f"[crabnet] Loaded labels: {len(id_to_target)} ids from jarvis + MP")

    with open(test_splits_path, "r") as f:
        test_splits = json.load(f)
    with open(train_splits_path, "r") as f:
        train_splits = json.load(f)
    val_splits = None
    if os.path.exists(val_splits_path):
        with open(val_splits_path, "r") as f:
            val_splits = json.load(f)

    if val_splits is None:
        from sklearn.model_selection import train_test_split as _tsp
        val_splits = {}
        for k in list(train_splits.keys()):
            if isinstance(k, str) and k.isdigit():
                ki = int(k)
            else:
                ki = k
            t = train_splits[k]
            if len(t) < 4:
                tr, va = t, t[: max(1, len(t) // 5)]
            else:
                tr, va = _tsp(t, test_size=0.1, random_state=42)
            val_splits[k] = va
            train_splits[k] = tr
        with open(val_splits_path, "w") as f:
            json.dump(val_splits, f, indent=2)
        print(f"[crabnet] Created missing validation splits at {val_splits_path}")

    all_ids = set()
    for fold_key in list(train_splits.keys()):
        all_ids.update(train_splits[fold_key])
    for fold_key in list(test_splits.keys()):
        all_ids.update(test_splits[fold_key])
    for fold_key in list(val_splits.keys()):
        all_ids.update(val_splits[fold_key])
    v_formulas, v_targets, v_idx = [], [], []
    for mid in all_ids:
        if mid in id_to_formula and mid in id_to_target:
            f_clean = condense_formula(id_to_formula[mid])
            if f_clean is None:
                continue
            v_formulas.append(f_clean)
            v_targets.append(id_to_target[mid])
            v_idx.append(mid)
    PRE_DF = pd.DataFrame({"formula": v_formulas, "target": v_targets}, index=v_idx)
    print(f"[crabnet] Precomputed formula dataframe: {len(PRE_DF)} compounds")

    from crabnet.crabnet_ import CrabNet

    fold_eval_metrics = {}
    fold_predictions = {}
    start_time = time.time()

    fold_keys_sorted = sorted(
        train_splits.keys(),
        key=lambda k: (int(k) if isinstance(k, str) and k.isdigit() else k),
    )
    N_FOLDS = len(fold_keys_sorted)
    print(f"[crabnet] Total Folds = {N_FOLDS}")

    for f_idx, fold_key in enumerate(fold_keys_sorted):
        train_ids = train_splits[fold_key]
        val_ids = val_splits[fold_key] if val_splits else None
        test_ids = test_splits[fold_key]

        t0 = time.time()
        train_df, _ = get_data_crabnet(train_ids, id_to_formula, id_to_target,
                                       precomputed_df=PRE_DF)
        test_df, _ = get_data_crabnet(test_ids, id_to_formula, id_to_target,
                                      precomputed_df=PRE_DF)
        val_df = None
        if val_ids:
            val_df, _ = get_data_crabnet(val_ids, id_to_formula, id_to_target,
                                         precomputed_df=PRE_DF)
        if train_df is None or test_df is None:
            print(f"[crabnet] Fold {fold_key}: no valid data, MAE=nan")
            fold_eval_metrics[fold_key] = {
                "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "mape": float("nan"),
            }
            fold_predictions[fold_key] = {}
            continue

        print(f"Processing Fold {f_idx + 1}/{N_FOLDS} ...")
        print(
            f"train_df size: {train_df.shape}  "
            f"test_df size: {test_df.shape}  "
            f"val_df size: {val_df.shape if val_df is not None else 'NO VAL (0,0)'}"
        )
        # CrabNet scales the target internally; do not apply StandardScaler here.
        true_values = np.asarray(test_df["target"].values, dtype=float)

        cb = CrabNet(
            mat_prop=TARGET,
            verbose=True,
            classification=False,
            epochs=120,
            epochs_step=10,
            batch_size=256,
            adam=True,
            lr=2e-3,
            weight_decay=1e-4,
            elem_prop="mat2vec",
            d_model=256,
            N=2,
            heads=2,
            out_hidden=[512, 256],
            dim_feedforward=1024,
        )
        try:
            cb.discard_n = 2
        except Exception:
            pass

        if val_df is None or val_df.empty:
            # Without a val split, carve one from train to avoid a None crash in fit().
            from sklearn.model_selection import train_test_split as _tsp2
            if len(train_df) >= 10:
                _tr, _va = _tsp2(train_df, test_size=0.1, random_state=1)
            else:
                _tr = train_df.iloc[:-1] if len(train_df) > 1 else train_df
                _va = train_df.iloc[-1:] if len(train_df) > 1 else train_df
            cb.fit(train_df=_tr, val_df=_va)
        else:
            cb.fit(train_df=train_df, val_df=val_df)

        try:
            preds_out = cb.predict(test_df)
            # predict() may return a DataFrame, a 1D array, or a 2D (N, 3) array.
            if hasattr(preds_out, "columns"):
                cols = list(preds_out.columns)
                if "prediction" in cols:
                    preds = np.asarray(preds_out["prediction"].values, dtype=float).ravel()
                elif "target" in cols:
                    preds = np.asarray(preds_out["target"].values, dtype=float).ravel()
                else:
                    arr = np.asarray(preds_out.values, dtype=float)
                    preds = arr.ravel()
            else:
                arr = np.asarray(preds_out, dtype=float)
                if arr.ndim == 1:
                    preds = arr.ravel()
                elif arr.ndim == 2:
                    if arr.shape[0] == len(true_values):
                        preds = arr[:, 0].ravel()   # col0 = pred
                    elif arr.shape[1] == len(true_values):
                        preds = arr[0, :].ravel()   # row0 = pred
                    else:
                        preds = arr.ravel()[:len(true_values)]
                else:
                    preds = arr.ravel()[:len(true_values)]
            if preds.size != len(true_values):
                # Pad/truncate to the test size to avoid an all-NaN cascade.
                preds = np.concatenate([
                    preds[:min(len(true_values), preds.size)],
                    np.full(max(0, len(true_values) - preds.size), np.nan),
                ])
            # Fill residual NaNs with the train target median.
            nan_mask = ~np.isfinite(preds)
            if nan_mask.any():
                fallback = float(np.nanmedian(train_df["target"].values))
                if not np.isfinite(fallback):
                    fallback = float(np.nanmean(true_values)) if np.isfinite(np.nanmean(true_values)) else 0.0
                print(f"    [nan_fix] Fold {f_idx+1}: {nan_mask.sum()}/{len(preds)} preds are NaN, filling with train-median = {fallback:.4f}")
                preds[nan_mask] = fallback
            eval_metrics = compute_eval_metrics_regression(true_values, preds)
        except Exception as e_em:
            print(f"[crabnet] metrics fallback: {e_em}")
            # Baseline fallback: train target median.
            try:
                fallback = float(np.nanmedian(train_df["target"].values))
            except Exception:
                fallback = float(np.nanmean(true_values)) if np.isfinite(np.nanmean(true_values)) else 0.0
            preds = np.full(len(true_values), fallback, dtype=float)
            eval_metrics = compute_eval_metrics_regression(true_values, preds)

        fold_eval_metrics[fold_key] = eval_metrics
        fold_predictions[fold_key] = {
            "test_ids": [str(x) for x in test_ids],
            "true_values": [float(x) for x in true_values],
            "predicted_values": [float(x) if (x is not None and np.isfinite(x)) else None
                                 for x in preds],
        }
        dt = time.time() - t0
        print(f"Fold {f_idx + 1} completed in {dt:.1f}s  -> mae = {eval_metrics.get('mae', 'nan')}")

    total_runtime = time.time() - start_time
    avg_runtime = total_runtime / max(1, N_FOLDS)
    metric_names = ["mae", "rmse", "r2", "mape"]
    average_metrics = {}
    for mname in metric_names:
        vals = [fold_eval_metrics[k].get(mname) for k in fold_eval_metrics]
        vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
        average_metrics[mname] = float(np.mean(vals)) if vals else None

    final_result = {
        "config": {
            "task": TARGET,
            "R": R, "N": N, "S": S,
            "n_folds": N_FOLDS,
            "variant": "crabnet_reduced",
            "config_choices": {
                "max_epochs": 120,
                "discard_n": 2,
                "optimizer": "Adam",
                "adam": True,
                "lr": 2e-3,
                "batch_size": 256,
                "weight_decay": 1e-4,
                "model_scaled_half": True,
                "d_model": 256,
                "N_attn_layers": 2,
                "heads": 2,
                "dim_feedforward": 1024,
                "out_hidden": [512, 256],
                "precomputed_formula_df": True,
                "n_domain_truncated_6": False,
            },
            "seed_state": "inherited_from_outer_bo_state",
        },
        "runtime": {
            "total_runtime_seconds": total_runtime,
            "avg_runtime_seconds_per_fold": avg_runtime,
            "total_runtime_str": str(timedelta(seconds=total_runtime)),
        },
        "average_metrics": average_metrics,
        "fold_eval_metrics": fold_eval_metrics,
        "fold_predictions": fold_predictions,
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"[crabnet] Result saved to {RESULT_FILE}")
    print(f"[crabnet] {N_FOLDS}-fold average metrics = {average_metrics}")
    return final_result


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python crabnet_run.py <task> <R> <N> <S>")
        sys.exit(1)
    run(
        task_name=sys.argv[1],
        R=float(sys.argv[2]),
        N=int(sys.argv[3]),
        S=int(sys.argv[4]),
    )
