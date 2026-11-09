import os
import sys
import time
import traceback
from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging
import itertools
import pickle

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from common import params

model_name = 'rfscm'
log_dir = os.path.join(parent_dir, 'work/log')
log_file = os.path.join(log_dir, model_name + '_execution.log')
os.makedirs(log_dir, exist_ok=True)

original_cwd = os.getcwd()
try:
    os.chdir(log_dir)
    from automatminer import MatPipe
    from automatminer.automl.adaptors import SinglePipelineAdaptor
    from automatminer.featurization import AutoFeaturizer
    from automatminer.preprocessing import DataCleaner, FeatureReducer
    try:
        from automatminer.utils.log import initialize_logger
        initialize_logger("automatminer", log_dir=log_dir)
    except ImportError:
        pass
finally:
    os.chdir(original_cwd)

def setup_logging(log_console_stdout=None):
    class FlushingStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
                self.flush()
            except Exception:
                pass

    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            try:
                super().emit(record)
                self.flush()
            except Exception:
                pass

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)

    if log_console_stdout is None:
        console_handler = FlushingStreamHandler(sys.stdout)
    else:
        console_handler = FlushingStreamHandler(log_console_stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = FlushingFileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(fh)
    def flush_all():
        try:
            sys.stdout.flush()
            for h in root_logger.handlers:
                h.flush()
        except Exception:
            pass
    import atexit; atexit.register(flush_all)
    root_logger._flush_all = flush_all

setup_logging()

from pymatgen.core import Structure
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import argparse
import threading

def start_fit_heartbeat(log_prefix, fold_idx, stop_event, interval_s=5, est_per_fit_s=None):
    t0 = time.time()
    import logging as _lg
    def _run():
        last_log = 0
        first = True
        while not stop_event.is_set():
            time.sleep(min(0.5, interval_s))
            if stop_event.is_set():
                break
            now = time.time()
            if now - last_log >= interval_s or first:
                first = False
                last_log = now
                el = now - t0
                line = f"{log_prefix}Fit heartbeat F{fold_idx} elapsed={el:.0f}s"
                if est_per_fit_s and est_per_fit_s > 0:
                    pct = min(99.9, 100.0 * el / est_per_fit_s)
                    rem = max(0, est_per_fit_s - el)
                    line += f" ({pct:4.1f}%) ≈remaining {rem:.0f}s (~{rem/60:.1f}m)"
                _lg.info(line)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error
except ImportError:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    try:
        from sklearn.metrics import max_error
    except ImportError:
        def max_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.max(np.abs(y_true - y_pred))

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

N_JOBS = int(os.environ.get("N_JOBS", "10"))
RANDOM_STATE = 18012019

def get_data(ids, all_targets, jarvis_dir, mp_dir, target_col, log_prefix=""):
    import time as _time
    structures = []
    targets = []
    valid_ids = []
    total = len(ids)
    t0 = _time.time()
    next_report = 5
    prog = 0
    miss = 0
    for i, mat_id in enumerate(ids, 1):
        if mat_id not in all_targets:
            continue

        target_val = all_targets[mat_id]

        if mat_id.startswith("JVASP-"):
            file_path = os.path.join(jarvis_dir, "jarvis_data", f"{mat_id}.cif")
        elif mat_id.startswith("mp-") or mat_id.startswith("mvc-"):
            file_path = os.path.join(mp_dir, "MP_data", f"{mat_id}.cif")
        else:
            logging.warning(f"Unknown ID format {mat_id}, skipping.")
            continue

        if not os.path.exists(file_path):
             miss += 1
             continue

        try:
            struct = Structure.from_file(file_path)
            structures.append(struct)
            targets.append(target_val)
            valid_ids.append(mat_id)
            prog += 1
        except Exception as e:
            logging.error(f"Error loading {mat_id}: {e}")
            continue

        elapsed = _time.time() - t0
        if elapsed >= next_report:
            pct = 100.0 * i / max(total, 1)
            logging.info(f"{log_prefix}Load CIF {i}/{total} ({pct:4.1f}%) ok={prog} miss={miss} rate={prog/max(elapsed,1e-3):.1f}/s elapsed={elapsed:.0f}s")
            next_report = int(elapsed) + 5

    df = pd.DataFrame({
        "structure": structures,
        target_col: targets
    }, index=valid_ids)

    elapsed = _time.time() - t0
    logging.info(f"{log_prefix}Load CIF done {prog}/{total} miss={miss} elapsed={elapsed:.1f}s avg={prog/max(elapsed,1e-3):.1f}/s -> df shape {df.shape}")
    return df


def run(task_name, R, N, S):
    if N == 60 and S in [9, 10]:
        return

    param_str = f"({R},{N},{S})"
    folder_name = f"{task_name}_{param_str}_folds"
    data_dir = os.path.join(parent_dir, "work/data", folder_name)
    result_dir = os.path.join(parent_dir, "work/result", model_name + '_' + folder_name.rsplit('_', 1)[0])
    result_file = os.path.join(result_dir, "results.json")

    if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, "train.json")):
        logging.warning(f"Data directory {data_dir} does not exist OR missing train.json. Auto-generating splits via process_data.process_data(task={task_name}, R={R}, N={N}, S={S}) ...")
        try:
            import shutil as _shutil
            if os.path.exists(data_dir) and not os.path.exists(os.path.join(data_dir, "train.json")):
                _shutil.rmtree(data_dir, ignore_errors=True)
                logging.warning(f"Cleaned stale/incomplete data dir: {data_dir}")
            import process_data as _pd
            _pd.process_data(task_name, R, N, S)
        except Exception as _e:
            logging.error(f"Failed to auto-generate splits: {_e}. Traceback:\n{traceback.format_exc()}")
            return
        if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, "train.json")):
            logging.warning(f"Still missing {data_dir}/train.json after auto-generate attempt. Skipping.")
            return
        else:
            logging.info(f"Auto-generated data dir OK: {data_dir}")

    if os.path.exists(result_file):
        logging.info(f"Task {folder_name} already completed. Skipping.")
        return

    logging.info(f"Starting task: {folder_name} | n_jobs={N_JOBS}")

    try:
        TARGET = task_name

        JARVIS_DIR = os.path.join(parent_dir, "jarvis_data", f"jarvis_{task_name}")
        MP_DIR = os.path.join(parent_dir, "MP_data", f"MP_{task_name}")

        logging.info(f"Using JARVIS_DIR: {JARVIS_DIR}")
        logging.info(f"Using MP_DIR: {MP_DIR}")

        os.makedirs(result_dir, exist_ok=True)

        logging.info("Loading labels...")
        jarvis_labels_path = os.path.join(JARVIS_DIR, "Y_value.csv")
        if os.path.exists(jarvis_labels_path):
            df_jarvis = pd.read_csv(jarvis_labels_path)
            jarvis_targets = dict(zip(df_jarvis["material_id"], df_jarvis[TARGET]))
        else:
            logging.warning(f"JARVIS labels not found at {jarvis_labels_path}")
            jarvis_targets = {}

        mp_labels_path = os.path.join(MP_DIR, "Y_value.csv")
        if os.path.exists(mp_labels_path):
            df_mp = pd.read_csv(mp_labels_path)
            mp_targets = dict(zip(df_mp["material_id"], df_mp[TARGET]))
        else:
            logging.warning(f"MP labels not found at {mp_labels_path}")
            mp_targets = {}

        all_targets = {**jarvis_targets, **mp_targets}
        logging.info(f"Loaded {len(all_targets)} targets.")

        logging.info("Loading splits...")
        with open(os.path.join(data_dir, "train.json"), "r") as f:
            train_splits = json.load(f)
        with open(os.path.join(data_dir, "test.json"), "r") as f:
            test_splits = json.load(f)

        val_path = os.path.join(data_dir, "val.json")
        if os.path.exists(val_path):
            logging.info("Loading validation splits...")
            with open(val_path, "r") as f:
                val_splits = json.load(f)
        else:
            val_splits = {}

        actual_folds = len(train_splits.keys())
        logging.info(f"Loaded {actual_folds}-fold split keys.")

        # Featurize the union of all fold ids once; target-agnostic to avoid leakage.
        cache_pkl = os.path.join(data_dir, f"featurized_all_{model_name}.pkl")
        cache_meta = os.path.join(data_dir, f"featurized_all_{model_name}_meta.json")
        need_cache = True
        df_all_feat = None
        if os.path.exists(cache_pkl) and os.path.exists(cache_meta):
            try:
                with open(cache_meta, "r") as f:
                    meta = json.load(f)
                if meta.get("n_jobs") == N_JOBS and meta.get("task") == task_name and meta.get("param") == param_str:
                    t_load = time.time()
                    df_all_feat = pd.read_pickle(cache_pkl)
                    logging.info(f"[CACHE] loaded {cache_pkl} shape={df_all_feat.shape} in {(time.time()-t_load):.1f}s (n_jobs={N_JOBS})")
                    need_cache = False
            except Exception as e:
                logging.warning(f"[CACHE] load failed: {e}, re-generating.")

        if need_cache:
            logging.info("[CACHE] collecting all material ids across all folds...")
            all_ids = []
            seen = set()
            for fold_idx in range(actual_folds):
                fk = str(fold_idx)
                ids_list = []
                ids_list += train_splits.get(fk, [])
                ids_list += test_splits.get(fk, [])
                ids_list += val_splits.get(fk, [])
                for mid in ids_list:
                    if mid not in seen:
                        seen.add(mid)
                        all_ids.append(mid)
            logging.info(f"[CACHE] total unique ids: {len(all_ids)} (across {actual_folds} folds train/test/val)")

            t_all_load = time.time()
            df_all_raw = get_data(all_ids, all_targets, JARVIS_DIR, MP_DIR, TARGET, log_prefix="[ALL] ")
            logging.info(f"[CACHE] raw data loaded {len(df_all_raw)}/{len(all_ids)} in {(time.time()-t_all_load)/60:.1f} min")

            # Drop NaN targets before featurization so indices stay aligned with the cache.
            df_all_raw_wnan = df_all_raw.dropna(subset=[TARGET])
            logging.info(f"[CACHE] raw with targets: {len(df_all_raw_wnan)} rows")

            autofeat = AutoFeaturizer(n_jobs=N_JOBS, preset="debug")
            cwd_before = os.getcwd()
            try:
                os.chdir(log_dir)
                cache_stop = threading.Event()
                est_all_s = int(max(60, 14 * 60 * (len(df_all_raw_wnan) / max(1, 479))))
                hb_cache = start_fit_heartbeat("[CACHE] ", -1, cache_stop, interval_s=10, est_per_fit_s=est_all_s)
                t_af = time.time()
                # AutoFeaturizer.fit_transform needs TARGET to decide regression vs classification.
                df_all_feat = autofeat.fit_transform(df_all_raw_wnan, TARGET)
                cache_stop.set()
                hb_cache.join(timeout=3)
                logging.info(f"[CACHE] AutoFeaturizer DONE in {(time.time()-t_af)/60:.1f} min | out shape={df_all_feat.shape}")
            finally:
                os.chdir(cwd_before)

            t_write = time.time()
            df_all_feat.to_pickle(cache_pkl, protocol=4)
            with open(cache_meta, "w") as f:
                json.dump({
                    "task": task_name, "param": param_str,
                    "n_jobs": N_JOBS,
                    "n_ids_raw": len(df_all_raw_wnan),
                    "n_feats": df_all_feat.shape[1],
                    "columns": list(df_all_feat.columns.astype(str)),
                    "index_material_ids": [str(x) for x in df_all_feat.index.tolist()],
                    "af_seconds": time.time() - t_af,
                }, f, indent=2)
            logging.info(f"[CACHE] saved {cache_pkl} size={os.path.getsize(cache_pkl)/1024/1024:.1f}MB in {(time.time()-t_write):.1f}s")

        # DataCleaner.fit_transform later requires the target column to be present.
        if TARGET not in df_all_feat.columns:
            logging.error(f"[FATAL] target column {TARGET!r} missing from featurized df. cols[:10]={list(df_all_feat.columns.astype(str))[:10]}")
            sys.exit(2)

        min_val = -float('inf')
        max_val = float('inf')
        if "shear_modulus" in task_name or "bulk_modulus" in task_name:
            min_val = -500.0
            max_val = 1000.0
            logging.info(f"Filtering {task_name} values outside [{min_val}, {max_val}]")

        dc_cfg = dict(feature_na_method="mean", max_na_frac=0.01,
                       na_method_fit="mean", na_method_transform="mean",
                       drop_na_targets=False)

        logging.info(f"Starting {actual_folds}-fold Cross Validation...")
        results = {}

        for fold_idx in range(actual_folds):
            fold_key = str(fold_idx)
            if fold_key not in train_splits or fold_key not in test_splits:
                logging.warning(f"Fold {fold_idx} not found in splits, skipping.")
                continue

            logging.info(f"Processing Fold {fold_idx}...")

            train_ids = list(train_splits[fold_key])
            test_ids = list(test_splits[fold_key])
            val_ids = list(val_splits.get(fold_key, []))
            if val_ids:
                logging.info(f"Adding {len(val_ids)} validation samples to training set.")
                train_ids = train_ids + val_ids

            # Cache index holds material_id strings.
            all_idx_set = set(df_all_feat.index.astype(str).tolist())
            train_idx_ok = [mid for mid in train_ids if str(mid) in all_idx_set]
            test_idx_ok  = [mid for mid in test_ids  if str(mid) in all_idx_set]
            missing_train = len(train_ids) - len(train_idx_ok)
            missing_test  = len(test_ids)  - len(test_idx_ok)
            if missing_train or missing_test:
                logging.warning(f"[F{fold_idx}] missing from cache: train {missing_train}/{len(train_ids)}, test {missing_test}/{len(test_ids)} (cif load miss earlier or target NaN)")

            df_train_feat = df_all_feat.loc[[str(x) for x in train_idx_ok]].copy()
            df_test_feat  = df_all_feat.loc[[str(x) for x in test_idx_ok]].copy()

            if len(df_train_feat) == 0 or len(df_test_feat) == 0:
                logging.warning(f"[F{fold_idx}] train/test empty after cache join. skipping.")
                continue

            df_train_feat = df_train_feat.dropna(subset=[TARGET])
            df_test_feat  = df_test_feat.dropna(subset=[TARGET])

            if min_val > -float('inf') or max_val < float('inf'):
                len_train_before = len(df_train_feat)
                df_train_feat = df_train_feat[(df_train_feat[TARGET] >= min_val) & (df_train_feat[TARGET] <= max_val)]
                if len(df_train_feat) < len_train_before:
                     logging.info(f"[F{fold_idx}] Filtered {len_train_before - len(df_train_feat)} training samples (outliers).")

            logging.info(f"Train size: {len(df_train_feat)}  Test size: {len(df_test_feat)} | cache shapes train={df_train_feat.shape} test={df_test_feat.shape}")

            if len(df_train_feat) == 0 or len(df_test_feat) == 0:
                logging.warning("Empty train or test set, skipping fold.")
                continue

            try:
                cwd_before = os.getcwd()
                try:
                    os.chdir(log_dir)
                    # Fresh objects per fold prevent leakage across folds.
                    dc = DataCleaner(**dc_cfg)
                    fr = FeatureReducer(reducers=[])
                    learner = SinglePipelineAdaptor(
                        regressor=RandomForestRegressor(n_estimators=500, n_jobs=N_JOBS, random_state=RANDOM_STATE),
                        classifier=RandomForestClassifier(n_estimators=500, n_jobs=N_JOBS, random_state=RANDOM_STATE),
                    )

                    fit_stop = threading.Event()
                    est_fit_s = max(60, 180)
                    hb_thread = start_fit_heartbeat(f"[F{fold_idx}] ", fold_idx, fit_stop, interval_s=5, est_per_fit_s=est_fit_s)
                    _t_fit = time.time()

                    X_tr = dc.fit_transform(df_train_feat.copy(), TARGET)
                    X_tr = fr.fit_transform(X_tr.copy(), TARGET)
                    learner.fit(X_tr.copy(), TARGET)

                    fit_stop.set()
                    hb_thread.join(timeout=2)
                    _et_fit = time.time() - _t_fit
                    logging.info(f"[F{fold_idx}] Fitting (DC+FR+RF) done in {_et_fit/60:.2f} min ({_et_fit:.0f}s) | X_tr shape={X_tr.shape}")

                    logging.info(f"[F{fold_idx}] Predicting...")
                    pred_stop = threading.Event()
                    est_pred_s = max(15, int(_et_fit * 0.08))
                    hb_thread2 = start_fit_heartbeat(f"[F{fold_idx}] ", fold_idx, pred_stop, interval_s=5, est_per_fit_s=est_pred_s)
                    hb_thread2.name = f"hb_pred_F{fold_idx}"
                    _t_pred = time.time()

                    X_te = dc.transform(df_test_feat.copy(), TARGET)
                    X_te = fr.transform(X_te.copy(), TARGET)
                    df_pred = learner.predict(X_te.copy(), TARGET)

                    pred_stop.set()
                    hb_thread2.join(timeout=2)
                    _et_pred = time.time() - _t_pred
                    logging.info(f"[F{fold_idx}] Predict done in {_et_pred:.1f}s | X_te shape={X_te.shape}")
                finally:
                    os.chdir(cwd_before)

                pred_col = f"{TARGET} predicted"
                if pred_col not in df_pred.columns:
                    logging.error(f"[F{fold_idx}] predicted column {pred_col} not found in learner output. cols={list(df_pred.columns.astype(str))[:8]}")
                    continue

                predictions = df_pred[pred_col]
                y_true = df_test_feat[TARGET]
                common_idx = y_true.index.intersection(predictions.index)
                if len(common_idx) == 0:
                    logging.warning(f"[F{fold_idx}] 0 common ids between y_true and predictions; skipping.")
                    continue
                y_true = y_true.loc[common_idx]
                y_pred = predictions.loc[common_idx]

                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mape = float(mean_absolute_percentage_error(y_true, y_pred))
                max_err_val = float(max_error(y_true, y_pred))

                results[f"fold_{fold_idx}"] = {
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "max_error": max_err_val,
                    "predictions": y_pred.tolist(),
                    "targets": y_true.tolist()
                }
                logging.info(f"Fold {fold_idx} completed. MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            except Exception as e:
                logging.error(f"Error in fold {fold_idx}: {e}")
                import traceback
                logging.error(traceback.format_exc())

        if results:
            avg_metrics = {
                "mae": float(np.mean([res["mae"] for res in results.values()])),
                "rmse": float(np.mean([res["rmse"] for res in results.values()])),
                "mape": float(np.mean([res["mape"] for res in results.values()])),
                "max_error": float(np.max([res["max_error"] for res in results.values()]))
            }

            final_output = {
                "task": task_name,
                "method": "RandomForest",
                "variant": "rfscm (single outer AutoFeaturizer cache + per-fold DC/FR/RF)",
                "params": {"R": R, "N": N, "S": S, "n_jobs": N_JOBS},
                "successful_folds": len(results),
                "fold_results": results,
                "average_metrics": avg_metrics,
                "cache_pkl": cache_pkl,
            }

            logging.info(f"Saving results to {result_file}...")
            with open(result_file, "w") as f:
                json.dump(final_output, f, indent=4)

            logging.info(f"Done processing {folder_name}. successful_folds={len(results)}/{actual_folds}")
            logging.info(f"Average MAE: {avg_metrics['mae']:.4f}")
            logging.info(f"Average RMSE: {avg_metrics['rmse']:.4f}")
        else:
            logging.warning(f"No successful folds for {folder_name}.")

    except Exception as e:
        logging.error(f"Critical error processing task {folder_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    R_values = params.R_values
    N_values = params.N_values
    S_values = params.S_values
    tasks = params.tasks

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, help="Run a single task, e.g. band_gap; if unspecified, run all in tasks")
    parser.add_argument("--R", type=float, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--S", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=None, help="Override N_JOBS. If unspecified, read the N_JOBS environment variable; otherwise default to 10")
    args = parser.parse_args()

    if args.n_jobs is not None:
        N_JOBS = args.n_jobs
        os.environ["N_JOBS"] = str(N_JOBS)

    if args.task is not None and args.R is not None and args.N is not None and args.S is not None:
        run(args.task, args.R, args.N, args.S)
    else:
        for task_name in tasks:
            for R in R_values:
                for N in N_values:
                    for S in S_values:
                        run(task_name, R, N, S)
