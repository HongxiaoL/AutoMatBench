import os
import json
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from crabnet.crabnet_ import CrabNet
from common.logging_utils import setup_dual_logging

# Add parent directory to path to import common.params
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from common import params

model_name = 'crabnet'
log_dir = os.path.join(parent_dir, 'work/log')
log_file = os.path.join(log_dir, model_name + '_execution.log')

def setup_logging():
    setup_dual_logging(log_file, level=logging.INFO, force=True, reset_handlers=True)
    logging.info("Initializing CrabNet script...")

setup_logging()

from sklearn.preprocessing import StandardScaler

# Helper functions
def condense_formula(mat):
    if isinstance(mat, str):
        return mat
    try:
        return mat.formula.replace(' ', '')
    except:
        return str(mat)

def get_data_crabnet(ids, id_to_formula, id_to_target):
    formulas = []
    targets = []
    
    for mat_id in ids:
        if mat_id in id_to_formula and mat_id in id_to_target:
            formulas.append(condense_formula(id_to_formula[mat_id]))
            targets.append(id_to_target[mat_id])
            
    return pd.DataFrame({'formula': formulas, 'target': targets})

def robust_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-5
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def run(task_name, R, N, S):
    if N == 60 and S in [9, 10]:
        return

    param_str = f"({R},{N},{S})"
    folder_name = f"{task_name}_{param_str}_folds"
    data_dir = os.path.join(parent_dir, "work/data", folder_name)
    result_dir = os.path.join(parent_dir, "work/result", model_name + '_' + folder_name.rsplit('_', 1)[0])
    result_file = os.path.join(result_dir, "results.json")

    if not os.path.exists(data_dir):
        return
    if os.path.exists(result_file):
        logging.info(f"Task {folder_name} already completed. Skipping.")
        return

    logging.info(f"Starting task: {folder_name}")
    os.makedirs(result_dir, exist_ok=True)

    TARGET = task_name
    # Assuming automatbench is parallel to work directory
    automatbench_dir = parent_dir
    JARVIS_DIR = os.path.join(automatbench_dir, "jarvis_data", f"jarvis_{task_name}")
    MP_DIR = os.path.join(automatbench_dir, "MP_data", f"MP_{task_name}")

    # Load targets and formulas
    id_to_target = {}
    id_to_formula = {}

    # Define value ranges for filtering outliers
    # Safe ranges: 
    # shear_modulus: -500 to 1000 GPa
    # bulk_modulus: -500 to 1000 GPa
    # Others: usually safe, but let's be careful.
    min_val = -float('inf')
    max_val = float('inf')
    
    if "shear_modulus" in task_name or "bulk_modulus" in task_name:
        min_val = -500.0
        max_val = 1000.0
        logging.info(f"Filtering {task_name} values outside [{min_val}, {max_val}]")

    # Jarvis
    jarvis_labels_path = os.path.join(JARVIS_DIR, "Y_value.csv")
    if os.path.exists(jarvis_labels_path):
        df_jarvis = pd.read_csv(jarvis_labels_path)
        if 'material_id' in df_jarvis.columns and TARGET in df_jarvis.columns and 'formula' in df_jarvis.columns:
            # Filter DataFrame first
            if min_val > -float('inf') or max_val < float('inf'):
                original_len = len(df_jarvis)
                df_jarvis = df_jarvis[(df_jarvis[TARGET] >= min_val) & (df_jarvis[TARGET] <= max_val)]
                logging.info(f"Filtered JARVIS data from {original_len} to {len(df_jarvis)}")
            
            id_to_target.update(dict(zip(df_jarvis["material_id"], df_jarvis[TARGET])))
            id_to_formula.update(dict(zip(df_jarvis["material_id"], df_jarvis["formula"])))

    # MP
    mp_labels_path = os.path.join(MP_DIR, "Y_value.csv")
    if os.path.exists(mp_labels_path):
        df_mp = pd.read_csv(mp_labels_path)
        if 'material_id' in df_mp.columns and TARGET in df_mp.columns and 'formula' in df_mp.columns:
            # Filter DataFrame first
            if min_val > -float('inf') or max_val < float('inf'):
                original_len = len(df_mp)
                df_mp = df_mp[(df_mp[TARGET] >= min_val) & (df_mp[TARGET] <= max_val)]
                logging.info(f"Filtered MP data from {original_len} to {len(df_mp)}")

            id_to_target.update(dict(zip(df_mp["material_id"], df_mp[TARGET])))
            id_to_formula.update(dict(zip(df_mp["material_id"], df_mp["formula"])))

    logging.info(f"Loaded {len(id_to_target)} targets and formulas.")

    with open(os.path.join(data_dir, "train.json"), "r") as f:
        train_splits = json.load(f)
    with open(os.path.join(data_dir, "test.json"), "r") as f:
        test_splits = json.load(f)
    
    val_splits_path = os.path.join(data_dir, "val.json")
    val_splits = None
    if os.path.exists(val_splits_path):
        with open(val_splits_path, "r") as f:
            val_splits = json.load(f)
    else:
        logging.warning(f"No validation split found for {folder_name}")

    results_folds = {}
    actual_folds = len(train_splits.keys())

    for fold_idx in range(actual_folds):
        fold_key = str(fold_idx)
        if fold_key not in train_splits: continue

        try:
            logging.info(f"Processing Fold {fold_idx}...")
            train_ids = train_splits[fold_key]
            test_ids = test_splits[fold_key]

            train_df = get_data_crabnet(train_ids, id_to_formula, id_to_target)
            test_df = get_data_crabnet(test_ids, id_to_formula, id_to_target)
            
            # Filter NaNs (might happen if IDs were filtered out)
            train_df = train_df.dropna(subset=['target'])
            test_df = test_df.dropna(subset=['target'])

            val_df = pd.DataFrame()
            if val_splits and fold_key in val_splits:
                val_ids = val_splits[fold_key]
                val_df = get_data_crabnet(val_ids, id_to_formula, id_to_target)
                val_df = val_df.dropna(subset=['target'])
                logging.info(f"Loaded {len(val_df)} validation samples.")
            
            if val_df.empty:
                logging.warning("Using train_test_split as fallback for validation.")
                val_df = train_df.sample(frac=0.1, random_state=42)
                train_df = train_df.drop(val_df.index)

            logging.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

            if train_df.empty or test_df.empty:
                logging.warning("Empty train or test set.")
                continue

            # Scaling
            scaler = StandardScaler()
            # Fit on training data only
            train_df['target'] = scaler.fit_transform(train_df[['target']])
            val_df['target'] = scaler.transform(val_df[['target']])
            # Do NOT scale test data target here, we need it for comparison
            # But for prediction input? CrabNet takes 'formula' and 'target' (dummy).
            
            # Training CrabNet
            # Use validation set for early stopping (implicit in passing val_df)
            cb = CrabNet(mat_prop=TARGET, verbose=True, classification=False)
            cb.fit(train_df=train_df, val_df=val_df)

            # Predicting
            test_df_for_pred = test_df.copy()
            test_df_for_pred['target'] = np.nan
            
            predictions_output = cb.predict(test_df_for_pred)
            
            # Extract predictions
            if hasattr(predictions_output, 'columns') and 'prediction' in predictions_output.columns:
                predictions_scaled = predictions_output['prediction'].values
            else:
                predictions_scaled = predictions_output
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

            targets_test = test_df['target'].values
            
            mae = float(mean_absolute_error(targets_test, predictions))
            rmse = float(np.sqrt(mean_squared_error(targets_test, predictions)))
            mape = float(robust_mape(targets_test, predictions))
            max_err_val = float(max_error(targets_test, predictions))

            results_folds[f"fold_{fold_idx}"] = {
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "max_error": max_err_val,
                "predictions": predictions.tolist(),
                "targets": targets_test.tolist(),
            }
            logging.info(f"Fold {fold_idx} MAE: {mae}, MAPE: {mape}")
            
            # Save partial results
            avg_metrics = {
                "mae": float(np.mean([res["mae"] for res in results_folds.values()])),
                "rmse": float(np.mean([res["rmse"] for res in results_folds.values()])),
                "mape": float(np.mean([res["mape"] for res in results_folds.values()])),
                "max_error": float(np.max([res["max_error"] for res in results_folds.values()]))
            }
            final_output = {
                "task": task_name,
                "method": "CrabNet",
                "params": {"R": R, "N": N, "S": S},
                "fold_results": results_folds,
                "average_metrics": avg_metrics
            }
            with open(result_file, "w") as f:
                json.dump(final_output, f, indent=4)
            logging.info(f"Saved partial results to {result_file}")

        except Exception as e:
            logging.error(f"Error in Fold {fold_idx}: {e}", exc_info=True)

    if results_folds:
        avg_metrics = {
            "mae": float(np.mean([res["mae"] for res in results_folds.values()])),
            "rmse": float(np.mean([res["rmse"] for res in results_folds.values()])),
            "mape": float(np.mean([res["mape"] for res in results_folds.values()])),
            "max_error": float(np.max([res["max_error"] for res in results_folds.values()]))
        }
        final_output = {
            "task": task_name,
            "method": "CrabNet",
            "params": {"R": R, "N": N, "S": S},
            "fold_results": results_folds,
            "average_metrics": avg_metrics
        }
        with open(result_file, "w") as f:
            json.dump(final_output, f, indent=4)
        logging.info(f"Saved results to {result_file}")

if __name__ == "__main__":
    R_values = params.R_values
    N_values = params.N_values
    S_values = params.S_values
    tasks = params.tasks

    for task_name in tasks:
        for R in R_values:
            for N in N_values:
                for S in S_values:
                    run(task_name, R, N, S)
