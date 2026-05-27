import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging
from common.logging_utils import setup_dual_logging

# Setup paths
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from common import params

model_name = 'rfscm'
log_dir = os.path.join(parent_dir, 'work/log')
log_file = os.path.join(log_dir, model_name + '_execution.log')
os.makedirs(log_dir, exist_ok=True)

# Context manager for imports to force automatminer logs to log_dir
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

def setup_logging():
    setup_dual_logging(log_file, level=logging.INFO, reset_handlers=True)

setup_logging()

# Import automatminer related modules
# We need to ensure logging setup is done or at least directory exists if possible
# But since this is a module, imports happen on load.
# We'll stick to standard imports but use the re-init in run() or setup_logging()

from pymatgen.core import Structure
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

# Common Configuration
N_JOBS = 10
RANDOM_STATE = 18012019

def get_data(ids, all_targets, jarvis_dir, mp_dir, target_col):
    structures = []
    targets = []
    valid_ids = []
    
    for mat_id in ids:
        if mat_id not in all_targets:
            continue
            
        target_val = all_targets[mat_id]
        
        # Determine file path
        if mat_id.startswith("JVASP-"):
            file_path = os.path.join(jarvis_dir, "jarvis_data", f"{mat_id}.cif")
        elif mat_id.startswith("mp-") or mat_id.startswith("mvc-"):
            file_path = os.path.join(mp_dir, "MP_data", f"{mat_id}.cif")
        else:
            logging.warning(f"Unknown ID format {mat_id}, skipping.")
            continue
            
        if not os.path.exists(file_path):
             continue
             
        try:
            struct = Structure.from_file(file_path)
            structures.append(struct)
            targets.append(target_val)
            valid_ids.append(mat_id)
        except Exception as e:
            logging.error(f"Error loading {mat_id}: {e}")
            continue
            
    df = pd.DataFrame({
        "structure": structures,
        target_col: targets
    }, index=valid_ids)
    
    return df

def run(task_name, R, N, S):
    if N == 60 and S in [9, 10]:
        return

    param_str = f"({R},{N},{S})"
    folder_name = f"{task_name}_{param_str}_folds"
    data_dir = os.path.join(parent_dir, "work/data", folder_name)
    result_dir = os.path.join(parent_dir, "work/result", model_name + '_' + folder_name.rsplit('_', 1)[0])
    result_file = os.path.join(result_dir, "results.json")

    if not os.path.exists(data_dir):
        logging.warning(f"Data directory {data_dir} does not exist. Skipping.")
        return

    if os.path.exists(result_file):
        logging.info(f"Task {folder_name} already completed. Skipping.")
        return

    logging.info(f"Starting task: {folder_name}")
    
    try:
        # Task specific configuration
        TARGET = task_name
        
        # Use relative paths from parent_dir
        JARVIS_DIR = os.path.join(parent_dir, "automatbench/jarvis_data", f"jarvis_{task_name}")
        MP_DIR = os.path.join(parent_dir, "automatbench/MP_data", f"MP_{task_name}")
        
        # Check if directories exist, if not try the path without double 'automatbench'
        if not os.path.exists(JARVIS_DIR):
             # Try alternative path (one level up or down depending on where parent_dir points)
             # Based on user error: /home2/hxli/automatbench/automatbench/jarvis_data/...
             # It seems parent_dir includes 'automatbench' already?
             # Let's check parent_dir value
             logging.info(f"parent_dir is: {parent_dir}")
             
             # If parent_dir ends with 'automatbench', we shouldn't add it again?
             # But the error shows /home2/hxli/automatbench/automatbench/...
             # This suggests parent_dir is /home2/hxli/automatbench and we append automatbench again.
             
             # Let's try to find the data directories relative to common anchor points
             # Assuming structure:
             # root/
             #   automatbench/
             #     jarvis_data/
             #     MP_data/
             #     work/
             
             # If parent_dir is 'root/automatbench' (where work/rfscm_run.py is work/.. -> root/automatbench)
             # Then we should look in ./jarvis_data not ./automatbench/jarvis_data
             
             JARVIS_DIR = os.path.join(parent_dir, "jarvis_data", f"jarvis_{task_name}")
             MP_DIR = os.path.join(parent_dir, "MP_data", f"MP_{task_name}")
             
        logging.info(f"Using JARVIS_DIR: {JARVIS_DIR}")
        logging.info(f"Using MP_DIR: {MP_DIR}")
        
        os.makedirs(result_dir, exist_ok=True)
        
        logging.info("Loading labels...")
        # Load JARVIS labels
        jarvis_labels_path = os.path.join(JARVIS_DIR, "Y_value.csv")
        if os.path.exists(jarvis_labels_path):
            df_jarvis = pd.read_csv(jarvis_labels_path)
            jarvis_targets = dict(zip(df_jarvis["material_id"], df_jarvis[TARGET]))
        else:
            logging.warning(f"JARVIS labels not found at {jarvis_labels_path}")
            jarvis_targets = {}

        # Load MP labels
        mp_labels_path = os.path.join(MP_DIR, "Y_value.csv")
        if os.path.exists(mp_labels_path):
            df_mp = pd.read_csv(mp_labels_path)
            mp_targets = dict(zip(df_mp["material_id"], df_mp[TARGET]))
        else:
            logging.warning(f"MP labels not found at {mp_labels_path}")
            mp_targets = {}

        # Combine targets
        all_targets = {**jarvis_targets, **mp_targets}
        logging.info(f"Loaded {len(all_targets)} targets.")

        # Load splits
        logging.info("Loading splits...")
        with open(os.path.join(data_dir, "train.json"), "r") as f:
            train_splits = json.load(f)
        with open(os.path.join(data_dir, "test.json"), "r") as f:
            test_splits = json.load(f)
        
        # Load validation splits if they exist
        val_path = os.path.join(data_dir, "val.json")
        if os.path.exists(val_path):
            logging.info("Loading validation splits...")
            with open(val_path, "r") as f:
                val_splits = json.load(f)
        else:
            val_splits = {}

        # Configure Automatminer pipeline
        logging.info("Setting up Automatminer pipeline...")
        learner = SinglePipelineAdaptor(
            regressor=RandomForestRegressor(n_estimators=500),
            classifier=RandomForestClassifier(n_estimators=500),
        )

        pipe_config = {
            "learner": learner,
            "reducer": FeatureReducer(reducers=[]),
            "cleaner": DataCleaner(feature_na_method="mean", max_na_frac=0.01, na_method_fit="mean", na_method_transform="mean", drop_na_targets=False),
            "autofeaturizer": AutoFeaturizer(n_jobs=N_JOBS, preset="debug"),
        }

        # Define value ranges for filtering outliers
        min_val = -float('inf')
        max_val = float('inf')
        
        if "shear_modulus" in task_name or "bulk_modulus" in task_name:
            min_val = -500.0
            max_val = 1000.0
            logging.info(f"Filtering {task_name} values outside [{min_val}, {max_val}]")

        results = {}
        actual_folds = len(train_splits.keys())
        logging.info(f"Starting {actual_folds}-fold Cross Validation...")

        for fold_idx in range(actual_folds):
            fold_key = str(fold_idx)
            if fold_key not in train_splits or fold_key not in test_splits:
                logging.warning(f"Fold {fold_idx} not found in splits, skipping.")
                continue
                
            logging.info(f"Processing Fold {fold_idx}...")
            
            train_ids = train_splits[fold_key]
            test_ids = test_splits[fold_key]
            
            # Add validation set to training set if available
            val_ids = val_splits.get(fold_key, [])
            if val_ids:
                logging.info(f"Adding {len(val_ids)} validation samples to training set.")
                train_ids = train_ids + val_ids
            
            logging.info(f"Loading train data ({len(train_ids)} samples)...")
            df_train = get_data(train_ids, all_targets, JARVIS_DIR, MP_DIR, TARGET)
            logging.info(f"Loading test data ({len(test_ids)} samples)...")
            df_test = get_data(test_ids, all_targets, JARVIS_DIR, MP_DIR, TARGET)
            
            # Drop entries with missing targets
            df_train = df_train.dropna(subset=[TARGET])
            df_test = df_test.dropna(subset=[TARGET])

            # Filter outliers in training set only
            if min_val > -float('inf') or max_val < float('inf'):
                len_train_before = len(df_train)
                df_train = df_train[(df_train[TARGET] >= min_val) & (df_train[TARGET] <= max_val)]
                if len(df_train) < len_train_before:
                     logging.info(f"Filtered {len_train_before - len(df_train)} training samples (outliers).")

            logging.info(f"Train size: {len(df_train)}")
            logging.info(f"Test size: {len(df_test)}")
            
            if len(df_train) == 0 or len(df_test) == 0:
                logging.warning("Empty train or test set, skipping fold.")
                continue

            try:
                # Initialize pipeline for this fold
                # Force log directory change for MatPipe execution
                cwd_before_matpipe = os.getcwd()
                try:
                    os.chdir(log_dir)
                    pipe = MatPipe(**pipe_config)
                    
                    # Fit the pipeline
                    logging.info("Fitting pipeline...")
                    pipe.fit(df_train, TARGET)
                    
                    # Predict
                    logging.info("Predicting...")
                    # MatPipe adds prediction column to the dataframe
                    df_pred = pipe.predict(df_test)
                finally:
                    os.chdir(cwd_before_matpipe)

                predictions = df_pred[f"{TARGET} predicted"]
                
                # Calculate metrics
                y_true = df_test[TARGET]
                y_pred = predictions
                
                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mape = float(mean_absolute_percentage_error(y_true, y_pred))
                max_err_val = float(max_error(y_true, y_pred))
                
                # Store results
                results[f"fold_{fold_idx}"] = {
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "max_error": max_err_val,
                    "predictions": predictions.tolist(),
                    "targets": y_true.tolist()
                }
                logging.info(f"Fold {fold_idx} completed. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                
            except Exception as e:
                logging.error(f"Error in fold {fold_idx}: {e}")
                import traceback
                logging.error(traceback.format_exc())

        # Calculate average metrics across all folds
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
                "params": {"R": R, "N": N, "S": S},
                "fold_results": results,
                "average_metrics": avg_metrics
            }

            # Save results to JSON file
            logging.info(f"Saving results to {result_file}...")
            with open(result_file, "w") as f:
                json.dump(final_output, f, indent=4)

            logging.info(f"Done processing {folder_name}.")
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

    for task_name in tasks:
        for R in R_values:
            for N in N_values:
                for S in S_values:
                    run(task_name, R, N, S)
