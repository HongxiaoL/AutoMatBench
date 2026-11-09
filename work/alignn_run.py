import json
import os
import argparse
import pandas as pd
import numpy as np
import logging
import sys
import shutil
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pymatgen.core.structure import Structure
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson, loadjson

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from common import params

model_name = 'alignn'
log_dir = os.path.join(parent_dir, 'work/log')
log_file = os.path.join(log_dir, model_name + '_execution.log')

try:
    from sklearn.metrics import max_error
except ImportError:
    def max_error(y_true, y_pred):
        return np.max(np.abs(np.array(y_true) - np.array(y_pred)))

def setup_logging():
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(logging.FileHandler(log_file))
    
    logging.info("Initializing ALIGNN script...")

setup_logging()

def get_data(ids, all_targets, jarvis_dir, mp_dir):
    structures = []
    targets = []
    valid_ids = []
    
    for mat_id in ids:
        if mat_id not in all_targets:
            continue
            
        target_val = all_targets[mat_id]
        
        if mat_id.startswith("JVASP-"):
            file_path = os.path.join(jarvis_dir, "jarvis_data", f"{mat_id}.cif")
        elif mat_id.startswith("mp-") or mat_id.startswith("mvc-"):
            file_path = os.path.join(mp_dir, "MP_data", f"{mat_id}.cif")
        else:
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
            
    return structures, targets, valid_ids

def run(task_name, R, N, S):
    if N == 60 and S in [9, 10]:
        logging.info(f"Skipping invalid parameter combination: R={R}, N={N}, S={S}")
        return

    default_config = {
        "version": "112bbedebdaecf59fb18e11c929080fb2f358246",
        "dataset": "user_data",
        "target": "target",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "id_tag": "jid",
        "random_seed": 123,
        "classification_threshold": None,
        "n_val": None,
        "n_test": None,
        "n_train": None,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "target_multiplication_factor": None,
        "epochs": 3,
        "batch_size": 2,
        "weight_decay": 1e-05,
        "learning_rate": 0.001,
        "filename": "sample",
        "warmup_steps": 2000,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "pin_memory": False,
        "save_dataloader": False,
        "write_checkpoint": True,
        "write_predictions": True,
        "store_outputs": True,
        "progress": True,
        "log_tensorboard": False,
        "standard_scalar_and_pca": False,
        "use_canonize": True,
        "num_workers": 0,
        "cutoff": 8.0,
        "max_neighbors": 12,
        "keep_data_order": False,
        "model": {
            "name": "alignn",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 64,
            "hidden_features": 256,
            "output_features": 1,
            "link": "identity",
            "zero_inflated": False,
            "classification": False
        }
    }

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
    JARVIS_DIR = os.path.join(parent_dir, "jarvis_data", f"jarvis_{task_name}")
    MP_DIR = os.path.join(parent_dir, "MP_data", f"MP_{task_name}")
    
    jarvis_labels_path = os.path.join(JARVIS_DIR, "Y_value.csv")
    jarvis_targets = {}
    if os.path.exists(jarvis_labels_path):
        df_jarvis = pd.read_csv(jarvis_labels_path)
        jarvis_targets = dict(zip(df_jarvis["material_id"], df_jarvis[TARGET]))
    
    mp_labels_path = os.path.join(MP_DIR, "Y_value.csv")
    mp_targets = {}
    if os.path.exists(mp_labels_path):
        df_mp = pd.read_csv(mp_labels_path)
        mp_targets = dict(zip(df_mp["material_id"], df_mp[TARGET]))
    
    all_targets = {**jarvis_targets, **mp_targets}
    logging.info(f"Loaded {len(all_targets)} targets.")

    with open(os.path.join(data_dir, "train.json"), "r") as f:
        train_splits = json.load(f)
    with open(os.path.join(data_dir, "test.json"), "r") as f:
        test_splits = json.load(f)
    
    val_splits_path = os.path.join(data_dir, "val.json")
    if os.path.exists(val_splits_path):
        with open(val_splits_path, "r") as f:
            val_splits = json.load(f)
    else:
        val_splits = None
        logging.warning(f"No validation split found for {folder_name}")

    results_folds = {}
    actual_folds = len(train_splits.keys())

    min_val = -float('inf')
    max_val = float('inf')
    
    if "shear_modulus" in task_name or "bulk_modulus" in task_name:
        min_val = -500.0
        max_val = 1000.0
        logging.info(f"Filtering {task_name} values outside [{min_val}, {max_val}]")

    for fold_idx in range(actual_folds):
        fold_key = str(fold_idx)
        if fold_key not in train_splits: continue
        
        logging.info(f"Processing Fold {fold_idx}...")
        
        fold_dir = os.path.join(result_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_ids = train_splits[fold_key]
        test_ids = test_splits[fold_key]
        val_ids = val_splits[fold_key] if val_splits and fold_key in val_splits else []
        
        if not val_ids:
                n_val_fallback = max(1, int(len(train_ids) * 0.1))
                val_ids = train_ids[-n_val_fallback:]
                train_ids = train_ids[:-n_val_fallback]
                logging.info(f"Created fallback validation set of size {len(val_ids)}")

        logging.info("Loading structures...")
        structs_train, targets_train, ids_train = get_data(train_ids, all_targets, JARVIS_DIR, MP_DIR)
        structs_val, targets_val, ids_val = get_data(val_ids, all_targets, JARVIS_DIR, MP_DIR)
        structs_test, targets_test, ids_test = get_data(test_ids, all_targets, JARVIS_DIR, MP_DIR)
        
        def filter_nans(structs, targets, ids):
            valid_s, valid_t, valid_i = [], [], []
            for s, t, i in zip(structs, targets, ids):
                if t is not None and not np.isnan(t):
                    if t < min_val or t > max_val:
                        continue
                    
                    valid_s.append(s)
                    valid_t.append(t)
                    valid_i.append(i)
            return valid_s, valid_t, valid_i

        structs_train, targets_train, ids_train = filter_nans(structs_train, targets_train, ids_train)
        structs_val, targets_val, ids_val = filter_nans(structs_val, targets_val, ids_val)
        structs_test, targets_test, ids_test = filter_nans(structs_test, targets_test, ids_test)

        logging.info(f"Train size: {len(structs_train)}, Val size: {len(structs_val)}, Test size: {len(structs_test)}")

        if not structs_train or not structs_test:
            logging.warning("Empty train or test set.")
            continue

        cwd = os.getcwd()
        os.chdir(fold_dir)
        
        try:
            with open("id_prop.csv", "w") as f:
                for i, s in enumerate(structs_train):
                    mid = ids_train[i]
                    atoms = pmg_to_atoms(s)
                    atoms.write_poscar(mid)
                    f.write(f"{mid},{targets_train[i]}\n")
                
                for i, s in enumerate(structs_val):
                    mid = ids_val[i]
                    atoms = pmg_to_atoms(s)
                    atoms.write_poscar(mid)
                    f.write(f"{mid},{targets_val[i]}\n")
                    
                for i, s in enumerate(structs_test):
                    mid = ids_test[i]
                    atoms = pmg_to_atoms(s)
                    atoms.write_poscar(mid)
                    f.write(f"{mid},{targets_test[i]}\n")
            
            config = default_config.copy()
            config["n_train"] = len(structs_train)
            config["n_val"] = len(structs_val)
            config["n_test"] = len(structs_test)
            config["keep_data_order"] = True
            config["epochs"] = 200 
            config["n_early_stopping"] = 10 
            config["early_stopping_min_delta"] = 0.0001
            config["output_dir"] = "output"
            
            if config.get("batch_size", 0) < 32:
                config["batch_size"] = 32

            config_filename = "config_fold.json"
            dumpjson(data=config, filename=config_filename)
            
            train_folder_path = os.path.join(os.path.dirname(sys.executable), "train_folder.py")
            cmd = (
                f"{train_folder_path} "
                f"--root_dir '{fold_dir}' "
                f"--config_name '{config_filename}' "
                f"--output_dir '{os.path.join(fold_dir, 'output')}' "
                f"--file_format poscar | tee -a '{log_file}'"
            )
            logging.info(f"Running command: {cmd}")
            ret = os.system(cmd)
            
            if ret != 0:
                logging.error(f"Training failed for fold {fold_idx}")
                continue
                
            test_csv = os.path.join(fold_dir, "output", "prediction_results_test_set.csv")
            if os.path.exists(test_csv):
                df = pd.read_csv(test_csv)
                target_vals = df.target.values
                pred_vals = df.prediction.values
                
                mae = float(mean_absolute_error(target_vals, pred_vals))
                rmse = float(np.sqrt(mean_squared_error(target_vals, pred_vals)))
                max_err_val = float(max_error(target_vals, pred_vals))
                
                mask = np.abs(target_vals) > 1e-5
                if np.sum(mask) > 0:
                    mape = float(np.mean(np.abs((target_vals[mask] - pred_vals[mask]) / target_vals[mask])))
                else:
                    mape = 0.0
                    
                logging.info(f"Fold {fold_idx} MAE: {mae}, MAPE: {mape}")
                
                results_folds[f"fold_{fold_idx}"] = {
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "max_error": max_err_val,
                    "predictions": pred_vals.tolist(),
                    "targets": target_vals.tolist()
                }
            else:
                logging.error(f"Prediction file not found: {test_csv}")

        finally:
            os.chdir(cwd)
            # Remove the fold directory to save disk space.
            if os.path.exists(fold_dir):
                    try:
                        shutil.rmtree(fold_dir)
                        logging.info(f"Cleaned up {fold_dir}")
                    except Exception as e:
                        logging.warning(f"Failed to clean up {fold_dir}: {e}")
        
    if results_folds:
        avg_metrics = {
            "mae": float(np.mean([res["mae"] for res in results_folds.values()])),
            "rmse": float(np.mean([res["rmse"] for res in results_folds.values()])),
            "mape": float(np.mean([res["mape"] for res in results_folds.values()])),
            "max_error": float(np.max([res["max_error"] for res in results_folds.values()]))
        }
        final_output = {
            "task": task_name,
            "method": "ALIGNN",
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
