import json
import os
import argparse
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from common.logging_utils import setup_dual_logging

# Setup logging FIRST, before importing tensorflow or kgcnn
# Add parent directory to path to import common.params
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

model_name = 'megnet'
log_dir = os.path.join(parent_dir, 'work/log')
log_file = os.path.join(log_dir, model_name + '_execution.log')

def setup_logging():
    setup_dual_logging(log_file, level=logging.INFO, force=True, reset_handlers=True)
    logging.info("Initializing MegNet script...")

setup_logging()

import tensorflow as tf
from copy import deepcopy

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')

from pymatgen.core.structure import Structure
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error
except ImportError:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Define max_error if not available (sklearn < 0.21)
    try:
        from sklearn.metrics import max_error
    except ImportError:
        def max_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.max(np.abs(y_true - y_pred))

    # Define mean_absolute_percentage_error if not available (sklearn < 0.24)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero
        mask = np.abs(y_true) > 1e-7 # Use a small epsilon instead of strict 0
        if np.sum(mask) == 0:
            return 0.0 
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# KGCNN imports
from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.Megnet import make_crystal_model
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.utils.devices import set_devices_gpu

# Import params
from common import params

# Argument parser for GPU
parser = argparse.ArgumentParser(description="Train MegNet.")
parser.add_argument("--gpu", required=False, help="GPU index used for training.", default=None, nargs="+", type=int)
# Only parse args if run as main script to avoid conflict when imported
if __name__ == "__main__":
    args = vars(parser.parse_args())
    gpu_to_use = args["gpu"]
    if gpu_to_use is not None:
        set_devices_gpu(gpu_to_use)
else:
    gpu_to_use = None

# Define callbacks for graph construction (from megnet_perovskites.py)
callbacks = {
    "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
    "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
    "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
    "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
    "abc": lambda st, ds: np.array(st.lattice.abc),
    "charge": lambda st, ds: np.array([st.charge], dtype="float"),
    "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
    "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int"),
}

# Define model hyperparameters (from megnet_perovskites.py)
hyper = {
    "model": {
        "module_name": "kgcnn.literature.Megnet",
        "class_name": "make_crystal_model",
        "config": {
            "name": "Megnet",
            "inputs": [
                {"shape": (None,), "name": "node_number", "dtype": "float32", "ragged": True},
                {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                {"shape": (None, 2), "name": "range_indices", "dtype": "int64", "ragged": True},
                {"shape": [1], "name": "charge", "dtype": "float32", "ragged": False},
                {"shape": (None, 3), "name": "range_image", "dtype": "int64", "ragged": True},
                {"shape": (3, 3), "name": "graph_lattice", "dtype": "float32", "ragged": False},
            ],
            "input_embedding": {"node": {"input_dim": 95, "output_dim": 64}, "graph": {"input_dim": 100, "output_dim": 64}},
            "make_distance": True, "expand_distance": True,
            "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4},
            "meg_block_args": {"node_embed": [64, 32, 32], "edge_embed": [64, 32, 32], "env_embed": [64, 32, 32], "activation": "kgcnn>softplus2"},
            "set2set_args": {"channels": 16, "T": 3, "pooling_method": "sum", "init_qstar": "0"},
            "node_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
            "edge_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
            "state_ff_args": {"units": [64, 32], "activation": "kgcnn>softplus2"},
            "nblocks": 3, "has_ff": True, "dropout": None, "use_set2set": True,
            "verbose": 10, "output_embedding": "graph",
            "output_mlp": {"use_bias": [True, True, True], "units": [32, 16, 1], "activation": ["kgcnn>softplus2", "kgcnn>softplus2", "linear"]},
        },
    },
    "training": {
        "fit": {
            "batch_size": 32, "epochs": 1000, "validation_freq": 10, "verbose": 2,
            "callbacks": [
                {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {"learning_rate_start": 0.0005, "learning_rate_stop": 0.5e-05, "epo_min": 100, "epo": 1000, "verbose": 0}}
            ],
        },
        "compile": {"optimizer": {"class_name": "Adam", "config": {"lr": 0.0005}}, "loss": "mean_absolute_error"},
        "scaler": {"class_name": "StandardScaler", "module_name": "kgcnn.scaler.scaler", "config": {"with_std": True, "with_mean": True, "copy": True}},
    },
    "data": {
        "dataset": {
            "class_name": "CrystalDataset", "module_name": "kgcnn.data.crystal", "config": {},
            "methods": [{"map_list": {"method": "set_range_periodic", "max_distance": 5.0}}],
        },
    },
}

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
    
    # Load validation splits if available
    val_splits_path = os.path.join(data_dir, "val.json")
    if os.path.exists(val_splits_path):
        with open(val_splits_path, "r") as f:
            val_splits = json.load(f)
    else:
        val_splits = None
        logging.warning(f"No validation split found for {folder_name}")

    results_folds = {}
    actual_folds = len(train_splits.keys())

    for fold_idx in range(actual_folds):
        fold_key = str(fold_idx)
        if fold_key not in train_splits: continue
        
        logging.info(f"Processing Fold {fold_idx}...")
        train_ids = train_splits[fold_key]
        test_ids = test_splits[fold_key]
        
        logging.info(f"Loading data for fold {fold_idx}...")
        structures_train, targets_train, _ = get_data(train_ids, all_targets, JARVIS_DIR, MP_DIR)
        structures_test, targets_test, ids_test = get_data(test_ids, all_targets, JARVIS_DIR, MP_DIR)
        
        # Handle validation data
        structures_val = []
        targets_val = []
        if val_splits and fold_key in val_splits:
            val_ids = val_splits[fold_key]
            structures_val, targets_val, _ = get_data(val_ids, all_targets, JARVIS_DIR, MP_DIR)
            logging.info(f"Loaded {len(structures_val)} validation samples.")
        
        # If no explicit validation set, split from training (fallback, though user said val exists)
        if not structures_val:
            logging.warning("Using train_test_split as fallback for validation.")
            from sklearn.model_selection import train_test_split
            structures_train, structures_val, targets_train, targets_val = train_test_split(
                structures_train, targets_train, test_size=0.1, random_state=42
            )

        logging.info(f"Train size: {len(structures_train)}, Val size: {len(structures_val)}, Test size: {len(structures_test)}")

        if not structures_train or not structures_test:
            logging.warning("Empty train or test set.")
            continue
        
        # Training data
        data_train = CrystalDataset()
        data_train._map_callbacks(structures_train, pd.Series(targets_train), callbacks)
        data_train.set_methods(hyper["data"]["dataset"]["methods"])
        data_train.clean(hyper["model"]["config"]["inputs"])
        
        y_train = np.array(data_train.get("graph_labels"))
        x_train = data_train.tensor(hyper["model"]["config"]["inputs"])

        scaler = StandardScaler(**hyper["training"]["scaler"]["config"])
        y_train = scaler.fit_transform(y_train)

        # Validation data
        data_val = CrystalDataset()
        data_val._map_callbacks(structures_val, pd.Series(targets_val), callbacks)
        data_val.set_methods(hyper["data"]["dataset"]["methods"])
        data_val.clean(hyper["model"]["config"]["inputs"])

        y_val = np.array(data_val.get("graph_labels"))
        x_val = data_val.tensor(hyper["model"]["config"]["inputs"])
        y_val = scaler.transform(y_val) # Use same scaler

        # Model
        model = make_crystal_model(**hyper["model"]["config"])
        model.compile(
            loss=tf.keras.losses.get(hyper["training"]["compile"]["loss"]),
            optimizer=tf.keras.optimizers.get(hyper["training"]["compile"]["optimizer"]),
        )
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        )

        train_callbacks = [
            (LinearLearningRateScheduler(**x["config"]) if "LinearLearningRateScheduler" in x["class_name"] else tf.keras.utils.deserialize_keras_object(x))
            for x in hyper["training"]["fit"]["callbacks"]
        ]
        train_callbacks.append(early_stopping)

        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=hyper["training"]["fit"]["batch_size"],
            epochs=hyper["training"]["fit"]["epochs"],
            verbose=hyper["training"]["fit"]["verbose"],
            callbacks=train_callbacks
        )

        # Testing data
        data_test = CrystalDataset()
        data_test._map_callbacks(structures_test, pd.Series(np.zeros(len(structures_test))), callbacks)
        data_test.set_methods(hyper["data"]["dataset"]["methods"])
        removed_indices = data_test.clean(hyper["model"]["config"]["inputs"])
        
        x_test = data_test.tensor(hyper["model"]["config"]["inputs"])
        predictions_model = model.predict(x_test)
        
        # Handle removed indices
        if removed_indices:
            indices_test_map = [j for j in range(len(structures_test))]
            for j in removed_indices:
                indices_test_map.pop(j)
            predictions_full = np.zeros((len(structures_test), 1))
            predictions_full[indices_test_map] = predictions_model
            predictions = predictions_full
        else:
            predictions = predictions_model

        predictions = scaler.inverse_transform(predictions)
        if predictions.shape[-1] == 1:
            predictions = np.squeeze(predictions, axis=-1)

        mae = float(mean_absolute_error(targets_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(targets_test, predictions)))
        
        # Robust MAPE calculation
        targets_test_arr = np.array(targets_test)
        predictions_arr = np.array(predictions)
        
        # Avoid division by zero or very small numbers
        mask = np.abs(targets_test_arr) > 1e-5 
        if np.sum(mask) > 0:
            mape = float(np.mean(np.abs((targets_test_arr[mask] - predictions_arr[mask]) / targets_test_arr[mask])))
        else:
            mape = 0.0 # Or np.nan
            
        max_err_val = float(max_error(targets_test, predictions))

        results_folds[f"fold_{fold_idx}"] = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "max_error": max_err_val,
            "predictions": predictions.tolist(),
            "targets": targets_test_arr.tolist(),
        }
        logging.info(f"Fold {fold_idx} MAE: {mae}")

    if results_folds:
        avg_metrics = {
            "mae": float(np.mean([res["mae"] for res in results_folds.values()])),
            "rmse": float(np.mean([res["rmse"] for res in results_folds.values()])),
            "mape": float(np.mean([res["mape"] for res in results_folds.values()])),
            "max_error": float(np.max([res["max_error"] for res in results_folds.values()]))
        }
        final_output = {
            "task": task_name,
            "method": "MegNet",
            "params": {"R": R, "N": N, "S": S},
            "fold_results": results_folds,
            "average_metrics": avg_metrics
        }
        with open(result_file, "w") as f:
            json.dump(final_output, f, indent=4)
        logging.info(f"Saved results to {result_file}")

if __name__ == "__main__":
    setup_logging()
    
    if len(physical_devices) > 0:
        logging.info(f"GPU detected: {physical_devices}")
    else:
        logging.warning("No GPU detected. Training will proceed on CPU, which may be slow.")

    R_values = params.R_values
    N_values = params.N_values
    S_values = params.S_values
    tasks = params.tasks

    for task_name in tasks:
        for R in R_values:
            for N in N_values:
                for S in S_values:
                    run(task_name, R, N, S)
