import json
import os
import argparse
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Setup logging FIRST, before importing tensorflow or kgcnn
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

model_name = 'schnet'
log_dir = os.path.join(parent_dir, 'work', 'log')
os.makedirs(log_dir, exist_ok=True)

import tensorflow as tf
from pymatgen.core.structure import Structure
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error

# KGCNN imports
from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.Schnet import make_crystal_model
from kgcnn.training.schedule import LinearWarmupExponentialDecay
from kgcnn.training.scheduler import LinearLearningRateScheduler
from kgcnn.utils.devices import set_devices_gpu

def robust_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-5
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# Callbacks for CrystalDataset
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

# Hyperparameters
hyper = {
    "model": {
        "module_name": "kgcnn.literature.Schnet",
        "class_name": "make_crystal_model",
        "config": {
            "name": "Schnet",
            "inputs": [
                {'shape': (None,), 'name': "node_number", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 3), 'name': "node_coordinates", 'dtype': 'float32', 'ragged': True},
                {'shape': (None, 2), 'name': "range_indices", 'dtype': 'int64', 'ragged': True},
                {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
            ],
            "input_embedding": {
                "node": {"input_dim": 95, "output_dim": 64}
            },
            "interaction_args": {
                "units": 128, "use_bias": True, "activation": "kgcnn>shifted_softplus", "cfconv_pool": "sum"
            },
            "node_pooling_args": {"pooling_method": "mean"},
            "depth": 4,
            "gauss_args": {"bins": 25, "distance": 5, "offset": 0.0, "sigma": 0.4}, "verbose": 10,
            "last_mlp": {"use_bias": [True, True, True], "units": [128, 64, 1],
                         "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
            "output_embedding": "graph",
            "use_output_mlp": False,
            "output_mlp": None,
        }
    },
    "training": {
        "fit": {
            "batch_size": 64, "epochs": 600, "validation_freq": 10, "verbose": 2, "validation_split": 0.1,
            "callbacks": [
                {"class_name": "kgcnn>LinearLearningRateScheduler", "config": {
                    "learning_rate_start": 1e-4, "learning_rate_stop": 1e-05, "epo_min": 100, "epo": 600,
                    "verbose": 0}
                 }
            ]
        },
        "compile": {
            "optimizer": {"class_name": "Adam", "config": {"lr": 1e-4}},
            "loss": "mean_absolute_error"
        },
        "scaler": {
            "class_name": "StandardScaler",
            "module_name": "kgcnn.scaler.scaler",
            "config": {"with_std": True, "with_mean": True, "copy": True}
        },
    },
    "data": {
        "dataset": {
            "class_name": "CrystalDataset",
            "module_name": "kgcnn.data.crystal",
            "config": {},
            "methods": [
                {"map_list": {"method": "set_range_periodic", "max_distance": 5}}
            ]
        },
    },
}

def get_structures(ids, source, task_name):
    structures = []
    
    subdir = 'jarvis_data' if source == 'jarvis' else 'MP_data'
    task_dir = f'jarvis_{task_name}' if source == 'jarvis' else f'MP_{task_name}'
    
    # Try different potential paths, prioritizing nested structure as observed
    paths_to_try = [
        # Nested: parent/subdir/task_dir/subdir
        os.path.join(parent_dir, subdir, task_dir, subdir),
        os.path.join(parent_dir, 'automatbench', subdir, task_dir, subdir),
        # Flat: parent/subdir/task_dir
        os.path.join(parent_dir, subdir, task_dir),
        os.path.join(parent_dir, 'automatbench', subdir, task_dir)
    ]
    
    base_dir = None
    for p in paths_to_try:
        if os.path.exists(p):
            base_dir = p
            break
            
    if not base_dir:
        print(f"Error: Data directory not found for {source} and task {task_name}")
        print(f"Tried paths: {paths_to_try}")
        return [None] * len(ids)

    for mat_id in ids:
        file_path = os.path.join(base_dir, f"{mat_id}.cif")
        try:
            struct = Structure.from_file(file_path)
            structures.append(struct)
        except Exception as e:
            # print(f"Error loading {mat_id}: {e}")
            structures.append(None)
    return structures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing matched_data.csv and splits.json')
    parser.add_argument('--source', required=True, choices=['jarvis', 'mp'], help='Source of data to use')
    parser.add_argument('--task', required=True, help='Task name')
    parser.add_argument('--fold', required=False, type=int, default=0, help='Fold index')
    parser.add_argument("--gpu", required=False, help="GPU index used for training.", default=None, nargs="+", type=int)
    args = parser.parse_args()
    
    # GPU setup
    if args.gpu is not None:
        set_devices_gpu(args.gpu)
    else:
        # Default to CPU if not specified or force to use what is available
        pass

    data_dir = args.data_dir
    source = args.source
    task_name = args.task
    fold = args.fold
    
    id_col = 'jarvis_id' if source == 'jarvis' else 'mp_id'
    target_col = f'jarvis_{task_name}' if source == 'jarvis' else f'mp_{task_name}'
    
    csv_path = os.path.join(data_dir, 'matched_data.csv')
    splits_path = os.path.join(data_dir, 'splits.json')
    
    if not os.path.exists(csv_path) or not os.path.exists(splits_path):
        print("Error: Input files not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    print(f"Loading structures for {source} (Fold {fold})...")
    
    if fold >= len(splits):
        print(f"Error: Fold {fold} out of range (max {len(splits)-1})")
        sys.exit(1)
        
    split = splits[fold]
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    train_ids = df.iloc[train_idx][id_col].tolist()
    val_ids = df.iloc[val_idx][id_col].tolist()
    test_ids = df.iloc[test_idx][id_col].tolist()
    
    train_targets = df.iloc[train_idx][target_col].tolist()
    val_targets = df.iloc[val_idx][target_col].tolist()
    test_targets = df.iloc[test_idx][target_col].tolist()

    train_structs = get_structures(train_ids, source, task_name)
    val_structs = get_structures(val_ids, source, task_name)
    test_structs = get_structures(test_ids, source, task_name)

    # Filter None
    def filter_data(structs, targets):
        valid_s = []
        valid_t = []
        for s, t in zip(structs, targets):
            if s is not None:
                valid_s.append(s)
                valid_t.append(t)
        return valid_s, valid_t

    train_structs, train_targets = filter_data(train_structs, train_targets)
    val_structs, val_targets = filter_data(val_structs, val_targets)
    test_structs, test_targets = filter_data(test_structs, test_targets)

    print(f"Source: {source} | Train: {len(train_structs)}, Val: {len(val_structs)}, Test: {len(test_structs)}")
    
    if len(train_structs) == 0:
        print("Error: Empty training set.")
        sys.exit(1)

    # Prepare data for KGCNN
    data_train = CrystalDataset()
    data_train._map_callbacks(train_structs, pd.Series(train_targets), callbacks)
    data_train.set_methods(hyper["data"]["dataset"]["methods"])
    data_train.clean(hyper["model"]["config"]["inputs"])
    
    y_train = np.array(data_train.get("graph_labels"))
    x_train = data_train.tensor(hyper["model"]["config"]["inputs"])

    scaler = StandardScaler(**hyper["training"]["scaler"]["config"])
    y_train = scaler.fit_transform(y_train)

    # Validation
    data_val = CrystalDataset()
    data_val._map_callbacks(val_structs, pd.Series(val_targets), callbacks)
    data_val.set_methods(hyper["data"]["dataset"]["methods"])
    data_val.clean(hyper["model"]["config"]["inputs"])

    y_val = np.array(data_val.get("graph_labels"))
    x_val = data_val.tensor(hyper["model"]["config"]["inputs"])
    y_val = scaler.transform(y_val)

    # Model
    model = make_crystal_model(**hyper["model"]["config"])
    model.compile(
        loss=tf.keras.losses.get(hyper["training"]["compile"]["loss"]),
        optimizer=tf.keras.optimizers.get(hyper["training"]["compile"]["optimizer"]),
    )
    
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

    print("Starting training...")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=hyper["training"]["fit"]["batch_size"],
        epochs=hyper["training"]["fit"]["epochs"],
        verbose=hyper["training"]["fit"]["verbose"],
        callbacks=train_callbacks
    )

    # Test
    data_test = CrystalDataset()
    data_test._map_callbacks(test_structs, pd.Series(test_targets), callbacks) # Pass targets to ensure alignment
    data_test.set_methods(hyper["data"]["dataset"]["methods"])
    data_test.clean(hyper["model"]["config"]["inputs"])
    
    x_test = data_test.tensor(hyper["model"]["config"]["inputs"])
    y_true_clean = np.array(data_test.get("graph_labels")) # Use cleaned targets from dataset
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    if predictions.shape[-1] == 1:
        predictions = np.squeeze(predictions, axis=-1)
    if y_true_clean.shape[-1] == 1:
        y_true_clean = np.squeeze(y_true_clean, axis=-1)
        
    mae = mean_absolute_error(y_true_clean, predictions)
    rmse = np.sqrt(mean_squared_error(y_true_clean, predictions))
    mape = robust_mape(y_true_clean, predictions)
    max_err = max_error(y_true_clean, predictions)
    
    result = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'max_error': float(max_err)
    }
    output_file = os.path.join(data_dir, f'results_{source}.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"{source.upper()} MAE: {mae:.4f}")
    print(f"{source.upper()} RMSE: {rmse:.4f}")
    print(f"{source.upper()} MAPE: {mape:.4f}")
    print(f"{source.upper()} MAX ERROR: {max_err:.4f}")

if __name__ == "__main__":
    main()
