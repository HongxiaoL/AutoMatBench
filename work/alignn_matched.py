import json
import os
import argparse
import pandas as pd
import numpy as np
import logging
import sys
import shutil
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from pymatgen.core.structure import Structure
from jarvis.core.atoms import pmg_to_atoms
from jarvis.db.jsonutils import dumpjson

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

model_name = 'alignn'
log_dir = os.path.join(parent_dir, 'work', 'log')
os.makedirs(log_dir, exist_ok=True)

def robust_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-5
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def get_structures(ids, source, task_name):
    structures = []
    
    subdir = 'jarvis_data' if source == 'jarvis' else 'MP_data'
    task_dir = f'jarvis_{task_name}' if source == 'jarvis' else f'MP_{task_name}'
    
    # Nested layout: root/<subdir>/<task_dir>/<subdir>; flat layout otherwise.
    paths_to_try = [
        os.path.join(parent_dir, subdir, task_dir, subdir),
        os.path.join(parent_dir, subdir, task_dir),
    ]
    
    base_dir = None
    for p in paths_to_try:
        if os.path.exists(p):
            base_dir = p
            break
            
    if not base_dir:
        print(f"Error: Data directory not found for {source} and task {task_name}")
        return [None] * len(ids)

    for mat_id in ids:
        file_path = os.path.join(base_dir, f"{mat_id}.cif")
        try:
            struct = Structure.from_file(file_path)
            structures.append(struct)
        except Exception as e:
            structures.append(None)
    return structures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing matched_data.csv and splits.json')
    parser.add_argument('--source', required=True, choices=['jarvis', 'mp'], help='Source of data to use')
    parser.add_argument('--task', required=True, help='Task name')
    parser.add_argument('--fold', required=False, type=int, default=0, help='Fold index')
    args = parser.parse_args()
    
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

    def filter_data(ids, structs, targets):
        valid_i = []
        valid_s = []
        valid_t = []
        for i, s, t in zip(ids, structs, targets):
            if s is not None:
                valid_i.append(i)
                valid_s.append(s)
                valid_t.append(t)
        return valid_i, valid_s, valid_t

    train_ids, train_structs, train_targets = filter_data(train_ids, train_structs, train_targets)
    val_ids, val_structs, val_targets = filter_data(val_ids, val_structs, val_targets)
    test_ids, test_structs, test_targets = filter_data(test_ids, test_structs, test_targets)

    print(f"Source: {source} | Train: {len(train_structs)}, Val: {len(val_structs)}, Test: {len(test_structs)}")
    
    if len(train_structs) == 0:
        print("Error: Empty training set.")
        sys.exit(1)

    run_dir = os.path.join(data_dir, f'alignn_run_{source}')
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(run_dir)

    try:
        # ALIGNN's val_loader uses drop_last=True, so batch_size must not exceed the val set size.
        n_val = len(val_structs)
        batch_size = 32
        if n_val > 0 and n_val < batch_size:
            batch_size = n_val
            print(f"Adjusting batch_size to {batch_size} to match validation set size.")

        with open("id_prop.csv", "w") as f:
            for i, s in enumerate(train_structs):
                mid = train_ids[i]
                atoms = pmg_to_atoms(s)
                atoms.write_poscar(mid)
                f.write(f"{mid},{train_targets[i]}\n")
            
            for i, s in enumerate(val_structs):
                mid = val_ids[i]
                atoms = pmg_to_atoms(s)
                atoms.write_poscar(mid)
                f.write(f"{mid},{val_targets[i]}\n")
                
            for i, s in enumerate(test_structs):
                mid = test_ids[i]
                atoms = pmg_to_atoms(s)
                atoms.write_poscar(mid)
                f.write(f"{mid},{test_targets[i]}\n")

        config = {
            "version": "112bbedebdaecf59fb18e11c929080fb2f358246",
            "dataset": "user_data",
            "target": "target",
            "atom_features": "cgcnn",
            "neighbor_strategy": "k-nearest",
            "id_tag": "jid",
            "random_seed": 123,
            "classification_threshold": None,
            "n_train": len(train_structs),
            "n_val": len(val_structs),
            "n_test": len(test_structs),
            "train_ratio": None,
            "val_ratio": None,
            "test_ratio": None,
            "target_multiplication_factor": None,
            "epochs": 200,
            "batch_size": batch_size,
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
            "keep_data_order": True,
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
        
        config_filename = "config.json"
        dumpjson(data=config, filename=config_filename)

        # Run training
        train_folder_path = os.path.join(os.path.dirname(sys.executable), "train_folder.py")
        cmd = (
            f"{sys.executable} {train_folder_path} "
            f"--root_dir '{run_dir}' "
            f"--config_name '{config_filename}' "
            f"--output_dir '{os.path.join(run_dir, 'output')}' "
            f"--file_format poscar"
        )
        print(f"Running ALIGNN: {cmd}")
        ret = os.system(cmd)
        
        if ret != 0:
            print("ALIGNN training failed.")
            sys.exit(1)
            
        test_csv = os.path.join(run_dir, "output", "prediction_results_test_set.csv")
        if os.path.exists(test_csv):
            df_res = pd.read_csv(test_csv)
            target_vals = df_res.target.values
            pred_vals = df_res.prediction.values
            
            mae = mean_absolute_error(target_vals, pred_vals)
            rmse = np.sqrt(mean_squared_error(target_vals, pred_vals))
            mape = robust_mape(target_vals, pred_vals)
            max_err = max_error(target_vals, pred_vals)
            
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
        else:
            print(f"Error: Results file {test_csv} not found.")
            sys.exit(1)

    finally:
        os.chdir(cwd)

if __name__ == "__main__":
    main()
