import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from pymatgen.core import Structure

# Change into log_dir while importing automatminer so its logs land there.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(parent_dir, 'work', 'log')
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
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
finally:
    os.chdir(original_cwd)

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
        print(f"Tried paths: {paths_to_try}")
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

    full_train_structs = train_structs + val_structs
    full_train_targets = train_targets + val_targets

    train_df = pd.DataFrame({'structure': full_train_structs, 'target': full_train_targets})
    train_df = train_df.dropna(subset=['structure'])
    
    test_df = pd.DataFrame({'structure': test_structs, 'target': test_targets})
    test_df = test_df.dropna(subset=['structure'])

    print(f"Source: {source} | Train: {len(train_df)}, Test: {len(test_df)}")
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: Empty training or testing set after loading structures.")
        sys.exit(1)

    learner = SinglePipelineAdaptor(
        regressor=RandomForestRegressor(n_estimators=500),
        classifier=RandomForestClassifier(n_estimators=500)
    )
    pipe_config = {
        "learner": learner,
        "reducer": FeatureReducer(reducers=[]),
        "cleaner": DataCleaner(feature_na_method="mean", max_na_frac=0.01, na_method_fit="drop", na_method_transform="mean"),
        "autofeaturizer": AutoFeaturizer(n_jobs=10, preset="debug"),
    }

    cwd_before = os.getcwd()
    try:
        os.chdir(log_dir)
        pipe = MatPipe(**pipe_config)
        print("Fitting pipeline...")
        pipe.fit(train_df, 'target')
        print("Predicting...")
        df_pred = pipe.predict(test_df)
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        sys.exit(1)
    finally:
        os.chdir(cwd_before)

    y_pred = df_pred["target predicted"].values
    y_true = test_df['target'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = robust_mape(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    
    result = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'max_error': max_err
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
