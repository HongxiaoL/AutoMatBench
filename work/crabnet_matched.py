import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error

# We assume this script is run in an environment where crabnet is installed
try:
    from crabnet.crabnet_ import CrabNet
except ImportError:
    print("Error: CrabNet not installed.")
    sys.exit(1)

def robust_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > 1e-5
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing matched_data.csv and splits.json')
    parser.add_argument('--source', required=True, choices=['jarvis', 'mp'], help='Source of data to use')
    parser.add_argument('--task', required=True, help='Task name (e.g. band_gap, refractive_index)')
    parser.add_argument('--fold', required=False, type=int, default=0, help='Fold index')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    source = args.source
    task_name = args.task
    fold = args.fold
    
    target_col = f'jarvis_{task_name}' if source == 'jarvis' else f'mp_{task_name}'
    
    csv_path = os.path.join(data_dir, 'matched_data.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    splits_path = os.path.join(data_dir, 'splits.json')
    if not os.path.exists(splits_path):
        print(f"Error: {splits_path} not found")
        sys.exit(1)
        
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Prepare DataFrames for CrabNet
    # CrabNet expects columns: 'formula', 'target'
    if fold >= len(splits):
        print(f"Error: Fold {fold} out of range (max {len(splits)-1})")
        sys.exit(1)
        
    split = splits[fold]
    train_df = df.iloc[split['train']][['formula', target_col]].rename(columns={target_col: 'target'})
    val_df = df.iloc[split['val']][['formula', target_col]].rename(columns={target_col: 'target'})
    test_df = df.iloc[split['test']][['formula', target_col]].rename(columns={target_col: 'target'})

    print(f"Source: {source} | Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train
    cb = CrabNet(mat_prop=task_name, verbose=False, classification=False)
    cb.fit(train_df=train_df, val_df=val_df)

    # Predict
    test_df_pred = test_df.copy()
    test_df_pred['target'] = np.nan
    preds = cb.predict(test_df_pred)
    
    if hasattr(preds, 'columns') and 'prediction' in preds.columns:
        y_pred = preds['prediction'].values
    else:
        y_pred = preds
    
    y_true = test_df['target'].values
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = robust_mape(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    
    # Save results
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
