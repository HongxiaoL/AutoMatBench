import os
import sys
import json
import pandas as pd
import numpy as np
import subprocess
import logging
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from common.logging_utils import setup_dual_logging

method_name = "rfscm"
task_name = "refractive_index"
N = 5
seed = 42

# Setup Logging
parent_dir = str(Path(__file__).resolve().parent.parent)
log_dir = os.path.join(parent_dir, 'work', 'log')
os.makedirs(log_dir, exist_ok=True)

def setup_logging(task_name, method_name):
    log_file = os.path.join(log_dir, f'matched_run_{method_name}_{task_name}.log')
    setup_dual_logging(log_file, level=logging.INFO, reset_handlers=True)

home_dir = os.path.expanduser("~")
env_paths = {
    "alignn": os.path.join(home_dir, "anaconda3/envs/alignn_env/bin/python"),
    "schnet": os.path.join(home_dir, "anaconda3/envs/schnet_env/bin/python"),
    "crabnet": os.path.join(home_dir, "anaconda3/envs/matbench_env/bin/python"),
    "rfscm": os.path.join(home_dir, "anaconda3/envs/automatminer_env/bin/python"),
    "megnet": os.path.join(home_dir, "anaconda3/envs/megnet_env/bin/python")
}

def prepare_matched_data(task_name, method_name):
    """
    Reads matched_materials_{task_name}.csv and prepares 5-fold cross-validation splits.
    Returns the directory where data is saved.
    """
    csv_name = f'matched_materials_{task_name}.csv'
    csv_path = os.path.join(parent_dir, 'automatbench', csv_name)
    if not os.path.exists(csv_path):
        # Try alternate location
        csv_path = os.path.join(parent_dir, csv_name)
        if not os.path.exists(csv_path):
            logging.error(f"Error: {csv_name} not found in {parent_dir} or {os.path.join(parent_dir, 'automatbench')}.")
            return None

    logging.info(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} matched materials for {task_name}.")

    output_dir = os.path.join(parent_dir, 'work', 'matched_result', f'matched_run_{method_name}_{task_name}')
    os.makedirs(output_dir, exist_ok=True)

    # Generate 5-fold splits
    kf = KFold(n_splits=N, shuffle=True, random_state=seed)
    splits = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        # Further split train into train/val (e.g. 90/10 of the training set)
        # Or just use train/test for CV if validation is handled internally or not needed
        # Standard CV usually just has Train/Test per fold. 
        # Some models (like CrabNet, ALIGNN) need a validation set.
        
        # Let's take 10% of the training set for validation
        train_idx_final, val_idx = train_test_split(train_index, test_size=0.1, random_state=42)
        
        splits.append({
            "train": train_idx_final.tolist(),
            "val": val_idx.tolist(),
            "test": test_index.tolist()
        })

    with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f)

    # Save CSV for the runner
    df.to_csv(os.path.join(output_dir, 'matched_data.csv'), index=False)
    
    logging.info(f"Data prepared in {output_dir} with 5-fold CV splits.")
    return output_dir

def run_model_on_source(data_dir, method, task_name, source='jarvis', fold=0):
    """
    Runs the model on either JARVIS or MP data for a specific fold.
    """
    runner_script = f"{method}_matched.py"
    runner_script_path = os.path.join(parent_dir, 'work', runner_script)
    
    if not os.path.exists(runner_script_path):
        logging.error(f"Error: {runner_script_path} not found.")
        return

    python_path = env_paths.get(method)
    if not python_path:
        logging.error(f"Error: Python path for {method} not found.")
        return
    
    cmd = [python_path, runner_script_path, '--data_dir', data_dir, '--source', source, '--task', task_name, '--fold', str(fold)]
    logging.info(f"Running {method} on {source} (Fold {fold}): {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {method} on {source} (Fold {fold}): {e}")

def main():
    setup_logging(task_name, method_name)
    
    logging.info("#########################################################################################")
    logging.info(f"Running Models on Matched JARVIS and MP Data for {task_name} using {method_name} with 5-fold CV")
    
    data_dir = prepare_matched_data(task_name, method_name)
    if not data_dir:
        return

    methods = [method_name]
    
    # Load existing results if available (Resuming capability)
    json_output_path = os.path.join(data_dir, f'comparison_results_{method_name}_{task_name}.json')
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, 'r') as f:
                comparison_results = json.load(f)
            logging.info(f"Loaded existing results from {json_output_path}")
        except Exception as e:
            logging.error(f"Error loading existing results: {e}")
            comparison_results = []
    else:
        comparison_results = []

    # Ensure we have an entry for the current method
    method_result = next((r for r in comparison_results if r['method'] == method_name), None)
    if not method_result:
        method_result = {
            "method": method_name,
            "jarvis": {},
            "mp": {},
            "folds": []
        }
        comparison_results.append(method_result)
    
    # Get existing fold indices
    existing_folds = {f['fold'] for f in method_result['folds']}

    metrics = ['mae', 'rmse', 'mape', 'max_error']

    for method in methods:
        logging.info(f"\n==================== {method.upper()} ====================")
        
        for fold in range(5):
            if fold in existing_folds:
                logging.info(f"Skipping Fold {fold} (Already completed)")
                continue

            logging.info(f"--- Fold {fold} ---")
            
            # Clean up potential previous run artifacts
            for s in ['jarvis', 'mp']:
                res_path = os.path.join(data_dir, f'results_{s}.json')
                if os.path.exists(res_path):
                    os.remove(res_path)

            logging.info(f"--- Running on JARVIS Data ---")
            run_model_on_source(data_dir, method, task_name, 'jarvis', fold)

            logging.info(f"--- Running on MP Data ---")
            run_model_on_source(data_dir, method, task_name, 'mp', fold)

            # Read results and compare
            res_j_path = os.path.join(data_dir, f'results_jarvis.json')
            res_m_path = os.path.join(data_dir, f'results_mp.json')
            
            if os.path.exists(res_j_path) and os.path.exists(res_m_path):
                with open(res_j_path, 'r') as f: res_j = json.load(f)
                with open(res_m_path, 'r') as f: res_m = json.load(f)

                # Move results to method-specific filenames to avoid overwriting
                os.rename(res_j_path, os.path.join(data_dir, f'results_{method}_jarvis_fold{fold}.json'))
                os.rename(res_m_path, os.path.join(data_dir, f'results_{method}_mp_fold{fold}.json'))

                fold_res = {
                    "fold": fold,
                    "jarvis": res_j,
                    "mp": res_m
                }
                method_result['folds'].append(fold_res)
                
                # Update averages and save IMMEDIATELY
                # Calculate average metrics across folds so far
                if method_result['folds']:
                    avg_jarvis = {m: np.mean([r['jarvis'][m] for r in method_result['folds']]) for m in metrics}
                    avg_mp = {m: np.mean([r['mp'][m] for r in method_result['folds']]) for m in metrics}
                    method_result['jarvis'] = avg_jarvis
                    method_result['mp'] = avg_mp

                try:
                    with open(json_output_path, 'w') as f:
                        json.dump(comparison_results, f, indent=4)
                    logging.info(f"Results updated and saved to {json_output_path} (Fold {fold} complete)")
                except Exception as e:
                    logging.error(f"Error saving comparison results: {e}")

            else:
                logging.warning(f"Warning: Results missing for {method} fold {fold}")

    print("\n#########################################################################################")
    print(f"Summary of Differences (JARVIS vs MP) for {task_name} (5-Fold CV Average)")
    print(f"{'Method':<10} | {'Metric':<10} | {'JARVIS':<10} | {'MP':<10} | {'Diff':<10}")
    print("-" * 60)

    for res in comparison_results:
        method = res['method']
        for metric in metrics:
            val_j = res['jarvis'].get(metric, 0.0)
            val_m = res['mp'].get(metric, 0.0)
            diff = abs(val_j - val_m)
            print(f"{method:<10} | {metric.upper():<10} | {val_j:.4f}     | {val_m:.4f}     | {diff:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
