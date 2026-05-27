import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
import random
import numpy as np
import logging
from common import params
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from common.logging_utils import setup_dual_logging

# Setup Logging
parent_dir = str(Path(__file__).resolve().parent.parent)
log_dir = os.path.join(parent_dir, 'work', 'log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'optimization_1st_dual.log')

setup_dual_logging(log_file, level=logging.INFO, reset_handlers=True)

R_values = params.R_values
N_values = params.N_values
S_values = params.S_values
method_name_A = "crabnet"
method_name_B = "rfscm"
task_name = "shear_modulus"
opt_index = "mae"
budget = 12
seed = 42
simulation = False

# Resume Capability
state_file = os.path.join(parent_dir, 'work', f'optimization_1st_dual_state_{method_name_A}_{method_name_B}_{task_name}.json')

def save_state(evaluated_idx, X_obs, y_obs, params_obs, n_init_done, steps_done):
    state = {
        "evaluated_idx": list(evaluated_idx),
        "X_obs": [x.tolist() for x in X_obs],
        "y_obs": y_obs,
        "params_obs": params_obs,
        "n_init_done": n_init_done,
        "steps_done": steps_done
    }
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)
    logging.info(f"State saved to {state_file}")

def load_state():
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        logging.info(f"Resuming from state file: {state_file}")
        return (
            set(state["evaluated_idx"]),
            [np.array(x) for x in state["X_obs"]],
            state["y_obs"],
            state["params_obs"],
            state.get("n_init_done", False),
            state.get("steps_done", 0)
        )
    return set(), [], [], [], False, 0

random.seed(seed)
np.random.seed(seed)

CANDIDATES = []
for r in R_values:
    for n in N_values:
        for s in S_values:
            if n == 60 and s in [9, 10]:
                continue
            CANDIDATES.append((r, n, s))

def one_hot(r, n, s):
    x = []
    for rv in R_values: x.append(1 if r == rv else 0)
    for nv in N_values: x.append(1 if n == nv else 0)
    for sv in S_values: x.append(1 if s == sv else 0)
    return np.array(x, dtype=float)

X_all = np.array([one_hot(*c) for c in CANDIDATES])

home_dir = os.path.expanduser("~")
env_paths = {
    "alignn": os.path.join(home_dir, "anaconda3/envs/alignn_env/bin/python"),
    "schnet": os.path.join(home_dir, "anaconda3/envs/schnet_env/bin/python"),
    "crabnet": os.path.join(home_dir, "anaconda3/envs/matbench_env/bin/python"),
    "rfscm": os.path.join(home_dir, "anaconda3/envs/automatminer_env/bin/python"),
    "megnet": os.path.join(home_dir, "anaconda3/envs/megnet_env/bin/python")
}

def run_single_method(method, R, N, S):
    """Executes a single method and returns its result."""
    if simulation:
        # Define two different convex functions (simulating MAE)
        if method == method_name_A:
            # Function A: Minimum at (0.5, 30, 5)
            val = (R - 0.5)**2 + 0.0005 * (N - 30)**2 + 0.02 * (S - 5)**2
        else:
            # Function B: Minimum at (0.2, 10, 8)
            val = (R - 0.2)**2 + 0.0005 * (N - 10)**2 + 0.02 * (S - 8)**2
        return val

    # Real Execution
    parent_dir = str(Path(__file__).resolve().parent.parent)
    result_dir = os.path.join(parent_dir, 'work', 'result', f'{method}_{task_name}_({R},{N},{S})')
    
    # Check if result already exists
    result_file = os.path.join(result_dir, "results.json")
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                avg_metrics = data.get('average_metrics', {})
                val = avg_metrics.get(opt_index)
                if val is not None:
                    print(f"Result already exists for {method}. {opt_index}: {val}")
                    return val
        except:
            pass # If reading fails, proceed to run

    if method not in env_paths:
        print(f"Error: Unknown method {method}")
        return None
        
    python_path = env_paths[method]
    script_path = os.path.join(parent_dir, "work", f"{method}_run.py")
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found")
        return None
    
    cmd = [
        python_path, 
        "-c", 
        f"import sys; import os; sys.path.append('{parent_dir}'); os.chdir('{parent_dir}/work'); import {method}_run as m; m.run('{task_name}', {R}, {N}, {S})"
    ]
    
    print(f"Running command for {method}: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {method}: {e}")
        return None

    # Read results
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                avg_metrics = data.get('average_metrics', {})
                val = avg_metrics.get(opt_index)
                print(f"Execution finished. {opt_index}: {val}")
                return val
        except Exception as e:
            print(f"Error reading results.json: {e}")
            return None
    else:
        print(f"Warning: results.json not found at {result_file}")
        return None

def expensive_f(R, N, S):
    print("#########################################################################################")
    print(f"Evaluating {task_name} with R={R}, N={N}, S={S}")

    parent_dir = str(Path(__file__).resolve().parent.parent)
    
    # --- Check and Generate Data ---
    if not simulation:
        data_gen_script = os.path.join(parent_dir, 'work', 'optimization_data_gen.py')
        if os.path.exists(data_gen_script):
            print(f"Checking/Generating data for ({task_name}, {R}, {N}, {S})...")
            work_dir = os.path.join(parent_dir, 'work')
            gen_cmd = [
                sys.executable,
                "-c",
                f"import sys; sys.path.append('{parent_dir}'); sys.path.append('{work_dir}'); import process_data; process_data.process_data('{task_name}', {R}, {N}, {S})"
            ]
            try:
                subprocess.run(gen_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error generating data: {e}")
                # return None # Optionally stop if data gen fails
        else:
            print(f"Warning: {data_gen_script} not found. Skipping data generation.")
            
    print("-----------------------------------------------------------------------------------------")

    # Execute both models
    print(f"Executing Method A: {method_name_A}")
    res_A = run_single_method(method_name_A, R, N, S)
    
    print("-----------------------------------------------------------------------------------------")
    
    print(f"Executing Method B: {method_name_B}")
    res_B = run_single_method(method_name_B, R, N, S)
    
    print("-----------------------------------------------------------------------------------------")
    
    if res_A is None or res_B is None:
        print("Failed to get results from one or both methods.")
        return None

    if simulation:
        print(f"Simulated A: {res_A}, B: {res_B}")
    
    # Calculate difference (Absolute difference)
    diff = abs(res_A - res_B)
    print(f"Difference ({opt_index}): {diff}")
    
    return diff

def expected_improvement(mu, sigma, y_best, xi=0.0):
    # EI for maximization
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


n_init = round(budget*0.3)  # Initial random exploration count
evaluated_idx, X_obs, y_obs, params_obs, n_init_done, steps_done = load_state()

# Initial Random Exploration
if not n_init_done:
    print("Initial random exploration...")
    # Filter out already evaluated indices from random sample if needed, 
    # but here we just need to complete n_init samples.
    # If resuming, we might have some samples already.
    remaining_init = n_init - len(evaluated_idx)
    if remaining_init > 0:
        # Avoid resampling already evaluated
        available_indices = [i for i in range(len(CANDIDATES)) if i not in evaluated_idx]
        init_indices = random.sample(available_indices, min(remaining_init, len(available_indices)))
        
        for idx in init_indices:
            r, n, s = CANDIDATES[idx]
            y = expensive_f(r, n, s)
            if y is None:
                print(f"Warning: Failed to evaluate ({r}, {n}, {s}). Skipping.")
                evaluated_idx.add(idx)
                continue
            X_obs.append(X_all[idx])
            y_obs.append(y)
            params_obs.append(CANDIDATES[idx])
            evaluated_idx.add(idx)
            save_state(evaluated_idx, X_obs, y_obs, params_obs, False, 0)
    
    n_init_done = True
    save_state(evaluated_idx, X_obs, y_obs, params_obs, True, 0)

# Iterative Selection (Remaining budget)
print("Iterative selection...")
remaining_steps = budget - len(evaluated_idx)
while len(evaluated_idx) < budget:
    t = len(evaluated_idx)
    # Check if we have enough data to fit
    if not y_obs:
        print("No valid observations to fit model. Randomly selecting next candidate.")
        remaining_indices = [i for i in range(len(CANDIDATES)) if i not in evaluated_idx]
        if not remaining_indices:
            print("All candidates evaluated.")
            break
        best_idx = random.choice(remaining_indices)
        
        # Evaluate randomly selected candidate
        r, n, s = CANDIDATES[best_idx]
        y = expensive_f(r, n, s)
        if y is None:
            print(f"Warning: Failed to evaluate ({r}, {n}, {s}). Skipping.")
            evaluated_idx.add(best_idx)
            save_state(evaluated_idx, X_obs, y_obs, params_obs, True, t)
            continue
        X_obs.append(X_all[best_idx])
        y_obs.append(y)
        params_obs.append(CANDIDATES[best_idx])
        evaluated_idx.add(best_idx)
        save_state(evaluated_idx, X_obs, y_obs, params_obs, True, t)
        continue

    # We want to MAXIMIZE the difference, so y_target is just y_obs (diff)
    y_target = np.array(y_obs)

    # Fit Random Forest surrogate
    rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=None, 
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(np.vstack(X_obs), y_target)

    # Predict mean and variance for all unevaluated points
    all_preds = np.stack([est.predict(X_all) for est in rf.estimators_], axis=0)
    mu_all = all_preds.mean(axis=0)
    var_all = all_preds.var(axis=0)

    y_best = max(y_target)
    best_ei, best_idx = -1.0, None
    for idx in range(len(CANDIDATES)):
        if idx in evaluated_idx: continue
        mu = mu_all[idx]; sigma = np.sqrt(var_all[idx])
        ei = expected_improvement(mu, sigma, y_best)
        if ei > best_ei:
            best_ei, best_idx = ei, idx
            
    # Fallback if no EI improvement found (should be rare unless all EI are 0)
    if best_idx is None:
        remaining_indices = [i for i in range(len(CANDIDATES)) if i not in evaluated_idx]
        if not remaining_indices:
            print("All candidates evaluated.")
            break
        best_idx = random.choice(remaining_indices)

    # Evaluate best candidate
    r, n, s = CANDIDATES[best_idx]
    y = expensive_f(r, n, s)
    if y is None:
        print(f"Warning: Failed to evaluate ({r}, {n}, {s}). Skipping.")
        evaluated_idx.add(best_idx)
        save_state(evaluated_idx, X_obs, y_obs, params_obs, True, t)
        continue
    X_obs.append(X_all[best_idx])
    y_obs.append(y)
    params_obs.append(CANDIDATES[best_idx])
    evaluated_idx.add(best_idx)
    save_state(evaluated_idx, X_obs, y_obs, params_obs, True, t)

# Output Best
if y_obs:
    # Find index with MAXIMUM difference
    best_i = int(np.argmax(y_obs))
    best_y = y_obs[best_i]
    best_params = params_obs[best_i]
    print("Steps: ", budget)
    print("Max Difference:", best_y)
    print("Optimal params (r,n,s):", best_params)
else:
    print("No successful evaluations found.")
