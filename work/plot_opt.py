import numpy as np
import matplotlib.pyplot as plt
import glob
import json
from pathlib import Path
import os
import sys

import math

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
json_dir = os.path.join(parent_dir, 'work/json')
plots_dir = os.path.join(parent_dir, 'work/figures')

def format_title(filename):
    name = filename.replace('.json', '')
    if name.endswith('_opt'):
        name = name[:-4]
    if name.startswith('optimization_'):
        name = name[len('optimization_'):]
        
    state_map = {
        '1st_state': 'First class',
        '2nd_state': 'Second class',
        '1st_dual_state': 'Dual of first class'
    }
    
    state_str = ""
    for k, v in state_map.items():
        if name.startswith(k + '_'):
            state_str = v
            name = name[len(k) + 1:]
            break
            
    models = ['crabnet_rfscm', 'rfscm', 'crabnet', 'alignn', 'megnet', 'schnet']
    model_str = ""
    for m in models:
        if name.startswith(m + '_'):
            model_str = m
            name = name[len(m) + 1:]
            break
            
    prop_str = name
    
    if model_str == 'crabnet_rfscm':
        model_str = 'crabnet & rfscm'
    
    parts = [p for p in [state_str, model_str, prop_str] if p]
    return ", ".join(parts[1:]) if len(parts) > 1 else filename

def draw_plot_on_ax(ax, json_path, index):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    y_obs = data.get("y_obs", [])[:]
    params_obs = data.get("params_obs", [])[:]

    if not y_obs or not params_obs:
        ax.set_visible(False)
        return

    num_points = min(len(y_obs), len(params_obs))
    y_obs = np.array(y_obs[:num_points])
    params_obs = np.array(params_obs[:num_points])

    r_val = params_obs[:, 0]
    n_val = params_obs[:, 1]
    s_val = params_obs[:, 2]
    mae_val = y_obs

    steps = np.arange(1, num_points + 1)
    
    y_blue = np.zeros_like(mae_val)
    split_step = num_points // 3 if num_points >= 3 else 0
    filename = os.path.basename(json_path)
    is_dual = "1st_dual_state" in filename
    
    if split_step > 0:
        y_blue[:split_step] = mae_val[:split_step]
        if is_dual:
            y_blue[split_step:] = np.maximum.accumulate(mae_val[split_step:])
        else:
            y_blue[split_step:] = np.minimum.accumulate(mae_val[split_step:])
    else:
        if is_dual:
            y_blue = np.maximum.accumulate(mae_val)
        else:
            y_blue = np.minimum.accumulate(mae_val)
    
    blue_label = 'Exploration / Maximum' if is_dual else 'Exploration / Minimum'
    if split_step > 0:
        ax.plot(steps[:split_step+1], y_blue[:split_step+1], marker='o', linestyle='--', color='b', label=blue_label)
        ax.plot(steps[split_step:], y_blue[split_step:], marker='o', linestyle='-', color='b')
    else:
        ax.plot(steps, y_blue, marker='o', linestyle='-', color='b', label=blue_label)
    
    ax.scatter(steps, mae_val, color='red', zorder=5, label='New value')

    if is_dual:
        best_idx = np.argmax(mae_val)
    else:
        best_idx = np.argmin(mae_val)
    
    ax.scatter(steps[best_idx], mae_val[best_idx], 
               s=150, facecolors='none', edgecolors='gold', 
               linewidth=2, zorder=10, label='Best configuration')

    y_label = "y_obs"
    if "1st_dual_state" in filename:
        y_label = "Range"
    elif "2nd_state" in filename:
        y_label = "Distance"
    elif "1st_state" in filename:
        y_label = "MAE"
        
    ax.set_ylabel(y_label, fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14)

    x_labels = [f"r={r_val[i]:.2g}\nn={n_val[i]:.2g}\ns={s_val[i]:.2g}" for i in range(num_points)]
    ax.set_xticks(steps)
    ax.set_xticklabels(x_labels, ha='center', fontsize=14)
    
    ax.get_xticklabels()[best_idx].set_weight('bold')
    
    if num_points >= 3:
        ax.axvline(x=split_step + 0.5, color='black', linestyle='--', linewidth=1.5, zorder=3)
    
    title_str = format_title(filename)
    label_char = chr(ord('a') + index)
    full_title = f"({label_char}) {title_str}"
    ax.set_title(full_title, y=-0.35, fontsize=18, fontweight='normal')
    
    ax.legend(fontsize=12, loc='best', framealpha=0.7)

if __name__ == "__main__":
    os.makedirs(plots_dir, exist_ok=True)
    
    pattern = os.path.join(json_dir, "*optimization*.json")
    json_files = sorted(glob.glob(pattern))
    
    categories = {
        'First class': [f for f in json_files if '1st_state' in f and 'dual' not in f],
        'Dual of first class': [f for f in json_files if '1st_dual_state' in f],
        'Second class': [f for f in json_files if '2nd_state' in f]
    }
    
    for cat_name, files in categories.items():
        if not files:
            continue
            
        num_files = len(files)
        ncols = 2
        nrows = math.ceil(num_files / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6 * nrows))
        # fig.suptitle(cat_name, fontsize=16, fontweight='bold')
        
        if num_files == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten()
            
        for i, file in enumerate(files):
            print(f"Processing ({cat_name}): {file}")
            draw_plot_on_ax(axes_list[i], file, i)
            
        for j in range(i + 1, len(axes_list)):
            axes_list[j].set_visible(False)
            
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.subplots_adjust(hspace=0.6)
        
        output_filename = cat_name.lower().replace(" ", "_") + "_summary.png"
        output_path = os.path.join(plots_dir, output_filename)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved summary: {output_path}")
