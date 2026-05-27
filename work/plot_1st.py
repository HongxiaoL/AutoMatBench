import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
import glob
import json
from pathlib import Path
import os
import sys

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from work.common.params import R_values, N_values, S_values
json_dir = os.path.join(parent_dir, 'work/json')
plots_dir = os.path.join(parent_dir, 'work/figures')

def format_title(filename):
    name = filename.replace('.json', '')
    if name.endswith('_opt'):
        name = name[:-4]
    if name.startswith('optimization_'):
        name = name[len('optimization_'):]
        
    state_map = {
        '1st_state': '',
        '2nd_state': 'Second class',
        '1st_dual_state': 'Dual'
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
    return ", ".join(parts) if parts else filename

def plot_json_data(json_path, ax, label=None, is_rightmost=False):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    y_obs = data.get("y_obs", [])[:]
    params_obs = data.get("params_obs", [])[:]

    if not y_obs or not params_obs:
        print(f"Skipping {json_path}: empty data")
        return

    num_points = min(len(y_obs), len(params_obs))
    y_obs = np.array(y_obs[:num_points])
    params_obs = np.array(params_obs[:num_points])

    r_val = params_obs[:, 0]
    n_val = params_obs[:, 1]
    s_val = params_obs[:, 2]
    mae_val = y_obs

    def get_index(val, val_list):
        try:
            return np.argmin([abs(v - val) for v in val_list])
        except:
            return 0

    r_idx = np.array([get_index(v, R_values) for v in r_val])
    n_idx = np.array([get_index(v, N_values) for v in n_val])
    s_idx = np.array([get_index(v, S_values) for v in s_val])

    norm = Normalize(vmin=float(mae_val.min()), vmax=float(mae_val.max()))
    mae_norm = norm(mae_val)
    cmap = colormaps.get_cmap("viridis")

    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 60)
    phi, theta = np.meshgrid(phi, theta)
    ux = np.sin(phi) * np.cos(theta)
    uy = np.sin(phi) * np.sin(theta)
    uz = np.cos(phi)

    ax.set_box_aspect((len(R_values), len(N_values), len(S_values)))

    ax.set_xlim(-0.5, len(R_values) - 0.5)
    ax.set_ylim(-0.5, len(N_values) - 0.5)
    ax.set_zlim(-0.5, len(S_values) - 0.5)

    r_ticks = range(0, len(R_values), 2)
    ax.set_xticks(r_ticks)
    ax.set_xticklabels([f"{R_values[i]:.1f}" for i in r_ticks], fontsize=12)
    
    n_ticks = range(0, len(N_values), 2)
    ax.set_yticks(n_ticks)
    ax.set_yticklabels([str(N_values[i]) for i in n_ticks], fontsize=12)
    
    s_ticks = range(0, len(S_values), 2)
    ax.set_zticks(s_ticks)
    ax.set_zticklabels([str(S_values[i]) for i in s_ticks], fontsize=12)

    ax.set_xlabel("Ratio r", fontsize=16, labelpad=10)
    ax.set_ylabel("Folds n", fontsize=16, labelpad=10)
    ax.set_zlabel("")
    if is_rightmost:
        ax.text2D(1.08, 0.5, "test samples s", transform=ax.transAxes, 
                  rotation=90, fontsize=16, ha='center', va='center')
    
    filename = os.path.basename(json_path)
    title_str = format_title(filename)
    if label:
        title_str = f"{label} {title_str}"
    ax.set_title(title_str, y=-0.35, fontsize=20)

    radius_base = 0.35
    r_min_ratio, r_max_ratio = 0.5, 2.5

    for i in range(num_points):
        scale = r_min_ratio + mae_norm[i] * (r_max_ratio - r_min_ratio)
        r = radius_base * scale
        
        X = r_idx[i] + r * ux
        Y = n_idx[i] + r * uy
        Z = s_idx[i] + r * uz
        color = cmap(mae_norm[i])
        rgba = np.empty(X.shape + (4,), dtype=float)
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = 0.6
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgba, linewidth=0, antialiased=True, shade=True)

    ax.view_init(elev=35.264, azim=45)




if __name__ == "__main__":
    os.makedirs(plots_dir, exist_ok=True)
    
    pattern = os.path.join(json_dir, "*optimization_1st_state*.json")
    json_files = glob.glob(pattern)
    json_files.sort()
    
    num_files = len(json_files)
    if num_files == 0:
        print("No files found.")
        sys.exit(0)
        
    cols = 5
    rows = (num_files + 4) // 5
    
    fig = plt.figure(figsize=(cols * 6, rows * 5))
    
    for i, file in enumerate(json_files):
        print(f"Processing: {file}")
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d', proj_type="ortho")
        label = f"({chr(ord('a') + i)})"
        is_rightmost = (i + 1) % cols == 0
        plot_json_data(file, ax, label=label, is_rightmost=is_rightmost)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=-0.45)
    
    output_path = os.path.join(plots_dir, "combined_1st_plots.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined plot: {output_path}")
