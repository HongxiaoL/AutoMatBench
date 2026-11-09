"""Generate the Problem 1/2/3 summary LaTeX tables from the same metric logic used by plot_all_rounds."""
import io, os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_all_rounds as P

def collect_row(state_label, model_name, task_name, metric_lbl, hb_default,
                state_json_path, gt_json_path, extra_refs, draw_refs, companions,
                hb_override, problem_label):
    """Re-implement draw_one's decision path up to best_y / best_i / params_obs to recover the best point for the table."""
    if task_name in P.SKIP_TASKS:
        return None
    if state_json_path is None or not os.path.exists(state_json_path):
        return None
    with open(state_json_path) as f:
        d = json.load(f)
    y = d.get('y_obs', []) or []
    params_obs = d.get('params_obs', []) or []
    if not y or not params_obs:
        return None
    y_obs = np.array([float(v) for v in y], dtype=float)
    is_cls = task_name in P.CLASSIFICATION_TASKS
    higher_better = hb_default if hb_override is None else bool(hb_override)
    left_ylabel = metric_lbl
    best_metric_word = metric_lbl
    if problem_label == 'Problem 2':
        left_ylabel = 'Absolute MAE gap'
        best_metric_word = '|MAE gap|'
    elif problem_label == 'Problem 3':
        if metric_lbl == 'ROC AUC':
            left_ylabel = 'Absolute deviation from reference ROC AUC'
            best_metric_word = 'Abs dev ROC AUC'
        else:
            left_ylabel = 'Absolute deviation from reference MAE'
            best_metric_word = 'Abs dev MAE'
    force_metallicity_as_rocdistance = (
        state_label == '2nd' and model_name == 'rfscm' and task_name == 'metallicity'
    )
    if force_metallicity_as_rocdistance:
        is_cls = False
        higher_better = False
    # Flip sign for negative-median regression (states may store -MAE).
    if (not is_cls):
        med = float(np.nanmedian(y_obs))
        if med < -1e-9:
            y_obs = -y_obs
    # Convert MAE-like classification values to ROC AUC (1-MAE), except the rocdistance case.
    if (not force_metallicity_as_rocdistance) and is_cls and higher_better \
            and (np.nanmax(y_obs) <= 1.0 + 1e-6) and (np.nanmin(y_obs) >= 0.0):
        if np.nanmean(y_obs) < 0.4999:
            y_obs = 1.0 - y_obs
    n_steps = len(y_obs)
    cap_drop = np.zeros(n_steps, dtype=bool)
    is_bulk_shear = (not is_cls) and task_name in ('bulk_modulus', 'shear_modulus')
    if is_bulk_shear:
        cap_drop |= (y_obs > P.BULK_SHEAR_CAP)
    valid = ~np.isnan(y_obs)
    # Bypass the MAD outlier filter for Problem 2 to keep the maximum |MAE gap|.
    bypass_mad = (problem_label == 'Problem 2')
    if not is_cls:
        if is_bulk_shear:
            drop = cap_drop | (~valid)
        else:
            if bypass_mad:
                drop = cap_drop | (~valid)
            else:
                vals = y_obs[valid & ~cap_drop]
                if vals.size >= 5:
                    med = np.median(vals)
                    mad = np.median(np.abs(vals - med))
                    if mad > 1e-12:
                        mz = 0.6745 * (y_obs - med) / mad
                        mad_mask = (np.abs(mz) > 5.0) & valid
                    else:
                        mad_mask = np.zeros(n_steps, dtype=bool)
                else:
                    mad_mask = np.zeros(n_steps, dtype=bool)
                drop = cap_drop | mad_mask | (~valid)
    else:
        drop = (y_obs < -1e-3) | (y_obs > 1.001)
        drop = drop | (~valid)
    y_plot = np.where(~drop, y_obs, np.nan)
    notnan = ~np.isnan(y_plot)
    if not notnan.any():
        return None
    best_i = int(np.nanargmax(y_plot) if higher_better else np.nanargmin(y_plot))
    best_y = float(y_plot[best_i])
    rv, nv, sv = params_obs[best_i][0], params_obs[best_i][1], params_obs[best_i][2]
    def fmt_rv(v):
        try:
            r = round(float(v), 1)
            if abs(r - round(r)) < 1e-9:
                return str(int(round(r)))
            return f'{r:.1f}'
        except Exception:
            return str(v)
    def fmt_int(v):
        try:
            return str(int(round(float(v))))
        except Exception:
            return str(v)
    rns_str = f'({fmt_rv(rv)}, {fmt_int(nv)}, {fmt_int(sv)})'
    return {
        'state': state_label,
        'model': model_name,
        'task': task_name,
        'metric_word': best_metric_word,
        'higher_better': bool(higher_better),
        'best_y': best_y,
        'best_k': best_i + 1,
        'n_steps': n_steps,
        'best_rns': rns_str,
        'problem': problem_label,
    }

def fmt_val(v, task):
    if task == 'metallicity':
        return f'{v:.4f}'
    absv = abs(v)
    if absv == 0:
        return '0'
    if absv < 0.01:
        return f'{v:.6g}'
    if absv < 1:
        return f'{v:.4f}'
    if absv < 100:
        return f'{v:.4f}'
    return f'{v:.6g}'

rows_p1 = []
rows_p2 = []
rows_p3 = []

for state_label, model_name, path_fn, extra_refs_fn, draw_refs, companions, hb_override, problem_label in P.STATES:
    for task_name, metric_lbl, hb_default in P.TASKS:
        state_json_path = path_fn(task_name)
        gt_json_path = P.GT_JSONS.get(task_name) if draw_refs else None
        extra_refs = extra_refs_fn(task_name) if callable(extra_refs_fn) else []
        row = collect_row(state_label, model_name, task_name, metric_lbl, hb_default,
                          state_json_path, gt_json_path, extra_refs, draw_refs,
                          companions, hb_override, problem_label)
        if row is None:
            continue
        if problem_label == 'Problem 1':
            rows_p1.append(row)
        elif problem_label == 'Problem 2':
            rows_p2.append(row)
        elif problem_label == 'Problem 3':
            rows_p3.append(row)

def task_tex(t):
    return {
        'band_gap': 'Band gap',
        'bulk_modulus': 'Bulk modulus',
        'formation_energy': 'Formation energy',
        'refractive_index': 'Refractive index',
        'shear_modulus': 'Shear modulus',
        'metallicity': 'Metallicity',
    }.get(t, t)

def model_tex(m):
    return {
        'rfscm': 'RF-SCM',
        'crabnet': 'CrabNet',
        'megnet': 'MEGNet',
        'schnet': 'SchNet',
        'alignn': 'ALIGNN',
        'crabnet_rfscm': 'CrabNet $\\oplus$ RF-SCM',
    }.get(m, m)

# ---------------- Problem 1 table ----------------
p1_task_order = ['band_gap','formation_energy','refractive_index','metallicity','bulk_modulus','shear_modulus']
p1_model_order = ['alignn','crabnet','megnet','schnet','rfscm']
# Sort band-gap models first in model order, then the remaining tasks.
def p1_sortkey(r):
    if r['task'] == 'band_gap':
        return (0, p1_model_order.index(r['model']) if r['model'] in p1_model_order else 99)
    else:
        return (1, p1_task_order.index(r['task']) if r['task'] in p1_task_order else 99)
rows_p1.sort(key=p1_sortkey)

print(r'\begin{table}[t]')
print(r'  \centering')
print(r'  \caption{Optimization results of all experiments for Problem 1: raw MAE (or ROC AUC for Metallicity) of the single-model BO search. RF-SCM covers all six tasks; ALIGNN / CrabNet / MEGNet / SchNet are only run on the band gap task.}')
print(r'  \label{tab:p1}')
print(r'  \begin{tabular}{llrrr}')
print(r'  \toprule')
print(r'  Model & Task & Best obs. & Best step & Best $(R,N,S)$ \\')
print(r'  \midrule')
for r in rows_p1:
    print(f'  {model_tex(r["model"]):<10} & {task_tex(r["task"]):<18} & {fmt_val(r["best_y"], r["task"]):>10} & {r["best_k"]:>2}/{r["n_steps"]:<2} & ${r["best_rns"]}$ \\\\')
print(r'  \bottomrule')
print(r'  \end{tabular}')
print(r'\end{table}')
print()

# ---------------- Problem 2 table ----------------
p2_order = ['band_gap','formation_energy','refractive_index','bulk_modulus','shear_modulus']
rows_p2.sort(key=lambda r: p2_order.index(r['task']) if r['task'] in p2_order else 99)
print(r'\begin{table}[t]')
print(r'  \centering')
print(r'  \caption{Optimization results of all experiments for Problem 2: maximizing the absolute MAE gap between CrabNet and RF-SCM under the same $(R,N,S)$ protocol. Metallicity is not run in the dual setting.}')
print(r'  \label{tab:p2}')
print(r'  \begin{tabular}{lrrr}')
print(r'  \toprule')
print(r'  Task & Best obs. $|\Delta\mathrm{MAE}|$ & Best step & Best $(R,N,S)$ \\')
print(r'  \midrule')
for r in rows_p2:
    print(f'  {task_tex(r["task"]):<18} & {fmt_val(r["best_y"], r["task"]):>14} & {r["best_k"]:>2}/{r["n_steps"]:<2} & ${r["best_rns"]}$ \\\\')
print(r'  \bottomrule')
print(r'  \end{tabular}')
print(r'\end{table}')
print()

# ---------------- Problem 3 table ----------------
p3_order = ['band_gap','formation_energy','refractive_index','metallicity','bulk_modulus','shear_modulus']
rows_p3.sort(key=lambda r: (0 if r['model']=='rfscm' else 1,
                            p3_order.index(r['task']) if r['task'] in p3_order else 99))
print(r'\begin{table}[t]')
print(r'  \centering')
print(r'  \caption{Optimization results of all experiments for Problem 3: minimizing the absolute deviation of the observed MAE (ROC AUC distance for Metallicity) from the reference / GT anchor value $(R=0,N=48,S=10)$. CrabNet has no Metallicity run in the second round.}')
print(r'  \label{tab:p3}')
print(r'  \begin{tabular}{llrrr}')
print(r'  \toprule')
print(r'  Model & Task & Best obs. (abs. dev.) & Best step & Best $(R,N,S)$ \\')
print(r'  \midrule')
for r in rows_p3:
    print(f'  {model_tex(r["model"]):<10} & {task_tex(r["task"]):<18} & {fmt_val(r["best_y"], r["task"]):>14} & {r["best_k"]:>2}/{r["n_steps"]:<2} & ${r["best_rns"]}$ \\\\')
print(r'  \bottomrule')
print(r'  \end{tabular}')
print(r'\end{table}')

plt.close('all')
