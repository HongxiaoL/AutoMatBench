"""Generate the Problem 1 multi-seed band gap LaTeX table from the same logic as plot_1st_bandgap_multiseed."""
import os, json, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_1st_bandgap_multiseed as P
from plot_1st_bandgap_multiseed import MODELS, SEEDS, load_state, best_so_far, fmt_rv, fmt_int

def fv(v):
    if abs(v) < 1e-9:
        return '0'
    if abs(v) < 0.01:
        return f'{v:.6g}'
    if abs(v) < 1:
        return f'{v:.4f}'
    if abs(v) < 100:
        return f'{v:.4f}'
    return f'{v:.6g}'

def fmt_pm(mean, std):
    return f'${fv(mean)} \\pm {fv(std)}$'

def fmt_range(lo, hi):
    return f'$[{fv(lo)}, {fv(hi)}]$'

def rns(r, n, s):
    return f'$({fmt_rv(r)}, {fmt_int(n)}, {fmt_int(s)})$'

rows = []
for model in MODELS:
    seed_stats = []
    for s in SEEDS:
        params, y, n, path = load_state(model, s)
        if y is None:
            print(f'  MISSING {path}')
            continue
        higher_better = False   # band gap MAE: lower is better
        bs = best_so_far(y, higher_better=higher_better)
        notnan = ~np.isnan(y)
        if not notnan.any():
            continue
        best_i = int(np.nanargmin(y))
        best_y = float(y[best_i])
        best_k = best_i + 1
        best_r, best_n, best_s = params[best_i][0], params[best_i][1], params[best_i][2]
        final_bsf = float(bs[-1])
        seed_stats.append({
            'seed': s,
            'best_y': best_y,
            'best_k': best_k,
            'best_rns': (best_r, best_n, best_s),
            'final_bsf': final_bsf,
            'n_steps': int(n),
        })
    if not seed_stats:
        continue
    best_ys = np.array([r['best_y'] for r in seed_stats])
    best_ks = np.array([r['best_k'] for r in seed_stats])
    final_bsfs = np.array([r['final_bsf'] for r in seed_stats])
    rows.append({
        'model': model,
        'mean_best': float(np.mean(best_ys)),
        'std_best':  float(np.std(best_ys)),
        'min_best':  float(np.min(best_ys)),
        'max_best':  float(np.max(best_ys)),
        'range_best': float(np.max(best_ys) - np.min(best_ys)),
        'mean_k':    float(np.mean(best_ks)),
        'std_k':     float(np.std(best_ks)),
        'ks_list':   [r['best_k'] for r in seed_stats],
        'mean_final_bsf': float(np.mean(final_bsfs)),
        'std_final_bsf':  float(np.std(final_bsfs)),
        'seed_rns':  [rns(r['best_rns'][0], r['best_rns'][1], r['best_rns'][2]) for r in seed_stats],
        'seed_best_y': [r['best_y'] for r in seed_stats],
        'seed_ks':   [r['best_k'] for r in seed_stats],
    })
    ks_txt = ', '.join([f'k={r["best_k"]}@{s}' for r,s in zip(seed_stats, SEEDS)])
    bsf_txt = f'mean±σ = {fv(np.mean(final_bsfs))} ± {fv(np.std(final_bsfs))}'
    print(f'{model:<8}  best MAE mean±σ = {fv(np.mean(best_ys))} ± {fv(np.std(best_ys))}   best_k = {ks_txt}   final_best-so-far: {bsf_txt}')

def model_tex(m):
    return {'crabnet':'CrabNet','megnet':'MEGNet','schnet':'SchNet'}.get(m,m)

def ks_tex(ks):
    return f'$[{", ".join(str(k) for k in ks)}]$'

print()
print(r'\begin{table}[t]')
print(r'  \centering')
print(r'  \caption{Multi-seed reproducibility summary for the Problem 1 band gap task (MAE, lower is better) across three BO seeds (0 / 42 / 200) for CrabNet, MEGNet and SchNet.  Shown: per-seed best observed MAE (mean±σ / range), the convergence steps the best MAE landed on, and the final best-so-far MAE at the last step (after forward-filling any missing evaluations).}')
print(r'  \label{tab:multiseed_bg}')
print(r'  \resizebox{\textwidth}{!}{%')
print(r'  \begin{tabular}{lrrrrrrrr}')
print(r'  \toprule')
print(r'  Model & Best obs.\ MAE ($\mu\pm\sigma$) & Best MAE min & Best MAE max & Best MAE range & Best step $k$ ($\mu\pm\sigma$; per-seed) & Final best-so-far MAE & Seed-wise best $(R,N,S)$ \\')
print(r'  \midrule')
for r in rows:
    seed_rns_joined = ', '.join([f'${s[1:-1]}$' for s in r['seed_rns']])
    seed_rns_joined = ', '.join(r['seed_rns'])
    print(f'  {model_tex(r["model"]):<8} & {fmt_pm(r["mean_best"], r["std_best"]):<22} & {fv(r["min_best"]):>10} & {fv(r["max_best"]):>10} & {fv(r["range_best"]):>10} & {fmt_pm(r["mean_k"], r["std_k"])} {ks_tex(r["seed_ks"])} & {fmt_pm(r["mean_final_bsf"], r["std_final_bsf"])} & {seed_rns_joined} \\\\')
print(r'  \bottomrule')
print(r'  \end{tabular}%')
print(r'  }')
print(r'\end{table}')
