#!/usr/bin/env python3
"""Multi-seed convergence plots for the 1st-round band gap task across seeds 0, 42 and 200.
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK_DIR = str(Path(__file__).resolve().parent)
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

EXPLORE_STEPS = round(0.3 * 24)
TASK = 'band_gap'
METRIC = 'MAE'
HIGHER_BETTER = False  # MAE: lower is better

MODELS = ['crabnet', 'megnet', 'schnet']
SEEDS = [0, 42, 200]

SEED_STYLES = {
    0:   {'color': '#2E5FAC', 'marker': 'o', 'label': 'seed 0'},
    42:  {'color': '#56A868', 'marker': 's', 'label': 'seed 42'},
    200: {'color': '#C44E52', 'marker': '^', 'label': 'seed 200'},
}


def fmt_rv(r):
    if r is None:
        return '?'
    if isinstance(r, (int, np.integer)) or float(r).is_integer():
        return str(int(r))
    return f'{float(r):g}'


def fmt_int(x):
    if x is None:
        return '?'
    return str(int(x)) if isinstance(x, (int, np.integer)) else str(x)


def load_state(model, seed):
    p = os.path.join(WORK_DIR, f'optimization_1st_state_{model}_{TASK}_seed{seed}.json')
    if not os.path.exists(p):
        return None, None, None, p
    with open(p) as f:
        d = json.load(f)
    params = d.get('params_obs', []) or []
    y = d.get('y_obs', []) or d.get('y_raw_obs', []) or []
    if len(y) == 0 or len(params) == 0:
        return None, None, None, p
    y_arr = np.array([float(v) for v in y], dtype=float)
    # Flip sign when values are predominantly negative (BayesOpt minimizes -MAE)
    med = float(np.nanmedian(y_arr))
    if med < -1e-9:
        y_arr = -y_arr
    return params, y_arr, len(y), p


def best_so_far(y, higher_better):
    bs = np.full_like(y, np.nan)
    cur = float('inf') if (not higher_better) else float('-inf')
    for i in range(y.size):
        if np.isnan(y[i]):
            bs[i] = cur
            continue
        if higher_better:
            if y[i] > cur:
                cur = y[i]
        else:
            if y[i] < cur:
                cur = y[i]
        bs[i] = cur
    return bs


def make_xtick_labels(params_list):
    labels = []
    for p in params_list:
        try:
            r, n, s = p[0], p[1], p[2]
        except Exception:
            labels.append('(?,?,?)')
            continue
        labels.append(f'({fmt_rv(r)}, {fmt_int(n)}, {fmt_int(s)})')
    return labels


def plot_model(model):
    seed_data = {}
    for s in SEEDS:
        params, y, n, path = load_state(model, s)
        if y is None:
            print(f'  {model} seed{s}: MISSING {path}')
            continue
        seed_data[s] = dict(params=params, y=y, n=n, path=path)

    if len(seed_data) == 0:
        print(f'  skip {model}: no seed data')
        return None

    # X-tick labels come from a seed with the longest param list (seed 42 preferred).
    max_n = max(d['n'] for d in seed_data.values())
    xtick_seed = 42 if (42 in seed_data and seed_data[42]['n'] == max_n) else \
        next(s for s in SEEDS if s in seed_data and seed_data[s]['n'] == max_n)
    xtick_labels = make_xtick_labels(seed_data[xtick_seed]['params'])
    steps_all = np.arange(1, max_n + 1)

    best_maes = []
    best_ks = []
    bsf_curves = []
    for s, d in seed_data.items():
        n = d['n']
        steps = np.arange(1, n + 1)
        y = d['y']
        if HIGHER_BETTER:
            best_i = int(np.nanargmax(y))
        else:
            best_i = int(np.nanargmin(y))
        best_maes.append(float(y[best_i]))
        best_ks.append(int(best_i + 1))
        bs = best_so_far(y, HIGHER_BETTER)
        bsf_curves.append(bs)

    bmae = np.array(best_maes)
    bks = np.array(best_ks)

    fig, ax = plt.subplots(figsize=(max(13, int(0.7 * max_n) + 6), 7.5))

    for s in SEEDS:
        if s not in seed_data:
            continue
        d = seed_data[s]
        y = d['y']
        n = d['n']
        steps = np.arange(1, n + 1)
        sty = SEED_STYLES[s]

        bs = best_so_far(y, HIGHER_BETTER)
        if HIGHER_BETTER:
            best_i = int(np.nanargmax(y))
        else:
            best_i = int(np.nanargmin(y))
        best_y = float(y[best_i])

        ax.plot(steps, bs, color=sty['color'], lw=2.2, alpha=0.92,
                label=f'{sty["label"]} best-so-far')
        ax.scatter(steps, y, marker=sty['marker'], s=70, zorder=5,
                   facecolors='none', edgecolors=sty['color'], linewidths=1.8,
                   label=f'{sty["label"]} observed {METRIC}')
        ax.scatter([steps[best_i]], [best_y], s=280, facecolors='none',
                   edgecolors='gold', linewidths=3.0, zorder=7,
                   label=f'{sty["label"]} best (k={best_i+1}, {METRIC}={best_y:.4g})')

    # Explore/exploit separator at the end of initialization.
    sep_x = EXPLORE_STEPS + 0.5
    if sep_x < max_n + 0.1:
        ax.axvline(x=sep_x, color='black', linestyle='--', lw=1.7, zorder=3)

    ax.set_xlabel('Evaluation step k', fontsize=21)
    ax.set_ylabel(METRIC, fontsize=21)
    ax.set_xticks(steps_all)
    ax.set_xticklabels([str(int(k)) for k in steps_all], ha='center', fontsize=18)
    ax.set_xlim(0.5, max_n + 0.5)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(bottom=0.0)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_title(f'Optimization trajectory · 1st {model} / band gap  [Problem 1]',
                 fontsize=25, fontweight='bold', pad=24)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.92, ncol=1)

    fig.tight_layout()
    out_p = os.path.join(FIGURES_DIR, f'optim_1st_{model}_{TASK}_multiseed_trajectory.png')
    fig.savefig(out_p, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'--- {model} ---')
    for s in SEEDS:
        if s in seed_data:
            d = seed_data[s]
            y = d['y']
            n = d['n']
            if HIGHER_BETTER:
                by = float(np.nanmax(y))
                bk = int(np.nanargmax(y)) + 1
            else:
                by = float(np.nanmin(y))
                bk = int(np.nanargmin(y)) + 1
            print(f'  seed{s:>3d}  steps={n:<2d}  best_{METRIC}={by:.5g} @k={bk:<2d}')
    # Final best-so-far stats for the stdout summary only.
    aligned = np.full((len(bsf_curves), max_n), np.nan)
    for i, bs in enumerate(bsf_curves):
        n = bs.size
        aligned[i, :n] = bs
        if n < max_n and not np.isnan(bs[-1]):
            aligned[i, n:] = bs[-1]
    valid_col = ~np.all(np.isnan(aligned), axis=0)
    if valid_col.any():
        final_col = aligned[:, np.max(np.where(valid_col)[0])]
        final_mae_mean = float(np.nanmean(final_col))
        final_mae_std = float(np.nanstd(final_col, ddof=1)) if np.isfinite(final_col).sum() > 1 else 0.0
    else:
        final_mae_mean = float(np.nanmean(bmae))
        final_mae_std = float(np.nanstd(bmae, ddof=1)) if bmae.size > 1 else 0.0

    print(f'  stats:  best_{METRIC}_mean±σ = {bmae.mean():.4g} ± {bmae.std(ddof=1) if bmae.size > 1 else 0:.3g}   '
          f'range = {bmae.min():.4g} – {bmae.max():.4g}   '
          f'best_k=[{",".join(str(int(k)) for k in bks)}]   '
          f'final_best-so-far_mean±σ = {final_mae_mean:.4g} ± {final_mae_std:.3g}')
    return out_p


def main():
    print('=== 1st round band gap multi-seed (3 seeds: 0/42/200) ===')
    for m in MODELS:
        plot_model(m)


if __name__ == '__main__':
    main()
