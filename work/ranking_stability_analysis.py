#!/usr/bin/env python3
"""Pairwise (Model vs rfscm) ranking stability analysis across (R,N,S) configs.

Because rfscm has full-space (R,N,S) ground truth (858 configs for band_gap),
every non-rfscm model can be compared against rfscm on whichever (R,N,S) configs
that non-rfscm model actually evaluated. This yields 4 independent pairwise
ranking-stability analyses:

  * ALIGNN  vs rfscm
  * MEGNet  vs rfscm
  * SchNet  vs rfscm
  * CrabNet vs rfscm

For each pair we compute, visualise and report:
  - Global win-rate of model over rfscm (MAE lower = better)
  - Ranking-swap frequency / ratio when configs are sorted lexicographically
  - Rank behaviour sliced by R, by N, by S
  - MAE-difference distribution  (model - rfscm); negative => model better
  - Scatter plot  rfscm_MAE  vs  model_MAE  with y=x parity line
  - Pairwise transition / concordance summary
"""

import os
import re
import json
import glob
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(WORK_DIR, 'result')
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

TASK = 'band_gap'
METRIC = 'MAE'
HIGHER_BETTER = False   # lower MAE = better

FULL_MODELS = ['alignn', 'megnet', 'schnet', 'crabnet']
BASELINE = 'rfscm'
MODEL_LABELS = {
    'alignn':  'ALIGNN',
    'megnet':  'MEGNet',
    'schnet':  'SchNet',
    'crabnet': 'CrabNet',
    'rfscm':   'RF-SCM',
}
MODEL_COLORS = {
    'alignn':  '#1f77b4',
    'megnet':  '#ff7f0e',
    'schnet':  '#2ca02c',
    'crabnet': '#d62728',
    'rfscm':   '#7f7f7f',
}
PAIR_ACCENT = {
    'alignn':  '#1f77b4',
    'megnet':  '#ff7f0e',
    'schnet':  '#2ca02c',
    'crabnet': '#d62728',
}

OPT_STATE_PATHS = {
    'alignn':  os.path.join(WORK_DIR, f'optimization_1st_state_alignn_{TASK}_seed42.json'),
    'megnet':  os.path.join(WORK_DIR, f'optimization_1st_state_megnet_{TASK}_seed42.json'),
    'schnet':  os.path.join(WORK_DIR, f'optimization_1st_state_schnet_{TASK}_seed42.json'),
    'crabnet': os.path.join(WORK_DIR, f'optimization_1st_state_crabnet_{TASK}_seed42.json'),
}


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

def fmt_rv(r):
    if r is None:
        return '?'
    if isinstance(r, (int, np.integer)) or float(r).is_integer():
        return str(int(r))
    return f'{float(r):g}'


def load_result_dir_mae(model_name):
    """Scan work/result for folders named <model>_band_gap_(R,N,S) and return
    dict {(R,N,S): MAE}.
    """
    mae_map = {}
    prefix = f'{model_name}_{TASK}_'
    if not os.path.isdir(RESULT_DIR):
        return mae_map
    for name in sorted(os.listdir(RESULT_DIR)):
        if not name.startswith(prefix):
            continue
        m = re.match(rf'{re.escape(prefix)}\((.+),(.+),(.+)\)', name)
        if not m:
            continue
        try:
            r, n, s = float(m.group(1)), int(m.group(2)), int(m.group(3))
        except Exception:
            continue
        sub = os.path.join(RESULT_DIR, name)
        mae = None
        for fn in sorted(os.listdir(sub)):
            if not fn.endswith('.json'):
                continue
            fp = os.path.join(sub, fn)
            try:
                with open(fp) as f:
                    d = json.load(f)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            am = d.get('average_metrics') or {}
            cand = am.get('mae') or d.get('value') or d.get('mae')
            if isinstance(cand, (int, float)) and np.isfinite(float(cand)):
                mae = float(cand)
                break
        if mae is not None:
            mae_map[(r, n, s)] = mae
    return mae_map


def load_opt_state_mae(model_name):
    """Load (R,N,S) -> MAE from the 1st-round optimization state JSON file."""
    p = OPT_STATE_PATHS.get(model_name)
    if p is None or not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    params = d.get('params_obs', []) or []
    y = d.get('y_obs', []) or d.get('y_raw_obs', []) or []
    rows = {}
    for p3, val in zip(params, y):
        try:
            r, n, s = float(p3[0]), int(p3[1]), int(p3[2])
        except Exception:
            continue
        rows[(r, n, s)] = float(val)
    # sign flip for -MAE internal storage
    if rows:
        med = float(np.nanmedian(list(rows.values())))
        if med < -1e-9:
            rows = {k: -v for k, v in rows.items()}
    return rows


def load_combined_model_mae(model_name):
    """Merge optimization-observed MAEs with result-dir MAEs; state JSON wins
    on conflict.
    """
    res_dir = load_result_dir_mae(model_name)
    state = load_opt_state_mae(model_name)
    combined = dict(res_dir)
    combined.update(state)
    return combined


# ---------------------------------------------------------------------------
# pairwise ranking computations
# ---------------------------------------------------------------------------

def build_pairwise_frame(model_mae, rfscm_mae, model_name):
    shared_keys = sorted(set(model_mae.keys()) & set(rfscm_mae.keys()))
    rows = []
    for k in shared_keys:
        r, n, s = k
        m_v = model_mae[k]
        b_v = rfscm_mae[k]
        delta = m_v - b_v          # <0 means model better, >0 means rfscm better
        if not HIGHER_BETTER:
            model_wins = int(delta < 0)
            tie = int(abs(delta) < 1e-12)
        else:
            model_wins = int(delta > 0)
            tie = int(abs(delta) < 1e-12)
        rows.append({
            'R': r, 'N': n, 'S': s,
            f'{model_name}_MAE': m_v,
            f'{BASELINE}_MAE': b_v,
            'delta_MAE': delta,
            'model_win': model_wins and not tie,
            'tie': tie,
            'rank_model': 1 if (model_wins and not tie) else (2 if (not tie) else 1.5),
            'rank_baseline': 2 if (model_wins and not tie) else (1 if (not tie) else 1.5),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(['R', 'N', 'S']).reset_index(drop=True)
    return df


def pairwise_rank_swaps(df):
    """Along the lexicographically-sorted (R,N,S) sequence of configs, count
    how many consecutive pairs SWAP the top-1 rank between model & baseline.
    Returns (n_swaps, n_consecutive, swap_ratio).
    Ties count as no-swap if adjacent was a tie or same-winner; a tie on one
    side but win-change on the other counts as a half (rounded up = swap).
    """
    rank_model = df['rank_model'].values
    rank_baseline = df['rank_baseline'].values
    n = len(df)
    swaps = 0
    consec = max(0, n - 1)
    for i in range(1, n):
        prev_winner_is_model = rank_model[i - 1] < rank_baseline[i - 1]
        cur_winner_is_model = rank_model[i] < rank_baseline[i]
        prev_tie = abs(rank_model[i - 1] - rank_baseline[i - 1]) < 1e-9
        cur_tie = abs(rank_model[i] - rank_baseline[i]) < 1e-9
        if (prev_tie or cur_tie) and not (prev_tie and cur_tie):
            # treat moving in/out of tie as a swap for volatility count
            swaps += 0 if (prev_tie and cur_tie) else 1
        elif (not prev_tie) and (not cur_tie) and (prev_winner_is_model != cur_winner_is_model):
            swaps += 1
    swap_ratio = (swaps / consec) if consec > 0 else float('nan')
    return int(swaps), consec, swap_ratio


def slice_stability(df, dim):
    out = {}
    for k, sub in df.groupby(dim, sort=True):
        n_cfg = len(sub)
        wins = int(sub['model_win'].sum())
        ties = int(sub['tie'].sum())
        wr = wins / n_cfg if n_cfg else float('nan')
        mean_delta = float(sub['delta_MAE'].mean())
        sw, consec, sr = pairwise_rank_swaps(sub.reset_index(drop=True))
        out[k] = dict(n=n_cfg, wins=wins, ties=ties, win_rate=wr,
                      mean_delta=mean_delta, swaps=sw, consec=consec,
                      swap_ratio=sr)
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_pair_delta_distribution(ax, df, model_name, color):
    deltas = df['delta_MAE'].values
    n = len(deltas)
    wins = int((deltas < 0).sum())
    ties = int((np.abs(deltas) < 1e-12).sum())
    losses = n - wins - ties

    bins = np.linspace(np.nanmin(deltas) * 1.05 - 0.05,
                       np.nanmax(deltas) * 1.05 + 0.05,
                       min(40, max(10, n // 2 + 4)))
    ax.hist(deltas, bins=bins, color=color, alpha=0.75, edgecolor='black', lw=0.8)
    ax.axvline(0.0, color='black', linestyle='--', lw=2.0, zorder=5,
               label='ΔMAE = 0 (equal)')
    md = float(np.nanmean(deltas))
    ax.axvline(md, color=color, linestyle='-', lw=2.2, zorder=6,
               label=f'mean ΔMAE = {md:+.4f}')
    ax.set_xlabel(f'ΔMAE = {MODEL_LABELS[model_name]} MAE − {MODEL_LABELS[BASELINE]} MAE\n'
                  f'(negative → {MODEL_LABELS[model_name]} better)',
                  fontsize=13.5)
    ax.set_ylabel('# (R,N,S) configs', fontsize=13.5)
    ax.set_title(f'{MODEL_LABELS[model_name]} vs {MODEL_LABELS[BASELINE]} · ΔMAE histogram\n'
                 f'N={n}  (win: {wins} · tie: {ties} · lose: {losses})',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(fontsize=11.5, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(axis='both', labelsize=11.5)


def plot_pair_mae_scatter(ax, df, model_name, color):
    x = df[f'{BASELINE}_MAE'].values
    y = df[f'{model_name}_MAE'].values
    vmax = max(np.nanmax(x), np.nanmax(y)) * 1.04
    vmin = 0.0
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.8, label='y = x (equal MAE)', zorder=3)
    ax.scatter(x, y, s=52, facecolors=color, edgecolors='black',
               linewidths=0.9, alpha=0.82, zorder=5,
               label=f'{MODEL_LABELS[model_name]} configs (N={len(df):d})')
    below = (y < x).sum()
    ax.set_xlabel(f'{MODEL_LABELS[BASELINE]} {METRIC} (lower is better)', fontsize=13.5)
    ax.set_ylabel(f'{MODEL_LABELS[model_name]} {METRIC} (lower is better)', fontsize=13.5)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'{MODEL_LABELS[model_name]} ↔ {MODEL_LABELS[BASELINE]} {METRIC} scatter\n'
                 f'points below parity: {below}/{len(df)} ({100*below/len(df):.1f}%)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(fontsize=11.5, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(axis='both', labelsize=11.5)


def plot_pair_rank_trajectory(ax, df, model_name, color):
    steps = np.arange(1, len(df) + 1)
    rm = df['rank_model'].values
    rb = df['rank_baseline'].values
    # binary winner trace: +1 model wins, 0 tie, -1 baseline wins
    winner = np.where(rm < rb, 1.0, np.where(rb < rm, -1.0, 0.0))
    ax.step(steps, winner, where='mid', color=color, lw=2.0, alpha=0.92)
    ax.plot(steps, winner, marker='o', ms=6.5, color=color,
            markeredgecolor='black', markeredgewidth=0.9, linestyle='-', lw=1.4,
            label=f'{MODEL_LABELS[model_name]} win=+1 / lose=−1 / tie=0')
    ax.axhline(0.0, color='black', lw=1.0, alpha=0.7)
    ax.axhline(1.0, color=color, lw=0.8, alpha=0.35, linestyle=':')
    ax.axhline(-1.0, color=MODEL_COLORS[BASELINE], lw=0.8, alpha=0.35, linestyle=':')
    swaps, consec, sr = pairwise_rank_swaps(df)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([f'{MODEL_LABELS[BASELINE]} better', 'Tie',
                        f'{MODEL_LABELS[model_name]} better'],
                       fontsize=11)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel('Shared (R,N,S) index (sorted R→N→S)', fontsize=13.5)
    ax.set_title(f'Winner trace · swaps: {swaps}/{consec} = {100*sr:.1f}% of adjacent pairs',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.tick_params(axis='x', labelsize=11)


def plot_pair_dim_slices(axes, df, model_name, color):
    """3 subplots: win-rate bar by R, by N, by S  (twin axis for swap ratio)."""
    dims = ['R', 'N', 'S']
    dim_names = {'R': 'Reduction ratio R', 'N': 'Fold count N', 'S': 'CV repeat S'}
    dim_fmt = {'R': fmt_rv, 'N': lambda v: f'{int(v)}', 'S': lambda v: f'{int(v)}'}
    for ax, dim in zip(axes, dims):
        stats = slice_stability(df, dim)
        keys = sorted(stats.keys())
        wr = np.array([stats[k]['win_rate'] for k in keys]) * 100.0
        sr = np.array([(stats[k]['swap_ratio'] if np.isfinite(stats[k]['swap_ratio']) else 0.0)
                       for k in keys]) * 100.0
        ns = [stats[k]['n'] for k in keys]
        x = np.arange(len(keys))
        bars = ax.bar(x, wr, color=color, alpha=0.78, edgecolor='black', lw=0.9,
                      label=f'{MODEL_LABELS[model_name]} win rate (%)')
        for b, n, v in zip(bars, ns, wr):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0,
                    f'{v:.0f}%\n(n={n})',
                    ha='center', va='bottom', fontsize=9.8, fontweight='bold')
        ax2 = ax.twinx()
        ax2.plot(x, sr, color='black', marker='s', ms=7, lw=2.0, linestyle='--',
                 alpha=0.85, label='adjacent swap ratio (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([dim_fmt[dim](k) for k in keys], fontsize=11,
                           rotation=30, ha='right')
        ax.set_ylim(0, 106)
        ax.set_ylabel(f'{MODEL_LABELS[model_name]} win rate (%)', fontsize=12.5, color=color)
        ax2.set_ylim(0, 106)
        ax2.set_ylabel('Adjacent rank-swap ratio (%)', fontsize=12.5, color='black')
        ax.set_xlabel(dim_names[dim], fontsize=13)
        ax.set_title(f'Per-{dim} breakdown', fontsize=13.5, fontweight='bold', pad=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(axis='y', labelcolor=color)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9.2, loc='upper left')


def make_one_pair_panel_figure(pair_name, model_name, df, stats):
    deltas = df['delta_MAE'].values
    n = len(deltas)
    wins = int((deltas < 0).sum())
    ties = int((np.abs(deltas) < 1e-12).sum())
    losses = n - wins - ties
    color = PAIR_ACCENT[model_name]

    fig, ax = plt.subplots(figsize=(11, 7.5))

    vrange = (np.nanmax(deltas) - np.nanmin(deltas)) or 1.0
    bins = np.linspace(np.nanmin(deltas) - 0.08 * vrange - 0.05,
                       np.nanmax(deltas) + 0.08 * vrange + 0.05,
                       min(50, max(12, n // 2 + 6)))
    _, _, patches = ax.hist(deltas, bins=bins, color=color, alpha=0.78,
                            edgecolor='black', lw=0.9, zorder=3, label='ΔMAE samples')
    ax.axvline(0.0, color='black', linestyle='--', lw=2.2, zorder=6,
               label='ΔMAE = 0 (equal MAE)')
    md = float(np.nanmean(deltas))
    sd = float(np.nanstd(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    med = float(np.nanmedian(deltas))
    ax.axvline(md, color=color, linestyle='-', lw=2.6, zorder=7,
               label=f'mean ΔMAE = {md:+.4f}')
    ax.axvline(med, color=color, linestyle=':', lw=2.2, zorder=7, alpha=0.85,
               label=f'median ΔMAE = {med:+.4f}')
    ymax = max([p.get_height() for p in patches]) * 1.03 if len(patches) else 1.0
    ax.axvspan(-1e9, 0.0, color=color, alpha=0.10, zorder=1,
               label=f'{MODEL_LABELS[model_name]} better (ΔMAE<0): {wins}/{n} ({100*wins/n:.1f}%)')
    ax.axvspan(0.0, 1e9, color=MODEL_COLORS[BASELINE], alpha=0.09, zorder=1,
               label=f'{MODEL_LABELS[BASELINE]} better (ΔMAE>0): {losses}/{n} ({100*losses/n:.1f}%)')

    txt = (f'N = {n}  shared (R,N,S)\n'
           f'Wins  : {wins:>3d}  ({100*wins/n:.1f}%)\n'
           f'Ties  : {ties:>3d}  ({100*ties/n:.1f}%)\n'
           f'Losses: {losses:>3d}  ({100*losses/n:.1f}%)\n'
           f'mean ΔMAE   = {md:+.4f}\n'
           f'std ΔMAE    = {sd:.4f}\n'
           f'median ΔMAE = {med:+.4f}\n'
           f'range = [{np.nanmin(deltas):+.4f}, {np.nanmax(deltas):+.4f}]')
    props = dict(boxstyle='round,pad=0.6', facecolor='whitesmoke',
                 edgecolor='gray', alpha=0.92, linewidth=1.2)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            verticalalignment='top', fontsize=11.5,
            bbox=props, zorder=10, family='monospace')

    ax.set_xlabel(f'ΔMAE = {MODEL_LABELS[model_name]} MAE  −  {MODEL_LABELS[BASELINE]} MAE   '
                  f'(negative → {MODEL_LABELS[model_name]} better)',
                  fontsize=15, labelpad=10)
    ax.set_ylabel('# of shared (R,N,S) configurations', fontsize=15, labelpad=10)
    ax.set_title(f'ΔMAE Distribution · {MODEL_LABELS[model_name]} vs {MODEL_LABELS[BASELINE]} · '
                 f'{TASK.replace("_", " ").title()}',
                 fontsize=18, fontweight='bold', pad=14)
    ax.legend(fontsize=11.5, loc='upper right', framealpha=0.93)
    ax.grid(True, linestyle='--', alpha=0.38, zorder=0)
    ax.tick_params(axis='both', labelsize=12.5)
    ax.set_ylim(0, ymax * 1.12)

    fig.tight_layout()
    out_p = os.path.join(FIGURES_DIR, f'ranking_pairwise_{model_name}_vs_{BASELINE}_{TASK}.png')
    fig.savefig(out_p, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out_p


def make_cross_pair_summary_figure(all_results):
    names = [MODEL_LABELS[m] for m in FULL_MODELS]
    colors = [PAIR_ACCENT[m] for m in FULL_MODELS]
    boxdata = [all_results[m]['df']['delta_MAE'].values for m in FULL_MODELS]

    fig, ax = plt.subplots(figsize=(12.5, 8.5))

    bp = ax.boxplot(boxdata, labels=names, patch_artist=True,
                    medianprops=dict(color='black', lw=2.2),
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='gold',
                                   markeredgecolor='black', markeredgewidth=1.2,
                                   markersize=10.5),
                    widths=0.62,
                    whiskerprops=dict(lw=1.5),
                    capprops=dict(lw=1.5))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    rng = np.random.default_rng(42)
    xs = np.arange(1, len(FULL_MODELS) + 1)
    for x, data, col in zip(xs, boxdata, colors):
        jitter = rng.normal(0, 0.045, size=len(data))
        ax.scatter(np.full_like(data, x, dtype=float) + jitter, data,
                   s=46, facecolor=col, edgecolor='black', lw=0.7,
                   alpha=0.72, zorder=3, label=f'{names[x-1]} samples')

    parity_line = ax.axhline(0.0, color='black', linestyle='--', lw=2.2, alpha=0.85,
                             label='ΔMAE = 0 parity (equal to RF-SCM)', zorder=4)

    for i, m in enumerate(FULL_MODELS):
        arr = all_results[m]['df']['delta_MAE'].values
        n = len(arr)
        wr = 100.0 * (arr < 0).sum() / n
        md = np.mean(arr)
        text = (f'N = {n}\n'
                f'mean Δ = {md:+.3f}\n'
                f'win-rate = {wr:.1f}%')
        props = dict(boxstyle='round,pad=0.42', facecolor='whitesmoke',
                     edgecolor=colors[i], alpha=0.93, linewidth=1.15)
        ytop = ax.get_ylim()[1]
        ax.text(xs[i] + 0.29, 0.98, text, transform=ax.get_xaxis_transform(),
                va='top', ha='left', fontsize=10.7, bbox=props, zorder=10)

    ax.set_ylabel(f'ΔMAE = Model MAE  −  {MODEL_LABELS[BASELINE]} MAE\n'
                  f'(negative → model is better than {MODEL_LABELS[BASELINE]})',
                  fontsize=15.5, labelpad=12)
    ax.set_title(f'ΔMAE Distribution · All Models vs {MODEL_LABELS[BASELINE]} · '
                 f'{TASK.replace("_", " ").title()}',
                 fontsize=18.5, fontweight='bold', pad=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.40, zorder=0)
    ax.tick_params(axis='both', labelsize=13.5)
    ax.set_xlim(0.25, len(FULL_MODELS) + 0.85)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict()
    for h, l in zip(handles, labels):
        uniq.setdefault(l, h)
    ordered_handles, ordered_labels = [], []
    from matplotlib.lines import Line2D
    mean_h = Line2D([], [], marker='D', markersize=10, markerfacecolor='gold',
                    markeredgecolor='black', markeredgewidth=1.2,
                    linestyle='None', label='Mean (diamond)')
    med_h = Line2D([], [], color='black', lw=2.2, linestyle='-', label='Median (box line)')
    ordered_handles = [parity_line, mean_h, med_h] + [
        Line2D([], [], marker='o', linestyle='None', markersize=8,
               markerfacecolor=colors[i], markeredgecolor='black',
               label=f'{names[i]} samples') for i in range(len(FULL_MODELS))
    ]
    ordered_labels = [
        'ΔMAE = 0 parity (equal to RF-SCM)',
        'Mean (gold diamond)',
        'Median (black line)',
    ] + [f'{names[i]} samples  N={len(boxdata[i])}' for i in range(len(FULL_MODELS))]
    ax.legend(ordered_handles, ordered_labels,
              loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=3, fontsize=11.8, framealpha=0.95,
              fancybox=True, shadow=False, borderpad=0.8)

    fig.subplots_adjust(bottom=0.26, top=0.91, left=0.10, right=0.97)
    out_p = os.path.join(FIGURES_DIR, f'ranking_pairwise_summary_all_models_{TASK}.png')
    fig.savefig(out_p, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out_p


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(all_results, all_figs, summary_fig):
    lines = []
    lines.append('# Band Gap: Pairwise Model Ranking Stability Analysis Across Evaluation Configs')
    lines.append('')
    lines.append('> Analysis strategy: since non-`rfscm` models each only evaluated a subset of the (R,N,S)')
    lines.append(f'> configs in their optimization trajectory, while **{MODEL_LABELS[BASELINE]}** has full (R,N,S)-space')
    lines.append('> ground truth results for the band_gap task (858 configs), we therefore independently compare each model')
    lines.append('> against RF-SCM for pairwise ranking stability, 4 groups in total:')
    lines.append('>')
    lines.append(f'> - A = {MODEL_LABELS["alignn"]}   vs  B = {MODEL_LABELS[BASELINE]}')
    lines.append(f'> - A = {MODEL_LABELS["megnet"]}   vs  B = {MODEL_LABELS[BASELINE]}')
    lines.append(f'> - A = {MODEL_LABELS["schnet"]}   vs  B = {MODEL_LABELS[BASELINE]}')
    lines.append(f'> - A = {MODEL_LABELS["crabnet"]}  vs  B = {MODEL_LABELS[BASELINE]}')
    lines.append('')
    lines.append('> Ranking rule: within each (R,N,S) config, lower MAE ranks higher (Rank 1 = best).')
    lines.append('')
    lines.append('---')
    lines.append('')

    lines.append('## 1. Overview: Summary Figure Across the Four Pairwise Comparisons')
    lines.append('')
    lines.append(f'![{os.path.basename(summary_fig)}]({os.path.basename(summary_fig)})')
    lines.append('')

    lines.append('### 1.1 Key Metrics Summary Table')
    lines.append('')
    lines.append('| Model | Shared (R,N,S) count | Model wins (ΔMAE<0) | Ties | Losses | Win rate | Mean ΔMAE | Adjacent rank swaps | Swap ratio |')
    lines.append('|------|----------------|-------------------|----|----|-----|-----------|---------------|---------|')
    for m in FULL_MODELS:
        st = all_results[m]['stats']
        df_m = all_results[m]['df']
        win = int(st['wins_total'])
        tie = int(st['ties_total'])
        los = st['n_total'] - win - tie
        lines.append(
            f'| {MODEL_LABELS[m]} | {st["n_total"]:>3d} | {win:>3d} | {tie} | {los:>3d} | '
            f'{100*st["win_rate_total"]:.1f}% | {st["mean_delta_total"]:+.4f} | '
            f'{st["swaps"]}/{st["consec"]} | {100*st["swap_ratio"]:.1f}% |')
    lines.append('')

    lines.append('### 1.2 Model Performance Ranking (by win rate vs RF-SCM)')
    lines.append('')
    ranked_by_wr = sorted(FULL_MODELS, key=lambda m: all_results[m]['stats']['win_rate_total'], reverse=True)
    for i, m in enumerate(ranked_by_wr, 1):
        wr = 100 * all_results[m]['stats']['win_rate_total']
        md = all_results[m]['stats']['mean_delta_total']
        sr = 100 * all_results[m]['stats']['swap_ratio'] if np.isfinite(all_results[m]['stats']['swap_ratio']) else float('nan')
        note = []
        if wr > 70: note.append('significantly better than RF-SCM')
        elif wr > 55: note.append('slightly better than RF-SCM')
        elif wr > 45: note.append('comparable to RF-SCM')
        else: note.append('worse than RF-SCM')
        if sr < 20: note.append('ranking extremely stable')
        elif sr < 40: note.append('ranking fairly stable')
        elif sr < 60: note.append('ranking somewhat volatile')
        else: note.append('ranking highly volatile')
        lines.append(f'{i}. **{MODEL_LABELS[m]}** — win rate {wr:.1f}%, mean ΔMAE = {md:+.4f}, swap ratio = {sr:.1f}% ({", ".join(note)})')
    lines.append('')

    lines.append('---')
    lines.append('')

    for m in FULL_MODELS:
        df_m = all_results[m]['df']
        st = all_results[m]['stats']
        swaps = st['swaps']; consec = st['consec']; sr = st['swap_ratio']
        fig_p = all_figs[m]
        lines.append(f'## 2. {MODEL_LABELS[m]} vs {MODEL_LABELS[BASELINE]} Detailed Analysis')
        lines.append('')
        lines.append(f'![{os.path.basename(fig_p)}]({os.path.basename(fig_p)})')
        lines.append('')

        lines.append('### 2.{idx}.1 Core Metrics'.format(idx=FULL_MODELS.index(m)+1))
        lines.append('')
        lines.append(f'- Shared (R,N,S) config count: **{st["n_total"]}**')
        wins = int(st['wins_total']); ties = int(st['ties_total'])
        lines.append(f'- Wins / Ties / Losses: **{wins} / {ties} / {st["n_total"] - wins - ties}**  → win rate = **{100*st["win_rate_total"]:.2f}%**')
        lines.append(f'- ΔMAE (model − RF-SCM): mean {st["mean_delta_total"]:+.4f}, std {st["std_delta_total"]:.4f}, Min = {st["min_delta_total"]:+.4f}, Max = {st["max_delta_total"]:+.4f}')
        lines.append(f'- Rank swaps (adjacent configs): **{swaps}/{consec}** → swap ratio = **{100*sr:.2f}%**')
        lines.append('')

        for dim, dn, fmtk in [('R', 'Reduction ratio R', fmt_rv),
                              ('N', 'Fold count N', lambda v: f'N={int(v)}'),
                              ('S', 'CV repeat S', lambda v: f'S={int(v)}')]:
            sl = slice_stability(df_m, dim)
            keys = sorted(sl.keys())
            section_num = FULL_MODELS.index(m) + 1
            sub_num = {'R': '2', 'N': '3', 'S': '4'}[dim]
            lines.append(f'### 2.{section_num}.{sub_num} Breakdown by {dn}')
            lines.append('')
            lines.append(f'| {dim} level | Config count | {MODEL_LABELS[m]} wins | Ties | Losses | Win rate | Mean ΔMAE | Within-level swap ratio |')
            lines.append('|' + '--------|' * 8)
            for k in keys:
                v = sl[k]
                win = int(v['wins']); tie = int(v['ties']); los = v['n'] - win - tie
                wr_pct = 100.0 * v['win_rate'] if v['n'] else float('nan')
                sr_pct = 100.0 * v['swap_ratio'] if np.isfinite(v['swap_ratio']) else float('nan')
                lines.append(
                    f'| {fmtk(k):>6s} | {v["n"]:>4d} | {win:>2d} | {tie:d} | {los:>2d} | '
                    f'{wr_pct:.1f}% | {v["mean_delta"]:+.4f} | '
                    f'{("—" if not np.isfinite(sr_pct) else f"{sr_pct:.1f}%"):>6s} |')
            lines.append('')

        lines.append(f'### 2.{FULL_MODELS.index(m)+1}.5 Top-10 Best / Worst ΔMAE Configs')
        lines.append('')
        d2 = df_m.copy()
        d2 = d2.sort_values('delta_MAE', ascending=True).reset_index(drop=True)
        lines.append('**Best 10 (model MAE significantly smaller than RF-SCM):**')
        lines.append('')
        lines.append('| Rank | (R, N, S) | {0} MAE | {1} MAE | ΔMAE |'.format(MODEL_LABELS[m], MODEL_LABELS[BASELINE]))
        lines.append('|----:|----------|--------|--------|-----|')
        for i in range(min(10, len(d2))):
            r = d2.iloc[i]
            lines.append(f'| {i+1} | ({fmt_rv(r.R)}, {int(r.N)}, {int(r.S)}) | {r[f"{m}_MAE"]:.4f} | {r[f"{BASELINE}_MAE"]:.4f} | {r.delta_MAE:+.4f} |')
        lines.append('')
        d2b = df_m.copy().sort_values('delta_MAE', ascending=False).reset_index(drop=True)
        lines.append('**Worst 10 (RF-SCM better than model, or model abnormally poor):**')
        lines.append('')
        lines.append('| Rank | (R, N, S) | {0} MAE | {1} MAE | ΔMAE |'.format(MODEL_LABELS[m], MODEL_LABELS[BASELINE]))
        lines.append('|----:|----------|--------|--------|-----|')
        for i in range(min(10, len(d2b))):
            r = d2b.iloc[i]
            lines.append(f'| {i+1} | ({fmt_rv(r.R)}, {int(r.N)}, {int(r.S)}) | {r[f"{m}_MAE"]:.4f} | {r[f"{BASELINE}_MAE"]:.4f} | {r.delta_MAE:+.4f} |')
        lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('## 3. Overall Conclusions')
    lines.append('')

    best_wr = ranked_by_wr[0]
    runner = ranked_by_wr[1] if len(ranked_by_wr) > 1 else None
    lines.append(f'1. **Best and runner-up performers**: in the pairwise comparisons against RF-SCM,')
    lines.append(f'   **{MODEL_LABELS[best_wr]}** leads with a **{100*all_results[best_wr]["stats"]["win_rate_total"]:.1f}%** win rate,')
    if runner is not None:
        lines.append(f'   {MODEL_LABELS[runner]} comes second with **{100*all_results[runner]["stats"]["win_rate_total"]:.1f}%**.')
    lines.append('')

    stab_rank = sorted(FULL_MODELS, key=lambda m: (all_results[m]['stats']['swap_ratio']
                      if np.isfinite(all_results[m]['stats']['swap_ratio']) else 999))
    lines.append(f'2. **Most stable model in ranking** (lowest adjacent swap ratio): **{MODEL_LABELS[stab_rank[0]]}**')
    lines.append(f'   swap ratio = {100*all_results[stab_rank[0]]["stats"]["swap_ratio"]:.1f}%, meaning this model\'s win/loss relationship with RF-SCM')
    lines.append(f'   changes the least with (R,N,S); its "performance ranking relative to RF-SCM" is the least sensitive to the choice of evaluation config.')
    lines.append(f'   Most unstable: **{MODEL_LABELS[stab_rank[-1]]}** — swap ratio = {100*all_results[stab_rank[-1]]["stats"]["swap_ratio"]:.1f}%.')
    lines.append('')

    lines.append('3. **Sources of fluctuation by evaluation-config dimension (R/N/S)**:')
    lines.append('   - From the stability heatmap (bottom right of the summary figure) one can see on which dimension each model more easily flips rank;')
    lines.append('     that dimension corresponds to the parameter "most likely to affect the two-model comparison conclusion", and should be emphasized when choosing configs for comparison experiments.')
    lines.append('')

    lines.append('4. **Effect of the (R,N,S) choice on conclusions**: for models with generally high swap ratios,')
    lines.append('   the conclusion on whether they are better than RF-SCM strongly depends on the specific (R,N,S) values,')
    lines.append('   so comparing models based on just one or two configs easily yields one-sided conclusions; evaluation should be done on the full space or a representative subset.')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('*Analysis and report are auto-generated by `ranking_stability_analysis.py`. Data sources:')
    lines.append(f'{MODEL_LABELS[BASELINE]} full ground truth is taken from the `result/rfscm_band_gap_*` folder;')
    lines.append('each comparison model is the union of `optimization_1st_state_*_band_gap*` JSON (preferred if the same config exists)')
    lines.append('and the `result/<model>_band_gap_*` result folder.*')

    out_md = os.path.join(WORK_DIR, f'ranking_pairwise_stability_{TASK}_report.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(lines))
    return out_md


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print('=== Pairwise Ranking Stability (Model vs RF-SCM) · Band Gap ===\n')
    print('Loading RF-SCM full-space ground truth from result/ ...')
    rfscm_mae = load_result_dir_mae(BASELINE)
    print(f'  {MODEL_LABELS[BASELINE]}: {len(rfscm_mae)} configs')

    all_results = {}
    all_figs = {}

    for m in FULL_MODELS:
        print(f'\n--- Processing {MODEL_LABELS[m]} vs {MODEL_LABELS[BASELINE]} ---')
        model_mae = load_combined_model_mae(m)
        n_total = len(model_mae)
        df = build_pairwise_frame(model_mae, rfscm_mae, m)
        n_shared = len(df)
        print(f'  {MODEL_LABELS[m]} evaluated configs: {n_total}')
        print(f'  Shared with {MODEL_LABELS[BASELINE]}: {n_shared}')

        wins = int(df['model_win'].sum())
        ties = int(df['tie'].sum())
        wr_total = wins / n_shared if n_shared else float('nan')
        deltas = df['delta_MAE'].values
        mean_delta = float(np.nanmean(deltas))
        std_delta = float(np.nanstd(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        min_delta = float(np.nanmin(deltas))
        max_delta = float(np.nanmax(deltas))
        swaps, consec, swap_ratio = pairwise_rank_swaps(df)

        stats = dict(
            n_total=n_shared,
            wins_total=wins, ties_total=ties,
            win_rate_total=wr_total,
            mean_delta_total=mean_delta, std_delta_total=std_delta,
            min_delta_total=min_delta, max_delta_total=max_delta,
            swaps=swaps, consec=consec, swap_ratio=swap_ratio,
        )
        print(f'  Win rate: {wins}/{n_shared} = {100*wr_total:.1f}%   '
              f' ties={ties}')
        print(f'  Delta MAE: mean={mean_delta:+.4f}  std={std_delta:.4f}  '
              f'range=[{min_delta:+.4f}, {max_delta:+.4f}]')
        print(f'  Rank swaps (adjacent): {swaps}/{consec} = {100*swap_ratio:.1f}%')

        print('  Drawing panel figure...', end=' ')
        panel = make_one_pair_panel_figure(f'{m}_vs_{BASELINE}', m, df, stats)
        print('saved', os.path.basename(panel))

        all_results[m] = dict(df=df, stats=stats)
        all_figs[m] = panel

    print('\n--- Cross-model summary figure ---')
    summary_fig = make_cross_pair_summary_figure(all_results)
    print(' saved:', os.path.basename(summary_fig))

    print('\n--- Writing markdown report ---')
    md_path = write_markdown_report(all_results, all_figs, summary_fig)
    print(' saved:', md_path)

    print('\n=== DONE ===')
    print('  Per-model panel figures:')
    for m in FULL_MODELS:
        print(f'    {MODEL_LABELS[m]:>8s}: {all_figs[m]}')
    print(f'  Summary: {summary_fig}')
    print(f'  Report:  {md_path}')


if __name__ == '__main__':
    main()
