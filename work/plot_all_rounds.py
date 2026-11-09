import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK_DIR = str(Path(__file__).resolve().parent)
FIGURES_DIR = os.path.join(WORK_DIR, 'figures')
RESULT_ROOT = os.path.join(WORK_DIR, 'result')
EXPLORE_STEPS = 4
BAND_GAP_FULL_MEAN = 0.87865
BULK_SHEAR_CAP = 50.0
CLASSIFICATION_TASKS = {'metallicity'}
SKIP_REF_FOR = {'bulk_modulus', 'shear_modulus'}
SKIP_TASKS = set()

GT_JSON = lambda t: os.path.join(WORK_DIR, 'GT', f'GT_rfscm_{t}.json')

FIRST_DUAL_JSON = {
    'band_gap':         os.path.join(WORK_DIR, 'optimization_1st_dual_state_crabnet_rfscm_band_gap_seed42.json'),
    'bulk_modulus':     os.path.join(WORK_DIR, 'optimization_1st_dual_state_crabnet_rfscm_bulk_modulus_seed42.json'),
    'formation_energy': os.path.join(WORK_DIR, 'optimization_1st_dual_state_crabnet_rfscm_formation_energy_seed42.json'),
    'refractive_index': os.path.join(WORK_DIR, 'optimization_1st_dual_state_crabnet_rfscm_refractive_index_seed42.json'),
    'shear_modulus':    os.path.join(WORK_DIR, 'optimization_1st_dual_state_crabnet_rfscm_shear_modulus_seed42.json'),
    'metallicity':      None,
}

# Problem 1: 1st round single-model optimization (raw MAE / ROC AUC).
JSON_DIR = os.path.join(WORK_DIR, 'json')
FIRST_ROUND_SINGLE = {
    'rfscm': {
        'band_gap':         os.path.join(WORK_DIR, 'optimization_1st_state_rfscm_band_gap.json'),
        'bulk_modulus':     os.path.join(WORK_DIR, 'optimization_1st_state_rfscm_bulk_modulus.json'),
        'formation_energy': os.path.join(WORK_DIR, 'optimization_1st_state_rfscm_formation_energy.json'),
        'shear_modulus':    os.path.join(WORK_DIR, 'optimization_1st_state_rfscm_shear_modulus.json'),
        'metallicity':      os.path.join(WORK_DIR, 'optimization_1st_state_rfscm_metallicity_seed42.json'),
        'refractive_index': os.path.join(WORK_DIR, 'optimization_1st_state_rfscm_refractive_index.json'),
    },
    'crabnet': {
        'band_gap':         os.path.join(WORK_DIR, 'optimization_1st_state_crabnet_band_gap_seed42.json'),
        'bulk_modulus':     None,
        'formation_energy': None,
        'refractive_index': None,
        'shear_modulus':    None,
        'metallicity':      None,
    },
    'megnet': {
        'band_gap':         os.path.join(WORK_DIR, 'optimization_1st_state_megnet_band_gap_seed42.json'),
        'bulk_modulus':     None,
        'formation_energy': None,
        'refractive_index': None,
        'shear_modulus':    None,
        'metallicity':      None,
    },
    'schnet': {
        'band_gap':         os.path.join(WORK_DIR, 'optimization_1st_state_schnet_band_gap_seed42.json'),
        'bulk_modulus':     None,
        'formation_energy': None,
        'refractive_index': None,
        'shear_modulus':    None,
        'metallicity':      None,
    },
    'alignn': {
        'band_gap':         os.path.join(WORK_DIR, 'optimization_1st_state_alignn_band_gap_seed42.json'),
        'bulk_modulus':     None,
        'formation_energy': None,
        'refractive_index': None,
        'shear_modulus':    None,
        'metallicity':      None,
    },
}

SECOND_ROUND_SEED42_RFSCM = {
    'band_gap':         os.path.join(WORK_DIR, 'optimization_2nd_rfscm_band_gap_seed42_state.json'),
    'bulk_modulus':     os.path.join(WORK_DIR, 'optimization_2nd_rfscm_bulk_modulus_seed42_state.json'),
    'formation_energy': os.path.join(WORK_DIR, 'optimization_2nd_rfscm_formation_energy_seed42_state.json'),
    'refractive_index': os.path.join(WORK_DIR, 'optimization_2nd_rfscm_refractive_index_seed42_state.json'),
    'shear_modulus':    os.path.join(WORK_DIR, 'optimization_2nd_rfscm_shear_modulus_seed42_state.json'),
    'metallicity':      os.path.join(WORK_DIR, 'optimization_2nd_state_rfscm_metallicity_seed42.json'),
}
SECOND_ROUND_SEED42_CRABNET = {
    'band_gap':         os.path.join(WORK_DIR, 'optimization_2nd_crabnet_band_gap_seed42_state.json'),
    'bulk_modulus':     os.path.join(WORK_DIR, 'optimization_2nd_crabnet_bulk_modulus_seed42_state.json'),
    'formation_energy': os.path.join(WORK_DIR, 'optimization_2nd_crabnet_formation_energy_seed42_state.json'),
    'refractive_index': os.path.join(WORK_DIR, 'optimization_2nd_crabnet_refractive_index_seed42_state.json'),
    'shear_modulus':    os.path.join(WORK_DIR, 'optimization_2nd_crabnet_shear_modulus_seed42_state.json'),
    'metallicity':      None,
}

STATES = [
    # (label, model_display_name, path_getter(task), extra_refs(task), draw_refs_bool, companions_list,
    #  higher_better_override, problem_label)
    #  None means fall back to TASKS's default higher_better
    # Problem 1: 1st round single-model raw MAE / ROC AUC optimization.
    ('1st', 'rfscm',
     lambda t: FIRST_ROUND_SINGLE['rfscm'].get(t),
     lambda t: [],
     False, [],
     None,
     'Problem 1'),
    ('1st', 'crabnet',
     lambda t: FIRST_ROUND_SINGLE['crabnet'].get(t),
     lambda t: [],
     False, [],
     None,
     'Problem 1'),
    ('1st', 'megnet',
     lambda t: FIRST_ROUND_SINGLE['megnet'].get(t),
     lambda t: [],
     False, [],
     None,
     'Problem 1'),
    ('1st', 'schnet',
     lambda t: FIRST_ROUND_SINGLE['schnet'].get(t),
     lambda t: [],
     False, [],
     None,
     'Problem 1'),
    ('1st', 'alignn',
     lambda t: FIRST_ROUND_SINGLE['alignn'].get(t),
     lambda t: [],
     False, [],
     None,
     'Problem 1'),
    # Problem 3: 2nd round, absolute deviation from the GT reference.
    ('2nd', 'rfscm',
     lambda t: SECOND_ROUND_SEED42_RFSCM.get(t),
     lambda t: [],
     False, [],
     None,
     'Problem 3'),
    ('2nd', 'crabnet',
     lambda t: SECOND_ROUND_SEED42_CRABNET.get(t),
     lambda t: [],
     False, [],
     None,
     'Problem 3'),
    # Problem 2: 1st round two-model combination, absolute MAE gap.
    ('1st_dual', 'crabnet_rfscm',
     lambda t: FIRST_DUAL_JSON.get(t),
     lambda t: [],
     False,
     ['rfscm', 'crabnet'],
     True,
     'Problem 2'),
]

TASKS = [
    # name, metric_label, higher_better
    ('band_gap',         'MAE',     False),
    ('bulk_modulus',     'MAE',     False),
    ('formation_energy', 'MAE',     False),
    ('metallicity',      'ROC AUC', True),
    ('refractive_index', 'MAE',     False),
    ('shear_modulus',    'MAE',     False),
]


def fmt_rv(rv):
    try:
        r = round(float(rv), 1)
        if abs(r - round(r)) < 1e-9:
            return str(int(round(r)))
        return f'{r:.1f}'
    except Exception:
        return str(rv)


def fmt_int(v):
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)


def read_ref_from_gt(gt_path, task_name):
    gt = json.load(open(gt_path))
    am = gt.get('average_metrics') or {}
    p = gt.get('params') or {}
    ref_R = p.get('R', 0.0)
    ref_N = p.get('N', 48)
    ref_S = p.get('S', 10)
    if task_name in CLASSIFICATION_TASKS:
        mae = am.get('mae')
        if mae is None:
            return (None, None, None, None)
        return (1.0 - float(mae), ref_R, ref_N, ref_S)
    mae = am.get('mae')
    if mae is None:
        return (None, None, None, None)
    return (float(mae), ref_R, ref_N, ref_S)


def fmt_r_variants(R):
    try:
        R = float(R); r1 = round(R, 1)
        out = []
        if abs(r1 - round(r1)) < 1e-9:
            out.append(str(int(round(r1))))
            out.append(f'{r1:.1f}')
        else:
            out.append(f'{r1:.1f}')
        seen = set()
        return [v for v in out if not (v in seen or seen.add(v))]
    except Exception:
        return [str(R)]


def lookup_result(task, model, R, N, S):
    N = int(float(N)); S = int(float(S))
    for rs in fmt_r_variants(R):
        for prefix in [f'{model}_{task}_({rs},{N},{S})']:
            fp = os.path.join(RESULT_ROOT, prefix, 'results.json')
            if os.path.exists(fp):
                try:
                    dd = json.load(open(fp))
                    am = dd.get('average_metrics') or {}
                    mae = am.get('mae')
                    if mae is None:
                        continue
                    if task in CLASSIFICATION_TASKS:
                        return 1.0 - float(mae)
                    return float(mae)
                except Exception:
                    continue
    return None


def draw_one(task_name, metric_lbl, higher_better_default,
             state_label, model_display_name, state_json_path, gt_json_path,
             extra_refs, draw_refs, companions, higher_better_override, problem_label):
    if task_name in SKIP_TASKS:
        print(f'  skip task: {task_name}')
        return None
    if state_json_path is None:
        print(f'  skip: state JSON not provided for {task_name}')
        return None
    if not os.path.exists(state_json_path):
        print(f'  skip: missing JSON {state_json_path}')
        return None

    # For 2nd, read the GT reference to anchor the Y-axis center even when draw_refs is False.
    need_ref_val = draw_refs or (state_label == '2nd' and task_name in
                                 ('band_gap', 'formation_energy', 'refractive_index'))
    ref_val = ref_R = ref_N = ref_S = None
    if need_ref_val and os.path.exists(gt_json_path):
        ref_val, ref_R, ref_N, ref_S = read_ref_from_gt(gt_json_path, task_name)

    st = json.load(open(state_json_path))
    params_obs = st.get('params_obs') or []
    y = st.get('y_obs') or st.get('y_raw_obs') or []
    if len(y) == 0 or len(params_obs) == 0:
        print(f'  skip: empty y_obs / params_obs')
        return None
    y_obs = np.array([float(v) for v in y], dtype=float)
    is_cls = task_name in CLASSIFICATION_TASKS

    higher_better = higher_better_default if higher_better_override is None else bool(higher_better_override)

    left_ylabel = metric_lbl
    best_metric_word = metric_lbl
    if problem_label == 'Problem 2':
        left_ylabel = 'Absolute MAE gap'
        best_metric_word = '|MAE gap|'
    elif problem_label == 'Problem 3':
        # y_obs is the absolute deviation from the GT reference value.
        if metric_lbl == 'ROC AUC':
            left_ylabel = 'Absolute deviation from reference ROC AUC'
            best_metric_word = 'Abs dev ROC AUC'
        else:
            left_ylabel = 'Absolute deviation from reference MAE'
            best_metric_word = 'Abs dev MAE'

    # 2nd rfscm metallicity: y_obs is a ROC AUC distance (smaller is better); no 1-MAE conversion and no 0..1 ylim cap.
    force_metallicity_as_rocdistance = (
        state_label == '2nd' and model_display_name == 'rfscm' and task_name == 'metallicity'
    )
    if force_metallicity_as_rocdistance:
        is_cls = False  # treat as regression: y is a numeric distance
        higher_better = False

    # State JSONs may store negative MAE (BayesOpt minimizes -MAE); flip sign for regression when the median is negative.
    if (not is_cls):
        med = float(np.nanmedian(y_obs))
        if med < -1e-9:
            y_obs = -y_obs

    # Convert MAE-like classification values to ROC AUC (1-MAE), except for the 2nd rfscm metallicity distance.
    if (not force_metallicity_as_rocdistance) and is_cls and higher_better \
            and (np.nanmax(y_obs) <= 1.0 + 1e-6) and (np.nanmin(y_obs) >= 0.0):
        if np.nanmean(y_obs) < 0.4999:
            print(f'  {task_name}: convert y_obs (MAE-like mean={np.nanmean(y_obs):.3f}) -> 1-MAE (ROC AUC)')
            y_obs = 1.0 - y_obs
    n_steps = len(y_obs)
    steps = np.arange(1, n_steps + 1)

    cap_drop = np.zeros(n_steps, dtype=bool)
    is_bulk_shear = (not is_cls) and task_name in ('bulk_modulus', 'shear_modulus')
    if is_bulk_shear:
        cap_drop |= (y_obs > BULK_SHEAR_CAP)
    valid = ~np.isnan(y_obs)

    # Robust MAD outlier removal for non-bulk/shear regression; bypassed for Problem 2 to keep the maximum |MAE gap|.
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
    n_drop = int(drop.sum())

    # Companion values (rfscm/crabnet) evaluated at the same (R,N,S).
    companion_values = {}
    for cm in companions:
        arr = np.full(n_steps, np.nan)
        for i in range(n_steps):
            rv, nv, sv = params_obs[i]
            v = lookup_result(task_name, cm, rv, nv, sv)
            if v is not None:
                arr[i] = float(v)
        if (not is_cls) and task_name in ('bulk_modulus', 'shear_modulus'):
            arr = np.where(arr > BULK_SHEAR_CAP, np.nan, arr)
        companion_values[cm] = arr

    bs = np.full(n_steps, np.nan)
    cur = float('inf') if (not higher_better) else float('-inf')
    for i in range(n_steps):
        if not np.isnan(y_plot[i]):
            if higher_better:
                if y_plot[i] > cur:
                    cur = y_plot[i]
            else:
                if y_plot[i] < cur:
                    cur = y_plot[i]
        bs[i] = cur

    notnan = ~np.isnan(y_plot)
    if notnan.any():
        best_i = int(np.nanargmax(y_plot) if higher_better else np.nanargmin(y_plot))
        best_y = float(y_plot[best_i])
    else:
        best_i, best_y = 0, float('nan')

    rns_labels = []
    for i in range(n_steps):
        rv, nv, sv = params_obs[i][0], params_obs[i][1], params_obs[i][2]
        rns_labels.append(f'({fmt_rv(rv)}, {fmt_int(nv)}, {fmt_int(sv)})')

    use_twinx = len(companions) > 0

    bsl_label = 'Best-so-far'

    if use_twinx:
        fig, ax_left = plt.subplots(figsize=(max(12, int(0.7 * n_steps) + 6), 7))
        ax_right = ax_left.twinx()
    else:
        fig, ax = plt.subplots(figsize=(max(12, int(0.7 * n_steps) + 6), 7))
        ax_left = ax
        ax_right = None

    companion_ax = ax_right if (ax_right is not None) else ax_left
    if 'crabnet' in companion_values:
        arr = companion_values['crabnet']
        mm = ~np.isnan(arr)
        if mm.any():
            companion_ax.scatter(steps[mm], arr[mm], marker='o', s=75,
                                 facecolors='none', edgecolors='#2E8B57', linewidths=1.8,
                                 zorder=5, label='CrabNet alone (same R,N,S)')
    if 'rfscm' in companion_values:
        arr = companion_values['rfscm']
        mm = ~np.isnan(arr)
        if mm.any():
            companion_ax.scatter(steps[mm], arr[mm], marker='^', s=95,
                                 facecolors='#FF8C00', edgecolors='black', linewidths=0.9,
                                 zorder=5, label='rfscm alone (same R,N,S)')

    if 'crabnet' in companion_values and 'rfscm' in companion_values:
        cb = companion_values['crabnet']
        rf = companion_values['rfscm']
        for i in range(n_steps):
            cbi, rfi = cb[i], rf[i]
            if np.isnan(cbi) or np.isnan(rfi):
                continue
            companion_ax.plot([steps[i], steps[i]], [cbi, rfi],
                              color='#888888', linestyle='-', lw=1.2, alpha=0.55, zorder=4)

    m1 = ~np.isnan(y_plot)
    if m1.any():
        ax_left.scatter(steps[m1], y_plot[m1], color='#2E5FAC', s=85, zorder=6,
                        label='Observed value')

    ax_left.plot(steps, bs, color='#C44E52', lw=2.9, alpha=0.95, zorder=4,
                 label=bsl_label)
    if notnan.any():
        ax_left.scatter(steps[best_i], best_y, s=310, facecolors='none',
                        edgecolors='gold', linewidths=3.1, zorder=7,
                        label=f'Best (k={best_i+1}, {best_metric_word}={best_y:.4g})')

    # Separator marks the initialization boundary (n_init, default 30%).
    planned_budget = 12 if (state_label == '1st' and task_name == 'refractive_index'
                            and model_display_name == 'rfscm') else 24
    sep_x = st.get('n_init', round(0.3 * planned_budget)) + 0.5
    if sep_x < n_steps + 0.1:
        ax_left.axvline(x=sep_x, color='black', linestyle='--', lw=1.7, zorder=3)

    if draw_refs:
        skip_ref = (ref_val is None) or (task_name in SKIP_REF_FOR)
        if not skip_ref:
            ref_r_str = f'({fmt_rv(ref_R)}, {fmt_int(ref_N)}, {fmt_int(ref_S)})'
            ax_left.axhline(y=float(ref_val), color='#551A8B', linestyle='-', lw=2.6, zorder=2,
                            label=f'Reference {ref_r_str}: {metric_lbl} = {ref_val:.4g}')
        for lbl, val, ls, col in extra_refs:
            ax_left.axhline(y=float(val), color=col, linestyle=ls, lw=2.4, zorder=2,
                            label=f'{lbl}: {metric_lbl} = {float(val):.4g}')

    ax_left.set_xlabel('Evaluation step k', fontsize=21)
    ax_left.set_ylabel(left_ylabel, fontsize=21, color='#1F3D7D')
    ax_left.tick_params(axis='y', which='major', labelsize=18, colors='#1F3D7D')
    ax_left.tick_params(axis='x', which='major', labelsize=18)
    ax_left.set_xticks(steps)
    ax_left.set_xticklabels(rns_labels, ha='right', rotation=45, fontsize=18)
    ax_left.set_xlim(0.5, n_steps + 0.5)

    if ax_right is not None:
        ax_right.set_ylabel(f'{metric_lbl} · standalone models (CrabNet / rfscm)',
                            fontsize=21, color='#2E6B49')
        ax_right.tick_params(axis='y', which='major', labelsize=18, colors='#2E6B49')
    else:
        ax_left.tick_params(axis='y', which='major', labelsize=18)

    if is_cls:
        ax_left.set_ylim(bottom=0.0, top=1.02)
        ax_left.ticklabel_format(axis='y', style='plain', useOffset=False)
    else:
        pools_left = [y_plot[~np.isnan(y_plot)]]
        flat_left = [p for p in pools_left if p.size > 0]
        if task_name in ('bulk_modulus', 'shear_modulus'):
            vis_l = np.concatenate(flat_left) if flat_left else np.array([BULK_SHEAR_CAP])
            top_l = BULK_SHEAR_CAP
            if vis_l.size:
                top_l = max(float(np.nanmax(vis_l) * 1.18), BULK_SHEAR_CAP * 1.1)
            ax_left.set_ylim(bottom=0.0, top=top_l)
            ax_left.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
            if ax_right is not None:
                pools_r = [companion_values[cm][~np.isnan(companion_values[cm])] for cm in companions]
                flat_r = [p for p in pools_r if p.size > 0]
                vis_r = np.concatenate(flat_r) if flat_r else np.array([BULK_SHEAR_CAP])
                top_r = BULK_SHEAR_CAP
                if vis_r.size:
                    top_r = max(float(np.nanmax(vis_r) * 1.18), BULK_SHEAR_CAP * 1.1)
                ax_right.set_ylim(bottom=0.0, top=top_r)
                ax_right.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
        else:
            ax_left.set_ylim(bottom=0.0)
            ax_left.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))
            if ax_right is not None:
                pools_r = [companion_values[cm][~np.isnan(companion_values[cm])] for cm in companions]
                flat_r = [p for p in pools_r if p.size > 0]
                if flat_r:
                    vis_r_max = float(np.nanmax(np.concatenate(flat_r)))
                    ax_right.set_ylim(bottom=0.0, top=vis_r_max * 1.2 if vis_r_max > 0 else None)
                ax_right.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4))

    ax_left.grid(True, linestyle='--', alpha=0.35)
    ax_left.set_title(f'Optimization trajectory · {model_display_name} / {task_name.replace("_", " ")}  [{problem_label}]',
                      fontsize=25, fontweight='bold', pad=24)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    if ax_right is not None:
        handles_right, labels_right = ax_right.get_legend_handles_labels()
    else:
        handles_right, labels_right = [], []
    all_handles = handles_left + handles_right
    all_labels = labels_left + labels_right
    seen = set(); final_h = []; final_l = []
    for h, l in zip(all_handles, all_labels):
        if l in seen:
            continue
        seen.add(l); final_h.append(h); final_l.append(l)
    if final_h:
        if companions:
            legend_upper_right = (state_label == '1st_dual' and
                                  model_display_name == 'crabnet_rfscm' and
                                  task_name == 'bulk_modulus')
            if legend_upper_right:
                ax_left.legend(final_h, final_l, fontsize=16, loc='upper right',
                               bbox_to_anchor=(0.99, 0.99), framealpha=0.93, ncol=1)
            else:
                ax_left.legend(final_h, final_l, fontsize=16, loc='upper left',
                               bbox_to_anchor=(0.01, 0.99), framealpha=0.93, ncol=1)
        else:
            ax_left.legend(final_h, final_l, fontsize=16.5, loc='best', framealpha=0.9)

    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_p = os.path.join(FIGURES_DIR, f'optim_{state_label}_{model_display_name}_{task_name}_trajectory.png')
    fig.savefig(out_p, dpi=150, bbox_inches='tight')
    plt.close(fig)

    hits = '  '.join([f'{cm}_hits={int(np.sum(~np.isnan(companion_values[cm])))}/{n_steps}'
                      for cm in companions])
    print(f'  {state_label:<8s} {model_display_name:<14s} {task_name:<20s} steps={n_steps:<2}  drop_dual={n_drop:<2}'
          f'  best_{best_metric_word}={best_y:.5g} @k={best_i+1:<2}  {hits}')
    return out_p


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for (state_label, model_name, path_fn, extra_refs_fn, draw_refs, companions, hb_override, problem_label) in STATES:
        print(f'\n==== STATE={state_label}  model={model_name}  draw_refs={draw_refs}  companions={companions}  higher_better_override={hb_override}  [{problem_label}] ====')
        for (tname, mlbl, hb_default) in TASKS:
            jpath = path_fn(tname) if callable(path_fn) else path_fn.get(tname)
            extra = extra_refs_fn(tname)
            draw_one(tname, mlbl, hb_default, state_label, model_name, jpath, GT_JSON(tname),
                     extra, draw_refs, companions, hb_override, problem_label)


if __name__ == "__main__":
    main()
