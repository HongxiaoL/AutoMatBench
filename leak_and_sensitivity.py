import pandas as pd
import numpy as np
import os, json, glob, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    ('band_gap',         'jarvis_band_gap',         'MP_band_gap'),
    ('bulk_modulus',     'jarvis_bulk_modulus',     'MP_bulk_modulus'),
    ('formation_energy', 'jarvis_formation_energy', 'MP_formation_energy'),
    ('metallicity',      'jarvis_metallicity',      'MP_metallicity'),
    ('refractive_index', 'jarvis_refractive_index', 'MP_refractive_index'),
    ('shear_modulus',    'jarvis_shear_modulus',    'MP_shear_modulus'),
]

THRESHOLDS = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 1.0]

print("=" * 100)
print("PART 1: Threshold sensitivity — OFM minimum-distance distribution & strict match pairs at each threshold")
print("=" * 100)

sens_rows = []
for task, j_sub, m_sub in TASKS:
    f = os.path.join(BASE_DIR, f'matched_materials_{task}.csv')
    df = pd.read_csv(f)
    d = df['ofm_distance'].values
    stats = {'task': task, 'N_best_pairs': len(d)}
    for t in THRESHOLDS:
        stats[f'le_{t:.0e}'] = int(np.sum(d <= t))
    for p in [10, 25, 50, 75, 90, 95, 99]:
        stats[f'p{p}'] = np.percentile(d, p)
    stats['min'] = d.min()
    stats['max'] = d.max()
    sens_rows.append(stats)

sens = pd.DataFrame(sens_rows)

header = f"{'Task':<20} {'Nbest':>8}"
for t in THRESHOLDS:
    header += f"  ≤{t:.0e}:{'>7'}"
print(f"{'Task':<20} {'Nbest':>7}  " + "  ".join(f"≤{t:.0e}" for t in THRESHOLDS))
print("-" * 100)
for _, r in sens.iterrows():
    line = f"{r['task']:<20} {int(r['N_best_pairs']):>7,}  "
    line += "  ".join(f"{int(r[f'le_{t:.0e}']):>6,}" for t in THRESHOLDS)
    print(line)

print()
print("OFM minimum-distance percentiles (the pair with minimum distance per shared chemical formula)")
print(f"{'Task':<20} {'p10':>10} {'p25':>10} {'p50':>10} {'p75':>10} {'p90':>10} {'p95':>10} {'p99':>10}")
print("-" * 90)
for _, r in sens.iterrows():
    print(f"{r['task']:<20} {r['p10']:10.2e} {r['p25']:10.2e} {r['p50']:10.2e} {r['p75']:10.2e} {r['p90']:10.2e} {r['p95']:10.2e} {r['p99']:10.2e}")

sens.to_csv(os.path.join(BASE_DIR, 'threshold_sensitivity.csv'), index=False)

# PART 2: attribute-value difference statistics.
print()
print("=" * 100)
print("PART 2: Attribute value difference |Δ| = |Jarvis - MP| for strict duplicate pairs (OFM ≤ 1e-3)")
print("=" * 100)

attr_rows = []
for task, j_sub, m_sub in TASKS:
    f = os.path.join(BASE_DIR, f'matched_materials_{task}.csv')
    df = pd.read_csv(f)
    df_strict = df[df['ofm_distance'] <= 1e-3].copy()
    jcol = f'jarvis_{task}'
    mcol = f'mp_{task}'
    if len(df_strict) == 0:
        print(f"{task}:  0 strict pairs, skip")
        continue
    is_binary = task == 'metallicity'
    if is_binary:
        jv = df_strict[jcol].astype(int)
        mv = df_strict[mcol].astype(int)
        delta = (jv - mv).abs().astype(float)
        eq = int(np.sum(jv.values == mv.values))
    else:
        jv = pd.to_numeric(df_strict[jcol], errors='coerce')
        mv = pd.to_numeric(df_strict[mcol], errors='coerce')
        delta = (jv - mv).abs()
        eq = -1
    same_sign_cnt = int(np.sum(
        ((jv * mv) > 0).fillna(False) |
        (jv.isna() & mv.isna())
    ))
    r = {
        'task': task,
        'strict_pairs': len(df_strict),
        'mean_abs_delta': float(delta.mean()),
        'median_abs_delta': float(delta.median()),
        'p90_abs_delta': float(delta.quantile(0.9)),
        'max_abs_delta': float(delta.max()),
        'nonzero_delta_count': int(np.sum(delta > 1e-9)),
    }
    if is_binary:
        r['exact_equal_count'] = eq
    attr_rows.append(r)

    print(f"\n--- {task} (n={len(df_strict)} pairs) ---")
    print(f"  mean |Δ|   = {r['mean_abs_delta']:.4e}")
    print(f"  med  |Δ|   = {r['median_abs_delta']:.4e}")
    print(f"  p90  |Δ|   = {r['p90_abs_delta']:.4e}")
    print(f"  max  |Δ|   = {r['max_abs_delta']:.4e}")
    print(f"  Pairs with difference (|Δ|>0) = {r['nonzero_delta_count']}/{len(df_strict)}")
    if eq >= 0:
        print(f"  Pairs with identical classes = {eq}/{len(df_strict)}")
    df_strict['abs_delta'] = delta
    top = df_strict.nlargest(min(5, len(df_strict)), 'abs_delta')
    print(f"  Top largest |Δ| samples:")
    for _, row in top.iterrows():
        print(f"    {row['jarvis_id']:14s} <-> {row['mp_id']:14s}  {row['formula']:12s}  "
              f"J={row[jcol]:.6f}  M={row[mcol]:.6f}  |Δ|={row['abs_delta']:.6f}")

pd.DataFrame(attr_rows).to_csv(os.path.join(BASE_DIR, 'attribute_delta_summary.csv'), index=False)

# PART 3: fold leakage check.
print()
print("=" * 100)
print("PART 3: Fold leakage check — whether strict duplicate pairs are split across train/test sides of the same fold")
print("        (using band_gap_(0.2,60,3)_folds under the work directory of the band_gap task as an example)")
print("=" * 100)

def find_splits_file_for_task(task, all_paths):
    # Prefer the (0.2,60,3) split directory, then the largest N.
    task_map = {
        'band_gap': 'band_gap',
        'bulk_modulus': 'bulk_modulus',
        'bulk_mod': 'bulk_modulus',
        'formation_energy': 'formation_energy',
        'formation': 'formation_energy',
        'metallicity': 'metallicity',
        'refractive_index': 'refractive_index',
        'shear_modulus': 'shear_modulus',
        'shear_mod': 'shear_modulus',
    }
    matches = []
    for p in all_paths:
        dirn = os.path.basename(os.path.dirname(p))
        # Directory looks like band_gap_(0.2,60,3)_folds
        base = dirn.replace('_folds','')
        for k in task_map:
            if base.startswith(k):
                if task_map[k] == task:
                    matches.append(p)
                break
    if not matches:
        return None
    # Prefer "(0.2,60,3)", then "(0.2,48,3)", then larger N.
    def sort_key(p):
        m = re.search(r'\(([0-9\.]+),([0-9]+),([0-9]+)\)', p)
        R,N,S = (0,0,0)
        if m: R,N,S = float(m.group(1)), int(m.group(2)), int(m.group(3))
        pref = 0
        if '0.2,60,3' in p: pref = 1000
        elif '0.2,48,3' in p: pref = 900
        return (pref, N, S)
    return sorted(matches, key=sort_key)[-1]

import re

splits_paths = glob.glob(os.path.join(BASE_DIR, 'work', 'data', '*_folds', 'train.json'))
print(f"  Found {len(splits_paths)} train.json files under work/data/")

leak_rows = []
for task, j_sub, m_sub in TASKS:
    f = os.path.join(BASE_DIR, f'matched_materials_{task}.csv')
    df = pd.read_csv(f)
    df_strict = df[df['ofm_distance'] <= 1e-3].copy()
    if len(df_strict) == 0:
        continue

    splits_file = find_splits_file_for_task(task, splits_paths)
    if splits_file is None:
        print(f"\n--- {task}: no corresponding splits file found, skipping leakage check")
        continue

    print(f"\n--- {task}  using splits: {splits_file} ---")
    with open(splits_file) as fh:
        splits = json.load(fh)

    # Two supported splits formats:
    # A. {fold_id: [train_ids]} -> ids not listed are test
    # B. {fold_id: {train/val/test: [ids]}}
    all_ids_in_splits = set()
    lookup = {}
    for fold_id, fold_data in splits.items():
        if isinstance(fold_data, list):
            train_ids = set(fold_data)
            all_ids_in_splits |= train_ids
            # Fold side is resolved per pair because not all test ids are listed.
            for mid in train_ids:
                if isinstance(mid, str):
                    lookup.setdefault(mid, {})[fold_id] = ('train', train_ids)
        elif isinstance(fold_data, dict):
            for side in ['train', 'val', 'test']:
                ids = fold_data.get(side, []) or []
                all_ids_in_splits |= set(ids)
                for mid in ids:
                    if isinstance(mid, str):
                        lookup.setdefault(mid, {})[fold_id] = side
        else:
            raise ValueError(f"Unknown split structure for fold {fold_id}: {type(fold_data)}")

    n_pairs = len(df_strict)
    both_in = 0
    both_in_and_same_cv_fold_exists = 0
    same_side = 0
    diff_side = 0
    train_test_leak = 0
    train_val_leak = 0
    val_test_leak = 0
    fold_conflict_pairs = []
    train_only_in = 0
    test_only_in = 0

    # Structure A: a pair leaks when one id is in a fold's train set and the other is not.
    structure_a_train_per_fold = None
    first_val = next(iter(splits.values()))
    if isinstance(first_val, list):
        structure_a_train_per_fold = {fid: set(v) for fid, v in splits.items()}
        total_folds = len(structure_a_train_per_fold)
        print(f"  Splits structure: A (each fold stores a train list, everything else=test), total folds={total_folds}")
    else:
        total_folds = len(splits)
        print(f"  Splits structure: B (explicit train/val/test), total folds={total_folds}")

    for _, row in df_strict.iterrows():
        jid = row['jarvis_id']
        mid = row['mp_id']
        # Structure A requires both ids in some fold's train set; Structure B requires both in lookup.
        if structure_a_train_per_fold is not None:
            j_seen = any(jid in tr for tr in structure_a_train_per_fold.values())
            m_seen = any(mid in tr for tr in structure_a_train_per_fold.values())
            if not (j_seen and m_seen):
                continue
            both_in += 1
            pair_has_conflict = False
            common_folds = list(structure_a_train_per_fold.keys())
            for fid in common_folds:
                tr = structure_a_train_per_fold[fid]
                j_in = jid in tr
                m_in = mid in tr
                if j_in == m_in:
                    same_side += 1
                else:
                    diff_side += 1
                    pair_has_conflict = True
                    if j_in and not m_in:
                        # jid in train, mid in test
                        train_test_leak += 1
                    elif not j_in and m_in:
                        # jid in test, mid in train
                        train_test_leak += 1
        else:
            if jid not in lookup or mid not in lookup:
                continue
            both_in += 1
            j_folds = lookup[jid]
            m_folds = lookup[mid]
            common_folds = set(j_folds.keys()) & set(m_folds.keys())
            pair_has_conflict = False
            for fid in common_folds:
                js = j_folds[fid]
                ms = m_folds[fid]
                if js == ms:
                    same_side += 1
                else:
                    diff_side += 1
                    pair_has_conflict = True
                    s = tuple(sorted([js, ms]))
                    if s == ('test', 'train'):
                        train_test_leak += 1
                    elif s == ('train', 'val'):
                        train_val_leak += 1
                    elif s == ('test', 'val'):
                        val_test_leak += 1

        if pair_has_conflict:
            fold_conflict_pairs.append((jid, mid, row['formula']))

    print(f"  Total strict duplicate pairs   : {n_pairs}")
    print(f"  Pairs with both IDs in splits  : {both_in}")
    if total_folds and both_in:
        total_pair_fold_occurrences = same_side + diff_side
        print(f"  Total pair×fold co-occurrences : {total_pair_fold_occurrences}")
    print(f"  Same-side (no leak) count      : {same_side}")
    print(f"  Different-side (leak) total    : {diff_side}")
    print(f"    └─ train ↔ test leak       : {train_test_leak}")
    if train_val_leak or val_test_leak:
        print(f"    └─ train ↔ val  leak       : {train_val_leak}")
        print(f"    └─ val   ↔ test leak       : {val_test_leak}")
    print(f"  Pairs leaking in at least 1 fold : {len(fold_conflict_pairs)} / {both_in}"
          + (f"  ({100*len(fold_conflict_pairs)/both_in:.1f}%)" if both_in else ""))
    if fold_conflict_pairs[:10]:
        print(f"  Leaking pair samples (first 10):")
        for jid, mid, fml in fold_conflict_pairs[:10]:
            print(f"    {jid:14s} <-> {mid:14s}  {fml}")

    leak_rows.append({
        'task': task,
        'strict_pairs': n_pairs,
        'pairs_both_in_splits': both_in,
        'same_side_occurrences': same_side,
        'cross_side_occurrences_total': diff_side,
        'train_test_leaks': train_test_leak,
        'train_val_leaks': train_val_leak,
        'val_test_leaks': val_test_leak,
        'pairs_with_at_least_one_leak_fold': len(fold_conflict_pairs),
    })

pd.DataFrame(leak_rows).to_csv(os.path.join(BASE_DIR, 'fold_leak_summary.csv'), index=False)
print()
print("Saved -> threshold_sensitivity.csv, attribute_delta_summary.csv, fold_leak_summary.csv")
