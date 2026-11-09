import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    ('band_gap',         'jarvis_band_gap',         'MP_band_gap'),
    ('bulk_modulus',     'jarvis_bulk_modulus',     'MP_bulk_modulus'),
    ('formation_energy', 'jarvis_formation_energy', 'MP_formation_energy'),
    ('metallicity',      'jarvis_metallicity',      'MP_metallicity'),
    ('refractive_index', 'jarvis_refractive_index', 'MP_refractive_index'),
    ('shear_modulus',    'jarvis_shear_modulus',    'MP_shear_modulus'),
]

DIST_THRESHOLD = 1e-3


def load_y_and_ofm(task, j_sub, m_sub):
    j_y_path = os.path.join(BASE_DIR, 'jarvis_data', j_sub, 'Y_value.csv')
    m_y_path = os.path.join(BASE_DIR, 'MP_data',   m_sub, 'Y_value.csv')
    j_ofm_path = os.path.join(BASE_DIR, 'jarvis_data', j_sub, f'OFM_{j_sub}_property.csv')
    m_ofm_path = os.path.join(BASE_DIR, 'MP_data',   m_sub, f'OFM_{m_sub}_property.csv')

    j_y = pd.read_csv(j_y_path).dropna(subset=['formula'])
    m_y = pd.read_csv(m_y_path).dropna(subset=['formula'])

    j_ofm = pd.read_csv(j_ofm_path)
    m_ofm = pd.read_csv(m_ofm_path)

    j_feats = j_ofm.iloc[:, 2:-1].values
    m_feats = m_ofm.iloc[:, 2:-1].values

    j_id_to_idx = {uid: i for i, uid in enumerate(j_ofm.iloc[:, 0])}
    m_id_to_idx = {uid: i for i, uid in enumerate(m_ofm.iloc[:, 0])}

    j_val_map = dict(zip(j_y['material_id'], j_y[task]))
    m_val_map = dict(zip(m_y['material_id'], m_y[task]))

    j_formula_map = {}
    for _, row in j_y.iterrows():
        j_formula_map.setdefault(row['formula'], []).append(row['material_id'])

    m_formula_map = {}
    for _, row in m_y.iterrows():
        m_formula_map.setdefault(row['formula'], []).append(row['material_id'])

    return (j_y, m_y, j_feats, m_feats,
            j_id_to_idx, m_id_to_idx,
            j_val_map, m_val_map,
            j_formula_map, m_formula_map)


def find_matches(task, j_y, m_y, j_feats, m_feats,
                 j_id_to_idx, m_id_to_idx,
                 j_val_map, m_val_map,
                 j_formula_map, m_formula_map):
    common_formulas = set(j_formula_map.keys()) & set(m_formula_map.keys())

    matches = []
    for formula in common_formulas:
        j_ids = j_formula_map[formula]
        m_ids = m_formula_map[formula]

        best = None
        best_dist = float('inf')
        for j_id in j_ids:
            for m_id in m_ids:
                if j_id not in j_id_to_idx or m_id not in m_id_to_idx:
                    continue
                j_idx = j_id_to_idx[j_id]
                m_idx = m_id_to_idx[m_id]
                d = euclidean(j_feats[j_idx], m_feats[m_idx])
                if d < best_dist:
                    best_dist = d
                    best = (j_id, m_id, formula, d,
                            j_val_map.get(j_id, np.nan),
                            m_val_map.get(m_id, np.nan))
        if best is not None:
            matches.append(best)

    df = pd.DataFrame(matches, columns=[
        'jarvis_id', 'mp_id', 'formula', 'ofm_distance',
        f'jarvis_{task}', f'mp_{task}'])
    strict = df[df['ofm_distance'] <= DIST_THRESHOLD].copy()
    return common_formulas, df, strict


def main():
    print("=" * 90)
    print(f"{'Task':<20} {'Jarvis':>8} {'MP':>8} {'CommonF':>10} "
          f"{'Match(OFM<1e-3)':>18} {'LooseMatch':>12}")
    print("-" * 90)

    all_pairs = set()
    all_strict_pairs = set()
    rows = []

    for task, j_sub, m_sub in TASKS:
        (j_y, m_y, j_feats, m_feats,
         j_id_to_idx, m_id_to_idx,
         j_val_map, m_val_map,
         j_formula_map, m_formula_map) = load_y_and_ofm(task, j_sub, m_sub)

        common_formulas, df_loose, df_strict = find_matches(
            task, j_y, m_y, j_feats, m_feats,
            j_id_to_idx, m_id_to_idx,
            j_val_map, m_val_map,
            j_formula_map, m_formula_map)

        n_j = len(j_y)
        n_m = len(m_y)
        n_cf = len(common_formulas)
        n_loose = len(df_loose)
        n_strict = len(df_strict)

        print(f"{task:<20} {n_j:>8,} {n_m:>8,} {n_cf:>10,} "
              f"{n_strict:>18,} {n_loose:>12,}")

        pairs_loose = set(zip(df_loose['jarvis_id'], df_loose['mp_id']))
        pairs_strict = set(zip(df_strict['jarvis_id'], df_strict['mp_id']))
        all_pairs |= pairs_loose
        all_strict_pairs |= pairs_strict

        out_loose = os.path.join(BASE_DIR, f'matched_materials_{task}.csv')
        df_loose.sort_values('ofm_distance').to_csv(out_loose, index=False)

        rows.append({
            'task': task,
            'jarvis_count': n_j,
            'mp_count': n_m,
            'common_formulas': n_cf,
            'strict_ofm_match_le_1e-3': n_strict,
            'loose_best_formula_match': n_loose,
            'jarvis_unique_pct_vs_mp': 100.0 * (n_j - n_strict) / n_j if n_j else 0,
            'mp_unique_pct_vs_jarvis': 100.0 * (n_m - n_strict) / n_m if n_m else 0,
        })

    print("-" * 90)
    summary = pd.DataFrame(rows)
    tot_j = summary['jarvis_count'].sum()
    tot_m = summary['mp_count'].sum()

    union_ids_jarvis_strict = set()
    union_ids_mp_strict = set()
    for task, j_sub, m_sub in TASKS:
        path = os.path.join(BASE_DIR, f'matched_materials_{task}.csv')
        df = pd.read_csv(path)
        df = df[df['ofm_distance'] <= DIST_THRESHOLD]
        union_ids_jarvis_strict |= set(df['jarvis_id'])
        union_ids_mp_strict |= set(df['mp_id'])

    print()
    print("=== Cross-Task Union of Unique Material Pairs (OFM distance <= 1e-3) ===")
    print(f"  Total unique (jarvis_id, mp_id) pairs : {len(all_strict_pairs):,}")
    print(f"  Distinct Jarvis IDs involved          : {len(union_ids_jarvis_strict):,}")
    print(f"  Distinct MP IDs involved              : {len(union_ids_mp_strict):,}")

    all_jarvis_ids = set()
    all_mp_ids = set()
    for task, j_sub, m_sub in TASKS:
        j_y = pd.read_csv(os.path.join(BASE_DIR, 'jarvis_data', j_sub, 'Y_value.csv'))
        m_y = pd.read_csv(os.path.join(BASE_DIR, 'MP_data',   m_sub, 'Y_value.csv'))
        all_jarvis_ids |= set(j_y['material_id'])
        all_mp_ids     |= set(m_y['material_id'])

    print()
    print("=== Overall Dataset-Wide Duplication (ID-union across all 6 tasks) ===")
    print(f"  Jarvis unique materials (union of 6)  : {len(all_jarvis_ids):,}")
    print(f"  MP unique materials (union of 6)      : {len(all_mp_ids):,}")
    print(f"  Duplicated Jarvis IDs (strict)        : {len(union_ids_jarvis_strict):,}  "
          f"({100*len(union_ids_jarvis_strict)/len(all_jarvis_ids):.2f}% of Jarvis)")
    print(f"  Duplicated MP IDs (strict)            : {len(union_ids_mp_strict):,}  "
          f"({100*len(union_ids_mp_strict)/len(all_mp_ids):.2f}% of MP)")

    summary.to_csv(os.path.join(BASE_DIR, 'duplicate_summary.csv'), index=False)
    print()
    print(f"Saved per-task summary -> duplicate_summary.csv")
    print(f"Saved per-task loose matches -> matched_materials_<task>.csv")


if __name__ == '__main__':
    main()
