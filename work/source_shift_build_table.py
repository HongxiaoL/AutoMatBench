"""
Source composition shift analysis for rfscm_band_gap 858-config experiment.

Goal: determine whether the n x r interaction effect on MAE is weakened
after controlling for train/test data-source composition shift (D).

Inputs:
  - work/result/rfscm_band_gap_(R,N,S)/results.json   (MAE + params)
  - work/data/band_gap_(R,N,S)_folds/train.json        (split membership)

Outputs:
  - config_with_source_shift.csv
  - source_shift_core_result.md  (written by a companion step)
"""
import os, glob, json, re
import numpy as np
import pandas as pd

WORK = os.path.dirname(os.path.abspath(__file__))

RDIR = os.path.join(WORK, 'result')
DDIR = os.path.join(WORK, 'data')

def parse_rns(name):
    m = re.search(r'\(([^)]+)\)', name)
    if not m:
        return None
    parts = m.group(1).split(',')
    R = float(parts[0])
    N = int(parts[1])
    S = int(parts[2])
    return R, N, S

def source_counts(ids):
    n_mp = sum(1 for x in ids if str(x).startswith('mp-'))
    n_jv = sum(1 for x in ids if str(x).startswith('JVASP'))
    return n_mp, n_jv

def main():
    rdirs = sorted(glob.glob(os.path.join(RDIR, 'rfscm_band_gap_*')))
    rows = []
    skipped = []
    for rd in rdirs:
        rns = parse_rns(os.path.basename(rd))
        if rns is None:
            continue
        R, N, S = rns
        config_id = f'rfscm_band_gap_({R},{N},{S})'

        # ---- MAE ----
        rj = os.path.join(rd, 'results.json')
        if not os.path.exists(rj):
            skipped.append((config_id, 'no results.json'))
            continue
        with open(rj) as f:
            res = json.load(f)
        mae = res.get('average_metrics', {}).get('mae', None)
        succ_folds = res.get('successful_folds', 0)
        if mae is None:
            skipped.append((config_id, 'no mae'))
            continue

        # ---- splits ----
        ddir = os.path.join(DDIR, f'band_gap_({R},{N},{S})_folds')
        tj = os.path.join(ddir, 'train.json')
        if not os.path.exists(tj):
            skipped.append((config_id, 'no train.json'))
            continue
        with open(tj) as f:
            splits = json.load(f)
        folds = list(splits.keys())
        n_folds = len(folds)
        if n_folds == 0:
            skipped.append((config_id, 'empty splits'))
            continue

        # full pool = union of all train ids across folds
        full = set()
        for fid in folds:
            full |= set(splits[fid])
        pool_mp, pool_jv = source_counts(full)
        pool_total = pool_mp + pool_jv
        pool_j_frac = pool_jv / pool_total if pool_total else np.nan

        D_list = []
        p_train_list = []
        p_test_list = []
        n_train_mp_list = []
        n_train_jv_list = []
        n_test_mp_list = []
        n_test_jv_list = []
        test_size_list = []
        train_size_list = []

        for fid in folds:
            tr = set(splits[fid])
            te = full - tr
            tr_mp, tr_jv = source_counts(tr)
            te_mp, te_jv = source_counts(te)
            tr_tot = tr_mp + tr_jv
            te_tot = te_mp + te_jv
            p_tr = tr_jv / tr_tot if tr_tot else np.nan
            p_te = te_jv / te_tot if te_tot else np.nan
            D_f = abs(p_te - p_tr) if (tr_tot and te_tot) else np.nan
            D_list.append(D_f)
            p_train_list.append(p_tr)
            p_test_list.append(p_te)
            n_train_mp_list.append(tr_mp)
            n_train_jv_list.append(tr_jv)
            n_test_mp_list.append(te_mp)
            n_test_jv_list.append(te_jv)
            train_size_list.append(tr_tot)
            test_size_list.append(te_tot)

        D_arr = np.array(D_list, dtype=float)
        valid = ~np.isnan(D_arr)
        D_mean = float(np.mean(D_arr[valid])) if valid.any() else np.nan
        # test-size weighted (consistency check; MAE here is simple mean, so D_mean is primary)
        ts = np.array(test_size_list, dtype=float)
        if valid.any() and np.sum(ts[valid]) > 0:
            D_weighted = float(np.average(D_arr[valid], weights=ts[valid]))
        else:
            D_weighted = np.nan

        rows.append({
            'config_id': config_id,
            'R': R,
            'N': N,
            'S': S,
            # alias per task notation
            'r': R,
            'n': N,
            's': S,
            'MAE': mae,
            'successful_folds': succ_folds,
            'n_folds_splits': n_folds,
            'pool_total': pool_total,
            'pool_j_frac': pool_j_frac,
            # primary shift metric: simple mean over folds (matches MAE aggregation)
            'D': D_mean,
            'D_weighted_by_test_size': D_weighted,
            'mean_p_train_J': float(np.nanmean(p_train_list)),
            'mean_p_test_J': float(np.nanmean(p_test_list)),
            'mean_train_size': float(np.mean(train_size_list)),
            'mean_test_size': float(np.mean(test_size_list)),
            'std_D_over_folds': float(np.std(D_arr[valid])) if valid.any() else np.nan,
        })

    df = pd.DataFrame(rows)
    out = os.path.join(WORK, 'config_with_source_shift.csv')
    df.to_csv(out, index=False)
    print(f'Saved {out} with {len(df)} configs.')
    if skipped:
        print(f'Skipped {len(skipped)} configs:')
        for cid, reason in skipped[:10]:
            print(f'  {cid}: {reason}')

    print()
    print('=== Summary ===')
    print(f'configs: {len(df)}')
    print(f'MAE: mean={df.MAE.mean():.4f} std={df.MAE.std():.4f} min={df.MAE.min():.4f} max={df.MAE.max():.4f}')
    print(f'D  : mean={df.D.mean():.4f} std={df.D.std():.4f} min={df.D.min():.4f} max={df.D.max():.4f}')
    print(f'corr(MAE, D)  Pearson={df.MAE.corr(df.D):.4f}  Spearman={df.MAE.corr(df.D, method="spearman"):.4f}')
    print(f'corr(MAE, pool_j_frac)  Pearson={df.MAE.corr(df.pool_j_frac):.4f}')
    print(f'corr(D, pool_j_frac)    Pearson={df.D.corr(df.pool_j_frac):.4f}')
    print(f'corr(D, R)              Pearson={df.D.corr(df.R):.4f}')
    print()
    print('D by R (mean):')
    print(df.groupby('R')['D'].agg(['mean','std','count']).round(4).to_string())
    print()
    print('D by N (mean):')
    print(df.groupby('N')['D'].agg(['mean','std','count']).round(4).to_string())
    print()
    print('D by S (mean):')
    print(df.groupby('S')['D'].agg(['mean','std','count']).round(4).to_string())

if __name__ == '__main__':
    main()
