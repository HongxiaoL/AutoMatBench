"""
Held-out test counts + fold-level uncertainty for representative configs.

Outputs:
  - rfscm_bandgap_ntest_audit.csv  : all 858 configs, nominal/split/eval counts
  - c29_representative_configs.csv : representative configs with fold-level MAE stats
"""
import json, csv
from pathlib import Path
import statistics

WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR / 'data'
RESULT_DIR = WORK_DIR / 'result'

# ---- Representative configs ----
# Rationale:
#   - (0.2, 24, 1..8): sweep S at fixed R,N to show how test-set size affects
#     fold-level scatter; includes the smallest test sets (S=1,2).
#   - (0.2, 4, 4)   : best configuration for K=24.
#   - (0.2, 48, 4)  : larger-N control at the same S=4 and R=0.2.
REPRESENTATIVE = [
    (0.2, 24, 1),
    (0.2, 24, 2),
    (0.2, 24, 4),
    (0.2, 24, 8),
    (0.2,  4, 4),
    (0.2, 48, 4),
]

def load_result(R, N, S):
    p = RESULT_DIR / f'rfscm_band_gap_({R},{N},{S})' / 'results.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def load_split(R, N, S):
    p = DATA_DIR / f'band_gap_({R},{N},{S})_folds' / 'test.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

# ---- 1. Full audit over all 858 configs ----
all_rows = []
for d in sorted(RESULT_DIR.glob('rfscm_band_gap_*')):
    rf = d / 'results.json'
    if not rf.exists():
        continue
    inner = d.name.split('(')[1].rstrip(')')
    parts = inner.split(',')
    R, N, S = float(parts[0]), int(parts[1]), int(parts[2])
    data = load_result(R, N, S)
    if data is None:
        continue
    fr = data['fold_results']
    per_eval = [len(v['targets']) for v in fr.values()]
    n_eval = sum(per_eval)
    n_folds_eval = len(fr)
    mae = data['average_metrics']['mae']

    sp = load_split(R, N, S)
    if sp is not None:
        per_split = [len(v) for v in sp.values()]
        n_split = sum(per_split)
        n_folds_split = len(sp)
    else:
        per_split = []
        n_split = None
        n_folds_split = None

    all_rows.append({
        'R': R, 'N': N, 'S': S,
        'n_nominal': N * S,
        'n_split': n_split,
        'n_folds_split': n_folds_split,
        'n_eval': n_eval,
        'n_folds_eval': n_folds_eval,
        'mae': mae,
    })

audit_csv = RESULT_DIR.parent / 'rfscm_bandgap_ntest_audit.csv'
with open(audit_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['R', 'N', 'S', 'n_nominal', 'n_split', 'n_folds_split',
                'n_eval', 'n_folds_eval', 'mae'])
    for r in sorted(all_rows, key=lambda x: (x['N'], x['S'], x['R'])):
        w.writerow([r['R'], r['N'], r['S'], r['n_nominal'], r['n_split'],
                    r['n_folds_split'], r['n_eval'], r['n_folds_eval'],
                    f"{r['mae']:.6f}"])

n_total = len(all_rows)
n_split_deficit = sum(1 for r in all_rows if r['n_split'] != r['n_nominal'])
n_eval_deficit = sum(1 for r in all_rows
                     if r['n_split'] is not None and r['n_eval'] != r['n_split'])
print(f"Full audit: {n_total} configs")
print(f"  split-level deficit (n_split != N*S)   : {n_split_deficit}")
print(f"  eval-level deficit  (n_eval  != n_split): {n_eval_deficit}")

# ---- 2. Representative configs with fold-level uncertainty ----
rep_rows = []
for R, N, S in REPRESENTATIVE:
    data = load_result(R, N, S)
    sp = load_split(R, N, S)
    if data is None:
        continue
    fr = data['fold_results']
    maes = [v['mae'] for v in fr.values()]
    per_eval = [len(v['targets']) for v in fr.values()]
    n_eval = sum(per_eval)
    n_folds = len(fr)
    mean_mae = statistics.mean(maes)
    std_mae = statistics.stdev(maes) if n_folds > 1 else 0.0
    sem = std_mae / (n_folds ** 0.5) if n_folds > 1 else 0.0
    ci95_half = 1.96 * sem
    if sp is not None:
        per_split = [len(v) for v in sp.values()]
        n_split = sum(per_split)
        n_folds_split = len(sp)
    else:
        n_split = n_folds_split = None
    rep_rows.append({
        'R': R, 'N': N, 'S': S,
        'n_nominal': N * S,
        'n_split': n_split,
        'n_folds_split': n_folds_split,
        'n_eval': n_eval,
        'n_folds_eval': n_folds,
        'mean_mae': mean_mae,
        'std_mae': std_mae,
        'sem': sem,
        'ci95_half': ci95_half,
        'min_mae': min(maes),
        'max_mae': max(maes),
        'per_fold_mae': maes,
    })

rep_csv = RESULT_DIR.parent / 'c29_representative_configs.csv'
with open(rep_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['R', 'N', 'S', 'n_nominal', 'n_split', 'n_folds_split',
                'n_eval', 'n_folds_eval', 'mean_mae', 'std_mae', 'sem',
                'ci95_half', 'min_mae', 'max_mae'])
    for r in rep_rows:
        w.writerow([r['R'], r['N'], r['S'], r['n_nominal'], r['n_split'],
                    r['n_folds_split'], r['n_eval'], r['n_folds_eval'],
                    f"{r['mean_mae']:.6f}", f"{r['std_mae']:.6f}",
                    f"{r['sem']:.6f}", f"{r['ci95_half']:.6f}",
                    f"{r['min_mae']:.6f}", f"{r['max_mae']:.6f}"])

print(f"\nSaved: {audit_csv}")
print(f"Saved: {rep_csv}")
print("\nRepresentative configs:")
print(f"{'R':>4} {'N':>3} {'S':>3} {'nom':>5} {'split':>6} {'eval':>6} "
      f"{'folds':>6} {'mean_mae':>10} {'std':>8} {'SEM':>8} {'95%CI±':>8}")
for r in rep_rows:
    print(f"{r['R']:>4} {r['N']:>3} {r['S']:>3} {r['n_nominal']:>5} "
          f"{str(r['n_split']):>6} {r['n_eval']:>6} "
          f"{r['n_folds_eval']:>3}/{r['n_folds_split']:<2} "
          f"{r['mean_mae']:>10.4f} {r['std_mae']:>8.4f} {r['sem']:>8.4f} "
          f"{r['ci95_half']:>8.4f}")
