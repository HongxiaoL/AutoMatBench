"""
Statistical analysis: does controlling for source-shift D weaken the n x r interaction?

Approach (ANOVA / OLS with categorical factors):
  Factors: n (8 levels), r (11 levels), s (10 levels), treated as categorical.
  Response: MAE.
  Covariate: D (continuous source-shift metric).

We compare the "extra explanatory power" of the n:r interaction term
WITH and WITHOUT D in the model.

"Extra explanatory power of n:r" is measured two ways:
  (a) Type-III partial SS of C(n):C(r)  as a fraction of total SS.
  (b) Delta R^2 = R^2(model with n:r) - R^2(model without n:r),
      computed both with D present in both models and with D absent.

Also report the p-value of n:r in both settings, and an ANOVA table.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

WORK = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(WORK, 'config_with_source_shift.csv')

def ss_table(model, label):
    """Return a small df with term, df, sumsq, meansq, F, p."""
    import statsmodels.stats.anova as anova
    try:
        aov = anova.anova_lm(model, typ=3)
    except Exception as e:
        return pd.DataFrame()
    aov = aov.reset_index().rename(columns={'index': 'term'})
    aov['model'] = label
    return aov

def fit_and_extract(df, include_D, include_nr, include_other_2way=True):
    """
    Fit OLS with categorical factors; returns (model, r2).
    """
    terms = ['C(n)', 'C(r)', 'C(s)']
    if include_other_2way:
        terms += ['C(n):C(s)', 'C(r):C(s)']
    if include_nr:
        terms.append('C(n):C(r)')
    if include_D:
        terms.append('D')
    formula = 'MAE ~ ' + ' + '.join(terms)
    model = ols(formula, data=df).fit()
    return model, model.rsquared

def main():
    df = pd.read_csv(CSV)
    print(f'Loaded {len(df)} configs.')

    for c in ['n', 'r', 's']:
        df[c] = df[c].astype('category')

    # ---- A. Type-III ANOVA, full 2-way model, WITHOUT D ----
    terms_noD = ['C(n)', 'C(r)', 'C(s)', 'C(n):C(r)', 'C(n):C(s)', 'C(r):C(s)']
    f_noD = 'MAE ~ ' + ' + '.join(terms_noD)
    m_noD = ols(f_noD, data=df).fit()

    # ---- B. Type-III ANOVA, full 2-way model, WITH D ----
    terms_D = ['D'] + terms_noD
    f_D = 'MAE ~ ' + ' + '.join(terms_D)
    m_D = ols(f_D, data=df).fit()

    print()
    print('=' * 90)
    print('TYPE-III ANOVA  (full 2-way factorial)')
    print('=' * 90)

    aov_noD = sm.stats.anova_lm(m_noD, typ=3)
    aov_D = sm.stats.anova_lm(m_D, typ=3)

    def fmt(aov, label):
        aov = aov.copy()
        aov['pct_of_total_SS'] = 100.0 * aov['sum_sq'] / aov['sum_sq'].sum()
        aov = aov[['df', 'sum_sq', 'pct_of_total_SS', 'F', 'PR(>F)']]
        aov.columns = [f'{c}_{label}' for c in aov.columns]
        return aov

    cmp = pd.concat([fmt(aov_noD, 'noD'), fmt(aov_D, 'D')], axis=1)
    print(cmp.round(4).to_string())

    nr_noD = aov_noD.loc['C(n):C(r)']
    nr_D = aov_D.loc['C(n):C(r)']
    total_ss_noD = aov_noD['sum_sq'].sum()
    total_ss_D = aov_D['sum_sq'].sum()

    print()
    print('=' * 90)
    print('n:r INTERACTION COMPARISON')
    print('=' * 90)
    print(f'{"metric":<40} {"no D":>15} {"with D":>15} {"change":>15}')
    print('-' * 90)
    print(f'{"Type-III SS(n:r)":<40} {nr_noD["sum_sq"]:>15.4f} {nr_D["sum_sq"]:>15.4f} {nr_D["sum_sq"]-nr_noD["sum_sq"]:>15.4f}')
    print(f'{"SS(n:r) % of total SS":<40} {100*nr_noD["sum_sq"]/total_ss_noD:>14.2f}% {100*nr_D["sum_sq"]/total_ss_D:>14.2f}%')
    print(f'{"F(n:r)":<40} {nr_noD["F"]:>15.3f} {nr_D["F"]:>15.3f}')
    print(f'{"p(n:r)":<40} {nr_noD["PR(>F)"]:>15.3e} {nr_D["PR(>F)"]:>15.3e}')
    print(f'{"R^2 (full 2-way model)":<40} {m_noD.rsquared:>15.4f} {m_D.rsquared:>15.4f}')
    print(f'{"R^2_adj":<40} {m_noD.rsquared_adj:>15.4f} {m_D.rsquared_adj:>15.4f}')

    # ---- C. Delta R^2 of n:r, with and without D ----
    print()
    print('=' * 90)
    print('DELTA R^2 OF n:r  ( = R2(with n:r) - R2(without n:r) )')
    print('=' * 90)

    _, r2_base_noD = fit_and_extract(df, include_D=False, include_nr=False)
    _, r2_full_noD = fit_and_extract(df, include_D=False, include_nr=True)
    dr2_noD = r2_full_noD - r2_base_noD

    _, r2_base_D = fit_and_extract(df, include_D=True, include_nr=False)
    _, r2_full_D = fit_and_extract(df, include_D=True, include_nr=True)
    dr2_D = r2_full_D - r2_base_D

    print(f'{"setting":<30} {"R2(no n:r)":>12} {"R2(with n:r)":>14} {"Delta R2(n:r)":>15}')
    print('-' * 75)
    print(f'{"without D":<30} {r2_base_noD:>12.4f} {r2_full_noD:>14.4f} {dr2_noD:>15.5f}')
    print(f'{"with D":<30} {r2_base_D:>12.4f} {r2_full_D:>14.4f} {dr2_D:>15.5f}')
    print()
    print(f'Delta R2(n:r) change when adding D: {dr2_D - dr2_noD:+.5f}')
    print(f'Relative change: {(dr2_D - dr2_noD)/dr2_noD*100:+.2f}%' if dr2_noD != 0 else 'n/a')

    # ---- D. D's own contribution ----
    print()
    print('=' * 90)
    print('D AS A PREDICTOR')
    print('=' * 90)
    m_donly = ols('MAE ~ D', data=df).fit()
    print(f'R2(MAE ~ D) alone            : {m_donly.rsquared:.5f}')
    r2_full_noD_val = r2_full_noD
    r2_full_D_val = r2_full_D
    print(f'R2 gain from adding D to 2-way: {r2_full_D_val - r2_full_noD_val:.5f}')
    print(f'D coefficient (full model)    : {m_D.params["D"]:.4f}  (p={m_D.pvalues["D"]:.3e})')
    from scipy.stats import spearmanr
    pr = np.corrcoef(df['MAE'], df['D'])[0, 1]
    sp, _ = spearmanr(df['MAE'], df['D'])
    print(f'corr(MAE, D)  Pearson={pr:.4f}  Spearman={sp:.4f}')

    # ---- E. Sensitivity: exclude R=0 and R=1 where D is degenerate ----
    print()
    print('=' * 90)
    print('SENSITIVITY: exclude R=0 and R=1 (D is exactly 0 there)')
    print('=' * 90)
    df_inner = df[(df['R'] > 0) & (df['R'] < 1)].copy()
    for c in ['n', 'r', 's']:
        df_inner[c] = df_inner[c].astype('category')
    print(f'Inner configs (0 < R < 1): {len(df_inner)}')

    m_noD_in = ols(f_noD, data=df_inner).fit()
    m_D_in = ols(f_D, data=df_inner).fit()
    aov_noD_in = sm.stats.anova_lm(m_noD_in, typ=3)
    aov_D_in = sm.stats.anova_lm(m_D_in, typ=3)
    nr_noD_in = aov_noD_in.loc['C(n):C(r)']
    nr_D_in = aov_D_in.loc['C(n):C(r)']
    tot_noD_in = aov_noD_in['sum_sq'].sum()
    tot_D_in = aov_D_in['sum_sq'].sum()
    print(f'{"metric":<40} {"no D":>15} {"with D":>15}')
    print('-' * 75)
    print(f'{"SS(n:r) % of total SS":<40} {100*nr_noD_in["sum_sq"]/tot_noD_in:>14.2f}% {100*nr_D_in["sum_sq"]/tot_D_in:>14.2f}%')
    print(f'{"p(n:r)":<40} {nr_noD_in["PR(>F)"]:>15.3e} {nr_D_in["PR(>F)"]:>15.3e}')
    print(f'{"R^2 (full 2-way)":<40} {m_noD_in.rsquared:>15.4f} {m_D_in.rsquared:>15.4f}')

    _, rb_n = fit_and_extract(df_inner, False, False)
    _, rf_n = fit_and_extract(df_inner, False, True)
    _, rb_d = fit_and_extract(df_inner, True, False)
    _, rf_d = fit_and_extract(df_inner, True, True)
    print(f'Delta R2(n:r)  without D: {rf_n-rb_n:.5f}   with D: {rf_d-rb_d:.5f}')

    summary = {
        'n_configs': int(len(df)),
        'n_configs_inner': int(len(df_inner)),
        'nr_ss_pct_noD': float(100 * nr_noD['sum_sq'] / total_ss_noD),
        'nr_ss_pct_D': float(100 * nr_D['sum_sq'] / total_ss_D),
        'nr_p_noD': float(nr_noD['PR(>F)']),
        'nr_p_D': float(nr_D['PR(>F)']),
        'r2_full_noD': float(m_noD.rsquared),
        'r2_full_D': float(m_D.rsquared),
        'dr2_nr_noD': float(dr2_noD),
        'dr2_nr_D': float(dr2_D),
        'r2_D_alone': float(m_donly.rsquared),
        'D_coef': float(m_D.params['D']),
        'D_p': float(m_D.pvalues['D']),
        'nr_ss_pct_noD_inner': float(100 * nr_noD_in['sum_sq'] / tot_noD_in),
        'nr_ss_pct_D_inner': float(100 * nr_D_in['sum_sq'] / tot_D_in),
        'dr2_nr_noD_inner': float(rf_n - rb_n),
        'dr2_nr_D_inner': float(rf_d - rb_d),
        'corr_pearson_MAE_D': float(pr),
        'corr_spearman_MAE_D': float(sp),
    }
    with open(os.path.join(WORK, 'source_shift_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print('Saved source_shift_stats.json')

    cmp.to_csv(os.path.join(WORK, 'anova_comparison.csv'))
    print('Saved anova_comparison.csv')

if __name__ == '__main__':
    import json
    main()
