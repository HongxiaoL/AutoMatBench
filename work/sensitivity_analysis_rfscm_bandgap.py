"""Sensitivity analysis for rfscm band_gap (R, N, S) parameter space."""

import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "legend.fontsize": 12.5,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

Ns = [4, 6, 8, 10, 12, 24, 48, 60]
Rs = [round(x / 10, 1) for x in range(0, 11)]
Ss = list(range(1, 11))

WORK = Path(__file__).resolve().parent / "result"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    rows = []
    for p in WORK.glob("rfscm_band_gap_*/results.json"):
        m = re.match(r"rfscm_band_gap_\(([0-9.]+),([0-9]+),([0-9]+)\)", p.parent.name)
        if not m:
            continue
        R, N, S = round(float(m.group(1)), 1), int(m.group(2)), int(m.group(3))
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        am = d.get("average_metrics") or d.get("avg_metrics") or {}
        mae = am.get("mae")
        if mae is None:
            continue
        try:
            mae = float(mae)
        except Exception:
            continue
        if mae != mae or mae < 0:
            continue
        rmse = am.get("rmse")
        try:
            rmse = float(rmse) if rmse is not None else None
        except Exception:
            rmse = None
        rows.append(dict(N=N, R=R, S=S, mae=mae, rmse=rmse))
    return rows


def anova_decomposition(rows):
    """Simple variance decomposition based on group means (sum-of-squares)."""
    mae = np.array([r["mae"] for r in rows])
    N_arr = np.array([r["N"] for r in rows])
    R_arr = np.array([r["R"] for r in rows])
    S_arr = np.array([r["S"] for r in rows])

    total_ss = np.sum((mae - mae.mean()) ** 2)

    def factor_ss(values):
        levels = sorted(set(values.tolist()))
        ss = 0.0
        for lv in levels:
            mask = values == lv
            if mask.sum() == 0:
                continue
            group_mean = mae[mask].mean()
            ss += mask.sum() * (group_mean - mae.mean()) ** 2
        return ss

    ss_N = factor_ss(N_arr)
    ss_R = factor_ss(R_arr)
    ss_S = factor_ss(S_arr)

    ss_NR = 0.0
    for n in Ns:
        for r in Rs:
            mask = (N_arr == n) & (R_arr == r)
            if mask.sum() == 0:
                continue
            n_mean = mae[N_arr == n].mean()
            r_mean = mae[R_arr == r].mean()
            cell_mean = mae[mask].mean()
            ss_NR += mask.sum() * (cell_mean - n_mean - r_mean + mae.mean()) ** 2

    ss_NS = 0.0
    for n in Ns:
        for s in Ss:
            mask = (N_arr == n) & (S_arr == s)
            if mask.sum() == 0:
                continue
            n_mean = mae[N_arr == n].mean()
            s_mean = mae[S_arr == s].mean()
            cell_mean = mae[mask].mean()
            ss_NS += mask.sum() * (cell_mean - n_mean - s_mean + mae.mean()) ** 2

    ss_RS = 0.0
    for r in Rs:
        for s in Ss:
            mask = (R_arr == r) & (S_arr == s)
            if mask.sum() == 0:
                continue
            r_mean = mae[R_arr == r].mean()
            s_mean = mae[S_arr == s].mean()
            cell_mean = mae[mask].mean()
            ss_RS += mask.sum() * (cell_mean - r_mean - s_mean + mae.mean()) ** 2

    explained = ss_N + ss_R + ss_S + ss_NR + ss_NS + ss_RS
    residual = max(total_ss - explained, 0.0)

    components = {
        "n (main)": ss_N,
        "r (main)": ss_R,
        "s (main)": ss_S,
        "n × r": ss_NR,
        "n × s": ss_NS,
        "r × s": ss_RS,
        "Residual": residual,
    }
    total = sum(components.values())
    pct = {k: v / total * 100 for k, v in components.items()}
    return components, pct, total_ss


def correlation_analysis(rows):
    N_arr = np.array([r["N"] for r in rows], dtype=float)
    R_arr = np.array([r["R"] for r in rows], dtype=float)
    S_arr = np.array([r["S"] for r in rows], dtype=float)
    mae = np.array([r["mae"] for r in rows])

    def pearson(x, y):
        return np.corrcoef(x, y)[0, 1]

    def spearman(x, y):
        rank_x = np.argsort(np.argsort(x)).astype(float)
        rank_y = np.argsort(np.argsort(y)).astype(float)
        return np.corrcoef(rank_x, rank_y)[0, 1]

    results = {}
    for name, arr in [("N", N_arr), ("R", R_arr), ("S", S_arr)]:
        results[name] = {
            "pearson": pearson(arr, mae),
            "spearman": spearman(arr, mae),
        }
    return results


def main_effects(rows):
    mae_by_n = defaultdict(list)
    mae_by_r = defaultdict(list)
    mae_by_s = defaultdict(list)
    for r in rows:
        mae_by_n[r["N"]].append(r["mae"])
        mae_by_r[r["R"]].append(r["mae"])
        mae_by_s[r["S"]].append(r["mae"])

    effects = {
        "N": {n: {"mean": statistics.mean(mae_by_n[n]),
                  "median": statistics.median(mae_by_n[n]),
                  "std": statistics.pstdev(mae_by_n[n]),
                  "n": len(mae_by_n[n])} for n in Ns},
        "R": {rv: {"mean": statistics.mean(mae_by_r[rv]),
                   "median": statistics.median(mae_by_r[rv]),
                   "std": statistics.pstdev(mae_by_r[rv]),
                   "n": len(mae_by_r[rv])} for rv in Rs},
        "S": {sv: {"mean": statistics.mean(mae_by_s[sv]),
                   "median": statistics.median(mae_by_s[sv]),
                   "std": statistics.pstdev(mae_by_s[sv]),
                   "n": len(mae_by_s[sv])} for sv in Ss},
    }
    return effects


def interaction_heatmap(rows):
    def make_grid(row_key, col_key, row_vals, col_vals):
        grid = np.full((len(row_vals), len(col_vals)), np.nan)
        for i, rv in enumerate(row_vals):
            for j, cv in enumerate(col_vals):
                vals = [r["mae"] for r in rows
                        if r[row_key] == rv and r[col_key] == cv]
                if vals:
                    grid[i, j] = statistics.mean(vals)
        return grid

    return {
        "N_R": make_grid("N", "R", Ns, Rs),
        "N_S": make_grid("N", "S", Ns, Ss),
        "R_S": make_grid("R", "S", Rs, Ss),
    }


def topk_analysis(rows, k=50):
    sorted_rows = sorted(rows, key=lambda r: r["mae"])
    topk = sorted_rows[:k]
    bottomk = sorted_rows[-k:]

    def dist(items, key, vals):
        counts = defaultdict(int)
        for it in items:
            counts[it[key]] += 1
        return [counts.get(v, 0) for v in vals]

    return {
        "top_N": dist(topk, "N", Ns),
        "top_R": dist(topk, "R", Rs),
        "top_S": dist(topk, "S", Ss),
        "bottom_N": dist(bottomk, "N", Ns),
        "bottom_R": dist(bottomk, "R", Rs),
        "bottom_S": dist(bottomk, "S", Ss),
        "top_list": [(r["R"], r["N"], r["S"], r["mae"]) for r in topk[:20]],
        "bottom_list": [(r["R"], r["N"], r["S"], r["mae"]) for r in bottomk[:10]],
    }


def plot_variance_bar(pct, components):
    fig, ax = plt.subplots(figsize=(14, 9.5))
    names = list(pct.keys())
    values = list(pct.values())
    colors = ["#4C72B0", "#55A868", "#8172B3",
              "#DDAA33", "#C44E52", "#CCB974", "#BBBBBB"]

    bars = ax.barh(names, values, color=colors[:len(names)],
                   edgecolor="black", linewidth=1.0, height=0.65)

    for bar, v, comp in zip(bars, values, components.values()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%\n(SS={comp:.2f})",
                va="center", ha="left", fontsize=18, fontweight="bold")

    ax.set_yticklabels(names, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlabel("Variance explained (%)", fontsize=22)
    ax.set_xlim(0, max(values) * 1.85)
    ax.set_title("MAE variance decomposition (r, n, s)",
                 fontsize=26, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.9)
    ax.invert_yaxis()
    fig.tight_layout()
    out = FIG_DIR / "sensitivity_variance_decomposition.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_main_effects(effects, global_med):
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharey=False)

    ax = axes[0]
    vals_N = [effects["N"][n]["median"] for n in Ns]
    std_N = [effects["N"][n]["std"] for n in Ns]
    ax.errorbar(Ns, vals_N, yerr=std_N, fmt="o-", color="#4C72B0",
                markersize=8, linewidth=2.2, capsize=5, label="median ± std")
    ax.axhline(global_med, color="gray", linestyle=":", linewidth=1.5,
               label=f"global median={global_med:.3f}")
    ax.set_xlabel("N (feature set size tier)")
    ax.set_ylabel("MAE (median)")
    ax.set_title("Main effect: N")
    ax.set_xticks(Ns)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    ax = axes[1]
    vals_R = [effects["R"][rv]["median"] for rv in Rs]
    std_R = [effects["R"][rv]["std"] for rv in Rs]
    ax.errorbar(Rs, vals_R, yerr=std_R, fmt="s-", color="#55A868",
                markersize=8, linewidth=2.2, capsize=5, label="median ± std")
    ax.axhline(global_med, color="gray", linestyle=":", linewidth=1.5,
               label=f"global median={global_med:.3f}")
    ax.set_xlabel("R (dataset mixing ratio)")
    ax.set_ylabel("MAE (median)")
    ax.set_title("Main effect: R")
    ax.set_xticks(Rs)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    ax = axes[2]
    vals_S = [effects["S"][sv]["median"] for sv in Ss]
    std_S = [effects["S"][sv]["std"] for sv in Ss]
    ax.errorbar(Ss, vals_S, yerr=std_S, fmt="^-", color="#8172B3",
                markersize=8, linewidth=2.2, capsize=5, label="median ± std")
    ax.axhline(global_med, color="gray", linestyle=":", linewidth=1.5,
               label=f"global median={global_med:.3f}")
    ax.set_xlabel("S (test-set size per fold)")
    ax.set_ylabel("MAE (median)")
    ax.set_title("Main effect: S")
    ax.set_xticks(Ss)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    fig.suptitle("Main effects of (N, R, S) on MAE (rfscm / band_gap, n=858)",
                 fontsize=20, fontweight="bold", y=1.015)
    fig.tight_layout()
    out = FIG_DIR / "sensitivity_main_effects.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_heatmaps(grids):
    fig, axes = plt.subplots(1, 3, figsize=(32, 11))

    def plot_heatmap(ax, data, row_labels, col_labels, title, xlabel, ylabel, cmap="viridis_r"):
        valid = data[~np.isnan(data)]
        vmin, vmax = valid.min(), valid.max()
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=20)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=20)
        ax.set_title(title, pad=20, fontsize=24, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                            fontsize=18,
                            color="white" if (i + j) % 2 == 0 else "black",
                            fontweight="bold")
        cb = plt.colorbar(im, ax=ax, shrink=0.88, label="mean MAE")
        cb.ax.tick_params(labelsize=18)
        cb.set_label("mean MAE", fontsize=20)

    plot_heatmap(axes[0], grids["N_R"], [f"n={n}" for n in Ns],
                 [f"{r:.1f}" for r in Rs],
                 "Mean MAE by n and r", "r", "n")
    plot_heatmap(axes[1], grids["N_S"], [f"n={n}" for n in Ns],
                 [f"s={s}" for s in Ss],
                 "Mean MAE by n and s", "s", "n")
    plot_heatmap(axes[2], grids["R_S"], [f"{r:.1f}" for r in Rs],
                 [f"s={s}" for s in Ss],
                 "Mean MAE by r and s", "s", "r")

    fig.tight_layout()
    out = FIG_DIR / "sensitivity_interaction_heatmaps.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_correlation_bar(corr):
    fig, ax = plt.subplots(figsize=(9, 6))
    factors = list(corr.keys())
    x = np.arange(len(factors))
    w = 0.35
    pears = [corr[f]["pearson"] for f in factors]
    spear = [corr[f]["spearman"] for f in factors]

    b1 = ax.bar(x - w / 2, pears, w, label="Pearson r", color="#4C72B0",
                edgecolor="black")
    b2 = ax.bar(x + w / 2, spear, w, label="Spearman ρ", color="#55A868",
                edgecolor="black")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.01 if h > 0 else -0.025),
                    f"{h:+.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylabel("Correlation with MAE")
    ax.set_title("Linear / rank correlation of (N, R, S) with MAE")
    ax.set_ylim(min(min(pears), min(spear)) - 0.15,
                max(max(pears), max(spear)) + 0.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / "sensitivity_correlations.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_topk_distribution(topk):
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    k = sum(topk["top_N"])

    def plot_dist(ax, vals, labels, title_top, title_bot, top_data, bot_data, color):
        x = np.arange(len(vals))
        w = 0.38
        ax.bar(x - w / 2, top_data, w, label=f"Top-{k}", color=color, alpha=0.85,
               edgecolor="black")
        ax.bar(x + w / 2, bot_data, w, label=f"Bottom-{k}", color="#BBBBBB",
               alpha=0.7, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{title_top} (left) vs {title_bot} (right)")
        ax.set_ylabel(f"Count in top/bottom-{k}")
        ax.legend(loc="best", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plot_dist(axes[0, 0], Ns, [f"N={n}" for n in Ns],
              "Top MAE distribution by N",
              "Bottom MAE distribution by N",
              topk["top_N"], topk["bottom_N"], "#4C72B0")
    plot_dist(axes[1, 0], Rs, [f"{r:.1f}" for r in Rs],
              "Top MAE distribution by R",
              "Bottom MAE distribution by R",
              topk["top_R"], topk["bottom_R"], "#55A868")
    plot_dist(axes[2, 0], Ss, [f"S={s}" for s in Ss],
              "Top MAE distribution by S",
              "Bottom MAE distribution by S",
              topk["top_S"], topk["bottom_S"], "#8172B3")

    for j in range(3):
        axes[j, 1].axis("off")

    ax_text = axes[0, 1]
    lines = ["Top-20 configurations (R, N, S, MAE):"]
    for i, (r, n, s, mae) in enumerate(topk["top_list"]):
        lines.append(f"  #{i+1:>2}: (R={r:.1f}, N={n:>2}, S={s:>2}) → MAE={mae:.4f}")
    ax_text.text(0.05, 0.95, "\n".join(lines), transform=ax_text.transAxes,
                 verticalalignment="top", fontsize=10.5, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5DC",
                           edgecolor="gray"))

    ax_text2 = axes[2, 1]
    lines2 = ["Worst-10 configurations (R, N, S, MAE):"]
    for i, (r, n, s, mae) in enumerate(topk["bottom_list"]):
        lines2.append(f"  #{i+1:>2}: (R={r:.1f}, N={n:>2}, S={s:>2}) → MAE={mae:.4f}")
    ax_text2.text(0.05, 0.95, "\n".join(lines2), transform=ax_text2.transAxes,
                  verticalalignment="top", fontsize=10.5, family="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFE4E1",
                            edgecolor="gray"))

    fig.suptitle(f"Top-{k} vs Bottom-{k} parameter distribution analysis",
                 fontsize=20, fontweight="bold", y=1.015)
    fig.tight_layout()
    out = FIG_DIR / "sensitivity_topk_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    rows = load_data()
    print(f"Loaded {len(rows)} valid data points")
    mae_all = [r["mae"] for r in rows]
    global_med = statistics.median(mae_all)
    global_mean = statistics.mean(mae_all)
    global_std = statistics.pstdev(mae_all)

    print(f"\n=== Global MAE Statistics ===")
    print(f"  n={len(rows)}, median={global_med:.4f}, mean={global_mean:.4f}, std={global_std:.4f}")
    print(f"  min={min(mae_all):.4f}, max={max(mae_all):.4f}")

    print(f"\n=== ANOVA Variance Decomposition ===")
    components, pct, total_ss = anova_decomposition(rows)
    for name in components:
        print(f"  {name:>12s}: SS={components[name]:.3f}  {pct[name]:.2f}%")

    print(f"\n=== Correlation Analysis ===")
    corr = correlation_analysis(rows)
    for f in corr:
        print(f"  {f}: Pearson r={corr[f]['pearson']:+.4f}, Spearman ρ={corr[f]['spearman']:+.4f}")

    print(f"\n=== Main Effects (mean/median/std per level) ===")
    effects = main_effects(rows)
    for factor in ["N", "R", "S"]:
        print(f"\n  --- {factor} ---")
        for lv in effects[factor]:
            e = effects[factor][lv]
            print(f"    {factor}={str(lv):>4s}: mean={e['mean']:.4f}  med={e['median']:.4f}  std={e['std']:.4f}  n={e['n']}")

    print(f"\n=== Top-K Analysis (K=50) ===")
    topk = topk_analysis(rows, k=50)
    print("  Top-10 configs:")
    for i, (r, n, s, mae) in enumerate(topk["top_list"][:10]):
        print(f"    #{i+1}: (R={r:.1f}, N={n}, S={s}) → MAE={mae:.4f}")
    print("  Bottom-5 configs:")
    for i, (r, n, s, mae) in enumerate(topk["bottom_list"][:5]):
        print(f"    #{i+1}: (R={r:.1f}, N={n}, S={s}) → MAE={mae:.4f}")

    print(f"\n=== Interaction Heatmaps (N×R, N×S, R×S) ===")
    grids = interaction_heatmap(rows)
    for key, g in grids.items():
        valid = g[~np.isnan(g)]
        print(f"  {key}: mean={valid.mean():.4f}, range=[{valid.min():.4f}, {valid.max():.4f}]")

    print("\n=== Generating figures ===")
    f1 = plot_variance_bar(pct, components)
    print(f"  [OK] {f1}")
    f2 = plot_main_effects(effects, global_med)
    print(f"  [OK] {f2}")
    f3 = plot_heatmaps(grids)
    print(f"  [OK] {f3}")
    f4 = plot_correlation_bar(corr)
    print(f"  [OK] {f4}")
    f5 = plot_topk_distribution(topk)
    print(f"  [OK] {f5}")

    print("\n=== Sensitivity analysis DONE ===")


if __name__ == "__main__":
    main()
