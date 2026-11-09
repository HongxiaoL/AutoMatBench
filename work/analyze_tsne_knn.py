import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from scipy.spatial.distance import cdist
import os
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

RANDOM_SEED = 42
NUM_FEATURES = 1024
# OFM CSVs are laid out as [material_id, formula, <1024 features>, property].
FEATURE_START_IDX = 2
N_SAMPLES = 1600

TASK_NAMES = [
    'band_gap',
    'bulk_modulus',
    'formation_energy',
    'metallicity',
    'refractive_index',
    'shear_modulus'
]

K_VALUES = [1, 5, 10, 20, 50, 100]


def load_task_data(task_name, R=0.0):
    input_file_mp = os.path.join(parent_dir, 'MP_data', f'MP_{task_name}', f'OFM_MP_{task_name}_property.csv')
    input_file_jarvis = os.path.join(parent_dir, 'jarvis_data', f'jarvis_{task_name}', f'OFM_jarvis_{task_name}_property.csv')

    df_mp = pd.read_csv(input_file_mp)
    if R > 0:
        df_jarvis = pd.read_csv(input_file_jarvis)
    else:
        df_jarvis = pd.DataFrame()

    n_jarvis = int(N_SAMPLES * R)
    n_mp = N_SAMPLES - n_jarvis

    if len(df_mp) < n_mp:
        sampled_mp = df_mp
    else:
        sampled_mp = df_mp.sample(n=n_mp, random_state=RANDOM_SEED)

    if R > 0:
        if len(df_jarvis) < n_jarvis:
            sampled_jarvis = df_jarvis
        else:
            sampled_jarvis = df_jarvis.sample(n=n_jarvis, random_state=RANDOM_SEED)
        sampled_df = pd.concat([sampled_mp, sampled_jarvis], ignore_index=True)
    else:
        sampled_df = sampled_mp.reset_index(drop=True)

    features = sampled_df.iloc[:, FEATURE_START_IDX:FEATURE_START_IDX + NUM_FEATURES]
    features = features.apply(pd.to_numeric, errors='coerce').astype(float)
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    return features.values, sampled_df


def run_tsne(features):
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, init='pca', verbose=0)
    return tsne.fit_transform(features)


def compute_knn_distances(distance_matrix, k):
    n = distance_matrix.shape[0]
    knn_dists = np.zeros((n, k))
    knn_indices = np.zeros((n, k), dtype=int)
    for i in range(n):
        sorted_idx = np.argsort(distance_matrix[i])
        sorted_idx = sorted_idx[sorted_idx != i]
        knn_indices[i] = sorted_idx[:k]
        knn_dists[i] = distance_matrix[i, knn_indices[i]]
    return knn_dists, knn_indices


def knn_overlap_ratio(indices_orig, indices_tsne, k):
    n = indices_orig.shape[0]
    overlaps = []
    for i in range(n):
        set_orig = set(indices_orig[i, :k])
        set_tsne = set(indices_tsne[i, :k])
        overlap = len(set_orig & set_tsne) / k
        overlaps.append(overlap)
    return np.mean(overlaps), np.std(overlaps)


def rank_correlation_per_point(knn_dists_orig, knn_dists_tsne):
    n = knn_dists_orig.shape[0]
    correlations = []
    for i in range(n):
        if np.std(knn_dists_orig[i]) == 0 or np.std(knn_dists_tsne[i]) == 0:
            continue
        corr, _ = spearmanr(knn_dists_orig[i], knn_dists_tsne[i])
        if not np.isnan(corr):
            correlations.append(corr)
    return np.mean(correlations), np.std(correlations)


def global_distance_correlation(dist_orig_flat, dist_tsne_flat):
    mask = ~np.eye(dist_orig_flat.shape[0], dtype=bool)
    d1 = dist_orig_flat[mask]
    d2 = dist_tsne_flat[mask]
    spearman_corr, _ = spearmanr(d1, d2)
    pearson_corr, _ = pearsonr(d1, d2)
    return pearson_corr, spearman_corr


def stress_function(dist_orig, dist_tsne):
    mask = ~np.eye(dist_orig.shape[0], dtype=bool)
    d_orig = dist_orig[mask]
    d_tsne = dist_tsne[mask]
    d_orig_norm = d_orig / np.max(d_orig)
    d_tsne_norm = d_tsne / np.max(d_tsne)
    numerator = np.sum((d_orig_norm - d_tsne_norm) ** 2)
    denominator = np.sum(d_orig_norm ** 2)
    return 1 - np.sqrt(numerator / denominator)


def analyze_task(task_name):
    print(f"\n{'='*60}")
    print(f"Analyzing task: {task_name}")
    print(f"{'='*60}")

    features, sampled_df = load_task_data(task_name, R=0.0)
    print(f"Loaded {features.shape[0]} samples, {features.shape[1]} features")

    features_2d = run_tsne(features)
    print(f"t-SNE completed, shape: {features_2d.shape}")

    dist_orig = pairwise_distances(features, metric='euclidean', n_jobs=1)
    dist_tsne = pairwise_distances(features_2d, metric='euclidean', n_jobs=1)
    print("Distance matrices computed")

    pearson_corr, spearman_corr = global_distance_correlation(dist_orig, dist_tsne)
    print(f"Global Pearson r: {pearson_corr:.4f}")
    print(f"Global Spearman ρ: {spearman_corr:.4f}")

    trust_score = stress_function(dist_orig, dist_tsne)
    print(f"Kruskal stress (1 - normalized): {trust_score:.4f}")

    max_k = max(K_VALUES)
    knn_dists_orig, knn_indices_orig = compute_knn_distances(dist_orig, max_k)
    knn_dists_tsne, knn_indices_tsne = compute_knn_distances(dist_tsne, max_k)

    results = {
        'task_name': task_name,
        'n_samples': features.shape[0],
        'n_features_orig': features.shape[1],
        'n_features_tsne': features_2d.shape[1],
        'global_pearson_r': pearson_corr,
        'global_spearman_rho': spearman_corr,
        'trust_score': trust_score,
        'per_k': {}
    }

    for k in K_VALUES:
        overlap_mean, overlap_std = knn_overlap_ratio(knn_indices_orig, knn_indices_tsne, k)
        rank_mean, rank_std = rank_correlation_per_point(
            knn_dists_orig[:, :k], knn_dists_tsne[:, :k]
        )

        dist_orig_k = knn_dists_orig[:, k-1] if k <= knn_dists_orig.shape[1] else knn_dists_orig[:, -1]
        dist_tsne_k = knn_dists_tsne[:, k-1] if k <= knn_dists_tsne.shape[1] else knn_dists_tsne[:, -1]

        results['per_k'][str(k)] = {
            'overlap_mean': overlap_mean,
            'overlap_std': overlap_std,
            'rank_corr_mean': rank_mean,
            'rank_corr_std': rank_std,
            'orig_kdist_mean': float(np.mean(dist_orig_k)),
            'orig_kdist_std': float(np.std(dist_orig_k)),
            'tsne_kdist_mean': float(np.mean(dist_tsne_k)),
            'tsne_kdist_std': float(np.std(dist_tsne_k)),
        }

        print(f"\nk={k:3d}:")
        print(f"  kNN overlap:    {overlap_mean:.4f} ± {overlap_std:.4f}")
        print(f"  Rank corr (ρ):  {rank_mean:.4f} ± {rank_std:.4f}")
        print(f"  Orig k-dist:    {np.mean(dist_orig_k):.3f} ± {np.std(dist_orig_k):.3f}")
        print(f"  t-SNE k-dist:   {np.mean(dist_tsne_k):.3f} ± {np.std(dist_tsne_k):.3f}")

    return results, features, features_2d, dist_orig, dist_tsne, knn_indices_orig, knn_indices_tsne


def plot_task_visualizations(task_name, features, features_2d, dist_orig, dist_tsne,
                              knn_indices_orig, knn_indices_tsne, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mask = ~np.eye(dist_orig.shape[0], dtype=bool)
    d_orig_flat = dist_orig[mask]
    d_tsne_flat = dist_tsne[mask]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f't-SNE vs Original Space Analysis · {task_name}', fontsize=18, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(features_2d[:, 0], features_2d[:, 1], s=12, alpha=0.6, c='steelblue', edgecolors='none')
    ax.set_xlabel('t-SNE Dim 1', fontsize=14)
    ax.set_ylabel('t-SNE Dim 2', fontsize=14)
    ax.set_title('t-SNE 2D Projection', fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=11)

    ax = axes[0, 1]
    idx = np.random.choice(len(d_orig_flat), size=min(50000, len(d_orig_flat)), replace=False)
    x_sample = d_orig_flat[idx]
    y_sample = d_tsne_flat[idx]
    ax.scatter(x_sample, y_sample, s=3, alpha=0.4, c='steelblue', edgecolors='none')
    lims = [min(x_sample.min(), y_sample.min()), max(x_sample.max(), y_sample.max())]
    ax.plot(lims, lims, 'k--', alpha=0.7, linewidth=1, label='y=x')
    p_corr, _ = pearsonr(x_sample, y_sample)
    s_corr, _ = spearmanr(x_sample, y_sample)
    ax.text(0.05, 0.95, f'Pearson r = {p_corr:.3f}\nSpearman ρ = {s_corr:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    ax.set_xlabel('Original-space Distance', fontsize=14)
    ax.set_ylabel('t-SNE-space Distance', fontsize=14)
    ax.set_title('Pairwise Distance Correlation', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.tick_params(labelsize=11)

    ax = axes[1, 0]
    k_values_plot = K_VALUES
    overlap_means = []
    overlap_stds = []
    for k in k_values_plot:
        overlap_mean, overlap_std = knn_overlap_ratio(knn_indices_orig, knn_indices_tsne, k)
        overlap_means.append(overlap_mean)
        overlap_stds.append(overlap_std)
    x_pos = np.arange(len(k_values_plot))
    bars = ax.bar(x_pos, overlap_means, yerr=overlap_stds,
                  capsize=4, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, overlap_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}' for k in k_values_plot], fontsize=11)
    ax.set_ylabel('kNN Set Overlap Ratio', fontsize=14)
    ax.set_ylim(0, min(1.0, max(overlap_means) * 1.4 + 0.05))
    ax.set_title(f'kNN Neighbor Overlap (t-SNE vs Original)', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=11)

    ax = axes[1, 1]
    bin_edges = np.linspace(0, 1, 51)
    k_for_hist = 10
    per_point_overlaps = []
    for i in range(knn_indices_orig.shape[0]):
        s_o = set(knn_indices_orig[i, :k_for_hist])
        s_t = set(knn_indices_tsne[i, :k_for_hist])
        per_point_overlaps.append(len(s_o & s_t) / k_for_hist)
    ax.hist(per_point_overlaps, bins=bin_edges, color='steelblue', alpha=0.8,
            edgecolor='black', linewidth=0.8)
    mean_ov = np.mean(per_point_overlaps)
    median_ov = np.median(per_point_overlaps)
    ax.axvline(mean_ov, color='crimson', linestyle='--', linewidth=2,
               label=f'Mean = {mean_ov:.3f}')
    ax.axvline(median_ov, color='darkgreen', linestyle=':', linewidth=2,
               label=f'Median = {median_ov:.3f}')
    ax.set_xlabel(f'k={k_for_hist} Neighbor Overlap Ratio (per-point)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'Distribution of Point-wise kNN Overlap (k={k_for_hist})', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, f'tsne_knn_analysis_{task_name}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    return fig_path


def plot_summary(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tasks = list(all_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('t-SNE vs Original Space · Cross-Task Summary', fontsize=20, fontweight='bold')

    metrics = [
        ('global_pearson_r', 'Global Pearson r', 'steelblue'),
        ('global_spearman_rho', 'Global Spearman ρ', 'seagreen'),
        ('trust_score', 'Kruskal Trust Score (1-stress)', 'darkorange')
    ]

    for ax_idx, (metric_key, ylabel, color) in enumerate(metrics[:3]):
        ax = axes[ax_idx // 2, ax_idx % 2]
        vals = [all_results[t][metric_key] for t in tasks]
        bars = ax.bar(range(len(tasks)), vals, color=color, alpha=0.8,
                      edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(tasks)))
        display_names = [t.replace('_', ' ').title() for t in tasks]
        ax.set_xticklabels(display_names, rotation=25, ha='right', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(min(0, min(vals) - 0.1), max(1.0, max(vals) * 1.15))
        ax.set_title(f'{ylabel} Across Tasks', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=11)

    ax = axes[1, 1]
    k_display = [1, 5, 10, 20, 50, 100]
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for ti, task in enumerate(tasks):
        overlaps = [all_results[task]['per_k'][str(k)]['overlap_mean'] for k in k_display]
        ax.plot(k_display, overlaps, marker=markers[ti % len(markers)],
                color=colors[ti % len(colors)], linewidth=2, markersize=8,
                label=task.replace('_', ' ').title())
    ax.set_xlabel('k (Number of Neighbors)', fontsize=14)
    ax.set_ylabel('kNN Set Overlap Ratio', fontsize=14)
    ax.set_title('kNN Overlap Curve by Task', fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(k_display)
    ax.set_xticklabels(k_display)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, 'tsne_knn_summary_all_tasks.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved summary figure: {fig_path}")


def main():
    np.random.seed(RANDOM_SEED)
    output_dir = os.path.join(parent_dir, 'work', 'figures', 'tsne_knn_analysis')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    fig_paths = {}

    for task_name in TASK_NAMES:
        try:
            results, features, features_2d, dist_orig, dist_tsne, knn_orig, knn_tsne = analyze_task(task_name)
            all_results[task_name] = results
            fig_path = plot_task_visualizations(
                task_name, features, features_2d, dist_orig, dist_tsne,
                knn_orig, knn_tsne, output_dir
            )
            fig_paths[task_name] = fig_path
        except Exception as e:
            print(f"Error analyzing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    results_json_path = os.path.join(output_dir, 'tsne_knn_analysis_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved analysis results to: {results_json_path}")

    plot_summary(all_results, output_dir)

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    header = f"{'Task':<22} {'Pearson':>8} {'Spearman':>10} {'Trust':>8}"
    for k in K_VALUES:
        header += f" {'k='+str(k):>8}"
    print(header)
    print("-" * len(header))
    for task in TASK_NAMES:
        if task not in all_results:
            continue
        r = all_results[task]
        line = f"{task:<22} {r['global_pearson_r']:>8.3f} {r['global_spearman_rho']:>10.3f} {r['trust_score']:>8.3f}"
        for k in K_VALUES:
            ov = r['per_k'][str(k)]['overlap_mean']
            line += f" {ov:>8.3f}"
        print(line)


if __name__ == "__main__":
    main()
