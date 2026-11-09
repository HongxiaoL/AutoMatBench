import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import os
import json
import logging
import traceback
import sys
from pathlib import Path

# Project root is the parent of the work/ directory.
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from common import params

RANDOM_SEED = 42
NUM_FEATURES = 1024
FEATURE_START_IDX = 2
N_SAMPLES = 1600
N_SELECT = 480

def process_data(task_name, R, N, S, feature_slice_includes_target=False):
    """
    Process data for a specific task and parameter set (R, N, S).
    Generates train/val/test splits and saves them to JSON files.
    """
    if task_name not in params.tasks or R not in params.R_values or N not in params.N_values or S not in params.S_values:
        raise ValueError("Configuration is outside the declared task/grid space.")
    if N == 60 and S >= 9:
        raise ValueError("N=60 requires S<=8.")
    print(f"\nProcessing task: {task_name} | R={R}, N={N}, S={S}", flush=True)
    
    np.random.seed(RANDOM_SEED)

    folder_name = f"{task_name}_({R},{N},{S})_folds"
    output_folds_dir = os.path.join(parent_dir, 'work', 'data', folder_name)
    
    if os.path.exists(os.path.join(output_folds_dir, 'train.json')) and \
       os.path.exists(os.path.join(output_folds_dir, 'val.json')) and \
       os.path.exists(os.path.join(output_folds_dir, 'test.json')):
        print(f"Reusing recorded splits for {task_name} (R={R}, N={N}, S={S}); no regeneration.")
        logging.info(f"Task {task_name} (R={R}, N={N}, S={S}) skipped (already completed).")
        return

    try:
        input_file_mp = os.path.join(parent_dir, 'MP_data', f'MP_{task_name}', f'OFM_MP_{task_name}_property.csv')
        input_file_jarvis = os.path.join(parent_dir, 'jarvis_data', f'jarvis_{task_name}', f'OFM_jarvis_{task_name}_property.csv')
        
        os.makedirs(output_folds_dir, exist_ok=True)
        print(f"Reading data from {input_file_mp} and {input_file_jarvis}...")
        logging.info(f"Starting task: {task_name} | R={R}, N={N}, S={S}")

        if not os.path.exists(input_file_mp):
            raise FileNotFoundError(input_file_mp)
        
        if R > 0 and not os.path.exists(input_file_jarvis):
            raise FileNotFoundError(input_file_jarvis)

        df_mp = pd.read_csv(input_file_mp)
        if R > 0:
            df_jarvis = pd.read_csv(input_file_jarvis)
            print(f"Jarvis Data shape: {df_jarvis.shape}")
        else:
            df_jarvis = pd.DataFrame()
            print("Skipping Jarvis data read (R=0)")

        print(f"MP Data shape: {df_mp.shape}")

        # The material ID is column 0 in both sources.
        id_col_idx = 0

        n_jarvis = int(N_SAMPLES * R)
        n_mp = N_SAMPLES - n_jarvis

        print(f"Sampling {n_jarvis} from Jarvis and {n_mp} from MP...")

        if len(df_mp) < n_mp:
            raise ValueError(f"Insufficient MP entries: {len(df_mp)} < {n_mp}")
        else:
            sampled_mp = df_mp.sample(n=n_mp, random_state=RANDOM_SEED)

        if R > 0:
            if len(df_jarvis) < n_jarvis:
                raise ValueError(f"Insufficient JARVIS entries: {len(df_jarvis)} < {n_jarvis}")
            else:
                sampled_jarvis = df_jarvis.sample(n=n_jarvis, random_state=RANDOM_SEED)
            
            sampled_df = pd.concat([sampled_mp, sampled_jarvis], ignore_index=True)
        else:
            sampled_df = sampled_mp.reset_index(drop=True)

        try:
            sampled_df['id_num'] = sampled_df.iloc[:, id_col_idx].str.extract(r'(\d+)').astype(float)
            sampled_df['id_prefix'] = sampled_df.iloc[:, id_col_idx].str.extract(r'([a-zA-Z-]+)')
            sampled_df = sampled_df.sort_values(by=['id_prefix', 'id_num'])
            sampled_df = sampled_df.drop(columns=['id_num', 'id_prefix'])
        except Exception as sort_err:
            print(f"Warning: Sorting failed ({sort_err}). Using default order.")
            logging.warning(f"Sorting failed: {sort_err}")

        start = 3 if feature_slice_includes_target else FEATURE_START_IDX
        if feature_slice_includes_target:
            logging.warning("Feature slice starts at column 3 and includes the target column.")
        features = sampled_df.iloc[:, start:start+NUM_FEATURES]
        if features.shape[1] != NUM_FEATURES:
            raise ValueError("Expected exactly 1024 descriptor columns.")
        print(f"Feature shape for t-SNE: {features.shape}")

        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.astype(float)
        
        if features.isnull().values.any():
            print("Warning: NaNs detected in features. Filling with 0.")
            features = features.fillna(0)
            
        if np.isinf(features.values).any():
            print("Warning: Infs detected in features. Replacing with 0.")
            features = features.replace([np.inf, -np.inf], 0)

        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, init='pca', verbose=0)
        features_2d = tsne.fit_transform(features)

        print("Calculating Gaussian KDE in 2D structural space...")
        kde = gaussian_kde(features_2d.T)
        density = kde(features_2d.T)

        result_df = pd.DataFrame({
            'id': sampled_df.iloc[:, id_col_idx].values,
            'density': density,
            'feat_x': features_2d[:, 0],
            'feat_y': features_2d[:, 1],
            'original_idx': np.arange(len(sampled_df))
        })

        if len(result_df) < N_SELECT:
            print(f"Warning: Not enough samples ({len(result_df)}) to select {N_SELECT}. Using all.")
            selected_df = result_df
        else:
            selected_df = result_df.sort_values(by='density').head(N_SELECT)

        print(f"Selected {len(selected_df)} samples with lowest density.")

        print(f"Running K-Means clustering with {N} clusters...")
        kmeans = KMeans(n_clusters=N, random_state=RANDOM_SEED)
        selected_features_2d = selected_df[['feat_x', 'feat_y']].values
        selected_df = selected_df.copy()
        selected_df['cluster'] = kmeans.fit_predict(selected_features_2d)

        print("Processing clusters to find anchors and neighbors...")
        
        clusters_test_ids = []
        valid_cluster_ids = []

        for cluster_id in range(N):
            cluster_samples = selected_df[selected_df['cluster'] == cluster_id]
            
            if len(cluster_samples) == 0:
                continue
            
            valid_cluster_ids.append(cluster_id)

            center_2d = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
            samples_2d = cluster_samples[['feat_x', 'feat_y']].values
            
            dists_to_center = pairwise_distances(samples_2d, center_2d).flatten()
            anchor_idx_in_cluster = np.argmin(dists_to_center)
            
            anchor_global_idx = cluster_samples.index[anchor_idx_in_cluster]
            anchor_high_feat = features.iloc[anchor_global_idx].values.reshape(1, -1)
            
            cluster_global_idxs = cluster_samples.index
            cluster_high_feats = features.iloc[cluster_global_idxs].values
            
            dists_to_anchor = pairwise_distances(cluster_high_feats, anchor_high_feat).flatten()
            
            cluster_df = cluster_samples.copy()
            cluster_df['dist_to_anchor'] = dists_to_anchor
            
            cluster_sorted = cluster_df.sort_values(by='dist_to_anchor')
            sorted_ids = cluster_sorted['id'].tolist()
            
            # First of the S samples is the anchor; the rest are its nearest neighbors.
            top_ids = sorted_ids[:S]
            clusters_test_ids.append(top_ids)

        N_folds = len(clusters_test_ids)
        print(f"Generating {N_folds} folds...")

        train_dict = {}
        val_dict = {}
        test_dict = {}

        for i in range(N_folds):
            fold_id = str(i)
            
            test_ids = clusters_test_ids[i]
            
            current_cluster_id = valid_cluster_ids[i]
            
            train_df = selected_df[selected_df['cluster'] != current_cluster_id]
            
            train_ids_set = set(train_df['id'].values)
            train_ids = sorted(list(train_ids_set))
            
            train_ids_split, val_ids_split = train_test_split(train_ids, test_size=0.1, random_state=RANDOM_SEED)
            
            train_ids_split.sort()
            val_ids_split.sort()
            
            train_dict[fold_id] = train_ids_split
            val_dict[fold_id] = val_ids_split
            test_dict[fold_id] = test_ids

        print(f"Saving to {output_folds_dir}...")
        
        with open(os.path.join(output_folds_dir, 'train.json'), 'w') as f:
            json.dump(train_dict, f)

        with open(os.path.join(output_folds_dir, 'val.json'), 'w') as f:
            json.dump(val_dict, f)

        with open(os.path.join(output_folds_dir, 'test.json'), 'w') as f:
            json.dump(test_dict, f)

        with open(os.path.join(output_folds_dir, 'split_metadata.json'), 'w') as f:
            json.dump({"feature_start_index": start, "feature_slice_includes_target": feature_slice_includes_target,
                       "seed": RANDOM_SEED, "n_sample": len(sampled_df),
                       "n_selected": len(selected_df)}, f, indent=2)
        print(f"Done processing task: {task_name} | R={R}, N={N}, S={S}")
        logging.info(f"Task {task_name} (R={R}, N={N}, S={S}) completed successfully.")

    except Exception as e:
        print(f"Error processing task {task_name} (R={R}, N={N}, S={S}): {str(e)}")
        logging.error(f"Error processing task {task_name} (R={R}, N={N}, S={S}): {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    log_file = os.path.join(parent_dir, 'work', 'log', 'process_data.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="Generate one clustering-based split.")
    parser.add_argument("--task", choices=params.tasks, required=True)
    parser.add_argument("--R", type=float, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--S", type=int, required=True)
    parser.add_argument("--feature-slice-includes-target", action="store_true",
                        help="Start the feature slice at column 3, which includes the target column.")
    args = parser.parse_args()
    process_data(args.task, args.R, args.N, args.S, args.feature_slice_includes_target)
