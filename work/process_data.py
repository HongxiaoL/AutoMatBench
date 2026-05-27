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
from common.logging_utils import setup_file_logging

# Calculate parent directory (Project Root)
# Assuming structure: PROJECT_ROOT/work/process_data.py
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

try:
    from common import params
except ImportError:
    print("Warning: Could not import 'params' from 'common'. Ensure the module exists in parent_dir.")
    # You might want to define default params here if import fails, or let it fail.

# Global constants
RANDOM_SEED = 42
NUM_FEATURES = 1024
FEATURE_START_IDX = 3
N_SAMPLES = 1600
N_SELECT = 480

def process_data(task_name, R, N, S):
    """
    Process data for a specific task and parameter set (R, N, S).
    Generates train/val/test splits and saves them to JSON files.
    """
    print(f"\nProcessing task: {task_name} | R={R}, N={N}, S={S}")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Update output directory to include parameters
    # Folder name format: task_name_(R,N,S)_folds
    folder_name = f"{task_name}_({R},{N},{S})_folds"
    output_folds_dir = os.path.join(parent_dir, 'work', 'data', folder_name)
    
    # Check if task is already completed
    if os.path.exists(os.path.join(output_folds_dir, 'train.json')) and \
       os.path.exists(os.path.join(output_folds_dir, 'val.json')) and \
       os.path.exists(os.path.join(output_folds_dir, 'test.json')):
        print(f"Task {task_name} (R={R}, N={N}, S={S}) already completed. Skipping.")
        logging.info(f"Task {task_name} (R={R}, N={N}, S={S}) skipped (already completed).")
        return

    try:
        # Construct input file paths
        # Assuming data folders are in the project root (parent_dir)
        # Note: Previous paths included 'automatbench' explicitly. 
        # If parent_dir is '.../automatbench', we don't need to repeat it.
        # If the file structure is different, these paths might need adjustment.
        input_file_mp = os.path.join(parent_dir, 'MP_data', f'MP_{task_name}', f'OFM_MP_{task_name}_property.csv')
        input_file_jarvis = os.path.join(parent_dir, 'jarvis_data', f'jarvis_{task_name}', f'OFM_jarvis_{task_name}_property.csv')
        
        # Ensure output directory exists
        os.makedirs(output_folds_dir, exist_ok=True)
        print(f"Reading data from {input_file_mp} and {input_file_jarvis}...")
        logging.info(f"Starting task: {task_name} | R={R}, N={N}, S={S}")

        # Read the CSV files
        if not os.path.exists(input_file_mp):
            # Try alternative path if the direct join didn't work (backward compatibility with explicit folder name)
            # This handles the case where parent_dir is one level up (e.g. /home/hxli) and data is in /home/hxli/automatbench/MP_data
            alt_input_file_mp = os.path.join(parent_dir, 'automatbench', 'MP_data', f'MP_{task_name}', f'OFM_MP_{task_name}_property.csv')
            if os.path.exists(alt_input_file_mp):
                input_file_mp = alt_input_file_mp
                # Also update jarvis path
                input_file_jarvis = os.path.join(parent_dir, 'automatbench', 'jarvis_data', f'jarvis_{task_name}', f'OFM_jarvis_{task_name}_property.csv')
            else:
                msg = f"Warning: {input_file_mp} not found. Skipping task."
                print(msg)
                logging.warning(msg)
                return
        
        # Check Jarvis file if needed
        if R > 0 and not os.path.exists(input_file_jarvis):
            msg = f"Warning: {input_file_jarvis} not found. Skipping task."
            print(msg)
            logging.warning(msg)
            return

        df_mp = pd.read_csv(input_file_mp)
        if R > 0:
            df_jarvis = pd.read_csv(input_file_jarvis)
            print(f"Jarvis Data shape: {df_jarvis.shape}")
        else:
            df_jarvis = pd.DataFrame() # Empty
            print("Skipping Jarvis data read (R=0)")

        print(f"MP Data shape: {df_mp.shape}")

        # Identify the material ID column (assuming same for both)
        id_col_idx = 0

        # Sampling
        n_jarvis = int(N_SAMPLES * R)
        n_mp = N_SAMPLES - n_jarvis

        print(f"Sampling {n_jarvis} from Jarvis and {n_mp} from MP...")

        if len(df_mp) < n_mp:
            print(f"Warning: Not enough MP data ({len(df_mp)}). Using all.")
            sampled_mp = df_mp
        else:
            sampled_mp = df_mp.sample(n=n_mp, random_state=RANDOM_SEED)

        if R > 0:
            if len(df_jarvis) < n_jarvis:
                print(f"Warning: Not enough Jarvis data ({len(df_jarvis)}). Using all.")
                sampled_jarvis = df_jarvis
            else:
                sampled_jarvis = df_jarvis.sample(n=n_jarvis, random_state=RANDOM_SEED)
            
            # Combine
            sampled_df = pd.concat([sampled_mp, sampled_jarvis], ignore_index=True)
        else:
            sampled_df = sampled_mp.reset_index(drop=True)

        # Sort by material ID for consistency
        try:
            sampled_df['id_num'] = sampled_df.iloc[:, id_col_idx].str.extract(r'(\d+)').astype(float)
            sampled_df['id_prefix'] = sampled_df.iloc[:, id_col_idx].str.extract(r'([a-zA-Z-]+)')
            sampled_df = sampled_df.sort_values(by=['id_prefix', 'id_num'])
            sampled_df = sampled_df.drop(columns=['id_num', 'id_prefix'])
        except Exception as sort_err:
            print(f"Warning: Sorting failed ({sort_err}). Using default order.")
            logging.warning(f"Sorting failed: {sort_err}")

        # Get all IDs
        # all_ids = set(sampled_df.iloc[:, id_col_idx].values) # Not used in current logic but kept if needed

        # Extract features
        features = sampled_df.iloc[:, FEATURE_START_IDX:FEATURE_START_IDX+NUM_FEATURES]
        print(f"Feature shape for t-SNE: {features.shape}")

        # Clean features
        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.astype(float)
        
        if features.isnull().values.any():
            print("Warning: NaNs detected in features. Filling with 0.")
            features = features.fillna(0)
            
        if np.isinf(features.values).any():
            print("Warning: Infs detected in features. Replacing with 0.")
            features = features.replace([np.inf, -np.inf], 0)

        # Run t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, init='pca', verbose=0)
        features_2d = tsne.fit_transform(features)

        # Calculate Gaussian KDE
        print("Calculating Gaussian KDE in 2D structural space...")
        kde = gaussian_kde(features_2d.T)
        density = kde(features_2d.T)

        # Create DataFrame
        result_df = pd.DataFrame({
            'id': sampled_df.iloc[:, id_col_idx].values,
            'density': density,
            'feat_x': features_2d[:, 0],
            'feat_y': features_2d[:, 1],
            'original_idx': np.arange(len(sampled_df))
        })

        # Select samples with lowest density
        if len(result_df) < N_SELECT:
            print(f"Warning: Not enough samples ({len(result_df)}) to select {N_SELECT}. Using all.")
            selected_df = result_df
        else:
            selected_df = result_df.sort_values(by='density').head(N_SELECT)

        print(f"Selected {len(selected_df)} samples with lowest density.")

        # K-Means clustering
        print(f"Running K-Means clustering with {N} clusters...")
        kmeans = KMeans(n_clusters=N, random_state=RANDOM_SEED)
        selected_features_2d = selected_df[['feat_x', 'feat_y']].values
        selected_df['cluster'] = kmeans.fit_predict(selected_features_2d)

        # Process clusters
        print("Processing clusters to find anchors and neighbors...")
        
        clusters_test_ids = [] # List of list of IDs for test set
        valid_cluster_ids = [] # List of cluster IDs that are actually used (non-empty)

        for cluster_id in range(N):
            cluster_samples = selected_df[selected_df['cluster'] == cluster_id]
            
            if len(cluster_samples) == 0:
                continue
            
            valid_cluster_ids.append(cluster_id)

            # Find anchor
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
            
            # Select top S samples (Anchor + S-1 neighbors)
            top_ids = sorted_ids[:S]
            clusters_test_ids.append(top_ids)

        # Generate Folds
        N_folds = len(clusters_test_ids)
        print(f"Generating {N_folds} folds...")

        train_dict = {}
        val_dict = {}
        test_dict = {}

        for i in range(N_folds):
            fold_id = str(i)
            
            # Test set
            test_ids = clusters_test_ids[i]
            
            # Current cluster ID
            current_cluster_id = valid_cluster_ids[i]
            
            # Train set: All materials in the remaining N-1 clusters
            train_df = selected_df[selected_df['cluster'] != current_cluster_id]
            
            train_ids_set = set(train_df['id'].values)
            train_ids = sorted(list(train_ids_set))
            
            # Split train_ids into train (90%) and val (10%)
            train_ids_split, val_ids_split = train_test_split(train_ids, test_size=0.1, random_state=RANDOM_SEED)
            
            train_ids_split.sort()
            val_ids_split.sort()
            
            train_dict[fold_id] = train_ids_split
            val_dict[fold_id] = val_ids_split
            test_dict[fold_id] = test_ids

        # Save to JSON
        print(f"Saving to {output_folds_dir}...")
        
        with open(os.path.join(output_folds_dir, 'train.json'), 'w') as f:
            json.dump(train_dict, f)

        with open(os.path.join(output_folds_dir, 'val.json'), 'w') as f:
            json.dump(val_dict, f)

        with open(os.path.join(output_folds_dir, 'test.json'), 'w') as f:
            json.dump(test_dict, f)

        print(f"Done processing task: {task_name} | R={R}, N={N}, S={S}")
        logging.info(f"Task {task_name} (R={R}, N={N}, S={S}) completed successfully.")

    except Exception as e:
        print(f"Error processing task {task_name} (R={R}, N={N}, S={S}): {str(e)}")
        logging.error(f"Error processing task {task_name} (R={R}, N={N}, S={S}): {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    log_file = os.path.join(parent_dir, 'work', 'log', 'process_data.log')
    setup_file_logging(log_file, level=logging.INFO)
    
    # Load parameters
    try:
        R_values = params.R_values
        N_values = params.N_values
        S_values = params.S_values
        tasks = params.tasks
    except AttributeError:
        # Fallback values if params are missing
        print("Warning: Parameters missing in common.params, using defaults/empty.")
        R_values = []
        N_values = []
        S_values = []
        tasks = []

    print("Starting batch processing...")
    
    for task_name in tasks:
        for R in R_values:
            for N in N_values:
                for S in S_values:
                    # Skip S=9, 10 when N=60
                    if N == 60 and S in [9, 10]:
                        continue
                    
                    process_data(task_name, R, N, S)
    
    print("All tasks completed.")
