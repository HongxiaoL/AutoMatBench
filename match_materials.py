
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# File paths
jarvis_y_path = '/jarvis_data/jarvis_refractive_index/Y_value.csv'
mp_y_path = '/MP_data/MP_refractive_index/Y_value.csv'
jarvis_ofm_path = '/jarvis_data/jarvis_refractive_index/OFM_jarvis_refractive_index_property.csv'
mp_ofm_path = '/MP_data/MP_refractive_index/OFM_MP_refractive_index_property.csv'

def load_data():
    print("Loading data...")
    # Load Y values
    jarvis_y = pd.read_csv(jarvis_y_path)
    mp_y = pd.read_csv(mp_y_path)
    
    # Load OFM features
    # Jarvis OFM: No header. First column ID, second Formula, last Property.
    jarvis_ofm = pd.read_csv(jarvis_ofm_path, header=None)
    
    # MP OFM: Has header. First column (name '0') ID, second (name '1') Formula, last 'property'.
    mp_ofm = pd.read_csv(mp_ofm_path)
    
    print(f"Jarvis Y shape: {jarvis_y.shape}")
    print(f"MP Y shape: {mp_y.shape}")
    print(f"Jarvis OFM shape: {jarvis_ofm.shape}")
    print(f"MP OFM shape: {mp_ofm.shape}")
    
    return jarvis_y, mp_y, jarvis_ofm, mp_ofm

def process_matches(jarvis_y, mp_y, jarvis_ofm, mp_ofm):
    # Prepare OFM feature arrays
    # Extract features: columns 2 to second-to-last (inclusive) -> indices 2:-1
    # Check dimensions
    j_feats = jarvis_ofm.iloc[:, 2:-1].values
    m_feats = mp_ofm.iloc[:, 2:-1].values
    
    if j_feats.shape[1] != m_feats.shape[1]:
        print(f"Warning: Feature dimensions mismatch! Jarvis: {j_feats.shape[1]}, MP: {m_feats.shape[1]}")
    
    # Map ID to features for quick access
    # Jarvis ID is at column 0
    j_id_to_idx = {uid: i for i, uid in enumerate(jarvis_ofm.iloc[:, 0])}
    # MP ID is at column 0 (which is named '0')
    m_id_to_idx = {uid: i for i, uid in enumerate(mp_ofm.iloc[:, 0])}
    
    # Group IDs by formula
    # Use Y value files for formula mapping as they are cleaner/explicit
    # Ensure formulas are stripped/normalized if needed. Assuming they are clean.
    # Remove NaN formulas
    jarvis_y = jarvis_y.dropna(subset=['formula'])
    mp_y = mp_y.dropna(subset=['formula'])

    # Map ID to Band Gap
    j_bg_map = dict(zip(jarvis_y['material_id'], jarvis_y['refractive_index']))
    m_bg_map = dict(zip(mp_y['material_id'], mp_y['refractive_index']))

    j_formula_map = {}
    for _, row in jarvis_y.iterrows():
        f = row['formula']
        if f not in j_formula_map: j_formula_map[f] = []
        j_formula_map[f].append(row['material_id'])
        
    m_formula_map = {}
    for _, row in mp_y.iterrows():
        f = row['formula']
        if f not in m_formula_map: m_formula_map[f] = []
        m_formula_map[f].append(row['material_id'])
        
    # Find common formulas
    common_formulas = set(j_formula_map.keys()) & set(m_formula_map.keys())
    print(f"Found {len(common_formulas)} common formulas.")
    
    matches = []
    
    for formula in common_formulas:
        j_ids = j_formula_map[formula]
        m_ids = m_formula_map[formula]
        
        best_match = None
        min_dist = float('inf')
        
        # Iterate over all pairs
        for j_id in j_ids:
            for m_id in m_ids:
                if j_id not in j_id_to_idx:
                    # print(f"Warning: Jarvis ID {j_id} not found in OFM data")
                    continue
                if m_id not in m_id_to_idx:
                    # print(f"Warning: MP ID {m_id} not found in OFM data")
                    continue
                
                j_idx = j_id_to_idx[j_id]
                m_idx = m_id_to_idx[m_id]
                
                feat_j = j_feats[j_idx]
                feat_m = m_feats[m_idx]
                
                dist = euclidean(feat_j, feat_m)
                
                if dist < min_dist:
                    min_dist = dist
                    j_bg = j_bg_map.get(j_id, np.nan)
                    m_bg = m_bg_map.get(m_id, np.nan)
                    best_match = (j_id, m_id, formula, dist, j_bg, m_bg)
        
        if best_match:
            matches.append(best_match)
            
    return matches

def main():
    jarvis_y, mp_y, jarvis_ofm, mp_ofm = load_data()
    matches = process_matches(jarvis_y, mp_y, jarvis_ofm, mp_ofm)
    
    # Create DataFrame
    df_matches = pd.DataFrame(matches, columns=['jarvis_id', 'mp_id', 'formula', 'ofm_distance', 'jarvis_refractive_index', 'mp_refractive_index'])
    
    output_file = 'matched_materials.csv'
    df_matches.to_csv(output_file, index=False)
    print(f"Saved {len(matches)} matches to {output_file}")
    print(df_matches.head())

if __name__ == '__main__':
    main()
