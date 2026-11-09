import pandas as pd
import numpy as np
from pymatgen.core import Structure
from matminer.featurizers.structure import OrbitalFieldMatrix
import os
from tqdm import tqdm
import warnings
import logging
import datetime
import csv
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_task(base_path, task_name, dataset_type, position=0):
    # dataset_type: "MP_data" or "jarvis_data"
    # task_folder: e.g. "MP_band_gap"
    
    task_dir = os.path.join(base_path, dataset_type, task_name)
    y_value_path = os.path.join(task_dir, "Y_value.csv")
    
    
    if dataset_type == "MP_data":
        cif_folder_name = "MP_data"
    elif dataset_type == "jarvis_data":
        cif_folder_name = "jarvis_data"
    else:
        return

    cif_dir = os.path.join(task_dir, cif_folder_name)
    
    output_filename = f"OFM_{task_name}_property.csv"
    output_path = os.path.join(task_dir, output_filename)


    if not os.path.exists(y_value_path):
        return

    df_y = pd.read_csv(y_value_path)
    
    if 'material_id' not in df_y.columns:
        return
        

    ofm = OrbitalFieldMatrix(period_tag=False)

    # Check if we are resuming
    processed_ids = set()
    file_exists = os.path.exists(output_path)
    
    if file_exists:
        try:
            with open(output_path, 'r') as f:
                reader = csv.reader(f)
                header_row = next(reader, None)
                if header_row:
                    for row in reader:
                        if row:
                            processed_ids.add(row[0])
        except Exception as e:
            file_exists = False
            processed_ids = set()

    header = list(range(1026)) + ['property']

    if not file_exists or len(processed_ids) == 0:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    materials_to_process = df_y[~df_y['material_id'].isin(processed_ids)]
    
    if len(materials_to_process) == 0:
        return

    total_materials = len(df_y)
    processed_count = len(processed_ids)
    
    # Use position argument to stack bars
    for index, row in tqdm(materials_to_process.iterrows(), total=total_materials, initial=processed_count, desc=task_name, position=position):
        mid = row['material_id']
        formula = row.get('formula', '')
        
        cols = [c for c in df_y.columns if c not in ['material_id', 'formula', 'id']]
        if len(cols) > 0:
            target_col = cols[-1] 
            y_val = row[target_col]
        else:
            y_val = ''
            
        cif_path = os.path.join(cif_dir, f"{mid}.cif")
        
        feature_row = None
        
        if not os.path.exists(cif_path):
            feature_row = [np.nan] * 1024 
        else:
            try:
                structure = Structure.from_file(cif_path)
                feature_row = ofm.featurize(structure)
            except Exception as e:
                feature_row = [np.nan] * 1024
        
        output_row = [mid, formula] + list(feature_row) + [y_val]

        with open(output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(output_row)
        
    logging.info(f"Task: {task_name}. Completed processing.")

from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    parser = argparse.ArgumentParser(description="Generate 1024-dimensional OFM features.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive.")
    root_dir = str(args.root)
    
    tasks_config = {
        "MP_data": [
            "MP_band_gap", 
            "MP_bulk_modulus", 
            "MP_formation_energy", 
            "MP_metallicity", 
            "MP_refractive_index", 
            "MP_shear_modulus"
        ],
        "jarvis_data": [
            "jarvis_band_gap", 
            "jarvis_bulk_modulus", 
            "jarvis_formation_energy", 
            "jarvis_metallicity", 
            "jarvis_refractive_index", 
            "jarvis_shear_modulus"
        ]
    }

    all_tasks = []
    task_idx = 0
    for dataset, tasks in tasks_config.items():
        for task in tasks:
            all_tasks.append((root_dir, task, dataset, task_idx))
            task_idx += 1

    # Each task is assigned its own tqdm position so all progress bars stay visible.
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_task, *args) for args in all_tasks]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Task failed with error: {e}")
                print(f"Task failed with error: {e}")

if __name__ == "__main__":
    main()
