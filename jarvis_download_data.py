
import os
import json
import pandas as pd
from jarvis.db.figshare import data as jarvis_data
from work.common.sort_utils import natural_sort_key

def process_task(data_3d, task_config):
    name = task_config["name"]
    prop_key = task_config["prop_key"]
    val_func = task_config.get("val_func", lambda x: x)
    
    print(f"Processing task: {name}...", flush=True)
    
    # Create directories
    base_dir = f"jarvis_{name}"
    data_dir = os.path.join(base_dir, "jarvis_data")
    os.makedirs(data_dir, exist_ok=True)
    
    data_list = []
    
    count = 0
    limit = 30000
    
    print(f"  Filtering and saving data...", flush=True)
    
    for d in data_3d:
        # Extract value
        try:
            if name == "refractive_index":
                # Special case for refractive index calculation from multiple keys
                val = val_func(d)
            else:
                val = d.get(prop_key)
                if val is None or val == 'na':
                    continue
                val = val_func(val)
        except:
            continue
            
        if val is None or val == 'na':
            continue
            
        jid = d['jid']
        
        data_list.append({
            "material_id": jid,
            "formula": d.get('formula', ''),
            name: val
        })

        # Save structure to jarvis_data directory
        from jarvis.core.atoms import Atoms
        atoms = Atoms.from_dict(d['atoms'])
        atoms.write_cif(os.path.join(data_dir, f"{jid}.cif"))
        
        count += 1
        if count >= limit:
            break
            
    # Save CSV
    if data_list:
        df = pd.DataFrame(data_list)
        
        # Sort by natural order of material_id
        df['sort_key'] = df['material_id'].apply(natural_sort_key)
        df = df.sort_values('sort_key').drop(columns=['sort_key'])
        
        csv_path = os.path.join(base_dir, "Y_value.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved Y_value.csv with {len(df)} entries.", flush=True)
    else:
        print("  No valid data extracted.", flush=True)
        
    print(f"  Done with {name}.\n", flush=True)

# Define tasks configuration
# JARVIS keys mapping based on documentation/exploration
tasks = [
    {
        "name": "bulk_modulus",
        "prop_key": "bulk_modulus_kv", # VRH Bulk Modulus
    },
    {
        "name": "shear_modulus",
        "prop_key": "shear_modulus_gv", # VRH Shear Modulus
    },
    {
        "name": "band_gap",
        "prop_key": "optb88vdw_bandgap", # or mbj_bandgap, using optb88vdw as standard DFT
    },
    {
        "name": "metallicity",
        "prop_key": "optb88vdw_bandgap", 
        # JARVIS doesn't have direct is_metal, infer from bandgap < 0.01 or similar?
        # Or check if there is a metal key. 
        # Usually if bandgap is 0 it is metal.
        "val_func": lambda x: True if float(x) <= 0.001 else False
    },
    {
        "name": "refractive_index",
        "prop_key": "n-solar", 
        "val_func": lambda d: ((float(d.get('mepsx', 0)) + float(d.get('mepsy', 0)) + float(d.get('mepsz', 0))) / 3.0)**0.5
    },
    {
        "name": "formation_energy",
        "prop_key": "formation_energy_peratom",
    }
]

if __name__ == "__main__":
    print("Loading JARVIS DFT 3D dataset...", flush=True)
    # Load dft_3d dataset
    data_3d = jarvis_data(dataset='dft_3d')
    print(f"Loaded {len(data_3d)} entries.", flush=True)
    
    for task in tasks:
        process_task(data_3d, task)
        
    print("All tasks completed.", flush=True)
