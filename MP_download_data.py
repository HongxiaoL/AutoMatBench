import os
import json
import pandas as pd
from mp_api.client import MPRester
from work.common.sort_utils import natural_sort_key

def process_task(mpr, task_config):
    name = task_config["name"]
    endpoint = task_config["endpoint"]
    fields = task_config["fields"]
    val_func = task_config["val_func"]
    
    print(f"Processing task: {name}...", flush=True)
    
    # Create directories
    base_dir = f"MP_{name}"
    data_dir = os.path.join(base_dir, "MP_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Fetch data
    # Limit to 30000 (30 chunks * 1000) to handle large datasets
    print(f"  Fetching data from {endpoint}...", flush=True)
    
    try:
        # Get the search method dynamically
        search_method = getattr(mpr.materials, endpoint).search
        
        # We fetch up to 30000 items.
        docs = search_method(
            fields=fields,
            num_chunks=30,
            chunk_size=1000
        )
    except Exception as e:
        print(f"  Error fetching {name}: {e}", flush=True)
        return

    count = len(docs)
    print(f"  Fetched {count} documents.", flush=True)
    
    if count == 0:
        print(f"  No documents found for {name}. Skipping.", flush=True)
        return

    # Process documents
    data_list = []
    
    print(f"  Saving CIF files and extracting values...", flush=True)
    for doc in docs:
        mp_id = str(doc.material_id)
        
        # Save CIF if structure is available
        if hasattr(doc, "structure") and doc.structure:
            try:
                cif_path = os.path.join(data_dir, f"{mp_id}.cif")
                doc.structure.to(filename=cif_path)
            except Exception as e:
                print(f"    Error saving CIF for {mp_id}: {e}")
            
        # Extract value
        try:
            val = val_func(doc)
        except Exception as e:
            # print(f"    Error extracting value for {mp_id}: {e}")
            val = None
        
        data_list.append({
            "material_id": mp_id,
            "formula": doc.formula_pretty,
            name: val
        })
        
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
tasks = [
    {
        "name": "bulk_modulus",
        "endpoint": "elasticity",
        "fields": ["material_id", "formula_pretty", "bulk_modulus", "structure"],
        # Handle both object (with vrh attr) and dict (with vrh key) and None
        "val_func": lambda d: getattr(d.bulk_modulus, "vrh", None) if hasattr(d, "bulk_modulus") and d.bulk_modulus and not isinstance(d.bulk_modulus, dict) else (d.bulk_modulus.get("vrh") if hasattr(d, "bulk_modulus") and isinstance(d.bulk_modulus, dict) else None)
    },
    {
        "name": "shear_modulus",
        "endpoint": "elasticity",
        "fields": ["material_id", "formula_pretty", "shear_modulus", "structure"],
        "val_func": lambda d: getattr(d.shear_modulus, "vrh", None) if hasattr(d, "shear_modulus") and d.shear_modulus and not isinstance(d.shear_modulus, dict) else (d.shear_modulus.get("vrh") if hasattr(d, "shear_modulus") and isinstance(d.shear_modulus, dict) else None)
    },
    {
        "name": "band_gap",
        "endpoint": "summary",
        "fields": ["material_id", "formula_pretty", "band_gap", "structure"],
        "val_func": lambda d: d.band_gap
    },
    {
        "name": "metallicity",
        "endpoint": "summary",
        "fields": ["material_id", "formula_pretty", "is_metal", "structure"],
        "val_func": lambda d: d.is_metal
    },
    {
        "name": "refractive_index",
        "endpoint": "summary", # Changed from dielectric to summary to access structure
        "fields": ["material_id", "formula_pretty", "n", "structure"],
        "val_func": lambda d: d.n
    },
    {
        "name": "formation_energy",
        "endpoint": "summary",
        "fields": ["material_id", "formula_pretty", "formation_energy_per_atom", "structure"],
        "val_func": lambda d: d.formation_energy_per_atom
    }
]

if __name__ == "__main__":
    print("Starting multi-task data download...", flush=True)
    with MPRester() as mpr:
        for task in tasks:
            process_task(mpr, task)
    print("All tasks completed.", flush=True)
