import pandas as pd
import numpy as np
import yaml
import os
import argparse
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = os.path.join(BASE_DIR, 'Config/preprocess/preprocess_multivariate.yaml')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def normalize_csv(file_path, appliance_name, max_power):
    print(f"\nProcessing {appliance_name} from {file_path}...")
    print(f"Using Global Max Power from YAML: {max_power}")
    
    df = pd.read_csv(file_path)
    
    # Assume 1st column is Appliance Power (or find by name)
    # The format usually has appliance name as header
    if appliance_name in df.columns:
        col = appliance_name
    else:
        # Fallback: assume first column
        col = df.columns[0]
        print(f"Warning: Column '{appliance_name}' not found. Using first column '{col}'.")

    # Check stats before
    real_max = df[col].max()
    print(f"Original Data Max: {real_max:.2f} W")
    
    # Normalize
    df[col] = df[col] / max_power
    
    # Check stats after
    norm_max = df[col].max()
    print(f"Normalized Data Max: {norm_max:.6f} (Should be <= 1.0)")
    
    if norm_max > 1.0:
        print("⚠ WARNING: Data exceeds config max_power! values > 1.0 will exist.")

    # Save
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    save_path = os.path.join(dir_name, f"{name}_static_norm{ext}")
    
    df.to_csv(save_path, index=False)
    print(f"✅ Saved normalized file to: {save_path}")
    print("-" * 50)

def main():
    config = load_config()
    appliances_config = config['appliances']
    
    # Default search dir
    data_dir = r"C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM"
    
    print("=" * 60)
    print("STATIC NORMALIZER (Watts -> [0, 1] using YAML Config)")
    print("=" * 60)
    
    # File selection
    target_file = input(f"Enter CSV path (default: {data_dir}): ").strip()
    if not target_file:
         target_file = data_dir
         
    if os.path.isdir(target_file):
        # Look for appliance files
        for app, info in appliances_config.items():
            fname = f"{app}_multivariate.csv"
            fpath = os.path.join(target_file, fname)
            if os.path.exists(fpath):
                normalize_csv(fpath, app, info['max_power'])
    else:
        # Single file
        # Try to detect appliance name
        fname = os.path.basename(target_file).lower()
        found = False
        for app, info in appliances_config.items():
            if app in fname:
                normalize_csv(target_file, app, info['max_power'])
                found = True
                break
        if not found:
            print("❌ Could not detect appliance name from filename. Please rename file to include appliance name (e.g. washingmachine_...)")

if __name__ == "__main__":
    main()
