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
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return None

def convert_zscore_to_minmax(file_path, appliance_name, specs):
    print(f"\nProcessing {appliance_name} from {file_path}...")
    
    # Logic extracted from APPLIANCE_SPECS
    mean = specs['mean']
    std = specs['std']
    max_power = specs['max_power']
    
    print(f"  Params: Mean={mean}, Std={std}, MaxPower={max_power}")
    
    df = pd.read_csv(file_path)
    
    # Identify Appliance Column (Col 1 usually in the 10-col format)
    # Col 0: Aggregate, Col 1: Appliance, Cols 2-9: Time
    # Or rely on headers
    
    app_col = None
    # Method 1: Header match
    if appliance_name in df.columns:
        app_col = appliance_name
    # Method 2: Header match 'power'
    elif 'power' in df.columns:
        app_col = 'power'
    # Method 3: Index 1 (if multivariate) or Index 0 (if single)
    elif len(df.columns) >= 2:
        app_col = df.columns[1] # standard multivariate index
    else:
        app_col = df.columns[0]
        
    print(f"  Target Column: {app_col}")
    
    # Extract Z-score data
    z_data = df[app_col].values
    
    # 1. Z-score -> Watts
    # Logic: Watts = Z * Std + Mean
    watts_data = z_data * std + mean
    
    # Clip negative watts (physically impossible)
    watts_data = np.maximum(watts_data, 0)
    
    print(f"  Recovered Watts Max: {watts_data.max():.2f}")
    
    # 2. Watts -> MinMax [0, 1]
    # Logic: MinMax = Watts / MaxPower
    minmax_data = watts_data / max_power
    
    print(f"  Converted MinMax Max: {minmax_data.max():.6f}")
    
    if minmax_data.max() > 1.0:
        print(f"  ⚠ Warning: Data exceeds {max_power}W! Max is {minmax_data.max():.4f} (normalized).")
        # Optional: Clip to 1.0? usually better to keep it to notify user, but for MinMax input usually expect <=1
    
    # Update DataFrame
    df[app_col] = minmax_data
    
    # Save
    dir_name = os.path.dirname(file_path)
    save_path = os.path.join(dir_name, f"{appliance_name}_multivariate.csv")
    
    df.to_csv(save_path, index=False)
    print(f"  ✅ Saved to: {save_path}")

def main():
    config = load_config()
    if not config: return
    
    appliances_config = config['appliances']
    
    # Default search dir
    data_dir = r"C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM\created_data\UK_DALE"
    
    print("=" * 60)
    print("Z-SCORE -> MINMAX CONVERTER")
    print("Logic: (Z * Std + Mean) / MaxPower")
    print("=" * 60)
    
    # Helper to get specs
    def get_specs(app_name):
        c = appliances_config.get(app_name)
        if c: return c
        # Fallback defaults locally if not in YAML (rare)
        return {'mean': 0, 'std': 1, 'max_power': 1} # Dummy
        
    
    target_path = input(f"Enter file or directory path (default: {data_dir}): ").strip()
    if not target_path:
        target_path = data_dir
        
    if os.path.isdir(target_path):
        # Batch process known appliances
        for app in appliances_config.keys():
            # Try to find standard name
            fname = f"{app}_training_.csv"
            fpath = os.path.join(target_path, fname)
            
            if not os.path.exists(fpath):
                 # Try nested folder
                 fpath = os.path.join(target_path, app, fname)
            
            if os.path.exists(fpath):
                convert_zscore_to_minmax(fpath, app, appliances_config[app])
    else:
        # Single file
        fname = os.path.basename(target_path).lower()
        found = False
        for app in appliances_config.keys():
            if app in fname:
                convert_zscore_to_minmax(target_path, app, appliances_config[app])
                found = True
                break
        if not found:
             print("❌ Could not detect appliance name from filename.")

if __name__ == "__main__":
    main()
