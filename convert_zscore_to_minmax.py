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
    
    # Logic extracted from APPLIANCE_SPECS (Algorithm 1 Logic)
    mean = specs['mean']
    std = specs['std']
    max_power = specs['max_power']
    clip_max = specs.get('max_power_clip')
    
    print(f"  Params: Mean={mean}, Std={std}, MaxPower={max_power}, ClipMax={clip_max}")
    
    # 1. Read CSV
    df = pd.read_csv(file_path)
    
    # 2. Identify Columns
    # Algorithm 1 expects: aggregate, appliance, 8 time features
    # But input might vary. We verify we have the appliance and time columns.
    
    app_col = None
    if appliance_name in df.columns:
        app_col = appliance_name
    elif 'power' in df.columns:
        app_col = 'power'
    else:
        # Fallback: assume column 1 if 10 cols, or col 0 if 9 cols
        # Typically Algorithm 1 input has Aggregate at 0, Appliance at 1
        if 'aggregate' in df.columns:
             # Find the column that is NOT aggregate and NOT time
             cols = [c for c in df.columns if c != 'aggregate' and '_sin' not in c and '_cos' not in c]
             if cols: app_col = cols[0]
    
    if not app_col:
        print(f"  ❌ Could not identify appliance column for {appliance_name}. Skipping.")
        return

    print(f"  Target Column: {app_col}")
    
    # 3. Extract Z-Score Data
    z_data = df[app_col].values
    
    # 4. Denormalize: Z-Score -> Watts (Matches algorithm1_v2 logic)
    # Check for Z-score format (Negative values)
    if z_data.min() < -0.1:
         print("  ✓ Detected Z-Score. Denormalizing to Watts (Z * Std + Mean)...")
         watts_data = z_data * std + mean
    else:
         print("  ✓ Detected Watts/Positive. Skipping Denormalization...")
         watts_data = z_data

    # 4.5 Apply Clipping if defined
    if clip_max is not None:
        print(f"  ✓ Clipping power to {clip_max}W (max_power_clip)...")
        watts_data = np.clip(watts_data, 0, clip_max)

    # Metric check
    print(f"  Watts Max: {watts_data.max():.2f}")
    
    # 5. Normalize: Watts -> MinMax (Matches algorithm1_v2 logic)
    # Logic: Watts / MaxPower
    minmax_data = watts_data / max_power
    
    # Final safety clip to [0, 1]
    minmax_data = np.clip(minmax_data, 0, 1.0)
    
    print(f"  MinMax Max: {minmax_data.max():.6f}")

    # 6. Construct Output DataFrame
    # Algorithm 1 Output Format: [Appliance, Time Features...] (Drops Aggregate)
    time_cols = [c for c in df.columns if '_sin' in c or '_cos' in c]
    
    df_out = pd.DataFrame()
    df_out[app_col] = minmax_data
    for tc in time_cols:
        df_out[tc] = df[tc]
        
    # 7. Save
    dir_name = os.path.dirname(file_path)
    save_fname = f"{appliance_name}_multivariate.csv"
    save_path = os.path.join(dir_name, save_fname)
    
    df_out.to_csv(save_path, index=False)
    print(f"  ✅ Saved to: {save_path} (Cols: {len(df_out.columns)})")

def main():
    config = load_config()
    if not config: return
    
    appliances_config = config['appliances']
    
    # Default search dir
    data_dir = r"C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM\created_data\UK_DALE"
    
    print("=" * 60)
    print("Z-SCORE -> MINMAX CONVERTER (Algorithm 1 Logic - No Filtering)")
    print("=" * 60)
    
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
