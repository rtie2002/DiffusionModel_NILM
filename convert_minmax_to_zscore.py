import numpy as np
import argparse
import os
import yaml
from pathlib import Path
from tqdm import tqdm

"""
Script to Convert MinMax [0,1] Multivariate NPY back to Z-score NPY
Specifically for files in: synthetic_data_multivariate/

Logic:
1. Load [0,1] data.
2. Identify appliance.
3. INVERSE MinMax: Watts = Data * MaxPower
4. APPLY Z-Score:  Z = (Watts - Mean) / Std

Required Config: Config/preprocess/preprocess_multivariate.yaml
"""

# Default Params as Fallback


def load_config():
    config_path = 'Config/preprocess/preprocess_multivariate.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config ({e}), using defaults.")
        return None

def get_params(appliance, config):
    if config and 'appliances' in config and appliance in config['appliances']:
        app_conf = config['appliances'][appliance]
        max_p = app_conf.get('max_power') 
        return {
            'mean': app_conf['mean'],
            'std': app_conf['std'],
            'max_power': max_p
        }
    else:
        raise ValueError(f"Error: Appliance '{appliance}' not found in config or config failed to load.")

def convert_file(file_path, output_dir, config):
    filename = os.path.basename(file_path)
    
    # 1. Detect Appliance
    appliance = None
    if config and 'appliances' in config:
        for app in config['appliances'].keys():
            if app in filename.lower():
                appliance = app
                break
    
    if not appliance:
        print(f"Skipping unknown appliance file: {filename}")
        return

    print(f"Processing {appliance} -> {filename}")
    
    # 2. Get Params
    params = get_params(appliance, config)
    print(f"  Params: Max={params['max_power']}, Mean={params['mean']}, Std={params['std']}")

    # 3. Load Data
    try:
        data = np.load(file_path) # (N, 512, 9)
    except Exception as e:
        print(f"  Error loading: {e}")
        return

    # Check validity (should be [0,1])
    power_ch = data[:, :, 0]
    if power_ch.min() < -0.1 or power_ch.max() > 1.2:
        print(f"  WARNING: Data range {power_ch.min():.2f}-{power_ch.max():.2f} suggests NOT MinMax. Skipping.")
        return

    # 4. Conversion Logic
    # Step A: MinMax [0,1] -> Watts
    # Watts = Value * MaxPower
    watts = power_ch * params['max_power']
    
    # Step B: Watts -> Z-score
    # Z = (Watts - Mean) / Std
    zscore_power = (watts - params['mean']) / params['std']
    
    # 5. Reconstruct Multivariate Array
    new_data = data.copy()
    new_data[:, :, 0] = zscore_power
    
    # Flatten to 2D for CSV: (N*512, 9)
    flat_data = new_data.reshape(-1, 9)
    
    # 6. Save as CSV
    # Get time columns from config or default
    time_cols = ['minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    if config and 'mixing' in config and 'time_cols' in config['mixing']:
        time_cols = config['mixing']['time_cols']
        
    cols = ['power'] + time_cols
    
    # Ensure column count matches
    if flat_data.shape[1] != len(cols):
        print(f"  WARNING: Column count mismatch. Data has {flat_data.shape[1]}, standard is {len(cols)}. using generic headers.")
        cols = ['power'] + [f'time_feat_{i}' for i in range(flat_data.shape[1]-1)]

    import pandas as pd
    df = pd.DataFrame(flat_data, columns=cols)
    
    out_name = filename.replace('.npy', '_zscore.csv')
    out_path = os.path.join(output_dir, out_name)
    df.to_csv(out_path, index=False)
    
    print(f"  Saved to: {out_path}")
    print(f"  New Range: {zscore_power.min():.2f} to {zscore_power.max():.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='synthetic_data_multivariate')
    parser.add_argument('--output_dir', type=str, default='synthetic_data_multivariate/zscore_converted')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = load_config()
    
    # List files
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy') and 'zscore' not in f]
    
    print(f"Found {len(files)} files in {args.input_dir}")
    
    for f in tqdm(files):
        convert_file(os.path.join(args.input_dir, f), args.output_dir, config)

if __name__ == '__main__':
    main()
