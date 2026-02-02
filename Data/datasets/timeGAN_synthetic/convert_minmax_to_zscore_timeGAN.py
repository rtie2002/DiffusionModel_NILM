import numpy as np
import argparse
import os
import yaml
import pandas as pd
from tqdm import tqdm
import sys

"""
Script to Convert TimeGAN Output (Already Rescaled) to Z-score CSV
Specifically for TimeGAN output in: Data/datasets/timeGAN_synthetic/

NOTE: TimeGAN output (.npy) is ALREADY in original scale (Watts, [-1,1] Time).
      It performs renormalization internally before saving.
      This script ONLY applies Z-score standardization to the Power channel.

Logic:
1. Load Data (Watts, Time).
2. Identify appliance.
3. Power: Apply Z-Score:  Z = (Watts - Mean) / Std.
4. Time: Keep as is ([-1, 1]).
5. Save as CSV.

Required Config: Config/preprocess/preprocess_multivariate.yaml
"""

# Hardcode project root for reliability
PROJECT_ROOT = r"C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "Config", "preprocess", "preprocess_multivariate.yaml")

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config from {CONFIG_PATH}: {e}")
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
        print(f"Warning: Appliance '{appliance}' not found in config. Using defaults.")
        return {'mean': 0, 'std': 1, 'max_power': 3000} 

def convert_file(file_path, output_dir, config):
    filename = os.path.basename(file_path)
    
    # 1. Detect Appliance
    appliance = None
    known_appliances = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
    
    for app in known_appliances:
        if app in filename.lower():
            appliance = app
            break
    
    if not appliance:
        print(f"Skipping unknown appliance file: {filename}")
        return

    print(f"Processing {appliance} -> {filename}")
    
    # 2. Get Params
    params = get_params(appliance, config)
    print(f"  Params: Mean={params['mean']}, Std={params['std']}")

    # 3. Load Data
    try:
        data = np.load(file_path, allow_pickle=True) 
        if data.dtype == 'O' or isinstance(data, list):
             data = np.stack(data)
    except Exception as e:
        print(f"  Error loading: {e}")
        return

    if len(data.shape) != 3:
        print(f"  Error: Expected 3D array, got {data.shape}")
        return

    # 4. Conversion Logic
    
    # Channel 0 is Power (Already in Watts)
    watts = data[:, :, 0]
    
    # Clip negative watts just in case (GAN noise)
    watts = np.maximum(watts, 0)
    
    # Z-Score Transformation
    zscore_power = (watts - params['mean']) / params['std']
    
    # Time Columns (Already [-1, 1])
    # Just clip them to be safe
    if data.shape[2] > 1:
        time_feats = data[:, :, 1:]
        time_feats = np.clip(time_feats, -1.0, 1.0)
    
    print(f"  Power Stats: Min={watts.min():.2f}, Max={watts.max():.2f}")
    
    # 5. Reconstruct Multivariate Matrix
    new_data = np.zeros_like(data)
    new_data[:, :, 0] = zscore_power
    if data.shape[2] > 1:
        new_data[:, :, 1:] = time_feats
    
    # Flatten to 2D for CSV
    flat_data = new_data.reshape(-1, new_data.shape[2])
    
    # 6. Save as CSV
    cols = ['power', 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    
    if flat_data.shape[1] != len(cols):
        cols = ['power'] + [f'feat_{i}' for i in range(flat_data.shape[1]-1)]

    df = pd.DataFrame(flat_data, columns=cols)
    out_name = f"{appliance}_multivariate.csv" 
    out_path = os.path.join(output_dir, out_name)
    
    df.to_csv(out_path, index=False)
    print(f"  Saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    default_input = os.path.join(PROJECT_ROOT, "Data", "datasets", "timeGAN_synthetic")
    parser.add_argument('--input_dir', type=str, default=default_input)
    
    args = parser.parse_args()
    output_dir = os.path.join(args.input_dir, "zscore_converted")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = load_config()
    if not config:
        return

    if os.path.exists(args.input_dir):
        files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
        for f in tqdm(files):
            convert_file(os.path.join(args.input_dir, f), output_dir, config)
    else:
        print(f"Directory not found: {args.input_dir}")

if __name__ == '__main__':
    main()
