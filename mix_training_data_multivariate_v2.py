# -*- coding: utf-8 -*-
"""
NILM Data Mixing Script v2 (Background Injection Version)
Logic: 
1. Load Synthetic Target Appliance (ỹ_target)
2. Extract REAL Background Power (X_bg = X_total - Y_target) from quiet periods
3. Construct Synthetic Aggregate: X_syn = X_bg + ỹ_target

Usage:
    python mix_training_data_multivariate_v2.py --appliance kettle --real_rows 200000 --synthetic_rows 200000
"""

import numpy as np
import pandas as pd
import argparse
import yaml
import random
from pathlib import Path

# Load configuration
CONFIG_PATH = 'Config/preprocess/preprocess_multivariate.yaml'

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except:
        return None

CONFIG = load_config()

# Standard NILM Stats (Fallback)
APPLIANCE_SPECS = {
    'kettle': {'mean': 700, 'std': 1000, 'max_power': 3998},
    'microwave': {'mean': 500, 'std': 800, 'max_power': 3969},
    'fridge': {'mean': 200, 'std': 400, 'max_power': 350},
    'dishwasher': {'mean': 700, 'std': 1000, 'max_power': 3964},
    'washingmachine': {'mean': 400, 'std': 700, 'max_power': 3999}
}

if CONFIG:
    for app_name, app_data in CONFIG['appliances'].items():
        APPLIANCE_SPECS[app_name] = {
            'mean': app_data['mean'], 'std': app_data['std'], 'max_power': app_data['max_power']
        }
    AGG_MEAN = CONFIG['normalization']['aggregate_mean']
    AGG_STD = CONFIG['normalization']['aggregate_std']
else:
    AGG_MEAN, AGG_STD = 522, 814

def load_synthetic_appliance(appliance_name, synthetic_dir):
    """Load synthetic [0,1] power and convert to Watts"""
    npy_path = Path(synthetic_dir) / f'ddpm_fake_{appliance_name}_multivariate.npy'
    if not npy_path.exists():
        # Try fallback path
        npy_path = Path(f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy')
    
    print(f"Loading synthetic data from: {npy_path}")
    data = np.load(npy_path)
    # Extract power column (index 0) and flatten
    power_01 = data[:, :, 0].reshape(-1)
    
    # Convert to Watts
    spec = APPLIANCE_SPECS[appliance_name]
    power_watts = power_01 * spec['max_power']
    return power_watts, data[:, :, 1:].reshape(-1, 8) # Return Watts, TimeFeatures

def extract_real_background_pool(appliance_name, real_path, window_size=600):
    """
    Extract periods where the target appliance is OFF to create a background power pool.
    Background = Total Aggregate - Target Appliance
    """
    print(f"Extracting Real Background Pool for {appliance_name}...")
    df = pd.read_csv(real_path)
    
    # Columns: 0=Agg, 1=Appliance
    agg_z = df.iloc[:, 0].values
    app_z = df.iloc[:, 1].values
    
    # Agg Z -> Watts
    agg_w = agg_z * AGG_STD + AGG_MEAN
    # App Z -> Watts
    spec = APPLIANCE_SPECS[appliance_name]
    app_w = app_z * spec['std'] + spec['mean']
    
    # Background Power (Everything except target appliance)
    bg_w = agg_w - app_w
    bg_w = np.maximum(bg_w, 0) # Physical constraint
    
    # -------------------------------------------------------------
    # Use Algorithm 1 logic to perfectly identify true OFF periods
    # -------------------------------------------------------------
    import os
    if CONFIG and appliance_name in CONFIG['appliances']:
        x_threshold = CONFIG['appliances'][appliance_name]['on_power_threshold']
        l_window = CONFIG['algorithm1']['window_length']
    else:
        x_threshold = 15.0
        l_window = 100
        
    power_sequence = app_w.copy()
    
    # 1. Identify ON points
    is_on_event = power_sequence >= x_threshold
    
    # 2. Expand ON points by l_window (Algorithm 1 logic)
    # Using np.convolve to dilate True values backwards and forwards by l_window
    kernel_size = l_window * 2 + 1
    kernel = np.ones(kernel_size)
    expanded_on = np.convolve(is_on_event.astype(float), kernel, mode='same') > 0
    
    # 3. True OFF periods are the inverse of expanded ON periods
    is_off_period = ~expanded_on
    
    # Slice into windows of background power where appliance is 100% safely OFF
    pool = []
    for i in range(0, len(bg_w) - window_size, window_size):
        if np.all(is_off_period[i : i + window_size]): 
            pool.append(bg_w[i : i + window_size])
    
    # FALLBACK: 如果连 Algorithm 1 过滤后也找不到绝对干净的窗口
    if len(pool) == 0:
        print(f"  WARNING: 找不到符合 Algorithm 1 绝对强力过滤的背景窗口，正在选择该段期间【最大峰值功率最低】的窗口...")
        scored_windows = []
        for i in range(0, len(bg_w) - window_size, window_size):
            max_app_power = np.max(app_w[i : i + window_size])
            scored_windows.append((max_app_power, bg_w[i : i + window_size]))
            
        # 按照目标电器最大功率从小到大排序，拿最安静的前 50 组当背景
        scored_windows.sort(key=lambda x: x[0])
        pool = [w[1] for w in scored_windows[:50]]
        print(f"  Fallback selected max appliances power in background: {[round(w[0],2) for w in scored_windows[:5]]} Watts")

    print(f"  Extracted {len(pool)} background windows (size {window_size}) using Algorithm 1 filtering")
    return pool

def mix_data_v2(appliance_name, real_rows, synthetic_rows, real_path=None, suffix="v2", shuffle=True, window_size=600, full_shuffle=False):
    print(f"\n{'='*60}\nNILM Mixed Dataset Construction v2 (Background Injection)\n{'='*60}")
    
    # 1. Load Real Data
    if real_path is None:
        real_path = f'created_data/UK_DALE/{appliance_name}_training_.csv'
    real_df = pd.read_csv(real_path)
    real_subset = real_df.iloc[:real_rows].copy()
    
    # 2. Extract Background Pool from Real Data (OFF periods)
    bg_pool = extract_real_background_pool(appliance_name, real_path, window_size)
    
    # 3. Load Synthetic Appliance Power
    synthetic_dir = CONFIG['paths']['synthetic_dir'] if CONFIG else 'synthetic_data_multivariate'
    syn_app_w, syn_time_features = load_synthetic_appliance(appliance_name, synthetic_dir)
    
    # 4. Construct Synthetic Sequence using Background Injection
    actual_syn_rows = min(synthetic_rows, len(syn_app_w)) if synthetic_rows > 0 else 0
    
    if actual_syn_rows > 0:
        print(f"Constructing Synthetic Section ({actual_syn_rows} rows)...")
        num_windows = (actual_syn_rows + window_size - 1) // window_size
        
        syn_agg_final = []
        syn_app_final = []
        syn_time_final = []
        
        spec = APPLIANCE_SPECS[appliance_name]
        
        for i in range(num_windows):
            start = i * window_size
            end = min(start + window_size, len(syn_app_w))
            if start >= len(syn_app_w) or len(syn_agg_final) >= actual_syn_rows: break
            
            curr_win_size = end - start
            ỹ_target = syn_app_w[start:end]
            x_bg = random.choice(bg_pool)[:curr_win_size]
            x_syn = x_bg + ỹ_target
            
            syn_agg_final.extend((x_syn - AGG_MEAN) / AGG_STD)
            syn_app_final.extend((ỹ_target - spec['mean']) / spec['std'])
            syn_time_final.append(syn_time_features[start:end])

        syn_time_final = np.concatenate(syn_time_final, axis=0)
        
        # Create Synthetic DataFrame
        syn_df = pd.DataFrame(syn_time_final, columns=real_df.columns[2:])
        syn_df.insert(0, real_df.columns[0], syn_agg_final[:len(syn_df)])
        syn_df.insert(1, real_df.columns[1], syn_app_final[:len(syn_df)])
        # Truncate to exact rows
        syn_df = syn_df.iloc[:actual_syn_rows]
    else:
        print("Skipping synthetic construction (Baseline mode).")
        syn_df = pd.DataFrame(columns=real_df.columns)
    
    # 5. Combine and Window Shuffle
    print("Finalizing Hybrid Dataset...")
    def to_windows(df, w):
        return [df.iloc[i : i+w] for i in range(0, len(df), w)]
    
    real_windows = to_windows(real_subset, window_size)
    syn_windows = to_windows(syn_df, window_size)
    
    if full_shuffle:
        # Scenario: Full Shuffle (Real and Synthetic mixed and randomized)
        print("Applying FULL SHUFFLE (Real + Synthetic)...")
        all_windows = real_windows + syn_windows
        random.shuffle(all_windows)
    elif shuffle:
        # Scenario: Mixed Shuffle (Real kept in order, Synthetic windows randomized)
        print("Applying PARTIAL SHUFFLE (Synthetic only)...")
        random.shuffle(syn_windows)
        
        # Keep real windows in order, randomly inject synthetic windows
        all_windows = list(real_windows)
        for syn_win in syn_windows:
            insert_loc = random.randint(0, len(all_windows))
            all_windows.insert(insert_loc, syn_win)
    else:
        print("No shuffle (Ordered mode)...")
        all_windows = real_windows + syn_windows
    
    final_df = pd.concat(all_windows, ignore_index=True)
    
    # 6. Save
    out_dir = Path(f'created_data/UK_DALE/{appliance_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{appliance_name}_training_{suffix}.csv'
    final_df.to_csv(out_file, index=False)
    
    print(f"DONE! Saved to: {out_file}")
    print(f"Total Rows: {len(final_df)} (Real: {len(real_subset)}, Synthetic: {len(syn_df)})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    parser.add_argument('--real_rows', type=int, default=200000)
    parser.add_argument('--synthetic_rows', type=int, default=200000)
    parser.add_argument('--suffix', type=str, default="200k+200k_bg_v2")
    parser.add_argument('--shuffle', action='store_true', help='Enable window shuffling (synthetic only)')
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle')
    parser.set_defaults(shuffle=True)
    parser.add_argument('--full-shuffle', action='store_true', help='Shuffle BOTH real and synthetic windows together')
    parser.add_argument('--window_size', type=int, default=600, help='Window size for slicing and shuffling')
    args = parser.parse_args()
    
    mix_data_v2(
        appliance_name=args.appliance, 
        real_rows=args.real_rows, 
        synthetic_rows=args.synthetic_rows, 
        suffix=args.suffix,
        shuffle=args.shuffle,
        window_size=args.window_size,
        full_shuffle=args.full_shuffle
    )
