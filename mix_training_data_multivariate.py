# -*- coding: utf-8 -*-
"""
NILM Data Mixing Script (Multivariate Version)
Combines real and synthetic appliance data for training, preserving time features.

Input format: 10 columns (aggregate, appliance, 8 time features)
Output format: 10 columns (mixed_aggregate, mixed_appliance, 8 time features)

Usage:
    python mix_training_data_multivariate.py --appliance kettle --real_rows 200000 --synthetic_rows 200000
"""

import numpy as np
import pandas as pd
import argparse
import yaml
from pathlib import Path

# Appliance specifications from ukdale_processing.py
APPLIANCE_SPECS = {
    'kettle': {'mean': 700, 'std': 1000, 'max_power': 3998},
    'microwave': {'mean': 500, 'std': 800, 'max_power': 2000},
    'fridge': {'mean': 200, 'std': 400, 'max_power': 350},
    'dishwasher': {'mean': 700, 'std': 1000, 'max_power': 3964},
    'washingmachine': {'mean': 400, 'std': 700, 'max_power': 3999}
}

TIME_COLS = ['minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

def load_synthetic_data(appliance_name, custom_folder=None):
    """Load and prepare synthetic data from NPY file"""
    print(f"\n=== Loading Synthetic Data for {appliance_name} ===")
    
    if custom_folder:
        npy_path = f'{custom_folder}/ddpm_fake_{appliance_name}.npy'
    else:
        npy_path = f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy'

    print(f"Loading from: {npy_path}")
    try:
        synthetic = np.load(npy_path)
    except FileNotFoundError:
        print(f"Error: File not found: {npy_path}")
        raise

    synthetic_flat = synthetic.reshape(-1)
    print(f"Loaded synthetic NPY shape: {synthetic.shape} -> flattened: {synthetic_flat.shape}")
    return synthetic_flat


def load_real_data_multivariate(appliance_name, custom_path=None):
    """Load real training data (Multivariate CSV with headers)"""
    print(f"\n=== Loading Real Data for {appliance_name} ===")
    
    if custom_path:
        csv_path = custom_path
    else:
        csv_path = f'created_data/UK_DALE/{appliance_name}_training_.csv'
        
    print(f"Loading from: {csv_path}")
    try:
        # Multivariate files usually have headers
        df = pd.read_csv(csv_path)
        
        # Verify columns
        if len(df.columns) < 10:
            print("WARNING: Input file has fewer than 10 columns. Is this a multivariate file?")
            print(f"Columns found: {list(df.columns)}")
            
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        raise

    print(f"Loaded real CSV: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def get_real_data_stats(df, appliance_col_idx=1):
    """Calculate actual Z-score statistics from real data dataframe"""
    print(f"\n=== Calculating Real Data Statistics ===")
    
    # Use column index or name depending on what's passed
    zscore_data = df.iloc[:, appliance_col_idx].values
    
    stats = {
        'zscore_min': float(zscore_data.min()),
        'zscore_max': float(zscore_data.max()),
        'zscore_mean': float(zscore_data.mean()),
        'zscore_std': float(zscore_data.std())
    }
    
    print(f"Real data Z-score statistics (min/max/mean/std):")
    print(f"  {stats['zscore_min']:.4f} / {stats['zscore_max']:.4f} / {stats['zscore_mean']:.4f} / {stats['zscore_std']:.4f}")
    
    return stats


def convert_synthetic_to_zscore(synthetic_minmax_01, appliance_name, real_stats=None):
    """Convert synthetic data from [0,1] to Z-score using clip/linear logic"""
    print(f"\n=== Converting Synthetic {appliance_name} [0,1] -> Z-score ===")
    
    specs = APPLIANCE_SPECS[appliance_name]
    mean = specs['mean']
    std = specs['std']
    clip_power = specs['max_power']
    
    # Basic logic: treat as clipped for now (same as before) or use stats if available
    # For now ensuring basic conversion works
    
    # Using simple denorm -> zscore
    synthetic_watts = synthetic_minmax_01 * clip_power
    synthetic_zscore = (synthetic_watts - mean) / std
    
    print(f"Synthetic Z-score range: [{synthetic_zscore.min():.4f}, {synthetic_zscore.max():.4f}]")
    return synthetic_zscore


def mix_data(appliance_name, real_rows, synthetic_rows, real_path=None, output_suffix="mixed", shuffle=True):
    """
    Mix real and synthetic multivariate data
    
    Preserves all 10 columns:
    - Col 0: Aggregate (Real or Synthetic sum)
    - Col 1: Appliance (Real or Synthetic)
    - Col 2-9: Time features (Real only - reusing real time features for synthetic part)
    """
    print(f"\n{'='*60}")
    print(f"Mixing Multivariate Data: {appliance_name}")
    print(f"Real rows: {real_rows:,} | Synthetic rows: {synthetic_rows:,}")
    print(f"{'='*60}")
    
    # 1. Load Real Data (Full DataFrame)
    if real_rows > 0:
        real_df = load_real_data_multivariate(appliance_name, custom_path=real_path)
        
        # Determine cols
        # Assuming standard format: aggregate, appliance_name, time...
        agg_col = real_df.columns[0]
        app_col = real_df.columns[1]
        time_cols = real_df.columns[2:]
        
        print(f"Identified columns:")
        print(f"  Aggregate: {agg_col}")
        print(f"  Appliance: {app_col}")
        print(f"  Time cols: {list(time_cols)}")
        
        # Slice chosen rows
        real_df_subset = real_df.iloc[:real_rows].copy()
        print(f"Real subset shape: {real_df_subset.shape}")
        
        # Get stats for synthetic scaling reference
        real_stats = get_real_data_stats(real_df, appliance_col_idx=1)
        
        # Capture time features to reuse for synthetic data
        # We need time features for the synthetic part too. 
        # Strategy: Reuse real time features from the *beginning* of the file or loop them if not enough
        full_time_features = real_df.iloc[:, 2:].values
    else:
        raise ValueError("Must have some real rows to provide Time Features for multivariate format!")

    # 2. Load Synthetic Data (All 5 appliances for aggregate)
    SYNTHETIC_FOLDER = "OUTPUT/synthetic_data_for_mix_data"
    print("\n=== Loading Synthetic Appliances ===")
    
    # Helper to load and prepare one synthetic appliance
    def get_syn_zscore(name):
        data = load_synthetic_data(name, custom_folder=SYNTHETIC_FOLDER)
        return convert_synthetic_to_zscore(data, name)

    # Load all 5 in Z-score
    all_syn_zscore = {name: get_syn_zscore(name) for name in APPLIANCE_SPECS.keys()}
    
    # Min length check
    min_len = min(len(d) for d in all_syn_zscore.values())
    print(f"\nMinimum synthetic length: {min_len:,}")
    
    # 3. Create Synthetic Aggregate
    print("Creating Synthetic Aggregate...")
    syn_agg_watts = np.zeros(min_len)
    
    for name, zdata in all_syn_zscore.items():
        specs = APPLIANCE_SPECS[name]
        # Z -> Watts
        watts = zdata[:min_len] * specs['std'] + specs['mean']
        syn_agg_watts += watts
        
    # Watts -> Z-score (Aggregate)
    AGG_MEAN = 522
    AGG_STD = 814
    syn_agg_zscore = (syn_agg_watts - AGG_MEAN) / AGG_STD
    
    # 4. Prepare Synthetic Dataframe Part
    actual_syn_rows = min(synthetic_rows, min_len)
    
    if actual_syn_rows > 0:
        syn_agg_cut = syn_agg_zscore[:actual_syn_rows]
        syn_app_cut = all_syn_zscore[appliance_name][:actual_syn_rows]
        
        # Reuse time features for synthetic part
        # If we have enough unused real time features, use them. Else loop.
        # Simple approach: concatenate real time features from row 0
        
        # Note: Ideally synthetic data should have its own time, but generator doesn't output it.
        # We reuse real time features to maintain valid sin/cos sequences.
        # We can take time features from the *end* of the real data file (unused part) 
        # or just repeat.
        
        # Let's take time features immediately following the real_rows selection
        available_time_len = len(full_time_features)
        start_idx = real_rows % available_time_len
        
        # Create indices for time features
        time_indices = np.arange(start_idx, start_idx + actual_syn_rows) % available_time_len
        syn_time_features = full_time_features[time_indices]
        
        # Construct Synthetic DataFrame
        syn_data_dict = {
            agg_col: syn_agg_cut,
            app_col: syn_app_cut
        }
        
        # Add time columns
        for i, t_col in enumerate(time_cols):
            syn_data_dict[t_col] = syn_time_features[:, i]
            
        syn_df_subset = pd.DataFrame(syn_data_dict)
        print(f"Created Synthetic DataFrame: {syn_df_subset.shape}")
        
    else:
        syn_df_subset = pd.DataFrame(columns=real_df.columns)

    # 5. Combine and Shuffle
    print("\n=== Mixing and Shuffling ===")
    
    # Window-based shuffling
    window_size = 6000 # 6000 points ~ 1.5h roughly, standard chunk
    
    # We will simply concatenate then output.
    # But wait, original script did window shuffling.
    # Let's stick to simple dataframe concatenation then shuffle ROW chunks?
    # No, window shuffling is important for training stability.
    
    # Convert DFs to list of window-dataframes
    def df_to_windows(df, w_size):
        wins = []
        for i in range(0, len(df), w_size):
            wins.append(df.iloc[i : i+w_size])
        return wins

    real_windows = df_to_windows(real_df_subset, window_size)
    syn_windows = df_to_windows(syn_df_subset, window_size) if not syn_df_subset.empty else []
    
    all_windows = real_windows + syn_windows
    print(f"Total windows (size {window_size}): {len(all_windows)} (Real: {len(real_windows)}, Syn: {len(syn_windows)})")
    
    if shuffle:
        import random
        random.shuffle(all_windows)
        print("Windows shuffled.")
    
    final_df = pd.concat(all_windows, ignore_index=True)
    print(f"Final Combined Shape: {final_df.shape}")

    # 6. Save Output
    output_dir = Path(f'created_data/UK_DALE')
    if not output_dir.exists():
        # Fallback path if created_data root differs
        output_dir = Path('NILM-main/dataset_preprocess/created_data/UK_DALE')
        
    output_file = output_dir / f'{appliance_name}_training_{output_suffix}.csv'
    
    # Save with header=True for multivariate
    final_df.to_csv(output_file, index=False, header=True)
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS]")
    print(f"Saved to: {output_file}")
    print(f"Format: Multivariate CSV (10 columns)")
    print(f"Total rows: {len(final_df):,}")
    print(f"{'='*60}\n")
    
    return output_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mix real and synthetic NILM data (Multivariate)')
    parser.add_argument('--appliance', type=str, required=True, choices=list(APPLIANCE_SPECS.keys()))
    parser.add_argument('--real_rows', type=int, default=200000)
    parser.add_argument('--synthetic_rows', type=int, default=200000)
    parser.add_argument('--real_path', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.suffix is None:
        real_k = args.real_rows // 1000
        syn_k = args.synthetic_rows // 1000
        args.suffix = f'{real_k}k+{syn_k}k'

    mix_data(args.appliance, args.real_rows, args.synthetic_rows, args.real_path, args.suffix)

