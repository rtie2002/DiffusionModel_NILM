# -*- coding: utf-8 -*-
"""
NILM Data Mixing Script
Combines real and synthetic appliance data for training

Usage:
    python mix_data.py --appliance kettle --real_rows 200000 --synthetic_rows 200000
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Appliance specifications from ukdale_processing.py
APPLIANCE_SPECS = {
    'kettle': {'mean': 700, 'std': 1000, 'max_power': 3998},
    'microwave': {'mean': 500, 'std': 800, 'max_power': 3969},
    'fridge': {'mean': 200, 'std': 400, 'max_power': 3323},
    'dishwasher': {'mean': 700, 'std': 1000, 'max_power': 3964},
    'washingmachine': {'mean': 400, 'std': 700, 'max_power': 3999}
}



def load_synthetic_data(appliance_name, custom_path=None):
    """Load and prepare synthetic data from NPY file"""
    print(f"\n=== Loading Synthetic Data for {appliance_name} ===")
    
    # Load NPY file (shape: [batch, window, 1])
    if custom_path and appliance_name in custom_path:  # Only use custom path if it looks relevant or if explicitly passed for this appliance
        # Actually, simpler logic: if custom_path is provided, IT IS for this appliance context
        # But we loop over all appliances. We only want to use custom_path for the TARGET appliance.
        # So we'll handle that logic in the caller.
        npy_path = custom_path
    else:
        npy_path = f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy'
        
    if custom_path: # Verify we are loading what we think we are loading
         pass 

    # Caller handles the logic of when to pass custom_path. Here we just use it if given.
    if custom_path:
        npy_path = custom_path
    else:
        npy_path = f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy'

    print(f"Loading form: {npy_path}")
    try:
        synthetic = np.load(npy_path)
    except FileNotFoundError:
        print(f"Error: File not found: {npy_path}")
        raise

    print(f"Loaded synthetic NPY: {synthetic.shape}")
    print(f"Value range: [{synthetic.min():.4f}, {synthetic.max():.4f}]")
    
    # Flatten: [batch, window, 1] -> [total_points]
    synthetic_flat = synthetic.reshape(-1)
    print(f"Flattened to: {synthetic_flat.shape}")
    
    return synthetic_flat


def load_real_data(appliance_name, custom_path=None):
    """Load real training data"""
    print(f"\n=== Loading Real Data for {appliance_name} ===")
    
    if custom_path:
        csv_path = custom_path
    else:
        csv_path = f'NILM-main/dataset_preprocess/created_data/UK_DALE/{appliance_name}_training_.csv'
        
    print(f"Loading form: {csv_path}")
    try:
        real_data = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        raise

    print(f"Loaded real CSV: {real_data.shape}")
    print(f"Columns: 0=aggregate, 1=appliance")
    
    return real_data


def denormalize_synthetic(synthetic_normalized, appliance_name):
    """Convert synthetic data from [0,1] to watts"""
    specs = APPLIANCE_SPECS[appliance_name]
    max_power = specs['max_power']
    
    # [0, 1] -> [0, max_power] watts
    synthetic_watts = synthetic_normalized * max_power
    print(f"Denormalized to watts: [{synthetic_watts.min():.2f}W, {synthetic_watts.max():.2f}W]")
    
    return synthetic_watts


def normalize_to_zscore(watts, appliance_name):
    """Convert watts to Z-score normalization"""
    specs = APPLIANCE_SPECS[appliance_name]
    mean = specs['mean']
    std = specs['std']
    
    # Z-score: (watts - mean) / std
    normalized = (watts - mean) / std
    print(f"Z-score normalized: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    return normalized


def create_synthetic_aggregate(all_appliances_synthetic):
    """
    Create synthetic aggregate by summing all 5 appliances
    
    Args:
        all_appliances_synthetic: dict of {appliance_name: synthetic_watts_array}
    
    Returns:
        aggregate power array (watts)
    """
    print("\n=== Creating Synthetic Aggregate ===")
    
    # Find minimum length
    min_len = min(len(data) for data in all_appliances_synthetic.values())
    print(f"Using length: {min_len} (minimum across all appliances)")
    
    # Sum all appliances
    aggregate = np.zeros(min_len)
    for appliance_name, watts in all_appliances_synthetic.items():
        aggregate += watts[:min_len]
        print(f"  + {appliance_name}: mean={watts[:min_len].mean():.2f}W")
    
    print(f"Aggregate range: [{aggregate.min():.2f}W, {aggregate.max():.2f}W]")
    print(f"Aggregate mean: {aggregate.mean():.2f}W")
    
    return aggregate


def mix_data(appliance_name, real_rows, synthetic_rows, real_path=None, synthetic_path=None, output_suffix="mixed"):
    """
    Mix real and synthetic data
    
    Args:
        appliance_name: Target appliance
        real_rows: Number of real data rows to use
        synthetic_rows: Number of synthetic data rows to use  
        real_path: Optional custom path for real data
        synthetic_path: Optional custom path for synthetic data (target appliance only)
        output_suffix: Suffix for output file
    """
    print(f"\n{'='*60}")
    print(f"Mixing Data: {appliance_name}")
    print(f"Real rows: {real_rows:,} | Synthetic rows: {synthetic_rows:,}")
    print(f"{'='*60}")
    
    # 1. Load real data
    real_data = load_real_data(appliance_name, custom_path=real_path)
    real_aggregate = real_data.iloc[:real_rows, 0].values  # Column 0
    real_appliance = real_data.iloc[:real_rows, 1].values  # Column 1
    print(f"\nReal data selected: {len(real_aggregate):,} rows")
    
    # 2. Load ALL 5 appliances synthetic data
    print("\n=== Loading ALL 5 Appliances ===")
    all_synthetic = {}
    for app_name in APPLIANCE_SPECS.keys():
        # Use custom path ONLY if this is the target appliance
        path_to_use = synthetic_path if (app_name == appliance_name) else None
        
        syn_normalized = load_synthetic_data(app_name, custom_path=path_to_use)
        syn_watts = denormalize_synthetic(syn_normalized, app_name)
        all_synthetic[app_name] = syn_watts
    
    # 3. Create synthetic aggregate
    syn_aggregate_watts = create_synthetic_aggregate(all_synthetic)
    
    # 4. Get target appliance synthetic data
    # CRITICAL: Must trim to same length as aggregate to ensure alignment!
    # Aggregate was created from min_len = minimum across all 5 appliances
    # So we MUST use the same indices for the target appliance
    aggregate_len = len(syn_aggregate_watts)
    syn_appliance_watts = all_synthetic[appliance_name][:aggregate_len]
    
    print(f"\nAlignment check:")
    print(f"  Aggregate length: {len(syn_aggregate_watts):,}")
    print(f"  Appliance length: {len(syn_appliance_watts):,}")
    assert len(syn_aggregate_watts) == len(syn_appliance_watts), "Length mismatch!"
    
    # 5. Trim to requested size
    min_syn_len = len(syn_aggregate_watts)  # They're now the same length
    actual_syn_rows = min(synthetic_rows, min_syn_len)
    print(f"  Actual synthetic rows to use: {actual_syn_rows:,}")
    
    syn_aggregate_watts = syn_aggregate_watts[:actual_syn_rows]
    syn_appliance_watts = syn_appliance_watts[:actual_syn_rows]
    
    # 6. Normalize synthetic data to Z-score (aggregate uses fixed params)
    # For aggregate, use fixed normalization parameters (from ukdale_processing.py)
    if actual_syn_rows > 0:
        aggregate_mean = 522  # Fixed aggregate mean
        aggregate_std = 814   # Fixed aggregate std
        syn_aggregate_zscore = (syn_aggregate_watts - aggregate_mean) / aggregate_std
        
        # For appliance, use appliance-specific params
        syn_appliance_zscore = normalize_to_zscore(syn_appliance_watts, appliance_name)
    else:
        # No synthetic data - create empty arrays
        syn_aggregate_zscore = np.array([])
        syn_appliance_zscore = np.array([])
        print("\nNo synthetic data to normalize (synthetic_rows=0)")
    
    # 7. Combine real + synthetic (preserve as separate arrays for window shuffling)
    print("\n=== Combining Data ===")
    print(f"Real shape: {real_aggregate.shape}")
    print(f"Synthetic shape: {syn_aggregate_zscore.shape}")
    
    # 8. Shuffle by windows (preserves temporal continuity)
    window_size = 600  # NILM standard window length
    print(f"\n=== Shuffling by Windows (size={window_size}) ===")
    
    # Split real data into windows
    real_agg_windows = []
    real_app_windows = []
    for i in range(0, len(real_aggregate) - window_size + 1, window_size):
        real_agg_windows.append(real_aggregate[i:i+window_size])
        real_app_windows.append(real_appliance[i:i+window_size])
    
    # Add remainder if exists (last incomplete window)
    real_remainder = len(real_aggregate) % window_size
    if real_remainder > 0:
        real_agg_windows.append(real_aggregate[-real_remainder:])
        real_app_windows.append(real_appliance[-real_remainder:])
        print(f"  + Real remainder: {real_remainder} points")
    
    # Split synthetic data into windows (skip if no synthetic data)
    syn_agg_windows = []
    syn_app_windows = []
    if len(syn_aggregate_zscore) > 0:
        for i in range(0, len(syn_aggregate_zscore) - window_size + 1, window_size):
            syn_agg_windows.append(syn_aggregate_zscore[i:i+window_size])
            syn_app_windows.append(syn_appliance_zscore[i:i+window_size])
        
        # Add remainder if exists
        syn_remainder = len(syn_aggregate_zscore) % window_size
        if syn_remainder > 0:
            syn_agg_windows.append(syn_aggregate_zscore[-syn_remainder:])
            syn_app_windows.append(syn_appliance_zscore[-syn_remainder:])
            print(f"  + Synthetic remainder: {syn_remainder} points")
    
    print(f"Real windows: {len(real_agg_windows)}")
    print(f"Synthetic windows: {len(syn_agg_windows)}")
    
    # Combine window lists
    all_agg_windows = real_agg_windows + syn_agg_windows
    all_app_windows = real_app_windows + syn_app_windows
    total_windows = len(all_agg_windows)
    
    # Shuffle windows (not points!)
    window_indices = np.arange(total_windows)
    np.random.shuffle(window_indices)
    
    shuffled_agg_windows = [all_agg_windows[i] for i in window_indices]
    shuffled_app_windows = [all_app_windows[i] for i in window_indices]
    
    # Flatten back to 1D arrays
    mixed_aggregate = np.concatenate(shuffled_agg_windows)
    mixed_appliance = np.concatenate(shuffled_app_windows)
    
    print(f"Total windows shuffled: {total_windows}")
    print(f"Final mixed shape: {mixed_aggregate.shape}")
    print("==> Temporal continuity preserved within each window!")
    
    # 9. Save
    output_dir = Path(f'NILM-main/dataset_preprocess/created_data/UK_DALE')
    output_file = output_dir / f'{appliance_name}_training_{output_suffix}.csv'
    
    mixed_df = pd.DataFrame({
        0: mixed_aggregate,
        1: mixed_appliance
    })
    
    mixed_df.to_csv(output_file, header=False, index=False)
    
    # Calculate actual rows used (after windowing)
    actual_real_rows = len(mixed_df) - len(syn_aggregate_zscore)
    actual_syn_rows_used = len(syn_aggregate_zscore)
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS]")
    print(f"Saved to: {output_file}")
    print(f"Total rows: {len(mixed_df):,}")
    print(f"  - Real: {actual_real_rows:,} / {real_rows:,} requested ({actual_real_rows/len(mixed_df)*100:.1f}%)")
    print(f"  - Synthetic: {actual_syn_rows_used:,} / {synthetic_rows:,} requested ({actual_syn_rows_used/len(mixed_df)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mix real and synthetic NILM data')
    parser.add_argument('--appliance', type=str, required=False,
                        choices=list(APPLIANCE_SPECS.keys()),
                        help='Target appliance')
    parser.add_argument('--real_rows', type=int, default=200000,
                        help='Number of real data rows (default: 200000)')
    parser.add_argument('--synthetic_rows', type=int, default=200000,
                        help='Number of synthetic rows (default: 200000)')
    parser.add_argument('--real_path', type=str, default=None,
                        help='Path to real data CSV')
    parser.add_argument('--synthetic_path', type=str, default=None,
                        help='Path to synthetic data NPY for target appliance')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Output file suffix (default: {real}k+{syn}k)')
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments provided (specifically appliance)
    if args.appliance is None:
        print("\n=== Interactive Data Mixing Mode ===")
        print("Available appliances:", ", ".join(list(APPLIANCE_SPECS.keys())))
        
        while True:
            user_app = input("Enter target appliance name: ").strip().lower()
            if user_app in APPLIANCE_SPECS:
                args.appliance = user_app
                break
            print(f"Invalid appliance. Please choose from: {list(APPLIANCE_SPECS.keys())}")
            
        user_real = input(f"Enter real rows (default {args.real_rows}): ").strip()
        if user_real.isdigit():
            args.real_rows = int(user_real)
            
        user_syn = input(f"Enter synthetic rows (default {args.synthetic_rows}): ").strip()
        if user_syn.isdigit():
            args.synthetic_rows = int(user_syn)
            
        # Interactive prompts for inputs
        default_real_path = f'NILM-main/dataset_preprocess/created_data/UK_DALE/{args.appliance}_training_.csv'
        print(f"\nReal data path (default: {default_real_path})")
        user_real_path = input("Enter path or press Enter for default: ").strip()
        if user_real_path:
            args.real_path = user_real_path.strip('"').strip("'")
            
        default_syn_path = f'OUTPUT/{args.appliance}_512/ddpm_fake_{args.appliance}_512.npy'
        print(f"\nSynthetic data path for {args.appliance} (default: {default_syn_path})")
        user_syn_path = input("Enter path or press Enter for default: ").strip()
        if user_syn_path:
            args.synthetic_path = user_syn_path.strip('"').strip("'")

    # Auto-generate suffix if not provided
    if args.suffix is None:
        real_k = args.real_rows // 1000
        syn_k = args.synthetic_rows // 1000
        args.suffix = f'{real_k}k+{syn_k}k'
    
    # Mix data
    output_file = mix_data(
        appliance_name=args.appliance,
        real_rows=args.real_rows,
        synthetic_rows=args.synthetic_rows,
        real_path=args.real_path,
        synthetic_path=args.synthetic_path,
        output_suffix=args.suffix
    )
    
    print(f"\nNext step: Train NILM model with:")
    print(f"  python NILM-main/S2S_train.py --appliance_name {args.appliance}")
    print(f"  Then manually change the training file path to: {output_file.name}")
