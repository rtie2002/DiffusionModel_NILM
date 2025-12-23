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



def load_synthetic_data(appliance_name, custom_folder=None):
    """Load and prepare synthetic data from NPY file
    
    Args:
        appliance_name: Name of the appliance
        custom_folder: If provided, look for 'ddpm_fake_{appliance_name}.npy' in this folder
                      Otherwise use default 'OUTPUT/{appliance_name}_512/' path
    """
    print(f"\n=== Loading Synthetic Data for {appliance_name} ===")
    
    # Determine NPY path
    if custom_folder:
        # Load from custom folder with standard naming
        npy_path = f'{custom_folder}/ddpm_fake_{appliance_name}.npy'
    else:
        # Default location
        npy_path = f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy'

    print(f"Loading from: {npy_path}")
    try:
        synthetic = np.load(npy_path)
    except FileNotFoundError:
        print(f"Error: File not found: {npy_path}")
        print(f"Expected filename: ddpm_fake_{appliance_name}.npy")
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


def get_real_data_stats(appliance_name):
    """Calculate actual Z-score statistics from real UK-DALE training data
    
    Returns:
        dict with 'zscore_min', 'zscore_max', 'zscore_mean', 'zscore_std'
    """
    csv_path = f'NILM-main/dataset_preprocess/created_data/UK_DALE/{appliance_name}_training_.csv'
    
    print(f"\n=== Calculating Real Data Statistics ===")
    print(f"Loading from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        print(f"Warning: Real data not found at {csv_path}")
        print(f"Falling back to theoretical Z-score range based on max_power_clip")
        return None
    
    # Column 1 is appliance data (already in Z-score)
    zscore_data = df.iloc[:, 1].values
    
    stats = {
        'zscore_min': float(zscore_data.min()),
        'zscore_max': float(zscore_data.max()),
        'zscore_mean': float(zscore_data.mean()),
        'zscore_std': float(zscore_data.std())
    }
    
    print(f"Real data Z-score statistics:")
    print(f"  Min:  {stats['zscore_min']:.4f}")
    print(f"  Max:  {stats['zscore_max']:.4f}")
    print(f"  Mean: {stats['zscore_mean']:.4f}")
    print(f"  Std:  {stats['zscore_std']:.4f}")
    
    return stats


def convert_synthetic_to_zscore(synthetic_minmax_01, appliance_name, real_stats=None):
    """Convert synthetic data from [0,1] to Z-score
    
    Strategy:
    - Clipped appliances (clip_power != real_max_power): Use clip_power for denormalization
    - Non-clipped appliances (clip_power == real_max_power): Use linear transformation to real Z-score range
    
    Args:
        synthetic_minmax_01: Synthetic data in [0,1] format
        appliance_name: Name of appliance
        real_stats: Stats from get_real_data_stats() or None
    
    Returns:
        Synthetic data in Z-score format
    """
    print(f"\n=== Converting Synthetic [0,1] -> Z-score ===")
    
    # Get appliance specs
    specs = APPLIANCE_SPECS[appliance_name]
    mean = specs['mean']
    std = specs['std']
    clip_power = specs['max_power']  # This is clip_power from APPLIANCE_SPECS
    
    # Load YAML to check if appliance was clipped
    config_path = 'Config/applainces_configuration.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        appliance_config = config[appliance_name]
        real_max_power = appliance_config.get('real_max_power')
        was_clipped = (clip_power != real_max_power)
    except Exception as e:
        print(f"Warning: Could not load YAML config: {e}")
        was_clipped = False  # Assume not clipped if can't load config
        real_max_power = clip_power
    
    if was_clipped:
        # CLIPPED APPLIANCE: Use clip_power method
        print(f"[CLIPPED] {appliance_name}: {clip_power}W clip < {real_max_power}W real_max")
        print(f"  Using clip_power method: [0,1] → [0,{clip_power}W] → Z-score")
        print(f"  Z-score params: mean={mean}W, std={std}W")
        
        # [0,1] → [0, clip_power] watts → Z-score
        synthetic_watts = synthetic_minmax_01 * clip_power
        synthetic_zscore = (synthetic_watts - mean) / std
        
    else:
        # NON-CLIPPED APPLIANCE: Use linear transformation to match real Z-score range
        if real_stats is not None:
            print(f"[NOT CLIPPED] {appliance_name}: {clip_power}W == {real_max_power}W")
            print(f"  Using linear transformation to real Z-score range")
            
            zscore_min = real_stats['zscore_min']
            zscore_max = real_stats['zscore_max']
            zscore_range = zscore_max - zscore_min
            
            # Linear transformation: [0,1] → [zscore_min, zscore_max]
            synthetic_zscore = synthetic_minmax_01 * zscore_range + zscore_min
            
            print(f"  Mapped [0,1] → Z-score range [{zscore_min:.4f}, {zscore_max:.4f}]")
        else:
            # Fallback: use clip_power method
            print(f"[NOT CLIPPED] {appliance_name}: No real_stats, using clip_power method")
            synthetic_watts = synthetic_minmax_01 * clip_power
            synthetic_zscore = (synthetic_watts - mean) / std
    
    print(f"Synthetic Z-score range: [{synthetic_zscore.min():.4f}, {synthetic_zscore.max():.4f}]")
    print(f"Synthetic Z-score mean:  {synthetic_zscore.mean():.4f}")
    print(f"Synthetic Z-score std:   {synthetic_zscore.std():.4f}")
    
    return synthetic_zscore


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


def mix_data(appliance_name, real_rows, synthetic_rows, real_path=None, output_suffix="mixed", shuffle=True):
    """
    Mix real and synthetic data
    
    Args:
        appliance_name: Target appliance
        real_rows: Number of real data rows to use
        synthetic_rows: Number of synthetic data rows to use  
        real_path: Optional custom path for real data
        output_suffix: Suffix for output file
        shuffle: Whether to shuffle windows (default: True)
    
    Note:
        Synthetic data is ALWAYS loaded from: OUTPUT/synthetic_data_for_mix_data/
        This ensures aggregate power is calculated from a consistent source.
    """
    print(f"\n{'='*60}")
    print(f"Mixing Data: {appliance_name}")
    print(f"Real rows: {real_rows:,} | Synthetic rows: {synthetic_rows:,}")
    print(f"{'='*60}")
    
    # 1. Load real data and calculate statistics (skip if real_rows=0)
    if real_rows > 0:
        real_data = load_real_data(appliance_name, custom_path=real_path)
        real_aggregate = real_data.iloc[:real_rows, 0].values  # Column 0
        real_appliance = real_data.iloc[:real_rows, 1].values  # Column 1
        print(f"\nReal data selected: {len(real_aggregate):,} rows")
        
        # Calculate real data Z-score statistics for synthetic conversion
        real_stats = get_real_data_stats(appliance_name)
    else:
        # No real data - create empty arrays
        real_aggregate = np.array([])
        real_appliance = np.array([])
        real_stats = None  # Will use fallback conversion
        print(f"\nNo real data (real_rows=0)")
        print(f"Synthetic data will use fallback conversion (may not match train.csv scale)")
    
    # 2. Load ALL 5 appliances synthetic data from HARDCODED source
    # ALWAYS use OUTPUT/synthetic_data_for_mix_data for aggregate calculation
    SYNTHETIC_FOLDER = "OUTPUT/synthetic_data_for_mix_data"
    print("\n=== Loading ALL 5 Appliances ===")
    print(f"** Loading ALL appliances from: {SYNTHETIC_FOLDER} (HARDCODED) **")
    
    # Get real stats for ALL appliances (for aggregate calculation)
    all_real_stats = {}
    for app_name in APPLIANCE_SPECS.keys():
        app_stats = get_real_data_stats(app_name)
        all_real_stats[app_name] = app_stats
    
    # Load and convert all synthetic appliances to Z-score
    all_synthetic_zscore = {}
    for app_name in APPLIANCE_SPECS.keys():
        syn_minmax = load_synthetic_data(app_name, custom_folder=SYNTHETIC_FOLDER)
        syn_zscore = convert_synthetic_to_zscore(syn_minmax, app_name, all_real_stats[app_name])
        all_synthetic_zscore[app_name] = syn_zscore
    
    # 3. Create synthetic aggregate from Z-score data
    # NOTE: All synthetic appliances are now in Z-score format
    print("\n=== Creating Synthetic Aggregate (Z-score) ===")
    
    # Find minimum length
    min_len = min(len(data) for data in all_synthetic_zscore.values())
    print(f"Using length: {min_len:,} (minimum across all appliances)")
    
    # Convert each appliance from Z-score back to Watts for summing
    all_synthetic_watts_for_agg = {}
    for app_name, zscore_data in all_synthetic_zscore.items():
        specs = APPLIANCE_SPECS[app_name]
        watts = zscore_data[:min_len] * specs['std'] + specs['mean']
        all_synthetic_watts_for_agg[app_name] = watts
        print(f"  {app_name}: mean={watts.mean():.2f}W")
    
    # Sum all appliances in watts
    syn_aggregate_watts = np.zeros(min_len)
    for app_name, watts in all_synthetic_watts_for_agg.items():
        syn_aggregate_watts += watts
    
    print(f"Aggregate range: [{syn_aggregate_watts.min():.2f}W, {syn_aggregate_watts.max():.2f}W]")
    print(f"Aggregate mean: {syn_aggregate_watts.mean():.2f}W")
    
    # Convert aggregate to Z-score
    aggregate_mean = 522  # Fixed aggregate mean
    aggregate_std = 814   # Fixed aggregate std
    syn_aggregate_zscore = (syn_aggregate_watts - aggregate_mean) / aggregate_std
    
    # 4. Get target appliance synthetic data (already in Z-score)
    syn_appliance_zscore = all_synthetic_zscore[appliance_name][:min_len]
    
    print(f"\nAlignment check:")
    print(f"  Aggregate length: {len(syn_aggregate_zscore):,}")
    print(f"  Appliance length: {len(syn_appliance_zscore):,}")
    assert len(syn_aggregate_zscore) == len(syn_appliance_zscore), "Length mismatch!"
    
    # 5. Trim to requested size
    actual_syn_rows = min(synthetic_rows, min_len)
    print(f"  Actual synthetic rows to use: {actual_syn_rows:,}")
    
    syn_aggregate_zscore = syn_aggregate_zscore[:actual_syn_rows]
    syn_appliance_zscore = syn_appliance_zscore[:actual_syn_rows]
    
    # 6. Synthetic data is now in Z-score (no further conversion needed)
    if actual_syn_rows > 0:
        print(f"\n=== Synthetic Data Already in Z-score ===")
        print(f"Aggregate Z-score: [{syn_aggregate_zscore.min():.4f}, {syn_aggregate_zscore.max():.4f}]")
        print(f"Appliance Z-score: [{syn_appliance_zscore.min():.4f}, {syn_appliance_zscore.max():.4f}]")
        
        # DIAGNOSTIC: Compare with real data scale if available
        if real_stats is not None:
            print("\n=== DIAGNOSTIC: Comparing Synthetic vs Real Z-score Scale ===")
            print(f"Real data Z-score:")
            print(f"  Range: [{real_stats['zscore_min']:.4f}, {real_stats['zscore_max']:.4f}]")
            print(f"  Mean:  {real_stats['zscore_mean']:.4f}")
            print(f"  Std:   {real_stats['zscore_std']:.4f}")
            
            print(f"\nSynthetic data Z-score:")
            print(f"  Range: [{syn_appliance_zscore.min():.4f}, {syn_appliance_zscore.max():.4f}]")
            print(f"  Mean:  {syn_appliance_zscore.mean():.4f}")
            print(f"  Std:   {syn_appliance_zscore.std():.4f}")
            
            # Check if ranges match (within tolerance)
            range_match = abs(syn_appliance_zscore.min() - real_stats['zscore_min']) < 0.1 and \
                         abs(syn_appliance_zscore.max() - real_stats['zscore_max']) < 0.1
            
            if range_match:
                print(f"\n  [PASS] Synthetic Z-score range matches real data!")
            else:
                print(f"\n  [WARNING] Synthetic Z-score range differs from real data")
                print(f"  This may cause scale mismatch in mixed training data")
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
    window_size = 6000  # NILM standard window length
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
    
    # Shuffle windows (not points!) - optional
    if shuffle:
        print(f"\n==> Shuffling {total_windows} windows...")
        window_indices = np.arange(total_windows)
        np.random.shuffle(window_indices)
        
        shuffled_agg_windows = [all_agg_windows[i] for i in window_indices]
        shuffled_app_windows = [all_app_windows[i] for i in window_indices]
    else:
        print(f"\n==> Keeping {total_windows} windows in sequential order (no shuffling)")
        shuffled_agg_windows = all_agg_windows
        shuffled_app_windows = all_app_windows
    
    # Flatten back to 1D arrays
    mixed_aggregate = np.concatenate(shuffled_agg_windows)
    mixed_appliance = np.concatenate(shuffled_app_windows)
    
    print(f"Total windows: {total_windows}")
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

    parser.add_argument('--suffix', type=str, default=None,
                        help='Output file suffix (default: {real}k+{syn}k)')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Disable window shuffling (keep sequential order)')
    
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
            
        print("\n[INFO] Synthetic data will be loaded from: OUTPUT/synthetic_data_for_mix_data (hardcoded)")
        
        # Ask about shuffling
        print("\n=== Window Shuffling ===")
        print("Shuffling mixes real and synthetic windows randomly.")
        print("No shuffling keeps windows in sequential order (real first, then synthetic).")
        user_shuffle = input("Shuffle windows? (y/n, default=y): ").strip().lower()
        if user_shuffle in ['n', 'no']:
            args.no_shuffle = True
            print("  -> Windows will be kept in sequential order")
        else:
            args.no_shuffle = False
            print("  -> Windows will be shuffled randomly")

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
        output_suffix=args.suffix,
        shuffle=not args.no_shuffle  # Shuffle by default, disable if --no-shuffle is set
    )
    
    print(f"\nNext step: Train NILM model with:")
    print(f"  python NILM-main/S2S_train.py --appliance_name {args.appliance}")
    print(f"  Then manually change the training file path to: {output_file.name}")
