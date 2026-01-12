# -*- coding: utf-8 -*-
"""
NILM Data Mixing Script (Multivariate Version - Real Power Output)
Combines real and synthetic appliance data for training, preserving time features.

Input format: Real data in Z-score, Synthetic data in [0,1]
Output format: 10 columns in REAL POWER (Watts):
  - aggregate (Watts)
  - appliance (Watts)  
  - 8 time features (sin/cos encoded, unchanged)

Usage:
    python mix_training_data_multivariate_convert_to_realPower.py --appliance kettle --real_rows 200000 --synthetic_rows 200000
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

def load_synthetic_data(appliance_name, custom_folder=None, return_full=False):
    """Load and prepare synthetic data from NPY file
    
    Args:
        appliance_name: Name of the appliance
        custom_folder: Custom folder path for synthetic data
        return_full: If True, return full multivariate data (N, 512, 9)
                     If False, return only appliance power column flattened
    
    Returns:
        If return_full=False: 1D array of appliance power values
        If return_full=True: 3D array (N, 512, 9) with power + time features
    """
    print(f"\n=== Loading Synthetic Data for {appliance_name} ===")
    
    if custom_folder:
        # Use filename pattern for files in synthetic_data_multivariate
        npy_path = f'{custom_folder}/ddpm_fake_{appliance_name}_multivariate.npy'
    else:
        npy_path = f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy'

    print(f"Loading from: {npy_path}")
    try:
        synthetic = np.load(npy_path)
    except FileNotFoundError:
        print(f"Error: File not found: {npy_path}")
        raise

    print(f"Loaded synthetic NPY shape: {synthetic.shape}")
    
    if return_full:
        # Return full multivariate data (N, 512, 9)
        print(f"Returning full multivariate data: {synthetic.shape}")
        return synthetic
    else:
        # Extract only appliance power column (first column, index 0)
        # Shape: (N, 512, 9) -> (N, 512, 1) -> flatten to (N*512,)
        if len(synthetic.shape) == 3 and synthetic.shape[2] >= 1:
            synthetic_power = synthetic[:, :, 0]  # Extract first column (power)
            synthetic_flat = synthetic_power.reshape(-1)
            print(f"Extracted power column and flattened: {synthetic_flat.shape}")
        else:
            # Fallback for old format (already just power)
            synthetic_flat = synthetic.reshape(-1)
            print(f"Flattened (old format): {synthetic_flat.shape}")
        
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


def get_all_appliances_stats():
    """Get real data statistics for all appliances
    
    Returns:
        dict: {appliance_name: stats_dict} for all appliances
    """
    print(f"\n=== Loading Real Data Stats for All Appliances ===")
    
    all_stats = {}
    for app_name in APPLIANCE_SPECS.keys():
        csv_path = f'created_data/UK_DALE/{app_name}_training_.csv'
        
        try:
            df = pd.read_csv(csv_path)
            # Column 1 is appliance data (already in Z-score)
            zscore_data = df.iloc[:, 1].values
            
            stats = {
                'zscore_min': float(zscore_data.min()),
                'zscore_max': float(zscore_data.max()),
                'zscore_mean': float(zscore_data.mean()),
                'zscore_std': float(zscore_data.std())
            }
            
            all_stats[app_name] = stats
            print(f"  {app_name}: [{stats['zscore_min']:.4f}, {stats['zscore_max']:.4f}]")
            
        except FileNotFoundError:
            print(f"  {app_name}: WARNING - File not found, will use fallback conversion")
            all_stats[app_name] = None
    
    return all_stats



def convert_synthetic_to_watts(synthetic_minmax_01, appliance_name, real_stats=None):
    """Convert synthetic data from [0,1] to Real Power (Watts)
    
    Strategy:
    - Clipped appliances (clip_power != real_max_power): Use clip_power for denormalization
    - Non-clipped appliances (clip_power == real_max_power): Use linear transformation to real power range
    
    Args:
        synthetic_minmax_01: Synthetic data in [0,1] format
        appliance_name: Name of appliance
        real_stats: Stats from get_real_data_stats() or None (should contain real power stats)
    
    Returns:
        Synthetic data in Watts
    """
    print(f"\n=== Converting Synthetic [0,1] -> Watts ===")
    
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
        # [0,1] -> [0, clip_power] Watts directly
        print(f"[CLIPPED] {appliance_name}: {clip_power}W clip < {real_max_power}W real_max")
        print(f"  Using clip_power method: [0,1] -> [0,{clip_power}W]")
        
        synthetic_watts = synthetic_minmax_01 * clip_power
        
    else:
        # NON-CLIPPED APPLIANCE: Use linear transformation to match real power range
        if real_stats is not None:
            print(f"[NOT CLIPPED] {appliance_name}: {clip_power}W == {real_max_power}W")
            print(f"  Using linear transformation to real power range")
            
            # Convert real Z-score stats to Watts
            # Real data is in Z-score, so we need to convert the min/max to Watts
            zscore_min = real_stats['zscore_min']
            zscore_max = real_stats['zscore_max']
            
            # Z-score -> Watts conversion
            real_watts_min = zscore_min * std + mean
            real_watts_max = zscore_max * std + mean
            real_watts_range = real_watts_max - real_watts_min
            
            # Linear transformation: [0,1] -> [real_watts_min, real_watts_max]
            synthetic_watts = synthetic_minmax_01 * real_watts_range + real_watts_min
            
            print(f"  Real power range (from Z-score): [{real_watts_min:.2f}W, {real_watts_max:.2f}W]")
            print(f"  Mapped [0,1] -> [{real_watts_min:.2f}W, {real_watts_max:.2f}W]")
        else:
            # Fallback: use clip_power method
            print(f"[NOT CLIPPED] {appliance_name}: No real_stats, using clip_power method")
            synthetic_watts = synthetic_minmax_01 * clip_power
    
    print(f"Synthetic Watts range: [{synthetic_watts.min():.2f}, {synthetic_watts.max():.2f}]")
    print(f"Synthetic Watts mean:  {synthetic_watts.mean():.2f}W")
    print(f"Synthetic Watts std:   {synthetic_watts.std():.2f}W")
    
    return synthetic_watts


def mix_data(appliance_name, real_rows, synthetic_rows, real_path=None, output_suffix="mixed", shuffle=True):
    """
    Mix real and synthetic multivariate data
    
    OUTPUT FORMAT: Real Power (Watts)
    - Col 0: Aggregate (Watts)
    - Col 1: Appliance (Watts)
    - Col 2-9: Time features (sin/cos encoded, unchanged)
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
        
        # Get stats BEFORE converting to Watts (while still in Z-score)
        print(f"\n=== Getting Real Data Stats (Z-score) ===")
        real_stats = get_real_data_stats(real_df, appliance_col_idx=1)
        
        # Convert real data from Z-score to Watts
        print(f"\n=== Converting Real Data: Z-score -> Watts ===")
        AGG_MEAN = 522
        AGG_STD = 814
        app_mean = APPLIANCE_SPECS[appliance_name]['mean']
        app_std = APPLIANCE_SPECS[appliance_name]['std']
        
        # Convert aggregate: Z-score -> Watts
        real_df_subset.iloc[:, 0] = real_df_subset.iloc[:, 0] * AGG_STD + AGG_MEAN
        # Clip to ensure no negative power values
        real_df_subset.iloc[:, 0] = real_df_subset.iloc[:, 0].clip(lower=0)
        print(f"Aggregate converted to Watts: [{real_df_subset.iloc[:, 0].min():.2f}, {real_df_subset.iloc[:, 0].max():.2f}]")
        
        # Convert appliance: Z-score -> Watts
        real_df_subset.iloc[:, 1] = real_df_subset.iloc[:, 1] * app_std + app_mean
        # Clip to ensure no negative power values
        real_df_subset.iloc[:, 1] = real_df_subset.iloc[:, 1].clip(lower=0)
        print(f"Appliance converted to Watts: [{real_df_subset.iloc[:, 1].min():.2f}, {real_df_subset.iloc[:, 1].max():.2f}]")
        
        # CRITICAL: Ensure appliance power never exceeds aggregate power
        print(f"\n=== Validating Real Data: Appliance vs Aggregate Power ===")
        real_agg_vals = real_df_subset.iloc[:, 0].values
        real_app_vals = real_df_subset.iloc[:, 1].values
        violations = np.sum(real_app_vals > real_agg_vals)
        if violations > 0:
            print(f"WARNING: Found {violations:,} points where appliance > aggregate ({violations/len(real_app_vals)*100:.2f}%)")
            print(f"  Max violation: {(real_app_vals - real_agg_vals).max():.2f}W")
            print(f"  Clipping appliance power to aggregate power...")
            real_df_subset.iloc[:, 1] = np.minimum(real_app_vals, real_agg_vals)
            print(f"  ✓ All violations fixed")
        else:
            print(f"✓ No violations found - all appliance values <= aggregate")
        
        # Capture time features to reuse for synthetic data
        # We need time features for the synthetic part too. 
        # Strategy: Reuse real time features from the *beginning* of the file or loop them if not enough
        full_time_features = real_df.iloc[:, 2:].values
    else:
        raise ValueError("Must have some real rows to provide Time Features for multivariate format!")

    # 2. Load Synthetic Data (All 5 appliances for aggregate)
    SYNTHETIC_FOLDER = "synthetic_data_multivariate"
    print("\n=== Loading Synthetic Appliances ===")
    
    # Helper to load and prepare one synthetic appliance in Watts
    def get_syn_watts(name):
        data = load_synthetic_data(name, custom_folder=SYNTHETIC_FOLDER, return_full=False)
        # Pass real_stats for proper scaling (use target appliance's stats for all)
        return convert_synthetic_to_watts(data, name, real_stats)

    # Load all 5 appliances in Watts (power only for aggregate calculation)
    all_syn_watts = {name: get_syn_watts(name) for name in APPLIANCE_SPECS.keys()}
    
    # Load FULL multivariate data for the TARGET appliance (to get time features)
    print(f"\n=== Loading Full Multivariate Data for Target Appliance: {appliance_name} ===")
    target_syn_full = load_synthetic_data(appliance_name, custom_folder=SYNTHETIC_FOLDER, return_full=True)
    print(f"Target appliance full shape: {target_syn_full.shape}")
    
    # Extract time features from synthetic data (columns 1-8, assuming column 0 is power)
    # Shape: (N, 512, 9) -> extract columns 1-8 -> flatten to (N*512, 8)
    syn_time_features_3d = target_syn_full[:, :, 1:]  # Columns 1-8 are time features
    syn_time_features_flat = syn_time_features_3d.reshape(-1, 8)
    print(f"Extracted synthetic time features shape: {syn_time_features_flat.shape}")
    
    # Min length check
    min_len = min(len(d) for d in all_syn_watts.values())
    print(f"\nMinimum synthetic power length: {min_len:,}")
    print(f"Synthetic time features length: {len(syn_time_features_flat):,}")
    
    # Use the minimum of power data and time features
    min_len = min(min_len, len(syn_time_features_flat))
    print(f"Final minimum length: {min_len:,}")
    
    # 3. Create Synthetic Aggregate (already in Watts)
    print("Creating Synthetic Aggregate...")
    syn_agg_watts = np.zeros(min_len)
    
    for name, watts_data in all_syn_watts.items():
        # Already in Watts, just sum
        syn_agg_watts += watts_data[:min_len]
    
    # Clip to ensure no negative power values
    syn_agg_watts = np.clip(syn_agg_watts, 0, None)
    
    print(f"Synthetic aggregate range: [{syn_agg_watts.min():.2f}, {syn_agg_watts.max():.2f}] Watts")
    
    # 4. Prepare Synthetic Dataframe Part (in Watts)
    actual_syn_rows = min(synthetic_rows, min_len)
    
    if actual_syn_rows > 0:
        syn_agg_cut = syn_agg_watts[:actual_syn_rows]  # Already in Watts
        syn_app_cut = all_syn_watts[appliance_name][:actual_syn_rows]  # Already in Watts
        
        # CRITICAL: Ensure appliance power never exceeds aggregate power
        # This is physically impossible in real NILM data
        print(f"\n=== Validating Appliance vs Aggregate Power ===")
        violations_before = np.sum(syn_app_cut > syn_agg_cut)
        if violations_before > 0:
            print(f"WARNING: Found {violations_before:,} points where appliance > aggregate ({violations_before/len(syn_app_cut)*100:.2f}%)")
            print(f"  Max violation: {(syn_app_cut - syn_agg_cut).max():.2f}W")
            print(f"  Clipping appliance power to aggregate power...")
            syn_app_cut = np.minimum(syn_app_cut, syn_agg_cut)
            print(f"  ✓ All violations fixed")
        else:
            print(f"✓ No violations found - all appliance values <= aggregate")
        
        # Time features: Use the synthetic data's own time features
        syn_time_features = syn_time_features_flat[:actual_syn_rows]
        print(f"Using synthetic time features: {syn_time_features.shape}")
        
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
    window_size = 600 # 6000 points ~ 1.5h roughly, standard chunk
    
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
    print(f"Format: Multivariate CSV (10 columns) - REAL POWER (Watts)")
    print(f"  - Columns 0-1: Aggregate & Appliance (Watts)")
    print(f"  - Columns 2-9: Time features (sin/cos encoded)")
    print(f"Total rows: {len(final_df):,}")
    print(f"{'='*60}\n")
    
    return output_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mix real and synthetic NILM data (Multivariate)')
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
        print("\n=== Interactive Multivariate Data Mixing Mode ===")
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
        default_real_path = f'created_data/UK_DALE/{args.appliance}_training_.csv'
        print(f"\nReal data path (default: {default_real_path})")
        user_real_path = input("Enter path or press Enter for default: ").strip()
        if user_real_path:
            args.real_path = user_real_path.strip('"').strip("'")
            
        print("\n[INFO] Synthetic data will be loaded from: synthetic_data_multivariate (hardcoded)")
        
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
        args.suffix = f'{real_k}k+{syn_k}k_realPower'
    
    # Mix data
    mix_data(
        appliance_name=args.appliance,
        real_rows=args.real_rows,
        synthetic_rows=args.synthetic_rows,
        real_path=args.real_path,
        output_suffix=args.suffix,
        shuffle=not args.no_shuffle  # Shuffle by default, disable if --no-shuffle is set
    )

