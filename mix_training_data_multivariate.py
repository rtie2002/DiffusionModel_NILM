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

# Load configuration
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'Config/preprocess/preprocess_multivariate.yaml'

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {CONFIG_PATH}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return None

CONFIG = load_config()

# Extract global specs from config
if CONFIG:
    APPLIANCE_SPECS = {}
    for app_name, app_data in CONFIG['appliances'].items():
        APPLIANCE_SPECS[app_name] = {
            'mean': app_data['mean'],
            'std': app_data['std'],
            'max_power': app_data['max_power']
        }
    
    TIME_COLS = CONFIG['mixing'].get('time_cols', [])
    if not TIME_COLS:
        print("WARNING: 'time_cols' not found in config, using defaults.")
        TIME_COLS = ['minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
        
    SYNTHETIC_DIR = CONFIG['paths'].get('synthetic_dir', 'synthetic_data_multivariate')
    # Resolve to absolute path
    if not Path(SYNTHETIC_DIR).is_absolute():
        SYNTHETIC_DIR = (BASE_DIR / SYNTHETIC_DIR).resolve()
    
else:
    # Fallback to hardcoded defaults if config fails
    print("WARNING: Using hardcoded defaults.")
    APPLIANCE_SPECS = {
        'kettle': {'mean': 700, 'std': 1000, 'max_power': 3998},
        'microwave': {'mean': 500, 'std': 800, 'max_power': 3969},
        'fridge': {'mean': 200, 'std': 400, 'max_power': 350},
        'dishwasher': {'mean': 700, 'std': 1000, 'max_power': 3964},
        'washingmachine': {'mean': 400, 'std': 700, 'max_power': 3999}
    }
    TIME_COLS = ['minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    SYNTHETIC_DIR = 'synthetic_data_multivariate'

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
        # Ensure custom_folder is a Path object or string handled correctly
        folder_path = Path(custom_folder)
        if not folder_path.is_absolute():
             folder_path = (BASE_DIR / folder_path).resolve()
        
        npy_path = folder_path / f'ddpm_fake_{appliance_name}_multivariate.npy'
    else:
        npy_path = BASE_DIR / f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy'

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
        csv_path = Path(custom_path)
        if not csv_path.is_absolute():
            csv_path = (BASE_DIR / csv_path).resolve()
    else:
        csv_path = BASE_DIR / f'created_data/UK_DALE/{appliance_name}_training_.csv'
        
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
        csv_path = BASE_DIR / f'created_data/UK_DALE/{app_name}_training_.csv'
        
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
    config_path = BASE_DIR / 'Config/applainces_configuration.yaml'
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
        print(f"  Using clip_power method: [0,1] -> [0,{clip_power}W] -> Z-score")
        print(f"  Z-score params: mean={mean}W, std={std}W")
        
        # [0,1] -> [0, clip_power] watts -> Z-score
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
            
            # Linear transformation: [0,1] -> [zscore_min, zscore_max]
            synthetic_zscore = synthetic_minmax_01 * zscore_range + zscore_min
            
            print(f"  Mapped [0,1] -> Z-score range [{zscore_min:.4f}, {zscore_max:.4f}]")
        else:
            # Fallback: use clip_power method
            print(f"[NOT CLIPPED] {appliance_name}: No real_stats, using clip_power method")
            synthetic_watts = synthetic_minmax_01 * clip_power
            synthetic_zscore = (synthetic_watts - mean) / std
    
    print(f"Synthetic Z-score range: [{synthetic_zscore.min():.4f}, {synthetic_zscore.max():.4f}]")
    print(f"Synthetic Z-score mean:  {synthetic_zscore.mean():.4f}")
    print(f"Synthetic Z-score std:   {synthetic_zscore.std():.4f}")
    
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
    # Use global SYNTHETIC_DIR loaded from config
    print(f"\n=== Loading Synthetic Appliances from {SYNTHETIC_DIR} ===")
    
    # Get real stats for ALL appliances (for proper conversion)
    all_real_stats = get_all_appliances_stats()
    
    # Helper to load and prepare one synthetic appliance (power only)
    def get_syn_zscore(name):
        data = load_synthetic_data(name, custom_folder=SYNTHETIC_DIR, return_full=False)
        # Pass the real stats for this appliance
        return convert_synthetic_to_zscore(data, name, all_real_stats[name])

    # Load all 5 appliances in Z-score (power only for aggregate calculation)
    all_syn_zscore = {name: get_syn_zscore(name) for name in APPLIANCE_SPECS.keys()}
    
    # Load FULL multivariate data for the TARGET appliance (to get time features)
    print(f"\n=== Loading Full Multivariate Data for Target Appliance: {appliance_name} ===")
    target_syn_full = load_synthetic_data(appliance_name, custom_folder=SYNTHETIC_DIR, return_full=True)
    print(f"Target appliance full shape: {target_syn_full.shape}")
    
    # Extract time features from synthetic data (columns 1-8, assuming column 0 is power)
    # Shape: (N, 512, 9) -> extract columns 1-8 -> flatten to (N*512, 8)
    syn_time_features_3d = target_syn_full[:, :, 1:]  # Columns 1-8 are time features
    syn_time_features_flat = syn_time_features_3d.reshape(-1, 8)
    print(f"Extracted synthetic time features shape: {syn_time_features_flat.shape}")
    
    # Min length check
    min_len = min(len(d) for d in all_syn_zscore.values())
    print(f"\nMinimum synthetic power length: {min_len:,}")
    print(f"Synthetic time features length: {len(syn_time_features_flat):,}")
    
    # Use the minimum of power data and time features
    min_len = min(min_len, len(syn_time_features_flat))
    print(f"Final minimum length: {min_len:,}")
    
    # 3. Create Synthetic Aggregate
    print("Creating Synthetic Aggregate...")
    syn_agg_watts = np.zeros(min_len)
    
    for name, zdata in all_syn_zscore.items():
        specs = APPLIANCE_SPECS[name]
        # Z -> Watts
        watts = zdata[:min_len] * specs['std'] + specs['mean']
        syn_agg_watts += watts
        
    # Watts -> Z-score (Aggregate)
    # Watts -> Z-score (Aggregate)
    if CONFIG:
        AGG_MEAN = CONFIG['normalization']['aggregate_mean']
        AGG_STD = CONFIG['normalization']['aggregate_std']
    else:
        AGG_MEAN = 522
        AGG_STD = 814
        
    syn_agg_zscore = (syn_agg_watts - AGG_MEAN) / AGG_STD
    
    # 4. Prepare Synthetic Dataframe Part
    actual_syn_rows = min(synthetic_rows, min_len)
    
    if actual_syn_rows > 0:
        syn_agg_cut = syn_agg_zscore[:actual_syn_rows]
        syn_app_cut = all_syn_zscore[appliance_name][:actual_syn_rows]
        
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
    # Window-based shuffling
    if CONFIG:
        window_size = CONFIG['mixing'].get('window_size', 600)
    else:
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
    output_dir = BASE_DIR / 'created_data/UK_DALE'
    if not output_dir.exists():
        # Fallback path if created_data root differs
        output_dir = BASE_DIR / 'NILM-main/dataset_preprocess/created_data/UK_DALE'
        
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
            
        print(f"\n[INFO] Synthetic data will be loaded from: {SYNTHETIC_DIR}")
        
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
    
    # Determine shuffle logic:
    # 1. Default to config value (or True if config missing)
    # 2. CLI flag --no-shuffle overrides to False
    config_shuffle_default = True
    if CONFIG and 'mixing' in CONFIG:
        config_shuffle_default = CONFIG['mixing'].get('shuffle', True)
    
    # If user explicitly passed --no-shuffle, force False. Otherwise use config.
    should_shuffle = False if args.no_shuffle else config_shuffle_default

    if not should_shuffle:
        print("-> Shuffling DISABLED (via Config or CLI)")

    # Mix data
    mix_data(
        appliance_name=args.appliance,
        real_rows=args.real_rows,
        synthetic_rows=args.synthetic_rows,
        real_path=args.real_path,
        output_suffix=args.suffix,
        shuffle=should_shuffle
    )

