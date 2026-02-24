# -*- coding: utf-8 -*-
"""
NILM Data Mixing Script v3 (Event-Based Injection)
====================================================
Key improvement over v2:
  - v2 sliced synthetic data into FIXED windows (w10, w50, w100, w600) and shuffled them.
    This BREAKS the physical continuity of appliance ON-periods (e.g., a kettle boiling
    cycle gets chopped into meaningless fragments).
  - v3 uses Algorithm 1 logic to detect complete ON-period EVENTS in the synthetic data,
    treats each event as an indivisible atomic block, and injects these complete events
    into the real data stream at random positions.

Logic:
  1. Load Synthetic Target Appliance power (ỹ_target) in Watts
  2. Detect ON-period events in synthetic data using Algorithm 1 (threshold + l_window expansion)
  3. Extract each event as a complete block (including quiet context before/after)
  4. Extract REAL Background Power (X_bg = X_total - Y_target) from certified OFF periods
  5. For each synthetic event block: X_syn_event = random(X_bg) + ỹ_event
  6. Randomly inject these complete synthetic event blocks into the real data stream
  7. Save combined dataset

Usage:
    python mix_training_data_multivariate_v3.py --appliance kettle --real_rows 200000 --synthetic_rows 200000
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
    return power_watts, data[:, :, 1:].reshape(-1, 8)  # Return Watts, TimeFeatures


def extract_real_background_pool(appliance_name, real_path, bg_window_size=600):
    """
    Extract periods where the target appliance is OFF to create a background power pool.
    Uses Algorithm 1 inverse logic: certified OFF = NOT(expanded ON regions).
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
    bg_w = np.maximum(bg_w, 0)  # Physical constraint
    
    # -------------------------------------------------------------
    # Use Algorithm 1 logic to perfectly identify true OFF periods
    # -------------------------------------------------------------
    if CONFIG and appliance_name in CONFIG['appliances']:
        x_threshold = CONFIG['appliances'][appliance_name]['on_power_threshold']
        l_window = CONFIG['algorithm1']['window_length']
    else:
        x_threshold = 15.0
        l_window = 100
        
    # 1. Identify ON points
    is_on_event = app_w >= x_threshold
    
    # 2. Expand ON points by l_window (Algorithm 1 logic)
    kernel_size = l_window * 2 + 1
    kernel = np.ones(kernel_size)
    expanded_on = np.convolve(is_on_event.astype(float), kernel, mode='same') > 0
    
    # 3. True OFF periods are the inverse of expanded ON periods
    is_off_period = ~expanded_on
    
    # Slice into windows of background power where appliance is 100% safely OFF
    pool = []
    for i in range(0, len(bg_w) - bg_window_size, bg_window_size):
        if np.all(is_off_period[i : i + bg_window_size]): 
            pool.append(bg_w[i : i + bg_window_size])
    
    # FALLBACK
    if len(pool) == 0:
        print(f"  WARNING: 找不到符合 Algorithm 1 绝对强力过滤的背景窗口，正在选择【最大峰值功率最低】的窗口...")
        scored_windows = []
        for i in range(0, len(bg_w) - bg_window_size, bg_window_size):
            max_app_power = np.max(app_w[i : i + bg_window_size])
            scored_windows.append((max_app_power, bg_w[i : i + bg_window_size]))
        scored_windows.sort(key=lambda x: x[0])
        pool = [w[1] for w in scored_windows[:50]]
        print(f"  Fallback max appliance power in background: {[round(w[0],2) for w in scored_windows[:5]]} Watts")

    print(f"  Extracted {len(pool)} background windows (size {bg_window_size}) using Algorithm 1 filtering")
    return pool


def detect_synthetic_events(syn_power_watts, appliance_name):
    """
    Use Algorithm 1 logic to detect complete ON-period events in the synthetic data.
    
    Returns a list of (start_idx, end_idx) tuples, each representing one complete
    appliance activation event (including l_window context before and after).
    """
    if CONFIG and appliance_name in CONFIG['appliances']:
        x_threshold = CONFIG['appliances'][appliance_name]['on_power_threshold']
        l_window = CONFIG['algorithm1']['window_length']
    else:
        x_threshold = 15.0
        l_window = 100
    
    n = len(syn_power_watts)
    
    # 1. Identify ON points
    is_on = syn_power_watts >= x_threshold
    
    # 2. Expand ON points by l_window
    kernel_size = l_window * 2 + 1
    kernel = np.ones(kernel_size)
    expanded_on = np.convolve(is_on.astype(float), kernel, mode='same') > 0
    
    # 3. Find contiguous ON regions (each region = one complete event)
    events = []
    in_event = False
    event_start = 0
    
    for i in range(n):
        if expanded_on[i] and not in_event:
            # Start of a new event
            event_start = i
            in_event = True
        elif not expanded_on[i] and in_event:
            # End of current event
            events.append((event_start, i))
            in_event = False
    
    # Handle case where last event extends to end of data
    if in_event:
        events.append((event_start, n))
    
    # Print statistics
    event_lengths = [e - s for s, e in events]
    if event_lengths:
        print(f"  Detected {len(events)} synthetic ON-period events")
        print(f"  Event lengths: min={min(event_lengths)}, max={max(event_lengths)}, "
              f"mean={np.mean(event_lengths):.0f}, total_rows={sum(event_lengths)}")
    else:
        print(f"  WARNING: No ON-period events detected in synthetic data!")
    
    return events


def assemble_background_for_event(event_length, bg_pool, bg_window_size=600):
    """
    Assemble a continuous background noise sequence of exactly `event_length` steps
    by stitching together random windows from the background pool.
    """
    bg_sequence = []
    remaining = event_length
    
    while remaining > 0:
        bg_window = random.choice(bg_pool)
        chunk = bg_window[:min(remaining, len(bg_window))]
        bg_sequence.append(chunk)
        remaining -= len(chunk)
    
    return np.concatenate(bg_sequence)[:event_length]


def mix_data_v3(appliance_name, real_rows, synthetic_rows, real_path=None, suffix="v3", shuffle=True):
    print(f"\n{'='*60}\nNILM Mixed Dataset Construction v3 (Event-Based Injection)\n{'='*60}")
    
    # 1. Load Real Data
    if real_path is None:
        real_path = f'created_data/UK_DALE/{appliance_name}_training_.csv'
    real_df = pd.read_csv(real_path)
    real_subset = real_df.iloc[:real_rows].copy()
    
    # 2. Extract Background Pool from Real Data (OFF periods)
    bg_pool = extract_real_background_pool(appliance_name, real_path)
    
    # 3. Load Synthetic Appliance Power
    synthetic_dir = CONFIG['paths']['synthetic_dir'] if CONFIG else 'synthetic_data_multivariate'
    syn_app_w, syn_time_features = load_synthetic_appliance(appliance_name, synthetic_dir)
    
    # 4. Detect ON-period events in synthetic data using Algorithm 1 logic
    print(f"\nDetecting ON-period events in synthetic data...")
    events = detect_synthetic_events(syn_app_w, appliance_name)
    
    if len(events) == 0:
        print("ERROR: No events detected. Cannot proceed with event-based injection.")
        return
    
    # 5. Select events to meet synthetic_rows target
    actual_syn_rows = min(synthetic_rows, len(syn_app_w)) if synthetic_rows > 0 else 0
    
    if actual_syn_rows > 0:
        print(f"\nConstructing Synthetic Section (target: {actual_syn_rows} rows)...")
        spec = APPLIANCE_SPECS[appliance_name]
        
        # Shuffle events and select until we reach the target row count
        event_indices = list(range(len(events)))
        random.shuffle(event_indices)
        
        syn_event_dfs = []
        total_syn_rows_used = 0
        events_used = 0
        
        for evt_idx in event_indices:
            if total_syn_rows_used >= actual_syn_rows:
                break
            
            start, end = events[evt_idx]
            event_len = end - start
            
            # Cap if adding this event would exceed the target
            if total_syn_rows_used + event_len > actual_syn_rows:
                event_len = actual_syn_rows - total_syn_rows_used
                end = start + event_len
            
            # Extract synthetic appliance power for this event
            ỹ_event = syn_app_w[start:end]
            time_event = syn_time_features[start:end]
            
            # Assemble matching background noise
            x_bg = assemble_background_for_event(event_len, bg_pool)
            
            # Construct synthetic aggregate: X_syn = X_bg + ỹ_target
            x_syn = x_bg + ỹ_event
            
            # Normalize back to Z-score format (matching real data format)
            agg_z = (x_syn - AGG_MEAN) / AGG_STD
            app_z = (ỹ_event - spec['mean']) / spec['std']
            
            # Create event DataFrame
            event_df = pd.DataFrame(time_event, columns=real_df.columns[2:])
            event_df.insert(0, real_df.columns[0], agg_z)
            event_df.insert(1, real_df.columns[1], app_z)
            
            syn_event_dfs.append(event_df)
            total_syn_rows_used += event_len
            events_used += 1
        
        # If we still haven't reached the target (not enough events), cycle through again
        cycle_count = 0
        while total_syn_rows_used < actual_syn_rows and cycle_count < 10:
            cycle_count += 1
            random.shuffle(event_indices)
            for evt_idx in event_indices:
                if total_syn_rows_used >= actual_syn_rows:
                    break
                start, end = events[evt_idx]
                event_len = min(end - start, actual_syn_rows - total_syn_rows_used)
                end = start + event_len
                
                ỹ_event = syn_app_w[start:end]
                time_event = syn_time_features[start:end]
                x_bg = assemble_background_for_event(event_len, bg_pool)
                x_syn = x_bg + ỹ_event
                
                agg_z = (x_syn - AGG_MEAN) / AGG_STD
                app_z = (ỹ_event - spec['mean']) / spec['std']
                
                event_df = pd.DataFrame(time_event, columns=real_df.columns[2:])
                event_df.insert(0, real_df.columns[0], agg_z)
                event_df.insert(1, real_df.columns[1], app_z)
                
                syn_event_dfs.append(event_df)
                total_syn_rows_used += event_len
                events_used += 1
        
        print(f"  Events used: {events_used}, Total synthetic rows: {total_syn_rows_used}")
    else:
        print("Skipping synthetic construction (Baseline mode).")
        syn_event_dfs = []
    
    # 6. Combine real data and synthetic event blocks
    print("Finalizing Hybrid Dataset...")
    
    if shuffle and len(syn_event_dfs) > 0:
        # Keep real data as one continuous block, inject synthetic events at random positions
        # Split real data into segments at random cut points to create injection slots
        n_events = len(syn_event_dfs)
        
        # Generate n_events random cut points in the real data
        real_len = len(real_subset)
        cut_points = sorted(random.sample(range(1, real_len), min(n_events, real_len - 1)))
        
        # Split real data at cut points
        real_segments = []
        prev = 0
        for cp in cut_points:
            real_segments.append(real_subset.iloc[prev:cp])
            prev = cp
        real_segments.append(real_subset.iloc[prev:])  # Last segment
        
        # Shuffle event blocks
        random.shuffle(syn_event_dfs)
        
        # Interleave: real_segment[0], syn_event[0], real_segment[1], syn_event[1], ...
        all_parts = []
        for i, seg in enumerate(real_segments):
            all_parts.append(seg)
            if i < len(syn_event_dfs):
                all_parts.append(syn_event_dfs[i])
        
        # Append any remaining synthetic events at the end
        if len(syn_event_dfs) > len(real_segments):
            for j in range(len(real_segments), len(syn_event_dfs)):
                all_parts.append(syn_event_dfs[j])
        
        final_df = pd.concat(all_parts, ignore_index=True)
    else:
        # No shuffle: real data first, then all synthetic events appended
        all_parts = [real_subset] + syn_event_dfs
        final_df = pd.concat(all_parts, ignore_index=True)
    
    # 7. Save
    out_dir = Path(f'created_data/UK_DALE/{appliance_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{appliance_name}_training_{suffix}.csv'
    final_df.to_csv(out_file, index=False)
    
    total_syn = sum(len(df) for df in syn_event_dfs)
    print(f"\nDONE! Saved to: {out_file}")
    print(f"Total Rows: {len(final_df)} (Real: {len(real_subset)}, Synthetic: {total_syn})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NILM Data Mixing v3 - Event-Based Injection')
    parser.add_argument('--appliance', type=str, required=True)
    parser.add_argument('--real_rows', type=int, default=200000)
    parser.add_argument('--synthetic_rows', type=int, default=200000)
    parser.add_argument('--suffix', type=str, default="200k+200k_event_v3")
    parser.add_argument('--shuffle', action='store_true', help='Enable event injection shuffling')
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle', help='Disable shuffling')
    parser.set_defaults(shuffle=True)
    args = parser.parse_args()
    
    mix_data_v3(
        appliance_name=args.appliance, 
        real_rows=args.real_rows, 
        synthetic_rows=args.synthetic_rows, 
        suffix=args.suffix,
        shuffle=args.shuffle
    )
