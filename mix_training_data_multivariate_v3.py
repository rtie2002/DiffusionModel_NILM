# -*- coding: utf-8 -*-
"""
NILM Data Mixing Script v3 (Evenly-Spaced OFF-Period Injection)
================================================================
Key improvement:
  - This script identifies all "True OFF" segments in the real data (using Algorithm 1 logic)
    and distributes synthetic events EVENLY across these segments.
  - This ensures synthetic data never overlaps with real ON-periods and is
    spread uniformly throughout the dataset's available "quiet time".

Logic:
  1. Load Real Data and identify "True OFF" periods (Algorithm 1 inverse).
  2. Load Synthetic Target Appliance power and detect events (Algorithm 1).
  3. Detect contiguous OFF segments in the real data.
  4. Calculate total OFF duration and distribute N synthetic events 
     proportionally across these segments.
  5. Within each OFF segment, place events at equal intervals.
  6. Save combined dataset.

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
        npy_path = Path(f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy')
    
    print(f"Loading synthetic data from: {npy_path}")
    data = np.load(npy_path)
    power_01 = data[:, :, 0].reshape(-1)
    spec = APPLIANCE_SPECS[appliance_name]
    power_watts = power_01 * spec['max_power']
    return power_watts, data[:, :, 1:].reshape(-1, 8) 


def get_off_periods_mask(appliance_name, df):
    """Get boolean mask identifying where target appliance is strictly OFF."""
    app_z = df.iloc[:, 1].values
    spec = APPLIANCE_SPECS[appliance_name]
    app_w = app_z * spec['std'] + spec['mean']
    
    if CONFIG and appliance_name in CONFIG['appliances']:
        x_threshold = CONFIG['appliances'][appliance_name]['on_power_threshold']
        l_window = CONFIG['algorithm1']['window_length']
    else:
        x_threshold = 15.0
        l_window = 100
        
    is_on_event = app_w >= x_threshold
    kernel_size = l_window * 2 + 1
    kernel = np.ones(kernel_size)
    expanded_on = np.convolve(is_on_event.astype(float), kernel, mode='same') > 0
    return ~expanded_on


def detect_synthetic_events(syn_power_watts, appliance_name):
    if CONFIG and appliance_name in CONFIG['appliances']:
        x_threshold = CONFIG['appliances'][appliance_name]['on_power_threshold']
        l_window = CONFIG['algorithm1']['window_length']
    else:
        x_threshold = 15.0
        l_window = 100
    
    n = len(syn_power_watts)
    is_on = syn_power_watts >= x_threshold
    kernel_size = l_window * 2 + 1
    kernel = np.ones(kernel_size)
    expanded_on = np.convolve(is_on.astype(float), kernel, mode='same') > 0
    
    events = []
    in_event = False
    event_start = 0
    for i in range(n):
        if expanded_on[i] and not in_event:
            event_start = i
            in_event = True
        elif not expanded_on[i] and in_event:
            events.append((event_start, i))
            in_event = False
    if in_event: events.append((event_start, n))
    return events


def extract_background_from_df(df, appliance_name):
    """Calculate background power from a dataframe segment."""
    agg_z = df.iloc[:, 0].values
    app_z = df.iloc[:, 1].values
    agg_w = agg_z * AGG_STD + AGG_MEAN
    spec = APPLIANCE_SPECS[appliance_name]
    app_w = app_z * spec['std'] + spec['mean']
    bg_w = agg_w - app_w
    return np.maximum(bg_w, 0)


def mix_data_v3(appliance_name, real_rows, synthetic_rows, real_path=None, suffix="v3"):
    print(f"\n{'='*60}\nNILM Mixed Dataset Construction v3 (Evenly-Spaced OFF Injection)\n{'='*60}")
    
    # 1. Load Data
    if real_path is None:
        real_path = f'created_data/UK_DALE/{appliance_name}_training_.csv'
    real_df = pd.read_csv(real_path).iloc[:real_rows].copy()
    
    # 2. Identify contiguous OFF segments in Real Data
    is_off_mask = get_off_periods_mask(appliance_name, real_df)
    off_segments = []
    in_seg = False
    start = 0
    for i in range(len(is_off_mask)):
        if is_off_mask[i] and not in_seg:
            start = i; in_seg = True
        elif not is_off_mask[i] and in_seg:
            off_segments.append((start, i)); in_seg = False
    if in_seg: off_segments.append((start, len(is_off_mask)))
    
    total_off_len = sum(e - s for s, e in off_segments)
    print(f"Total rows: {len(real_df)}, Total OFF periods: {total_off_len} rows, OFF segments count: {len(off_segments)}")

    # 3. Load and prepare Synthetic Events
    syn_app_w, syn_time_features = load_synthetic_appliance(appliance_name, 'synthetic_data_multivariate')
    syn_events_raw = detect_synthetic_events(syn_app_w, appliance_name)
    
    actual_syn_rows = min(synthetic_rows, len(syn_app_w)) if synthetic_rows > 0 else 0
    if actual_syn_rows <= 0:
        print("Baseline mode - No synthetic data to inject.")
        real_df.to_csv(f'created_data/UK_DALE/{appliance_name}/{appliance_name}_training_{suffix}.csv', index=False)
        return

    # Select enough events for target rows
    selected_events = []
    current_rows = 0
    while current_rows < actual_syn_rows:
        for s, e in syn_events_raw:
            length = min(e - s, actual_syn_rows - current_rows)
            selected_events.append((s, s + length))
            current_rows += length
            if current_rows >= actual_syn_rows: break
    
    num_events = len(selected_events)
    print(f"Injecting {num_events} events across {total_off_len} OFF-period rows.")

    # 4. Distribute events across OFF segments
    # Each segment gets a share of events proportional to its length
    segment_event_counts = []
    for s, e in off_segments:
        count = int(np.round(num_events * (e - s) / total_off_len))
        segment_event_counts.append(count)
    
    # Adjust to match exact num_events due to rounding
    diff = num_events - sum(segment_event_counts)
    if diff != 0:
        # Adjustment: add/sub from largest segments
        indices = np.argsort([e - s for s, e in off_segments])[::-1]
        for i in range(abs(diff)):
            segment_event_counts[indices[i % len(indices)]] += (1 if diff > 0 else -1)

    # 5. Build final sequence by injecting into segments
    # We will build a list of dataframes (Segments and Events)
    final_parts = []
    event_ptr = 0
    spec = APPLIANCE_SPECS[appliance_name]
    
    curr_real_idx = 0
    for seg_idx, (seg_start, seg_end) in enumerate(off_segments):
        # Add real data BEFORE this OFF segment
        if seg_start > curr_real_idx:
            final_parts.append(real_df.iloc[curr_real_idx : seg_start])
        
        # Process this OFF segment
        seg_len = seg_end - seg_start
        seg_df = real_df.iloc[seg_start : seg_end].copy()
        n_events_here = segment_event_counts[seg_idx]
        
        if n_events_here > 0:
            # Even spacing logic inside this segment
            # Gap between events
            avg_event_len = sum(selected_events[i][1] - selected_events[i][0] for i in range(event_ptr, event_ptr+n_events_here)) / n_events_here
            spacing = (seg_len - (avg_event_len * n_events_here)) / (n_events_here + 1)
            spacing = max(0, spacing)
            
            curr_seg_pos = 0
            for _ in range(n_events_here):
                if event_ptr >= num_events: break
                
                # 1. Add quiet part of background
                quiet_len = int(spacing)
                if quiet_len > 0 and curr_seg_pos + quiet_len <= seg_len:
                    final_parts.append(seg_df.iloc[curr_seg_pos : curr_seg_pos + quiet_len])
                    curr_seg_pos += quiet_len
                
                # 2. Add Synthetic Event
                s_syn, e_syn = selected_events[event_ptr]
                event_len = e_syn - s_syn
                
                # Extract background from real segment (or cycle it if segment is too short)
                # For simplicity, we use the next N rows of the current quiet segment as background
                bg_slice_len = min(event_len, seg_len - curr_seg_pos)
                if bg_slice_len > 0:
                    real_bg_w = extract_background_from_df(seg_df.iloc[curr_seg_pos : curr_seg_pos + bg_slice_len], appliance_name)
                    ỹ_event = syn_app_w[s_syn : s_syn + bg_slice_len]
                    time_event = syn_time_features[s_syn : s_syn + bg_slice_len]
                    
                    x_syn = real_bg_w + ỹ_event
                    agg_z = (x_syn - AGG_MEAN) / AGG_STD
                    app_z = (ỹ_event - spec['mean']) / spec['std']
                    
                    evt_df = pd.DataFrame(time_event, columns=real_df.columns[2:])
                    evt_df.insert(0, real_df.columns[0], agg_z)
                    evt_df.insert(1, real_df.columns[1], app_z)
                    final_parts.append(evt_df)
                    
                    curr_seg_pos += bg_slice_len
                event_ptr += 1
            
            # Add remaining part of segment
            if curr_seg_pos < seg_len:
                final_parts.append(seg_df.iloc[curr_seg_pos:])
        else:
            final_parts.append(seg_df)
        
        curr_real_idx = seg_end
        
    # Add any remaining real data at the very end
    if curr_real_idx < len(real_df):
        final_parts.append(real_df.iloc[curr_real_idx:])

    # 6. Save
    final_df = pd.concat(final_parts, ignore_index=True)
    out_dir = Path(f'created_data/UK_DALE/{appliance_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{appliance_name}_training_{suffix}.csv'
    final_df.to_csv(out_file, index=False)
    print(f"DONE! Saved to: {out_file}, Rows: {len(final_df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    parser.add_argument('--real_rows', type=int, default=200000)
    parser.add_argument('--synthetic_rows', type=int, default=200000)
    parser.add_argument('--suffix', type=str, default="200k+200k_event_v3")
    args = parser.parse_args()
    
    mix_data_v3(args.appliance, args.real_rows, args.synthetic_rows, suffix=args.suffix)
