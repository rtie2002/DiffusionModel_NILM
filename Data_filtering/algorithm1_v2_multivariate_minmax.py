import numpy as np
import pandas as pd
import argparse
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

"""
Algorithm 1 for MinMax Normalized NPY Data (Diffusion Model Output)
Same logic as algorithm1_v2_multivariate.py but tailored for .npy files in [0, 1] range.

WorkFlow:
1. Load .npy file (usually (N, 512, 9))
2. Flatten to (N*512, 9)
3. Convert MinMax [0, 1] power to Watts using appliance max_power
4. Apply Algorithm 1 (Window Selection, Spike Removal)
5. Normalize back to [0, 1]
6. Save as CSV (9 columns: appliance + 8 time features)
"""

# Set base directory for relative paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / 'Config' / 'preprocess' / 'preprocess_multivariate.yaml'

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Appliance parameters
APPLIANCE_PARAMS = {}
for app_name, app_conf in CONFIG['appliances'].items():
    APPLIANCE_PARAMS[app_name] = {
        'on_power_threshold': app_conf['on_power_threshold'],
        'max_power': app_conf['max_power'],
        'max_power_clip': app_conf.get('max_power_clip'),
    }

def remove_isolated_spikes(power_sequence, window_size=5, spike_threshold=3.0, 
                           background_threshold=50):
    """
    Remove isolated spikes that suddenly appear when surrounding data is near zero.
    """
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    num_spikes = 0
    
    half_window = window_size // 2
    padded = np.pad(power_sequence, half_window, mode='edge')
    
    for i in range(n):
        current_value = power_sequence[i]
        if current_value < background_threshold:
            continue
        
        window = padded[i : i + window_size]
        surrounding = np.concatenate([window[:half_window], window[half_window+1:]])
        median_surrounding = np.median(surrounding)
        
        low_values_count = np.sum(surrounding < background_threshold)
        is_background_low = low_values_count >= (len(surrounding) * 0.6)
        
        if is_background_low and current_value > spike_threshold * median_surrounding:
            if current_value > background_threshold * 2:
                power_sequence[i] = 0
                num_spikes += 1
    
    return power_sequence, num_spikes

def algorithm1_core(data, appliance_name, x_threshold, l_window=100, x_noise=0,
                   remove_spikes=True, spike_window=5, spike_threshold=3.0,
                   background_threshold=50, clip_max=None, max_power=None):
    """
    Core Algorithm 1 logic.
    Input 'data' is a 2D array (N, 9) where col 0 is power in Watts.
    """
    power_sequence = data[:, 0].copy()
    
    # Step 0: Remove isolated spikes
    if remove_spikes:
        power_sequence, num_spikes = remove_isolated_spikes(
            power_sequence, 
            window_size=spike_window,
            spike_threshold=spike_threshold,
            background_threshold=background_threshold
        )
        print(f"  Spike removal: {num_spikes} isolated spikes detected and removed")
    
    # Step 1-9: Window Selection
    power_sequence[power_sequence < x_noise] = 0
    t_start = np.where(power_sequence >= x_threshold)[0]
    
    t_selected = []
    for idx in t_start:
        start = max(0, idx - l_window)
        end = min(len(power_sequence), idx + l_window + 1)
        t_selected.extend(range(start, end))
    
    t_selected = sorted(set(t_selected))
    
    if not t_selected:
        print("  WARNING: No data points selected based on threshold!")
        return None, []

    # Filter data
    filtered_data = data[t_selected].copy()
    # Update power column with spike-removed/noise-removed version
    filtered_data[:, 0] = power_sequence[t_selected]
    
    # Step 10: Clip outliers
    if clip_max is not None:
        num_clipped = np.sum(filtered_data[:, 0] > clip_max)
        filtered_data[:, 0] = np.clip(filtered_data[:, 0], 0, clip_max)
        if num_clipped > 0:
            print(f"  Clipped {num_clipped} values above {clip_max}W ({num_clipped/len(filtered_data)*100:.2f}%)")
    
    # Step 11: MinMax Normalization
    print(f"  Normalizing using max_power: {max_power} W")
    filtered_data[:, 0] = np.clip(filtered_data[:, 0], 0, max_power) / max_power
    
    return filtered_data, t_selected

def main():
    parser = argparse.ArgumentParser(description='Apply Algorithm 1 to MinMax Normalized NPY data')
    parser.add_argument('--appliance_name', type=str, default=None, help='Appliance name (detects from path if None)')
    parser.add_argument('--input_file', '-i', type=str, default=None, help='Input .npy file path')
    parser.add_argument('--output_dir', '-o', type=str, default='Data/datasets', help='Output directory')
    parser.add_argument('--window', type=int, default=CONFIG['algorithm1']['window_length'])
    parser.add_argument('--clip_max', type=float, default=None)
    
    args = parser.parse_args()

    # Interactive input if missing
    input_file = args.input_file
    if not input_file:
        print("=" * 60)
        print("ALGORITHM 1: NPY (MINMAX) PROCESSOR")
        print("=" * 60)
        print("Enter path to the .npy file:")
        input_file = input("Path: ").strip().replace('"', '').replace("'", "")

    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        return

    # Detect appliance
    appliance_name = args.appliance_name
    if not appliance_name:
        for app in APPLIANCE_PARAMS.keys():
            if app in os.path.basename(input_file).lower():
                appliance_name = app
                break
    
    if not appliance_name:
        print(f"Available: {', '.join(APPLIANCE_PARAMS.keys())}")
        appliance_name = input("Enter appliance name: ").strip().lower()

    if appliance_name not in APPLIANCE_PARAMS:
        print(f"❌ Error: Unknown appliance '{appliance_name}'")
        return

    params = APPLIANCE_PARAMS[appliance_name]
    max_power = params['max_power']
    x_threshold = params['on_power_threshold']
    
    # Determine clip_max
    clip_max = args.clip_max or params.get('max_power_clip') or CONFIG['algorithm1']['clip_max']

    print(f"\nProcessing {appliance_name} from {input_file}")
    print(f"  Threshold: {x_threshold}W | Max Power: {max_power}W | Clip: {clip_max}W")

    # Load and Flatten
    try:
        data = np.load(input_file)
    except Exception as e:
        print(f"❌ Error loading NPY: {e}")
        return

    print(f"  Shape: {data.shape}")
    if len(data.shape) == 3:
        n_windows, window_size, n_feats = data.shape
        data = data.reshape(-1, n_feats)
        print(f"  Flattened to: {data.shape}")

    # Scale to Watts for Algorithm 1
    print("  Rescaling MinMax [0, 1] to Watts...")
    data_watts = data.copy()
    data_watts[:, 0] = data[:, 0] * max_power

    # Apply Algorithm 1
    filtered_data, t_indices = algorithm1_core(
        data_watts,
        appliance_name=appliance_name,
        x_threshold=x_threshold,
        l_window=args.window,
        remove_spikes=True,
        max_power=max_power,
        clip_max=clip_max,
        background_threshold=CONFIG['algorithm1']['background_threshold'],
        spike_threshold=CONFIG['algorithm1']['spike_threshold'],
        spike_window=CONFIG['algorithm1']['spike_window']
    )

    if filtered_data is None:
        return

    # Save to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.basename(input_file).replace('.npy', '_algorithm1.csv')
    output_path = os.path.join(args.output_dir, filename)
    
    cols = [appliance_name, 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    df_out = pd.DataFrame(filtered_data, columns=cols)
    
    print(f"  Saving to {output_path}...")
    df_out.to_csv(output_path, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"  Selected: {len(filtered_data):,} samples ({len(filtered_data)/len(data)*100:.2f}%)")
    print(f"  Output Format: 9-column CSV (Normalized 0-1)")

if __name__ == '__main__':
    main()
