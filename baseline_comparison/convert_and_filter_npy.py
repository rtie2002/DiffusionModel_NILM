import numpy as np
import pandas as pd
import os
import argparse
import sys
import yaml

# ==============================================================================
# 2-IN-1: BALANCED CONVERSION + ALGORITHM 1 FILTERING
# ==============================================================================
# 1. Converts NPY to CSV and scales to real Watts.
# 2. Applies Algorithm 1 to select only active "ON" periods.
# ==============================================================================

APPLIANCES = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']

def load_config():
    # Attempt to load global config for thresholds
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'Config', 'preprocess', 'preprocess_multivariate.yaml')
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        print(f"⚠️ Warning: Could not find config at {config_path}. Using safe defaults.")
        return None

def remove_isolated_spikes(power_sequence, window_size=5, spike_threshold=3.0, background_threshold=50):
    """Algorithm 1 Internal: Remove isolated noise sparks."""
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    padded = np.pad(power_sequence, window_size//2, mode='edge')
    for i in range(n):
        if power_sequence[i] < background_threshold: continue
        window = padded[i : i + window_size]
        surrounding = np.concatenate([window[:window_size//2], window[window_size//2 + 1:]])
        if np.median(surrounding) < background_threshold and power_sequence[i] > spike_threshold * background_threshold:
            power_sequence[i] = 0
    return power_sequence

def main():
    parser = argparse.ArgumentParser(description='Convert and Filter Synthetic NPY')
    parser.add_argument('--input', type=str, default=None, help='Path to synthetic .npy file')
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        print("\n" + "=" * 60)
        print("2-IN-1 CONVERTER & FILTER (ALGORITHM 1)")
        print("=" * 60)
        input_path = input("Please enter the path to your .npy file: ").strip()
    
    # Path cleaning
    input_path = input_path.strip('"').strip("'")
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return

    # 1. Identify Appliance
    filename = os.path.basename(input_path).lower()
    appliance = next((a for a in APPLIANCES if a in filename), None)
    if not appliance:
        print("❌ Could not detect appliance from filename.")
        return

    # 2. Extract Real Stats for Scaling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    real_csv_path = os.path.join(script_dir, 'data', f'{appliance}_multivariate.csv')
    if not os.path.exists(real_csv_path):
        print(f"❌ Real data reference not found at: {real_csv_path}")
        return

    print(f"🚀 Processing: {appliance.upper()}")
    real_df = pd.read_csv(real_csv_path)
    power_col = appliance if appliance in real_df.columns else real_df.columns[0]
    real_max = real_df[power_col].max()
    real_min = real_df[power_col].min()
    real_range = real_max - real_min

    # 3. Load Synthetic Data
    syn_data = np.load(input_path)
    n_features = syn_data.shape[2] if len(syn_data.shape) == 3 else 1
    data_2d = syn_data.reshape(-1, n_features)

    # 4. PART A: Rescale to Watts (Linear Transform)
    # [0, 1] -> [Real Min, Real Max]
    data_2d[:, 0] = data_2d[:, 0] * (real_range + 1e-8) + real_min
    print(f"⚖️ Applied Scale: Data converted to Watts range.")

    # 5. PART B: Application of Algorithm 1 (Filtering)
    config = load_config()
    if config:
        params = config['appliances'][appliance]
        x_threshold = params['on_power_threshold']
        l_window = config['algorithm1']['window_length']
    else:
        x_threshold = 50.0  # Safe default
        l_window = 100

    print(f"🧹 Applying Algorithm 1 Filtering (Threshold={x_threshold}W, Window={l_window})...")
    
    # Spike Removal
    power_seq = remove_isolated_spikes(data_2d[:, 0])
    
    # Active Selection
    t_start = np.where(power_seq >= x_threshold)[0]
    t_selected = []
    for idx in t_start:
        t_selected.extend(range(max(0, idx - l_window), min(len(power_seq), idx + l_window + 1)))
    t_selected = sorted(set(t_selected))
    
    if not t_selected:
        print("⚠️ Warning: No 'ON' periods detected! Data might be too quiet.")
        data_filtered = data_2d # Fallback to all data if none selected
    else:
        data_filtered = data_2d[t_selected]
        print(f"✨ Selected {len(data_filtered):,} active samples ({len(data_filtered)/len(data_2d)*100:.1f}% retention).")

    # 6. Final Normalization back to [0, 1] for training compatibility
    # Ensure it's exactly [0, 1] relative to the real max power
    data_filtered[:, 0] = np.clip(data_filtered[:, 0], 0, real_max) / (real_max + 1e-8)

    # 7. Save to CSV (9 Standard Columns)
    cols = [appliance, 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    
    # Adjust if there's an aggregate column present
    if data_filtered.shape[1] > len(cols):
        cols = ['aggregate'] + cols
        
    df_output = pd.DataFrame(data_filtered, columns=cols[:data_filtered.shape[1]])
    output_path = os.path.splitext(input_path)[0] + '_filtered.csv'
    df_output.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS: Saved filtered & normalized CSV to: {os.path.basename(output_path)}")

if __name__ == "__main__":
    main()
