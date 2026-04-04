import numpy as np
import pandas as pd
import os
import argparse

# ==============================================================================
# BALANCED NPY TO CSV CONVERTER
# ==============================================================================
# This script converts TimeGAN/CGAN .npy output back to CSV and automatically
# performs a linear transformation to match the scale of the real data.
# ==============================================================================

APPLIANCES = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']

def main():
    parser = argparse.ArgumentParser(description='Balanced NPY to CSV Converter')
    parser.add_argument('--input', type=str, default=None, help='Path to the synthetic .npy file')
    args = parser.parse_args()

    input_path = args.input
    
    # NEW: Interactive prompt if no input provided
    if not input_path:
        print("\n" + "=" * 60)
        print("BALANCED NPY TO CSV CONVERTER (AUTO-RESCALE)")
        print("=" * 60)
        input_path = input("Please enter the path to your .npy file: ").strip()
        
    # Clean up paths (remove quotes from shift+right-click copies)
    if input_path.startswith('"') and input_path.endswith('"'):
        input_path = input_path[1:-1]
    if input_path.startswith("'") and input_path.endswith("'"):
        input_path = input_path[1:-1]

    if not input_path:
        print("❌ Error: No file path provided.")
        return

    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return

    # 1. Identify Appliance from filename
    filename = os.path.basename(input_path).lower()
    appliance = None
    for a in APPLIANCES:
        if a in filename:
            appliance = a
            break
    
    if not appliance:
        print("⚠️ Could not detect appliance from filename. Please include appliance name in the filename.")
        return

    # 2. Locate Real Reference Data for Scaling
    # The script assumes it is in <root>/baseline_comparison/
    # Reference data is in <root>/baseline_comparison/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    real_csv_path = os.path.join(script_dir, 'data', f'{appliance}_multivariate.csv')

    if not os.path.exists(real_csv_path):
        print(f"⚠️ Reference data not found at: {real_csv_path}")
        return

    print(f"🚀 Processing: {appliance.upper()}")
    
    # 3. Load Real Stats (Min/Max)
    print(f"🔎 Calculating real scale from: {os.path.basename(real_csv_path)}...")
    real_df = pd.read_csv(real_csv_path)
    power_col = appliance if appliance in real_df.columns else real_df.columns[0]
    
    real_max = real_df[power_col].max()
    real_min = real_df[power_col].min()
    real_range = real_max - real_min
    print(f"📊 Real Power Range: [{real_min:.2f}, {real_max:.2f}]")

    # 4. Load & Rescale Synthetic Data
    print(f"📥 Loading synthetic data: {os.path.basename(input_path)}...")
    try:
        syn_data = np.load(input_path)
    except Exception as e:
        print(f"❌ Failed to load .npy: {e}")
        return

    # Handle shape (N, 512, 9) or (N, 512, 1)
    if len(syn_data.shape) == 3:
        n_features = syn_data.shape[2]
        data_2d = syn_data.reshape(-1, n_features)
    else:
        data_2d = syn_data.reshape(-1, 1) if len(syn_data.shape) == 2 else syn_data

    # 5. Linear Transform [0, 1] -> [Real Min, Real Max]
    # Concept identical to mix_training_data_multivariate.py
    syn_p_scaled = data_2d[:, 0] * (real_range + 1e-8) + real_min
    data_2d[:, 0] = syn_p_scaled
    print(f"⚖️ Applied Linear Transform: Scale fixed to Watts.")

    # 6. Create CSV with standard columns
    cols = [
        appliance,
        'minute_sin', 'minute_cos',
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos'
    ]
    
    # Final check on column count
    if data_2d.shape[1] > len(cols):
        cols = ['aggregate'] + cols
    
    df_output = pd.DataFrame(data_2d, columns=cols[:data_2d.shape[1]])

    # 7. Save output
    output_path = os.path.splitext(input_path)[0] + '_rescaled.csv'
    df_output.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS: Saved re-scaled CSV to: {os.path.basename(output_path)}")
    print(f"📈 Power column range: [{df_output[appliance].min():.2f}, {df_output[appliance].max():.2f}]")

if __name__ == "__main__":
    main()
