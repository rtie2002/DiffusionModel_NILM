import numpy as np
import pandas as pd
import argparse
import os
import sys

# Supported appliances
APPLIANCES = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']

def detect_appliance_from_path(file_path):
    """Detect appliance name from file path"""
    file_lower = os.path.basename(file_path).lower()
    for appliance in APPLIANCES:
        if appliance in file_lower:
            return appliance
    return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert Diffusion Model NPY output to CSV')
    parser.add_argument('--input', type=str, default=None, help='Path to the input .npy file')
    parser.add_argument('--appliance', type=str, default=None, choices=APPLIANCES,
                        help='Appliance name (auto-detected from filename if not specified)')
    args = parser.parse_args()

    # Interactive prompt if no input provided
    input_path = args.input
    if not input_path:
        print("=" * 60)
        print("NPY TO CSV CONVERTER")
        print("=" * 60)
        print("Enter the path to your .npy file:")
        input_path = input("Path: ").strip()
        
        # Remove quotes
        if input_path.startswith('"') and input_path.endswith('"'):
            input_path = input_path[1:-1]
        if input_path.startswith("'") and input_path.endswith("'"):
            input_path = input_path[1:-1]

    if not input_path:
        print("❌ Error: No input file specified.")
        return

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return

    # Auto-detect appliance from filename
    appliance_name = args.appliance
    if not appliance_name:
        appliance_name = detect_appliance_from_path(input_path)
        if appliance_name:
            print(f"✓ Detected appliance: {appliance_name}")
        else:
            print(f"\n⚠ Could not auto-detect appliance from filename.")
            print(f"Available: {', '.join(APPLIANCES)}")
            while True:
                user_input = input("Enter appliance name: ").strip().lower()
                if user_input in APPLIANCES:
                    appliance_name = user_input
                    break
                print(f"Invalid appliance. Please choose from: {', '.join(APPLIANCES)}")

    print(f"\nLoading NPY file: {input_path}...")
    
    try:
        data = np.load(input_path)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load file. It seems corrupt or incomplete.")
        print(f"Details: {e}")
        return

    print(f"Original shape: {data.shape}")

    # Handle various shapes
    # Expected: (N, 512, 9) where 9 = 1 power + 8 time features
    # Or: (N, 512, 1) or (N, 512)
    if len(data.shape) == 3:
        n_features = data.shape[2]
        data_2d = data.reshape(-1, n_features)
    elif len(data.shape) == 2:
        n_features = 1
        data_2d = data.reshape(-1, 1)
    else:
        print(f"❌ Error: Unsupported data shape {data.shape}")
        return

    print(f"Reshaped to: {data_2d.shape}")

    # Create DataFrame with column names
    print("Creating DataFrame...")
    if data_2d.shape[1] == 9:
        cols = [
            appliance_name,
            'minute_sin', 'minute_cos',
            'hour_sin', 'hour_cos',
            'dow_sin', 'dow_cos',
            'month_sin', 'month_cos'
        ]
    elif data_2d.shape[1] == 10:
        # Some versions have aggregate? 
        cols = ['aggregate', appliance_name] + ['t'+str(i) for i in range(8)]
    else:
        cols = [f'col_{i}' for i in range(data_2d.shape[1])]
        cols[0] = appliance_name
    
    df = pd.DataFrame(data_2d, columns=cols[:data_2d.shape[1]])

    # Save to CSV
    output_file = os.path.splitext(input_path)[0] + '.csv'
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)

    print(f"\n✅ CONVERSION SUCCESSFUL!")
    print(f"  Input:     {os.path.basename(input_path)}")
    print(f"  Output:    {os.path.basename(output_file)}")
    print(f"  Appliance: {appliance_name}")
    print(f"  Rows:      {len(df):,}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    if appliance_name in df.columns:
        print(f"\nData summary:")
        print(f"  {appliance_name} power range: [{df[appliance_name].min():.4f}, {df[appliance_name].max():.4f}]")

if __name__ == "__main__":
    main()
