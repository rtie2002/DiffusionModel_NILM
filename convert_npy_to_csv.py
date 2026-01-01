import numpy as np
import pandas as pd
import argparse
import os

# Supported appliances
APPLIANCES = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']

def detect_appliance_from_path(file_path):
    """Detect appliance name from file path"""
    file_lower = file_path.lower()
    for appliance in APPLIANCES:
        if appliance in file_lower:
            return appliance
    return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert Diffusion Model NPY output to CSV (supports all 5 appliances)')
    parser.add_argument('--input', type=str, default='OUTPUT/washingmachine_multivariate/ddpm_fake_washingmachine_multivariate.npy', 
                        help='Path to the input .npy file')
    parser.add_argument('--appliance', type=str, default=None, choices=APPLIANCES,
                        help='Appliance name (auto-detected from filename if not specified)')
    args = parser.parse_args()

    input_path = args.input
    
    # Check if file exists
    if not os.path.exists(input_path):
        # Try checking in current directory if not found
        local_path = os.path.basename(input_path)
        if os.path.exists(local_path):
            input_path = local_path
        else:
            print(f"❌ Error: File not found: {input_path}")
            return

    # Auto-detect appliance from filename
    appliance_name = args.appliance
    if not appliance_name:
        appliance_name = detect_appliance_from_path(input_path)
        if appliance_name:
            print(f"✓ Detected appliance: {appliance_name}")
        else:
            print(f"⚠ Warning: Could not auto-detect appliance from filename")
            print(f"   Available: {', '.join(APPLIANCES)}")
            user_input = input("Enter appliance name: ").strip().lower()
            if user_input in APPLIANCES:
                appliance_name = user_input
            else:
                print(f"❌ Error: Invalid appliance '{user_input}'")
                return

    print(f"Loading NPY file: {input_path}...")
    
    try:
        data = np.load(input_path)
    except ValueError as e:
        print(f"\n❌ ERROR: Failed to load file. It seems corrupt or incomplete.")
        print(f"   Details: {e}")
        print(f"   👉 Solution: Please regenerate the data using 'python main.py ...'")
        return

    print(f"Shape: {data.shape}")

    # Reshape to 2D
    print("Reshaping to 2D...")
    try:
        data_2d = data.reshape(-1, 9)
    except ValueError:
        print(f"❌ Error: Expected 9 columns (Power + 8 Time features), but got shape {data.shape}")
        return
        
    print(f"Reshaped to: {data_2d.shape}")

    # Create DataFrame with column names (dynamic based on appliance)
    print("Creating DataFrame...")
    cols = [
        appliance_name,  # Use detected appliance name
        'minute_sin', 'minute_cos',
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos'
    ]
    
    df = pd.DataFrame(data_2d, columns=cols)

    # Save to CSV
    output_file = input_path.replace('.npy', '.csv')
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)

    print(f"\n✓ Conversion complete!")
    print(f"  Appliance: {appliance_name}")
    print(f"  Input:     {input_path}")
    print(f"  Output:    {output_file}")
    print(f"  Rows:      {len(df):,}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData summary:")
    print(f"  {appliance_name} power range: [{df[appliance_name].min():.4f}, {df[appliance_name].max():.4f}]")

if __name__ == "__main__":
    main()
