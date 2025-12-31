import numpy as np
import pandas as pd
import argparse
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert Diffusion Model NPY output to CSV')
    parser.add_argument('--input', type=str, default='OUTPUT/washingmachine_multivariate/ddpm_fake_washingmachine_multivariate.npy', 
                        help='Path to the input .npy file')
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

    # Create DataFrame with column names
    print("Creating DataFrame...")
    cols = [
        'washingmachine',
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

    print(f"\nConversion complete!")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_file}")
    print(f"\nFirst few rows:")
    print(df.head())

if __name__ == "__main__":
    main()
