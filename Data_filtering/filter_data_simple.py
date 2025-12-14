"""
Simple Data Filter using Algorithm 1 Core Logic

This script applies the core filtering logic from Algorithm 1 to clean data files.
Supports both .npy and .csv files with interactive prompts.

Usage:
    python filter_data_simple.py
    
Then follow the prompts to enter file path and parameters.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def denormalize_data(normalized_data, mean, std):
    """
    Denormalize data from [0,1] to Watts using appliance parameters.
    Assumes data was normalized using: (x - mean) / std
    """
    # Estimate max power from mean + 3*std (covers ~99.7% of data)
    max_power = mean + 3 * std
    
    # Convert [0,1] to Watts
    denormalized = normalized_data * max_power
    
    return denormalized

def filter_data(power_sequence, threshold, window=100, noise_threshold=0):
    """
    Core Algorithm 1 filtering logic (simplified, no spike detection)
    
    Args:
        power_sequence: 1D array of power values IN WATTS
        threshold: Power threshold for activation detection (Watts)
        window: Window length around activation points
        noise_threshold: Remove values below this threshold (Watts)
    
    Returns:
        filtered_data: Cleaned and MinMax normalized data
    """
    print(f"\n{'='*60}")
    print("FILTERING DATA (Algorithm 1)")
    print(f"{'='*60}")
    
    # Step 1: Remove noise
    power_sequence = power_sequence.copy()
    power_sequence[power_sequence < noise_threshold] = 0
    print(f"Step 1: Removed noise below {noise_threshold}W")
    
    # Step 2: Find activation points
    T_start = np.where(power_sequence >= threshold)[0]
    print(f"Step 2: Found {len(T_start)} activation points (>= {threshold}W)")
    
    if len(T_start) == 0:
        print("WARNING: No activation points found! Returning empty array.")
        return np.array([])
    
    # Step 3: Select windows around activation points
    T_selected = []
    for index in T_start:
        T_start_window = max(0, index - window)
        T_end_window = min(len(power_sequence), index + window + 1)
        T_selected.extend(range(T_start_window, T_end_window))
    
    T_selected = sorted(set(T_selected))
    print(f"Step 3: Selected {len(T_selected)} samples around activation points")
    
    # Step 4: Extract selected data
    x_selected = power_sequence[T_selected]
    
    # Step 5: MinMax normalize to [0,1]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_selected.reshape(-1, 1)).flatten()
    
    print(f"Step 4: Applied MinMax normalization to [0,1]")
    print(f"  Original samples: {len(power_sequence):,}")
    print(f"  Filtered samples: {len(x_scaled):,}")
    print(f"  Retention rate: {len(x_scaled)/len(power_sequence)*100:.2f}%")
    print(f"  Output range: [{x_scaled.min():.4f}, {x_scaled.max():.4f}]")
    print(f"{'='*60}\n")
    
    return x_scaled


def load_data(file_path):
    """Load data from .npy or .csv file and return data + original shape info"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        data = np.load(file_path)
        original_shape = data.shape
        print(f"Loaded .npy file: {data.shape}")
        
        # Flatten if multi-dimensional
        if len(data.shape) > 1:
            data = data.flatten()
            print(f"Flattened to: {data.shape}")
        
        return data, original_shape
    
    elif ext == '.csv':
        df = pd.read_csv(file_path)
        print(f"Loaded .csv file: {df.shape}")
        
        # Try to find power column
        if 'power' in df.columns:
            data = df['power'].values
            print(f"Using 'power' column")
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data = df[numeric_cols[0]].values
                print(f"Using column: {numeric_cols[0]}")
            else:
                raise ValueError("No numeric columns found in CSV")
        
        return data, None  # CSV has no original shape
    
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .npy or .csv")


def save_data(data, output_path, original_shape=None, window_size=512):
    """Save filtered data to CSV and NPY (reshaped to match original if possible)"""
    # Save CSV (always 1D)
    df = pd.DataFrame({'power': data})
    df.to_csv(output_path, index=False)
    print(f"✓ Saved filtered data to: {output_path}")
    print(f"  Rows: {len(df):,}")
    
    # Save NPY (reshape to windows if original was 3D)
    npy_path = output_path.replace('.csv', '.npy')
    
    if original_shape is not None and len(original_shape) == 3:
        # Original was 3D (N, seq_len, features)
        # Reshape filtered data back to windows
        seq_len = original_shape[1]
        n_features = original_shape[2]
        
        # Calculate how many complete windows we can make
        n_windows = len(data) // seq_len
        
        if n_windows > 0:
            # Trim to fit complete windows
            trimmed_data = data[:n_windows * seq_len]
            
            # Reshape to (N, seq_len, features)
            reshaped_data = trimmed_data.reshape(n_windows, seq_len, n_features)
            
            np.save(npy_path, reshaped_data)
            print(f"✓ Saved .npy file to: {npy_path}")
            print(f"  Shape: {reshaped_data.shape} (reshaped to match original format)")
            
            if len(data) > len(trimmed_data):
                print(f"  Note: Trimmed {len(data) - len(trimmed_data)} samples to fit complete windows")
        else:
            # Not enough data for even one window, save as 1D
            np.save(npy_path, data)
            print(f"✓ Saved .npy file to: {npy_path}")
            print(f"  Shape: {data.shape} (not enough data for window reshape)")
    else:
        # Original was 1D or CSV, save as 1D
        np.save(npy_path, data)
        print(f"✓ Saved .npy file to: {npy_path}")
        print(f"  Shape: {data.shape}")


def main():
    print("\n" + "="*60)
    print("SIMPLE DATA FILTER (Algorithm 1 Core Logic)")
    print("="*60)
    
    # Get file path
    print("\nEnter the path to your data file (.npy or .csv):")
    file_path = input("File path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return
    
    # Load data
    print(f"\nLoading data from: {file_path}")
    try:
        data, original_shape = load_data(file_path)
        print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"Total samples: {len(data):,}")
    except Exception as e:
        print(f"ERROR loading file: {e}")
        return
    
    # Check if data is normalized
    is_normalized = (data.min() >= -0.1 and data.max() <= 1.1)
    
    if is_normalized:
        print("\n⚠ Data appears to be NORMALIZED [0,1]")
        print("Algorithm 1 requires data in WATTS. Will denormalize first.")
        
        # Get appliance type for denormalization
        print("\nSelect appliance type:")
        print("  1. kettle (mean=700W, std=1000W)")
        print("  2. microwave (mean=500W, std=800W)")
        print("  3. fridge (mean=200W, std=400W)")
        print("  4. dishwasher (mean=700W, std=1000W)")
        print("  5. washingmachine (mean=400W, std=700W)")
        
        appliance_choice = input("Enter number (1-5): ").strip()
        
        appliance_params = {
            '1': ('kettle', 700, 1000, 200),
            '2': ('microwave', 500, 800, 200),
            '3': ('fridge', 200, 400, 50),
            '4': ('dishwasher', 700, 1000, 10),
            '5': ('washingmachine', 400, 700, 20),
        }
        
        if appliance_choice in appliance_params:
            name, mean, std, default_threshold = appliance_params[appliance_choice]
            print(f"\nSelected: {name}")
            print(f"  Mean: {mean}W, Std: {std}W")
            
            # Denormalize
            data = denormalize_data(data, mean, std)
            print(f"  Denormalized range: [{data.min():.2f}, {data.max():.2f}] W")
            
            # Use default threshold
            threshold = default_threshold
            print(f"  Using default threshold: {threshold}W")
        else:
            print("ERROR: Invalid choice. Exiting.")
            return
    else:
        print("\n✓ Data appears to be in WATTS (not normalized)")
        
        # Get threshold
        print("\nEnter power threshold for activation detection (in Watts):")
        print("  Examples: kettle=200, microwave=200, fridge=50, dishwasher=10, washingmachine=500")
        threshold_input = input("Threshold (W): ").strip()
        try:
            threshold = float(threshold_input)
        except:
            print("ERROR: Invalid threshold. Using default: 100W")
            threshold = 100.0
    
    # Get window length (optional)
    print("\nEnter window length (default: 100, press Enter to use default):")
    window_input = input("Window length: ").strip()
    if window_input:
        try:
            window = int(window_input)
        except:
            print("ERROR: Invalid window. Using default: 100")
            window = 100
    else:
        window = 100
    
    # Get noise threshold (optional)
    print("\nEnter noise threshold to remove low values (default: 0, press Enter to skip):")
    noise_input = input("Noise threshold (W): ").strip()
    if noise_input:
        try:
            noise_threshold = float(noise_input)
        except:
            print("ERROR: Invalid noise threshold. Using default: 0")
            noise_threshold = 0
    else:
        noise_threshold = 0
    
    # Filter data
    try:
        filtered_data = filter_data(data, threshold, window, noise_threshold)
        
        if len(filtered_data) == 0:
            print("ERROR: No data remaining after filtering!")
            print("TIP: Try lowering the threshold or check if data is in correct units")
            return
        
    except Exception as e:
        print(f"ERROR during filtering: {e}")
        return
    
    # Save output
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path) or '.'
    output_path = os.path.join(output_dir, f"{base_name}_filtered.csv")
    
    print(f"\nSaving to: {output_path}")
    try:
        save_data(filtered_data, output_path, original_shape)
        print("\n✓ FILTERING COMPLETE!")
    except Exception as e:
        print(f"ERROR saving file: {e}")


if __name__ == '__main__':
    main()
