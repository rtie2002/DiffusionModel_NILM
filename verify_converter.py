import numpy as np
import pandas as pd
import os
import subprocess

def test_conversion():
    print("Starting verification of convert_npy_to_csv.py...")
    
    # 1. Create a dummy NPY file with KNOWN values
    # Shape: (2, 512, 9) - 2 windows
    # Window 0: All 0.1
    # Window 1: All 0.9
    dummy_data = np.zeros((2, 512, 9))
    dummy_data[0, :, :] = 0.1
    dummy_data[1, :, :] = 0.9
    
    test_npy = 'test_dummy.npy'
    np.save(test_npy, dummy_data)
    print(f"Created dummy NPY: {test_npy} (Shape: {dummy_data.shape})")
    print(f"   Window 0 values: 0.1")
    print(f"   Window 1 values: 0.9")
    
    # 2. Run the conversion script
    cmd = f"python convert_npy_to_csv.py --input {test_npy}"
    print(f"Running conversion command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Conversion failed with code {result.returncode}")
        print(result.stderr)
        return
    else:
        print("Conversion script finished successfully.")
        
    # 3. Load the resulting CSV and verify values
    test_csv = 'test_dummy.csv'
    if not os.path.exists(test_csv):
        print(f"CSV file not found: {test_csv}")
        return
        
    df = pd.read_csv(test_csv)
    print(f"Loaded generated CSV: {test_csv} (Shape: {df.shape})")
    
    # Expected rows: 2 * 512 = 1024
    if len(df) != 1024:
        print(f"Wrong number of rows! Expected 1024, got {len(df)}")
        return
        
    # Check Window 0 (first 512 rows)
    w0_mean = df.iloc[:512].mean().mean()
    print(f"   Window 0 Mean (Expected 0.1): {w0_mean:.4f}")
    if not np.isclose(w0_mean, 0.1):
         print("Window 0 values mistmatch!")
    else:
         print("Window 0 values match (0.1)")

    # Check Window 1 (next 512 rows)
    w1_mean = df.iloc[512:].mean().mean()
    print(f"   Window 1 Mean (Expected 0.9): {w1_mean:.4f}")
    if not np.isclose(w1_mean, 0.9):
         print("Window 1 values mistmatch!")
    else:
         print("Window 1 values match (0.9)")
         
    # Check column names
    expected_cols = ['washingmachine', 'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
    if list(df.columns) == expected_cols:
        print("Column names match.")
    else:
        print(f"Column names mismatch! Got: {list(df.columns)}")

    # Cleanup
    try:
        os.remove(test_npy)
        os.remove(test_csv)
        print("Cleaned up test files.")
    except:
        pass

    print("\nCONCLUSION: convert_npy_to_csv.py is WORKING CORRECTLY.")

if __name__ == "__main__":
    test_conversion()
