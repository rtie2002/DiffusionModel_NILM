import numpy as np
import pandas as pd

# Load NPY file
print("Loading NPY file...")
data = np.load('ddpm_fake_washingmachine_multivariate.npy')
print(f"Shape: {data.shape}")  # (400, 512, 9)

# Reshape to 2D
print("Reshaping to 2D...")
data_2d = data.reshape(-1, 9)  # (204800, 9)
print(f"Reshaped to: {data_2d.shape}")

# Create DataFrame with column names
print("Creating DataFrame...")
df = pd.DataFrame(data_2d, columns=[
    'washingmachine',
    'minute_sin', 'minute_cos',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos'
])

# Save to CSV
output_file = 'ddpm_fake_washingmachine_multivariate.csv'
print(f"Saving to {output_file}...")
df.to_csv(output_file, index=False)

print(f"\n✓ Conversion complete!")
print(f"  Input:  ddpm_fake_washingmachine_multivariate.npy (400, 512, 9)")
print(f"  Output: {output_file} ({len(df)} rows, 9 columns)")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData statistics:")
print(df.describe())
