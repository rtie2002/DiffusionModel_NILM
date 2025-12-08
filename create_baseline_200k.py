"""
Create 200k-only real data baseline for fair comparison
Matches paper's experimental setup
"""

import pandas as pd

appliances = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']

for appliance in appliances:
    print(f"\nProcessing {appliance}...")
    
    # Load original full CSV
    csv_path = f'NILM-main/dataset_preprocess/created_data/UK_DALE/{appliance}_training_.csv'
    df = pd.read_csv(csv_path, header=None)
    print(f"  Original size: {len(df):,} rows")
    
    # Extract first 200k rows
    df_200k = df.iloc[:200000]
    print(f"  Extracted: {len(df_200k):,} rows")
    
    # Save as baseline file
    output_path = f'NILM-main/dataset_preprocess/created_data/UK_DALE/{appliance}_training_200k_baseline.csv'
    df_200k.to_csv(output_path, header=False, index=False)
    print(f"  Saved to: {appliance}_training_200k_baseline.csv")

print("\n✅ All baseline datasets (200k real only) created!")
print("\nFor training:")
print("  1. Backup original file: copy {appliance}_training_.csv {appliance}_training_original.csv")
print("  2. Use baseline: copy {appliance}_training_200k_baseline.csv {appliance}_training_.csv")
print("  3. Train: python NILM-main/S2S_train.py --appliance_name {appliance}")
