import pandas as pd
import numpy as np
import argparse
import os
import yaml
from tqdm import tqdm

# ==============================================================================
# CSV MINMAX [0,1] TO Z-SCORE CONVERTER
# ==============================================================================
# Converts CSV power data from [0, 1] range to Z-score using config stats.
# ==============================================================================

APPLIANCES = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']

def load_config():
    # Relative path from root
    config_path = 'Config/preprocess/preprocess_multivariate.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️ Warning: Could not find config at {config_path}. Using fallback defaults may fail.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert CSV MinMax to Z-Score')
    parser.add_argument('--input', type=str, help='Path to MinMax [0,1] CSV file')
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        print("\n" + "=" * 60)
        print("CSV MINMAX [0,1] TO Z-SCORE CONVERTER")
        print("=" * 60)
        input_path = input("Please enter the path to your MinMax CSV file: ").strip()
    
    # Path cleaning
    input_path = input_path.strip('"').strip("'")
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return

    # 1. Detect Appliance from filename
    filename = os.path.basename(input_path).lower()
    appliance = next((a for a in APPLIANCES if a in filename), None)
    if not appliance:
        print("❌ Could not detect appliance from filename. Ensure filename contains kettle, fridge, etc.")
        return

    # 2. Load Config Params
    config = load_config()
    if not config or appliance not in config['appliances']:
        print(f"❌ Error: Could not find parameters for '{appliance}' in config.")
        return

    params = config['appliances'][appliance]
    mean = params['mean']
    std = params['std']
    max_power = params['max_power']

    print(f"🚀 Processing: {appliance.upper()}")
    print(f"📊 Stats: MaxPower={max_power}W | Mean={mean}W | Std={std}W")

    # 3. Load Data
    df = pd.read_csv(input_path)
    
    # Identify the power column (should be the appliance name or 'power')
    power_col = appliance if appliance in df.columns else (df.columns[0] if 'power' not in df.columns else 'power')
    
    # 4. Conversion Logic
    # Step A: MinMax [0,1] -> Watts
    watts = df[power_col] * max_power
    
    # Step B: Watts -> Z-Score
    zscore = (watts - mean) / std
    
    # 5. Create Output DataFrame
    df_output = df.copy()
    df_output[power_col] = zscore
    
    # Rename column to generic 'power' or keep as is? 
    # Usually for training its better to keep it as the appliance name or 'power'.
    print(f"✨ Converted range: {zscore.min():.2f} to {zscore.max():.2f}")

    # 6. Save Result
    output_path = os.path.splitext(input_path)[0] + '_zscore.csv'
    df_output.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS: Saved Z-score CSV to: {os.path.basename(output_path)}")

if __name__ == "__main__":
    main()
