#!/bin/bash

# Set working directory to project root to ensure python script relative paths work
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="$PROJECT_ROOT/created_data/UK_DALE"

echo "Working Directory set to: $PWD"
echo "Data Directory: $DATA_DIR"

# ==============================================================================
# AUTO-FIX: Expand Synthetic Data Files
# The mixing tool caps output at the shortest synthetic file length.
# We must ensure all source NPY files are large enough (e.g. > 2500 windows)
# to support 100% data injection scenarios.
# ==============================================================================
TEMP_EXPAND_SCRIPT="temp_expand_synthetic.py"
cat <<EOF > "$TEMP_EXPAND_SCRIPT"
import numpy as np
import glob
import os
import shutil

# Target size: 2500 windows * 512 points = 1.28M points (enough for >100% of ~1M real data)
TARGET_WINDOWS = 2500 

search_path = os.path.join('synthetic_data_multivariate', 'ddpm_fake_*_multivariate.npy')
files = glob.glob(search_path)

print(f'\n[Auto-Expand] Checking {len(files)} synthetic files...')

for f in files:
    try:
        data = np.load(f)
        if data.shape[0] < TARGET_WINDOWS:
            print(f'  Expanding {os.path.basename(f)}: {data.shape[0]} windows -> {TARGET_WINDOWS} windows')
            
            # Backup original if not exists
            if not os.path.exists(f + '.bak'):
                shutil.copy2(f, f + '.bak')
            
            # Tile/Repeat data to reach target size
            repeats = int(np.ceil(TARGET_WINDOWS / data.shape[0]))
            new_data = np.tile(data, (repeats, 1, 1))[:TARGET_WINDOWS]
            
            np.save(f, new_data)
            print('  -> Expanded and saved.')
        else:
            # print(f'  {os.path.basename(f)} is OK ({data.shape[0]} windows)')
            pass
            
    except Exception as e:
        print(f'  Error processing {f}: {e}')
print('[Auto-Expand] Check complete.\n')
EOF

# Write and execute temp script
python "$TEMP_EXPAND_SCRIPT"
if [ -f "$TEMP_EXPAND_SCRIPT" ]; then rm "$TEMP_EXPAND_SCRIPT"; fi
# ==============================================================================

run_mix_ratios() {
    local appliance=$1
    local filename=$2
    local full_path="$DATA_DIR/$filename"
    
    if [ -f "$full_path" ]; then
        echo -e "\n----------------------------------------------------------------"
        echo "Processing $appliance (Path: $full_path)..."
        
        # Count lines efficiently
        echo "Counting rows in real data file..."
        # Using wc -l which counts newline characters
        local line_count=$(wc -l < "$full_path")
        
        # Subtract 1 for header
        local total_real_points=$((line_count - 1))
        
        echo "Total Real Data Points: $total_real_points"
        
        # Different Ratios of Synthetic Data
        local ratios=("0.25" "0.50" "1.00" "2.00")
        
        for ratio in "${ratios[@]}"; do
            # Calculate synthetic rows and percentage using python for accuracy
            local syn_rows=$(python -c "import math; print(math.floor($total_real_points * $ratio))")
            local percent=$(python -c "print(int(float($ratio) * 100))")
            
            echo -e "\n  [Scenario: Synthetic Data = ${percent}% of Real Data]"
            echo "  > Real Rows: $total_real_points"
            echo "  > Synthetic Rows: $syn_rows"
            
            # Execute python script with specific suffix
            local suffix="synthetic_${percent}%"
            python mix_training_data_multivariate.py --appliance "$appliance" --real_rows "$total_real_points" --synthetic_rows "$syn_rows" --real_path "$full_path" --suffix "$suffix"
        done
    else
        echo "WARNING: Data file not found for $appliance at $full_path"
    fi
}

# Run for all appliances
run_mix_ratios "fridge" "fridge_training_.csv"
run_mix_ratios "microwave" "microwave_training_.csv"
run_mix_ratios "kettle" "kettle_training_.csv"
run_mix_ratios "dishwasher" "dishwasher_training_.csv"
run_mix_ratios "washingmachine" "washingmachine_training_.csv"
