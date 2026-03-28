#!/usr/bin/env bash

# ------------------------------------------------------------
# mix_training_data_multivariate.sh
# Bash version of the PowerShell mixing script, using the V2 Python implementation.
# ------------------------------------------------------------

# Get the directory where this script resides – this is the project root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || { echo "Failed to cd to project root"; exit 1; }

DATA_DIR="$PROJECT_ROOT/created_data/UK_DALE"

echo "Working directory set to: $PROJECT_ROOT"
echo "Data directory: $DATA_DIR"

# ------------------------------------------------------------------
# 1) Auto‑expand synthetic NPY files if they are shorter than the target.
# ------------------------------------------------------------------
TEMP_EXPAND_SCRIPT="temp_expand_synthetic.py"
cat > "$TEMP_EXPAND_SCRIPT" << 'PYEOF'
import numpy as np, glob, os, shutil

TARGET_WINDOWS = 2500  # 2500 windows * 512 points ≈ 1.28M points
search_path = os.path.join('synthetic_data_multivariate', '*.npy')
files = glob.glob(search_path)

print(f"\n[Auto‑Expand] Checking {len(files)} synthetic files...")

for f in files:
    try:
        data = np.load(f)
        if data.shape[0] < TARGET_WINDOWS:
            print(f"  Expanding {os.path.basename(f)}: {data.shape[0]} → {TARGET_WINDOWS}")
            # backup original if not already backed up
            if not os.path.exists(f + '.bak'):
                shutil.copy2(f, f + '.bak')
            repeats = int(np.ceil(TARGET_WINDOWS / data.shape[0]))
            new_data = np.tile(data, (repeats, 1, 1))[:TARGET_WINDOWS]
            np.save(f, new_data)
            print('  -> Expanded and saved.')
    except Exception as e:
        print(f'  Error processing {f}: {e}')
print('[Auto‑Expand] Done.\n')
PYEOF

python "$TEMP_EXPAND_SCRIPT"
rm -f "$TEMP_EXPAND_SCRIPT"

# ------------------------------------------------------------------
# 2) Define the synthetic‑to‑real ratios we want to generate.
# ------------------------------------------------------------------
ratios=(0.25 0.50 1.00 2.00)

# ------------------------------------------------------------------
# 3) Core mixing function – called for each appliance.
# ------------------------------------------------------------------
run_mix_ratios() {
    local appliance="$1"
    local filename="$2"
    local real_path="$DATA_DIR/$filename"

    if [[ ! -f "$real_path" ]]; then
        echo "WARNING: Data file not found for $appliance at $real_path"
        return
    fi

    echo "\n----------------------------------------------------------------"
    echo "Processing $appliance (Path: $real_path)..."

    # Count lines (including header) – wc -l is fast.
    local line_count=$(wc -l < "$real_path")
    local total_real_points=$((line_count - 1))
    echo "Total Real Data Points (rows): $total_real_points"

    for ratio in "${ratios[@]}"; do
        # synthetic rows = floor(real_rows * ratio)
        local syn_rows=$(python - <<END
import math, sys
ratio = $ratio
real = $total_real_points
print(math.floor(real * ratio))
END
)
        local percent=$(python - <<END
print(int($ratio * 100))
END
)
        local suffix="synthetic_${percent}%"
        echo "\n  [Scenario] Synthetic = ${percent}% of Real"
        echo "   Real rows      : $total_real_points"
        echo "   Synthetic rows : $syn_rows"

        python mix_training_data_multivariate.py \
            --appliance "$appliance" \
            --real_rows "$total_real_points" \
            --synthetic_rows "$syn_rows" \
            --real_path "$real_path" \
            --suffix "$suffix"
    done
}

# ------------------------------------------------------------------
# 4) Run for each appliance.
# ------------------------------------------------------------------
run_mix_ratios "fridge"       "fridge_training_.csv"
run_mix_ratios "microwave"    "microwave_training_.csv"
run_mix_ratios "kettle"       "kettle_training_.csv"
run_mix_ratios "dishwasher"   "dishwasher_training_.csv"
run_mix_ratios "washingmachine" "washingmachine_training_.csv"

# End of script
