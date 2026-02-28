#!/bin/bash

# ==============================================================================
# Script: generate_ratio_mixed_datasets.sh
# Purpose: Generate combinations of real and synthetic NILM data using 
#          specific ratios relative to the real dataset (200k rows).
#
# Ratios: 5%, 10%, 50%, 100%, 200% of Synthetic data added to 100% Real data.
# Strategies: 
#   1. Ordered (v2)
#   2. Shuffled w/ Various Windows (v2 - Partial & Full)
#   3. Event-based Injection (v3)
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# --- Self-Cleaning: Fix Windows Line Endings (\r) ---
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i 's/\r$//' mix_training_data_multivariate_v3.py 2>/dev/null
    sed -i 's/\r$//' mix_training_data_multivariate_v2.py 2>/dev/null
    sed -i 's/\r$//' "$0" 2>/dev/null
fi

# Appliances to process
if [ "$1" == "all" ] || [ -z "$1" ]; then
    APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
    echo "Processing ALL appliances: ${APPLIANCES[*]}"
else
    APPLIANCES=("$1")
    echo "Processing single appliance: $1"
fi

for APPLIANCE in "${APPLIANCES[@]}"; do
    echo "================================================================"
    echo "Starting dynamic ratio-based generation for $APPLIANCE..."
    echo "================================================================"

    # 1. Determine REAL_ROWS (100% of available)
    REAL_FILE="created_data/UK_DALE/${APPLIANCE}_training_.csv"
    if [ ! -f "$REAL_FILE" ]; then
        echo "Error: Real data file not found: $REAL_FILE"
        continue
    fi
    # Subtract 1 for header
    TOTAL_REAL_ROWS=$(($(wc -l < "$REAL_FILE") - 1))
    REAL_ROWS=$TOTAL_REAL_ROWS

    # 2. Determine Max Available Synthetic Rows from NPY
    # Assuming standard synthetic directory structure from mix_training_data_multivariate_v2.py
    SYN_DIR="synthetic_data_multivariate"
    SYN_FILE="$SYN_DIR/ddpm_fake_${APPLIANCE}_multivariate.npy"
    if [ ! -f "$SYN_FILE" ]; then
        # Check fallback path
        SYN_FILE="OUTPUT/${APPLIANCE}_512/ddpm_fake_${APPLIANCE}_512.npy"
    fi

    if [ -f "$SYN_FILE" ]; then
        # Get flattened length (assuming shape [N, 512, 9] or similar where power is col 0)
        # mix_data_v2 uses data[:, :, 0].reshape(-1)
        MAX_SYN_ROWS=$(python -c "import numpy as np; data=np.load('$SYN_FILE'); print(data.shape[0]*data.shape[1])")
    else
        echo "Warning: Synthetic NPY not found for $APPLIANCE. Ratios will fail if > 0."
        MAX_SYN_ROWS=0
    fi

    echo "Appliance: $APPLIANCE"
    echo "Real Count (100%): $REAL_ROWS"
    echo "Max Synthetic Count: $MAX_SYN_ROWS"

    # Define Ratios
    RATIOS=(0 0.05 0.1 0.5 1.0 2.0)

    for ratio in "${RATIOS[@]}"; do
        
        # Calculate syn_rows based on REAL_ROWS
        # Using python for float multiplication
        syn_rows=$(python -c "print(int($REAL_ROWS * $ratio))")
        
        # Determine Ratio Label
        if [ "$(echo "$ratio == 0" | bc -l)" -eq 1 ]; then
            PCT="0pct"
            TAG="[BASELINE]"
        elif [ "$(echo "$ratio == 0.05" | bc -l)" -eq 1 ]; then PCT="5pct"; TAG="[PROCESSING]"
        elif [ "$(echo "$ratio == 0.1" | bc -l)" -eq 1 ]; then PCT="10pct"; TAG="[PROCESSING]"
        elif [ "$(echo "$ratio == 0.5" | bc -l)" -eq 1 ]; then PCT="50pct"; TAG="[PROCESSING]"
        elif [ "$(echo "$ratio == 1.0" | bc -l)" -eq 1 ]; then PCT="100pct"; TAG="[PROCESSING]"
        elif [ "$(echo "$ratio == 2.0" | bc -l)" -eq 1 ]; then PCT="200pct"; TAG="[PROCESSING]"
        else PCT="${ratio}ratio"; TAG="[PROCESSING]"
        fi

        # Cap syn_rows at MAX_SYN_ROWS
        if [ "$syn_rows" -gt "$MAX_SYN_ROWS" ]; then
            echo "Note: Ratio $PCT for $APPLIANCE capped at $MAX_SYN_ROWS rows (Actual available)"
            syn_rows=$MAX_SYN_ROWS
        fi

        REAL_K=$((REAL_ROWS / 1000))
        SYN_K=$((syn_rows / 1000))

        # 1. Ordered Case (or Baseline) — v2
        SUFFIX="${REAL_K}k+${SYN_K}k_${PCT}_ordered"
        echo "$TAG $APPLIANCE | Ratio: $PCT | Rows: ${REAL_K}k real + ${SYN_K}k syn"
        
        python mix_training_data_multivariate_v2.py \
            --appliance "$APPLIANCE" \
            --real_rows $REAL_ROWS \
            --synthetic_rows $syn_rows \
            --suffix "$SUFFIX" \
            --no-shuffle

        # Skip all shuffled/event modes if no synthetic data
        if [ "$syn_rows" -gt 0 ]; then
            for window in "${WINDOW_SIZES[@]}"; do
                # A) Partial Shuffle
                echo "[v2 Partial Shuffle] $APPLIANCE | Window: $window | Ratio: $PCT"
                SUFFIX="${REAL_K}k+${SYN_K}k_${PCT}_shuffled_w${window}"
                python mix_training_data_multivariate_v2.py --appliance "$APPLIANCE" --real_rows $REAL_ROWS --synthetic_rows $syn_rows --suffix "$SUFFIX" --shuffle --window_size $window

                # B) Full Shuffle
                echo "[v2 Full Shuffle] $APPLIANCE | Window: $window | Ratio: $PCT"
                SUFFIX="${REAL_K}k+${SYN_K}k_${PCT}_full_shuffled_w${window}"
                python mix_training_data_multivariate_v2.py --appliance "$APPLIANCE" --real_rows $REAL_ROWS --synthetic_rows $syn_rows --suffix "$SUFFIX" --full-shuffle --window_size $window
            done

            # 3. Event-Based Injection (v3)
            SUFFIX="${REAL_K}k+${SYN_K}k_${PCT}_event_even_v3"
            echo "[v3 Event Even] $APPLIANCE | Ratio: $PCT"
            python mix_training_data_multivariate_v3.py --appliance "$APPLIANCE" --real_rows $REAL_ROWS --synthetic_rows $syn_rows --suffix "$SUFFIX"
        fi
    done
done

echo "================================================================"
echo "Done! Ratio-based datasets generated (v2 + v3)."
