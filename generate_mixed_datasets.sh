#!/bin/bash

# ==============================================================================
# Script: generate_mixed_datasets.sh
# Purpose: Generate 25 combinations of real and synthetic NILM data
#          (5 injection ratios x 5 shuffle/window configurations)
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Appliances to process
if [ "$1" == "all" ] || [ -z "$1" ]; then
    APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
    echo "Processing ALL appliances: ${APPLIANCES[*]}"
else
    APPLIANCES=("$1")
    echo "Processing single appliance: $1"
fi

REAL_ROWS=200000

# Injection Cases (Synthetic Rows)
SYN_ROWS_CASES=(0 20000 100000 200000 400000)

# Window Sizes for Shuffling
WINDOW_SIZES=(10 50 100 600)

for APPLIANCE in "${APPLIANCES[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Starting dataset generation for $APPLIANCE..."
    echo "Real Rows: $REAL_ROWS (fixed)"
    echo "----------------------------------------------------------------"

    for syn_rows in "${SYN_ROWS_CASES[@]}"; do
        
        # 1. Ordered Case (No Shuffle)
        # ----------------------------------------------------------------
        echo "[Processing] $APPLIANCE | Injection: ${REAL_ROWS}+${syn_rows} | Mode: Ordered"
        
        # Generate suffix like: 200k+20k_ordered
        REAL_K=$((REAL_ROWS / 1000))
        SYN_K=$((syn_rows / 1000))
        SUFFIX="${REAL_K}k+${SYN_K}k_ordered"
        
        python mix_training_data_multivariate.py \
            --appliance "$APPLIANCE" \
            --real_rows $REAL_ROWS \
            --synthetic_rows $syn_rows \
            --suffix "$SUFFIX" \
            --no-shuffle

        # 2. Shuffled Cases (Various Window Sizes)
        # Skip shuffling if there is no synthetic data (Baseline Case)
        if [ "$syn_rows" -gt 0 ]; then
            for window in "${WINDOW_SIZES[@]}"; do
                echo "[Processing] $APPLIANCE | Mode: Shuffled | Window: $window | Injection: ${REAL_ROWS}+${syn_rows}"
                
                # Generate suffix like: 200k+20k_shuffled_w10
                SUFFIX="${REAL_K}k+${SYN_K}k_shuffled_w${window}"
                
                python mix_training_data_multivariate_v2.py \
                    --appliance "$APPLIANCE" \
                    --real_rows $REAL_ROWS \
                    --synthetic_rows $syn_rows \
                    --suffix "$SUFFIX" \
                    --shuffle \
                    --window_size $window
            done
        else
            echo "[Skipping] Shuffled modes for $APPLIANCE (Zero synthetic rows)"
        fi
    done
done

echo "----------------------------------------------------------------"
echo "Done! Generated all requested cases."
