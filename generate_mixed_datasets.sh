#!/bin/bash

# ==============================================================================
# Script: generate_mixed_datasets.sh
# Purpose: Generate combinations of real and synthetic NILM data
#          Strategy 1 (v2): Fixed window shuffling (w10, w50, w100, w600)
#          Strategy 2 (v3): Event-based injection (Evenly-spaced in OFF periods)
#          Per appliance: 1 baseline + 4 ordered + 16 shuffled + 4 v3 = 25
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# --- Self-Cleaning: Fix Windows Line Endings (\r) ---
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i 's/\r$//' mix_training_data_multivariate_v3.py 2>/dev/null
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

REAL_ROWS=200000

# Injection Cases (Synthetic Rows)
SYN_ROWS_CASES=(0 10000 20000 100000 200000 400000)

# Window Sizes for Shuffling (v2 strategy)
WINDOW_SIZES=(600 6000)

for APPLIANCE in "${APPLIANCES[@]}"; do
    echo "================================================================"
    echo "Starting dataset generation for $APPLIANCE..."
    echo "Real Rows: $REAL_ROWS (fixed)"
    echo "================================================================"

    for syn_rows in "${SYN_ROWS_CASES[@]}"; do
        
        # Determine if this is the baseline case
        if [ "$syn_rows" -eq 0 ]; then
            TAG="[BASELINE]"
            SUFFIX_TYPE="ordered"
        else
            TAG="[PROCESSING]"
            SUFFIX_TYPE="ordered"
        fi

        # 1. Ordered Case (or Baseline) — v2
        # ----------------------------------------------------------------
        REAL_K=$((REAL_ROWS / 1000))
        SYN_K=$((syn_rows / 1000))
        SUFFIX="${REAL_K}k+${SYN_K}k_${SUFFIX_TYPE}"
        
        echo "$TAG $APPLIANCE | Injection: ${REAL_K}k+${SYN_K}k | Mode: $SUFFIX_TYPE"
        
        python mix_training_data_multivariate_v2.py \
            --appliance "$APPLIANCE" \
            --real_rows $REAL_ROWS \
            --synthetic_rows $syn_rows \
            --suffix "$SUFFIX" \
            --no-shuffle

        # Skip all shuffled/event modes if no synthetic data (Baseline)
        if [ "$syn_rows" -gt 0 ]; then

            # 2. Shuffled Cases — v2 (Various Window Sizes)
            # ----------------------------------------------------------------
            for window in "${WINDOW_SIZES[@]}"; do
                # A) Partial Shuffle (Real kept order)
                echo "[v2 Partial Shuffle] $APPLIANCE | Window: $window | Injection: ${REAL_K}k+${SYN_K}k"
                SUFFIX="${REAL_K}k+${SYN_K}k_shuffled_w${window}"
                
                python mix_training_data_multivariate_v2.py \
                    --appliance "$APPLIANCE" \
                    --real_rows $REAL_ROWS \
                    --synthetic_rows $syn_rows \
                    --suffix "$SUFFIX" \
                    --shuffle \
                    --window_size $window

                # B) Full Shuffle (Everything randomized)
                echo "[v2 Full Shuffle] $APPLIANCE | Window: $window | Injection: ${REAL_K}k+${SYN_K}k"
                SUFFIX="${REAL_K}k+${SYN_K}k_full_shuffled_w${window}"
                
                python mix_training_data_multivariate_v2.py \
                    --appliance "$APPLIANCE" \
                    --real_rows $REAL_ROWS \
                    --synthetic_rows $syn_rows \
                    --suffix "$SUFFIX" \
                    --full-shuffle \
                    --window_size $window
            done

            # 3. Event-Based Injection (v3) — Evenly Spaced (OFF periods only)
            # ----------------------------------------------------------------
            SUFFIX="${REAL_K}k+${SYN_K}k_event_even_v3"
            echo "[v3 Event Even] $APPLIANCE | Injection: ${REAL_K}k+${SYN_K}k"

            python mix_training_data_multivariate_v3.py \
                --appliance "$APPLIANCE" \
                --real_rows $REAL_ROWS \
                --synthetic_rows $syn_rows \
                --suffix "$SUFFIX"

        else
            echo "[Skipping] Shuffled/Event modes for $APPLIANCE (Zero synthetic rows)"
        fi
    done
done

echo "================================================================"
echo "Done! Generated all requested cases (v2 + v3)."
