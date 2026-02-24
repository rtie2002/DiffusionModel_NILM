#!/bin/bash

# ==============================================================================
# Script: generate_mixed_datasets_v3.sh
# Purpose: Generate dataset combinations using Event-Based Injection (v3)
#          Unlike v2 which uses fixed window sizes (w10, w50, w100, w600),
#          v3 detects complete ON-period events and injects them atomically.
#
#          Configurations per appliance:
#            1. Baseline (200k+0k, no synthetic data)
#            2. Ordered  (200k+Xk, synthetic appended, no shuffle)
#            3. Event-shuffled (200k+Xk, events injected randomly)
#          = 1 + 4*2 = 9 configurations per appliance
#          = 9 * 5 = 45 experiments total
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

for APPLIANCE in "${APPLIANCES[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Starting v3 event-based dataset generation for $APPLIANCE..."
    echo "Real Rows: $REAL_ROWS (fixed)"
    echo "----------------------------------------------------------------"

    for syn_rows in "${SYN_ROWS_CASES[@]}"; do
        
        REAL_K=$((REAL_ROWS / 1000))
        SYN_K=$((syn_rows / 1000))

        # 1. Ordered Case (or Baseline)
        # ----------------------------------------------------------------
        SUFFIX="${REAL_K}k+${SYN_K}k_ordered"
        
        if [ "$syn_rows" -eq 0 ]; then
            TAG="[BASELINE]"
        else
            TAG="[ORDERED]"
        fi
        
        echo "$TAG $APPLIANCE | Injection: ${REAL_K}k+${SYN_K}k | Mode: ordered"
        
        python mix_training_data_multivariate_v3.py \
            --appliance "$APPLIANCE" \
            --real_rows $REAL_ROWS \
            --synthetic_rows $syn_rows \
            --suffix "$SUFFIX" \
            --no-shuffle

        # 2. Event-Shuffled Case (skip for baseline)
        # ----------------------------------------------------------------
        if [ "$syn_rows" -gt 0 ]; then
            SUFFIX="${REAL_K}k+${SYN_K}k_event_shuffled"
            
            echo "[EVENT-SHUFFLED] $APPLIANCE | Injection: ${REAL_K}k+${SYN_K}k | Mode: event_shuffled"
            
            python mix_training_data_multivariate_v3.py \
                --appliance "$APPLIANCE" \
                --real_rows $REAL_ROWS \
                --synthetic_rows $syn_rows \
                --suffix "$SUFFIX" \
                --shuffle
        else
            echo "[Skipping] Event-shuffled mode for $APPLIANCE (Zero synthetic rows)"
        fi
    done
done

echo "----------------------------------------------------------------"
echo "Done! Generated all v3 event-based dataset cases."
