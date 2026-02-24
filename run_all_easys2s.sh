#!/bin/bash

# ==============================================================================
# Script: run_all_easys2s.sh
# Purpose: Automate training + testing for all 25 dataset combinations,
#          for all 5 appliances (125 experiments total).
#          Includes v2 (window w50+) and v3 (evenly-spaced events).
#          Displays a formatted MAE summary table at the end.
#          Uses CLI arguments for non-brittle parameter passing.
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# --- Self-Cleaning: Fix Windows Line Endings (\r) ---
# If this script was edited on Windows, it might contain \r which breaks bash.
# This line will clean itself and the companion scripts on the fly.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i 's/\r$//' generate_mixed_datasets.sh 2>/dev/null
    sed -i 's/\r$//' mix_training_data_multivariate_v3.py 2>/dev/null
    sed -i 's/\r$//' "$0" 2>/dev/null
fi

# Use the full Python path from the nilm_main conda env
PYTHON="/home/raymond/miniconda3/envs/nilm_main/bin/python3"

# Sanity check
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

# Fix: libdevice.10.bc for TF XLA
TRITON_LIBDEVICE="$($PYTHON -c 'import triton; import os; print(os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/lib/libdevice.10.bc"))' 2>/dev/null)"
if [ -f "$TRITON_LIBDEVICE" ]; then
    CUDA_DATA_DIR="/tmp/xla_cuda_data_$$"
    mkdir -p "$CUDA_DATA_DIR/nvvm/libdevice"
    ln -sf "$TRITON_LIBDEVICE" "$CUDA_DATA_DIR/nvvm/libdevice/libdevice.10.bc"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_DATA_DIR"
fi

export TF_CPP_MIN_LOG_LEVEL=2

# --- Appliances ---
# Fix: If user types 'al', assume they meant 'all'
USER_INPUT="$1"
if [ "$USER_INPUT" == "al" ]; then USER_INPUT="all"; fi

if [ "$USER_INPUT" == "all" ] || [ -z "$USER_INPUT" ]; then
    APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
    echo "Targeting ALL appliances: ${APPLIANCES[*]}"
else
    APPLIANCES=("$USER_INPUT")
    echo "Targeting single appliance: $USER_INPUT"
fi

# --- Experiment Parameters ---
REAL_K="200k"
SYN_K_CASES=("0k" "20k" "100k" "200k" "400k")
WINDOW_SIZES=("50" "100" "6000") # w10 removed for execution
EPOCHS=100
BATCH_SIZE=2048   # Increased for speed on RTX 4090
TRAIN_PERCENT="20"

# --- Paths ---
DATA_DIR="$PROJECT_ROOT/created_data/UK_DALE/"
NILM_DIR="$PROJECT_ROOT/NILM-main"
MODELS_ROOT="$NILM_DIR/models/EasyS2S/UK_DALE"

# --- Result storage ---
declare -A RESULTS
declare -a CONFIG_ORDER

# --- Helper: run one train+test experiment, capture MAE ---
run_experiment() {
    local app=$1
    local train_filename=$2
    local origin_model=$3
    local config_key=$4

    local model_path="$MODELS_ROOT/${train_filename}_model"
    local weight_file="${model_path}_weights.h5"

    echo ""
    echo "------------------------------------------------------------"
    echo " APP: $app | $config_key | origin_model=$origin_model"
    echo "------------------------------------------------------------"

    # --- SKIP if already done ---
    if [ -f "$weight_file" ]; then
        echo "[RESUME] Found existing model: $weight_file. Skipping training."
    else
        # --- TRAIN ---
        echo "[TRAIN]"
        cd "$NILM_DIR"
        $PYTHON EasyS2S_train.py \
            --appliance_name "$app" \
            --n_epoch $EPOCHS \
            --batchsize $BATCH_SIZE \
            --datadir "$DATA_DIR" \
            --train_filename "$train_filename" \
            --origin_model "$origin_model" \
            --dataset_name "UK_DALE" \
            --train_percent "$TRAIN_PERCENT"

        if [ $? -ne 0 ]; then
            echo "ERROR: Training failed. Skipping test."
            RESULTS["${config_key}|${app}"]="FAIL"
            cd "$PROJECT_ROOT"; return
        fi
    fi

    # --- TEST ---
    echo "[TEST]"
    TMP_TEST_LOG="/tmp/easys2s_test_$$.log"
    cd "$NILM_DIR"
    $PYTHON EasyS2S_test.py \
        --appliance_name "$app" \
        --datadir "$DATA_DIR" \
        --train_filename "$train_filename" \
        --origin_model "$origin_model" \
        --dataset_name "UK_DALE" \
        --train_percent "$TRAIN_PERCENT" 2>&1 | tee "$TMP_TEST_LOG"
    
    TEST_EXIT=${PIPESTATUS[0]}
    MAE=$(grep -oP "MAE:\s*\K[0-9]*\.?[0-9]+" "$TMP_TEST_LOG" | head -1)
    rm -f "$TMP_TEST_LOG"

    if [ $TEST_EXIT -ne 0 ] && [ -z "$MAE" ]; then
        RESULTS["${config_key}|${app}"]="FAIL"
    else
        RESULTS["${config_key}|${app}"]="${MAE:-N/A}"
    fi

    cd "$PROJECT_ROOT"
}

# --- Helper: Print Table ---
print_summary_table() {
    local col_w=14
    echo ""
    echo "================================================================"
    echo "  MAE SUMMARY TABLE (Current Progress)"
    echo "================================================================"
    local header="Configuration                  "
    for local_app in "${APPLIANCES[@]}"; do
        printf -v col "%-${col_w}s" "${local_app}"; header+="| ${col}"
    done
    echo "$header"
    local sep=""; local sep_len=${#header}; printf -v sep '%*s' "$sep_len" ''; echo "${sep// /-}"
    for config_key in "${CONFIG_ORDER[@]}"; do
        printf "%-31s" "$config_key"
        for local_app in "${APPLIANCES[@]}"; do
            local val="${RESULTS["${config_key}|${local_app}"]:-...}"
            if [[ "$val" =~ ^[0-9]*\.?[0-9]+$ ]]; then printf "| %-14.2f" "$val"
            else printf "| %-14s" "$val"; fi
        done
        echo ""
    done
    echo "${sep// /-}"; echo ""
}

# Pre-populate CONFIG_ORDER for a consistent table structure
for syn_k in "${SYN_K_CASES[@]}"; do
    if [ "$syn_k" == "0k" ]; then
        CONFIG_ORDER+=("200k+${syn_k} | Baseline    ")
    else
        CONFIG_ORDER+=("200k+${syn_k} | Ordered     ")
        for window in "${WINDOW_SIZES[@]}"; do
            CONFIG_ORDER+=("200k+${syn_k} | Shuffled w${window} ")
        done
        CONFIG_ORDER+=("200k+${syn_k} | Event Even  ")
    fi
done

# Main Loop: Process one appliance fully before moving to the next
for app in "${APPLIANCES[@]}"; do
    echo ">>> STARTING FULL EXPERIMENT SET FOR APPLIANCE: $app <<<"
    for syn_k in "${SYN_K_CASES[@]}"; do
        ORIGIN_MODEL="False"; [ "$syn_k" == "0k" ] && ORIGIN_MODEL="True"

        # 1. Ordered Case (or Baseline)
        CONFIG_KEY="200k+${syn_k} | Ordered     "
        [ "$syn_k" == "0k" ] && CONFIG_KEY="200k+${syn_k} | Baseline    "
        
        TRAIN_SUFFIX="ordered"
        TRAIN_FILENAME="${app}_training_${REAL_K}+${syn_k}_${TRAIN_SUFFIX}"
        run_experiment "$app" "$TRAIN_FILENAME" "$ORIGIN_MODEL" "$CONFIG_KEY"
        print_summary_table

        # 2. Shuffled/Event Cases
        if [ "$syn_k" != "0k" ]; then
            # v2: Shuffled Windows
            for window in "${WINDOW_SIZES[@]}"; do
                CONFIG_KEY="200k+${syn_k} | Shuffled w${window} "
                TRAIN_FILENAME="${app}_training_${REAL_K}+${syn_k}_shuffled_w${window}"
                run_experiment "$app" "$TRAIN_FILENAME" "$ORIGIN_MODEL" "$CONFIG_KEY"
                print_summary_table
            done

            # v3: Event-Based Even (OFF periods only)
            CONFIG_KEY="200k+${syn_k} | Event Even  "
            TRAIN_FILENAME="${app}_training_${REAL_K}+${syn_k}_event_even_v3"
            run_experiment "$app" "$TRAIN_FILENAME" "$ORIGIN_MODEL" "$CONFIG_KEY"
            print_summary_table
        fi
    done
done

print_summary_table
echo "DONE: All experiments finished."
