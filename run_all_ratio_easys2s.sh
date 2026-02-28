#!/bin/bash

# ==============================================================================
# Script: run_all_ratio_easys2s.sh
# Purpose: Automate training + testing for specific ratio-named datasets
#          Naming logic: appliance_training_100%+[5,10,50,100,200]%.csv
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# --- Self-Cleaning: Fix Windows Line Endings ---
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i 's/\r$//' "$0" 2>/dev/null
fi

# Use the full Python path from the nilm_main conda env
PYTHON="/home/raymond/miniconda3/envs/nilm_main/bin/python3"

# Appliances
USER_INPUT="$1"
if [ "$USER_INPUT" == "al" ]; then USER_INPUT="all"; fi
if [ "$USER_INPUT" == "all" ] || [ -z "$USER_INPUT" ]; then
    APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
    echo "Targeting ALL appliances: ${APPLIANCES[*]}"
else
    APPLIANCES=("$USER_INPUT")
    echo "Targeting single appliance: $USER_INPUT"
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

# Ratios and Modes
RATIO_LABELS=("0pct" "5pct" "10pct" "50pct" "100pct" "200pct")
WINDOW_SIZES=("600" "6000")
EPOCHS=100
BATCH_SIZE=2048
TRAIN_PERCENT="20"

# Paths
DATA_DIR="$PROJECT_ROOT/created_data/UK_DALE/"
NILM_DIR="$PROJECT_ROOT/NILM-main"
MODELS_ROOT="$NILM_DIR/models/EasyS2S/UK_DALE"

declare -A RESULTS
declare -a CONFIG_ORDER

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

run_experiment() {
    local app=$1
    local train_filename=$2
    local origin_model=$3
    local config_key=$4

    local model_path="$MODELS_ROOT/${train_filename}_model"
    local weight_file="${model_path}_weights.h5"

    echo ""
    echo "------------------------------------------------------------"
    echo " APP: $app | $config_key"
    echo "------------------------------------------------------------"

    if [ -f "$weight_file" ]; then
        echo "[RESUME] Found existing model: $weight_file. Skipping training."
    else
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

    echo "[TEST]"
    TMP_LOG="/tmp/ratio_test_$$.log"
    cd "$NILM_DIR"
    $PYTHON EasyS2S_test.py \
        --appliance_name "$app" \
        --datadir "$DATA_DIR" \
        --train_filename "$train_filename" \
        --origin_model "$origin_model" \
        --dataset_name "UK_DALE" \
        --train_percent "$TRAIN_PERCENT" 2>&1 | tee "$TMP_LOG"
    
    TEST_EXIT=${PIPESTATUS[0]}
    MAE=$(grep -oP "MAE:\s*\K[0-9]*\.?[0-9]+" "$TMP_LOG" | head -1)
    rm -f "$TMP_LOG"

    if [ $TEST_EXIT -ne 0 ] && [ -z "$MAE" ]; then
        RESULTS["${config_key}|${app}"]="FAIL"
    else
        RESULTS["${config_key}|${app}"]="${MAE:-N/A}"
    fi

    cd "$PROJECT_ROOT"
}

# Define Table Structure
for pct in "${RATIO_LABELS[@]}"; do
    if [ "$pct" == "0pct" ]; then
        CONFIG_ORDER+=("100%+${pct} | Baseline")
    else
        CONFIG_ORDER+=("100%+${pct} | Ordered")
        for w in "${WINDOW_SIZES[@]}"; do
            CONFIG_ORDER+=("100%+${pct} | Partial w${w}")
            CONFIG_ORDER+=("100%+${pct} | Full w${w}")
        done
        CONFIG_ORDER+=("100%+${pct} | Event Even")
    fi
done

# Main Loop
for app in "${APPLIANCES[@]}"; do
    echo ">>> STARTING FULL EXPERIMENT SET FOR APPLIANCE: $app <<<"
    for pct in "${RATIO_LABELS[@]}"; do
        ORIGIN_MODEL="False"; [ "$pct" == "0pct" ] && ORIGIN_MODEL="True"
        
        # 1. Ordered
        CK="100%+${pct} | Ordered"; [ "$pct" == "0pct" ] && CK="100%+${pct} | Baseline"
        FNAME="${app}_training_100%+${pct}_ordered"
        run_experiment "$app" "$FNAME" "$ORIGIN_MODEL" "$CK"
        print_summary_table

        if [ "$pct" != "0pct" ]; then
            # 2. Shuffled
            for w in "${WINDOW_SIZES[@]}"; do
                run_experiment "$app" "${app}_training_100%+${pct}_shuffled_w${w}" "$ORIGIN_MODEL" "100%+${pct} | Partial w${w}"
                print_summary_table
                run_experiment "$app" "${app}_training_100%+${pct}_full_shuffled_w${w}" "$ORIGIN_MODEL" "100%+${pct} | Full w${w}"
                print_summary_table
            done
            # 3. Event
            run_experiment "$app" "${app}_training_100%+${pct}_event_even_v3" "$ORIGIN_MODEL" "100%+${pct} | Event Even"
            print_summary_table
        fi
    done
done

print_summary_table
echo "DONE: All experiments finished."
