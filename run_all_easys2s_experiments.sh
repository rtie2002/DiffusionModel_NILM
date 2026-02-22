#!/bin/bash

# ==============================================================================
# Script: run_all_easys2s_experiments.sh
# Purpose: Automate training + testing for all 21 dataset combinations,
#          for all 5 appliances (105 experiments total).
#          Displays a formatted MAE summary table at the end.
# Usage:
#   bash run_all_easys2s_experiments.sh           # all appliances
#   bash run_all_easys2s_experiments.sh fridge    # single appliance
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Use the full Python path from the nilm_main conda env to avoid
# "No module named tensorflow" errors when conda is not activated in subshells
PYTHON="/home/raymond/miniconda3/envs/nilm_main/bin/python3"

# Sanity check
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Please update PYTHON variable in this script."
    exit 1
fi
echo "Using Python: $PYTHON"

# Fix: Find libdevice.10.bc for XLA BatchNorm GPU kernel compilation
# TF looks in nvidia pip-package path — try to detect and set it explicitly
NILM_MAIN_PY_SITE="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
LIBDEVICE_CANDIDATES=(
    "$NILM_MAIN_PY_SITE/nvidia/cuda_nvcc"
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-12.2"
    "/usr/local/cuda-12"
    "/usr/local/cuda-11.8"
)
CUDA_DATA_DIR=""
for candidate in "${LIBDEVICE_CANDIDATES[@]}"; do
    if [ -f "$candidate/nvvm/libdevice/libdevice.10.bc" ]; then
        CUDA_DATA_DIR="$candidate"
        echo "Found libdevice at: $CUDA_DATA_DIR"
        break
    fi
done

if [ -n "$CUDA_DATA_DIR" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_DATA_DIR"
    echo "Set XLA_FLAGS=$XLA_FLAGS"
else
    echo "WARNING: libdevice.10.bc not found. BatchNorm may fail on GPU."
    echo "Run: find /home/raymond/miniconda3/envs/nilm_main -name 'libdevice.10.bc'"
    echo "Then set XLA_FLAGS manually in this script."
    # Last resort: force CPU for BatchNorm by disabling GPU JIT
    export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
fi

# --- Appliances ---
if [ "$1" == "all" ] || [ -z "$1" ]; then
    APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
else
    APPLIANCES=("$1")
fi

# --- Experiment Parameters ---
REAL_K="200k"
SYN_K_CASES=("0k" "20k" "100k" "200k" "400k")
WINDOW_SIZES=("10" "50" "100" "600")
EPOCHS=100
BATCH_SIZE=1024
TRAIN_PERCENT="20"

# --- Paths ---
DATA_DIR="$PROJECT_ROOT/created_data/UK_DALE/"
NILM_DIR="$PROJECT_ROOT/NILM-main"
TRAIN_SCRIPT="$NILM_DIR/EasyS2S_train.py"
TEST_SCRIPT="$NILM_DIR/EasyS2S_test.py"

# --- Result storage (key=config|appliance, value=MAE) ---
declare -A RESULTS
declare -a CONFIG_ORDER   # track ordered list of config keys

# --- Helper: patch a top-level variable in a Python file ---
update_python_var() {
    local file=$1 var=$2 val=$3 is_string=$4
    if [[ "$is_string" == "true" ]]; then
        sed -i "s/^$var\s*=\s*['\"].*['\"]/$var = '$val'/" "$file"
    else
        sed -i "s/^$var\s*=\s*[A-Za-z0-9_\.]*/$var = $val/" "$file"
    fi
}

# --- Helper: run one train+test experiment, capture MAE ---
run_experiment() {
    local app=$1
    local train_filename=$2
    local origin_model=$3
    local config_key=$4   # e.g. "200k+20k | Shuffled w10"

    echo ""
    echo "------------------------------------------------------------"
    echo " APP: $app | $config_key | originModel=$origin_model"
    echo "------------------------------------------------------------"

    # Patch Python files
    update_python_var "$TRAIN_SCRIPT" "originModel"   "$origin_model"  "false"
    update_python_var "$TRAIN_SCRIPT" "datasetName"   "UK_DALE"        "true"
    update_python_var "$TRAIN_SCRIPT" "applianceName" "$app"           "true"
    update_python_var "$TRAIN_SCRIPT" "TrainPercent"  "$TRAIN_PERCENT" "true"
    update_python_var "$TEST_SCRIPT"  "originModel"   "$origin_model"  "false"
    update_python_var "$TEST_SCRIPT"  "datasetName"   "UK_DALE"        "true"
    update_python_var "$TEST_SCRIPT"  "TrainPercent"  "$TRAIN_PERCENT" "true"

    # --- TRAIN ---
    echo "[TRAIN]"
    cd "$NILM_DIR"
    $PYTHON EasyS2S_train.py \
        --appliance_name "$app" \
        --n_epoch $EPOCHS \
        --batchsize $BATCH_SIZE \
        --datadir "$DATA_DIR" \
        --train_filename "$train_filename"

    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "ERROR: Training failed (exit $TRAIN_EXIT). Skipping test."
        RESULTS["${config_key}|${app}"]="FAIL"
        cd "$PROJECT_ROOT"
        return
    fi

    # --- TEST: capture output to parse MAE ---
    echo "[TEST]"
    TEST_OUTPUT=$($PYTHON EasyS2S_test.py \
        --appliance_name "$app" \
        --datadir "$DATA_DIR" \
        --train_filename "$train_filename" 2>&1)
    echo "$TEST_OUTPUT"

    TEST_EXIT=$?
    if [ $TEST_EXIT -ne 0 ]; then
        echo "ERROR: Testing failed (exit $TEST_EXIT)."
        RESULTS["${config_key}|${app}"]="FAIL"
    else
        # Extract MAE value from output (line: "MAE: X.XXXXX")
        MAE=$(echo "$TEST_OUTPUT" | grep -oP "MAE:\s*\K[0-9]+\.[0-9]+" | head -1)
        if [ -z "$MAE" ]; then
            MAE="N/A"
        fi
        RESULTS["${config_key}|${app}"]="$MAE"
    fi

    cd "$PROJECT_ROOT"
}

# ==============================================================================
# Main Experiment Loop
# ==============================================================================
echo "Processing appliances: ${APPLIANCES[*]}"

for syn_k in "${SYN_K_CASES[@]}"; do

    if [ "$syn_k" == "0k" ]; then
        ORIGIN_MODEL="True"
    else
        ORIGIN_MODEL="False"
    fi

    # 1. Ordered Case (always runs)
    CONFIG_KEY="200k+${syn_k} | Ordered     "
    # Track unique config keys in order
    if [[ ! " ${CONFIG_ORDER[*]} " =~ " ${CONFIG_KEY} " ]]; then
        CONFIG_ORDER+=("$CONFIG_KEY")
    fi
    for app in "${APPLIANCES[@]}"; do
        TRAIN_FILENAME="${app}_training_${REAL_K}+${syn_k}_ordered"
        run_experiment "$app" "$TRAIN_FILENAME" "$ORIGIN_MODEL" "$CONFIG_KEY"
    done

    # 2. Shuffled Cases (skip for 0k baseline)
    if [ "$syn_k" != "0k" ]; then
        for window in "${WINDOW_SIZES[@]}"; do
            CONFIG_KEY="200k+${syn_k} | Shuffled w${window} "
            if [[ ! " ${CONFIG_ORDER[*]} " =~ " ${CONFIG_KEY} " ]]; then
                CONFIG_ORDER+=("$CONFIG_KEY")
            fi
            for app in "${APPLIANCES[@]}"; do
                TRAIN_FILENAME="${app}_training_${REAL_K}+${syn_k}_shuffled_w${window}"
                run_experiment "$app" "$TRAIN_FILENAME" "$ORIGIN_MODEL" "$CONFIG_KEY"
            done
        done
    fi

done

# ==============================================================================
# Print MAE Summary Table
# ==============================================================================
COL_W=14   # column width per appliance

echo ""
echo "================================================================"
echo "  MAE SUMMARY TABLE"
echo "================================================================"

# Header row
HEADER="Configuration                  "
for app in "${APPLIANCES[@]}"; do
    printf -v COL "%-${COL_W}s" "${app}"
    HEADER+="| ${COL}"
done
echo "$HEADER"

# Separator
SEP=""
SEP_LEN=$(( ${#HEADER} ))
printf -v SEP '%*s' "$SEP_LEN" ''
echo "${SEP// /-}"

# Data rows
for config_key in "${CONFIG_ORDER[@]}"; do
    printf "%-31s" "$config_key"
    for app in "${APPLIANCES[@]}"; do
        val="${RESULTS["${config_key}|${app}"]:-N/A}"
        printf "| %-${COL_W}s" "$val"
    done
    echo ""
done

echo "${SEP// /-}"
echo ""
echo "All experiments completed."
echo "================================================================"
