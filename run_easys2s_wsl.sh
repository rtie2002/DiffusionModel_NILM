#!/bin/bash

# --- Parameters (Default Values) ---
APPLIANCE=${1:-"fridge"}
MODE=${2:-"Mixed"}  # Mixed or Baseline
TRAIN_FILENAME=${3:-""}
EPOCHS=${4:-100}
BATCH_SIZE=${5:-1024}
TRAIN_PERCENT="20"

# Paths
TRAIN_SCRIPT="NILM-main/EasyS2S_train.py"
TEST_SCRIPT="NILM-main/EasyS2S_test.py"

# --- Logic for Mode ---
if [[ "$MODE" == "Baseline" ]]; then
    ORIGIN_MODEL="True"
    echo "Running in BASELINE mode (originModel=True)"
else
    ORIGIN_MODEL="False"
    echo "Running in MIXED/ROBUST mode (originModel=False)"
fi

# --- Helper Function to Update Python Variables ---
update_python_var() {
    local file=$1
    local var=$2
    local val=$3
    local is_string=$4

    if [[ "$is_string" == "true" ]]; then
        # Replace var = 'value' or var = "value"
        sed -i "s/^$var\s*=\s*['\"].*['\"]/$var = '$val'/" "$file"
    else
        # Replace var = True/False/Number
        sed -i "s/^$var\s*=\s*[A-zA-Z0-9_\.]*/$var = $val/" "$file"
    fi
    echo "  -> Updated $var to $val in $file"
}

echo -e "\n[1/3] Configuring Training Script..."
update_python_var "$TRAIN_SCRIPT" "originModel" "$ORIGIN_MODEL" "false"
update_python_var "$TRAIN_SCRIPT" "datasetName" "UK_DALE" "true"
update_python_var "$TRAIN_SCRIPT" "applianceName" "$APPLIANCE" "true"
update_python_var "$TRAIN_SCRIPT" "TrainPercent" "$TRAIN_PERCENT" "true"

echo -e "\n[2/3] Starting Training..."
cd NILM-main
CMD="python3 EasyS2S_train.py --appliance_name $APPLIANCE --n_epoch $EPOCHS --batchsize $BATCH_SIZE"
if [[ -n "$TRAIN_FILENAME" ]]; then
    CMD="$CMD --train_filename $TRAIN_FILENAME"
    echo "  -> Using specific training file: $TRAIN_FILENAME"
fi

echo "Executing: $CMD"
$CMD
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
cd ..

echo -e "\n[3/3] Configuring & Running Test..."
update_python_var "$TEST_SCRIPT" "originModel" "$ORIGIN_MODEL" "false"
update_python_var "$TEST_SCRIPT" "datasetName" "UK_DALE" "true"
update_python_var "$TEST_SCRIPT" "TrainPercent" "$TRAIN_PERCENT" "true"

cd NILM-main
echo "Executing: python3 EasyS2S_test.py --appliance_name $APPLIANCE"
python3 EasyS2S_test.py --appliance_name "$APPLIANCE"
cd ..

echo -e "\n--- Experiment Finished for $APPLIANCE ---"
