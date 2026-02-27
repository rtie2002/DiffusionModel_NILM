#!/bin/bash

# ==============================================================================
# Script: run_all_preprocessing.sh
# Purpose: Execute NILMformer preprocessing (Train/Val and Test) for all appliances.
# ==============================================================================

# Ensure we are in the project root (assumes script is run from project root or its parent)
# Since the .ps1 used relative paths like 'preprocess_NILMformer\...', 
# we should stay in the root directory.

DATA_DIR="NILM-main/dataset_preprocess/UK_DALE/"
APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")

echo "Starting NILMformer Preprocessing for all appliances..."

# 1. Training & Validation Preprocessing
echo "------------------------------------------------------------"
echo "Phase 1: Training + Validating Preprocessing"
echo "------------------------------------------------------------"
for app in "${APPLIANCES[@]}"; do
    echo "[TRAIN/VAL] Processing: $app"
    python preprocess_NILMformer/multivariate_ukdale_preprocess_training+validating.py \
        --appliance_name "$app" \
        --data_dir "$DATA_DIR"
done

# 2. Testing Preprocessing
echo ""
echo "------------------------------------------------------------"
echo "Phase 2: Testing Preprocessing"
echo "------------------------------------------------------------"
for app in "${APPLIANCES[@]}"; do
    echo "[TEST] Processing: $app"
    python preprocess_NILMformer/multivariate_ukdale_preprocess_testing.py \
        --appliance_name "$app" \
        --data_dir "$DATA_DIR"
done

echo ""
echo "============================================================"
echo "Preprocessing Complete!"
echo "============================================================"
