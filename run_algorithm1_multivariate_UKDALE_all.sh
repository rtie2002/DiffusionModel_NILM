#!/bin/bash

# Run Algorithm 1 v2 Multivariate for all appliances (UK_DALE Training Data)
# This script applies Algorithm 1 to the training data in created_data/UK_DALE
# to prepare it for subsequent modeling or synthesis steps.

SCRIPT="synthetic_data_multivariate/algorithm1_v2_multivariate.py"
INPUT_BASE="created_data/UK_DALE"
OUTPUT_DIR="Data/datasets"

# Allow overriding directories via arguments
if [ ! -z "$1" ]; then
    INPUT_BASE="$1"
fi
if [ ! -z "$2" ]; then
    OUTPUT_DIR="$2"
fi

echo "============================================================"
echo "Running Algorithm 1 for all appliances"
echo "Input Dir:  $INPUT_BASE"
echo "Output Dir: $OUTPUT_DIR"
echo "============================================================"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

APPLIANCES=("dishwasher" "fridge" "kettle" "microwave" "washingmachine")

for APP in "${APPLIANCES[@]}"; do
    INPUT_FILE="$INPUT_BASE/${APP}_training_.csv"
    
    if [ -f "$INPUT_FILE" ]; then
        echo "------------------------------------------------------------"
        echo "Processing $APP..."
        python "$SCRIPT" \
            --appliance_name "$APP" \
            --input_file "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" --no_remove_spikes
    else
        echo "Warning: Input file not found for $APP ($INPUT_FILE)"
    fi
done

echo "============================================================"
echo "All appliances processed successfully."
echo "Cleaned datasets are in: $OUTPUT_DIR"
echo "============================================================"
