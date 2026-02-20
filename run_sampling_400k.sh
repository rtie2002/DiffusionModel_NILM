#!/bin/bash

# ====================================================
#   Diffusion Model Sampling: 400k Data (800 Windows)
# ====================================================

# Default values
APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
MILESTONE=10
GPU=0
# 800 windows * 512 points = 409,600 points (~400k)
SAMPLE_NUM=800

# Parse arguments for flexibility
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --milestone) MILESTONE="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --appliances) IFS=',' read -ra APPLIANCES <<< "$2"; shift ;;
        *) echo "Usage: $0 [--milestone M] [--gpu G] [--appliances a,b,c]"; exit 1 ;;
    esac
    shift
done

echo "===================================================="
echo "   Sampling 400k Data (800 Windows)"
echo "===================================================="
echo "Appliances: ${APPLIANCES[*]}"
echo "GPU ID: $GPU"
echo "Milestone: $MILESTONE"
echo "Target: $SAMPLE_NUM windows (~409,600 points)"
echo "===================================================="

for app in "${APPLIANCES[@]}"; do
    echo -e "\n>>> Sampling Appliance: [${app^^}]"
    
    configPath="Config/$app.yaml"
    if [ ! -f "$configPath" ]; then
        echo "Warning: Config file not found: $configPath. Skipping..."
        continue
    fi

    echo "--- Starting Sampling for $app ---"
    
    python main.py \
        --name "${app}_multivariate" \
        --config "$configPath" \
        --sample 1 \
        --milestone $MILESTONE \
        --sample_num $SAMPLE_NUM \
        --sampling_mode "ordered_non_overlapping" \
        --gpu $GPU
        
    if [ $? -ne 0 ]; then
        echo "Error: Sampling failed for $app"
        exit 1
    fi
done

echo -e "\n===================================================="
echo "   Sampling completed successfully!"
echo "===================================================="
