#!/bin/bash

# ====================================================
#   Diffusion Model Automation: Linux (WSL2)
# ====================================================

# Default values
APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
TRAIN=false
SAMPLE=false
MILESTONE=10
GPU=0
PROPORTION=1.0

# Help message
usage() {
    echo "Usage: $0 [--train] [--sample] [--milestone M] [--gpu G] [--appliances a,b,c]"
    echo "Example: $0 --train --sample --appliances fridge,microwave"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train) TRAIN=true ;;
        --sample) SAMPLE=true ;;
        --milestone) MILESTONE="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --appliances) IFS=',' read -ra APPLIANCES <<< "$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# If neither Train nor Sample is specified, do both
if [ "$TRAIN" = false ] && [ "$SAMPLE" = false ]; then
    TRAIN=true
    SAMPLE=true
fi

echo "===================================================="
echo "   Linux Diffusion Automation: ACTIVE"
echo "===================================================="
echo "Appliances: ${APPLIANCES[*]}"
echo "GPU ID: $GPU"
echo "Milestone: $MILESTONE"
echo "===================================================="

for app in "${APPLIANCES[@]}"; do
    echo -e "\n>>> Processing Appliance: [${app^^}]"
    
    configPath="Config/$app.yaml"
    if [ ! -f "$configPath" ]; then
        echo "Warning: Config file not found: $configPath. Skipping..."
        continue
    fi

    # --- Step 1: Training ---
    if [ "$TRAIN" = true ]; then
        echo "--- [1/2] Starting Training for $app ---"
        python main.py --train \
            --name "${app}_multivariate" \
            --config "$configPath" \
            --tensorboard \
            --gpu $GPU \
            --opts dataloader.train_dataset.params.save2npy False \
            dataloader.train_dataset.params.proportion $PROPORTION
        
        if [ $? -ne 0 ]; then
            echo "Error: Training failed for $app"
            exit 1
        fi
    fi

    # --- Step 2: Sampling (200% Logic) ---
    if [ "$SAMPLE" = true ]; then
        echo "--- [2/2] Starting Sampling for $app ---"
        
        # Calculate dynamic sample number (200% coverage)
        # 1. Get window size and data path from YAML
        window=$(grep "window:" "$configPath" | head -n 1 | awk '{print $2}')
        dataPath=$(grep "data_root:" "$configPath" | head -n 1 | awk '{print $2}' | tr -d "'" | tr -d '"')
        
        if [ -n "$dataPath" ] && [ -f "$dataPath" ]; then
            # Fast line count in Linux
            totalLines=$(wc -l < "$dataPath")
            totalPoints=$((totalLines - 1))
            # 200% Calculation: (Points/Window) * 2
            dynamicSampleNum=$(( (totalPoints / window + 1) * 2 ))
            echo "  -> Found $totalPoints points. Dynamic SampleNum (200%): $dynamicSampleNum"
        else
            echo "  -> Data file not found ($dataPath). Using fallback."
            dynamicSampleNum=4800
        fi

        python main.py \
            --name "${app}_multivariate" \
            --config "$configPath" \
            --sample 1 \
            --milestone $MILESTONE \
            --sample_num $dynamicSampleNum \
            --sampling_mode "ordered" \
            --gpu $GPU
            
        if [ $? -ne 0 ]; then
            echo "Error: Sampling failed for $app"
            exit 1
        fi
    fi
done

echo -e "\n===================================================="
echo "   All Linux tasks completed successfully!"
echo "===================================================="
