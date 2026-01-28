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
SAMPLE_NUM=0

# Help message
usage() {
    echo "Usage: $0 [--train] [--sample] [--milestone M] [--gpu G] [--proportion P] [--sample_num N] [--appliances a,b,c]"
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
        --proportion) PROPORTION="$2"; shift ;;
        --sample_num) SAMPLE_NUM="$2"; shift ;;
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
echo "Proportion: $PROPORTION"
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

    # --- Step 2: Sampling ---
    if [ "$SAMPLE" = true ]; then
        echo "--- [2/2] Starting Sampling for $app ---"
        
        # Calculate dynamic sample number if not specified
        dynamicSampleNum=$SAMPLE_NUM
        if [ "$dynamicSampleNum" -eq 0 ]; then
            # Extract window size and data path from YAML
            window=$(grep "window:" "$configPath" | head -n 1 | awk '{print $2}')
            dataPath=$(grep "data_root:" "$configPath" | head -n 1 | awk '{print $2}' | tr -d "'" | tr -d '"')
            
            # Use fallback for window if not found
            if [ -z "$window" ]; then window=512; fi

            if [ -n "$dataPath" ] && [ -f "$dataPath" ]; then
                # Fast line count in Linux
                totalLines=$(wc -l < "$dataPath")
                totalPoints=$((totalLines - 1))
                # Dynamic SampleNum: (Points/Window) * 2
                dynamicSampleNum=$(( (totalPoints / window + 1) * 2 ))
                echo "  -> Found $totalPoints points. Dynamic SampleNum: $dynamicSampleNum"
            else
                echo "  -> Data file not found ($dataPath). Using fallback."
                dynamicSampleNum=1000
            fi
        fi

        python main.py \
            --name "${app}_multivariate" \
            --config "$configPath" \
            --sample 1 \
            --milestone $MILESTONE \
            --sample_num $dynamicSampleNum \
            --sampling_mode "ordered_non_overlapping" \
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
