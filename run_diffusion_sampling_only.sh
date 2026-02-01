#!/bin/bash

# ====================================================
#   Diffusion Model Automation: Linux (WSL2) - SAMPLING ONLY
# ====================================================

# Default values
APPLIANCES=("washingmachine" "microwave" "fridge" "kettle" "dishwasher")
SAMPLE=true
MILESTONE=10
GPU=0
PROPORTION=1.0
SAMPLE_NUM=0
GUIDANCE=2.5
TIMESTEPS=2000

# Help message
usage() {
    echo "Usage: $0 [--milestone M] [--gpu G] [--proportion P] [--sample_num N] [--guidance G_SCALE] [--sampling_steps S] [--appliances a,b,c]"
    echo "Example: $0 --guidance 2.5 --appliances fridge,microwave"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --milestone) MILESTONE="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --proportion) PROPORTION="$2"; shift ;;
        --sample_num) SAMPLE_NUM="$2"; shift ;;
        --guidance) GUIDANCE="$2"; shift ;;
        --sampling_steps) TIMESTEPS="$2"; shift ;;
        --appliances) IFS=',' read -ra APPLIANCES <<< "$2"; shift ;;
        *) usage ;;
    esac
    shift
done

echo "===================================================="
echo "   Linux Diffusion Automation: SAMPLING ONLY"
echo "===================================================="
echo "Appliances: ${APPLIANCES[*]}"
echo "GPU ID: $GPU"
echo "Milestone: $MILESTONE"
echo "Proportion: $PROPORTION"
echo "Guidance Scale: $GUIDANCE"
echo "===================================================="

for app in "${APPLIANCES[@]}"; do
    echo -e "\n>>> Processing Appliance: [${app^^}]"
    
    configPath="Config/$app.yaml"
    if [ ! -f "$configPath" ]; then
        echo "Warning: Config file not found: $configPath. Skipping..."
        continue
    fi

    # --- Step 2: Sampling (Only Sampling) ---
    echo "--- Starting Sampling for $app ---"
    
    # Calculate dynamic sample number if not specified
    dynamicSampleNum=$SAMPLE_NUM
    if [ "$dynamicSampleNum" -eq 0 ]; then
        # Extract window size and data path from YAML (Stripping \r for Windows compatibility)
        window=$(grep "window:" "$configPath" | head -n 1 | awk '{print $2}' | tr -d '\r')
        dataPath=$(grep "data_root:" "$configPath" | head -n 1 | awk '{print $2}' | tr -d "'" | tr -d '"' | tr -d '\r')
        
        # Use fallback for window if not found
        if [ -z "$window" ]; then window=512; fi

        if [ -n "$dataPath" ]; then
            # Handle relative paths properly (remove leading ./)
            checkPath="${dataPath#./}"
            
            # Check original path, then fallback to root if not found
            # This handles cases where data is moved but YAML isn't updated
            if [ ! -f "$checkPath" ]; then
                filename=$(basename "$checkPath")
                if [ -f "$filename" ]; then
                    checkPath="$filename"
                fi
            fi

            if [ -f "$checkPath" ]; then
                # Fast line count in Linux
                totalLines=$(wc -l < "$checkPath")
                totalPoints=$((totalLines - 1))
                # Dynamic SampleNum: (Points/Window + 1) * 2 to ensure 200% coverage
                dynamicSampleNum=$(( (totalPoints / window + 1) * 2 ))
                echo "  -> Found $totalPoints points in $checkPath. Window size: $window"
                echo "  -> Dynamic SampleNum: $dynamicSampleNum (200% data)"
            else
                echo "  -> Warning: Data file not found ($checkPath). Using fallback 1000."
                dynamicSampleNum=1000
            fi
        fi
    fi

    # 🚀 自动选择采样算法：步数少时用 ddim，步数多时用 ddpm
    SAMPLER_TYPE="ddpm"
    if [ $TIMESTEPS -lt 500 ]; then
        SAMPLER_TYPE="ddim"
    fi

    echo -e "\n🚀 [SAMPLER MODE]: ${SAMPLER_TYPE^^} Activated (Steps: $TIMESTEPS)"
    echo "----------------------------------------------------"

    python main.py \
        --name "${app}_multivariate" \
        --config "$configPath" \
        --sample 1 \
        --milestone $MILESTONE \
        --sample_num $dynamicSampleNum \
        --sampling_mode "ordered_non_overlapping" \
        --guidance_scale $GUIDANCE \
        --sampling_timesteps $TIMESTEPS \
        --sampler $SAMPLER_TYPE \
        --gpu $GPU
        
    if [ $? -ne 0 ]; then
        echo "Error: Sampling failed for $app"
        exit 1
    fi
done

echo -e "\n===================================================="
echo "   All Linux sampling tasks completed successfully!"
echo "===================================================="
