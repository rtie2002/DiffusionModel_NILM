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
SEED=2025
PROPORTION=1.0
SAMPLE_NUM=0
REPORT_ONOFF=true
ONOFF_ONLY=false

# Help message
usage() {
    echo "Usage: $0 [--train] [--sample] [--milestone M] [--gpu G] [--seed S] [--proportion P] [--sample_num N] [--appliances a,b,c] [--onoff-report-only] [--no-onoff-report]"
    echo "Example: $0 --train --sample --appliances fridge,microwave"
    echo "Example: $0 --onoff-report-only"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train) TRAIN=true ;;
        --sample) SAMPLE=true ;;
        --milestone) MILESTONE="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --proportion) PROPORTION="$2"; shift ;;
        --sample_num) SAMPLE_NUM="$2"; shift ;;
        --appliances) IFS=',' read -ra APPLIANCES <<< "$2"; shift ;;
        --onoff-report-only) ONOFF_ONLY=true; REPORT_ONOFF=true ;;
        --no-onoff-report) REPORT_ONOFF=false ;;
        *) usage ;;
    esac
    shift
done

# If neither Train nor Sample is specified, do both
if [ "$ONOFF_ONLY" = true ]; then
    TRAIN=false
    SAMPLE=false
elif [ "$TRAIN" = false ] && [ "$SAMPLE" = false ]; then
    TRAIN=true
    SAMPLE=true
fi

format_duration() {
    local seconds="$1"
    if [ -z "$seconds" ] || [ "$seconds" = "NA" ]; then
        echo "NA"
        return
    fi

    local whole="${seconds%.*}"
    local hours=$((whole / 3600))
    local minutes=$(((whole % 3600) / 60))
    local secs=$((whole % 60))

    if [ "$hours" -gt 0 ]; then
        printf "%02d:%02d:%02d" "$hours" "$minutes" "$secs"
    else
        printf "%02d:%02d" "$minutes" "$secs"
    fi
}

get_npy_shape() {
    local path="$1"
    if [ ! -f "$path" ]; then
        return
    fi

    python -c "import sys, numpy as np; a=np.load(sys.argv[1], mmap_mode='r'); print(','.join(map(str, a.shape)))" "$path" 2>/dev/null || true
}

get_latest_runtime_from_log() {
    local log_path="$1"
    local label="$2"
    if [ ! -f "$log_path" ]; then
        return
    fi

    grep "$label, time:" "$log_path" | tail -n 1 | sed -E 's/.*time: ([0-9.]+).*/\1/'
}

get_latest_parameter_count_from_log() {
    local log_path="$1"
    if [ ! -f "$log_path" ]; then
        echo "NA"
        return
    fi

    local count
    count=$(grep "overall.*trainable" "$log_path" | tail -n 1 | sed -E "s/.*overall.*trainable': '([^']+)'.*/\1/")
    if [ -z "$count" ]; then
        echo "NA"
    else
        echo "$count"
    fi
}

summary_dir="OUTPUT"
mkdir -p "$summary_dir"
summary_csv="$summary_dir/revision_reproducibility_summary.csv"
summary_md="$summary_dir/revision_reproducibility_summary.md"
onoff_csv="$summary_dir/active_window_proportions.csv"
onoff_md="$summary_dir/active_window_proportions.md"

echo "Appliance,GeneratedSamples,RandomSeed,TrainingTimeSeconds,TrainingTime,SamplingTimeSeconds,SamplingTime,ModelParameters,OutputShape,OutputFile" > "$summary_csv"
echo "| Appliance | Generated samples | Random seed | Training time | Sampling time | Model parameters | Output shape |" > "$summary_md"
echo "|---|---:|---:|---:|---:|---:|---|" >> "$summary_md"

echo "===================================================="
echo "   Linux Diffusion Automation: ACTIVE"
echo "===================================================="
echo "Appliances: ${APPLIANCES[*]}"
echo "GPU ID: $GPU"
echo "Random Seed: $SEED"
echo "Milestone: $MILESTONE"
echo "Proportion: $PROPORTION"
echo "ON/OFF Report: $REPORT_ONOFF"
echo "===================================================="

if [ "$REPORT_ONOFF" = true ]; then
    config_args=()
    for app in "${APPLIANCES[@]}"; do
        configPath="Config/$app.yaml"
        if [ -f "$configPath" ]; then
            config_args+=("$configPath")
        else
            echo "Warning: Config file not found for ON/OFF report: $configPath"
        fi
    done

    if [ "${#config_args[@]}" -gt 0 ]; then
        echo "--- Generating ON/OFF Proportion Reports ---"
        python report_booster_onoff_proportions.py \
            --configs "${config_args[@]}" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --output-csv "$onoff_csv" \
            --output-md "$onoff_md"
    fi
fi

if [ "$ONOFF_ONLY" = true ]; then
    echo -e "\nON/OFF proportion report saved to:"
    echo "  $onoff_csv"
    echo "  $onoff_md"
    exit 0
fi

for app in "${APPLIANCES[@]}"; do
    echo -e "\n>>> Processing Appliance: [${app^^}]"

    runName="${app}_multivariate"
    outputDir="OUTPUT/$runName"
    logPath="$outputDir/logs/log.txt"
    expectedOutput="$outputDir/ddpm_fake_${runName}.npy"
    trainElapsedSeconds=""
    sampleElapsedSeconds=""
    dynamicSampleNum=""
    
    configPath="Config/$app.yaml"
    if [ ! -f "$configPath" ]; then
        echo "Warning: Config file not found: $configPath. Skipping..."
        continue
    fi

    # --- Step 1: Training ---
    if [ "$TRAIN" = true ]; then
        echo "--- [1/2] Starting Training for $app ---"
        trainStart=$(date +%s)
        python main.py --train \
            --name "$runName" \
            --config "$configPath" \
            --tensorboard \
            --gpu $GPU \
            --seed "$SEED" \
            --opts dataloader.train_dataset.params.save2npy False \
            dataloader.train_dataset.params.proportion $PROPORTION \
            dataloader.train_dataset.params.seed "$SEED"
        trainStatus=$?
        trainEnd=$(date +%s)
        trainElapsedSeconds=$((trainEnd - trainStart))
        
        if [ $trainStatus -ne 0 ]; then
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

        sampleStart=$(date +%s)
        python main.py \
            --name "$runName" \
            --config "$configPath" \
            --sample 1 \
            --milestone $MILESTONE \
            --sample_num $dynamicSampleNum \
            --sampling_mode "ordered_non_overlapping" \
            --gpu $GPU \
            --seed "$SEED" \
            --opts dataloader.train_dataset.params.seed "$SEED"
        sampleStatus=$?
        sampleEnd=$(date +%s)
        sampleElapsedSeconds=$((sampleEnd - sampleStart))
            
        if [ $sampleStatus -ne 0 ]; then
            echo "Error: Sampling failed for $app"
            exit 1
        fi

        if [ -f "$expectedOutput" ]; then
            echo "Successfully generated: $expectedOutput"
        else
            echo "Warning: Output file not found at expected location: $expectedOutput"
        fi
    fi

    if [ -z "$trainElapsedSeconds" ]; then
        trainElapsedSeconds=$(get_latest_runtime_from_log "$logPath" "Training done")
    fi
    if [ -z "$sampleElapsedSeconds" ]; then
        sampleElapsedSeconds=$(get_latest_runtime_from_log "$logPath" "Sampling done")
    fi

    outputShape=$(get_npy_shape "$expectedOutput")
    if [ -n "$outputShape" ]; then
        generatedSamples="${outputShape%%,*}"
    elif [ -n "$dynamicSampleNum" ]; then
        generatedSamples="$dynamicSampleNum"
        outputShape="not found"
    else
        generatedSamples="NA"
        outputShape="not found"
    fi

    parameterCount=$(get_latest_parameter_count_from_log "$logPath")
    trainingTime=$(format_duration "$trainElapsedSeconds")
    samplingTime=$(format_duration "$sampleElapsedSeconds")
    trainSecondsForCsv=${trainElapsedSeconds:-NA}
    sampleSecondsForCsv=${sampleElapsedSeconds:-NA}

    printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
        "$app" "$generatedSamples" "$SEED" "$trainSecondsForCsv" "$trainingTime" \
        "$sampleSecondsForCsv" "$samplingTime" "$parameterCount" "$outputShape" "$expectedOutput" >> "$summary_csv"
    echo "| $app | $generatedSamples | $SEED | $trainingTime | $samplingTime | $parameterCount | $outputShape |" >> "$summary_md"
done

echo -e "\nReproducibility summary saved to:"
echo "  $summary_csv"
echo "  $summary_md"
if [ "$REPORT_ONOFF" = true ]; then
    echo "  $onoff_csv"
    echo "  $onoff_md"
fi

echo -e "\n===================================================="
echo "   All Linux tasks completed successfully!"
echo "===================================================="
