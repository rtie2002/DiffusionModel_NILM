#!/bin/bash

# ====================================================
#   Quarterly Diffusion Pipeline: Linux (WSL2)
#   Drop-in replacement for run_diffusion_all.sh with
#   high-quality quarterly training + sampling.
#
#   Workflow per appliance:
#     1. Split <app>_multivariate.csv  -> Q1/Q2/Q3/Q4 CSVs
#     2. Generate per-quarter YAML configs
#     3. Train one model per available quarter
#     4. Sample from Q1->Q2->Q3->Q4 cyclically until
#        the required coverage (e.g. 200%) is reached
#     5. Concat all sampled NPYs in temporal order
#        -> ddpm_fake_<app>_multivariate.npy (same output
#           path as run_diffusion_all.sh)
#
#   Usage (same flags as run_diffusion_all.sh):
#     bash run_quarterly_diffusion.sh --train --sample \
#          --milestone 10 --gpu 0 --appliances washingmachine
# ====================================================

# ── Default values ────────────────────────────────────────────────────────────
APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
TRAIN=false
SAMPLE=false
MILESTONE=10
GPU=0
PROPORTION=1.0
SAMPLE_NUM=0   # 0 = auto (200% of real data)

# ── Help ──────────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [--train] [--sample] [--milestone M] [--gpu G]"
    echo "          [--sample_num N] [--appliances a,b,c]"
    echo "Example: $0 --train --sample --appliances washingmachine --milestone 10"
    exit 1
}

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train)       TRAIN=true ;;
        --sample)      SAMPLE=true ;;
        --milestone)   MILESTONE="$2"; shift ;;
        --gpu)         GPU="$2"; shift ;;
        --proportion)  PROPORTION="$2"; shift ;;
        --sample_num)  SAMPLE_NUM="$2"; shift ;;
        --appliances)  IFS=',' read -ra APPLIANCES <<< "$2"; shift ;;
        *)             usage ;;
    esac
    shift
done

if [ "$TRAIN" = false ] && [ "$SAMPLE" = false ]; then
    TRAIN=true
    SAMPLE=true
fi

echo "===================================================================="
echo "   Quarterly Diffusion Pipeline: ACTIVE"
echo "===================================================================="
echo "   Appliances : ${APPLIANCES[*]}"
echo "   GPU        : $GPU"
echo "   Milestone  : $MILESTONE"
echo "   Train      : $TRAIN"
echo "   Sample     : $SAMPLE"
echo "===================================================================="

# ── Per-appliance loop ────────────────────────────────────────────────────────
for app in "${APPLIANCES[@]}"; do
    echo -e "\n>>> Processing Appliance: [${app^^}]"

    BASE_CONFIG="Config/${app}.yaml"
    
    # ── High Discovery Logic for CSV ──────────────────────────────────────────
    # Check possible locations and suffixes
    POSSIBLE_LOCATIONS=("Data/datasets" ".")
    POSSIBLE_SUFFIXES=("multivariate.csv" "training_.csv")
    
    INPUT_CSV=""
    for loc in "${POSSIBLE_LOCATIONS[@]}"; do
        for suf in "${POSSIBLE_SUFFIXES[@]}"; do
            path="${loc}/${app}_${suf}"
            if [ -f "$path" ]; then
                INPUT_CSV="$path"
                break 2
            fi
        done
    done

    if [ ! -f "$BASE_CONFIG" ]; then
        echo "  Warning: Config not found: $BASE_CONFIG — skipping"
        continue
    fi
    if [ -z "$INPUT_CSV" ]; then
        echo "  Warning: Input CSV for $app not found (checked Data/datasets/ and .) — skipping"
        continue
    fi

    DATA_ROOT=$(dirname "$INPUT_CSV")
    echo "  Found Data : $INPUT_CSV"

    # ── STEP 1: Split CSV into quarters ──────────────────────────────────────
    echo -e "\n  [1/4] Splitting $(basename "$INPUT_CSV") into quarters..."
    python preprocess_NILMformer/split_csv_by_quarter.py \
        --input "$INPUT_CSV" \
        --output_dir "$DATA_ROOT"

    # Detect which quarters were actually created in the SAME directory as input
    AVAILABLE_QUARTERS=()
    for q in Q1 Q2 Q3 Q4; do
        # Handle cases where naming was _multivariate or _training_
        # basename strips directory, then we check for results of splitting
        # script's output naming logic (it appends _Q1 before .csv)
        base_name=$(basename "$INPUT_CSV" .csv)
        if [ -f "${DATA_ROOT}/${base_name}_${q}.csv" ]; then
            AVAILABLE_QUARTERS+=("$q")
        fi
    done

    if [ ${#AVAILABLE_QUARTERS[@]} -eq 0 ]; then
        echo "  Error: No quarterly CSVs created for $app in $DATA_ROOT — skipping"
        continue
    fi
    echo "  Available quarters: ${AVAILABLE_QUARTERS[*]}"

    # ── STEP 2: Generate per-quarter YAML configs ─────────────────────────────
    echo -e "\n  [2/4] Generating quarter configs..."
    python generate_quarter_configs.py \
        --base_config "$BASE_CONFIG" \
        --appliance "$app" \
        --quarters "${AVAILABLE_QUARTERS[@]}" \
        --csv_dir "$DATA_ROOT" \
        --source_csv "$INPUT_CSV"

    # ── STEP 3: Train each quarter model ─────────────────────────────────────
    if [ "$TRAIN" = true ]; then
        echo -e "\n  [3/4] Training quarter models..."
        for q in "${AVAILABLE_QUARTERS[@]}"; do
            Q_CONFIG="Config/${app}_${q}.yaml"
            Q_NAME="${app}_${q}"
            echo -e "\n    --- Training [$q] ---"
            python main.py --train \
                --name "$Q_NAME" \
                --config "$Q_CONFIG" \
                --tensorboard \
                --gpu $GPU \
                --opts dataloader.train_dataset.params.save2npy False \
                       dataloader.train_dataset.params.proportion $PROPORTION

            if [ $? -ne 0 ]; then
                echo "  Error: Training failed for $app $q"
                exit 1
            fi
            echo "    [$q] Training complete."
        done
    else
        echo -e "\n  [3/4] Skipping training (--train not set)"
    fi

    # ── STEP 4: Sample + Concat ───────────────────────────────────────────────
    if [ "$SAMPLE" = true ]; then
        echo -e "\n  [4/4] Sampling (quarterly cycling)..."

        # Get window size from config
        window=$(grep "window:" "$BASE_CONFIG" | head -n 1 | awk '{print $2}' | tr -d '\r')
        if [ -z "$window" ]; then window=512; fi

        # Total data points in original CSV
        totalLines=$(wc -l < "$INPUT_CSV")
        totalPoints=$((totalLines - 1))   # subtract header

        numQ=${#AVAILABLE_QUARTERS[@]}

        # ── Calculate how many samples per quarter per cycle ──────────────────
        # 100% per quarter = roughly (totalPoints / numQ) / window windows
        # numCycles = 2 means 200% coverage (same as original script's *2)
        pointsPerQ=$((totalPoints / numQ))
        samplesPerQPerCycle=$(( pointsPerQ / window + 1 ))

        # Total windows the original script would have generated (200% = *2)
        if [ "$SAMPLE_NUM" -gt 0 ]; then
            TOTAL_TARGET=$SAMPLE_NUM
            NUM_CYCLES=$(( (TOTAL_TARGET + numQ * samplesPerQPerCycle - 1) / (numQ * samplesPerQPerCycle) ))
            if [ "$NUM_CYCLES" -lt 1 ]; then NUM_CYCLES=1; fi
        else
            # Auto: 200% = 2 full Q1→Q4 cycles
            NUM_CYCLES=2
        fi

        echo "    Window size         : $window"
        echo "    Total data points   : $totalPoints"
        echo "    Available quarters  : ${AVAILABLE_QUARTERS[*]}"
        echo "    Samples/Q/cycle     : $samplesPerQPerCycle"
        echo "    Cycles              : $NUM_CYCLES ($(( NUM_CYCLES * 100 ))% coverage)"

        # ── Run sampling: cycle Q1→Q4 for NUM_CYCLES ─────────────────────────
        ALL_GENERATED_FILES=()   # tracks NPY paths in temporal order

        for (( cycle=1; cycle<=NUM_CYCLES; cycle++ )); do
            echo -e "\n    === Cycle $cycle / $NUM_CYCLES ==="
            for q in "${AVAILABLE_QUARTERS[@]}"; do
                Q_CONFIG="Config/${app}_${q}.yaml"
                Q_NAME="${app}_${q}"
                CYCLE_NAME="${app}_${q}_c${cycle}"

                echo "    Sampling [$q] cycle $cycle  (${samplesPerQPerCycle} windows)..."
                python main.py \
                    --name "$Q_NAME" \
                    --config "$Q_CONFIG" \
                    --sample 1 \
                    --milestone $MILESTONE \
                    --sample_num $samplesPerQPerCycle \
                    --sampling_mode "ordered_non_overlapping" \
                    --gpu $GPU

                if [ $? -ne 0 ]; then
                    echo "  Error: Sampling failed for $app $q cycle $cycle"
                    exit 1
                fi

                # main.py saves to OUTPUT/<Q_NAME>/ddpm_fake_<Q_NAME>.npy
                SRC="OUTPUT/${Q_NAME}/ddpm_fake_${Q_NAME}.npy"
                DST="OUTPUT/${Q_NAME}/ddpm_fake_${Q_NAME}_c${cycle}.npy"

                if [ -f "$SRC" ]; then
                    cp "$SRC" "$DST"
                    ALL_GENERATED_FILES+=("$DST")
                    echo "    [$q] c${cycle} -> $DST"
                else
                    echo "    Warning: Expected output not found: $SRC"
                fi
            done
        done

        # ── Concatenate all in temporal order: Q1c1,Q2c1,Q3c1,Q4c1,Q1c2... ──
        FINAL_OUT_DIR="OUTPUT/${app}_multivariate"
        FINAL_OUT="${FINAL_OUT_DIR}/ddpm_fake_${app}_multivariate.npy"
        mkdir -p "$FINAL_OUT_DIR"

        echo -e "\n    Concatenating ${#ALL_GENERATED_FILES[@]} files in temporal order..."
        python concat_quarterly_samples.py \
            --files "${ALL_GENERATED_FILES[@]}" \
            --output "$FINAL_OUT"

        if [ $? -ne 0 ]; then
            echo "  Error: Concatenation failed for $app"
            exit 1
        fi

        echo "    Final output: $FINAL_OUT"
    else
        echo -e "\n  [4/4] Skipping sampling (--sample not set)"
    fi

    echo -e "\n>>> [$app] Done."
done

echo -e "\n===================================================================="
echo "   Quarterly Pipeline Complete!"
echo "===================================================================="
