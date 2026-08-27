#!/bin/bash

set -euo pipefail

APPLIANCE="fridge"
COUNTS=("392" "784" "1176" "1568")
MILESTONE=10
SEED=2025
GPU=0
BATCH_SIZE=392
BATCH_SIZES=()
WARMUP_COUNT=16
OUTPUT_DIR="OUTPUT"

usage() {
    echo "Usage: $0 [--appliance fridge] [--counts \"392 784 1176 1568\"] [--batch-size 392] [--batch-sizes \"1 10 100 392\"] [--milestone 10] [--seed 2025] [--gpu 0] [--warmup-count 16]"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --appliance) APPLIANCE="$2"; shift ;;
        --counts) read -ra COUNTS <<< "$2"; shift ;;
        --milestone) MILESTONE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --batch-sizes) read -ra BATCH_SIZES <<< "$2"; shift ;;
        --warmup-count) WARMUP_COUNT="$2"; shift ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
    shift
done

checkpoint=".Checkpoints/Checkpoints_${APPLIANCE}_multivariate_512/checkpoint-${MILESTONE}.pt"

echo "===================================================="
echo "   TC-DiT Sampling Scaling Benchmark"
echo "===================================================="
echo "Appliance: $APPLIANCE"
echo "Counts: ${COUNTS[*]}"
echo "Milestone: $MILESTONE"
echo "Seed: $SEED"
echo "GPU: $GPU"
if [[ "${#BATCH_SIZES[@]}" -gt 0 ]]; then
    echo "Batch sizes: ${BATCH_SIZES[*]}"
else
    echo "Batch size: $BATCH_SIZE"
fi
echo "Warm-up windows: $WARMUP_COUNT"
echo "Checkpoint: $checkpoint"
echo "===================================================="

if [[ ! -f "$checkpoint" ]]; then
    echo "Error: checkpoint not found: $checkpoint" >&2
    echo "Available checkpoints:" >&2
    find .Checkpoints -name "checkpoint-*.pt" -print 2>/dev/null | sort >&2 || true
    exit 1
fi

cmd=(
    python benchmark_sampling_windows.py
    --appliances "$APPLIANCE"
    --counts "${COUNTS[@]}"
    --milestone "$MILESTONE"
    --seed "$SEED"
    --gpu "$GPU"
    --warmup-count "$WARMUP_COUNT"
)

if [[ "${#BATCH_SIZES[@]}" -gt 0 ]]; then
    cmd+=(--batch-sizes "${BATCH_SIZES[@]}")
else
    cmd+=(--batch-size "$BATCH_SIZE")
fi

"${cmd[@]}"

timestamp=$(date +"%Y%m%d_%H%M%S")
csv_src="$OUTPUT_DIR/sampling_window_benchmark.csv"
md_src="$OUTPUT_DIR/sampling_window_benchmark.md"
csv_dst="$OUTPUT_DIR/sampling_window_benchmark_${APPLIANCE}_${timestamp}.csv"
md_dst="$OUTPUT_DIR/sampling_window_benchmark_${APPLIANCE}_${timestamp}.md"

if [[ -f "$csv_src" ]]; then
    cp "$csv_src" "$csv_dst"
    echo "Saved timestamped CSV: $csv_dst"
fi

if [[ -f "$md_src" ]]; then
    cp "$md_src" "$md_dst"
    echo "Saved timestamped Markdown: $md_dst"
fi

if [[ -f "$csv_src" ]]; then
    python analyze_sampling_linearity.py --input "$csv_src" --output-dir "$OUTPUT_DIR"
fi

echo "Benchmark complete."
