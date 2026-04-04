#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================="
echo "   STARTING TS2VEC EVALUATION (ALL TARGETS)  "
echo "============================================="

# Define the appliances exactly as they are in the python script
APPLIANCES=(
    "washingmachine"
    "dishwasher"
    "kettle"
    "microwave"
    "fridge"
)

# Loop through each appliance and execute the script
for appliance in "${APPLIANCES[@]}"; do
    echo ""
    echo "▶️ Evaluating: ${appliance}"
    
    # Run the evaluation script (defaults to --mode all)
    python "Data Quality Checking/evaluate_ts2vec.py" "$appliance"
    
    echo "✅ Finished evaluating: ${appliance}"
done

echo ""
echo "=========================================================================="
echo "                            1. MULTIVARIATE MODE                          "
echo "=========================================================================="
if [ -f "Data Quality Checking/ts2vec_results/global_metrics_multivariate.csv" ]; then
    column -s, -t < "Data Quality Checking/ts2vec_results/global_metrics_multivariate.csv"
else
    echo "No multivariate summary found."
fi

echo ""
echo "=========================================================================="
echo "                            2. POWER MODE ONLY                            "
echo "=========================================================================="
if [ -f "Data Quality Checking/ts2vec_results/global_metrics_power.csv" ]; then
    column -s, -t < "Data Quality Checking/ts2vec_results/global_metrics_power.csv"
else
    echo "No power summary found."
fi

echo ""
echo "=========================================================================="
echo "                            3. TIME MODE ONLY                             "
echo "=========================================================================="
if [ -f "Data Quality Checking/ts2vec_results/global_metrics_time.csv" ]; then
    column -s, -t < "Data Quality Checking/ts2vec_results/global_metrics_time.csv"
else
    echo "No time summary found."
fi

echo "=========================================================================="
echo " Detailed plots saved in 'Data Quality Checking/ts2vec_results/'"
