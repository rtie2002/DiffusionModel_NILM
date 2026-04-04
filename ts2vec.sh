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
echo "============================================="
echo " 🎉 ALL EVALUATIONS COMPLETED SUCCESSFULLY! "
echo " Check 'Data Quality Checking/ts2vec_results/' for outputs."
echo "============================================="
