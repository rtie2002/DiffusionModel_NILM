# Run Algorithm 1 v2 Multivariate for all appliances 
# This processes the Z-Score Converted CNN-SNGAN data to calculate On/Off events and Energy.

$script = "synthetic_data_multivariate/algorithm1_v2_multivariate.py"
$input_base = "Data/datasets/real_distribution_FULL_zscore"
$output_dir = "Data/datasets/real_distributions"

# Ensure output directory exists
if (-not (Test-Path $output_dir)) {
    New-Item -ItemType Directory -Path $output_dir -Force | Out-Null
}

# Dishwasher
Write-Host "Processing Dishwasher (CNN-SNGAN)..."
python $script --appliance_name dishwasher --input_file "$input_base/dishwasher_training_.csv" --output_dir $output_dir

# Fridge
Write-Host "Processing Fridge (CNN-SNGAN)..."
python $script --appliance_name fridge --input_file "$input_base/fridge_training_.csv" --output_dir $output_dir

# Kettle
Write-Host "Processing Kettle (CNN-SNGAN)..."
python $script --appliance_name kettle --input_file "$input_base/kettle_training_.csv" --output_dir $output_dir

# Microwave
Write-Host "Processing Microwave (CNN-SNGAN)..."
python $script --appliance_name microwave --input_file "$input_base/microwave_training_.csv" --output_dir $output_dir

# Washing Machine
Write-Host "Processing Washing Machine (CNN-SNGAN)..."
python $script --appliance_name washingmachine --input_file "$input_base/washingmachine_training_.csv" --output_dir $output_dir

Write-Host "All CNN-SNGAN synthetic appliances processed."
