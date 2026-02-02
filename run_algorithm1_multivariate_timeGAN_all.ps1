# Run Algorithm 1 v2 Multivariate for all appliances (TimeGAN Synthetic Data)
# This processes the Z-Score Converted TimeGAN data to calculate On/Off events and Energy.

$script = "synthetic_data_multivariate/algorithm1_v2_multivariate.py"
$input_base = "Data/datasets/timeGAN_synthetic/zscore_converted"
$output_dir = "Data/datasets/timeGAN_synthetic"

# Dishwasher
Write-Host "Processing Dishwasher (TimeGAN)..."
python $script --appliance_name dishwasher --input_file "$input_base/dishwasher_multivariate.csv" --output_dir $output_dir

# Fridge
Write-Host "Processing Fridge (TimeGAN)..."
python $script --appliance_name fridge --input_file "$input_base/fridge_multivariate.csv" --output_dir $output_dir

# Kettle
Write-Host "Processing Kettle (TimeGAN)..."
python $script --appliance_name kettle --input_file "$input_base/kettle_multivariate.csv" --output_dir $output_dir

# Microwave
Write-Host "Processing Microwave (TimeGAN)..."
python $script --appliance_name microwave --input_file "$input_base/microwave_multivariate.csv" --output_dir $output_dir

# Washing Machine
Write-Host "Processing Washing Machine (TimeGAN)..."
python $script --appliance_name washingmachine --input_file "$input_base/washingmachine_multivariate.csv" --output_dir $output_dir

Write-Host "All TimeGAN synthetic appliances processed."
