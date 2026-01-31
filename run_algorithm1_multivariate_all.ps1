# Run Algorithm 1 v2 Multivariate for all appliances

$script = "synthetic_data_multivariate/algorithm1_v2_multivariate.py"
$output_dir = "Data/datasets/real_distributions"

# Dishwasher
Write-Host "Processing Dishwasher..."
python $script --appliance_name dishwasher --input_file "created_data/UK_DALE/dishwasher_training_.csv" --output_dir $output_dir

# Fridge
Write-Host "Processing Fridge..."
python $script --appliance_name fridge --input_file "created_data/UK_DALE/fridge_training_.csv" --output_dir $output_dir

# Kettle
Write-Host "Processing Kettle..."
python $script --appliance_name kettle --input_file "created_data/UK_DALE/kettle_training_.csv" --output_dir $output_dir

# Microwave
Write-Host "Processing Microwave..."
python $script --appliance_name microwave --input_file "created_data/UK_DALE/microwave_training_.csv" --output_dir $output_dir

# Washing Machine
Write-Host "Processing Washing Machine..."
python $script --appliance_name washingmachine --input_file "created_data/UK_DALE/washingmachine_training_.csv" --output_dir $output_dir

Write-Host "All appliances processed."
