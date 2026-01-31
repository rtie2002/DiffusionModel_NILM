# Run Algorithm 1 v2 Multivariate for all appliances (Synthetic Data)

$script = "synthetic_data_multivariate/algorithm1_v2_multivariate.py"
$output_dir = "Data/datasets/synthetic_processed"

# Dishwasher
Write-Host "Processing Dishwasher (Synthetic)..."
python $script --appliance_name dishwasher --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_dishwasher_multivariate_zscore.csv" --output_dir $output_dir

# Fridge
Write-Host "Processing Fridge (Synthetic)..."
python $script --appliance_name fridge --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_fridge_multivariate_zscore.csv" --output_dir $output_dir

# Kettle
Write-Host "Processing Kettle (Synthetic)..."
python $script --appliance_name kettle --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_kettle_multivariate_zscore.csv" --output_dir $output_dir

# Microwave
Write-Host "Processing Microwave (Synthetic)..."
python $script --appliance_name microwave --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_microwave_multivariate_zscore.csv" --output_dir $output_dir

# Washing Machine
Write-Host "Processing Washing Machine (Synthetic)..."
python $script --appliance_name washingmachine --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_washingmachine_multivariate_zscore.csv" --output_dir $output_dir

Write-Host "All synthetic appliances processed."
