# Synthetic Data Processing - Algorithm 1 Multivariate

Use these commands to apply Algorithm 1 filtering to the **Z-score converted synthetic data**.
This ensures the synthetic data used for downstream tasks follows the same filtering rules (ON-events only, MinMax normalization) as the real training data.

> **Input**: `synthetic_data_multivariate/zscore_converted/*.csv` (Z-score Power + Time)
> **Output**: `Data/datasets/{appliance}_synthetic_multivariate.csv` (MinMax Power + Time)

---

## 1. Dishwasher
```bash
python synthetic_data_multivariate/algorithm1_v2_multivariate.py --appliance_name dishwasher --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_dishwasher_multivariate_zscore.csv" --output_dir "Data/datasets/synthetic_processed"
```

## 2. Fridge
```bash
python synthetic_data_multivariate/algorithm1_v2_multivariate.py --appliance_name fridge --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_fridge_multivariate_zscore.csv" --output_dir "Data/datasets/synthetic_processed"
```

## 3. Kettle
```bash
python synthetic_data_multivariate/algorithm1_v2_multivariate.py --appliance_name kettle --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_kettle_multivariate_zscore.csv" --output_dir "Data/datasets/synthetic_processed"
```

## 4. Microwave
```bash
python synthetic_data_multivariate/algorithm1_v2_multivariate.py --appliance_name microwave --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_microwave_multivariate_zscore.csv" --output_dir "Data/datasets/synthetic_processed"
```

## 5. Washing Machine
```bash
python synthetic_data_multivariate/algorithm1_v2_multivariate.py --appliance_name washingmachine --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_washingmachine_multivariate_zscore.csv" --output_dir "Data/datasets/synthetic_processed"
```
```bash
python synthetic_data_multivariate/algorithm1_v2_multivariate.py --appliance_name washingmachine --input_file "synthetic_data_multivariate/zscore_converted/washingmachine_training_.csv" --output_dir "Data/datasets/synthetic_processed"
```

```bash
python synthetic_data_multivariate/algorithm1_v2_multivariate.py --appliance_name dishwasher --input_file "Data\dishwasher_training_.csv" --output_dir "Data/datasets/synthetic_processed"
```
---

### Batch Script (PowerShell)
You can run all of them at once using this PowerShell snippet:

```powershell
$appliances = @("dishwasher", "fridge", "kettle", "microwave", "washingmachine")

foreach ($app in $appliances) {
    Write-Host "Processing Synthetic Data for: $app" -ForegroundColor Cyan
    python synthetic_data_multivariate/algorithm1_v2_multivariate.py `
        --appliance_name $app `
        --input_file "synthetic_data_multivariate/zscore_converted/ddpm_fake_${app}_multivariate_zscore.csv" `
        --output_dir "Data/datasets/synthetic_processed"
}
```
