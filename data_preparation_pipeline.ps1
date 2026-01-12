$appliances = @("fridge", "microwave", "kettle", "dishwasher", "washingmachine")
$dataDir = "NILM-main/dataset_preprocess/UK_DALE/"
$createdDataDir = "created_data/UK_DALE"

# Ensure output directory exists for checking
if (-not (Test-Path $createdDataDir)) {
    Write-Host "Output directory does not exist, scripts will likely create it."
}

foreach ($app in $appliances) {
    Write-Host "`n=== Checking Data for $app ==="
    
    # 1. Testing Data
    $testFile = Join-Path $createdDataDir "${app}_test_.csv"
    if (Test-Path $testFile) {
        Write-Host "  [SKIP] Testing data already exists: $testFile"
    }
    else {
        Write-Host "  [RUN] Generating testing data for $app..."
        python preprocess_NILMformer\multivariate_ukdale_preprocess_testing.py --appliance_name $app --data_dir $dataDir
    }
    
    # 2. Training & Validation Data
    $trainFile = Join-Path $createdDataDir "${app}_training_.csv"
    $valFile = Join-Path $createdDataDir "${app}_validation_.csv"
    
    if ((Test-Path $trainFile) -and (Test-Path $valFile)) {
        Write-Host "  [SKIP] Training/Validation data already exists: $trainFile"
    }
    else {
        Write-Host "  [RUN] Generating training & validation data for $app..."
        python "preprocess_NILMformer\multivariate_ukdale_preprocess_training+validating.py" --appliance_name $app --data_dir $dataDir
    }
}

Write-Host "`n=== Data Preparation Check Complete ==="