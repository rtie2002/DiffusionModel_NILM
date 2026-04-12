param (
    [string]$Appliance = "washingmachine",
    [string]$BaseConfig = "Config/washingmachine.yaml",
    [string]$InputCsv = "washingmachine_multivariate.csv",
    [string]$DataDir = "."
)

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host " STARTING QUARTERLY PIPELINE FOR: $Appliance" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# 1. Split the Data
Write-Host "`n[1/4] Splitting $InputCsv into quarters..." -ForegroundColor Yellow
python preprocess_NILMformer/split_csv_by_quarter.py --input $InputCsv --output_dir $DataDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error splitting data. Exiting." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Detect Available Quarters
$availableQuarters = @()
$quarters = @("Q1", "Q2", "Q3", "Q4")
foreach ($q in $quarters) {
    $csvPath = Join-Path $DataDir "${Appliance}_multivariate_${q}.csv"
    if (Test-Path $csvPath) {
        $availableQuarters += $q
    }
}

if ($availableQuarters.Count -eq 0) {
    Write-Host "No quarterly CSVs created. Exiting." -ForegroundColor Red
    exit 1
}

# 3. Generate Configs
Write-Host "`n[2/4] Generating YAML configs for $Appliance..." -ForegroundColor Yellow
python generate_quarter_configs.py --base_config $BaseConfig --appliance $Appliance --csv_dir $DataDir --quarters $availableQuarters
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error generating configs. Exiting." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 4. Train the Models
Write-Host "`n[3/4] Starting Training Loop..." -ForegroundColor Yellow

foreach ($q in $availableQuarters) {
    $configPath = "Config/${Appliance}_${q}.yaml"
    Write-Host "`n-------------------------------------------------------" -ForegroundColor Green
    Write-Host " TRAINING QUARTER: $q" -ForegroundColor Green
    Write-Host "-------------------------------------------------------" -ForegroundColor Green
    python main.py --train --config $configPath --name "${Appliance}_${q}"
}

# 5. Sampling and Concatenation (Example for 200%)
Write-Host "`n[4/4] Starting Concatenated Sampling..." -ForegroundColor Yellow
# This part involves a more complex loop for 200% logic, 
# for now we'll refer users to run_quarterly_diffusion.sh for full orchestration.

Write-Host "`nPipeline scripts restored. Use run_quarterly_diffusion.sh for full Cycle/Concat logic." -ForegroundColor Cyan
