
param (
    [string[]]$Appliances = @("dishwasher", "fridge", "kettle", "microwave", "washingmachine")
)

$ErrorActionPreference = "Stop"

# Paths
$ScriptDir = "synthetic_CNN_SNGAN"
$PythonScript = Join-Path $ScriptDir "convert_minmax_to_zscore.py"
$TempOutputDir = Join-Path $ScriptDir "zscore_converted"
$FinalOutputDir = "Data\datasets\CNN_SNGAN"

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   CNN-SNGAN Baseline: Data Conversion & Scaling" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# 1. Create Directories
if (-not (Test-Path $FinalOutputDir)) {
    New-Item -ItemType Directory -Path $FinalOutputDir -Force | Out-Null
    Write-Host "Created output directory: $FinalOutputDir" -ForegroundColor Gray
}

# 2. Run Python Conversion
Write-Host "`n>>> Running MinMax to Z-Score conversion script..." -ForegroundColor Yellow
python $PythonScript --input_dir $ScriptDir --output_dir $TempOutputDir

if ($LASTEXITCODE -ne 0) {
    Write-Error "Python conversion script failed."
}

# 3. Rename and Move to Final Destination
Write-Host "`n>>> Organizing files for evaluation (renaming to standard format)..." -ForegroundColor Yellow

foreach ($app in $Appliances) {
    # The python script names them synthetic_{app}_zscore.csv
    $sourceFile = Join-Path $TempOutputDir "synthetic_$($app)_zscore.csv"
    $destFile = Join-Path $FinalOutputDir "$($app)_multivariate.csv"

    if (Test-Path $sourceFile) {
        Copy-Item -Path $sourceFile -Destination $destFile -Force
        Write-Host "  [OK] $($sourceFile.PadRight(40)) -> $destFile" -ForegroundColor Green
    }
    else {
        Write-Warning " Warning: Output for $app not found at $sourceFile"
    }
}

Write-Host "`n====================================================" -ForegroundColor Cyan
Write-Host "   Conversion Complete! Files ready in: $FinalOutputDir" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
