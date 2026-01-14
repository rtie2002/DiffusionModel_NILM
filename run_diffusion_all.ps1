
param (
    [string[]]$Appliances = @("fridge", "microwave", "kettle", "dishwasher", "washingmachine"),
    [switch]$Train,
    [switch]$Sample,
    [int]$Milestone = 10, # Note: Milestone is the checkpoint index (e.g., 10 for 20,000 steps with 2,000 save cycle)
    [int]$SampleNum = 5000,
    [int]$Gpu = 0,
    [float]$Proportion = 1.0 # Added: Use to reduce data size if RAM is limited (e.g., 0.5)
)

$ErrorActionPreference = "Stop"

# If neither Train nor Sample is specified, do both
if (-not $Train -and -not $Sample) {
    $Train = $true
    $Sample = $true
}

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   Diffusion Model Automation: Train & Sample" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Appliances: $($Appliances -join ', ')"
Write-Host "GPU ID: $Gpu"
Write-Host "Proportion: $Proportion"
Write-Host "Flags: --tensorboard (Training), _multivariate (Naming)" -ForegroundColor Gray
Write-Host "Steps: $(if ($Train) { 'Training ' })$(if ($Train -and $Sample) { '& ' })$(if ($Sample) { 'Sampling' })"
Write-Host "====================================================" -ForegroundColor Cyan

foreach ($app in $Appliances) {
    Write-Host "`n>>> Processing Appliance: [$($app.ToUpper())]" -ForegroundColor Yellow
    
    $configPath = "Config/$app.yaml"
    if (-not (Test-Path $configPath)) {
        Write-Warning "Config file not found: $configPath. Skipping..."
        continue
    }

    # --- Step 1: Training ---
    if ($Train) {
        Write-Host "--- [1/2] Starting Training for $app ---" -ForegroundColor Green
        $trainArgs = @(
            "main.py",
            "--train",
            "--name", "${app}_multivariate",
            "--config", $configPath,
            "--tensorboard",
            "--gpu", $Gpu,
            "--opts", "dataloader.train_dataset.params.save2npy", "False", 
            "dataloader.train_dataset.params.proportion", $Proportion
        )
        
        Write-Host "Running: python $($trainArgs -join ' ')" -ForegroundColor Gray
        python @trainArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Training failed for $app with exit code $LASTEXITCODE"
        }
    }

    # --- Step 2: Sampling ---
    if ($Sample) {
        Write-Host "--- [2/2] Starting Sampling for $app ---" -ForegroundColor Green
        
        # --- NEW: Calculate SampleNum dynamically to match 100% of original data size ---
        $configContent = Get-Content $configPath -Raw
        
        # 1. Extract window size
        $window = 512 # fallback
        if ($configContent -match "window:\s*(\d+)") {
            $window = [int]$matches[1]
        }
        
        # 2. Extract data path
        $dataPath = ""
        if ($configContent -match "data_root:\s*([^\s#]+)") {
            $dataPath = $matches[1]
        }
        
        $dynamicSampleNum = $SampleNum # Fallback to param
        
        if ($dataPath) {
            # Handle relative paths in config (relative to project root usually)
            $fullDataPath = $dataPath
            if (-not [System.IO.Path]::IsPathRooted($fullDataPath)) {
                $fullDataPath = Join-Path $PWD $fullDataPath
            }
            
            if (Test-Path $fullDataPath) {
                Write-Host "  -> Calculating SampleNum from: $fullDataPath" -ForegroundColor Gray
                # Count lines minus header
                $rowCount = 0
                Get-Content $fullDataPath -ReadCount 10000 | ForEach-Object { $rowCount += $_.Count }
                $rowCount = $rowCount - 1 # Subtract header
                
                if ($rowCount -gt 0) {
                    $dynamicSampleNum = [math]::Floor($rowCount / $window)
                    Write-Host "  -> Original Rows: $rowCount, Window: $window" -ForegroundColor Gray
                    Write-Host "  -> Dynamic SampleNum: $dynamicSampleNum (to cover $rowCount points)" -ForegroundColor Cyan
                }
            }
            else {
                Write-Warning "  -> Could not verify data path: $fullDataPath. Using default SampleNum."
            }
        }

        # Note: milestone defaults to 10 which is current checkpoint index
        $sampleArgs = @(
            "main.py",
            "--name", "${app}_multivariate",
            "--config", $configPath,
            "--sample", 1,
            "--milestone", $Milestone,
            "--sample_num", $dynamicSampleNum,
            "--gpu", $Gpu
        )

        Write-Host "Running: python $($sampleArgs -join ' ')" -ForegroundColor Gray
        python @sampleArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Sampling failed for $app with exit code $LASTEXITCODE"
        }
        
        # Verify output
        $expectedOutput = "OUTPUT/$app/ddpm_fake_$app.npy"
        if (Test-Path $expectedOutput) {
            Write-Host "Successfully generated: $expectedOutput" -ForegroundColor Cyan
        }
        else {
            Write-Warning "Output file not found at expected location: $expectedOutput"
        }
    }
}

Write-Host "`n====================================================" -ForegroundColor Cyan
Write-Host "   All tasks completed!" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
