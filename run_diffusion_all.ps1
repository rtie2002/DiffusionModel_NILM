
param (
    [string[]]$Appliances = @("fridge", "microwave", "kettle", "dishwasher", "washingmachine"),
    [switch]$Train,
    [switch]$Sample,
    [int]$Milestone = 10, # Note: Milestone is the checkpoint index (e.g., 10 for 20,000 steps with 2,000 save cycle)
    [int]$SampleNum = 0, # Default to 0 to trigger automatic calculation from CSV
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
        
        $dynamicSampleNum = $SampleNum
        
        if ($dataPath) {
            $dataPath = $dataPath.Trim("'").Trim("`"")
            $fullDataPath = $dataPath
            if (-not [System.IO.Path]::IsPathRooted($fullDataPath)) {
                $fullDataPath = Join-Path $PWD $fullDataPath
            }
            
            if (Test-Path $fullDataPath) {
                # If SampleNum is 0, we MUST calculate it.
                if ($dynamicSampleNum -eq 0) {
                    # Clean up path (removes .\ and other redundancies)
                    $fullDataPath = [System.IO.Path]::GetFullPath($fullDataPath)
                    Write-Host "  -> Calculating SampleNum from: $fullDataPath" -ForegroundColor Gray
                    
                    # PERFORMANCE FIX: Using Get-Content with -ReadCount is robust for large files in PowerShell
                    $lineCount = 0
                    Get-Content $fullDataPath -ReadCount 10000 | ForEach-Object { $lineCount += $_.Count }
                    
                    if ($lineCount -gt 1) {
                        $totalPoints = $lineCount - 1 # Subtract Header
                        
                        $dynamicSampleNum = [math]::Ceiling($totalPoints / $window)
                        Write-Host "  -> Found $totalPoints data points. Window size: $window" -ForegroundColor Gray
                        Write-Host "  -> Dynamic SampleNum set to: $dynamicSampleNum windows" -ForegroundColor Cyan
                    }
                }
                else {
                    Write-Host "  -> Using manually specified SampleNum: $dynamicSampleNum" -ForegroundColor Gray
                }
            }
            else {
                if ($dynamicSampleNum -eq 0) {
                    Write-Warning "  -> Could not find data file and SampleNum is 0. Using fallback of 1000."
                    $dynamicSampleNum = 1000
                }
            }
        }
        
        if ($dynamicSampleNum -eq 0) { $dynamicSampleNum = 1000 } # Final safety fallback

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
