
param (
    [string[]]$Appliances = @("fridge", "microwave", "kettle", "dishwasher", "washingmachine"),
    [switch]$Train,
    [switch]$Sample,
    [int]$Milestone = 10, # Note: Milestone is the checkpoint index (e.g., 10 for 20,000 steps with 2,000 save cycle)
    [int]$SampleNum = 0, # Default to 0 to trigger automatic calculation from CSV
    [int]$Gpu = 0,
    [int]$Seed = 2025,
    [float]$Proportion = 1.0 # Added: Use to reduce data size if RAM is limited (e.g., 0.5)
)

$ErrorActionPreference = "Stop"

function Format-Duration {
    param($Seconds)

    if ($null -eq $Seconds -or $Seconds -eq "") {
        return "NA"
    }

    $ts = [TimeSpan]::FromSeconds([double]$Seconds)
    if ($ts.TotalHours -ge 1) {
        return "{0:00}:{1:00}:{2:00}" -f [int]$ts.TotalHours, $ts.Minutes, $ts.Seconds
    }

    return "{0:00}:{1:00}" -f $ts.Minutes, $ts.Seconds
}

function Get-NpyShape {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $null
    }

    $shapeCode = "import sys, numpy as np; a=np.load(sys.argv[1], mmap_mode='r'); print(','.join(map(str, a.shape)))"
    try {
        $shape = (& python -c $shapeCode $Path 2>$null)
        if ($LASTEXITCODE -eq 0 -and $shape) {
            return $shape.Trim()
        }
    }
    catch {
        return $null
    }

    return $null
}

function Get-LatestRuntimeFromLog {
    param(
        [string]$LogPath,
        [string]$Label
    )

    if (-not (Test-Path $LogPath)) {
        return $null
    }

    $hits = @(Select-String -Path $LogPath -Pattern "$Label, time:")
    if ($hits.Count -eq 0) {
        return $null
    }

    $line = $hits[-1].Line
    if ($line -match "time:\s*([0-9.]+)") {
        return [double]$Matches[1]
    }

    return $null
}

function Get-LatestParameterCountFromLog {
    param([string]$LogPath)

    if (-not (Test-Path $LogPath)) {
        return "NA"
    }

    $hits = @(Select-String -Path $LogPath -Pattern "overall.*trainable")
    if ($hits.Count -eq 0) {
        return "NA"
    }

    $line = $hits[-1].Line
    if ($line -match "overall.*trainable': '([^']+)'") {
        return $Matches[1]
    }

    return "NA"
}

function Write-ReproducibilitySummary {
    param([array]$Rows)

    $summaryDir = "OUTPUT"
    New-Item -ItemType Directory -Path $summaryDir -Force | Out-Null

    $csvPath = Join-Path $summaryDir "revision_reproducibility_summary.csv"
    $mdPath = Join-Path $summaryDir "revision_reproducibility_summary.md"

    $Rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

    $mdLines = @()
    $mdLines += "| Appliance | Generated samples | Random seed | Training time | Sampling time | Model parameters | Output shape |"
    $mdLines += "|---|---:|---:|---:|---:|---:|---|"
    foreach ($row in $Rows) {
        $mdLines += "| $($row.Appliance) | $($row.GeneratedSamples) | $($row.RandomSeed) | $($row.TrainingTime) | $($row.SamplingTime) | $($row.ModelParameters) | $($row.OutputShape) |"
    }

    Set-Content -Path $mdPath -Value $mdLines -Encoding UTF8

    Write-Host "`nReproducibility summary saved to:" -ForegroundColor Cyan
    Write-Host "  $csvPath" -ForegroundColor Cyan
    Write-Host "  $mdPath" -ForegroundColor Cyan
}

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
Write-Host "Random Seed: $Seed"
Write-Host "Proportion: $Proportion"
Write-Host "Flags: --tensorboard (Training), _multivariate (Naming)" -ForegroundColor Gray
Write-Host "Steps: $(if ($Train) { 'Training ' })$(if ($Train -and $Sample) { '& ' })$(if ($Sample) { 'Sampling' })"
Write-Host "====================================================" -ForegroundColor Cyan

$summaryRows = @()

foreach ($app in $Appliances) {
    Write-Host "`n>>> Processing Appliance: [$($app.ToUpper())]" -ForegroundColor Yellow

    $runName = "${app}_multivariate"
    $outputDir = Join-Path "OUTPUT" $runName
    $logPath = Join-Path $outputDir "logs/log.txt"
    $expectedOutput = Join-Path $outputDir "ddpm_fake_${runName}.npy"
    $trainElapsedSeconds = $null
    $sampleElapsedSeconds = $null
    $dynamicSampleNum = $null
    
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
            "--name", $runName,
            "--config", $configPath,
            "--tensorboard",
            "--gpu", $Gpu,
            "--seed", $Seed,
            "--opts", "dataloader.train_dataset.params.save2npy", "False", 
            "dataloader.train_dataset.params.proportion", $Proportion,
            "dataloader.train_dataset.params.seed", $Seed
        )
        
        Write-Host "Running: python $($trainArgs -join ' ')" -ForegroundColor Gray
        $trainTimer = [System.Diagnostics.Stopwatch]::StartNew()
        python @trainArgs
        $trainTimer.Stop()
        $trainElapsedSeconds = [Math]::Round($trainTimer.Elapsed.TotalSeconds, 2)
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
                        
                        $dynamicSampleNum = 2 * [math]::Ceiling($totalPoints / $window)
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
            "--name", $runName,
            "--config", $configPath,
            "--sample", 1,
            "--milestone", $Milestone,
            "--sample_num", $dynamicSampleNum,
            "--sampling_mode", "ordered_non_overlapping",
            "--gpu", $Gpu,
            "--seed", $Seed,
            "--opts", "dataloader.train_dataset.params.seed", $Seed
        )

        Write-Host "Running: python $($sampleArgs -join ' ')" -ForegroundColor Gray
        $sampleTimer = [System.Diagnostics.Stopwatch]::StartNew()
        python @sampleArgs
        $sampleTimer.Stop()
        $sampleElapsedSeconds = [Math]::Round($sampleTimer.Elapsed.TotalSeconds, 2)
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Sampling failed for $app with exit code $LASTEXITCODE"
        }
        
        # Verify output
        if (Test-Path $expectedOutput) {
            Write-Host "Successfully generated: $expectedOutput" -ForegroundColor Cyan
        }
        else {
            Write-Warning "Output file not found at expected location: $expectedOutput"
        }
    }

    if ($null -eq $trainElapsedSeconds) {
        $trainElapsedSeconds = Get-LatestRuntimeFromLog -LogPath $logPath -Label "Training done"
    }
    if ($null -eq $sampleElapsedSeconds) {
        $sampleElapsedSeconds = Get-LatestRuntimeFromLog -LogPath $logPath -Label "Sampling done"
    }

    $outputShape = Get-NpyShape -Path $expectedOutput
    $generatedSamples = "NA"
    if ($outputShape) {
        $generatedSamples = $outputShape.Split(",")[0]
    }
    elseif ($null -ne $dynamicSampleNum -and $dynamicSampleNum -gt 0) {
        $generatedSamples = $dynamicSampleNum
        $outputShape = "not found"
    }
    else {
        $outputShape = "not found"
    }

    $summaryRows += [PSCustomObject]@{
        Appliance = $app
        GeneratedSamples = $generatedSamples
        RandomSeed = $Seed
        TrainingTimeSeconds = $(if ($null -eq $trainElapsedSeconds) { "NA" } else { $trainElapsedSeconds })
        TrainingTime = Format-Duration $trainElapsedSeconds
        SamplingTimeSeconds = $(if ($null -eq $sampleElapsedSeconds) { "NA" } else { $sampleElapsedSeconds })
        SamplingTime = Format-Duration $sampleElapsedSeconds
        ModelParameters = Get-LatestParameterCountFromLog -LogPath $logPath
        OutputShape = $outputShape
        OutputFile = $expectedOutput
    }
}

Write-ReproducibilitySummary -Rows $summaryRows

Write-Host "`n====================================================" -ForegroundColor Cyan
Write-Host "   All tasks completed!" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
