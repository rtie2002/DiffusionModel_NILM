# Function to find the maximum milestone from checkpoint files to continue training or for sampling
function Get-MaxMilestone {
    param (
        [string]$CheckpointDir
    )
    if (Test-Path $CheckpointDir) {
        $files = Get-ChildItem -Path $CheckpointDir -Filter "checkpoint-*.pt"
        if ($files) {
            $milestones = $files | ForEach-Object {
                if ($_.Name -match 'checkpoint-(\d+).pt') {
                    [int]$matches[1]
                }
            }
            if ($milestones) {
                return ($milestones | Measure-Object -Maximum).Maximum
            }
        }
    }
    return 0
}

# Function to run training and sampling pipeline
function Run-Pipeline {
    param (
        [string]$ApplianceName,
        [string]$ConfigPath
    )
    
    $runName = "${ApplianceName}_multivariate"
    Write-Host "`n========================================================"
    Write-Host "Processing $ApplianceName"
    Write-Host "========================================================"
    
    # 1. Train
    Write-Host "Starting training for $runName..."
    python main.py --train --config $ConfigPath --name $runName --tensorboard
    
    # 2. Find Max Milestone
    $chkptDir = ".Checkpoints/Checkpoints_${runName}_512"
    
    Write-Host "Searching for checkpoints in $chkptDir..."
    $milestone = Get-MaxMilestone -CheckpointDir $chkptDir
    
    if ($milestone -gt 0) {
        Write-Host "Found max milestone for ${ApplianceName}: $milestone"
        
        # 3. Calculate Sample Number (Total Data Points / 512)
        $dataPath = "Data/datasets/${runName}.csv"
        if (Test-Path $dataPath) {
            Write-Host "Reading data file: $dataPath to calculate distinct windows..."
            # Count lines - 1 (header)
            $lineCount = 0
            Get-Content $dataPath -ReadCount 1000 | ForEach-Object { $lineCount += $_.Count }
            $dataPoints = $lineCount - 1
            
            # Formula: num_samples = floor(dataPoints / 512)
            $numSamples = [math]::Floor($dataPoints / 512)
            
            Write-Host "  Total data points: $dataPoints"
            Write-Host "  Window size: 512"
            Write-Host "  Calculated sample_num: $numSamples"
            
            # 4. Sample
            Write-Host "Starting sampling for $runName with milestone $milestone and samples $numSamples..."
            python main.py --config $ConfigPath --name $runName --milestone $milestone --sample_num $numSamples
        }
        else {
            Write-Host "ERROR: Data file not found at $dataPath. Cannot calculate sample_num."
            Write-Host "Running sampling with default (full dataset size)..."
            python main.py --config $ConfigPath --name $runName --milestone $milestone
        }
    }
    else {
        Write-Host "ERROR: No checkpoint found for $ApplianceName in $chkptDir"
    }
}

# Run pipeline for all appliances
Run-Pipeline "washingmachine" "Config/washingmachine.yaml"
Run-Pipeline "kettle" "Config/kettle.yaml"
Run-Pipeline "microwave" "Config/microwave.yaml"
Run-Pipeline "dishwasher" "Config/dishwasher.yaml"
Run-Pipeline "fridge" "Config/fridge.yaml"