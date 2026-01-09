# Set working directory to project root to ensure python script relative paths work
$ProjectRoot = "C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM"
Set-Location -Path $ProjectRoot

$DataDir = Join-Path $ProjectRoot "created_data\UK_DALE"

Write-Host "Working Directory set to: $PWD"
Write-Host "Data Directory: $DataDir"

function Run-Mix-Ratios {
    param(
        [string]$Appliance,
        [string]$FileName
    )
    $fullPath = Join-Path $DataDir $FileName
    
    if (Test-Path $fullPath) {
        Write-Host "`n----------------------------------------------------------------"
        Write-Host "Processing $Appliance (Path: $fullPath)..."
        
        # Count lines efficiently
        Write-Host "Counting rows in real data file..."
        $lineCount = 0
        Get-Content $fullPath -ReadCount 5000 | ForEach-Object { $lineCount += $_.Count }
        
        # Subtract 1 for header
        $totalRealPoints = $lineCount - 1
        
        Write-Host "Total Real Data Points: $totalRealPoints"
        
        # Different Ratios of Synthetic Data
        $ratios = @(0.25, 0.50, 1.00)
        
        foreach ($ratio in $ratios) {
            $synRows = [math]::Floor($totalRealPoints * $ratio)
            $percent = $ratio * 100
            
            Write-Host "`n  [Scenario: Synthetic Data = ${percent}% of Real Data]"
            Write-Host "  > Real Rows: $totalRealPoints"
            Write-Host "  > Synthetic Rows: $synRows"
            
            # Execute python script with specific suffix
            $suffix = "synthetic_${percent}%"
            python mix_training_data_multivariate.py --appliance $Appliance --real_rows $totalRealPoints --synthetic_rows $synRows --real_path "$fullPath" --suffix "$suffix"
        }
    }
    else {
        Write-Host "WARNING: Data file not found for $Appliance at $fullPath"
    }
}

# Run for all appliances
Run-Mix-Ratios "fridge" "fridge_training_.csv"
Run-Mix-Ratios "microwave" "microwave_training_.csv"
Run-Mix-Ratios "kettle" "kettle_training_.csv"
Run-Mix-Ratios "dishwasher" "dishwasher_training_.csv"
Run-Mix-Ratios "washingmachine" "washingmachine_training_.csv"