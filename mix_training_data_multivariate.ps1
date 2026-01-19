# Set working directory to project root to ensure python script relative paths work
$ProjectRoot = $PSScriptRoot
Set-Location -Path $ProjectRoot

$DataDir = Join-Path $ProjectRoot "created_data\UK_DALE"

Write-Host "Working Directory set to: $PWD"
Write-Host "Data Directory: $DataDir"

# ==============================================================================
# AUTO-FIX: Expand Synthetic Data Files
# The mixing tool caps output at the shortest synthetic file length.
# We must ensure all source NPY files are large enough (e.g. > 2500 windows)
# to support 100% data injection scenarios.
# ==============================================================================
$TempExpandScript = "temp_expand_synthetic.py"
$PythonCode = @"
import numpy as np
import glob
import os
import shutil

# Target size: 2500 windows * 512 points = 1.28M points (enough for >100% of ~1M real data)
TARGET_WINDOWS = 2500 

search_path = os.path.join('synthetic_data_multivariate', 'ddpm_fake_*_multivariate.npy')
files = glob.glob(search_path)

print(f'\n[Auto-Expand] Checking {len(files)} synthetic files...')

for f in files:
    try:
        data = np.load(f)
        if data.shape[0] < TARGET_WINDOWS:
            print(f'  Expanding {os.path.basename(f)}: {data.shape[0]} windows -> {TARGET_WINDOWS} windows')
            
            # Backup original if not exists
            if not os.path.exists(f + '.bak'):
                shutil.copy2(f, f + '.bak')
            
            # Tile/Repeat data to reach target size
            repeats = int(np.ceil(TARGET_WINDOWS / data.shape[0]))
            new_data = np.tile(data, (repeats, 1, 1))[:TARGET_WINDOWS]
            
            np.save(f, new_data)
            print('  -> Expanded and saved.')
        else:
            # print(f'  {os.path.basename(f)} is OK ({data.shape[0]} windows)')
            pass
            
    except Exception as e:
        print(f'  Error processing {f}: {e}')
print('[Auto-Expand] Check complete.\n')
"@

# Write and execute temp script
Set-Content -Path $TempExpandScript -Value $PythonCode
python $TempExpandScript
if (Test-Path $TempExpandScript) { Remove-Item $TempExpandScript }
# ==============================================================================

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
        $ratios = @(0.25, 0.50, 1.00, 2.00)
        
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