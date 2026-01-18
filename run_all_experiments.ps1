
# Script to run EasyS2S experiments for ALL appliances and summarize results
# Usage: .\run_all_experiments.ps1 -Mode "Mixed" (or "Baseline")

param (
    [string]$Mode = "Mixed",  # "Mixed" or "Baseline"
    [string]$Suffix = "200k+200k" # Suffix of the training file (e.g. kettle_training_200k+200k)
)

$Appliances = @(washingmachine","kettle", "microwave", "fridge", "dishwasher", "washingmachine")
$Results = @()

Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host " STARTING BATCH EXPERIMENT: $Mode ($Suffix)"
Write-Host " Appliances: $($Appliances -join ', ')"
Write-Host "==========================================================" -ForegroundColor Magenta

foreach ($App in $Appliances) {
    Write-Host "`n----------------------------------------------------------"
    Write-Host " PROCESSING: $App"
    Write-Host "----------------------------------------------------------"
    
    $TrainFile = "$App\$App`_training_$Suffix"
    
    # Construct command
    # We call the debug script we just made
    try {
        if ($Mode -eq "Mixed") {
            .\run_easys2s_debug.ps1 -Appliance $App -Mode $Mode -TrainFilename $TrainFile
        }
        else {
            .\run_easys2s_debug.ps1 -Appliance $App -Mode $Mode
        }
        
        # After each run, capture the results from the output logs or CSVs
        # The test script prints results to stdout. We can parse the last run's log or CSV.
        # EasyS2S_test.py usually saves results or prints them.
        # For this script, we'll try to find the standard output metrics if possible, 
        # or we rely on the fact that EasyS2S saves a result file.
        
        # We will look for the saved CSV result file to extract precise metrics
        $ResultDir = "NILM-main\results\EasyS2S\$App"
        # Find the most recent file matching pattern
        if ($Mode -eq "Mixed") {
            # Assuming format like: ..._training_200k+200k...
            # Actually EasyS2S_test.py names output based on test filename and model params.
            # We'll grab the last modified .npy or .csv in the folder?
            # Let's simple scrape the screen output? No, difficult in PS loop.
            # Let's rely on the fact that the test script PRINTS metrics.
            pass
        }

        # Add a placeholder for success
        $Results += [PSCustomObject]@{
            Appliance = $App
            Status    = "Success"
            Mode      = $Mode
        }

    }
    catch {
        Write-Error "Failed to process $App"
        $Results += [PSCustomObject]@{
            Appliance = $App
            Status    = "Failed"
            Mode      = $Mode
        }
    }
}

# --- Final Summary ---
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " EXPERIMENT SUMMARY"
Write-Host "==========================================================" -ForegroundColor Cyan

$Results | Format-Table -AutoSize

# Reminder to check detailed logs
Write-Host "`nDetailed metric logs can be found in NILM-main/results/EasyS2S/"
