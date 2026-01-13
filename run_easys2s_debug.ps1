
param (
    [string]$Appliance = "fridge",
    [string]$Mode = "Mixed",  # "Mixed" (Robust loss) or "Baseline" (MSE / originModel=True)
    [string]$TrainFilename,   # Optional: Specific filename to use for training
    [int]$Epochs = 100,
    [int]$BatchSize = 1024
)

$ErrorActionPreference = "Stop"

# --- Configuration ---
$NILM_MAIN_DIR = "NILM-main"
$TRAIN_SCRIPT = "$NILM_MAIN_DIR\EasyS2S_train.py"
$TEST_SCRIPT = "$NILM_MAIN_DIR\EasyS2S_test.py"

# --- Determine Variables based on Mode ---
if ($Mode -eq "Baseline") {
    $OriginModel = "True"
    $TrainPercent = "20" # Or whatever baseline uses
    Write-Host "Running in BASELINE mode (originModel=True)" -ForegroundColor Cyan
}
else {
    $OriginModel = "False"
    $TrainPercent = "20"
    Write-Host "Running in MIXED/ROBUST mode (originModel=False)" -ForegroundColor Cyan
}

# --- Helper Function to Modify Python Scripts ---
function Set-PythonVar {
    param ($FilePath, $VarName, $NewValue, $IsString = $false)
    
    $CurrentContent = Get-Content $FilePath -Raw
    
    # Regex to find variable assignment (e.g., originModel=True or originModel = False)
    # Handles potential spaces around = and existing quotes
    if ($IsString) {
        # Replace: var = 'value' or var="value"
        $Pattern = "$VarName\s*=\s*['`"][^'`"]*['`"]"
        $Replacement = "$VarName='$NewValue'"
    }
    else {
        # Replace: var = True or var=10
        $Pattern = "$VarName\s*=\s*[A-Za-z0-9_.]+"
        $Replacement = "$VarName=$NewValue"
    }

    if ($CurrentContent -match $Pattern) {
        $NewContent = $CurrentContent -replace $Pattern, $Replacement
        Set-Content -Path $FilePath -Value $NewContent
        Write-Host "  -> Updated $VarName to $NewValue in $FilePath" -ForegroundColor Gray
    }
    else {
        Write-Warning "  -> Could not find variable '$VarName' in $FilePath to update."
    }
}

# --- Step 1: Modify Training Script ---
Write-Host "`n[1/3] Configuring Training Script..." -ForegroundColor Green
Set-PythonVar -FilePath $TRAIN_SCRIPT -VarName "originModel" -NewValue $OriginModel
Set-PythonVar -FilePath $TRAIN_SCRIPT -VarName "datasetName" -NewValue "UK_DALE" -IsString $true
Set-PythonVar -FilePath $TRAIN_SCRIPT -VarName "applianceName" -NewValue $Appliance -IsString $true
Set-PythonVar -FilePath $TRAIN_SCRIPT -VarName "TrainPercent" -NewValue $TrainPercent -IsString $true

# --- Step 2: Run Training ---
Write-Host "`n[2/3] Starting Training..." -ForegroundColor Green

$TrainCmdArgs = @("EasyS2S_train.py", "--appliance_name", $Appliance, "--n_epoch", $Epochs, "--batchsize", $BatchSize)

# If a specific filename is provided (e.g. for Mixed data), use it
if ($TrainFilename) {
    $TrainCmdArgs += "--train_filename"
    $TrainCmdArgs += $TrainFilename
    Write-Host "  -> Using specific training file: $TrainFilename" -ForegroundColor Yellow
}

# Run python command inside NILM-main directory
Push-Location $NILM_MAIN_DIR
try {
    Write-Host "Executing: python $($TrainCmdArgs -join ' ')" -ForegroundColor Gray
    python @TrainCmdArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

# --- Step 3: Modify Testing Script ---
Write-Host "`n[3/3] Configuring & Running Test..." -ForegroundColor Green

# Update Test Script Variables
Set-PythonVar -FilePath $TEST_SCRIPT -VarName "originModel" -NewValue $OriginModel
Set-PythonVar -FilePath $TEST_SCRIPT -VarName "datasetName" -NewValue "UK_DALE" -IsString $true
Set-PythonVar -FilePath $TEST_SCRIPT -VarName "TrainPercent" -NewValue $TrainPercent -IsString $true

# The test script sometimes hardcodes applianceName or doesn't use the arg properly in some legacy versions,
# but our previous edits ensured it uses args.appliance_name. 
# We'll update the default variable just in case.
# Note: In your Colab script you replaced a specific line comment. 
# Here we'll just rely on the regex variable replacer which is robust.

# Run Test
Push-Location $NILM_MAIN_DIR
try {
    $TestCmdArgs = @("EasyS2S_test.py", "--appliance_name", $Appliance)
    Write-Host "Executing: python $($TestCmdArgs -join ' ')" -ForegroundColor Gray
    python @TestCmdArgs
    if ($LASTEXITCODE -ne 0) { throw "Testing failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "`n=== Experiment Completed Successfully! ===" -ForegroundColor Green
