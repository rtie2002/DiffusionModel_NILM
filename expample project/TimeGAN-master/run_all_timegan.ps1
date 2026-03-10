
$appliances = @(
    "kettle_training_",
    "fridge_training_",
    "dishwasher_training_",
    "microwave_training_",
    "washingmachine_training_"
)

foreach ($app in $appliances) {
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "🚀 STARTING TIMEGAN TRAINING FOR: $app" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    
    # Run the training script
    # We use --iteration 20000 and --batch_size 512 as discussed for 4090 optimization
    python train.py --data_name $app --seq_len 100 --batch_size 512 --iteration 20000
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ FINISHED: $app" -ForegroundColor Green
    }
    else {
        Write-Host "❌ FAILED: $app" -ForegroundColor Red
    }
}

Write-Host "`nAll TimeGAN experiments completed!" -ForegroundColor Yellow
