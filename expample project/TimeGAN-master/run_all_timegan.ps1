$appliances = @(
    "kettle_training_",
    "fridge_training_",
    "dishwasher_training_",
    "microwave_training_",
    "washingmachine_training_"
)

foreach ($app in $appliances) {
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "🚀 STARTING C-TIMEGAN+ FOR: $app" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    
    # ⚡ Step 1: Training (Aligned with 2024 Paper)
    Write-Host "🏗️ Phase 1: Training..."
    python train.py --data_name $app --seq_len 60 --batch_size 512 --iteration 50000
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Training Finished: $app" -ForegroundColor Green
        
        # ⚡ Step 2: Sampling (with OCSVM Filtering)
        Write-Host "🧪 Phase 2: Generating Synthetic Data (OCSVM Filtering)..."
        python sample_only.py --data_name $app --seq_len 60
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "🎉 Successfully generated data for: $app" -ForegroundColor Green
        }
        else {
            Write-Host "⚠️ Sampling failed for: $app" -ForegroundColor Red
        }
    }
    else {
        Write-Host "❌ Training failed for: $app" -ForegroundColor Red
    }
}

Write-Host "`n🏆 All C-TimeGAN+ experiments completed!" -ForegroundColor Yellow
