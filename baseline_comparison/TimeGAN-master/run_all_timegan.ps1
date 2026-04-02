$appliances = @(
    "washingmachine_multivariate",
    "kettle_multivariate",
    "fridge_multivariate",
    "dishwasher_multivariate",
    "microwave_multivariate"
)

foreach ($app in $appliances) {
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "🚀 STARTING C-TIMEGAN+ FOR: $app" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    
    # ⚡ Step 1: Training (Conv-TimeGAN: window=512, hidden=24, iter=50k, lr=0.001)
    Write-Host "🏗️ Phase 1: Training..."
    python train.py --data_name $app --seq_len 512 --hidden_dim 24 --batch_size 128 --iteration 2000 --lr 0.001
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Training Finished: $app" -ForegroundColor Green
        
        # ⚡ Step 2: Sampling (Matching 512 length)
        Write-Host "🧪 Phase 2: Generating Synthetic Data (OCSVM Filtering)..."
        python sample_only.py --data_name $app --seq_len 512
        
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
