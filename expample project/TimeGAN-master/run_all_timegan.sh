# 定义电器列表
appliances=(
    "washingmachine_multivariate"
    "kettle_multivariate"
    "fridge_multivariate"
    "dishwasher_multivariate"
    "microwave_multivariate"
)

# 确保输出目录存在
mkdir -p ./output/TimeGAN

for app in "${appliances[@]}"
do
    echo -e "\n\033[0;36m============================================================\033[0m"
    echo -e "\033[0;36m🚀 STARTING C-TIMEGAN+ FOR: $app\033[0m"
    echo -e "\033[0;36m============================================================\033[0m"
    
    # ⚡ 步骤 1: 训练 (Using 60 window and 10,000 iterations for efficiency)
    echo "🏗️ Phase 1: Training..."
    python train.py --data_name "$app" --seq_len 60 --batch_size 128 --iteration 10000
    
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32m✅ Training Finished: $app\033[0m"
        
        # ⚡ 步骤 2: 采样 (采用 60 窗口大小)
        echo "🧪 Phase 2: Generating Synthetic Data (with OCSVM Filtering)..."
        python sample_only.py --data_name "$app" --seq_len 60
        
        if [ $? -eq 0 ]; then
            echo -e "\033[0;32m🎉 Successfully generated data for: $app\033[0m"
        else
            echo -e "\033[0;31m⚠️ Sampling failed for: $app\033[0m"
        fi
    else
        echo -e "\033[0;31m❌ Training failed for: $app\033[0m"
    fi
done

echo -e "\n\033[0;33m🏆 All C-TimeGAN+ experiments completed!\033[0m"
