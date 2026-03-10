#!/bin/bash

# 定义电器列表
appliances=(
    "kettle_training_"
    "fridge_training_"
    "dishwasher_training_"
    "microwave_training_"
    "washingmachine_training_"
)

for app in "${appliances[@]}"
do
    echo -e "\n\033[0;36m============================================================\033[0m"
    echo -e "\033[0;36m🚀 STARTING TIMEGAN TRAINING FOR: $app\033[0m"
    echo -e "\033[0;36m============================================================\033[0m"
    
    # 运行训练
    python train.py --data_name "$app" --seq_len 100 --batch_size 512 --iteration 20000
    
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32m✅ FINISHED: $app\033[0m"
    else
        echo -e "\033[0;31m❌ FAILED: $app\033[0m"
    fi
done

echo -e "\n\033[0;33mAll TimeGAN experiments completed!\033[0m"
