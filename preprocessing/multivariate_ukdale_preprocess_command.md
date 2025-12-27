# Multivariate Preprocessing Commands
# 多变量预处理命令

## Quick Start 快速开始

```bash
# Fridge (冰箱)
python preprocessing/multivariate_ukdale_preprocess.py --appliance_name fridge --data_dir "NILM-main/dataset_preprocess/UK_DALE/"

# Microwave (微波炉)
python preprocessing/multivariate_ukdale_preprocess.py --appliance_name microwave --data_dir "NILM-main/dataset_preprocess/UK_DALE/"

# Kettle (水壶)
python preprocessing/multivariate_ukdale_preprocess.py --appliance_name kettle --data_dir "NILM-main/dataset_preprocess/UK_DALE/"

# Dishwasher (洗碗机)
python preprocessing/multivariate_ukdale_preprocess.py --appliance_name dishwasher --data_dir "NILM-main/dataset_preprocess/UK_DALE/"

# Washing Machine (洗衣机)
python preprocessing/multivariate_ukdale_preprocess.py --appliance_name washingmachine --data_dir "NILM-main/dataset_preprocess/UK_DALE/"
```

## Output 输出

生成的文件位于 `created_data/UK_DALE/`:

```
fridge_training_.csv      # 训练集 (6列: aggregate, appliance, minute, hour, day, month)
fridge_validation_.csv    # 验证集
fridge_test_.csv          # 测试集
```

## Apply Algorithm 1 应用Algorithm 1

过滤有效部分并保留时间特征:

```bash
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name fridge
```

输出: `Data/datasets/fridge_multivariate.csv` (5列: appliance, minute, hour, day, month)

## Verify Data 验证数据

```bash
# 检查CSV格式
python preprocessing/check_csv_format.py

# 分布对比
python "Data Quality Checking/distribution_comparison.py"

# 时间数据查看器
python "Data Quality Checking/temporal_data_viewer.py"
```

## Complete Workflow 完整流程

```bash
# Step 1: 预处理 (生成6列CSV)
python preprocessing/multivariate_ukdale_preprocess.py \
  --appliance_name fridge \
  --data_dir "NILM-main/dataset_preprocess/UK_DALE/"

# Step 2: 应用Algorithm 1 (过滤并生成5列CSV)
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge

# Step 3: 验证数据分布
python "Data Quality Checking/distribution_comparison.py"

# Step 4: 训练多变量扩散模型
# (使用 Data/datasets/fridge_multivariate.csv)
```

## Parameters 参数说明

### multivariate_ukdale_preprocess.py

- `--appliance_name`: 电器名称 (必需)
- `--data_dir`: UK-DALE数据目录 (推荐: "NILM-main/dataset_preprocess/UK_DALE/")
- `--save_path`: 输出目录 (默认: "created_data/UK_DALE/")
- `--aggregate_mean`: 总功率均值 (默认: 522W)
- `--aggregate_std`: 总功率标准差 (默认: 814W)

### algorithm1_v2_multivariate.py

- `--appliance_name`: 电器名称 (必需)
- `--input_file`: 输入CSV (默认: created_data/UK_DALE/{appliance}_training_.csv)
- `--output_dir`: 输出目录 (默认: "Data/datasets")
- `--window`: Algorithm 1窗口长度 (默认: 100)
- `--clip_max`: 可选，裁剪最大值 (Watts)
- `--remove_spikes`: 移除孤立尖峰 (默认: True)

## Troubleshooting 故障排除

### 错误: FileNotFoundError

```bash
# 确保使用正确的数据路径
python preprocessing/multivariate_ukdale_preprocess.py \
  --appliance_name fridge \
  --data_dir "NILM-main/dataset_preprocess/UK_DALE/"
```

### 错误: PermissionError

```
PermissionError: [Errno 13] Permission denied: 'created_data/UK_DALE/fridge_training_.csv'
```

**原因**: CSV文件正在被其他程序打开（如Excel、VSCode等）

**解决方案**:
1. 关闭所有打开的CSV文件
2. 关闭VSCode中打开的CSV文件
3. 重新运行脚本

### 错误: 找不到Python

```bash
# 使用完整路径
& "c:/Users/Raymond Tie/Desktop/DiffusionModel_NILM/.venv/Scripts/python.exe" \
  preprocessing/multivariate_ukdale_preprocess.py \
  --appliance_name fridge \
  --data_dir "NILM-main/dataset_preprocess/UK_DALE/"
```

## Notes 注意事项

- **图表已关闭**: `debug = False` 以加快处理速度
- **关闭CSV文件**: 运行前确保所有CSV文件已关闭
- **数据路径**: 使用相对路径 `NILM-main/dataset_preprocess/UK_DALE/`
