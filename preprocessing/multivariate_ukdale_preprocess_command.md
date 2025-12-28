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

## 📊 Data Processing Pipeline

### Overview

```mermaid
graph TD
    A[Raw UK-DALE .dat files] --> B[Load & Resample 60s]
    B --> C[Align Timestamps]
    C --> D[Extract Temporal Features]
    D --> E[Z-score Normalization]
    E --> F[Split Train/Val/Test]
    F --> G[6-Column CSV Output]
    G --> H[Algorithm 1 Filtering]
    H --> I[5-Column CSV Output]
```

### Step-by-Step Process

#### Step 1: Load Raw Data

**Code Location**: Lines 106-115, 146-151

```python
# Load aggregate (mains) data
mains_df = load_dataframe(args.data_dir, house_id, channel=1)

# Load appliance data  
app_df = load_dataframe(args.data_dir, house_id, channel=appliance_channel)
```

**Input**: `.dat` files with Unix timestamps and power values
**Output**: Pandas DataFrames

#### Step 2: Timestamp Alignment & Resampling

**Code Location**: Lines 153-193

```python
# Convert to datetime
mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
app_df['time'] = pd.to_datetime(app_df['time'], unit='s')

# Align timestamps and resample to 60 seconds
df_align = mains_df.join(app_df, how='outer').\
    resample('60S').mean().bfill(limit=1)
```

**Why 60 seconds?**
- Reduces data size while preserving patterns
- Standard sampling rate for NILM research
- Balances temporal resolution and computational efficiency

#### Step 3: Extract Temporal Features

**Code Location**: Lines 197-207

```python
# Extract temporal features from timestamp
df_align['minute'] = df_align['time'].dt.minute  # 0-59
df_align['hour'] = df_align['time'].dt.hour      # 0-23
df_align['day'] = df_align['time'].dt.day        # 1-31
df_align['month'] = df_align['time'].dt.month    # 1-12

# Select columns (remove timestamp)
df_align = df_align[['aggregate', appliance_name, 'minute', 'hour', 'day', 'month']]
```

**Purpose**: Provide temporal context for multivariate diffusion models

#### Step 4: Z-score Normalization

**Code Location**: Lines 261-266

```python
# Get normalization parameters
mean = params_appliance[appliance_name]['mean']
std = params_appliance[appliance_name]['std']

# Apply Z-score normalization
df_align['aggregate'] = (df_align['aggregate'] - AGG_MEAN) / AGG_STD
df_align[appliance_name] = (df_align[appliance_name] - mean) / std
```

**Formula**:
```
normalized_value = (original_value - mean) / std
```

**Denormalization** (for visualization/evaluation):
```
original_value = normalized_value * std + mean
```

#### Step 5: Train/Val/Test Split

**Code Location**: Lines 277-291

```python
# Split ratios
validation_percent = 20  # 20%
testing_percent = 20     # 20%
training_percent = 60    # 60%

# Split data
test = train.tail(test_len)
val = train.tail(val_len) 
# Remaining data is training set
```

### Output Format

#### 6-Column CSV (from multivariate_ukdale_preprocess.py)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| aggregate | float | Z-score | Normalized total power |
| appliance | float | Z-score | Normalized appliance power |
| minute | int | 0-59 | Minute of hour |
| hour | int | 0-23 | Hour of day |
| day | int | 1-31 | Day of month |
| month | int | 1-12 | Month of year |

**Example**:
```csv
-0.234,0.567,15,14,28,6
-0.189,0.432,16,14,28,6
```

#### 5-Column CSV (after algorithm1_v2_multivariate.py)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| appliance | float | [0,1] | MinMax normalized appliance power |
| minute | int | 0-59 | Minute of hour |
| hour | int | 0-23 | Hour of day |
| day | int | 1-31 | Day of month |
| month | int | 1-12 | Month of year |

**Note**: Aggregate column is removed, appliance power is re-normalized using MinMax [0,1]

## 🔧 Normalization Parameters

### Current Parameters (from code)

```python
params_appliance = {
    'kettle': {
        'mean': 700,
        'std': 1000,
        'houses': [2],
        'channels': [8],
    },
    'microwave': {
        'mean': 500,
        'std': 800,
        'houses': [2],
        'channels': [15],
    },
    'fridge': {
        'mean': 200,
        'std': 400,
        'houses': [2],
        'channels': [14],
    },
    'dishwasher': {
        'mean': 700,
        'std': 1000,
        'houses': [2],
        'channels': [13],
    },
    'washingmachine': {
        'mean': 400,
        'std': 700,
        'houses': [2],
        'channels': [12],
    }
}

AGG_MEAN = 522  # Aggregate mean
AGG_STD = 814   # Aggregate std
```

### Recommended Parameters (calculated from actual UK-DALE data)

Based on analysis of Building 1 and 2 combined data:

```python
# Option 1: Actual calculated values
params_appliance = {
    'kettle': {'mean': 13, 'std': 168},
    'microwave': {'mean': 25, 'std': 177},
    'fridge': {'mean': 47, 'std': 50},
    'dishwasher': {'mean': 49, 'std': 305},
    'washingmachine': {'mean': 38, 'std': 232},
}
AGG_MEAN = 409
AGG_STD = 502

# Option 2: Transformer project values (well-tested)
params_appliance = {
    'kettle': {'mean': 100, 'std': 500},
    'microwave': {'mean': 60, 'std': 300},
    'fridge': {'mean': 50, 'std': 50},      # ← Almost perfect match!
    'dishwasher': {'mean': 700, 'std': 1000},
    'washingmachine': {'mean': 400, 'std': 700},
}
AGG_MEAN = 400
AGG_STD = 500
```

### Impact of Different Parameters

**Using smaller mean/std** (actual values):
- ✅ More accurate normalization
- ✅ Data matches actual distribution
- ⚠️ Larger normalized value range
- ⚠️ May need model retraining

**Using larger mean/std** (original values):
- ✅ Compatible with existing models
- ✅ Smaller normalized value range
- ⚠️ Less accurate normalization
- ⚠️ May not match actual data distribution

### How to Calculate Your Own Parameters

```bash
# Run the statistics calculator
python preprocessing/calculate_ukdale_stats.py
```

This will output recommended mean/std values based on your actual UK-DALE data.

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
- **归一化参数**: 建议使用实际计算的统计值以获得最佳性能
- **时间特征**: 保持原始整数值，不进行归一化


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
