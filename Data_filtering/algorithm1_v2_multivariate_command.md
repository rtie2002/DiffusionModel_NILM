# Algorithm 1 Multivariate - Command Guide
# Algorithm 1 多变量版本 - 命令指南

## Overview 概述

`algorithm1_v2_multivariate.py` 用于处理多变量CSV数据，应用Algorithm 1过滤有效部分，同时保留时间特征。

**输入**: 6列 (aggregate, appliance, minute, hour, day, month)
**输出**: 5列 (appliance, minute, hour, day, month)

---

## Quick Start 快速开始

```bash
# 基本用法
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name fridge

# 指定输入文件
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --input_file "created_data/UK_DALE/fridge_training_.csv"

# 使用虚拟环境
& "C:/Users/Raymond Tie/Desktop/DiffusionModel_NILM/.venv/Scripts/python.exe" \
  Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge
```

---

## All Appliances 所有电器

```bash
# Fridge
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name fridge

# Microwave
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name microwave

# Kettle
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name kettle

# Dishwasher
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name dishwasher

# Washing Machine
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name washingmachine
```

---

## Parameters 参数说明

### `--appliance_name` (必需)
- **选项**: `fridge`, `microwave`, `kettle`, `dishwasher`, `washingmachine`
- **示例**: `--appliance_name fridge`

### `--input_file` (可选)
- **默认**: `created_data/UK_DALE/{appliance}_training_.csv`
- **说明**: 输入的多变量CSV文件路径
- **示例**: `--input_file "created_data/UK_DALE/fridge_training_.csv"`

### `--output_dir` (可选)
- **默认**: `Data/datasets`
- **说明**: 输出目录
- **示例**: `--output_dir "output/filtered/"`

### `--window` (可选)
- **默认**: `100`
- **说明**: Algorithm 1的窗口长度
- **示例**: `--window 150`

### `--clip_max` (可选)
- **默认**: `None`
- **说明**: 裁剪最大值（Watts），用于去除异常值
- **示例**: `--clip_max 3000`

### `--remove_spikes` (可选)
- **默认**: `True`
- **说明**: 是否移除孤立尖峰
- **示例**: `--remove_spikes` 或 `--no_remove_spikes`

### `--spike_window` (可选)
- **默认**: `5`
- **说明**: 尖峰检测窗口大小
- **示例**: `--spike_window 7`

### `--spike_threshold` (可选)
- **默认**: `3.0`
- **说明**: 尖峰阈值倍数
- **示例**: `--spike_threshold 4.0`

### `--background_threshold` (可选)
- **默认**: `50`
- **说明**: 背景阈值（Watts）
- **示例**: `--background_threshold 100`

---

## Input Format 输入格式

**期望输入**: 多变量CSV（无表头，6列）

```csv
-0.308,-0.475,27,21,20,5
-0.308,-0.475,28,21,20,5
-0.334,-0.475,29,21,20,5
...
```

**列说明**:
1. `aggregate` - Z-score归一化的总功率
2. `appliance` - Z-score归一化的电器功率
3. `minute` - 分钟 (0-59)
4. `hour` - 小时 (0-23)
5. `day` - 日期 (1-31)
6. `month` - 月份 (1-12)

---

## Output Format 输出格式

**输出**: 5列CSV（无表头）

```csv
0.12,30,14,15,5
0.25,31,14,15,5
0.89,32,14,15,5
...
```

**列说明**:
1. `appliance` - MinMax归一化的电器功率 [0,1]
2. `minute` - 分钟（保持不变）
3. `hour` - 小时（保持不变）
4. `day` - 日期（保持不变）
5. `month` - 月份（保持不变）

**保存位置**: `Data/datasets/{appliance}_multivariate.csv`

---

## Complete Workflow 完整工作流程

### Step 1: 多变量预处理

```bash
# 生成6列CSV
python preprocessing/multivariate_ukdale_preprocess.py \
  --appliance_name fridge \
  --data_dir "NILM-main/dataset_preprocess/UK_DALE/"
```

**输出**: `created_data/UK_DALE/fridge_training_.csv` (6列)

### Step 2: 应用Algorithm 1

```bash
# 过滤有效部分，输出5列
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge
```

**输出**: `Data/datasets/fridge_multivariate.csv` (5列)

### Step 3: 验证数据

```bash
# 检查输出
python -c "import pandas as pd; df = pd.read_csv('Data/datasets/fridge_multivariate.csv', header=None); print(f'Shape: {df.shape}'); print(df.head())"
```

---

## Algorithm 1 工作原理

1. **读取多变量CSV** - 6列输入
2. **反归一化appliance power** - 转换为Watts
3. **应用阈值检测** - 找到启动事件 (power >= threshold)
4. **窗口选择** - 在启动事件前后选择±window范围
5. **MinMax归一化** - 将appliance power归一化到[0,1]
6. **保留时间特征** - minute, hour, day, month保持不变
7. **输出5列CSV** - 移除aggregate列

---

## Example 示例

### 基本用法

```bash
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name fridge
```

**输出**:
```
============================================================
Applying Algorithm 1 to multivariate TRAINING data: fridge
============================================================
Reading: created_data/UK_DALE/fridge_training_.csv
  CSV columns: ['aggregate', 'fridge', 'minute', 'hour', 'day', 'month']
  CSV shape: (103669, 6)
  Original data length: 103,669

Denormalizing appliance power (Z-score inverse):
  Mean: 200 W, Std: 400 W

Applying Algorithm 1:
  Threshold: 50 W
  Window length: 100

  Selected data length: 85,234
  Reduction: 18,435 samples removed
  Retention rate: 82.21%

============================================================
SUCCESS: Algorithm 1 processing complete!
============================================================
  Saved: Data/datasets/fridge_multivariate.csv
  Rows: 85,234
  Format: 5 columns (fridge, minute, hour, day, month)
  Appliance power: MinMax normalized [0,1]
  Temporal columns: Original values preserved
```

### 高级用法

```bash
# 裁剪异常值并调整窗口
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --window 150 \
  --clip_max 3000 \
  --spike_threshold 4.0
```

---

## Troubleshooting 故障排除

### 错误: FileNotFoundError

```
FileNotFoundError: Training file not found: created_data/UK_DALE/fridge_training_.csv
```

**解决方案**:
1. 确保先运行多变量预处理
2. 检查输入文件路径是否正确

```bash
# 先运行预处理
python preprocessing/multivariate_ukdale_preprocess.py --appliance_name fridge --data_dir "NILM-main/dataset_preprocess/UK_DALE/"

# 再运行Algorithm 1
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name fridge
```

### 错误: 缺少appliance_name参数

```
error: the following arguments are required: --appliance_name
```

**解决方案**: 必须指定电器名称

```bash
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name fridge
```

---

## Comparison 与单变量版本对比

| 特性 | algorithm1_v2.py | algorithm1_v2_multivariate.py |
|------|------------------|-------------------------------|
| 输入列数 | 2列 (aggregate, power) | 6列 (aggregate, appliance, minute, hour, day, month) |
| 输出列数 | 1列 (power) | 5列 (appliance, minute, hour, day, month) |
| 时间特征 | ❌ 无 | ✅ 保留 |
| 用途 | 单变量扩散模型 | 多变量扩散模型 |
| 归一化 | MinMax [0,1] | MinMax [0,1] |

---

## Notes 注意事项

- **Algorithm 1逻辑相同**: 只是保留了更多列
- **时间特征不变**: minute, hour, day, month保持原值
- **只归一化appliance**: aggregate列被移除
- **输出5列**: 适合多变量扩散模型训练

---

## Next Steps 下一步

1. **验证数据分布**:
   ```bash
   python "Data Quality Checking/distribution_comparison.py"
   ```

2. **训练多变量扩散模型**:
   使用 `Data/datasets/fridge_multivariate.csv` 进行训练

3. **生成合成数据**:
   训练完成后生成多变量合成数据
