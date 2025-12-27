# UK-DALE Preprocessing Command Guide
# UK-DALE 预处理命令指南

## Overview 概述

`ukdale_processing.py` 用于处理UK-DALE数据集，生成用于NILM训练的CSV文件。

**修改**: `sample_seconds = 60` (从6秒改为60秒采样)

---

## Quick Start 快速开始

### ✅ 推荐方法: 在脚本所在目录运行

```bash
# Step 1: 进入脚本目录
cd C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM\NILM-main\dataset_preprocess

# Step 2: 运行脚本（使用当前目录的脚本）
python ukdale_processing.py --appliance_name washingmachine

# 或使用虚拟环境Python
& "C:/Users/Raymond Tie/Desktop/DiffusionModel_NILM/.venv/Scripts/python.exe" ukdale_processing.py --appliance_name washingmachine
```

**为什么这样做**:
- ✅ 脚本默认在当前目录查找 `UK_DALE/` 数据
- ✅ 输出文件保存到 `created_data/UK_DALE/`
- ✅ 路径简单，不易出错

### ❌ 常见错误

```bash
# 错误1: 在脚本目录运行时使用完整路径
cd NILM-main/dataset_preprocess
python NILM-main/dataset_preprocess/ukdale_processing.py  # ❌ 错误！会找不到文件

# 错误2: 在项目根目录运行时不指定数据路径
cd DiffusionModel_NILM
python NILM-main/dataset_preprocess/ukdale_processing.py  # ❌ 错误！找不到UK_DALE/
```

### 方法2: 从项目根目录运行（需要指定数据路径）

```bash
# 在项目根目录 (DiffusionModel_NILM/)
cd C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM

# 必须指定完整数据路径
python NILM-main/dataset_preprocess/ukdale_processing.py \
  --appliance_name washingmachine \
  --data_dir "NILM-main/dataset_preprocess/UK_DALE/"
```

---

## All Appliances 所有电器

```bash
# Fridge (冰箱)
python ukdale_processing.py --appliance_name fridge

# Microwave (微波炉)
python ukdale_processing.py --appliance_name microwave

# Kettle (水壶)
python ukdale_processing.py --appliance_name kettle

# Dishwasher (洗碗机)
python ukdale_processing.py --appliance_name dishwasher

# Washing Machine (洗衣机)
python ukdale_processing.py --appliance_name washingmachine
```

---

## Parameters 参数说明

### `--appliance_name` (必需)
- **选项**: `fridge`, `microwave`, `kettle`, `dishwasher`, `washingmachine`
- **示例**: `--appliance_name fridge`

### `--data_dir` (可选)
- **默认**: `UK_DALE/`
- **推荐**: `UK_DALE/` (相对于脚本位置)
- **示例**: `--data_dir "UK_DALE/"`

### `--save_path` (可选)
- **默认**: `created_data/UK_DALE/`
- **示例**: `--save_path "output/"`

### `--aggregate_mean` (可选)
- **默认**: `522` (Watts)
- **说明**: 总功率均值

### `--aggregate_std` (可选)
- **默认**: `814` (Watts)
- **说明**: 总功率标准差

---

## Output Files 输出文件

运行成功后，在 `NILM-main/dataset_preprocess/created_data/UK_DALE/` 生成:

```
fridge_training_.csv      # 训练集 (2列: aggregate, power)
fridge_validation_.csv    # 验证集
fridge_test_.csv          # 测试集
```

**CSV格式** (带表头):
```
aggregate,power
-0.123,0.456
-0.145,0.478
...
```

---

## Data Location 数据位置

确保UK-DALE数据在正确位置:

```
NILM-main/dataset_preprocess/UK_DALE/
├── house_1/
├── house_2/
│   ├── channel_1.dat    # Mains
│   ├── channel_8.dat    # Kettle
│   ├── channel_12.dat   # Washing machine
│   ├── channel_13.dat   # Dishwasher
│   ├── channel_14.dat   # Fridge
│   └── channel_15.dat   # Microwave
└── ...
```

---

## Complete Workflow 完整工作流程

### Step 1: 预处理数据 (ukdale_processing.py)

```bash
cd NILM-main/dataset_preprocess

python ukdale_processing.py --appliance_name fridge
```

**输出**: `created_data/UK_DALE/fridge_*.csv` (2列: aggregate, power)

### Step 2: (可选) 应用Algorithm 1

```bash
cd ../..  # 返回项目根目录

python Data_filtering/algorithm1_v2.py \
  --appliance_name fridge \
  --input_file "NILM-main/dataset_preprocess/created_data/UK_DALE/fridge_training_.csv"
```

**输出**: `Data/datasets/fridge.csv` (1列: power, MinMax归一化)

---

## Sampling Rate 采样率

**当前设置**: `sample_seconds = 60` (60秒采样)

**修改采样率**:
编辑 `ukdale_processing.py` 第131行:
```python
sample_seconds = 60  # 改为其他值，如 6, 10, 30, 60
```

**采样率对比**:
- `6秒`: 更多数据点，训练时间长
- `60秒`: 较少数据点，训练更快

---

## Troubleshooting 故障排除

### 错误: FileNotFoundError

```
FileNotFoundError: [Errno 2] No such file or directory: 'UK_DALE/house_2/channel_14.dat'
```

**解决方案**:
1. 检查数据目录是否存在
2. 确保在正确的目录运行脚本:
   ```bash
   cd NILM-main/dataset_preprocess
   python ukdale_processing.py --appliance_name fridge
   ```

### 错误: PermissionError

```
PermissionError: [Errno 13] Permission denied: 'created_data/UK_DALE/fridge_training_.csv'
```

**解决方案**:
1. 关闭所有打开的CSV文件
2. 关闭Excel、VSCode等程序中的CSV文件
3. 重新运行脚本

### 错误: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'pandas'
```

**解决方案**:
```bash
pip install pandas numpy matplotlib
```

---

## Verify Output 验证输出

```bash
# 查看文件
head created_data/UK_DALE/fridge_training_.csv

# 统计行数
wc -l created_data/UK_DALE/fridge_training_.csv

# 使用Python查看
python -c "import pandas as pd; df = pd.read_csv('created_data/UK_DALE/fridge_training_.csv'); print(f'Shape: {df.shape}'); print(df.head())"
```

---

## Differences vs Multivariate 与多变量版本的区别

| 特性 | ukdale_processing.py | multivariate_ukdale_preprocess.py |
|------|---------------------|-----------------------------------|
| 输出列数 | 2列 (aggregate, power) | 6列 (aggregate, appliance, minute, hour, day, month) |
| 时间特征 | ❌ 无 | ✅ 有 (minute, hour, day, month) |
| 用途 | 单变量NILM | 多变量NILM |
| 位置 | NILM-main/dataset_preprocess/ | preprocessing/ |

---

## Notes 注意事项

- **采样率**: 已修改为60秒 (`sample_seconds = 60`)
- **数据格式**: 输出CSV包含表头 (aggregate, power)
- **Z-score归一化**: 使用appliance的mean和std进行归一化
- **工作目录**: 建议在 `NILM-main/dataset_preprocess/` 目录运行

---

## Example Output 示例输出

```
============================================================
Processing appliance: fridge
============================================================
Reading: UK_DALE/house_2/channel_14.dat
...
✓ Loaded data successfully
...
Size of total training set is 0.1037 M rows.
Size of total validation set is 0.0346 M rows.
Size of total testing set is 0.0346 M rows.

Please find files in: created_data/UK_DALE/
Total elapsed time: 0.45 min.
```
