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

**说明**: 指定要处理的电器类型

**选项**: 
- `fridge` - 冰箱
- `microwave` - 微波炉
- `kettle` - 电热水壶
- `dishwasher` - 洗碗机
- `washingmachine` - 洗衣机

**示例**:
```bash
# 处理冰箱数据
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name fridge

# 处理微波炉数据
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name microwave
```

**效果**: 
- 自动查找对应的输入文件 `created_data/UK_DALE/{appliance}_training_.csv`
- 输出文件命名为 `Data/datasets/{appliance}_multivariate.csv`

---

### `--input_file` (可选)

**说明**: 自定义输入文件路径，覆盖默认路径

**默认**: `created_data/UK_DALE/{appliance}_training_.csv`

**使用场景**:
1. 输入文件在非标准位置
2. 使用自定义命名的文件
3. 处理测试/验证数据集

**示例**:
```bash
# 使用自定义路径
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --input_file "custom_data/my_fridge_data.csv"

# 处理测试集
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --input_file "created_data/UK_DALE/fridge_test_.csv"

# 使用绝对路径
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --input_file "C:/Data/fridge_training_.csv"
```

**注意**: 
- 文件必须是6列格式 (aggregate, appliance, minute, hour, day, month)
- 路径包含空格时需要使用引号

---

### `--output_dir` (可选)

**说明**: 指定输出文件保存目录

**默认**: `Data/datasets`

**使用场景**:
1. 组织不同实验的输出
2. 分离训练/测试数据
3. 保存到特定项目目录

**示例**:
```bash
# 保存到自定义目录
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --output_dir "experiments/exp001/"

# 保存到日期目录
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --output_dir "output/2024-12-29/"

# 分离不同配置的输出
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --output_dir "output/window_150/" \
  --window 150
```

**效果**: 
- 输出文件: `{output_dir}/{appliance}_multivariate.csv`
- 目录不存在时会自动创建

---

### `--window` (可选)

**说明**: Algorithm 1 的窗口长度，控制在启动事件前后选择多少样本

**默认**: `100`

**推荐值**:
- **小窗口 (50-100)**: 只保留核心事件，数据更纯净
- **中窗口 (100-200)**: 平衡，包含启动和关闭过程
- **大窗口 (200-500)**: 保留更多上下文，数据量更大

**使用场景**:

**场景 1: 快速启动电器 (微波炉, 电热水壶)**
```bash
# 使用小窗口，因为事件短暂
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name kettle \
  --window 50
```

**场景 2: 长时间运行电器 (洗衣机, 洗碗机)**
```bash
# 使用大窗口，捕获完整运行周期
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name washingmachine \
  --window 300
```

**场景 3: 持续运行电器 (冰箱)**
```bash
# 使用中等窗口
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --window 150
```

**效果对比**:
```
Window = 50:  保留 60% 数据，更纯净
Window = 100: 保留 75% 数据，平衡
Window = 200: 保留 90% 数据，更完整
```

---

### `--clip_max` (可选)

**说明**: 裁剪功率最大值（Watts），去除异常高值

**默认**: `None` (不裁剪)

**使用场景**:
1. 数据包含测量错误
2. 存在异常尖峰
3. 需要限制功率范围

**示例**:

**场景 1: 去除微波炉异常值**
```bash
# 微波炉正常功率 < 2000W
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name microwave \
  --clip_max 2000
```

**场景 2: 限制冰箱功率**
```bash
# 冰箱正常功率 < 500W
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --clip_max 500
```

**场景 3: 洗衣机功率限制**
```bash
# 洗衣机正常功率 < 3000W
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name washingmachine \
  --clip_max 3000
```

**效果**:
```
Before clipping: [100, 200, 5000, 300, 150]
After clip_max=500: [100, 200, 500, 300, 150]
```

**推荐值**:
- Fridge: 500-800W
- Microwave: 1500-2000W
- Kettle: 2000-2500W
- Dishwasher: 2500-3000W
- Washing Machine: 2500-3500W

---

### `--remove_spikes` (可选)

**说明**: 是否移除孤立尖峰（短暂的异常高值）

**默认**: `True` (移除)

**使用场景**:

**启用 (默认)**:
```bash
# 移除尖峰 (推荐)
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --remove_spikes
```

**禁用**:
```bash
# 保留所有数据，包括尖峰
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --no_remove_spikes
```

**何时禁用**:
1. 尖峰是真实信号（如微波炉启动）
2. 需要保留所有原始数据
3. 手动处理异常值

**效果示例**:
```
原始数据: [100, 100, 5000, 100, 100]  ← 孤立尖峰
remove_spikes=True:  [100, 100, 100, 100, 100]
remove_spikes=False: [100, 100, 5000, 100, 100]
```

---

### `--spike_window` (可选)

**说明**: 尖峰检测的窗口大小（前后各多少个样本）

**默认**: `5`

**工作原理**: 检查当前点与前后 `spike_window` 个样本的差异

**使用场景**:

**小窗口 (3-5)**: 检测非常短暂的尖峰
```bash
# 检测单点尖峰
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --spike_window 3
```

**中窗口 (5-10)**: 平衡检测
```bash
# 标准检测
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --spike_window 7
```

**大窗口 (10-20)**: 检测较长的异常段
```bash
# 检测持续异常
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --spike_window 15
```

**效果对比**:
```
数据: [100, 100, 100, 500, 500, 100, 100, 100]

spike_window=3:  检测到尖峰 (500与100差异大)
spike_window=10: 可能不检测 (窗口内有多个500)
```

---

### `--spike_threshold` (可选)

**说明**: 尖峰阈值倍数，判断多大的差异算尖峰

**默认**: `3.0`

**工作原理**: `spike = value > mean + threshold * std`

**使用场景**:

**严格检测 (2.0-2.5)**: 移除更多尖峰
```bash
# 激进去除尖峰
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --spike_threshold 2.0
```

**标准检测 (3.0)**: 平衡
```bash
# 标准检测
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --spike_threshold 3.0
```

**宽松检测 (4.0-5.0)**: 只移除极端尖峰
```bash
# 保守去除
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --spike_threshold 4.5
```

**效果示例**:
```
数据: mean=100, std=50
值: 250

threshold=2.0: 250 > 100+2*50=200 → 是尖峰 ✓
threshold=3.0: 250 > 100+3*50=250 → 不是尖峰 ✗
threshold=4.0: 250 > 100+4*50=300 → 不是尖峰 ✗
```

**推荐值**:
- 噪声数据: 2.0-2.5
- 正常数据: 3.0
- 保留更多: 4.0-5.0

---

### `--background_threshold` (可选)

**说明**: 背景功率阈值（Watts），低于此值视为背景噪声

**默认**: `50`

**使用场景**:

**低功率电器 (冰箱)**: 降低阈值
```bash
# 冰箱待机功率很低
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --background_threshold 30
```

**中功率电器 (微波炉)**: 标准阈值
```bash
# 微波炉待机功率中等
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name microwave \
  --background_threshold 50
```

**高功率电器 (洗衣机)**: 提高阈值
```bash
# 洗衣机启动功率高
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name washingmachine \
  --background_threshold 100
```

**效果**:
```
功率序列: [10, 20, 30, 200, 300, 40, 20]

threshold=50:  保留 [200, 300] (启动事件)
threshold=100: 保留 [200, 300] (启动事件)
threshold=30:  保留 [200, 300, 40] (包含更多)
```

**推荐值**:
- Fridge: 20-40W
- Microwave: 50-80W
- Kettle: 50-100W
- Dishwasher: 80-120W
- Washing Machine: 100-150W

---

## 参数组合示例

### 组合 1: 保守过滤 (保留更多数据)
```bash
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --window 200 \
  --spike_threshold 4.0 \
  --background_threshold 30
```
**效果**: 保留 ~90% 数据

### 组合 2: 激进过滤 (只保留核心事件)
```bash
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --window 50 \
  --spike_threshold 2.0 \
  --background_threshold 80 \
  --clip_max 500
```
**效果**: 保留 ~60% 数据，更纯净

### 组合 3: 平衡配置 (推荐)
```bash
python Data_filtering/algorithm1_v2_multivariate.py \
  --appliance_name fridge \
  --window 100 \
  --spike_threshold 3.0 \
  --background_threshold 50
```
**效果**: 保留 ~75% 数据，平衡质量和数量

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
