# Mix Training Data (Multivariate) Guide
# 多变量混合训练数据指南

## Overview 概述

`mix_training_data_multivariate.py` 用于将真实数据和生成的合成数据混合，创建用于训练 NILMFormer 的混合数据集。它会保留所有的10个列（聚合功率，电器功率，以及8个时间特征）。

**Input**:
- Real Data (CSV): `created_data/UK_DALE/{appliance}_training_.csv` (Multivariate format)
- Synthetic Data (NPY): `synthetic_data_multivariate/ddpm_fake_{appliance}_multivariate.npy`

**Output**:
- Mixed CSV: `created_data/UK_DALE/{appliance}_training_{suffix}.csv`

---

## Quick Start 快速开始

```bash
# Basic Usage (200k Real + 200k Synthetic)
python mix_training_data_multivariate.py --appliance fridge --real_rows 200000 --synthetic_rows 200000
```

---

## Command Examples 命令示例

### 1. Default Mix (200k + 200k)
Creates `fridge_training_200k+200k.csv`

```bash
python mix_training_data_multivariate.py --appliance fridge
```

### 2. Custom Mix Ratio
Example: 50k Real + 500k Synthetic
Creates `washingmachine_training_50k+500k.csv`

```bash
python mix_training_data_multivariate.py --appliance washingmachine --real_rows 50000 --synthetic_rows 500000
```

### 3. No Shuffle (Sequence Order)
By default, the script shuffles the windows. Use `--no-shuffle` to keep them in order (Real first, then Synthetic).

```bash
python mix_training_data_multivariate.py --appliance microwave --no-shuffle
```

### 4. Custom Output Suffix
Override the default `Nk+Mk` suffix.
Creates `kettle_training_experiment1.csv`

```bash
python mix_training_data_multivariate.py --appliance kettle --suffix experiment1
```

---

## Arguments 参数说明

| Argument | Description | Default |
|----------|-------------|---------|
| `--appliance` | Target appliance name (fridge, kettle, microwave, dishwasher, washingmachine) | Interactive Mode |
| `--real_rows` | Number of rows to take from real data | 200000 |
| `--synthetic_rows` | Number of rows to take from synthetic data | 200000 |
| `--suffix` | Suffix for the output filename (e.g., `200k+200k`) | Auto-generated |
| `--no-shuffle` | If set, disables window shuffling (keeps real then synthetic) | False (Shuffling ON) |
| `--real_path` | Custom path to real data CSV | Default Path |

---

## Workflow Workflow

1.  **Prepare Synthetic Data**: Ensure you have generated synthetic data and converted it to NPY in `synthetic_data_multivariate/`.
2.  **Run Mixing Script**: Use one of the commands above.
3.  **Update Config**: To train with this new data, update your YAML config (e.g., `Config/fridge.yaml`) to point to the new file:

```yaml
dataloader:
  train_dataset:
    params:
      proportion: 1.0
      # Update filename here
      data_root: ./created_data/UK_DALE/fridge_training_200k+200k.csv 
```
