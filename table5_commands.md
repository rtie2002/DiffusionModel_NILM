# NILM Table 5 Reproduction Commands

## 1. Setup (Google Colab)
Before running experiments, ensure you are in the correct directory.
```python
import os
os.environ['PATH'] = '/usr/bin:' + os.environ['PATH']
%cd /content/drive/MyDrive/DiffusionModel_NILM/NILM-main
```

---

## 2. File Naming Rules (CRITICAL)

The `EasyS2S` scripts have hardcoded logic for filenames. You **MUST** rename your prepared data files to match these exact names before training.

**All files must be placed in:**  
`dataset_preprocess/created_data/UK_DALE/{appliance_name}/`  
*(Example: `dataset_preprocess/created_data/UK_DALE/kettle/`)*

### Case A: Baseline Experiment (Origin 200k)
| Your Data Source | **Required Filename** |
|------------------|-----------------------|
| `kettle_training_.csv` (First 200k rows) | **`kettle_20training_.csv`** |

### Case B: Robust/Mixed Experiments (Rows 2-5)
For **ALL** mixed data experiments (100k+100k, 100k+200k, etc.), the script expects **ONE** constant filename because `TrainPercent` is hardcoded to `'20'`.

| Your Data Source (Example) | **Required Filename** |
|----------------------------|-----------------------|
| `kettle_training_100k+100k.csv` | **`UK_DALECombinedkettle_file20.csv`** |
| `kettle_training_100k+200k.csv` | **`UK_DALECombinedkettle_file20.csv`** |
| `kettle_training_200k+200k.csv` | **`UK_DALECombinedkettle_file20.csv`** |

> **IMPORTANT:** When switching experiments (e.g., from 100k+100k to 100k+200k), simply overwrite `UK_DALECombinedkettle_file20.csv` with your new data file.

---

## 3. Baseline Experiment (Origin 200k)
**Goal:** Reproduce the first row of Table 5 (Standard training on 200k real data).

### Training (Baseline)
```python
# 1. Set Baseline Mode (originModel = True)
!sed -i "s/originModel=False/originModel=True/" EasyS2S_train.py

# 2. Run Training
!MPLBACKEND=Agg python3.10 EasyS2S_train.py --appliance_name kettle --n_epoch 100 --batchsize 1024
```

### Testing (Baseline)
```python
# 1. Ensure Baseline Mode (originModel = True)
!sed -i "s/originModel=False/originModel=True/" EasyS2S_test.py

# 2. Run Test
!python3.10 EasyS2S_test.py --appliance_name kettle
```

---

## 4. Robust/Mixed Experiment (e.g., 100k + 100k)
**Goal:** Reproduce Rows 2-5 of Table 5 (Training on mixed real + synthetic data with Robust Loss).

**Prerequisite:** Explicitly rename your mixed file (e.g., `kettle_training_100k+100k.csv`) to **`UK_DALECombinedkettle_file20.csv`**.

### Training (Robust)
```python
# 1. Set Robust Mode (originModel = False)
!sed -i "s/originModel=True/originModel=False/" EasyS2S_train.py

# 2. Run Training
!MPLBACKEND=Agg python3.10 EasyS2S_train.py --appliance_name kettle --n_epoch 100 --batchsize 1024
```

### Testing (Robust)
```python
# 1. Set Robust Mode (originModel = False)
!sed -i "s/originModel=True/originModel=False/" EasyS2S_test.py

# 2. Fix Filename Capitalization Bug (Required for Robust model loading)
!sed -i "s/combine{TrainPercent}/Combine{TrainPercent}/" EasyS2S_test.py

# 3. Run Test
!python3.10 EasyS2S_test.py --appliance_name kettle
```
