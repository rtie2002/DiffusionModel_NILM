# Command Reference - DiffusionModel_NILM

## Live Loss Visualization with TensorBoard

### Diffusion Model Training (PyTorch + TensorBoard)

**Terminal 1: Start Training**
```bash
# Train with TensorBoard enabled
python main.py --train --config Config/microwave.yaml --name microwave_exp --tensorboard

# Other appliances
python main.py --train --config Config/kettle.yaml --name kettle_exp --tensorboard
python main.py --train --config Config/washingmachine.yaml --name washingmachine_exp --tensorboard
```

**Terminal 2: View Live Loss Graphs**
```bash
# Start TensorBoard
tensorboard --logdir OUTPUT

# Open browser to: http://localhost:6006
```

**Log Location:** `OUTPUT/{experiment_name}/logs/`

---

### NILM-main Training (TensorFlow + TensorBoard)

**Terminal 1: Start Training**
```bash
# TensorBoard automatically enabled
python NILM-main/S2S_train.py
python NILM-main/GRU_train.py
python NILM-main/fcn_train.py
```

**Terminal 2: View Live Loss Graphs**
```bash
# Start TensorBoard
tensorboard --logdir ./tensorboard_test

# Open browser to: http://localhost:6006
```

**Log Location:** `./tensorboard_test/`

---

## NILM Data Preprocessing (ukdale_processing.py)

```bash
python ukdale_processing.py --appliance_name kettle
python ukdale_processing.py --appliance_name microwave
python ukdale_processing.py --appliance_name fridge
python ukdale_processing.py --appliance_name dishwasher
python ukdale_processing.py --appliance_name washingmachine
```

---

## Algorithm 1 Processing

### Version 1: WITH Spike Removal (Recommended)
```bash
# Basic usage
python Data_filtering/algorithm1_v2.py --appliance_name microwave
python Data_filtering/algorithm1_v2.py --appliance_name kettle
python Data_filtering/algorithm1_v2.py --appliance_name washingmachine

# With custom parameters
python Data_filtering/algorithm1_v2.py --appliance_name kettle --window 100 --plot_samples 10000

# Adjust spike removal sensitivity
python Data_filtering/algorithm1_v2.py --appliance_name microwave --spike_threshold 2.0 --background_threshold 30

# Disable spike removal
python Data_filtering/algorithm1_v2.py --appliance_name microwave --no_remove_spikes
```

### Version 2: NO Spike Removal
```bash
python Data_filtering/algorithm1.py --appliance_name kettle
python Data_filtering/algorithm1.py --appliance_name microwave
```

**Output:** `Data/datasets/{appliance_name}.csv`

---

## Train Diffusion Model (main.py)

```bash
# Kettle
python main.py --config Config/kettle.yaml --name kettle_512 --train --tensorboard

# Washing Machine
python main.py --config Config/washingmachine.yaml --name washingmachine_1024 --train --tensorboard
```

---

## Sampling / Data Generation (main.py)

```bash
# Generate samples from trained model
python main.py --config Config/kettle.yaml --name kettle_512 --milestone 10
python main.py --config Config/washingmachine.yaml --name washingmachine1024 --milestone 16

# Custom number of samples
python main.py --config Config/kettle.yaml --name kettle_512 --milestone 10 --sample_num 5000
```

**Output:** `OUTPUT/{name}/ddpm_fake_{name}.npy`

---

## Visualize Synthetic Data

### Interactive Mode (Recommended)
```bash
# Just run the script - it will prompt you for file paths
python evaluation/visualize_synthetic_data.py

# Follow the prompts:
# 1. Enter synthetic data path (or press Enter for default)
# 2. Select appliance (1-5)
# 3. Enter real data path (optional, or press Enter to skip)
```

### Command-Line Mode
```bash
# Using appliance name (uses default paths)
python evaluation/visualize_synthetic_data.py --appliance kettle
python evaluation/visualize_synthetic_data.py --appliance microwave

# Using custom file paths
python evaluation/visualize_synthetic_data.py \
  --synthetic_path OUTPUT/kettle_512/ddpm_fake_kettle_512.npy \
  --real_path Data/datasets/kettle.csv
```

**Output:** `OUTPUT/visualizations/{appliance}_visualization.png`


---

## FID_ts (Fréchet Inception Distance for Time Series)

```bash
# Calculate FID for all appliances
python evaluation/fid_ts.py --all

# Calculate FID for specific appliance
python evaluation/fid_ts.py --appliance dishwasher \
  --real_path OUTPUT/dishwasher_512/samples/Dishwasher_norm_truth_512_train.npy \
  --synthetic_path OUTPUT/kettle_512/ddpm_fake_kettle_512.npy
```

---

## Inverse Transform (Denormalization)

```bash
# Convert normalized data back to original scale
python InverseTransform.py
```

---

## TensorBoard Tips

### View Multiple Experiments
```bash
# Compare multiple training runs
tensorboard --logdir OUTPUT --port 6006
```

### Remote Server (SSH)
```bash
# On server: Start TensorBoard
tensorboard --logdir OUTPUT --host 0.0.0.0 --port 6006

# On local machine: Port forward
ssh -L 6006:localhost:6006 user@server

# Open browser to: http://localhost:6006
```

### Google Colab
```python
# Load TensorBoard extension
%load_ext tensorboard
%tensorboard --logdir OUTPUT
```

---

## Appliance Parameters (Algorithm 1)

| Appliance | ON Threshold | Mean | Std |
|-----------|-------------|------|-----|
| Kettle | 200W | 700W | 1000W |
| Microwave | 200W | 500W | 800W |
| Fridge | 50W | 200W | 400W |
| Dishwasher | 10W | 700W | 1000W |
| Washing Machine | 20W | 400W | 700W |
