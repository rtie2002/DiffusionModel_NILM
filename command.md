# Command Reference - DiffusionModel_NILM

## 🚀 Conditional DDPM Training & Sampling (NEW)

### Training Conditional DDPM

**Basic Training** (with time features):
```bash
# Washing Machine (multivariate: power + 8 time features)
python main.py --train --config Config/washingmachine.yaml --name washingmachine_multivariate --tensorboard

# Other appliances
python main.py --train --config Config/kettle.yaml --name kettle_multivariate --tensorboard
python main.py --train --config Config/microwave.yaml --name microwave_multivariate --tensorboard
python main.py --train --config Config/fridge.yaml --name fridge_multivariate --tensorboard
python main.py --train --config Config/dishwasher.yaml --name dishwasher_multivariate --tensorboard
python main.py --train --config Config/washingmachine.yaml --name washingmachine_multivariate --tensorboard
```

**Training Controls**:
- `--train`: Enable training mode
- `--name`: Experiment name (for organizing checkpoints/outputs)
- `--tensorboard`: Enable TensorBoard logging (optional but recommended)
- `--config`: Path to config file (controls all hyperparameters)

**Config File Controls** (`Config/washingmachine.yaml`):
```yaml
model:
  params:
    condition_dim: 8          # Number of time features (minute, hour, dow, month sin/cos)
    seq_length: 512           # Window size
    feature_size: 1           # Power feature (1D)
    timesteps: 2000           # Diffusion steps
    d_model: 128              # Model hidden dimension
    n_layer_enc: 2            # Encoder layers
    n_layer_dec: 2            # Decoder layers

solver:
  base_lr: 1.0e-6            # Learning rate
  max_epochs: 20000          # Total training epochs
  save_cycle: 2000           # Save checkpoint every N epochs
  results_folder: .Checkpoints/Checkpoints_washingmachine_multivariate

dataloader:
  batch_size: 64             # Batch size
  train_dataset:
    params:
      data_root: ./Data/datasets/washingmachine_multivariate.csv  # 9-column CSV
```

**Checkpoint Locations**:
```
.Checkpoints/Checkpoints_washingmachine_multivariate/
├── checkpoint-2000.pt
├── checkpoint-4000.pt
├── ...
└── checkpoint-20000.pt
```

---

### Sampling / Conditional Generation

**Basic Sampling** (uses dataset size by default):
```bash
python main.py --config Config/washingmachine.yaml --name washingmachine_multivariate --milestone 20000
```

**Custom Sample Count**:
```bash
# Generate 5000 samples
python main.py --config Config/washingmachine.yaml --name washingmachine_multivariate --milestone 20000 --sample_num 5000

# Generate 10000 samples
python main.py --config Config/washingmachine.yaml --name washingmachine_multivariate --milestone 20000 --sample_num 10000
```

**Use Different Checkpoints**:
```bash
# Use epoch 10000 model
python main.py --config Config/washingmachine.yaml --name washingmachine_multivariate --milestone 10000

# Use epoch 15000 model
python main.py --config Config/washingmachine.yaml --name washingmachine_multivariate --milestone 15000
```

**Sampling Controls**:
- `--milestone`: Which checkpoint to use (epoch number)
- `--sample_num`: Number of samples to generate (default: dataset size)
- `--name`: Must match training experiment name

**Output**:
```
OUTPUT/washingmachine_multivariate/
└── ddpm_fake_washingmachine_multivariate.npy  # (N, 512, 9)
    # Column 0: Power
    # Columns 1-8: Time features (minute, hour, dow, month sin/cos)
```

**Time Distribution**:
- Automatically samples time features from training dataset
- Maintains original time distribution (e.g., if 10% of data is from January, ~10% of generated data will be too)
- Uses `replace=False` to avoid duplicates when `sample_num ≤ dataset_size`

---

### Data Augmentation Workflow

**1. Prepare Multivariate Data**:
```bash
# Run Algorithm 1 with multivariate output
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name washingmachine
# Output: washingmachine_multivariate.csv (9 columns)
```

**2. Train Conditional Model**:
```bash
python main.py --train --config Config/washingmachine.yaml --name washingmachine_multivariate --tensorboard
```

**3. Generate Synthetic Data**:
```bash
# Generate same amount as real data
python main.py --config Config/washingmachine.yaml --name washingmachine_multivariate --milestone 20000

# Or generate more for augmentation
python main.py --config Config/washingmachine.yaml --name washingmachine_multivariate --milestone 20000 --sample_num 10000
```

**4. Verify Generated Data**:
```python
import numpy as np
import pandas as pd

# Load
data = np.load('OUTPUT/washingmachine_multivariate/ddpm_fake_washingmachine_multivariate.npy')
print(f"Shape: {data.shape}")  # (N, 512, 9)

# Convert to DataFrame
data_2d = data.reshape(-1, 9)
df = pd.DataFrame(data_2d, columns=[
    'washingmachine', 'minute_sin', 'minute_cos',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos'
])

# Save as CSV
df.to_csv('synthetic_washingmachine_9d.csv', index=False)
```

---

## Live Loss Visualization with TensorBoard

### Diffusion Model Training (PyTorch + TensorBoard)

**Terminal 1: Start Training**
```bash
# Train with TensorBoard enabled
python main.py --train --config Config/washingmachine.yaml --name washingmachine_multivariate --tensorboard
```

**Terminal 2: View Live Loss Graphs**
```bash
# Start TensorBoard
tensorboard --logdir OUTPUT

# Open browser to: http://localhost:6006
```

**Log Location:** `OUTPUT/{experiment_name}/logs/`

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

### Multivariate Version (WITH Time Features - For Conditional DDPM)
```bash
# Generate 9-column CSV (power + 8 time features)
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name washingmachine
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name kettle
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name microwave

# With custom parameters
python Data_filtering/algorithm1_v2_multivariate.py --appliance_name kettle --window 100 --plot_samples 10000
```

**Output:** `Data/datasets/{appliance_name}_multivariate.csv` (9 columns)

### Standard Version (Power Only)
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

**Output:** `Data/datasets/{appliance_name}.csv` (1 column)

---

## Resume Training from Checkpoint

```bash
# Training will automatically resume from latest checkpoint if it exists
python main.py --train --config Config/washingmachine.yaml --name washingmachine_multivariate

# To start fresh, delete checkpoint folder or use a new name
```

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

---

## Quick Reference

### Training
```bash
python main.py --train --config Config/{appliance}.yaml --name {exp_name} --tensorboard
```

### Generation
```bash
python main.py --config Config/{appliance}.yaml --name {exp_name} --milestone {epoch}
```

### Key Parameters
- `--train`: Training mode
- `--name`: Experiment identifier
- `--milestone`: Checkpoint epoch to use
- `--sample_num`: Number of samples to generate
- `--tensorboard`: Enable logging
