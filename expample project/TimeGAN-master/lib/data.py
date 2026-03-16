"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

data.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
(3) load_data: download or generate data
(4): batch_generator: mini-batch generator
"""

import numpy as np
from os.path import dirname, abspath


def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets, extracting Aggregate as a Condition."""
  import os
  # Try different relative paths to reliably find the data
  possible_paths = [
      os.path.join(dirname(dirname(abspath(__file__))), 'data', f'{data_name}.csv'),
      os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'Data', 'datasets', f'{data_name}.csv'),
      f'/home/raymond/projects/DiffusionModel_NILM/Data/datasets/{data_name}.csv',
      f'C:/Users/Raymond Tie/Desktop/DiffusionModel_NILM/Data/datasets/{data_name}.csv',
  ]
  
  file_path = None
  for p in possible_paths:
      if os.path.exists(p):
          file_path = p
          break

  if file_path is None:
    raise FileNotFoundError(f"Missing data file for {data_name}. Tried: {possible_paths}")

  print(f"\n📂 DATA SOURCE RECOGNIZED:")
  print(f"   -> Name: {data_name}")
  print(f"   -> Path: {file_path}")
  
  ori_data = np.loadtxt(file_path, delimiter=",", skiprows=1)
  
  # ⚡ C-TimeGAN LEAN ARCHITECTURE (Requested Fix):
  if ori_data.shape[1] == 9:
      print(f"✅ 9-Column Mode Detected:")
      print(f"   -> TARGET: Appliance Power (Col 0) | Dim: 1")
      print(f"   -> COND  : Time Features (Col 1-8) | Dim: 8")
      # TARGET: Only Appliance Power (1 column)
      targets = ori_data[:, 0:1]    
      # CONDITION: Time features (8 columns)
      conditions = ori_data[:, 1:] 
  elif ori_data.shape[1] >= 10:
      print(f"✅ 10+ Column Mode Detected:")
      print(f"   -> TARGET: Appliance Power (Col 1) | Dim: 1")
      print(f"   -> COND  : Aggregate + Time (Col 0, 2-9) | Dim: 9")
      # TARGET: Only Appliance Power
      targets = ori_data[:, 1:2]
      # CONDITION: Aggregate (Col 0) + Time (Col 2-9)
      conditions = np.column_stack([ori_data[:, 0:1], ori_data[:, 2:10]])
  else:
      print(f"⚠️ Low Dimension Data ({ori_data.shape[1]} cols).")
      conditions = ori_data[:, 0:1]
      targets = ori_data[:, 0:1]

  # ⚡ SMART NORMALIZATION: 
  # If data is already in [0, 1], don't do anything (prevents double MinMax).
  # If data is NOT in [0, 1] (like Time Features in [-1, 1]), normalize it to [0, 1] for TimeGAN.
  
  def safe_minmax(data):
      d_min = np.min(data, axis=0)
      d_max = np.max(data, axis=0)
      # Tolerance check for existing [0, 1] range
      if np.all(d_min >= -0.001) and np.all(d_max <= 1.001):
          print("      -> Column already in [0, 1] range. Skipping re-normalization.")
          return data
      return MinMaxScaler(data)

  print(f"   -> Processing {data_name} columns...")
  conditions = safe_minmax(conditions)
  targets = safe_minmax(targets)
    
  # ─────────────────────────────────────────────────────────────────────────
  # ⚡ C-TimeGAN CORE: First-Order Difference as Condition (Paper Eq. 7)
  #
  #   The paper explicitly states:
  #   "the difference between adjacent time points of the current appliance
  #    (first-order difference) is also used as a condition"
  #
  #   This is THE key feature that separates C-TimeGAN from vanilla TimeGAN.
  #   Without it, the generator has no guidance on transient/sharp changes.
  # ─────────────────────────────────────────────────────────────────────────
  print(f"   -> Computing First-Order Difference (C-TimeGAN Eq. 7)...")
  # diff[t] = target[t] - target[t-1], with diff[0] = 0
  diff = np.diff(targets[:, 0], prepend=targets[0, 0])  # (N,)
  # Normalize diff to [0, 1] for stable training
  diff_min = np.min(diff)
  diff_max = np.max(diff)
  if diff_max - diff_min > 1e-7:
      diff_norm = (diff - diff_min) / (diff_max - diff_min)
  else:
      diff_norm = np.zeros_like(diff)
  diff_norm = diff_norm.reshape(-1, 1)  # (N, 1)
  
  # Append first-order difference as an extra condition column
  conditions = np.column_stack([conditions, diff_norm])
  print(f"   -> Condition dim after adding 1st-order diff: {conditions.shape[1]}")

  temp_targets = []
  temp_conds = []
  
  # ⚡ FULL DATASET MODE: No filtering, using sliding windows for 100% coverage
  print(f"   -> Processing FULL dataset for overall waveform (Window Size: {seq_len})...")
  
  # Use stride=1 for complete sliding window coverage
  stride = 1 
  active_count = 0
  
  for i in range(0, len(targets) - seq_len, stride):
    window_t = targets[i:i + seq_len]
    window_c = conditions[i:i + seq_len]
    
    # Logic: No threshold check. We want to handle the whole dataset (ON and OFF).
    temp_targets.append(window_t)
    temp_conds.append(window_c)
    active_count += 1
  
  print(f"✅ Processed {active_count} windows covering the entire sequence.")
        
  # Shuffle indices
  idx = np.random.permutation(len(temp_targets))    
  return [temp_targets[i] for i in idx], [temp_conds[i] for i in idx]

def load_data(opt):
  ## Data loading
  if opt.data_name == 'sine':
    # If you ever want to add sine back
    return sine_data_generation(opt.num_samples, opt.seq_len, opt.z_dim), None

  else:
    print(f'Loading {opt.data_name} dataset with Conditions...')
    targets, conditions = real_data_loading(opt.data_name, opt.seq_len)
    return targets, conditions



def batch_generator(data, conditions, batch_size):
  """Mini-batch generator for C-TimeGAN.

  Args:
    - data: time-series data
    - conditions: condition information (Aggregate power)
    - batch_size: the number of samples in each batch

  Returns:
    - X_mb: targets in each batch
    - T_mb: indices (dummies for time if not used)
    - C_mb: conditions in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]

  X_mb = list(data[i] for i in train_idx)
  C_mb = list(conditions[i] for i in train_idx)
  T_mb = [len(x) for x in X_mb] # Keep lengths for RNN masking

  return X_mb, T_mb, C_mb