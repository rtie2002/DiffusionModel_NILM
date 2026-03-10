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
  file_path = os.path.join(dirname(dirname(abspath(__file__))), 'data', f'{data_name}.csv')
  
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing data file for {data_name} at {file_path}")

  ori_data = np.loadtxt(file_path, delimiter=",", skiprows=1)
  
  # ⚡ C-TimeGAN REDESIGN:
  # Column 0 is Aggregate (Condition)
  # Column 1 is Appliance (Target)
  # Column 2-9 are Time Features
  if ori_data.shape[1] > 1:
      print(f"Original columns: {ori_data.shape[1]}. Extracting Aggregate as Condition...")
      conditions = ori_data[:, 0:1] # Aggregate
      targets = ori_data[:, 1:]    # Appliance + Time Feats
  else:
      conditions = np.zeros((len(ori_data), 1))
      targets = ori_data.reshape(-1, 1)

  # Normalize separately
  conditions = MinMaxScaler(conditions)
  targets = MinMaxScaler(targets)
    
  temp_targets = []
  temp_conds = []
  for i in range(0, len(targets) - seq_len):
    temp_targets.append(targets[i:i + seq_len])
    temp_conds.append(conditions[i:i + seq_len])
        
  # Suffle indices
  idx = np.random.permutation(len(temp_targets))    
  return [temp_targets[i] for i in idx], [temp_conds[i] for i in idx]

def load_data(opt):
  ## Data loading
  if opt.data_name in ['kettle_training_', 'fridge_training_', 'dishwasher_training_', 'microwave_training_', 'washingmachine_training_']:
    print(f'Loading {opt.data_name} dataset with Conditions...')
    targets, conditions = real_data_loading(opt.data_name, opt.seq_len)
    return targets, conditions
  else:
    raise ValueError(f"Unknown dataset: {opt.data_name}")


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