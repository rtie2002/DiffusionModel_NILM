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
  """Load and preprocess real-world datasets for NILM appliances.
  
  Args:
    - data_name: kettle, fridge, dishwasher, microwave, or washingmachine
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  import os
  file_path = os.path.join(dirname(dirname(abspath(__file__))), 'data', f'{data_name}.csv')
  
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing data file for {data_name} at {file_path}")

  # Load data from CSV (assuming power is in the last column or only column)
  # NILM datasets usually have a header
  ori_data = np.loadtxt(file_path, delimiter=",", skiprows=1)
  
  # NEW: Skip the first column (aggregated power) and use columns 1 to 9
  # Columns: [0: Aggregated, 1: Appliance Power, 2-9: Time Features]
  if ori_data.shape[1] > 1:
      print(f"Original columns: {ori_data.shape[1]}. Selecting columns 1 to end (Appliance + Time Features)...")
      ori_data = ori_data[:, 1:] 
  
  # Ensure 2D shape [samples, features]
  if len(ori_data.shape) == 1:
    ori_data = ori_data.reshape(-1, 1)
        
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data


def load_data(opt):
  ## Data loading
  if opt.data_name in ['kettle_training_', 'fridge_training_', 'dishwasher_training_', 'microwave_training_', 'washingmachine_training_']:
    print(f'Loading {opt.data_name} dataset...')
    ori_data = real_data_loading(opt.data_name, opt.seq_len)
  elif opt.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, opt.seq_len, dim)
  else:
    raise ValueError(f"Unknown dataset: {opt.data_name}")
    
  print(opt.data_name + ' dataset is ready.')

  return ori_data


def batch_generator(data, time, batch_size):
  """Mini-batch generator.

  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch

  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]

  X_mb = list(data[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)

  return X_mb, T_mb