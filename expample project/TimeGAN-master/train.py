"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

train.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
from options import Options
from lib.data import load_data
from lib.timegan import TimeGAN


def train():
    """ Training
    """

    # ARGUMENTS
    opt = Options().parse()

    # LOAD DATA
    ori_data = load_data(opt)
    
    # NEW: Automatically update dimension based on loaded data
    # ori_data is a list of [seq_len, dim]
    actual_dim = ori_data[0].shape[-1]
    opt.z_dim = actual_dim
    print(f"Detected data dimension: {actual_dim}. Updating model config...")

    # LOAD MODEL
    model = TimeGAN(opt, ori_data)

    # TRAIN MODEL
    model.train()
    
    # GENERATE SYNTHETIC DATA
    # Generate a substantial amount of data for evaluation
    num_samples = len(ori_data)
    print(f"Generating {num_samples} synthetic samples...")
    generated_data = model.generation(num_samples)
    
    # Process for saving
    generated_array = np.array(generated_data) # [N, seq_len, dim]
    
    # Define save paths
    output_dir = os.path.join(opt.outf, opt.name)
    npy_path = os.path.join(output_dir, f'timegan_fake_{opt.data_name}.npy')
    csv_path = os.path.join(output_dir, f'timegan_fake_{opt.data_name}.csv')
    
    # 1. Save as .npy (Same format as diffusion model)
    np.save(npy_path, generated_array)
    print(f"NPY saved: {npy_path} (Shape: {generated_array.shape})")
    
    # 2. Save as .csv (Flattened for easy viewing)
    # Handle potentially multi-dimensional data by flattening or taking power column
    if actual_dim == 1:
        # Just power
        df = pd.DataFrame(generated_array.reshape(-1, 1), columns=['power'])
    else:
        # Full 9 dimensions
        cols = ['power'] + [f'time_feat_{i}' for i in range(actual_dim-1)]
        df = pd.DataFrame(generated_array.reshape(-1, actual_dim), columns=cols)
        
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

if __name__ == '__main__':
    train()
