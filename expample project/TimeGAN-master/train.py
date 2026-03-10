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
    import torch

    # ARGUMENTS
    opt = Options().parse()

    # --- DEVICE INFORMATION ---
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"✓ Using device: CUDA (GPU)")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"⚠ Using device: CPU (No GPU detected or CUDA disabled)")
    print("=" * 60)
    print()

    # LOAD DATA
    # C-TimeGAN REDESIGN: load_data returns (targets, conditions)
    targets, conditions = load_data(opt)
    
    # Update dimension based on loaded targets
    actual_dim = targets[0].shape[-1]
    opt.z_dim = actual_dim
    print(f"Detected target dimension: {actual_dim}. Updating model config...")

    # LOAD MODEL (Pass tuple to BaseModel)
    model = TimeGAN(opt, (targets, conditions))

    # TRAIN MODEL
    model.train()
    
    # Post-training sampling is now handled by sample_only.py 
    # to inclusion OCSVM filtering. Use run_all_timegan.sh to trigger both.
    print(f"Training of {opt.data_name} completed.")
    
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
