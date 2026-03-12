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
        print(f"OK - Using device: CUDA (GPU)")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"WARNING: Using device: CPU (No GPU detected or CUDA disabled)")
    print("=" * 60)
    print()

    # The following lines related to OCSVM filtering and `generated_data`
    # are typically applied *after* data generation, not during training.
    # Placing them here would result in a NameError as `generated_data`
    # is not defined at this stage of the `train()` function.
    # If this logic is intended for `sample_only.py` or after model.train(),
    # please adjust the placement accordingly.
    #
    # # 4.5. 🛡️ NEW: OCSVM Filtering (C-TimeGAN+)
    # # Filter out noisy samples that don't match the real data manifold
    # generated_data = apply_ocsvm_filtering(np.stack(targets), generated_data, opt.data_name)
    
    # LOAD DATA
    targets, conditions = load_data(opt)
    
    # --- DYNAMIC DIMENSION DETECTION ---
    actual_dim = targets[0].shape[-1]
    cond_dim = conditions[0].shape[-1]
    
    opt.z_dim = actual_dim
    opt.cond_dim = cond_dim 
    
    print(f"\nDIMENSION SUMMARY:")
    print(f"   -> Target Dim (Appliance+Time): {actual_dim}")
    print(f"   -> Cond   Dim (Context Source): {cond_dim}")
    print(f"   -> Total RNN Input          : {actual_dim + cond_dim}")
    print("-" * 60)

    # LOAD MODEL (Pass tuple to BaseModel)
    model = TimeGAN(opt, (targets, conditions))

    # TRAIN MODEL
    model.train()
    
    print(f"SUCCESS: Training of {opt.data_name} completed.")
    print(f"📂 Weights saved in: {os.path.join(opt.outf, opt.name, 'train', 'weights')}")
    print(f"INFO: You can now run sample_only.py or run_all_timegan.sh to generate data.")

if __name__ == '__main__':
    train()
