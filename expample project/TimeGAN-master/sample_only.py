
import os
import torch
import numpy as np
import pandas as pd
from options import Options
from lib.data import load_data
from lib.timegan import TimeGAN

try:
    from sklearn.svm import OneClassSVM
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def apply_ocsvm_filtering(ori_data, generated_data, data_name):
    """
    Implements the '+' in C-TimeGAN+ (Anomaly detection for synthetic data).
    Ref: IEEE TIM 2024 Section III-C
    """
    if not HAS_SKLEARN:
        print("⚠️ sklearn not found. Skipping OCSVM filtering. (pip install scikit-learn to enable)")
        return generated_data

    print(f"🛡️ Applying OCSVM Filtering (C-TimeGAN+ Enhancement)...")
    
    # Pre-process: Flatten the sequences [N, T, D] -> [N, T*D]
    n_ori, t_ori, d_ori = ori_data.shape
    n_gen, t_gen, d_gen = generated_data.shape
    
    X_train = ori_data.reshape(n_ori, -1)
    X_test = generated_data.reshape(n_gen, -1)
    
    # Train OCSVM on Real Data
    # kernel='rbf', nu=0.01 (defines the expected outlier ratio in real data)
    clf = OneClassSVM(gamma='auto', nu=0.01).fit(X_train)
    
    # Predict on Synthetic Data
    # 1 for inlier (realistic), -1 for outlier (anomalous)
    preds = clf.predict(X_test)
    
    realistic_idx = np.where(preds == 1)[0]
    unrealistic_count = len(preds) - len(realistic_idx)
    
    print(f"✅ OCSVM Filtered out {unrealistic_count} anomalies ({unrealistic_count/n_gen:.1%} of data).")
    
    return generated_data[realistic_idx]

def sample():
    # 1. 获取配置
    opt = Options().parse()
    
    # 自动定位之前训练好的权重路径
    # 假设你之前跑的是 kettle_training_
    weights_path = os.path.join(opt.outf, opt.name, 'train/weights')
    opt.resume = weights_path
    
    # 2. 加载数据 (Targets + Conditions)
    targets, conditions = load_data(opt)
    actual_dim = targets[0].shape[-1]
    opt.z_dim = actual_dim

    # 3. 初始化模型并自动加载权重
    print(f"🔄 Loading pre-trained weights from: {weights_path}")
    model = TimeGAN(opt, (targets, conditions))
    
    # 4. 仅生成数据
    num_samples = len(targets)
    print(f"🚀 Generating {num_samples} samples using saved weights...")
    generated_data = model.generation(num_samples)
    
    # 4.5. 🛡️ NEW: OCSVM Filtering (C-TimeGAN+)
    # Filter out noisy samples that don't match the real data manifold
    generated_data = apply_ocsvm_filtering(np.stack(ori_data), generated_data, opt.data_name)
    
    # 5. 保存结果
    output_dir = os.path.join(opt.outf, opt.name)
    npy_path = os.path.join(output_dir, f'timegan_fake_{opt.data_name}.npy')
    csv_path = os.path.join(output_dir, f'timegan_fake_{opt.data_name}_snippet.csv')
    
    # ⚡ CRITICAL: Save NPY first (Fast and Memory Efficient)
    print(f"📁 Saving full NPY file: {npy_path}...")
    np.save(npy_path, generated_data)
    print("✅ NPY saved successfully.")

    # ⚡ OPTIMIZATION: Only save a snippet to CSV (Preventing pandas RAM crash)
    print(f"📊 Saving a 10,000 sample snippet to CSV for preview...")
    # Take first 10,000 samples to keep CSV size manageable
    snippet_array = generated_data[:10000]
    if actual_dim == 1:
        df = pd.DataFrame(snippet_array.reshape(-1, 1), columns=['power'])
    else:
        cols = ['power'] + [f'time_feat_{i}' for i in range(actual_dim-1)]
        df = pd.DataFrame(snippet_array.reshape(-1, actual_dim), columns=cols)
    
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV Snippet saved: {csv_path}")
    
    print("=" * 60)
    print(f"✨ SUCCESS! Data saved to:")
    print(f"   NPY: {npy_path}")
    print(f"   CSV: {csv_path}")
    print(f"   Shape: {generated_data.shape}")
    print("=" * 60)

if __name__ == '__main__':
    sample()
