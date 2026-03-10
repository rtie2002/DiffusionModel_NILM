
import os
import torch
import numpy as np
import pandas as pd
from options import Options
from lib.data import load_data
from lib.timegan import TimeGAN

def sample():
    # 1. 获取配置
    opt = Options().parse()
    
    # 自动定位之前训练好的权重路径
    # 假设你之前跑的是 kettle_training_
    weights_path = os.path.join(opt.outf, opt.name, 'train/weights')
    opt.resume = weights_path
    
    # 2. 加载数据（为了获取真实的 Min/Max 用于反向归一化）
    ori_data = load_data(opt)
    actual_dim = ori_data[0].shape[-1]
    opt.z_dim = actual_dim

    # 3. 初始化模型并自动加载权重（We set resume in opt so it loads automatically）
    print(f"🔄 Loading pre-trained weights from: {weights_path}")
    model = TimeGAN(opt, ori_data)
    
    # 4. 仅生成数据 (跳过训练!)
    num_samples = len(ori_data)
    print(f"🚀 Generating {num_samples} samples using saved weights...")
    generated_data = model.generation(num_samples)
    
    # 5. 保存结果
    generated_array = np.array(generated_data)
    output_dir = os.path.join(opt.outf, opt.name)
    npy_path = os.path.join(output_dir, f'timegan_fake_{opt.data_name}.npy')
    csv_path = os.path.join(output_dir, f'timegan_fake_{opt.data_name}.csv')
    
    np.save(npy_path, generated_array)
    
    if actual_dim == 1:
        df = pd.DataFrame(generated_array.reshape(-1, 1), columns=['power'])
    else:
        cols = ['power'] + [f'time_feat_{i}' for i in range(actual_dim-1)]
        df = pd.DataFrame(generated_array.reshape(-1, actual_dim), columns=cols)
    
    df.to_csv(csv_path, index=False)
    
    print("=" * 60)
    print(f"✨ SUCCESS! Data saved to:")
    print(f"   NPY: {npy_path}")
    print(f"   CSV: {csv_path}")
    print(f"   Shape: {generated_array.shape}")
    print("=" * 60)

if __name__ == '__main__':
    sample()
