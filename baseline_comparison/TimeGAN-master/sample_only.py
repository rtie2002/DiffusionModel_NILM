
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


def inject_on_period_texture(generated_data, real_targets, on_threshold=0.05, noise_scale=0.015):
    """
    CNN-inspired Texture Injection for ON periods.

    When the generated waveform has flat ON blocks (TimeGAN weakness),
    this function:
      1. Detects ON periods (power > threshold) per sample
      2. Finds real training windows at a similar power level
      3. Extracts the high-frequency texture (residual from mean)
      4. Injects that texture onto the flat block

    Args:
        generated_data : np.ndarray [N, T, D] - generated samples
        real_targets   : np.ndarray [N, T, D] - real training samples (normalized)
        on_threshold   : float - normalized power threshold to detect ON state
        noise_scale    : float - how strongly to blend real texture in

    Returns:
        textured_data  : np.ndarray [N, T, D] - texture-injected output
    """
    print("🎨 Injecting ON-period texture from real data...")
    textured = generated_data.copy()
    real_arr = np.array(real_targets)   # [M, T, D]
    real_power = real_arr[:, :, 0]      # [M, T] - power channel only

    N, T, D = generated_data.shape

    for i in range(N):
        gen_power = textured[i, :, 0]   # [T]

        # --- Step 1: Detect ON period ---
        on_mask = gen_power > on_threshold
        if not np.any(on_mask):
            continue  # All OFF, no texture needed

        # --- Step 2: Find the mean power level of this ON block ---
        mean_gen_power = np.mean(gen_power[on_mask])

        # --- Step 3: Find real windows with similar mean power (±30%) ---
        real_means = np.mean(real_power, axis=1)  # [M]
        lower = mean_gen_power * 0.7
        upper = mean_gen_power * 1.3
        candidate_idx = np.where((real_means >= lower) & (real_means <= upper))[0]

        if len(candidate_idx) == 0:
            # Fallback: top-20 closest real windows
            distances = np.abs(real_means - mean_gen_power)
            candidate_idx = np.argsort(distances)[:20]

        # --- Step 4: Randomly pick one real window and extract texture ---
        chosen_idx = np.random.choice(candidate_idx)
        real_window = real_power[chosen_idx]      # [T]
        real_mean   = np.mean(real_window)

        # High-frequency texture = real signal - its own mean (residual)
        texture = real_window - real_mean         # [T]

        # --- Step 5: Inject texture ONLY during ON periods ---
        # Scale texture to match generated power level
        scale = mean_gen_power / (real_mean + 1e-7)
        scaled_texture = texture * scale * noise_scale

        textured[i, on_mask, 0] += scaled_texture[on_mask]

        # Clip to valid range [0, max of generated]
        textured[i, :, 0] = np.clip(textured[i, :, 0], 0, gen_power.max() * 1.1)

    print(f"✅ Texture injection complete ({N} samples processed).")
    return textured


def concat_to_target_length(data, target_len=512):
    """
    Concatenate consecutive short windows [N, 60, D] into
    long sequences [M, 512, D] to match Diffusion Model format.

    Strategy:
      - Groups of ceil(target_len / win_len) consecutive windows
        are stitched together, then trimmed to exactly target_len steps.
      - Consecutive windows are used (not random) to preserve
        temporal continuity.

    Args:
        data       : np.ndarray [N, T, D]  - short windows (e.g. T=60)
        target_len : int                   - target sequence length (e.g. 512)

    Returns:
        long_data  : np.ndarray [M, target_len, D]
    """
    N, T, D = data.shape
    windows_per_seq = int(np.ceil(target_len / T))   # e.g. ceil(512/60) = 9
    step = windows_per_seq                            # non-overlapping groups

    long_seqs = []
    for start in range(0, N - windows_per_seq + 1, step):
        group = data[start : start + windows_per_seq]   # [9, 60, D]
        stitched = np.concatenate(group, axis=0)         # [540, D]
        trimmed  = stitched[:target_len, :]              # [512, D]
        long_seqs.append(trimmed)

    long_data = np.stack(long_seqs, axis=0)             # [M, 512, D]
    print(f"✅ Concatenated: {N} windows × {T} steps → {len(long_seqs)} sequences × {target_len} steps")
    return long_data



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
    
    # ⚡ OCSVM on 1 Million samples has O(N^3) time complexity and will freeze the computer forever.
    # FIX: We randomly subsample a maximum of 10,000 points to train the SVM boundary. 
    # This takes seconds instead of days, and 10k is plenty to define the real data manifold.
    max_train_samples = min(10000, n_ori)
    np.random.seed(42)  # For reproducibility
    demo_indices = np.random.choice(n_ori, max_train_samples, replace=False)
    X_train_sub = X_train[demo_indices]
    
    print(f"   -> Training OCSVM boundary on {max_train_samples} real samples...")
    # kernel='rbf', nu=0.01 (defines the expected outlier ratio in real data)
    clf = OneClassSVM(gamma='auto', nu=0.01).fit(X_train_sub)
    
    # Predict on Synthetic Data (Batched to prevent RAM spikes)
    print(f"   -> Predicting anomalies in {n_gen} synthetic samples...")
    batch_size = 50000
    preds = []
    
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_gen, desc="OCSVM Predict")
    except ImportError:
        pbar = None
        
    for i in range(0, n_gen, batch_size):
        end_idx = min(i + batch_size, n_gen)
        batch_preds = clf.predict(X_test[i:end_idx])
        preds.append(batch_preds)
        if pbar: pbar.update(end_idx - i)
        
    if pbar: pbar.close()
    
    preds = np.concatenate(preds)
    
    # 1 for inlier (realistic), -1 for outlier (anomalous)
    realistic_idx = np.where(preds == 1)[0]
    unrealistic_count = len(preds) - len(realistic_idx)
    
    print(f"✅ OCSVM Filtered out {unrealistic_count} anomalies ({(unrealistic_count/n_gen)*100:.1f}% of data).")
    
    return generated_data[realistic_idx]

def sample():
    # 1. 获取配置
    opt = Options().parse()
    
    weights_path = os.path.join(opt.outf, opt.name, 'train/weights')
    opt.resume = weights_path
    
    print("📜 Output mode: Pure MinMax [0, 1] (Aligning with Diffusion Model logic).")


    # 3. 加载归一化数据 (Targets + Conditions)
    targets, conditions = load_data(opt)
    actual_dim = targets[0].shape[-1]
    cond_dim = conditions[0].shape[-1]
    
    opt.z_dim = actual_dim
    opt.cond_dim = cond_dim 
    
    print(f"📊 Sampling Logic:")
    print(f"   -> Appliance Dim : {actual_dim}")
    print(f"   -> Context   Dim : {cond_dim}")

    # 4. 初始化模型并加载权重
    print(f"🔄 Loading pre-trained weights from: {weights_path}")
    model = TimeGAN(opt, (targets, conditions))
    
    # 5. 生成数据 (output is in [0,1] normalized range)
    num_samples = len(targets)
    print(f"🚀 Generating {num_samples} samples...")
    generated_data = model.generation(num_samples)   # [N, seq_len, 9], values in [0,1]

    # 6. OCSVM Filtering
    # ⚠️ DISABLED: User requested to turn off anomaly detection.
    # generated_data = apply_ocsvm_filtering(np.stack(targets), generated_data, opt.data_name)
    
    # 7. ON-Period Texture Injection
    # ⚠️ DISABLED: User requested the model to learn texture natively during training.
    # generated_data = inject_on_period_texture(generated_data, np.stack(targets))


    # ─────────────────────────────────────────────────────────────────────────
    # 8. ⚡ OUTPUT ALIGNMENT (Strictly MinMax [0, 1])
    #
    #    User requested to output in MinMax [0, 1] to match Diffusion Model logic.
    #    - Channel 0   : Power, stays in [0, 1]
    #    - Channel 1-8 : Time features, restored to [-1, 1]
    # ─────────────────────────────────────────────────────────────────────────
    print("🔁 Outputting data in MinMax [0, 1] (Aligning with Diffusion Model logic)...")
    
    # ⚡ FIX: Remove the last channel (First-Order Difference) which was appended 
    # strictly for TimeGAN training and does not exist in the Diffusion Model.
    final_data = generated_data[:, :, :-1].copy()

    # Channel 0: Stays normalized [0, 1]
    final_data[:, :, 0] = final_data[:, :, 0]

    # Channel 1 to end (Time/Aggregate features): restored to [-1, 1]
    final_data[:, :, 1:] = final_data[:, :, 1:] * 2.0 - 1.0


    # ─────────────────────────────────────────────────────────────────────────
    # 8.5 ⚡ RESHAPE to match Diffusion Model window size [N, 512, 9]
    #   TimeGAN  : [N, 60, 9]  ← seq_len 60
    #   Diffusion: [M, 512, 9]  ← window  512
    #   Strategy : stitch consecutive 60-step windows → trim to 512
    # ─────────────────────────────────────────────────────────────────────────
    DIFFUSION_WINDOW = 512
    final_data = concat_to_target_length(final_data, target_len=DIFFUSION_WINDOW)

    N, L, V = final_data.shape
    print(f"✅ Final shape : {final_data.shape}")
    print(f"   Power range : {final_data[:,:,0].min():.4f} ~ {final_data[:,:,0].max():.4f}")
    print(f"   Time range  : {final_data[:,:,1:].min():.3f} ~ {final_data[:,:,1:].max():.3f}")

    # 9. Save
    output_dir = os.path.join(opt.outf, opt.name)
    npy_path   = os.path.join(output_dir, f'timegan_fake_{opt.data_name}.npy')
    csv_path   = os.path.join(output_dir, f'timegan_fake_{opt.data_name}_snippet.csv')

    print(f"📁 Saving NPY: {npy_path}")
    np.save(npy_path, final_data)
    print("✅ NPY saved.")

    # Snippet CSV (first 10,000 samples only)
    snippet = final_data[:10000]
    cols = ['power'] + [f'time_feat_{i}' for i in range(V - 1)]
    df = pd.DataFrame(snippet.reshape(-1, V), columns=cols)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV snippet saved: {csv_path}")

    print("=" * 60)
    print(f"✨ SUCCESS!")
    print(f"   NPY : {npy_path}")
    print(f"   Shape: {final_data.shape}  ← same as ddpm_fake_xxx.npy")
    print("=" * 60)

if __name__ == '__main__':
    sample()

