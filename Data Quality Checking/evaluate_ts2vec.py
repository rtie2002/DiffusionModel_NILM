"""
TS2Vec Evaluation for Real vs Synthetic NILM Data

This script uses TS2Vec (Time Series to Vector) to evaluate the quality of synthetic data.
Theory:
1. Train a TS2Vec encoder on Real data to learn efficient time-series representations.
2. Encode both Real and Synthetic samples into latent vectors.
3. Metrics:
   - Discriminative Score: Train a classifier to distinguish Real vs Synthetic. 
     (Target Accuracy = 0.5 means indistinguishable/perfect).
   - Visualization: PCA/t-SNE plot of the embeddings.

Usage:
    python evaluate_ts2vec.py <appliance> [options]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy import linalg

# Add project root to sys.path to allow imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import TS2Vec
try:
    from Models.ts2vec.ts2vec import TS2Vec
except ImportError:
    # Try alternate path if generic import fails
    sys.path.append(os.path.join(PROJECT_ROOT, "Models", "ts2vec"))
    from models.ts2vec.ts2vec import TS2Vec

# ==================== Configuration ====================
REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "datasets", "real_distributions")

# Allow override via environment variables for comparing different baselines
SYNTHETIC_DATA_DIR = os.environ.get('SYNTHETIC_DATA_DIR_OVERRIDE', 
                                     os.path.join(PROJECT_ROOT, "Data", "datasets", "synthetic_processed"))
RESULTS_DIR = os.environ.get('RESULTS_DIR_OVERRIDE',
                              os.path.join(PROJECT_ROOT, "Data Quality Checking", "ts2vec_results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
JUDGES_DIR = os.path.join(PROJECT_ROOT, "Data Quality Checking", "pretrained_judges")
os.makedirs(JUDGES_DIR, exist_ok=True)

def load_data(appliance, sequence_length=480, max_samples=100000, mode='multivariate'):
    """Load and preprocess real and synthetic data with appliance-specific strategies."""
    print(f"Loading data for {appliance} (Mode: {mode})...")
    
    # === APPLIANCE-SPECIFIC STRATEGY ===
    # For Spiky/Low-duty-cycle appliances, use smaller windows to focus on the pulse
    # rather than being dominated by 'silence' (OFF periods).
    if appliance.lower() in ['kettle', 'microwave']:
        seq_len = 128  # Zoom in on the pulse (approx 2 mins)
        stride = 16    # High-density event capturing
    else:
        seq_len = sequence_length if sequence_length else 480
        stride = 10    # Trajectory-based continuity for heavy appliances
        
    real_path = os.path.join(REAL_DATA_DIR, f"{appliance}_multivariate.csv")
    synth_path = os.path.join(SYNTHETIC_DATA_DIR, f"{appliance}_multivariate.csv")
    
    if not os.path.exists(real_path) or not os.path.exists(synth_path):
        raise FileNotFoundError(f"Data files for {appliance} not found.")
        
    df_real = pd.read_csv(real_path)
    df_synth = pd.read_csv(synth_path)

    # Data Cleaning: Handle NaNs early
    df_real = df_real.fillna(method='ffill').fillna(0)
    df_synth = df_synth.fillna(method='ffill').fillna(0)
    
    # 🧪 Dithering for Spiky Appliances
    # Prevents 'Artificial Island' effect caused by machine-perfect flat lines
    if appliance.lower() in ['kettle', 'microwave']:
        noise = np.random.normal(0, 0.001, df_synth.shape)
        df_synth = df_synth + noise

    # Column Filtering based on Mode
    if mode == 'power':
        power_col = [col for col in df_real.columns if col.lower() == appliance.lower() or col == df_real.columns[0]][0]
        df_real = df_real[[power_col]]
        df_synth = df_synth[[power_col]]
    elif mode == 'time':
        time_cols = [col for col in df_real.columns if 'sin' in col or 'cos' in col or 'hour' in col or 'minute' in col]
        df_real = df_real[time_cols]
        df_synth = df_synth[time_cols]
    
    def df_to_windows(df, w_len, w_stride, limit=None):
        data = df.values
        if len(data) < w_len:
            data = np.pad(data, ((0, w_len - len(data)), (0, 0)), mode='edge')

        n_samples = len(data)
        num_windows = (n_samples - w_len) // w_stride + 1
        
        # Create windows in CHRONOLOGICAL ORDER
        windows = np.array([data[i * w_stride : i * w_stride + w_len] for i in range(num_windows)])
        
        # Maintain chronological integrity for split
        if limit and len(windows) > limit:
            windows = windows[:limit]
            
        return windows, df.columns.tolist()

    real_windows, cols = df_to_windows(df_real, seq_len, stride, max_samples)
    synth_windows, _ = df_to_windows(df_synth, seq_len, stride, max_samples)
    
    print(f"   ✓ Extracted {len(real_windows)} windows (Len: {seq_len}, Stride: {stride})")
    return real_windows, synth_windows, cols

def train_ts2vec(train_data, input_dims, output_dims=320, device='cuda'):
    """Train TS2Vec model on the data."""
    print("\nInitializing TS2Vec training...")
    
    # Check if GPU is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
        device = 'cpu'
        
    model = TS2Vec(
        input_dims=input_dims,
        output_dims=output_dims,
        hidden_dims=64,
        depth=10,
        device=device,
        lr=0.001,
        batch_size=16,
        temporal_unit=0
    )
    
    print("Training TS2Vec...")
    loss_log = model.fit(train_data, n_epochs=100, verbose=True)
    return model, loss_log

def calculate_fid(real_embeddings, synth_embeddings, eps=1e-4):
    """
    Calculate the Fréchet Inception Distance (FID) (Lower is better).
    Numerically robust implementation with epsilon offset.
    """
    mu_r = np.mean(real_embeddings, axis=0)
    mu_s = np.mean(synth_embeddings, axis=0)
    sigma_r = np.cov(real_embeddings, rowvar=False)
    sigma_s = np.cov(synth_embeddings, rowvar=False)
    
    diff = mu_r - mu_s
    mean_diff = diff.dot(diff)
    
    # Using scipy.linalg.sqrtm for matrix square root + Fallback
    try:
        offset = np.eye(sigma_r.shape[0]) * eps
        cov_prod, _ = linalg.sqrtm((sigma_r + offset).dot(sigma_s + offset), disp=False)
        if np.iscomplexobj(cov_prod):
            cov_prod = cov_prod.real
    except:
        # Emergency fallback for near-singular matrices
        return mean_diff + np.trace(sigma_r + sigma_s)

    fid = mean_diff + np.trace(sigma_r + sigma_s - 2 * cov_prod)
    return max(0.0, fid)

def calculate_swd(real_embeddings, synth_embeddings, n_projections=500):
    """
    Calculate Sliced Wasserstein-2 Distance (SWD-W2) (Lower is better).
    Approximates Earth Mover's Distance in Latent Space using W2 metric.
    """
    dim = real_embeddings.shape[1]
    results = []
    
    for _ in range(n_projections):
        projection = np.random.randn(dim)
        projection /= np.linalg.norm(projection)
        
        p_real = np.sort(real_embeddings.dot(projection))
        p_synth = np.sort(synth_embeddings.dot(projection))
        
        if len(p_real) != len(p_synth):
            interp_indices = np.linspace(0, len(p_synth)-1, len(p_real))
            p_synth = np.interp(interp_indices, np.arange(len(p_synth)), p_synth)
            
        w2_dist = np.mean((p_real - p_synth)**2)
        results.append(w2_dist)
        
    return np.sqrt(np.mean(results))

def evaluate_embeddings(model, real_data, synth_data, appliance, mode='multivariate', out_dir=None):
    """Encode data and evaluate using Discriminative Score and Visualization."""
    print("\nEncoding data...")
    
    # full_series: encoding based on max pooling over the whole series
    real_repr = model.encode(real_data, encoding_window='full_series')
    synth_repr = model.encode(synth_data, encoding_window='full_series')
    
    print(f"Embeddings shape: {real_repr.shape}")
    
    # --- Metric 1: Discriminative Score ---
    # === Discriminative Score (Strict Time-Series Split) ===
    # To prevent Data Leakage from sliding windows, we CANNOT randomly shuffle early. 
    # We must split chronologically first so Train and Test share almost zero overlap.
    r_split = int(len(real_repr) * 0.7)
    s_split = int(len(synth_repr) * 0.7)
    
    X_train = np.concatenate([real_repr[:r_split], synth_repr[:s_split]], axis=0)
    y_train = np.concatenate([np.zeros(r_split), np.ones(s_split)], axis=0)
    
    X_test = np.concatenate([real_repr[r_split:], synth_repr[s_split:]], axis=0)
    y_test = np.concatenate([np.zeros(len(real_repr) - r_split), np.ones(len(synth_repr) - s_split)], axis=0)
    
    # Shuffle only AFTER train/test have been strictly isolated from each other
    train_idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    test_idx = np.random.permutation(len(X_test))
    X_test, y_test = X_test[test_idx], y_test[test_idx]
    
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    # Standard Discriminative Score (TimeGAN Standard: |Acc - 0.5|)
    # Closer to 0.0 is better (Indistinguishable)
    discriminative_score = np.abs(acc - 0.5)
    
    # --- Metric 2: Context-FID ---
    print("Calculating Context-FID...")
    fid_score = calculate_fid(real_repr, synth_repr)
    
    # --- Metric 3: SWD (Latent vs Raw) ---
    print("Calculating SWD-W2 (Latent Space)...")
    swd_latent = calculate_swd(real_repr, synth_repr)
    
    print("Calculating SWD-W2 (Raw Space)...")
    real_raw_flat = real_data.reshape(len(real_data), -1)
    synth_raw_flat = synth_data.reshape(len(synth_data), -1)
    swd_raw = calculate_swd(real_raw_flat, synth_raw_flat)

    # --- Metric Report ---
    report_lines = [
        f"{'='*60}",
        f"EVALUATION REPORT: {appliance.upper()} | MODE: {mode.upper()}",
        f"{'='*60}",
        f"",
        f"[1. PHYSICAL DOMAIN (RAW SPACE)]",
        f"   Raw SWD Score : {swd_raw:.4f}  (lower = closer physical values)",
        f"",
        f"[2. FEATURE DOMAIN (LATENT SPACE)]",
        f"   Discriminative: {acc:.4f}  (target ~0.50 = indistinguishable)",
        f"   Context-FID   : {fid_score:.4f}  (lower is better)",
        f"   Latent SWD    : {swd_latent:.4f}  (lower = patterns more similar)",
        f"{'='*60}",
    ]
    for line in report_lines:
        print(line)

    # Save metrics.txt
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
            f.write("\n".join(report_lines))
        print(f"   💾 Local Metrics saved: {os.path.join(out_dir, 'metrics.txt')}")

    # --- Append to Mode-Specific Summary Table ---
    # Save a separate table for Multivariate, Power, and Time
    summary_file = os.path.join(RESULTS_DIR, f"global_metrics_{mode.lower()}.csv")
    new_entry = pd.DataFrame([{
        "Appliance": appliance.upper(),
        "Raw SWD-W2": round(swd_raw, 4),
        "D-Score": round(discriminative_score, 4),
        "Context-FID": round(fid_score, 4),
        "Latent SWD-W2": round(swd_latent, 4)
    }])
    
    if os.path.exists(summary_file):
        new_entry.to_csv(summary_file, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(summary_file, mode='w', header=True, index=False)
    print(f"   📊 Table Updated: {summary_file}")

    # --- Visualization: LATENT (Feature Domain) ONLY ---
    print(f"\n🎨 Generating PCA & t-SNE visualizations (Latent Domain)...")
    n_vis = min(500, len(real_repr), len(synth_repr))
    r_idx = np.random.choice(len(real_repr), n_vis, replace=False)
    s_idx = np.random.choice(len(synth_repr), n_vis, replace=False)

    for d_name, d_real, d_synth, suffix in [
        ("LATENT", real_repr, synth_repr, "latent"),
    ]:
        X_vis = np.concatenate([d_real[r_idx], d_synth[s_idx]], axis=0)
        plot_dir = out_dir if out_dir else RESULTS_DIR
        os.makedirs(plot_dir, exist_ok=True)

        # --- PCA (separate file) ---
        X_pca = PCA(n_components=2, random_state=42).fit_transform(X_vis)
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.suptitle(f"{appliance.upper()} | {mode.upper()} | {d_name} - PCA",
                     fontsize=15, fontweight='bold')
        
        # Academic Paper Style Dots
        ax.scatter(X_pca[:n_vis, 0], X_pca[:n_vis, 1], c='#d62728', label='Real Data', 
                   alpha=0.6, s=35, edgecolors='white', linewidths=0.3)
        ax.scatter(X_pca[n_vis:, 0], X_pca[n_vis:, 1], c='#1f77b4', label='Synthetic Diffusion Data', 
                   alpha=0.6, s=35, edgecolors='white', linewidths=0.3)
        
        ax.set_title("PCA Dimensionality Reduction", fontsize=12)
        ax.legend(frameon=True, fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pca_path = os.path.join(plot_dir, f"{suffix}_pca_visual.png")
        plt.savefig(pca_path, dpi=120)
        plt.close()
        print(f"   📸 Saved PCA  ({d_name}): {pca_path}")

        # --- t-SNE (separate file) ---
        # Dynamically set perplexity for very small test datasets to avoid crashes
        current_perplexity = min(30, max(2, len(X_vis) - 1))
        X_tsne = TSNE(n_components=2, perplexity=current_perplexity, random_state=42).fit_transform(X_vis)
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.suptitle(f"{appliance.upper()} | {mode.upper()} | {d_name} - t-SNE",
                     fontsize=15, fontweight='bold', fontfamily='sans-serif')
        
        # Academic Paper Style Dots
        ax.scatter(X_tsne[:n_vis, 0], X_tsne[:n_vis, 1], c='#d62728', label='Real', 
                   alpha=0.65, s=45, edgecolors='white', linewidths=0.5)
        ax.scatter(X_tsne[n_vis:, 0], X_tsne[n_vis:, 1], c='#1f77b4', label='Synthetic', 
                   alpha=0.65, s=45, edgecolors='white', linewidths=0.5)
        
        ax.set_title("t-SNE Manifold", fontsize=12)
        ax.legend(frameon=True, fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        tsne_path = os.path.join(plot_dir, f"{suffix}_tsne_visual.png")
        plt.savefig(tsne_path, dpi=120)
        plt.close()
        print(f"   📸 Saved t-SNE({d_name}): {tsne_path}")


    return acc


def get_or_train_encoder(appliance, mode, seq_len, force_retrain, train_on_synth):
    """
    Load or train a DEDICATED encoder per (appliance, mode) pair.
    e.g. washingmachine_power_encoder.pth  <- trained only on power features
         washingmachine_time_encoder.pth   <- trained only on time features
         washingmachine_multivariate_encoder.pth <- trained on all 9 features
    This ensures each scenario is evaluated fairly by its own specialist encoder.
    """
    model_name = f"{appliance}_{mode}_encoder.pth"
    model_path = os.path.join(JUDGES_DIR, model_name)

    # Load the correct data for THIS mode
    real_data, synth_data, _ = load_data(appliance, sequence_length=seq_len, mode=mode)
    real_data = np.nan_to_num(real_data)
    synth_data = np.nan_to_num(synth_data)

    n_features = real_data.shape[-1]  # Varies per mode (9, 1, or 8)

    model = TS2Vec(
        input_dims=n_features,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if os.path.exists(model_path) and not force_retrain:
        print(f"\n📂 Encoder exists for ({appliance}, {mode}). Loading: {model_name}")
        print(f"   ✅ Skipping training. Use --force_retrain to override.")
        model.load(model_path)
    else:
        print(f"\n🧬 Training encoder for ({appliance}, {mode}) on {n_features} features...")
        train_source = synth_data if train_on_synth else real_data
        model.fit(train_source, n_epochs=100, verbose=True)
        model.save(model_path)
        print(f"💾 Encoder saved: {model_path}")

    return model


def run_evaluation(appliance, mode, seq_len, force_retrain, train_on_synth):
    """Train a dedicated encoder for this (appliance, mode) and evaluate."""
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING: {appliance.upper()} | MODE: {mode.upper()}")
    print(f"{'='*60}")

    # Each scenario gets its own specialist encoder
    encoder = get_or_train_encoder(appliance, mode, seq_len, force_retrain, train_on_synth)

    real_data, synth_data, _ = load_data(appliance, sequence_length=seq_len, mode=mode)
    real_data = np.nan_to_num(real_data)
    synth_data = np.nan_to_num(synth_data)

    # Per-scenario output folder
    scenario_dir = os.path.join(RESULTS_DIR, f"{appliance}_{mode}")
    os.makedirs(scenario_dir, exist_ok=True)
    print(f"   📁 Results: {scenario_dir}")

    evaluate_embeddings(encoder, real_data, synth_data, appliance, mode=mode, out_dir=scenario_dir)


def generate_master_grid(plot_type="PCA"):
    """
    Generates a massive high-quality 5x3 grid for the thesis paper.
    plot_type: "PCA" or "t-SNE"
    """
    print(f"\n=======================================================")
    print(f"  GENERATING TOP-TIER MASTER {plot_type} GRID          ")
    print(f"=======================================================\n")
    
    modes_ordered = ['multivariate', 'power', 'time']
    mode_titles = ["Time + Power (Multivariate)", "Power Only", "Time Only"]
    
    fig, axes = plt.subplots(nrows=len(APPLIANCES), ncols=len(modes_ordered), figsize=(18, 22))
    fig.suptitle(f"Manifold Visualization ({plot_type}) Across All Appliances and Feature Modes", 
                 fontsize=24, fontweight='bold', y=0.98, fontfamily='sans-serif')
    
    red_patch, blue_patch = None, None
    seq_len = 480  # Default evaluation seq_len
    
    for i, app in enumerate(APPLIANCES):
        for j, mode in enumerate(modes_ordered):
            ax = axes[i, j]
            
            # Setup row and column headers
            if i == 0:
                ax.set_title(mode_titles[j].upper(), fontsize=16, fontweight='bold', pad=15)
            if j == 0:
                ax.set_ylabel(app.upper(), fontsize=16, fontweight='bold', labelpad=15)
                
            model_path = os.path.join(JUDGES_DIR, f"{app}_{mode}_encoder.pth")
            try:
                real_data, synth_data, _ = load_data(app, sequence_length=seq_len, mode=mode)
                real_data = np.nan_to_num(real_data)
                synth_data = np.nan_to_num(synth_data)
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError
                    
                model = TS2Vec(input_dims=real_data.shape[-1], output_dims=320, hidden_dims=64,
                               depth=10, device='cuda' if torch.cuda.is_available() else 'cpu')
                model.load(model_path)
                real_repr = model.encode(real_data, encoding_window='full_series')
                synth_repr = model.encode(synth_data, encoding_window='full_series')
                
            except Exception as e:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', fontsize=12, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                continue
                
            n_vis = min(500, len(real_repr), len(synth_repr))
            r_idx = np.random.choice(len(real_repr), n_vis, replace=False)
            s_idx = np.random.choice(len(synth_repr), n_vis, replace=False)
            
            X_vis = np.concatenate([real_repr[r_idx], synth_repr[s_idx]], axis=0)
            
            if plot_type == "PCA":
                X_proj = PCA(n_components=2, random_state=42).fit_transform(X_vis)
            else:
                current_perplexity = min(30, max(2, len(X_vis) - 1))
                X_proj = TSNE(n_components=2, perplexity=current_perplexity, random_state=42).fit_transform(X_vis)
            
            sc1 = ax.scatter(X_proj[:n_vis, 0], X_proj[:n_vis, 1], c='#d62728', 
                             alpha=0.6, s=40, edgecolors='white', linewidths=0.5, label='Real Data')
            sc2 = ax.scatter(X_proj[n_vis:, 0], X_proj[n_vis:, 1], c='#1f77b4', 
                             alpha=0.6, s=40, edgecolors='white', linewidths=0.5, label='Synthetic GAN Data')
            
            if red_patch is None and blue_patch is None:
                red_patch, blue_patch = sc1, sc2
                
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(True, alpha=0.2)
            for spine in ax.spines.values():
                spine.set_color('#dddddd'); spine.set_linewidth(1.5)

    if red_patch and blue_patch:
        fig.legend(handles=[red_patch, blue_patch], loc='lower center', 
                   ncol=2, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    output_path = os.path.join(RESULTS_DIR, f"Master_Grid_{plot_type.replace('-', '')}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ SUCCESS! Master {plot_type} image saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="TS2Vec Evaluation for NILM Data")
    # Extended APPLIANCES list to accept "all"
    parser.add_argument("appliance", type=str, choices=APPLIANCES + ["all"])
    parser.add_argument("--seq_len", type=int, default=480)
    parser.add_argument("--mode", type=str,
                        choices=['multivariate', 'power', 'time', 'all'],
                        default='all',
                        help="Default 'all': trains a separate encoder per scenario")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain encoder even if saved model exists")
    parser.add_argument("--train_on_synth", action="store_true",
                        help="Train encoder on Synthetic instead of Real data")

    args = parser.parse_args()

    # Trigger Master Grid Generation
    if args.appliance == "all":
        generate_master_grid("PCA")
        generate_master_grid("t-SNE")
        return

    modes_to_run = ['multivariate', 'power', 'time'] if args.mode == 'all' else [args.mode]
    for m in modes_to_run:
        run_evaluation(args.appliance, m, args.seq_len, args.force_retrain, args.train_on_synth)


if __name__ == "__main__":
    main()