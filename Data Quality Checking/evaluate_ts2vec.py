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
    """Load and preprocess real and synthetic data."""
    print(f"Loading data for {appliance} (Mode: {mode})...")
    
    real_path = os.path.join(REAL_DATA_DIR, f"{appliance}_multivariate.csv")
    synth_path = os.path.join(SYNTHETIC_DATA_DIR, f"{appliance}_multivariate.csv")
    
    if not os.path.exists(real_path) or not os.path.exists(synth_path):
        raise FileNotFoundError(f"Data files for {appliance} not found.")
        
    df_real = pd.read_csv(real_path)
    df_synth = pd.read_csv(synth_path)

    # Column Filtering based on Mode
    if mode == 'power':
        # Keep only the appliance power column (usually the first one)
        power_col = [col for col in df_real.columns if col.lower() == appliance.lower() or col == df_real.columns[0]][0]
        df_real = df_real[[power_col]]
        df_synth = df_synth[[power_col]]
    elif mode == 'time':
        # Keep only time-related features (sin/cos columns)
        time_cols = [col for col in df_real.columns if 'sin' in col or 'cos' in col or 'hour' in col or 'minute' in col]
        df_real = df_real[time_cols]
        df_synth = df_synth[time_cols]
    
    def df_to_windows(df, seq_len, limit=None):
        data = df.values
        
        # SLIDING WINDOW STRATEGY:
        # Instead of cutting data into strict, non-overlapping blocks (which produces very few dots), 
        # we slide the window by a small stride. This captures the continuous "trajectory" 
        # of the appliance and produces a rich, dense manifold graph even from just 3000 rows.
        stride = 10
        
        if len(data) < seq_len:
            print(f"   ⚠️ WARNING: Data length ({len(data)}) is less than sequence length ({seq_len}). Padding...")
            pad_size = seq_len - len(data)
            data = np.pad(data, ((0, pad_size), (0, 0)))

        num_windows = (len(data) - seq_len) // stride + 1
        
        # Create overlapping windows
        windows = np.array([data[i * stride : i * stride + seq_len] for i in range(num_windows)])
        
        if limit and len(windows) > limit:
            indices = np.random.choice(len(windows), limit, replace=False)
            windows = windows[indices]
            
        return windows, df.columns.tolist()

    real_windows, cols = df_to_windows(df_real, sequence_length, max_samples)
    synth_windows, _ = df_to_windows(df_synth, sequence_length, max_samples)
    
    print(f"Features: {cols}")
    print(f"Real Samples: {real_windows.shape}, Synthetic Samples: {synth_windows.shape}")
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

def calculate_fid(real_embeddings, synth_embeddings):
    """
    Calculate the Fréchet Inception Distance (FID) between two distributions of embeddings.
    Formula: FID = ||mu_r - mu_s||^2 + Tr(Sigma_r + Sigma_s - 2*sqrt(Sigma_r * Sigma_s))
    """
    mu_r = np.mean(real_embeddings, axis=0)
    mu_s = np.mean(synth_embeddings, axis=0)
    
    sigma_r = np.cov(real_embeddings, rowvar=False)
    sigma_s = np.cov(synth_embeddings, rowvar=False)
    
    # Calculate the squared difference of means
    diff = mu_r - mu_s
    mean_diff = diff.dot(diff)
    
    # Calculate the product of covariances and its square root
    # Using scipy.linalg.sqrtm for matrix square root
    cov_prod, _ = linalg.sqrtm(sigma_r.dot(sigma_s), disp=False)
    
    # Handle numerical errors (complex numbers can appear if values are near zero)
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
        
    fid = mean_diff + np.trace(sigma_r + sigma_s - 2 * cov_prod)
    return fid

def calculate_swd(real_embeddings, synth_embeddings, n_projections=200):
    """
    Calculate Sliced Wasserstein Distance (SWD) between two distributions.
    Efficiently approximates the Wasserstein distance by projecting into random 1D lines.
    """
    dim = real_embeddings.shape[1]
    results = []
    
    for _ in range(n_projections):
        # Generate a random direction on the unit sphere
        projection = np.random.randn(dim)
        projection /= np.linalg.norm(projection)
        
        # Project data onto this direction
        p_real = real_embeddings.dot(projection)
        p_synth = synth_embeddings.dot(projection)
        
        # Calculate 1D Wasserstein distance (sort and compute mean absolute diff)
        p_real_sorted = np.sort(p_real)
        p_synth_sorted = np.sort(p_synth)
        
        # If sample sizes differ, we interpolate to match count
        if len(p_real_sorted) != len(p_synth_sorted):
            # Linearly interpolate to the size of real data for comparison
            # In NILM eval, they are often similar max_samples, but this is safer
            interp_indices = np.linspace(0, len(p_synth_sorted)-1, len(p_real_sorted))
            p_synth_resampled = np.interp(interp_indices, np.arange(len(p_synth_sorted)), p_synth_sorted)
            wd = np.mean(np.abs(p_real_sorted - p_synth_resampled))
        else:
            wd = np.mean(np.abs(p_real_sorted - p_synth_sorted))
            
        results.append(wd)
        
    return np.mean(results)

def evaluate_embeddings(model, real_data, synth_data, appliance, mode='multivariate', out_dir=None):
    """Encode data and evaluate using Discriminative Score and Visualization."""
    print("\nEncoding data...")
    
    # full_series: encoding based on max pooling over the whole series
    real_repr = model.encode(real_data, encoding_window='full_series')
    synth_repr = model.encode(synth_data, encoding_window='full_series')
    
    print(f"Embeddings shape: {real_repr.shape}")
    
    # --- Metric 1: Discriminative Score ---
    # Label: 0 for Real, 1 for Synthetic
    X = np.concatenate([real_repr, synth_repr], axis=0)
    y = np.concatenate([np.zeros(len(real_repr)), np.ones(len(synth_repr))], axis=0)
    
    # Shuffle and split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    
    # Train classifier (Logistic Regression)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    # Ideal accuracy is 0.5 (random guess), meaning arrays are indistinguishable
    # High accuracy (~1.0) means they are easily distinguishable (bad for synthesis)
    # --- Metric 2: Context-FID ---
    print("Calculating Context-FID...")
    fid_score = calculate_fid(real_repr, synth_repr)
    
    # --- Metric 3: SWD (Latent vs Raw) ---
    print("Calculating SWD (Latent Space)...")
    swd_latent = calculate_swd(real_repr, synth_repr)
    
    print("Calculating SWD (Raw Space)...")
    # Reshape (N, L, C) -> (N, L*C) to treat entire window as a flattened feature vector
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
        print(f"   💾 Metrics saved: {os.path.join(out_dir, 'metrics.txt')}")

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
                     fontsize=15, fontweight='bold', fontfamily='sans-serif')
        
        # Academic Paper Style Dots
        ax.scatter(X_pca[:n_vis, 0], X_pca[:n_vis, 1], c='#d62728', label='Real', 
                   alpha=0.65, s=45, edgecolors='white', linewidths=0.5)
        ax.scatter(X_pca[n_vis:, 0], X_pca[n_vis:, 1], c='#1f77b4', label='Synthetic', 
                   alpha=0.65, s=45, edgecolors='white', linewidths=0.5)
        
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
        
        ax.set_title(f"t-SNE Manifold | Discriminative Score: {acc:.3f}", fontsize=12)
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


def main():
    parser = argparse.ArgumentParser(description="TS2Vec Evaluation for NILM Data")
    parser.add_argument("appliance", type=str, choices=APPLIANCES)
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

    modes_to_run = ['multivariate', 'power', 'time'] if args.mode == 'all' else [args.mode]
    for m in modes_to_run:
        run_evaluation(args.appliance, m, args.seq_len, args.force_retrain, args.train_on_synth)


if __name__ == "__main__":
    main()