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

def load_data(appliance, sequence_length=480, max_samples=20000, mode='multivariate'):
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
        # Simple non-overlapping windows for efficiency in evaluation
        num_windows = len(data) // seq_len
        windows = data[:num_windows*seq_len].reshape(num_windows, seq_len, -1)
        
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

def evaluate_embeddings(model, real_data, synth_data, appliance, mode='multivariate'):
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
    
    # --- Metric 3: SWD ---
    print("Calculating SWD...")
    swd_score = calculate_swd(real_repr, synth_repr)
    
    print(f"\nMetric Report for {appliance.upper()}:")
    print(f"----------------------------------------")
    print(f"1. Discriminative Score: {acc:.4f} (Target: ~0.50)")
    print(f"2. Context-FID Score   : {fid_score:.4f} (Lower is better)")
    print(f"3. SWD Score           : {swd_score:.4f} (Lower is better)")
    print(f"----------------------------------------")
    print(f"  Note: 0.5 = Perfect Dist., FID < 1.0 is considered excellent for NILM")
    
    # --- Metric 2: Visualization (PCA & t-SNE) ---
    print(f"Generating visualizations (Mode: {mode})...")
    
    # Subsample for visualization clarity
    n_vis_samples = min(500, len(real_repr))
    real_idx = np.random.choice(len(real_repr), n_vis_samples, replace=False)
    synth_idx = np.random.choice(len(synth_repr), n_vis_samples, replace=False)
    
    vis_real = real_repr[real_idx]
    vis_synth = synth_repr[synth_idx]
    
    vis_X = np.concatenate([vis_real, vis_synth], axis=0)
    
    # 1. PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(vis_X)
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(vis_X)
    
    # Plotting both side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # PCA Plot
    ax1.scatter(X_pca[:len(vis_real), 0], X_pca[:len(vis_real), 1], 
                c='red', label='Real', alpha=0.5, s=20)
    ax1.scatter(X_pca[len(vis_real):, 0], X_pca[len(vis_real):, 1], 
                c='blue', label='Synthetic', alpha=0.5, s=20)
    ax1.set_title(f"PCA of TS2Vec Embeddings ({mode.capitalize()})\n{appliance.capitalize()}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE Plot
    ax2.scatter(X_tsne[:len(vis_real), 0], X_tsne[:len(vis_real), 1], 
                c='red', label='Real', alpha=0.5, s=20)
    ax2.scatter(X_tsne[len(vis_real):, 0], X_tsne[len(vis_real):, 1], 
                c='blue', label='Synthetic', alpha=0.5, s=20)
    ax2.set_title(f"t-SNE of TS2Vec Embeddings ({mode.capitalize()})\nDisc. Score: {acc:.4f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"{appliance}_{mode}_ts2vec_visual.png")
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to: {save_path}")

    return acc

def main():
    parser = argparse.ArgumentParser(description="TS2Vec Evaluation for NILM Data")
    parser.add_argument("appliance", type=str, choices=APPLIANCES, help="Appliance to evaluate")
    parser.add_argument("--seq_len", type=int, default=120, help="Sequence length for windows (default: 120)")
    parser.add_argument("--train_on_synth", action="store_true", help="If set, train encoder on Synthetic data instead of Real")
    parser.add_argument("--mode", type=str, choices=['multivariate', 'power', 'time'], default='multivariate', 
                        help="Evaluation mode: multivariate (all), power (only power), time (only sin/cos)")
    
    args = parser.parse_args()
    
    # 1. Load Data
    real_data, synth_data, cols = load_data(args.appliance, sequence_length=args.seq_len, mode=args.mode)
    
    # Handle NaNs if any (simple fill)
    real_data = np.nan_to_num(real_data)
    synth_data = np.nan_to_num(synth_data)
    
    # 2. Train TS2Vec
    # Typically we train on Real data to verify if Synthetic data maps to the same manifold
    train_source = synth_data if args.train_on_synth else real_data
    
    model, _ = train_ts2vec(train_source, input_dims=real_data.shape[-1])
    
    # 3. Evaluate
    evaluate_embeddings(model, real_data, synth_data, args.appliance, mode=args.mode)

if __name__ == "__main__":
    main()