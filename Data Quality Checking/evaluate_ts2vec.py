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
SYNTHETIC_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "datasets", "synthetic_processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Data Quality Checking", "ts2vec_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]

def load_data(appliance, sequence_length=480, max_samples=2000):
    """Load and preprocess real and synthetic data."""
    print(f"Loading data for {appliance}...")
    
    real_path = os.path.join(REAL_DATA_DIR, f"{appliance}_multivariate.csv")
    synth_path = os.path.join(SYNTHETIC_DATA_DIR, f"{appliance}_multivariate.csv")
    
    if not os.path.exists(real_path) or not os.path.exists(synth_path):
        raise FileNotFoundError(f"Data files for {appliance} not found.")
        
    df_real = pd.read_csv(real_path)
    df_synth = pd.read_csv(synth_path)
    
    # Preprocess: Segment into windows
    # We treat the continuous stream as a collection of sliding windows 
    # or non-overlapping windows for training
    
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
    loss_log = model.fit(train_data, n_epochs=20, verbose=True)
    return model, loss_log

def evaluate_embeddings(model, real_data, synth_data, appliance):
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
    print(f"\nDiscriminative Score (Accuracy of Real vs Synth): {acc:.4f}")
    print(f"  Note: 0.5 = Perfect (Indistinguishable), 1.0 = Bad (Distinct)")
    
    # --- Metric 2: Visualization (PCA/t-SNE) ---
    print("Generating visualizations...")
    
    # PCA first to reduce dim if needed, or straight to t-SNE
    # Combined for fit
    n_vis_samples = min(500, len(real_repr))
    
    # Subsample for visualization clarity
    real_idx = np.random.choice(len(real_repr), n_vis_samples, replace=False)
    synth_idx = np.random.choice(len(synth_repr), n_vis_samples, replace=False)
    
    vis_real = real_repr[real_idx]
    vis_synth = synth_repr[synth_idx]
    
    vis_X = np.concatenate([vis_real, vis_synth], axis=0)
    vis_labels = ['Real'] * len(vis_real) + ['Synthetic'] * len(vis_synth)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(vis_X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:len(vis_real), 0], X_tsne[:len(vis_real), 1], 
                c='red', label='Real', alpha=0.5, s=20)
    plt.scatter(X_tsne[len(vis_real):, 0], X_tsne[len(vis_real):, 1], 
                c='blue', label='Synthetic', alpha=0.5, s=20)
    
    plt.title(f"t-SNE of TS2Vec Embeddings: {appliance.capitalize()}\nDiscriminative Score: {acc:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(RESULTS_DIR, f"{appliance}_ts2vec_tsne.png")
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to: {save_path}")
    # plt.show() # Uncomment if running interactively

    return acc

def main():
    parser = argparse.ArgumentParser(description="TS2Vec Evaluation for NILM Data")
    parser.add_argument("appliance", type=str, choices=APPLIANCES, help="Appliance to evaluate")
    parser.add_argument("--seq_len", type=int, default=120, help="Sequence length for windows (default: 120)")
    parser.add_argument("--train_on_synth", action="store_true", help="If set, train encoder on Synthetic data instead of Real (not recommended for standard eval)")
    
    args = parser.parse_args()
    
    # 1. Load Data
    real_data, synth_data, cols = load_data(args.appliance, sequence_length=args.seq_len)
    
    # Handle NaNs if any (simple fill)
    real_data = np.nan_to_num(real_data)
    synth_data = np.nan_to_num(synth_data)
    
    # 2. Train TS2Vec
    # Typically we train on Real data to verify if Synthetic data maps to the same manifold
    train_source = synth_data if args.train_on_synth else real_data
    
    model, _ = train_ts2vec(train_source, input_dims=real_data.shape[-1])
    
    # 3. Evaluate
    evaluate_embeddings(model, real_data, synth_data, args.appliance)

if __name__ == "__main__":
    main()