import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy import linalg
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import TS2Vec
try:
    from Models.ts2vec.ts2vec import TS2Vec
except ImportError:
    print("Warning: Could not import TS2Vec. FID will be skipped.")
    TS2Vec = None

# Import Discriminative Metrics assuming they exist in typical TimeGAN folder
# If not, we will implement a simple one here.
try:
    from expample_project.TimeGAN_master.metrics.discriminative_metrics import discriminative_score_metrics
except:
    # Fallback: Simple RNN Discriminator Implementation
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    def discriminative_score_metrics(real_data, synthetic_data):
        print("Running internal Discriminative Score...")
        # Basic GRU Classifier: Real=1, Fake=0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Prepare Data
        X = np.concatenate([real_data, synthetic_data], axis=0)
        y = np.concatenate([np.ones(len(real_data)), np.zeros(len(synthetic_data))], axis=0)
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        
        # Split Train/Test
        split = int(len(X) * 0.8)
        X_train, y_train = torch.FloatTensor(X[:split]).to(device), torch.FloatTensor(y[:split]).to(device)
        X_test, y_test = torch.FloatTensor(X[split:]).to(device), torch.FloatTensor(y[split:]).to(device)
        
        class Discriminator(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.gru = nn.GRU(input_dim, 64, batch_first=True)
                self.fc = nn.Linear(64, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                _, h = self.gru(x)
                return self.sigmoid(self.fc(h[-1]))
        
        model = Discriminator(X.shape[2]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Train
        epochs = 10 # Short training
        batch_size = 128
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test).squeeze()
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean().item()
            
        # Metric: |0.5 - Accuracy|
        # Ideally Accuracy is 0.5 (indistinguishable), so metric is 0.
        return np.abs(0.5 - accuracy)

def calculate_fid(real_emb, fake_emb):
    # Calculate Frechet Distance between two multivariate Gaussians
    mu1, sigma1 = real_emb.mean(axis=0), np.cov(real_emb, rowvar=False)
    mu2, sigma2 = fake_emb.mean(axis=0), np.cov(fake_emb, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def load_data(appliance, window_size=512):
    # 1. Load Real Data (CSV)
    real_path = fr'C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM\Data\datasets\{appliance}_multivariate.csv'
    if not os.path.exists(real_path):
        print(f"Error: Real data not found at {real_path}")
        return None, None
        
    df = pd.read_csv(real_path)
    # Find power col
    col = next((c for c in df.columns if 'power' in c.lower()), df.columns[-1])
    real_vals = df[col].values[:10000] # Limit size for speed
    
    # Create windows
    real_windows = []
    for i in range(0, len(real_vals)-window_size, window_size):
        real_windows.append(real_vals[i:i+window_size])
    real_data = np.array(real_windows)
    real_data = np.expand_dims(real_data, axis=2) # (N, L, 1)
    
    # 2. Load Synthetic Data (NPY)
    # Adjust this path to where your generator saves data
    syn_path = fr'C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM\Synthetic_Data\{appliance}\synthetic_{appliance}.npy'
    if os.path.exists(syn_path):
        syn_data = np.load(syn_path)
        # Ensure shape (N, L, 1)
        if syn_data.ndim == 2: syn_data = np.expand_dims(syn_data, axis=2)
        if syn_data.shape[1] != window_size:
            print(f"Warning: Synthetic data length {syn_data.shape[1]} != {window_size}. Truncating/Padding.")
            syn_data = syn_data[:, :window_size, :]
    else:
        print(f"Warning: Synthentic data not found at {syn_path}. Using Random Noise for Demo.")
        syn_data = np.random.rand(len(real_data), window_size, 1)
        
    return real_data, syn_data

def main():
    appliance = 'dishwasher'
    print(f"Evaluating appliance: {appliance}")
    
    # 1. Load Data
    real_data, syn_data = load_data(appliance)
    if real_data is None: return

    print(f"Real Data Shape: {real_data.shape}")
    print(f"Syn Data Shape:  {syn_data.shape}")

    # 2. Discriminative Score
    disc_score = discriminative_score_metrics(real_data, syn_data)
    print(f"\n>>> Discriminative Score: {disc_score:.4f} (Lower is better, ideal 0.0)")

    # 3. Context-FID (TS2Vec)
    if TS2Vec is not None:
        print("\n>>> Calculating Context-FID with TS2Vec...")
        # Initialize TS2Vec (Unsupervised training on Real Data to learn features)
        # Or load a pretrained one if you have it.
        # Here we train briefly or just init to extract features if weights specificied
        
        model = TS2Vec(
            input_dims=1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            output_dims=320
        )
        
        # Ideally, we should train TS2Vec on Real Data first to get meaningful embeddings
        # print("Training TS2Vec on Real Data (Briefly)...")
        # model.fit(real_data, verbose=True, n_epochs=5) 
        
        # For now, let's assume initialized weights form a random projection (better than nothing)
        # Or if you have a path: model.load(path)
        
        real_emb = model.encode(real_data) # (N, Hidden)
        syn_emb = model.encode(syn_data)   # (N, Hidden)
        
        fid = calculate_fid(real_emb, syn_emb)
        print(f">>> Context-FID: {fid:.4f} (Lower is better)")
    
if __name__ == "__main__":
    main()
