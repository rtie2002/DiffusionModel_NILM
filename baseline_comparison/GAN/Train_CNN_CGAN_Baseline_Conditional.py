import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm

# ==========================================
# CONFIGURATION
# ==========================================
APPLIANCES = ["dishwasher", "washingmachine", "fridge", "kettle", "microwave"]
WINDOW_SIZE = 512
BATCH_SIZE = 128
EPOCHS_PER_APP = 2000
COND_DIM = 8

BASE_DIR = os.getcwd() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Project Root: {BASE_DIR}")
print(f"✅ Device: {device}")

# ==========================================
# MODEL DEFINITIONS (Conditional CNN-GAN)
# ==========================================
class Generator(nn.Module):
    def __init__(self, cond_dim=8):
        super().__init__()
        # Input: noise(100) + condition_at_first_step(8) = 108
        self.fc = nn.Linear(100 + cond_dim, 128 * 16)
        def up(ic, oc): return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(ic, oc, 3, 1, 1),
            nn.BatchNorm1d(oc),
            nn.ReLU(True))
        self.model = nn.Sequential(up(128,64), up(64,32), up(32,16), up(16,8),
                                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                                nn.Conv1d(8, 1, 3, 1, 1), nn.Tanh())
        
    def forward(self, z, c):
        # c: (bs, 512, 8). Use first step's condition as global window context
        c_global = c[:, 0, :] 
        gen_input = torch.cat([z, c_global], dim=1)
        return self.model(self.fc(gen_input).view(-1, 128, 16))

class Discriminator(nn.Module):
    def __init__(self, cond_dim=8):
        super().__init__()
        # Input: power(1) + time_features(8) = 9 channels
        def cb(ic, oc, s=2): return nn.Sequential(spectral_norm(nn.Conv1d(ic, oc, 4, s, 1)), nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(cb(1 + cond_dim, 16), cb(16, 32), cb(32, 64), cb(64, 128),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128,1), nn.Sigmoid())
        
    def forward(self, x, c):
        # x: (bs, 1, 512), c: (bs, 512, 8)
        c = c.permute(0, 2, 1) # (bs, 8, 512)
        disc_input = torch.cat([x, c], dim=1) # (bs, 9, 512)
        return self.conv(disc_input)

class NILM_Dataset(Dataset):
    def __init__(self, p, t):
        self.data = []
        stride = WINDOW_SIZE // 2 # 50% overlap for better continuity learning
        for i in range(0, len(p) - WINDOW_SIZE, stride):
            # NO FILTER: Include all windows (Active and Inactive) to learn real distribution
            self.data.append((p[i:i+WINDOW_SIZE], t[i:i+WINDOW_SIZE]))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        p, t = self.data[idx]
        return torch.from_numpy(p).float().unsqueeze(0), torch.from_numpy(t).float()

# ==========================================
# MAIN EXECUTION
# ==========================================
def train_appliance(appliance):
    print(f'\n{"="*60}\nPROCESSING: {appliance.upper()}\n{"="*60}')
    
    CSV_PATH = os.path.join(BASE_DIR, 'Data', 'datasets', f'{appliance}_multivariate.csv')
    OUT_DIR = os.path.join(BASE_DIR, 'Synthetic_data', f'cgan_{appliance}') 
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f'⚠️ Skipping {appliance}: CSV not found')
        return

    # Data Prep
    print(f'Loading data for {appliance}...')
    df = pd.read_csv(CSV_PATH)
    power_col = [c for c in df.columns if 'power' in c.lower() or appliance in c.lower()][0]
    p_min, p_max = df[power_col].min(), df[power_col].max()
    raw_p_norm = (df[power_col].values - p_min) / (p_max - p_min) * 2 - 1
    
    time_cols = [c for c in df.columns if c != power_col]
    time_features = df[time_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    dataset = NILM_Dataset(raw_p_norm, time_features)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    if len(dataset) == 0:
        print(f'Warning: No valid windows found for {appliance}')
        return

    G, D = Generator(COND_DIM).to(device), Discriminator(COND_DIM).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Training
    print(f'🔥 Training for {EPOCHS_PER_APP} epochs...')
    for epoch in range(1, EPOCHS_PER_APP + 1):
        for i, (real_p, real_t) in enumerate(train_loader):
            real_p, real_t = real_p.to(device), real_t.to(device)
            bs = real_p.size(0)

            # Train Discriminator
            opt_D.zero_grad()
            z = torch.randn(bs, 100).to(device)
            fake_p = G(z, real_t)
            
            # Discriminator loss: real/real_t vs fake/real_t
            loss_d = (criterion(D(real_p, real_t), torch.full((bs,1), 0.9).to(device)) + 
                      criterion(D(fake_p.detach(), real_t), torch.zeros(bs,1).to(device))) / 2
            loss_d.backward(); opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            fake_p = G(torch.randn(bs, 100).to(device), real_t)
            # Adversarial + Continuity Penalty (0.2 * TV loss)
            loss_g = criterion(D(fake_p, real_t), torch.ones(bs,1).to(device)) + \
                     0.2 * torch.mean(torch.abs(fake_p[:, :, 1:] - fake_p[:, :, :-1]))
            loss_g.backward(); opt_G.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{EPOCHS_PER_APP}] | Loss_D: {loss_d.item():.4f} | Loss_G: {loss_g.item():.4f}')

    # Sampling 
    print(f'Generating Conditional Synthetic data...')
    G.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        # Generate double the amount of source data
        for _ in range((len(dataset)*2) // BATCH_SIZE + 1):
            idx = np.random.choice(len(dataset), BATCH_SIZE)
            batch_t = torch.stack([dataset[i][1] for i in idx]).to(device)
            z = torch.randn(BATCH_SIZE, 100).to(device)
            p = (G(z, batch_t).cpu().numpy() + 1) / 2
            all_p.append(p)
            all_t.append(batch_t.cpu().numpy())

    final_p = np.concatenate(all_p, axis=0)[:len(dataset)*2]
    final_t = np.concatenate(all_t, axis=0)[:len(dataset)*2]
    # Shape: [N, 512, 1] + [N, 512, 8] -> [N, 512, 9]
    final_merged = np.concatenate([np.expand_dims(final_p.squeeze(1), axis=2), final_t], axis=2)

    np_path = os.path.join(OUT_DIR, f'synthetic_{appliance}.npy')
    np.save(np_path, final_merged)
    print(f'⭐ SUCCESS: {appliance.upper()} saved to {np_path}')

if __name__ == "__main__":
    for appliance in APPLIANCES:
        train_appliance(appliance)
