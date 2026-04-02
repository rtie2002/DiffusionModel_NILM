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
EPOCHS_PER_APP = 20000
COND_DIM = 8

BASE_DIR = os.getcwd() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Project Root: {BASE_DIR}")
print(f"✅ Device: {device}")

# ==========================================
# MODEL DEFINITIONS (Conditional Pixel-wise CNN-GAN)
# ==========================================
class Generator(nn.Module):
    def __init__(self, cond_dim=8):
        super().__init__()
        self.fc = nn.Linear(100, 128 * 16)
        def up(ic, oc): return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(ic, oc, 3, 1, 1),
            nn.BatchNorm1d(oc),
            nn.LeakyReLU(0.2, inplace=True))
            
        # Concat condition at every resolution to force temporal alignment
        self.u1 = up(128 + cond_dim, 64)
        self.u2 = up(64 + cond_dim, 32)
        self.u3 = up(32 + cond_dim, 16)
        self.u4 = up(16 + cond_dim, 8)
        self.final_conv = nn.Conv1d(8 + cond_dim, 1, 3, 1, 1)
        
    def forward(self, z, c):
        # c: (bs, 512, 8). Per-step condition.
        x = self.fc(z).view(-1, 128, 16)
        c_p = c.permute(0, 2, 1) # (bs, 8, 512)
        
        def get_c(res): return nn.functional.interpolate(c_p, size=res, mode='nearest')

        x = self.u1(torch.cat([x, get_c(16)], dim=1)) # 32
        x = self.u2(torch.cat([x, get_c(32)], dim=1)) # 64
        x = self.u3(torch.cat([x, get_c(64)], dim=1)) # 128
        x = self.u4(torch.cat([x, get_c(128)], dim=1)) # 256
        x = nn.functional.interpolate(x, size=512, mode='nearest')
        return torch.tanh(self.final_conv(torch.cat([x, get_c(512)], dim=1)))

class Discriminator(nn.Module):
    def __init__(self, cond_dim=8):
        super().__init__()
        def cb(ic, oc): return nn.Sequential(
            spectral_norm(nn.Conv1d(ic, oc, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2))
        self.conv = nn.Sequential(
            cb(1 + cond_dim, 32), cb(32, 64), cb(64, 128), cb(128, 256),
            nn.Conv1d(256, 1, 32, 1, 0), nn.Sigmoid())
            
    def forward(self, x, c):
        disc_input = torch.cat([x, c.permute(0, 2, 1)], dim=1)
        return self.conv(disc_input).view(-1, 1)

class NILM_Dataset(Dataset):
    def __init__(self, p, t):
        self.data = []
        # VANILLA SAMPLING: NO BOOSTING. Just keep the natural distribution.
        stride = WINDOW_SIZE // 2
        for i in range(0, len(p) - WINDOW_SIZE, stride):
            self.data.append((p[i:i+WINDOW_SIZE], t[i:i+WINDOW_SIZE]))
        print(f"📊 Dataset Loaded: {len(self.data)} windows (Natural Distribution)")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        p, t = self.data[idx]
        return torch.from_numpy(p).float().unsqueeze(0), torch.from_numpy(t).float()

# ==========================================
# MAIN EXECUTION
# ==========================================
def train_appliance(appliance):
    CSV_PATH = os.path.join(BASE_DIR, 'baseline_comparison', 'data', f'{appliance}_multivariate.csv')
    OUT_DIR = os.path.join(BASE_DIR, 'Synthetic_data', f'cgan_{appliance}') 
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f'⚠️ Skipping {appliance}: CSV not found')
        return

    print(f'Loading data for {appliance}...')
    df = pd.read_csv(CSV_PATH)
    power_col = 'dishwasher' if 'dishwasher' in df.columns else df.columns[0]
    time_cols = [c for c in df.columns if any(k in c for k in ['sin', 'cos'])]
    
    p_max, p_min = df[power_col].max(), df[power_col].min()
    raw_p_norm = (df[power_col].values - p_min) / (p_max - p_min + 1e-8) * 2 - 1
    time_features = df[time_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    dataset = NILM_Dataset(raw_p_norm, time_features)
    current_batch_size = min(BATCH_SIZE, len(dataset))
    train_loader = DataLoader(dataset, batch_size=current_batch_size, shuffle=True, drop_last=True)
    if len(train_loader) == 0: return

    G, D = Generator(COND_DIM).to(device), Discriminator(COND_DIM).to(device)
    def weights_init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)): nn.init.kaiming_normal_(m.weight)
    G.apply(weights_init); D.apply(weights_init)

    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    print(f'🔥 Training for {EPOCHS_PER_APP} epochs...')
    loss_d, loss_g = torch.tensor(0.0), torch.tensor(0.0)

    for epoch in range(1, EPOCHS_PER_APP + 1):
        G.train()
        for i, (real_p, real_t) in enumerate(train_loader):
            real_p, real_t = real_p.to(device), real_t.to(device)
            bs = real_p.size(0)

            # Train D
            opt_D.zero_grad()
            fake_p = G(torch.randn(bs, 100).to(device), real_t)
            loss_d = (criterion(D(real_p, real_t), torch.full((bs,1), 0.9).to(device)) + 
                      criterion(D(fake_p.detach(), real_t), torch.zeros(bs,1).to(device))) / 2
            loss_d.backward(); opt_D.step()

            # Train G (2x)
            for _ in range(2):
                opt_G.zero_grad()
                fake_p = G(torch.randn(bs, 100).to(device), real_t)
                loss_g = criterion(D(fake_p, real_t), torch.ones(bs,1).to(device))
                loss_g.backward(); opt_G.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{EPOCHS_PER_APP}] | Loss_D: {loss_d.item():.4f} | Loss_G: {loss_g.item():.4f}')
            G.eval(); prog_path = os.path.join(BASE_DIR, 'baseline_comparison', 'GAN', 'training_progress.png')
            with torch.no_grad():
                # Pick an active window for plotting
                sample_p_ref, sample_t = dataset[0][0], dataset[0][1]
                for k in range(min(100, len(dataset))):
                    if torch.max(dataset[k][0]) > -0.9:
                        sample_p_ref, sample_t = dataset[k][0], dataset[k][1]; break
                p_gen = G(torch.randn(1, 100).to(device), sample_t.unsqueeze(0).to(device)).cpu().numpy()[0,0]
                plt.clf(); plt.plot((sample_p_ref[0]+1)/2, label='Real'); plt.plot((p_gen+1)/2, label='Fake')
                plt.title(f'Epoch {epoch}'); plt.legend(); plt.savefig(prog_path); plt.close()

    # Sampling 1:1 ratio using stored p_max/p_min for exact amplitude matching
    print(f'Generating Conditional Synthetic data (1:1 Ratio)...')
    G.eval(); all_p, all_t = [], []
    with torch.no_grad():
        num_windows = len(dataset)
        for _ in range(num_windows // BATCH_SIZE + 1):
            idx = np.random.choice(num_windows, BATCH_SIZE)
            batch_t = torch.stack([dataset[i][1] for i in idx]).to(device)
            # ⚡ INVERSE NORMALIZATION: (G + 1) / 2 * range + min
            p_raw = (G(torch.randn(BATCH_SIZE, 100).to(device), batch_t).cpu().numpy() + 1) / 2
            p_denorm = p_raw * (p_max - p_min + 1e-8) + p_min
            all_p.append(p_denorm); all_t.append(batch_t.cpu().numpy())
    final_p = np.concatenate(all_p, axis=0)[:num_windows]
    final_t = np.concatenate(all_t, axis=0)[:num_windows]
    final_merged = np.concatenate([np.expand_dims(final_p.squeeze(1), axis=2), final_t], axis=2)
    np_path = os.path.join(OUT_DIR, f'synthetic_{appliance}.npy')
    np.save(np_path, final_merged)
    print(f'Done! Saved to {np_path}')

if __name__ == "__main__":
    for app in APPLIANCES: train_appliance(app)
