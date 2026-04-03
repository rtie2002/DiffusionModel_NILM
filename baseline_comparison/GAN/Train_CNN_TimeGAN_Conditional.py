import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils import spectral_norm

# ==========================================
# CONFIGURATION
# ==========================================
APPLIANCES  = ["dishwasher", "washingmachine", "fridge", "kettle", "microwave"]
WINDOW_SIZE = 512
BATCH_SIZE  = 128
COND_DIM    = 8
HIDDEN_DIM  = 128

AE_ITER    = 10000
SUP_ITER   = 10000
JOINT_ITER = 20000

# Weights
ETA    = 5.0         # Regional activity weight
LAMBDA = 1.0
GAMMA  = 1.0
FOCAL  = 20.0        # Sharpness penalty

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DATASET with ACTIVE OVERSAMPLING
# ==========================================
class NILM_Dataset(Dataset):
    def __init__(self, p, t):
        self.data = []
        self.weights = []
        stride = WINDOW_SIZE // 2
        for i in range(0, len(p) - WINDOW_SIZE, stride):
            window_p = p[i:i + WINDOW_SIZE]
            window_t = t[i:i + WINDOW_SIZE]
            self.data.append((window_p, window_t))
            
            # ── Weighted Sampling Logic ──────────────────────────────
            # If the window has significant power activity (> -0.9 in [-1,1] range),
            # give it a high weight so the model SEES waves more often.
            if np.max(window_p) > -0.9: 
                self.weights.append(10.0) # 10x more likely to be sampled
            else:
                self.weights.append(1.0)
                
        print(f"📊 Dataset: {len(self.data)} windows | Weighted Active Sampling ENABLED")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        p, t = self.data[idx]
        return torch.from_numpy(p).float().unsqueeze(0), torch.from_numpy(t).float()

# ==========================================
# MODELS (CNN TimeGAN)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim))
    def forward(self, x): return x + self.block(x)

class Embedder(nn.Module):
    def __init__(self, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        cb = lambda ic, oc: nn.Sequential(
            nn.Conv1d(ic, oc, 4, 2, 1),
            nn.BatchNorm1d(oc),
            nn.LeakyReLU(0.2, inplace=True))
        self.encoder = nn.Sequential(
            cb(1 + cond_dim, 32), cb(32, 64), cb(64, 128), cb(128, hidden_dim))
    def forward(self, x, c):
        return self.encoder(torch.cat([x, c.permute(0, 2, 1)], dim=1))

class Recovery(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        def ub(ic, oc):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(ic, oc, 3, 1, 1),
                nn.BatchNorm1d(oc),
                nn.LeakyReLU(0.2, inplace=True))
        self.decoder = nn.Sequential(
            ub(hidden_dim, 128), ub(128, 64), ub(64, 32), ub(32, 16),
            nn.Conv1d(16, 1, 3, 1, 1),
            nn.Tanh()) # Use Tanh for [-1, 1] range
    def forward(self, h): return self.decoder(h)

class Generator(nn.Module):
    def __init__(self, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(100, hidden_dim * 16)
        def up_res(ic, oc):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(ic, oc, 3, 1, 1),
                nn.BatchNorm1d(oc),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(oc))
        self.u1 = up_res(hidden_dim + cond_dim, hidden_dim)
        self.u2 = up_res(hidden_dim + cond_dim, hidden_dim)
        self.u3 = up_res(hidden_dim + cond_dim, hidden_dim)
        self.u4 = up_res(hidden_dim + cond_dim, hidden_dim)
        # G outputs Latent Embedding (128 channels), NOT final waveform
        self.extra = nn.Sequential(nn.Conv1d(hidden_dim + cond_dim, hidden_dim, 15, 1, 7), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(0.2))
        self.dropout = nn.Dropout(0.3)

    def forward(self, z, c):
        x = self.dropout(self.fc(z).view(-1, self.hidden_dim, 16))
        c_p = c.permute(0, 2, 1)
        gc = lambda res: nn.functional.interpolate(c_p, size=res, mode='nearest')
        x = self.u1(torch.cat([x, gc(16)], dim=1))
        x = self.u2(torch.cat([x, gc(32)], dim=1))
        x = self.u3(torch.cat([x, gc(64)], dim=1))
        x = self.u4(torch.cat([x, gc(128)], dim=1))
        x = nn.functional.interpolate(x, size=512, mode='nearest')
        return self.extra(torch.cat([x, gc(512)], dim=1))

class Supervisor(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
    def forward(self, h):
        out, _ = self.rnn(h.permute(0, 2, 1))
        return out.permute(0, 2, 1)

class Discriminator(nn.Module):
    def __init__(self, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        cb = lambda ic, oc: nn.Sequential(spectral_norm(nn.Conv1d(ic, oc, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3))
        self.conv = nn.Sequential(cb(hidden_dim + cond_dim, 16), cb(16, 32), cb(32, 64), cb(64, 128), nn.Conv1d(128, 1, 32, 1, 0), nn.Sigmoid())
    def forward(self, h, c):
        c_p = nn.functional.interpolate(c.permute(0, 2, 1), size=h.size(-1), mode='nearest')
        return self.conv(torch.cat([h, c_p], dim=1))

# ==========================================
# TRAINING LOGIC
# ==========================================
def train_appliance(appliance):
    CSV_PATH = os.path.join(BASE_DIR, "baseline_comparison", "data", f"{appliance}_multivariate.csv")
    OUT_DIR  = os.path.join(_SCRIPT_DIR, "results")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    df = pd.read_csv(CSV_PATH)
    p_max, p_min = df[appliance].max(), df[appliance].min()
    # ⚡ [-1, 1] Normalization to match successful legacy CGAN
    raw_p_norm = (df[appliance].values - p_min) / (p_max - p_min + 1e-8) * 2 - 1
    time_feat = df[[c for c in df.columns if 'sin' in c or 'cos' in c]].values
    
    dataset = NILM_Dataset(raw_p_norm, time_feat)
    # ⚡ Weighted Sampler to FORCE model to see active waveforms
    sampler = WeightedRandomSampler(dataset.weights, len(dataset.weights), replacement=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    _it = iter(loader)
    def get_batch(jitter=0.0):
        nonlocal _it
        try: X, C = next(_it)
        except StopIteration: _it = iter(loader); X, C = next(_it)
        # Anti-Parking shift
        s = np.random.randint(0, WINDOW_SIZE)
        X, C = torch.roll(X, s, -1), torch.roll(C, s, 1)
        X, C = X.to(device), C.to(device)
        if jitter > 0: C = torch.clamp(C + torch.randn_like(C)*jitter, -1, 1)
        return X, C

    E, R, G, S, D = Embedder().to(device), Recovery().to(device), Generator().to(device), Supervisor().to(device), Discriminator().to(device)
    opt_ER, opt_GS, opt_D = optim.Adam(list(E.parameters())+list(R.parameters()), 2e-4), optim.Adam(list(G.parameters())+list(S.parameters()), 2e-4), optim.Adam(D.parameters(), 1e-4)
    l_mse, l_bce = nn.MSELoss(), nn.BCELoss()

    print(f"Phase 1: Weighted AE Training for {appliance}...")
    for i in range(AE_ITER):
        X, C = get_batch()
        opt_ER.zero_grad()
        H = E(X, C if i > AE_ITER//2 else torch.zeros_like(C))
        X_rec = R(H)
        w = torch.where(X > -0.9, torch.full_like(X, FOCAL), torch.ones_like(X))
        loss = torch.mean((X_rec - X)**2 * w)
        loss.backward(); opt_ER.step()

    print(f"Phase 3: Joint Training (Weighted Active Balance)...")
    for i in range(1, JOINT_ITER + 1):
        for _ in range(4):
            X, C = get_batch(0.0); _, Cj = get_batch(0.05)
            z = torch.randn(BATCH_SIZE, 100, device=device)
            z2 = torch.randn(BATCH_SIZE, 100, device=device)
            opt_GS.zero_grad()

            Eh = G(z, Cj)
            Hh = S(Eh)
            Xh = R(Hh)
            Yf = D(Hh, Cj)
            Yf_e = D(Eh, Cj)
            
            # X, Xh: [-1, 1]
            Xz = X.view(BATCH_SIZE,1,8,64).mean(-1); Xhz = Xh.view(BATCH_SIZE,1,8,64).mean(-1)
            loss_reg = l_mse(Xhz, Xz)
            with torch.no_grad(): Xh2 = R(S(G(z2, Cj)))
            loss_div = torch.clamp(0.1 - (Xh - Xh2).abs().mean(), min=0)
            
            w = torch.where(X > -0.9, torch.full_like(X, FOCAL), torch.ones_like(X))
            loss_anchor = torch.mean((Xh - X)**2 * w) 
            
            loss_g = l_bce(Yf, torch.ones_like(Yf)) + l_bce(Yf_e, torch.ones_like(Yf_e)) + ETA*loss_reg + 2.0*loss_div + 5.0*loss_anchor
            loss_g.backward(); opt_GS.step()

        X, C = get_batch(0.05)
        opt_D.zero_grad()
        with torch.no_grad():
            H = E(X, C)
            Hh = S(G(torch.randn(BATCH_SIZE, 100, device=device), C))
        Yr, Yf = D(H, C), D(Hh, C)
        loss_d = l_bce(Yr, torch.full_like(Yr, 0.9)) + l_bce(Yf, torch.zeros_like(Yf))
        if loss_d > 0.3: loss_d.backward(); opt_D.step()

        if i % 100 == 0:
            print(f"[{i}/{JOINT_ITER}] G={loss_g.item():.4f} | Reg={loss_reg.item():.4f} | Div={loss_div.item():.4f}")
            plt.figure(figsize=(15,3)); plt.plot((X[0,0].cpu()+1)/2); plt.plot((Xh[0,0].detach().cpu()+1)/2); plt.savefig(os.path.join(OUT_DIR, f"progress_{appliance}.png")); plt.close()

    G.eval(); S.eval(); R.eval(); all_p = []
    with torch.no_grad():
        for _ in range(len(dataset)//BATCH_SIZE + 1):
            idx = np.random.choice(len(dataset), BATCH_SIZE)
            Ci = torch.stack([dataset[j][1] for j in idx]).to(device)
            Pi = (R(S(G(torch.randn(BATCH_SIZE,100,device=device), Ci))).cpu().numpy() + 1) / 2
            all_p.append(Pi * (p_max - p_min + 1e-8) + p_min)
    np.save(os.path.join(OUT_DIR, f"synthetic_{appliance}.npy"), np.concatenate(all_p, axis=0)[:len(dataset)])

if __name__ == "__main__":
    for app in APPLIANCES: train_appliance(app)
