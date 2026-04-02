"""Train_CNN_TimeGAN_Conditional.py

CNN-based Conditional TimeGAN (C-TimeGAN) for NILM Data Augmentation.

Upgraded from: Train_CNN_CGAN_Baseline_Conditional.py
Paper: "Conditional TimeGAN for Realistic and High-Quality Appliance
        Trajectories Generation and Data Augmentation in NILM" IEEE TIM 2024

Architecture (5 CNN networks):
  E  - Embedder   : X + C  → H   (latent embedding space)
  R  - Recovery   : H      → X̂   (reconstruct from embedding)
  G  - Generator  : Z + C  → Ê   (synthetic embedding from noise)
  S  - Supervisor : H      → Ĥ   (temporal next-step predictor)
  D  - Discriminator : H + C → [0,1]  (real vs fake in embedding space)

Training (3 phases, paper Table I):
  Phase 1 — AE pre-training    : E + R  (focal-weighted MSE)
  Phase 2 — Supervisor pre-train: S      (next-step temporal MSE)
  Phase 3 — Joint adversarial  : all 5  (paper Eq. 13-15)

CNN skeleton preserved from the working CNN-CGAN baseline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm

# ==========================================
# CONFIGURATION
# ==========================================
APPLIANCES  = ["dishwasher", "washingmachine", "fridge", "kettle", "microwave"]
WINDOW_SIZE = 512
BATCH_SIZE  = 128
COND_DIM    = 8     # time-feature channels (sin/cos encoding)
HIDDEN_DIM  = 128   # embedding space channels ↑ (was 64) — more capacity for sharp peaks

# Training iterations (3 phases)
AE_ITER    = 5000    # Phase 1: AutoEncoder   ↑ (was 2000) — needs more time on sparse NILM peaks
SUP_ITER   = 5000    # Phase 2: Supervisor    ↑ (was 3000) — need L_S < 0.005 before joint
JOINT_ITER = 20000   # Phase 3: Joint         ↑ (was 5000) — match CGAN budget

# Loss weights (C-TimeGAN paper, Table I)
ETA    = 15.0        # supervised loss weight in G  (η)
LAMBDA = 1.0         # supervised loss weight in ER (λ)
GAMMA  = 1.0         # E_hat discriminator weight   (γ)
FOCAL  = 100.0       # ON-period focal penalty       ↑ (was 50) — stronger peak emphasis

# Script is at  <root>/baseline_comparison/GAN/Train_CNN_TimeGAN_Conditional.py
# So go up 3 levels: GAN → baseline_comparison → project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(os.path.dirname(_SCRIPT_DIR))   # project root
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Project Root : {BASE_DIR}")
print(f"✅ Device       : {device}")


# ==========================================
# MODEL DEFINITIONS  (CNN-based C-TimeGAN)
# ==========================================

class Embedder(nn.Module):
    """E(X, C) → H  |  Maps real sequences into embedding space.
    Uses GELU (not Tanh) so magnitude ordering is preserved
    — a 2400W spike stays larger than a 200W value in embedding space.
    Input : X [B,1,T] + C [B,T,cond_dim]
    Output: H [B,hidden_dim,T]
    """
    def __init__(self, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        def cb(ic, oc):
            return nn.Sequential(
                nn.Conv1d(ic, oc, 3, 1, 1),
                nn.BatchNorm1d(oc),
                nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(
            cb(1 + cond_dim, 64),
            cb(64, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GELU())   # preserves magnitude — Tanh would squash peaks

    def forward(self, x, c):
        # x:[B,1,T]  c:[B,T,cond_dim] → permute → [B,cond_dim,T]
        return self.net(torch.cat([x, c.permute(0, 2, 1)], dim=1))


class Recovery(nn.Module):
    """R(H) → X̂  |  Reconstructs power sequences from embedding.
    
    KEY FIX: Uses Softplus (not Sigmoid) as final activation.
    Sigmoid gradient at peak (0.99) = 0.99×0.01 = 0.009 ≈ 0 → can't learn sharp spikes.
    Softplus has full gradient everywhere, then we clamp to [0,1] post-hoc.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        def cb(ic, oc):
            return nn.Sequential(
                nn.Conv1d(ic, oc, 3, 1, 1),
                nn.BatchNorm1d(oc),
                nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(
            cb(hidden_dim, 64),
            cb(64, 32),
            nn.Conv1d(32, 1, 3, 1, 1),
            nn.Softplus(beta=10))  # smooth, non-saturating; clamp applied in forward

    def forward(self, h):
        out = self.net(h)                          # [B, 1, T], values ≥ 0
        return torch.clamp(out, 0.0, 1.0)         # bound to valid [0,1] range


class Generator(nn.Module):
    """G(Z, C) → Ê  |  Synthesises embeddings from noise + condition.
    Keeps the fc + upsample CNN structure from the working CNN-CGAN baseline,
    but outputs to embedding space (hidden_dim channels) instead of the signal.
    Input : z [B,100], c [B,T,cond_dim]
    Output: Ê [B,hidden_dim,T]
    """
    def __init__(self, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(100, hidden_dim * 16)

        def up(ic, oc):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(ic, oc, 3, 1, 1),
                nn.BatchNorm1d(oc),
                nn.LeakyReLU(0.2, inplace=True))

        self.u1 = up(hidden_dim + cond_dim, hidden_dim)
        self.u2 = up(hidden_dim + cond_dim, hidden_dim)
        self.u3 = up(hidden_dim + cond_dim, hidden_dim)
        self.u4 = up(hidden_dim + cond_dim, hidden_dim)
        self.final_conv = nn.Conv1d(hidden_dim + cond_dim, hidden_dim, 3, 1, 1)

    def forward(self, z, c):
        # z:[B,100]  c:[B,T,cond_dim]
        x   = self.fc(z).view(-1, self.hidden_dim, 16)
        c_p = c.permute(0, 2, 1)   # [B,cond_dim,T]

        def gc(res): return nn.functional.interpolate(c_p, size=res, mode='nearest')

        x = self.u1(torch.cat([x,          gc(16)],  dim=1))   # → 32
        x = self.u2(torch.cat([x,          gc(32)],  dim=1))   # → 64
        x = self.u3(torch.cat([x,          gc(64)],  dim=1))   # → 128
        x = self.u4(torch.cat([x,          gc(128)], dim=1))   # → 256
        x = nn.functional.interpolate(x, size=512, mode='nearest')
        return torch.tanh(self.final_conv(torch.cat([x, gc(512)], dim=1)))


class Supervisor(nn.Module):
    """S(H) → Ĥ  |  Temporal next-step predictor in embedding space.
    Learns the stepwise dynamics so that Ĥ[:,t] ≈ H[:,t+1].
    Input : H [B,hidden_dim,T]
    Output: Ĥ [B,hidden_dim,T]
    """
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        def cb(ic, oc):
            return nn.Sequential(
                nn.Conv1d(ic, oc, 3, 1, 1),
                nn.BatchNorm1d(oc),
                nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(
            cb(hidden_dim, hidden_dim),
            cb(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU())   # consistent with Embedder — no squashing

    def forward(self, h):
        return self.net(h)


class Discriminator(nn.Module):
    """D(H, C) → [0,1]  |  Classifies real/fake in embedding space.
    Identical strided-CNN structure to the CNN-CGAN baseline discriminator.
    Spectral norm + label smoothing for stability.
    Input : H [B,hidden_dim,T], C [B,T,cond_dim]
    Output: [B,1]
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, cond_dim=COND_DIM):
        super().__init__()
        def cb(ic, oc):
            return nn.Sequential(
                spectral_norm(nn.Conv1d(ic, oc, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2))
        self.conv = nn.Sequential(
            cb(hidden_dim + cond_dim, 32),
            cb(32, 64),
            cb(64, 128),
            cb(128, 256),
            nn.Conv1d(256, 1, 32, 1, 0),
            nn.Sigmoid())

    def forward(self, h, c):
        # h:[B,hidden_dim,T]  c:[B,T,cond_dim]
        inp = torch.cat([h, c.permute(0, 2, 1)], dim=1)
        return self.conv(inp).view(-1, 1)


# ==========================================
# DATASET  (identical to CNN-CGAN baseline)
# ==========================================
class NILM_Dataset(Dataset):
    def __init__(self, p, t):
        self.data = []
        stride = WINDOW_SIZE // 2
        for i in range(0, len(p) - WINDOW_SIZE, stride):
            self.data.append((p[i:i + WINDOW_SIZE], t[i:i + WINDOW_SIZE]))
        print(f"📊 Dataset: {len(self.data)} windows (stride={stride})")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        p, t = self.data[idx]
        # p → [1, T]  (unsqueeze channel dim for Conv1d)
        # t → [T, cond_dim] (time features, permuted to [cond_dim,T] in get_batch)
        return torch.from_numpy(p).float().unsqueeze(0), torch.from_numpy(t).float()


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================
def train_appliance(appliance):
    CSV_PATH = os.path.join(BASE_DIR, 'baseline_comparison', 'data',
                            f'{appliance}_multivariate.csv')
    OUT_DIR  = os.path.join(BASE_DIR, 'Synthetic_data', f'ctimegan_{appliance}')
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f'⚠️  Skipping {appliance}: CSV not found at {CSV_PATH}')
        return

    # ── Data loading (same as CNN-CGAN baseline) ──
    print(f'\n{"="*60}')
    print(f'  C-TimeGAN  →  {appliance.upper()}')
    print(f'{"="*60}')
    df        = pd.read_csv(CSV_PATH)
    power_col = appliance if appliance in df.columns else df.columns[0]
    time_cols = [c for c in df.columns if any(k in c for k in ['sin', 'cos'])]

    p_max, p_min = df[power_col].max(), df[power_col].min()

    # Normalise power to [0,1]  (Recovery uses Sigmoid → forces [0,1] output)
    raw_p_01  = (df[power_col].values - p_min) / (p_max - p_min + 1e-8)
    time_feat = df[time_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    dataset = NILM_Dataset(raw_p_01, time_feat)
    cur_bs  = min(BATCH_SIZE, len(dataset))
    loader  = DataLoader(dataset, batch_size=cur_bs, shuffle=True, drop_last=True)
    if len(loader) == 0:
        print('⚠️  Empty loader, skipping.')
        return

    # ── Infinite batch iterator ──
    def inf_loader():
        while True:
            for batch in loader:
                yield batch

    _it = iter(inf_loader())

    def get_batch():
        real_p, real_t = next(_it)
        return real_p.to(device), real_t.to(device)
        # real_p: [B, 1, T]     real_t: [B, T, cond_dim]

    # ── Initialise 5 CNN networks ──
    E = Embedder   (COND_DIM,  HIDDEN_DIM).to(device)
    R = Recovery   (HIDDEN_DIM           ).to(device)
    G = Generator  (COND_DIM,  HIDDEN_DIM).to(device)
    S = Supervisor (HIDDEN_DIM           ).to(device)
    D = Discriminator(HIDDEN_DIM, COND_DIM).to(device)

    def weights_init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
    for net in [E, R, G, S, D]:
        net.apply(weights_init)

    lr = 0.0001
    opt_ER = optim.Adam(list(E.parameters()) + list(R.parameters()),
                        lr=lr, betas=(0.9, 0.999))
    opt_S  = optim.Adam(S.parameters(), lr=lr,     betas=(0.9, 0.999))
    opt_G  = optim.Adam(list(G.parameters()) + list(S.parameters()),
                        lr=0.0002, betas=(0.5, 0.999))
    opt_D  = optim.Adam(D.parameters(), lr=lr,     betas=(0.5, 0.999))

    l_mse = nn.MSELoss()
    l_bce = nn.BCELoss()

    loss_er = loss_s = loss_g = loss_d = torch.tensor(0.0)
    PROG    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'ctimegan_progress_{appliance}.png')

    # ──────────────────────────────────────────────────────────
    # PHASE 1 : AutoEncoder Pre-training  (E + R)
    # Eq L_R = E[ w · ||X - R(E(X,C))||² ]
    # w = 50 on ON-periods (>0.05), 1 elsewhere  (focal penalty)
    # ──────────────────────────────────────────────────────────
    print(f'\n🔧 Phase 1: AutoEncoder pre-training  ({AE_ITER} iters)...')
    for step in range(1, AE_ITER + 1):
        X, C = get_batch()
        opt_ER.zero_grad()
        H       = E(X, C)
        X_tilde = R(H)
        w       = torch.where(X > 0.05,
                              torch.full_like(X, FOCAL),
                              torch.ones_like(X))
        # MSE focal + L1 focal: L1 preserves sharp edges that MSE blurs
        loss_er_mse = torch.mean((X_tilde - X) ** 2 * w)
        loss_er_l1  = torch.mean(torch.abs(X_tilde - X) * w)
        loss_er     = loss_er_mse + 0.5 * loss_er_l1
        loss_er.backward()
        opt_ER.step()
        if step % 200 == 0:
            print(f'  AE  [{step:4d}/{AE_ITER}]  L_R = {loss_er.item():.5f}')

    # ──────────────────────────────────────────────────────────
    # PHASE 2 : Supervisor Pre-training  (S,  E frozen)
    # Eq L_S = E[ ||H_{t+1} - S(H_t)||² ]   (temporal next-step)
    # ──────────────────────────────────────────────────────────
    print(f'\n🔧 Phase 2: Supervisor pre-training  ({SUP_ITER} iters)...')
    for step in range(1, SUP_ITER + 1):
        X, C = get_batch()
        opt_S.zero_grad()
        with torch.no_grad():
            H = E(X, C)
        H_sup  = S(H)
        # H_sup[:,t] should predict H[:,t+1]
        loss_s = l_mse(H_sup[:, :, :-1], H[:, :, 1:])
        loss_s.backward()
        opt_S.step()
        if step % 200 == 0:
            print(f'  SUP [{step:4d}/{SUP_ITER}]  L_S = {loss_s.item():.5f}')

    # ──────────────────────────────────────────────────────────
    # PHASE 3 : Joint Adversarial Training  (all 5 networks)
    #
    # G  (Eq 14): L_G  = η·√L_S + L_U + γ·L_U_e + V1 + V2
    # ER (Eq 13): L_ER = L_R + λ·L_S
    # D         : L_D  = BCE(real,0.9) + BCE(fake_H,0) + γ·BCE(fake_E,0)
    #              Gate: only backprop if L_D > 0.15  (prevents D dominating)
    # G trains 2× per step  (proven effective in CNN-CGAN baseline)
    # ──────────────────────────────────────────────────────────
    print(f'\n🔥 Phase 3: Joint adversarial training  ({JOINT_ITER} iters)...')

    for step in range(1, JOINT_ITER + 1):

        # ── Generator + Supervisor  (2× update) ──────────────
        for _ in range(2):
            X, C = get_batch()
            z    = torch.randn(X.size(0), 100, device=device)
            opt_G.zero_grad()

            E_hat  = G(z, C)
            H_hat  = S(E_hat)
            X_hat  = R(H_hat)            # for moments matching
            Y_fake   = D(H_hat, C)
            Y_fake_e = D(E_hat,  C)

            loss_g_U   = l_bce(Y_fake,   torch.ones_like(Y_fake))
            loss_g_U_e = l_bce(Y_fake_e, torch.ones_like(Y_fake_e))
            loss_g_s   = l_mse(H_hat[:, :, :-1], E_hat[:, :, 1:])

            # Moments matching (V1=std, V2=mean)  paper Eq 15
            loss_g_V1 = torch.mean(torch.abs(torch.std(X_hat, 0) - torch.std(X, 0)))
            loss_g_V2 = torch.mean(torch.abs(torch.mean(X_hat, 0) - torch.mean(X, 0)))

            # Frequency-domain loss: penalise spectral mismatch between fake and real
            # Sharp appliance spikes have a distinct FFT profile that MSE alone misses
            X_hat_fft = torch.abs(torch.fft.rfft(X_hat.squeeze(1), dim=-1))
            X_fft     = torch.abs(torch.fft.rfft(X.squeeze(1),     dim=-1))
            loss_g_freq = torch.mean(torch.abs(X_hat_fft.mean(0) - X_fft.mean(0)))

            loss_g = (loss_g_U
                      + GAMMA  * loss_g_U_e
                      + ETA    * torch.sqrt(loss_g_s + 1e-8)
                      + loss_g_V1 + loss_g_V2
                      + 0.1   * loss_g_freq)   # spectral consistency
            loss_g.backward()
            opt_G.step()

        # ── Encoder + Recovery  (joint update) ───────────────
        X, C = get_batch()
        opt_ER.zero_grad()
        H       = E(X, C)
        X_tilde = R(H)
        H_sup   = S(H)
        w       = torch.where(X > 0.05,
                              torch.full_like(X, FOCAL),
                              torch.ones_like(X))
        loss_er_mse   = torch.mean((X_tilde - X) ** 2 * w)
        loss_er_l1    = torch.mean(torch.abs(X_tilde - X) * w)
        loss_er       = loss_er_mse + 0.5 * loss_er_l1
        loss_s_j  = l_mse(H_sup[:, :, :-1], H[:, :, 1:])
        (loss_er + LAMBDA * loss_s_j).backward()
        opt_ER.step()

        # ── Discriminator ─────────────────────────────────────
        X, C = get_batch()
        z    = torch.randn(X.size(0), 100, device=device)
        opt_D.zero_grad()
        with torch.no_grad():
            H     = E(X, C)
            E_hat = G(z, C)
            H_hat = S(E_hat)
        Y_real   = D(H,     C)
        Y_fake   = D(H_hat, C)
        Y_fake_e = D(E_hat, C)
        loss_d   = (l_bce(Y_real,   torch.full_like(Y_real, 0.9))
                  + l_bce(Y_fake,   torch.zeros_like(Y_fake))
                  + GAMMA * l_bce(Y_fake_e, torch.zeros_like(Y_fake_e)))
        if loss_d > 0.15:    # gate: preserve G training signal
            loss_d.backward()
            opt_D.step()

        # ── Logging + waveform progress ───────────────────────
        if step % 100 == 0:
            print(f'  Joint [{step:4d}/{JOINT_ITER}] '
                  f'G={loss_g.item():.4f} | '
                  f'D={loss_d.item():.4f} | '
                  f'ER={loss_er.item():.5f}')

            E.eval(); G.eval(); S.eval(); R.eval()
            with torch.no_grad():
                X_s, C_s = get_batch()
                z_s      = torch.randn(1, 100, device=device)
                E_hat_p  = G(z_s, C_s[:1])
                H_hat_p  = S(E_hat_p)
                X_hat_p  = R(H_hat_p).cpu().numpy()[0, 0]
                real_s   = X_s[0, 0].cpu().numpy()
                # find active window for cleaner plot
                for ki in range(min(100, len(dataset))):
                    sp = dataset[ki][0]
                    if sp.max() > 0.05:
                        real_s = sp.numpy()[0]; break
            plt.clf()
            plt.plot(real_s,  label='Real',      linewidth=1.5)
            plt.plot(X_hat_p, label='Generated', linewidth=1.5, alpha=0.8)
            plt.title(f'{appliance} | Joint Step {step}/{JOINT_ITER}')
            plt.legend(); plt.tight_layout()
            plt.savefig(PROG); plt.close()
            E.train(); G.train(); S.train(); R.train()

    # ──────────────────────────────────────────────────────────
    # SAMPLING  —  synthesis pipeline: G → S → R → signal
    # ──────────────────────────────────────────────────────────
    print(f'\n🎨 [{appliance}] Generating synthetic data (1:1 ratio)...')
    E.eval(); G.eval(); S.eval(); R.eval()

    all_p, all_t = [], []
    num_windows  = len(dataset)

    with torch.no_grad():
        for _ in range(num_windows // cur_bs + 1):
            idx     = np.random.choice(num_windows, cur_bs)
            batch_c = torch.stack([dataset[i][1] for i in idx]).to(device)
            # batch_c: [B, T, cond_dim]

            z       = torch.randn(cur_bs, 100, device=device)
            E_hat_s = G(z, batch_c)
            H_hat_s = S(E_hat_s)
            p_01    = R(H_hat_s).cpu().numpy()        # [B, 1, T] in [0,1]

            # Inverse-normalise → original Watts
            p_denorm = p_01 * (p_max - p_min + 1e-8) + p_min
            all_p.append(p_denorm)
            all_t.append(batch_c.cpu().numpy())

    final_p = np.concatenate(all_p, axis=0)[:num_windows]   # [N, 1, T]
    final_t = np.concatenate(all_t, axis=0)[:num_windows]   # [N, T, cond_dim]

    # Reshape to [N, T, 1+cond_dim]  (same format as CNN-CGAN output)
    final_p_t    = np.transpose(final_p, (0, 2, 1))          # [N, T, 1]
    final_merged = np.concatenate([final_p_t, final_t], axis=2)

    np_path = os.path.join(OUT_DIR, f'synthetic_{appliance}.npy')
    np.save(np_path, final_merged)
    print(f'✅  Saved → {np_path}  |  shape: {final_merged.shape}')


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    for app in APPLIANCES:
        train_appliance(app)
