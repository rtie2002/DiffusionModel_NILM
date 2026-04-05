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
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
APPLIANCES  = ["dishwasher", "washingmachine", "fridge", "kettle", "microwave"]
WINDOW_SIZE = 512
BATCH_SIZE  = 128
COND_DIM    = 8     # time features only (minute/hour/dow/month sin+cos)
                    # NOTE: Δpower REMOVED from input to prevent data leakage.
                    # Instead, derivative & phase-shift alignment enforced via LOSS.
HIDDEN_DIM  = 192
AE_ITER    = 20000
SUP_ITER   = 20000
JOINT_ITER = 50000

# Loss weights
ETA    = 1.0         # TimeGAN Joint Supervised Weight
LAMBDA = 1.0         # AE Supervised Weight
GAMMA  = 1.0         # E-hat Discriminator Weight
FOCAL  = 30.0        # ON-period Focal Weight (for imbalanced spikes)

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

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, 1, 1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, 3, 1, 1),
            nn.BatchNorm1d(channels))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))

class Embedder(nn.Module):
    """E(X, C) → H  |  Upgraded to 4 ResBlocks for deep feature extraction.
    """
    def __init__(self, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.init_conv = nn.Conv1d(1 + cond_dim, hidden_dim, 15, 1, 7)
        self.res = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(4)])
        self.final = nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1)

    def forward(self, x, c):
        h = self.init_conv(torch.cat([x, c.permute(0, 2, 1)], dim=1))
        h = self.res(h)
        return self.final(h)


class Recovery(nn.Module):
    """R(H) → X̂  |  Upgraded to 192 channels (No Bottleneck).
    """
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.init_conv = nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1)
        self.res = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(4)])
        self.final = nn.Sequential(
            nn.Conv1d(hidden_dim, 1, 3, 1, 1),
            nn.Softplus(beta=10))

    def forward(self, h):
        x = self.init_conv(h)
        x = self.res(x)
        return torch.clamp(self.final(x), 0.0, 1.0)


class Generator(nn.Module):
    """G(Z, C) → Ê  |  Upgraded resolution and depth.
    Input : Z [B, 100], C [B, T, cond_dim]
    """
    def __init__(self, cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Startup 192 x 16
        self.fc = nn.Linear(100, hidden_dim * 16)
        
        def up_block(ic, oc):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(ic, oc, 3, 1, 1),
                nn.BatchNorm1d(oc),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(oc),
                ResBlock(oc))

        self.u1 = up_block(hidden_dim + cond_dim, hidden_dim)
        self.u2 = up_block(hidden_dim + cond_dim, hidden_dim)
        self.u3 = up_block(hidden_dim + cond_dim, hidden_dim)
        self.u4 = up_block(hidden_dim + cond_dim, hidden_dim)
        self.final_conv = nn.Conv1d(hidden_dim + cond_dim, hidden_dim, 3, 1, 1)

    def forward(self, z, c):
        x   = self.fc(z).view(-1, self.hidden_dim, 16)
        c_p = c.permute(0, 2, 1)
        def gc(res): return nn.functional.interpolate(c_p, size=res, mode='nearest')
        
        x = self.u1(torch.cat([x, gc(16)],  dim=1))
        x = self.u2(torch.cat([x, gc(32)],  dim=1))
        x = self.u3(torch.cat([x, gc(64)],  dim=1))
        x = self.u4(torch.cat([x, gc(128)], dim=1))
        x = nn.functional.interpolate(x, size=512, mode='nearest')
        return self.final_conv(torch.cat([x, gc(512)], dim=1))


class Supervisor(nn.Module):
    """S(H) → Ĥ  |  Next-step predictor. Deepened to 4 ResBlocks.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1, dilation=1),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 4, dilation=4),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1))

    def forward(self, h):
        return self.res(h)


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
                nn.Dropout(0.25))
        
        # Heavy-duty Critique Chain: Match 192-dim Gen power
        self.conv = nn.Sequential(
            cb(hidden_dim + cond_dim, 64),
            cb(64, 128),
            cb(128, 256),
            cb(256, 512),
            nn.Conv1d(512, 1, 32, 1, 0),
            nn.Sigmoid())

    def forward(self, h, c):
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

    # ── APPLIANCE-SPECIFIC TUNING ──────────────────────────────────────────
    # Microwave is extremely sparse/short → boost Focal weight to prevent zero-collapse.
    # Washing machine needs smoothness.
    current_focal = FOCAL
    if "microwave" in appliance.lower():
        current_focal = 100.0  # Intense focus on the short microwave bursts
        print(f'   → ⚡ Microwave detected: Boosting FOCAL to {current_focal}')
    elif "washing" in appliance.lower():
        current_focal = 50.0   # Help with long complex cycles

    # Normalise power to [0,1]  (Recovery uses Sigmoid → forces [0,1] output)
    raw_p_01  = (df[power_col].values - p_min) / (p_max - p_min + 1e-8)
    time_feat = df[time_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    # ── LEAKAGE FIX: Δpower REMOVED from condition input ──────────────────────
    # Previously, the first-order difference of REAL power was fed as the 9th
    # condition channel.  This gave the Generator "god vision" during training
    # (it knew exactly where ON/OFF transitions were), but at sampling time
    # it had to be zeroed out → the model collapsed.
    #
    # Instead, derivative alignment and phase-shift losses are added to the
    # Generator loss function in Phase 3.  The model must now LEARN to produce
    # sharp edges autonomously.
    print(f'   → Condition dim: {time_feat.shape[1]}  (8 time features, NO Δpower leakage)')

    dataset = NILM_Dataset(raw_p_01, time_feat)

    # ── MINORITY CLASS BALANCING: Weighted Sampling ──────────────────────────
    num_on = sum(1 for p_w, _ in dataset if p_w.max() > 0.05)
    num_off = len(dataset) - num_on
    print(f"   → Stats: {num_on} ON windows, {num_off} OFF windows")
    
    if appliance.lower() == "fridge":
        sampler = None
        print("   ⚠️  Booster disabled for Fridge.")
    elif num_on > 0:
        # Target: roughly 50% ON windows in each batch
        w_on = (num_off / num_on) 
        weights = [w_on if p_w.max() > 0.05 else 1.0 for p_w, _ in dataset]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        print(f"   → Applied WeightedRandomSampler (ON boost factor: {w_on:.2f})")
    else:
        sampler = None
        print("   ⚠️  No ON periods found in dataset. Using uniform sampling.")

    cur_bs  = min(BATCH_SIZE, len(dataset))
    loader  = DataLoader(dataset, batch_size=cur_bs, 
                         sampler=sampler, 
                         shuffle=(sampler is None), 
                         drop_last=True)
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

    lr_init = 0.0001
    opt_ER = optim.Adam(list(E.parameters()) + list(R.parameters()), lr=lr_init, betas=(0.5, 0.999))
    opt_S  = optim.Adam(S.parameters(), lr=lr_init, betas=(0.5, 0.999))
    opt_G  = optim.Adam(list(G.parameters()) + list(S.parameters()), lr=lr_init, betas=(0.5, 0.999))
    opt_D  = optim.Adam(D.parameters(), lr=lr_init, betas=(0.5, 0.999))

    # Schedulers to handle 100k iteration decay
    sch_ER = optim.lr_scheduler.StepLR(opt_ER, step_size=30000, gamma=0.5)
    sch_S  = optim.lr_scheduler.StepLR(opt_S,  step_size=30000, gamma=0.5)
    sch_G  = optim.lr_scheduler.StepLR(opt_G,  step_size=30000, gamma=0.5)
    sch_D  = optim.lr_scheduler.StepLR(opt_D,  step_size=30000, gamma=0.5)

    l_mse = nn.MSELoss()
    l_bce = nn.BCELoss()

    loss_er = loss_s = loss_g = loss_d = torch.tensor(0.0)
    PROG    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'ctimegan_progress_{appliance}.png')

    # ──────────────────────────────────────────────────────────
    # PHASE 1 : AutoEncoder Pre-training  (E + R)
    #
    # REDESIGN: Two sub-phases for clean shape learning:
    #   1a. Shape-only   : E encodes X WITHOUT C → pure morphology
    #   1b. Conditional  : E fine-tuned WITH C  → learns when/where
    # This prevents the Embedder from learning a C→X shortcut,
    # which is the root cause of memorization.
    # ──────────────────────────────────────────────────────────
    print(f'\n🔧 Phase 1a: Shape-only AutoEncoder  ({AE_ITER//2} iters, no condition)...')
    # Use zero condition so E sees only power shape
    for step in tqdm(range(1, AE_ITER // 2 + 1), desc="Phase 1a"):
        X, C = get_batch()
        opt_ER.zero_grad()
        C_zero  = torch.zeros_like(C)   # ← blind to time features
        H       = E(X, C_zero)
        X_tilde = R(H)
        w = torch.where(X > 0.05, torch.full_like(X, FOCAL), torch.ones_like(X))
        loss_er = torch.mean((X_tilde - X)**2 * w) + 0.5 * torch.mean(torch.abs(X_tilde - X) * w)
        loss_er.backward()
        opt_ER.step()
        if step % 200 == 0:
            print(f'  AE-shape [{step:4d}/{AE_ITER//2}]  L_R = {loss_er.item():.5f}')

    print(f'\n🔧 Phase 1b: Conditional AutoEncoder fine-tune ({AE_ITER//2} iters)...')
    for step in tqdm(range(1, AE_ITER // 2 + 1), desc="Phase 1b"):
        X, C = get_batch()
        opt_ER.zero_grad()
        H       = E(X, C)              # ← now with real C for fine-tuning
        X_tilde = R(H)
        w = torch.where(X > 0.05, torch.full_like(X, current_focal), torch.ones_like(X))
        loss_er = torch.mean((X_tilde - X)**2 * w) + 0.5 * torch.mean(torch.abs(X_tilde - X) * w)
        loss_er.backward()
        opt_ER.step()
        if step % 200 == 0:
            print(f'  AE-cond  [{step:4d}/{AE_ITER//2}]  L_R = {loss_er.item():.5f}')

    # ──────────────────────────────────────────────────────────
    # PHASE 2 : Supervisor Pre-training  (S,  E frozen)
    # Eq L_S = E[ ||H_{t+1} - S(H_t)||² ]   (temporal next-step)
    # ──────────────────────────────────────────────────────────
    print(f'\n🔧 Phase 2: Supervisor pre-training  ({SUP_ITER} iters)...')
    for step in tqdm(range(1, SUP_ITER + 1), desc="Phase 2"):
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
    # REDESIGNED for Natural, Diverse Generation:
    #
    # G Loss = Adversarial + Supervisor + Distribution matching
    #          + Derivative (batch-level) + Diversity penalty
    #          (NO loss_g_brute → removes memorization!)
    #
    # KEY PRINCIPLE:
    #   - D is the SOLE quality judge (real vs fake)
    #   - Derivative loss matches STATISTICAL distribution of edges,
    #     not per-sample alignment (which caused copying)
    #   - Diversity loss: same C + two different z → must differ
    # ──────────────────────────────────────────────────────────
    print(f'\n🔥 Phase 3: Joint adversarial training  ({JOINT_ITER} iters)...')

    for step in tqdm(range(1, JOINT_ITER + 1), desc="Phase 3"):
        # ── Generator + Supervisor ────────────
        for _ in range(3):
            X, C = get_batch()
            z    = torch.randn(X.size(0), 100, device=device)
            opt_G.zero_grad()

            E_hat  = G(z, C)
            H_hat  = S(E_hat)
            X_hat  = R(H_hat)
            Y_fake   = D(H_hat, C)
            Y_fake_e = D(E_hat,  C)

            # 1. Adversarial
            loss_g_U   = l_bce(Y_fake,   torch.ones_like(Y_fake))
            loss_g_U_e = l_bce(Y_fake_e, torch.ones_like(Y_fake_e))

            # 2. Supervisor Coherence
            loss_g_s = l_mse(H_hat[:, :, :-1], E_hat[:, :, 1:])

            # 3. STATISTICAL MATCH (Distribution level)
            loss_g_V2 = torch.abs(torch.mean(X_hat) - torch.mean(X))
            loss_g_V1 = torch.abs(torch.std(X_hat) - torch.std(X))

            # 4. SPECTRAL MATCH
            X_hat_fft   = torch.abs(torch.fft.rfft(X_hat.squeeze(1), dim=-1))
            X_fft       = torch.abs(torch.fft.rfft(X.squeeze(1),     dim=-1))
            loss_g_freq = torch.mean(torch.abs(X_hat_fft.mean(0) - X_fft.mean(0)))

            # 5. DERIVATIVE DISTRIBUTION MATCH (Statistical Edge Lock)
            real_deriv_abs = (X[:, :, 1:] - X[:, :, :-1]).abs()
            fake_deriv_abs = (X_hat[:, :, 1:] - X_hat[:, :, :-1]).abs()
            loss_g_deriv = torch.abs(fake_deriv_abs.mean() - real_deriv_abs.mean()) + \
                           torch.abs(fake_deriv_abs.std()  - real_deriv_abs.std())

            loss_g = (loss_g_U
                      + GAMMA * loss_g_U_e
                      + ETA   * torch.sqrt(loss_g_s + 1e-8)
                      + 10.0  * loss_g_V1 + 10.0 * loss_g_V2
                      + 2.0   * loss_g_freq
                      + 5.0   * loss_g_deriv)
            loss_g.backward()
            opt_G.step()

        # ── Encoder + Recovery ────────────────
        X, C = get_batch()
        opt_ER.zero_grad()
        H       = E(X, C)
        X_tilde = R(H)
        H_sup   = S(H)
        w       = torch.where(X > 0.05, torch.full_like(X, current_focal), torch.ones_like(X))
        loss_er = torch.mean((X_tilde - X) ** 2 * w) + 0.5 * torch.mean(torch.abs(X_tilde - X) * w)
        loss_s_j = l_mse(H_sup[:, :, :-1], H[:, :, 1:])
        (loss_er + LAMBDA * loss_s_j).backward()
        opt_ER.step()

        # ── Discriminator ──
        X, C = get_batch()
        z    = torch.randn(X.size(0), 100, device=device)
        opt_D.zero_grad()
        with torch.no_grad():
            H     = E(X, C)
            E_hat = G(z, C)
            H_hat = S(E_hat)
        Y_real   = D(H,     C)
        Y_fake   = D(H_hat, C)
        loss_d   = l_bce(Y_real, torch.full_like(Y_real, 0.9)) + l_bce(Y_fake, torch.zeros_like(Y_fake))
        if loss_d > 0.15:
            loss_d.backward()
            opt_D.step()

        # Step Schedulers
        sch_G.step(); sch_D.step(); sch_S.step(); sch_ER.step()

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
        if loss_d > 0.5:    # gate: ↑ (was 0.15) — stop D if it becomes too strong
            loss_d.backward()
            opt_D.step()

        # ── Logging + waveform progress ───────────────────────
        if step % 100 == 0:
            print(f'  Joint [{step:4d}/{JOINT_ITER}] '
                  f'G={loss_g.item():.4f} | '
                  f'D={loss_d.item():.4f} | '
                  f'ER={loss_er.item():.5f} | '
                  f'Deriv={loss_g_deriv.item():.5f} | '
                  f'Div={loss_g_div.item():.4f} | '
                  f'V3(ON)={loss_g_V3.item():.4f}')

            E.eval(); G.eval(); S.eval(); R.eval()
            with torch.no_grad():
                # Get 3 consecutive windows to show continuity
                # Using a fixed starting point for visual consistency
                start_id = 0
                for ki in range(min(500, len(dataset))):
                    if dataset[ki][0].max() > 0.05:
                        start_id = ki; break

                rs_list, gs_list = [], []
                for i in range(3):
                    cur_idx = (start_id + i * (WINDOW_SIZE // 2)) % len(dataset)
                    real_p_w, real_t_w = dataset[cur_idx]
                    real_p_w = real_p_w.to(device).unsqueeze(0)
                    real_t_w = real_t_w.to(device).unsqueeze(0)

                    z_s = torch.randn(1, 100, device=device)
                    E_hat_p = G(z_s, real_t_w)
                    H_hat_p = S(E_hat_p)
                    fake_p_w = R(H_hat_p).cpu().numpy()[0, 0]

                    # For plotting continuity, take only the non-overlapping stride part
                    # but here we'll just append for simplicity.
                    rs_list.append(real_p_w[0,0].cpu().numpy())
                    gs_list.append(fake_p_w)

                real_long = np.concatenate(rs_list)
                fake_long = np.concatenate(gs_list)

            plt.figure(figsize=(15, 5))
            plt.plot(real_long, label='Real (3-windows)', color='teal', linewidth=1.5)
            plt.plot(fake_long, label='Generated (TimeGAN)', color='darkorange', linewidth=1.5, alpha=0.8)
            plt.title(f'{appliance} | Long-term Progress | Step {step}/{JOINT_ITER}')
            plt.xlabel('Timestep (3 × 512)')
            plt.ylabel('Power (Normalised)')
            plt.grid(True, alpha=0.3)
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
            # batch_c: [B, T, 8]  — pure time features, no leakage

            z       = torch.randn(cur_bs, 100, device=device)
            E_hat_s = G(z, batch_c)
            H_hat_s = S(E_hat_s)
            p_01    = R(H_hat_s).cpu().numpy()        # [B, 1, T] in [0,1]

            # Inverse-normalise → original Watts
            p_denorm = p_01 * (p_max - p_min + 1e-8) + p_min
            all_p.append(p_denorm)
            all_t.append(batch_c.cpu().numpy())  # [B, T, 8]

    final_p = np.concatenate(all_p, axis=0)[:num_windows]   # [N, 1, T]
    final_t = np.concatenate(all_t, axis=0)[:num_windows]   # [N, T, 8]

    # Reshape to [N, T, 1+8]  (same format as CNN-CGAN output)
    final_p_t    = np.transpose(final_p, (0, 2, 1))          # [N, T, 1]
    final_merged = np.concatenate([final_p_t, final_t], axis=2)  # [N, T, 9]

    np_path = os.path.join(OUT_DIR, f'synthetic_{appliance}.npy')
    np.save(np_path, final_merged)
    print(f'✅  Saved → {np_path}  |  shape: {final_merged.shape}')


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    for app in APPLIANCES:
        train_appliance(app)
