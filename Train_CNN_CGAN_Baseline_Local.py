
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
APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
WINDOW_SIZE = 512
BATCH_SIZE = 64
EPOCHS_PER_APP = 10000  # Number of epochs per appliance
BASE_DIR = os.getcwd()  # Assumes running from project root

# Check Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ GPU not found. Using CPU!")

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 128 * 16)
        def up(ic, oc): return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(ic, oc, 3, 1, 1),
            nn.BatchNorm1d(oc),
            nn.ReLU(True))
        self.model = nn.Sequential(up(128,64), up(64,32), up(32,16), up(16,8),
                                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                                nn.Conv1d(8, 1, 3, 1, 1), nn.Tanh())
    def forward(self, z): return self.model(self.fc(z).view(-1, 128, 16))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def cb(ic, oc, s=2): return nn.Sequential(spectral_norm(nn.Conv1d(ic, oc, 4, s, 1)), nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(cb(1, 16), cb(16, 32), cb(32, 64), cb(64, 128),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x): return self.conv(x)

class NILM_Dataset(Dataset):
    def __init__(self, p, t):
        self.data = []
        stride = 64
        for i in range(0, len(p) - WINDOW_SIZE, stride):
            if np.max(p[i:i+WINDOW_SIZE]) > -0.9: self.data.append((p[i:i+WINDOW_SIZE], t[i:i+WINDOW_SIZE]))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        p, t = self.data[idx]
        return torch.from_numpy(p).float().unsqueeze(0), torch.from_numpy(t).float()

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def main():
    print(f"Starting Training in: {BASE_DIR}")
    
    for appliance in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {appliance.upper()}")
        print(f"{'='*60}")
        
        # Path Setup
        CSV_PATH = os.path.join(BASE_DIR, "Data", "datasets", f"{appliance}_multivariate.csv")
        OUT_DIR = os.path.join(BASE_DIR, "Synthetic_Data", appliance)
        PLOT_DIR = os.path.join(OUT_DIR, "training_plots")
        
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(PLOT_DIR, exist_ok=True)

        if not os.path.exists(CSV_PATH):
            print(f"⚠️ Skipping {appliance}: CSV not found at {CSV_PATH}")
            continue

        # Data Prep
        print(f"Loading data...")
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            continue
            
        power_col = next((c for c in df.columns if 'power' in c.lower() or appliance in c.lower()), df.columns[-1])
        p_min, p_max = df[power_col].min(), df[power_col].max()
        raw_p_norm = (df[power_col].values - p_min) / (p_max - p_min) * 2 - 1

        # Convert time features
        time_features_df = df.drop(columns=[power_col]).apply(pd.to_numeric, errors='coerce').fillna(0)
        dataset = NILM_Dataset(raw_p_norm, time_features_df.values)
        
        if len(dataset) == 0:
            print(f"Warning: No valid windows found for {appliance}")
            continue
            
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        # Model Init
        G, D = Generator().to(device), Discriminator().to(device)
        opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
        sched_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=300, gamma=0.5)
        sched_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=300, gamma=0.5)
        criterion = nn.BCELoss()

        # Training Loop
        print(f"🔥 Training for {EPOCHS_PER_APP} epochs...")
        for epoch in range(1, EPOCHS_PER_APP + 1):
            last_real_p = None
            last_fake_p = None

            for real_p, _ in train_loader:
                real_p = real_p.to(device); bs = real_p.size(0)

                # Train Discriminator
                opt_D.zero_grad()
                z = torch.randn(bs, 100).to(device); fake_p = G(z)
                loss_d = (criterion(D(real_p), torch.full((bs,1), 0.9).to(device)) + criterion(D(fake_p.detach()), torch.zeros(bs,1).to(device))) / 2
                loss_d.backward(); opt_D.step()

                # Train Generator
                for _ in range(2):
                    opt_G.zero_grad()
                    fake_p = G(torch.randn(bs, 100).to(device))
                    loss_g = criterion(D(fake_p), torch.ones(bs,1).to(device)) + 0.2 * torch.mean(torch.abs(fake_p[:, :, 1:] - fake_p[:, :, :-1]))
                    loss_g.backward(); opt_G.step()

                last_real_p = real_p.detach().cpu()
                last_fake_p = fake_p.detach().cpu()

            sched_G.step(); sched_D.step()

            # Save Plot Image periodically (Every 100 epochs)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{EPOCHS_PER_APP} complete.")
                plt.figure(figsize=(12, 4))
                if last_real_p is not None and last_fake_p is not None:
                    plt.plot((last_real_p[0,0]+1)/2, label='Real (Reference)', alpha=0.5)
                    plt.plot((last_fake_p[0,0]+1)/2, label='Generated', color='orange', alpha=0.8)
                plt.title(f"{appliance.upper()} Epoch {epoch}")
                plt.legend()
                plt.savefig(os.path.join(PLOT_DIR, f"epoch_{epoch}.png"))
                plt.close()

        # Sampling 200%
        NUM_VALS = len(dataset)
        NUM_GEN = NUM_VALS * 2
        print(f"Generating synthetic data ({NUM_GEN} samples)...")

        G.eval()
        all_f = []
        with torch.no_grad():
            for _ in range(NUM_GEN // BATCH_SIZE + 1):
                all_f.append((G(torch.randn(BATCH_SIZE, 100).to(device)).cpu().numpy() + 1) / 2)

        final_p = np.concatenate(all_f, axis=0)[:NUM_GEN]
        indices = np.random.choice(len(dataset), NUM_GEN, replace=True)
        final_t = np.array([dataset[idx][1].numpy() for idx in indices])

        final_merged = np.concatenate([np.expand_dims(final_p.squeeze(1), axis=2), final_t], axis=2)

        np_path = os.path.join(OUT_DIR, f"synthetic_{appliance}.npy")
        np.save(np_path, final_merged)
        torch.save(G.state_dict(), os.path.join(OUT_DIR, f"{appliance}_generator.pth"))
        print(f"✅ Saved synthetic data to: {np_path}")

    print("\n🏆 ALL JOBS DONE.")

if __name__ == "__main__":
    main()
