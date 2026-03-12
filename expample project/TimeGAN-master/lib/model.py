"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

-----------------------------

model.py: Network Modules (CNN-Enhanced TimeGAN)

⚡ HYBRID ARCHITECTURE:
  - Encoder, Recovery, Generator, Discriminator → CNN-based (Conv1d)
    for effective local pattern capture (sharp edges, transients, ON/OFF)
  - Supervisor → GRU-based (THE TimeGAN signature)
    for autoregressive temporal coherence in latent space

This preserves the TimeGAN training framework while gaining CNN's
ability to generate detailed waveform patterns.

(1) Encoder     - Conv1d: real data → latent space
(2) Recovery    - Conv1d: latent space → data space
(3) Generator   - Conv1d: noise+cond → latent space
(4) Supervisor  - GRU:    latent → latent (temporal supervision) ← TimeGAN core
(5) Discriminator - Conv1d: latent+cond → real/fake classification
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Encoder(nn.Module):
    """Embedding network: maps real data → latent space using Conv1d.

    Conv1d extracts local temporal features much more effectively than
    GRU for signals with sharp transitions and high-frequency texture.

    Args:
      - input: [B, T, z_dim]  (appliance power)
      - cond:  [B, T, cond_dim] (time features + first-order diff)

    Returns:
      - H: [B, T, hidden_dim]  (latent embeddings)
    """
    def __init__(self, opt):
        super(Encoder, self).__init__()
        in_dim = opt.z_dim + opt.cond_dim

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(128, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, cond, sigmoid=True):
        combined = torch.cat([input, cond], dim=-1)   # [B, T, in_dim]
        x = combined.transpose(1, 2)                  # [B, in_dim, T]
        x = self.conv(x)                              # [B, 128, T]
        x = x.transpose(1, 2)                         # [B, T, 128]
        H = self.fc(x)                                # [B, T, hidden_dim]
        if sigmoid:
            H = self.sigmoid(H)
        return H


class Recovery(nn.Module):
    """Recovery network: maps latent space → data space using Conv1d.

    Uses multiple Conv1d layers for sharp detail reconstruction,
    which is critical for capturing ON/OFF edges and transient spikes.

    Args:
      - H: [B, T, hidden_dim]

    Returns:
      - X_tilde: [B, T, z_dim]
    """
    def __init__(self, opt):
        super(Recovery, self).__init__()
        h = opt.hidden_dim

        self.conv = nn.Sequential(
            nn.Conv1d(h, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32, opt.z_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, sigmoid=True):
        x = input.transpose(1, 2)                     # [B, hidden_dim, T]
        x = self.conv(x)                              # [B, 32, T]
        x = x.transpose(1, 2)                         # [B, T, 32]
        X_tilde = self.fc(x)                          # [B, T, z_dim]
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde


class Generator(nn.Module):
    """Generator: maps noise+condition → latent space using Conv1d.

    CNN-based generation captures local patterns (sharp peaks, texture)
    far more effectively than GRU which tends to smooth everything out.

    Args:
      - Z:    [B, T, latent_dim]  (random noise)
      - cond: [B, T, cond_dim]    (conditions)

    Returns:
      - E: [B, T, hidden_dim]  (generated latent embeddings)
    """
    def __init__(self, opt):
        super(Generator, self).__init__()
        in_dim = opt.latent_dim + opt.cond_dim

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, cond, sigmoid=True):
        combined = torch.cat([input, cond], dim=-1)   # [B, T, in_dim]
        x = combined.transpose(1, 2)                  # [B, in_dim, T]
        x = self.conv(x)                              # [B, 128, T]
        x = x.transpose(1, 2)                         # [B, T, 128]
        E = self.fc(x)                                # [B, T, hidden_dim]
        if sigmoid:
            E = self.sigmoid(E)
        return E


class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence.

    ⚡⚡⚡ THIS IS THE CORE TimeGAN COMPONENT ⚡⚡⚡

    KEPT AS GRU — this is what makes this model "TimeGAN" and not just
    a plain GAN. The Supervisor enforces autoregressive temporal coherence
    in latent space by predicting h_{t+1} from h_t, ensuring that generated
    sequences follow realistic temporal dynamics.

    Without this, the model is just a CNN-GAN with an autoencoder.

    Args:
      - H: [B, T, hidden_dim]  (latent representation)

    Returns:
      - S: [B, T, hidden_dim]  (supervised latent sequence)
    """
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(
            input_size=opt.hidden_dim,
            hidden_size=opt.hidden_dim,
            num_layers=opt.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Uses strided Conv1d (like CNN-CGAN) with spectral normalization
    for stable adversarial training. Outputs a single real/fake
    probability per sample (whole-sequence discrimination).

    Args:
      - H:    [B, T, hidden_dim]
      - cond: [B, T, cond_dim]

    Returns:
      - Y_hat: [B, 1]  (real/fake probability)
    """
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        in_dim = opt.hidden_dim + opt.cond_dim

        self.model = nn.Sequential(
            spectral_norm(nn.Conv1d(in_dim, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, cond, sigmoid=True):
        combined = torch.cat([input, cond], dim=-1)   # [B, T, D]
        x = combined.transpose(1, 2)                  # [B, D, T]
        x = self.model(x)                             # [B, 256]
        Y_hat = self.fc(x)                            # [B, 1]
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat
