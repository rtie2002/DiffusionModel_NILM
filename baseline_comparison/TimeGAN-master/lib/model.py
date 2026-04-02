"""
TCN-Enhanced TimeGAN (Waveform Specialist)
------------------------------------------
🚀 ARCHITECTURE UPGRADE:
  - Supervisor:    GRU → TCN (Temporal Convolutional Network)
                   Uses Dilated Convolutions to handle sequence length 512
                   with 100% temporal coherence and NO gradient vanishing.
  - Discriminator: CNN-based (Conv1d) WITHOUT Global Pooling.
                   Preserves spatial resolution to detect sharp NILM edges.
  - Generator:     ResNet-style Conv1d blocks for superior waveform detail.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    """Basic Residual Conv1d Block for sharp waveforms."""
    def __init__(self, in_channels, out_channels, kernel=5, dilation=1):
        super(ConvBlock, self).__init__()
        padding = (kernel - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.res(x)


class Encoder(nn.Module):
    """
    ⚡ DUAL-STREAM ENCODER: Processes Power and Time separately before fusion.
    """
    def __init__(self, opt):
        super(Encoder, self).__init__()
        h = opt.hidden_dim
        
        # Power Path (focus on sharp transients)
        self.p_path = nn.Sequential(
            nn.Conv1d(opt.z_dim, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            ConvBlock(32, 64)
        )
        # Time Path (focus on periodic cycles)
        self.t_path = nn.Sequential(
            nn.Conv1d(opt.cond_dim, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            ConvBlock(32, 64)
        )
        # Fusion Path
        self.fusion = nn.Sequential(
            ConvBlock(128, 128),
            nn.Conv1d(128, h, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, cond, sigmoid=True):
        p_feat = self.p_path(input.transpose(1, 2))
        t_feat = self.t_path(cond.transpose(1, 2))
        combined = torch.cat([p_feat, t_feat], dim=1) # [B, 128, T]
        h = self.fusion(combined).transpose(1, 2)
        return self.sigmoid(h) if sigmoid else h


class Recovery(nn.Module):
    def __init__(self, opt):
        super(Recovery, self).__init__()
        h = opt.hidden_dim
        
        self.model = nn.Sequential(
            ConvBlock(h, 128),
            ConvBlock(128, 64),
            nn.Conv1d(64, opt.z_dim, kernel_size=3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, sigmoid=True):
        x = input.transpose(1, 2)
        x_tilde = self.model(x).transpose(1, 2)
        return self.sigmoid(x_tilde) if sigmoid else x_tilde


class Generator(nn.Module):
    """
    ⚡ DUAL-STREAM GENERATOR: Condition gates the noise-to-latent generation.
    """
    def __init__(self, opt):
        super(Generator, self).__init__()
        h = opt.hidden_dim
        
        # Noise Projector
        self.z_path = nn.Sequential(
            nn.Conv1d(opt.latent_dim, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            ConvBlock(64, 64)
        )
        # Condition Projector
        self.c_path = nn.Sequential(
            nn.Conv1d(opt.cond_dim, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            ConvBlock(64, 64)
        )
        # Depth-wise Fusion
        self.fusion = nn.Sequential(
            ConvBlock(128, 128),
            nn.Conv1d(128, h, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, cond, sigmoid=True):
        z_feat = self.z_path(z.transpose(1, 2))
        c_feat = self.c_path(cond.transpose(1, 2))
        combined = torch.cat([z_feat, c_feat], dim=1)
        e = self.fusion(combined).transpose(1, 2)
        return self.sigmoid(e) if sigmoid else e


class Supervisor(nn.Module):
    """
    ⚡ TCN SUPERVISOR: Replaces GRU to handle long (512) sequences.
    Uses dilated convolutions to achieve a 512+ receptive field.
    """
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        h = opt.hidden_dim
        
        # Dilations: 1, 2, 4, 8, 16, 32, 64, 128, 256 -> Receptive Field = 511
        layers = []
        for d in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            layers.append(ConvBlock(h, h, kernel=3, dilation=d))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(h, h)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_seq, sigmoid=True):
        x = h_seq.transpose(1, 2)
        s = self.tcn(x).transpose(1, 2)
        out = self.fc(s)
        return self.sigmoid(out) if sigmoid else out


class Discriminator(nn.Module):
    """
    ⚡ LOCAL DISCRIMINATOR: No adaptive pooling.
    Evaluates waveform validity at multiple temporal resolutions.
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
        )
        # Ensure final FC dimension is correct for 512 sequence
        # D downsamples by 2^3 = 8 -> 512 / 8 = 64
        self.fc = nn.Linear(256 * (opt.seq_len // 8), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_seq, cond, sigmoid=True):
        x = torch.cat([h_seq, cond], dim=-1).transpose(1, 2)
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        y_hat = self.fc(x)
        return self.sigmoid(y_hat) if sigmoid else y_hat

