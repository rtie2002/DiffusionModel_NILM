"""
TimeGAN-TCN: Pure Sequential Architecture with Hyper-Conditioning
-----------------------------------------------------------------
🚀 Fixes applied:
  - Preserves exact 512 -> 512 temporal resolution in Encoder/Recovery.
  - Hyper-Conditioning: Condition 'C' is injected at EVERY block in 
    the Generator to forcefully lock temporal position alignment.
  - TCN Supervisor: Retained for autoregressive constraint.
  - Discriminator: Strided CNN matching CNN-CGAN for sharp penalties.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

def get_c(c, res):
    """Interpolate condition to match resolution - Exactly as in CNN-CGAN."""
    # c: [B, T, cond_dim] -> [B, cond_dim, T]
    c_p = c.transpose(1, 2)
    return F.interpolate(c_p, size=res, mode='nearest')


class CondConvBlock(nn.Module):
    """Residual Conv1d block that re-injects conditions at every step."""
    def __init__(self, in_channels, cond_dim, out_channels, kernel=5):
        super(CondConvBlock, self).__init__()
        # First conv processes previous features + conditions
        self.conv1 = nn.Conv1d(in_channels + cond_dim, out_channels, kernel, padding=kernel//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # Second conv
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel, padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual mapping
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        # x: [B, C, T], cond: [B, cond_dim, T]
        identity = self.res(x)
        # Re-inject condition
        out = torch.cat([x, cond], dim=1)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Encoder(nn.Module):
    """512 -> 512 Pure Sequence mapping to preserve Temporal Axis."""
    def __init__(self, opt):
        super(Encoder, self).__init__()
        cd = opt.cond_dim
        h = opt.hidden_dim
        
        self.in_conv = nn.Conv1d(opt.z_dim + cd, 32, 5, padding=2)
        self.block1 = CondConvBlock(32, cd, 64)
        self.block2 = CondConvBlock(64, cd, 128)
        self.block3 = CondConvBlock(128, cd, h)

    def forward(self, input, cond):
        c_p = cond.transpose(1, 2)
        x = torch.cat([input, cond], dim=-1).transpose(1, 2)
        
        x = F.leaky_relu(self.in_conv(x), 0.2)
        x = self.block1(x, c_p)
        x = self.block2(x, c_p)
        x = self.block3(x, c_p)
        
        H = x.transpose(1, 2)
        return H  # <--- FIX: Unbounded Latent Space


class Recovery(nn.Module):
    """512 -> 512 Pure Sequence mapping."""
    def __init__(self, opt):
        super(Recovery, self).__init__()
        h = opt.hidden_dim
        
        def cb(ic, oc): return nn.Sequential(
            nn.Conv1d(ic, oc, 5, padding=2),
            nn.BatchNorm1d(oc),
            nn.LeakyReLU(0.2, inplace=True))
            
        self.net = nn.Sequential(
            cb(h, 128),
            cb(128, 64),
            cb(64, 32),
            nn.Conv1d(32, opt.z_dim, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = input.transpose(1, 2)
        X_tilde = self.net(x).transpose(1, 2)
        return self.sigmoid(X_tilde)  # Only Recovery limits to [0,1] domain


class Generator(nn.Module):
    """
    ⚡ GLOBAL UPSAMPLING GENERATOR
    """
    def __init__(self, opt):
        super(Generator, self).__init__()
        cd = opt.cond_dim
        h = opt.hidden_dim
        
        self.fc = nn.Linear(opt.latent_dim, 128 * 16)
        
        def up(ic, oc): return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(ic, oc, 3, 1, 1),
            nn.BatchNorm1d(oc),
            nn.LeakyReLU(0.2, inplace=True))
            
        self.u1 = up(128 + cd, 64)   # 16 -> 32
        self.u2 = up(64 + cd, 32)    # 32 -> 64
        self.u3 = up(32 + cd, 16)    # 64 -> 128
        self.u4 = up(16 + cd, 8)     # 128 -> 256
        self.final_conv = nn.Conv1d(8 + cd, h, 3, 1, 1)

    def forward(self, z, cond):
        z_seed = z.mean(dim=1) 
        x = self.fc(z_seed).view(-1, 128, 16)
        
        x = self.u1(torch.cat([x, get_c(cond, 16)], dim=1)) 
        x = self.u2(torch.cat([x, get_c(cond, 32)], dim=1)) 
        x = self.u3(torch.cat([x, get_c(cond, 64)], dim=1)) 
        x = self.u4(torch.cat([x, get_c(cond, 128)], dim=1)) 
        x = F.interpolate(x, size=512, mode='nearest')
        
        E = self.final_conv(torch.cat([x, get_c(cond, 512)], dim=1)).transpose(1, 2)
        return E  # <--- FIX: Unbounded Latent Space


class Supervisor(nn.Module):
    """
    ⚡ TCN SUPERVISOR
    """
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        h = opt.hidden_dim
        
        layers = []
        for d in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(h, h, 3, padding=d, dilation=d),
                    nn.LeakyReLU(0.2)
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(h, h)

    def forward(self, h_seq):
        x = h_seq.transpose(1, 2)
        s = self.tcn(x).transpose(1, 2)
        out = self.fc(s)
        return out  # <--- FIX: Unbounded Latent Space


class Discriminator(nn.Module):
    """
    ⚡ PIXEL-PERFECT DISCRIMINATOR
    Uses CNN-CGAN's strided logic for zero-tolerance on blurry edges.
    """
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        cd = opt.cond_dim
        h = opt.hidden_dim
        
        def cb(ic, oc): return nn.Sequential(
            spectral_norm(nn.Conv1d(ic, oc, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2))
            
        self.conv = nn.Sequential(
            cb(h + cd, 32),   # 256
            cb(32, 64),       # 128
            cb(64, 128),      # 64
            cb(128, 256),     # 32
            nn.Conv1d(256, 1, opt.seq_len // 16, 1, 0) # Down to [B, 1, 1]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_seq, cond, sigmoid=True):
        disc_input = torch.cat([h_seq, cond], dim=-1).transpose(1, 2)
        y_hat = self.conv(disc_input).view(-1, 1)
        return self.sigmoid(y_hat) if sigmoid else y_hat

