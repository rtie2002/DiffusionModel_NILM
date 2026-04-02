"""
TimeGAN-TCN: Pixel-Perfect Alignment with CNN-CGAN
-------------------------------------------------
🚀 ARCHITECTURE TRANSPLANT:
  - Generator/Recovery:   Uses Upsample + Multi-level Condition Injection
                          (Exact logic from the successful CNN-CGAN script)
  - Discriminator/Encoder: Uses strided Conv1d + Spectral Norm + LeakyReLU
  - Supervisor:           TCN with Dilated Convolutions (1-256)
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


class ConvBlock(nn.Module):
    """Basic Residual Conv1d Block for TCN."""
    def __init__(self, in_channels, out_channels, kernel=3, dilation=1):
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
    """Discriminator-style Downsampling for Encoding."""
    def __init__(self, opt):
        super(Encoder, self).__init__()
        cd = opt.cond_dim
        h = opt.hidden_dim
        
        def cb(ic, oc): return nn.Sequential(
            spectral_norm(nn.Conv1d(ic, oc, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1))
            
        self.conv = nn.Sequential(
            cb(opt.z_dim + cd, 32), # 256
            cb(32, 64),              # 128
            cb(64, 128),             # 64
            cb(128, 256),            # 32
        )
        # Sequence-level flattening to match hidden_dim
        self.fc = nn.Linear(256 * (opt.seq_len // 16), opt.seq_len * h)
        self.h = h
        self.t = opt.seq_len
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, cond, sigmoid=True):
        # input: [B, T, 1], cond: [B, T, 8]
        x = torch.cat([input, cond], dim=-1).transpose(1, 2)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x).view(-1, self.t, self.h)
        return self.sigmoid(x) if sigmoid else x


class Recovery(nn.Module):
    """Generator-style Upsampling for Recovery."""
    def __init__(self, opt):
        super(Recovery, self).__init__()
        h = opt.hidden_dim
        
        def up(ic, oc): return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(ic, oc, 3, 1, 1),
            nn.BatchNorm1d(oc),
            nn.LeakyReLU(0.2, inplace=True))
            
        # Recovery uses the same block style as GAN-Generator.
        self.u1 = up(h, 64)   # 64
        self.u2 = up(64, 32)  # 128
        self.u3 = up(32, 16)  # 256
        self.u4 = up(16, 8)   # 512
        self.final = nn.Conv1d(8, opt.z_dim, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, sigmoid=True):
        # input: [B, T, H] -> [B, H, T]
        x = input.transpose(1, 2)
        # Fixed seeding at resolution 32
        x = F.interpolate(x, size=32, mode='nearest') 
        
        x = self.u1(x) # 64
        x = self.u2(x) # 128
        x = self.u3(x) # 256
        x = self.u4(x) # 512
        x_tilde = self.final(x).transpose(1, 2)
        return self.sigmoid(x_tilde) if sigmoid else x_tilde


class Generator(nn.Module):
    """
    ⚡ PIXEL-PERFECT GENERATOR:
    Matches CNN-CGAN exactly with Multi-level Condition Injection.
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
            
        self.u1 = up(128 + cd, 64)
        self.u2 = up(64 + cd, 32)
        self.u3 = up(32 + cd, 16)
        self.u4 = up(16 + cd, 8)
        self.final_conv = nn.Conv1d(8 + cd, h, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, cond, sigmoid=True):
        # z: [B, T, latent_dim] -> we use mean noise per specimen to match GAN seed logic
        z_seed = z.mean(dim=1) 
        x = self.fc(z_seed).view(-1, 128, 16) # [B, 128, 16]
        
        # Inject condition at every layer - EXACTLY like CNN-CGAN
        x = self.u1(torch.cat([x, get_c(cond, 16)], dim=1)) # 32
        x = self.u2(torch.cat([x, get_c(cond, 32)], dim=1)) # 64
        x = self.u3(torch.cat([x, get_c(cond, 64)], dim=1)) # 128
        x = self.u4(torch.cat([x, get_c(cond, 128)], dim=1)) # 256
        x = F.interpolate(x, size=512, mode='nearest')
        
        e = self.final_conv(torch.cat([x, get_c(cond, 512)], dim=1)).transpose(1, 2)
        return self.sigmoid(e) if sigmoid else e


class Supervisor(nn.Module):
    """
    ⚡ TCN SUPERVISOR: Keeps temporal consistency over 512 steps.
    """
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        h = opt.hidden_dim
        layers = []
        # Dilations: 1, 2, ..., 256 -> cover 512 receptive field
        for d in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            layers.append(ConvBlock(h, h, dilation=d))
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
    ⚡ PIXEL-PERFECT DISCRIMINATOR:
    Matches CNN-CGAN exactly (Spectral Norm + LeakyReLU + Flatten).
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
            cb(h + cd, 32), cb(32, 64), cb(64, 128), cb(128, 256),
            nn.Conv1d(256, 1, opt.seq_len // 16, 1, 0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_seq, cond, sigmoid=True):
        # Combine hidden state and condition
        disc_input = torch.cat([h_seq, cond], dim=-1).transpose(1, 2)
        y_hat = self.conv(disc_input).view(-1, 1)
        return self.sigmoid(y_hat) if sigmoid else y_hat
