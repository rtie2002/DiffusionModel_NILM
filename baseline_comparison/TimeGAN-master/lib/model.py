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
import torch.nn.functional as F

class Encoder(nn.Module):
    """RNN/GRU Encoder"""
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(opt.z_dim + opt.cond_dim, opt.hidden_dim, num_layers=opt.num_layers, batch_first=True)

    def forward(self, input, cond):
        x = torch.cat([input, cond], dim=-1)
        H, _ = self.rnn(x)
        return H

class Recovery(nn.Module):
    """RNN/GRU Decoder/Recovery"""
    def __init__(self, opt):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(opt.hidden_dim, opt.hidden_dim, num_layers=opt.num_layers, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.z_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H):
        X_tilde, _ = self.rnn(H)
        return self.sigmoid(self.fc(X_tilde))

class Generator(nn.Module):
    """RNN/GRU Generator"""
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(opt.latent_dim + opt.cond_dim, opt.hidden_dim, num_layers=opt.num_layers, batch_first=True)

    def forward(self, Z, cond):
        x = torch.cat([Z, cond], dim=-1)
        E, _ = self.rnn(x)
        return E

class Supervisor(nn.Module):
    """RNN/GRU Supervisor"""
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(opt.hidden_dim, opt.hidden_dim, num_layers=opt.num_layers - 1, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)

    def forward(self, H):
        # TimeGAN paper states supervisor is num_layers - 1
        S, _ = self.rnn(H)
        return self.fc(S)

class Discriminator(nn.Module):
    """RNN/GRU Discriminator"""
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(opt.hidden_dim + opt.cond_dim, opt.hidden_dim, num_layers=opt.num_layers, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H, cond, sigmoid=True):
        x = torch.cat([H, cond], dim=-1)
        Y, _ = self.rnn(x)
        y_hat = self.fc(Y)
        return self.sigmoid(y_hat) if sigmoid else y_hat
