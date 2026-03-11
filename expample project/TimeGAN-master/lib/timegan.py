"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen(robinbg@foxmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .data import batch_generator
from utils import extract_time, random_generator, NormMinMax
from .model import Encoder, Recovery, Generator, Discriminator, Supervisor


class BaseModel():
  """ Base Model for timegan (C-TimeGAN Redesign)
  """

  def __init__(self, opt, data_tuple):
    # Seed for deterministic behavior
    self.seed(opt.manualseed)

    # Initalize variables.
    self.opt = opt
    # data_tuple = (targets, conditions)
    self.ori_data, self.ori_conds = data_tuple
    
    # Pre-calculate ranges for target data for renormalization
    targets_concat = np.concatenate(self.ori_data, axis=0)
    self.min_val = np.min(targets_concat, axis=0)
    self.max_val = np.max(targets_concat, axis=0)

    self.max_seq_len = self.opt.seq_len
    self.data_num = len(self.ori_data)
    self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

  def seed(self, seed_value):
    if seed_value == -1: return
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

  def save_weights(self, epoch):
    weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
    if not os.path.exists(weight_dir): os.makedirs(weight_dir)
    torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()}, '%s/netE.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()}, '%s/netR.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()}, '%s/netG.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()}, '%s/netD.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()}, '%s/netS.pth' % (weight_dir))


  def train_one_iter_er(self):
    self.nete.train()
    self.netr.train()
    self.X0, self.T, self.C0 = batch_generator(self.ori_data, self.ori_conds, self.opt.batch_size)
    self.X = torch.tensor(np.stack(self.X0), dtype=torch.float32).to(self.device).contiguous()
    self.C = torch.tensor(np.stack(self.C0), dtype=torch.float32).to(self.device).contiguous()
    self.optimize_params_er()

  def train_one_iter_er_(self):
    self.nete.train()
    self.netr.train()
    self.X0, self.T, self.C0 = batch_generator(self.ori_data, self.ori_conds, self.opt.batch_size)
    self.X = torch.tensor(np.stack(self.X0), dtype=torch.float32).to(self.device).contiguous()
    self.C = torch.tensor(np.stack(self.C0), dtype=torch.float32).to(self.device).contiguous()
    self.optimize_params_er_()
 
  def train_one_iter_s(self):
    self.nets.train()
    self.X0, self.T, self.C0 = batch_generator(self.ori_data, self.ori_conds, self.opt.batch_size)
    self.X = torch.tensor(np.stack(self.X0), dtype=torch.float32).to(self.device).contiguous()
    self.C = torch.tensor(np.stack(self.C0), dtype=torch.float32).to(self.device).contiguous()
    self.optimize_params_s()

  def train_one_iter_g(self):
    self.netg.train()
    self.X0, self.T, self.C0 = batch_generator(self.ori_data, self.ori_conds, self.opt.batch_size)
    self.X = torch.tensor(np.stack(self.X0), dtype=torch.float32).to(self.device).contiguous()
    self.C = torch.tensor(np.stack(self.C0), dtype=torch.float32).to(self.device).contiguous()
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)
    self.optimize_params_g()

  def train_one_iter_d(self):
    self.netd.train()
    self.X0, self.T, self.C0 = batch_generator(self.ori_data, self.ori_conds, self.opt.batch_size)
    self.X = torch.tensor(np.stack(self.X0), dtype=torch.float32).to(self.device).contiguous()
    self.C = torch.tensor(np.stack(self.C0), dtype=torch.float32).to(self.device).contiguous()
    self.Z = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)
    self.optimize_params_d()


  def train(self):
    from tqdm import tqdm
    pbar_er = tqdm(range(self.opt.iteration), desc='Encoder Pre-training')
    for iter in pbar_er:
      self.train_one_iter_er()
      pbar_er.set_postfix({'loss': f'{self.err_er.item():.4f}'})

    pbar_s = tqdm(range(self.opt.iteration), desc='Supervisor Pre-training')
    for iter in pbar_s:
      self.train_one_iter_s()
      pbar_s.set_postfix({'loss': f'{self.err_s.item():.4f}'})

    pbar_j = tqdm(range(self.opt.iteration), desc='Joint Training')
    for iter in pbar_j:
      for kk in range(2):
        self.train_one_iter_g()
        self.train_one_iter_er_()
      self.train_one_iter_d()
      pbar_j.set_postfix({'G_loss': f'{self.err_g.item():.4f}'})

    self.save_weights(self.opt.iteration)
    print('\nFinish C-TimeGAN Training')

  def generation(self, num_samples):
    if num_samples == 0: return None
    batch_size = self.opt.batch_size
    
    self.netg.eval()
    self.nets.eval()
    self.netr.eval()
    
    # ⚡ 100W DATA OPTIMIZATION: Zero-Spike Pre-allocation
    print(f"📦 Pre-allocating memory for {num_samples} samples...")
    out_array = np.zeros((num_samples, self.opt.seq_len, self.opt.z_dim), dtype=np.float32)
    
    iterations = (num_samples // batch_size) + (1 if num_samples % batch_size != 0 else 0)
    from tqdm import tqdm
    for i in tqdm(range(iterations), desc="Generating"):
        curr_batch_size = batch_size if i < iterations - 1 else num_samples - i * batch_size
        if curr_batch_size <= 0: break

        # Get real conditions (Aggregate) for context
        _, T_samples, C_mb = batch_generator(self.ori_data, self.ori_conds, curr_batch_size)
        
        Z = random_generator(curr_batch_size, self.opt.z_dim, T_samples, self.opt.seq_len)
        Z_tensor = torch.tensor(np.stack(Z), dtype=torch.float32).to(self.device).contiguous()
        C_tensor = torch.tensor(np.stack(C_mb), dtype=torch.float32).to(self.device).contiguous()
        
        with torch.no_grad():
            E_hat = self.netg(Z_tensor, C_tensor)
            H_hat = self.nets(E_hat)
            gen_batch = self.netr(H_hat).cpu().numpy()

        start_idx = i * batch_size
        for j in range(curr_batch_size):
            temp = gen_batch[j, :T_samples[j], :]
            # Re-normalize
            temp = temp * (self.max_val + 1e-7)
            temp = temp + self.min_val
            # Fill pre-allocated block
            out_array[start_idx + j, :T_samples[j], :] = temp
            
    return out_array


class TimeGAN(BaseModel):
    """TimeGAN Class (C-TimeGAN 2024 Redesign)
    """
    @property
    def name(self):
      return 'TimeGAN'

    def __init__(self, opt, data_tuple):
      super(TimeGAN, self).__init__(opt, data_tuple)

      self.nete = Encoder(self.opt).to(self.device)
      self.netr = Recovery(self.opt).to(self.device)
      self.netg = Generator(self.opt).to(self.device)
      self.netd = Discriminator(self.opt).to(self.device)
      self.nets = Supervisor(self.opt).to(self.device)

      if self.opt.resume != '':
        print("\nLoading pre-trained networks.")
        self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
        self.netr.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netR.pth'))['state_dict'])
        self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
        self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
        self.nets.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netS.pth'))['state_dict'])

      self.l_mse = nn.MSELoss()
      self.l_bce = nn.BCELoss()

      # Setup optimizer
      self.optimizer_e = optim.Adam(self.nete.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
      self.optimizer_r = optim.Adam(self.netr.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
      self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
      self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
      self.optimizer_s = optim.Adam(self.nets.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


    def forward_e(self):
      self.H = self.nete(self.X, self.C)

    def forward_er(self):
      self.H = self.nete(self.X, self.C)
      self.X_tilde = self.netr(self.H)

    def forward_g(self):
      self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device).contiguous()
      self.E_hat = self.netg(self.Z, self.C)

    def forward_dg(self):
      self.Y_fake = self.netd(self.H_hat, self.C)
      self.Y_fake_e = self.netd(self.E_hat, self.C)

    def forward_rg(self):
      self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
      self.H_supervise = self.nets(self.H)

    def forward_sg(self):
      self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
      self.Y_real = self.netd(self.H, self.C)
      self.Y_fake = self.netd(self.H_hat, self.C)
      self.Y_fake_e = self.netd(self.E_hat, self.C)

    def backward_er(self):
      self.err_er = self.l_mse(self.X_tilde, self.X)
      self.err_er.backward(retain_graph=True)

    def backward_er_(self):
      # Weighted MSE for Power column (Index 0)
      power_mse = self.l_mse(self.X_tilde[:, :, 0], self.X[:, :, 0])
      feat_mse = self.l_mse(self.X_tilde[:, :, 1:], self.X[:, :, 1:])
      self.err_er_raw = 10.0 * power_mse + 1.0 * feat_mse
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      
      self.err_er = 10 * torch.sqrt(self.err_er_raw) + 0.1 * self.err_s
      self.err_er.backward(retain_graph=True)

    def backward_g(self):
      # Adversarial (This is what actually learns the realistic shape!)
      self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))
      self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
      
      # Moments (V1 = Variance/Texture, V2 = Mean/Energy)
      # We INCREASE the weight on V1 to force it to learn the fluctuations (std dev)!
      real_std = torch.sqrt(torch.std(self.X,[0])[1] + 1e-6)
      fake_std = torch.sqrt(torch.std(self.X_hat,[0])[1] + 1e-6)
      self.err_g_V1 = torch.mean(torch.abs(fake_std - real_std))   
      self.err_g_V2 = torch.mean(torch.abs((torch.mean(self.X_hat,[0])[0]) - (torch.mean(self.X,[0])[0])))  
      
      # Difference/Texture Loss: Instead of rigid MSE which smooths things out,
      # we force the VARIANCE (roughness) of the fake signal's steps to match the real signal's steps.
      diff_real = self.X[:, 1:, 0] - self.X[:, :-1, 0]
      diff_fake = self.X_hat[:, 1:, 0] - self.X_hat[:, :-1, 0]
      
      # Match the "amount of jitter" (std of diff) rather than the exact exact placement
      texture_real = torch.std(diff_real, dim=1)
      texture_fake = torch.std(diff_fake, dim=1)
      self.err_g_texture = torch.mean(torch.abs(texture_fake - texture_real))

      # Supervisor
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      
      # Total G Loss
      # ⚠️ REMOVED err_g_tv completely because it forces flat blocks!
      # ⚠️ INCREASED Adversarial & Texture weight to encourage high-frequency learning
      self.err_g = self.err_g_U * 2.0 + \
                   self.err_g_U_e * self.opt.w_gamma + \
                   self.err_g_V1 * (self.opt.w_g * 5.0) + \
                   self.err_g_V2 * self.opt.w_g + \
                   10.0 * torch.sqrt(self.err_s) + \
                   5.0 * self.err_g_texture
                   
      self.err_g.backward(retain_graph=True)

    def backward_s(self):
      self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
      self.err_s.backward(retain_graph=True)

    def backward_d(self):
      self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
      self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
      self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
      self.err_d = self.err_d_real + self.err_d_fake + self.err_d_fake_e * self.opt.w_gamma
      if self.err_d > 0.15:
        self.err_d.backward(retain_graph=True)

    def optimize_params_er(self):
      self.forward_er()
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_er_(self):
      self.forward_er()
      self.forward_s()
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er_()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_s(self):
      self.forward_e()
      self.forward_s()
      self.optimizer_s.zero_grad()
      self.backward_s()
      self.optimizer_s.step()

    def optimize_params_g(self):
      self.forward_e()
      self.forward_s()
      self.forward_g()
      self.forward_sg()
      self.forward_rg()
      self.forward_dg()
      self.optimizer_g.zero_grad()
      self.optimizer_s.zero_grad()
      self.backward_g()
      self.optimizer_g.step()
      self.optimizer_s.step()

    def optimize_params_d(self):
      self.forward_e()
      self.forward_g()
      self.forward_sg()
      self.forward_d()
      self.forward_dg()
      self.optimizer_d.zero_grad()
      self.backward_d()
      self.optimizer_d.step()
