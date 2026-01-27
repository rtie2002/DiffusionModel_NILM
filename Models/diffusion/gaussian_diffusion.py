import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.diffusion.agent_transformer import Transformer
from Models.diffusion.model_utils import default, identity, extract


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def huber_loss_fn(input, target):
    return F.huber_loss(input, target,  delta=0.5)

def cosine_beta_schedule(timesteps, s=0.008):

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            condition_dim=8,  # NEW: Number of conditional features (time features)
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            # CBDM Parameters
            tau=0.1,      # Regularization weight
            gamma=0.25,   # Commitment weight
            on_threshold=15.0, # Power threshold to define "ON" state (Appliance specific)
            **kwargs
    ):
        super(Diffusion, self).__init__()
        
        self.tau = tau
        self.gamma = gamma
        self.on_threshold = on_threshold

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.condition_dim = condition_dim  # NEW: Store condition dimension
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        # NEW: Transformer now decouples feature_size (power) and condition_dim (time)
        self.model = Transformer(n_feat=feature_size, 
                                 n_channel=seq_length, 
                                 condition_dim=condition_dim,
                                 n_layer_enc=n_layer_enc, 
                                 n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, 
                                 attn_pdrop=attn_pd, 
                                 resid_pdrop=resid_pd, 
                                 mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, 
                                 n_embd=d_model, 
                                 conv_params=[kernel_size, padding_size], 
                                 **kwargs)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, padding_masks=None):
        # x shape: (B, L, 1+8) = (B, L, 9)
        trend, season = self.model(x, t, padding_masks=padding_masks)
        
        # The model now only predicts the power dimension (1D)
        power_pred = trend + season # (B, L, 1)
        
        # We MUST return all 9 dimensions for the diffusion loop, 
        # but the time features (conditions) must be the original ones from x.
        conditions = x[:, :, self.feature_size:] # (B, L, 8)
        model_output = torch.cat([power_pred, conditions], dim=-1) # (B, L, 9)
        
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        
        # CRITICAL FIX: Only add noise to power dimension, not time features!
        if t > 0:
            noise = torch.randn_like(x)
            # Zero out noise for time features (columns 1-8)
            if x.shape[-1] == self.feature_size + self.condition_dim:  # 9 dimensions
                noise[:, :, self.feature_size:] = 0  # No noise for time features!
        else:
            noise = 0.
        
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))

    @torch.no_grad()
    def generate_with_conditions(self, condition, batch_size=None):
        """
        Generate data with time features preserved (outputs 9 dimensions)
        
        Args:
            condition: (B, seq_length, condition_dim) - Time features to condition on
            batch_size: Optional, inferred from condition if not provided
        
        Returns:
            (B, seq_length, feature_size + condition_dim) - Generated power + time features
        """
        if condition is not None:
            batch_size = condition.shape[0]
            seq_length = condition.shape[1]
        else:
            seq_length = self.seq_length
            if batch_size is None:
                raise ValueError("Either condition or batch_size must be provided")
        
        # Initialize noise for full dimensions (power + time features)
        img = torch.randn(batch_size, seq_length, self.feature_size + self.condition_dim).to(condition.device)
        
        # Replace time feature part with actual conditions
        img[:, :, self.feature_size:] = condition
        
        # 🚀 REAL-TIME PROGRESS MONITOR (For RTX 4090 Denoising)
        from tqdm.auto import tqdm
        pbar = tqdm(reversed(range(0, self.num_timesteps)), 
                    total=self.num_timesteps, 
                    desc='[Denoising Step]', 
                    leave=False)

        # Reverse diffusion process
        for t in pbar:
            img, _ = self.p_sample(img, t)
            # Force time features to stay as conditions (prevent drift)
            img[:, :, self.feature_size:] = condition
        
        return img  # (B, seq_length, feature_size + condition_dim)



    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'huber':
            return huber_loss_fn
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        # NEW: Separate power and conditions
        # x_start shape: (B, seq_length, feature_size + condition_dim)
        x_power = x_start[:, :, :self.feature_size]  # Extract power (B, seq_length, 1)
        x_condition = x_start[:, :, self.feature_size:]  # Extract conditions (B, seq_length, 8)
        
        # Only noise the power part
        noise = default(noise, lambda: torch.randn_like(x_power))
        if target is None:
            target = x_power

        # Noise sample (only power)
        x_power_noisy = self.q_sample(x_start=x_power, t=t, noise=noise)
        
        # NEW: Concatenate noisy power with clean conditions
        x = torch.cat([x_power_noisy, x_condition], dim=-1)  # (B, seq_length, 9)
        
        model_out = self.output(x, t, padding_masks)
        
        # NEW: Only compute loss on power part
        model_out_power = model_out[:, :, :self.feature_size]

        train_loss = self.loss_fn(model_out_power, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            # NEW: Use only power part for Fourier loss
            fft1 = torch.fft.fft(model_out_power.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        
        main_loss = train_loss.mean()

        # 🚀 CBDM: CLASS-BALANCING REGULARIZATION
        # 1. Identify "ON" and "OFF" samples in the current batch
        # x_power range is roughly [0, 1] after normalization in CustomDataset, 
        # but the threshold in config is in Watts. 
        # For simplicity, we assume x_power is already standardized/normalized.
        # We classify based on mean power in the window.
        is_on = (x_power.mean(dim=(1, 2)) > self.on_threshold) 
        on_idx = torch.where(is_on)[0]
        off_idx = torch.where(~is_on)[0]

        cbdm_loss = torch.tensor(0.0, device=x_start.device)
        
        # We only apply CBDM if we have a mix of states in the batch
        if len(on_idx) > 0 and len(off_idx) > 0 and self.tau > 0:
            # 🚀 OPTIMIZED: Vectorized Balanced Sampling
            B = x_start.shape[0]
            # 50% chance to pick an ON condition, 50% chance to pick an OFF condition
            use_on = torch.rand(B, device=x_start.device) > 0.5
            
            # Select random indices from each set for the entire batch
            rand_on = on_idx[torch.randint(0, len(on_idx), (B,), device=x_start.device)]
            rand_off = off_idx[torch.randint(0, len(off_idx), (B,), device=x_start.device)]
            
            # Combine based on the 50/50 roll
            target_indices = torch.where(use_on, rand_on, rand_off)
            
            # 3. Get the balanced conditions (y_prime)
            x_condition_prime = x_condition[target_indices]
            x_prime = torch.cat([x_power_noisy, x_condition_prime], dim=-1)
            
            # 4. Forward pass with balanced conditions
            model_out_prime = self.output(x_prime, t, padding_masks)
            model_out_prime_power = model_out_prime[:, :, :self.feature_size]

            # 5. Compute Lr and Lrc (Algorithm 1, lines 8 & 10)
            # Lr = t * tau * ||eps(y) - sg(eps(y_prime))||^2
            # Lrc = t * tau * ||sg(eps(y)) - eps(y_prime))||^2
            # Note: At time step t, epsilon prediction vs target prediction are related.
            # We apply it to the model outputs (x_start predictions) as per Proposition 2.
            
            # Weight regularization by time step t (normalized)
            t_weight = (t.float() / self.num_timesteps).view(-1, 1, 1)
            
            lr = t_weight * self.tau * (model_out_power - model_out_prime_power.detach())**2
            lrc = t_weight * self.tau * self.gamma * (model_out_power.detach() - model_out_prime_power)**2
            
            cbdm_loss = (lr + lrc).mean()

        return main_loss + cbdm_loss

    def forward(self, x, condition=None, **kwargs):
        # x can be either:
        # - (B, seq_length, 9) from dataloader (power + time features)
        # - (B, seq_length, 1) + condition (B, seq_length, 8) for manual conditioning
        
        if condition is not None:
            # Manual conditioning: concatenate power and conditions
            x = torch.cat([x, condition], dim=-1)  # (B, seq_length, 9)
        
        b, c, n, device = *x.shape, x.device
        # Accept both 1 (unconditional) and 9 (conditional)
        assert n == self.feature_size or n == self.feature_size + self.condition_dim, \
            f'number of features must be {self.feature_size} (unconditional) or {self.feature_size + self.condition_dim} (conditional), got {n}'
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season = self.model(x, t)
        return trend, season

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        x_self_cond=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    

if __name__ == '__main__':
    pass
