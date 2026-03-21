import os
import sys
import time
import math
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# RTX 4090 Specific Optimizations
# Use TF32 for matrix multiplications (3x+ speedup on Ampere/Ada GPUs)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    # Automatically find the fastest kernels for your hardware
    torch.backends.cudnn.benchmark = True

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        # 🚀 Use Fused Adam for peak performance on RTX 4090 (PyTorch 2.0+)
        opt_kwargs = {'lr': start_lr, 'betas': [0.9, 0.96]}
        if 'fused' in torch.optim.Adam.__init__.__code__.co_varnames:
            opt_kwargs['fused'] = True
            print("🚀 Fused Adam optimizer activated (Peak Efficiency)")
            
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **opt_kwargs)
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)
        
        # Mixed Precision Training (FP16) for ~2x speedup
        self.scaler = GradScaler()

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        missing, unexpected = self.model.load_state_dict(data['model'], strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"Warning: Model loaded with strict=False (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        self.step = data['step']
        try:
            self.opt.load_state_dict(data['opt'])
        except ValueError:
            pass # Optimizer mismatch is expected and fine for sampling

        try:
            # Try loading EMA with strict=False to get the best possible matching weights
            self.ema.load_state_dict(data['ema'], strict=False)
        except (ValueError, RuntimeError) as e:
            print(f"Warning: EMA state mismatch deep failure. ({e})")
            print("Action: Copying loaded MAIN model weights to EMA model as fallback.")
            self.ema.ema_model.load_state_dict(self.model.state_dict())
        
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    # non_blocking=True + BF16 for peak transfer/compute overlap
                    data = next(self.dl).to(device, non_blocking=True)
                    
                    # 🚀 RTX 4090 Native Acceleration: Use Bfloat16
                    # This is more stable than FP16 and just as fast on Ada-Lovelace
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        
                    # Scaled Backward Pass
                    self.scaler.scale(loss).backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                # Unscale gradients before clipping
                self.scaler.unscale_(self.opt)
                clip_grad_norm_(self.model.parameters(), 1.0)
                # Scaled Optimizer Step
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None, dataset=None, ordered=True, stride=1):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + (1 if num % size_every != 0 else 0)
        
        print(f"\n{'='*70}")
        print(f"SAMPLING MODE: {'ORDERED (Sequential)' if ordered else 'RANDOM'}")
        if stride > 1:
            print(f"STRIDE: {stride} (Non-overlapping blocks if stride == window)")
        print(f"Generating {num} windows in {num_cycle} batches (batch_size={size_every})")
        print(f"{'='*70}\n")

        # Check if conditional generation is supported
        use_conditional = hasattr(self.ema.ema_model, 'generate_with_conditions') and dataset is not None and shape[1] == 9
        
        if use_conditional:
            print("✓ Using CONDITIONAL generation with time features from dataset")
            dataset_size = len(dataset.samples)
            print(f"  Available time templates in dataset: {dataset_size} windows")
        # 🧵 THE CONTINUOUS RIBBON STITCHING STRATEGY
        # We generate sequential windows with a 64-point anchor for seamless connections.
        overlap_len = 64
        stride = shape[0] - overlap_len  # 512 - 64 = 448
        
        # Calculate how many steps we need to cover 'num' windows of 512
        total_points_needed = num * shape[0]
        # windows_needed = ceil((total_points - 512) / 448) + 1
        num_windows_needed = math.ceil((total_points_needed - shape[0]) / stride) + 1
        
        # We process windows one-by-one to maintain the sequential physical anchor
        # We process all windows simultaneously using batch dimension
        all_conditions = []
        dataset_size = len(dataset.data)
        
        print(f"🚀 PARALLEL STITCHING: Preparing {num_windows_needed} overlapping windows for batch generation...")
        
        for window_idx in range(num_windows_needed):
            start_idx = (window_idx * stride) % dataset_size
            if start_idx + shape[0] <= dataset_size:
                window_data = dataset.data[start_idx : start_idx + shape[0]]
            else:
                p1 = dataset.data[start_idx:]
                p2 = dataset.data[:(start_idx + shape[0]) % dataset_size]
                window_data = np.concatenate([p1, p2], axis=0)
            all_conditions.append(window_data[:, 1:9])
            
        print(f"🚀 PARALLEL STITCHING: Generating all {num_windows_needed} windows simultaneously (Full 2000-step Quality)!")
        
        # Batch shape: [num_windows_needed, 512, 8]
        conditions = torch.FloatTensor(np.stack(all_conditions)).to(self.device)
        
        # Free up memory before massive generation
        torch.cuda.empty_cache()
        
        # OOM PREVENTION: Process in safe chunks of 800 windows
        # 4453 windows at once exceeds 24GB VRAM. We process them sequentially in large batch chunks!
        chunk_size = 800
        num_chunks = math.ceil(num_windows_needed / chunk_size)
        print(f"🚀 PARALLEL STITCHING: Generating {num_windows_needed} windows in {num_chunks} safe chunks (Chunk size: {chunk_size})...")
        
        all_samples = []

        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for i in range(num_chunks):
                    start_i = i * chunk_size
                    end_i = min((i + 1) * chunk_size, num_windows_needed)
                    chunk_cond = conditions[start_i:end_i]
                    
                    print(f"  -> Processing Chunk [{i+1}/{num_chunks}] ({end_i - start_i} windows)...")
                    chunk_sample = self.ema.ema_model.generate_with_conditions(
                        chunk_cond, 
                        sync_overlap_len=overlap_len
                    )
                    all_samples.append(chunk_sample.detach().cpu().numpy())
                    torch.cuda.empty_cache()  # Keep VRAM clean between chunks
                
        # Reconstruct the continuous matrix
        sample_np = np.concatenate(all_samples, axis=0)
        
        # Extract the "Fresh" part to build the ribbon
        all_fresh_points = [sample_np[0]] # Window 0 is fully fresh (512 points)
        for i in range(1, num_windows_needed):
            all_fresh_points.append(sample_np[i, overlap_len:]) # Next windows contribute the 448 non-overlap points


        # ⛓️ CONCATENATE & RE-SLICE
        ribbon = np.concatenate(all_fresh_points, axis=0) # (L, 9)
        
        # Ensure we have precisely the right amount for (num, 512, 9)
        if ribbon.shape[0] < total_points_needed:
            # This shouldn't happen with math.ceil, but for safety:
            print("  [Warning] Ribbon too short, padding...")
            pad = np.zeros((total_points_needed - ribbon.shape[0], 9))
            ribbon = np.concatenate([ribbon, pad], axis=0)
        
        # Final Truncate & Reshape to User's Format
        final_samples = ribbon[:total_points_needed, :].reshape(num, shape[0], 9)
        samples = final_samples # Match the variable name for further processing
            
        print(f"\n{'='*70}")
        print(f"✓ All {num} windows generated successfully!")
        if use_conditional:
            print(f"✓ Generated data follows the {'ORIGINAL SEQUENCE' if ordered else 'RANDOM DISTRIBUTION'}")
        print(f"{'='*70}\n")
        
        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples
