import os

# -----------------------------------------------------------------------------
# ⚡ SILENCE TENSORFLOW VERBOSITY (Must be before any other imports)
# -----------------------------------------------------------------------------
# "2" = ERROR only (hides INFO and WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# Disable oneDNN custom ops messages (fixes "oneDNN custom operations are on" spam)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# -----------------------------------------------------------------------------
import sys
import warnings
import torch
import torch._inductor.config
import argparse
import numpy as np
from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader
from Models.diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def apply_base_noise_filter(power_sequence, background_threshold=15.0, bridge_gap=30):
    """
    Zeroes out noise below the threshold globally, but PROTECTS temporary dips 
    that occur inside an active appliance cycle by bridging gaps forward.
    """
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    
    # 1. Identify raw active points
    is_active = (power_sequence >= background_threshold).astype(int)
    
    # 2. Bridge small gaps (protect internal dips)
    for i in range(1, n - bridge_gap):
        if is_active[i-1] == 1 and is_active[i] == 0:
            upcoming = is_active[i:i+bridge_gap]
            if np.any(upcoming == 1):
                next_active = np.where(upcoming == 1)[0][0]
                is_active[i:i+next_active] = 1
                
    # 3. Apply mask to zero out ONLY the TRUE background noise outside of cycles
    power_sequence[is_active == 0] = 0.0
    
    return power_sequence


def remove_isolated_spikes(power_sequence, window_size=5, spike_threshold=3.0, 
                          background_threshold=50):
    """
    Remove isolated spikes that appear in the middle of OFF periods.
    Works entirely in WATTS space.
    """
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    num_spikes = 0
    
    # Pad array for edge handling
    half_window = window_size // 2
    padded = np.pad(power_sequence, half_window, mode='edge')
    
    for i in range(n):
        current_value = power_sequence[i]
        # Skip checking if already silently low
        if current_value < max(1.0, background_threshold * 0.1): 
            continue
        
        # Get surrounding values (excluding center point)
        window_start, window_end = i, i + window_size
        window = padded[window_start:window_end]
        surrounding = np.concatenate([window[:half_window], window[half_window+1:]])
        
        median_surrounding = np.median(surrounding)
        
        if current_value > background_threshold:
            # Check if surroundings are mostly 'OFF' (near zero, e.g. < 15W)
            is_background_quiet = np.all(surrounding < 15.0)
            
            if is_background_quiet and current_value > spike_threshold * (median_surrounding + 1.0):
                power_sequence[i] = 0
                num_spikes += 1
                
    return power_sequence, num_spikes

def validate_full_cycles(power_sequence, background_threshold=15.0, 
                        min_peak=1000.0, bridge_gap=20, min_duration=80):
    """
    Cycle-wise Validation.
    Groups clusters and zeroes out those without a required Watts signature.
    Works entirely in WATTS space.
    """
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    is_active = (power_sequence >= background_threshold).astype(int)
    
    # Bridge small gaps
    for i in range(1, n - bridge_gap):
        if is_active[i-1] == 1 and is_active[i] == 0:
            upcoming = is_active[i:i+bridge_gap]
            if np.any(upcoming == 1):
                next_active = np.where(upcoming == 1)[0][0]
                is_active[i:i+next_active] = 1

    # Analyze segments
    diff = np.diff(np.concatenate(([0], is_active, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    num_fake_segments = 0
    for start, end in zip(starts, ends):
        segment = power_sequence[start:end]
        # Kill if peak is too low OR if totally duration is too short to be a real cycle
        if np.max(segment) < min_peak or (end-start) < min_duration:
            power_sequence[start:end] = 0
            num_fake_segments += 1
                
    return power_sequence, num_fake_segments


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config', type=str, default=None,
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=2024,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')

    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--sample_num', type=int, default=None,
                        help='Number of samples to generate. If None, uses len(dataset). '
                             'Set to match real data size or any custom value.')
    parser.add_argument('--milestone', type=int, default=1000)
    parser.add_argument(
        '--opts',
        nargs='+',
        default=None,
        help='Optional key-value overrides like dataloader.batch_size 32'
    )

    parser.add_argument('--sampling_mode', type=str, default='ordered_non_overlapping',
                        choices=['random', 'ordered', 'ordered_non_overlapping'],
                        help='Sampling mode: random (default), ordered (sequential overlap), or ordered_non_overlapping (sequential non-overlap)')

    parser.add_argument('--cfg_scale', type=float, default=None,
                        help='Classifier-Free Guidance Scale (1.0 = No CFG)')

    args = parser.parse_args()
    
    # Get the directory where main.py is located (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If output is a relative path, make it relative to script_dir
    if not os.path.isabs(args.output):
        output_dir = os.path.join(script_dir, args.output)
    else:
        output_dir = args.output
    
    # Create full save directory path
    args.save_dir = os.path.join(output_dir, f'{args.name}')
    
    # Create directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but --gpu was specified. Remove --gpu or install a CUDA-enabled PyTorch build.")
        torch.cuda.set_device(args.gpu)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Display device information
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"✓ Using device: CUDA (GPU)")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print(f"⚠ Using device: CPU (No GPU detected)")
        print(f"  Warning: Training will be MUCH slower on CPU!")
    print("=" * 60)
    print()

    if args.config is None:
        raise ValueError("Missing --config argument. Provide a YAML config path (e.g. Config/microwave.yaml).")

    config = load_yaml_config(args.config)
    config = merge_opts_to_config(config, args.opts)
    
    # --- 🔥 CFG OVERRIDE ---
    # Allow command line --cfg_scale to override YAML config
    if args.cfg_scale is not None:
        if 'model' in config and 'params' in config['model']:
            config['model']['params']['guidance_scale'] = args.cfg_scale
            print(f"⚡ Command Line Override: Setting Guidance Scale to {args.cfg_scale}")

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).to(device)
    
    # --- 🚀 STABILITY MODE: RTX 4090 Native (Eager) ---
    # Disabled torch.compile due to persistent stride compatibility issues.
    # The system is still heavily accelerated by BF16, Fused Adam, and optimized Batches.
    if sys.platform != 'win32':
        print("🚀 WSL2 Detected: Running in Native High-Speed Mode (Eager).")
        # Enable CUDNN Benchmark for finding best convolution algo
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for matmuls (huge speedup on Ampere/Ada)
        torch.set_float32_matmul_precision('high')
    else:
        print("💡 Windows Detected: Running in Standard Mode.")
    
    # ⚡ EFFICIENCY FIX: Only build the heavy training dataloader if we are actually training.
    # This prevents creating millions of sliding windows and applying booster/jitter for training 
    # when we only intended to sample.
    if args.train:
        dataloader_info = build_dataloader(config, args)
    else:
        # Provide a minimal placeholder to avoid Trainer initialization failure
        dataloader_info = {
            'dataloader': [], 
            'dataset': None
        }
    
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    else:
        trainer.load(args.milestone)
        
        
        # CRITICAL: For sampling, use FULL dataset (100%) to ensure complete temporal coverage
        # Training uses 80/20 split, but generation should access ALL months (1-12)
        

        # Create a separate dataset with proportion=1.0 (no train/test split)
        sampling_dataset_config = config['dataloader']['train_dataset'].copy()
        sampling_dataset_config['params']['proportion'] = 0.0  # Use 0.0 to put 100% of data into the inference/test set
        sampling_dataset_config['params']['style'] = 'non_overlapping'  # CRITICAL: Use non-overlapping for sampling
        sampling_dataset_config['params']['save2npy'] = False  # Don't save, just for sampling
        sampling_dataset_config['params']['period'] = 'test'   # FORCE test period to disable booster during sampling
        sampling_dataset = instantiate_from_config(sampling_dataset_config)
        
        
        print(f"✓ Full dataset loaded for sampling: {len(sampling_dataset)} windows (ensures 1-12 month coverage)\n")
        

        # Use this full dataset for sampling
        dataset = sampling_dataset
        
        # Determine number of samples and stride based on mode
        stride = 1
        ordered = False
        
        if args.sampling_mode == 'ordered_non_overlapping':
            print("Mode: Ordered Non-Overlapping (Sequential)")
            # CRITICAL: Dataset is already non-overlapping blocks, so we step 1 by 1
            stride = 1 
            ordered = True
            
            # Verify coverage matches dataset size
            max_windows = len(dataset)

            
            if args.sample_num is not None:
                num_samples = args.sample_num
                if num_samples > max_windows:
                    print(f"Warning: Requested {num_samples} samples, but dataset only fits {max_windows} non-overlapping windows.")
            else:
                num_samples = max_windows
                print(f"Auto-calculated samples for full coverage: {num_samples}")
                
        elif args.sampling_mode == 'ordered':
            print("Mode: Ordered (Sequential, Sliding Window)")
            stride = 1
            ordered = True
            if args.sample_num is not None:
                num_samples = args.sample_num
            else:
                num_samples = len(dataset)
                
        else: # random
            print("Mode: Random (Random Sampling from Full Dataset)")
            stride = 1
            ordered = False
            if args.sample_num is not None:
                num_samples = args.sample_num
            else:
                num_samples = 2500 # Default reasonable number for random sampling if not specified
                print(f"Generating default number of samples: {num_samples}")

        
        # Call trainer.sample with updated arguments (1000 samples max per batch for 4090)
        samples = trainer.sample(num=num_samples, size_every=1000, shape=[dataset.window, dataset.var_num], 
                                dataset=dataset, ordered=ordered, stride=stride)
        
        if dataset.auto_norm:
            # 1. Get shape information from generated samples
            # This is critical to define 'V' before using it in the check below
            N, L, V = samples.shape
            
            if V == 9:
                print("Applying recovery for Multivariate (1 Power + 8 Time features)...")
                # Split power and time - BOTH are in [-1, 1] range from model
                power = samples[:, :, 0:1] # (N, L, 1)
                time_feats = samples[:, :, 1:9] # (N, L, 8) - Keep as is in [-1, 1]
                
                # 2. Unnormalize ONLY power to [0, 1] for inverse_transform
                power_01 = unnormalize_to_zero_to_one(power)
                
                # Reshape for scaler
                power_flat = power_01.reshape(-1, 1)
                power_recovered = dataset.scaler.inverse_transform(power_flat)
                power_recovered = power_recovered.reshape(N, L, 1)
                
                # 3. Recombine: Recovered Power + Original [-1, 1] Time Features
                samples = np.concatenate([power_recovered, time_feats], axis=2)
            else:
                # Univariate case
                samples = unnormalize_to_zero_to_one(samples)
                samples_flat = samples.reshape(-1, V)
                samples_recovered = dataset.scaler.inverse_transform(samples_flat)
                samples = samples_recovered.reshape(N, L, V)

            print(f"Generated data shape: {samples.shape}")
            print(f"Data Unnormalized Range: {samples[:,:,0].min():.4f} to {samples[:,:,0].max():.4f}W")
            
            # --- TASK 1: SAVE ORIGINAL (RAW) NPY ---
            raw_path = os.path.join(args.save_dir, f'ddpm_fake_{args.name}_raw.npy')
            np.save(raw_path, samples)
            print(f"✓ Saved original RAW samples to: {raw_path}")
            # --- TASK 2: FILTER NOISE (Apply Algorithm 1 Logic & EWMA) ---
            try:
                # Load configuration without importing the external postprocess file
                import yaml
                config_path = os.path.join(os.path.dirname(__file__), 'Config/preprocess/preprocess_multivariate.yaml')
                with open(config_path, 'r') as f:
                    specs = yaml.safe_load(f)
                
                if specs and 'appliances' in specs and args.name in specs['appliances']:
                    print(f"\n🚀 Applying Post-Processing Filters for {args.name}...")
                    app_specs = specs['appliances'][args.name]
                    max_power = app_specs.get('max_power', 1000.0)
                    
                    noise_thres_watts = app_specs.get('on_power_threshold', 15.0)
                    clip_max_watts = app_specs.get('max_power_clip', None)
                    
                    # Work on the power sequence
                    N, L, V = samples.shape
                    
                    # Store original time features since we only filter power
                    if V > 1:
                        time_feats = samples[:, :, 1:].copy()
                    
                    # Flatten power for 1D filtering functions and CONVERT TO WATTS
                    power_seq_minmax = samples[:, :, 0].flatten()
                    power_seq_watts = power_seq_minmax * max_power
                    
                    # --- ROUND 1: Base Noise Filter (Smart Masking) ---
                    print(f"  ✓ Round 1: Smart Noise Filter (< {noise_thres_watts}W protected inside cycles)...")
                    power_seq_watts = apply_base_noise_filter(power_seq_watts, background_threshold=noise_thres_watts, bridge_gap=30)
                    
                    # --- ROUND 2: Appliance-Specific Advanced Filters ---
                    if args.name.lower() == 'washingmachine':
                        print(f"  ✓ Round 2: Searching for isolated spikes (using {noise_thres_watts}W BG)...")
                        power_seq_watts, n_spikes = remove_isolated_spikes(
                            power_seq_watts, window_size=5, spike_threshold=3.0, background_threshold=noise_thres_watts
                        )
                        if n_spikes > 0:
                            print(f"    - Cleaned {n_spikes} isolated glitches.")
                            
                        print("  ✓ Round 2: Validating Washing Machine signature...")
                        power_seq_watts, n_fake = validate_full_cycles(
                            power_seq_watts, background_threshold=noise_thres_watts, 
                            min_peak=1000.0, bridge_gap=20, min_duration=80
                        )
                        if n_fake > 0:
                            print(f"    - Removed {n_fake} fake cycles without 1000W peaks.")
                    
                    # --- ROUND 3: Hard Clip ---
                    if clip_max_watts is not None:
                        print(f"  ✓ Round 3: Clipping max power to {clip_max_watts}W...")
                        power_seq_watts = np.clip(power_seq_watts, 0.0, clip_max_watts)
                    
                    # Put filtered power back into samples array (CONVERT BACK TO MINMAX)
                    power_seq_minmax = power_seq_watts / max_power
                    samples[:, :, 0] = power_seq_minmax.reshape(N, L)
                    
                    # Restore time features just to be safe
                    if V > 1:
                        samples[:, :, 1:] = time_feats
                    
            except Exception as e:
                print(f"⚠️ Could not apply noise filters automatically: {e}")

            # --- TASK 3: SAVE FILTERED NPY ---
            filt_path = os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy')
            np.save(filt_path, samples)
            print(f"✓ Saved FILTERED samples to: {filt_path}")

if __name__ == '__main__':
    main()
