import os
import sys
import warnings

# 1. SUPPRESS ALL INFRASTRUCTURE WARNINGS (Clean Console)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import argparse
import numpy as np
from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader
from Models.diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


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

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).to(device)
    
    # 🚀 STABILITY FIRST: Disabling torch.compile to prevent stride AssertionErrors
    # The current AgentTransformer architecture's complex views/einops are incompatible 
    # with the current Triton compiler. We run in Eager mode for 100% reliability.
    if sys.platform != 'win32':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print("🚀 Linux Detected: Running in High-Performance Eager Mode (Stable).")

    else:
        print("💡 Windows Detected: Running in Standard Mode.")
    
    # ⚡ EFFICIENCY FIX: Only build the heavy training dataloader if we are actually training.
    if args.train:
        dataloader_info = build_dataloader(config, args)
    else:
        dataloader_info = {'dataloader': [], 'dataset': None}
    
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    # 3. Also compile the EMA Transformer core safely for Sampling
    if sys.platform != 'win32':
        try:
            if hasattr(trainer.ema.ema_model, 'model'):
                print("  -> Compiling sampling core (EMA) with Safe-Triton...")
                trainer.ema.ema_model.model = torch.compile(trainer.ema.ema_model.model, mode='default')
        except Exception as e:
            print(f"  ⚠ EMA Compile failed: {e}")

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

        
        # Call trainer.sample with updated arguments (size_every=1000 Optimized for 4090)
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
                # This is the correct step for MinMaxScaler!
                power_01 = unnormalize_to_zero_to_one(power)
                
                # Reshape for scaler
                power_flat = power_01.reshape(-1, 1)
                power_recovered = dataset.scaler.inverse_transform(power_flat)
                power_recovered = power_recovered.reshape(N, L, 1)
                
                # 3. Recombine: Recovered Power + Original [-1, 1] Time Features
                samples = np.concatenate([power_recovered, time_feats], axis=2)
            else:
                # Univariate case
                samples_01 = unnormalize_to_zero_to_one(samples)
                samples_flat = samples_01.reshape(-1, V)
                samples_recovered = dataset.scaler.inverse_transform(samples_flat)
                samples = samples_recovered.reshape(N, L, V)

            print(f"Generated data shape: {samples.shape}")
            print(f"Data Unnormalized Range: {samples[:,:,0].min():.4f} to {samples[:,:,0].max():.4f}W")
            np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
