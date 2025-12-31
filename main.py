import os
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
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        
        # Determine number of samples to generate
        if args.sample_num is not None:
            num_samples = args.sample_num
            print(f"Generating custom number of samples: {num_samples}")
        else:
            num_samples = len(dataset)
            print(f"Generating default number of samples: {num_samples} (dataset size)")
        
        samples = trainer.sample(num=num_samples, size_every=400, shape=[dataset.window, dataset.var_num], dataset=dataset)
        if dataset.auto_norm:
            # CRITICAL: Only normalize power column, preserve time features in [-1, 1]
            if dataset.var_num == 9:  # Conditional case (1 power + 8 time features)
                # Power column (index 0): [-1, 1] -> [0, 1]
                samples[:, :, 0:1] = unnormalize_to_zero_to_one(samples[:, :, 0:1])
                # Time features (indices 1-8): Keep in [-1, 1] for proper Sin/Cos representation
                print("Applied selective normalization:")
                print(f"  - Power (col 0): normalized to [0, 1]")
                print(f"  - Time features (cols 1-8): preserved in [-1, 1]")
            else:  # Unconditional case (1 power only)
                samples = unnormalize_to_zero_to_one(samples)
            
            print(f"Generated data shape: {samples.shape}")
            print(f"Total data points: {samples.size:,}")
            np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
