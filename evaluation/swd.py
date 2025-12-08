"""
Sliced Wasserstein Distance (SWD) for NILM Evaluation

Standalone implementation of Sliced Wasserstein Distance metric.
No external dependencies required beyond PyTorch.

This metric evaluates the quality of synthetic NILM data by comparing
distributions using the Sliced Wasserstein Distance.

References:
    - Rabin et al. "Wasserstein Barycenter and Its Application to Texture Mixing" (2011)
    - Kolouri et al. "Sliced Wasserstein Distance for Learning Gaussian Mixture Models" (2018)
    - EBSW: https://github.com/khainb/EBSW (reference only, not imported)

Usage:
    # Evaluate all appliances
    python evaluation/swd.py --all
    
    # Evaluate single appliance
    python evaluation/swd.py --appliance fridge
    
    # Custom paths
    python evaluation/swd.py --appliance kettle --real_path path/to/real.npy --synthetic_path path/to/synthetic.npy
"""

import numpy as np
import torch
import argparse
import os


def sliced_wasserstein_distance(X, Y, num_projections=50, p=2, device='cpu'):
    """
    Compute Sliced Wasserstein Distance between two distributions.
    
    The Sliced Wasserstein Distance projects the high-dimensional distributions
    onto random 1D lines and computes the Wasserstein distance on each projection.
    The final SWD is the average of these 1D Wasserstein distances.
    
    Mathematical Formula:
        SWD_p(X, Y) = [ (1/K) Σ_{k=1}^K W_p(X_θk, Y_θk)^p ]^(1/p)
    
    where:
        - θk are random unit vectors (projections)
        - W_p is the p-Wasserstein distance
        - X_θk, Y_θk are 1D projections of X, Y onto θk
    
    Args:
        X: Real data, shape (n_samples, n_features) or torch.Tensor
        Y: Synthetic data, shape (n_samples, n_features) or torch.Tensor
        num_projections: Number of random projections (default: 50)
        p: Power for Wasserstein distance (default: 2)
        device: 'cpu' or 'cuda'
    
    Returns:
        swd_score: Sliced Wasserstein Distance (float)
    """
    # Convert to PyTorch tensors
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(Y, torch.Tensor):
        Y = torch.from_numpy(Y).float()
    
    # Move to device
    X = X.to(device)
    Y = Y.to(device)
    
    # Ensure same number of samples
    n_samples = min(X.shape[0], Y.shape[0])
    X = X[:n_samples]
    Y = Y[:n_samples]
    
    # Get dimensions
    n_features = X.shape[1]
    
    # Generate random projections on unit sphere
    # Shape: (num_projections, n_features)
    projections = torch.randn(num_projections, n_features, device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    
    # Project data onto random directions
    # X_proj, Y_proj: (n_samples, num_projections)
    X_proj = torch.matmul(X, projections.T)
    Y_proj = torch.matmul(Y, projections.T)
    
    # Sort projections (required for Wasserstein distance)
    X_proj_sorted = torch.sort(X_proj, dim=0)[0]  # (n_samples, num_projections)
    Y_proj_sorted = torch.sort(Y_proj, dim=0)[0]
    
    # Compute 1D Wasserstein distance for each projection
    # W_p = [ (1/n) Σ |sort(X)[i] - sort(Y)[i]|^p ]^(1/p)
    diff = X_proj_sorted - Y_proj_sorted
    wasserstein_p = torch.mean(torch.abs(diff) ** p, dim=0)  # (num_projections,) - MEAN over samples
    
    # Average over all projections and take p-th root
    swd = torch.pow(wasserstein_p.mean(), 1.0 / p)
    
    return swd.item()


def _sliced_wasserstein_distance_deprecated(X, Y, num_projections=50, p=2, device='cpu'):
    """
    Simple Sliced Wasserstein Distance implementation.
    
    Args:
        X: shape (1, n_samples, n_features)
        Y: shape (1, n_samples, n_features)
        num_projections: Number of random projections
        p: Power for Wasserstein distance
        device: 'cpu' or 'cuda'
    
    Returns:
        swd: Scalar SWD value
    """
    batch_size, n_samples, dim = X.shape
    
    # Random projections on unit sphere
    projections = torch.randn(batch_size, num_projections, dim, device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=2, keepdim=True))
    
    # Project data
    X_proj = torch.bmm(X, projections.transpose(1, 2))  # (batch, n_samples, num_proj)
    Y_proj = torch.bmm(Y, projections.transpose(1, 2))
    
    # Sort projections
    X_proj_sorted = torch.sort(X_proj.transpose(1, 2))[0]  # (batch, num_proj, n_samples)
    Y_proj_sorted = torch.sort(Y_proj.transpose(1, 2))[0]
    
    # Wasserstein distance for each projection
    # Use MEAN instead of SUM - this is the correct formulation
    diff = X_proj_sorted - Y_proj_sorted
    wasserstein_p = torch.mean(torch.abs(diff) ** p, dim=2)  # (batch, num_proj) - MEAN over samples
    
    # Average over projections, then take p-th root
    swd = torch.pow(wasserstein_p.mean(dim=1), 1.0 / p).mean()
    
    return swd.item()


def load_real_data(appliance_name, data_root='OUTPUT'):
    """Load real ground truth data (normalized [0,1])."""
    appliance_folder = f"{appliance_name}_512"
    samples_dir = os.path.join(data_root, appliance_folder, 'samples')
    
    patterns = [
        f"{appliance_name.capitalize()}_norm_truth_512_train.npy",
        f"{appliance_name}_norm_truth_512_train.npy",
    ]
    
    for pattern in patterns:
        filepath = os.path.join(samples_dir, pattern)
        if os.path.exists(filepath):
            data = np.load(filepath)
            print(f"[OK] Loaded real data: {filepath}")
            print(f"  Shape: {data.shape}, Range: [{data.min():.4f}, {data.max():.4f}]")
            return data
    
    raise FileNotFoundError(f"Real data not found for {appliance_name} in {samples_dir}")


def load_synthetic_data(appliance_name, output_root='OUTPUT'):
    """Load synthetic data from diffusion model."""
    patterns = [
        f"{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy",
        f"{appliance_name}/ddpm_fake_{appliance_name}.npy",
    ]
    
    for pattern in patterns:
        filepath = os.path.join(output_root, pattern)
        if os.path.exists(filepath):
            data = np.load(filepath)
            print(f"[OK] Loaded synthetic data: {filepath}")
            print(f"  Shape: {data.shape}, Range: [{data.min():.4f}, {data.max():.4f}]")
            return data
    
    raise FileNotFoundError(f"Synthetic data not found for {appliance_name}")


def evaluate_swd(appliance_name, real_path=None, synthetic_path=None, 
                 num_projections=50, verbose=True):
    """
    Evaluate SWD score for synthetic data.
    
    Args:
        appliance_name: Name of appliance
        real_path: Custom path to real data (optional)
        synthetic_path: Custom path to synthetic data (optional)
        num_projections: Number of random projections (default: 50)
        verbose: Print detailed info
    
    Returns:
        swd_score: SWD score (float)
    """
    if verbose:
        print("=" * 70)
        print(f"SWD Evaluation: {appliance_name.upper()}")
        print("=" * 70)
        print()
    
    # Load data
    if real_path:
        real_data = np.load(real_path)
        if verbose:
            print(f"[OK] Loaded real data: {real_path}")
    else:
        real_data = load_real_data(appliance_name)
    
    if synthetic_path:
        synthetic_data = np.load(synthetic_path)
        if verbose:
            print(f"[OK] Loaded synthetic data: {synthetic_path}")
    else:
        synthetic_data = load_synthetic_data(appliance_name)
    
    if verbose:
        print()
    
    # Use same number of samples
    n_samples = min(len(real_data), len(synthetic_data))
    real_data = real_data[:n_samples]
    synthetic_data = synthetic_data[:n_samples]
    
    if verbose:
        print(f"Using {n_samples} samples for comparison")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        print()
    
    # Flatten to 2D: (n_samples, features)
    real_features = real_data.reshape(n_samples, -1)
    synthetic_features = synthetic_data.reshape(n_samples, -1)
    
    if verbose:
        print(f"Feature shape: ({n_samples}, {real_features.shape[1]})")
        print(f"Computing SWD with {num_projections} projections...")
        print()
    
    # Compute SWD
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    swd_score = sliced_wasserstein_distance(
        real_features, 
        synthetic_features, 
        num_projections=num_projections,
        device=device
    )
    
    if verbose:
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"SWD Score: {swd_score:.4f}")
        print()
        print("Interpretation (calibrated for [0,1] normalized NILM data):")
        print("  Lower is better (0 = perfect match)")
        print("  < 0.05   : Excellent - Same appliance with high-quality synthesis")
        print("  0.05-0.10: Good/Borderline - Same appliance or quality concerns")
        print("  0.10-0.20: Fair - Likely different appliances")
        print("  > 0.20   : Poor - Definitely different appliances")
        print()
        print("  NOTE: SWD has limited discrimination for NILM [0,1] data.")
        print("        Use FID as primary metric. SWD provides supplementary info.")
        print("=" * 70)
    
    return swd_score


def detect_appliances(output_root='OUTPUT'):
    """Auto-detect appliances with synthetic data."""
    appliances = []
    
    if not os.path.exists(output_root):
        return appliances
    
    for item in os.listdir(output_root):
        item_path = os.path.join(output_root, item)
        if os.path.isdir(item_path) and item.endswith('_512'):
            appliance = item.replace('_512', '')
            
            synthetic_patterns = [
                f'ddpm_fake_{appliance}_512.npy',
                f'ddpm_fake_{appliance}.npy'
            ]
            
            has_synthetic = any(
                os.path.exists(os.path.join(item_path, pattern))
                for pattern in synthetic_patterns
            )
            
            if has_synthetic:
                appliances.append(appliance)
    
    return sorted(appliances)


def evaluate_all_appliances(num_projections=50):
    """Evaluate SWD for all detected appliances."""
    appliances = detect_appliances()
    
    if not appliances:
        print("No appliances with synthetic data found in OUTPUT/")
        return
    
    print("=" * 80)
    print(f"Batch SWD Evaluation - Found {len(appliances)} appliance(s)")
    print("=" * 80)
    print()
    
    results = []
    
    for appliance in appliances:
        print(f"[{appliances.index(appliance)+1}/{len(appliances)}] Evaluating {appliance}...")
        print("-" * 80)
        
        try:
            swd_score = evaluate_swd(
                appliance_name=appliance,
                num_projections=num_projections,
                verbose=True
            )
            
            results.append({
                'appliance': appliance.capitalize(),
                'swd': swd_score
            })
            
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")
            results.append({
                'appliance': appliance.capitalize(),
                'swd': float('nan')
            })
        
        print()
    
    # Display summary table
    print()
    print("=" * 80)
    print("SWD SCORE SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Appliance':<20} {'SWD Score':<15}")
    print("-" * 80)
    
    for r in results:
        if np.isnan(r['swd']):
            swd_str = "N/A"
        else:
            swd_str = f"{r['swd']:.4f}"
        
        print(f"{r['appliance']:<20} {swd_str:<15}")
    
    print("=" * 80)
    print()
    print("Interpretation (calibrated for [0,1] normalized NILM data):")
    print("  < 0.05   : Excellent - Same appliance with high-quality synthesis")
    print("  0.05-0.10: Good/Borderline - Same appliance or quality concerns")
    print("  0.10-0.20: Fair - Likely different appliances")
    print("  > 0.20   : Poor - Definitely different appliances")
    print()
    print("  NOTE: SWD has limited discrimination for NILM [0,1] data.")
    print("        Use FID as primary metric. SWD provides supplementary info.")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate synthetic data quality using Sliced Wasserstein Distance (SWD)'
    )
    parser.add_argument('--appliance', type=str, default=None,
                       choices=['fridge', 'kettle', 'microwave', 'dishwasher', 'washingmachine'],
                       help='Appliance name (if not using --all)')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all appliances found in OUTPUT folder')
    parser.add_argument('--real_path', type=str, default=None,
                       help='Custom path to real data .npy file')
    parser.add_argument('--synthetic_path', type=str, default=None,
                       help='Custom path to synthetic data .npy file')
    parser.add_argument('--num_projections', type=int, default=50,
                       help='Number of random projections for SWD (default: 50)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.appliance:
        parser.error("Either --appliance or --all must be specified")
    
    if args.all:
        # Batch evaluation
        results = evaluate_all_appliances(num_projections=args.num_projections)
        return results
    else:
        # Single appliance evaluation
        swd_score = evaluate_swd(
            appliance_name=args.appliance,
            real_path=args.real_path,
            synthetic_path=args.synthetic_path,
            num_projections=args.num_projections,
            verbose=not args.quiet
        )
        return swd_score


if __name__ == '__main__':
    main()
