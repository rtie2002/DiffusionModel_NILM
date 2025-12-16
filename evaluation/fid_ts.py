"""
Context-FID Evaluation for NILM Synthetic Data
Adapted from GluonTS ContextFIDExperiment
https://github.com/mbohlkeschneider/psa-gan

Calculates Fréchet Inception Distance (FID) to evaluate 
the quality of synthetic appliance power data.

Usage:
    # Interactive mode (recommended - no path quoting issues):
    python evaluation/fid_ts.py
    
    # Or explicitly:
    python evaluation/fid_ts.py --interactive
    
    # Command-line mode with custom paths:
    python evaluation/fid_ts.py --appliance fridge --real_path "path/to/real.npy" --synthetic_path "path/to/synthetic.npy"
    
    # Or use default paths:
    python evaluation/fid_ts.py --appliance fridge
    
    # Evaluate all appliances:
    python evaluation/fid_ts.py --all
"""

import warnings
import argparse
import numpy as np
from numpy import cov, trace, iscomplexobj
from scipy import linalg
import os

# ============================================================================
# Core FID Calculation (From GluonTS)
# ============================================================================

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
    
        d² = ||mu_1 - mu_2||² + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    
    Stable version by Dougal J. Sutherland.
    
    Args:
        mu1: Mean of real data features
        sigma1: Covariance of real data features  
        mu2: Mean of synthetic data features
        sigma2: Covariance of synthetic data features
        eps: Small value for numerical stability
        
    Returns:
        FID score (lower is better)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, \
        f"Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, \
        f"Covariance matrices have different shapes: {sigma1.shape} vs {sigma2.shape}"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = (
            f"FID calculation produces singular product; "
            f"adding {eps} to diagonal of covariance estimates"
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {max_imag}")
        covmean = covmean.real
    
    tr_covmean = trace(covmean)
    
    return (
        diff.dot(diff) + trace(sigma1) + trace(sigma2) - 2 * tr_covmean
    )


def calculate_fid(real_features, synthetic_features):
    """
    Calculate FID score between real and synthetic data.
    
    Args:
        real_features: Real data features, shape (n_samples, n_features)
        synthetic_features: Synthetic data features, shape (n_samples, n_features)
        
    Returns:
        FID score (float)
    """
    # Calculate statistics
    mu_real = real_features.mean(axis=0)
    sigma_real = cov(real_features, rowvar=False)
    
    mu_synthetic = synthetic_features.mean(axis=0)
    sigma_synthetic = cov(synthetic_features, rowvar=False)
    
    # Calculate FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_synthetic, sigma_synthetic)
    
    return fid


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_real_data(appliance_name, data_root='OUTPUT'):
    """
    Load real ground truth data in NORMALIZED [0,1] format.
    
    IMPORTANT: For fair FID comparison, real and synthetic data must be 
    on the SAME scale. Synthetic data is saved as [0,1] normalized,
    so we load the normalized version of real data too.
    """
    # Real data is in: OUTPUT/{appliance}_512/samples/
    appliance_folder = f"{appliance_name}_512"
    samples_dir = os.path.join(data_root, appliance_folder, 'samples')
    
    # Priority: Load normalized [0,1] version to match synthetic data
    patterns = [
        f"{appliance_name.capitalize()}_norm_truth_512_train.npy",  # [0, 1] ← PRIORITY!
        f"{appliance_name}_norm_truth_512_train.npy",                # [0, 1] ← PRIORITY!
        f"{appliance_name.capitalize()}_ground_truth_512_train.npy", # Real watts (fallback)
        f"{appliance_name}_ground_truth_512_train.npy",              # Real watts (fallback)
    ]
    
    for pattern in patterns:
        filepath = os.path.join(samples_dir, pattern)
        if os.path.exists(filepath):
            data = np.load(filepath)
            
            # Check if data is normalized or not
            if 'norm_truth' in pattern:
                scale = "[0, 1] normalized (OK)"
            else:
                scale = "Real Watts (warning: may not match synthetic scale!)"
            
            print(f"[OK] Loaded real data: {filepath}")
            print(f"  Shape: {data.shape}")
            print(f"  Scale: {scale}")
            print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
            return data
    
    raise FileNotFoundError(
        f"Could not find real data for {appliance_name}\n"
        f"Searched in: {samples_dir}\n"
        f"Tried patterns: {patterns}"
    )


def load_synthetic_data(appliance_name, output_root='OUTPUT'):
    """
    Load synthetic data generated by diffusion model.
    
    Synthetic data is saved in [0,1] normalized format after 
    unnormalize_to_zero_to_one() in main.py.
    """
    patterns = [
        f"{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy",
        f"{appliance_name}/ddpm_fake_{appliance_name}.npy",
    ]
    
    for pattern in patterns:
        filepath = os.path.join(output_root, pattern)
        if os.path.exists(filepath):
            data = np.load(filepath)
            print(f"[OK] Loaded synthetic data: {filepath}")
            print(f"  Shape: {data.shape}")
            print(f"  Scale: [0, 1] normalized (from diffusion model)")
            print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
            return data
    
    raise FileNotFoundError(
        f"Could not find synthetic data for {appliance_name} in {output_root}\n"
        f"Tried patterns: {patterns}"
    )


def prepare_features(data, method='flatten'):
    """
    Prepare data for FID calculation.
    
    Args:
        data: Input data, shape (n_samples, seq_length, n_features)
        method: 'flatten' or 'statistics'
        
    Returns:
        features: 2D array (n_samples, feature_dim)
    """
    if len(data.shape) == 2:
        # Already 2D
        return data
    
    if method == 'flatten':
        # Flatten each sample: (n_samples, seq_length * n_features)
        return data.reshape(len(data), -1)
    
    elif method == 'statistics':
        # Statistical features for each sample
        features = []
        for sample in data:
            # Calculate statistics across time dimension
            sample_flat = sample.flatten()
            feats = [
                sample_flat.mean(),
                sample_flat.std(),
                sample_flat.min(),
                sample_flat.max(),
                np.percentile(sample_flat, 25),
                np.percentile(sample_flat, 50),
                np.percentile(sample_flat, 75),
            ]
            features.append(feats)
        return np.array(features)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def interpret_fid(fid_score):
    """
    Provide interpretation of FID_TS score.
    
    Thresholds based on empirical testing:
    - Correct matches (same appliance): 0.8-2.9
    - Mismatches (different appliances): 16-24
    """
    if fid_score < 5:
        quality = "Excellent"
        emoji = "***"
    elif fid_score < 15:
        quality = "Poor"
        emoji = "(!)"
    else:
        quality = "Terrible"
        emoji = "(X)"
    
    return quality, emoji


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_fid(appliance_name, real_path=None, synthetic_path=None, 
                 method='flatten', verbose=True):
    """
    Evaluate FID score for synthetic data.
    
    Args:
        appliance_name: Name of appliance (e.g., 'fridge', 'kettle')
        real_path: Custom path to real data (optional)
        synthetic_path: Custom path to synthetic data (optional)
        method: Feature extraction method ('flatten' or 'statistics')
        verbose: Print detailed info
        
    Returns:
        fid_score: FID score (float)
    """
    if verbose:
        print("=" * 70)
        print(f"FID Evaluation: {appliance_name.upper()}")
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
    
    # Prepare features
    if verbose:
        print(f"Extracting features using method: '{method}'...")
    
    real_features = prepare_features(real_data, method=method)
    synthetic_features = prepare_features(synthetic_data, method=method)
    
    if verbose:
        print(f"  Real features shape: {real_features.shape}")
        print(f"  Synthetic features shape: {synthetic_features.shape}")
        print()
    
    # Calculate FID
    if verbose:
        print("Calculating FID score...")
    
    fid_score = calculate_fid(real_features, synthetic_features)
    
    # Interpret results
    quality, emoji = interpret_fid(fid_score)
    
    if verbose:
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"FID Score: {fid_score:.4f}")
        print(f"Quality: {quality}")
        print()
        print("Interpretation:")
        print("  < 5    : Excellent - Same appliance (empirically validated)")
        print("  5-15   : Poor - Possible mismatch")
        print("  > 15   : Terrible - Different appliances")
        print("=" * 70)
    
    return fid_score


# ============================================================================
# Auto-Detection and Batch Evaluation
# ============================================================================

def detect_appliances(output_root='OUTPUT'):
    """Auto-detect appliances with synthetic data in OUTPUT folder."""
    appliances = []
    
    if not os.path.exists(output_root):
        return appliances
    
    for item in os.listdir(output_root):
        item_path = os.path.join(output_root, item)
        if os.path.isdir(item_path) and item.endswith('_512'):
            # Extract appliance name
            appliance = item.replace('_512', '')
            
            # Check if synthetic data exists
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


def evaluate_all_appliances(method='flatten', output_root='OUTPUT'):
    """Evaluate FID for all detected appliances and display as table."""
    appliances = detect_appliances(output_root)
    
    if not appliances:
        print(f"No appliances with synthetic data found in {output_root}/")
        return
    
    print("=" * 80)
    print(f"Batch FID Evaluation - Found {len(appliances)} appliance(s)")
    print("=" * 80)
    print()
    
    results = []
    
    for appliance in appliances:
        print(f"[{appliances.index(appliance)+1}/{len(appliances)}] Evaluating {appliance}...")
        print("-" * 80)
        
        try:
            fid_score = evaluate_fid(
                appliance_name=appliance,
                method=method,
                verbose=True
            )
            
            quality, emoji = interpret_fid(fid_score)
            results.append({
                'appliance': appliance.capitalize(),
                'fid': fid_score,
                'quality': quality,
                'emoji': emoji
            })
            
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")
            results.append({
                'appliance': appliance.capitalize(),
                'fid': float('nan'),
                'quality': 'Error',
                'emoji': 'X'
            })
        
        print()
    
    # Display summary table
    print()
    print("=" * 80)
    print("FID SCORE SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Appliance':<20} {'FID Score':<15} {'Quality':<15}")
    print("-" * 80)
    
    for r in results:
        if np.isnan(r['fid']):
            fid_str = "N/A"
        else:
            fid_str = f"{r['fid']:.4f}"
        
        print(f"{r['appliance']:<20} {fid_str:<15} {r['quality']:<15}")
    
    print("=" * 80)
    print()
    print("Legend (Empirically Validated):")
    print("  < 5    : Excellent - Same appliance ***")
    print("  5-15   : Poor - Possible mismatch (!)")
    print("  > 15   : Terrible - Different appliances (X)")
    print("=" * 80)
    
    return results


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate synthetic data quality using FID (Fréchet Inception Distance)'
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
    parser.add_argument('--method', type=str, default='flatten',
                       choices=['flatten', 'statistics'],
                       help='Feature extraction method (default: flatten)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode with prompts')
    
    args = parser.parse_args()
    
    # Interactive mode - prompt user for inputs
    if args.interactive or (not args.all and not args.appliance):
        print("=" * 70)
        print("FID EVALUATION - INTERACTIVE MODE")
        print("=" * 70)
        print()
        print("This tool calculates FID score between real and synthetic data.")
        print("You can provide file paths directly to avoid path quoting issues.")
        print()
        print("-" * 70)
        
        # Prompt for real data path
        print("Enter the path to GROUND TRUTH (real) data:")
        print("(You can paste the full path, no quotes needed)")
        real_path = input("Ground truth path: ").strip().strip('"').strip("'")
        
        # Validate real path
        while not os.path.exists(real_path):
            print(f"\n[ERROR] File not found: {real_path}")
            print("Please check the path and try again.")
            real_path = input("Ground truth path: ").strip().strip('"').strip("'")
        
        print(f"[OK] Found: {real_path}")
        print()
        
        # Prompt for synthetic data path
        print("Enter the path to SYNTHETIC data:")
        print("(You can paste the full path, no quotes needed)")
        synthetic_path = input("Synthetic path: ").strip().strip('"').strip("'")
        
        # Validate synthetic path
        while not os.path.exists(synthetic_path):
            print(f"\n[ERROR] File not found: {synthetic_path}")
            print("Please check the path and try again.")
            synthetic_path = input("Synthetic path: ").strip().strip('"').strip("'")
        
        print(f"[OK] Found: {synthetic_path}")
        print()
        
        # Prompt for method
        print("-" * 70)
        print("Feature extraction method:")
        print("  1. flatten (default) - Use raw time series")
        print("  2. statistics - Use statistical features")
        method_input = input("Select method (1 or 2, default=1): ").strip()
        
        if method_input == '2':
            method = 'statistics'
        else:
            method = 'flatten'
        
        print(f"Using method: {method}")
        print()
        
        # Try to auto-detect appliance name from filename for display purposes
        synthetic_filename = os.path.basename(synthetic_path).lower()
        appliances = ['fridge', 'kettle', 'microwave', 'dishwasher', 'washingmachine']
        detected_appliance = None
        for app in appliances:
            if app in synthetic_filename:
                detected_appliance = app
                break
        
        # Use detected appliance or generic name
        appliance_name = detected_appliance if detected_appliance else "unknown"
        
        # Run evaluation
        fid_score = evaluate_fid(
            appliance_name=appliance_name,
            real_path=real_path,
            synthetic_path=synthetic_path,
            method=method,
            verbose=True
        )
        return fid_score
    
    # Command-line mode
    # Validate arguments
    if not args.all and not args.appliance:
        parser.error("Either --appliance, --all, or --interactive must be specified")
    
    if args.all:
        # Batch evaluation
        results = evaluate_all_appliances(method=args.method)
        return results
    else:
        # Single appliance evaluation
        fid_score = evaluate_fid(
            appliance_name=args.appliance,
            real_path=args.real_path,
            synthetic_path=args.synthetic_path,
            method=args.method,
            verbose=not args.quiet
        )
        return fid_score


if __name__ == '__main__':
    main()
