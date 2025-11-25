"""
Algorithm 1: Data Cleaning and Selection for Appliance Power Data
Based on the paper: "A diffusion model-based framework to enhance the robustness 
of non-intrusive load disaggregation"

This script implements Algorithm 1 to select effective parts of appliance data
and prepare it for diffusion model training.

IMPORTANT: According to the paper (Section 4.1), Algorithm 1 is applied ONLY to 
TRAINING data: "When synthesizing data, we execute Algorithm 1 on the training data, 
send it to the diffusion model for synthetic data training..."

- Training data: Apply Algorithm 1 → Used for diffusion model training
- Validation/Test data: NOT processed by Algorithm 1 → Used for NILM model evaluation

Input: CSV file from ukdale_processing.py (format: aggregate,power with header)
Output: Single-column CSV with 'power' header (MinMax normalized, effective parts only)

Usage:
    # Default: Process only training file (as per paper)
    python apply_algorithm1.py --appliance_name microwave
    
    # Optional: Process all files (for other purposes)
    python apply_algorithm1.py --appliance_name microwave --all
    
Workflow:
    1. Read CSV from ukdale_processing.py output (aggregate,power columns)
    2. Extract power column
    3. Apply Algorithm 1 (select effective parts based on threshold and window)
    4. Apply MinMaxScaler normalization
    5. Save to Data/datasets/{appliance_name}.csv (single column, 'power' header)
"""

import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Appliance parameters from Table 1 in the paper
APPLIANCE_PARAMS = {
    'kettle': {
        'on_power_threshold': 200,
        'mean': 700,
        'std': 1000,
    },
    'microwave': {
        'on_power_threshold': 200,
        'mean': 500,
        'std': 800,
    },
    'fridge': {
        'on_power_threshold': 50,
        'mean': 200,
        'std': 400,
    },
    'dishwasher': {
        'on_power_threshold': 10,
        'mean': 700,
        'std': 1000,
    },
    'washingmachine': {
        'on_power_threshold': 20,
        'mean': 400,
        'std': 700,
    }
}

def algorithm1_data_cleaning(power_sequence, x_threshold, l_window=100, x_noise=0):
    """
    Algorithm 1: Data Cleaning and Selection for Appliance Power Data
    
    Input:
        power_sequence: power sequence x = [x1,...,xt,...]
        x_threshold: appliance start threshold
        l_window: window length (default: 100, from paper Table 2)
        x_noise: power noise threshold (default: 0)
    
    Output:
        cleaned and selected power data
    """
    # Step 1: Initialize T_selected as an empty list
    T_selected = []
    
    # Step 2: x[x < x_noise] = 0
    power_sequence = power_sequence.copy()
    power_sequence[power_sequence < x_noise] = 0
    
    # Step 3: T_start = where(x >= x_threshold)
    T_start = np.where(power_sequence >= x_threshold)[0]
    
    # Step 4-8: For each index in T_start, select window before and after
    for index in T_start:
        T_start_window = max(0, index - l_window)
        T_end_window = min(len(power_sequence), index + l_window + 1)
        T_selected.extend(range(T_start_window, T_end_window))
    
    # Step 9: T_selected = sorted(set(T_selected))
    T_selected = sorted(set(T_selected))
    
    # Step 10: x = x[T_selected]
    x_selected = power_sequence[T_selected]
    
    # Step 11-12: Apply MinMaxScaler
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_selected.reshape(-1, 1))
    
    return x_scaled.flatten(), scaler

def plot_data_processing(power_data_original, x_cleaned, 
                        x_threshold, appliance_name, output_dir, max_samples=None):
    """
    Plot three informative graphs showing Algorithm 1's effect:
    1. Original data with threshold and selected regions highlighted
    2. Zoomed view of a startup event
    3. Final selected and normalized data
    
    Args:
        max_samples: Maximum number of samples to plot for overview. If None, plot all data.
    """
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Algorithm 1 Data Processing: {appliance_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Find startup events for highlighting
    startup_indices = np.where(power_data_original >= x_threshold)[0]
    
    # ============ Plot 1: Full original data with highlighted regions ============
    ax1 = fig.add_subplot(gs[0, :])
    
    # Determine sample size for overview
    if max_samples is None:
        sample_size = len(power_data_original)
    else:
        sample_size = min(max_samples, len(power_data_original))
    
    indices = np.arange(sample_size)
    
    # Plot original data
    ax1.plot(indices, power_data_original[:sample_size], 'b-', linewidth=0.5, alpha=0.6, label='Original data')
    
    # Highlight startup regions
    if len(startup_indices) > 0:
        startup_in_range = startup_indices[startup_indices < sample_size]
        if len(startup_in_range) > 0:
            ax1.scatter(startup_in_range, power_data_original[startup_in_range], 
                       c='red', s=1, alpha=0.5, label='Startup events (≥ threshold)')
    
    # Threshold line
    ax1.axhline(y=x_threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Threshold: {x_threshold} W')
    
    ax1.set_title(f'Step 1: Original Data (Z-score denormalized to Watts)', 
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Power (Watts)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Statistics box
    on_percentage = (len(startup_indices) / len(power_data_original)) * 100
    stats_text = f'Total samples: {len(power_data_original):,}\n'
    stats_text += f'Range: [{power_data_original.min():.0f}, {power_data_original.max():.0f}] W\n'
    stats_text += f'Startup events: {len(startup_indices):,} ({on_percentage:.2f}%)'
    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ============ Plot 2: Zoomed view of a startup event ============
    ax2 = fig.add_subplot(gs[1, 0])
    
    if len(startup_indices) > 0:
        # Find a good startup event to zoom into (middle of the data)
        mid_idx = len(startup_indices) // 2
        center = startup_indices[mid_idx]
        window = 500  # Show ±500 samples around startup
        
        start_idx = max(0, center - window)
        end_idx = min(len(power_data_original), center + window)
        
        zoom_indices = np.arange(start_idx, end_idx)
        zoom_data = power_data_original[start_idx:end_idx]
        
        ax2.plot(zoom_indices, zoom_data, 'b-', linewidth=1, label='Power')
        ax2.axhline(y=x_threshold, color='r', linestyle='--', linewidth=2, 
                   label=f'Threshold: {x_threshold} W')
        
        # Highlight the startup event
        startup_in_zoom = startup_indices[(startup_indices >= start_idx) & (startup_indices < end_idx)]
        if len(startup_in_zoom) > 0:
            ax2.scatter(startup_in_zoom, power_data_original[startup_in_zoom], 
                       c='red', s=20, alpha=0.7, label='Startup', zorder=5)
        
        ax2.set_title(f'Step 2: Zoomed View of Startup Event', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Power (Watts)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        zoom_text = f'Window: ±{window} samples\nCenter: index {center}'
        ax2.text(0.02, 0.98, zoom_text,
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'No startup events found\n(all data below threshold)',
                ha='center', va='center', fontsize=12, color='red')
        ax2.set_title('Step 2: Zoomed View (No Events)', fontweight='bold', fontsize=12)
    
    # ============ Plot 3: Distribution comparison ============
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create histograms
    ax3.hist(power_data_original, bins=50, alpha=0.5, label='Original', color='blue', density=True)
    
    # For cleaned data, we need to denormalize it back to see the distribution
    # But we don't have the scaler, so we'll just show it's focused on high power
    ax3.axvline(x=x_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {x_threshold} W')
    
    ax3.set_title('Step 3: Power Distribution', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Power (Watts)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    dist_text = f'Original: mostly OFF state\nAlgorithm 1: keeps ON state'
    ax3.text(0.98, 0.98, dist_text,
            transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    
    # ============ Plot 4: Final selected data (MinMax normalized) ============
    ax4 = fig.add_subplot(gs[2, :])
    
    # Show first N samples of cleaned data
    sample_size_cleaned = min(5000, len(x_cleaned))
    indices_cleaned = np.arange(sample_size_cleaned)
    
    ax4.plot(indices_cleaned, x_cleaned[:sample_size_cleaned], 
            'g-', linewidth=0.5, alpha=0.7)
    ax4.set_title(f'Step 4: Final Output - Selected & MinMax Normalized [0,1] (First {sample_size_cleaned:,} samples)', 
                  fontweight='bold', fontsize=12)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Power (Normalized 0-1)')
    ax4.grid(True, alpha=0.3)
    
    retention_rate = len(x_cleaned) / len(power_data_original) * 100
    removed_samples = len(power_data_original) - len(x_cleaned)
    
    final_text = f'Selected samples: {len(x_cleaned):,}\n'
    final_text += f'Removed samples: {removed_samples:,}\n'
    final_text += f'Retention rate: {retention_rate:.2f}%\n'
    final_text += f'Range: [{x_cleaned.min():.4f}, {x_cleaned.max():.4f}]'
    
    ax4.text(0.02, 0.98, final_text,
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Show plot
    plt.show()

def main():
    """
    Apply Algorithm 1 to TRAINING data only (as per paper requirement).
    
    According to the paper (Section 4.1): 
    "When synthesizing data, we execute Algorithm 1 on the training data, 
    send it to the diffusion model for synthetic data training..."
    """
    parser = argparse.ArgumentParser(
        description='Apply Algorithm 1 to TRAINING data for diffusion model (as per paper)')
    parser.add_argument('--appliance_name', type=str, required=True,
                        help='Appliance name: microwave, fridge, dishwasher, washingmachine, kettle')
    parser.add_argument('--input_file', type=str, 
                        default=None,
                        help='Input CSV file path (if not provided, uses training file from ukdale_processing)')
    parser.add_argument('--output_dir', type=str,
                        default='Data/datasets',
                        help='Output directory for processed CSV files')
    parser.add_argument('--window', type=int, default=100,
                        help='Window length for Algorithm 1 (default: 100, from paper)')
    parser.add_argument('--plot_samples', type=int, default=5000,
                        help='Number of samples to plot in visualization (default: 5000, use 0 for all data)')
    
    args = parser.parse_args()
    
    appliance_name = args.appliance_name.lower()
    
    if appliance_name not in APPLIANCE_PARAMS:
        raise ValueError(f"Unknown appliance: {appliance_name}. Must be one of: {list(APPLIANCE_PARAMS.keys())}")
    
    # Get appliance parameters
    params = APPLIANCE_PARAMS[appliance_name]
    x_threshold = params['on_power_threshold']
    
    # Determine input file (TRAINING data only)
    base_dir = 'NILM-main/dataset_preprocess/created_data/UK_DALE'
    if args.input_file is None:
        # Default: use training file from ukdale_processing.py
        input_file = f'{base_dir}/{appliance_name}_training_.csv'
    else:
        input_file = args.input_file
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Training file not found: {input_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{appliance_name}.csv')
    
    # Process training data
    print(f"\n{'='*60}")
    print(f"Applying Algorithm 1 to TRAINING data: {appliance_name}")
    print(f"{'='*60}")
    print(f"Reading: {input_file}")
    print(f"  Expected format: CSV with header (aggregate,power) from ukdale_processing.py")
    
    # Read CSV (with header: aggregate,power) - output from ukdale_processing.py
    df = pd.read_csv(input_file, header=0)
    
    print(f"  CSV columns: {df.columns.tolist()}")
    print(f"  CSV shape: {df.shape}")
    
    # Extract power column from ukdale_processing.py output
    if 'power' in df.columns:
        power_data = df['power'].values
        print(f"  Using 'power' column")
    else:
        # Fallback: if no 'power' column, use second column (appliance column)
        power_data = df.iloc[:, 1].values
        print(f"  Warning: No 'power' column found, using second column")
    
    print(f"  Original data length: {len(power_data)}")
    print(f"  Power range (Z-score normalized): [{power_data.min():.4f}, {power_data.max():.4f}]")
    
    # IMPORTANT: Data from ukdale_processing.py is already Z-score normalized
    # Formula used in ukdale_processing.py: normalized = (original - mean) / std
    # So we need to denormalize: original = normalized * std + mean
    # 
    # All parameters (mean, std, threshold) are in ORIGINAL power units (Watts)
    # This is correct because:
    # 1. Threshold values (e.g., 200W for kettle) are from paper Table 1 (original units)
    # 2. Mean and std are from paper Table 1 (original units)
    # 3. We denormalize the data first, then apply Algorithm 1 with original threshold
    
    mean = params['mean']  # Original power mean (Watts) - from paper Table 1
    std = params['std']    # Original power std (Watts) - from paper Table 1
    print(f"\nDenormalizing data (Z-score inverse):")
    print(f"  Mean: {mean} W, Std: {std} W (original power units)")
    print(f"  Formula: original = normalized * std + mean")
    
    # Z-score inverse: original = normalized * std + mean
    power_data_original = power_data * std + mean
    print(f"  Denormalized power range: [{power_data_original.min():.4f}, {power_data_original.max():.4f}] W")
    
    # Verify denormalization is reasonable (power should be non-negative)
    if power_data_original.min() < -100:  # Allow some tolerance for noise
        print(f"  ⚠ Warning: Denormalized data has negative values below -100W")
        print(f"    This might indicate a mismatch in mean/std parameters")
    
    # Apply Algorithm 1 on denormalized data (original power units)
    print(f"\nApplying Algorithm 1:")
    print(f"  Threshold: {x_threshold} W (original power units - from paper Table 1)")
    print(f"  Window length: {args.window}")
    
    x_cleaned, scaler = algorithm1_data_cleaning(
        power_data_original, 
        x_threshold=x_threshold, 
        l_window=args.window
    )
    
    print(f"  Selected data length: {len(x_cleaned)}")
    print(f"  Reduction: {len(power_data) - len(x_cleaned)} samples removed")
    print(f"  Retention rate: {len(x_cleaned)/len(power_data)*100:.2f}%")
    print(f"  Scaled range: [{x_cleaned.min():.4f}, {x_cleaned.max():.4f}]")
    
    # Save as CSV with 'power' header
    output_df = pd.DataFrame({'power': x_cleaned})
    output_df.to_csv(output_file, index=False, header=True)
    
    print(f"\n{'='*60}")
    print(f"✓ Algorithm 1 processing complete!")
    print(f"{'='*60}")
    print(f"  Saved: {output_file}")
    print(f"  Rows: {len(output_df)}")
    print(f"  Format: Single column with 'power' header (MinMax normalized)")
    print(f"\n  Use in config: data_root: {output_file}")
    print(f"  Note: Only TRAINING data processed (as per paper requirement)")
    
    # Plot data processing visualization
    print(f"\nGenerating visualization...")
    try:
        max_samples = None if args.plot_samples == 0 else args.plot_samples
        plot_data_processing(
            power_data_original,  # Before Algorithm 1 (denormalized)
            x_cleaned,  # After Algorithm 1 (MinMax normalized)
            x_threshold,
            appliance_name,
            args.output_dir,
            max_samples=max_samples
        )
    except Exception as e:
        print(f"  Warning: Could not generate plot: {e}")
        print(f"  (This is optional and does not affect the output CSV file)")

if __name__ == '__main__':
    main()

