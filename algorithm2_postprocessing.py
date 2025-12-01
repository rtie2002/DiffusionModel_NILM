"""
Algorithm 2: Decomposition Signal Post-Processing Algorithm
Based on the paper: "A diffusion model-based framework to enhance the robustness 
of non-intrusive load disaggregation"

This algorithm uses exponentially weighted moving average (EWMA) to smooth
NILM decomposition results while preserving ON/OFF transitions.

Mathematical Model:
    Decomposed signal: x_t = p_t + n_t
        p_t = actual electrical power
        n_t = noise term ~ N(0, σ²)
    
    Post-processed signal: s_t = α·x_t + (1-α)·s_{t-1}
        α = smoothing coefficient (0 < α < 1)
    
    Noise variance reduction: α·σ² / (2-α)
    When α < 1, noise is suppressed

Usage:
    python algorithm2_postprocessing.py --input results.csv --appliance washingmachine --alpha 0.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Appliance thresholds (from paper Table 1)
APPLIANCE_THRESHOLDS = {
    'kettle': 2000,         # 2000W
    'microwave': 200,       # 200W
    'fridge': 50,           # 50W
    'dishwasher': 10,       # 10W
    'washingmachine': 20,   # 20W
}

def algorithm2_postprocessing(x, x_threshold, alpha=0.5):
    """
    Algorithm 2: Decomposition Signal Post-Processing
    
    Args:
        x (np.ndarray): Decomposed power results from NILM model, shape (n,)
        x_threshold (float): Appliance start threshold (in Watts)
        alpha (float): Smoothing coefficient, 0 < alpha < 1
                      Smaller alpha = more smoothing
                      Larger alpha = less smoothing (closer to original)
    
    Returns:
        s (np.ndarray): Post-processed power data, shape (n,)
    
    Algorithm Steps (from paper):
        1. Initialize s as zeros with same shape as x
        2. Initialize f_active = False, x_last = 0
        3. For each time step t:
            4. If x[t] > x_threshold:
                5. If not f_active (first activation):
                    6. s[t] = x[t]
                    7. x_last = x[t]
                    8. f_active = True
                9. Else (appliance already active):
                    10. s[t] = α·x[t] + (1-α)·x_last
                    11. x_last = s[t]
            12. Else (below threshold):
                13. s[t] = x[t]
                14. f_active = False
        15. Return s
    """
    # Input validation
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    if len(x.shape) > 1:
        x = x.flatten()
    
    if not (0 < alpha < 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    # Line 1: Initialize output array
    s = np.zeros_like(x, dtype=float)
    
    # Line 2: Initialize state variables
    f_active = False  # Is appliance currently active?
    x_last = 0.0      # Last processed value
    
    # Line 3: Iterate through each time step
    for t in range(len(x)):
        # Line 4-8: Current value above threshold (appliance may be running)
        if x[t] > x_threshold:
            # Line 5-8: First detection of activation
            if not f_active:
                s[t] = x[t]        # Line 6: No smoothing on first activation
                x_last = x[t]      # Line 7: Record current value
                f_active = True    # Line 8: Mark as active
            # Line 9-11: Appliance already active (apply EWMA)
            else:
                s[t] = alpha * x[t] + (1 - alpha) * x_last  # Line 10: EWMA
                x_last = s[t]                                # Line 11: Update last value
        # Line 12-14: Current value below threshold (appliance OFF)
        else:
            s[t] = x[t]          # Line 13: Keep original value (preserve OFF state)
            f_active = False     # Line 14: Reset active flag
    
    # Line 15: Return smoothed signal
    return s

def visualize_postprocessing(x_original, s_processed, appliance_name, alpha, save_path=None):
    """
    Visualize the effect of Algorithm 2 post-processing
    
    Args:
        x_original: Original decomposed signal
        s_processed: Post-processed signal
        appliance_name: Name of the appliance
        alpha: Smoothing coefficient used
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Algorithm 2 Post-Processing Results - {appliance_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    sample_size = min(5000, len(x_original))
    
    # Plot 1: Overlaid comparison
    ax1 = axes[0]
    ax1.plot(x_original[:sample_size], color='red', alpha=0.7, 
             linewidth=0.8, label='Original (with noise)')
    ax1.plot(s_processed[:sample_size], color='blue', alpha=0.9, 
             linewidth=1.2, label=f'Post-processed (α={alpha})')
    ax1.set_title('Time Series Comparison', fontweight='bold')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Power (W)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detail view of a segment
    ax2 = axes[1]
    start_idx = 1000
    end_idx = min(2000, len(x_original))
    ax2.plot(range(start_idx, end_idx), x_original[start_idx:end_idx], 
             color='red', alpha=0.7, linewidth=1.0, label='Original', marker='o', markersize=2)
    ax2.plot(range(start_idx, end_idx), s_processed[start_idx:end_idx], 
             color='blue', alpha=0.9, linewidth=1.5, label='Post-processed')
    ax2.set_title(f'Detail View (Samples {start_idx}-{end_idx})', fontweight='bold')
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Power (W)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistical comparison
    ax3 = axes[2]
    ax3.axis('off')
    
    # Calculate statistics
    stats_data = []
    stats_data.append(['Metric', 'Original Signal', 'Post-processed Signal'])
    stats_data.append(['Mean', f'{x_original.mean():.2f} W', f'{s_processed.mean():.2f} W'])
    stats_data.append(['Std Dev', f'{x_original.std():.2f} W', f'{s_processed.std():.2f} W'])
    stats_data.append(['Min', f'{x_original.min():.2f} W', f'{s_processed.min():.2f} W'])
    stats_data.append(['Max', f'{x_original.max():.2f} W', f'{s_processed.max():.2f} W'])
    
    noise_reduction = (1 - s_processed.std() / x_original.std()) * 100
    stats_data.append(['Noise Reduction', '-', f'{noise_reduction:.1f}%'])
    
    # Create table
    table = ax3.table(cellText=stats_data, loc='center', cellLoc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Beautify table
    for i in range(len(stats_data)):
        if i == 0:  # Header
            for j in range(3):
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            table[(i, 0)].set_facecolor('#E8F5E9')
    
    ax3.set_title('Statistical Comparison', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Apply Algorithm 2 post-processing to NILM decomposition results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file with decomposed power data')
    parser.add_argument('--appliance', type=str, required=True,
                       choices=list(APPLIANCE_THRESHOLDS.keys()),
                       help='Appliance name')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Smoothing coefficient (0 < alpha < 1), default: 0.5')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Custom threshold (Watts). If not set, uses default from paper')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path. If not set, saves to input_postprocessed.csv')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization of results')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Algorithm 2: Post-Processing NILM Results - {args.appliance.upper()}")
    print("=" * 70)
    
    # Load input data
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return
    
    df = pd.read_csv(args.input)
    
    # Expect 'power' column
    if 'power' not in df.columns:
        print(f"ERROR: Input CSV must have 'power' column")
        print(f"Found columns: {df.columns.tolist()}")
        return
    
    x_original = df['power'].values
    print(f"Loaded data: {len(x_original)} samples")
    print(f"  Mean: {x_original.mean():.2f} W")
    print(f"  Std Dev: {x_original.std():.2f} W")
    
    # Get threshold
    if args.threshold is not None:
        x_threshold = args.threshold
        print(f"\nUsing custom threshold: {x_threshold} W")
    else:
        x_threshold = APPLIANCE_THRESHOLDS[args.appliance]
        print(f"\nUsing default threshold: {x_threshold} W (from paper)")
    
    # Apply Algorithm 2
    print(f"\nApplying Algorithm 2 with α = {args.alpha}...")
    s_processed = algorithm2_postprocessing(x_original, x_threshold, args.alpha)
    
    # Calculate noise reduction
    noise_reduction = (1 - s_processed.std() / x_original.std()) * 100
    print(f"\nPost-processing complete!")
    print(f"  Post-processed Mean: {s_processed.mean():.2f} W")
    print(f"  Post-processed Std Dev: {s_processed.std():.2f} W")
    print(f"  Noise Reduction: {noise_reduction:.1f}%")
    
    # Save output
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_postprocessed.csv"
    
    output_df = pd.DataFrame({'power': s_processed})
    output_df.to_csv(args.output, index=False)
    print(f"\nSaved post-processed data: {args.output}")
    
    # Visualize
    if args.visualize:
        print("\nGenerating visualization...")
        vis_path = os.path.splitext(args.output)[0] + '_visualization.png'
        visualize_postprocessing(x_original, s_processed, args.appliance, 
                                args.alpha, vis_path)
    
    print("=" * 70)

if __name__ == '__main__':
    main()
