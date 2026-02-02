"""
Time Distribution Viewer - All Appliances Comparison (Research Paper Format)
显示所有电器的真实数据与合成数据的时间特征分布对比图

Creates a 5x4 grid (5 appliances × 4 time features) in publication-ready format.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

# Data folder paths
REAL_DATA_FOLDER = r"C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM\Data\datasets\real_distributions"
SYNTHETIC_DATA_FOLDER = r"C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM\Data\datasets\synthetic_processed"

# Appliance names (order for display)
APPLIANCES = ['dishwasher', 'fridge', 'kettle', 'microwave', 'washingmachine']
APPLIANCE_LABELS = {
    'dishwasher': 'Dishwasher',
    'fridge': 'Fridge', 
    'kettle': 'Kettle',
    'microwave': 'Microwave',
    'washingmachine': 'Washing Machine'
}

# Time feature names
TIME_FEATURES = ['minute', 'hour', 'dow', 'month']
TIME_FEATURE_LABELS = {
    'minute': 'Minute of Hour',
    'hour': 'Hour of Day', 
    'dow': 'Day of Week',
    'month': 'Month'
}

def sincos_to_time(df):
    """Convert sin/cos encoded time features to readable values"""
    df = df.copy()
    
    # Minute (1-60)
    angle = np.arctan2(df['minute_sin'], df['minute_cos'])
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    df['minute'] = np.round((angle / (2 * np.pi)) * 60) % 60 + 1
    
    # Hour (1-24)
    angle = np.arctan2(df['hour_sin'], df['hour_cos'])
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    df['hour'] = np.round((angle / (2 * np.pi)) * 24) % 24 + 1
    
    # Day of week (1-7, 1=Mon, 7=Sun)
    angle = np.arctan2(df['dow_sin'], df['dow_cos'])
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    df['dow'] = np.round((angle / (2 * np.pi)) * 7) % 7 + 1
    
    # Month (1-12)
    angle = np.arctan2(df['month_sin'], df['month_cos'])
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    df['month'] = (np.round((angle / (2 * np.pi)) * 12) % 12) + 1
    
    df['month'] = df['month'].clip(1, 12)
    
    return df

def load_and_process_csv(file_path):
    """Load CSV and convert sin/cos to time values"""
    df = pd.read_csv(file_path)
    if 'minute_sin' in df.columns:
        df = sincos_to_time(df)
    return df

def get_file_path(folder, appliance):
    """Get the file path for an appliance in a folder"""
    file_path = os.path.join(folder, f"{appliance}_multivariate.csv")
    if os.path.exists(file_path):
        return file_path
    return None

def load_all_data():
    """Load real and synthetic data for all appliances"""
    data = {}
    
    for appliance in APPLIANCES:
        data[appliance] = {'real': None, 'synthetic': None}
        
        real_path = get_file_path(REAL_DATA_FOLDER, appliance)
        if real_path:
            print(f"Loading {appliance}...")
            data[appliance]['real'] = load_and_process_csv(real_path)
            print(f"  Real: {len(data[appliance]['real']):,} samples")
        
        synthetic_path = get_file_path(SYNTHETIC_DATA_FOLDER, appliance)
        if synthetic_path:
            data[appliance]['synthetic'] = load_and_process_csv(synthetic_path)
            print(f"  Synthetic: {len(data[appliance]['synthetic']):,} samples")
    
    return data

def plot_all_appliances_comparison(data):
    """Plot 5x4 grid in publication-ready format"""
    
    # Create figure with proper size for publication (single column ~3.5in, double ~7in)
    fig, axes = plt.subplots(5, 4, figsize=(12, 12))
    
    # Colors - professional color scheme
    real_color = '#D62728'      # Matplotlib red
    synthetic_color = '#1F77B4'  # Matplotlib blue
    
    # Bin configurations
    bin_configs = {
        'minute': {'bins': np.arange(0.5, 61.5, 1), 'xlim': (0, 61), 'xticks': [1, 15, 30, 45, 60]},
        'hour': {'bins': np.arange(0.5, 25.5, 1), 'xlim': (0, 25), 'xticks': [1, 6, 12, 18, 24]},
        'dow': {'bins': np.arange(0.5, 8.5, 1), 'xlim': (0.5, 7.5), 'xticks': [1, 2, 3, 4, 5, 6, 7], 
                'xticklabels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']},
        'month': {'bins': np.arange(0.5, 13.5, 1), 'xlim': (0.5, 12.5), 'xticks': range(1, 13)}
    }
    
    for row_idx, appliance in enumerate(APPLIANCES):
        df_real = data[appliance]['real']
        df_synthetic = data[appliance]['synthetic']
        
        for col_idx, feature in enumerate(TIME_FEATURES):
            ax = axes[row_idx, col_idx]
            config = bin_configs[feature]
            
            # Plot histograms with bar style
            if df_real is not None:
                ax.hist(df_real[feature], bins=config['bins'], 
                       color=real_color, alpha=0.6, 
                       edgecolor='darkred', linewidth=0.5,
                       label='Real', density=True)
            
            if df_synthetic is not None:
                ax.hist(df_synthetic[feature], bins=config['bins'], 
                       color=synthetic_color, alpha=0.5, 
                       edgecolor='darkblue', linewidth=0.5,
                       label='Synthetic', density=True)
            
            # Configure axes
            ax.set_xlim(config['xlim'])
            ax.set_xticks(config['xticks'])
            if 'xticklabels' in config:
                ax.set_xticklabels(config['xticklabels'], fontsize=8)
            
            # Remove top spine for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Y-axis formatting
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            
            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(TIME_FEATURE_LABELS[feature], fontweight='bold', pad=10)
            
            # Row labels (left column only)
            if col_idx == 0:
                ax.set_ylabel(f'{APPLIANCE_LABELS[appliance]}\nDensity', fontweight='bold')
            else:
                ax.set_ylabel('Density')
            
            # X-axis labels (bottom row only)
            if row_idx == 4:
                if feature == 'minute':
                    ax.set_xlabel('Minute')
                elif feature == 'hour':
                    ax.set_xlabel('Hour')
                elif feature == 'dow':
                    ax.set_xlabel('Day')
                elif feature == 'month':
                    ax.set_xlabel('Month')
    
    # Add single legend at the bottom
    handles = [
        plt.Rectangle((0,0), 1, 1, facecolor=real_color, alpha=0.6, edgecolor=real_color, linewidth=0.8),
        plt.Rectangle((0,0), 1, 1, facecolor=synthetic_color, alpha=0.5, edgecolor=synthetic_color, linewidth=0.8)
    ]
    fig.legend(handles, ['Real Data', 'Synthetic Data'], 
               loc='lower center', ncol=2, frameon=True,
               bbox_to_anchor=(0.5, -0.01), fontsize=11,
               fancybox=False, edgecolor='black')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, hspace=0.35, wspace=0.3)
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), "time_distribution_comparison.pdf")
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\n[OK] Saved to: {output_path}")
    
    # Also save as PNG for preview
    output_png = os.path.join(os.path.dirname(__file__), "time_distribution_comparison.png")
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"[OK] Saved to: {output_png}")
    
    plt.show()

def main():
    print("="*60)
    print("TIME DISTRIBUTION COMPARISON (Publication Format)")
    print("="*60)
    print(f"\nReal Data: {REAL_DATA_FOLDER}")
    print(f"Synthetic Data: {SYNTHETIC_DATA_FOLDER}")
    print(f"\nGenerating 5x4 grid (5 appliances x 4 time features)...")
    print()
    
    # Check folders
    if not os.path.exists(REAL_DATA_FOLDER):
        print(f"[X] Real data folder not found")
        return
    if not os.path.exists(SYNTHETIC_DATA_FOLDER):
        print(f"[X] Synthetic data folder not found")
        return
    
    # Load data
    data = load_all_data()
    
    # Check we have data
    valid = [a for a in APPLIANCES if data[a]['real'] is not None or data[a]['synthetic'] is not None]
    if not valid:
        print("[X] No data loaded")
        return
    
    # Plot
    print("\nGenerating publication-quality figure...")
    plot_all_appliances_comparison(data)
    
    print("\n[OK] Done!")

if __name__ == '__main__':
    main()