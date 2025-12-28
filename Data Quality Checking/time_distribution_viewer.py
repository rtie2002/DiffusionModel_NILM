"""
Simple Time Distribution Viewer
显示所有 4 个时间特征的分布图

Automatically converts sin/cos to readable time values and displays:
- Minute distribution (0-59)
- Hour distribution (0-23)
- Day of week distribution (0-6, Mon-Sun)
- Month distribution (1-12)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
import sys

def sincos_to_time(df):
    """Convert sin/cos encoded time features to readable values"""
    print("Converting sin/cos to readable time values...")
    
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
    
    df['month'] = df['month'].clip(1, 12)  # Ensure valid range
    
    return df

def plot_distributions(df):
    """Plot all 4 time distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Time Feature Distributions (Converted from Sin/Cos)', 
                 fontsize=14, fontweight='bold')
    
    # Minute distribution (1-60, bars centered on values)
    bins = np.arange(0.5, 61.5, 1)  # Bins from 0.5 to 60.5 to center bars on 1-60
    axes[0, 0].hist(df['minute'], bins=bins, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Minute Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Minute (1-60)', fontsize=10)
    axes[0, 0].set_xlim(0.5, 60.5)
    axes[0, 0].set_ylabel('Count', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].text(0.02, 0.98, f'Total: {len(df):,}', 
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Hour distribution (1-24, bars centered on values)
    bins = np.arange(0.5, 25.5, 1)  # Bins from 0.5 to 24.5 to center bars on 1-24
    axes[0, 1].hist(df['hour'], bins=bins, color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title('Hour Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Hour (1-24)', fontsize=10)
    axes[0, 1].set_ylabel('Count', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_xlim(0.5, 24.5)
    axes[0, 1].set_xticks(range(1, 25, 2))
    
    # Day of week distribution (1-7, bars centered on values)
    bins = np.arange(0.5, 8.5, 1)  # Bins from 0.5 to 7.5 to center bars on 1-7
    axes[1, 0].hist(df['dow'], bins=bins, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('Day of Week Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Day of Week (1=Mon, 7=Sun)', fontsize=10)
    axes[1, 0].set_ylabel('Count', fontsize=10)
    axes[1, 0].set_xlim(0.5, 7.5)
    axes[1, 0].set_xticks(range(1, 8))
    axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Month distribution (1-12, bars centered on values)
    month_min = int(df['month'].min())
    month_max = int(df['month'].max())
    bins = np.arange(month_min - 0.5, month_max + 1.5, 1)  # Center bars on month values
    axes[1, 1].hist(df['month'], bins=bins, 
                   color='cyan', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title('Month Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Month (1-12)', fontsize=10)
    axes[1, 1].set_ylabel('Count', fontsize=10)
    axes[1, 1].set_xlim(month_min - 0.5, month_max + 0.5)
    axes[1, 1].set_xticks(range(month_min, month_max + 1))
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def select_csv_file():
    """Open file dialog to select CSV file"""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Multivariate CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir="Data/datasets/"
    )
    root.destroy()
    return file_path

def main():
    print("="*80)
    print("TIME DISTRIBUTION VIEWER")
    print("="*80)
    print("\nDisplays all 4 time feature distributions:")
    print("  - Minute (1-60)")
    print("  - Hour (1-24)")
    print("  - Day of Week (1-7, 1=Mon, 7=Sun)")
    print("  - Month (1-12)")
    
    # Select file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = select_csv_file()
    
    if not file_path:
        print("\n✗ No file selected. Exiting...")
        return
    
    try:
        print(f"\nLoading: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        
        # Check if sin/cos format
        if 'minute_sin' in df.columns:
            print("  ✓ Detected sin/cos format")
            df = sincos_to_time(df)
            print(f"    Minute range: {df['minute'].min()}-{df['minute'].max()}")
            print(f"    Hour range: {df['hour'].min()}-{df['hour'].max()}")
            print(f"    DOW range: {df['dow'].min()}-{df['dow'].max()}")
            print(f"    Month range: {df['month'].min()}-{df['month'].max()}")
        else:
            print("  ✓ Detected raw format (already has time values)")
        
        # Plot distributions
        print("\nGenerating plots...")
        plot_distributions(df)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
