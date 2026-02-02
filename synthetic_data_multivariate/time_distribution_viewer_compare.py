"""
Time Distribution Viewer with Real Data Comparison
显示选定CSV与真实数据的时间特征分布对比图

Automatically converts sin/cos to readable time values and displays:
- Minute distribution (0-59)
- Hour distribution (0-23)
- Day of week distribution (0-6, Mon-Sun)
- Month distribution (1-12)

Compares selected CSV file with real data from real_data folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
import sys
import os

# Real data folder path
REAL_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "real_data")

# Appliance names for matching
APPLIANCES = ['dishwasher', 'fridge', 'kettle', 'microwave', 'washingmachine']

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
    
    df['month'] = df['month'].clip(1, 12)  # Ensure valid range
    
    return df

def detect_appliance(file_path):
    """Detect appliance name from file path"""
    file_name = os.path.basename(file_path).lower()
    for appliance in APPLIANCES:
        if appliance in file_name:
            return appliance
    return None

def get_real_data_path(appliance):
    """Get the real data file path for an appliance"""
    real_file = os.path.join(REAL_DATA_FOLDER, f"{appliance}_training_.csv")
    if os.path.exists(real_file):
        return real_file
    return None

def plot_comparison_distributions(df_selected, df_real, selected_label, real_label):
    """Plot comparison of time distributions between selected and real data"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Time Distribution Comparison\n{selected_label} vs {real_label}', 
                 fontsize=14, fontweight='bold')
    
    # Colors
    selected_color = '#3498db'  # Blue for selected
    real_color = '#e74c3c'       # Red for real
    
    # ========== ROW 0: Selected Data ==========
    # Minute distribution (1-60)
    bins = np.arange(0.5, 61.5, 1)
    axes[0, 0].hist(df_selected['minute'], bins=bins, color=selected_color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[0, 0].set_title('Minute (Selected)', fontsize=11, fontweight='bold', color=selected_color)
    axes[0, 0].set_xlabel('Minute (1-60)', fontsize=9)
    axes[0, 0].set_xlim(0.5, 60.5)
    axes[0, 0].set_ylabel('Count', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].text(0.02, 0.98, f'N={len(df_selected):,}', transform=axes[0, 0].transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Hour distribution (1-24)
    bins = np.arange(0.5, 25.5, 1)
    axes[0, 1].hist(df_selected['hour'], bins=bins, color=selected_color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[0, 1].set_title('Hour (Selected)', fontsize=11, fontweight='bold', color=selected_color)
    axes[0, 1].set_xlabel('Hour (1-24)', fontsize=9)
    axes[0, 1].set_ylabel('Count', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_xlim(0.5, 24.5)
    axes[0, 1].set_xticks(range(1, 25, 2))
    
    # Day of week distribution (1-7)
    bins = np.arange(0.5, 8.5, 1)
    axes[0, 2].hist(df_selected['dow'], bins=bins, color=selected_color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[0, 2].set_title('Day of Week (Selected)', fontsize=11, fontweight='bold', color=selected_color)
    axes[0, 2].set_xlabel('Day (1=Mon, 7=Sun)', fontsize=9)
    axes[0, 2].set_ylabel('Count', fontsize=9)
    axes[0, 2].set_xlim(0.5, 7.5)
    axes[0, 2].set_xticks(range(1, 8))
    axes[0, 2].set_xticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'], fontsize=8)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Month distribution (1-12)
    bins = np.arange(0.5, 13.5, 1)
    axes[0, 3].hist(df_selected['month'], bins=bins, color=selected_color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[0, 3].set_title('Month (Selected)', fontsize=11, fontweight='bold', color=selected_color)
    axes[0, 3].set_xlabel('Month (1-12)', fontsize=9)
    axes[0, 3].set_ylabel('Count', fontsize=9)
    axes[0, 3].set_xlim(0.5, 12.5)
    axes[0, 3].set_xticks(range(1, 13))
    axes[0, 3].grid(True, alpha=0.3, axis='y')
    
    # ========== ROW 1: Real Data ==========
    # Minute distribution (1-60)
    bins = np.arange(0.5, 61.5, 1)
    axes[1, 0].hist(df_real['minute'], bins=bins, color=real_color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[1, 0].set_title('Minute (Real)', fontsize=11, fontweight='bold', color=real_color)
    axes[1, 0].set_xlabel('Minute (1-60)', fontsize=9)
    axes[1, 0].set_xlim(0.5, 60.5)
    axes[1, 0].set_ylabel('Count', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].text(0.02, 0.98, f'N={len(df_real):,}', transform=axes[1, 0].transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Hour distribution (1-24)
    bins = np.arange(0.5, 25.5, 1)
    axes[1, 1].hist(df_real['hour'], bins=bins, color=real_color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[1, 1].set_title('Hour (Real)', fontsize=11, fontweight='bold', color=real_color)
    axes[1, 1].set_xlabel('Hour (1-24)', fontsize=9)
    axes[1, 1].set_ylabel('Count', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_xlim(0.5, 24.5)
    axes[1, 1].set_xticks(range(1, 25, 2))
    
    # Day of week distribution (1-7)
    bins = np.arange(0.5, 8.5, 1)
    axes[1, 2].hist(df_real['dow'], bins=bins, color=real_color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[1, 2].set_title('Day of Week (Real)', fontsize=11, fontweight='bold', color=real_color)
    axes[1, 2].set_xlabel('Day (1=Mon, 7=Sun)', fontsize=9)
    axes[1, 2].set_ylabel('Count', fontsize=9)
    axes[1, 2].set_xlim(0.5, 7.5)
    axes[1, 2].set_xticks(range(1, 8))
    axes[1, 2].set_xticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'], fontsize=8)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Month distribution (1-12)
    bins = np.arange(0.5, 13.5, 1)
    axes[1, 3].hist(df_real['month'], bins=bins, color=real_color, alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[1, 3].set_title('Month (Real)', fontsize=11, fontweight='bold', color=real_color)
    axes[1, 3].set_xlabel('Month (1-12)', fontsize=9)
    axes[1, 3].set_ylabel('Count', fontsize=9)
    axes[1, 3].set_xlim(0.5, 12.5)
    axes[1, 3].set_xticks(range(1, 13))
    axes[1, 3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_overlay_distributions(df_selected, df_real, selected_label, real_label):
    """Plot overlaid time distributions for direct comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Time Distribution Overlay Comparison\n{selected_label} vs {real_label}', 
                 fontsize=14, fontweight='bold')
    
    # Colors
    selected_color = '#3498db'  # Blue
    real_color = '#e74c3c'       # Red
    
    # Minute distribution (1-60)
    bins = np.arange(0.5, 61.5, 1)
    axes[0, 0].hist(df_selected['minute'], bins=bins, color=selected_color, alpha=0.5, 
                    edgecolor=selected_color, linewidth=0.5, label=f'Selected (N={len(df_selected):,})', density=True)
    axes[0, 0].hist(df_real['minute'], bins=bins, color=real_color, alpha=0.5, 
                    edgecolor=real_color, linewidth=0.5, label=f'Real (N={len(df_real):,})', density=True)
    axes[0, 0].set_title('Minute Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Minute (1-60)', fontsize=10)
    axes[0, 0].set_xlim(0.5, 60.5)
    axes[0, 0].set_ylabel('Density', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].legend(fontsize=8)
    
    # Hour distribution (1-24)
    bins = np.arange(0.5, 25.5, 1)
    axes[0, 1].hist(df_selected['hour'], bins=bins, color=selected_color, alpha=0.5, 
                    edgecolor=selected_color, linewidth=0.5, label='Selected', density=True)
    axes[0, 1].hist(df_real['hour'], bins=bins, color=real_color, alpha=0.5, 
                    edgecolor=real_color, linewidth=0.5, label='Real', density=True)
    axes[0, 1].set_title('Hour Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Hour (1-24)', fontsize=10)
    axes[0, 1].set_ylabel('Density', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_xlim(0.5, 24.5)
    axes[0, 1].set_xticks(range(1, 25, 2))
    axes[0, 1].legend(fontsize=8)
    
    # Day of week distribution (1-7)
    bins = np.arange(0.5, 8.5, 1)
    axes[1, 0].hist(df_selected['dow'], bins=bins, color=selected_color, alpha=0.5, 
                    edgecolor=selected_color, linewidth=0.5, label='Selected', density=True)
    axes[1, 0].hist(df_real['dow'], bins=bins, color=real_color, alpha=0.5, 
                    edgecolor=real_color, linewidth=0.5, label='Real', density=True)
    axes[1, 0].set_title('Day of Week Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Day of Week (1=Mon, 7=Sun)', fontsize=10)
    axes[1, 0].set_ylabel('Density', fontsize=10)
    axes[1, 0].set_xlim(0.5, 7.5)
    axes[1, 0].set_xticks(range(1, 8))
    axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].legend(fontsize=8)
    
    # Month distribution (1-12)
    bins = np.arange(0.5, 13.5, 1)
    axes[1, 1].hist(df_selected['month'], bins=bins, color=selected_color, alpha=0.5, 
                    edgecolor=selected_color, linewidth=0.5, label='Selected', density=True)
    axes[1, 1].hist(df_real['month'], bins=bins, color=real_color, alpha=0.5, 
                    edgecolor=real_color, linewidth=0.5, label='Real', density=True)
    axes[1, 1].set_title('Month Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Month (1-12)', fontsize=10)
    axes[1, 1].set_ylabel('Density', fontsize=10)
    axes[1, 1].set_xlim(0.5, 12.5)
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend(fontsize=8)
    
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

def load_and_process_csv(file_path, label="Data"):
    """Load CSV and convert sin/cos to time values"""
    print(f"\nLoading {label}: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    
    # Check if sin/cos format
    if 'minute_sin' in df.columns:
        print(f"  ✓ Detected sin/cos format, converting...")
        df = sincos_to_time(df)
        print(f"    Minute range: {df['minute'].min():.0f}-{df['minute'].max():.0f}")
        print(f"    Hour range: {df['hour'].min():.0f}-{df['hour'].max():.0f}")
        print(f"    DOW range: {df['dow'].min():.0f}-{df['dow'].max():.0f}")
        print(f"    Month range: {df['month'].min():.0f}-{df['month'].max():.0f}")
    else:
        print("  ✓ Detected raw format (already has time values)")
    
    return df

def main():
    print("="*80)
    print("TIME DISTRIBUTION VIEWER WITH REAL DATA COMPARISON")
    print("="*80)
    print("\nCompares time feature distributions between:")
    print("  - Selected CSV file (e.g., synthetic/generated data)")
    print("  - Real data from real_data folder")
    print("\nTime features compared:")
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
        # Detect appliance from selected file
        appliance = detect_appliance(file_path)
        if appliance:
            print(f"\n✓ Detected appliance: {appliance}")
        else:
            print("\n⚠ Could not detect appliance from filename.")
            print(f"  Filename: {os.path.basename(file_path)}")
            print(f"  Expected appliance names: {APPLIANCES}")
            return
        
        # Load selected CSV
        df_selected = load_and_process_csv(file_path, label="Selected CSV")
        
        # Find and load corresponding real data
        real_data_path = get_real_data_path(appliance)
        if not real_data_path:
            print(f"\n✗ Real data not found for {appliance}")
            print(f"  Expected at: {os.path.join(REAL_DATA_FOLDER, f'{appliance}_training_.csv')}")
            return
        
        df_real = load_and_process_csv(real_data_path, label="Real Data")
        
        # Generate labels for plots
        selected_label = f"Selected: {os.path.basename(file_path)}"
        real_label = f"Real: {appliance}_training_.csv"
        
        # Plot side-by-side comparison
        print("\n" + "="*80)
        print("Generating side-by-side comparison plots...")
        plot_comparison_distributions(df_selected, df_real, selected_label, real_label)
        
        # Plot overlay comparison
        print("\nGenerating overlay comparison plots...")
        plot_overlay_distributions(df_selected, df_real, selected_label, real_label)
        
        print("\n✓ Done!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
