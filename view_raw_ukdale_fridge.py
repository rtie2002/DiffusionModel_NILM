"""
View raw UK-DALE power data from any house and channel
This shows the actual power consumption in Watts before any normalization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# UK-DALE data directory - updated to correct location
DATA_DIR = 'NILM-main/dataset_preprocess/UK_DALE/'

# Appliance channel mappings (House 1)
APPLIANCE_CHANNELS = {
    'kettle': {'house': 1, 'channel': 10},
    'microwave': {'house': 1, 'channel': 13},
    'fridge': {'house': 1, 'channel': 12},
    'dishwasher': {'house': 1, 'channel': 6},
    'washingmachine': {'house': 1, 'channel': 5},
}

def load_raw_data(house, channel, max_samples=None):
    """Load raw power data from UK-DALE"""
    filepath = os.path.join(DATA_DIR, f'house_{house}', f'channel_{channel}.dat')
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print(f"\nPlease check:")
        print(f"1. UK-DALE data is in: {DATA_DIR}")
        print(f"2. House {house} exists")
        print(f"3. Channel {channel} exists")
        return None
    
    print(f"Loading data from: {filepath}")
    
    # Read the data file
    df = pd.read_table(filepath, sep=' ', names=['timestamp', 'power'], nrows=max_samples)
    
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
    
    print(f"\nLoaded {len(df):,} samples")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    
    return df

def analyze_power_data(df, appliance_name="Appliance"):
    """Analyze and display power statistics"""
    power = df['power'].values
    
    print(f"\n{'='*60}")
    print(f"RAW {appliance_name.upper()} POWER STATISTICS (in Watts)")
    print(f"{'='*60}")
    print(f"Min:        {power.min():.2f} W")
    print(f"Max:        {power.max():.2f} W")
    print(f"Mean:       {power.mean():.2f} W")
    print(f"Median:     {np.median(power):.2f} W")
    print(f"Std Dev:    {power.std():.2f} W")
    print(f"25th %ile:  {np.percentile(power, 25):.2f} W")
    print(f"75th %ile:  {np.percentile(power, 75):.2f} W")
    print(f"95th %ile:  {np.percentile(power, 95):.2f} W")
    print(f"99th %ile:  {np.percentile(power, 99):.2f} W")
    
    # Analyze ON/OFF states
    threshold = 10  # Watts - consider < 10W as OFF
    on_samples = power[power >= threshold]
    off_samples = power[power < threshold]
    
    print(f"\n{'='*60}")
    print(f"ON/OFF STATE ANALYSIS (threshold: {threshold}W)")
    print(f"{'='*60}")
    print(f"OFF samples: {len(off_samples):,} ({len(off_samples)/len(power)*100:.1f}%)")
    print(f"ON samples:  {len(on_samples):,} ({len(on_samples)/len(power)*100:.1f}%)")
    
    if len(on_samples) > 0:
        print(f"\nON state statistics:")
        print(f"  Min:    {on_samples.min():.2f} W")
        print(f"  Max:    {on_samples.max():.2f} W")
        print(f"  Mean:   {on_samples.mean():.2f} W")
        print(f"  Median: {np.median(on_samples):.2f} W")
    
    return power

def visualize_power(df, appliance_name, house, channel, max_plot_samples=10000):
    """Visualize power consumption over time"""
    # Limit samples for visualization
    if len(df) > max_plot_samples:
        print(f"\nPlotting first {max_plot_samples:,} samples for visualization")
        df_plot = df.iloc[:max_plot_samples].copy()
    else:
        df_plot = df.copy()
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Time series
    ax1.plot(df_plot['time'], df_plot['power'], linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (Watts)')
    ax1.set_title(f'UK-DALE House {house}, Channel {channel} - Raw {appliance_name} Power', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Full dataset stats:\n"
    stats_text += f"Min: {df['power'].min():.1f}W\n"
    stats_text += f"Max: {df['power'].max():.1f}W\n"
    stats_text += f"Mean: {df['power'].mean():.1f}W\n"
    stats_text += f"Median: {df['power'].median():.1f}W"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Histogram
    ax2.hist(df['power'], bins=100, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Power (Watts)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Power Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(df['power'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["power"].mean():.1f}W')
    ax2.axvline(df['power'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["power"].median():.1f}W')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='View raw UK-DALE power data')
    parser.add_argument('--appliance', type=str, choices=list(APPLIANCE_CHANNELS.keys()),
                       help='Appliance name (auto-selects house and channel)')
    parser.add_argument('--house', type=int, help='House number')
    parser.add_argument('--channel', type=int, help='Channel number')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to load')
    
    args = parser.parse_args()
    
    # Determine house and channel
    if args.appliance:
        # Use predefined appliance mapping
        appliance_name = args.appliance
        house = APPLIANCE_CHANNELS[appliance_name]['house']
        channel = APPLIANCE_CHANNELS[appliance_name]['channel']
        print(f"\n{'='*60}")
        print(f"UK-DALE RAW {appliance_name.upper()} DATA VIEWER")
        print(f"{'='*60}")
    elif args.house and args.channel:
        # Use manually specified house and channel
        house = args.house
        channel = args.channel
        appliance_name = f"House{house}_Ch{channel}"
        print(f"\n{'='*60}")
        print(f"UK-DALE RAW DATA VIEWER")
        print(f"{'='*60}")
    else:
        # Interactive mode
        print(f"\n{'='*60}")
        print(f"UK-DALE RAW DATA VIEWER - INTERACTIVE MODE")
        print(f"{'='*60}")
        print(f"\nAvailable appliances: {', '.join(APPLIANCE_CHANNELS.keys())}")
        user_input = input("\nEnter appliance name (or 'custom' for manual house/channel): ").strip().lower()
        
        if user_input in APPLIANCE_CHANNELS:
            appliance_name = user_input
            house = APPLIANCE_CHANNELS[appliance_name]['house']
            channel = APPLIANCE_CHANNELS[appliance_name]['channel']
        else:
            appliance_name = "Custom"
            house = int(input("Enter house number: ").strip())
            channel = int(input("Enter channel number: ").strip())
    
    print(f"House: {house}")
    print(f"Channel: {channel}")
    
    # Load data
    df = load_raw_data(house, channel, max_samples=args.max_samples)
    
    if df is None:
        return
    
    # Analyze
    power = analyze_power_data(df, appliance_name)
    
    # Visualize
    print(f"\n{'='*60}")
    print(f"Opening visualization...")
    print(f"{'='*60}")
    visualize_power(df, appliance_name, house, channel)

if __name__ == '__main__':
    main()
