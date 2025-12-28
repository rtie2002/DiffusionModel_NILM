"""
Calculate Mean and Std for UK-DALE Dataset
计算UK-DALE数据集的均值和标准差

This script calculates the correct mean and std for each appliance
across ALL buildings to ensure consistent normalization.
"""

import pandas as pd
import numpy as np
import os

# Channel mapping from labels.dat
CHANNEL_MAPPING = {
    1: {  # Building 1
        'aggregate': 1,
        'fridge': 12,
        'washingmachine': 5,
        'dishwasher': 6,
        'kettle': 10,
        'microwave': 13
    },
    2: {  # Building 2
        'aggregate': 1,
        'fridge': 14,
        'washingmachine': 12,
        'dishwasher': 13,
        'kettle': 8,
        'microwave': 15
    }
}

def load_channel_data(house, channel, sample_rate='6s', max_samples=None):
    """Load data from a specific channel"""
    data_path = f'NILM-main/dataset_preprocess/UK_DALE/house_{house}/channel_{channel}.dat'
    
    if not os.path.exists(data_path):
        print(f"  Warning: {data_path} not found")
        return None
    
    # Read data
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=['time', 'power'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')
    
    # Resample
    df = df.resample(sample_rate).mean().fillna(method='ffill', limit=30)
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
    
    return df['power'].values

def calculate_stats_for_appliance(appliance_name, buildings=[1, 2], sample_rate='6s', max_samples_per_building=100000):
    """Calculate mean and std for an appliance across multiple buildings"""
    print(f"\n{'='*60}")
    print(f"Calculating stats for: {appliance_name.upper()}")
    print(f"{'='*60}")
    
    all_data = []
    
    for building in buildings:
        if building not in CHANNEL_MAPPING:
            print(f"  Warning: Building {building} not in mapping")
            continue
        
        if appliance_name not in CHANNEL_MAPPING[building]:
            print(f"  Warning: {appliance_name} not found in Building {building}")
            continue
        
        channel = CHANNEL_MAPPING[building][appliance_name]
        print(f"\nBuilding {building}, Channel {channel}:")
        
        data = load_channel_data(building, channel, sample_rate, max_samples_per_building)
        
        if data is not None:
            # Remove negative values and zeros
            data = data[data > 0]
            
            if len(data) == 0:
                print(f"  Warning: No valid data (all zeros or negative)")
                continue
            
            print(f"  Samples: {len(data):,}")
            print(f"  Mean: {data.mean():.2f}W")
            print(f"  Std: {data.std():.2f}W")
            print(f"  Min: {data.min():.2f}W")
            print(f"  Max: {data.max():.2f}W")
            
            all_data.append(data)
    
    if not all_data:
        print(f"\n  ERROR: No data found for {appliance_name}")
        return None
    
    # Combine all data
    combined_data = np.concatenate(all_data)
    
    # Calculate overall statistics
    mean = combined_data.mean()
    std = combined_data.std()
    min_val = combined_data.min()
    max_val = combined_data.max()
    
    print(f"\n{'='*60}")
    print(f"COMBINED STATISTICS (All Buildings)")
    print(f"{'='*60}")
    print(f"  Total samples: {len(combined_data):,}")
    print(f"  Mean: {mean:.2f}W")
    print(f"  Std: {std:.2f}W")
    print(f"  Min: {min_val:.2f}W")
    print(f"  Max: {max_val:.2f}W")
    
    return {
        'mean': round(mean),
        'std': round(std),
        'min': round(min_val),
        'max': round(max_val),
        'samples': len(combined_data)
    }

def main():
    print("="*80)
    print("UK-DALE STATISTICS CALCULATOR")
    print("="*80)
    print("\nCalculating mean and std for each appliance across all buildings...")
    print("This ensures consistent normalization parameters.\n")
    
    appliances = ['aggregate', 'fridge', 'washingmachine', 'dishwasher', 'kettle', 'microwave']
    buildings = [1, 2]
    
    results = {}
    
    for appliance in appliances:
        stats = calculate_stats_for_appliance(appliance, buildings, sample_rate='6s', max_samples_per_building=100000)
        if stats:
            results[appliance] = stats
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - RECOMMENDED PARAMETERS")
    print("="*80)
    print("\nparams_appliance = {")
    
    for appliance in ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']:
        if appliance in results:
            stats = results[appliance]
            print(f"    '{appliance}': {{")
            print(f"        'mean': {stats['mean']},")
            print(f"        'std': {stats['std']},")
            print(f"        'max_on_power': {stats['max']},")
            print(f"        # Calculated from {stats['samples']:,} samples across Buildings {buildings}")
            print(f"    }},")
    
    if 'aggregate' in results:
        stats = results['aggregate']
        print(f"    'aggregate': {{")
        print(f"        'mean': {stats['mean']},")
        print(f"        'std': {stats['std']},")
        print(f"    }}")
    
    print("}")
    
    print("\n" + "="*80)
    print("NOTE: These values should be used for BOTH buildings to ensure")
    print("consistent normalization across the entire dataset.")
    print("="*80)

if __name__ == '__main__':
    main()
