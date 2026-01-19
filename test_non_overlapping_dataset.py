"""
Test Non-Overlapping Window Dataset Creation
Verify that the new dataset covers all 12 months uniformly
"""

import sys
sys.path.append('.')

from Utils.Data_utils.real_datasets import CustomDataset
import numpy as np
import pandas as pd

def check_temporal_coverage(dataset, name="Dataset"):
    """Check month distribution in the dataset"""
    print("\n" + "="*70)
    print(f"Temporal Coverage Analysis: {name}")
    print("="*70)
    
    # Get all data
    if hasattr(dataset, 'samples'):
        if hasattr(dataset.samples, '__len__'):
            num_samples = len(dataset.samples)
            print(f"Total windows: {num_samples:,}")
            
            # Sample some windows to check
            sample_indices = np.linspace(0, num_samples-1, min(100, num_samples), dtype=int)
            
            all_months = []
            for idx in sample_indices:
                window_data = dataset.samples[idx]  # (512, 9)
                month_sin = window_data[:, 7]
                month_cos = window_data[:, 8]
                
                angles = np.arctan2(month_sin, month_cos)
                angles = np.where(angles < 0, angles + 2*np.pi, angles)
                months = (angles / (2 * np.pi) * 12).round().astype(int)
                months = np.where(months == 0, 12, months)
                
                all_months.extend(months)
            
            unique_months = sorted(set(all_months))
            counts = pd.Series(all_months).value_counts().sort_index()
            
            print(f"\nSampled {len(sample_indices)} windows:")
            print(f"  Unique months detected: {unique_months}")
            print(f"  Coverage: {len(unique_months)}/12 months")
            
            if len(unique_months) == 12:
                print("  ✓ FULL COVERAGE (All 12 months)")
            else:
                missing = set(range(1, 13)) - set(unique_months)
                print(f"  ✗ INCOMPLETE COVERAGE")
                print(f"  Missing months: {sorted(missing)}")
            
            print("\n  Month distribution:")
            for month in range(1, 13):
                count = counts.get(month, 0)
                symbol = "✓" if count > 0 else "✗"
                print(f"    {symbol} Month {month:2d}: {count:6,} samples")
            
            return len(unique_months) == 12
    else:
        print("  Unable to access samples")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING NON-OVERLAPPING WINDOW DATASET CREATION")
    print("="*70)
    
    # Test with fridge data
    appliance = "fridge_multivariate"
    
    print(f"\nCreating dataset for: {appliance}")
    print("Configuration:")
    print("  Window size: 512")
    print("  Proportion: 0.8 (80% train, 20% test)")
    print("  Period: train")
    
    try:
        dataset = CustomDataset(
            name=appliance,
            data_root="Data/datasets",
            window=512,
            proportion=0.8,
            save2npy=True,
            neg_one_to_one=True,
            seed=2024,
            period='train',
            output_dir='./OUTPUT'
        )
        
        print(f"\n✓ Dataset created successfully")
        print(f"  Number of windows: {len(dataset)}")
        
        # Check temporal coverage
        full_coverage = check_temporal_coverage(dataset, name=f"{appliance} (Train)")
        
        if full_coverage:
            print("\n" + "="*70)
            print("✓ SUCCESS: Full temporal coverage achieved!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✗ WARNING: Incomplete temporal coverage")
            print("="*70)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
