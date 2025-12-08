"""
Check Logical Consistency of Mixed Dataset
Validates that appliance power never exceeds aggregate power
"""

import pandas as pd
import numpy as np

def check_logic(filename='NILM-main/dataset_preprocess/created_data/UK_DALE/kettle_training_200k+200k.csv'):
    print("="*70)
    print("CHECKING LOGICAL CONSISTENCY")
    print("="*70)
    
    print("\nLoading dataset...")
    df = pd.read_csv(filename, header=None)
    print(f"Loaded: {df.shape}")
    
    print("\n" + "-"*70)
    print("Rule: Appliance Power <= Aggregate Power")
    print("(Physical constraint: Part cannot exceed whole)")
    print("-"*70)
    
    # Check violations
    violations = df[df[1] > df[0]]
    
    print(f"\nTotal points: {len(df):,}")
    print(f"Violations (Kettle > Aggregate): {len(violations):,}")
    
    if len(violations) > 0:
        print(f"Percentage: {len(violations)/len(df)*100:.4f}%")
        print("\n[ERROR] Logical inconsistencies found!")
        print("\nFirst 10 violations:")
        print(f"{'Index':<10} {'Aggregate':<15} {'Kettle':<15} {'Diff':<15}")
        print("-"*55)
        for idx in violations.head(10).index:
            agg = df.loc[idx, 0]
            app = df.loc[idx, 1]
            diff = app - agg
            print(f"{idx:<10} {agg:>12.4f}   {app:>12.4f}   {diff:>12.4f}")
        
        print(f"\nMax violation: {(df[1] - df[0]).max():.4f}")
        print(f"Mean violation: {(violations[1] - violations[0]).mean():.4f}")
    else:
        print("\n[OK] No violations found!")
        print("All data points are logically consistent.")
        print("Appliance power <= Aggregate power everywhere.")
    
    # Additional checks
    print("\n" + "-"*70)
    print("Additional Statistics")
    print("-"*70)
    
    diff = df[0] - df[1]  # Should always be >= 0
    print(f"\nAggregate - Kettle:")
    print(f"  Min:    {diff.min():>10.4f} {'[OK]' if diff.min() >= 0 else '[ERROR]'}")
    print(f"  Max:    {diff.max():>10.4f}")
    print(f"  Mean:   {diff.mean():>10.4f}")
    print(f"  Median: {diff.median():>10.4f}")
    
    # Check percentage where they're very close
    close_thresh = 0.1
    close_count = np.sum(np.abs(diff) < close_thresh)
    print(f"\nPoints where Agg ≈ Kettle (diff < {close_thresh}): {close_count:,} ({close_count/len(df)*100:.2f}%)")
    
    print("\n" + "="*70)
    if len(violations) == 0:
        print("[PASS] Dataset passes logical consistency check!")
    else:
        print("[FAIL] Dataset has logical inconsistencies!")
    print("="*70)

if __name__ == '__main__':
    check_logic()
