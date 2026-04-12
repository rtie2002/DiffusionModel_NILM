"""
concat_quarterly_samples.py
============================
Concatenate per-quarter NPY files into one final output, preserving
temporal order: Q1 → Q2 → Q3 → Q4 → Q1 → Q2 → ... (for multiple cycles).

Usage (called internally by run_quarterly_diffusion.sh):
    python concat_quarterly_samples.py \\
        --files path/to/q1_c1.npy path/to/q2_c1.npy ... path/to/q4_c2.npy \\
        --output OUTPUT/washingmachine_multivariate/ddpm_fake_washingmachine_multivariate.npy
"""

import argparse
import os
import numpy as np


def concat_npy_files(file_paths: list, output_path: str) -> np.ndarray:
    """
    Loads and concatenates a list of NPY files along axis 0 (window dimension).
    Files are concatenated IN THE ORDER given, so pass them in Q1→Q4 temporal order.
    """
    arrays = []
    for fp in file_paths:
        if not os.path.isfile(fp):
            print(f"  [WARNING] File not found, skipping: {fp}")
            continue
        arr = np.load(fp)
        arrays.append(arr)
        print(f"  Loaded {fp}  shape={arr.shape}")

    if not arrays:
        raise RuntimeError("No valid NPY files found to concatenate.")

    combined = np.concatenate(arrays, axis=0)
    print(f"\n  Combined shape: {combined.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, combined)
    print(f"  Saved -> {output_path}")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate quarterly NPY samples in temporal order.')
    parser.add_argument('--files', nargs='+', required=True,
                        help='NPY file paths in the desired concat order '
                             '(Q1_c1, Q2_c1, ..., Q4_c1, Q1_c2, Q2_c2, ...)')
    parser.add_argument('--output', required=True,
                        help='Output NPY path for the combined array.')
    args = parser.parse_args()

    print(f"[concat_quarterly_samples] Concatenating {len(args.files)} file(s)...")
    result = concat_npy_files(args.files, args.output)
    print(f"\nFinal dataset: {result.shape[0]} windows x {result.shape[1]} timesteps x {result.shape[2]} features")


if __name__ == '__main__':
    main()
