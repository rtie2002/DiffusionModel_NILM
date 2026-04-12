"""
split_csv_by_quarter.py
=======================
Splits any *_multivariate.csv (with month_sin / month_cos columns) into
four quarterly CSVs. No reprocessing of raw data needed.

Month is recovered from the sin/cos encoding via arctan2:
    month = round(atan2(month_sin, month_cos) / (2*pi/12)) mod 12  ->  1-12

Q1 = Jan-Mar   (months  1, 2, 3)
Q2 = Apr-Jun   (months  4, 5, 6)
Q3 = Jul-Sep   (months  7, 8, 9)
Q4 = Oct-Dec   (months 10,11,12)

Output filenames:
    <base>_Q1.csv, <base>_Q2.csv, <base>_Q3.csv, <base>_Q4.csv
    e.g.  washingmachine_multivariate_Q1.csv

Usage:
    python preprocess_NILMformer/split_csv_by_quarter.py \\
        --input washingmachine_multivariate.csv

    # custom output directory
    python preprocess_NILMformer/split_csv_by_quarter.py \\
        --input washingmachine_multivariate.csv \\
        --output_dir ./Data/datasets/
"""

import argparse
import os
import numpy as np
import pandas as pd

QUARTERS = {
    'Q1': [1, 2, 3],
    'Q2': [4, 5, 6],
    'Q3': [7, 8, 9],
    'Q4': [10, 11, 12],
}


def decode_month(df: pd.DataFrame) -> pd.Series:
    """Recover integer month 1-12 from month_sin and month_cos columns."""
    rad = np.arctan2(df['month_sin'].values, df['month_cos'].values)
    month_idx = np.round(rad / (2 * np.pi / 12)).astype(int) % 12
    month_idx[month_idx == 0] = 12  # sin/cos(2*pi) -> 0 which is December
    return pd.Series(month_idx, index=df.index, name='_month')


def split_by_quarter(input_csv: str, output_dir: str) -> dict:
    """
    Split input_csv by quarter. Returns a dict of {quarter: output_path}.
    Only includes quarters that actually contain data.
    """
    print(f"[split_by_quarter] Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    print(f"  Total rows: {total_rows:,}")

    if 'month_sin' not in df.columns or 'month_cos' not in df.columns:
        raise ValueError("CSV must contain 'month_sin' and 'month_cos' columns.")

    month_series = decode_month(df)
    unique_months = sorted(month_series.unique())
    print(f"  Detected months: {unique_months}")

    # Derive base path without extension
    base = os.path.splitext(os.path.abspath(input_csv))[0]
    # Strip any existing _Q? suffix to avoid double-suffix
    for q in QUARTERS:
        if base.endswith(f'_{q}'):
            base = base[:-len(f'_{q}')]

    os.makedirs(output_dir, exist_ok=True)
    created = {}

    for q_label, months in QUARTERS.items():
        mask = month_series.isin(months)
        subset = df[mask].reset_index(drop=True)

        if len(subset) == 0:
            print(f"  [{q_label}] No data for months {months} - skipping.")
            continue

        out_name = os.path.basename(base) + f'_{q_label}.csv'
        out_path = os.path.join(output_dir, out_name)
        subset.to_csv(out_path, index=False)
        created[q_label] = out_path
        pct = 100 * len(subset) / total_rows
        print(f"  [{q_label}] months={months}  rows={len(subset):,} ({pct:.1f}%)  -> {out_path}")

    print(f"  Done. {len(created)} quarter(s) created: {list(created.keys())}")
    return created


def main():
    parser = argparse.ArgumentParser(
        description='Split a multivariate NILM CSV into quarterly subsets using month_sin/cos.'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV (e.g. washingmachine_multivariate.csv)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. Defaults to same folder as input.')
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    split_by_quarter(args.input, out_dir)


if __name__ == '__main__':
    main()
