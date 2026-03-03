# -*- coding: utf-8 -*-
"""
Batch Convert Z-Score CSVs to Real Power (Watts) — Subfolder Version

Automatically scans all appliance subfolders under created_data/UK_DALE/
(i.e. dishwasher/, fridge/, kettle/, microwave/, washingmachine/)
and converts every CSV inside from Z-score to Watts in-place,
saving results with a _realPower suffix.

Same conversion logic as convert_zscore_folder_to_watts.py, but targets
the appliance subfolders instead of the root directory.

Usage:
    python convert_zscore_subfolders_to_watts.py
    python convert_zscore_subfolders_to_watts.py --dry-run   # preview only
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path

# ── Normalization constants ────────────────────────────────────────────────────
AGG_MEAN = 522
AGG_STD  = 814

APPLIANCE_SPECS = {
    'kettle':         {'mean': 700, 'std': 1000, 'max_power': 3998},
    'microwave':      {'mean': 500, 'std':  800, 'max_power': 2000},
    'fridge':         {'mean': 200, 'std':  400, 'max_power':  350},
    'dishwasher':     {'mean': 700, 'std': 1000, 'max_power': 3964},
    'washingmachine': {'mean': 400, 'std':  700, 'max_power': 3999},
}

# Root directory (relative to this script's location)
BASE_DIR = Path(__file__).parent  # .../created_data/UK_DALE


# ── Core conversion ───────────────────────────────────────────────────────────
def convert_to_watts(df: pd.DataFrame, appliance_name: str) -> pd.DataFrame | None:
    """Convert Z-score dataframe to Watts.  Returns modified df or None."""
    if df.empty:
        return None

    cols = df.columns.tolist()

    # ── Aggregate column ──────────────────────────────────────────────────────
    agg_col = None
    if 'aggregate' in cols:
        agg_col = 'aggregate'
    elif '0' in cols:
        agg_col = '0'
    elif cols:
        agg_col = cols[0]

    if agg_col is not None:
        print(f"    - Aggregate column : '{agg_col}'")
        df[agg_col] = (df[agg_col] * AGG_STD + AGG_MEAN).clip(lower=0)

    # ── Appliance column ──────────────────────────────────────────────────────
    app_col = None
    if appliance_name in cols:
        app_col = appliance_name
    elif 'appliance' in cols:
        app_col = 'appliance'
    elif '1' in cols:
        app_col = '1'
    elif len(cols) > 1:
        app_col = cols[1]

    if app_col is not None and appliance_name in APPLIANCE_SPECS:
        print(f"    - Appliance column : '{app_col}'")
        s = APPLIANCE_SPECS[appliance_name]
        df[app_col] = (df[app_col] * s['std'] + s['mean']).clip(lower=0)
    else:
        print(f"    [Warning] Cannot identify appliance column for '{appliance_name}', skipping.")

    # Time-feature columns (cols 2+) are left untouched (sin/cos encoded)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main(dry_run: bool = False):
    print("=" * 60)
    print("  Batch Z-Score → Watts Converter  (Subfolder Edition)")
    print("=" * 60)
    print(f"Base directory : {BASE_DIR}")
    print(f"Constants      : Agg Mean={AGG_MEAN}, Agg Std={AGG_STD}")
    if dry_run:
        print("  *** DRY-RUN MODE — no files will be written ***")
    print()

    total_converted = 0
    total_skipped   = 0
    total_errors    = 0

    for appliance_name in APPLIANCE_SPECS:
        subfolder = BASE_DIR / appliance_name
        if not subfolder.is_dir():
            print(f"[SKIP] Subfolder not found: {subfolder}")
            continue

        csv_files = sorted(subfolder.glob("*.csv"))
        if not csv_files:
            print(f"[SKIP] No CSV files found in: {subfolder}")
            continue

        print(f"\n── {appliance_name.upper()} ({len(csv_files)} files) ──────────────────────")

        for file_path in csv_files:
            filename = file_path.name

            # Skip already converted files
            if "_realPower" in filename:
                print(f"  [SKIP] Already converted: {filename}")
                total_skipped += 1
                continue

            print(f"\n  Processing : {filename}")

            try:
                df = pd.read_csv(file_path)
                df_watts = convert_to_watts(df.copy(), appliance_name)

                if df_watts is None:
                    print(f"  [SKIP] Empty or unreadable dataframe.")
                    total_skipped += 1
                    continue

                out_name = filename.replace(".csv", "_realPower.csv")
                out_path = subfolder / out_name

                if dry_run:
                    print(f"  [DRY-RUN] Would save → {out_name}")
                else:
                    df_watts.to_csv(out_path, index=False)
                    print(f"  Saved → {out_name}")

                total_converted += 1

            except Exception as exc:
                print(f"  [ERROR] {filename}: {exc}")
                total_errors += 1

    print()
    print("=" * 60)
    print(f"  Done.  Converted={total_converted}  "
          f"Skipped={total_skipped}  Errors={total_errors}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all Z-score CSVs in appliance subfolders to Watts."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview which files would be converted without writing anything."
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
