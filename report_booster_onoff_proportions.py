#!/usr/bin/env python3
"""Report active/inactive window proportions before and after booster expansion.

The calculation mirrors CustomDataset in Utils/Data_utils/real_datasets.py:
- read the configured appliance power column
- MinMax scale the power channel and map it to [-1, 1]
- define active windows by max(window_power) > boost_threshold
- split indices with the same seeded permutation logic
- apply the training-time Active-Event Continuity Booster index expansion
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_APPLIANCES = ["fridge", "microwave", "kettle", "dishwasher", "washingmachine"]
DEFAULT_BOOST_FACTOR = 4
DEFAULT_BOOST_THRESHOLD = 0.2
DEFAULT_JITTER_LIMIT = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ON/OFF window proportions before and after Active-Event Continuity Booster."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="YAML config files. If omitted, Config/<appliance>.yaml is used for the default appliance list.",
    )
    parser.add_argument(
        "--appliances",
        default=",".join(DEFAULT_APPLIANCES),
        help="Comma-separated appliance names used when --configs is omitted.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--proportion", type=float, default=None)
    parser.add_argument("--output-csv", default="OUTPUT/active_window_proportions.csv")
    parser.add_argument("--output-md", default="OUTPUT/active_window_proportions.md")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.full_load(f)


def resolve_path(path_text, base_dir):
    path = Path(str(path_text).strip().strip("'\""))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def find_power_column(csv_path, configured_name):
    header = pd.read_csv(csv_path, nrows=0)
    target = configured_name.lower()
    for col in header.columns:
        col_lower = col.lower()
        if col_lower == target or col_lower == "power":
            return col
    return header.columns[0]


def neg_one_normalize(power_values):
    min_value = float(np.min(power_values))
    max_value = float(np.max(power_values))
    if np.isclose(max_value, min_value):
        return np.zeros_like(power_values, dtype=np.float32)
    scaled = (power_values - min_value) / (max_value - min_value)
    return (scaled * 2.0 - 1.0).astype(np.float32)


def sliding_window_active_mask(power_norm, window, threshold):
    rolling_max = pd.Series(power_norm).rolling(window=window).max().to_numpy()[window - 1 :]
    return rolling_max > threshold


def divide_indices(indices, ratio, seed):
    size = indices.shape[0]
    state = np.random.get_state()
    np.random.seed(seed)
    regular_train_num = int(np.ceil(size * ratio))
    shuffled = np.random.permutation(size)
    train_ids = shuffled[:regular_train_num]
    test_ids = shuffled[regular_train_num:]
    np.random.set_state(state)
    return indices[train_ids], indices[test_ids]


def compute_one(config_path, seed, proportion_override):
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent if config_path.parent.name.lower() == "config" else Path.cwd()
    config = load_yaml(config_path)

    params = config["dataloader"]["train_dataset"]["params"]
    name = str(params["name"])
    data_path = resolve_path(params["data_root"], project_root)
    window = int(params.get("window", config["model"]["params"].get("seq_length", 512)))
    proportion = float(proportion_override if proportion_override is not None else params.get("proportion", 1.0))
    boost_factor = int(params.get("boost_factor", DEFAULT_BOOST_FACTOR))
    boost_threshold = float(params.get("boost_threshold", DEFAULT_BOOST_THRESHOLD))
    jitter_limit = int(params.get("jitter_limit", DEFAULT_JITTER_LIMIT))

    power_col = find_power_column(data_path, name)
    power = pd.read_csv(data_path, usecols=[power_col])[power_col].to_numpy(dtype=np.float32)
    power_norm = neg_one_normalize(power)

    sample_num_total = max(len(power_norm) - window + 1, 0)
    indices = np.arange(sample_num_total)
    active_mask = sliding_window_active_mask(power_norm, window, boost_threshold)

    np.random.seed(seed)
    train_indices, _ = divide_indices(indices, proportion, seed)
    before_indices = train_indices.copy()
    before_on = int(active_mask[before_indices].sum())
    before_total = int(before_indices.shape[0])

    booster_applied = False
    after_indices = before_indices
    if name.lower() != "fridge" and before_on > 0 and boost_factor > 1:
        active_ids = before_indices[active_mask[before_indices]]
        boosted_versions = [before_indices]
        for _ in range(boost_factor - 1):
            jitter = np.random.randint(-jitter_limit, jitter_limit + 1, size=len(active_ids))
            jittered_active = np.clip(active_ids + jitter, 0, sample_num_total - 1)
            boosted_versions.append(jittered_active)
        after_indices = np.concatenate(boosted_versions)
        booster_applied = True

    after_on = int(active_mask[after_indices].sum())
    after_total = int(after_indices.shape[0])

    def pct(part, total):
        return 100.0 * part / total if total else 0.0

    return {
        "Appliance": name,
        "DataFile": str(data_path),
        "WindowLength": window,
        "Seed": seed,
        "BoostFactor": 1 if name.lower() == "fridge" else boost_factor,
        "JitterLimit": 0 if name.lower() == "fridge" else jitter_limit,
        "BoostThreshold": boost_threshold,
        "BoosterApplied": "Yes" if booster_applied else "No",
        "BeforeTotal": before_total,
        "BeforeON": before_on,
        "BeforeOFF": before_total - before_on,
        "BeforeONPct": pct(before_on, before_total),
        "BeforeOFFPct": pct(before_total - before_on, before_total),
        "AfterTotal": after_total,
        "AfterON": after_on,
        "AfterOFF": after_total - after_on,
        "AfterONPct": pct(after_on, after_total),
        "AfterOFFPct": pct(after_total - after_on, after_total),
    }


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Appliance",
        "WindowLength",
        "Seed",
        "BoostFactor",
        "JitterLimit",
        "BoostThreshold",
        "BoosterApplied",
        "BeforeTotal",
        "BeforeON",
        "BeforeOFF",
        "BeforeONPct",
        "BeforeOFFPct",
        "AfterTotal",
        "AfterON",
        "AfterOFF",
        "AfterONPct",
        "AfterOFFPct",
        "DataFile",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["BeforeONPct", "BeforeOFFPct", "AfterONPct", "AfterOFFPct"]:
                out[key] = f"{out[key]:.2f}"
            writer.writerow(out)


def write_markdown(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| Appliance | Before ON/OFF (%) | After ON/OFF (%) | Windows before | Windows after | Booster |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            before = f"{row['BeforeONPct']:.2f}/{row['BeforeOFFPct']:.2f}"
            after = f"{row['AfterONPct']:.2f}/{row['AfterOFFPct']:.2f}"
            booster = "skipped" if row["Appliance"].lower() == "fridge" else f"k={row['BoostFactor']}, delta=+/-{row['JitterLimit']}"
            f.write(
                f"| {row['Appliance']} | {before} | {after} | {row['BeforeTotal']} | {row['AfterTotal']} | {booster} |\n"
            )


def print_table(rows):
    print("\nActive-window proportions before and after booster")
    print("-" * 92)
    print(
        f"{'Appliance':<16} {'Before ON/OFF (%)':>20} {'After ON/OFF (%)':>20} "
        f"{'Windows before':>15} {'Windows after':>14}"
    )
    print("-" * 92)
    for row in rows:
        before = f"{row['BeforeONPct']:.2f}/{row['BeforeOFFPct']:.2f}"
        after = f"{row['AfterONPct']:.2f}/{row['AfterOFFPct']:.2f}"
        print(f"{row['Appliance']:<16} {before:>20} {after:>20} {row['BeforeTotal']:>15} {row['AfterTotal']:>14}")


def main():
    args = parse_args()
    root = Path.cwd()
    if args.configs:
        configs = [Path(c) for c in args.configs]
    else:
        configs = [root / "Config" / f"{app.strip()}.yaml" for app in args.appliances.split(",") if app.strip()]

    rows = [compute_one(config, args.seed, args.proportion) for config in configs]
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print_table(rows)
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
