#!/usr/bin/env python3
"""Report ON/OFF proportions before and after Active-Event Continuity Booster.

The script reports two related views:
1. Time-step ON/OFF percentages, using the appliance Watt-level ON threshold.
2. Active-window ON/OFF percentages, using the booster definition in training.

The booster calculation mirrors Utils/Data_utils/real_datasets.py and reads the
booster parameters from each appliance YAML file.
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
        description="Compute ON/OFF proportions before and after Active-Event Continuity Booster."
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
    parser.add_argument("--preprocess-config", default="Config/preprocess/preprocess_multivariate.yaml")
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


def canonical_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_power_column(csv_path, configured_name):
    header = pd.read_csv(csv_path, nrows=0)
    target = canonical_name(configured_name)
    for col in header.columns:
        if canonical_name(col) == target or canonical_name(col) == "power":
            return col
    return header.columns[0]


def neg_one_normalize(power_values):
    min_value = float(np.min(power_values))
    max_value = float(np.max(power_values))
    if np.isclose(max_value, min_value):
        return np.zeros_like(power_values, dtype=np.float32)
    scaled = (power_values - min_value) / (max_value - min_value)
    return (scaled * 2.0 - 1.0).astype(np.float32)


def sliding_window_active_mask(power_values, window, threshold):
    rolling_max = pd.Series(power_values).rolling(window=window).max().to_numpy()[window - 1 :]
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


def pct(part, total):
    return 100.0 * part / total if total else 0.0


def count_window_timesteps(power_values, indices, window, threshold, chunk_size=5000):
    if len(indices) == 0:
        return 0, 0

    offsets = np.arange(window)
    on_count = 0
    total_count = int(len(indices) * window)

    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        loc = chunk[:, None] + offsets[None, :]
        values = power_values[loc]
        on_count += int(np.count_nonzero(values > threshold))

    return on_count, total_count


def load_state_params(preprocess_config, appliance_name):
    preprocess_path = Path(preprocess_config)
    if not preprocess_path.is_absolute():
        preprocess_path = (Path.cwd() / preprocess_path).resolve()

    params = load_yaml(preprocess_path).get("appliances", {})
    key = canonical_name(appliance_name)
    for app_name, app_params in params.items():
        if canonical_name(app_name) == key:
            return app_params

    raise KeyError(f"Cannot find appliance '{appliance_name}' in {preprocess_path}")


def threshold_in_data_units(power_values, threshold_watts, max_power):
    if float(np.nanmax(power_values)) <= 1.5:
        return float(threshold_watts) / float(max_power), "normalized"
    return float(threshold_watts), "watts"


def compute_one(config_path, seed, proportion_override, preprocess_config):
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
    neg_one_to_one = bool(params.get("neg_one_to_one", True))

    state_params = load_state_params(preprocess_config, name)
    on_threshold_w = float(state_params["on_power_threshold"])
    max_power = float(state_params["max_power"])

    power_col = find_power_column(data_path, name)
    power_raw = pd.read_csv(data_path, usecols=[power_col])[power_col].to_numpy(dtype=np.float32)
    power_for_booster = neg_one_normalize(power_raw) if neg_one_to_one else power_raw
    on_threshold_data, threshold_unit = threshold_in_data_units(power_raw, on_threshold_w, max_power)

    sample_num_total = max(len(power_raw) - window + 1, 0)
    indices = np.arange(sample_num_total)
    active_mask = sliding_window_active_mask(power_for_booster, window, boost_threshold)

    np.random.seed(seed)
    before_indices, _ = divide_indices(indices, proportion, seed)
    before_indices = before_indices.copy()
    active_ids = before_indices[active_mask[before_indices]]

    after_indices = before_indices
    booster_applied = False
    if len(active_ids) > 0 and boost_factor > 1:
        boosted_versions = [before_indices]
        for _ in range(boost_factor - 1):
            jitter = np.random.randint(-jitter_limit, jitter_limit + 1, size=len(active_ids))
            jittered_active = np.clip(active_ids + jitter, 0, sample_num_total - 1)
            boosted_versions.append(jittered_active)
        after_indices = np.concatenate(boosted_versions)
        booster_applied = True

    input_on = int(np.count_nonzero(power_raw > on_threshold_data))
    input_total = int(power_raw.shape[0])

    before_step_on, before_step_total = count_window_timesteps(
        power_raw, before_indices, window, on_threshold_data
    )
    after_step_on, after_step_total = count_window_timesteps(
        power_raw, after_indices, window, on_threshold_data
    )

    before_window_total = int(before_indices.shape[0])
    before_window_on = int(active_mask[before_indices].sum())
    after_window_total = int(after_indices.shape[0])
    after_window_on = int(active_mask[after_indices].sum())

    return {
        "Appliance": name,
        "DataFile": str(data_path),
        "PowerColumn": power_col,
        "WindowLength": window,
        "Seed": seed,
        "BoostFactor": boost_factor,
        "JitterLimit": jitter_limit,
        "BoostThreshold": boost_threshold,
        "BoosterApplied": "Yes" if booster_applied else "No",
        "StateThresholdW": on_threshold_w,
        "StateThresholdData": on_threshold_data,
        "StateThresholdDataUnit": threshold_unit,
        "InputPoints": input_total,
        "InputON": input_on,
        "InputOFF": input_total - input_on,
        "InputONPct": pct(input_on, input_total),
        "InputOFFPct": pct(input_total - input_on, input_total),
        "BeforeStepTotal": before_step_total,
        "BeforeStepON": before_step_on,
        "BeforeStepOFF": before_step_total - before_step_on,
        "BeforeStepONPct": pct(before_step_on, before_step_total),
        "BeforeStepOFFPct": pct(before_step_total - before_step_on, before_step_total),
        "AfterStepTotal": after_step_total,
        "AfterStepON": after_step_on,
        "AfterStepOFF": after_step_total - after_step_on,
        "AfterStepONPct": pct(after_step_on, after_step_total),
        "AfterStepOFFPct": pct(after_step_total - after_step_on, after_step_total),
        "BeforeWindowTotal": before_window_total,
        "BeforeWindowON": before_window_on,
        "BeforeWindowOFF": before_window_total - before_window_on,
        "BeforeWindowONPct": pct(before_window_on, before_window_total),
        "BeforeWindowOFFPct": pct(before_window_total - before_window_on, before_window_total),
        "AfterWindowTotal": after_window_total,
        "AfterWindowON": after_window_on,
        "AfterWindowOFF": after_window_total - after_window_on,
        "AfterWindowONPct": pct(after_window_on, after_window_total),
        "AfterWindowOFFPct": pct(after_window_total - after_window_on, after_window_total),
    }


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Appliance",
        "StateThresholdW",
        "InputONPct",
        "InputOFFPct",
        "BeforeStepONPct",
        "BeforeStepOFFPct",
        "AfterStepONPct",
        "AfterStepOFFPct",
        "BeforeWindowONPct",
        "BeforeWindowOFFPct",
        "AfterWindowONPct",
        "AfterWindowOFFPct",
        "InputPoints",
        "BeforeStepTotal",
        "AfterStepTotal",
        "BeforeWindowTotal",
        "AfterWindowTotal",
        "WindowLength",
        "Seed",
        "BoostFactor",
        "JitterLimit",
        "BoostThreshold",
        "BoosterApplied",
        "StateThresholdData",
        "StateThresholdDataUnit",
        "PowerColumn",
        "DataFile",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in [
                "InputONPct",
                "InputOFFPct",
                "BeforeStepONPct",
                "BeforeStepOFFPct",
                "AfterStepONPct",
                "AfterStepOFFPct",
                "BeforeWindowONPct",
                "BeforeWindowOFFPct",
                "AfterWindowONPct",
                "AfterWindowOFFPct",
            ]:
                out[key] = f"{out[key]:.2f}"
            out["StateThresholdW"] = f"{out['StateThresholdW']:.0f}"
            out["StateThresholdData"] = f"{out['StateThresholdData']:.8f}"
            writer.writerow(out)


def write_markdown(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("## Time-step ON/OFF proportions\n\n")
        f.write("| Appliance | ON threshold | Input ON/OFF (%) | Before booster ON/OFF (%) | After booster ON/OFF (%) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            input_pct = f"{row['InputONPct']:.2f}/{row['InputOFFPct']:.2f}"
            before = f"{row['BeforeStepONPct']:.2f}/{row['BeforeStepOFFPct']:.2f}"
            after = f"{row['AfterStepONPct']:.2f}/{row['AfterStepOFFPct']:.2f}"
            f.write(f"| {row['Appliance']} | > {row['StateThresholdW']:.0f} W | {input_pct} | {before} | {after} |\n")

        f.write("\n## Active-window ON/OFF proportions\n\n")
        f.write("| Appliance | Before ON/OFF (%) | After ON/OFF (%) | Windows before | Windows after | Booster |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            before = f"{row['BeforeWindowONPct']:.2f}/{row['BeforeWindowOFFPct']:.2f}"
            after = f"{row['AfterWindowONPct']:.2f}/{row['AfterWindowOFFPct']:.2f}"
            booster = f"k={row['BoostFactor']}, delta=+/-{row['JitterLimit']}"
            f.write(
                f"| {row['Appliance']} | {before} | {after} | "
                f"{row['BeforeWindowTotal']} | {row['AfterWindowTotal']} | {booster} |\n"
            )


def print_tables(rows):
    print("\nTime-step ON/OFF proportions before and after booster")
    print("-" * 102)
    print(
        f"{'Appliance':<16} {'ON threshold':>14} {'Input ON/OFF (%)':>20} "
        f"{'Before booster (%)':>20} {'After booster (%)':>20}"
    )
    print("-" * 102)
    for row in rows:
        input_pct = f"{row['InputONPct']:.2f}/{row['InputOFFPct']:.2f}"
        before = f"{row['BeforeStepONPct']:.2f}/{row['BeforeStepOFFPct']:.2f}"
        after = f"{row['AfterStepONPct']:.2f}/{row['AfterStepOFFPct']:.2f}"
        threshold = f"> {row['StateThresholdW']:.0f} W"
        print(f"{row['Appliance']:<16} {threshold:>14} {input_pct:>20} {before:>20} {after:>20}")

    print("\nActive-window ON/OFF proportions before and after booster")
    print("-" * 92)
    print(
        f"{'Appliance':<16} {'Before ON/OFF (%)':>20} {'After ON/OFF (%)':>20} "
        f"{'Windows before':>15} {'Windows after':>14}"
    )
    print("-" * 92)
    for row in rows:
        before = f"{row['BeforeWindowONPct']:.2f}/{row['BeforeWindowOFFPct']:.2f}"
        after = f"{row['AfterWindowONPct']:.2f}/{row['AfterWindowOFFPct']:.2f}"
        print(
            f"{row['Appliance']:<16} {before:>20} {after:>20} "
            f"{row['BeforeWindowTotal']:>15} {row['AfterWindowTotal']:>14}"
        )


def main():
    args = parse_args()
    root = Path.cwd()
    if args.configs:
        configs = [Path(c) for c in args.configs]
    else:
        configs = [root / "Config" / f"{app.strip()}.yaml" for app in args.appliances.split(",") if app.strip()]

    rows = [compute_one(config, args.seed, args.proportion, args.preprocess_config) for config in configs]
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print_tables(rows)
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
