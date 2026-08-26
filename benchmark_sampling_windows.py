import argparse
import csv
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from engine.solver import Trainer
from Utils.io_utils import (
    instantiate_from_config,
    load_yaml_config,
    merge_opts_to_config,
    seed_everything,
)


DEFAULT_APPLIANCES = ["fridge", "microwave", "kettle", "dishwasher", "washingmachine"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TC-DiT sampling time for different numbers of generated windows."
    )
    parser.add_argument(
        "--appliances",
        nargs="+",
        default=DEFAULT_APPLIANCES,
        help="Appliances to benchmark, or use the default five-appliance list.",
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[100, 500, 1000, 2000],
        help="Generated-window counts to benchmark.",
    )
    parser.add_argument("--config-dir", default="Config", help="Directory containing appliance YAML configs.")
    parser.add_argument("--output", default="OUTPUT", help="Directory for benchmark outputs.")
    parser.add_argument("--milestone", type=int, default=10, help="Checkpoint milestone to load.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed used for benchmarking.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id. Use -1 for CPU.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Maximum generated windows per sampling batch.")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of sampling batch sizes to benchmark. Overrides --batch-size.",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=0,
        help="Optional unreported warm-up generation count before timing.",
    )
    parser.add_argument(
        "--sampling-mode",
        default="ordered_non_overlapping",
        choices=["ordered_non_overlapping", "ordered", "random"],
        help="Condition-template selection mode for sampling.",
    )
    parser.add_argument(
        "--opts",
        nargs="+",
        default=None,
        help="Optional config overrides, using the same key-value format as main.py.",
    )
    return parser.parse_args()


def format_duration(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_params(num_params):
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)


def make_sampling_dataset_config(config, seed):
    dataset_config = deepcopy(config["dataloader"]["train_dataset"])
    dataset_config["params"]["proportion"] = 0.0
    dataset_config["params"]["style"] = "non_overlapping"
    dataset_config["params"]["save2npy"] = False
    dataset_config["params"]["period"] = "test"
    dataset_config["params"]["seed"] = seed
    return dataset_config


def benchmark_appliance(args, appliance, device):
    config_path = Path(args.config_dir) / f"{appliance}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config = load_yaml_config(str(config_path))
    opts = list(args.opts or [])
    opts.extend(["dataloader.train_dataset.params.seed", str(args.seed)])
    config = merge_opts_to_config(config, opts)

    run_name = f"{appliance}_multivariate"
    runner_args = SimpleNamespace(name=run_name, save_dir=str(Path(args.output) / run_name))

    model = instantiate_from_config(config["model"]).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    trainer = Trainer(
        config=config,
        args=runner_args,
        model=model,
        dataloader={"dataloader": [], "dataset": None},
        logger=None,
    )
    checkpoint_path = trainer.results_folder / f"checkpoint-{args.milestone}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {appliance}: {checkpoint_path}\n"
            "Run training first or set --milestone to an existing checkpoint."
        )
    trainer.load(args.milestone)

    sampling_dataset = instantiate_from_config(make_sampling_dataset_config(config, args.seed))
    ordered = args.sampling_mode != "random"
    stride = 1
    rows = []
    batch_sizes = args.batch_sizes if args.batch_sizes else [args.batch_size]

    for batch_size in batch_sizes:
        if batch_size <= 0:
            continue

        print(f"\nSampling batch size: {batch_size}")
        if args.warmup_count > 0:
            warmup_count = min(args.warmup_count, max(args.counts))
            print(f"Warm-up generation: {warmup_count} windows (not recorded)")
            warmup = trainer.sample(
                num=warmup_count,
                size_every=min(batch_size, warmup_count),
                shape=[sampling_dataset.window, sampling_dataset.var_num],
                dataset=sampling_dataset,
                ordered=ordered,
                stride=stride,
            )
            del warmup
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        for count in args.counts:
            if count <= 0:
                continue

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            start = time.perf_counter()
            samples = trainer.sample(
                num=count,
                size_every=min(batch_size, count),
                shape=[sampling_dataset.window, sampling_dataset.var_num],
                dataset=sampling_dataset,
                ordered=ordered,
                stride=stride,
            )

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            output_shape = tuple(samples.shape)
            del samples

            equivalent_days = count * sampling_dataset.window / 1440.0
            rows.append(
                {
                    "Appliance": appliance,
                    "GeneratedWindows": count,
                    "BatchSize": batch_size,
                    "EquivalentDays": round(equivalent_days, 3),
                    "SamplingTimeSeconds": round(elapsed, 2),
                    "SamplingTime": format_duration(elapsed),
                    "SecondsPerWindow": round(elapsed / count, 4),
                    "WindowsPerSecond": round(count / elapsed, 4) if elapsed > 0 else "NA",
                    "SecondsPerEquivalentDay": round(elapsed / equivalent_days, 4) if equivalent_days > 0 else "NA",
                    "RandomSeed": args.seed,
                    "TrainableParameters": format_params(trainable_params),
                    "TotalParameters": format_params(total_params),
                    "OutputShape": str(output_shape),
                }
            )

    return rows


def write_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sampling_window_benchmark.csv"
    md_path = output_dir / "sampling_window_benchmark.md"

    fieldnames = [
        "Appliance",
        "GeneratedWindows",
        "BatchSize",
        "EquivalentDays",
        "SamplingTimeSeconds",
        "SamplingTime",
        "SecondsPerWindow",
        "WindowsPerSecond",
        "SecondsPerEquivalentDay",
        "RandomSeed",
        "TrainableParameters",
        "TotalParameters",
        "OutputShape",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Sampling Window Benchmark\n\n")
        f.write("Sampling time is measured for generation only and excludes saving synthetic arrays to disk.\n\n")
        f.write(
            "| Appliance | Generated windows | Batch size | Equivalent days | Sampling time | s/window | windows/s | s/equivalent day |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['Appliance']} | {row['GeneratedWindows']} | {row['BatchSize']} | {row['EquivalentDays']} | "
                f"{row['SamplingTime']} | {row['SecondsPerWindow']} | {row['WindowsPerSecond']} | "
                f"{row['SecondsPerEquivalentDay']} |\n"
            )

    print(f"Benchmark CSV saved to: {csv_path}")
    print(f"Benchmark Markdown saved to: {md_path}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    seed_everything(args.seed)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use --gpu -1 for CPU benchmarking.")
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    rows = []
    for appliance in args.appliances:
        print(f"\nBenchmarking {appliance}...")
        rows.extend(benchmark_appliance(args, appliance, device))

    write_outputs(rows, args.output)


if __name__ == "__main__":
    main()
