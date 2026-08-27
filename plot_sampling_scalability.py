import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROWS = [
    {
        "Appliance": "fridge",
        "GeneratedWindows": 1000,
        "BatchSize": 1000,
        "SamplingTimeSeconds": 1242.12,
        "SamplingTime": "20:42",
        "SecondsPerWindow": 1.2421,
    },
    {
        "Appliance": "fridge",
        "GeneratedWindows": 2000,
        "BatchSize": 1000,
        "SamplingTimeSeconds": 2480.33,
        "SamplingTime": "41:20",
        "SecondsPerWindow": 1.2402,
    },
    {
        "Appliance": "fridge",
        "GeneratedWindows": 3000,
        "BatchSize": 1000,
        "SamplingTimeSeconds": 3717.26,
        "SamplingTime": "01:01:57",
        "SecondsPerWindow": 1.2391,
    },
    {
        "Appliance": "fridge",
        "GeneratedWindows": 4000,
        "BatchSize": 1000,
        "SamplingTimeSeconds": 4961.81,
        "SamplingTime": "01:22:41",
        "SecondsPerWindow": 1.2405,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a one-column paper figure for TC-DiT sampling scalability."
    )
    parser.add_argument(
        "--input",
        default="OUTPUT/sampling_window_benchmark.csv",
        help="Optional benchmark CSV. Built-in fridge benchmark data are used if the file is missing.",
    )
    parser.add_argument("--appliance", default="fridge", help="Appliance to plot.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Sampling batch size to plot.")
    parser.add_argument("--output-dir", default="OUTPUT/figures", help="Directory for figure outputs.")
    parser.add_argument("--basename", default="Figure_sampling_scalability_final", help="Output filename prefix.")
    parser.add_argument("--window-length", type=int, default=512, help="Generated window length in time steps.")
    parser.add_argument("--resolution-min", type=float, default=1.0, help="Sampling resolution in minutes.")
    parser.add_argument("--title", default="Sampling Scalability", help="Figure title.")
    parser.add_argument("--dpi", type=int, default=2400, help="PNG export resolution.")
    return parser.parse_args()


def read_rows(input_path):
    path = Path(input_path)
    if not path.exists():
        print(f"[INFO] CSV not found: {path}. Using built-in fridge benchmark data.")
        return DEFAULT_ROWS

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def select_rows(rows, appliance, batch_size):
    selected = []
    for row in rows:
        if str(row["Appliance"]).strip().lower() != appliance.lower():
            continue
        if int(row["BatchSize"]) != batch_size:
            continue
        selected.append(
            {
                "windows": int(row["GeneratedWindows"]),
                "seconds": float(row["SamplingTimeSeconds"]),
                "seconds_per_window": float(row["SecondsPerWindow"]),
            }
        )

    selected.sort(key=lambda item: item["windows"])
    if len(selected) < 2:
        raise ValueError(
            f"Need at least two benchmark points for appliance={appliance}, batch_size={batch_size}."
        )
    return selected


def fit_line(windows, seconds):
    slope, intercept = np.polyfit(windows, seconds, 1)
    fitted = slope * windows + intercept
    ss_res = np.sum((seconds - fitted) ** 2)
    ss_tot = np.sum((seconds - np.mean(seconds)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def make_plot(points, appliance, batch_size, output_dir, basename, dpi, window_length, resolution_min, title):
    windows = np.array([p["windows"] for p in points], dtype=float)
    seconds = np.array([p["seconds"] for p in points], dtype=float)
    minutes = seconds / 60.0

    slope_seconds, intercept_seconds, r2 = fit_line(windows, seconds)
    fit_windows = np.linspace(windows.min(), windows.max(), 100)
    fit_minutes = (slope_seconds * fit_windows + intercept_seconds) / 60.0
    mean_seconds_per_window = np.mean([p["seconds_per_window"] for p in points])

    set_plot_style()
    fig, ax = plt.subplots(figsize=(3.25, 2.72))

    line_color = "#1f5d8f"
    marker_face = "#f2a23a"
    fit_color = "#6f7782"

    ax.plot(
        windows,
        minutes,
        color=line_color,
        linewidth=1.6,
        marker="o",
        markersize=4.4,
        markerfacecolor=marker_face,
        markeredgecolor=line_color,
        markeredgewidth=0.8,
        label="Measured",
    )
    ax.plot(
        fit_windows,
        fit_minutes,
        color=fit_color,
        linewidth=1.0,
        linestyle="--",
        label="Linear fit",
    )

    window_hours = window_length * resolution_min / 60.0
    annotation = (
        f"Mean cost = {mean_seconds_per_window:.2f} s/window\n"
        f"Linear fit: $R^2$={r2:.4f}"
    )
    ax.text(
        0.04,
        0.94,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#c9ced6",
            "linewidth": 0.5,
            "alpha": 0.95,
        },
    )

    for i, (x_value, y_value) in enumerate(zip(windows, minutes)):
        if i == len(windows) - 1:
            label_offset = (0, -13)
            label_va = "top"
        else:
            label_offset = (0, 7)
            label_va = "bottom"

        ax.annotate(
            f"{y_value:.1f}",
            xy=(x_value, y_value),
            xytext=label_offset,
            textcoords="offset points",
            ha="center",
            va=label_va,
            fontsize=6.3,
            color="#263238",
        )

    ax.set_xlabel("Generated windows", labelpad=2)
    ax.set_ylabel("Wall-clock sampling time (min)", labelpad=3)
    ax.set_xlim(windows.min() - 180, windows.max() + 180)
    ax.set_ylim(0, max(minutes) * 1.12)
    ax.set_xticks(windows)
    ax.set_xticklabels([f"{int(x/1000)}k" for x in windows])
    ax.set_yticks(np.arange(0, 91, 20))
    ax.set_yticks(np.arange(0, 91, 10), minor=True)
    ax.grid(axis="y", which="major", color="#d6dbe2", linewidth=0.55)
    ax.grid(axis="y", which="minor", color="#edf0f4", linewidth=0.35)
    ax.grid(axis="x", color="#f0f2f5", linewidth=0.35)

    def windows_to_mpoints(value):
        return value * window_length / 1_000_000.0

    def mpoints_to_windows(value):
        return value * 1_000_000.0 / window_length

    top_ax = ax.secondary_xaxis("top", functions=(windows_to_mpoints, mpoints_to_windows))
    top_ticks = windows_to_mpoints(windows)
    top_ax.set_xticks(top_ticks)
    top_ax.set_xticklabels([f"{tick:.2f}M" for tick in top_ticks])
    top_ax.set_xlabel("Output size (million 1-min steps)", labelpad=1)
    top_ax.tick_params(width=0.7, length=3.0)

    fig.suptitle(title, fontsize=8.1, fontweight="bold", y=0.895)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    top_ax.spines["top"].set_linewidth(0.7)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0, borderaxespad=0.2)
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.15, top=0.705)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f"{basename}_{appliance}_batch{batch_size}"

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Linear fit slope: {slope_seconds:.5f} s/window")
    print(f"Linear fit intercept: {intercept_seconds:.3f} s")
    print(f"R2: {r2:.5f}")


def main():
    args = parse_args()
    rows = read_rows(args.input)
    points = select_rows(rows, args.appliance, args.batch_size)
    make_plot(
        points=points,
        appliance=args.appliance,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        basename=args.basename,
        dpi=args.dpi,
        window_length=args.window_length,
        resolution_min=args.resolution_min,
        title=args.title,
    )


if __name__ == "__main__":
    main()
