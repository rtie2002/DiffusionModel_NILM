import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit sampling time vs generated windows to assess approximate linearity."
    )
    parser.add_argument("--input", default="OUTPUT/sampling_window_benchmark.csv")
    parser.add_argument("--output-dir", default="OUTPUT")
    return parser.parse_args()


def fit_line(xs, ys):
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx if ss_xx else 0.0
    intercept = y_mean - slope * x_mean
    preds = [intercept + slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return slope, intercept, r2


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_outputs(summary_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sampling_linearity_summary.csv"
    md_path = output_dir / "sampling_linearity_summary.md"

    fieldnames = [
        "Appliance",
        "BatchSize",
        "NumPoints",
        "SlopeSecondsPerWindow",
        "InterceptSeconds",
        "R2",
        "MinWindows",
        "MaxWindows",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Sampling Linearity Summary\n\n")
        f.write("A linear model is fitted as: sampling time = intercept + slope x generated windows.\n\n")
        f.write("| Appliance | Batch size | Points | Slope (s/window) | Intercept (s) | R2 | Window range |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['Appliance']} | {row['BatchSize']} | {row['NumPoints']} | "
                f"{row['SlopeSecondsPerWindow']} | {row['InterceptSeconds']} | {row['R2']} | "
                f"{row['MinWindows']}-{row['MaxWindows']} |\n"
            )

    print(f"Linearity CSV saved to: {csv_path}")
    print(f"Linearity Markdown saved to: {md_path}")


def main():
    args = parse_args()
    rows = read_rows(args.input)
    grouped = defaultdict(list)

    for row in rows:
        appliance = row["Appliance"]
        batch_size = row.get("BatchSize", "NA")
        windows = int(row["GeneratedWindows"])
        seconds = float(row["SamplingTimeSeconds"])
        grouped[(appliance, batch_size)].append((windows, seconds))

    summary_rows = []
    for (appliance, batch_size), points in grouped.items():
        points = sorted(points)
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        slope, intercept, r2 = fit_line(xs, ys)
        summary_rows.append(
            {
                "Appliance": appliance,
                "BatchSize": batch_size,
                "NumPoints": len(points),
                "SlopeSecondsPerWindow": round(slope, 5),
                "InterceptSeconds": round(intercept, 3),
                "R2": round(r2, 5),
                "MinWindows": min(xs),
                "MaxWindows": max(xs),
            }
        )

    if not summary_rows:
        raise RuntimeError("Not enough benchmark points to fit a linear model.")

    write_outputs(summary_rows, args.output_dir)


if __name__ == "__main__":
    main()
