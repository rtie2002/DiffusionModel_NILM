"""
generate_quarter_configs.py
===========================
Given a base appliance YAML config (e.g. Config/washingmachine.yaml) and a
list of available quarters, generate one YAML per quarter:
    Config/washingmachine_Q1.yaml
    Config/washingmachine_Q2.yaml
    ...

Changes made per quarter config:
  - dataloader.train_dataset.params.data_root  -> <appliance>_multivariate_<Q>.csv
  - solver.results_folder                      -> <original_folder>_<Q>

Usage (called internally by run_quarterly_diffusion.sh):
    python generate_quarter_configs.py \\
        --base_config Config/washingmachine.yaml \\
        --appliance washingmachine \\
        --quarters Q1 Q2 Q3 \\
        --csv_dir .
"""

import argparse
import copy
import os
import yaml


def generate_configs(base_config_path: str,
                     appliance: str,
                     quarters: list,
                     csv_dir: str = '.',
                     source_csv: str = None) -> dict:
    """
    Generate one YAML config per quarter.
    Returns {quarter_label: output_config_path}.
    """
    with open(base_config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # Detect suffix from source_csv if provided (e.g. "washingmachine_training_")
    if source_csv:
        base_name = os.path.splitext(os.path.basename(source_csv))[0]
    else:
        base_name = f"{appliance}_multivariate"

    config_dir = os.path.dirname(os.path.abspath(base_config_path))
    created = {}

    for q in quarters:
        cfg = copy.deepcopy(base_cfg)

        # ── data_root ──────────────────────────────────────────────────────────
        csv_name = f"{base_name}_{q}.csv"
        csv_path = os.path.join(csv_dir, csv_name).replace('\\', '/')
        cfg['dataloader']['train_dataset']['params']['data_root'] = csv_path

        # ── results_folder ─────────────────────────────────────────────────────
        old_folder = cfg['solver']['results_folder']
        # Strip any stale quarter suffix (safe to rerun)
        for other_q in ['Q1', 'Q2', 'Q3', 'Q4']:
            old_folder = old_folder.rstrip(f'_{other_q}')
        cfg['solver']['results_folder'] = f"{old_folder}_{q}"

        # ── write new config ───────────────────────────────────────────────────
        out_path = os.path.join(config_dir, f"{appliance}_{q}.yaml")
        with open(out_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)

        created[q] = out_path
        print(f"  [{q}] Config -> {out_path}  (data: {csv_path})")

    return created


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-quarter YAML configs from a base appliance config.')
    parser.add_argument('--base_config', required=True,
                        help='Path to base YAML (e.g. Config/washingmachine.yaml)')
    parser.add_argument('--appliance', required=True,
                        help='Appliance name (e.g. washingmachine)')
    parser.add_argument('--quarters', nargs='+', default=['Q1', 'Q2', 'Q3', 'Q4'],
                        help='Quarters to generate configs for (default: Q1 Q2 Q3 Q4)')
    parser.add_argument('--csv_dir', default='.',
                        help='Directory containing the quarterly CSVs (default: .)')
    parser.add_argument('--source_csv', default=None,
                        help='The original CSV file used as a template for naming.')
    args = parser.parse_args()

    print(f"[generate_quarter_configs] Appliance: {args.appliance}")
    generate_configs(args.base_config, args.appliance, args.quarters, args.csv_dir, args.source_csv)
    print("Done.")


if __name__ == '__main__':
    main()
