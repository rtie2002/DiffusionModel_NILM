"""
Run TS2Vec Evaluation for ALL Appliances (CNN-SNGAN Baseline)

This script evaluates CNN-SNGAN synthetic data against real data using TS2Vec embeddings.
It mirrors the structure of run_ts2vec_all.py but uses CNN_SNGAN as the synthetic data source.
"""

import os
import subprocess
import sys
import argparse

# Configurations
APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
PYTHON_EXECUTABLE = sys.executable  

# Get paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Data directories
REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "datasets", "real_distributions")
SYNTHETIC_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "datasets", "CNN_SNGAN")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "ts2vec_results_CNN_SNGAN")

def run_evaluation():
    parser = argparse.ArgumentParser(description="Run TS2Vec Evaluation for ALL Appliances (CNN-SNGAN)")
    parser.add_argument("--mode", type=str, choices=['multivariate', 'power', 'time', 'all'], default='multivariate',
                        help="Evaluation mode: multivariate, power, time, or all (runs all three)")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length for evaluation")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    modes_to_run = ['multivariate', 'power', 'time'] if args.mode == 'all' else [args.mode]
    
    overall_results = {}

    print("="*60)
    print(f"   TS2Vec Evaluation: CNN-SNGAN vs Real Data")
    print(f"   Modes: {modes_to_run}")
    print(f"   Synthetic Data: {SYNTHETIC_DATA_DIR}")
    print("="*60)

    for mode in modes_to_run:
        print(f"\n\n{'#'*60}")
        print(f"   STARTING MODE: {mode.upper()}")
        print(f"{'#'*60}")
        
        results = {}
        
        for app in APPLIANCES:
            print(f"\n>>> Processing Appliance: {app.upper()} | Mode: {mode}")
            print("-" * 40)
            
            # Check if synthetic data file exists
            synth_file = os.path.join(SYNTHETIC_DATA_DIR, f"{app}_multivariate.csv")
            if not os.path.exists(synth_file):
                print(f"  ⚠ Synthetic data not found: {synth_file}")
                results[app] = ("N/A", "N/A", "N/A")
                continue
            
            # Construct command - we'll use a temporary modified evaluate script
            # Actually, we can use the existing evaluate_ts2vec.py with environment variables
            # or we create a wrapper
            
            # For simplicity, let's create a temporary config
            cmd = [
                PYTHON_EXECUTABLE,
                "-u", 
                os.path.join(SCRIPT_DIR, "evaluate_ts2vec.py"),
                app,
                "--seq_len", str(args.seq_len),
                "--mode", mode
            ]
            
            # We need to temporarily override the synthetic data directory
            # We'll do this by creating a modified environment
            env = os.environ.copy()
            env['SYNTHETIC_DATA_DIR_OVERRIDE'] = SYNTHETIC_DATA_DIR
            env['RESULTS_DIR_OVERRIDE'] = RESULTS_DIR
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env
                )
                
                discriminative_score = "N/A"
                fid_score = "N/A"
                swd_score = "N/A"
                
                for line in process.stdout:
                    print(line, end='') 
                    # Capture Discriminative Score
                    if "1. Discriminative Score" in line:
                        parts = line.strip().split(':')
                        if len(parts) > 1:
                            discriminative_score = parts[1].strip().split(' ')[0]
                    
                    # Capture FID Score
                    if "2. Context-FID Score" in line:
                        parts = line.strip().split(':')
                        if len(parts) > 1:
                            fid_score = parts[1].strip()
                            
                    # Capture SWD Score
                    if "3. SWD Score" in line:
                        parts = line.strip().split(':')
                        if len(parts) > 1:
                            swd_score = parts[1].strip()
                
                process.wait()
                
                if process.returncode == 0:
                    results[app] = (discriminative_score, fid_score, swd_score)
                    print(f"✓ {app} ({mode}) completed. DS: {discriminative_score}, FID: {fid_score}, SWD: {swd_score}")
                else:
                    results[app] = ("FAILED", "N/A", "N/A")
                    print(f"✗ {app} ({mode}) failed with exit code {process.returncode}")
                    
            except Exception as e:
                print(f"Error running {app} in {mode}: {e}")
                results[app] = ("ERROR", "N/A", "N/A")
        
        overall_results[mode] = results

    # Final Summary Table (Much wider now)
    print("\n\n" + "="*140)
    print("   FINAL RESULTS: CNN-SNGAN vs Real (Scores: Lower is Better)")
    print("="*140)
    
    # Create header
    header = f"{'Appliance':<15}"
    for mode in modes_to_run:
        header += f"| {mode.upper():<35}"
    print(header)
    
    sub_header = f"{'':<15}"
    for mode in modes_to_run:
        sub_header += f"| {'DS':<6} {'FID':<10} {'SWD':<15}"
    print(sub_header)
    print("-" * len(header))
    
    for app in APPLIANCES:
        row = f"{app:<15}"
        for mode in modes_to_run:
            score, fid, swd = overall_results[mode].get(app, ("N/A", "N/A", "N/A"))
            row += f"| {score:<6} {fid:<10} {swd:<15}"
        print(row)
    print("="*140)
    print("DS: Discriminative Score | FID: Context-FID | SWD: Sliced Wasserstein Distance")
    print(f"Plots saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    if not os.path.exists("Data Quality Checking"):
        print("Error: Please run this script from the project root directory.")
    else:
        run_evaluation()
