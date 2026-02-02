"""
Run TS2Vec Evaluation for ALL Appliances

This script sequentially runs evaluate_ts2vec.py for all 5 appliances:
- dishwasher
- fridge
- kettle
- microwave
- washingmachine

It prints the Discriminative Score for each and confirms where the plot is saved.
"""

import os
import subprocess
import sys

# Configurations
APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
SEQ_LEN = 60  # Length of window to evaluate (can be increased to 120 or more)
PYTHON_EXECUTABLE = sys.executable  # Use the current python interpreter

def run_evaluation():
    print("="*60)
    print("   Starting TS2Vec Evaluation for ALL Appliances")
    print("="*60)
    
    results = {}
    
    for app in APPLIANCES:
        print(f"\n\n>>> Processing Appliance: {app.upper()}")
        print("-" * 40)
        
        # Construct command
        cmd = [
            PYTHON_EXECUTABLE,
            "-u", # Unbuffered output
            os.path.join("Data Quality Checking", "evaluate_ts2vec.py"),
            app,
            "--seq_len", str(SEQ_LEN)
        ]
        
        try:
            # Run the command and stream output to console
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output line by line to capture score
            discriminative_score = "N/A"
            for line in process.stdout:
                print(line, end='') # Print to console
                
                # Capture the score from the output line
                if "Discriminative Score" in line:
                    parts = line.strip().split(':')
                    if len(parts) > 1:
                        discriminative_score = parts[1].strip()
            
            process.wait()
            
            if process.returncode == 0:
                results[app] = discriminative_score
                print(f"✓ {app} completed. Score: {discriminative_score}")
            else:
                results[app] = "FAILED"
                print(f"✗ {app} failed with exit code {process.returncode}")
                
        except Exception as e:
            print(f"Error running {app}: {e}")
            results[app] = "ERROR"

    # Final Summary
    print("\n\n" + "="*60)
    print("   FINAL RESULTS SUMMARY (Discriminative Score)")
    print("   Lower is better (closer to 0.5 = indistinguishable)")
    print("="*60)
    for app, score in results.items():
        print(f"{app:<15}: {score}")
    print("="*60)

if __name__ == "__main__":
    # Ensure run from project root
    if not os.path.exists("Data Quality Checking"):
        print("Error: Please run this script from the project root directory.")
        print("Example: python \"Data Quality Checking/run_ts2vec_all.py\"")
    else:
        run_evaluation()
