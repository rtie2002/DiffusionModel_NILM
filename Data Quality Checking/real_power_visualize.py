"""
Function to detect data normalization type from CSV or NPY files.

This function analyzes the data range and distribution to determine if it's:
- Z-score normalized (mean ~0, std ~1)
- MinMax normalized [0, 1]
- Scaled to [-1, 1]
- Raw/unnormalized data

Usage:
    # Interactive prompt
    python detect_normalization.py
    
    # Command line argument
    python detect_normalization.py --path path/to/file.npy
    python detect_normalization.py -p path/to/file.csv
    
    # As a module
    from detect_normalization import detect_normalization
    result = detect_normalization('file.npy')
"""

import numpy as np
import pandas as pd
import os
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

# Normalization parameters from ukdale_processing.py
AGG_MEAN = 522
AGG_STD = 814

APPLIANCE_PARAMS = {
    'kettle': {'mean': 700, 'std': 1000, 'max_power': 3998},
    'microwave': {'mean': 500, 'std': 800, 'max_power': 2000},  # Clipped in algorithm1_v2.py
    'fridge': {'mean': 200, 'std': 400, 'max_power': 350},  # Clipped in algorithm1_v2.py
    'dishwasher': {'mean': 700, 'std': 1000, 'max_power': 3964},
    'washingmachine': {'mean': 400, 'std': 700, 'max_power': 3999}  # Original from paper
}

def detect_data_type(data):
    """
    Detect if data is MinMax [0,1], Z-score, or raw power
    
    Returns: 'minmax_0_1', 'zscore', or 'raw'
    """
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    # Check for MinMax [0, 1]
    if 0.0 <= min_val < 0.05 and 0.8 < max_val <= 1.05:
        return 'minmax_0_1'
    
    # Check for Z-score (mean ~0, reasonable std)
    elif -2 < mean_val < 2 and 0.2 < std_val < 3.0 and min_val < 0:
        return 'zscore'
    
    # Raw power (positive, large values)
    elif min_val >= 0 and max_val > 100:
        return 'raw'
    
    # Default to Z-score if uncertain
    return 'zscore'

def denormalize_zscore_to_watts(zscore_data, mean, std):
    """Convert Z-score normalized data back to watts"""
    return zscore_data * std + mean

def denormalize_minmax_to_watts(minmax_data, max_power):
    """Convert MinMax [0,1] normalized data back to watts"""
    return minmax_data * max_power

def detect_appliance_from_path(file_path):
    """Detect appliance name from file path"""
    file_lower = file_path.lower()
    for appliance in APPLIANCE_PARAMS.keys():
        if appliance in file_lower:
            return appliance
    return None


def detect_normalization(file_path, verbose=True):
    """
    Detect the normalization type of data in a CSV or NPY file.
    Supports multi-column CSVs.
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data based on file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    results = {}
    
    if file_ext == '.npy':
        # Load NPY file
        data = np.load(file_path)
        original_shape = data.shape
        data_flat = data.flatten()
        columns = ['Data'] # NPY treated as single stream
        
        # Analyze the single/flattened stream
        results['Data'] = analyze_column(data_flat)
        results['Data']['shape'] = original_shape
        
    elif file_ext == '.csv':
        # Load CSV file
        df = pd.read_csv(file_path, header=0) # Assume header exists
        
        # Heuristic: if columns look like default integers '0', '1' but parsed as ints, keeps them.
        # If no header really existed, pandas might have consumed the first row. 
        # But for strictly numeric logs, header=None is better.
        # Let's check dtypes.
        
        # Analyze each numeric column
        for col in df.columns:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                data_col = df[col].values
                results[col] = analyze_column(data_col)
                results[col]['shape'] = data_col.shape
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Only .npy and .csv are supported.")
    
    # Print results if verbose
    if verbose:
        print("=" * 60)
        print("NORMALIZATION DETECTION RESULTS")
        print("=" * 60)
        print(f"File: {file_path}")
        print(f"File type: {file_ext.upper()}")
        
        for col_name, res in results.items():
            print(f"\n--- Column: {col_name} ---")
            print(f"  Shape: {res['shape']}")
            print(f"  Mean: {res['mean']:.6f}")
            print(f"  Std:  {res['std']:.6f}")
            print(f"  Min:  {res['min']:.6f}")
            print(f"  Max:  {res['max']:.6f}")
            print(f"  Range: [{res['min']:.6f}, {res['max']:.6f}]")
            print(f"  Detected Type: {res['type']} (Confidence: {res['confidence']})")
        print("=" * 60)
    
    return results

def analyze_column(data_flat):
    """Helper to analyze a single flat array of data"""
    mean_val = float(np.mean(data_flat))
    std_val = float(np.std(data_flat))
    min_val = float(np.min(data_flat))
    max_val = float(np.max(data_flat))
    data_range = [min_val, max_val]
    
    norm_type = 'unknown'
    confidence = 'low'
    
    # Check for Z-score normalization (mean ~0, std ~1)
    if abs(mean_val) < 0.1 and 0.9 < std_val < 1.1:
        norm_type = 'z-score'
        confidence = 'high'
    elif abs(mean_val) < 0.5 and 0.5 < std_val < 2.0:
        norm_type = 'z-score'
        confidence = 'medium'
    
    # Check for MinMax [0, 1] normalization
    elif 0.0 <= min_val < 0.01 and 0.99 < max_val <= 1.01:
        norm_type = 'minmax_0_1'
        confidence = 'high'
    elif 0.0 <= min_val < 0.05 and 0.95 < max_val <= 1.05:
        norm_type = 'minmax_0_1'
        confidence = 'medium'
    
    # Check for MinMax [-1, 1] normalization
    elif -1.01 <= min_val < -0.99 and 0.99 < max_val <= 1.01:
        norm_type = 'minmax_-1_1'
        confidence = 'high'
    elif -1.05 <= min_val < -0.95 and 0.95 < max_val <= 1.05:
        norm_type = 'minmax_-1_1'
        confidence = 'medium'
    
    # Check if data looks raw (not normalized)
    elif min_val >= -10 and max_val > 100 and mean_val > 10:
        norm_type = 'raw'
        confidence = 'medium'
    
    # If still unknown, provide best guess
    if norm_type == 'unknown':
        if min_val >= 0 and max_val <= 1:
            norm_type = 'likely_minmax_0_1'
            confidence = 'low'
        elif min_val >= -1 and max_val <= 1:
            norm_type = 'likely_minmax_-1_1'
            confidence = 'low'
        elif abs(mean_val) < 1 and std_val < 2:
            norm_type = 'likely_z-score'
            confidence = 'low'
            
    return {
        'type': norm_type,
        'confidence': confidence,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'range': data_range
    }

def interactive_viewer(file_path, max_windows=100, denormalize=True):
    """
    Interactive MATLAB-like data viewer with scrolling, zoom, and multi-column support.
    Can denormalize Z-score data back to watts.
    """
    print(f"\nLoading data for interactive viewer: {file_path}")
    
    # Auto-detect appliance from path
    detected_appliance = detect_appliance_from_path(file_path)
    appliance_name = None
    
    if denormalize:
        if detected_appliance:
            print(f"\n✓ Detected appliance: {detected_appliance}")
            confirm = input(f"Use '{detected_appliance}' for denormalization? (y/n, default=y): ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                appliance_name = detected_appliance
        
        if not appliance_name:
            print(f"\nAvailable appliances: {', '.join(APPLIANCE_PARAMS.keys())}")
            user_input = input("Enter appliance name (or 'skip' to show Z-score): ").strip().lower()
            if user_input in APPLIANCE_PARAMS:
                appliance_name = user_input
            elif user_input != 'skip':
                print(f"Unknown appliance '{user_input}', showing Z-score data")
                denormalize = False
    
    # Load data (Detect normalization internally calls analyze_column but returns dict)
    # We mainly need the data dictionary logic here.
    
    file_ext = os.path.splitext(file_path)[1].lower()
    data_dict = {} # {col_name: windows_array}
    
    window_size = 1024
    num_windows = 0
    window_length = 0
    
    if file_ext == '.npy':
        data = np.load(file_path)
        # Handle NPY usually as single stream
        if len(data.shape) == 3:
            windows = data
        elif len(data.shape) == 2:
             num_windows = min(data.shape[0] // window_size, max_windows)
             windows = data[:num_windows * window_size].reshape(num_windows, window_size, -1)
        else:
             data_flat = data.flatten()
             num_windows = min(len(data_flat) // window_size, max_windows)
             windows = data_flat[:num_windows * window_size].reshape(num_windows, window_size, 1)
        
        data_dict['Data'] = windows
        num_windows = windows.shape[0]
        window_length = windows.shape[1]
        
    else:  # CSV
        # Heuristic to handle headerless files gracefully
        df = pd.read_csv(file_path, header=0)
        try:
             # Check if current columns are actually data (float-convertible)
             # If so, assume no header
             [float(c) for c in df.columns]
             df = pd.read_csv(file_path, header=None)
             print("Detected headerless CSV.")
        except ValueError:
             # Columns are strings -> likely real header
             pass
        
        # Process each column
        first_valid = True
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                data_flat = df[col].values
                n_wins = min(len(data_flat) // window_size, max_windows)
                
                if n_wins == 0:
                    continue
                    
                if first_valid:
                    num_windows = n_wins
                    window_length = window_size
                    first_valid = False
                else:
                    n_wins = min(n_wins, num_windows) # Sync lengths
                
                # Reshape
                wins = data_flat[:n_wins * window_size].reshape(n_wins, window_size, 1)
                data_dict[str(col)] = wins

    if not data_dict:
        print("No numeric data found to plot.")
        return

    # Denormalize data if requested
    if denormalize and appliance_name:
        print(f"\n{'='*60}")
        print(f"DENORMALIZING TO WATTS")
        print(f"{'='*60}")
        print(f"Appliance: {appliance_name}")
        
        # Denormalize each column
        for col_name in data_dict.keys():
            col_str = str(col_name)
            
            # Determine column type and parameters
            if col_str == '0':
                # Column 0 = aggregate
                mean, std = AGG_MEAN, AGG_STD
                max_power = None  # Aggregate doesn't use max_power
                label = "Aggregate"
            elif col_str == '1':
                # Column 1 = appliance
                mean = APPLIANCE_PARAMS[appliance_name]['mean']
                std = APPLIANCE_PARAMS[appliance_name]['std']
                max_power = APPLIANCE_PARAMS[appliance_name]['max_power']
                label = appliance_name.capitalize()
            elif col_str.lower() in ['data', 'power']:
                # NPY or single-column CSV
                mean = APPLIANCE_PARAMS[appliance_name]['mean']
                std = APPLIANCE_PARAMS[appliance_name]['std']
                max_power = APPLIANCE_PARAMS[appliance_name]['max_power']
                label = appliance_name.capitalize()
            else:
                continue
            
            # Get data
            original_shape = data_dict[col_name].shape
            data_flat = data_dict[col_name].reshape(-1)
            
            # Detect data type
            data_type = detect_data_type(data_flat)
            print(f"\n{label}:")
            print(f"  Detected type: {data_type}")
            print(f"  Original range: [{data_flat.min():.4f}, {data_flat.max():.4f}]")
            
            # Apply appropriate denormalization
            if data_type == 'minmax_0_1':
                # MinMax [0,1] → Watts
                data_watts = denormalize_minmax_to_watts(data_flat, max_power)
                print(f"  Denormalization: MinMax × {max_power}W")
            elif data_type == 'zscore':
                # Z-score → Watts
                data_watts = denormalize_zscore_to_watts(data_flat, mean, std)
                print(f"  Denormalization: Z-score (mean={mean}W, std={std}W)")
            else:  # raw
                # Already in watts
                data_watts = data_flat
                print(f"  No conversion needed (already in watts)")
            
            print(f"  Final range: [{data_watts.min():.1f}W, {data_watts.max():.1f}W]")
            
            # Update data_dict
            data_dict[col_name] = data_watts.reshape(original_shape)
        
        print(f"{'='*60}\n")

    print(f"Loaded {num_windows} windows, each with {window_length} time steps")
    print(f"Columns found: {list(data_dict.keys())}")
    
    # Create figure with adjusted margins for cleaner layout
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.15, left=0.12, right=0.95, top=0.95)
    
    # Initial setup
    current_window = 0
    lines = {}
    
    # Color cycle
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot all columns with global time steps
    start_step = current_window * window_length
    x_data = np.arange(start_step, start_step + window_length)
    
    for i, (col_name, windows) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        line, = ax.plot(x_data, windows[current_window, :, 0], linewidth=1.5, label=col_name, color=color, alpha=0.8)
        lines[col_name] = line
    
    ax.legend(loc='upper right')
    ax.set_xlim(start_step, start_step + window_length)
    ax.set_xlabel('Global Time Step', fontsize=12)
    ylabel = 'Power (Watts)' if (denormalize and appliance_name) else 'Z-Score Value'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Controls - sliders at bottom
    ax_slider = plt.axes([0.12, 0.08, 0.55, 0.03])
    window_slider = Slider(ax_slider, 'Window', 0, num_windows - 1, valinit=current_window, valstep=1, valfmt='%d')
    
    ax_scale = plt.axes([0.12, 0.03, 0.55, 0.03])
    scale_slider = Slider(ax_scale, 'Y-Scale', 0.1, 5.0, valinit=1.0, valfmt='%.2f')
    
    # Checkboxes for Column Selection - smaller and compact
    ax_check = plt.axes([0.01, 0.30, 0.09, 0.5])  # Narrower panel
    check = CheckButtons(ax_check, list(data_dict.keys()), [True]*len(data_dict))
    
    # Buttons - positioned at bottom right
    ax_prev = plt.axes([0.70, 0.08, 0.08, 0.04])
    ax_next = plt.axes([0.79, 0.08, 0.08, 0.04])
    ax_reset = plt.axes([0.88, 0.08, 0.10, 0.04])
    
    btn_prev = Button(ax_prev, '◀ Prev')
    btn_next = Button(ax_next, 'Next ▶')
    btn_reset = Button(ax_reset, 'Auto-Fit Y')
    
    # State tracking
    def update_view_range():
        """Auto-scale Y-axis based on visible lines"""
        y_min, y_max = float('inf'), float('-inf')
        any_visible = False
        
        for col_name, line in lines.items():
            if line.get_visible():
                # Get data for CURRENT window only
                y_data = line.get_ydata()
                y_min = min(y_min, y_data.min())
                y_max = max(y_max, y_data.max())
                any_visible = True
        
        if not any_visible:
            return 
            
        y_range = y_max - y_min
        if y_range == 0: y_range = 1
        pad = y_range * 0.1
        
        center = (y_max + y_min) / 2
        rng = (y_max - y_min) + 2*pad
        
        # Apply scale slider
        scale = scale_slider.val
        rng = rng / scale
        
        ax.set_ylim(center - rng/2, center + rng/2)
        
        # Update title based on visible lines stats?
        # Just update window number
        ax.set_title(f'Window {current_window + 1}/{num_windows}', fontsize=14, fontweight='bold')
    
    def update_window(val):
        nonlocal current_window
        current_window = int(window_slider.val)
        
        # Calculate global time step range for this window
        start_step = current_window * window_length
        x_data = np.arange(start_step, start_step + window_length)
        
        for col_name, windows in data_dict.items():
            lines[col_name].set_xdata(x_data)
            lines[col_name].set_ydata(windows[current_window, :, 0])
        
        # Update X-axis limits to match new window
        ax.set_xlim(start_step, start_step + window_length)
        
        # Update title but DON'T auto-scale Y-axis
        ax.set_title(f'Window {current_window + 1}/{num_windows} (Steps {start_step:,}-{start_step+window_length:,})', fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()
        
    def update_scale(val):
        # Only update when scale slider changes, not when window changes
        update_view_range()
        fig.canvas.draw_idle()
        
    def toggle_visibility(label):
        line = lines[label]
        line.set_visible(not line.get_visible())
        line.figure.canvas.draw_idle()
        # Don't auto-scale when toggling visibility
        
    def prev_window(event):
        if current_window > 0:
            window_slider.set_val(current_window - 1)
            
    def next_window(event):
        if current_window < num_windows - 1:
            window_slider.set_val(current_window + 1)
            
    def auto_fit_y(event):
        """Manually trigger Y-axis auto-fit to current window"""
        scale_slider.set_val(1.0)  # Reset scale first
        update_view_range()
        fig.canvas.draw_idle()
    
    # Connect
    window_slider.on_changed(update_window)
    scale_slider.on_changed(update_scale)
    check.on_clicked(toggle_visibility)
    btn_prev.on_clicked(prev_window)
    btn_next.on_clicked(next_window)
    btn_reset.on_clicked(auto_fit_y)
    
    # Initial View Update
    update_view_range()
    
    print("\n" + "=" * 60)
    print("INTERACTIVE VIEWER OPENED")
    print("=" * 60)
    print("• Use Checkboxes on left to Show/Hide columns")
    print("• Use Slider to scroll through windows")
    print("• Use Y-Scale to zoom vertically")
    print("=" * 60 + "\n")
    
    plt.show()


# Main function for command-line usage
def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Detect normalization type of data in CSV or NPY files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--path', '-p', type=str, default=None, help='Path to the file to analyze')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    parser.add_argument('--view', '-v', action='store_true', help='Open viewer')
    parser.add_argument('--no-view', action='store_true', help='Skip viewer')
    parser.add_argument('--max-windows', type=int, default=100000, help='Max windows to display (default: 100000, effectively all)')
    parser.add_argument('--denormalize', action='store_true', default=True, help='Denormalize Z-score to watts (default: True)')
    parser.add_argument('--no-denormalize', dest='denormalize', action='store_false', help='Keep Z-score values (do not convert to watts)')
    
    args = parser.parse_args()
    
    # Interactive prompt
    interactive_mode = False
    if args.path:
        file_path = args.path
    else:
        interactive_mode = True
        print("=" * 60)
        print("NORMALIZATION DETECTION")
        print("=" * 60)
        print("Enter the path to your CSV or NPY file:")
        print("(You can also use: python detect_normalization.py --path <file_path>)")
        print("-" * 60)
        file_path = input("File path: ").strip()
        
        # Remove quotes
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if file_path.startswith("'") and file_path.endswith("'"):
            file_path = file_path[1:-1]
    
    if not file_path or not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Run
    try:
        verbose = not args.quiet
        result = detect_normalization(file_path, verbose=verbose)
        
        should_view = args.view or (interactive_mode and not args.no_view)
        
        if should_view:
            try:
                print("\nOpening interactive viewer...")
                interactive_viewer(file_path, max_windows=args.max_windows, denormalize=args.denormalize)
            except Exception as e:
                print(f"Error opening viewer: {e}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
