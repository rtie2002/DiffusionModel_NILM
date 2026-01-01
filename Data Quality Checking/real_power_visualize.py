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
from matplotlib.widgets import Slider, Button

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
    
    # Check for MinMax [0, 1] - MUST check before Z-score!
    # Lowered max threshold from 0.8 to 0.1 to catch low-power appliances like fridge (max~0.38)
    # Key indicators: min near 0, max < 1, no negative values
    if 0.0 <= min_val < 0.05 and 0.1 < max_val <= 1.05 and min_val >= 0:
        return 'minmax_0_1'
    
    # Check for Z-score (mean ~0, reasonable std, HAS negative values)
    # MUST have negative values to be Z-score, otherwise it's likely MinMax
    elif -2 < mean_val < 2 and 0.2 < std_val < 3.0 and min_val < 0:
        return 'zscore'
    
    # Raw power (positive, large values)
    elif min_val >= 0 and max_val > 100:
        return 'raw'
    
    # Default: if min >= 0 and max <= 1, assume MinMax, otherwise Z-score
    if min_val >= 0 and max_val <= 1:
        return 'minmax_0_1'
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
            col_str = str(col_name).lower()
            
            # Determine column type and parameters
            # Support multiple formats:
            # - 2-column: (aggregate, appliance)
            # - 10-column: (aggregate, appliance, 8 time features)
            # - Numeric indices: '0', '1', etc.
            
            # Check if column is appliance power
            if col_str == appliance_name or col_str == 'power' or col_str == 'data':
                # Direct appliance column
                mean = APPLIANCE_PARAMS[appliance_name]['mean']
                std = APPLIANCE_PARAMS[appliance_name]['std']
                max_power = APPLIANCE_PARAMS[appliance_name]['max_power']
                label = appliance_name.capitalize()
            elif col_str == 'aggregate':
                # Aggregate column
                mean, std = AGG_MEAN, AGG_STD
                max_power = None
                label = "Aggregate"
            elif col_str in ['minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 
                            'dow_sin', 'dow_cos', 'month_sin', 'month_cos']:
                # Time features - skip
                continue
            elif col_str == '0':
                # Check if this is aggregate (2-column format) or appliance (10-column format)
                col_data = data_dict[col_name].reshape(-1)
                if col_data.min() < -0.1:
                    # Column 0 = aggregate (2-column format)
                    mean, std = AGG_MEAN, AGG_STD
                    max_power = None
                    label = "Aggregate"
                else:
                    # Column 0 = appliance (10-column format without headers)
                    mean = APPLIANCE_PARAMS[appliance_name]['mean']
                    std = APPLIANCE_PARAMS[appliance_name]['std']
                    max_power = APPLIANCE_PARAMS[appliance_name]['max_power']
                    label = appliance_name.capitalize()
            elif col_str == '1':
                # Column 1 = appliance (2-column format) or time feature (10-column format)
                col_data = data_dict[col_name].reshape(-1)
                if col_data.max() <= 1.1 and col_data.min() >= -1.1:
                    # Likely sin/cos time feature, skip
                    continue
                else:
                    # Column 1 = appliance power
                    mean = APPLIANCE_PARAMS[appliance_name]['mean']
                    std = APPLIANCE_PARAMS[appliance_name]['std']
                    max_power = APPLIANCE_PARAMS[appliance_name]['max_power']
                    label = appliance_name.capitalize()
            elif col_str in ['2', '3', '4', '5', '6', '7', '8', '9']:
                # Columns 2-9 in 10-column format are time features
                # Skip denormalization for temporal features
                continue
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
    plt.subplots_adjust(bottom=0.20, left=0.08, right=0.95, top=0.95)
    
    # Initial setup
    current_window = 0
    lines = {}
    
    # Selection state
    selection_state = {
        'active': False,
        'start_x': None,
        'start_y': None,
        'rect': None,
        'x_min': None,
        'x_max': None,
        'y_min': None,
        'y_max': None,
        'saved_xlim': None,
        'saved_ylim': None
    }
    
    # Color cycle
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot all columns with global time steps
    start_step = current_window * window_length
    x_data = np.arange(start_step, start_step + window_length)
    
    for i, (col_name, windows) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        line, = ax.plot(x_data, windows[current_window, :, 0], linewidth=1.5, label=col_name, color=color, alpha=0.8)
        lines[col_name] = line
    
    # Create interactive legend (clickable to toggle visibility)
    legend = ax.legend(loc='upper right')
    # Make each legend line and text individually pickable for easier clicking
    for legend_line in legend.get_lines():
        legend_line.set_picker(True)
        legend_line.set_pickradius(5)  # 5 pixel radius for easier clicking
    for legend_text in legend.get_texts():
        legend_text.set_picker(True)  # Make text clickable too
    
    ax.set_xlim(start_step, start_step + window_length)
    ax.set_xlabel('Global Time Step', fontsize=12)
    ylabel = 'Power (Watts)' if (denormalize and appliance_name) else 'Z-Score Value'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Controls - sliders at bottom
    ax_slider = plt.axes([0.08, 0.13, 0.59, 0.03])
    window_slider = Slider(ax_slider, 'Window', 0, num_windows - 1, valinit=current_window, valstep=1, valfmt='%d')
    
    ax_scale = plt.axes([0.08, 0.08, 0.59, 0.03])
    scale_slider = Slider(ax_scale, 'Y-Scale', 0.1, 5.0, valinit=1.0, valfmt='%.2f')
    
    # Buttons - positioned at bottom right (two rows)
    ax_prev = plt.axes([0.70, 0.13, 0.08, 0.04])
    ax_next = plt.axes([0.79, 0.13, 0.08, 0.04])
    ax_reset = plt.axes([0.88, 0.13, 0.10, 0.04])
    
    ax_zoom_sel = plt.axes([0.70, 0.08, 0.13, 0.04])
    ax_clear_sel = plt.axes([0.84, 0.08, 0.14, 0.04])
    
    btn_prev = Button(ax_prev, '◀ Prev')
    btn_next = Button(ax_next, 'Next ▶')
    btn_reset = Button(ax_reset, 'Auto-Fit Y')
    btn_zoom_sel = Button(ax_zoom_sel, 'Zoom to Sel')
    btn_clear_sel = Button(ax_clear_sel, 'Clear Selection')
    
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
        
    def on_pick(event):
        """Handle legend pick events to toggle line visibility"""
        legend_lines = legend.get_lines()
        legend_texts = legend.get_texts()
        
        # Check if a legend line was clicked
        for i, (legend_line, orig_line) in enumerate(zip(legend_lines, lines.values())):
            if event.artist == legend_line:
                # Toggle visibility
                visible = not orig_line.get_visible()
                orig_line.set_visible(visible)
                
                # Update legend line and text alpha to show state
                legend_line.set_alpha(1.0 if visible else 0.2)
                legend_texts[i].set_alpha(1.0 if visible else 0.3)
                
                fig.canvas.draw_idle()
                return
        
        # Check if a legend text was clicked
        for i, (legend_text, orig_line) in enumerate(zip(legend_texts, lines.values())):
            if event.artist == legend_text:
                # Toggle visibility
                visible = not orig_line.get_visible()
                orig_line.set_visible(visible)
                
                # Update legend line and text alpha to show state
                legend_lines[i].set_alpha(1.0 if visible else 0.2)
                legend_text.set_alpha(1.0 if visible else 0.3)
                
                fig.canvas.draw_idle()
                return
        
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
    
    def on_mouse_press(event):
        """Handle mouse button press for selection"""
        if event.inaxes != ax or event.button != 1:  # Left click only
            return
        
        # Start selection
        selection_state['active'] = True
        selection_state['start_x'] = event.xdata
        selection_state['start_y'] = event.ydata
        
        # Remove old rectangle if exists
        if selection_state['rect'] is not None:
            selection_state['rect'].remove()
            selection_state['rect'] = None
    
    def on_mouse_move(event):
        """Handle mouse move for selection rectangle"""
        if not selection_state['active'] or event.inaxes != ax:
            return
        
        # Update rectangle
        if selection_state['rect'] is not None:
            selection_state['rect'].remove()
        
        x0, y0 = selection_state['start_x'], selection_state['start_y']
        x1, y1 = event.xdata, event.ydata
        
        width = x1 - x0
        height = y1 - y0
        
        # Draw selection rectangle
        selection_state['rect'] = plt.Rectangle(
            (x0, y0), width, height,
            fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7
        )
        ax.add_patch(selection_state['rect'])
        fig.canvas.draw_idle()
    
    def on_mouse_release(event):
        """Handle mouse button release to finalize selection"""
        if not selection_state['active'] or event.button != 1:
            return
        
        selection_state['active'] = False
        
        if event.inaxes != ax or selection_state['start_x'] is None:
            return
        
        # Calculate selection bounds
        x0, y0 = selection_state['start_x'], selection_state['start_y']
        x1, y1 = event.xdata, event.ydata
        
        # Ensure x_min < x_max and y_min < y_max
        selection_state['x_min'] = min(x0, x1)
        selection_state['x_max'] = max(x0, x1)
        selection_state['y_min'] = min(y0, y1)
        selection_state['y_max'] = max(y0, y1)
        
        # Display selection statistics
        print_selection_stats()
    
    def print_selection_stats():
        """Print statistics for the selected region"""
        if selection_state['x_min'] is None:
            return
        
        x_min, x_max = selection_state['x_min'], selection_state['x_max']
        y_min, y_max = selection_state['y_min'], selection_state['y_max']
        
        print("\n" + "="*60)
        print("SELECTION STATISTICS")
        print("="*60)
        print(f"X Range: [{x_min:.1f}, {x_max:.1f}] (Width: {x_max-x_min:.1f} steps)")
        print(f"Y Range: [{y_min:.2f}, {y_max:.2f}]")
        
        # Calculate statistics for each visible line in selection
        for col_name, line in lines.items():
            if not line.get_visible():
                continue
            
            # Get data within selection
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            
            # Find indices within x range
            mask = (x_data >= x_min) & (x_data <= x_max)
            selected_y = y_data[mask]
            
            if len(selected_y) == 0:
                continue
            
            # Further filter by y range
            y_mask = (selected_y >= y_min) & (selected_y <= y_max)
            selected_y_filtered = selected_y[y_mask]
            
            print(f"\n{col_name}:")
            print(f"  Points in selection: {len(selected_y)}")
            print(f"  Points in Y range: {len(selected_y_filtered)}")
            print(f"  Mean: {selected_y.mean():.2f}")
            print(f"  Std: {selected_y.std():.2f}")
            print(f"  Min: {selected_y.min():.2f}")
            print(f"  Max: {selected_y.max():.2f}")
        
        print("="*60)
        print("Tip: Click 'Zoom to Sel' to zoom into this region")
        print("="*60 + "\n")
    
    def zoom_to_selection(event):
        """Zoom to the selected region"""
        if selection_state['x_min'] is None:
            print("No selection made. Click and drag on the plot to select a region.")
            return
        
        # Save current view
        selection_state['saved_xlim'] = ax.get_xlim()
        selection_state['saved_ylim'] = ax.get_ylim()
        
        # Zoom to selection
        x_min, x_max = selection_state['x_min'], selection_state['x_max']
        y_min, y_max = selection_state['y_min'], selection_state['y_max']
        
        # Add small padding
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        print(f"\nZoomed to selection: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.2f}, {y_max:.2f}]")
        fig.canvas.draw_idle()
    
    def clear_selection(event):
        """Clear the current selection and restore view"""
        # Remove rectangle
        if selection_state['rect'] is not None:
            selection_state['rect'].remove()
            selection_state['rect'] = None
        
        # Restore saved view if exists
        if selection_state['saved_xlim'] is not None:
            ax.set_xlim(selection_state['saved_xlim'])
            ax.set_ylim(selection_state['saved_ylim'])
        
        # Reset selection state
        selection_state['x_min'] = None
        selection_state['x_max'] = None
        selection_state['y_min'] = None
        selection_state['y_max'] = None
        selection_state['saved_xlim'] = None
        selection_state['saved_ylim'] = None
        
        print("\nSelection cleared and view restored.")
        fig.canvas.draw_idle()
    
    # Connect events
    window_slider.on_changed(update_window)
    scale_slider.on_changed(update_scale)
    fig.canvas.mpl_connect('pick_event', on_pick)  # Connect legend click handler
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    btn_prev.on_clicked(prev_window)
    btn_next.on_clicked(next_window)
    btn_reset.on_clicked(auto_fit_y)
    btn_zoom_sel.on_clicked(zoom_to_selection)
    btn_clear_sel.on_clicked(clear_selection)
    
    # Initial View Update
    update_view_range()
    
    print("\n" + "=" * 60)
    print("INTERACTIVE VIEWER OPENED")
    print("==" * 60)
    print("• Click and Drag to select a region (red rectangle)")
    print("• Click 'Zoom to Sel' to zoom into selected region")
    print("• Click 'Clear Selection' to reset view")
    print("• Click on Legend items to Show/Hide lines")
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
