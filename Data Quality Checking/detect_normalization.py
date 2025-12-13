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


def detect_normalization(file_path, verbose=True):
    """
    Detect the normalization type of data in a CSV or NPY file.
    
    Args:
        file_path (str): Path to the CSV or NPY file
        verbose (bool): If True, print detailed information
    
    Returns:
        dict: Dictionary containing:
            - 'type': Normalization type ('z-score', 'minmax_0_1', 'minmax_-1_1', 'raw', 'unknown')
            - 'mean': Mean value of the data
            - 'std': Standard deviation of the data
            - 'min': Minimum value
            - 'max': Maximum value
            - 'range': Data range [min, max]
            - 'shape': Shape of the data array
    
    Examples:
        >>> result = detect_normalization('data.npy')
        >>> result = detect_normalization('data.csv')
        >>> result = detect_normalization('data.csv', verbose=False)
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data based on file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.npy':
        # Load NPY file
        data = np.load(file_path)
        original_shape = data.shape
        # Flatten to 1D for analysis (handles any shape)
        data_flat = data.flatten()
        
    elif file_ext == '.csv':
        # Load CSV file
        df = pd.read_csv(file_path, header=0)
        # If 'power' column exists, use it; otherwise use first column
        if 'power' in df.columns:
            data_flat = df['power'].values
        else:
            # Use first numeric column
            data_flat = df.iloc[:, 0].values
        original_shape = data_flat.shape
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Only .npy and .csv are supported.")
    
    # Calculate statistics
    mean_val = float(np.mean(data_flat))
    std_val = float(np.std(data_flat))
    min_val = float(np.min(data_flat))
    max_val = float(np.max(data_flat))
    data_range = [min_val, max_val]
    
    # Determine normalization type based on statistics
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
    # Raw power data typically has: mean > 0, std > mean, min >= 0, max >> 1
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
    
    # Prepare result dictionary
    result = {
        'type': norm_type,
        'confidence': confidence,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'range': data_range,
        'shape': original_shape
    }
    
    # Print results if verbose
    if verbose:
        print("=" * 60)
        print("NORMALIZATION DETECTION RESULTS")
        print("=" * 60)
        print(f"File: {file_path}")
        print(f"File type: {file_ext.upper()}")
        print(f"\nData Statistics:")
        print(f"  Shape: {result['shape']}")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std:  {std_val:.6f}")
        print(f"  Min:  {min_val:.6f}")
        print(f"  Max:  {max_val:.6f}")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"\nDetected Normalization:")
        print(f"  Type: {norm_type}")
        print(f"  Confidence: {confidence}")
        print("=" * 60)
    
    return result


def interactive_viewer(file_path, max_windows=100):
    """
    Interactive MATLAB-like data viewer with scrolling and zoom controls.
    
    Args:
        file_path (str): Path to CSV or NPY file
        max_windows (int): Maximum number of windows to load (for performance)
    
    Features:
        - Scroll through windows with slider
        - Zoom in/out with mouse wheel or buttons
        - Pan with mouse drag
        - Scale control (auto, manual)
    """
    print(f"\nLoading data for interactive viewer: {file_path}")
    
    # Load data using the detection function
    result = detect_normalization(file_path, verbose=False)
    
    # Load full data
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.npy':
        data = np.load(file_path)
        # Reshape to (num_windows, window_length, features)
        if len(data.shape) == 3:
            windows = data
        elif len(data.shape) == 2:
            # Assume it's (samples, features) - create windows
            window_size = 1024
            num_windows = min(data.shape[0] // window_size, max_windows)
            windows = data[:num_windows * window_size].reshape(num_windows, window_size, -1)
        else:
            # Flatten and reshape
            data_flat = data.flatten()
            window_size = 1024
            num_windows = min(len(data_flat) // window_size, max_windows)
            windows = data_flat[:num_windows * window_size].reshape(num_windows, window_size, 1)
    else:  # CSV
        df = pd.read_csv(file_path, header=0)
        if 'power' in df.columns:
            data_flat = df['power'].values
        else:
            data_flat = df.iloc[:, 0].values
        
        window_size = 1024
        num_windows = min(len(data_flat) // window_size, max_windows)
        windows = data_flat[:num_windows * window_size].reshape(num_windows, window_size, 1)
    
    num_windows, window_length, num_features = windows.shape
    print(f"Loaded {num_windows} windows, each with {window_length} time steps")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95)
    
    # Initial window to display
    current_window = 0
    y_min, y_max = windows.min(), windows.max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1
    
    # Plot initial window
    line, = ax.plot(windows[current_window, :, 0], linewidth=1.5, color='blue')
    ax.set_xlim(0, window_length)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Window {current_window + 1}/{num_windows} | Range: [{y_min:.4f}, {y_max:.4f}]', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create slider for window navigation
    ax_slider = plt.axes([0.1, 0.1, 0.6, 0.03])
    window_slider = Slider(ax_slider, 'Window', 0, num_windows - 1, 
                          valinit=current_window, valstep=1, valfmt='%d')
    
    # Create scale control slider
    ax_scale = plt.axes([0.1, 0.05, 0.6, 0.03])
    scale_slider = Slider(ax_scale, 'Y-Scale', 0.1, 5.0, valinit=1.0, valfmt='%.2f')
    
    # Create buttons
    ax_prev = plt.axes([0.75, 0.1, 0.08, 0.04])
    ax_next = plt.axes([0.84, 0.1, 0.08, 0.04])
    ax_zoom_in = plt.axes([0.75, 0.05, 0.08, 0.04])
    ax_zoom_out = plt.axes([0.84, 0.05, 0.08, 0.04])
    ax_reset = plt.axes([0.94, 0.05, 0.05, 0.09])
    
    btn_prev = Button(ax_prev, '◄ Prev')
    btn_next = Button(ax_next, 'Next ►')
    btn_zoom_in = Button(ax_zoom_in, 'Zoom +')
    btn_zoom_out = Button(ax_zoom_out, 'Zoom -')
    btn_reset = Button(ax_reset, 'Reset\nView')
    
    # Store current view state
    view_state = {
        'x_min': 0,
        'x_max': window_length,
        'y_center': (y_min + y_max) / 2,
        'y_range': y_range + 2 * y_padding
    }
    
    def update_window(val):
        """Update displayed window"""
        nonlocal current_window
        current_window = int(window_slider.val)
        line.set_ydata(windows[current_window, :, 0])
        
        # Update window-specific stats
        win_min = windows[current_window, :, 0].min()
        win_max = windows[current_window, :, 0].max()
        win_mean = windows[current_window, :, 0].mean()
        
        ax.set_title(f'Window {current_window + 1}/{num_windows} | '
                    f'Range: [{win_min:.4f}, {win_max:.4f}] | Mean: {win_mean:.4f}',
                    fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()
    
    def update_scale(val):
        """Update Y-axis scale"""
        scale_factor = scale_slider.val
        current_y_center = view_state['y_center']
        new_range = view_state['y_range'] / scale_factor
        view_state['y_range'] = new_range
        ax.set_ylim(current_y_center - new_range/2, current_y_center + new_range/2)
        fig.canvas.draw_idle()
    
    def prev_window(event):
        """Go to previous window"""
        if current_window > 0:
            window_slider.set_val(current_window - 1)
    
    def next_window(event):
        """Go to next window"""
        if current_window < num_windows - 1:
            window_slider.set_val(current_window + 1)
    
    def zoom_in(event):
        """Zoom in"""
        scale_slider.set_val(min(5.0, scale_slider.val + 0.2))
    
    def zoom_out(event):
        """Zoom out"""
        scale_slider.set_val(max(0.1, scale_slider.val - 0.2))
    
    def reset_view(event):
        """Reset view to default"""
        win_min = windows[current_window, :, 0].min()
        win_max = windows[current_window, :, 0].max()
        y_pad = (win_max - win_min) * 0.1
        view_state['y_center'] = (win_min + win_max) / 2
        view_state['y_range'] = (win_max - win_min) + 2 * y_pad
        scale_slider.set_val(1.0)
        ax.set_ylim(win_min - y_pad, win_max + y_pad)
        ax.set_xlim(0, window_length)
        fig.canvas.draw_idle()
    
    # Connect callbacks
    window_slider.on_changed(update_window)
    scale_slider.on_changed(update_scale)
    btn_prev.on_clicked(prev_window)
    btn_next.on_clicked(next_window)
    btn_zoom_in.on_clicked(zoom_in)
    btn_zoom_out.on_clicked(zoom_out)
    btn_reset.on_clicked(reset_view)
    
    # Enable mouse wheel zoom
    def on_scroll(event):
        """Handle mouse wheel scroll for zoom"""
        if event.inaxes == ax:
            if event.button == 'up':
                zoom_in(event)
            elif event.button == 'down':
                zoom_out(event)
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Enable pan with middle mouse button
    def on_press(event):
        """Handle mouse press for panning"""
        if event.inaxes == ax and event.button == 2:  # Middle mouse button
            view_state['press_x'] = event.xdata
            view_state['press_y'] = event.ydata
    
    def on_motion(event):
        """Handle mouse motion for panning"""
        if event.inaxes == ax and event.button == 2 and 'press_x' in view_state:
            dx = event.xdata - view_state['press_x']
            dy = event.ydata - view_state['press_y']
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            fig.canvas.draw_idle()
    
    def on_release(event):
        """Handle mouse release"""
        if 'press_x' in view_state:
            del view_state['press_x']
            if 'press_y' in view_state:
                del view_state['press_y']
    
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    
    # Add keyboard shortcuts
    def on_key(event):
        """Handle keyboard shortcuts"""
        if event.key == 'left' or event.key == 'a':
            prev_window(None)
        elif event.key == 'right' or event.key == 'd':
            next_window(None)
        elif event.key == '+' or event.key == '=':
            zoom_in(None)
        elif event.key == '-' or event.key == '_':
            zoom_out(None)
        elif event.key == 'r' or event.key == 'R':
            reset_view(None)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add instructions
    instructions = (
        "Controls:\n"
        "• Slider: Navigate windows\n"
        "• Y-Scale: Adjust vertical zoom\n"
        "• Buttons: Prev/Next, Zoom +/-, Reset\n"
        "• Mouse Wheel: Zoom in/out\n"
        "• Middle Mouse: Pan\n"
        "• Keyboard: ←/→ (prev/next), +/- (zoom), R (reset)"
    )
    fig.text(0.02, 0.98, instructions, transform=fig.transFigure,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    print("\n" + "=" * 60)
    print("INTERACTIVE VIEWER OPENED")
    print("=" * 60)
    print("Use the controls to navigate and zoom the data.")
    print("Close the window when done.")
    print("=" * 60 + "\n")
    
    plt.show()


# Main function for command-line usage
def main():
    """
    Main function for command-line interface.
    Supports both interactive prompt and --path argument.
    """
    parser = argparse.ArgumentParser(
        description='Detect normalization type of data in CSV or NPY files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for file path)
  python detect_normalization.py
  
  # Specify file path via argument
  python detect_normalization.py --path data.npy
  python detect_normalization.py -p data.csv
  
  # Silent mode (no verbose output)
  python detect_normalization.py --path data.npy --quiet
  
  # Open interactive MATLAB-like viewer
  python detect_normalization.py --path data.npy --view
  python detect_normalization.py -p data.npy -v
  
  # Interactive prompt mode (viewer opens automatically)
  python detect_normalization.py
  # Then enter file path when prompted - viewer will open automatically!
  
  # Skip viewer in interactive mode
  python detect_normalization.py --no-view
  
  # Viewer with custom max windows
  python detect_normalization.py --path data.npy --view --max-windows 50
        """
    )
    
    parser.add_argument(
        '--path', '-p',
        type=str,
        default=None,
        help='Path to the CSV or NPY file to analyze'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode: suppress verbose output'
    )
    
    parser.add_argument(
        '--view', '-v',
        action='store_true',
        help='Open interactive MATLAB-like viewer after detection'
    )
    
    parser.add_argument(
        '--no-view',
        action='store_true',
        help='Skip opening viewer (useful when using --path)'
    )
    
    parser.add_argument(
        '--max-windows',
        type=int,
        default=100,
        help='Maximum number of windows to load in viewer (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Track if we're in interactive prompt mode
    interactive_mode = False
    
    # Get file path: from argument or prompt user
    if args.path:
        file_path = args.path
    else:
        # Interactive prompt
        interactive_mode = True
        print("=" * 60)
        print("NORMALIZATION DETECTION")
        print("=" * 60)
        print("Enter the path to your CSV or NPY file:")
        print("(You can also use: python detect_normalization.py --path <file_path>)")
        print("-" * 60)
        file_path = input("File path: ").strip()
        
        # Remove quotes if user added them
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if file_path.startswith("'") and file_path.endswith("'"):
            file_path = file_path[1:-1]
    
    # Check if file exists
    if not file_path:
        print("Error: No file path provided.")
        sys.exit(1)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Run detection
    try:
        verbose = not args.quiet
        result = detect_normalization(file_path, verbose=verbose)
        
        # If quiet mode, just print the type
        if args.quiet:
            print(f"{result['type']}")
        
        # Open interactive viewer:
        # - If --view flag is used, OR
        # - If in interactive prompt mode (unless --no-view is specified)
        should_view = args.view or (interactive_mode and not args.no_view)
        
        if should_view:
            try:
                print("\n" + "=" * 60)
                print("Opening interactive viewer...")
                print("=" * 60)
                interactive_viewer(file_path, max_windows=args.max_windows)
            except Exception as e:
                print(f"Error opening viewer: {e}")
                print("Make sure matplotlib backend supports interactive mode.")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# Example usage
if __name__ == '__main__':
    main()

