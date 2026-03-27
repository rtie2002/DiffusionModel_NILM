import numpy as np
import argparse
import os
import yaml
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

# Provide the exact path to your config
BASE_DIR = Path(__file__).resolve().parent.parent # DiffusionModel_NILM directory
CONFIG_PATH = os.path.join(BASE_DIR, 'Config', 'preprocess', 'preprocess_multivariate.yaml')


def apply_base_noise_filter(power_sequence, background_threshold=15.0, bridge_gap=30):
    """
    Zeroes out noise below the threshold globally, but PROTECTS temporary dips 
    that occur inside an active appliance cycle by bridging gaps forward.
    """
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    
    # 1. Identify raw active points
    is_active = (power_sequence >= background_threshold).astype(int)
    
    # 2. Bridge small gaps (protect internal dips)
    for i in range(1, n - bridge_gap):
        if is_active[i-1] == 1 and is_active[i] == 0:
            upcoming = is_active[i:i+bridge_gap]
            if np.any(upcoming == 1):
                next_active = np.where(upcoming == 1)[0][0]
                is_active[i:i+next_active] = 1
                
    # 3. Apply mask to zero out ONLY the TRUE background noise outside of cycles
    power_sequence[is_active == 0] = 0.0
    
    return power_sequence


def remove_isolated_spikes(power_sequence, window_size=5, spike_threshold=3.0, 
                          background_threshold=50):
    """
    Remove isolated spikes that appear in the middle of OFF periods.
    Works entirely in WATTS space.
    """
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    num_spikes = 0
    
    # Pad array for edge handling
    half_window = window_size // 2
    padded = np.pad(power_sequence, half_window, mode='edge')
    
    for i in range(n):
        current_value = power_sequence[i]
        # Skip checking if already silently low
        if current_value < max(1.0, background_threshold * 0.1): 
            continue
        
        # Get surrounding values (excluding center point)
        window_start, window_end = i, i + window_size
        window = padded[window_start:window_end]
        surrounding = np.concatenate([window[:half_window], window[half_window+1:]])
        
        median_surrounding = np.median(surrounding)
        
        if current_value > background_threshold:
            # Check if surroundings are mostly 'OFF' (near zero, e.g. < 15W)
            is_background_quiet = np.all(surrounding < 15.0)
            
            if is_background_quiet and current_value > spike_threshold * (median_surrounding + 1.0):
                power_sequence[i] = 0
                num_spikes += 1
                
    return power_sequence, num_spikes

def validate_full_cycles(power_sequence, background_threshold=15.0, 
                        min_peak=1000.0, bridge_gap=20, min_duration=80):
    """
    Cycle-wise Validation.
    Groups clusters and zeroes out those without a required Watts signature.
    Works entirely in WATTS space.
    """
    power_sequence = power_sequence.copy()
    n = len(power_sequence)
    is_active = (power_sequence >= background_threshold).astype(int)
    
    # Bridge small gaps
    for i in range(1, n - bridge_gap):
        if is_active[i-1] == 1 and is_active[i] == 0:
            upcoming = is_active[i:i+bridge_gap]
            if np.any(upcoming == 1):
                next_active = np.where(upcoming == 1)[0][0]
                is_active[i:i+next_active] = 1

    # Analyze segments
    diff = np.diff(np.concatenate(([0], is_active, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    num_fake_segments = 0
    for start, end in zip(starts, ends):
        segment = power_sequence[start:end]
        # Kill if peak is too low OR if totally duration is too short to be a real cycle
        if np.max(segment) < min_peak or (end-start) < min_duration:
            power_sequence[start:end] = 0
            num_fake_segments += 1
                
    return power_sequence, num_fake_segments


def visualize_comparison(original_data, filtered_data, window_size=512):
    """
    Interactive matplotlib viewer to compare original vs filtered power windows.
    Both arrays should be of shape (N, L) containing pure Watts data.
    """
    N = original_data.shape[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95, top=0.90, hspace=0.3)
    
    current_idx = 0
    x_axis = np.arange(window_size)
    
    # Initial plots
    line1, = ax1.plot(x_axis, original_data[current_idx], color='red', alpha=0.8, linewidth=1.5, label='Original (Noisy)')
    line2, = ax2.plot(x_axis, filtered_data[current_idx], color='green', alpha=0.9, linewidth=1.5, label='Filtered (Cleaned)')
    
    ax1.set_title(f"Window {current_idx+1}/{N} - Original", fontweight='bold')
    ax2.set_title(f"Window {current_idx+1}/{N} - Filtered", fontweight='bold')
    ax2.set_xlabel("Time step")
    
    for ax in [ax1, ax2]:
        ax.set_ylabel("Power (Watts)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
    ax1.set_xlim(0, window_size)
    y_max = max(np.max(original_data[current_idx]), np.max(filtered_data[current_idx])) * 1.2
    if y_max < 10: y_max = 50
    ax1.set_ylim(-10, y_max)
    
    # Slider
    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
    window_slider = Slider(ax_slider, 'Window', 0, N - 1, valinit=current_idx, valstep=1, valfmt='%d')
    
    # Buttons
    ax_prev_btn = plt.axes([0.15, 0.04, 0.09, 0.04])
    ax_next_btn = plt.axes([0.25, 0.04, 0.09, 0.04])
    ax_autofit = plt.axes([0.36, 0.04, 0.09, 0.04])
    ax_zoom_btn = plt.axes([0.47, 0.04, 0.12, 0.04])
    ax_clear_btn = plt.axes([0.60, 0.04, 0.12, 0.04])
    
    btn_prev = Button(ax_prev_btn, '◀ Prev')
    btn_next = Button(ax_next_btn, 'Next ▶')
    btn_autofit = Button(ax_autofit, 'Auto-Fit Y')
    btn_zoom = Button(ax_zoom_btn, 'Zoom to Sel')
    btn_clear = Button(ax_clear_btn, 'Clear Selection')
    
    # Selection State Tracking for Drag-and-Zoom
    selection_state = {
        'active': False,
        'start_x': None,
        'start_y': None,
        'rect1': None,
        'rect2': None,
        'x_min': None,
        'x_max': None,
        'y_min': None,
        'y_max': None,
        'saved_xlim': None,
        'saved_ylim': None
    }
    
    def clear_selection(event=None):
        if selection_state['rect1'] is not None:
            selection_state['rect1'].remove()
            selection_state['rect1'] = None
        if selection_state['rect2'] is not None:
            selection_state['rect2'].remove()
            selection_state['rect2'] = None
            
        if selection_state['saved_xlim'] is not None:
            ax1.set_xlim(selection_state['saved_xlim'])
            ax1.set_ylim(selection_state['saved_ylim'])
            
        selection_state['x_min'] = None
        selection_state['x_max'] = None
        selection_state['y_min'] = None
        selection_state['y_max'] = None
        selection_state['saved_xlim'] = None
        selection_state['saved_ylim'] = None
        fig.canvas.draw_idle()

    def zoom_to_selection(event=None):
        if selection_state['x_min'] is None:
            print("No selection made. Drag on the chart to select an area first.")
            return
        
        selection_state['saved_xlim'] = ax1.get_xlim()
        selection_state['saved_ylim'] = ax1.get_ylim()
        
        x_min, x_max = selection_state['x_min'], selection_state['x_max']
        y_min, y_max = selection_state['y_min'], selection_state['y_max']
        
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        
        ax1.set_xlim(x_min - x_pad, x_max + x_pad)
        ax1.set_ylim(y_min - y_pad, y_max + y_pad)
        fig.canvas.draw_idle()
        
    def on_mouse_press(event):
        if event.inaxes not in [ax1, ax2] or event.button != 1:
            return
        selection_state['active'] = True
        selection_state['start_x'] = event.xdata
        selection_state['start_y'] = event.ydata
        
        if selection_state['rect1'] is not None:
            selection_state['rect1'].remove()
            selection_state['rect1'] = None
        if selection_state['rect2'] is not None:
            selection_state['rect2'].remove()
            selection_state['rect2'] = None
            
    def on_mouse_move(event):
        if not selection_state['active'] or event.inaxes not in [ax1, ax2]:
            return
        
        if selection_state['rect1'] is not None:
            selection_state['rect1'].remove()
        if selection_state['rect2'] is not None:
            selection_state['rect2'].remove()
            
        x0, y0 = selection_state['start_x'], selection_state['start_y']
        x1, y1 = event.xdata, event.ydata
        
        width = x1 - x0
        height = y1 - y0
        
        selection_state['rect1'] = plt.Rectangle((x0, y0), width, height, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7)
        selection_state['rect2'] = plt.Rectangle((x0, y0), width, height, fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7)
        
        ax1.add_patch(selection_state['rect1'])
        ax2.add_patch(selection_state['rect2'])
        fig.canvas.draw_idle()
        
    def on_mouse_release(event):
        if not selection_state['active'] or event.button != 1:
            return
        selection_state['active'] = False
        
        if event.inaxes not in [ax1, ax2] or selection_state['start_x'] is None:
            return
            
        x0, y0 = selection_state['start_x'], selection_state['start_y']
        x1, y1 = event.xdata, event.ydata
        
        selection_state['x_min'] = min(x0, x1)
        selection_state['x_max'] = max(x0, x1)
        selection_state['y_min'] = min(y0, y1)
        selection_state['y_max'] = max(y0, y1)
    
    def update(val):
        idx = int(window_slider.val)
        line1.set_ydata(original_data[idx])
        line2.set_ydata(filtered_data[idx])
        
        ax1.set_title(f"Window {idx+1}/{N} - Original", fontweight='bold')
        ax2.set_title(f"Window {idx+1}/{N} - Filtered", fontweight='bold')
        
        # Keep user zoom state if they have one, unless they used AutoFit recently.
        fig.canvas.draw_idle()
        
    def autofit(event):
        idx = int(window_slider.val)
        y_max = max(np.max(original_data[idx]), np.max(filtered_data[idx])) * 1.2
        if y_max < 10: y_max = 50
        ax1.set_xlim(0, window_size)
        ax1.set_ylim(-10, y_max)
        
        # Reset selection states on autofit
        if selection_state['rect1'] is not None:
            selection_state['rect1'].remove()
            selection_state['rect1'] = None
        if selection_state['rect2'] is not None:
            selection_state['rect2'].remove()
            selection_state['rect2'] = None
        selection_state['saved_xlim'] = None
        selection_state['saved_ylim'] = None
        selection_state['x_min'] = None
        
        fig.canvas.draw_idle()
        
    def prev_w(event):
        if window_slider.val > 0:
            window_slider.set_val(window_slider.val - 1)
            
    def next_w(event):
        if window_slider.val < N - 1:
            window_slider.set_val(window_slider.val + 1)
            
    window_slider.on_changed(update)
    btn_prev.on_clicked(prev_w)
    btn_next.on_clicked(next_w)
    btn_autofit.on_clicked(autofit)
    btn_zoom.on_clicked(zoom_to_selection)
    btn_clear.on_clicked(clear_selection)
    
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    
    print("\n[Visualizer] Closing the window will continue the save process...")
    plt.show()


def process_npy_file(input_file, appliance_name, visualize=False):
    print("=" * 60)
    print(f"NPY STANDALONE NOISE FILTER - {appliance_name.upper()}")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(input_file):
        print(f"❌ Error: Could not find file {input_file}")
        return
        
    print(f"Loading '{os.path.basename(input_file)}'...")
    samples = np.load(input_file)
    print(f"  Shape: {samples.shape}")
    
    # Handle dimensionality
    if len(samples.shape) == 3:
        N, L, V = samples.shape
    elif len(samples.shape) == 2:
        N, L = samples.shape
        V = 1
        samples = samples.reshape(N, L, V)
    else:
        print("❌ Error: Unrecognized shape. Needs to be (N, L, V) or (N, L).")
        return

    # 2. Load Yaml Specs
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    if appliance_name not in config['appliances']:
        print(f"❌ Error: Appliance '{appliance_name}' not found in {CONFIG_PATH}")
        return
        
    app_specs = config['appliances'][appliance_name]
    max_power = app_specs.get('max_power', 1000.0)
    noise_thres_watts = app_specs.get('on_power_threshold', 15.0)
    clip_max_watts = app_specs.get('max_power_clip', None)

    print(f"Loaded params for {appliance_name}: MaxPower={max_power}W, Threshold={noise_thres_watts}W")

    # 3. Filtering Preparation
    # Back up time features (if multivariate)
    if V > 1:
        time_feats = samples[:, :, 1:].copy()
        
    # Extract just the power column and flatten to 1D
    power_seq_minmax = samples[:, :, 0].flatten()
    
    # Convert MinMax space [0, 1] to Watts space for logic calculations
    power_seq_watts = power_seq_minmax * max_power
    
    # Save a perfect copy of the ORIGINAL Data in Watts for visual comparison
    if visualize:
        original_watts_visual = copy.deepcopy(power_seq_watts).reshape(N, L)

    # 4. Filters
    # --- ROUND 1: Base Noise Filter (Smart Masking) ---
    print(f"  ✓ Round 1: Smart Noise Filter (< {noise_thres_watts}W protected inside cycles)...")
    power_seq_watts = apply_base_noise_filter(power_seq_watts, background_threshold=noise_thres_watts, bridge_gap=30)
    
    # --- ROUND 2: Appliance Specific ---
    if appliance_name.lower() == 'washingmachine':
        print(f"  ✓ Round 2: Searching for isolated spikes...")
        power_seq_watts, n_spikes = remove_isolated_spikes(
            power_seq_watts, window_size=5, spike_threshold=3.0, background_threshold=noise_thres_watts
        )
        if n_spikes > 0:
            print(f"    - Cleaned {n_spikes} isolated glitches.")
            
        print("  ✓ Round 2: Validating Washing Machine signature...")
        power_seq_watts, n_fake = validate_full_cycles(
            power_seq_watts, background_threshold=noise_thres_watts, 
            min_peak=1000.0, bridge_gap=20, min_duration=80
        )
        if n_fake > 0:
            print(f"    - Removed {n_fake} fake cycles without 1000W peaks.")

    # --- ROUND 3: Hard Clip Limit ---
    if clip_max_watts is not None:
        print(f"  ✓ Round 3: Clipping max power to {clip_max_watts}W...")
        power_seq_watts = np.clip(power_seq_watts, 0.0, clip_max_watts)

    # 5. Restore shape and Format
    # Convert Watts back to MinMax space [0, 1]
    filtered_watts_visual = power_seq_watts.reshape(N, L)
    power_seq_minmax = power_seq_watts / max_power
    samples[:, :, 0] = power_seq_minmax.reshape(N, L)
    
    # Restore time features identically
    if V > 1:
        samples[:, :, 1:] = time_feats
        
    # Optional Visualization Step
    if visualize:
        visualize_comparison(original_watts_visual, filtered_watts_visual, window_size=L)
        
    # 6. Save
    dir_name = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Prevent appending CLEANED multiple times
    if base_name.endswith('_CLEANED'):
        save_fname = f"{base_name}.npy" 
    else:
        save_fname = f"{base_name}_CLEANED.npy"
        
    save_path = os.path.join(dir_name, save_fname)
    np.save(save_path, samples)
    
    print(f"✅ SUCCESSFULLY SAVED!")
    print(f"   Shape: {samples.shape}")
    print(f"   Path:  {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone NPY Noise Filter")
    parser.add_argument("--input", type=str, required=False, help="Path to the original .npy file")
    parser.add_argument("--appliance", type=str, required=False, 
                        choices=["kettle", "microwave", "fridge", "dishwasher", "washingmachine"],
                        help="Appliance name for fetching YAML thresholds")
    parser.add_argument("--visualize", action="store_true", 
                        help="Open an interactive matplotlib window to compare Before/After before saving")
    args = parser.parse_args()
    
    # --- INTERACTIVE FALLBACK ---
    input_path = args.input
    appliance = args.appliance
    visualize = args.visualize

    if input_path is None:
        print("\n" + "="*60)
        print("NPY POST-PROCESSOR: INTERACTIVE MODE")
        print("="*60)
        input_path = input("\n➤ Enter the path to your .npy file: ").strip().strip('"').strip("'")
        # If run interactively, default visualize to True unless explicitly disabled
        visualize = True 

    if not input_path:
        print("❌ Error: No input path provided. Exiting.")
        exit(1)

    if appliance is None:
        print("\nAvailable Appliances: kettle, microwave, fridge, dishwasher, washingmachine")
        appliance = input("➤ Enter the appliance name: ").strip().lower()

    process_npy_file(input_path, appliance, visualize=visualize)
