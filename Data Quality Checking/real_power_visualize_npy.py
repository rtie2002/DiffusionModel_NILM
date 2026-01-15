import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import yaml
from pathlib import Path

# Set base directory for relative paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load configuration for denormalization parameters
CONFIG_PATH = BASE_DIR / 'Config/preprocess/preprocess_multivariate.yaml'

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file not found at {CONFIG_PATH}. Using default parameters.")
        return None

CONFIG = load_config()

# Default parameters if config is missing
DEFAULT_PARAMS = {
    'kettle': {'mean': 700, 'std': 1000, 'max_power': 3998},
    'microwave': {'mean': 500, 'std': 800, 'max_power': 3969},
    'fridge': {'mean': 200, 'std': 400, 'max_power': 3323},
    'dishwasher': {'mean': 700, 'std': 1000, 'max_power': 3964},
    'washingmachine': {'mean': 400, 'std': 700, 'max_power': 3999}
}

def get_appliance_params(appliance_name):
    if CONFIG and 'appliances' in CONFIG and appliance_name in CONFIG['appliances']:
        app_config = CONFIG['appliances'][appliance_name]
        return {
            'mean': app_config['mean'],
            'std': app_config['std'],
            'max_power': app_config.get('real_max_power', app_config['max_power'])
        }
    return DEFAULT_PARAMS.get(appliance_name, DEFAULT_PARAMS['kettle'])

def detect_appliance_from_path(file_path):
    file_lower = os.path.basename(file_path).lower()
    for appliance in DEFAULT_PARAMS.keys():
        if appliance in file_lower:
            return appliance
    return None

def detect_normalization(data):
    """Detect if data is Z-score or MinMax [0, 1]"""
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    
    # If mean is near 0 and has negative values, likely Z-score
    if abs(mean_val) < 0.5 and min_val < 0:
        return 'zscore'
    # If values are in [0, 1] range, likely MinMax
    elif 0 <= min_val < 0.2 and 0.1 < max_val <= 1.1:
        return 'minmax'
    return 'zscore' # Default to zscore

def denormalize(data, appliance_name, norm_type):
    params = get_appliance_params(appliance_name)
    if norm_type == 'zscore':
        return data * params['std'] + params['mean']
    elif norm_type == 'minmax':
        return data * params['max_power']
    return data

def main():
    parser = argparse.ArgumentParser(description='Visualize power data from .npy files')
    parser.add_argument('--path', '-p', type=str, default=None, help='Path to .npy file')
    parser.add_argument('--appliance', '-a', type=str, default=None, help='Appliance name')
    parser.add_argument('--no-denorm', action='store_true', help='Skip denormalization')
    args = parser.parse_args()

    # Interactive prompt for file path
    file_path = args.path
    if not file_path:
        print("=" * 60)
        print("MULTIVARIATE NPY VISUALIZER")
        print("=" * 60)
        print("Enter the path to your .npy file:")
        file_path = input("Path: ").strip()
        
        # Remove quotes
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if file_path.startswith("'") and file_path.endswith("'"):
            file_path = file_path[1:-1]

    if not file_path or not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return

    # Load data
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"❌ Error loading .npy file: {e}")
        return

    print(f"✓ Loaded data shape: {data.shape}")

    # Prepare data for plotting
    if len(data.shape) == 3:
        num_windows = data.shape[0]
        window_size = data.shape[1]
        num_channels = data.shape[2]
    elif len(data.shape) == 2:
        num_windows = data.shape[0]
        window_size = data.shape[1]
        num_channels = 1
        data = data[:, :, np.newaxis]
    else:
        print(f"❌ Error: Unsupported data shape: {data.shape}")
        return

    # Detect appliance and denormalize
    appliance_name = args.appliance or detect_appliance_from_path(file_path)
    if not appliance_name:
        print(f"\n⚠ Could not auto-detect appliance from filename.")
        print(f"Available: {', '.join(DEFAULT_PARAMS.keys())}")
        while True:
            user_input = input("Enter appliance name: ").strip().lower()
            if user_input in DEFAULT_PARAMS:
                appliance_name = user_input
                break
            print(f"Invalid appliance. Please choose from: {', '.join(DEFAULT_PARAMS.keys())}")
            
    print(f"✓ Using appliance: {appliance_name}")

    power_data = data[:, :, 0]
    if not args.no_denorm:
        norm_type = detect_normalization(power_data)
        print(f"✓ Detected normalization: {norm_type}")
        power_denorm = denormalize(power_data, appliance_name, norm_type)
    else:
        power_denorm = power_data
        norm_type = "none"

    # Setup Plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.20, left=0.08, right=0.90, top=0.92)
    
    current_window = 0
    
    # State tracking
    selection_state = {
        'active': False, 'start_x': None, 'start_y': None, 'rect': None,
        'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None,
        'saved_xlim': None, 'saved_ylim': None
    }
    
    # Primary plot
    start_step = current_window * window_size
    x_data = np.arange(start_step, start_step + window_size)
    line_power, = ax1.plot(x_data, power_denorm[current_window], color='blue', linewidth=1.5, label='Appliance Power')
    ax1.set_xlabel('Global Time Step')
    ax1.set_ylabel('Power (Watts)' if not args.no_denorm else f'Value ({norm_type})')
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for time features
    ax2 = None
    lines_time = []
    if num_channels > 1:
        ax2 = ax1.twinx()
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        time_labels = ['minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
        for i in range(1, num_channels):
            color = colors[(i-1) % len(colors)]
            label = time_labels[i-1] if i-1 < len(time_labels) else f'CH {i}'
            line, = ax2.plot(x_data, data[current_window, :, i], color=color, alpha=0.5, linestyle='--', label=label)
            lines_time.append(line)
        ax2.set_ylabel('Time Features')
    
    # Interactive legend
    lines = [line_power] + lines_time
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='upper right')
    for leg_line in legend.get_lines(): leg_line.set_picker(True); leg_line.set_pickradius(5)
    for leg_text in legend.get_texts(): leg_text.set_picker(True)

    # Controls
    ax_slider = plt.axes([0.08, 0.12, 0.50, 0.03])
    window_slider = Slider(ax_slider, 'Window', 0, num_windows - 1, valinit=0, valstep=1, valfmt='%d')
    
    ax_scale = plt.axes([0.08, 0.07, 0.50, 0.03])
    scale_slider = Slider(ax_scale, 'Y-Scale', 0.1, 5.0, valinit=1.0, valfmt='%.2f')
    
    ax_prev = plt.axes([0.62, 0.12, 0.06, 0.04]); btn_prev = Button(ax_prev, '◀ Prev')
    ax_next = plt.axes([0.69, 0.12, 0.06, 0.04]); btn_next = Button(ax_next, 'Next ▶')
    ax_reset = plt.axes([0.76, 0.12, 0.12, 0.04]); btn_reset = Button(ax_reset, 'Auto-Fit Y')
    
    ax_zoom_sel = plt.axes([0.62, 0.07, 0.10, 0.04]); btn_zoom_sel = Button(ax_zoom_sel, 'Zoom Sel')
    ax_clear_sel = plt.axes([0.73, 0.07, 0.15, 0.04]); btn_clear_sel = Button(ax_clear_sel, 'Clear Sel')

    def update_view_range():
        if not line_power.get_visible(): return
        y_data = line_power.get_ydata()
        y_min, y_max = y_data.min(), y_data.max()
        y_range = max(y_max - y_min, 1)
        pad = y_range * 0.1
        center, rng = (y_max + y_min) / 2, (y_range + 2*pad) / scale_slider.val
        ax1.set_ylim(center - rng/2, center + rng/2)

    def update(val):
        idx = int(window_slider.val)
        start_step = idx * window_size
        x_data = np.arange(start_step, start_step + window_size)
        
        line_power.set_xdata(x_data)
        line_power.set_ydata(power_denorm[idx])
        
        if ax2:
            for i, line in enumerate(lines_time):
                line.set_xdata(x_data)
                line.set_ydata(data[idx, :, i+1])
        
        ax1.set_xlim(start_step, start_step + window_size)
        ax1.set_title(f'File: {os.path.basename(file_path)} | Window: {idx+1}/{num_windows} (Steps {start_step:,}-{start_step+window_size:,})', fontsize=12, fontweight='bold')
        fig.canvas.draw_idle()

    def on_scale(val): update_view_range(); fig.canvas.draw_idle()

    def on_pick(event):
        leg_lines, leg_texts = legend.get_lines(), legend.get_texts()
        for i, (ll, ol) in enumerate(zip(leg_lines, lines)):
            if event.artist == ll or event.artist == leg_texts[i]:
                vis = not ol.get_visible()
                ol.set_visible(vis)
                ll.set_alpha(1.0 if vis else 0.2)
                leg_texts[i].set_alpha(1.0 if vis else 0.3)
                fig.canvas.draw_idle()
                return

    window_slider.on_changed(update)
    scale_slider.on_changed(on_scale)
    btn_prev.on_clicked(lambda e: window_slider.set_val(max(0, int(window_slider.val) - 1)))
    btn_next.on_clicked(lambda e: window_slider.set_val(min(num_windows - 1, int(window_slider.val) + 1)))
    btn_reset.on_clicked(lambda e: (scale_slider.set_val(1.0), update_view_range(), fig.canvas.draw_idle()))

    # Mouse Selection
    def on_press(event):
        if event.inaxes != ax1 or event.button != 1: return
        selection_state.update({'active': True, 'start_x': event.xdata, 'start_y': event.ydata})
        if selection_state['rect']: selection_state['rect'].remove(); selection_state['rect'] = None

    def on_move(event):
        if not selection_state['active'] or event.inaxes != ax1: return
        if selection_state['rect']: selection_state['rect'].remove()
        width, height = event.xdata - selection_state['start_x'], event.ydata - selection_state['start_y']
        selection_state['rect'] = plt.Rectangle((selection_state['start_x'], selection_state['start_y']), width, height, fill=False, edgecolor='red', linewidth=1.5, linestyle='--')
        ax1.add_patch(selection_state['rect'])
        fig.canvas.draw_idle()

    def on_release(event):
        if not selection_state['active']: return
        selection_state['active'] = False
        if event.inaxes != ax1: return
        selection_state['x_min'], selection_state['x_max'] = sorted([selection_state['start_x'], event.xdata])
        selection_state['y_min'], selection_state['y_max'] = sorted([selection_state['start_y'], event.ydata])
        print(f"\nSelection: X=[{selection_state['x_min']:.1f}, {selection_state['x_max']:.1f}], Y=[{selection_state['y_min']:.1f}, {selection_state['y_max']:.1f}]")

    btn_zoom_sel.on_clicked(lambda e: (selection_state.update({'saved_xlim': ax1.get_xlim(), 'saved_ylim': ax1.get_ylim()}), ax1.set_xlim(selection_state['x_min'], selection_state['x_max']), ax1.set_ylim(selection_state['y_min'], selection_state['y_max']), fig.canvas.draw_idle()) if selection_state['x_min'] else None)
    btn_clear_sel.on_clicked(lambda e: (ax1.set_xlim(selection_state['saved_xlim']) if selection_state['saved_xlim'] else None, ax1.set_ylim(selection_state['saved_ylim']) if selection_state['saved_ylim'] else None, selection_state.update({'x_min': None, 'rect': (selection_state['rect'].remove() if selection_state['rect'] else None)}), fig.canvas.draw_idle()))

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Init
    update(0); update_view_range()
    print("\nInteractive NPY Viewer Controls:")
    print("- Slider/Buttons: Navigate windows")
    print("- Scale Slider/Auto-Fit: Control vertical zoom")
    print("- Left-Click + Drag: Select region")
    print("- Legend: Click to toggle visibility")
    plt.show()

if __name__ == "__main__":
    main()
