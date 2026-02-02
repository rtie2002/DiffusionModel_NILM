"""
Waveform Viewer for ALL Appliances - Real vs Synthetic NILM Data

Displays all 5 appliances simultaneously in a grid layout with
overlaid real (red) and synthetic (blue) waveforms for comparison.

Features:
- All appliances displayed at once (5 subplots)
- Overlaid real vs synthetic waveforms
- Interactive navigation (Previous/Next/Random)
- Zoom In/Out controls
- Save figure option
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

# ==================== Configuration ====================
BASE_DIR = r"C:\Users\Raymond Tie\Desktop\DiffusionModel_NILM"
REAL_DATA_DIR = os.path.join(BASE_DIR, "Data", "datasets", "real_distributions")
SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "Data", "datasets", "synthetic_processed")

APPLIANCES = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
APPLIANCE_LABELS = ["(a) Dishwasher", "(b) Fridge", "(c) Kettle", "(d) Microwave", "(e) Washing Machine"]
WINDOW_SIZE = 480  # Default window size (8 hours at 1-min resolution)


class AllAppliancesViewer:
    def __init__(self):
        self.window_size = WINDOW_SIZE
        self.start_idx = 0
        self.data = {}  # {appliance: {'real': df, 'synthetic': df}}
        self.fig = None
        self.axes = None
        self.buttons = []
        
    def load_all_data(self):
        """Load both real and synthetic data for all appliances."""
        print(f"\n{'='*60}")
        print("   Loading data for ALL appliances...")
        print(f"{'='*60}")
        
        for appliance in APPLIANCES:
            real_file = os.path.join(REAL_DATA_DIR, f"{appliance}_multivariate.csv")
            synthetic_file = os.path.join(SYNTHETIC_DATA_DIR, f"{appliance}_multivariate.csv")
            
            self.data[appliance] = {'real': None, 'synthetic': None}
            
            # Load real data
            if os.path.exists(real_file):
                self.data[appliance]['real'] = pd.read_csv(real_file)
                print(f"  {appliance:15} - Real: {len(self.data[appliance]['real']):>8} samples", end="")
            else:
                print(f"  {appliance:15} - Real: NOT FOUND", end="")
                
            # Load synthetic data
            if os.path.exists(synthetic_file):
                self.data[appliance]['synthetic'] = pd.read_csv(synthetic_file)
                print(f" | Synthetic: {len(self.data[appliance]['synthetic']):>8} samples")
            else:
                print(f" | Synthetic: NOT FOUND")
        
        print(f"\n{'='*60}")
        print("   All data loaded successfully!")
        print(f"{'='*60}\n")
    
    def get_window_data(self, data, start_idx, window_size):
        """Extract a window of data starting from the given index."""
        if data is None:
            return None
        end_idx = min(start_idx + window_size, len(data))
        return data.iloc[start_idx:end_idx]
    
    def plot_all_appliances(self):
        """Create a grid of plots showing all appliances with overlaid waveforms."""
        # Close existing figure if any
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create figure with 2 rows x 3 columns grid (5 appliances + 1 for controls info)
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle(f'Real vs Synthetic Waveform Comparison - All Appliances\n'
                         f'(Samples {self.start_idx} to {self.start_idx + self.window_size})', 
                         fontsize=14, fontweight='bold')
        
        # Create 2x3 grid of subplots
        self.axes = []
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]  # Grid positions for 5 appliances
        
        for i, (appliance, label) in enumerate(zip(APPLIANCES, APPLIANCE_LABELS)):
            row, col = positions[i]
            ax = self.fig.add_subplot(2, 3, i + 1)
            self.axes.append(ax)
            
            # Get data for this appliance
            real_data = self.data[appliance]['real']
            synth_data = self.data[appliance]['synthetic']
            
            power_col = appliance
            
            real_window = self.get_window_data(real_data, self.start_idx, self.window_size)
            synth_window = self.get_window_data(synth_data, self.start_idx, self.window_size)
            
            # Plot both waveforms overlaid
            if real_window is not None and len(real_window) > 0:
                real_power = real_window[power_col].values
                time_idx = np.arange(len(real_power))
                ax.plot(time_idx, real_power, color='red', linewidth=1.0, 
                       label='Real', alpha=0.9)
            
            if synth_window is not None and len(synth_window) > 0:
                synth_power = synth_window[power_col].values
                time_idx = np.arange(len(synth_power))
                ax.plot(time_idx, synth_power, color='blue', linewidth=1.0, 
                       label='Synthetic', alpha=0.8)
            
            # Style the subplot
            ax.set_xlabel('Time (samples)', fontsize=9)
            ax.set_ylabel('Power', fontsize=9)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(0, self.window_size)
        
        # Add info panel in the 6th subplot position
        ax_info = self.fig.add_subplot(2, 3, 6)
        ax_info.axis('off')
        info_text = (
            "Controls:\n"
            "─────────────────────\n"
            "← Previous: Move back\n"
            "→ Next: Move forward\n"
            "🎲 Random: Random position\n"
            "Zoom In: Smaller window\n"
            "Zoom Out: Larger window\n"
            "💾 Save: Save as PNG\n"
            "\n"
            "Legend:\n"
            "─────────────────────\n"
            "🔴 Red = Real Data\n"
            "🔵 Blue = Synthetic Data\n"
            f"\nWindow Size: {self.window_size} samples\n"
            f"Start Index: {self.start_idx}"
        )
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
        
        # Add navigation buttons
        self._add_navigation_buttons()
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.show()
    
    def _add_navigation_buttons(self):
        """Add interactive navigation buttons to the figure."""
        # Button axes at the bottom
        ax_prev = self.fig.add_axes([0.15, 0.02, 0.1, 0.04])
        ax_next = self.fig.add_axes([0.27, 0.02, 0.1, 0.04])
        ax_random = self.fig.add_axes([0.39, 0.02, 0.1, 0.04])
        ax_zoom_in = self.fig.add_axes([0.51, 0.02, 0.08, 0.04])
        ax_zoom_out = self.fig.add_axes([0.61, 0.02, 0.08, 0.04])
        ax_save = self.fig.add_axes([0.73, 0.02, 0.1, 0.04])
        
        btn_prev = Button(ax_prev, '← Previous', color='lightblue', hovercolor='skyblue')
        btn_next = Button(ax_next, 'Next →', color='lightblue', hovercolor='skyblue')
        btn_random = Button(ax_random, '🎲 Random', color='lightyellow', hovercolor='yellow')
        btn_zoom_in = Button(ax_zoom_in, 'Zoom In', color='lightgreen', hovercolor='limegreen')
        btn_zoom_out = Button(ax_zoom_out, 'Zoom Out', color='lightcoral', hovercolor='salmon')
        btn_save = Button(ax_save, '💾 Save', color='lightgray', hovercolor='silver')
        
        def on_prev(event):
            self.start_idx = max(0, self.start_idx - self.window_size // 2)
            self._update_all_plots()
        
        def on_next(event):
            # Find the max length across all appliances
            max_len = 0
            for appliance in APPLIANCES:
                if self.data[appliance]['real'] is not None:
                    max_len = max(max_len, len(self.data[appliance]['real']))
                if self.data[appliance]['synthetic'] is not None:
                    max_len = max(max_len, len(self.data[appliance]['synthetic']))
            
            self.start_idx = min(max_len - self.window_size, self.start_idx + self.window_size // 2)
            self.start_idx = max(0, self.start_idx)
            self._update_all_plots()
        
        def on_random(event):
            # Find the min length across all appliances (to ensure valid data for all)
            min_len = float('inf')
            for appliance in APPLIANCES:
                if self.data[appliance]['real'] is not None:
                    min_len = min(min_len, len(self.data[appliance]['real']))
                if self.data[appliance]['synthetic'] is not None:
                    min_len = min(min_len, len(self.data[appliance]['synthetic']))
            
            if min_len > self.window_size:
                self.start_idx = random.randint(0, int(min_len) - self.window_size)
                self._update_all_plots()
        
        def on_zoom_in(event):
            self.window_size = max(60, self.window_size // 2)  # Min 1 hour
            self._update_all_plots()
        
        def on_zoom_out(event):
            self.window_size = min(2880, self.window_size * 2)  # Max 2 days
            self._update_all_plots()
        
        def on_save(event):
            filename = f"all_appliances_comparison_{self.start_idx}_{self.window_size}.png"
            save_path = os.path.join(BASE_DIR, "Data Quality Checking", filename)
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        btn_random.on_clicked(on_random)
        btn_zoom_in.on_clicked(on_zoom_in)
        btn_zoom_out.on_clicked(on_zoom_out)
        btn_save.on_clicked(on_save)
        
        # Store button references to prevent garbage collection
        self.buttons = [btn_prev, btn_next, btn_random, btn_zoom_in, btn_zoom_out, btn_save]
    
    def _update_all_plots(self):
        """Update all subplot plots with current settings."""
        for i, (appliance, label) in enumerate(zip(APPLIANCES, APPLIANCE_LABELS)):
            ax = self.axes[i]
            ax.clear()
            
            # Get data for this appliance
            real_data = self.data[appliance]['real']
            synth_data = self.data[appliance]['synthetic']
            
            power_col = appliance
            
            real_window = self.get_window_data(real_data, self.start_idx, self.window_size)
            synth_window = self.get_window_data(synth_data, self.start_idx, self.window_size)
            
            # Plot both waveforms overlaid
            if real_window is not None and len(real_window) > 0:
                real_power = real_window[power_col].values
                time_idx = np.arange(len(real_power))
                ax.plot(time_idx, real_power, color='red', linewidth=1.0, 
                       label='Real', alpha=0.9)
            
            if synth_window is not None and len(synth_window) > 0:
                synth_power = synth_window[power_col].values
                time_idx = np.arange(len(synth_power))
                ax.plot(time_idx, synth_power, color='blue', linewidth=1.0, 
                       label='Synthetic', alpha=0.8)
            
            # Style the subplot
            ax.set_xlabel('Time (samples)', fontsize=9)
            ax.set_ylabel('Power', fontsize=9)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(0, self.window_size)
        
        # Update title
        self.fig.suptitle(f'Real vs Synthetic Waveform Comparison - All Appliances\n'
                         f'(Samples {self.start_idx} to {self.start_idx + self.window_size})', 
                         fontsize=14, fontweight='bold')
        
        self.fig.canvas.draw_idle()


def main():
    """Main function to run the all-appliances viewer."""
    print("\n" + "=" * 60)
    print("   NILM Waveform Viewer - ALL APPLIANCES")
    print("   Real vs Synthetic Data Comparison (Overlaid)")
    print("=" * 60)
    
    viewer = AllAppliancesViewer()
    viewer.load_all_data()
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        try:
            viewer.start_idx = int(sys.argv[1])
        except ValueError:
            pass
    
    if len(sys.argv) > 2:
        try:
            viewer.window_size = int(sys.argv[2])
        except ValueError:
            pass
    
    print(f"\nStarting viewer with:")
    print(f"  - Start Index: {viewer.start_idx}")
    print(f"  - Window Size: {viewer.window_size} samples")
    print("\nUse the buttons to navigate, zoom, and save figures.")
    
    viewer.plot_all_appliances()


if __name__ == "__main__":
    main()
