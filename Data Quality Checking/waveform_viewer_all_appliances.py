"""
Waveform Viewer for ALL Appliances - Real vs Synthetic NILM Data

Displays all 5 appliances simultaneously in a SPLIT grid layout:
- Top Row: Real Data (Red)
- Bottom Row: Synthetic Data (Blue)

Features:
- 2x5 Grid Layout (10 plots)
- Vertical alignment for direct comparison
- Synchronized Y-axis scales for fair comparison
- Interactive navigation (Previous/Next/Random)
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
APPLIANCE_LABELS = ["Dishwasher", "Fridge", "Kettle", "Microwave", "Washing Machine"]
WINDOW_SIZE = 480  # Default window size (8 hours at 1-min resolution)


class AllAppliancesViewer:
    def __init__(self):
        self.window_size = WINDOW_SIZE
        self.start_idx = 0
        self.data = {}  # {appliance: {'real': df, 'synthetic': df}}
        self.y_limits = {}  # {appliance: max_val} to sync y-axis
        self.fig = None
        self.axes_real = []
        self.axes_synth = []
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
            max_val = 0
            
            # Load real data
            if os.path.exists(real_file):
                df = pd.read_csv(real_file)
                self.data[appliance]['real'] = df
                max_val = max(max_val, df[appliance].max())
                print(f"  {appliance:15} - Real: {len(df):>8} samples", end="")
            else:
                print(f"  {appliance:15} - Real: NOT FOUND", end="")
                
            # Load synthetic data
            if os.path.exists(synthetic_file):
                df = pd.read_csv(synthetic_file)
                self.data[appliance]['synthetic'] = df
                max_val = max(max_val, df[appliance].max())
                print(f" | Synthetic: {len(df):>8} samples")
            else:
                print(f" | Synthetic: NOT FOUND")
            
            # Store max value for y-axis scaling (add 10% buffering)
            self.y_limits[appliance] = max_val * 1.1
        
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
        """Create a 2x5 grid of plots (Top: Real, Bottom: Synthetic)."""
        # Close existing figure if any
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create figure with 2 rows x 5 columns
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.suptitle(f'Real (Top) vs Synthetic (Bottom) Comparison\n'
                         f'(Samples {self.start_idx} to {self.start_idx + self.window_size})', 
                         fontsize=16, fontweight='bold')
        
        self.axes_real = []
        self.axes_synth = []
        
        # Create subplots
        # Row 1 (Top): Real Data
        # Row 2 (Bottom): Synthetic Data
        
        for i, (appliance, label) in enumerate(zip(APPLIANCES, APPLIANCE_LABELS)):
            # Real Plot (Top Row)
            ax_real = self.fig.add_subplot(2, 5, i + 1)
            self.axes_real.append(ax_real)
            
            # Synthetic Plot (Bottom Row)
            ax_synth = self.fig.add_subplot(2, 5, i + 6)
            self.axes_synth.append(ax_synth)
            
            # Common Y-limit for this appliance
            y_max = self.y_limits.get(appliance, 1.0)
            
            # --- Plot Real Data ---
            real_data = self.data[appliance]['real']
            real_window = self.get_window_data(real_data, self.start_idx, self.window_size)
            
            if real_window is not None and len(real_window) > 0:
                vals = real_window[appliance].values
                ax_real.plot(np.arange(len(vals)), vals, color='red', linewidth=1.0)
                ax_real.text(0.05, 0.9, f"Mean: {np.mean(vals):.3f}", transform=ax_real.transAxes, fontsize=8)
            
            ax_real.set_title(f"{label}\n(Real Origin)", fontsize=10, fontweight='bold', color='darkred')
            ax_real.set_ylim(0, y_max)
            ax_real.grid(True, alpha=0.3)
            ax_real.set_xticks([]) # Hide x-ticks for top row
            
            # --- Plot Synthetic Data ---
            synth_data = self.data[appliance]['synthetic']
            synth_window = self.get_window_data(synth_data, self.start_idx, self.window_size)
            
            if synth_window is not None and len(synth_window) > 0:
                vals = synth_window[appliance].values
                ax_synth.plot(np.arange(len(vals)), vals, color='blue', linewidth=1.0)
                ax_synth.text(0.05, 0.9, f"Mean: {np.mean(vals):.3f}", transform=ax_synth.transAxes, fontsize=8)
            
            ax_synth.set_title(f"(Synthetic)", fontsize=10, fontweight='bold', color='darkblue')
            ax_synth.set_ylim(0, y_max) # Sync Y-axis with Real
            ax_synth.grid(True, alpha=0.3)
            ax_synth.set_xlabel("Time (samples)", fontsize=8)
            
            # Add Y-label only to the first column
            if i == 0:
                ax_real.set_ylabel("Power", fontsize=10)
                ax_synth.set_ylabel("Power", fontsize=10)
        
        # Add navigation buttons
        self._add_navigation_buttons()
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.92])
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
            # Find the min length across all appliances
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
            self.window_size = max(60, self.window_size // 2)
            self._update_all_plots()
        
        def on_zoom_out(event):
            self.window_size = min(2880, self.window_size * 2)
            self._update_all_plots()
        
        def on_save(event):
            filename = f"split_comparison_{self.start_idx}_{self.window_size}.png"
            save_path = os.path.join(BASE_DIR, "Data Quality Checking", filename)
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        btn_random.on_clicked(on_random)
        btn_zoom_in.on_clicked(on_zoom_in)
        btn_zoom_out.on_clicked(on_zoom_out)
        btn_save.on_clicked(on_save)
        
        # Store button references
        self.buttons = [btn_prev, btn_next, btn_random, btn_zoom_in, btn_zoom_out, btn_save]
    
    def _update_all_plots(self):
        """Update all 10 subplots."""
        for i, appliance in enumerate(APPLIANCES):
            ax_real = self.axes_real[i]
            ax_synth = self.axes_synth[i]
            
            ax_real.clear()
            ax_synth.clear()
            
            y_max = self.y_limits.get(appliance, 1.0)
            label = APPLIANCE_LABELS[i]
            
            # --- Real ---
            real_data = self.data[appliance]['real']
            real_window = self.get_window_data(real_data, self.start_idx, self.window_size)
            
            if real_window is not None and len(real_window) > 0:
                vals = real_window[appliance].values
                ax_real.plot(np.arange(len(vals)), vals, color='red', linewidth=1.0)
                ax_real.text(0.05, 0.9, f"Mean: {np.mean(vals):.3f}", transform=ax_real.transAxes, fontsize=8)
            
            ax_real.set_title(f"{label}\n(Real Origin)", fontsize=10, fontweight='bold', color='darkred')
            ax_real.set_ylim(0, y_max)
            ax_real.grid(True, alpha=0.3)
            ax_real.set_xticks([])
            
            # --- Synthetic ---
            synth_data = self.data[appliance]['synthetic']
            synth_window = self.get_window_data(synth_data, self.start_idx, self.window_size)
            
            if synth_window is not None and len(synth_window) > 0:
                vals = synth_window[appliance].values
                ax_synth.plot(np.arange(len(vals)), vals, color='blue', linewidth=1.0)
                ax_synth.text(0.05, 0.9, f"Mean: {np.mean(vals):.3f}", transform=ax_synth.transAxes, fontsize=8)
            
            ax_synth.set_title(f"(Synthetic)", fontsize=10, fontweight='bold', color='darkblue')
            ax_synth.set_ylim(0, y_max)
            ax_synth.grid(True, alpha=0.3)
            ax_synth.set_xlabel("Time (samples)", fontsize=8)
            
            if i == 0:
                ax_real.set_ylabel("Power", fontsize=10)
                ax_synth.set_ylabel("Power", fontsize=10)

        self.fig.suptitle(f'Real (Top) vs Synthetic (Bottom) Comparison\n'
                         f'(Samples {self.start_idx} to {self.start_idx + self.window_size})', 
                         fontsize=16, fontweight='bold')
        self.fig.canvas.draw_idle()


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("   NILM Waveform Viewer - SPLIT VIEW")
    print("   Real (Top) vs Synthetic (Bottom)")
    print("=" * 60)
    
    viewer = AllAppliancesViewer()
    viewer.load_all_data()
    
    # Parse command line arguments
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
    
    viewer.plot_all_appliances()


if __name__ == "__main__":
    main()
