"""
Optimized Temporal Data Viewer with GUI Filters
优化版带GUI过滤器的时间数据查看器

Performance optimizations:
- Downsample data for faster plotting
- Reduce redraw frequency
- Simplified GUI layout
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox
from tkinter import Tk, filedialog
import sys

class TemporalDataViewer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df_original = None
        self.df_filtered = None
        self.max_plot_points = 5000  # Limit for performance
        
        # Filter state
        self.filters = {
            'minute': [0, 59],
            'hour': [0, 23],
            'day': [1, 31],
            'month': [1, 12]
        }
        
        # Load data
        self.load_data()
        
        # Create GUI
        self.create_gui()
    
    def load_data(self):
        """Load CSV data"""
        print(f"\nLoading data from: {self.file_path}")
        self.df_original = pd.read_csv(
            self.file_path,
            header=None,
            names=['aggregate', 'appliance', 'minute', 'hour', 'day', 'month']
        )
        print(f"✓ Loaded {len(self.df_original):,} rows")
        
        # Set filter ranges based on actual data
        self.filters['minute'] = [int(self.df_original['minute'].min()), int(self.df_original['minute'].max())]
        self.filters['hour'] = [int(self.df_original['hour'].min()), int(self.df_original['hour'].max())]
        self.filters['day'] = [int(self.df_original['day'].min()), int(self.df_original['day'].max())]
        self.filters['month'] = [int(self.df_original['month'].min()), int(self.df_original['month'].max())]
        
        self.df_filtered = self.df_original.copy()
    
    def create_gui(self):
        """Create optimized GUI"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(14, 8))
        self.fig.suptitle('Temporal Data Viewer - Use text boxes to filter, then click Apply', 
                         fontsize=12, fontweight='bold')
        
        # Flatten axes for easier access
        self.ax_agg = self.axes[0, 0]
        self.ax_app = self.axes[0, 1]
        self.ax_info = self.axes[0, 2]
        self.ax_minute = self.axes[1, 0]
        self.ax_hour = self.axes[1, 1]
        self.ax_controls = self.axes[1, 2]
        
        # Hide info and controls axes
        self.ax_info.axis('off')
        self.ax_controls.axis('off')
        
        # Create simple filter controls
        self.create_controls()
        
        # Initial plot
        self.update_plots()
        
        plt.tight_layout()
        plt.show()
    
    def create_controls(self):
        """Create simplified filter controls"""
        # Control positions
        y_start = 0.85
        y_step = 0.15
        
        # Minute filter
        y = y_start
        self.ax_controls.text(0.05, y, 'Minute:', fontsize=9, weight='bold')
        ax_min_min = plt.axes([0.72, y - 0.05, 0.08, 0.03])
        ax_min_max = plt.axes([0.82, y - 0.05, 0.08, 0.03])
        self.txt_min_min = TextBox(ax_min_min, '', initial=str(self.filters['minute'][0]))
        self.txt_min_max = TextBox(ax_min_max, '', initial=str(self.filters['minute'][1]))
        
        # Hour filter
        y -= y_step
        self.ax_controls.text(0.05, y, 'Hour:', fontsize=9, weight='bold')
        ax_hour_min = plt.axes([0.72, y - 0.05, 0.08, 0.03])
        ax_hour_max = plt.axes([0.82, y - 0.05, 0.08, 0.03])
        self.txt_hour_min = TextBox(ax_hour_min, '', initial=str(self.filters['hour'][0]))
        self.txt_hour_max = TextBox(ax_hour_max, '', initial=str(self.filters['hour'][1]))
        
        # Day filter
        y -= y_step
        self.ax_controls.text(0.05, y, 'Day:', fontsize=9, weight='bold')
        ax_day_min = plt.axes([0.72, y - 0.05, 0.08, 0.03])
        ax_day_max = plt.axes([0.82, y - 0.05, 0.08, 0.03])
        self.txt_day_min = TextBox(ax_day_min, '', initial=str(self.filters['day'][0]))
        self.txt_day_max = TextBox(ax_day_max, '', initial=str(self.filters['day'][1]))
        
        # Month filter
        y -= y_step
        self.ax_controls.text(0.05, y, 'Month:', fontsize=9, weight='bold')
        ax_month_min = plt.axes([0.72, y - 0.05, 0.08, 0.03])
        ax_month_max = plt.axes([0.82, y - 0.05, 0.08, 0.03])
        self.txt_month_min = TextBox(ax_month_min, '', initial=str(self.filters['month'][0]))
        self.txt_month_max = TextBox(ax_month_max, '', initial=str(self.filters['month'][1]))
        
        # Apply button
        y -= y_step + 0.05
        ax_apply = plt.axes([0.72, y - 0.05, 0.18, 0.05])
        self.btn_apply = Button(ax_apply, 'Apply Filters', color='lightblue')
        self.btn_apply.on_clicked(self.apply_filters)
        
        # Reset button
        y -= 0.08
        ax_reset = plt.axes([0.72, y - 0.05, 0.18, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset All', color='lightcoral')
        self.btn_reset.on_clicked(self.reset_filters)
        
        # Info text
        self.info_text = self.ax_info.text(0.05, 0.95, '', fontsize=9, 
                                           verticalalignment='top', family='monospace')
        self.update_info()
    
    def apply_filters(self, event=None):
        """Apply filters - only called when button clicked"""
        try:
            # Get filter values
            min_min = int(self.txt_min_min.text)
            min_max = int(self.txt_min_max.text)
            hour_min = int(self.txt_hour_min.text)
            hour_max = int(self.txt_hour_max.text)
            day_min = int(self.txt_day_min.text)
            day_max = int(self.txt_day_max.text)
            month_min = int(self.txt_month_min.text)
            month_max = int(self.txt_month_max.text)
            
            print(f"\nApplying filters...")
            print(f"  Minute: {min_min}-{min_max}")
            print(f"  Hour: {hour_min}-{hour_max}")
            print(f"  Day: {day_min}-{day_max}")
            print(f"  Month: {month_min}-{month_max}")
            
            # Apply filters
            self.df_filtered = self.df_original[
                (self.df_original['minute'] >= min_min) & (self.df_original['minute'] <= min_max) &
                (self.df_original['hour'] >= hour_min) & (self.df_original['hour'] <= hour_max) &
                (self.df_original['day'] >= day_min) & (self.df_original['day'] <= day_max) &
                (self.df_original['month'] >= month_min) & (self.df_original['month'] <= month_max)
            ]
            
            print(f"  Result: {len(self.df_filtered):,} rows")
            
            # Update plots
            self.update_plots()
            self.update_info()
            
        except ValueError as e:
            print(f"✗ Invalid filter value: {e}")
    
    def reset_filters(self, event):
        """Reset all filters"""
        self.txt_min_min.set_val(str(self.filters['minute'][0]))
        self.txt_min_max.set_val(str(self.filters['minute'][1]))
        self.txt_hour_min.set_val(str(self.filters['hour'][0]))
        self.txt_hour_max.set_val(str(self.filters['hour'][1]))
        self.txt_day_min.set_val(str(self.filters['day'][0]))
        self.txt_day_max.set_val(str(self.filters['day'][1]))
        self.txt_month_min.set_val(str(self.filters['month'][0]))
        self.txt_month_max.set_val(str(self.filters['month'][1]))
        
        self.df_filtered = self.df_original.copy()
        self.update_plots()
        self.update_info()
        print("\n✓ Filters reset")
    
    def update_info(self):
        """Update info text"""
        total = len(self.df_original)
        filtered = len(self.df_filtered)
        pct = (filtered / total * 100) if total > 0 else 0
        
        info = f"Filtered: {filtered:,} / {total:,}\n"
        info += f"({pct:.1f}%)\n\n"
        
        if len(self.df_filtered) > 0:
            info += f"Aggregate:\n"
            info += f"  mean: {self.df_filtered['aggregate'].mean():.3f}\n"
            info += f"  std:  {self.df_filtered['aggregate'].std():.3f}\n\n"
            info += f"Appliance:\n"
            info += f"  mean: {self.df_filtered['appliance'].mean():.3f}\n"
            info += f"  std:  {self.df_filtered['appliance'].std():.3f}\n"
        else:
            info += "No data"
        
        self.info_text.set_text(info)
    
    def downsample_data(self, data, max_points):
        """Downsample data for faster plotting"""
        if len(data) <= max_points:
            return data
        
        # Use every nth point
        step = len(data) // max_points
        return data[::step]
    
    def update_plots(self):
        """Update all plots with downsampled data"""
        if len(self.df_filtered) == 0:
            print("✗ No data to plot!")
            return
        
        # Downsample for time series plots
        df_plot = self.df_filtered.iloc[::max(1, len(self.df_filtered) // self.max_plot_points)]
        
        print(f"  Plotting {len(df_plot):,} points (downsampled from {len(self.df_filtered):,})")
        
        # Clear axes
        self.ax_agg.clear()
        self.ax_app.clear()
        self.ax_minute.clear()
        self.ax_hour.clear()
        
        # Plot aggregate power
        self.ax_agg.plot(df_plot['aggregate'].values, linewidth=0.5, color='blue', alpha=0.7)
        self.ax_agg.set_title('Aggregate Power', fontsize=10)
        self.ax_agg.set_ylabel('Power', fontsize=8)
        self.ax_agg.grid(True, alpha=0.3)
        self.ax_agg.tick_params(labelsize=8)
        
        # Plot appliance power
        self.ax_app.plot(df_plot['appliance'].values, linewidth=0.5, color='red', alpha=0.7)
        self.ax_app.set_title('Appliance Power', fontsize=10)
        self.ax_app.set_ylabel('Power', fontsize=8)
        self.ax_app.grid(True, alpha=0.3)
        self.ax_app.tick_params(labelsize=8)
        
        # Plot minute distribution
        self.ax_minute.hist(self.df_filtered['minute'], bins=60, color='green', 
                           alpha=0.7, edgecolor='black', linewidth=0.5)
        self.ax_minute.set_title('Minute Distribution', fontsize=10)
        self.ax_minute.set_xlabel('Minute', fontsize=8)
        self.ax_minute.tick_params(labelsize=8)
        
        # Plot hour distribution
        self.ax_hour.hist(self.df_filtered['hour'], bins=24, color='orange', 
                         alpha=0.7, edgecolor='black', linewidth=0.5)
        self.ax_hour.set_title('Hour Distribution', fontsize=10)
        self.ax_hour.set_xlabel('Hour', fontsize=8)
        self.ax_hour.tick_params(labelsize=8)
        
        # Redraw
        self.fig.canvas.draw_idle()
        print("✓ Plots updated")

def select_csv_file():
    """Open file dialog to select CSV file"""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Multivariate CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir="created_data/UK_DALE/"
    )
    root.destroy()
    return file_path

def main():
    print("="*80)
    print("TEMPORAL DATA VIEWER - Optimized Version")
    print("="*80)
    print("\nPerformance tips:")
    print("  - Data is downsampled to 5000 points for faster plotting")
    print("  - Click 'Apply Filters' button to update (not automatic)")
    print("  - Use smaller data ranges for faster response")
    
    # Select file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = select_csv_file()
    
    if not file_path:
        print("\n✗ No file selected. Exiting...")
        return
    
    # Create viewer
    try:
        viewer = TemporalDataViewer(file_path)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
