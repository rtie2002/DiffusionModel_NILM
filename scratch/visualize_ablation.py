import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Proposed', 'w/o AdaLN', 'w/o Booster']
mae_values = [14.20, 15.64, 18.64]
mr_values = [0.413, 0.395, 0.331]

# Colors (matching the vibe of the provided image)
colors = ['#ADD8E6', '#F08080', '#90EE90'] # Light Blue, Light Coral, Light Green
edge_colors = ['#5F9EA0', '#CD5C5C', '#6B8E23']

def create_ablation_plot():
    # Set default font settings for clarity
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # --- MAE Plot ---
    bars1 = ax1.bar(labels, mae_values, color=colors, edgecolor='black', alpha=0.85, width=0.55)
    ax1.set_title('Average MAE (W)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylim(0, 22)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add horizontal dashed line for Proposed baseline
    ax1.axhline(y=mae_values[0], color='#5F9EA0', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.4,
                f'{height:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # --- MR Plot ---
    bars2 = ax2.bar(labels, mr_values, color=colors, edgecolor='black', alpha=0.85, width=0.55)
    ax2.set_title('Average Match Rate (MR)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylim(0, 0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add horizontal dashed line for Proposed baseline
    ax2.axhline(y=mr_values[0], color='#5F9EA0', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                f'{height:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # General styling
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=13)
        # Make tick labels bold for extra clarity
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('semibold')

    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=600, bbox_inches='tight')
    print("Plot saved to ablation_study_results.png with high resolution (600 DPI)")

if __name__ == "__main__":
    create_ablation_plot()
