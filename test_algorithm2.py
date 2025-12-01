"""
Test Example for Algorithm 2 Post-Processing

This example creates synthetic noisy data and demonstrates
how Algorithm 2 smooths the signal while preserving transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from algorithm2_postprocessing import algorithm2_postprocessing

# Create synthetic appliance power data with noise
np.random.seed(42)
time_steps = 1000

# Simulate washing machine cycle
# OFF (0-100), ON (100-500), OFF (500-600), ON (600-900), OFF (900-1000)
power_clean = np.zeros(time_steps)
power_clean[100:500] = 400  # First cycle
power_clean[600:900] = 450  # Second cycle

# Add Gaussian noise
noise_std = 30
noise = np.random.normal(0, noise_std, time_steps)
power_noisy = power_clean + noise
power_noisy = np.maximum(power_noisy, 0)  # Ensure non-negative

# Apply Algorithm 2
threshold = 20  # Washing machine threshold (20W)
alpha = 0.5     # Smoothing coefficient

print("=" * 60)
print("Algorithm 2 Test Example")
print("=" * 60)
print(f"\nSimulated Data:")
print(f"  Time steps: {time_steps}")
print(f"  Noise std: {noise_std} W")
print(f"  Threshold: {threshold} W")
print(f"  Smoothing α: {alpha}")

# Test different alpha values
alphas = [0.3, 0.5, 0.7]
results = {}

for a in alphas:
    s = algorithm2_postprocessing(power_noisy, threshold, alpha=a)
    results[a] = s
    noise_reduction = (1 - s.std() / power_noisy.std()) * 100
    print(f"\nα = {a}:")
    print(f"  Original Std: {power_noisy.std():.2f} W")
    print(f"  Processed Std: {s.std():.2f} W")
    print(f"  Noise Reduction: {noise_reduction:.1f}%")

# Visualize
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle('Algorithm 2 Test: Effect of Different α Values', 
             fontsize=16, fontweight='bold')

# Plot 1: Original clean signal
ax1 = axes[0]
ax1.plot(power_clean, color='green', linewidth=2, label='Clean Signal (Ground Truth)')
ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold ({threshold}W)')
ax1.set_title('Ground Truth (No Noise)', fontweight='bold')
ax1.set_ylabel('Power (W)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Noisy signal
ax2 = axes[1]
ax2.plot(power_noisy, color='red', linewidth=0.5, alpha=0.7, label='Noisy Signal')
ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold ({threshold}W)')
ax2.set_title(f'Noisy Signal (σ = {noise_std}W)', fontweight='bold')
ax2.set_ylabel('Power (W)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Comparison of different α values
ax3 = axes[2]
ax3.plot(power_noisy, color='red', linewidth=0.5, alpha=0.3, label='Noisy')
ax3.plot(results[0.3], color='blue', linewidth=1.5, alpha=0.8, label='α=0.3 (More smoothing)')
ax3.plot(results[0.5], color='orange', linewidth=1.5, alpha=0.8, label='α=0.5 (Medium)')
ax3.plot(results[0.7], color='purple', linewidth=1.5, alpha=0.8, label='α=0.7 (Less smoothing)')
ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_title('Comparison of Different α Values', fontweight='bold')
ax3.set_ylabel('Power (W)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Detail view
ax4 = axes[3]
start, end = 80, 130
ax4.plot(range(start, end), power_clean[start:end], color='green', linewidth=2, 
         marker='o', markersize=3, label='Ground Truth')
ax4.plot(range(start, end), power_noisy[start:end], color='red', linewidth=0.8, 
         alpha=0.5, marker='x', markersize=4, label='Noisy')
ax4.plot(range(start, end), results[0.5][start:end], color='blue', linewidth=2, 
         marker='s', markersize=3, label='Post-processed (α=0.5)')
ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold')
ax4.set_title(f'Detail View: ON Transition (Samples {start}-{end})', fontweight='bold')
ax4.set_xlabel('Time Index')
ax4.set_ylabel('Power (W)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('algorithm2_test_example.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved: algorithm2_test_example.png")
plt.show()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)

# Key observations
print("\nKey Observations:")
print("1. Smaller α (e.g., 0.3) = More smoothing, slower response")
print("2. Larger α (e.g., 0.7) = Less smoothing, faster response")
print("3. Algorithm preserves ON/OFF transitions (no lag at startup)")
print("4. Noise is reduced during active periods")
