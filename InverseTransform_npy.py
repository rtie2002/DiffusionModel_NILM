import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Appliance power specifications from ukdale_processing.py
APPLIANCE_SPECS = {
    'kettle': {'max_power': 3998},
    'microwave': {'max_power': 3969},
    'fridge': {'max_power': 350},
    'dishwasher': {'max_power': 3964},
    'washingmachine': {'max_power': 3999}
}

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Denormalize synthetic appliance data from [0,1] to watts')
parser.add_argument('--appliance', type=str, default='kettle',
                    choices=['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine'],
                    help='Appliance name (default: kettle)')
args = parser.parse_args()

# Configuration
applianceName = args.appliance
max_power = APPLIANCE_SPECS[applianceName]['max_power']
print(f'\n=== Denormalizing {applianceName} data ===')
print(f'Max power: {max_power}W')

# Load generated NPY file (normalized [0, 1])
generated_npy = np.load(f'OUTPUT/{applianceName}_512/ddpm_fake_{applianceName}_512.npy')
flattened_data = generated_npy.reshape(-1, 1)

# Simple denormalization: [0, 1] → [0, max_power] watts
denormalized_watts = flattened_data * max_power

# Load original normalized data for comparison
df_original = pd.read_csv(f'Data/datasets/{applianceName}.csv')
original_normalized = df_original['power'].values.reshape(-1, 1)
original_watts = original_normalized * max_power

# Plot comparison
arr1 = denormalized_watts.flatten()
arr2 = original_watts.flatten()

fig, axs = plt.subplots(2, figsize=(8, 8))
axs[0].plot(arr1[:20000], linestyle='-', color='b', label='Generated')
axs[0].set_title(f'Generated {applianceName}')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Power (Watts)')
axs[0].legend()

axs[1].plot(arr2[:20000], linestyle='-', color='r', label='Original')
axs[1].set_title(f'Original {applianceName}')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Power (Watts)')
axs[1].legend()

plt.tight_layout()
plt.show()

# Save denormalized data
df_output = pd.DataFrame(arr1, columns=['power'])
df_output.to_csv(f'OUTPUT/{applianceName}_512/{applianceName}_denormalized.csv', index=False)

print(f'✓ Denormalized data saved to OUTPUT/{applianceName}_512/{applianceName}_denormalized.csv')
print(f'✓ Data range: {arr1.min():.2f}W to {arr1.max():.2f}W')
print(f'✓ Expected max: {max_power}W')
