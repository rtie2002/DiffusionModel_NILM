# Evaluation Tools

This folder contains tools for validating the quality and performance of generated synthetic data from the diffusion model.

## Available Tools

### 1. `visualize_synthetic_data.py`
Comprehensive visual quality assessment for synthetic appliance power data.

**Features:**
- Time series comparison (real vs synthetic)
- Power distribution histograms
- Statistical metrics (mean, std, min, max)
- Side-by-side visualization

**Usage:**
```bash
python visualize_synthetic_data.py --appliance <appliance_name>
```

**Supported appliances:**
- `kettle`
- `microwave`
- `fridge`
- `dishwasher`
- `washingmachine`

**Example:**
```bash
python visualize_synthetic_data.py --appliance dishwasher
```

**Output:**
- 4-panel comparison chart
- Saved to `OUTPUT/visualizations/{appliance}_visualization.png`

## Requirements

Before running, ensure you have:
1. Generated synthetic data using `main.py`
2. Synthetic data saved in `OUTPUT/{appliance}_512/ddpm_fake_{appliance}_512.npy`
3. Real data in `Data/datasets/{appliance}.csv`

## Purpose

This folder centralizes all data quality validation tools to assess:
- ✅ Distribution similarity
- ✅ Statistical properties match
- ✅ Time series pattern quality
- ✅ Overall synthetic data fidelity
