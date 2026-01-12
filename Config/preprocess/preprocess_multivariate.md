# Preprocessing Configuration Documentation

This document details the configuration parameters found in `Config/preprocess/preprocess_multivariate.yaml` and their usage across the data processing pipeline.

## 1. Normalization Formulas

These parameters define how raw power values (Watts) are normalized into the Z-score format required for training.

### Aggregate (Mains) Normalization
Used to normalize the total household power consumption.

*   **Config Keys**: `normalization.aggregate_mean`, `normalization.aggregate_std`
*   **Formula**:
    $$Z_{agg} = \frac{P_{agg} - \mu_{agg}}{\sigma_{agg}}$$
    Where:
    *   $P_{agg}$: Raw aggregate power (Watts)
    *   $\mu_{agg}$: `aggregate_mean` (522)
    *   $\sigma_{agg}$: `aggregate_std` (814)

### Appliance Normalization (Standard)
Each appliance has its own specific statistics for Z-score normalization.

*   **Config Keys**: `appliances.<name>.mean`, `appliances.<name>.std`
*   **Formula**:
    $$Z_{app} = \frac{P_{app} - \mu_{app}}{\sigma_{app}}$$
    Where:
    *   $P_{app}$: Raw appliance power (Watts)
    *   $\mu_{app}$: Appliance `mean` (e.g., Kettle=700)
    *   $\sigma_{app}$: Appliance `std` (e.g., Kettle=1000)

### MinMax Normalization (for Algorithm 1)
Algorithm 1 uses a [0, 1] range for filtering relative events.

*   **Config Key**: `appliances.<name>.max_power`
*   **Formula**:
    $$P_{norm} = \frac{P_{app}}{P_{max}}$$
    Where:
    *   $P_{max}$: `max_power` (e.g., Kettle=3998)

---

## 2. Algorithm 1 Parameters (Data Filtering)

These parameters control `Data_filtering/algorithm1_v2_multivariate.py`, which selects "active" windows for training.

### Spike Removal
Removes high-frequency noise spikes before processing.

*   **Config Keys**: 
    *   `spike_window`: Size of the sliding window (e.g., 5 samples)
    *   `spike_threshold`: Condition for spike detection
*   **Condition**:
    $$P_t > \text{median}(W_t) \times \text{threshold}$$

### Event Detection
Determines if an appliance is "ON".

*   **Config Key**: `appliances.<name>.on_power_threshold`
*   **Condition**:
    $$Status = \begin{cases} \text{ON}, & \text{if } P_{app} \ge \text{threshold} \\ \text{OFF}, & \text{if } P_{app} < \text{threshold} \end{cases}$$

### Clipping
Hard caps power values to prevent outliers.

*   **Config Key**: `appliances.<name>.max_power_clip`
*   **Formula**:
    $$P_{clipped} = \min(P_{app}, \text{clip\_max})$$

---

## 3. Data Mixing Parameters

These parameters control `mix_training_data_multivariate.py`, which combines real and synthetic data.

### Synthetic Denormalization
Synthetic data is generated in [0, 1] range and must be converted to Z-score.

**Scenario A: Clipped Appliance (clip < real_max)**
The model learned a clipped range, so we upscale to the clip limit first.
1.  **Scale UP**: $P_{watts} = P_{syn} \times \text{clip\_max}$
2.  **Normalize**: $Z_{syn} = \frac{P_{watts} - \mu_{app}}{\sigma_{app}}$

**Scenario B: Unclipped Appliance**
The model learned the full range. We map [0,1] directly to the real data Z-score range.
1.  **Map Range**:
    $$Z_{syn} = P_{syn} \times (Z_{max} - Z_{min}) + Z_{min}$$
    *   $Z_{min}, Z_{max}$: Actual min/max Z-scores observed in real training data.

### Windowing & Shuffling
Controls how the continuous time-series is broken into chunks for training.

*   **`window_size` (600)**:
    *   The continuous data is sliced into non-overlapping windows of 600 samples (10 hours at 1-min resolution).
    *   Formula: `num_windows = total_rows // 600`
*   **`shuffle` (True)**:
    *   The order of these windows is randomized to break temporal correlations and mix real/synthetic data uniformly.

---

## 4. Dataset Splitting Strategy

Defines how the original UK-DALE data is divided.

### Training Set (`training`)
*   **Files**: `multivariate_ukdale_preprocess_training+validating.py`
*   **Houses**: `appliances.<name>.train.houses` (e.g., [1, 3, 5])
*   **Logic**: Uses 100% of data from these houses for training (unless `validation_percent` > 0).

### Testing Set (`testing`)
*   **Files**: `multivariate_ukdale_preprocess_testing.py`
*   **Houses**: `appliances.<name>.test.houses` (e.g., [2])
*   **Logic**: Uses 100% of data (`testing_percent: 100`) from the unseen house (House 2) for testing.
