# Preprocessing Configuration and Normalization

## Normalization Parameters
The `preprocess_multivariate.yaml` configuration file defines the following global normalization constants for the aggregate (mains) power readings:

```yaml
normalization:
  aggregate_mean: 522
  aggregate_std: 814
```

These specific values (522 and 814) are derived from the statistics of the UK-DALE dataset used in the NILM-main project.

## Mathematical Formula
The normalization process uses **Z-score normalization** (also known as standardization). This technique rescales the data distribution so that the mean of the observed values is 0 and the standard deviation is 1.

The formula is:

$$
z = \frac{x - \mu}{\sigma}
$$

Where:
*   $z$ is the normalized output value fed into the model.
*   $x$ is the raw power reading (in Watts).
*   $\mu$ is the population mean (`aggregate_mean` = 522).
*   $\sigma$ is the population standard deviation (`aggregate_std` = 814).

## Implementation
In the preprocessing scripts (`multivariate_ukdale_preprocess_*.py`), this formula is applied directly to the aggregate column:

```python
# Normalization step in code
df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
```

---

# Algorithm 1: Data Filtering & Selection

The `Data_filtering/algorithm1_v2_multivariate.py` script implements the data selection logic described in the Diffusion Model framework. Its primary goal is to **filter out "silent" periods** (where appliances are OFF) to improve the training efficiency of the generative model, while preserving effective activation windows.

## Workflow Overview

The process consists of 5 main steps:
1.  **Denormalization**: Convert Z-score inputs back to Real Power (Watts).
2.  **Spike Removal**: Clean isolated noise/sensor glitches.
3.  **Window Selection (Algorithm 1)**: Select time windows around ON events.
4.  **Clipping**: Cap power values at a predefined maximum.
5.  **Re-Normalization**: Apply MinMax scaling [0, 1] for the Diffusion Model.

## Detailed Steps & Formulas

### Step 1: Denormalization
The input data comes from `ukdale_preprocess` which uses Z-score normalization. To perform physical threshold logic (e.g., "ON if > 50 Watts"), we must first reverse this.

$$ x_{watts} = z \cdot \sigma_{app} + \mu_{app} $$

Where $\mu_{app}$ and $\sigma_{app}$ are the appliance-specific mean and std defined in `preprocess_multivariate.yaml`.

### Step 2: Spike Removal (Optional)
To prevent triggering selection windows on sensor errors, isolated spikes are removed.
A point $x_t$ is considered a spike if:
1.  **Background is Low**: The surrounding window (excluding $x_t$) is mostly (< 60%) below `background_threshold` (50W).
2.  **Magnitude is High**: $x_t > \text{spike\_threshold} \times \text{median}(\text{neighbors})$.

If detected, $x_t \leftarrow 0$.

### Step 3: Event-Based Selection (Algorithm 1)
This is the core logic. We construct a set of indices $T_{selected}$ to keep.

1.  **Identify ON Events**: Find all time indices $t$ where power exceeds the activation threshold:
    $$ T_{start} = \{ t \mid x_t \ge \text{on\_power\_threshold} \} $$
    
2.  **Apply Windowing**: For every trigger point $t \in T_{start}$, select a surrounding window of size $L$ (`window_length`, typically 10):
    $$ [ t - L, \quad t + L ] $$
    
3.  **Union Indices**: The final selected dataset is the union of all such windows:
    $$ T_{selected} = \bigcup_{t \in T_{start}} \{ t-L, \dots, t+L \} $$

This effectively discards long periods of inactivity while capturing the startup, steady state, and shutdown transients of the appliance.

### Step 4: Outlier Clipping
To prevent extreme outliers from skewing the [0, 1] normalization, values are hard-capped.

$$ x_{clipped} = \min(x_{selected}, \text{max\_power\_clip}) $$

*   `max_power_clip` is defined per appliance in the config (e.g., Fridge = 300W).

### Step 5: MinMax Normalization
The Diffusion Model requires inputs strictly in the range [0, 1].

$$ x_{norm} = \frac{x_{clipped}}{\text{max\_power}} $$

*   **Note**: `max_power` is a fixed constant per appliance (e.g., 2000W), NOT the maximum of the current batch. This ensures consistent scaling across different dataset splits.
*   Values are clamped to ensure $x_{norm} \in [0, 1]$.

## Configurable Parameters
All parameters are managed in `Config/preprocess/preprocess_multivariate.yaml`.

| Parameter | Description | Source Section |
| :--- | :--- | :--- |
| `window_length` | Samples to keep before/after event | `algorithm1` |
| `on_power_threshold` | Wattage to trigger selection | `appliances.<name>` |
| `max_power_clip` | Hard cap in Watts (Step 4) | `appliances.<name>` |
| `max_power` | Denominator for MinMax (Step 5) | `appliances.<name>` |
