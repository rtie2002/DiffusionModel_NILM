# Classifier-Free Guidance (CFG) for NILM Temporal Conditioning

This document explains the mathematical and logical framework used to control time-series generation via Classifier-Free Guidance (CFG) in the DiffusionModel_NILM project.

## 1. The Core Formula

In our implementation, CFG does not change the model architecture; instead, it modifies the inference trajectory. At each denoising step $t$, we compute two predictions:

1.  **Conditional Prediction ($\epsilon_{cond}$):** The model's output when provided with valid time features (Minute, Hour, Day of Week, Month).
2.  **Unconditional Prediction ($\epsilon_{uncond}$):** The model's output when time features are masked (set to 0).

The guided prediction $\tilde{\epsilon}$ is calculated as:

$$\tilde{\epsilon} = \epsilon_{uncond} + s \cdot (\epsilon_{cond} - \epsilon_{uncond})$$

Where **$s$** is the `guidance_scale`.

## 2. Why it Controls "Time"

The term $(\epsilon_{cond} - \epsilon_{uncond})$ represents the **pure gradient of the time features**. 

*   **$\epsilon_{uncond}$ (The Base):** This represents the general "NILM style" memoized by the model. Since morning/evening peaks are dominant in the training set, the unconditional model naturally gravitates toward these high-probability structures.
*   **$\epsilon_{cond}$ (The Intent):** This is the model's attempt to reconcile the noise with a specific time label (e.g., 13:00 PM).
*   **Amplification ($s$):** By increasing $s > 1.0$, we forcefully amplify the difference caused by the time label. It "pushes" the generation away from the generic global distribution and toward the specific temporal local distribution.

### Real-world Example: Finding the Afternoon Peak
If the real data shows a peak at 13:00, but only with 10% frequency:
- At `guidance_scale = 1.0`, the model might generate a flat line or noise because the "Morning/Evening Bias" is too strong.
- At `guidance_scale = 5.0`, the model is forced to interpret the 13:00 label as a command to generate power. The weak 13:00 signal is multiplied by 5, effectively "lifting" it above the **Algorithm 1 filtering threshold** (e.g., 200W).

## 3. Implementation Details

In `Models/diffusion/gaussian_diffusion.py`, the `model_predictions` function implements this logic:

```python
# 1. Standard Forward (with Time Features)
x_start_cond = self.output(x, t, padding_masks)

# 2. Null Forward (Time Features = 0)
x_uncond = x.clone()
x_uncond[:, :, self.feature_size:] = 0.0 # Clear minute/hour/dow/month
x_start_uncond = self.output(x_uncond, t, padding_masks)

# 3. Dynamic Gain Calculation
w = guidance_scale - 1.0
p_guided = p_cond + w * (p_cond - p_uncond) 
```

## 4. Interaction with Sampling Modes

### DDPM (Standard Mode)
CFG is applied at every single step of the 1000/2000 step chain. This provides the most precise alignment but is computationally expensive (requires 2x forward passes per step).

### DDIM (Fast Mode)
CFG is integrated into the deterministic ODE trajectory. Even with only 50 steps, the `guidance_scale` acts as a steering force for the ODE solver, ensuring that the 50-step "shortcuts" still land at the correct temporal peaks.

## 5. Summary of Scale ($s$) Effects

| Scale | Effect | Use Case |
| :--- | :--- | :--- |
| **1.0** | No Guidance | Standard model behavior; prone to "Morning/Evening Bias". |
| **1.5 - 3.0** | Moderate Guidance | Balanced generation; good for general distributions. |
| **5.0 - 7.0** | **Strong Guidance** | **Best for finding elusive peaks (Afternoon peak)**. Improves temporal alignment at the cost of slight waveform rigidity. |

---
*Document generated for DiffusionModel_NILM Sampling Optimization.*
