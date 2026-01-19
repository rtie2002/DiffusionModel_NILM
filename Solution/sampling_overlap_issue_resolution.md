# Sampling Overlap Problem & Resolution

## 1. The Problem: "Time Conditioning Not Working"
**Symptom**:
The user observed that the generated synthetic data appeared to be "stuck" in specific time periods (e.g., only containing data for December or January), despite the model being trained on the full year's dataset. This led to the initial suspicion that the time-conditioning mechanism (Sin/Cos features for Month, Day, Hour) was broken.

**Root Cause Analysis**:
The issue was **not** with the model training or the conditioning logic, but with the **Sampling Strategy**.

The original `main.py` sampling logic, when set to `ordered=True` (Sequential), used a **Sliding Window** approach with a default stride of `1`.
*   **Total Data Points**: ~1,200,000 minutes (~2.4 years).
*   **Window Size**: 512 minutes.
*   **Total Available Sliding Windows**: ~1,200,000 (Window 0=[0-512], Window 1=[1-513], ...).

When the user requested a specific number of samples (e.g., `2500` windows) to form a dataset:
*   The script generated indices: `0, 1, 2, ..., 2499`.
*   **Time Coverage**: These 2500 windows only covered the first `2500 + 512` minutes of the data (approx. **2 days**).
*   **Result**: The synthetic dataset successfully reproduced 2 days of data effectively, but naturally, the month and day did not change over just 48 hours. This created the illusion that "Time Conditioning" failed.

## 2. The Solution: Non-Overlapping "Unrolled" Sampling
To generate a synthetic dataset that is a **statistical twin** of the original (covering the same duration, seasons, and days), the sampling must be **Non-Overlapping**.

**New Logic Implemented**:
We introduced a new sampling mode: **`--sampling_mode ordered_non_overlapping`**.

### Mathematical Change
**Old Logic (Sliding/Overlapping)**:
$$ Index_i = Start + i \times 1 $$
*Coverage per $N$ samples $\approx N$ minutes.*

**New Logic (Strided/Non-Overlapping)**:
$$ Index_i = Start + i \times WindowSize $$
*Coverage per $N$ samples $\approx N \times 512$ minutes.*

### Implementation Details
1.  **`main.py`**:
    *   Added `ordered_non_overlapping` logic.
    *   Automatically sets `stride = dataset.window` (512).
    *   Automatically calculates the required `num_samples` to cover the entire file: $N = \lceil \frac{TotalPoints}{WindowSize} \rceil$.

2.  **`solver.py`**:
    *   Modified `trainer.sample()` to accept a `stride` parameter.
    *   Indices are now generated as `[(i * stride) % dataset_size]`.

### Outcome
*   **100% Coverage**: The sampler now jumps 512 minutes for every step. Generating ~2500 samples now covers the entire 1.2 million minutes of the original dataset.
*   **No Redundancy**: Generated windows share zero overlap, eliminating data leakage for downstream training.
*   **Correct Distribution**: Since every timestamp from the real history is used exactly once as a condition, the synthetic data possesses the exact same seasonal (Month) and weekly (Day of Week) distribution as the real data.

## 3. Handling >100% Data (e.g., 200%)
When the user requests **200%** of the original data size:
1.  The loop wraps around the dataset length using the modulo operator `% dataset_size`.
2.  **Pass 1**: Generates indices $0, 512, 1024, \dots$ (Full Timeline, V1).
3.  **Pass 2**: Generates indices $0, 512, 1024, \dots$ (Full Timeline, V2).
4.  ** Diversity**: Although the time conditions are identical for Pass 1 and Pass 2, the **random noise initialization** ($\epsilon \sim \mathcal{N}(0, I)$) is different. This results in two distinct variations of power profiles for the exact same historical moments, effectively serving as robust **Data Augmentation**.
