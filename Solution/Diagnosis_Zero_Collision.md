# Diagnosis: The "Zero Collision" & Feature Blurring
**Why the Waveform Disappeared and How We Fixed It**

## 1. The Suspect: "Zero" in Current & Voltage
In our encoding, we use Sine and Cosine to represent time (Cyclical Features).
*   **Noon (12:00)**: $\sin(\pi) \approx \mathbf{0}$.
*   **Midnight (00:00)**: $\sin(0) = \mathbf{0}$.
*   **6:00 AM/PM**: $\cos(\pi/2) = \mathbf{0}$.

### The Crime: "Null Token Collision"
When we implemented Classifier-Free Guidance (CFG), we needed a "Null Token" to represent **"I don't know the time"**.
*   **Old Strategy**: Set feature = `0.0`.
*   **The Conflict**:
    *   **Case A (Noon)**: Input is `[0, -1]`.
    *   **Case B (Null)**: Input is `[0, 0]`.
    *   **The Problem**: Half the data (the sine component) is IDENTICAL.
    *   **Result**: The model gets confused. "Is this input 0 because it's Noon, or because the sensor is off?"

### The Consequence: Vanishing Guidance
Recall the magic formula:
$$ \text{Guidance} = s \times (\epsilon_{\text{Noon}} - \epsilon_{\text{Null}}) $$

If "Noon" looks like "Null" (because of the zeros), then:
$$ \epsilon_{\text{Noon}} \approx \epsilon_{\text{Null}} $$
$$ \text{Guidance} \approx s \times 0 = \mathbf{0} $$

The force that is supposed to create the peak **vanishes**. The model defaults to the background noise (Unconditional), which has no peaks. **The waveform disappears.**

---

## 2. The Solution: Two-Pronged Attack

### Part A: The Geometric Fix (-9.0)
We changed the Null Token to **-9.0**.
*   **New Logic**:
    *   **Noon**: `[0, -1]` (Valid Unit Circle point)
    *   **Null**: `[-9, -9]` (Far away in space)
*   **Why it works**: -9.0 is mathematically impossible for a Sine/Cos. The model effectively sees a "Giant Red Flag" saying "THIS IS A NULL TOKEN".
*   **Result**: $\epsilon_{\text{Null}}$ is now totally different from $\epsilon_{\text{Noon}}$. The difference vector becomes huge. **Guidance Force is restored.**

### Part B: The "New Block" (HTE - Hierarchical Temporal Encoding)
Even with the guidance force restored, we had a secondary problem: **Noise**.
*   **Old Architecture**: All times (Minute, Hour, Day, Month) were mixed into one bucket.
*   **The Risk**: The model might pay attention to "Month" (which is huge and slow) and ignore "Minute" (which is fast and twitchy).

**The HTE Fix**:
We built specialized "Lanes" (The New Blocks).
1.  **Minute Expert**: Only looks at Minutes.
2.  **Hour Expert**: Only looks at Hours.
3.  **Day Expert**: Only looks at Days.

**How it helps**:
The "Hour Expert" cannot be distracted by the Month. It focuses 100% on the **13:00** signal. It outputs a strong, clean embedding that says "IT IS 13:00".
When we subtract the Null token, we get a **pure, sharp time signal**, not a muddy mix of dates.

---

## Summary
1.  **Zero Collision**: `0.0` masked the signal because valid time also contains `0.0`.
2.  **Fix (-9.0)**: Moved the mask to an impossible location, restoring contrast.
3.  **HTE Block**: Isolated the "Hour" signal to ensure the contrast is sharp and specific to the daily cycle.
