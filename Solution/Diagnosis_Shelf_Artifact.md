# Diagnosis: Sampling "Shelf" Artifact (The Noisy Plateau)

**Symptom**: The waveform shows sharp, correct peaks (Steps 10240-10700) followed by a sudden "Shelf" of continuous noise at ~300W (Steps 10800+).

## 1. The Likely Culprit: OOD Extrapolation (The "Drifting" Effect)
When we use Classifier-Free Guidance (CFG) with a high scale ($w > 1$), we are **extrapolating** outside the training distribution.
$$ \text{Output} = \text{Uncond} + w \times (\text{Cond} - \text{Uncond}) $$

Sometimes, this linear extrapolation pushes the value into a "forbidden zone" where the model's behavior becomes undefined (Out-Of-Distribution).
*   **The "Shelf"**: It looks like the model got "stuck" in a high-energy state and couldn't diffuse back down to zero. This is a common phenomenon in diffusion when the guidance scale is too high or the "Unconditional" baseline is unstable.

## 2. Check: Did you Retrain? (Critical)
**You introduced massive changes:**
1.  **HTE Block**: Changed the neural network shape (4 MLPs instead of 1).
2.  **-9.0 Token**: Changed the input distribution geometry.

**If you did not retrain from scratch (Epoch 0):**
*   The model weights are completely mismatched.
*   The "Shelf" is the model hallucinating because it doesn't understand the new embeddings.
*   **Action**: You MUST retrain. The old weights cannot handle HTE or -9.0.

## 3. Check: Is the Guidance Scale ($w$) too high?
If you *did* retrain:
*   A scale of `w=5.0` or `w=7.0` might be too aggressive, causing this instability.
*   **Action**: Try lowering guidance to `w=2.0` or `w=1.5`.
    *   If the Shelf disappears, the scale was the problem.
    *   If the Shelf stays, the Training is the problem.

## 4. Check: The "Dynamic Thresholding"
In `gaussian_diffusion.py`:
```python
s = p_guided.abs().flatten(1).max(dim=1)[0]
x_start = p_guided / s
```
This logic clamps high peaks. However, if the "Unconditional" path predicts a strong negative value (e.g. -0.5) and Conditional is 0, the difference is Positive. Multiplied by $w$, it creates a "Phantom Positive".
*   This suggests the **-9.0 Null Token** might not have been fully learned as "Zero Power" yet.

**Recommendation:**
1.  **Confirm Retraining**: Ensure you ran `run_diffusion_all` with `--train`.
2.  **Lower Guidance**: Sample with `guidance_scale=2.0`.
