# Slide: Inference Stage (Dual-Path Sampling)

## 1. Process Overview
*   **Dual Forward Pass**: For every single timestep $t$, the model runs **twice**:
    *   **Path A (Conditional)**: Input includes real time features (e.g., Hour=13).
    *   **Path B (Unconditional)**: Input has **ALL** time features masked to **-9.0**.

## 2. The Logic
*   **The "Difference" ($\Delta$)**:
    *   $\Delta = \epsilon_{cond} - \epsilon_{uncond}$
    *   This vector acts as the **"Direction of Intent"**.
    *   It points away from "Generic Noise" towards "Specific Patterns" (e.g., Peaks).

## 3. The Extrapolation Formula
*   **Mathematical Core**:
    *   $\text{Output} = \text{Uncond} + s \times (\text{Cond} - \text{Uncond})$
*   **Role of $s$ (Guidance Scale)**:
    *   **$s=1$**: Standard conditional generation.
    *   **$s>1$** (e.g., 2.0 - 3.0): **Amplifies** the specific time probability.
    *    Forces the model to prioritize the *Condition* over the *Base Probability*.

## 4. Physical Meaning
*   **$\epsilon_{uncond}$**: Represents the **Background / Noise Floor**.
*   **$\epsilon_{cond}$**: Represents **Background + Signal**.
*   **Subtraction**: Mathematically cancels out the background, isolating the **Pure Signal**.
