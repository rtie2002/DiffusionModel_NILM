# Classifier-Free Guidance (CFG) in NILM: A Mathematical and Architectural Thesis

**Document ID**: NILM-CFG-THESIS-001
**Topic**: Deep Dive into Classifier-Free Guidance Implementation for Sparse Energy Disaggregation
**Focus**: Mathematics, Code Implementation, Pipeline Flows, and Novel Modifications

---

## 1. Abstract

This document serves as a comprehensive technical reference for the implementation of Classifier-Free Guidance (CFG) within the Diffusion Transformer (DiT) architecture for Non-Intrusive Load Monitoring (NILM). It details the transition from standard conditional diffusion to a guidance-based generation framework designed to recover high-frequency, sparse electrical signatures.

---

## 2. Mathematical Framework

### 2.1 The Score Matching Objective
Diffusion models aim to learn the score function of the data distribution, defined as $\nabla_x \log p(x)$. In a conditional setting (given time $c$), we aim to learn $\nabla_x \log p(x|c)$.

Using Bayes' rule:
$$ p(x|c) = \frac{p(c|x) p(x)}{p(c)} $$
Taking the logarithm and gradient with respect to $x$:
$$ \nabla_x \log p(x|c) = \nabla_x \log p(c|x) + \nabla_x \log p(x) $$

This equation reveals that the **conditional score** is the sum of the **unconditional score** ($\nabla_x \log p(x)$) and the **classifier gradient** ($\nabla_x \log p(c|x)$).

### 2.2 Derivation of Classifier-Free Guidance
Ho & Salimans (2021) proposed that we do not need an explicit classifier to estimate $\nabla_x \log p(c|x)$. Instead, we can train a single neural network $\epsilon_\theta(x, c)$ to approximate the conditional score, and train the *same* network to approximate the unconditional score $\epsilon_\theta(x, \emptyset)$ by masking $c$ with a null token $\emptyset$.

Substituting the scores into the guidance equation:
$$ \tilde{\epsilon}_\theta(x, c) = (1 + w) \epsilon_\theta(x, c) - w \epsilon_\theta(x, \emptyset) $$
This captures the gradient direction:
$$ \tilde{\epsilon}_\theta(x, c) = \epsilon_\theta(x, \emptyset) + (1 + w) (\epsilon_\theta(x, c) - \epsilon_\theta(x, \emptyset)) $$

*   **$\epsilon_\theta(x, \emptyset)$**: The unconditional noise prediction.
*   **$(\epsilon_\theta(x, c) - \epsilon_\theta(x, \emptyset))$**: The conditional gradient direction (implicit classifier).
*   **$w$**: The guidance strength.

---

## 3. Physical Interpretation: The "Sculptor" Analogy (Revised)

To understand the **Sampling Process** correctly, imagine a **Sculptor (The Model)** refining a statue out of a block of stone (Noise).

**The Loop (What happens at every step $t$):**

1.  **The Consultations**:
    *   **Consultant A (Unconditional)**: "Based on the average stone shape, remove this much noise." This tends to produce safe, generic forms (mean-reversion).
    *   **Consultant B (Conditional)**: "Considering this is specifically for 13:00 PM, remove noise differently."

2.  **The Gradient Calculation (The "Diff")**:
    *   **Difference = Consultant B - Consultant A**.
    *   This is NOT "The Signal". This is the **Direction of Adjustment**.
    *   If positive: "The condition implies LESS noise here (Stronger Signal)."
    *   If negative: "The condition implies MORE noise here (Weaker Signal)."

3.  **The Guidance (The Amplification)**:
    *   **Final Action = Consultant A + $w \times$ Difference**.
    *   We tell the sculptor: "Look at how Consultant B differs from A. Now, do that **$w$ times harder**."
    *   This forces the result to be simpler and more aligned with the condition than the model would naturally dare to make it.

---

## 4. Pipeline & Architectural Changes

The implementation of CFG required modifications across three distinct stages of the pipeline: Model Architecture, Training Loop, and Inference Loop.

### 4.1 Architecture: The Diffusion Transformer (DiT)

**Location**: `Models/diffusion/agent_transformer.py`

We utilize a Transformer-based backbone where temporal conditions are injected via **Adaptive Layer Normalization (AdaLN-Zero)**.

#### 4.1.1 Hierarchical Temporal Encoding (HTE)
Standard embeddings treat time as a flat vector. We modified the `cond_emb_mlp` to support valid expert separation.

**Code Change (`Transformer.__init__`)**:
```python
# SPLIT: 8 features -> 4 specialized paths
self.minute_mlp = make_small_mlp(2, chunk_dim)  # Expert 1
self.hour_mlp   = make_small_mlp(2, chunk_dim)  # Expert 2 (Crucial for 13:00 Peak)
self.dow_mlp    = make_small_mlp(2, chunk_dim)  # Expert 3
self.month_mlp  = make_small_mlp(2, chunk_dim)  # Expert 4
```

#### 4.1.2 AdaLN-Zero Modulation
**Location**: `Models.diffusion.model_utils.AdaLayerNorm`

The condition $c$ is projected to predict the scaling parameters for every layer.
$$ \text{AdaLN}(x, c) = \alpha \cdot (\text{LayerNorm}(x) \cdot (1 + \gamma(c)) + \beta(c)) $$
*   **Zero-Initialization**: The projection layers are initialized to zero, ensuring the model starts training as an identity mapper.

---

### 4.2 Training Pipeline: The Joint optimization

**Location**: `Models/diffusion/gaussian_diffusion.py` -> `_train_loss`

To enable the model to perform both conditional and unconditional prediction, we implement **Stochastic Conditioning**.

#### 4.2.1 The Bernoulli Mask Mechanism
We introduce a hyperparameter `cond_drop_prob = 0.1`.

**Code Implementation**:
```python
if self.cond_drop_prob > 0:
    keep_mask = torch.bernoulli(torch.ones(B, 1, 1) * (1 - self.cond_drop_prob))
    # CRITICAL CHANGE: Using -9.0 instead of 0.0
    x_condition = torch.where(keep_mask.bool(), x_condition, torch.full_like(x_condition, -9.0))
```

#### 4.2.2 The "-9.0" Geographic Separation Rationale
*   **Conflict**: If we masked with `0.0`, the model would confuse "Noon" ($12:00 \to \sin \approx 0$) with "No Condition".
*   **Solution**: We use `-9.0`. Since $\sin(t) \in [-1, 1]$, a value of `-9.0` is geometrically impossible, preventing ambiguity.

---

### 4.3 Inference Pipeline: Dual-Path Sampling

**Location**: `Models/diffusion/gaussian_diffusion.py` -> `model_predictions`

#### 4.3.1 The Extrapolation
We apply the guidance scale $w$ (typically 2.0 - 3.0 for NILM).

**Code Implementation**:
```python
# The Extrapolation
diff = pred_cond - pred_uncond
pred_guided = pred_uncond + w * diff 
```

---

## 5. Critical Analysis: Capabilities & Limitations

It is crucial to understand the scientific boundaries of CFG for NILM.

### 5.1 What CFG CAN Do (The Solution Space)
| Problem | Mechanism | Prerequisite |
| :--- | :--- | :--- |
| **Weak Conditional Signal** | Amplifies $(\epsilon_{cond} - \epsilon_{uncond})$ to force generation to strictly follow the condition. | "13:00 $\to$ High Load" samples must exist in training data (>5%). |
| **Low Semantic Alignment** | Forces the model to be more "confident" in the discrete time labels. | Valid feature encoding (HTE). |

### 5.2 What CFG CANNOT Do (The Limitation Space)
| Problem | Reason | Correct Solution |
| :--- | :--- | :--- |
| **Extremely Low ON-Ratio** | If the model never learned "13:00 means ON", masking won't help. $(\epsilon_{cond} - \epsilon_{uncond}) \approx 0$. | **Loss Weighting** (3-5x on peaks) or **Resampling**. |
| **Amplitude Collapse** | MSE Loss biases towards the conditional mean (e.g., 200W instead of 1000W). | **Resampling** over-represented OFF states. |
| **Transient Detail Loss** | Not a guidance issue, but a sampling resolution issue. | Increase diffusion steps or use **DPM-Solver++**. |

### 5.3 Guidance Scale Recommendation
For Time-Series data, unlike Image Generation:
*   **Do NOT use $w > 4.0$**: This tends to break the temporal autocorrelation structure, creating unrealistic jagged waves.
*   **Recommended Range**: $w \in [2.0, 3.0]$. This provides sufficient boost for peak recovery without destroying the waveform smoothness.

---

## 6. Thesis Conclusion

CFG effectively acts as a **Contrastive Amplifier**. By learning the difference between "I know the time" and "I don't know the time", the model isolates the *gradient direction* required to satisfy the time condition. During inference, we extrapolate along this gradient. Crucially, strictly abiding by the conditions requires that the model has seen enough examples during training to form a valid conditional gradient; CFG cannot create information that was never learned.
