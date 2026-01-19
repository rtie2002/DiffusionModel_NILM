# Conditional Generation via Guided Subspace Diffusion

## 1. Introduction

In the context of Non-Intrusive Load Monitoring (NILM) synthetic data generation, preserving the temporal consistency of appliance activation is critical. Standard Denoising Diffusion Probabilistic Models (DDPMs) generate data from a pure Gaussian noise distribution $\mathcal{N}(0, I)$, which often leads to temporally incoherent samples where appliances may activate at physically impossible times (e.g., a solar panel generating power at midnight).

To resolve this, we implement a **Conditional Subspace Diffusion** mechanism, often referred to in literature as "Replacement-based Conditional Sampling" or "Inpainting-guided Diffusion". Unlike Classifier-Free Guidance (CFG) which requires training separate conditional embeddings, our approach modifies the diffusion kinematic structure itself by treating the time features as **observed variables** separate from the **latent variables** (power).

## 2. Mathematical Formulation

### 2.1. State Space Definition

We define our multivariate time-series sample $\mathbf{x} \in \mathbb{R}^{L \times C}$ where $L$ is the sequence length (512) and $C$ is the channel dimension. We decompose the channel space $C$ into two disjoint subspaces: the **target subspace** $\mathcal{X}_p$ (Power) and the **conditional subspace** $\mathcal{X}_c$ (Time).

Let $\mathbf{x}_0$ be the concatenation of the target variable $\mathbf{x}^p$ and the conditional variable $\mathbf{x}^c$:

$$
\mathbf{x}_0 = [\mathbf{x}^p_0 \parallel \mathbf{x}^c]
$$

Where:
*   $\mathbf{x}^p \in \mathbb{R}^{L \times 1}$: The appliance power consumption (to be generated).
*   $\mathbf{x}^c \in \mathbb{R}^{L \times 8}$: The temporal encoding vectors (fixed ground truth).

### 2.2. The Modified Forward Process (Diffusion)

In visual DDPMs, the forward process $q(\mathbf{x}_t | \mathbf{x}_0)$ adds Gaussian noise to the entire object. In our conditional framework, we define a **Masked Forward Process**. We enforce that the conditional information $\mathbf{x}^c$ remains fully observed (clean) throughout the diffusion chain $t=1 \dots T$.

The transition kernel is defined as:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t^p; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}^p, \beta_t \mathbf{I}) \otimes \delta(\mathbf{x}_t^c - \mathbf{x}^c)
$$

This implies that while the power component diffuses towards randomness, the time component remains constant:

$$
\mathbf{x}_t = [\underbrace{\sqrt{\bar{\alpha}_t}\mathbf{x}_0^p + \sqrt{1-\bar{\alpha}_t}\epsilon}_{\text{Noisy Power}} \parallel \underbrace{\mathbf{x}^c}_{\text{Clean TimeFeatures}}]
$$

### 2.3. The Guided Reverse Process (Sampling)

The reverse process seeks to approximate the posterior $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$. Standard sampling predicts the mean $\mu_\theta(\mathbf{x}_t, t)$ and variance $\Sigma_\theta(\mathbf{x}_t, t)$.

To enforce the condition $\mathbf{c}$, we apply a **Hard Manifold Constraint** at every sampling step. After the model predicts the denoised state $\tilde{\mathbf{x}}_{t-1}$, we manually inject the known ground truth $\mathbf{x}^c$ back into the state vector.

The update rule for timestep $t \rightarrow t-1$ becomes:

1.  **Predict**: $\hat{\mathbf{x}}_{t-1} \leftarrow \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$
2.  **Constraint**: $\mathbf{x}_{t-1} \leftarrow [\hat{\mathbf{x}}_{t-1}^p \parallel \mathbf{x}^c]$

By resetting the conditional channels $\mathbf{x}^c$ to the target time features at every step, the attention mechanism of the Transformer backbone uses $\mathbf{x}^c$ as anchors to guide the reconstruction of $\mathbf{x}^p$.

## 3. Implementation Details

### 3.1. Training Objective
The neural network $\epsilon_\theta$ receives the full 9-dimensional vector but is optimized only on the power dimension. The loss function is a partial Huber Loss computed over the target subspace:

$$
\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t)[0] \|_H \right]
$$

where $\epsilon_\theta(\mathbf{x}_t, t)[0]$ denotes the first channel of the predicted noise.

### 3.2. Vector Structure

The input vector $\mathbf{x} \in \mathbb{R}^{512 \times 9}$ is structured as follows:

| Index | Feature Name | Description | Role in Diffusion |
| :--- | :--- | :--- | :--- |
| **0** | **Power (Z-Score)** | Normalized Watts | **Latent Target** (Noised & Denoised) |
| 1 | `minute_sin` | $\sin(2\pi \cdot m / 1440)$ | Condition (Fixed) |
| 2 | `minute_cos` | $\cos(2\pi \cdot m / 1440)$ | Condition (Fixed) |
| 3 | `hour_sin` | $\sin(2\pi \cdot h / 24)$ | Condition (Fixed) |
| 4 | `hour_cos` | $\cos(2\pi \cdot h / 24)$ | Condition (Fixed) |
| 5 | `dow_sin` | $\sin(2\pi \cdot d / 7)$ | Condition (Fixed) |
| 6 | `dow_cos` | $\cos(2\pi \cdot d / 7)$ | Condition (Fixed) |
| 7 | `month_sin` | $\sin(2\pi \cdot M / 12)$ | Condition (Fixed) |
| 8 | `month_cos` | $\cos(2\pi \cdot M / 12)$ | Condition (Fixed) |

## 4. Illustrative Example

Consider generating the power profile for a *Washing Machine* specifically for **Tuesday at 09:00 AM**.

1.  **Preparation**:
    We construct the condition matrix $\mathbf{C} \in \mathbb{R}^{512 \times 8}$ derived from the timestamps starting at Tuesday 09:00 AM.
    *   $x_t[:, 1:9]$ is filled with these values.

2.  **Initialization ($t=1000$)**:
    *   $x_{1000}^p \sim \mathcal{N}(0, 1)$ (Pure random noise for power).
    *   $x_{1000}^c = \mathbf{C}$ (Perfect time data).

3.  **Denoising Step ($t=999$)**:
    *   The Transformer sees noisy power alongside clear "Tuesday 9 AM" signals.
    *   The Attention layers relate the noise to the time context. "Tuesday 9 AM" is a valid high-probability time for washer usage.
    *   The model predicts a slightly less noisy power $\tilde{x}_{999}^p$.
    *   **CRITICAL**: The process might introduce slight numerical drift to the time columns. We execute: `x[999][:, 1:] = C`.

4.  **Convergence ($t \rightarrow 0$)**:
    *   As noise is removed, the power profile takes shape. Because the time features were present and correct at every step, the model "hallucinates" a washing cycle specifically aligned with the provided time window, rather than a random hallucination.

## 5. Advantages over Concat-Conditioning

Standard concatenation conditioning (simply appending the condition to the input without replacement) often suffers from **signal washout** in diffusion models. By treating the condition as part of the data manifold but exempting it from corruption, we ensure the guidance signal remains strong ($SNR_{cond} = \infty$) even when the target signal is pure noise ($SNR_{target} \approx 0$). This results in significantly faster convergence and higher adherence to temporal constraints.
