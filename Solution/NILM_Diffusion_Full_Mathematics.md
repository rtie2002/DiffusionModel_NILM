# Comprehensive Mathematical Documentation for NILM Diffusion Model

**Version**: 2.1 (Expanded Technical Bible)
**Scope**: Thorough listing of all mathematical formulas, process flows, and architectural mechanisms used in the `DiffusionModel_NILM` project.
**Target Audience**: Researchers and Advanced Engineers requiring complete reproducibility.

---

# Table of Contents

1.  [Forward Diffusion Process](#1-forward-diffusion-process)
    *   [Beta Schedules](#11-beta-schedules)
    *   [Transition Kernels](#12-transition-kernels)
2.  [Model Architecture: The Diffusion Transformer (DiT)](#2-model-architecture-the-diffusion-transformer-dit)
    *   [Input Decoupling](#21-input-decoupling)
    *   [Hierarchical Temporal Encoding (HTE)](#22-hierarchical-temporal-encoding-hte)
    *   [Adaptive Layer Normalization (AdaLN-Zero)](#23-adaptive-layer-normalization-adaln-zero)
    *   [Agent Attention Mechanism](#24-agent-attention-mechanism)
3.  [Training Dynamics & Objective Function](#3-training-dynamics--objective-function)
    *   [The CFG Training Mechanism (Bernoulli Masking)](#31-the-cfg-training-mechanism-bernoulli-masking)
    *   [Loss Function Formulation (Time + Frequency)](#32-loss-function-formulation-time--frequency)
4.  [Inference & Sampling Dynamics](#4-inference--sampling-dynamics)
    *   [The Crucial Difference: Training vs. Sampling](#41-the-crucial-difference-training-vs-sampling)
    *   [Classifier-Free Guidance (CFG) Sampling Formula](#42-classifier-free-guidance-cfg-sampling-formula)
    *   [The Signal Direction & -9.0 Logic](#43-the-signal-direction-and--90-logic)
    *   [DDPM Sampling (Markovian)](#44-ddpm-sampling-markovian)
    *   [DDIM Fast Sampling (Non-Markovian)](#45-ddim-fast-sampling-non-markovian)
    *   [Inpainting logic (Langevin Dynamics)](#46-inpainting-logic-langevin-dynamics)

---

# 1. Forward Diffusion Process

The diffusion process transforms the structured data distribution $x_0$ into a standard Gaussian noise distribution $x_T \sim \mathcal{N}(0, I)$ over $T$ steps.

### 1.1 Beta Schedules
We define a noise schedule $\beta_t$ that controls the variance added at each step.

**Cosine Schedule (Default):**
To prevent abrupt changes in noise levels, we use a cosine-based schedule (Nichol & Dhariwal, 2021).
$$ f(t) = \cos^2 \left( \frac{t/T + s}{1+s} \cdot \frac{\pi}{2} \right) $$
$$ \bar{\alpha}_t = \frac{f(t)}{f(0)} $$
$$ \beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} $$
*Constraints*: $\beta_t$ is clipped to be no larger than 0.999.

**Linear Schedule (Alternative):**
$$ \beta_t = \text{Linear}(10^{-4}, 0.02, t) $$

### 1.2 Transition Kernels
Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.
The probability of $x_t$ given $x_0$ is given by:
$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I}) $$

**Sample Generation (Re-parameterization Trick):**
$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$
where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.

---

# 2. Model Architecture: The Diffusion Transformer (DiT)

The backbone of our generation model is a specialized Transformer adapted for diffusion processes, featuring **Hierarchical Temporal Encoding (HTE)** and **AdaLN-Zero**.

### 2.1 Input Decoupling
The input tensor $X \in \mathbb{R}^{B \times L \times 9}$ is explicitly split into content and condition for processing.
$$ X_{\text{power}} = X[:, :, 0] \in \mathbb{R}^{B \times L \times 1} $$
$$ X_{\text{cond}} = X[:, :, 1:9] \in \mathbb{R}^{B \times L \times 8} $$

### 2.2 Hierarchical Temporal Encoding (HTE)
Instead of a flat projection, the 8-dimensional time features are sliced and processed by specialized "Expert" MLPs to preserve multi-scale fidelity.

**Input Slicing:**
*   $x_{min} = X_{\text{cond}}[:, :, 0:2]$ (Minute sin/cos)
*   $x_{hr} = X_{\text{cond}}[:, :, 2:4]$ (Hour sin/cos)
*   $x_{dow} = X_{\text{cond}}[:, :, 4:6]$ (Day of Week sin/cos)
*   $x_{mo} = X_{\text{cond}}[:, :, 6:8]$ (Month sin/cos)

**Expert Processing:**
Each expert $E_k$ is a Multi-Layer Perceptron (Linear $\to$ SiLU $\to$ Linear):
$$ h_{min} = E_{min}(x_{min}) \in \mathbb{R}^{D/4} $$
$$ h_{hr} = E_{hr}(x_{hr}) \in \mathbb{R}^{D/4} $$
$$ h_{dow} = E_{dow}(x_{dow}) \in \mathbb{R}^{D/4} $$
$$ h_{mo} = E_{mo}(x_{mo}) \in \mathbb{R}^{D/4} $$

**Feature Fusion:**
$$ L_{emb} = \text{Concat}(h_{min}, h_{hr}, h_{dow}, h_{mo}) \cdot W_{proj} $$
This ensures that the "Hour" features (crucial for peak detection) have a dedicated, non-interfering pathway into the model's core.

### 2.3 Adaptive Layer Normalization (AdaLN-Zero)
This is the mechanism that injects the time condition into the network. It replaces standard LayerNorm. The condition $c$ (which includes discrete diffusion time step embedding $t_{emb}$ and our HTE features $L_{emb}$) modulates the statistics of layer activations.

**Modulation Parameter Regression:**
$$ \text{emb} = \text{SiLU}(\text{Linear}(t_{emb} + L_{emb})) $$
$$ [\gamma, \beta, \alpha] = \text{Chunk}(\text{Linear}(\text{emb}), 3) $$
Note: The final Linear layer is initialized with **zeros**, meaning $\gamma, \beta, \alpha$ start at 0. This is the **"Zero Initialization"** strategy for stability.

**Normalization Equation:**
For an input $x$ to a layer:
$$ \text{AdaLN}(x, c) = \alpha \cdot \left( \text{LayerNorm}(x) \cdot (1 + \gamma) + \beta \right) $$

*   $\gamma$ (Scaling): Stretches the distribution based on time (e.g., amplifying peaks at 13:00).
*   $\beta$ (Shifting): Moves the mean (e.g., lifting the baseline).
*   $\alpha$ (Gating): Controls how much this block influences the residual stream.

### 2.4 Agent Attention Mechanism
To handle long sequences efficiently, we use an Agent-based attention mechanism combined with Flash Attention.
$$ A = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V $$
Implemented as:
1.  **Agent Pool**: Queries interact with a small set of "Agent Tokens" (Summarization).
2.  **Broadcast**: Agents broadcast information back to the original sequence.
Formula:
$$ \text{Context} = \text{Attention}(Q, K_{agents}, V_{agents}) $$

---

# 3. Training Dynamics & Objective Function

This section details how the model is trained to support CFG. The critical concept is that **CFG requires no change to the Loss Function architecture**, but it fundamentally alters the **Data Distribution** seen by the model during training.

### 3.1 The CFG Training Mechanism (Bernoulli Masking)
During training, we do not calculate "differences" or "gradients". Instead, we simply expose the model to two types of examples: conditioned and unconditioned.

For every batch, we generate a random mask vector $m \in \{0, 1\}$ sampled from a Bernoulli distribution with parameter $(1 - p_{drop})$, where $p_{drop}=0.1$.

**The Conditional Transformation Formula:**
$$ \tilde{c} = m \cdot c + (1 - m) \cdot \emptyset $$
Where $\emptyset = \mathbf{-9.0}$ (The Null Token).

*   **Case 1 (Mask=1, 90% chance)**: The model sees the true time $c$ (e.g., Hour=13).
    *   $\text{Input} = [x_t, \text{Hour}=13]$
    *   **Objective**: Learn $p(x_t | \text{Hour}=13)$.
*   **Case 2 (Mask=0, 10% chance)**: The model sees the Null Token.
    *   $\text{Input} = [x_t, \emptyset]$
    *   **Objective**: Learn the marginal distribution $p(x_t)$, effectively the "average appliance behavior" without knowing the time.

**Mathematical Significance**:
By training on this mixture, we effectively learn a **Joint Estimator** $\epsilon_\theta(x_t, t, \tilde{c})$ that can switch between a conditional estimator and an unconditional estimator based entirely on the input value of $\tilde{c}$.

### 3.2 Loss Function Formulation (Time + Frequency)

The objective is to minimize the error between the true noise $\epsilon$ and the predicted noise $\epsilon_\theta$.

**Time-Domain Huber Loss:**
Used for robustness against outliers (spikes).
$$ \mathcal{L}_{time} = \mathbb{E}_{x, c, \epsilon, t, m} \left[ \begin{cases} 
0.5 ||\epsilon - \epsilon_\theta(x_t, t, \tilde{c})||^2 & \text{if } \text{error} < \delta \\
\delta (||\epsilon - \epsilon_\theta|| - 0.5 \delta) & \text{otherwise}
\end{cases} \right] $$
where $\delta = 0.5$.

**Frequency-Domain Fourier Loss:**
To ensure the generated signal has the correct spectral characteristics.
$$ \mathcal{L}_{freq} = || \text{Re}(\mathcal{F}(x)) - \text{Re}(\mathcal{F}(y)) ||^2 + || \text{Im}(\mathcal{F}(x)) - \text{Im}(\mathcal{F}(y)) ||^2 $$

**Total Optimization Goal:**
$$ \min_\theta \left( \mathcal{L}_{time} + \lambda_{ff} \cdot \mathcal{L}_{freq} \right) $$

---

# 4. Inference & Sampling Dynamics

This is where the magic happens. While training was passive (just learning to predict), sampling is **active** (manipulating the prediction) using Classifier-Free Guidance.

### 4.1 The Crucial Difference: Training vs. Sampling

| Feature | **Training Stage** | **Sampling Stage (CFG)** |
| :--- | :--- | :--- |
| **Logic** | **Stochastic Masking** | **Deterministic Linear Extrapolation** |
| **Formula** | $\epsilon \approx \epsilon_\theta(x_t, \tilde{c})$ | $\tilde{\epsilon} = \epsilon_{uncond} + s(\epsilon_{cond} - \epsilon_{uncond})$ |
| **Input** | Either $c$ **OR** $\emptyset$ (Mutually Exclusive) | Both $c$ **AND** $\emptyset$ (Simultaneous Calculation) |
| **Goal** | Learn two separate distributions: $p(x|c)$ and $p(x)$ | **Combine** them to create a new "super-conditional" distribution |

### 4.2 Classifier-Free Guidance (CFG) Sampling Formula
Unlike training, where we input one condition or the other, in sampling we perform **two forward passes** for the *same* noisy input $x_t$ at *every* timestep.

1.  **Conditional Pass**: $\epsilon_{cond} = \epsilon_\theta(x_t, t, c)$
2.  **Unconditional (Null) Pass**: $\epsilon_{uncond} = \epsilon_\theta(x_t, t, \emptyset)$

**The Extrapolation Formula:**
$$ \tilde{\epsilon}_\theta(x_t, c) = (1 + s) \cdot \epsilon_\theta(x_t, c) - s \cdot \epsilon_\theta(x_t, \emptyset) $$
*Rearranging gives the more intuitive "Correction" form:*
$$ \tilde{\epsilon}_\theta(x_t, c) = \epsilon_{uncond} + \underbrace{(s + 1)}_{\text{Effective Scale}} \cdot (\epsilon_{cond} - \epsilon_{uncond}) $$
*(Note: Implementations vary on whether scale is $w$ or $w+1$, our code uses $p_{guided} = p_{cond} + (s-1)(p_{cond} - p_{uncond})$ which implies base $cond$ plus boosted difference)*.

**Our Exact Implementation Formula (`gaussian_diffusion.py`):**
$$ \tilde{x}_{start} = x_{start}^{cond} + (s - 1.0) \cdot (x_{start}^{cond} - x_{start}^{uncond}) $$
*Note: We apply guidance in $x_{start}$ space (reconstructed data) rather than $\epsilon$ space (noise) for stability, but the mathematical principle of linear extrapolation is identical.*

### 4.3 The Signal Direction & -9.0 Logic
The term $\Delta = (x_{start}^{cond} - x_{start}^{uncond})$ represents the **Semantic Gradient vector**.

*   **Geometric Interpretation**: In the high-dimensional latent space, $\epsilon_{uncond}$ points to the "center of mass" of all appliance data (mostly zero/flat). $\epsilon_{cond}$ points to the specific cluster for "13:00 PM".
*   **The Difference Vector**: The vector $\vec{v} = \epsilon_{cond} - \epsilon_{uncond}$ points *away* from the generic "average" and *towards* the "specific".
*   **The -9.0 Multiplier**: Because we use `-9.0` (a geometrically distant point) instead of `0.0` (a valid point), the magnitude $||\vec{v}||$ is naturally larger, providing a robust, non-vanishing gradient even when the signal is weak.

### 4.4 DDPM Sampling (Markovian)
Standard sampling follows the reverse Markov chain:
$$ x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \tilde{\epsilon}_\theta \right) + \sigma_t z $$
where $z \sim \mathcal{N}(0, I)$.
This is accurate but slow (1000-2000 steps).

### 4.5 DDIM Fast Sampling (Non-Markovian)
We use the implicit ODE formulation for acceleration (50 steps).
$$ x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t}\tilde{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\alpha_{t-1} - \sigma_t^2} \cdot \tilde{\epsilon}_\theta + \sigma_t z $$
**Quadratic Striding**: We select timesteps using a quadratic function $t_i = \lfloor (i/S)^2 \cdot T \rfloor$ to focus more sampling steps near $t=0$ where fine details (texture, high-freq peaks) are formed.

### 4.6 Inpainting logic (Langevin Dynamics)
For filling missing data:
$$ x_{t-1} = x_{t-1}^{\text{known}} \odot M + x_{t-1}^{\text{sampled}} \odot (1-M) $$
Refined by $K$ steps of Langevin correction:
$$ x_{t-1} \leftarrow x_{t-1} - \frac{\lambda}{2} \nabla_x || y - M \odot x ||^2 + \sqrt{\lambda} \xi $$

---
*Document prepared for DiffusionModel_NILM Technical Reference.*
