# Comparative Analysis of Conditional Diffusion Architectures for NILM

This document provides a formal comparative analysis between the **Early Stage Concatenation-based Conditioning** (Method A) and the **Proposed Decoupled Cross-Guidance Architecture** (Method B) for multivariate NILM synthetic data generation.

---

## 1. Abstract
In the context of Non-Intrusive Load Monitoring (NILM), synthetic data must not only preserve appliances' physical characteristics (waveforms) but also strictly adhere to temporal regularities (time-of-use patterns). This report dissects the evolution from naive feature concatenation to a decoupled latent guidance framework, aiming to reach state-of-the-art precision for downstream disaggregation tasks (e.g., NILMFormer).

---

## 2. Methodology A: Early Stage Concatenation (Current)

### 2.1 The Concept
Feature concatenation, or "Channel-wise Fusion," assumes that the model can implicitly learn the relationship between noisy power signals and deterministic time features by treating them as a single $N$-dimensional vector.

### 2.2 Mathematical Formulation
$X_t = [x_t^{power}, x_t^{time\_1}, \dots, x_t^{time\_8}] \in \mathbb{R}^{B \times L \times 9}$

The denoising function $\epsilon_\theta$ is defined as:
$$\hat{\epsilon} = \epsilon_\theta(X_t, t, \tau)$$
where $X_t$ is the concatenated noisy input, $t$ is the diffusion step, and $\tau$ is the global condition.

### 2.3 Structural Flowchart
```
[ Noisy Power (1) ] --\
                       > [ Concat (9) ] --> [ Linear Embedding ] --> [ Transfromer Blocks ] --> [ Output ]
[ Time Features (8) ] --/                          ^
                                                   |
                                            [ Step Embedding ]
```

### 2.4 Critical Limitations
1. **Signal-to-Noise Ratio (SNR) Dilution**: The model attempts to denoise the entire 9D vector, even though 8 dimensions (time) are deterministic and noise-free. This wastes computational capacity.
2. **Weak Conditioning**: The relationship between time and power is additive rather than multiplicative, leading to "Temporal Drift" where the model generates waveforms at logically incorrect timestamps.

---

## 3. Methodology B: Decoupled Cross-Guidance (Proposed)

### 3.1 The Concept: "Power as Signal, Time as Context"
Methodology B treats time features as a global "Instruction Set" rather than raw data. By decoupling the conditioning, the model uses deterministic time information to modulate the power signals' generation process dynamically.

### 3.2 Architectural Innovation: AdaLN & Cross-Attention
We transition to a **Diffusion Transformer (DiT)** inspired architecture:
1. **Time Encoder**: A dedicated MLP/Transformer that extracts high-level semantic features from the 8 time features.
2. **Power Decoder**: A denoising network that strictly processes the 1D power signal.
3. **AdaLN (Adaptive Layer Norm)**: The time context modulates the mean and variance of the power features in every layer.

### 3.3 Mathematical Formulation
The conditional probability $p(x_{t-1} | x_t, c)$ is modeled where $c = \text{Encoder}(x_{time})$.

**In AdaLN-Zero Blocks:**
$$\text{Output} = \gamma(c) \cdot \text{LayerNorm}(x) + \beta(c)$$
Where $\gamma, \beta$ are scale and shift parameters dynamically predicted by the time context $c$.

### 3.4 Structural Flowchart
```
[ Time Features (8) ] --> [ Time Encoder ] ---------------------\
                                                                 |
                                                                 V (Guidance)
[ Noisy Power (1) ] --> [ Power Embedding ] --> [ Cross-Attention / AdaLN ] --> [ Denoised Output ]
      ^                                                          ^
      |__________________________________________________________|
```

---

## 4. Architectural Comparison: DiT-based Conditioning Paradigms

Following the state-of-the-art **Diffusion Transformer (DiT)** framework, three primary mechanisms exist for integrating deterministic conditions (8 time features) into the stochastic generation process.

### A. DiT Block with adaLN-Zero (Author's Recommended Method)
*   **Mechanism**: The conditioning signals (deterministic time features and diffusion timestep $t$) are processed via a Multi-Layer Perceptron (MLP) to regress six modulation parameters: $\gamma_1, \beta_1$ (for attention), $\gamma_2, \beta_2$ (for MLP), and $\alpha_1, \alpha_2$ (dimension-wise gating).
*   **Logic**: Instead of treating time as "data points," it treats time as "environmental parameters." The "Zero" signifies that the final MLP layer is initialized to zero, ensuring the block functions as an identity transformation at the start of training.
*   **Advantages for NILM**: 
    *   **Maximized Efficiency**: Sequence length remains at $L$ (e.g., 512), avoiding the $O(L^2)$ complexity spike.
    *   **Optimal Stability**: Solves the initial convergence issues common in noisy multivariate signals.
    *   **High Fidelity**: Directly modulates the "style" of the power signal based on the time context.

### B. DiT Block with Cross-Attention
*   **Mechanism**: Inserts a Multi-Head Cross-Attention layer after the Self-Attention block.
*   **Logic**: Power tokens act as the **Query (Q)**, while the Time Context (processed by an Encoder) acts as the **Key (K) and Value (V)**.
*   **Evaluation**: Strong "alignment" capability, but computationally heavier due to the additional attention operations in every Transformer block.

### C. DiT Block with In-Context Conditioning (Baseline)
*   **Mechanism**: Deterministic time features are prepended or appended to the input sequence as "condition tokens."
*   **Evaluation**: This leads to quadratic scaling of self-attention memory usage. For long NILM sequences (512+), this is the primary cause of slow training speeds (5-6 hours per appliance) and often results in the lowest generative logic accuracy.

---

## 5. Analytical Comparison Table

| Metric | Methodology A (In-Context/Concat) | Methodology B (Cross-Attention) | Methodology C (adaLN-Zero) |
| :--- | :--- | :--- | :--- |
| **Computational Complexity** | $O((L_{data}+L_{cond})^2)$ | $O(L_{data}^2 + L_{data} \cdot L_{cond})$ | **$O(L_{data}^2)$** |
| **Logic Alignment** | Implicit / Stochastic | Explicit / Attention-based | **Explicit / Modulation-based** |
| **Training Speed** | Slow (Current) | Moderate | **Fast (Target)** |
| **Zero-Step Stability** | Weak | Moderate | **Excellent (Zero-init)** |
| **Implementation Complexity** | Low | High | **Moderate** |

---

## 6. Conclusion: The Roadmap to SOTA NILM Generation
To overcome the bottleneck of long training times and weak temporal logic, the project roadmap involves a transition from the baseline **In-Context Concatenation** to a **Decoupled adaLN-Zero architecture**.

By treating the 8 time features as modulation parameters rather than raw data, we enable the model to:
1.  **Strictly maintain physical laws**: Deeply grounding power generation in temporal context.
2.  **Optimize Hardware (RTX 4090)**: Reducing the GFLOPs per step to accelerate iterations.
3.  **Prepare for NILMFormer**: Providing the highest quality, time-consistent synthetic waveforms for downstream training.

---

### 中文核心总结 (Thesis Digest in Mandarin)
本项目技术路线演变：从 **在本文 (In-Context)** 拼接转向 **adaLN-Zero (自适应归一化调节)**。
- **现状 (Concat)**：计算复杂度随序列长度平方向增长，导致训练缓慢且逻辑松散。
- **目标 (adaLN-Zero)**：将时间特征作为“控制参数”而非“数据点”。通过调节每一层神经元的缩放与偏移，在不增加额外计算开销的前提下，强制波形与时间对齐。
- **价值**：这是当前最具扩展性的扩散架构，能将数小时的训练任务大幅缩减，并显著提升合成数据的逻辑准确性。
