# Diffusion Model Training and Sampling Strategy

## 1. Probabilistic Modeling of Load Profiles

The objective of the proposed diffusion model is to learn the conditional distribution $p_\theta(x_0 | c)$, where $x_0 \in \mathbb{R}^{L \times 1}$ represents a sequence of appliance power consumption values of length $L$, and $c \in \mathbb{R}^{L \times K}$ represents the corresponding temporal condition features (e.g., time of day, day of week, seasonal indicators). The model is trained to generate realistic power profiles that respect the underlying temporal patterns of human activity.

### 1.1 Training: Randomized Slice Learning

During the training phase, strict temporal continuity between batches is not required. The model learns local dependencies and conditional relationships by maximizing the evidence lower bound (ELBO) on randomized slices of the dataset.

Let $\mathcal{D} = \{(x^{(i)}, c^{(i)})\}_{i=1}^{T_{total}}$ be the complete time-series dataset, where $T_{total}$ is the total number of recorded timestamps. We define a sliding window operation $\mathcal{W}_j$ that extracts a subsequence of length $L$ starting at index $j$:
$$
\mathbf{x}_j = \mathcal{W}_j(\mathcal{D}) = \{x^{(j)}, \dots, x^{(j+L-1)}\}
$$
$$
\mathbf{c}_j = \mathcal{W}_j(\mathcal{C}) = \{c^{(j)}, \dots, c^{(j+L-1)}\}
$$

The training objective is to minimize the simplified noise prediction loss:
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{j \sim U(0, T_{total}-L), t \sim U(1, T), \epsilon \sim \mathcal{N}(0, I)} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_j + \sqrt{1-\bar{\alpha}_t}\epsilon, t, \mathbf{c}_j) \|^2 \right]
$$

Critically, the starting index $j$ is sampled uniformly from the entire dataset at each step. This **Randomized Slice Learning** ensures that:
1.  **IID Assumption Approximation**: By treating each window $\mathbf{x}_j$ as an independent sample, we satisfy the independent and identically distributed (IID) assumption required for stable stochastic gradient descent (SGD).
2.  **Temporal Bias Elimination**: Due to the DataLoader's `shuffle=True` setting, while a single batch may not contain all temporal contexts, within each **epoch**, the model traverses all samples in the dataset (covering all seasons, weekdays, and hours). Over multiple epochs with different shuffling orders, the model is exposed to all temporal contexts (e.g., morning/evening, summer/winter) with approximately equal frequency, thereby preventing overfitting to any specific period.

## 2. Sampling: Conditional Generation Strategies

Once trained, the model $p_\theta(x_0 | c)$ can be used to synthesize data. The quality and utility of the synthetic data depend heavily on how the conditions $\mathbf{c}$ are selected. We propose two distinct sampling strategies: **Ordered Non-Overlapping Sampling** (for training data augmentation) and **Random Sampling** (for diversity augmentation).

### 2.1 Strategy A: Randomized Conditional Sampling

In this mode, we aim to generate diverse, representative samples from the learned distribution without enforcing long-term continuity. We sample $N$ indices $\{j_1, \dots, j_N\}$ uniformly from the dataset:
$$
j_k \sim U(0, T_{total}-L)
$$

The synthetic data batch $X_{sf}$ is generated as:
$$
X_{sf} = \{ \hat{\mathbf{x}}_{j_1}, \dots, \hat{\mathbf{x}}_{j_N} \} \quad \text{where} \quad \hat{\mathbf{x}}_{j_k} \sim p_\theta(x_0 | \mathbf{c}_{j_k})
$$

**Properties:**
*   **Distribution Matching**: This strategy ensures that the marginal distribution of temporal conditions in the synthetic dataset matches the ground truth $p(\mathbf{c})$. E.g., if "December" constitutes 1/12th of the real data, it will constitute ~1/12th of the synthetic conditions.
*   **Diversity**: Suitable for observing the model's behavior under various random conditions.

### 2.2 Strategy B: Ordered Non-Overlapping Sampling (Dataset Replication)

To create a synthetic dataset that structurally mirrors the real dataset (e.g., for training a downstream NILM model), we employ an **Ordered Non-Overlapping** strategy. We partition the temporal conditions $\mathcal{C}$ into $M$ contiguous, non-overlapping blocks of length $L$:
$$
M = \lfloor T_{total} / L \rfloor
$$
The starting indices are deterministic and strided by $L$:
$$
J = \{0, L, 2L, \dots, (M-1)L\}
$$

The synthetic dataset $\mathcal{D}_{syn}$ is constructed by sequentially generating power profiles for each block:
$$
\mathcal{D}_{syn} = \bigcup_{k=0}^{M-1} \hat{\mathbf{x}}_{k \cdot L} \quad \text{where} \quad \hat{\mathbf{x}}_{k \cdot L} \sim p_\theta(x_0 | \mathbf{c}_{k \cdot L})
$$

**Mathematical Implication:**
This is effectively performing a **re-simulation** of the entire recorded history.
$$
\hat{X}_{total} \approx \int p_\theta(x_t | c_t) \, dt \quad \text{over the domain } [0, T_{total}]
$$
This ensures:
1.  **Full Temporal Coverage**: Every timestamp $c^{(i)}$ from the original dataset is used exactly once as a condition.
2.  **Seasonal completeness**: The synthetic data contains the exact same proportion of days, weeks, and months as the real data.
3.  **No Data Leakage**: Since blocks are non-overlapping, we avoid the redundancy inherent in sliding window sampling (where $x_t$ would appear in $L$ different windows).

### 2.3 Implementation Logic

Let $S$ be the stride parameter.
*   **Overlapping (Sliding Window):** $S = 1$. The generated set corresponds to indices $\{0, 1, 2, \dots \}$. This introduces extreme redundancy ($99.8\%$ overlap) and temporal bias if $N \ll T_{total}$ (e.g., only generating the first 2 days).
*   **Non-Overlapping (Block-wise):** $S = L$. The generated set corresponds to indices $\{0, L, 2L, \dots \}$. This maximizes information gain and coverage.

Our improved sampling algorithm implements Strategy B by setting $S=L$, ensuring the generated synthetic dataset $\mathcal{D}_{syn}$ is a complete, unbiased "parallel universe" version of $\mathcal{D}_{real}$.
