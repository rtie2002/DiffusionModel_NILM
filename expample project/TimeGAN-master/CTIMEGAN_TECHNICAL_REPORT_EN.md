# Technical Report: C-TimeGAN Integration for NILM

This report details the architectural overview of the **Conditional TimeGAN (C-TimeGAN)**, optimized and implemented for Non-Intrusive Load Monitoring (NILM) tasks.

---

## 1. Core Architecture: Multi-Dimensional Condition Integration

While original TimeGAN models unconstrained latent spaces, our **C-TimeGAN** shifts to conditional probability modeling $P(X | C)$, where $C$ represents context features (8-dimensional time features and aggregate power).

---

## 2. Mathematical Formulation of Loss Functions

The model is optimized using a weighted multi-component loss function to ensure structural and temporal accuracy. We categorize these into two distinct groups, exactly as they are implemented in the `backward_g` and `backward_er_` functions:

### Group A: Generative & Supervised Losses (The "Logic")
The objective for the generator is to minimize the following composite loss:
$$L_G = w_U \cdot L_U + w_{Ue} \cdot L_{Ue} + w_{V1} \cdot L_{V1} + w_{V2} \cdot L_{V2} + \eta \cdot \sqrt{L_\sigma}$$

*   **Adversarial Loss ($L_U, L_{Ue}$)**: Binary Cross Entropy (BCE) from the Discriminator.
*   **Moment Matching ($L_{V1}, L_{V2}$)**: Ensures the Mean ($\mu$) and Standard Deviation ($\sigma$) of generated batches match the real data.
*   **Supervised Loss ($L_\sigma$)**: Measures how well the Supervisor predicts the next latent state:
    $$L_\sigma = \mathbb{E}[\| H_{t} - \text{Supervisor}(H_{t-1}) \|_2]$$

### Group B: Embedding & Recovery Losses (The "High-Fidelity")
Ensures the precision of the mapping between raw power signals ($X$) and latent features ($H$):
$$L_{EMB} = 10 \cdot \sqrt{L_{MSE}(X, \tilde{X})} + 0.1 \cdot L_\sigma$$
*   **Reconstruction Loss**: Acts as the "Eyes" of the model, ensuring it can reproduce the exact pixels/values of a waveform.

---

## 3. Training & Data Injection Pipeline

TimeGAN operates through three specific stages where different data streams are injected, and specific loss combinations are applied.

### 3.1 Stage-wise Loss Combination & Execution

| Stage | Objective | Active Loss Combination | Core Networks Trained |
| :--- | :--- | :--- | :--- |
| **1. Pre-ER** | Feature Mapping | **Group B** ($L_{EMB}$ only) | `Encoder`, `Recovery` |
| **2. Pre-S** | Temporal Rules | **$L_\sigma$** (MSE from Group A) | `Supervisor` |
| **3. Joint** (Adv) | Adversarial Creation | **Group A** | `Generator`, `Discriminator`, `Supervisor` |
| **3. Joint** (Emb) | Stable Features | **Group B** (Combined with Eq A logic) | `Encoder`, `Recovery` |

### 3.2 Detailed Step-by-Step Logic

#### **Step 1: Embedding Pre-training (The "Alphabet")**
*   **Data Injected**: Real $X$ + Condition $C$.
*   **Logic**: Minimize **Group B** ($L_{EMB}$). The model ignores generation and focuses purely on compression and reconstruction. It ensures that any raw washing machine signal can be perfectly translated to a latent vector and back.

#### **Step 2: Supervisor Pre-training (The "Grammar")**
*   **Data Injected**: Real $X$ + Condition $C$.
*   **Logic**: Minimize **$L_\sigma$**. The model learns the "Next-Step Prediction" logic. It ensures that if the current state is "Wash," the next predicted state isn't suddenly "Off" or "Spin" randomly.

#### **Step 3: Joint Adversarial Training (The "Symphony")**
This stage involves two parallel optimization loops to balance the GAN:

*   **Sub-Step 3.1: Generator Update (Group A)**
    *   **Data Injected**: Random Noise $Z$ + Condition $C$.
    *   **Formulation**: $Objective = L_U + w \cdot (L_{V1} + L_{V2}) + 15 \cdot \sqrt{L_\sigma}$
    *   **Goal**: Force the generator to produce signals that trick the discriminator **AND** follow the temporal predictions of the supervisor, while keeping similar global amplitude.

*   **Sub-Step 3.2: Embedding Update (Group B + L_σ)**
    *   **Data Injected**: Real $X$ + Condition $C$.
    *   **Formulation**: $Objective = L_{EMB} + 0.1 \cdot L_\sigma$
    *   **Goal**: Ensure that while the Generator is changing, the Encoder/Recovery remains stable and maintains high reconstruction fidelity.

---

## 4. Why so many losses and steps?

A standard GAN only has one loss. TimeGAN requires this multi-step, multi-loss pipeline because:
1.  **Phase Separation**: You cannot teach a model to "be creative" (Joint) before it knows how to "observe" (Pre-ER) and "predict" (Pre-S).
2.  **Conflict Resolution**: High-dimensional time-series needs logical order to not collapse into noise. $L_\sigma$ anchors the adversarial training so it doesn't wander off.
3.  **Bijective Stability**: Fusing reconstruction checks ($L_{MSE}$) with logic ($L_\sigma$) ensures the latent space doesn't "drift" while the Generator is fighting the Discriminator.

---

## 5. Comparison: Our Implementation vs. Vanilla Generator

| Feature | Standard GAN Generator | Our Conditional TimeGAN |
| :--- | :--- | :--- |
| **Data Flow** | Noise $\rightarrow$ Data | **Noise + Context $\rightarrow$ Supervised Logic $\rightarrow$ Data** |
| **Loss Criteria** | L2/MSE or BCE only | **BCE + Multi-Moment (V1/V2) + Forward Prediction** |
| **Convergence** | Static LR | **Cosine Annealing** (Aggressive start, fine-tune finish) |
| **Task Fitness** | Good for independent images | **Required** for sequential energy transients |

---

## 6. Key Innovation: Hybrid RNN-CNN Recovery Network

We upgraded the standard RNN Recovery net to a **Hybrid RNN-CNN Architecture**:
*   **RNN Layer**: Captures global pulse duration.
*   **CNN Layer**: A specialized refinement stage with a 5-step kernel and **Residual Connection** to restore sharp transients.
