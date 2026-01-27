# CBDM Implementation for NILM (Technical Documentation)

This document outlines the integration of **Class-Balancing Diffusion Models (CBDM)** into the NILM framework to solve the severe class imbalance (Long-Tail) problem between "ON" and "OFF" appliance states.

---

## 1. Mathematical Foundation

The core issue in NILM data generation is that the "OFF" state ($y=0$) occupies $>95\%$ of the total duration. Standard diffusion models minimize the global expected loss, which leads to **mode-collapse** in the rare "ON" state ($y=1$).

### Density Ratio Correction
CBDM corrects this by introducing a distribution adjustment schema into the reverse transfer probability:
$$p^*_\theta(x_{t-1}|x_t, y) = p_\theta(x_{t-1}|x_t, y) \frac{p_\theta(x_{t-1})}{p^*_\theta(x_{t-1})}$$

### Regularization Term
In practice, this is implemented as a training-time regularizer added to the standard DDPM objective:
$$L_{CBDM} = L_{DDPM} + L_r + \gamma L_{rc}$$

1.  **Distribution Adjustment Loss ($L_r$)**: 
    $$L_r = t \cdot \tau \cdot \| \epsilon_\theta(x_t, y) - \text{sg}(\epsilon_\theta(x_t, y')) \|^2$$
    This forces the model to transfer features from the "Head" class (OFF) to the "Tail" class (ON), using a stop-gradient (`sg`) to prevent identity collapse.

2.  **Commitment Loss ($L_{rc}$)**:
    Ensures the model doesn't drift too far from the original conditional manifold while balancing.

---

## 2. Deep Learning Model Logic

The implementation leverages the **AgentTransformer's** existing conditional architecture.

1.  **Adaptive Feature Decoupling**: The model continues to take 9D input (1D Power + 8D Time Features). While training, the CBDM logic separates the power signal (target) from the temporal conditions (conditioning signal).
2.  **Balanced Condition Swap**: For every batch, the model performs a "hidden" second forward pass. It takes the noisy power signal of an "ON" sample and pair it with a randomly selected "OFF" time feature set (and vice-versa). 
3.  **Cross-Class Similarity**: By minimizing the distance between these predictions, the hidden layers learn a unified physical representation of electrical current that is independent of the imbalance ratio.

---

## 3. Code Implementation Details

The implementation is primarily contained within `Models/diffusion/gaussian_diffusion.py` and configured via `fridge.yaml`.

### A. Vectorized State Detection
We use an appliance-specific threshold to identify states on-the-fly inside the GPU:
```python
is_on = (x_power.mean(dim=(1, 2)) > self.on_threshold)
on_idx = torch.where(is_on)[0]
off_idx = torch.where(~is_on)[0]
```

### B. Optimized Balanced Sampling
To maintain **RTX 4090 performance**, the 50/50 class balancing is fully vectorized to avoid Python CPU bottlenecks:
```python
use_on = torch.rand(B, device=x_start.device) > 0.5
target_indices = torch.where(use_on, rand_on, rand_off)
```

### C. Time-Adaptive Regularization
The regularization weight is normalized by the diffusion timestep $t$, ensuring that the model focuses on **Global Structure Balancing** in high-noise phases and **Physical Fidelity** in low-noise phases:
```python
t_weight = (t.float() / self.num_timesteps).view(-1, 1, 1)
lr = t_weight * self.tau * (model_out_power - model_out_prime_power.detach())**2
```

---

## 4. Configuration Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `tau` | 0.1 | Regularization strength. Higher values = more ON diversity. |
| `gamma` | 0.25 | Commitment weight. Prevents artifacts in wave shapes. |
| `on_threshold` | 0.05 | Normalized power threshold (15W @ 300W scale) to define the ON state. |
