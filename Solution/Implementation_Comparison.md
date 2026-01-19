# Comprehensive Comparison: Original vs. Modified Diffusion Model for NILM

## Executive Summary

This document provides a detailed, thesis-style comparison between the original Diffusion Model implementation for NILM (Non-Intrusive Load Monitoring) and the modified version that incorporates multivariate temporal feature conditioning. The modifications represent a significant architectural enhancement, transforming an unconditional generative model into a temporally-aware conditional generation framework.

---

## 1. Architectural Modifications

### 1.1 Core Model Architecture (`gaussian_diffusion.py`)

#### Addition of Conditional Dimension Parameter
**Original Code:**
```python
class Diffusion(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,  # Only 1 (power)
            ...
```

**Modified Code:**
```python
class Diffusion(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,  # Still 1 (power)
            condition_dim=8,  # NEW: 8 time features
            ...
```

**Impact**: The model now explicitly separates the generation target (power, 1D) from conditioning inputs (time features, 8D), enabling temporal context awareness.

---

#### Transformer Input Dimension Expansion
**Original Code:**
```python
self.model = Transformer(n_feat=feature_size, ...)  # 1 dimension
```

**Modified Code:**
```python
self.model = Transformer(n_feat=feature_size + condition_dim, ...)  # 9 dimensions
```

**Impact**: The internal Transformer now processes a 9-dimensional input space, jointly modeling power and time features through attention mechanisms.

---

### 1.2 Training Process Modifications

#### Noise Application Strategy
**Original Code:**
```python
def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
    noise = default(noise, lambda: torch.randn_like(x_start))
    x = self.q_sample(x_start=x_start, t=t, noise=noise)  # Noise ALL features
    model_out = self.output(x, t, padding_masks)
    train_loss = self.loss_fn(model_out, target, reduction='none')
```

**Modified Code:**
```python
def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
    # Separate power and conditions
    x_power = x_start[:, :, :self.feature_size]  # (B, L, 1)
    x_condition = x_start[:, :, self.feature_size:]  # (B, L, 8)
    
    # Only noise the power part
    noise = default(noise, lambda: torch.randn_like(x_power))
    x_power_noisy = self.q_sample(x_start=x_power, t=t, noise=noise)
    
    # Concatenate noisy power with CLEAN conditions
    x = torch.cat([x_power_noisy, x_condition], dim=-1)
```

**Mathematical Formulation:**

Original: $ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I) $ where $x_t \in \mathbb{R}^9$

Modified: 
$$
q(x_t | x_0, c) = \begin{cases}
\mathcal{N}(x_t^{power}; \sqrt{\bar{\alpha}_t} x_0^{power}, (1-\bar{\alpha}_t)I) & \text{for power} \\
c & \text{for conditions (no noise)}
\end{cases}
$$

**Impact**: Time features remain pristine during training, acting as *pure conditioning signals* rather than noisy inputs. This prevents the model from learning spurious correlations between noisy time features and power values.

---

#### Loss Computation Changes
**Original Code:**
```python
train_loss = self.loss_fn(model_out, target, reduction='none')
```

**Modified Code:**
```python
# Only compute loss on power part
model_out_power = model_out[:, :, :self.feature_size]
train_loss = self.loss_fn(model_out_power, target, reduction='none')
```

**Impact**: The optimization objective is strictly focused on reconstructing power values, not time features. This aligns with the true task: "Given time context $c$, predict power $p$."

---

### 1.3 Sampling Process Enhancements

#### New Conditional Generation Function
**Original Code:**
- No conditional generation capability
- Only `generate_mts(batch_size)` for unconditional sampling

**Modified Code:**
```python
@torch.no_grad()
def generate_with_conditions(self, condition, batch_size=None):
    """
    Generate data with time features preserved
    
    Args:
        condition: (B, seq_length, 8) - Time features
    Returns:
        (B, seq_length, 9) - Generated power + time features
    """
    # Initialize noise for full dimensions
    img = torch.randn(B, L, 9).to(condition.device)
    
    # Replace time features with actual conditions
    img[:, :, 1:] = condition
    
    # Reverse diffusion
    for t in reversed(range(0, self.num_timesteps)):
        img, _ = self.p_sample(img, t)
        # Force time features to stay fixed
        img[:, :, 1:] = condition
    
    return img
```

**Impact**: Enables *guided generation* where the model synthesizes power profiles that respect given temporal contexts (e.g., "Generate power for Monday, 3 PM, August").

---

#### Noise Masking During Sampling
**Original Code:**
```python
def p_sample(self, x, t: int, clip_denoised=True):
    ...
    noise = torch.randn_like(x) if t > 0 else 0.
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
```

**Modified Code:**
```python
def p_sample(self, x, t: int, clip_denoised=True):
    ...
    if t > 0:
        noise = torch.randn_like(x)
        # Zero out noise for time features (columns 1-8)
        if x.shape[-1] == 9:
            noise[:, :, 1:] = 0  # No noise for time features!
    else:
        noise = 0.
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
```

**Impact**: Prevents stochastic perturbations from corrupting the time features during reverse diffusion, maintaining temporal consistency.

---

## 2. Data Loading and Preprocessing Modifications

### 2.1 Dataset Structure (`real_datasets.py`)

#### Column Handling
**Original Code:**
- Expected CSV with single column: `power`
- Total features: 1

**Modified Code:**
- Expected CSV with 9 columns: `[appliance_power, minute_sin, minute_cos, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos]`
- Automatic detection and parsing of time features
- Conditional scaler fitting (only on power column)

```python
# Extract power
app_col = find_appliance_column()  # e.g., 'dishwasher'
data = df[[app_col] + time_cols].values

# Fit scaler ONLY on power
scaler = MinMaxScaler()
scaler.fit(data[:, 0:1])  # Column 0 only
```

**Impact**: Time features bypass normalization, preserving their semantic meaning (sin/cos values in [-1, 1]).

---

### 2.2 Normalization Strategy
**Original Code:**
```python
data = sc aler.transform(rawdata)  # All columns normalized
if auto_norm:
    data = normalize_to_neg_one_to_one(data)  # [0,1] -> [-1,1]
```

**Modified Code:**
```python
if rawdata.shape[-1] == 9:
    power = rawdata[:, 0:1]
    time_features = rawdata[:, 1:]  # Already in [-1,1]
    
    # Scale only power
    power_scaled = scaler.transform(power)
    if auto_norm:
        power_scaled = normalize_to_neg_one_to_one(power_scaled)
    
    data = np.concatenate([power_scaled, time_features], axis=1)
```

**Impact**: Decouples power normalization (MinMax → [-1,1]) from time feature handling (preserve original values).

---

## 3. Main Script Enhancements (`main.py`)

### 3.1 Sampling Mode Introduction
**Original Code:**
```python
samples = trainer.sample(num=len(dataset), size_every=400, 
                        shape=[dataset.window, dataset.var_num])
```

**Modified Code:**
```python
parser.add_argument('--sampling_mode', type=str, default='ordered_non_overlapping',
                    choices=['random', 'ordered', 'ordered_non_overlapping'])

# Intelligent sample count calculation
if args.sampling_mode == 'ordered_non_overlapping':
    stride = dataset.window
    max_windows = len(dataset) // stride
    num_samples = max_windows
else:
    num_samples = len(dataset)

samples = trainer.sample(num=num_samples, stride=stride, ordered=ordered, ...)
```

**Impact**: Enables three distinct generation paradigms:
1. **Random**: Diverse samples from random time points
2. **Ordered**: Sequential sliding windows (high overlap)
3. **Ordered Non-Overlapping**: Full timeline coverage, zero redundancy

---

### 3.2 Selective Denormalization
**Original Code:**
```python
if dataset.auto_norm:
    samples = unnormalize_to_zero_to_one(samples)  # All features
```

**Modified Code:**
```python
if dataset.auto_norm:
    if dataset.var_num == 9:
        # Power: [-1,1] -> [0,1]
        samples[:, :, 0:1] = unnormalize_to_zero_to_one(samples[:, :, 0:1])
        # Time features: Keep in [-1,1]
        print("Power normalized to [0,1], time features preserved in [-1,1]")
    else:
        samples = unnormalize_to_zero_to_one(samples)
```

**Impact**: Output data maintains the correct value ranges: power in [0,1] for MinMax scaling, time features in [-1,1] for sin/cos interpretation.

---

## 4. Solver Enhancements (`solver.py`)

### 4.1 Strided Sampling Implementation
**Original Code:**
```python
def sample(self, num, size_every, shape, dataset, ordered=True):
    ...
    if ordered:
        indices = [(windows_completed + i) % dataset_size 
                   for i in range(windows_this_batch)]
```
*Stride implicitly = 1*

**Modified Code:**
```python
def sample(self, num, size_every, shape, dataset, ordered=True, stride=1):
    ...
    if ordered:
        indices = [(windows_completed + i * stride) % dataset_size 
                   for i in range(windows_this_batch)]
```

**Mathematical Impact:**
- **Original**: $\text{Index}_i = i \mod N$ → Covers first $M$ minutes only
- **Modified**: $\text{Index}_i = (i \times S) \mod N$ → Covers full $M \times S$ minutes where $S=512$

For a dataset with 1.28M minutes, generating 2500 samples:
- Original coverage: ~2500 minutes (2 days)
- Modified coverage: 1,280,000 minutes (full 2.4 years)

---

## 5. Configuration File Changes

### 5.1 YAML Config Structure
**Original (`Config/dishwasher.yaml`):**
```yaml
model:
  params:
    seq_length: 512
    feature_size: 1  # Only power
    # No condition_dim
```

**Modified:**
```yaml
model:
  params:
    seq_length: 512
    feature_size: 1  # Power dimension
    condition_dim: 8  # Time features dimension
```

---

## 6. Summary of Key Innovations

| **Aspect** | **Original** | **Modified** | **Benefit** |
|------------|-------------|--------------|-------------|
| **Model Type** | Unconditional Generation | Conditional Generation with Temporal Features | Respects time-dependent usage patterns |
| **Training Data** | 1D power values | 9D power + time features | Learns temporal correlations |
| **Noise Application** | All dimensions noised | Only power dimension noised | Prevents time feature corruption |
| **Loss Function** | All dimensions | Power dimension only | Focused optimization |
| **Sampling Strategy** | Random/Sequential overlapping | Ordered non-overlapping | Full timeline coverage, zero redundancy |
| **Generated Output** | 1D power sequence | 9D power + time sequence | Temporally aligned synthetic data |
| **Downstream Compatibility** | Generic NILM models | Temporal-aware models (e.g., NILMFormer) | Exploits model's time-encoding capability |

---

## 7. Theoretical Justification

### 7.1 Conditional DDPM Formulation
The modified model implements a **factorized conditional diffusion model**:

$$
p_\theta(x_0 | c) = \int p_\theta(x_{0:T} | c) dx_{1:T}
$$

where:
- $x_0 \in \mathbb{R}^1$: Target power value
- $c \in \mathbb{R}^8$: Temporal conditioning (sin/cos encoded)
- $p_\theta$: Learned reverse process

The reverse process is formulated as:
$$
p_\theta(x_{t-1} | x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, c, t), \Sigma_\theta(x_t, t))
$$

This contrasts with the original unconditional formulation:
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**Key Difference**: The denoising function $\mu_\theta$ now has access to temporal context $c$, enabling it to generate power profiles consistent with specific times of day/week/year.

---

### 7.2 Data Augmentation via Ordered Non-Overlapping Sampling
The new sampling strategy ensures:

1. **Temporal Coverage**: Every timestamp $t \in [0, T_{total}]$ is used exactly once as a condition
2. **Statistical Consistency**: Marginal distribution $ p(c_{synthetic}) = p(c_{real}) $
3. **No Data Leakage**: Zero overlap between generated windows

This creates a synthetic dataset that is a **parallel universe** of the real dataset: same timeline, different realizations.

---

## 8. Experimental Implications

### 8.1 Training Data Requirements
**Original**: 
- Requires large amounts of unlabeled power data
- No time information needed

**Modified**:
- Requires power data **with timestamps**
- Preprocessing to generate sin/cos time features
- Higher dimensional input (9D vs 1D)

### 8.2 Computational Cost
**Training**:
- Original: $O(N \times L \times 1)$
- Modified: $O(N \times L \times 9)$ ≈ **9× memory for forward pass**

**Sampling**:
- Similar cost (diffusion iterations dominate)

### 8.3 Expected Performance Gains
When used with temporal-aware NILM models:
- **Hypothesis**: Synthetic data with correct time features should improve disaggregation accuracy
- **Mechanism**: Time-conditioned synthetic data better represents realistic usage patterns
- **Validation Required**: Comparative experiments on NILMFormer with:
  1. Baseline (real data only)
  2. Original diffusion synthetic data
  3. Modified time-conditioned synthetic data

---

## 9. Limitations and Future Work

### 9.1 Current Limitations
1. **Training Complexity**: Increased parameter count and memory usage
2. **Preprocessing Burden**: Requires accurate timestamp extraction
3. **Hyperparameter Sensitivity**: Condition dimension scaling might need tuning

### 9.2 Potential Extensions
1. **Multivariate Output**: Extend to simultaneously generate multiple appliances
2. **Hierarchical Conditioning**: Add house-level or user-level conditions
3. **Adaptive Stride**: Dynamic non-overlapping strategies based on data density

---

## 10. Conclusion

The modifications transform the original unconditional diffusion model into a sophisticated **temporal-conditional generative framework**. The key innovations—selective noise application, conditional generation, and ordered non-overlapping sampling—represent a cohesive design philosophy: **respect the temporal structure of load data**.

While the original model could generate statistically plausible power sequences, it lacked awareness of *when* these sequences should occur. The modified version fills this gap, enabling the generation of temporally-grounded synthetic datasets suitable for training advanced, time-aware NILM models.

**The contribution is not a single algorithmic breakthrough, but a systematic adaptation** of diffusion models to the unique requirements of time-series load disaggregation. This represents solid incremental research suitable for publication in domain-specific venues (NILM, energy informatics) or applied machine learning conferences.
