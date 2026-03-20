# NILM Diffusion Model: Full Pipeline Documentation (Paper Quality)
> **Code-Verified. Every detail cross-checked line-by-line against source files.**
> Files audited: `real_datasets.py`, `agent_transformer.py`, `model_utils.py`, `gaussian_diffusion.py`, `solver.py`, `main.py`

---

## PART 1: Complete Step-by-Step Data Flow (Code-Accurate)

---

### STAGE 1 — Data Loading & Preprocessing (`real_datasets.py`)

**Step 1.1 — Raw Data Ingestion**
- Source: `.npy` / CSV
- Shape: `(Total_Timesteps, 9)` — Column 0 = raw watts; Columns 1–8 = pre-computed 8D time features

**Step 1.2 — Time Feature Format (8 Dimensions)**
```
col 1: minute_sin    col 2: minute_cos
col 3: hour_sin      col 4: hour_cos
col 5: day_of_week_sin   col 6: day_of_week_cos
col 7: month_sin     col 8: month_cos
```
⚠️ **These 8D time features are NEVER scaled/normalized. They remain as raw Sin/Cos throughout training and sampling.**

**Step 1.3 — Normalization (Power Only)**
```python
# Power column (col 0) ONLY:
MinMaxScaler → [0, 1]
normalize_to_neg_one_to_one → [-1, 1]
# Time features (col 1-8): NO normalization. Kept as-is.
```

**Step 1.4 — Windowing Strategy (Style-Dependent)**
```
Training:  Sliding window, stride=1 → (L - 512 + 1) samples
Sampling:  Non-overlapping, stride=512 → (L // 512) samples
```

**Step 1.5 — Continuity Booster (Training Only)**
```python
# Scan every training window:
For each window:
    if max(power in window) > boost_threshold:  # threshold = 0.2 * 2 - 1 = -0.6 in [-1,1] space
        mark as "active_window"

# Temporal Jittering: Duplicate active windows with small random shifts
For i in range(boost_factor - 1):   # boost_factor default = 4 (empirical)
    jitter = randint(-2, +2)         # ±2 point shift in window start index
    Add jittered_active_windows to training set

# Result: Training set expanded to (original + 3 × active_windows)
# Indices sorted to preserve temporal order (Jan → Dec)
```

**Stage 1 Output:** `LazyWindows` object yielding `(B, 512, 9)` batches — normalized power + raw time features.

---

### STAGE 2 — Training Pipeline (`solver.py` → `gaussian_diffusion.py`)

**Step 2.1 — Batch Delivery**
```python
data = next(dataloader)   # Shape: (B, 512, 9), on GPU
```

**Step 2.2 — Physical Decoupling (`_train_loss`, `agent_transformer.py` Line 525)**
```python
x_power     = x_start[:, :, :1]   # (B, 512, 1) — The stochastic variable
x_condition = x_start[:, :, 1:]   # (B, 512, 8) — The frozen deterministic prior
```

**Step 2.3 — Forward Diffusion (Noise Injection to Power Only) (`q_sample`, Line 308)**
```python
noise = randn_like(x_power)
x_power_noisy = sqrt(ᾱ_t) * x_power + sqrt(1 - ᾱ_t) * noise
# Time features are NEVER noised. They are locked.
x_input = cat([x_power_noisy, x_condition], dim=-1)  # (B, 512, 9) re-assembled
```

**Step 2.4 — Neural Network Forward Pass (`output()`, Line 159)**
Noisy 9D tensor enters Transformer model.

---

### STAGE 3 — Inside the Transformer (`agent_transformer.py`)

#### Branch A: Time Condition → Semantic Embedding
```
Input: x_cond (B, 512, 8)
    ↓
[1] Linear(8 → 1024) + SiLU         ← cond_emb_mlp layer 1 (Line 484–485)
    ↓
[2] ResMLP Block #1:                 ← (Line 486)
      residual = x
      x = Linear(1024) → SiLU → Linear(1024)
      output = residual + x          ← Skip Connection
    ↓
[3] ResMLP Block #2:                 ← (Line 487)
      (Same structure as Block #1)
    ↓
[4] Linear(1024 → 1024)              ← Final output refiner (Line 488)
    ↓
Output: label_emb (B, 512, 1024)     ← Semantic Meta-Command for every timestep
```

#### Branch B: Power Signal → Transformer Core
```
Input: x_power_noisy (B, 512, 1)
    ↓
Conv_MLP: 1D Conv(1 → 1024, kernel=1) ← Power embedding (Line 466)
    ↓
LearnablePositionalEncoding(1024)      ← Position-aware embedding (Line 505)
    ↓
═══════════════════════
ENCODER (5 × EncoderBlock, Line 504):
    Each block receives: (x, timestep t, label_emb)

    Inside each EncoderBlock (Line 248–292):
    ┌─────────────────────────────────┐
    │ AdaLayerNorm (ln1):             │  ← model_utils.py Line 187
    │   emb = SinusoidalPosEmb(t)    │  ← Diffusion timestep t → 1024D
    │   emb = emb + label_emb        │  ← Add semantic time command (broadcast)
    │   emb → SiLU → Linear(3×1024)  │  ← ZERO-INITIALIZED linear
    │   → chunk into γ, β, α         │
    │   x_norm = LayerNorm(x)*(1+γ)+β│
    │                                 │
    │ Agent Attention (SDPA):         │
    │   QKV = Linear(x_norm)         │
    │   Agent tokens: (1, nh, 64, hs)│  ← Learnable agent tokens
    │   SDPA(agents, K, V) → context1│  ← Agent aggregation
    │   SDPA(Q, context1, context1)  │  ← Agent broadcast
    │   x = x + α * attn_output      │  ← Gated residual
    │                                 │
    │ AdaLayerNorm (ln2) + MLP:       │
    │   Same γ, β, α from new AdaLN  │
    │   MLP: Linear(1024→4096)→GELU  │
    │        → Linear(4096→1024)     │
    │   x = x + α * mlp_output       │  ← Gated residual
    └─────────────────────────────────┘
═══════════════════════
Encoder output: enc_cond (B, 512, 1024)

DECODER (14 × DecoderBlock, Line 507):
    Each block receives: (x, enc_cond, timestep t, label_emb)

    Inside each DecoderBlock (Line 324–397):
    ┌─────────────────────────────────┐
    │ [1] Self-Attention (AdaLN+SDPA)│  ← Same AdaLN-Zero mechanism
    │     x = x + α1 * self_attn_out │
    │                                 │
    │ [2] Cross-Attention (AdaLN):   │
    │     Q from x, KV from enc_cond │  ← Cross-attention to Encoder
    │     x = x + α2 * cross_attn   │
    │                                 │
    │ [3] Physics Decomposition:     │
    │     proj → split into x1, x2  │
    │     trend  = TrendBlock(x1)    │  ← Polynomial regression (degree 3)
    │     season = FourierLayer(x2)  │  ← Top-K inverse DFT frequencies
    │     (accumulate across blocks) │
    │                                 │
    │ [4] MLP Branch (AdaLN):        │
    │     x = x + α3 * mlp_out      │
    └─────────────────────────────────┘

SYNTHESIS (Lines 543–548):
    res          = inverse Conv_MLP(output)         # (B, 512, 1)
    res_m        = mean(res, dim=1)                 # Global mean
    season_error = combine_s(Σ seasons) + res - res_m
    trend_final  = combine_m(Σ means) + res_m + Σ trends
    power_pred   = trend_final + season_error       # (B, 512, 1)
```

**Step 3 Output Re-assembly (Line 169–170)**
```python
conditions = x[:, :, 1:]                          # Original clean 8D time
model_output = cat([power_pred, conditions], -1)  # (B, 512, 9)
```

---

### STAGE 4 — Loss Computation (`_train_loss`, Line 315)

```python
# Extract predicted power
model_out_power = model_out[:, :, :1]         # (B, 512, 1)
target          = x_power_original            # (B, 512, 1) — original clean power

# 1. Compute Huber Loss (delta=0.5)
base_loss = huber_loss(model_out_power, target, reduction='none')

# 2. Asymmetric ON-period Weight Mask (Novelty)
weight_mask = where(target > 0.05, 5.0, 1.0)  # 5× penalty for ON-states

# 3. Weighted Time-domain Loss
train_loss = base_loss * weight_mask

# 4. Optional Fourier Loss (frequency domain consistency)
if use_ff:
    fft_pred = fft(model_out_power)
    fft_real = fft(target)
    f_loss   = huber(Re(fft_pred), Re(fft_real)) + huber(Im(fft_pred), Im(fft_real))
    train_loss += ff_weight * (f_loss * weight_mask)

# 5. Reweight by diffusion timestep
train_loss = mean(train_loss * loss_weight[t])
```

---

### STAGE 5 — Sampling (`generate_with_conditions`, Line 255 + `main.py`)

**Step 5.1 — Initialize Noise**
```python
img = randn(B, 512, 9)               # Full random noise (all 9 channels)
img[:, :, 1:] = condition            # LOCK time features to ground truth (Line 278)
```

**Step 5.2 — Reverse Diffusion Loop (T → 0, Line 288–291)**
```python
for t in range(T, 0, -1):
    model_mean, variance = p_mean_variance(img, t)
    noise               = randn_like(img)
    noise[:, :, 1:]     = 0         # ZERO NOISE on time channels (Line 202)
    img = model_mean + sqrt(variance) * noise
    img[:, :, 1:]       = condition  # RE-LOCK every single step (Line 291)
```

**Step 5.3 — Ordered Non-Overlapping Batch Sampling (`main.py`, Line 176–191)**
```
Dataset style='non_overlapping' during sampling
→ Windows at indices [0, 512, 1024, 1536, ...]
→ Condition for each window = time features from dataset[i]
→ Each window generated independently (no overlap blending in current default mode)
```

**Step 5.4 — Post-Processing & Denormalization (`main.py`, Line 218–244)**
```python
# Generated samples: (N, 512, 9) in [-1, 1] space
power    = samples[:, :, 0:1]         # Power in [-1, 1]
time_feats = samples[:, :, 1:9]       # Time in [-1, 1]

# Denormalize power ONLY:
power_01  = (power + 1) / 2           # → [0, 1]
power_W   = scaler.inverse_transform(power_01)  # → Real Watts

# Recombine
final = cat([power_W, time_feats], axis=2)  # (N, 512, 9)

# Save
np.save('ddpm_fake_{appliance}.npy', final)
```

---

## PART 2: Image Generation Prompts (5 Diagrams)

### 🖼️ PROMPT A — Full System Flowchart (Top-Level Overview)
> "A professional academic flowchart on a pure white background for a Conditional NILM Diffusion Model. Shows 5 labeled stages connected by arrows: [Stage 1] 'Data Preprocessing': raw power + timestamps → MinMaxScaler on power only → 512-point Sliding Windows → Continuity Booster (4× active window duplication, ±2pt Jitter). [Stage 2] '9D Tensor Construction': Shape (B,512,9), immediately split into 1D Red Power path and 8D Green frozen Time path. [Stage 3] 'Forward Diffusion + Transformer': red power path shows noise injection formula P_t = sqrt(alpha_bar)*P0 + sqrt(1-alpha_bar)*epsilon; then 9D tensor enters Encoder(5 blocks)+Decoder(14 blocks). [Stage 4] 'Loss': Huber Loss with 5× ON-period weight mask + Fourier regularization. [Stage 5] 'Sampling + Denorm': T→0 reverse diffusion with locked time features, then MinMaxScaler inverse on power only. IEEE style, black lines, teal/red path colors."

---

### 🖼️ PROMPT B — Physical Decoupling Diagram
> "A precise technical diagram on a white background. Title: 'Decoupled Conditioning: Physical Variable Separation'. Shows a 9-column tensor (B, 512, 9) entering from the left. A vertical scissors-cut divides it: left slice labeled '[:, :, :1] = Power P_0 [Stochastic]' shown in red, right slice '[:, :, 1:] = Time Condition C_0 [Deterministic / Frozen]' shown in green. The red power stream enters a 'q_sample: Add Gaussian Noise ε' block with formula P_t = sqrt(ᾱ_t)·P_0 + sqrt(1-ᾱ_t)·ε. The green time stream bypasses noise completely, labeled 'NEVER NOISED'. Both streams reconnect into a 9D tensor cat([P_t, C_0]) before entering the Transformer. Clean academic style, IEEE paper quality."

---

### 🖼️ PROMPT C — ResMLP Conditioning Engine (Internal Architecture)
> "A white-background architecture diagram of the 'Temporal Conditioning Engine (cond_emb_mlp)'. Vertical hierarchy bottom-to-top: [Input] '8D Temporal Prior: Sin/Cos of Minute, Hour, Day-of-Week, Month'. [Block 1] 'Initial Projector: Linear(8→1024) + SiLU'. [Block 2] 'ResMLP Block #1': split path — left 'Identity Skip x' and right 'Linear(1024)→SiLU→Linear(1024)', merged at a '+' circle node. [Block 3] 'ResMLP Block #2': identical to Block 2. [Block 4] 'Output Refiner: Linear(1024→1024)'. [Output] 'label_emb: Semantic Meta-Command (B, 512, 1024)'. Arrow shows label_emb flowing toward 'AdaLN-Zero'. Annotation near '+' nodes: 'h_m = h_{m-1} + MLP(h_{m-1})'. IEEE journal figure style."

---

### 🖼️ PROMPT D — AdaLN-Zero Modulation Block (Internal)
> "An architecture diagram of a 'DiT-style Encoder/Decoder Block with AdaLN-Zero' on white background. Two inputs enter from top: (1) 'x: current embedding (B, 512, 1024)' and (2) 'combined_emb = SinusoidalPosEmb(t) + label_emb (B, 512, 1024)'. The combined_emb passes through 'SiLU → Linear_ZeroInit(1024 → 3072)' then splits via chunk(3) into three signals: 'γ (Scale)', 'β (Shift)', 'α (Gate)'. γ and β modulate a LayerNorm: 'x_norm = LayerNorm(x) × (1+γ) + β'. x_norm then enters 'Agent Attention SDPA' (with 64 learnable agent tokens). The attention output multiplied by α: 'x_out = x + α × attn_out'. Below shows identical AdaLN-Zero + MLP branch with own α gate. Zero-initialization label on the Linear layer. Clean black and teal, IEEE style."

---

### 🖼️ PROMPT E — Sampling & Post-Processing Pipeline
> "A technical diagram on white background showing the complete inference/sampling pipeline. Left to right: [1] 'Initialize: img = randn(B,512,9); img[:,:,1:] = condition'. [2] A loop arrow showing 'Reverse Diffusion T→0' with two key operations per step: 'p_mean_variance (model forward pass)' and 'noise[:,  :, 1:]=0 then img[:,:,1:]=condition LOCKED'. [3] After loop: 'Generated samples (N,512,9) in [-1,1]'. [4] 'Post-Processing': split power from time → 'power: unnormalize [-1,1]→[0,1] → MinMaxScaler inverse → Real Watts'; 'time: kept as [-1,1], no inverse'. [5] 'cat([power_W, time_feats]) → Save .npy'. Clean flowchart, professional, computer science style."

---

## PART 3: Summary of Key Design Invariants (For Paper Writing)

| Property | Power Path | Time Condition Path |
|---|---|---|
| **Normalization** | MinMaxScaler → [-1, 1] | None (raw Sin/Cos) |
| **Noised in Training** | YES (q_sample) | NO |
| **Input to Transformer** | Noisy P_t | Clean C_0 always |
| **Modifies during Sampling** | YES (reverse diffusion) | NO (locked every step) |
| **Post-processing** | inverse_transform → Watts | Kept as [-1, 1] |
| **Loss Computed On** | YES (Huber + Fourier) | NO |
