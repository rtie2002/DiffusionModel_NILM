# Innovation: Hierarchical Temporal Encoding (HTE) Block
**The "Split & Conquer" Architecture for Multi-Scale Time Series**

## 1. The Problem: "Feature Blurring"
In standard Transformers (like classic DiT), we typically throw all time features into one big bucket:
*   Input Vector: `[Minute, Hour, Day, Month]` (8 Dimensions)
*   Processed by: **Single MLP**.

**The Flaw**:
The neural network tends to be "lazy". It latches onto the strongest signal (usually **Month/Seasonality** because it changes slowly and predicts the baseline) and ignores the subtle signal (**Minute/Hour** which defines the sharp edges).
*   Result: The model learns "It's January" but forgets "It's 13:00". The appliance waveform gets smeared.

---

## 2. The Innovation: The HTE Block
We replaced the Single MLP with **4 Independent Expert MLPs** running in parallel.

### The Architecture logic
1.  **Slicing**: We physically cut the 8-dim vector into 4 dual-dim pairs.
2.  **Expert Processing**:
    *   **Minute Expert**: Sees ONLY `[Min_Sin, Min_Cos]`. Learns high-frequency noise/texture.
    *   **Hour Expert**: Sees ONLY `[Hour_Sin, Hour_Cos]`. **Crucial for Daily Patterns (Microwave at Noon).**
    *   **Day Expert**: Sees ONLY `[Day_Sin, Day_Cos]`. Learns Weekly routines (Weekend vs Weekday).
    *   **Month Expert**: Sees ONLY `[Month_Sin, Month_Cos]`. Learns Seasonality.

3.  **Fusion**: We concatenate the experts *only after* they have extracted their specific features.
    $$ E_{final} = \text{Concat}(E_{min}, E_{hr}, E_{day}, E_{mo}) $$

---

## 3. Why It Solves Your Problem
This architecture forces a **Structural Inductive Bias**.
*   The "Hour Expert" **cannot** be distracted by the Month. It has no access to it.
*   It serves as a dedicated channel to transmit the "13:00 PM" command directly to the AdaLN modulation layers.
*   Combined with CFG, this means when we amplify the signal, we are amplifying a **pure, disentangled Hour signal**, creating sharp, accurately timed peaks.

## 4. Code Implementation (`agent_transformer.py`)
```python
# The "Split"
m_emb = self.minute_mlp(x_cond[:, :, 0:2])
h_emb = self.hour_mlp(x_cond[:, :, 2:4])  # <--- The Hero
d_emb = self.dow_mlp(x_cond[:, :, 4:6])
mo_emb = self.month_mlp(x_cond[:, :, 6:8])

# The "Fusion"
label_emb = torch.cat([m_emb, h_emb, d_emb, mo_emb], dim=-1)
```
