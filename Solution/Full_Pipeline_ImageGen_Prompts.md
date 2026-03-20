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