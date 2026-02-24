# Algorithm 2 — Synthetic ON-Period Injection into Real Training Data

---

**Input:**
- X_real = [(x_agg, x_app)] — real sequence, Z-score normalised, length N_real
- S ∈ R^{N_s × 600 × C} — synthetic .npy data, MinMax [0, 1]
- N_syn — target number of synthetic rows to inject
- x_thr — appliance ON power threshold (Watts)
- l_win — dilation window length (from Algorithm 1)
- L_max — maximum chunk length per injection unit
- δ_thr — edge magnitude threshold (Watts)
- w — density counting window (timesteps)
- D_min — minimum event density to be "active"
- g_min — minimum gap to merge (morphological closing)
- d_min — minimum event duration to keep

**Output:**
- X_mix — mixed training dataset

---

## Phase 1: Identify Real OFF Segments

```
1:  ỹ ← X_real_app · σ_app + μ_app               # de-normalise appliance to Watts
2:  M_on[t] ← 1 if ỹ[t] >= x_thr else 0          # ON indicator
3:  M_on ← M_on ⊛ ones(2·l_win + 1)              # dilate (Algorithm 1 buffer)
4:  M_off ← NOT M_on                               # strict OFF mask
5:  Ω ← ExtractSegments(M_off)                     # contiguous OFF segments {(s_k, e_k)}
```

---

## Phase 2: Extract Synthetic Events via Edge Density

```
6:  P̃ ← Flatten(S[:, :, 0]) · P_max              # continuous power sequence [N_s×600] Watts
7:  δ[t] ← |P̃[t] - P̃[t-1]|                      # first-order difference (edges)
8:  E[t] ← 1 if δ[t] >= δ_thr else 0              # edge indicator
9:  D[t] ← Σ E[τ] for τ in [t-w/2, t+w/2]        # event density in sliding window w
10: A[t] ← 1 if D[t] >= D_min else 0              # active mask
11: A ← MorphClose(A, g_min)                       # merge gaps < g_min (intra-cycle pauses)
12: R ← ExtractSegments(A, min_len=d_min)          # discard segments < d_min steps

13: P ← []                                         # event pool
14: for (s, e) in R:
15:     for p = s to e, step L_max:
16:         P.append( P̃[p : min(p + L_max, e)] )  # split into ≤ L_max chunks
17:     end for
18: end for
```

> **Why chunking?**
> Without `L_max`, morphological closing may merge consecutive ON windows into one
> giant event (e.g., 80,000 steps for washing machine), which can only fit into a
> few large OFF segments — leaving the rest of the dataset empty.
> Capping at `L_max` creates many small units that can be spread uniformly.

---

## Phase 3: Even Distribution and Physical Reconstruction

```
19: Cycle P until total rows >= N_syn → selected pool P* of size M

20: for k = 1 to |Ω|:
21:     m_k ← round( M · (e_k - s_k) / Σ(e_j - s_j) )   # proportional event count
22:     gap_k ← floor( (e_k - s_k - Σ L_i^(k)) / (m_k + 1) )  # even inner spacing

23:     for each chunk P*_i, i in [1..m_k]:
24:         Append gap_k real rows from X_real[Ω_k] to X_mix    # quiet background gap
25:         x_bg ← max(X_real[Ω_k]_agg · σ_agg + μ_agg - ỹ[Ω_k], 0)  # background Watts
26:         x̃_syn ← x_bg + P*_i                               # physical reconstruction
27:         Append ( (x̃_syn - μ_agg)/σ_agg,
                     (P*_i  - μ_app)/σ_app  ) to X_mix         # re-normalise and save
28:     end for

29:     Append remaining rows of Ω_k to X_mix
30: end for

31: return X_mix
```

---

## Summary of Key Design Decisions

| Step | Design Choice | Reason |
|---|---|---|
| Phase 1 — OFF mask | Use Algorithm 1 dilation buffer | Prevents injection into real ON-period margins |
| Phase 2 — Density detection | Count edges per window, not power value | Robust to multi-phase appliances (washer, dishwasher) where low-power phases are still "active" |
| Phase 2 — Chunking (L_max) | Split detected events into fixed-length units | Ensures enough injection units for even distribution across the entire dataset |
| Phase 3 — Proportional allocation | m_k ∝ segment length | Longer OFF segments receive more events — maintains temporal balance |
| Phase 3 — Physical reconstruction | x̃_syn = x_bg + ỹ_syn | Aggregate is physically correct; no artificial power appears from nowhere |
