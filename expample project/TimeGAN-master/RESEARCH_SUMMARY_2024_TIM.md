# Research Summary: C-TimeGAN for NILM (IEEE TIM 2024)

This document summarizes the methodology from the paper:  
**"Conditional-TimeGAN for Realistic and High-Quality Appliance Trajectories Generation and Data Augmentation in Nonintrusive Load Monitoring"**  
*Published in IEEE Transactions on Instrumentation and Measurement, Vol. 73, 2024.*

## 1. Core Problem
Original TimeGAN (Vanilla) often struggles with NILM data because:
- **Rapid Power Changes:** Sharp ON/OFF transitions are smoothed out by GRU/RNN.
- **Data Sparsity:** Appliances are OFF most of the time, causing mode collapse or noise.
- **Long Dependencies:** Difficulty in maintaining consistent power levels over long windows.

## 2. Key Mathematical Formulas

### A. Conditional Architecture (Eq. 7 & 8)
The model incorporates **Conditions** ($c$) into the latent space mapping:
- **Static Latent Space:** $h_{\sigma} = e_{\sigma}(\sigma_m, c_{dj}, c_{cj})$
- **Temporal Latent Space:** $h_{t} = e_{\chi}(h_{\sigma}, h_{t-1}, \chi_{tm}, c_{di}, c_{ci})$
- **Generator Output:** $\hat{h}_t = g_{\chi}(\hat{h}_{\sigma}, \hat{h}_{t-1}, z_t, c_{di}, c_{ci})$

*Where $c_{di}$ represents the first-order difference (power transitions).*

### B. Multi-Objective Loss Function
The model optimizes three specific losses to enforce physical realism:

1. **Reconstruction Loss ($L_R$ - Eq. 10):**
   Ensures the embedding can perfectly recover the original signal.
   $$L_R = E [ \| \sigma - \tilde{\sigma} \|_2^2 + \sum_t \| \chi_t - \tilde{\chi}_t \|_2^2 ]$$

2. **Supervised Loss ($L_{\sigma}$ - Eq. 12):**
   Step-wise supervised learning to ensure the generator's latent transitions match the real data's encoding.
   $$L_{\sigma} = E [ \sum_t \| h_t - g_{\chi}(h_{\sigma}, h_{t-1}, z_t) \|_2^2 ]$$

3. **Unsupervised (Adversarial) Loss ($L_U$ - Eq. 11):**
   The standard GAN minimax game between Generator and Discriminator.

## 3. Implementation Logic for our Codebase

To mimic this paper's success, we have modified `lib/timegan.py` with:
- **Weighted MSE:** Power column (Index 0) is given **10x weight** in $L_R$.
- **First-order Difference Penalty ($L_{diff}$):** A new term that calculates $\| \Delta P_{fake} - \Delta P_{real} \|$. This forces the model to learn sharp ON/OFF edges rather than fuzzy noise.
- **TV Regularization:** Encourages stable power states between transitions.

### C. Appliance Coupling Constraints (The "Secret Sauce")
Unlike our current setup which treats each appliance as independent, the paper uses **other appliances as conditions** ($c_c, c_d$):
- when generating a Fridge, the model "knows" what the Washing Machine and Microwave are doing.
- This captures the **interdependency** (e.g., specific usage patterns at specific times of day).

## 4. Specific Architecture Details (Table I & II)
To match the paper's benchmarks, we should note:
- **Hidden Dim:** $4 \times N_{appliances}$ (e.g., if we have 6 appliances, hidden dim = 24).
- **Sequence Length:** Used a sliding window of **60** to capture temporal dependencies.
- **Layers:** 3-layer GRU (Gated Recurrent Unit) for all sub-networks.
- **Learning Rate:** $0.0001$ with Adam optimizer.

## 5. Optimal Data Augmentation Strategy
The paper provides a crucial guideline for the 200% generation you were planning:
- **Optimal Ratio:** Performance peaks when the ratio of synthetic data to the activation rate is between **1:3 and 1:4**.
- **Over-augmentation:** Adding too much synthetic data (e.g., 5%+) can lead to **underfitting** or "over-regularization," where the model stops learning real-world nuances.

## 6. OCSVM Post-processing (Eq. 15 & 16)
The paper suggests using a **One-Class Support Vector Machine** as a filter:
- **Input:** Real appliance windows are used to find the "normal" hyperplane.
- **Filter:** Synthetic samples falling outside the hyperplane ($\text{sgn}(f(x)) = -1$) are discarded.
- **Result:** This significantly improves FID and Recall scores by removing "unrealistic" artifacts.

---
*Summary updated with deep-dive technical points from IEEE TIM 2024.*
