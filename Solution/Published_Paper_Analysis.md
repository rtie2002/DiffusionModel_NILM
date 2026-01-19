# Analysis of Published Paper vs. Your Work

## Paper Info
- **Title**: "A diffusion model-based framework to enhance the robustness of non-intrusive load disaggregation"
- **Authors**: Zeyi Geng, Linfeng Yang, Wuqing Yu
- **Journal**: Energy (Q1, IF ~11.2)
- **Published**: March 2025
- **Code**: https://github.com/linfengYang/DiffusionModel_NILM

---

## Their Main Contributions

### 1. **Diffusion Model for Data Augmentation**
- Generate synthetic appliance load data
- Focus on **low-noise**, **multi-state** data generation
- Use **Algorithm 1** to select effective ON periods

### 2. **Robust Loss Function**
- Combine L1 and L2 (Huber-like)
- Add ON/OFF detection component
- Formula: Mixed loss with power prediction + state classification

### 3. **Post-Processing Algorithm**
- Exponentially weighted moving average
- Reduce signal fluctuations

---

## Their Experimental Design (What You Can Learn)

### Experiment 1: Generated Data Quality Evaluation

**Metrics They Used:**
| Metric | Purpose | Tool |
|--------|---------|------|
| **Context-FID** | Feature similarity | Pre-trained TS2Vec encoder |
| **SWD** | Signal distribution distance | Sliced Wasserstein Distance |
| **Classification Score** | Can classifier distinguish real vs. synthetic? | LSTM classifier |
| **Predictive Score** | Temporal dependency preservation | GRU predictor |

**Baselines:**
- GAN
- TimeGAN
- **Their Diffusion Model**

**Result**: Diffusion > TimeGAN > GAN (亮Table 4)

---

### Experiment 2: NILM Performance (Main Results)

**Setup:**
| Dataset | Training Data | Test Scenario |
|---------|---------------|---------------|
| UK-DALE House 2 | Real + Synthetic (various ratios) | Origin-household |
| UK-DALE House 1 | Trained on House 2 data | Cross-household |

**Mixing Ratios Tested:**
- 100k Real + 100k Synthetic
- 100k Real + 200k Synthetic
- 200k Real + 100k Synthetic
- **200k Real + 200k Synthetic** (Best)

**Models Tested:**
- Their simple CNN model (baseline)
- **S2P** (Seq2Point)
- **FCN** (Fully Convolutional Network)
- **AugLPN** (State-of-the-art)

**Metrics:**
- MAE (Mean Absolute Error)
- SAE (Signal Aggregate Error)
- F1 Score

**Results** (Table 5-9):
- Origin-household: 24.24% MAE improvement (average)
- Cross-household: 16.60% MAE improvement
- All models improved with their framework

---

### Experiment 3: Ablation Study

**Components Tested:**
| Component | Purpose |
|-----------|---------|
| Robust Data | Their synthetic data vs. no synthetic |
| Robust Loss | Their Huber-like loss vs. MSE |
| Post-Processing | Algorithm 2 vs. no smoothing |

**Result** (Table 11): All components contribute

---

## Key Differences: Their Work vs. Yours

| Aspect | Their Work | Your Work |
|--------|-----------|-----------|
| **Time Features** | ❌ No temporal conditioning | ✅ **Multivariate time features (8D)** |
| **Generation Strategy** | Algorithm 1 (ON-period selection) | **Ordered non-overlapping (full timeline)** |
| **Focus** | Noise reduction + data augmentation | **Temporal alignment + full coverage** |
| **Downstream Model** | Simple CNN, S2P, FCN, AugLPN | NILMFormer (temporal-aware) |
| **Novelty** | Robust loss + post-processing | **Time-conditioned generation** |

---

## What YOU Can Borrow from Their Experiments

### ✅ 1. Data Quality Metrics (MUST DO)

**Copy their evaluation metrics:**
```python
# Implement these:
1. Context-FID (use TS2Vec or similar encoder)
2. SWD (Sliced Wasserstein Distance)
3. Classification Score (LSTM to distinguish real/fake)
4. Predictive Score (GRU predicts real data from synthetic)
```

**Why**: These are **standard metrics** for time-series generation

**Your Table:**
| Metric | GAN | Ours (Time-Conditioned) |
|--------|-----|-------------------------|
| Context-FID | ? | ? (should be lower ✓) |
| SWD | ? | ? (should be lower ✓) |
| Classification Score | ? | ? |
| Predictive Score | ? | ? |
| **Temporal Distribution χ²** | High | **Low** ← Your unique contribution |

---

### ✅ 2. Mixing Ratio Experiments

**Test different ratios** (like they did):
- 0% (Real only)
- 25% synthetic
- 50% synthetic
- 100% synthetic
- 200% synthetic (您已经有这个了)

**Present as curve:**
```
     MAE
      |
 21   |  *  (0%)
      |
 19   |     *  (25%)
      |
 17   |        *  (50%)  ← Optimal
      |
 18   |           *  (100%)  ← Overfitting?
      |________________________
          Synthetic %
```

---

### ✅ 3. Cross-Dataset Validation

**They tested on REDD (cross-household)**
**You should test on REDD too** (您的plan已有)

---

### ✅ 4. Multiple Downstream Models

**They tested:**
- S2P
- FCN
- AugLPN

**You should test:**
- NILMFormer (main)
- ~~EasyS2S~~ (您说incompatible，可skip)
- **考虑加FCN或S2P**（如果容易adapt to multivariate）

---

### ✅ 5. Ablation Study Structure

**Copy their ablation logic:**
| Experiment | What You Remove | Expected Impact |
|------------|-----------------|-----------------|
| Full Method | Nothing | Best performance |
| w/o Time Conditioning | Remove 8D time features | Performance drops |
| w/o Ordered Sampling | Use random sampling | Temporal distribution skewed |

---

## What YOU Have That THEY Don't

### Your Unique Strengths:

✅ **1. Temporal Feature Conditioning**
- They don't condition on time
- You explicitly model time-power correlation
- **This is your main novelty!**

✅ **2. Ordered Non-Overlapping Sampling**
- They use Algorithm 1 (ON-period selection, ~5% coverage)
- You use full timeline (100% coverage)
- **Ensures temporal completeness**

✅ **3. Temporal-Aware Model Focus**
- They test on generic models (CNN, FCN)
- You test on NILMFormer (temporal Transformer)
- **Better match for your data**

---

## Recommended Experimental Plan (Updated)

### Week 1-2: Implement Data Quality Metrics
- [ ] Context-FID (borrow TS2Vec or train simple encoder)
- [ ] SWD
- [ ] Classification Score
- [ ] **Your unique**: Temporal Distribution χ²

### Week 3: GAN Baseline
- [ ] Implement Conditional GAN
- [ ] Train on UK-DALE
- [ ] Generate synthetic data
- [ ] Compute quality metrics

### Week 4: Main Results (UK-DALE)
- [ ] Test mixing ratios: 0%, 25%, 50%, 100%, 200%
- [ ] NILMFormer performance
- [ ] Create results table (like their Table 5)

### Week 5: REDD Validation
- [ ] Repeat on REDD
- [ ] Cross-household generalization

### Week 6: Ablation
- [ ] Time conditioning ablation
- [ ] Sampling strategy validation

---

## Your Paper's Positioning (vs. Theirs)

### Their Paper:
**"We improve NILM robustness through:**
1. Noise-robust synthetic data
2. Robust loss function
3. Post-processing"

### Your Paper Should Be:
**"We enable temporal-aware NILM data augmentation through:**
1. **Time-conditioned diffusion** (vs. their unconditional)
2. **Ordered non-overlapping sampling** (vs. their ON-period selection)
3. **Validation on temporal-aware models** (NILMFormer)

**Positioning:** You're **complementary** to their work, not competing

---

## Conclusion: What to Do

### ✅ Must Do:
1. **Implement their data quality metrics** (Context-FID, SWD, etc.)
2. **Add GAN baseline** (fair comparison)
3. **Test multiple mixing ratios**
4. **REDD validation**

### ✅ Your Unique Claims:
1. "First to use **time-conditioned** diffusion for NILM"
2. "**Ordered non-overlapping** ensures full temporal coverage"
3. "Validated on **temporal-aware** models (NILMFormer)"

### Total Time: 5-6 weeks

**您的工作和他们的work角度不同，可以共存！您focus on temporal alignment，他们focus on noise robustness。**

**要我帮您实现Context-FID和SWD metrics吗？**
