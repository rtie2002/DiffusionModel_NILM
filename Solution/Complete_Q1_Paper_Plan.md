# 6-Week Q1 Paper Development Plan

**Target**: Applied Energy (Q1, IF ~11.2) or IEEE Trans Smart Grid (Q1, IF ~9.6)

---

## Core Contributions

1. **Time-Conditioned Diffusion for NILM** - First to jointly generate power + temporal features
2. **Ordered Non-Overlapping Sampling** - Full temporal coverage strategy
3. **Complete Pipeline** - Diffusion → Algorithm 1 → Temporal NILM Integration
4. **Extensive Validation** - Multi-baseline, multi-dataset, comprehensive ablations

**Expected Impact**: ~16.6% average MAE improvement

---

## Week 1: Mathematical Foundation & Baseline Prep

### Day 1-2: Mathematical Formulation
- [ ] Write conditional diffusion equations (LaTeX)
  - $p_\theta(x_0^{\text{power}} | c^{\text{time}})$
  - Forward process (only noise power)
  - Training loss (power-only)
- [ ] Formalize ordered non-overlapping sampling
  - Indices: $\mathcal{I} = \{i \cdot L : i = 0, \dots, M-1\}$
  - Proof of full coverage
  - Zero overlap property
- [ ] Create comparison: Sliding vs. Non-Overlapping

### Day 3-4: Literature Review
- [ ] Read & summarize key papers
  - TimeGAN (NeurIPS 2019)
  - RCGAN (2017)
  - TimeGrad (NeurIPS 2021)
  - CSDI (NeurIPS 2021)
  - Original Unconditional DDPM (your baseline)
- [ ] Create comparison table (Method | Power | Time | Conditional | Code)
- [ ] Identify research gap

### Day 5-6: GAN Baseline Code
- [ ] Implement Conditional GAN
  - Generator: `(z, time) → power`
  - Discriminator: `(power, time) → real/fake`
- [ ] Write training script `train_conditional_gan.py`
- [ ] Test on small dataset

### Day 7: Documentation
- [ ] `mathematical_formulation.md` with LaTeX
- [ ] `literature_review.md`
- [ ] Week 1 summary

---

## Week 2: Core Experiments

### Day 1-2: Train Unconditional Baseline
- [ ] Use original author's code (C:\...\expample project\Diffusiom_model_original)
- [ ] Generate synthetic data (25%, 50%, 100%)
- [ ] Record results (all 5 appliances)

### Day 3-4: Train GAN Baseline
- [ ] Train Conditional GAN on UK-DALE
- [ ] Generate synthetic data (25%, 50%, 100%)
- [ ] Record results

### Day 5-6: Your Method (Already Done)
- [ ] Organize existing results
- [ ] Ensure all mixing ratios tested (0%, 25%, 50%, 100%, 200%)
- [ ] Verify reproducibility

### Day 7: Create Results Table
- [ ] Table: Baseline Comparison
  ```
  Method          | Dishw | Fridge | Kettle | Micro | Wash | Avg
  Real Only       | X     | X      | X      | X     | X    | X
  Uncond+50%      | X     | X      | X      | X     | X    | X
  GAN+50%         | X     | X      | X      | X     | X    | X
  Ours+50%        | X     | X      | X      | X     | X    | X
  ```
- [ ] Statistical significance tests (t-test, p-values)

---

## Week 3: Ablation Studies

### Day 1-2: Time Feature Ablation
- [ ] Experiment 1: Without time features (power-only generation)
- [ ] Experiment 2: With time features (your method)
- [ ] Quantify improvement from time conditioning

### Day 2-3: Sampling Strategy Ablation
- [ ] Random sampling (existing if you have it)
- [ ] Ordered sliding (stride=1)
- [ ] Ordered non-overlapping (stride=512) ← Your method
- [ ] Compare coverage & quality

### Day 4-5: Design Validation
- [ ] **Shuffling Experiment**
  - Mix real+synthetic, shuffle → Test
  - Mix real+synthetic, keep order → Test
  - Measure performance gap
  
- [ ] **Continuity Experiment**
  - Algorithm 1 ON-only → Test
  - Whole dataset continuous → Test
  - Measure performance gap

### Day 6-7: Analysis & Visualization
- [ ] Create ablation table
- [ ] Write analysis: Why each design choice matters
- [ ] Draft Section 5.3: "Design Validation"

---

## Week 4: Multi-Dataset Validation

### Day 1-2: REDD Dataset Prep
- [ ] Download REDD dataset
- [ ] Adapt preprocessing scripts
  - Extract same 5 appliances (if available)
  - Generate time features
  - Split train/test

### Day 3-4: REDD Experiments
- [ ] Train your diffusion model on REDD
- [ ] Generate synthetic data
- [ ] Train NILMFormer on REDD
- [ ] Record results

### Day 5: Cross-Dataset Analysis
- [ ] Compare UK-DALE vs. REDD results
- [ ] If results differ, analyze why (dataset characteristics)
- [ ] Create comparison figure

### Day 6-7: Optional - Third Dataset
- [ ] If time permits: ECO or REFIT
- [ ] Otherwise: Strengthen UK-DALE + REDD analysis

---

## Week 5: Theory & Visualization

### Day 1-2: Mutual Information Analysis
- [ ] Implement MI calculation
  ```python
  from sklearn.feature_selection import mutual_info_regression
  MI = mutual_info_regression(time_features, power)
  ```
- [ ] Compute for each appliance
- [ ] Create MI table
- [ ] Write explanation: "Why time features provide information"

### Day 3-4: Create Figures (6+ total)
- [ ] **Figure 1**: Temporal Distribution
  - (a) Real data - month histogram
  - (b) Unconditional - skewed months
  - (c) Ours - uniform months

- [ ] **Figure 2**: Sample Quality
  - (a) Real sample
  - (b) GAN sample
  - (c) Unconditional DDPM
  - (d) Ours

- [ ] **Figure 3**: Performance Comparison (Bar chart)
  - X-axis: Appliances
  - Y-axis: MAE
  - Bars: Real, Uncond, GAN, Ours

- [ ] **Figure 4**: Mixing Ratio Analysis
  - X-axis: Synthetic %
  - Y-axis: MAE
  - Lines: Different appliances

- [ ] **Figure 5**: Continuity Impact
  - Comparison: Fragmented vs. Continuous

- [ ] **Figure 6**: Shuffling Impact
  - Comparison: Shuffled vs. Ordered

### Day 5-6: Failure Case Analysis
- [ ] Deep dive: Dishwasher Window=256 anomaly
  - Visualize generated samples
  - Compare ON/OFF ratios
  - Statistical analysis
  - Propose explanation
- [ ] Write Discussion subsection

### Day 7: Organize Figures
- [ ] All figures high-resolution (300 DPI)
- [ ] Consistent style (fonts, colors)
- [ ] Captions written

---

## Week 6: Paper Writing

### Day 1: Structure & Outline
- [ ] **Title**: "Temporally-Coherent Diffusion Models for Non-Intrusive Load Monitoring Data Augmentation"
- [ ] **Abstract** (250 words)
- [ ] Section outline:
  1. Introduction (2 pages)
  2. Related Work (2 pages)
  3. Preliminaries (1.5 pages)
  4. Method (3 pages)
  5. Experiments (5 pages)
  6. Analysis (2 pages)
  7. Discussion (1 page)
  8. Conclusion (0.5 page)

### Day 2-3: Write Intro, Related Work, Method
- [ ] Introduction
  - NILM background
  - Data scarcity problem
  - Our solution
  - **4 Contributions** (list explicitly)
  
- [ ] Related Work
  - NILM methods
  - Data augmentation for time series
  - Diffusion models
  - **Gap**: No prior work on time-conditioned + ordered sampling
  
- [ ] Method
  - Problem formulation
  - Architecture
  - Training procedure
  - Ordered non-overlapping sampling

### Day 4-5: Write Experiments & Results
- [ ] Experimental Setup
  - Datasets (UK-DALE, REDD)
  - Models (NILMFormer, etc.)
  - Baselines (GAN, Unconditional)
  - Metrics (MAE, F1)
  
- [ ] Main Results
  - Baseline comparison (Table + analysis)
  - Statistical significance
  
- [ ] Ablation Studies
  - Time feature ablation
  - Sampling strategy ablation
  - Design validation (shuffle, continuity)
  
- [ ] Multi-Dataset Results
  - REDD validation
  - Generalization analysis

### Day 6: Write Analysis, Discussion, Conclusion
- [ ] Analysis
  - Mutual Information findings
  - Qualitative analysis (visualizations)
  - Failure cases
  
- [ ] Discussion
  - Why our method works
  - Limitations (honest)
  - Future work
  
- [ ] Conclusion
  - Summarize contributions
  - Impact statement

### Day 7: Polish & Format
- [ ] Proofread entire paper
- [ ] Check all references
- [ ] Ensure all figures/tables cited
- [ ] Format according to journal guidelines
- [ ] Write highlights (5 bullet points)
- [ ] Prepare graphical abstract
- [ ] Write cover letter

---

## Deliverables Checklist

### Code
- [ ] Clean, documented codebase
- [ ] `train_conditional_gan.py`
- [ ] Preprocessing scripts for REDD
- [ ] Reproducibility README

### Data
- [ ] UK-DALE preprocessed
- [ ] REDD preprocessed
- [ ] Generated synthetic datasets (all baselines)
- [ ] Trained model checkpoints

### Results
- [ ] All experiment results (CSV/Excel)
- [ ] Statistical test outputs
- [ ] Ablation study results

### Paper
- [ ] Complete draft (~18 pages)
- [ ] 6+ figures (high-res)
- [ ] 4+ tables
- [ ] LaTeX source files
- [ ] Supplementary material (if needed)

### Submission Materials
- [ ] Cover letter
- [ ] Highlights (5 bullets)
- [ ] Graphical abstract
- [ ] Author info

---

## Success Criteria

### Minimum (70% acceptance confidence)
- ✅ 2 datasets
- ✅ 2 baselines (Unconditional + GAN)
- ✅ 5 appliances
- ✅ Ablations complete
- ✅ Statistical tests

### Target (85% acceptance confidence)
- ✅ All minimum +
- ✅ 3 baselines
- ✅ Mutual Information analysis
- ✅ Cross-dataset validation
- ✅ Comprehensive visualization

---

## Time Tracking Template

```markdown
## Week X Progress

### Completed
- [x] Task 1
- [x] Task 2

### In Progress
- [ ] Task 3 (50% done)

### Blocked
- Issue: [description]
- Solution: [next steps]

### Next Week Priority
- Critical task for Week X+1
```

---

**Estimated Total Effort**: 160-200 hours (6 weeks full-time equivalent)

**Target Submission Date**: Week 7, Day 1
