> **Research survey — house-localization pivot ("wallhacks").** Background research agent (Sonnet), 2026-07-08. Literature/tooling survey; design input, not a validated experiment. Indexed in [README.md](README.md).

---

# Architecture Recommendation: Indoor Seismic Classification and Localization

## Background on the Problem Scope

The pivot from a 340-terrain single-geophone outdoor corpus to N geophones on one concrete/brick house is a **scope compression of roughly 10–100×**. Fewer surface types, bounded geometry, fixed mounting positions, and a known propagation medium together mean: (a) the feature manifold is far smaller, (b) training data is more likely to be sparse, and (c) physics-based shortcuts become viable. The rationale for a 220K-param SNN (edge power, temporal sparsity, large diversity) disappears in this setting. The recommendations below build the case bottom-up from precedent.

---

## 1. Classification Model: Type and Size

### What the literature says

Geophone/seismic footstep/intrusion classification with classical ML consistently reaches **92–98%** with an engineered feature bank:

- **SVM (RBF kernel) on time + frequency features**: 95–97% accuracy for human/vehicle/animal discrimination on geophone data ([Intrusion Detection Using Seismic Signals and Classification with SVM, ResearchGate](https://www.researchgate.net/publication/315605199_Intrusion_Detection_Using_Seismic_Signals_and_Classification_with_Support_Vector_Machine)). A wavelet-denoised 13-feature set (temporal + spectral) drove the reported SVM result for physical security intrusion.

- **Random Forest**: 92.76% on multi-class seismic (pedestrian / bicycle / vehicle) in the marine-seismological literature ([Event Recognition in Marine Seismological Data, GJI 2023](https://academic.oup.com/gji/article/235/1/589/7199654)). RF-with-EMD+DWT reached 100% in a controlled 5-class problem, though that was laboratory conditions.

- **Gradient Boosting (XGBoost, LGBM)**: 93.5–94.2% accuracy on microseismic multi-class signal ([Ensemble Learning Improves Microseismic Classification, PMC 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11314873/)). The EML-PSP ensemble physical-security paper (2023, ScienceDirect) combined augmentation + ensemble boosting specifically for geophone-based intrusion classification, reporting further improvement over individual classifiers.

- **1D-CNN on raw waveforms**: 95–99.8% on microseismic; CNN trained end-to-end on geophone data for earthquake/vehicle/noise separation reached >99% per class ([Deep Learning-Based Earthquake and Vehicle Detection, Springer 2024](https://link.springer.com/article/10.1007/s10950-024-10267-8)). The gap over classical ML ranges from **0.5 to 6 percentage points** depending on dataset size and problem difficulty.

The critical meta-finding for your scope is from the large tabular data benchmark ([Grinsztajn et al., arXiv:2402.03970v2](https://arxiv.org/html/2402.03970v2)):

> *"On small-sized and medium-sized tabular datasets with ≤10,000 samples, CatBoost and XGBoost demonstrate robust and consistent performance (median ranks 2 and 2.5). TabPFN is statistically superior at ≤1,000 instances. Dataset-specific neural networks outperform at large scale but do not consistently dominate GBDTs."*

For your problem the dataset is unlikely to exceed a few thousand labeled footstep events. The feature space is clearly tabular (engineered per-step descriptors), not raw-waveform. **This places it squarely in the GBDT-wins zone.**

### Recommendation

**Use LightGBM (or XGBoost) on an engineered per-step feature bank.** Justifications:

1. Tabular-data regime with estimated N < 10,000 labeled events.
2. 3-class problem (human / animal / nothing) is narrower than your original 340-terrain design—less representational capacity is needed.
3. LightGBM gives Shapley-value feature importances, which directly addresses the confound reviewer concern ("is your classifier responding to step frequency or floor resonance?").
4. Inference is microsecond-class; no GPU needed at the base station.
5. Matches or beats small neural networks at this scale per benchmark.

A **small MLP (2 hidden layers, 64–128 units, ~10–30K parameters)** is worth running in parallel as a sanity check. If it beats LightGBM by >2 percentage points on held-out data, escalate. Otherwise, keep LightGBM for publication interpretability.

**Do not use the SNN for classification in the indoor setting.** The SNN's energy-efficiency rationale was edge-node local inference across 340 terrains. With base-station streaming, edge power is no longer binding; 340-terrain diversity is gone; and the SNN's representational advantage disappears in a narrow 3-class tabular problem.

**Feature bank (per sensor, per detected step event):**

- Time-domain: RMS energy, peak amplitude, zero-crossing rate, rise-time (10→90%), duration, skewness, kurtosis (~7 features).
- Frequency-domain: dominant frequency, spectral centroid, spectral entropy, 5–8 sub-band energy ratios (0–5 Hz, 5–20 Hz, 20–80 Hz, 80–200 Hz), peak-to-RMS ratio in each band (~13 features).
- Optional MFCC-analog: 8–12 cepstral coefficients from the footstep spectrum (gives ~10 features).

Total per-sensor: ~30–40 features. Aggregate over N sensors → ~30N + inter-sensor features. For N = 4–8 sensors this stays well under 300 features, which GBDT handles natively.

---

## 2. Is a Classification Gate Needed?

### The argument for mandatory gating

**Yes, a classification gate is necessary.** The logic is identical to detection thresholding in radar before direction-of-arrival estimation: you do not run DoA on noise.

Indoor environments contain persistent vibration sources that are spectrally overlapping with human footsteps:

- HVAC / AC units: 5–50 Hz, quasi-periodic, spatially fixed.
- Pet activity (cats, dogs): step frequency 2–5 Hz, amplitude lower than humans but not negligible.
- Appliance vibration (washing machine, refrigerator): 25–50 Hz, fixed location.

Without gating, a localizer trained on human footsteps will misfire on any of these, producing spurious position estimates. Worse, for a security/safety application, pet-triggered "person present" alarms are exactly the confound you must control for.

The gating role has two components:

1. **Spurious-source rejection**: suppress HVAC, appliances, pets from triggering the localizer.
2. **Event labeling**: tag the localized event with class so downstream logic can differentiate "animal at position X" (log only) vs. "human at position X" (act).

### Multi-task (joint head) vs. two-stage pipeline

**Recommendation: two-stage pipeline, with optional joint feature extraction.**

| | Multi-task (shared backbone) | Two-stage pipeline |
|---|---|---|
| Classification input | Per-step feature vector from all sensors | Same |
| Localization input | Same feature vector, different head | Same vector OR a separate inter-sensor feature set (energy ratios, TDOA estimates) |
| Coupling | Gradient from localizer updates the shared feature extractor | None — classifier triggers localizer independently |
| Advantage | Shared features can regularize localization head; single inference pass | Simpler, independently trainable, classifier can be updated without retraining localizer |
| Risk | Localizer gradient corrupts features useful for class discrimination; requires simultaneous labeled data for both tasks | Extra inference latency (two models, sequential) |
| When to prefer | If labeled data for both tasks is abundant and collected simultaneously | Default choice; especially when class labels are easier to collect than position labels |

The multi-task precedent exists ([LSTM + joint training for vibration damage classification + localization](https://www.sciencedirect.com/science/article/abs/pii/S2352012421010493)), and in seismology (magnitude + location from transformer, arXiv:2101.02010), but those applications have correlated targets. For your problem, classification features (spectral shape of step) are partially shared with localization features (inter-sensor timing/energy) but not fully: gait class does not inform step position, and position does not inform animal vs. human class. The shared representation is shallow.

**Practical path**: Train LightGBM classifier independently. Use its "human" output as a trigger. Separately train the localizer on human-only steps. If you have time and labeled data: also try a small shared-trunk MLP → two output heads (class probabilities + x,y) and compare OOB performance. The two-stage pipeline is the publication-defensible default.

---

## 3. Localization Model

### Physics-based classical estimators

**Time Difference of Arrival (TDOA) via cross-correlation (GCC-PHAT)** is the natural starting point. Problems:

- Floor vibration is **highly dispersive**: group velocity varies with frequency in concrete/brick (Rayleigh wave dispersion). Different spectral components of the same footstep arrive with different delays, smearing the cross-correlation peak.
- Mirshekari et al. 2018 ([Occupant Localization via Footstep-Induced Structural Vibration, Mechanical Systems and Signal Processing](https://www.sciencedirect.com/science/article/abs/pii/S0888327018302280)) addressed this with wavelet narrowbanding (WT-TDOA): isolate narrowband frequencies where dispersion is minimal, then compute cross-correlation. Result: **0.34 m mean localization error** in a single-story concrete/wood structure with 4 sensors.

**Energy-based localization** (no TDOA needed): fit an amplitude-distance decay model (energy ∝ distance^β) from multiple sensors → solve least squares for position. The multi-sensor stochastic energy-based paper ([PMC10708826](https://pmc.ncbi.nlm.nih.gov/articles/PMC10708826/)) with Byzantine sensor elimination achieved **0.99–1.58 m mean error** using 11 accelerometers. Simpler than TDOA but less accurate.

**Matched Field Processing (MFP)**: scan a grid of candidate positions; for each, compute the predicted Green's function (wave propagation to each sensor); correlate against the observed multi-channel signal; the position with maximum correlation is the estimate. This is the physics-optimal solution when the Green's function is known. For homogeneous concrete/brick, analytical Rayleigh wave Green's functions are tractable. For heterogeneous walls and joints, numerical FEM/FDM simulation is needed but feasible.

### Learned estimators

**1D-CNN for TDOA refinement**: the SPIE 2024 paper (Pedestrian footstep localization using deep convolutional network for TDoA estimation) used a 1D-CNN on synchronized geophone time-series to estimate TDoA, achieving <1 m accuracy for single pedestrians with a 60%+ improvement over baseline cross-correlation TDOA. The CNN learns to denoise and sharpen the TDOA estimate, compensating for dispersion that fools GCC-PHAT.

**Deep learning MDPI 2024 Step Vibration** ([Motion Target Localization via Step Vibration, MDPI 2024](https://www.mdpi.com/2076-3417/14/20/9361)): deep learning reduced average positioning error to **0.376 m** versus **0.607 m** for WT-TDOA (−37.9%) and **0.502 m** for SAE-BPNN (−24.8%).

**Ridge regression on PCA of raw waveforms (reservoir computing)**: the 2026 arxiv paper ([arxiv:2603.04610](https://arxiv.org/html/2603.04610v1)) used 11 accelerometers + PCA compression + ridge regression (linear readout). Cross-participant RMSE: **0.98 m (longitudinal), 0.47 m (lateral)**. Key finding: *"Performance converges beyond 6 sensors, indicating moderate sensing suffices."* This is essentially a linear learner on physics-derived features — the building itself does the nonlinear computation.

**Fingerprint dictionary + KNN**: if you simulate the Green's function for a grid of positions → every sensor (say 0.1 m grid, 4–8 sensors, 10 features per sensor-position pair), the dictionary is small (10,000 × N × 10 ≈ 400K floats for a 10 m × 10 m room at 0.1 m resolution with 4 sensors and 10 features). KNN or nearest-centroid lookup over this dictionary is the **zero-shot baseline from simulation**. In RF indoor localization (the closest analog in terms of dictionary size and matching strategy), fingerprint + KNN achieved 1–3 m accuracy in WiFi, but structured simulation + KNN on vibration is more reliable because the physics model is better-specified.

### How much learning is needed when Green's functions are simulated?

The answer depends on model error (how well simulation matches reality):

- **Ideal (homogeneous medium, no joints, perfect sensor placement)**: simulation matches reality well → MFP or fingerprint KNN from simulation alone should localize within 0.3–0.5 m, no learning required.
- **Realistic (wall joints, embedded rebar, fixture masses, floor-to-wall coupling)**: simulation has systematic bias → **a small calibration layer** on top of the physics baseline corrects the residual. This is the **sim-to-real transfer** pattern.

Calibration architecture: run the physics localizer (MFP or fingerprint KNN) → record its output (predicted x,y) + prediction confidence → regress from (physics prediction, per-sensor feature bank) → (corrected x,y) using ridge regression or a 2-layer MLP. Training data: 100–500 real calibration footsteps at known positions (these are cheap to collect — walk a grid with a measuring tape). This approach is used in sonar matched-field processing and in RF fingerprinting sim-to-real transfer ([Using Synthetic Data to Enhance Fingerprint-Based Localization, arXiv:2105.01903](https://arxiv.org/pdf/2105.01903)).

### Recommendation

**Three-tier localization strategy:**

1. **Tier 1 (baseline, no training data needed)**: Simulated Green's function fingerprint dictionary → nearest-neighbor lookup. Gives ~0.5–1.5 m error depending on propagation model quality. This is your publication-quality physics baseline.

2. **Tier 2 (small calibration, 100–500 labeled steps)**: Correct the tier-1 output with a ridge regression or a very small MLP (2 layers × 32–64 units, ~2K parameters) trained on real calibration data. Expected error: 0.3–0.6 m. Matches the Mirshekari WT-TDOA result (0.34 m) and the reservoir-computing result (0.47–0.98 m).

3. **Tier 3 (more data, 500–5000 labeled steps)**: Replace the dictionary lookup with a direct regressor: LightGBM regressor or a 3-layer MLP (64–128 units, ~15–30K parameters) trained end-to-end from (per-sensor feature bank + inter-sensor features) → (x,y). Expected error: 0.2–0.4 m based on the deep learning MDPI 2024 result (0.376 m).

**Do not start with a CNN on raw waveforms** unless you have > 5000 labeled footsteps. The reservoir-computing paper showed that PCA + ridge regression with 11 sensors on only 12 traversals (~1000 steps total) reaches 0.65 m RMSE — a small linear model beats a CNN trained on insufficient data.

**Graph NN over the sensor array** is architecturally elegant (nodes = sensors, edges = inter-sensor features, message passing = spatial aggregation) but is premature for N < 12 sensors and under 5K training samples. Reserve for future work or multi-building generalization.

---

## 4. Model Size and Complexity Guidance

### When classical beats deep for this problem

The decision boundary is empirical but consistent across the tabular-data literature:

- **N < 1,000 labeled steps**: use LightGBM (classification) + ridge regression calibration (localization). Optionally try TabPFN for classification (in-context learner, excellent at N < 1K per arXiv:2402.03970v2).
- **1,000 < N < 10,000**: LightGBM still competitive; try a small MLP alongside. Use cross-validation to decide.
- **N > 10,000**: small MLP / 1D-CNN becomes competitive. But for a single-house corpus this is unlikely without simulation augmentation.

### Concrete size estimates

| Component | Model | Param count | Disk size | Inference latency |
|---|---|---|---|---|
| Classifier | LightGBM, 200 trees, depth-6 | N/A (tree-based) | ~200 KB | <1 ms |
| Classifier | MLP 2×64 on 100-feature input | ~8,600 | <50 KB | <1 ms |
| Localizer (tier 1) | KNN on simulation dictionary | 0 (lookup) | ~400 KB (dictionary) | ~5 ms |
| Localizer (tier 2) | Ridge regression | ~300 coefficients | <5 KB | <0.1 ms |
| Localizer (tier 3) | MLP 3×64 | ~12,000 | <50 KB | <1 ms |

The original SNN at 220K parameters is 10–25× larger than what any of these solutions requires. Model compression is not needed — you simply don't need that capacity.

### Edge vs. base station

With base-station streaming already planned, all models can run at the base station. The edge/low-power constraint that motivated the SNN is lifted. LightGBM + KNN dictionary runs comfortably on any ARM Cortex-A class device (Raspberry Pi) without a GPU. This simplifies deployment substantially.

### Interpretability for publication

LightGBM + engineered features gives:
- Shapley feature importances: "does the classifier use dominant frequency, energy, or inter-sensor timing?"
- Partial dependence plots for each feature
- Ablation by feature group (time-domain only, frequency-domain only)

This is directly answerable to reviewers asking: *"is your model exploiting a physical confound?"* A black-box CNN cannot offer this without post-hoc explainability wrappers (LIME, SHAP on CNN is less reliable than on GBT). For a publication-first project, the interpretability argument alone justifies keeping classical ML as the primary model.

---

## 5. Input Representation

### Per-sensor feature bank vs. raw waveforms vs. learned features

**For classification (3-class: human / animal / nothing):**

- Use per-sensor engineered features, then aggregate across sensors (mean, max, variance of each feature across the array).
- Why: a single-sensor feature bank already captures step spectral shape (which distinguishes human from cat from HVAC). The array adds robustness (majority vote, or concatenate per-sensor features).
- CNN on raw multi-channel waveforms is only beneficial if the classification signal is subtle enough to require joint spatio-temporal processing. For coarse 3-class separation, it is not — confirmed by SVM/RF results at 92–97% with simple features.
- **Recommended feature bank**: ~30–40 features per sensor (details in section 1). At N = 6 sensors: 180–240 features + ~15 inter-sensor features (energy ratios, TDOA pairs = C(N,2) = 15 pairs for N = 6). Total: ~200–260 features.

**For localization (regression to x,y):**

- The most informative input is the **inter-sensor representation**: energy at each sensor, TDOA estimates between sensor pairs, phase coherence, group velocity estimates from TDOA×frequency pairs.
- Per-sensor absolute feature values are less informative (they depend on source strength, which varies per person and shoe type); inter-sensor ratios are source-strength-invariant.
- For tier-2 and tier-3 learned localizers: input = [energy at each sensor / total energy, TDOA estimate for each sensor pair (GCC-PHAT), dominant frequency at each sensor, spectral entropy at each sensor]. At N = 6: 6 (energy ratios) + 15 (TDOA pairs) + 6 (dom. freq.) + 6 (spectral entropy) = 33 features. This is trivial for ridge regression or a small MLP.
- Raw multi-channel waveform input (N × T matrix) for a 1D-CNN is viable at tier 3 with > 2,000 labeled steps. It can learn a learned TDOA estimator (the SPIE CNN approach) that outperforms GCC-PHAT. Reserve this for the extended study, not the baseline publication.

**For the simulation fingerprint dictionary (tier 1 localization):**

- Store at each grid point: simulated [energy_per_sensor, phase_delay_per_sensor, predicted_TDOA_per_pair].
- Match real observations against this dictionary using L2 distance in the feature space (normalized to zero-mean, unit-variance before matching).
- This is equivalent to fingerprint-based positioning in the RF localization literature: KNN in a 33-dimensional feature space over 1,000–10,000 grid points is fast and robust.

### Summary of input flow

```
Raw per-sensor geophone traces (N sensors, sampled at Fs)
        │
        ▼
[Step detection: threshold on band-pass energy]
        │
        ▼
[Per-step feature extraction]
        ├─ Per-sensor: RMS, peak, dominant freq, spectral entropy, sub-band energy ×5
        ├─ Aggregated across sensors: mean/max/var of each per-sensor feature
        └─ Inter-sensor: energy ratios, GCC-PHAT TDOA estimates per sensor pair
        │
        ├──▶ [LightGBM classifier on per-sensor + aggregated features]
        │         │
        │    "human" predicted?
        │         │Yes
        │         ▼
        └──▶ [Localizer: KNN in simulation dictionary (tier 1) → ridge regression correction (tier 2)]
                  │
                  ▼
             (x, y) position estimate
```

---

## Verdict Summary

| Decision | Recommendation | Rationale |
|---|---|---|
| Classification model | **LightGBM on engineered feature bank** (~30–40 features/sensor); small MLP as competitor | Tabular regime, N < 10K labeled events; classical ML matches or beats DL per arXiv:2402.03970v2; interpretable for reviewers |
| SNN retention | **Drop for classification** in the indoor pivot | Edge-power rationale is gone with base-station streaming; 220K params is 25× overparameterized for this scope |
| Classification gate | **Mandatory two-stage pipeline** | HVAC/pets/appliances are constant indoor confounders; localizer must be gated on "human" class to avoid spurious position outputs |
| Multi-task joint model | **Optional secondary experiment** | Useful regularization if both labels collected simultaneously; do not replace the two-stage pipeline as primary |
| Localization baseline | **Simulated Green's function fingerprint + KNN** (zero training data) | Physics is well-specified for concrete/brick; MFP / fingerprint matching is the classical-first optimal strategy |
| Localization calibration | **Ridge regression on 100–500 real steps** (tier 2) | Corrects simulation-to-reality bias; expected 0.3–0.6 m RMSE based on Mirshekari (0.34 m) and reservoir-computing (0.47–0.65 m) results |
| Localization if data allows | **Small MLP (3×64 units, ~12K params) or LGBM regressor** (tier 3, 500–5000 steps) | Matches 0.376 m MDPI 2024 result; stays far smaller than 220K SNN |
| CNN on raw waveforms | **Defer; use for TDOA refinement only if tier 2 error > 0.5 m** | Insufficient data at deployment; CNN-for-TDOA (SPIE 2024 pattern) is the right use if pursued |
| Graph NN | **Future work only** | Overkill for N ≤ 12 sensors at this data scale; no demonstrated advantage over MLP in this regime |
| Input representation | **Engineered per-sensor feature bank + inter-sensor features** for both tasks; raw waveforms only for CNN TDOA refinement | Best accuracy-per-sample in sparse-data regime; interpretable; see section 5 feature list |
| Model size guidance | **< 50 KB for classifier (LightGBM); < 5 KB for calibration regressor; 400 KB for simulation dictionary** | 100–500× smaller than the SNN was; justified by scope compression from 340 terrains to 1 house |

---

## Key Cited Precedents

- [Mirshekari et al. (2018) Occupant Localization via Footstep-Induced Structural Vibration — 0.34 m error](https://www.sciencedirect.com/science/article/abs/pii/S0888327018302280)
- [Multi-Sensor Stochastic Energy-Based Vibro-Localization with Byzantine Sensor Elimination — 0.99–1.58 m (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10708826/)
- [Can a Building Work as a Reservoir: Footstep Localization with Accelerometer Networks — 0.47–1.13 m RMSE, linear readout (arXiv:2603.04610)](https://arxiv.org/html/2603.04610v1)
- [Motion Target Localization via Step Vibration Signals, Deep Learning — 0.376 m, −37.9% vs WT-TDOA (MDPI 2024)](https://www.mdpi.com/2076-3417/14/20/9361)
- [Pedestrian Footstep Localization Using a Deep CNN for TDoA Estimation — <1 m accuracy, +60% over baseline (SPIE 2024)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12950/3009826/Pedestrian-footstep-localization-using-a-deep-convolutional-network-for-time/10.1117/12.3009826.short)
- [Is Deep Learning Finally Better than Decision Trees on Tabular Data? — GBDTs competitive at N < 10K; TabPFN best at N < 1K (arXiv:2402.03970v2)](https://arxiv.org/html/2402.03970v2)
- [Ensemble Learning Improves Microseismic Signal Classification — RF/XGBoost/LGBM at 93.5–94.2% (PMC 2024)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11314873/)
- [Intrusion Detection Using Seismic Signals and Classification with SVM (ResearchGate)](https://www.researchgate.net/publication/315605199_Intrusion_Detection_Using_Seismic_Signals_and_Classification_with_Support_Vector_Machine)
- [Deep Learning Earthquake and Vehicle Detection via Geophone, >99% (Springer 2024)](https://link.springer.com/article/10.1007/s10950-024-10267-8)
- [GaitVibe+: Structural Vibration-Based Footstep Localization (arXiv:2212.03377)](https://arxiv.org/pdf/2212.03377)
- [Occupant-Detection Strategy Using Footstep-Induced Floor Vibrations — ACM DFHS 2019](https://dl.acm.org/doi/10.1145/3360773.3360881)
- [Using Synthetic Data to Enhance Fingerprint-Based Localization (arXiv:2105.01903)](https://arxiv.org/pdf/2105.01903)
- [Matched Field Processing Review, GJI 2022](https://academic.oup.com/gji/article/231/2/1268/6619059)
- [FLoc: Device-Free Passive Indoor Localization in Complex Environments](https://luwang-szu.github.io/paper/FLoc.pdf)
- [Survey of Sound Source Localization with Deep Learning, arXiv:2109.03465](https://arxiv.org/pdf/2109.03465)
- [SNNs for Low-Power Vibration-Based Predictive Maintenance — >97%, 3 orders of magnitude energy reduction on Loihi (arXiv:2506.13416)](https://arxiv.org/html/2506.13416v1)
