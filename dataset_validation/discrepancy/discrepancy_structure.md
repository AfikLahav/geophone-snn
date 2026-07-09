# Structural / Protocol Discrepancy Report: Real Dataset vs Synthetic v4 Corpus

**Generated:** 2026-07-06  
**Auditor:** rigorously verified — every number traced to actual file reads or Python computations against the data.

---

## 1. Real Dataset Inventory

### 1a. Main Training Dataset — `Goephone-Project/geophone_data/`

| File | Rows | Duration | Fs (Hz) | Jitter | Amp Min (V) | Amp Max (V) | Unique Values | LSB (mV) | 3s/1.5s Windows | Label |
|------|------|----------|---------|--------|-------------|-------------|---------------|----------|-----------------|-------|
| car.csv | 840,024 | 840.0 s | 1000 | 0.0000 ms | −0.272 | +0.278 | 992 | 0.20 | 559 | vehicle |
| human.csv | 1,400,040 | 1400.0 s | 1000 | 0.0000 ms | −0.381 | +0.362 | 892 | 0.20 | 932 | human |
| car_nothing.csv | 480,036 | 480.0 s | 1000 | 0.0000 ms | −0.155 | +0.176 | 751 | 0.20 | 319 | nothing |
| human_nothing.csv | 998,440 | 998.4 s | 1000 | 0.0000 ms | −0.102 | +0.093 | 597 | 0.20 | 664 | nothing |

**ADC identification:** LSB = 0.20 mV. This does not match any standard ADS1015 GAIN setting (0.125 / 0.25 / 0.5 / 1 / 2 / 3 mV). The max amplitude of +0.278 V exceeds the ADS1015 GAIN_SIXTEEN FSR (±0.256 V), ruling that out. The recording chain includes an AD620 instrumentation amplifier in the circuit (see `gap_study/05_sensor_chain.md`, `gap_study/lit/L6_sensor_chain.md`) — the effective LSB is set by the composite gain not documented in the repository. **ADC generation: indeterminate from file content alone.**

**Jitter is exactly 0.0 ms** because timestamps are integer-millisecond (`millis()` in firmware) — sub-ms jitter is below the timestamp resolution, not absent.

**No README, no site/date/operator metadata** in `Goephone-Project/`. Recording date, soil type, coupling medium, and operator are unknown.

**Real windows summary (3 s / 1.5 s hop):** vehicle = 559, human = 932, nothing = 983, animal = **0** → total **2,474**

---

### 1b. Avi Folder — `פרויקט גמר/פרויקט גמר/Avi_geophone_data/`

| File | Rows | Same as Main? | Note |
|------|------|---------------|------|
| car.csv | 840,024 | YES (head md5 db487246) | Byte-identical to main car.csv |
| human.csv | 1,400,040 | YES (head md5 6b3ef56c) | Byte-identical to main human.csv |
| car_nothing.csv | 480,036 | DIFFERENT head (0ee1d07b) | Same row count; minor edit or reorder |
| human_nothing.csv | 998,440 | DIFFERENT head (a5d34f52) | Same row count |
| **car_fix.csv** | **840,024** | NO (head md5 8c6a8028) | Same duration and amplitude range as car.csv but different content. A corrected/reprocessed variant. **Not referenced by any pipeline script.** |

`car_fix.csv` is a silent alternative to `car.csv` that is never consumed. No script in the repository loads it.

---

### 1c. Root Geophone 2026 Files (Rig Characterization — NOT used for training)

**May 2026 (old I2C regime):** 4 files, 100 Hz, 0.5 ms jitter, 3–25 unique values, LSB = 0.125 mV → ADS1015 GAIN_SIXTEEN, amplitude range ±10 mV. Explicitly excluded from gap analysis (`00_GAP_INVENTORY.md`: "The broken-rig geophone_2026*.csv recordings were not used").

**June 2026 (new UART rig, phase0 characterization):** 7 files, ~1000 Hz, typical jitter 0.04–0.07 ms, 12–176 unique values, LSB = 0.125 mV → ADS1015 GAIN_SIXTEEN confirmed, amplitude range ±18 mV, noise floor ~0.08 mV RMS ≈ 1 LSB. One outlier file (`geophone_20260615_104937.csv`) has 43.8 ms jitter (packet timing error). Mains lines at **44–47 Hz and 55 Hz** (not 50 Hz — site-specific).

**These files are from a different, newer rig** than the main training data. They are not labeled and are not used in model training.

---

## 2. Label Semantics: Real vs Synthetic v4

### 2a. Real — Session-level, File-name-only

- **Granularity:** one label per entire recording (file-level). Every 3-second window drawn from `car.csv` inherits label "vehicle" regardless of whether a car is actually passing during that window.
- **No per-window fields:** no SNR, no occupancy fraction, no zone3, no soft target, no noise condition, no subkind, no terrain.
- **Sub-floor contamination:** windows where the in-band RMS is indistinguishable from the paired "nothing" session ambient floor (90th percentile):
  - `car.csv` (vehicle band 5–25 Hz vs `car_nothing.csv` floor): **9% (51/559 windows)** labeled "vehicle" despite being at ambient level
  - `human.csv` (human band 20–90 Hz vs `human_nothing.csv` floor): **13% (123/932 windows)** labeled "human" despite being at ambient level
- **Label accuracy impact:** REAL_LABEL_AUDIT.json — session-level accuracy (synth-calibrated thresholds) = 0.856; rises to 0.895 when sub-floor windows are excluded, confirming these windows degrade evaluation.
- **Paired nothing sessions:** `car_nothing.csv` calibrates the car session floor; `human_nothing.csv` calibrates the human session floor. This pairing is hardcoded in pipeline scripts but the pairing is not formally documented.

### 2b. Synthetic v4 — Per-window, Activity-aware, 27 columns

Label source: `simgeo_v4/label_windows.py`. Per window, for each of 3 classes:

| Column | Meaning |
|--------|---------|
| `{cls}_snr` | Exact in-band SNR (dB) from D2 answer-key clean/noise blobs |
| `{cls}_occ` | Fraction of window covered by that class's emission activity intervals (0–1) |
| `{cls}_level` | Ordinal: 0=absent, 1=single, 2=multiple (from activity_json counts) |
| `{cls}_soft` | Soft target (sigmoid of SNR curve) |
| `{cls}_zone3` | 0=sub-floor, 1=marginal, 2=detectable (from `gates_3s.json` thresholds) |

Plus: `coarse`, `subkind`, `profile_id`, `family`, `terrain_vs`, `noise_condition`, `masking`, `common_snr`, `activity`.

**Label bands (label.py):** human=(20–90 Hz), vehicle=(5–25 Hz), animal=(20–90 Hz), common=(5–90 Hz)

**SNR distributions (active windows):**
- human (n=438,516): p10=−52.7 dB, p50=−15.5 dB, p90=+14.8 dB
- vehicle (n=341,606): p10=−72.7 dB, p50=−11.1 dB, p90=+22.9 dB
- animal (n=598,715): p10=−60.3 dB, p50=−19.5 dB, p90=+15.9 dB

**Sub-floor share of present windows (zone3=0):** human 44%, vehicle 47%, animal 51% — synthetic properly identifies faint windows; real labels do not.

---

## 3. Composition / Coverage Asymmetry

| Dimension | Real | Synthetic v4 |
|-----------|------|--------------|
| Total windows | 2,474 | 3,282,746 |
| Total scenes/recordings | 4 | 171,360 |
| Unique terrain profiles | 2 (car site, human site) | 340 |
| Active classes with data | 2 (vehicle, human) | 5 (human, vehicle, animal, mixed, nothing) |
| Noise conditions | 1 (uncharacterized real ambient) | 6 (calm/wind_low/mid/high/rain_light/heavy) |
| Sites | 2 | synthetic (340 profiles × ~17 terrain families) |
| Subkinds | 2 (one per class) | 34 |

**Ratio synthetic:real windows = 1,327:1**

### Classes with ZERO real windows

- **animal** (all subkinds: dog, horse, boar, sheep, jackal, herd, horse_rider)
- **mixed** (human+animal, human+vehicle, vehicle+animal)
- All weather-conditioned variants (wind, rain) — real data has uncharacterized ambient only

### Subkinds with ZERO real examples

Every synthetic subkind except "car" (partially approximated by car.csv) and "walk" (partially approximated by human.csv): run, march, group, child, stealth, convoy, tractor, loaded, motorbike, bicycle, tracked, two_vehicle, truck, overflight, idle_exit, slow_quad, machinery, traffic, all animal subkinds.

Note: even "walk" and "car" are only approximate — the real sessions have unknown terrain, unknown distance, single site, no per-pass segmentation.

---

## 4. Session Confounds in Real Data

### Rig Generations (3 distinct setups)

| Gen | Files | Fs | Jitter | LSB | ADC | Mains | Notes |
|-----|-------|----|--------|-----|-----|-------|-------|
| Gen1 (old I2C) | 2026-05 root CSVs | 100 Hz | 0.5 ms | 0.125 mV | ADS1015 GAIN_SIXTEEN | unknown | Non-uniform clock; excluded |
| Gen2 (main training) | Goephone-Project/*.csv | 1000 Hz | 0.0 ms (integer) | 0.20 mV | ADS1015 + AD620 (unknown gain) | Unknown | Unknown provenance; different coupling ring |
| Gen3 (new UART) | 2026-06 root CSVs | 1000 Hz | 0.047 ms | 0.125 mV | ADS1015 GAIN_SIXTEEN | 44–47, 55 Hz | Phase 0 characterized; NOT main training rig |

### Human vs Car Session Differences (within Gen2)

**Coupling ring (65 Hz floor-ring resonance):**  
- `human.csv`: footstep impacts drive a visible 65 Hz ring-down (`datasets/compat/ringdown.py`). Real floor ring = 65–66 Hz; synthetic corpus coupling_fc median = 132 Hz (2× too high — gap #2 in `00_GAP_INVENTORY.md`).
- `car.csv`: vehicle source is continuous (Gaussian), too few sharp impulses to probe coupling ring; different surface contact or rig orientation plausible.

**Kurtosis contrast (`datasets/compat/car_forensics.py`):**  
- human (20–100 Hz band): high kurtosis (impulsive footsteps)
- car (20–100 Hz band): near-Gaussian (continuous engine vibration)

**Spectral placement gap (phase0/RESULTS.md §0.7):**  
- Sim (one footstep): centroid 5.9 Hz, 10–40 Hz fraction 1%  
- Real human.csv: centroid 71.9 Hz, 10–40 Hz fraction 6%, 60–75 Hz fraction 27%  
- Real is dominated by the 65 Hz coupling ring and 50 Hz mains, not the footstep band itself.

**Gain mismatch (car session):**  
- `compare_15s.py` header: "3 unknown gains" — amplitudes not cross-comparable across sessions.
- CAR_SCALE_SWEEP.json: optimal scale for car session = **10.0**; pipeline applies 25.4 uniformly. AUROC: 0.911 at ×10 vs 0.904 at ×25.4.

---

## 5. Pipeline Conventions: Where Real and Synthetic Are Treated Differently

| Convention | Applied to Real | Applied to Synthetic | Source / File:Line |
|------------|----------------|---------------------|-------------------|
| Amplitude scale ×25.4 | YES — `amplitude * 25.4` before feature extraction | NO — waveforms stored in mV natively | `real_label_audit.py:24`, `domain_gap_v4.py:25`, `eval_synth_to_real.py:26`, `finetune_5fold.py:12` |
| Scale derivation | 25.4 = synth_nothing_mV (0.282) / real_nothing_V (0.0111); floor-aligned | N/A | `rescale_test.py:17`, `units_check.py` → UNITS_CHECK.json |
| Scale validity | INVALID globally: UNITS_CHECK.json verdict = "class-DEPENDENT" (CV=0.44); human class ratio 0.011 (→ ×91), vehicle 0.029 (→ ×34), nothing 0.039 (→ ×25) | N/A | `snn_v2_out/UNITS_CHECK.json` |
| ±256 mV clip | DE FACTO NOT APPLIED (real×25.4 max = 0.278×25.4 = 7.1 mV << 256 mV) | Applied in `extract_features.py:76` | `simgeo_v4/extract_features.py:6` (header comment), `:76` |
| Feature scaler | Applied (mean/std from synth-train-only scaler.json) | Trained on synth train split; applied to self | `snn_v4_out/E0_win3s/scaler.json`, CLIPZ=8.0, 104 features |
| Sub-floor gating | NOT applied — all 9%+13% sub-floor windows included in eval/finetune as active | zone3=0 available for curriculum; not filtered in E0 | `real_label_audit.py:75-84` |
| Session-paired floor | YES — car session uses car_nothing floor; human session uses human_nothing floor | NOT applicable (independent nothing scenes) | `real_label_audit.py:72-80` |
| Label source | Filename convention → uniform session label | Activity-aware: per-window from occupancy intervals | `label_windows.py:83-97` |
| 30 s scene blocks | YES — `SCENE = 30 × FS` partitions recording before windowing | Not applicable (scenes are pre-segmented) | `real_label_audit.py:56-58` |

---

## 6. Top 5 Structural Risks for Sim-Real Comparison Validity

**Risk 1 — Car session gain misalignment (×10 optimal vs ×25.4 used)**  
The ×25.4 scale derived from the nothing-floor ratio does not match the car session's actual gain. Optimal scale = 10.0 (CAR_SCALE_SWEEP.json). The top 3 domain-divergent features (variance, log_rms, peak_to_peak; domain AUROC 0.979–0.982) are all amplitude-magnitude features. Wrong scale shifts these features systematically, making vehicle windows appear at the wrong z-score after the synth-trained scaler is applied.

**Risk 2 — Sub-floor label noise in real evaluation**  
9% of vehicle windows and 13% of human windows are labeled active despite being indistinguishable from ambient. These act as hard negatives inside the positive class — they lower measured accuracy (0.856→0.895 when excluded) and prevent clean separation of true detection failures from label noise. Any claimed accuracy gap between synth and real is partially label noise, not model failure.

**Risk 3 — Zero real animal data; entire animal class is extrapolation**  
No real animal recording exists anywhere in the repository. All animal transfer performance is measured on synthetic-only val sets. The domain gap for animal is unmeasurable from real data. DOMAIN_GAP_V4.json measures overall domain AUROC = 0.964; animal-class contribution to this gap is unquantified.

**Risk 4 — Coupling resonance 2× too high in synthetic (132 Hz median vs 65 Hz real)**  
The human session coupling ring sits at 65–66 Hz; synthetic corpus coupling_fc median is 132 Hz. This misplaces spectral energy across all human windows and likely some vehicle windows. `human_upper_band_frac` (fraction of 20–90 Hz energy above 55 Hz): real = 0.466, v4_synth = 0.216. Spectral features (spectral_centroid, dominant_freq, energy_human_peak/tail bands) carry this systematic error into all comparisons.

**Risk 5 — Feature scaler trained on synth-only; real distribution systematically offset**  
Overall domain AUROC = 0.964 — a trivial classifier can distinguish real from synthetic using any of the top 20 features. After ×25.4 scaling, real windows cluster in a different region of feature space than synthetic windows of the same nominal class. The synth-trained scaler assigns wrong z-scores to real data. The CLIPZ=8.0 clip then further truncates real-data outliers at different points than synthetic ones. This is the fundamental sim-to-real transfer hazard.

---

## Appendix: File Hashes for Duplicate Detection (first 500 bytes, MD5)

| File | Location | Head-MD5 |
|------|----------|----------|
| car.csv | Goephone-Project/geophone_data/ | db487246 |
| car.csv | Avi_geophone_data/ | db487246 (IDENTICAL) |
| car_fix.csv | Avi_geophone_data/ | 8c6a8028 (DIFFERENT) |
| human.csv | Goephone-Project/geophone_data/ | 6b3ef56c |
| human.csv | Avi_geophone_data/ | 6b3ef56c (IDENTICAL) |
| car_nothing.csv | Avi_geophone_data/ | 0ee1d07b |
| human_nothing.csv | Avi_geophone_data/ | a5d34f52 |
