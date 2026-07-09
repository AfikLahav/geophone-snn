# V4 — model training & experiment registry

*2026-07-06. Tracking document for every planned training run and experiment on the v4
corpus. Companion to `V4_IMPLEMENTATION_PLAN.md` (build) and `GENERATION_PLAN_V4.md`
(design). Status column is the live tracker — update in place as runs complete.*

## 0. Shared protocol (applies to every run unless a row overrides)

- **Data:** `corpus_v4` (with source-gap/pause scenes + occupancy labels), features per
  window geometry `features_v4_<W>s`; splits = `splits_v3.json` (profile-held-out val).
- **Window geometries:** WIN ∈ {1, 2, 3, 5, 8, 10} s, HOP = WIN/2 (50% overlap).
  Per-geometry matched-filter gates `gates_<W>s.json` (τ shifts vs 3 s: +4.8 / +1.8 /
  0 / −2.2 / −4.3 / −5.2 dB).
- **Reference architecture** (unless varied): FeatureSNN 104 feats, widths (512,256,128),
  T=4, ~220k params, seed 0.
- **Reference training protocol (FROZEN 2026-07-07 after the E0.5 sampler sweep):**
  `GEO_GATED_EVAL=1 GEO_ORDINAL_MASK=1`, **balanced sampler OFF** (uniform draws — the
  corpus's native coverage suffices; balancing over-weights marginal windows and makes the
  vehicle head conservative). Real-side metrics per run: pure-transfer
  (`eval_synth_to_real.py`: threshold-free AUROCs + synth-calibrated 3-way) and, for
  candidate builds, the finetune_5fold chain.
- **Metrics recorded for EVERY run** (single JSON per run):
  gated val AUROC (own-geometry gates) · ungated AUROC · per-5 dB-bin stratified AUROC ·
  realizable SNR50/90 @ Pfa 1% per class · params · train wall-time · firing rates.
  Real-side (where applicable): zero-shot 5-fold acc · finetuned 5-fold acc ·
  FAR-on-nothing per head.
- **Output naming:** `snn_v4_out/<EXP_ID>/` (e.g. `snn_v4_out/E1_win3s/`), one
  `RESULT.json` + `history.json` each. Registry table below is the index.
- **Comparability rules:** never compare pooled AUROC across window geometries
  (different window populations) — cross-geometry comparisons use detection floors
  (SNR50/90) and Pd-vs-SNR curves only. Cross-model comparisons at fixed geometry may
  use gated AUROC + matched-bin table.

## 1. Prerequisites (block everything below)

| ID | Item | Status |
|---|---|---|
| P0.0 | C0 coupling-resonance verification on real recordings → `FC_ANCHOR_HZ` frozen (plan D5/B1.7) | pending |
| P0.1 | v4 corpus rendered (pause/gap scenes D4, per-class mixed blobs D2, activity_json D3, re-anchored coupling D5, 20 s min duration) + coverage gate PASS | pending |
| P0.2 | Phase L label passes (6 geometries, occupancy + zone3, T_COVER=36 s quota) + feature extraction per geometry | pending |
| P0.3 | Per-geometry gates derived (`derive_gates.py --T <W>`) | pending |
| P0.4 | Real CSVs re-windowed per geometry (+ ADS1115 rate question resolved: firmware actual SPS pinned; resample-to-1 kHz step defined for new recordings) | pending |
| P0.5 | Noise validation: synthetic "nothing" PSDs vs real per condition (re-run of the v2 `noise_condition_match` comparison on v4) | pending |

---

## 2. Experiment tracks

### E0 — v4 acceptance chain (P0 priority; gates the rest)

| ID | Run | Models | Pass criterion | Status |
|---|---|---|---|---|
| E0.1 | Pretrain @ 3 s (reference) | 1 | trains, gated AUROC non-degenerate | pending |
| E0.2 | Matched-bin A1 vs v2 (`stratified_auroc.py`) | 0 | every bin ≥ v2 same-bin − 0.01 | pending |
| E0.3 | finetune_5fold + snn_real_baseline @ 3 s | 2 chains | zero-shot ≥ 0.859; 5-fold ≥ 0.977 ± 0.014; human FAR ≤ 0.0075 (A2) | pending |
| E0.4 | Pd-vs-SNR curves per class @ Pfa 1% (detectability pattern on features_v4_3s) | 0 | reported; feeds gate iteration | pending |
| E0.5 | If A2 misses with healthy coverage: `GEO_STRATA_MIX` sweep (5 mixes) | ≤5 | recovers A2 before any corpus change | contingent |

### E1 — Window-size sweep (paper headline)

Reference arch at every geometry; pretrain only, real zero-shot evaluated per geometry.

| ID | WIN/HOP | Models | Notes | Status |
|---|---|---|---|---|
| E1.1 | 1 / 0.5 s | 1 | cadence features expected to degenerate (1–2 footfalls) | pending |
| E1.2 | 2 / 1 s | 1 | | pending |
| E1.3 | 3 / 1.5 s | 0 | = E0.1, reused | pending |
| E1.4 | 5 / 2.5 s | 1 | | pending |
| E1.5 | 8 / 4 s | 1 | | pending |
| E1.6 | 10 / 5 s | 1 | expected best on hard/faint cases (+5.2 dB integration) | pending |
| E1.7 | **Deliverable:** realizable SNR50/90 vs WIN, per class, against the −10·log10(T) matched-filter slope (one figure) + latency column | 0 | | pending |
| E1.8 | Winner geometry (+3 s baseline): full finetune_5fold on real | 2 chains | real 5-fold per geometry | pending |
| E1.9 | Occupancy-stratified eval @ 8 s & 10 s: recall vs occupancy bin (0–25/25–50/50–75/75–100%) — the "6 s nothing + 2 s detection" question, quantified | 0 | pending |

### E2 — Model size sweep (feeds ESP32 sizing)

At 3 s and 10 s geometries only. Widths / approx params:
tiny (64,32) ≈ 12k · small (256,128) ≈ 64k · reference (512,256,128) = 220k ·
large (1024,512,256) ≈ 1.4M.

| ID | Size × geometry | Models | Status |
|---|---|---|---|
| E2.1 | tiny/small/large @ 3 s (ref exists) | 3 | pending |
| E2.2 | tiny/small/large @ 10 s (ref from E1.6) | 3 | pending |
| E2.3 | **Deliverable:** accuracy-vs-params curve ×2 geometries; pick the ESP32 operating point (int8 size + expected margin) | 0 | pending |

### E3 — Architecture comparison: SNN vs non-SNN (same features)

| ID | Run | Models | Question | Status |
|---|---|---|---|---|
| E3.1 | Features-MLP (same 104 inputs, matched param count ~220k, same loss/labels) @ 3 s | 1 | does spiking cost accuracy? (paper-critical) | pending |
| E3.2 | Features-MLP @ 10 s | 1 | | pending |
| E3.3 | Features-GRU over T? (optional; only if E5 shows context matters) | ≤1 | contingent | pending |
| — | Classical baselines (logreg 0.986 / HGB 0.983 on real) already exist; re-report on v4 | 0 | | pending |

### E4 — Raw-waveform models (no feature engineering)

Input = raw mV window (3,000 @ 3 s / 10,000 @ 10 s samples, 1 kHz), same labels/gates.

| ID | Run | Models | Notes | Status |
|---|---|---|---|---|
| E4.1 | 1D-CNN @ 3 s (~0.3–0.5M params) | 1 | synthetic val + real zero-shot | pending |
| E4.2 | 1D-CNN @ 10 s | 1 | | pending |
| E4.3 | 1D-CNN large (~2M) @ winner geometry | 1 | capacity check | pending |
| E4.4 | **Key readout:** sim→real transfer of raw vs features — raw carries the full synthetic fingerprint (domain-classifier AUROC = 1.0), so degraded transfer is the expected risk; either outcome is a paper finding. Also decides the ESP32 firmware path (raw model ⇒ no on-device DSP) | 0 | pending |

### E5 — Temporal context (model memory across windows)

| ID | Run | Models | Notes | Status |
|---|---|---|---|---|
| E5.1 | Decision smoothing baseline: re-run hysteresis sweep on v4 (exists from v2) | 0 | cheap floor for "does context help" | pending |
| E5.2 | Feature stacking: input = current + N previous windows' features, N ∈ {1, 3} @ 3 s | 2 | input dim 208/416 | pending |
| E5.3 | **Streaming SNN:** no membrane reset between consecutive windows within a scene (train + eval scene-ordered) | 1–2 | SNN-native memory; identical to ESP32 deployment mode (continuous inference, no resets) | pending |
| E5.4 | Context runs scored specifically on: gap/pause scenes, marginal-zone windows, masked-mixed windows | 0 | where memory should pay | pending |

### E6 — Robustness / harder-training ablations

| ID | Run | Models | Notes | Status |
|---|---|---|---|---|
| E6.1 | Input-feature dropout p ∈ {0.05, 0.10, 0.15} @ 3 s | 3 | distinct from hidden Dropout(0.1) | pending |
| E6.2 | Channel augmentation at feature time: per-window gain jitter ±3 dB + band-energy dropout | 1 | mirrors sensor drift | pending |
| E6.3 | Readout: real zero-shot delta vs E0.1 (robustness should show on real, not synthetic) | 0 | | pending |

### E7 — Transfer asymmetry

| ID | Run | Models | Notes | Status |
|---|---|---|---|---|
| E7.1 | Real-only model (C, exists) evaluated on v4 gated val (h/v/n) | 0 | expected poor (2,353 windows, 2 sessions, no animal) — the asymmetry number | pending |
| E7.2 | Synth→real zero-shot per geometry | 0 | falls out of E1 rows | pending |

### E8 — Polyphony & hard-scene eval (plan Phase E)

| ID | Run | Models | Notes | Status |
|---|---|---|---|---|
| E8.1 | Cross-trigger matrix + masked-vs-solo recall + segment multi-label F1/ER (`eval_polyphony.py`) on E0.1 and E1 winner | 0 | CT off-diagonals ≤ 5% @ FAR-matched θ = reporting baseline | pending |
| E8.2 | Tier-1 aliasing scoreboard: slow_quad-vs-human, group-vs-quadruped, march, herd-continuum — per-subkind confusion table | 0 | the "non-trivial dataset" receipt | pending |
| E8.3 | Gap/pause scenes: false-drop rate during 1–3 s pauses (does the model hold or flicker) vs E5 context variants | 0 | | pending |

### E9 — Deployment sizing (flagged; after winners known)

| ID | Run | Notes | Status |
|---|---|---|---|
| E9.1 | int8 quantization of winner (E2 point): accuracy delta synthetic + real | ESP32-S3: 220k int8 ≈ 220 KB (SRAM-comfortable); ~0.9M MAC/window ≈ 2–5 ms @ 240 MHz; multi-M params need PSRAM (8 MB on N8R8), ~10–50 ms | later |
| E9.2 | Streaming-mode (E5.3) memory footprint + state carry test | deployment mode | later |

---

### Results logged (2026-07-07)

**E0 (v4 low-pass, 3 s) — DONE:** gated synth 0.895 (h .861/v .947/a .876); real zero-shot
0.838, fine-tuned 0.979±0.011 (≥ v2's 0.977 ✓, beats scratch +0.013 ✓); FAR human 0.0 ✓;
matched-bin A1: vehicle+animal ≥ v2, human 1–3% BELOW ✗. Artifacts: `snn_v4_out/E0_win3s/`.

**EC (coupling-bump experiment, corpus_v4b) — DONE:** physics fix validated (upper human
band 0.16→0.51 vs real 0.47; `exp_coupling.json`); synthetic gated 0.920 (human +3.3 vs E0);
real fine-tuned 0.978±0.011 (wash); PURE synth→real (below): net worse than E0.
Artifacts: `snn_v4_out/E1_coupling_bump/`, corpus_v4b.

**PURE synthetic→real (no fine-tune, no real threshold tuning; `eval_synth_to_real.py`,
`SYNTH_TO_REAL.json` per model):**

| threshold-free AUROC | v2 | v4 low-pass | v4 bump |
|---|---|---|---|
| human vs nothing | 0.973 | **0.985** | 0.967 |
| vehicle vs nothing | **0.973** | 0.873 | 0.914 |
| human vs vehicle | 0.894 | **0.925** | 0.872 |
| 3-way acc (synth-calibrated τ, 1% FAR) | 0.837 | **0.856** | 0.824 |
| nothing recall (false-alarm behavior) | 0.747 | **0.978** | 0.816 |

Verdict: **v4 low-pass is the pure-transfer winner** (+1.9 acc over v2, dramatically better
false-alarm discipline); the bump trades human transfer for vehicle transfer, net worse —
synthetic-side fidelity gains did not carry to real (domain gap is task-irrelevant). The
bump-vs-lowpass physics question goes to the rig tap-test (empirical), not transfer numbers.
Open weakness either way: v4 vehicle-vs-nothing transfer < v2 (0.87–0.91 vs 0.97).

**E2 (uniform sampler, corpus_v4 low-pass, balanced sampler OFF) — DONE:** pure transfer
h-vs-n 0.981 / v-vs-n **0.921** (E0: 0.873) / h-vs-v 0.883; 3-way acc **0.864**, bal **0.841**
— best pure-transfer build so far. ⇒ ~half the vehicle-transfer gap was the balanced-sampler
diet (marginal-heavy batches → conservative vehicle head), NOT sim physics. Remaining
vehicle gap vs v2 (0.92 vs 0.97) unattributable on the current real data (mismatched-setup
session; see REAL_LABEL_AUDIT/CAR_SCALE_SWEEP) — decisive test = new consistent-setup
recordings (rig characterization prompt).

**E0.5 sampler sweep — DONE, protocol frozen:** balanced (E0) / mild-mix 1,1,1,1.5,3 (E3) /
uniform (E2): pure-transfer acc 0.856 / 0.862 / **0.864**, v-vs-n 0.873 / 0.907 / **0.921**,
bal-acc 0.831 / 0.835 / **0.841**. E2 (uniform) also best fine-tuned: **0.980 ± 0.013**
(E0 0.979, v2 0.977), zero-shot 0.840. ⇒ reference protocol = sampler OFF; E2 is the v4
reference build. Fold-3 zero-shot weakness persists in every build (0.70–0.74) — a real-data
slice issue, not a training-recipe issue.

**ET terrain-matching (matched / anti / random vs E2 all-340) — DONE:** pure-transfer
threshold-free AUROCs:
| | E2 all-340 | ET_matched | ET_anti | ET_random |
|---|---|---|---|---|
| human-vs-nothing | **0.981** | 0.958 | 0.950 | 0.973 |
| vehicle-vs-nothing | 0.921 | **0.939** | 0.854 | 0.916 |
| human-vs-vehicle | **0.883** | 0.809 | 0.781 | 0.859 |
| 3-way acc | **0.864** | 0.848 | 0.778 | 0.834 |
Verdict: (1) terrain is REAL for transfer — anti-matched craters vehicle-vs-nothing to
0.854 (−8.5 vs matched), the falsification arm worked; (2) matched training gives the BEST
vehicle head (0.939 > E2's 0.921) — part of the "weak vehicle transfer" was terrain
dilution; (3) but full diversity (E2) still wins overall — matching costs the human head
and cross-class separation (fewer windows + narrower coverage). Practical: site-adapted
terrain selection helps the head whose deployment terrain is known; don't trade away
diversity wholesale. Remaining vehicle gap vs v2 (0.939 vs 0.973) = the amplitude/clipping
issue (DISCREPANCY_REPORT §2), the surviving v4.1 lever.

**E4 (v4.1 amplitude calibration) — DONE, NOT PROMOTED:** corpus_v41 (gain 1.6–2.2,
mains-relative-to-floor, spur P=0.2, mach P=0.25) passed amplitude+coverage gates
(floor 9.04 mV physical, clip medians 0, detectable-window kurtosis 0.63→2.81). But
pure transfer REGRESSED — 2×2 attribution (threshold-free AUROC h-vs-n / v-vs-n / h-vs-v):
| | real ×25.4 | real ×1000 physical |
|---|---|---|
| E2 (v4) | **0.981** / 0.921 / 0.883 | 0.966 / **0.933** / 0.867 |
| E4 (v4.1) | 0.955 / 0.887 / **0.904** | 0.928 / 0.918 / 0.870 |
v4.1 is worse than v4 in both scale conventions ⇒ per the pre-registered decision rule,
**v4 (E2) stays production candidate**; v4.1 = attribution datapoint. Pattern now 3-for-3
(coupling bump, terrain matching-only, amplitude calibration): PHYSICAL-FIDELITY
improvements do not improve — and can hurt — real transfer on this real dataset, while
COMPOSITION changes (loudness coverage, sampler, diversity) do move it. Side finding:
the physical ×1000 scale improves the vehicle head even on v4 (0.921→0.933, best yet)
at the cost of human — scale convention is itself an operating-point choice.

**Real-eval audit (REAL_LABEL_AUDIT.json):** 9%/13% of car/human real windows are
ambient-indistinguishable but present-labeled (session-level labels) → E0 cleaned-label acc
0.895, oracle-threshold ceiling 0.904. Threshold calibration costs only ~1 pt. Car-session
gain sweep (CAR_SCALE_SWEEP.json): AUROC flat over 32× scale — gain misalignment ruled out.

## 3. Budget summary

| Track | New trainings | GPU time (est) |
|---|---|---|
| E0 acceptance | 3 (+≤5 contingent) | ~1 h |
| E1 window sweep | 5 + 2 finetune chains | ~1.5 h |
| E2 sizes | 6 | ~1 h |
| E3 MLP | 2 (+1) | ~0.5 h |
| E4 raw | 3 | ~1–2 h (raw input, bigger) |
| E5 context | 3–4 | ~1 h |
| E6 robustness | 4 | ~0.75 h |
| **Total** | **~26–31 runs** | **~7–9 h GPU** + ~4 h CPU extraction (6 geometries) |

All runs are ~6–15 min each on the 3090 (feature models) — the sweep is cheap; the
discipline is in the registry, fixed protocol, and not comparing across incomparable
populations.

## 4. Decision flow

1. E0 must pass (A1+A2) before E1–E8 results are quotable.
2. E1 picks the window geometry (or the 3 s + 10 s pair) that everything downstream
   standardizes on.
3. E2 + E9 pick the deployment model. E4 decides feature-pipeline vs raw firmware.
4. E5 decides whether streaming context ships (it's free on-device if it wins).
5. Paper figure map: E1.7 (floor-vs-T), E2.3 (acc-vs-params), E3.1 (SNN parity),
   E4.4 (raw transfer), E8.2 (hard-scene table), E0.4 (Pd-vs-SNR).

## 5. Open knobs (decide before P0.1 render)

- Gap/pause parameters: p(scene has gaps) ≈ 0.25, 1–2 gaps × U(1–3) s, subject AND
  mixed scenes (my recommendation: include mixed).
- Occupancy label definition frozen: presence per window = any active emission in
  window; occupancy = active fraction (drives E1.9/E8.3).
- E2 size ladder values; E4 CNN family (plain 1D-CNN vs CRNN).
- Whether E5.3 streaming trains with scene-ordered batches from day one or evaluates
  a reset-trained model first (cheaper, weaker).
