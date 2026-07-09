# Time-domain / amplitude discrepancy: REAL geophone vs SYNTHETIC v4

_Generated 2026-07-06 21:04:35. Synth = `G:\geophone_synth\corpus_v4\shard_0.sqlite`, model input = clip(clean+clean2?+noise, +-256 mV). Real amp = VOLTS; analysed in mV (x1000). Windows = 3 s. Scripts: `wf_common.py`, `wf_analyze.py`, `wf_report.py`._

All numbers below are produced by `wf_analyze.py` (written to `discrepancy_waveform.json`) unless noted; derived headline numbers by `wf_report.py`.

## 0. Amplitude-scale convention (the x25.4 question)

- The pipeline's `x25.4` convention and the true-mV `x1000` convention differ by **39.37x** (`wf_report.py`).
- Applying `x25.4` to real volts puts real RMS at **0.4-3.3 (mV-equiv)**, i.e. only **0.4-3.1%** of the synthetic model-input mV scale (ratios below). `x25.4` therefore does NOT align the two domains; it under-scales real by ~30-240x vs synthetic mV.
- `x1000` (true mV) lands real on the same order of magnitude as synth (ratios 0.17-1.21). The single factor on raw volts needed to match the synth median is class-dependent (**825-6065x**), so no single scalar reconciles them. (`wf_analyze.py` -> `pipeline_x25p4`)

| pairing | real med (V) | x25.4 -> ratio vs synth mV | x1000(mV) -> ratio | factor on V to match synth |
|---|---|---|---|---|
| human_active__vs__human | 0.02820 | 0.716 -> 0.0075 | 28.20 -> 0.295 | 3392 |
| car_active__vs__vehicle | 0.12951 | 3.290 -> 0.0308 | 129.51 -> 1.212 | 825 |
| human_nothing__vs__nothing | 0.00736 | 0.187 -> 0.0042 | 7.36 -> 0.165 | 6065 |
| car_nothing__vs__nothing | 0.02519 | 0.640 -> 0.0143 | 25.19 -> 0.564 | 1772 |

## 1. Amplitude distributions (RMS per 3 s window, mV) & scale gap (dB)

RMS-per-window percentiles (mV). Real analysed as mV (x1000); synth = clipped model input. (`wf_analyze.py` -> `real`/`synth` `rms_mv_pct`)

| source | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| REAL human_active | 13.7 | 20.0 | 28.2 | 57.4 | 97.0 |
| REAL human_nothing | 3.9 | 6.0 | 7.4 | 11.2 | 35.2 |
| REAL car_active | 27.6 | 58.0 | 129.5 | 306.2 | 862.5 |
| REAL car_nothing | 17.4 | 21.4 | 25.2 | 34.6 | 48.3 |
| SYNTH human | 9.5 | 47.5 | 95.7 | 174.4 | 256.0 |
| SYNTH vehicle | 5.7 | 43.4 | 106.8 | 193.4 | 252.8 |
| SYNTH nothing | 4.1 | 16.7 | 44.7 | 97.6 | 177.1 |
| SYNTH animal | 9.9 | 36.7 | 75.0 | 144.2 | 224.4 |
| SYNTH mixed | 7.8 | 30.4 | 69.2 | 143.8 | 247.3 |

Scale gap = 20*log10(synth_median / real_median). Positive = synth louder. (`wf_analyze.py` -> `scale_gap`)

| pairing | real med mV | synth med mV | gap dB (median) | gap dB (p95) |
|---|---|---|---|---|
| human_active__vs__human | 28.2 | 95.7 | +10.6 | +8.4 |
| car_active__vs__vehicle | 129.5 | 106.8 | -1.7 | -10.7 |
| human_nothing__vs__nothing | 7.4 | 44.7 | +15.7 | +14.0 |
| car_nothing__vs__nothing | 25.2 | 44.7 | +5.0 | +11.3 |

- Real CAR events overshoot the synthetic +-256 mV ceiling: real car peak = **18342 mV** (**71.6x**, **37.1 dB** above the synth clip), while synth vehicle p95 window RMS is pinned near 253 mV by clipping.

## 2. Impulsiveness (per-window excess kurtosis & crest factor)

| source | kurt p50 | kurt p95 | crest p50 | crest p95 |
|---|---|---|---|---|
| REAL human_active | 13.66 | 37.91 | 7.75 | 11.48 |
| REAL car_active | 1.38 | 7.06 | 4.16 | 6.87 |
| REAL human_nothing | 0.29 | 6.60 | 3.70 | 7.09 |
| REAL car_nothing | 0.08 | 5.32 | 3.53 | 7.54 |
| SYNTH human | 0.23 | 14.82 | 2.67 | 7.64 |
| SYNTH vehicle | -0.25 | 10.22 | 2.40 | 7.40 |
| SYNTH nothing | 0.87 | 18.46 | 3.87 | 9.31 |

- **Human is the worst realism gap**: real footstep windows are spiky (kurt p50 = 13.7, crest p50 = 7.7) but synth human is near-Gaussian (kurt p50 = 0.23, crest p50 = 2.7). Synth under-models impulsiveness by ~13 kurtosis units.
- Car gap is milder (real kurt 1.38 vs synth -0.25). +-256 clipping drives synth crest p5 toward 1.0 (flat-topped), an artifact absent in real.

## 3. Quantization (LSB & distinct levels per window)

| source | LSB (mV) | eff bits | distinct/win (median) | fill ratio |
|---|---|---|---|---|
| REAL human_active | 0.290 | 13.6 | 446 | 0.149 |
| REAL car_active | 0.200 | 17.1 | 1329 | 0.443 |
| REAL human_nothing | 0.290 | 11.6 | 114 | 0.038 |
| REAL car_nothing | 0.290 | 12.2 | 391 | 0.130 |
| REAL field_20260524_a | 0.125 | 3.3 | 3 | 0.010 |
| REAL field_20260615_a | 0.125 | 4.7 | 20 | 0.006 |
| REAL field_20260615_b | 0.125 | 5.0 | 30 | 0.010 |
| SYNTH human | continuous (float32) | ~24 | 2898 | 0.966 |
| SYNTH vehicle | continuous (float32) | ~24 | 2874 | 0.958 |
| SYNTH nothing | continuous (float32) | ~24 | 2999 | 1.000 |

- Real is a discrete comb: main sessions LSB ~0.20-0.29 mV, field files LSB = 0.125 mV (ADS1015, 12-bit, +-0.256 V FSR). Distinct levels per 3 s window collapse to **14-114** in quiet/field data (fill 0.5-4%). Synth float32 fills **96-100%** of a window with unique levels (continuous). The quantization is worst exactly in the quiet regime the model must discriminate.

## 4. Clipping / rail saturation

| source | rail frac (exact) | clipfrac median | clipfrac p95 |
|---|---|---|---|
| REAL human_active | 0.00000 | n/a | n/a |
| REAL car_active | 0.00000 | n/a | n/a |
| REAL human_nothing | 0.00000 | n/a | n/a |
| REAL car_nothing | 0.00000 | n/a | n/a |
| SYNTH human | n/a | 0.1078 | 1.0000 |
| SYNTH vehicle | n/a | 0.1313 | 0.7904 |
| SYNTH nothing | n/a | 0.0153 | 0.1920 |

- Real main sessions **never saturate** (0.000 samples at rail; car reaches 18.3 V with no clip). Synth is hard-clipped at +-256 mV: median within-scene clip fraction human 10.8%, vehicle 13.1%, nothing 1.5%; the loudest synth windows are up to 100% railed. Structural inversion: synth clips where real has headroom.

## 5. Non-stationarity (CV of per-window RMS)

| class | real rms-CV | synth rms-CV (median within scene) |
|---|---|---|
| human | 0.734 | 0.374 |
| vehicle | 1.283 | 0.546 |
| nothing | 0.341 | 0.514 |

- Real active recordings are burstier than synthetic scenes (real car CV 1.28 vs synth 0.55; real human 0.73 vs synth 0.37). Real records long quiet + sparse events; synth scenes are event-centric/uniform. Note the spans differ (real 840-1400 s vs synth 20-115 s).

## 6. Sample rate & timing (from real time_s)

| source | fs Hz | jitter ms | dt_max s | frac dt>1.5x median |
|---|---|---|---|---|
| human_active | 1000.00 | 0.005 | 0.001 | 0.00e+00 |
| human_nothing | 1000.00 | 218.741 | 35.001 | 3.91e-05 |
| car_active | 1000.00 | 0.005 | 0.001 | 0.00e+00 |
| car_nothing | 1000.00 | 0.009 | 0.001 | 0.00e+00 |
| field_20260524_a | 100.00 | 0.500 | 0.011 | 0.00e+00 |
| field_20260524_b | 100.00 | 0.500 | 0.011 | 0.00e+00 |
| field_20260615_a | 1000.00 | 0.068 | 0.002 | 4.69e-03 |
| field_20260615_b | 1000.00 | 0.047 | 0.002 | 2.08e-03 |
| field_20260615_c | 1000.00 | 0.029 | 0.002 | 8.41e-04 |
| SYNTH (all) | 1000.00 | 0.000 | - | 0 |

- Main sessions (human/car active + car_nothing) sit on a clean 1000 Hz grid (jitter <=0.01 ms), matching synth fs. But: `human_nothing` is a spliced recording (39 discontinuities of 35.0 s). Field 20260524 files are **100 Hz** (not 1000). Field 20260615 files show 0.08-0.5% dropped samples. Synth never has splices/drops/rate mixing.
