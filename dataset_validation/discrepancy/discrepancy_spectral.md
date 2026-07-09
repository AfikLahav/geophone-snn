# REAL geophone vs SYNTHETIC corpus_v4 -- spectral discrepancy report

_generated 2026-07-06_

## Data & method

- Real rigs: human-session (human.csv/human_nothing.csv), car-session (car.csv/car_nothing.csv), field 2026-06 (geophone_2026*.csv)
- Synth: G:/geophone_synth/corpus_v4/shard_0.sqlite (11424 scenes); model input = clip(noise_mv + clean_mv(+clean_mv2), +/-256)
- mV; PSD mV^2/Hz; fs=1000 Hz; band 1-450 Hz; window 3.0s / hop 1.5s; per-window Welch nperseg=1000 -> median across windows (median Welch PSD).
- Synth human/vehicle model-input restricted to target_snr_db >= 12.0 dB (signal-dominated); clean-only = signal component alone (all scenes).
- **Confound handling**: the two real rigs have very different noise floors, so an active-minus-paired-floor **signal-excess** PSD is reported to isolate the true event spectrum.

## Task 1 & 5 -- band fractions (of 1-450 Hz area), slope, centroid, bandwidth

| group | n_win | 1-5 | 5-25 | 20-55 | 55-90 | 90-180 | 180-450 | slope 2-20Hz | centroid 5-180 | eff BW Hz |
|---|---|---|---|---|---|---|---|---|---|---|
| REAL human active | 613 | 0.001 | 0.016 | 0.252 | 0.298 | 0.410 | 0.028 | +0.75 | 85.9 | 415 |
| SYNTH human (model-in) | 1149 | 0.015 | 0.282 | 0.536 | 0.164 | 0.057 | 0.011 | +0.84 | 42.6 | 340 |
| SYNTH human clean | 1267 | 0.000 | 0.042 | 0.768 | 0.207 | 0.009 | 0.000 | +3.52 | 44.8 | 382 |
| REAL car active | 343 | 0.006 | 0.897 | 0.197 | 0.002 | 0.003 | 0.001 | +2.39 | 17.6 | 414 |
| SYNTH vehicle (model-in) | 732 | 0.008 | 0.182 | 0.501 | 0.241 | 0.104 | 0.022 | +0.94 | 51.0 | 368 |
| SYNTH vehicle clean | 851 | 0.000 | 0.252 | 0.736 | 0.118 | 0.002 | 0.000 | +3.74 | 36.3 | 392 |
| REAL human floor | 664 | 0.026 | 0.103 | 0.385 | 0.066 | 0.308 | 0.120 | -0.45 | 77.5 | 415 |
| REAL car floor | 319 | 0.511 | 0.339 | 0.074 | 0.022 | 0.048 | 0.020 | -1.70 | 28.1 | 414 |
| REAL field 2026-06 | 98 | 0.015 | 0.071 | 0.202 | 0.121 | 0.332 | 0.276 | -0.07 | 85.5 | 403 |
| SYNTH nothing calm | 5462 | 0.031 | 0.412 | 0.506 | 0.105 | 0.014 | 0.000 | +0.59 | 31.8 | 296 |
| SYNTH nothing wind | 5137 | 0.028 | 0.485 | 0.416 | 0.101 | 0.022 | 0.001 | +0.75 | 30.6 | 353 |
| SYNTH nothing rain | 5400 | 0.023 | 0.421 | 0.497 | 0.107 | 0.016 | 0.000 | +0.68 | 32.0 | 336 |

**Upper-band fraction (20-90 Hz energy above 55 Hz)**: real human 0.542 vs synth human model-in 0.234 / clean 0.212; real car 0.01 vs synth vehicle model-in 0.324 / clean 0.138.
**Signal-excess (active - paired floor)**: real human centroid 86.03 Hz (ub55=0.547); real car centroid 17.59 Hz (ub55=0.01). Confirms the shift is genuine signal, not rig floor.

## Task 2 -- narrowband line inventory

**Real per-file (prominence > 6 dB above local median):**
| file | n_lines | mains 50/100/150.. | rig 44-47/53-55 | spurious 140-180 |
|---|---|---|---|---|
| human.csv | 1 | - | - | - |
| car.csv | 1 | 201.17 | - | - |
| human_nothing.csv | 3 | 50.05 | - | - |
| car_nothing.csv | 8 | 99.85,49.32 | 53.47 | - |
| geophone_20260524_180254.csv | 2 | - | - | - |
| geophone_20260524_180403.csv | 2 | - | - | - |
| geophone_20260524_180436.csv | 1 | - | - | - |
| geophone_20260524_180443.csv | 0 | - | - | - |
| geophone_20260615_065644.csv | 26 | 50.29,150.63,100.59,200.93 | 54.93 | 155.52,160.16 |
| geophone_20260615_073558.csv | 28 | 100.1,150.15,50.05,200.2 | - | 157.71,165.04 |
| geophone_20260615_082734.csv | 8 | 50.05,150.15,200.44 | - | - |
| geophone_20260615_090405.csv | 82 | 50.29,100.59,200.93,150.63,251.22 | 55.42,45.17 | 155.76,160.89,140.62,145.51 |
| geophone_20260615_104937.csv | 12 | 50.05 | - | - |
| geophone_20260615_110000.csv | 7 | 50.05,100.1,150.15 | - | - |
| geophone_20260615_110301.csv | 11 | 50.29,100.34,150.63,200.68 | - | - |

**Synth per-scene line stats (median PSD washes out random lines):**
| synth group | lines/scene | machinery 4-80Hz scene-frac | mach freq p10/50/90 | 140-180 scene-frac | 140-180 prom p50/p90 dB |
|---|---|---|---|---|---|
| machinery | 4.47 | 0.687 | [6.8, 37.5, 61.2] | 0.427 | [12.47, 21.63] |
| calm | 4.32 | 0.593 | [6.8, 40.3, 71.5] | 0.473 | [15.38, 31.08] |
| all_nothing | 4.67 | 0.72 | [6.8, 38.0, 65.3] | 0.46 | [12.67, 29.4] |
| human | 4.57 | 0.707 | [6.8, 27.1, 65.1] | 0.347 | [8.79, 16.54] |
| vehicle | 8.29 | 0.853 | [11.1, 33.5, 67.1] | 0.5 | [7.99, 19.97] |

## Task 3 & 4 -- coupling resonance & 150 Hz spurious

**Real (line-suppressed broad-bump fit for coupling; raw for 150 Hz):**
| group | coupling center Hz | coupling height dB | width Hz | 150-band center | 150-band height dB |
|---|---|---|---|---|---|
| real_human_nothing | 51.0 | 12.83 | 3.0 | 150.0 | 4.25 |
| real_car_nothing | 44.0 | 8.93 | 2.0 | 159.0 | 4.12 |
| real_field | 49.0 | 4.94 | 2.0 | 150.0 | 6.81 |
| real_human_active | 41.0 | 2.25 | 30.0 | 155.0 | 4.17 |
| real_car_active | 42.0 | 6.73 | 8.0 | 162.0 | 3.74 |

**Synth coupling_fc parameter**: p10/25/50/75/90 = [29.3, 37.3, 52.2, 73.9, 99.7] Hz, range 21.3-211.4 Hz, only 19.8% in 50-65 Hz.
**Synth per-scene realized bump**: coupling present in 90% of scenes, center p25/50/75 = [36.1, 49.3, 68.1] Hz, height p50/p90 = [12.25, 17.62] dB; 150 Hz spurious height p50/p90 = [10.32, 22.7] dB.
Prior coupling_verify.json FC_ANCHOR = 53.0 Hz (cross-check consistent).
