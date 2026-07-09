# Feature definitions — all 132 per-window features

Reference for every feature computed by `simgeo/features.py` (`window_features`), in the
exact order of `FEATURE_NAMES`. Each entry gives the formula and a one-line explanation.

## Notation & shared quantities

All features describe one **window** `w` = 3000 samples = 3.0 s at 1 kHz.

- `w` — the windowed time signal; `|w|` its absolute value; `N = 3000`.
- `X = rfft(w)`, `mag = |X|` (magnitude spectrum), `PSD = mag²` (power spectrum).
- `f` — FFT frequency bins, 0–500 Hz, spacing 0.333 Hz.
- `BP_b(x)` — band `b` of the signal: a 4th-order Butterworth band-pass applied **scene-wide** then sliced to the window. `RMS_b = sqrt(mean(BP_b²))`.
- **Legacy bands** (Hz): LOW 20–30, CAR_APPROACH 30–34, CAR_PEAK 34–40, CAR_TAIL 40–48, MID_GAP 48–60, HUMAN_PEAK 60–70, HUMAN_TAIL 70–80, HIGH 90–100.
- **Physics bands** (Hz): wind 1–5, veh 5–25, foot 20–90, high 90–180, hop 12–15, eng 20–60, rain 60–200.
- `env = lowpass₂₀(|x|)` — broadband amplitude envelope (used for cadence/impulses).
- `Espec = |rfft(env − mean, 8192)|²` — **modulation (envelope) power spectrum**; `f_m` its bins; analysis band 0.3–12 Hz; `modmed = median(Espec over 0.3–12 Hz)`.
- `pnorm = PSD / medfilt(PSD, 51)` — **locally whitened** spectrum (a value of, say, 8 means a peak 8× above its local spectral background); used to find tonal lines/combs.
- **impulses** `it` — envelope peaks with height > 2·median(env), spaced ≥ 80 ms apart (footstep/transient picks); **rain picks** use the 60–200 Hz band, height > 3·median, ≥ 20 ms apart.
- **mains/site lines** = {44,45,46,47,50,55,100,150} Hz ±0.75 Hz, excluded from comb/line detectors and tracked separately.

Cepstra/entropies use base-2 logs; tiny `+ε` terms (1e-10…1e-30) are numerical guards omitted below.

---

## LEGACY32 — the notebook's original 32 features

Kept verbatim from the prototype so their cross-terrain transfer is *measured*, not assumed.

1. **`energy_low_freq`** — `RMS_{20–30}`. Energy in the 20–30 Hz low band.
2. **`energy_car_approach`** — `RMS_{30–34}`. Lower vehicle band (approaching engine/tire).
3. **`energy_car_peak`** — `RMS_{34–40}`. Main vehicle band.
4. **`energy_car_tail`** — `RMS_{40–48}`. Upper vehicle band.
5. **`energy_mid_gap`** — `RMS_{48–60}`. Gap band between vehicle and human bands.
6. **`energy_human_peak`** — `RMS_{60–70}`. Main footstep band.
7. **`energy_human_tail`** — `RMS_{70–80}`. Upper footstep band.
8. **`energy_high_freq`** — `RMS_{90–100}`. High band (impulsive/transient content).
9. **`rms_total`** — `sqrt(mean(w²))`. Overall amplitude of the window.
10. **`peak_to_peak`** — `max(w) − min(w)`. Full dynamic range.
11. **`variance`** — `var(w)`. Signal power about the mean.
12. **`zcr`** — `count(sign changes in w) / N`. Zero-crossing rate ≈ a rough pitch/high-freq proxy.
13. **`kurtosis`** — `kurtosis(w)` (excess). Impulsiveness/heavy-tailedness of the amplitude distribution.
14. **`skewness`** — `skew(w)`. Asymmetry of the amplitude distribution.
15. **`spectral_centroid`** — `sc = Σ(f·mag) / Σ mag`. "Center of mass" frequency (brightness).
16. **`spectral_bandwidth`** — `sqrt(Σ((f−sc)²·mag) / Σ mag)`. Spread of the spectrum about the centroid.
17. **`spectral_rolloff`** — smallest `f` where `cumsum(PSD)` reaches 95% of total. Frequency below which most energy lies.
18. **`spectral_flatness`** — `geomean(PSD) / mean(PSD)`. Near 1 = noise-like/flat; near 0 = tonal/peaky.
19. **`spectral_entropy`** — `−Σ p·log₂p`, `p = PSD/ΣPSD`. Spectral disorder (flat = high, tonal = low).
20. **`dominant_freq`** — `f` at `argmax(mag)`. Single strongest frequency.
21. **`sta_lta_ratio`** — `max(STA)/LTA`, `STA` = 0.1 s moving mean of `|w|`, `LTA` = `mean(|w|)`. Classic seismic trigger — how much the loudest 0.1 s stands above the window average.
22. **`event_count`** — `#peaks(|w|, height > 2·rms_total)`. Number of strong transients.
23. **`mean_burst_len`** — mean run length (samples) where `|w| > rms_total`. Typical burst duration.
24. **`activity_concentration`** — `Σ(top 25% of w²) / Σ w²`. How concentrated energy is in the loudest quarter of samples.
25. **`temporal_entropy`** — entropy of the 50-bin amplitude histogram of `w`. Disorder of the sample-value distribution.
26. **`burst_efficiency`** — `Σ(w² where |w|>rms) / Σ w²`. Fraction of energy in above-average samples.
27. **`autocorr_250`** — `corr(w[:−250], w[250:])`. Self-similarity at 0.25 s lag (periodicity).
28. **`autocorr_500`** — `corr(w[:−500], w[500:])`. Self-similarity at 0.5 s lag.
29. **`ratio_car_human`** — `E_{CAR_PEAK} / E_{HUMAN_PEAK}` (34–40 ÷ 60–70). Vehicle-vs-human band balance.
30. **`ratio_human_car`** — `E_{HUMAN_PEAK} / E_{CAR_PEAK}`. The inverse balance.
31. **`centroid_car_band`** — `Σ(f·mag)/Σmag` over 34–48 Hz. Brightness within the vehicle band.
32. **`centroid_human_band`** — `Σ(f·mag)/Σmag` over 60–80 Hz. Brightness within the footstep band.

## PHYS7 — physics-band energies/fractions (distance-robust)

Fractions normalize out absolute level, so they survive changes in source distance.

33. **`log_rms`** — `log₁₀(rms_total)`. Loudness on a log scale.
34. **`frac_wind_1_5`** — `RMS_{wind 1–5} / rms_total`. Share of energy in the wind band.
35. **`frac_veh_5_25`** — `RMS_{veh 5–25} / rms_total`. Share in the vehicle band.
36. **`frac_foot_20_90`** — `RMS_{foot 20–90} / rms_total`. Share in the footstep band.
37. **`frac_high_90_180`** — `RMS_{high 90–180} / rms_total`. Share in the high band.
38. **`centroid_foot_band`** — `Σ(f·mag)/Σmag` over 20–90 Hz. Brightness within the footstep band.
39. **`ratio_veh_foot`** — `RMS_{veh} / RMS_{foot}`. Vehicle-band vs footstep-band ratio (vehicle/human cue).

## CAD12 — cadence / gait modulation (envelope spectrum 0.3–12 Hz)

Built on `Espec`, the spectrum of the amplitude envelope — i.e. the *rhythm* of the signal.

40. **`cad_freq`** — `f_m` of `max(Espec)` in 0.8–4 Hz (call it `f0`). Dominant cadence (steps/s).
41. **`cad_salience`** — `Espec[f0] / modmed`. How far the cadence peak stands above the modulation-spectrum background.
42. **`cad_harm2`** — `P(2·f0) / P(f0)`. 2nd-harmonic strength of the gait rhythm (biped vs quadruped structure).
43. **`cad_harm4`** — `P(4·f0) / P(f0)`. 4th-harmonic strength of the gait rhythm.
44. **`footfall_rate`** — `f_m` of `max(Espec)` in 0.8–10 Hz. Footfall rate over a wider band (catches faster quadruped gaits).
45. **`mod_peak_count`** — `#peaks(Espec in 0.5–8 Hz, height > 3·modmed)`. Number of rhythmic components → candidate for single vs multiple movers.
46. **`env_entropy`** — entropy of normalized `Espec` over 0.3–12 Hz. Rhythm disorder (one clean cadence = low; many movers = high).
47. **`env_cv`** — `std(env)/mean(env)`. Envelope burstiness (impulsive vs steady).
48. **`env_ac_strength`** — `max(AC_env[lags 250–1250]) / AC_env[0]`, `AC_env = irfft(Espec)`. Strength of envelope periodicity at 0.25–1.25 s.
49. **`iii_mean`** — mean inter-impulse interval (s) over window impulses (0 if < 3). Average time between footsteps.
50. **`iii_cv`** — `std/mean` of inter-impulse intervals (0 if < 3). Regularity of step timing.
51. **`gust_mod`** — `Σ Espec[0.05–0.5 Hz] / Σ Espec[0.05–12 Hz]`. Very-slow modulation share (wind gust AM).

## ENG8 — tonality / engine

`pnorm` highlights narrowband peaks; mains/site lines are excluded from comb/line detectors.

52. **`comb_salience`** — best harmonic-comb score over fundamentals 25–58 Hz: for each candidate `f0`, average `pnorm` at orders 1–3 (skipping mains bins, need ≥2 clean orders); take the max. Engine harmonic-comb strength.
53. **`comb_f0`** — the fundamental frequency that wins `comb_salience`. Estimated engine fundamental.
54. **`n_lines`** — `#peaks(pnorm in 10–180 Hz, non-mains, height > 6)`. Count of distinct tonal lines.
55. **`line_stability`** — mean pairwise correlation of the log-PSDs of the three 1 s sub-windows (10–100 Hz). Spectral steadiness over the 3 s (steady engine vs transient).
56. **`hop_band_frac`** — `RMS_{hop 12–15} / rms_total`. Energy share in the wheel-hop band.
57. **`beat_depth`** — `std/mean` of the engine-band envelope (`env` of 20–60 Hz band). Amplitude beating → multiple engines/convoy cue.
58. **`line_mains`** — `mean(pnorm at mains/site-line bins)`. Mains (50/100/150 Hz) + rig-line monitor.
59. **`tonality_max`** — `max(pnorm over 10–180 Hz)`. Prominence of the single strongest tonal peak.

## WX2 — weather gates

60. **`rain_impulse_rate`** — `#rain_picks_in_window / 3 s`. Rate of high-band impact transients (rain).
61. **`highband_kurtosis`** — `kurtosis(BP_{rain 60–200})`. Impulsiveness of the high band (rain shot noise is very kurtotic).

## CEP16 — cepstral coefficients (compact spectral-shape descriptors)

Each = `DCT(log(filterbank · mag))`, keeping coefficients 1–8 (the c0 energy term is dropped). 24 triangular filters spanning ~1–200 Hz.

62–69. **`mfcc_1` … `mfcc_8`** — cepstra from a **mel-spaced** filterbank. Smooth, decorrelated summary of spectral shape (low orders = coarse tilt/peaks).
70–77. **`lfcc_1` … `lfcc_8`** — same, from a **log/geometric-spaced** filterbank (`geomspace(2,200,26)`). A second spectral-shape view with different frequency weighting. (Named "lfcc" by convention; the bank here is geometric-spaced.)

## WPE16 — wavelet-packet energy fractions

`db4` wavelet packet, level 4 → 16 leaf sub-bands (natural order), each ≈ 31 Hz wide across 0–500 Hz.

78–93. **`wpe_0` … `wpe_15`** — `leaf_energy_i / Σ leaf_energy`, i.e. the fraction of energy in each of the 16 sub-bands. A time-frequency "texture" fingerprint that complements the FFT bands. (Low indices = low frequency.)

## AR8 — autoregressive (linear-prediction) coefficients

Autocorrelation `r = irfft(PSD)[:9]` → Levinson-Durbin → order-8 AR model.

94–101. **`ar_1` … `ar_8`** — the 8 AR coefficients. They encode the spectral envelope / resonances of the window as a parametric model (poles ≈ formant-like peaks).

## NL1 — nonlinear complexity

102. **`higuchi_fd`** — Higuchi fractal dimension (`kmax=8`): slope of `log L(k)` vs `log(1/k)`, where `L(k)` is the mean curve length at sub-sampling step `k`. Higher = rougher/more complex waveform.

---

## v2 NEW — 30 added features

### MOD2 — modulation-spectrum refinement (single/multi, biped/quad/vehicle)

Refines the envelope spectrum `Espec`; `Emod = Espec over 0.3–12 Hz`.

103. **`mod_e_1_3`** — `Σ Espec[1–3 Hz] / Σ Emod`. Modulation energy in the human-cadence band.
104. **`mod_e_3_8`** — `Σ Espec[3–8 Hz] / Σ Emod`. Modulation energy in the quadruped/fast-footfall band.
105. **`mod_e_8_12`** — `Σ Espec[8–12 Hz] / Σ Emod`. Modulation energy in the herd/group-blur band.
106. **`mod_ratio_low_mid`** — `Σ Espec[1–3] / Σ Espec[3–8]`. Slow- vs fast-rhythm ratio (human vs animal cue).
107. **`mod_peak2_ratio`** — `2nd-highest / highest` peak in `Espec[0.5–8 Hz]` (0 if < 2 peaks). A strong secondary rhythm signals multiple movers.
108. **`mod_flatness`** — `geomean(Emod) / mean(Emod)`. Tonal (single clean rhythm, low) vs noise-like (high) modulation.
109. **`mod_centroid`** — `Σ(f_m·Emod) / Σ Emod`. Center frequency of the rhythm = overall movement tempo.

### III2 — inter-impulse-interval statistics (single vs multiple)

Computed from window impulses `it`.

110. **`impulse_density`** — `len(it) / 3 s`. Footstep/transient rate (more feet → more impulses).
111. **`iii_entropy`** — entropy of an 8-bin histogram of inter-impulse intervals (≥4 impulses, else 0). Irregular spacing (multiple unsynchronized movers) → high.
112. **`iii_skew`** — `skew(inter-impulse intervals)`. Asymmetry of step-timing distribution.
113. **`iii_range_norm`** — `(p90 − p10) / median` of intervals. Normalized spread of step timing.
114. **`impulse_amp_cv`** — `std/mean` of envelope amplitude at impulse times. Variability of footstep strength.

### TONAL — tonal-vs-impulsive / machinery + aircraft confusers (nothing vs class)

115. **`spectral_crest`** — `max(PSD) / mean(PSD)`. Peakiness — one dominant tone vs broadband.
116. **`n_sharp_lines`** — `#peaks(pnorm in 10–200 Hz, non-mains, height > 8)`. Count of *sharp* tonal lines (stricter than `n_lines`).
117. **`max_line_prom`** — `max(pnorm in 10–200 Hz, non-mains)`. Prominence of the strongest non-mains line.
118. **`tonal_index`** — `Σ PSD[pnorm>8] / Σ PSD[10–200 Hz]`. Fraction of in-band energy living in sharp tonal lines.
119. **`heli_comb_salience`** — best harmonic-comb score over fundamentals **12–22 Hz** (helicopter main-rotor blade-pass comb), same comb test as `comb_salience`.
120. **`pump_band_frac`** — `Σ PSD[75–180 Hz] / Σ PSD`. Energy share in the irrigation-pump tonal band.
121. **`tonal_in_foot`** — `Σ PSD[pnorm>8 within 20–90 Hz] / Σ PSD[20–90 Hz]`. Distinguishes tonal machinery sitting *inside* the footstep band from genuine (broadband, impulsive) footsteps.

### STAT2 — stationarity / temporal structure (between-class)

Uses three 1 s sub-windows; `subc` = their spectral centroids, `subr` = their RMS, `sub` = their log-PSDs.

122. **`centroid_var`** — `std(subc)`. How much the spectral centroid drifts across the 3 s (non-stationarity; a passing vehicle sweeps, steady noise doesn't).
123. **`rms_cv_sub`** — `std(subr) / mean(subr)`. Loudness burstiness over the three seconds.
124. **`spectral_flux`** — `mean(|Δ sub|)` between consecutive sub-windows. Rate of spectral change.
125. **`temporal_centroid`** — `Σ(n·w²) / (Σw² · N)`, `n` = sample index. Where in the window the energy sits (0 = front-loaded onset, 1 = back-loaded).
126. **`attack_sharpness`** — mean envelope slope `(env[t] − env[t−10])/0.01` at impulse onsets (first ≤20 impulses). Transient rise rate — sharp footsteps vs gradual swells.

### HOS — higher-order spectral (engine harmonic coupling)

127. **`spec_skew`** — `skew(PSD)`. Asymmetry of the power distribution across frequency.
128. **`spec_kurt`** — `kurtosis(PSD)`. Peakiness of the power distribution (strong tones → high).
129. **`comb_stability`** — `mean over freq of std across the 3 sub-window log-PSDs (10–100 Hz)`. Low = the spectrum (engine harmonics) is phase-locked/steady across the 3 s.

### COOC — co-occurrence / multi-source complexity (mixed scenes)

130. **`band_occupancy`** — `#{ b ∈ (wind,veh,foot,high) : RMS_b/rms_total > 0.15 }`. How many physics bands are simultaneously active → multi-source/mixed cue.
131. **`spectral_entropy_full`** — `−Σ p·log₂p`, `p = PSD/ΣPSD`. Overall spectral complexity. *(Essentially the same quantity as `spectral_entropy` #19 — kept as an independent multi-source descriptor; expect the two to correlate strongly.)*
132. **`veh_foot_simultaneity`** — `(RMS_{veh}/rms_total) · min(impulse_density, 5)`. High only when vehicle-band tonal energy **and** footstep impulses occur together — a direct human+vehicle co-occurrence cue for mixed scenes.

---

_Source of truth: `simgeo/features.py`. If a band edge, threshold, or order changes there, update this file. Window = 3.0 s / 3000 samples @ 1 kHz; features are computed per window at 1.5 s hop._
