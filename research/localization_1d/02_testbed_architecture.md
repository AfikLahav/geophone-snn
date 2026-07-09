> **Architecture plan — 1-D pyprop8 localization testbed.** Authored 2026-07-08.
> Research & design phase only — NO code written. Reuses `simgeo_v42/` pipeline.
> Inputs: `gfbank_build.py`, `scenes.py`, `wavelet.py`, `profiles.py`; `research/house_localization/02_indoor_localization_methods.md`; `research/house_localization/04_sensor_hardware_hybrid.md`.

---

# 1-D Localization Testbed: Concrete Architecture

## Overview

The testbed is a pure-Python Monte Carlo harness that reuses the existing pyprop8 Green's-function pipeline to synthesize multi-sensor recordings of a walking path in 2-D and then runs a suite of localization algorithms against them — measuring accuracy, GDOP, CRLB, and blind-velocity degradation — at negligible compute cost compared with a Devito 3-D run.

"1-D" refers to the physics engine: pyprop8 solves 1-D depth-integrated wave propagation in a horizontally layered half-space, producing cylindrically symmetric Rayleigh-wave Green's functions indexed only by range r. This gives correct frequency-dependent attenuation and waveform shape (including the t* Azimi model already wired in `scenes.py`) while bypassing 3-D FEM/FDTD entirely. The engine cost is O(nr × nt) per bank, not O(Nx × Ny × Nz × Nt).

---

## 1. 3-Component GF Banks

### 1.1 What the current bank stores

`gfbank_build.py` calls `compute_seismograms` with two force sources in one shot:

```python
F2 = np.stack([np.array([[0.], [0.], [1.]]),   # Fz: vertical force
               np.array([[1.], [0.], [0.]])])   # Fx: horizontal radial force
```

The returned seismogram array has shape `(2, nr, 3, nt)` — axis 1 is source, axis 2 is component (Z, R, T). The bank builder saves only `a[0, :, 2, :]` (vertical response to Fz, component index 2 = Z) and `a[1, :, 2, :]` (vertical response to Fx). The horizontal components `a[:, :, 0, :]` (R) and `a[:, :, 1, :]` (T) are already computed but silently discarded.

### 1.2 Extension to 3-component output

**No change to the pyprop8 call is needed.** The two extra save lines in `_build_one_inner` are:

```python
np.savez_compressed(out,
    radii_km=radii, dt=DT, meta=json.dumps(profile),
    z_fz =a[0, :, 2, :].astype(np.float32),   # existing: vertical / Fz
    z_fx =a[1, :, 2, :].astype(np.float32),   # existing: vertical / Fx
    r_fz =a[0, :, 0, :].astype(np.float32),   # NEW: radial / Fz
    r_fx =a[1, :, 0, :].astype(np.float32),   # NEW: radial / Fx
    t_fz =a[0, :, 1, :].astype(np.float32),   # NEW: transverse / Fz (zero by symmetry)
    t_fx =a[1, :, 1, :].astype(np.float32))   # NEW: transverse / Fx
```

The transverse components `t_fz` and `t_fx` are identically zero for a point source in a 1-D layered model (SH is decoupled from P-SV for a vertical or radial point force). Storing them is optional but aids debugging. The material additions: 4 × (nr × nt × 4 bytes) ≈ 4 × 48 × 2500 × 4 ≈ 1.9 MB extra per bank, negligible against the existing ~1 MB per bank.

### 1.3 How the radial component gives bearing

The radial particle motion `r_fz(r, t)` points from source to receiver along the surface. For a sensor at position `s` and a source at position `p`:

```
unit_radial = (s - p) / |s - p|
```

The observed horizontal velocity at sensor `s` is:

```
H_obs(t) ≈ r_fz(|s-p|, t) × unit_radial
```

If the sensor has two horizontal channels (N, E), the P/Rayleigh arrival particle-motion ellipse is aligned along `unit_radial`. Polarization analysis (covariance of N×E samples in a short window around the first P/Rayleigh arrival) gives the back-azimuth angle as:

```
bearing_to_source = atan2(unit_radial[1], unit_radial[0]) + π
```

In a 1-D model, this bearing is exact. In the real world, lateral heterogeneity and structural reflections scatter it. The testbed delivers ground-truth bearing for every (source, sensor) pair because we construct `unit_radial` exactly.

### 1.4 The translation-invariant distance-indexed dictionary

The GF bank is indexed by scalar range `r` only — it is translation-invariant by the 1-D symmetry of the half-space model. This gives a flat lookup table of shape `(nr,)` rather than a 2-D spatial grid.

**Matched-field cost without the dictionary:**
If you treated each sensor–source pair independently, you would need `N_sensors × N_source_candidates` GF evaluations. For N=6 sensors, a 10 m × 10 m floor on a 0.1 m grid (10,000 candidates): 60,000 GF lookups, each requiring interpolation in (r, t) — expensive.

**With the translation-invariant dictionary:**
For each candidate source position `p_j` and sensor position `s_i`, compute scalar `r_ij = |s_i - p_j|` and look up `GF[r_ij]` by nearest-neighbor or linear interpolation in the pre-built `(nr,)` table. Cost:
- N_source_candidates × N_sensors scalar distance computations: O(N_grid × N_sensors) arithmetic operations, ~60,000 float multiplications — microseconds.
- N_source_candidates × N_sensors range lookups via `np.searchsorted`: O(N_grid × N_sensors × log(nr)) — still microseconds.
- The actual GF waveforms are read from a pre-loaded array, not recomputed.

The azimuth enters only as a scalar cosine factor in `emit()` (see `scenes.py` line 118: `cosaz = dot(heading, to_rx) / ...`). For matched-field, azimuth factors can be swept as an O(N_azimuths) outer loop or folded into the replica construction per candidate. Total cost for matched-field over the full grid:

```
O(N_distances × N_sensors × nt)   for replica FFT convolutions
```

where N_distances ≤ nr (up to 48) because many candidate positions map to the same discretized range bin. In practice the distinct ranges from N_sensors sensors to a 100×100 grid are at most N_sensors × N_grid = 60,000, but most share range bins, so the unique convolutions are far fewer.

This is the key economy vs. a Devito run, where every spatial grid point in the 3-D domain requires propagating the full wavefield.

---

## 2. The 2-D Scene Harness

### 2.1 Coordinate system

Place the house floor as a 2-D plane (x, y) in meters. Sensors are fixed at positions `{s_i}_{i=1..N}`. Source positions are drawn from a walking path `{p(t_k)}` sampled at footstep times `{t_k}`.

No change from `scenes.py`'s existing Cartesian geometry is needed — `assemble()` already uses `r = hypot(x, y)` with sensor at origin. For the multi-sensor harness, shift coordinates so sensor `s_i` is at origin for each render, or equivalently compute `r_ij = |s_i - p_j|` directly and call `bank.emit(r_ij, ...)`.

### 2.2 Multi-sensor scene synthesis

For each sensor `s_i` and each footstep emission at `p_j, t_j`:

1. Compute `r_ij = |s_i - p_j|`, `heading = unit vector of walking direction`, `to_rx = (s_i - p_j) / r_ij`.
2. Look up the (Fz, Fx) wavelet pair from `walk_footfall(rng, mass)` in `wavelet.py`.
3. Call `bank.emit(r_ij, heading, to_rx, Fz, Fx)` — returns ground velocity waveform length `bank.n` (~2.5 s at 1000 Hz).
4. Add this waveform at offset `t_j` into sensor `i`'s timeline.
5. Optionally add noise: colored noise matched to a geophone self-noise floor (10 ng/√Hz above 4.5 Hz, converted to velocity via ω integration).

Output: `recordings[i, 0:T_total*FS]` for i in 0..N-1, plus `gt_positions[k] = p(t_k)` as ground truth.

The single-sensor `assemble()` in `scenes.py` already implements steps 2–4 with the source at origin. The only extension is iterating over sensors and offsetting the source position.

**Footstep timing:** reuse `sample_path()` from `scenes.py` to get a realistic walking trajectory, or define a deterministic grid path for systematic sweeps.

### 2.3 Complexity

Per-scene cost:
- N footsteps × N sensors × `emit()` calls: each `emit()` is 2 FFTs of length `bank.n` (~2500) = O(bank.n × log(bank.n)) ≈ 30,000 flops.
- Total: N_steps × N_sensors × 30,000 flops.
- For N_steps = 20, N_sensors = 6: 3.6 M flops. At 10 GFLOPS/core: **0.36 ms**.
- Full scene including noise synthesis and array accumulation: ~1–5 ms per scene.

Compare with Devito: a 10 m × 10 m × 3 m house at 1 cm resolution = 10^8 spatial cells; 0.25 s simulation at 10 kHz time step = 2500 steps; at ~10 GFLOPS: **~25,000 seconds** per scene (several CPU-hours). The pyprop8 testbed is **~10^7 × faster** per scene. Enabling 100,000-scene Monte Carlo sweeps in minutes rather than years.

---

## 3. Localizer Modules

### 3.1 Module interface

Each localizer takes as input:
```python
recordings: ndarray (N_sensors, T_total*FS)
sensor_positions: ndarray (N_sensors, 2)          # meters, 2-D
fs: float                                          # 1000.0
bank: Bank                                         # for matched-field replicas
```
and returns:
```python
estimated_position: ndarray (T_total_steps, 2)    # one estimate per footstep
```

Ground truth `gt_positions` is held out from all localizer modules — it is only used for error computation after the run.

### 3.2 Matched-Field Processing (MFP)

MFP correlates observed multi-sensor waveforms with synthetic replicas generated from the GF dictionary.

**Algorithm:**
1. For each footstep window (detect by STA/LTA or known cadence during Monte Carlo), extract `rec_window[i, 0:W]` for each sensor.
2. For each candidate source position `p_j` on a spatial grid:
   a. For each sensor `i`, compute `r_ij = |s_i - p_j|` and look up `replica_ij = VF["z_fz"][r_bin] * Fz_spectrum` (using the pre-built `bank.VF` spectra from `scenes.py`'s `Bank` class).
   b. Compute the Bartlett (conventional MFP) power: `B(p_j) = |sum_i conj(DATA_i) * REPLICA_i|^2 / (sum |DATA_i|^2 × sum |REPLICA_i|^2)`.
3. Return `argmax_j B(p_j)` as the estimated position.

**Minimum-Variance (Capon/MVDR) variant:** replace Bartlett with `1 / (replica^H × R^{-1} × replica)` where `R` is the cross-spectral matrix estimated from data. Capon has narrower mainlobe but requires stable `R^{-1}` (needs N_sensors^2 samples → stable for long windows).

**Cost:** O(N_grid × N_sensors × n_freq) per footstep. For N_grid = 10,000, N_sensors = 6, n_freq = 1250: 75M multiply-accumulates per footstep — ~10 ms per estimate at 10 GFLOPS. Negligible.

**GF source for replicas:** the identical `bank.VF` dict built in `Bank.__init__()` is already in memory from scene synthesis. MFP replicas cost nothing extra to generate — they are lookups into the pre-built `(nr, n_freq)` complex array.

### 3.3 SO-TDOA (Sign-Only TDOA)

Based on Bahroun et al. (J. Sound Vib. 2014, arXiv 1211.3233). For each sensor pair (i, j), compute `tau_ij = argmax(GCC-PHAT(rec_i, rec_j))`. The sign of `tau_ij` constrains which side of the baseline the source lies on. Intersect the sign-consistent half-planes for all N(N-1)/2 pairs.

**Full TDOA variant (for comparison):** use the measured `tau_ij` as a hyperbolic constraint and solve the overdetermined system via nonlinear least squares (Gauss-Newton or BFGS). Input: set of (sensor_i, sensor_j, tau_ij) triplets. Assumes constant group velocity `v_g` (the "blind velocity" scenario — see §5).

**With known velocity:** position CRLB from TDOA is:
```
sigma_pos ≈ GDOP × v_g × sigma_tau
```
where `sigma_tau = 1 / (2π × beta_eff × sqrt(2T × SNR))` (Carter 1987 CRLB, §4.1 of doc 02). For beta_eff = 2π × 30 Hz, T = 0.1 s, SNR = 20 dB: `sigma_tau ≈ 0.09 ms`, `v_g = 300 m/s`, GDOP = 1.5: `sigma_pos ≈ 4 cm`.

### 3.4 RSS / Energy Trilateration

For each sensor `i`, extract peak RMS amplitude `A_i` in the footstep window. Model: `A_i = A_0 × r_i^{-0.5} × exp(-beta × r_i)` (Rayleigh-wave geometric spreading r^{-0.5} plus exponential t* attenuation already built into `bank.emit()`). Fit `(x, y, A_0)` by nonlinear least squares over all sensors.

Parameters `beta = 1 / (q × vr)` are profile-dependent — known exactly in simulation, unknown in practice. This module tests how much knowing beta helps (simulation) vs. assuming a flat model (practical).

### 3.5 Polarization Bearing

For each sensor `i` with 3-component recordings `(Z_i, R_i, T_i)`, compute the first-arrival particle-motion ellipse in the (Z, R) plane (P/Rayleigh retrograde ellipse). The major axis of the ellipse in the horizontal plane gives the back-azimuth from sensor `i` to the source.

In the simulation, the radial component is synthesized as:
```python
R_i(t) = bank.emit_radial(r_ij, heading, to_rx, Fz, Fx)
```
where `emit_radial` uses `bank.VF["r_fz"]` and `bank.VF["r_fx"]` from the extended bank (§1.2).

The back-azimuth estimate at sensor `i`:
```
theta_i = atan2(cov_ZR[0,1], cov_ZR[0,0])  # covariance of Z, R in first-arrival window
```

**Triangulation from N bearing lines:** solve the least-squares intersection of N lines through `s_i` at angle `theta_i` (standard azimuth-only localization). With N=3: unique intersection; with N=6: overdetermined, robust.

**CRLB for bearing-only localization:** varies as 1/sqrt(N) bearing lines; degrades at large range because R/Z amplitude ratio decreases (Rayleigh ellipticity is frequency and range dependent but not strongly so in a half-space).

### 3.6 Likelihood Surface Fusion

Combine MFP, TDOA, RSS, and polarization into a joint likelihood surface:
```
L(p_j) = L_MFP(p_j) × L_TDOA(p_j) × L_RSS(p_j) × L_bearing(p_j)
```
where each `L_k(p_j)` is computed independently and expressed as a probability-like score (e.g., normalized Bartlett power for MFP; Gaussian in TDOA residual for TDOA; Gaussian in amplitude residual for RSS; von Mises in azimuth for bearing).

The fusion surface `L(p_j)` is over the same (N_grid,) spatial grid. Its argmax is the fused estimate. Its width (covariance of the top-M% mass) gives a confidence ellipse.

**This is the key diagnostic:** which localizer is the binding bottleneck under which conditions (low SNR, sparse sensors, mis-specified velocity)?

### 3.7 Kalman Track

After per-footstep position estimates, apply a constant-velocity Kalman filter:

```
State: [x, y, vx, vy]
Process noise: Q = diag([0, 0, 0.5, 0.5]) m^2/s^2 (pedestrian acceleration)
Measurement noise: R = sigma_pos^2 × I_2 (from the chosen localizer's empirical RMSE)
Observation: H = [[1,0,0,0],[0,1,0,0]]
```

The Kalman smoother (RTS) run backward over a walking segment reduces per-step RMSE by ~15–25% (consistent with Alajlouni & Tarazaga 2020; Casaclang et al. arXiv 2603.04610).

---

## 4. Blind-Velocity Harness, GDOP / Sensor-Count Monte Carlo, CRLB Validation

### 4.1 Blind-velocity harness

The "blind velocity" scenario: the localizer does not know the true Rayleigh group velocity `v_g` of the profile — it uses a guessed value `v_hat`. This is the dominant real-world error source (§4.3 of doc 02: velocity uncertainty Δv/v = 5–20% dominates timing noise floor).

**Harness design:**
1. Draw a soil profile from the existing `library.json` (true `v_g = 0.92 × vs_top_ms` from `profiles.py`).
2. Synthesize a scene with the true profile's GF bank (N sensors, M footsteps).
3. Run TDOA localizer with `v_hat = v_g × (1 + epsilon)` for `epsilon` ∈ {−30%, −20%, −10%, 0%, +10%, +20%, +30%}.
4. Record `RMSE(epsilon)` to produce a bias-vs-velocity-error curve.

The curve's slope gives `d(RMSE)/d(Δv/v)`, which is the sensitivity of each localizer to velocity miscalibration. MFP (using the GF bank directly) is by construction velocity-exact and should show zero sensitivity; TDOA degrades quadratically with Δv/v at short range and linearly at long range.

**Expected result:** TDOA position bias ≈ `d × Δv/v` at sensor–source distance d. For d = 5 m, Δv/v = 10%: bias ≈ 0.5 m — reproduced exactly by the harness, validating the analytic prediction.

### 4.2 GDOP / Sensor-Count Monte Carlo

**GDOP sweep (fixed N, varying geometry):**
For N sensors on a 10 m × 10 m floor, place them at:
- Corners only (N=4): baseline
- Corners + center (N=5): one extra sensor
- Corners + mid-walls (N=6): standard recommended
- Corners + mid-walls + center (N=7+): diminishing returns
For each placement, compute GDOP as:
```
GDOP(p) = sqrt(trace((A^T A)^{-1}))   where A_ij = (p_j - s_i) / |p_j - s_i| / v_g
```
over a fine grid of source positions `p_j` on the floor. Plot GDOP(x,y) as a heat map. Integrate to get mean GDOP and 95th-percentile GDOP.

**Sensor-count Monte Carlo (varying N, random placement):**
For N = 3, 4, 5, 6, 8, 12:
- Draw 1000 random sensor placements (uniform on floor boundary or interior).
- For each placement, run M=50 footstep localizations with the TDOA module.
- Record median RMSE and 90th-percentile RMSE.
Output: median RMSE vs N curve, showing the 3→4 large gain, 4→6 moderate gain, 6+ diminishing returns (consistent with literature §3.3 of doc 02).

### 4.3 CRLB validation

The CRLB for 2-D TDOA position estimation is:

```
FIM_kl = sum_{(i,j) pairs} (1/sigma_tau^2) × (d tau_ij / d p_k) × (d tau_ij / d p_l)
CRLB_pos = trace(FIM^{-1})
```

where `d tau_ij / d p_k = (unit_ik_k - unit_jk_k) / v_g` (directional cosines, index k = x or y).

`sigma_tau` is computed from the Carter (1987) formula:
```
sigma_tau = 1 / (2π × beta_eff × sqrt(T_obs × SNR_linear))
```
using `beta_eff` from the signal bandwidth (roughly the Rayleigh-wave corner frequency of the footstep wavelet, controlled by `_rise_a2p()` in `wavelet.py`).

**Validation procedure:**
1. For each (N, geometry, SNR, beta_eff) combination, compute the analytic CRLB floor.
2. Run 500 Monte Carlo scenes at that (N, SNR), record empirical RMSE.
3. Plot `empirical RMSE / sqrt(CRLB_pos)` — this ratio should approach 1.0 from above as N_scenes → ∞. A ratio > 2–3 indicates the chosen localizer is far from information-efficient; a ratio close to 1 indicates it is CRLB-tight.

MFP with the correct GF model should be CRLB-tight (it is a maximum-likelihood estimator in Gaussian noise). TDOA with correctly known velocity should also approach the CRLB. RSS will be significantly above CRLB due to the r^{−0.5} model mismatch at short range.

---

## 5. Compute Budget

### 5.1 Per-scene cost breakdown

| Step | Operations | Time (single core) |
|---|---|---|
| Load GF bank (cached) | 48 × 2500 × 6 float32 reads | < 1 ms (memory-bound, cached) |
| footstep wavelet synthesis | 20 footsteps × 1000-pt array ops | < 0.1 ms |
| Multi-sensor scene synthesis | 20 × 6 `emit()` calls = 240 FFTs of 2500 pts | ~5 ms |
| STA/LTA event detection | 6 × 2500 pts | < 0.1 ms |
| MFP over 10,000 grid points | 10,000 × 6 × 1250 complex MACs | ~10 ms |
| TDOA GCC-PHAT (6 pairs) | 6 × FFT(2500) pairs | < 0.5 ms |
| TDOA nonlinear LS solve | 20 footsteps × Gauss-Newton (10 iter × 4×4 system) | < 1 ms |
| RSS nonlinear LS | 20 footsteps × BFGS | < 1 ms |
| Kalman smooth | 20 steps × 4×4 matrices | < 0.1 ms |
| **Total per scene** | | **~20 ms** |

### 5.2 Why pyprop8 is ~10^6–10^7× cheaper than Devito

A Devito 3-D elastic simulation of a 10 m × 10 m × 3 m house at 1 cm spatial resolution requires:
- Grid: 1000 × 1000 × 300 = 3 × 10^8 cells.
- Time steps: 0.25 s at CFL dt = 1 cm / (2 × 3000 m/s) ≈ 1.7 µs → ~150,000 steps.
- Operations per step per cell: ~50 FP multiplications (elastic stencil, 2nd order).
- Total: 3×10^8 × 1.5×10^5 × 50 ≈ 2.25 × 10^15 FLOP.
- At 100 GFLOP/s (4-core laptop): **22,500 s ≈ 6 hours** per scene.

pyprop8 testbed: 20 ms per scene (§5.1). Ratio: **22,500 / 0.02 = 1,125,000×** (about 10^6).

**Monte Carlo implication:** 100,000 scenes for a statistically stable GDOP / CRLB map:
- pyprop8 testbed: 100,000 × 20 ms = **33 minutes** on a single core.
- Devito: 100,000 × 6 hours = **68 years**.

The testbed makes the entire Monte Carlo scientifically feasible. Individual Devito runs are reserved for spot-checking specific configurations that the testbed identifies as interesting.

### 5.3 N values for O(N) estimates

For the default configuration (N_sensors = 6, N_grid = 10,000, n_freq = 1250, N_footsteps = 20 per scene):

| Symbol | N | Dominant operation | Memory |
|---|---|---|---|
| N_sensors | 6 | Loop over sensors in `emit()` and MFP | 6 × bank.n × 4 bytes ≈ 60 kB per scene buffer |
| N_grid | 10,000 | MFP grid search | 10,000 × n_freq × 16 bytes ≈ 200 MB replica cache |
| bank.nr | 48 | Range-lookup interpolation | 48 × n_freq × 16 bytes ≈ 1.5 MB per bank |
| bank.n | 2,500 | FFT size | Fixed |
| n_freq | 1,250 | rfftfreq bins = bank.n // 2 + 1 | Fixed |
| N_footsteps | 20 per scene | Scene synthesis and per-step localizer | Negligible |
| N_scenes | 100,000 | Monte Carlo outer loop | Streaming (one scene at a time) |

Total resident memory per scene: ~200 MB (replica cache, dominated by MFP grid). This can be reduced to ~2 MB by computing MFP replicas on-the-fly using `bank.VF` lookups instead of pre-caching all grid replicas, trading ~2× compute for ~100× memory.

---

## 6. Module Dependency Map

```
library.json  ←  profiles.py (build_library)
      |
      ↓
gfbank_build.py  →  simgeo_banks/{profile_id}.npz
   (extended to save r_fz, r_fx, t_fz, t_fx in addition to z_fz, z_fx)
      |
      ↓
Bank (scenes.py)  →  bank.VF  ←─────────────────────────────┐
   (extended to load r_fz, r_fx; build VF["r_fz"], VF["r_fx"])  │
      |                                                         │
      ↓                                                         │
testbed/scene_2d.py           (new, thin wrapper)              │
  ├─ place_sensors(N, room_m)  →  sensor_positions             │
  ├─ draw_path(room_m, speed)  →  gt_positions, footstep_times │
  └─ synthesize(bank, sensors, gt_pos, times, rng)             │
       ↓                                                        │
  recordings[N, T]  +  gt_positions                            │
      |                                                         │
      ↓                                                         │
testbed/localizers.py         (new)                            │
  ├─ MFP(recordings, sensors, bank, grid)  ─────────uses VF────┘
  ├─ TDOA(recordings, sensors, v_hat)
  ├─ RSS(recordings, sensors, beta_hat)
  ├─ Polarization(recordings_3c, sensors)
  └─ FusedLikelihood(all_localizers, weights)
      |
      ↓
testbed/harness.py            (new)
  ├─ run_gdop_sweep(N_sensors_range, N_scenes)
  ├─ run_blind_velocity(epsilon_range, N_scenes)
  └─ run_crlb_validation(SNR_range, beta_eff_values, N_scenes)
      |
      ↓
testbed/plots.py              (new — outputs figures for paper)
```

All new files live in `simgeo_v42/testbed/` or `localization_1d/testbed/` — to be decided.

---

## 7. Design Decisions and Open Questions

### 7.1 Which profiles to use

The 282-bank library covers the full Israeli soil-profile catalog. For the 1-D testbed, a representative subset of ~10–20 profiles (one per family: soft_soil, sand, gravel, clay, rock, kurkar, asphalt, concrete) is sufficient for the GDOP/CRLB sweeps. The full 282 is needed only for the blind-velocity harness (to sample across the Vs distribution).

### 7.2 Noise model

The current `scenes.py` does not add geophone self-noise. The testbed needs a noise model. Recommendation: colored Gaussian noise whose PSD matches the GS-11D geophone self-noise floor (~10 ng/√Hz = 10×10^{-9} m/s²/√Hz → velocity noise PSD = (10×10^{-9}/ω)^2 m²/(m/s)²/Hz at each frequency). This is flat in velocity above the geophone corner (4.5 Hz), rolling off as 1/ω below.

SNR at sensor `i` for a footstep at range `r_i` is then:
```
SNR_i = rms(emit(r_i)) / rms(noise_in_band)
```
which the testbed can compute analytically from the bank spectra without rendering noise explicitly (for CRLB purposes), or render explicitly (for MFP/TDOA testing).

### 7.3 Indoor vs. outdoor physics

The existing pyprop8 pipeline and the localization testbed as described here model **outdoor half-space propagation** (soil Rayleigh waves, t* attenuation, cylindrical spreading). The house-localization goal involves **indoor structural vibration** (Lamb plate waves in a concrete slab, floor boundary reflections, dispersive flexural modes — see doc 02, §1.2).

The pyprop8 / 1-D testbed is a proxy for the indoor problem in the following sense:
- Correct frequency-dependent attenuation and waveform shape.
- Correct multi-sensor geometry and GDOP structure.
- Wrong dispersion law: Rayleigh-wave group velocity is roughly non-dispersive above the corner frequency, whereas flexural Lamb waves have `v_ph(f) ∝ f^{1/2}`.
- No reflections from room boundaries.

The testbed is therefore ideal for:
- Benchmarking TDOA, MFP, RSS, bearing, Kalman architectures under controlled conditions.
- Measuring GDOP and CRLB for any N/geometry.
- Testing the blind-velocity harness (Δv/v sensitivity).
- Comparing localizers' relative performance.

It is **not** a direct model of the flexural-wave indoor scenario. The upgrade path to flexural waves is to replace the `GF[r]` lookup with a thin-plate transfer function `H(sensor, source, ω) = G_0 × H_0(r, ω) × exp(−iφ_disp(r, ω))` where `φ_disp` encodes the Kelvin-Voigt dispersion. This replacement does not change any of the localizer architecture — only the replica source in MFP and the assumed velocity model in TDOA.

---

## Sources

- pyprop8 Green's-function engine: `simgeo_v42/gfbank_build.py`, `scenes.py`, `wavelet.py`, `profiles.py` (this codebase).
- Indoor localization methods ranked: `research/house_localization/02_indoor_localization_methods.md` (this codebase), 2026-07-08.
- Sensor hardware and timing: `research/house_localization/04_sensor_hardware_hybrid.md` (this codebase), 2026-07-08.
- TDOA CRLB (Carter 1987 / Knapp-Carter 1976): cited in doc 02, §4.1.
- SO-TDOA (Bahroun et al., J. Sound Vib. 2014, arXiv 1211.3233): cited in doc 02, §1.2 and §2.5.
- MFP for structural vibration (Jacquelin et al. 1996; EDMF, Smith & Drira, MSSP 2022): cited in doc 02, §1.3.
- Energy/RSS + Kalman (Alajlouni & Tarazaga 2020): cited in doc 02, §1.6.
- Reservoir computing footstep localization (Casaclang et al. arXiv 2603.04610, 2026): cited in doc 02, §1.5.
- GaitVibe+ (arXiv 2212.03377): cited in doc 02, §2.2.
- Geophone self-noise, RS4D hybrid architecture: `research/house_localization/04_sensor_hardware_hybrid.md`, §1.2 and §2.1.
