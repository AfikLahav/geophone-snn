> **Research + design document — house-localization pivot ("wallhacks").** Synthesized from docs 01–06 + targeted web research, 2026-07-08. Design input, not a validated experiment. Indexed in [README.md](README.md).

---

# Dataset Design for Devito 3-D Elastic House Simulation

**Scope.** This document specifies every dimension of the training corpus the Devito voxel-FDTD simulator will produce: sensor placement, source grid, Green's-function reciprocity library, confuser modeling, storage accounting, and train/localize split with sim-to-real risk management. It is the downstream authority for the simulator build (doc 06) and feeds the model architecture (doc 01) and localization methods (doc 02).

---

## 1. Sensor Placement

### 1.1 Node architecture: 4-channel per physical node

Each physical node mounts two sensors co-located on the same structural element:

- **1× 4.5 Hz moving-coil geophone (vertical)** — primary TDOA and fingerprinting channel. Self-noise ~10 ng/√Hz at 30 Hz; flat velocity response above 4.5 Hz; captures floor fundamentals (5–8 Hz) through the full footstep band (20–250 Hz). No DC response.
- **1× ADXL355 MEMS (3-axis)** — secondary channels. Self-noise ~245,000 ng/√Hz, so it contributes no new SNR below ~78 kHz, but it supplies: (a) 3-axis components for bearing/polarization; (b) DC/tilt for sensor orientation logging; (c) strong-motion anti-clip backup for heavy impacts. This is the RS4D architecture (Raspberry Shake 4D, >1,000 units deployed).

Each node therefore yields **4 channels** (1 vertical geophone + 3 MEMS: Z, N, E). For a 6-node array: **24 raw channels** per recording. In the simulation, only the vertical geophone channel (particle velocity vz) is the primary product; the 3 MEMS horizontals are cheap to record (same Devito SparseTimeFunction sweep) and provide polarization features at zero additional simulation cost.

### 1.2 Mounting priority and "low + top ring" rationale

From doc 04 §4.2, coupling quality in descending order: floor slab (bonded) > underfloor soffit > column base > load-bearing wall > foundation >> roof. Roof is ruled out categorically: 20–40 dB insertion loss through all structural joints, and wind-induced vibration (0.1–10 Hz) swamps footstep signal. Mid-floor-height wall mounting is ruled out for the same reason: moderate coupling, heavy attenuation above 80 Hz, wall resonance noise at 10–50 Hz.

The two viable rings are:

**Low ring** — sensors bonded directly to floor slabs at or near slab level. On an exterior wall, the most accessible equivalent is the floor-to-wall junction (sill plate / ring beam / foundation stem wall). These positions capture the dominant flexural (A0 Lamb) wave launched by a footstep before it traverses the wall junction, giving broadband signal 5–250 Hz. Best TDOA utility; highest amplitude. Corners are preferred because they sample both room dimensions and are structural nodes (column-to-beam joints), which are high-amplitude locations in the building's modal shapes.

**Top ring** — sensors bonded to the ceiling slab soffit or top-of-wall slab edge. On a multi-storey house this means the underside of the floor above, immediately adjacent to the exterior wall. On a single-storey with a flat concrete roof, it means the roof slab at the parapet. These positions add vertical parallax: a signal arriving at a low-ring sensor vs. a top-ring sensor in the same XY column has different propagation paths through the floor slab, enabling the system to distinguish which floor the source is on. The top ring is not expected to match the TDOA fidelity of the low ring, but it adds one critical piece of information (floor discrimination) at low incremental cost.

Skipping the mid-ring is justified because: (1) mid-wall sensors collect only low-frequency (<20 Hz) structural bending modes and heavily attenuated high-frequency content; (2) their TDOA utility is poor (velocity is uncertain, path is through multiple wall elements); (3) they add computational cost to the simulation (more sensor positions) without improving floor-level discrimination or 2-D localization. The information gain from a mid-ring sensor is superseded by the combination of low-ring high-SNR TDOA and top-ring floor discriminant.

### 1.3 Count: 6 nodes (4 low + 2 top) per house

| Count | Configuration | What each placement buys |
|---|---|---|
| 4 low-ring corner sensors | 1 per exterior corner, bonded to floor slab / ring beam junction | Full 2-D TDOA coverage; 6 TDOA pairs overdetermine 2-D position; corner placement minimizes GDOP for sources in the floor interior |
| +1 low-ring mid-wall (long axis) | Mid-point of the longest wall | Breaks symmetry ambiguity in elongated rooms; halves effective baseline for sources near that wall |
| +1 top-ring corner (single point) | Same corner as one low-ring sensor but at ceiling/roof slab level | Provides vertical parallax for floor-level discrimination; single top sensor sufficient to compare arrival-time difference between slab levels |

Total: **6 nodes × 4 channels = 24 channels per house**. This matches the "recommended 4–6" verdict from doc 04 §3 and the reservoir-computing study (11 accelerometers) evidence that performance saturates past 6 sensors (arXiv:2603.04610). Extending to 8 nodes (2 more top-ring positions) is the natural scale-up for a two-storey house dataset and can be run as a second configuration without rewriting the Devito forward problem.

### 1.4 Simulation representation

Each geophone node is a point receiver in the Devito SparseTimeFunction grid, recording vz (vertical particle velocity). The ADXL355 horizontals map to vx and vy at the same grid point. Under reciprocity, all sensor positions become source positions; 6 sensors → 6 reciprocal simulations per house. All floor interior grid positions are recorded passively in each shot. This is the correct setup per doc 06 §8.

---

## 2. Source Grid: Interior Footstep Positions

### 2.1 Floor grid

The source grid covers all walkable interior floor positions on every simulated floor. For a canonical 10×12 m single-storey house:

| Grid parameter | Value | Rationale |
|---|---|---|
| Spatial step | 0.25 m | Half the 0.5 m step used in doc 06; finer grid costs nothing at simulation time (all positions recorded simultaneously); 0.25 m resolves the ~0.5 m localization target; sub-0.25 m gains are blocked by wave-velocity calibration uncertainty (doc 05 §3) |
| Positions per floor (10×12 m interior, walls excluded, ~0.5 m margin) | ~(9×11) = ~1,764 positions (0.25 m grid, 9 m × 11 m walkable) → trim to ~1,500 excluding walls, doors, fixed furniture zones | |
| Floors | 1 (single-storey baseline) → 2 (two-storey extension) | Single-storey first; two-storey doubles source positions and adds inter-floor classification demand |
| Wall exclusion margin | 0.3 m from each wall | Near-wall near-field artifacts in FDTD; humans do not walk within 0.3 m of walls in practice |

At 0.25 m grid on a 9×11 m walkable area: **37 × 45 = 1,665 grid points** (approximately; actual count varies by floor plan). Round to **~1,500 walkable positions** per floor per house.

### 2.2 Reciprocity inversion

With 6 sensor nodes and ~1,500 source positions:
- **Without reciprocity:** 1,500 simulations per house
- **With reciprocity:** 6 simulations per house, recording vz (and vx, vy) at all 1,500 floor positions simultaneously

Savings: **250× per house** (vs the 60× quoted in doc 06 which used a coarser 0.5 m grid; the 0.25 m grid quadruples positions while keeping sensor count fixed). This is the dominant cost reduction in the entire pipeline.

### 2.3 Source-time function: the footstep GRF model

The footstep is modeled as a vertical point force injected into τzz at the floor cell. The source-time function follows the ground reaction force (GRF) biomechanics literature:

**Single-impact pulse parameters** (walking, 700–800 N body weight, hard heel-strike on concrete):
- **Heel-strike transient:** rise time 15–30 ms, peak force 1.1–1.5× body weight (~770–1,200 N), high-frequency content to 250 Hz
- **Loading response:** 80–120 ms plateau at 0.8–1.0× body weight, energy concentrated 1–30 Hz
- **Toe-off:** 80–100 ms, peak ~1.0–1.2× body weight, softer spectrum than heel-strike

For simulation the most important feature is the **heel-strike impulse**, which dominates the 80–250 Hz content used for TDOA. A practical proxy:

```python
def grf_wavelet(t, f0=20.0, t0=0.05, bw=700):
    """
    Ricker-derivative (Mexican hat derivative) scaled to ~700 N peak.
    f0=20 Hz centers the GRF spectrum; heel-strike transient.
    """
    u = np.pi * f0 * (t - t0)
    return -bw * 2 * u * np.exp(-u**2)
```

This is a Ricker-derivative centered at 20 Hz, which produces: (a) near-zero DC (no static load offset — appropriate for dynamic transient only); (b) peak energy at 5–50 Hz; (c) usable energy to ~100 Hz; (d) negligible energy above 200 Hz. For higher-fidelity GRF, parameterize the actual ISO 10137 walking force model (sum of Fourier harmonics at step frequency f_s and harmonics 2f_s, 3f_s... up to 8th harmonic):

```python
def iso_walking_force(t, f_s=1.8, alpha=[0.41, 0.069, 0.033, 0.013], body_weight=700):
    """
    ISO 10137 walking force model. f_s = step frequency (1.5-2.2 Hz).
    alpha[k] = dynamic load factor for harmonic k+1.
    """
    F = body_weight * np.ones_like(t)
    for k, a in enumerate(alpha):
        F += body_weight * a * np.sin(2 * np.pi * (k+1) * f_s * t - np.pi/2)
    return F
```

The ISO model gives realistic spectral content at low harmonics (1–8 Hz) but lacks the heel-strike transient (80–250 Hz). **For the dataset: use the Ricker-derivative for the high-frequency TDOA component and modulate its envelope at step frequency f_s to inject multi-step sequences.** Both are synthesized in post-processing by convolving with the stored Green's function — no re-simulation is needed.

**Source-diversity parameters to randomize per training sample (post-processing):**
- Body weight: 500–1,000 N (uniform or log-normal)
- Step frequency: 1.0–2.5 Hz (children to brisk adults)
- Heel-strike peak force multiplier: 1.0–1.6× body weight
- Heel-strike rise time: 10–40 ms (soft shoe vs. hard heel)
- Walking trajectory: straight-line, L-shaped, random-walk (all synthesized from GF slice convolution)

This parametric diversity is O(seconds) to generate after the GF library exists — it costs no simulation budget.

---

## 3. The Green's-Function Library and Its Triple Duty

### 3.1 Construction (one time per house geometry)

The Devito forward problem shoots **N_sensor = 6 reciprocal simulations** per house. Each simulation places a vertical point source (force in z) at one sensor position and records vz at all N_pos = 1,500 floor positions simultaneously for T = 0.5 s:

```
GF_library[sensor_k, position_j, t] = vz(x_j, t | source at x_k)
```

Dimensions per house: 6 sensors × 1,500 positions × N_time × N_comp. At 100 Hz scenario (doc 06): N_time = 20,833 samples (0.5 s at dt=24 µs) — this is the raw simulation output. For the stored dataset, downsample to 2,000 SPS (adequate for 250 Hz Nyquist at the stored level): N_time_stored = 1,000 samples (0.5 s × 2,000 SPS). Three components (vz primary + vx, vy from ADXL355 analog positions): N_comp = 3.

Stored shape per house: `[6, 1500, 1000, 3]` float32 = 6 × 1500 × 1000 × 3 × 4 B = **108 MB per house** (3-component, full time trace).

If only vertical (vz) component is stored: `[6, 1500, 1000, 1]` = **36 MB per house**.

For the fingerprint/MFP use case, only frequency-domain features need to be stored rather than full time traces. A 128-point FFT (64 frequency bins) per sensor-position pair: `[6, 1500, 64]` complex64 = 6 × 1500 × 64 × 8 B = **4.6 MB per house**. This is the matched-field replica field — far smaller.

### 3.2 Triple duty of one corpus

**Duty 1 — ML training dataset.** Synthetic footstep waveforms at each of the 1,500 positions are generated by convolving the GRF source-time function with the appropriate GF column. The resulting multi-channel waveform (6 channels, 0.5 s) is the training input; the floor-grid position (i, j) or (x, y) is the label. Feature extraction (per-sensor energy, TDOA, spectral entropy) is applied in post-processing. Training samples are unlimited in number from this corpus — each new random walk trajectory through the floor grid generates a new labeled sequence at O(microseconds) per step.

**Duty 2 — Matched-field localization dictionary.** The GF library *is* the replica field for Matched Field Processing. At inference time, the observed multi-channel waveform is cross-correlated against GF_library[:, j, :] for each candidate position j. The position with maximum Bartlett (or MVDR) processor output is the MFP estimate. No model fitting required post-construction — the simulation *is* the model. This is the Tier-1 localizer (zero training data, physics baseline) described in doc 01.

**Duty 3 — Fingerprint database for KNN retrieval.** Extracted features from GF_library (per-sensor energy, TDOA between sensor pairs, dominant frequency at each sensor) form a 33-feature vector per candidate position, stored as a [1500 × 33] matrix per house. KNN search over this matrix in <1 ms for a house with 1,500 positions. This is identical to WiFi fingerprint-based positioning, except the "radio map" is physics-derived, not empirically measured. The approach was validated in the RF localization literature: large-scale synthetic pretraining reduces real-world localization error by 50% (ScienceDirect, 2025), with calibrated simulation fidelity outweighing raw dataset size.

### 3.3 Why one simulation produces all three outputs

The key structural property: the GF library is a **linear superposition kernel**. Any footstep trajectory is a linear sum of GF columns convolved with the GRF time function. The same library supports:
- Arbitrary source strength (amplitude scaling)
- Arbitrary step-rate (modulated convolution timing)
- Arbitrary trajectory (position-indexed superposition)
- Any measurement window length (time slicing)
- Domain randomization of surface roughness, furniture mass loading (perturbative correction to GF, or separate library per perturbation)

This linearity is exact under the FDTD model's elastic-wave assumption and fails only for: (a) contact nonlinearity very close to the source (< 0.5 m, doc 06 §8); (b) nonlinear soil response at high strains (not relevant here).

---

## 4. Confuser Modeling

Confusers are vibration sources that a real sensor will observe but that are not footsteps. They must be in the training set for the classifier (doc 01 §2) to reject them and for the localizer to not misfire on them.

### 4.1 Confuser taxonomy and spectral signatures

| Confuser class | Dominant frequency range | Character | Structural coupling path |
|---|---|---|---|
| HVAC blower/fan | 5–50 Hz, harmonics of rotation rate (e.g., 6.6 Hz × N for a 400 rpm drum) | Quasi-periodic, continuous, stationary source | Motor vibration → duct/mount → floor/ceiling slab |
| HVAC compressor | 25–60 Hz (50 Hz power-line harmonic for 2-pole motor); 100 Hz and 200 Hz if 4-pole | Line-rate tone + broadband turbulence | Compressor → refrigerant lines → building frame |
| Washing machine (spin) | 0–20 Hz during acceleration; 8–25 Hz at spin plateau (400–800 rpm × 1/60) | Intermittent strong broadband; strong unbalance harmonics | Appliance mass → floor slab; strongest of all appliances |
| Refrigerator compressor | 25–60 Hz narrow tone; intermittent (duty cycle) | Narrow-band, amplitude-modulated by on/off cycling | Compressor vibration → cabinet → floor |
| Mains hum | 50 or 60 Hz (depending on grid) and harmonics (100, 150, 200 Hz) | Pure tones, constant, ubiquitous | Electromagnetic coupling into power lines → motor vibration; also direct inductive coupling into geophone coil if co-located with transformers |
| Plumbing / water hammer | Transient: broadband 5–170 Hz; quasi-continuous flow: 10–80 Hz | Short impulse (0.1–0.5 s); localized to pipe route | Pipe rigidly mounted to wall/floor joists → slab |
| Wind / exterior | 0.1–10 Hz | Stochastic, broadband, correlated across all sensors | Roof/wall excitation |
| Pets (cat/dog footsteps) | 2–5 Hz cadence; spectral content 20–100 Hz per impact | Similar to human footstep but 5–15× lower peak force | Same as human footstep — same structural path |

The mains-hum (50/60 Hz and harmonics) is also an EM confuser that appears as an additive tone on the geophone coil output independent of structural coupling. In simulation it is best modeled as a post-hoc additive signal.

### 4.2 Confuser simulation strategy: source-signature ⊛ Green's function

Each confuser is a **fixed-location vibration source** (or a fixed pipe route) with a known spectral signature. The confuser signal at sensor k is:

```
confuser_k(t) = STF_confuser(t) * GF_library[k, pos_confuser, :]
```

where `pos_confuser` is the floor-grid position nearest to the appliance/pipe and `STF_confuser` is the confuser source-time function. Because the GF library covers all 1,500 interior positions, any appliance location already has a precomputed Green's function — **no additional simulation is needed**.

Source-time functions per confuser class:

```python
# HVAC fan: quasi-periodic, frequency f_hvac with harmonic distortion
def stf_hvac(t, f_hvac=6.6, harmonics=5, sigma=0.01):
    s = np.zeros_like(t)
    for k in range(1, harmonics+1):
        phase_jitter = np.random.normal(0, sigma)
        s += (1/k) * np.sin(2*np.pi*k*f_hvac*t + phase_jitter)
    return s

# Washing machine: unbalance + spin-up
def stf_washing(t, f_spin=8.0, ramp_time=30.0, amplitude=1.0):
    f_inst = f_spin * np.clip(t / ramp_time, 0, 1)
    phase = 2*np.pi * np.cumsum(f_inst) / len(t) * ramp_time
    return amplitude * np.sin(phase)

# Mains hum: pure additive (not via GF — EM coupling)
def stf_mains(t, f_grid=50.0, amplitude=0.05):
    return amplitude * np.sin(2*np.pi*f_grid*t)

# Plumbing transient: broadband impulse, pipe-position GF
def stf_water_hammer(t, t0=0.1, tau=0.02):
    return np.exp(-((t-t0)/tau)**2) * np.sin(2*np.pi*50*(t-t0))

# Pet footstep: scaled-down GRF (1/8 body weight, 2-5 Hz cadence)
def stf_pet(t, f_s=3.0, bw=80):
    return grf_wavelet(t, f0=20.0, t0=0.1, bw=bw)
```

### 4.3 Post-hoc noise (ground-noise recipe, re-pointed to house GFs)

The existing ground-noise recipe from the outdoor corpus generates a stochastic vibration background from a spatial distribution of distant random sources. For the indoor case, re-point the recipe:

1. **Ambient structural noise floor** — draw M = 50–100 random positions from the floor grid, each with a 1/f² power spectral density source (brownian noise). Convolve each with its GF column and sum. This models micro-seismic hum, distant traffic, and building creep.

2. **Per-band noise floors:**
   - 0.1–5 Hz: wind/tidal, PSD ∝ f⁻² — added directly to all channels (spatially coherent across sensors, since source is exterior)
   - 5–20 Hz: HVAC rotation harmonics — modeled via STF above
   - 50/100/150/200 Hz: mains hum tones — additive sinusoid, randomize amplitude in [0.01, 0.10] × signal RMS
   - 80–120 Hz: appliance broadband — random-position GF convolution
   - All bands: sensor self-noise — white noise at 10 ng/√Hz (geophone) and 245,000 ng/√Hz (MEMS), scaled to sample rate

3. **SNR parameterization.** The training corpus should span SNR = 0–30 dB (signal = peak footstep energy in 20–80 Hz band; noise = RMS of all confuser contributions in same band). The low-SNR regime (0–10 dB) will only appear in the training set for the classifier; the Tier-1 MFP localizer should document its accuracy floor as a function of SNR.

4. **Confuser-only samples.** Approximately 20% of training samples should have no footstep (class = "nothing") and only confuser + noise. This is the primary driver of the classifier's false-positive rate. If the classifier never sees HVAC-only waveforms during training, it will not learn to reject them.

### 4.4 Appliance location sampling

For each training house, randomly draw appliance positions from a set of "appliance rooms" (kitchen: washing machine, refrigerator; living room: AC unit; bathrooms: plumbing). These are fixed positions within each house instance, sampled once per house and held constant across all samples from that house. This models the real deployment scenario (appliances don't move).

---

## 5. Scale and Storage

### 5.1 Dataset dimensions

| Dimension | Baseline | Extended |
|---|---|---|
| Houses | 100 | 500 |
| Sensor nodes per house | 6 | 6 |
| Channels per node | 4 (1 geo-Z + 3 MEMS) | 4 |
| Total channels per house | 24 | 24 |
| Floor positions per house | ~1,500 (0.25 m grid, 9×11 m) | ~1,500 |
| Reciprocal sims per house | 6 | 6 |
| Sim duration | 0.5 s | 0.5 s |
| Stored sample rate | 2,000 SPS | 2,000 SPS |
| Stored samples per trace | 1,000 | 1,000 |
| Components stored (vz only) | 1 | 3 (vz + vx + vy) |
| float32 bytes per sample | 4 | 4 |

### 5.2 Raw GF library storage (float32, full time traces)

**1-component (vz only):**
```
per house = 6 sensors × 1,500 positions × 1,000 samples × 1 component × 4 B
          = 36,000,000 B = 36 MB per house
100 houses = 3.6 GB
500 houses = 18 GB
```

**3-component (vz + vx + vy):**
```
per house = 6 × 1,500 × 1,000 × 3 × 4 B = 108 MB per house
100 houses = 10.8 GB
500 houses = 54 GB
```

### 5.3 Compact fingerprint/MFP replica store (frequency domain)

128-point FFT of each trace (64 real + 64 imag = 128 complex64 = 512 B):
```
per house = 6 × 1,500 × 128 × 2 × 4 B = 9.2 MB per house (per component)
500 houses × 1 component = 4.6 GB
500 houses × 3 components = 13.8 GB
```

This is the form used by the MFP/fingerprint lookups.

### 5.4 Synthetic training waveforms (after post-processing)

Each training sample is one detected footstep event: a 0.5 s window per sensor channel (6 channels × 1,000 samples × 4 B = 24 KB per event). At 1,000 events per house (many footfall trajectories synthesized):
```
100 houses × 1,000 events = 100,000 events × 24 KB = 2.4 GB
500 houses × 1,000 events = 500,000 events × 24 KB = 12 GB
```

If features are pre-extracted (33-feature vector per step, float32): 500,000 × 33 × 4 B = 66 MB — negligible.

### 5.5 Storage summary

| Dataset form | 100 houses | 500 houses |
|---|---|---|
| Raw GF library, 1-comp vz | 3.6 GB | **18 GB** |
| Raw GF library, 3-comp | 10.8 GB | **54 GB** |
| MFP replica store (FFT, 1-comp) | 0.9 GB | 4.6 GB |
| Synthetic training waveforms (raw, 6ch × 0.5s) | 2.4 GB | 12 GB |
| Pre-extracted features only | ~7 MB | ~35 MB |
| **Total (3-comp GF + training raw + MFP)** | **~14 GB** | **~70 GB** |

**The ~100 GB estimate from doc 06 is confirmed as a reasonable ceiling for the 500-house, 3-component, full time-trace dataset.** The refined estimate is 54–70 GB for 3-component GF library + synthetic training waveforms at 500 houses. At 100 houses the working set is 10–14 GB, comfortable on a single NVMe drive.

The doc-06 estimate used a 0.5 m grid (480 positions) and 1-comp storage; moving to 0.25 m (1,500 positions) at 3 comp adds a factor of 9.4, but the frequency-domain (MFP replica) form stays under 14 GB regardless of grid density.

### 5.6 Compute cost cross-check (doc 06 baseline, updated for 6 sensors, 0.25 m grid)

At 100 Hz scenario, RTX 3090, 6 reciprocal sims (not 8 as in doc 06):
```
Per house wall-clock = 6 × 6.7 s = 40 s
100 houses = 67 min
500 houses = 5.6 hr
```

At 250 Hz, 8×A100:
```
Per house = 6 × 261 s / 13 = 120 s = 2 min
500 houses = 17 hr ≈ $560 AWS p4d.24xlarge (spot)
```

These costs support a 500-house corpus as feasible in a single cloud session.

---

## 6. Train / Localize Split and Sim-to-Real Risk Management

### 6.1 Dataset splits

The 500-house corpus is split along **house geometry**, not along positions within a house. Houses must not be shared between train and test splits because the GF library for a specific house is the "answer key" for all positions in that house — testing on the same house would leak physics.

| Split | Houses | Purpose |
|---|---|---|
| Train | 350 (70%) | Classifier training (LightGBM); Tier-3 MLP localizer training |
| Validation | 75 (15%) | Hyperparameter selection, early stopping |
| Test (held-out geometry) | 75 (15%) | Generalization to unseen house geometries |
| Real-world calibration | N/A (0 houses) | Small set of ~100–500 real footsteps at known positions, used only for Tier-2 sim-to-real calibration (ridge regression) |

Tier-1 (MFP/KNN from simulation) operates on the **localization dictionary**, which is the GF library for the specific test house. It is not "trained" on the train split — it uses only the simulated replica field for that house. This split is necessary to avoid the trivial case of locating steps in a house whose GF library is identical to training data.

**Within-house simulation train/localize distinction:** the fingerprint dictionary for localization uses GF columns at **all 1,500 grid positions**. The training set for supervised ML uses a random 80% of grid positions as labeled samples. The held-out 20% of positions (300 per house) form the within-house localization test set. For publication, both the held-out-geometry test (75 houses) and within-house test (20% grid positions across all houses) are reported separately.

### 6.2 House geometry diversity

House geometries must span the sim-to-real gap in structural properties. Per doc 06, the Devito model is parameterized by (slab thickness, wall thickness, room layout, soil type). For 500 houses, randomize:

| Parameter | Range | Distribution |
|---|---|---|
| Floor plan | rectangular, L-shaped, T-shaped | Uniform over 3 classes |
| Floor area | 60–200 m² | Log-uniform |
| Slab thickness | 0.12–0.25 m | Uniform |
| Wall thickness | 0.15–0.30 m (concrete) or 0.20–0.40 m (brick) | Uniform per wall type |
| Wall material | concrete / brick | 50/50 |
| Soil Vs_min | 100–300 m/s | Log-uniform |
| Number of floors | 1 / 2 | 60% / 40% |
| Room count | 3–8 | Uniform |

This is domain randomization over the structural properties. It serves two purposes: (1) the classifier and Tier-3 localizer train on diverse house physics and generalize better; (2) the held-out test set probes a part of the parameter space the model has not seen.

### 6.3 Sim-to-real risk factors and mitigations

**Risk 1: Velocity model error.** The Devito model uses nominal Vp, Vs, ρ for concrete and soil. Real slabs vary ±10–30% in Vs due to rebar density, aggregate gradation, and age. The MFP localizer is sensitive to velocity error because arrival times shift; a 10% velocity error at 3 m propagation distance → 3 cm timing error at 300 m/s → 9 mm position error (negligible) at 100 Hz, but 10% of a 5 m path at 200 m/s → 25 ms → 5 m error if using constant-velocity TDOA without correction. Mitigation: (a) domain-randomize velocity ±20% around nominal in training; (b) include an on-site calibration step (3–5 hammer strikes at known floor positions) to estimate the actual velocity profile before deploying the localizer — this is the Tier-2 calibration described in doc 01.

**Risk 2: Wall junction transmission loss.** The Devito FDTD model with voxelized concrete accurately models transmission through slab-to-wall junctions within ±5 dB for wavelengths >> cell size (holds at 100 Hz, marginally at 250 Hz). Real junctions have isolation joints, air gaps, and elastomeric pads that the voxel model cannot represent. Mitigation: train with stochastic "junction transmission loss" applied post-hoc as a random per-sensor gain (-3 to -12 dB per junction traversal), effectively simulating variable coupling. This prevents the model from relying on absolute amplitude at sensors that require cross-junction propagation.

**Risk 3: Modal frequency shifts from furniture mass loading.** Interior furniture (100–400 kg) shifts floor-mode frequencies by 1–5 Hz (sparse modes at <250 Hz, doc 05 §1.4). Fingerprint features that depend on modal resonances (dominant frequency at each sensor) will drift. Mitigation: (a) use arrival-time-based features (TDOA, group velocity) as primary features — these are less sensitive to modal shifts than resonance peaks; (b) store dominant frequency as a secondary feature and accept 5–10% fingerprint accuracy degradation due to furniture changes; (c) document this as a known limitation in the publication.

**Risk 4: Staircase artifacts in voxelized walls.** Diagonal walls (45°) introduce staircase diffraction at high frequencies. At dx = 0.15 m and 100 Hz, the staircase step is comparable to λ_B ≈ 1.52 m → negligible. At 250 Hz, λ_B ≈ 0.61 m and dx = 0.06 m → ~10% of wavelength → small but non-zero artifact. Mitigation: (a) limit house geometries to rectangular/L-shaped/T-shaped layouts that use axis-aligned walls; (b) validate against 5 FEniCSx shell-FEM reference runs (doc 06 §10 recommendation).

**Risk 5: Sensor coupling variability.** Real epoxy-bonded geophones have ±1–3 dB coupling variation. Mounting resonances from imperfect bonding add peaks at 40–150 Hz. Mitigation: add random per-sensor frequency-response filter (±2 dB, random pole-zero pair at 50–150 Hz) to training samples. This is a 1-line post-processing step.

**Risk 6: Near-source nonlinearity.** The GRF at the foot-floor contact zone is a contact nonlinearity (Hertzian contact, soft toe vs. hard heel). The FDTD model uses linear elasticity throughout, which is accurate at r > 0.5 m from the source (doc 06 §8). Steps very close to a sensor (<0.5 m) will have systematic amplitude bias in the simulation. Mitigation: exclude source positions within 0.5 m of any sensor from the fingerprint dictionary and training set. This removes ~30–50 of the 1,500 positions per house — negligible.

### 6.4 The Tier-1/2/3 localization ladder under sim-to-real degradation

| Tier | Data required | Expected error (in-sim) | Expected error (real, no calibration) | Expected error (real, 100 calibration steps) |
|---|---|---|---|---|
| Tier-1: MFP/KNN on simulation GF | 0 labeled real steps | 0.15–0.30 m (interpolation of 0.25 m grid) | 0.40–1.2 m (velocity error dominant) | 0.25–0.5 m (calibration corrects velocity) |
| Tier-2: Ridge regression on Tier-1 output | 100–500 real steps | — | — | 0.20–0.40 m |
| Tier-3: MLP/LGBM end-to-end | 500–5,000 real steps | 0.10–0.20 m | 0.30–0.60 m | 0.15–0.35 m |

The publication-defensible baseline is **Tier-1 with 100-step calibration → 0.25–0.50 m accuracy**, which matches the state-of-the-art Mirshekari result (0.34 m) and the reservoir-computing result (0.47–0.65 m) while using zero labeled real data for the core localizer.

---

## 7. Summary of Design Decisions

| Dimension | Decision | Key constraint |
|---|---|---|
| Sensor count | 6 nodes (4 low-ring corners + 1 low-ring mid-wall + 1 top-ring) | Saturation past 6 (arXiv:2603.04610); floor discrimination from top ring |
| Channels per node | 4 (geo-Z + MEMS X/Y/Z) | RS4D architecture; no extra sim cost for horizontal channels |
| No mid-ring | Excluded | 20–40 dB coupling loss through wall elements; wind contamination |
| No roof | Excluded | Doc 04 §4.4; wind-dominated, TDOA useless |
| Source grid density | 0.25 m | Matches 0.25 m TDOA accuracy target; 4× positions vs doc 06 but zero extra sim cost (reciprocity) |
| Reciprocal sims | 6/house (not 1,500) | 250× cost reduction |
| Source-time function | Ricker-derivative centered at 20 Hz; ISO-walking envelope for multi-step | Reproduces heel-strike 5–250 Hz content; parameterize in post-processing |
| Houses | 100 baseline / 500 full | 100: 10–14 GB, feasible on single NVMe; 500: ~70 GB, feasible on multi-TB NVMe |
| GF storage form | Full time trace (2,000 SPS, 1-comp): 36 MB/house; FFT replica (128-pt): 4.6 MB/house | Time traces for MLP training; FFT replicas for MFP lookup |
| Confuser strategy | STF ⊛ GF from fixed appliance positions (no extra sim) + post-hoc EM tones | Same GF library, zero additional simulations |
| Train/test split | By house geometry (70/15/15) | Physics leakage prevention |
| Sim-to-real | Domain randomization (velocity ±20%, junction loss, coupling filter) + Tier-2 calibration | Matches RF fingerprint synthetic-pretrain literature (50% error reduction) |

---

## Key Sources

- [Mirshekari et al. (2018) Occupant Localization via Footstep-Induced Structural Vibration — 0.34 m](https://www.sciencedirect.com/science/article/abs/pii/S0888327018302280)
- [GaitVibe+ (arXiv:2212.03377) — 0.22 m vibration-only](https://arxiv.org/pdf/2212.03377)
- [Reservoir Computing footstep localization (arXiv:2603.04610) — convergence past 6 sensors](https://arxiv.org/html/2603.04610v1)
- [FEEL Algorithm (PMC9886238) — 98.6% accuracy, 3 sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC9886238/)
- [Raspberry Shake RS4D architecture — geo + MEMS hybrid](https://shop.raspberryshake.org/product/turnkey-iot-home-earth-monitor-rs-4d/)
- [Enforcing Reciprocity in Operator Learning (arXiv:2602.11631)](https://arxiv.org/pdf/2602.11631)
- [Bridging the sim-to-real gap in RF localization — synthetic pretraining (ScienceDirect 2025)](https://www.sciencedirect.com/article/pii/S1566253525011662)
- [HVAC noise characteristics (ASHRAE Handbook Chapter 49)](https://handbook.ashrae.org/Handbooks/A23/IP/A23_Ch49/a23_ch49_ip.aspx)
- [Plumbing water hammer frequency: up to 170 Hz (Frontiers 2022)](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.956209/full)
- [ISO walking force model — Dynamic Load Factors for floor vibration design](https://www.researchgate.net/publication/363068431)
- [Footstep GRF on concrete: 25–250 Hz range (SPIE)](https://www.spiedigitallibrary.org/proceedings/Download?fullDOI=10.1117/12.663978)
- [Domain Adaptation SHM Review (arXiv:2512.18780)](https://arxiv.org/pdf/2512.18780)
- [Devito elastic tutorial (Virieux staggered grid)](https://www.devitoproject.org/examples/seismic/tutorials/06_elastic.html)
- [Elastodynamic reciprocity (Wapenaar PRL 93, 2004)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.93.254301)
- [Structural reverberation and MFP (PMC9886238, PMC11644851, PMC11053483)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644851/)
