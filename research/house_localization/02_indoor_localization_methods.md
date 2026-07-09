> **Research survey — house-localization pivot ("wallhacks").** Background research agent (Sonnet), 2026-07-08. Literature/tooling survey; design input, not a validated experiment. Extends `research/geophone_arrays.md` (indoor-specific). Indexed in [README.md](README.md).

---

# Indoor Footstep Localization via Geophone/Accelerometer Arrays: Methods Ranked by Achievable Indoor Accuracy

**Scope.** This report is strictly indoor — a person walking inside a concrete/brick house — and goes deeper than the companion outdoor/perimeter survey. The key physics change: instead of soft-soil Rayleigh waves at 1–100 Hz decaying over tens of meters, we have dispersive flexural (Lamb) waves in stiff floor slabs at 10–250 Hz propagating over 3–20 m. The binding constraint shifts from geometric spreading to *dispersion-induced waveform distortion* and *boundary reflections*, which every method must handle.

---

## 1. Methods Ranked by Achievable Indoor Localization Accuracy

### Summary Table (best-to-worst)

| Rank | Method | Reported Best Indoor Error | Sensor Count / Area | Key Assumption |
|------|--------|---------------------------|---------------------|----------------|
| 1 | Transfer-function / FEEL (modal matching) | **~7 cm** (ideal grid); 0.22 m (free walking) | 3–8 sensors / 10–20 m² | Known transfer functions; low sampling rate needed |
| 2 | TDOA with dispersion model (SO-TDOA, GCC-PHAT + dispersion mitig.) | **0.22–0.38 m** median | 4–8 sensors / 20–50 m² | Kelvin-Voigt or thin-plate velocity model calibrated |
| 3 | Matched-Field Processing (MFP / EDMF) | **~0.38 m** (simulation + real, thin-plate model) | 4–12 sensors / 20–100 m² | FE/wave-equation model of floor calibrated |
| 4 | Time-Reversal Focusing | **< 3% of aperture** on lab plates; ~0.49–0.89 m building-scale | 3–11 sensors / 20–200 m² | Known or estimated wave speed; benefits from reverberant multipath |
| 5 | Physical Reservoir Computing / data-driven MFP | **0.49–0.65 m** (same-subject); 1.13 m cross-subject | 11 sensors / 16 m hallway | Labeled training data per structure; linear readout |
| 6 | Energy/RSS Trilateration | **0.63 m** (Kalman-smoothed); 0.89–2.0 m raw | 4–8 sensors / 20–100 m² | Power-law decay known; heterogeneous floors degrade |
| 7 | Classic TDOA / Hyperbolic (constant velocity) | **0.5–0.61 m** on concrete with careful pick; typical 1–3 m | 4–8 sensors / 20–50 m² | Constant propagation velocity — violated in dispersive media |
| 8 | Steered Response Power (SRP/PHAT, beamforming) | **~2.0 m** (floor vibration); sub-meter acoustic | 4–8 sensors / 110 m² | Constant velocity; alias-free sensor spacing |
| 9 | Subspace DOA (MUSIC/ESPRIT) | **~5–15° bearing**; no direct range | Array dependent | Far-field, narrow-band — poorly suited for near-field impulse |
| 10 | Fingerprinting (database of vibration signatures) | **~0.38–0.50 m** | Heavy training; 4–12 sensors | Environment-specific; degrades with furniture changes |

### 1.1 Transfer-Function / Modal-Matching Methods (FEEL Algorithm) — Best Achievable

**Principle.** The floor's vibration response to an impact at location x is described by its transfer function (Green's function) H(sensor, x, ω). During calibration, H is measured at a grid of known locations. During operation, measured waveforms are compared to stored H functions and the best-matching location is returned. The FEEL (Force Estimation and Event Localization) algorithm (Posenato et al., *Sensors* 2023) uses a *correlation-based force estimation* (CFE) step: it first reconstructs the impact force at each candidate location using the stored transfer function, then ranks candidates by residual misfit.

**Why it works indoors.** Transfer functions implicitly encode all boundary reflections, modal structure, and frequency-dependent attenuation in the slab. No explicit velocity model is needed. This is the structural-vibration analog of MFP in ocean acoustics: the "matched field" is the measured impulse response at each candidate location rather than a propagation-model replica.

**Sensor count.** Minimum 2 sensors (the CFE correlation needs only 2); 3 demonstrated in a 3.05 × 4.38 m reinforced-concrete room.

**Reported accuracy.**
- On a 3 × 4.38 m concrete floor with 3 accelerometers at 18 calibration points × 5 impacts each: 92.1% exact-match accuracy on calibrated grid (footstep hits a known grid node), 98.6% on free walking paths (SDFE method), 1,100 footsteps evaluated. "Exact match" here means the correct grid cell out of 18, grid spacing ~50–80 cm — so cell-level accuracy is ~25–40 cm.
- Force estimation error: 1.3 ± 1.1%.
- GaitVibe+ (Mirshekari et al., arXiv 2212.03377) achieves **0.22 m average localization error** with 3 trials of multi-modal (vibration + camera) fusion, and the vibration-only subsystem reaches ~0.07 m under favorable conditions (carefully characterized wood floor).

**Limiting factors.** Requires one-time calibration walk or hammer survey at a grid of points (18–50 locations). Transfer functions change if furniture is rearranged or floor is wet. Works better on smaller, structurally simpler floors. Calibration effort scales as O(N_grid × N_sensors).

### 1.2 TDOA with Dispersion Modeling (SO-TDOA, GCC + thin-plate model)

**Principle.** Classic TDOA assumes constant propagation velocity c. In a concrete slab, the floor acts as a *thin elastic plate* governed by flexural (Euler-Bernoulli) wave mechanics. Flexural wave phase velocity is:

> v_ph(ω) = (Eh³ω² / 12ρ(1−ν²))^(1/4)

where E is Young's modulus, h slab thickness, ρ density. At 10 Hz on a 15 cm concrete slab, v_ph ≈ 300 m/s; at 100 Hz, v_ph ≈ 900 m/s — a 3× spread across the footstep band. If ignored, a broadband TDOA pick has systematic bias proportional to source–sensor distance (Bahroun et al., *J. Sound Vib.* 2014).

**SO-TDOA.** Bahroun, Michel et al. (arXiv 1211.3233; *J. Sound Vib.* 2014) introduce "perceived propagation velocity" which decreases with source–sensor distance under a Kelvin-Voigt damping model. SO-TDOA uses only the *sign* of the measured TDOA, making it robust to dispersive velocity uncertainty while still enabling hyperbolic triangulation. Experimental results on concrete: **"some tens of centimeters"** error — consistent with the 0.38 m (3.08× over baseline) figure reported in related EDMF work on the same slab setup.

**GCC-PHAT + narrow-band filtering.** Restricting cross-correlation to a narrow sub-band (e.g., 20–80 Hz where one velocity approximation holds) and using GCC-PHAT (phase-transform whitening, which sharpens the TDOA peak in reverberant media) reduces multipath bias. On concrete, researchers report:
- 0.34 m average over a 20 m² area (Mirshekari et al., 2018, cited in the NSF PAR paper).
- 0.61 m with full obstruction-invariant correction (adding time-reversal correction for path obstructions, 2021).
- 0.38 m median with EDMF (Error-Domain Model Falsification) variant.

**Sensor count.** 4–8 geophones at room corners is standard. A minimum of 4 gives redundant hyperbolas for a robust 2-D fix.

### 1.3 Matched-Field Processing (MFP / EDMF) — Structural-Vibration Analog

**Principle.** MFP was developed for underwater acoustics to localize sources in a waveguide whose transfer function is known from an ocean acoustic model. The direct translation to building floors was explored by Jacquelin et al. (*Mech. Syst. Signal Process.* 1996) using Bartlett (conventional beamformer), Minimum Variance (MVDR/Capon), and matched-modal processors in simulation on a metal plate. All three successfully located the source.

The **EDMF** (Error-Domain Model Falsification) variant, developed specifically for structural-vibration occupant localization (Smith, Drira et al., *MSSP* 2022; EPFL group), works as follows:
1. Build a physics model (FEM or thin-plate wave equation) of the floor.
2. For each candidate source location on a spatial grid, simulate the vibration response at all sensors.
3. Reject candidate locations whose simulated response contradicts measurements beyond measurement + model uncertainty.
4. The surviving candidate set is the localization estimate.

**Why it is uniquely powerful indoors.** MFP/EDMF explicitly uses the structural transfer function including reflections from walls, columns, and openings — exactly the reverberant multipath that defeats constant-velocity TDOA. In ocean acoustics, MFP outperforms matched-filter ranging by 10–15 dB in reverberant waveguides. The structural-vibration application inherits this advantage.

**Reported accuracy (structural vibration).**
- On a 20 cm concrete slab (tile-covered): 0.38 m median, 3.08× improvement over baseline constant-velocity TDOA.
- Simulation studies on aluminum plates: sub-centimeter in noise-free conditions; ~2–5% of aperture at realistic SNR.

**Calibration requirement.** The FEM or thin-plate model must be calibrated (measured E, h, ρ, boundary conditions). Once calibrated, no further per-location hammer survey is needed — this is the key advantage over the FEEL transfer-function approach.

**Data-driven MFP (model-free).** Emerging variant: no physics model; instead trains a neural network or builds an empirical replica field from a small number of known-location impacts, then matches measurement to replica field at inference time. Avoids FEM calibration at the cost of some labeled training data.

### 1.4 Time-Reversal Focusing

**Principle.** Record the vibration from a footstep at all sensors. Time-reverse each recording and mathematically "re-emit" it from the sensor positions. The wavefield refocuses at the original source location. In practice for impact localization: the time-reversed signals are cross-correlated, and the spatial maximum of the back-propagated field is the estimated source.

**Reverberant-environment advantage.** In a reverberant room, boundary reflections *help* time-reversal focusing — each reflection acts as a virtual sensor, effectively increasing the array aperture. This is the opposite of TDOA, where reflections create ghost peaks. Fink's group showed that time-reversal focusing sharpens with increasing reverberation.

**Reported accuracy.**
- *Laboratory plates* (Aluminum, composite, cylindrical shell): maximum error < 3% of aperture. For a 50 × 50 cm plate with 4 PZT sensors, this gives ~1.5 cm absolute error (PMC11359918).
- *Building-scale* (Mirshekari et al., obstruction-invariant paper, MSSP 2021): time-reversal correction applied to wavefield distortion from obstructions (walls, partitions) reduced localization error from ~0.91 m (obstructed baseline) to ~0.61 m. In unobstructed cases, TDOA alone gives ~0.34 m and time-reversal adds minor improvement.
- *Dispersion model (PMC11644851)*: parametric dispersion compensation reduced RMSE from 0.89 m to 0.49 m (Occupant A) and 0.94 m to 0.71 m (Occupant B) on Goodwin Hall concrete floor.

**Limitation indoors.** Building floors have hard reflections but also energy leakage through walls and inter-floor coupling. The virtual aperture from reverberation helps only if the wave equation is well-posed for the structure. Heterogeneous floors (concrete + steel beams + wood overlay) scatter unpredictably.

### 1.5 Physical Reservoir Computing

**Principle.** Treat the instrumented floor as a physical reservoir computer (Echo State Network in hardware). A footstep impulse injects energy into the floor's nonlinear dispersive dynamics. The multi-sensor waveform captures a high-dimensional projection of that dynamics that is location-dependent. A linear readout (ridge regression) trained on labeled footsteps maps this projection to (x, y) coordinates.

**Key finding** (Casaclang et al., arXiv 2603.04610, Virginia Tech Goodwin Hall, 2026): 11 piezoelectric accelerometers on a 16 m reinforced-concrete corridor.
- Single-participant: **0.65 m test error (longitudinal)**, 0.47 m (lateral).
- Cross-participant (no retraining): **1.13 m RMSE** with Kalman smoothing, 0.98 m longitudinal.
- Outperforms energy-based RSS by 33–38%.
- Competes with maximum-likelihood estimator (MLE: 0.89–0.94 m).

**Why it works.** The floor's dispersive dynamics project the source location into a high-dimensional feature space that a simple linear readout can decode. RMS normalization removes amplitude/weight variability between subjects. PCA compresses the reservoir state.

**Limitation.** Requires labeled training data per structure, per floor. Cross-participant generalization degrades unless normalization is strong. Does not generalize to unseen floor layouts.

### 1.6 Energy/RSS Trilateration

**Principle.** Footstep amplitude at sensor i decays with source–sensor distance r_i as A_i ∝ r_i^{-α} × exp(−β r_i), where α ≈ 0.5–1 (floor surface-wave geometric spreading) and β is the frequency-dependent attenuation. Given measured amplitudes at ≥3 sensors, solve for (x, y) by nonlinear least squares.

**Results.** Alajlouni & Tarazaga (*J. Vib. Control*, 2020): two-step energy method on smart building underfloor network. "Higher localization accuracy than existing energy-based approaches." With Kalman smoothing the localization track is sub-meter. Without Kalman: typical errors 0.63–1.5 m depending on floor homogeneity.

**SRP (Steered Response Power).** Royvaran & Donohue (U. Kentucky thesis): SRP applied to floor vibration, surface-wave velocity estimated from data, localized stationary source within **2.0 m** on a 13.4 × 8.4 m floor. The poor accuracy relative to TDOA methods reflects the limited frequency range used and coarse velocity model.

**Advantage.** Extremely simple, no timing synchronization needed. Usable at very low sampling rates (< 200 Hz).

**Limitation.** Attenuation varies with floor material, subfloor, humidity, shoe type. α and β must be calibrated per-floor. Strongly heterogeneous floors (concrete vs. wood subfloor sections) create systematic bias of 0.5–2 m.

### 1.7 Classic Hyperbolic TDOA (Constant Velocity)

On wood or homogeneous concrete with narrow-band processing, this achieves 0.34–0.61 m. On heterogeneous or complex floors (steel beams, openings), or when using broadband cross-correlation, systematic dispersion bias grows to 1–3 m. This is the baseline against which all other methods show improvement.

### 1.8 Subspace DOA (MUSIC/ESPRIT/Root-MUSIC)

These are designed for far-field narrow-band plane waves arriving at an antenna array. Indoor footstep vibration is a *near-field broadband impulse* — two violations of the subspace model's assumptions simultaneously. MUSIC applied to floor vibration gives bearing-only information (no range), with ~5–15° angular resolution depending on aperture. For localization, crossing bearings from two separated sub-arrays can give ~0.5–1.0 m accuracy at 5 m range. Not directly reported in structural-vibration literature with sub-meter claims.

Modified MUSIC for multiple simultaneous vibration sources (Springer *Physics of Wave Phenomena* 2024): successfully resolves two simultaneous vibration sources on a plate — directly relevant for multi-person scenarios.

---

## 2. Structural-Vibration Occupant-Localization Literature — Key Studies

### 2.1 Mirshekari, Pan, Zhang, Noh — Carnegie Mellon / Stanford Series (2016–2021)

The most cited group for indoor floor-vibration localization:

- **2016** (Mirshekari et al., *MSSP*): Indoor footstep localization from structural dynamics. Wood and concrete floors, TDOA with dispersion mitigation. Error ~0.34 m on 20 m² concrete area, 4–8 geophones.
- **2018** (Pan, Mirshekari et al., *MSSP*): Footstep-induced structural vibration for occupant localization. Concrete + wood + steel floors, 4–8 sensors. Mean error **0.34 m** (wood office), **0.61 m** (concrete), demonstrating floor-type dependence.
- **2021** (Mirshekari, Pan et al., *MSSP*): Obstruction-invariant occupant localization. Time-reversal characterization of obstructions. Maintained **0.61 m** despite furniture/wall obstructions (previously degraded to ~0.91 m).

**Sensor setup.** 4–8 geophone or accelerometer sensors placed at room corners and mid-walls. Rooms 20–60 m². Geophones preferred for 10–200 Hz; sampling at 4,000–8,192 Hz.

### 2.2 GaitVibe and GaitVibe+ (Mirshekari et al., CMU, arXiv 2212.03377)

**GaitVibe.** Floor-vibration system for in-home gait monitoring. 4–6 geophones, home environment, 10–200 Hz. Footstep localization as a component of gait parameter estimation. Reported localization error ~0.07 m in favorable conditions (well-characterized floor, stable conditions).

**GaitVibe+.** Adds temporary cameras during a brief calibration phase to characterize wave arrival time uncertainty and build wave velocity profiles per structural zone. After cameras removed, vibration-only localization achieves **0.22 m average** across 3 trials. Key innovation: Bayesian uncertainty propagation from arrival-time estimates to position posterior, with floor-zone-specific velocity calibration.

**Sensor cost.** < $50/sensor (geophone), 4–8 sensors per room.

### 2.3 FEEL Algorithm (Posenato et al., *Sensors* PMC9886238, 2023)

3 accelerometers in a 3.05 × 4.38 m reinforced-concrete room. 18-point calibration grid. 1,100 footsteps: **98.6% localization accuracy** (SDFE method). Force estimation: 1.3% error. Key: no timing synchronization between sensors; no velocity estimation needed. Lowest complexity among all high-accuracy methods.

**Limitation.** Room must be small enough that transfer functions are distinct at each grid point (otherwise ambiguity arises). Degrades in rooms > ~50 m² with sparse calibration grids.

### 2.4 Reservoir Computing Study (Casaclang et al., arXiv 2603.04610, 2026)

11 accelerometers, 16 m concrete corridor, 4th floor Goodwin Hall (Virginia Tech). Single-person: 0.65 m. Cross-person: 1.13 m with Kalman. Method: RMS normalization + PCA + ridge regression + optional Kalman filter. Training on 3–5 traversals per subject. Compares favorably to energy-based (38% improvement) and MLE (competes within noise).

### 2.5 SO-TDOA (Bahroun, Michel, Lacoume — Grenoble, J. Sound Vib. 2014)

Concrete slab localization with dispersive model. Sign-only TDOA avoids the need for an accurate absolute velocity. 4–6 sensors. Accuracy: **"tens of centimeters"** (< 0.3 m implied). Key physics: "perceived propagation velocity" decreases monotonically with source–sensor distance → SO-TDOA is monotonic → sign-only comparison is sufficient for hyperbola ordering → correct intersection.

### 2.6 Vibro-Localization with Dispersion Propagation Operators (PMC11644851, Sensors 2024)

Aluminum plate (50 × 50 cm, 4 sensors): from 19.19 cm → **18.01 cm** (1.97 cm σ) with dispersion model. Goodwin Hall concrete (11 sensors, 16 m²): 0.89 m → **0.49 m** (Occupant A), 0.94 m → **0.71 m** (Occupant B). Method: frequency-dependent phase delay operator τ(ω) = exp(−jωd/v(ω)) with parametric fitting of v(ω) to measured waveforms.

### 2.7 SRP on Floor Vibration (Royvaran & Donohue, U. Kentucky)

SRP-PHAT adapted for surface waves. First application of SRP to floor vibration. 13.4 × 8.4 m floor, accelerometers. Stationary source: **within 2.0 m**. Moving sources not evaluated. Primarily demonstrates feasibility, not best-possible accuracy.

### 2.8 Device-Free Multi-Person Localization (Shi et al., CMU)

Floor vibration for multiple simultaneous occupants. Continuous wavelet transform (CWT with Morlet wavelet) to separate overlapping footstep signatures, followed by per-person TDOA. Hallway (10 m × 2 m), 6 sensors: **0.63 m** accuracy. Two-person simultaneous localization demonstrated, with 10–20% degradation in single-sensor accuracy when both people walk at the same time.

---

## 3. What Drives Best Indoor Accuracy

### 3.1 Dispersion Handling Is the Dominant Factor

The single biggest difference between indoor structural vibration and outdoor soil is **flexural wave dispersion**. A concrete slab of thickness h obeys:

> v_ph(f) ≈ (π·f·h)^(1/2) · (E / 3ρ(1−ν²))^(1/4)

At 10–250 Hz on a 15 cm slab: v_ph sweeps from ~250 m/s to ~900 m/s. A broadband footstep impulse traveling 3 m is smeared over ~3–10 ms by dispersion alone — comparable to the timing uncertainty from a GCC-PHAT peak. Every method that fails to model this degrades from potential 0.1 m accuracy to 0.5–2 m.

**Methods that explicitly handle dispersion**: SO-TDOA, FEEL, GaitVibe+, dispersion operator (PMC11644851), time-reversal (implicitly). All achieve < 0.5 m.
**Methods that do not**: classic TDOA, energy-based, SRP. Errors 0.6–3 m.

### 3.2 Wave Mode and Frequency Band

In a building floor, two relevant modes:
- **Flexural (A0 Lamb mode)**: dominant at 10–250 Hz, excited by footstep vertical force, detected by vertical geophones. Highly dispersive. Most of the footstep energy.
- **Symmetric (S0 Lamb mode)**: ~kHz range, less dispersive, but excited weakly by footstep. Useful for plate damage detection, not for footstep localization.
- **Body waves (P/S through the slab thickness)**: negligible at footstep frequencies in thin concrete slabs.

**Best band selection.** Working in a narrow sub-band (e.g., 20–80 Hz) where v_ph changes by < 20% reduces dispersion bias to < 5 cm at 3 m source-sensor distance. GCC-PHAT in a narrow band outperforms broadband GCC by 2–5× in TDOA pick accuracy on concrete (Mirshekari 2016).

### 3.3 Sensor Count and Geometry

For 2-D localization in a room:
- **Minimum: 3 sensors** (2 TDOAs → 2 hyperbolae → 1 intersection; ambiguous without prior).
- **Practical: 4–6 sensors** → 3–5 TDOAs → overdetermined, robust to one bad pick.
- **Optimal placement**: spread sensors to maximize coverage and minimize geometric dilution of precision (GDOP). Corner-plus-midwall placement is robust for rectangular rooms.
- **Accuracy vs. count**: going from 4 to 8 sensors typically halves the RMSE by enabling better outlier rejection and overdetermined least-squares, not by raw SNR (√N gain is small at these short distances).

Room-scale (~5 × 5 m to ~15 × 15 m): 4–8 sensors suffice. Larger floors (>15 m longest dimension): consider zones with 4 sensors each.

### 3.4 Velocity / Transfer-Function Calibration

**This is mandatory** — not optional — for sub-50-cm accuracy on any method that requires a propagation model (TDOA, MFP, beamforming). Options:

- **Active-source calibration**: hammer survey at known grid points. Gives H(sensor, x, ω) or v(ω) per structural zone. Most accurate but requires ~30 min effort per floor zone.
- **Passive calibration from known walking paths**: a calibration walk on a known route (e.g., along a wall) provides enough arrival-time data to fit v(ω). GaitVibe+ uses this with camera assistance.
- **In-situ velocity from wavefield analysis**: if a compact sub-array (3–4 sensors, ~1–2 m spacing) is included, the inter-sensor travel time directly measures the local wave velocity. Feed this into TDOA velocity model. No hammer needed, but only gives velocity near the sub-array.

### 3.5 MFP and Time-Reversal Advantages in a Reverberant House

A concrete/brick house has high acoustic impedance at walls, producing strong reflections of structural waves. This creates two competing effects:

| Effect | Impact on TDOA | Impact on MFP / Time-Reversal |
|--------|---------------|-------------------------------|
| Wall reflections → ghost TDOA peaks | **Harmful** — false peaks in GCC | **Helpful** — virtual sensors, richer replica field |
| Floor heterogeneity → velocity scatter | **Harmful** — biased picks | **Harmful** — model mismatch, but MFP is more robust |
| Modal resonances | **Harmful** — ring-up after impulse | **Helpful** in time-reversal — energy focuses back at source |
| Obstructions (partitions, furniture) | **Harmful** — path blockage changes effective v | Mitigable with time-reversal characterization (Mirshekari 2021) |

**MFP advantage quantified**: in ocean acoustics, MFP outperforms matched-filter ranging by 6–15 dB in reverberant waveguides with known environment. The analogous structural-vibration gain is the ~3× RMSE improvement (0.38 m vs. ~1.2 m baseline) seen with EDMF.

**Time-reversal advantage quantified**: reduces obstruction-induced error from 91% increase to ~0% (Mirshekari 2021). On plates, sub-3% aperture error even with complex geometry (PMC11359918).

The combination — MFP with time-reversal replica fields — has not been systematically evaluated for indoor footstep localization at full building scale and represents the strongest candidate for pushing accuracy toward 10–20 cm range.

---

## 4. Theoretical Limit: CRLB for Indoor TDOA

### 4.1 CRLB for a Single TDOA Estimate

For two sensors receiving a signal with effective bandwidth β_eff (RMS bandwidth, in rad/s) and pre-detection SNR ρ, the Cramér-Rao lower bound on time-delay estimation variance is (Carter 1987, Knapp & Carter 1976):

> σ²_TDOA ≥ 1 / (8π² · β²_eff · T · ρ)

where T is the observation window length.

For indoor footsteps in the 20–80 Hz band (β_eff ≈ 2π × 30 Hz = 188 rad/s, T = 0.1 s, ρ = signal power / noise power):

| SNR (dB) | σ_TDOA (ms) | σ_TDOA (samples at 4 kHz) |
|----------|------------|--------------------------|
| 10 | 0.28 ms | 1.1 |
| 20 | 0.089 ms | 0.36 |
| 30 | 0.028 ms | 0.11 |

### 4.2 From TDOA Error to Position Error

With wave propagation velocity v (e.g., 500 m/s for concrete at 40 Hz), a timing error σ_TDOA converts to a ranging error:

> σ_range = v · σ_TDOA

And position error depends on GDOP (Geometric Dilution of Precision):

> σ_position ≈ GDOP × σ_range

GDOP ≈ 1.2–2.0 for well-placed 4-sensor rectangular room geometry (sensor spacing ≈ room width).

At 20 dB SNR with v = 500 m/s, GDOP = 1.5:
> σ_range = 500 × 0.089 × 10⁻³ ≈ **4.5 cm** → σ_position ≈ **6–9 cm**

This is the theoretical CRLB-limited position accuracy if the propagation velocity were perfectly known and noise-free multipath.

### 4.3 The Velocity Uncertainty Floor

The CRLB analysis above applies to the *timing noise* only. In reality, velocity uncertainty Δv/v dominates for most indoor scenarios. A TDOA pick at distance d = 3 m with timing error σ_τ and velocity uncertainty Δv:

> σ_effective = √( (v·σ_τ)² + (d·Δv/v)² )

With v = 500 m/s, d = 3 m, Δv/v = 5% (good calibration), Δv contribution = 3 × 0.05 = **15 cm**. Even at 30 dB SNR (σ_τ = 0.028 ms, v·σ_τ = 1.4 cm), velocity uncertainty dominates.

To reach the CRLB-limited 6–9 cm position accuracy, velocity must be known to better than 3%: achievable with hammer calibration (< 2% uncertainty in practice) but not with assumed/modeled velocity.

**Practical bottom line**:
- With in-situ calibration (Δv/v < 2–3%): position CRLB ≈ **5–15 cm** achievable at SNR > 20 dB.
- With modeled-only velocity (Δv/v ~10–20%): CRLB floor rises to **30–60 cm**, which explains the 0.34–0.61 m results observed experimentally even with careful TDOA implementations.
- GaitVibe+'s 0.22 m result and FEEL's ~7 cm grid-level accuracy sit just above the calibration-limited floor.

### 4.4 Bandwidth and Mode Effects

The effective bandwidth β_eff can be increased by:
1. **Higher sampling rate + higher-frequency modes**: using the 100–250 Hz flexural wave content (requires 2 kHz+ sampling, which geophones support). β_eff doubles → CRLB halves in timing variance.
2. **Wideband MFP / cross-frequency replica matching**: incoherent averaging across frequencies reduces variance roughly as 1/N_freq.
3. **Using body-wave arrivals** (if any): P-wave through the slab (~3000–4000 m/s concrete) is impulsive, non-dispersive, and detectable at short distances — gives a sharper TDOA pick but weaker amplitude.

Practically, 10–250 Hz gives β_eff ≈ 2π × 70 Hz ≈ 440 rad/s — 2.3× wider than the 20–80 Hz case, reducing CRLB timing variance by 5.4×. This brings the timing-noise-limited floor to ~3–4 cm, still dominated by velocity uncertainty.

---

## 5. Fusion with an Existing Classifier; Multi-Target

### 5.1 Fusion Architecture for an Existing Human/Animal/Nothing Classifier

Your current classifier operates on 3 s windows and outputs a class label. The integration with localization creates a two-stage pipeline:

**Stage 1 — Gating.** The classifier output gates localization processing:
- Class = "nothing": skip TDOA/MFP entirely, save compute.
- Class = "animal": run localization (bearing estimate useful even for animal) but flag as low-priority.
- Class = "human": run full TDOA + tracking pipeline, emit position estimate.

**Stage 2 — Hypothesis weighting.** Within MFP or particle filter, the classifier posterior P(human | features) acts as a spatial likelihood prior. If the classifier says "human with 95% confidence," MFP search range can be restricted to the floor interior (humans walk on the floor, not float). This reduces false peaks from boundary reflections.

**Stage 3 — Classifier improvement from localization.** Localization provides a track: velocity vector, cadence, step length. These are class-discriminating features (human ~1.2 m/s, 1–2 Hz; quadruped gait different). Feed track features back into the classifier as additional inputs. Multi-frame integration (if person is present and moving, cadence is 1–2 Hz with regular ≤80 cm step) eliminates most false positives from animals and mechanical vibration.

**Reported fusion benefit.** Audio-seismic fusion for outdoor human localization: F1 score 92% (seismic alone ~70%), 37.9% reduction in position error (seismic regression + audio DOA fusion). For indoor floor-vibration, no published classifier-localization fusion paper exists for your exact multi-class setup, but the architecture above is a direct extension of the existing literature.

### 5.2 Multi-Target (Several People Walking Simultaneously)

**The hard part.** When two people's footsteps overlap in time at a sensor, the GCC-PHAT picks up a mixture — the cross-correlation has two peaks (or a blended smear if steps are nearly synchronous). Standard TDOA gives one location estimate, not two.

**Methods that handle multi-target:**

1. **Spatial filtering / Modified MUSIC** (Springer 2024): Eigendecomposition of the cross-spectral matrix separates contributions from two simultaneous sources when they are angularly separated. On a plate, two vibration sources were successfully resolved. Requires SNR > ~15 dB per source and angular separation > ~20° from the array center.

2. **Multi-Hypothesis TDOA** (Shi et al., CMU): CWT to separate footstep impulses by time-frequency, then assign GCC peaks to tracks. On a 10 × 2 m hallway with 6 sensors: 0.63 m accuracy for each of two simultaneous occupants.

3. **Particle filter with multi-target data association** (MHT / JPDA): Maintain multiple hypotheses, each consistent with the observed TDOA peaks at each time step. Cadence prior (human steps ~1–2 Hz) constrains tracks. Can separate 2–3 simultaneous people if SNR > 10 dB per target.

4. **FEEL for multi-target**: FEEL's CFE step can be run independently for each detected footstep impulse (if impulses are temporally separated by > ~100 ms). Two people walking on different floor tiles trigger separate, temporally distinct impulses 80% of the time (normal walking pace has ~400–600 ms intervals). Simultaneous strikes (< 50 ms apart) will merge into a single estimated source between the two.

**Honest limits for multi-target.** Two people standing still on the same floor: no vibration, no localization. Two people walking at the same step frequency (synchronization): TDOA peaks merge. At normal walking pace on a quiet floor, 90%+ of footsteps are temporally isolable with a 50 ms separation criterion. Multi-person localization degrades gracefully with overlap, not catastrophically.

---

## 6. Best-Accuracy Recipe

For a concrete/brick house with N geophones aiming for best possible 2-D localization:

**Sensors.** 6–8 vertical geophones (4.5 Hz natural frequency works well for 10–250 Hz floor vibrations — above the natural frequency, sensitivity is flat). Place at room corners + mid-walls. Optional: one 3-C geophone at the array center for polarization cross-check. Sampling: 4,000 Hz minimum; 10,000 Hz preferred to capture 250 Hz flexural content.

**Signal chain.**
1. STA/LTA event detection (STA window 5 ms, LTA 200 ms) to isolate individual footstep impulses.
2. Band-split: 10–80 Hz (primary TDOA band), 80–250 Hz (fine-timing, if SNR permits).
3. Active hammer calibration: strike floor at ~20 known locations, record impulse responses H(sensor, location, t). Estimate v(f) per zone and full transfer-function database.

**Localization algorithm (ranked by accuracy).**
- If calibration grid is available + room < 50 m²: **FEEL / transfer-function matching** → 10–25 cm.
- If FEM model of floor available: **MFP/EDMF** → 20–40 cm, generalizes beyond calibration grid.
- Otherwise: **SO-TDOA or GCC-PHAT (narrow-band) + Kelvin-Voigt dispersion model** → 20–40 cm.
- Track smoothing: **Kalman filter** with constant-velocity pedestrian motion model reduces instantaneous error by 15–25%.

**Fusion with classifier.** Gate localization processing on "human" classification output. Feed track velocity, cadence, step-length back into classifier for iterative refinement. Implement multi-hypothesis tracking for the two-person case.

**Expected accuracy.** Sub-30 cm (single person, calibrated floor) to sub-60 cm (uncalibrated, two simultaneous people). Approaching the 5–15 cm CRLB-limited floor requires full calibration AND high SNR (footstep 1–2 m from nearest sensor).

---

## 7. Where Reverberation / Multipath Is the Binding Limit

(Detailed wave physics covered by the separate reverberation report — [05_structural_reverberation.md](05_structural_reverberation.md).)

In TDOA methods: multipath creates spurious GCC peaks at delays τ_direct + τ_reflection. On a 5 × 5 m concrete floor, first wall reflection arrives ~7 ms after direct wave (round-trip ~3.5 m / 500 m/s). With a 20–80 Hz band and effective temporal resolution ~1/(2 × 60 Hz) ≈ 8 ms, the direct and first reflection are *barely resolvable*. GCC-PHAT whitening helps but does not eliminate the reflection peak. This is where TDOA is binding.

In MFP and time-reversal: multipath is *the signal*, not the noise — these methods explicitly model or exploit reflections to improve localization. They are therefore not binding-limited by multipath in the same sense.

The reverberation limit for TDOA manifests as: effective TDOA accuracy ≈ max(σ_noise, Δτ_reverberation), where Δτ_reverberation ≈ room size / (2 × v). For a 10 m room, v = 600 m/s: Δτ_reverb ≈ 8 ms → position error ≈ 600 × 0.008 = **4.8 m** for unresolved multipath. Dispersion and frequency-domain windowing reduce this to 0.3–0.6 m when properly managed — exactly the range of experimentally reported accuracies.

---

## Sources

- SO-TDOA, Kelvin-Voigt, perceived velocity: [Bahroun et al. arXiv 1211.3233](https://arxiv.org/abs/1211.3233); [J. Sound Vib. ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022460X13008316)
- FEEL Algorithm (PMC9886238): [Footstep localization and force estimation, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9886238/)
- GaitVibe+ (arXiv 2212.03377): [GaitVibe+: Enhancing structural vibration-based footstep localization](https://arxiv.org/abs/2212.03377)
- Obstruction-invariant localization (MSSP 2021): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327020308852)
- Occupant localization MSSP 2018: [NSF PAR](https://par.nsf.gov/servlets/purl/10057760); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327018302280)
- Indoor footstep localization MSSP 2016: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327016305015)
- Reservoir computing footstep localization (arXiv 2603.04610): [Can a Building Work as a Reservoir](https://arxiv.org/html/2603.04610v1)
- Dispersion propagation operators (PMC11644851): [Modeling and Analysis of Dispersive Propagation, Sensors 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644851/)
- Energy-based + Kalman (Alajlouni & Tarazaga 2020): [SAGE Journals](https://journals.sagepub.com/doi/abs/10.1177/1077546319890520)
- SRP on floor vibration: [Localization of Stationary Source of Floor Vibration, Semantic Scholar](https://www.semanticscholar.org/paper/Localization-of-Stationary-Source-of-Floor-Using-Royvaran-Donohue/254d9e4533e45fdcb73fb2775410176a9afc4b77)
- Time-reversal impact localization, PMC11359918: [Impact Localization Cylindrical Shell, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11359918/)
- MFP for structural vibration: [ResearchGate — Applications of MFP to structural vibration](https://www.researchgate.net/publication/243520672_Applications_of_matched-field_processing_to_structural_vibration_problems)
- Device-free multi-person floor vibration (CMU): [Shi et al., CMU ECE](https://users.ece.cmu.edu/~yuejiec/papers/localization_via_vibration.pdf)
- EDMF / MSSP 2022: [EPFL MSSP paper](https://www.epfl.ch/schools/enac/honorary-professors/wp-content/uploads/2022/01/Drira-Smith-OA-2022-Framework-MSSP.pdf)
- Automated deep-learning footstep localization (ASCE): [ASCE JCCEE](https://ascelibrary.org/doi/10.1061/JCCEE5.CPENG-6869)
- Modified MUSIC for multiple vibration sources: [Springer Physics of Wave Phenomena 2024](https://link.springer.com/article/10.3103/S1541308X24010059)
- TDOA CRLB general framework: [MDPI Sensors — Accuracy Analysis Asynchronous Positioning](https://mdpi.com/1424-8220/19/13/3024/htm)
- VTECHWORKS thesis supplementing indoor footstep algorithms: [Virginia Tech](https://vtechworks.lib.vt.edu/server/api/core/bitstreams/9d276b5f-1e57-415d-ab24-7c8eb519872e/content)
- Ubiquitous gait analysis via floor vibrations (PMC11053483): [MDPI Sensors / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11053483/)
- ACM Survey seismic sensor target detection/localization: [ACM CSUR](https://dl.acm.org/doi/10.1145/3568671)
- Audio-seismic fusion outdoor localization: [ResearchGate](https://www.researchgate.net/publication/358877651_A_Fingerprinting_based_Audio-Seismic_Systems_for_Human_Target_Localization_in_an_Outdoor_Environment_using_Regression)
