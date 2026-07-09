> **Research document — 1-D pyprop8 localization testbed.** Research phase only; no implementation. Written 2026-07-08. Covers method bake-off, experimental protocol, and blind-velocity rigor. Indexed in `localization_1d/`.

---

# Localization-Method Bake-Off and Experimental Protocol for the 1-D pyprop8 Testbed

## Purpose

This document designs the method competition and experimental protocol for the cheap 1-D testbed: flat layered soil, a vertical-force surface source, N surface receivers, waveforms synthesized by pyprop8. The goal is to validate localization methods on a fully controlled, analytically tractable problem before committing to the expensive 3-D Devito house simulation.

The testbed is translation-invariant: in 1-D layered media, propagation depends only on source–receiver distance r, not on absolute position. This means every "forward model" collapses to a function of r alone, which dramatically simplifies both MFP replica generation and the fingerprint dictionary. Rayleigh waves dominate at surface-to-surface range; they are dispersive (phase velocity c(f) depends on frequency and on the layer structure), and that dispersion is the central difficulty that all methods must handle.

---

## 1. Testbed Physics and pyprop8 Capabilities

### 1.1 pyprop8 as the synthetic data engine

pyprop8 (Valentine & Woodhouse 2022, JOSS) implements the Thompson–Haskell propagator matrix for a 1-D layered elastic half-space. Key facts:

- **Source types.** Moment tensor (earthquake radiation) or body-force vector (vertical point force — the correct model for a surface footstep or hammer blow). Both are supported.
- **Receiver positions.** Surface receivers at arbitrary offsets (depth = 0, x = r_i). Surface-to-surface propagation is fully supported.
- **Outputs.** `compute_seismograms()` returns three-component time-domain displacement at each receiver; `compute_spectra()` returns the complex frequency-domain wavefield. Both include all body-wave and surface-wave contributions automatically.
- **Sensitivity kernels.** `DerivativeSwitches` gives ∂wavefield/∂source_parameters — directly usable for gradient-based localization refinement.
- **Runtime.** Pure Python (NumPy/SciPy), no compiled code, runs on a single CPU core. A single forward simulation for a 5-layer model, 8 receivers, 4096 time samples runs in seconds — fast enough for brute-force grid searches.

### 1.2 Rayleigh wave dispersion in layered soil

For a flat-layered soil half-space, the Rayleigh-wave phase velocity c_R(f) is determined by the layer structure (thicknesses h_i, S-wave velocities v_S_i, P-wave velocities v_P_i, densities ρ_i). Key properties:

- At low frequencies (long wavelength), c_R(f) approaches the half-space S-wave velocity.
- At high frequencies (short wavelength), c_R(f) approaches the top-layer S-wave velocity.
- Between these limits, c_R(f) varies by 2–5× across the footstep band (1–100 Hz), depending on the velocity contrast between layers.

The group velocity U(f) = d(f·c_R)/df differs from the phase velocity, causing waveform dispersion: a broadband impulse spreads in time as it propagates, with high-frequency energy arriving before low-frequency energy (normal dispersion). This is the soil analog of the indoor floor problem documented in `02_indoor_localization_methods.md`.

The translation invariance means: if we know c_R(f), the travel time from source to receiver at distance r is τ(f, r) = r / c_R(f). This is the key replica formula for all model-based methods.

**Practical velocity range.** For soft alluvial soil: c_R(10 Hz) ≈ 100–200 m/s, c_R(50 Hz) ≈ 150–250 m/s (20–50% spread). For stiff soil: c_R(10 Hz) ≈ 200–350 m/s, c_R(50 Hz) ≈ 300–500 m/s. A "constant velocity" assumption introduces a systematic ranging bias of roughly (Δc/c) × r per mode, which grows with distance.

---

## 2. The Method Set

The following six methods are evaluated on this testbed. For each: principle, how it maps to the 1-D layered soil case, expected accuracy from the literature, failure modes, and data requirements.

### 2.1 Matched-Field Processing (MFP) on the Distance Dictionary

**Principle.** MFP was developed for underwater acoustic waveguide source localization (Baggeroer, Kuperman & Mikhalevsky 1993, IEEE J. Ocean. Eng.). The key idea: precompute a "replica" of the expected wavefield at all sensors for each candidate source location, then find the candidate whose replica best matches the observed data. In a translation-invariant 1-D medium, the replica field at receiver i for a source at distance r_s is completely determined by the Green's function G(r_i − r_s, t) or G(|r_s − x_i|, t), which depends only on the source–receiver offset d_i = |r_s − x_i|.

**How it works on this testbed.** Build a 1-D grid of candidate source positions x_s ∈ {x_1, x_2, ..., x_M}. For each x_s, call `pyprop8.compute_spectra()` to obtain the complex replica wavefield p̂(ω; x_s) at all N receivers. The Bartlett (conventional) MFP power is:

> P_Bartlett(x_s) = |w†(x_s) · C · w(x_s)|

where C is the cross-spectral density matrix (CSDM) of the observed data, w(x_s) is the replica vector (unit-norm steering vector from pyprop8), and † is conjugate-transpose. The estimated source position is argmax_x_s P_Bartlett(x_s).

Higher-resolution variants include Minimum Variance (MVDR/Capon): P_MVDR = 1/(w† C⁻¹ w), which narrows the peak but requires good CSDM estimation (many snapshots). In the testbed a single-shot event gives one snapshot — standard MVDR is unstable; use diagonal loading (MVDR-DL) or coherent (matched-phase) processing.

**Translation invariance advantage.** Because the medium is 1-D, the replica at receiver x_i for source x_s depends only on d_i = |x_s − x_i|. The full replica dictionary can be precomputed as a 1-D table G(d, ω) indexed by distance d. This is O(M × N_freq) precomputation, with M candidate distances and N_freq frequency bins. Storage is trivial. At inference time, given N observed waveforms, compute d̂_i = argmax over d of cross-correlation between G(d, ω) and observed spectrum at receiver i, then solve for x_s.

**Expected accuracy.** In ocean acoustics with a known environmental model: MFP routinely achieves range estimation within 1–5% of true range and depth within one wavelength (Clay & Medwin 1977; Baggeroer et al. 1993). For near-surface seismic: a dense array MFP study (Seydoux et al., GJI 2018) localized shallow surface sources within ±20 m on a 600×600 m aperture array of 1108 geophones — about 3% of aperture. For a 10–20 m source–receiver range testbed, this implies 0.3–0.6 m accuracy with a well-known model. With perfectly known velocity model (calibrated pyprop8 layers), accuracy approaches the timing CRLB: roughly 5–20 cm depending on SNR and bandwidth.

**Failure modes.**
1. Model mismatch: if the true layered model differs from the model used for replicas, the replica peak shifts. A 10% error in the top-layer S-wave velocity translates to roughly a 10% range bias.
2. Spatial aliasing: if receiver spacing exceeds half the dominant Rayleigh wavelength (λ/2 at 50 Hz, v=200 m/s → λ/2 = 2 m), grating lobes appear in the MFP power surface.
3. Coherent multipath / boundary reflections: if the array has finite length and reflections from boundaries are present, spurious peaks appear (less of a problem in a field testbed than in a reverberant room).
4. Low SNR: CSDM estimation degrades below SNR ~0 dB; MFP power surface flattens.

**Data requirements.** Velocity model (all layer parameters) OR calibrated pyprop8 replica dictionary. 3-component not needed — vertical only captures Rayleigh fundamentals. Absolute amplitude calibration not needed (only phase/coherence used for Bartlett). Multiple independent snapshots improve CSDM estimate but are not mandatory for single-event Bartlett.

**References.** Baggeroer, Kuperman & Mikhalevsky, IEEE J. Ocean. Eng. 1993; Seydoux et al., GJI 2018 (doi:10.1093/gji/ggy139); Seme et al., Seismica 2024 (Qseek/MFP for shallow sources).

---

### 2.2 SO-TDOA / GCC-PHAT + Dispersion Model

**Principle.** Classic TDOA: measure the time difference τ_ij between arrivals at receivers i and j, then solve the set of hyperbolic equations (each TDOA defines a hyperbola in the source-plane on which the source must lie). For N receivers, there are N(N−1)/2 independent TDOAs, of which N−1 are non-redundant; an overdetermined system for N ≥ 4 gives a robust least-squares solution.

**The dispersion problem.** In a dispersive medium, a broadband pulse has no single arrival time — different frequency components arrive at different times. The "TDOA" measured by GCC (generalized cross-correlation) is a frequency-weighted average group arrival time, not the phase travel time. If the GCC weighting (e.g., PHAT whitening) is uniform across frequency, the estimated τ is biased toward the group velocity of the dominant frequency band. This bias depends on source–receiver distance and the dispersion curve: Δτ_bias = r × (1/U_dominant − 1/U_true_weighted), where U is group velocity. For soft soil over 10 m, this can be 10–50 ms for broadband GCC.

**SO-TDOA (Sign-Only TDOA, Bahroun et al., arXiv 1211.3233; J. Sound Vib. 2014).** Rather than using the absolute TDOA, SO-TDOA uses only its sign: sgn(τ_ij). The insight is that in the dispersive-damped medium, the "perceived propagation velocity" (the apparent velocity inferred from broadband TDOA) decreases monotonically with distance. This means the sign of τ_ij is preserved even when the magnitude is biased. SO-TDOA solves for source position using sign-consistent hyperbola ordering, achieving "tens of centimeters" accuracy on concrete slabs without needing to estimate the absolute velocity magnitude.

**GCC-PHAT + narrow-band dispersion correction.** A more quantitative approach: apply GCC-PHAT within a narrow frequency band [f_low, f_high] where c_R(f) varies by < 10%, then use the band-center phase velocity c_R(f_center) as the TDOA velocity. This converts a dispersive broadband problem into a series of nearly non-dispersive narrow-band sub-problems. The TDOA at band center is unbiased to first order in Δc/Δf × bandwidth. For soil at 10–50 Hz (roughly constant c_R if band is < 20 Hz wide), this approach achieves sub-meter accuracy without a full dispersion model.

**Full dispersion model correction.** If the dispersion curve c_R(f) is known (from calibration or from pyprop8), the frequency-domain phase delay at frequency f is φ(f, r) = 2πf × r / c_R(f). Correcting the observed cross-spectrum by this known phase before cross-correlation collapses the dispersed arrival back to a sharp, undispersed peak. This is the "dispersion propagation operator" approach (Casaclang et al., Sensors 2024, PMC11644851), which reduced RMSE from 0.89 m to 0.49 m on a concrete corridor.

**TDOA → position inversion.** Given τ_ij for all pairs, solve by nonlinear least squares (e.g., Levenberg–Marquardt) or by a hyperbolic linearization (Chan & Ho 1994, IEEE Trans. Signal Process.). For the 1-D surface problem, the geometry is particularly well-conditioned when receivers bracket the source.

**Expected accuracy.** With known velocity model and narrow-band GCC-PHAT: 0.1–0.4 m at source–receiver ranges of 5–20 m (Mirshekari et al. 2016 reports 0.34 m on 20 m² concrete; outdoor soil with 80 m/s Rayleigh waves and similar timing uncertainty scales to similar fractional accuracy). The CRLB sets the floor: at 20 dB SNR with 20-Hz effective bandwidth and v = 150 m/s, σ_range = v × σ_TDOA = 150 × 0.09 ms ≈ 1.4 cm per sensor pair; GDOP × σ_range ≈ 2–6 cm for good geometry. Velocity uncertainty Δv/v = 5% adds a ranging bias of d × 0.05 = 0.5 m at d = 10 m — dominating the timing noise. Hence calibration quality is the binding limit.

**Failure modes.**
1. Constant-velocity assumption: for broadband GCC without a dispersion model, bias grows as ~(Δc/c_mean) × r. For 30% velocity spread over the band and r = 10 m, bias = 3 m.
2. Multipath: in a layered medium, body-wave reflections from layer boundaries arrive after the direct Rayleigh wave and may produce secondary GCC peaks, confusing the onset pick.
3. Poor geometry (all receivers on one side): GDOP degrades to >5 on a linear array if the source is beyond the array end.
4. Low frequency + short array: at 10 Hz, λ = 15 m. If the array aperture is smaller than λ, the cross-correlation peak is broad and TDOA precision degrades.

**Data requirements.** Timing synchronization across receivers (< 0.1 ms for 0.01 m accuracy at v=100 m/s). Velocity model or calibration for dispersion correction. Vertical component only (Rayleigh fundamental). Single-event onset sufficient.

**References.** Bahroun et al. arXiv 1211.3233; GCC-PHAT indoor accuracy: Mirshekari et al. MSSP 2016; dispersion operator: PMC11644851 (Sensors 2024); outdoor Rayleigh TDOA: PMC12473310 (Sensors 2026, 0.095 m with AIC picker and ITWLS).

---

### 2.3 Energy / RSS Trilateration

**Principle.** In a homogeneous or weakly attenuating medium, the amplitude A at distance r from a surface source follows:

> A(r) = A_0 × r^{−α} × exp(−β r)

where α = 0.5 for cylindrical (2-D) Rayleigh wave spreading and β = π f / (Q × c_R(f)) is the anelastic attenuation (Q is the seismic quality factor, typically 20–100 for near-surface soil). Given measured amplitudes A_i at N ≥ 3 receivers at known positions x_i, solve for source position x_s by nonlinear least squares on the over-determined system:

> ln(A_i) = ln(A_0) − α ln(r_i) − β r_i

The unknowns are x_s (which determines r_i) and A_0. If β is known (or estimated from the array data itself), the system is identifiable with N ≥ 3 receivers. Kalman filtering over successive events smooths the track.

**How it works on this testbed.** pyprop8 gives absolute displacement amplitudes, so the ratio A_i / A_j between two receivers directly measures the amplitude decay with distance. In the 1-D model, the amplitude ratio depends only on r_i and r_j (through geometric spreading and attenuation), not on source moment. This makes RSS trilateration a clean test: the model is A(r) = G(r, f) integrated over frequency, where G is the pyprop8 Green's function evaluated at the correct offset.

**Expected accuracy.** Reported values in structural vibration literature: 0.63 m with Kalman smoothing (Alajlouni & Tarazaga, J. Vib. Control 2020); 0.89–2.0 m raw (Casaclang et al. 2026). For outdoor soil, the energy method is simpler to calibrate (α, β can be estimated from the array data itself) but less accurate than TDOA because amplitude is more sensitive to local soil heterogeneity than travel time. Realistic expectation for the testbed: 0.5–2 m depending on heterogeneity of the simulated model and SNR.

**Advantage.** Does not require timing synchronization. Works at very low sampling rates (< 100 Hz). Provides a useful cross-check on TDOA methods. Simple to implement. Only requires scalar (peak, RMS, or envelope-integral) amplitudes — not waveform onset times.

**Failure modes.**
1. Heterogeneous attenuation: if β varies across the array (soft patch vs. stiff patch), the assumed decay model is wrong. A 2× error in β at r = 10 m introduces a ~7 m bias.
2. Near-field saturation: at very short source–receiver distances (r < 1 wavelength), the far-field amplitude scaling A ∝ r^{−0.5} breaks down.
3. Amplitude calibration variation: if sensors have different coupling or sensitivity, apparent amplitude ratios are wrong. The array must be cross-calibrated.
4. Source anisotropy: if the source is not perfectly vertical-force (e.g., has a moment tensor component), the radiation pattern is angle-dependent and breaks the isotropic amplitude assumption.

**Data requirements.** Amplitude calibration (sensor sensitivities must match). Q and geometric spreading exponent (need either calibration shots at known distances or assumed model). Vertical component only. No onset picking required.

**References.** Alajlouni & Tarazaga, J. Vib. Control 2020; Casaclang et al. arXiv 2603.04610 (2026); PMC12473310 (Sensors 2026, energy-based trilateration with 4 MEMS sensors, 3.2 m spacing, Rayleigh at 150 Hz, accuracy ~0.5 m before refinement).

---

### 2.4 3-Component Polarization Bearing

**Principle.** A Rayleigh wave at the surface has retrograde elliptical particle motion in the vertical (Z) and radial (R) plane, with no transverse (T) component in a 1-D isotropic medium. A 3-component geophone records Z, R, T at a single station. The direction of propagation (azimuth from receiver to source) is determined by the horizontal radial direction of the Rayleigh ellipse, which is given by the horizontal components (H_x, H_y).

The standard method is covariance matrix eigendecomposition (polarization analysis):

1. Form the 3×3 covariance matrix C = <u u^T> over a short window around the arrival.
2. Compute eigenvectors. The principal eigenvector gives the polarization direction (Z-R ellipse major axis direction in 3-D space).
3. The horizontal projection of this eigenvector gives the azimuth from receiver to source (180° ambiguity resolved by particle-motion sense: Rayleigh waves are retrograde).

For Love waves (SH surface waves), particle motion is transverse and horizontal — the polarization plane is horizontal, and azimuth is perpendicular to the T-component direction.

**How it works on this testbed.** pyprop8 outputs 3-component (radial, transverse, vertical) seismograms at each receiver. For a vertical-force source in a 1-D isotropic medium, transverse motion is identically zero; only R and Z are excited. The Rayleigh ellipticity ratio e(f) = A_R / A_Z (radial to vertical amplitude ratio) is frequency-dependent and encodes the layer structure. From a 3-C receiver at position x_i, the azimuth toward the source can be read directly from the radial direction. With two or more 3-C receivers at different positions, crossing azimuths gives the 2-D source location (triangulation by bearing). Minimum 2 receivers (2 bearings); 3 give overdetermined bearing intersection for outlier robustness.

**Expected accuracy.** In seismology, single-station polarization analysis achieves ~5–15° bearing accuracy for teleseismic events with good SNR (Jurkevics 1988, BSSA). For near-field shallow sources on a local geophone array, bearing accuracy is typically 5–10° at SNR > 10 dB. At 10 m source–receiver range with 5° bearing error, position error = 10 × tan(5°) ≈ 0.9 m. At 2° error, it is 0.35 m. Cross-bearings from two receivers improve precision and resolve the 180° ambiguity.

The key advantage: bearing is independent of distance, so it is unaffected by amplitude calibration or attenuation model errors. It is also less sensitive to dispersion than onset times, because the polarization direction is determined by the dominant frequency component (wherever energy is concentrated, the Rayleigh ellipse azimuth still points at the source).

**Failure modes.**
1. 1-D isotropic medium only: in an anisotropic or 3-D heterogeneous medium, the Rayleigh ellipse tilts out of the Z-R plane, introducing bearing errors.
2. Near-field effects: at r < ~1 wavelength, the Rayleigh wavefield is not yet dominated by the surface wave; body waves (P, S direct) contaminate the polarization estimate.
3. Love wave contamination: in a real field setup, Love waves (T-component) may be present from source asymmetry or lateral heterogeneity. These must be separated from Rayleigh before polarization analysis.
4. Requires 3-component sensors: standard single-axis vertical geophones give no bearing information. This method requires at least two 3-C nodes (or one 3-C node plus any other constraint).
5. 180° ambiguity: resolved by retrograde Rayleigh motion sense, but requires clean signal at a single dominant frequency.

**Data requirements.** At minimum 2 × 3-component sensors (geophone + accelerometer hybrid, as recommended in `04_sensor_hardware_hybrid.md`). Accurate sensor orientation in the horizontal plane (± 5° for 5° bearing accuracy). No velocity model needed for bearing; velocity model needed to convert bearings to intersection point.

**References.** Jurkevics 1988, BSSA (covariance polarization analysis); Teanby et al. 2004, BSSA (cluster analysis of eigenvectors); efficient six-component polarization analysis (arXiv 2212.11701, 2022); 3-C geophone orientation review (ResearchGate: Draganov & Ruigrok 2010).

---

### 2.5 Time Reversal

**Principle.** Time reversal (TR) localization: record the wavefield from an unknown source at all receivers. Time-reverse each recording and numerically back-propagate it through the medium. The back-propagated wavefield focuses at the original source location. In practice: compute the spatial distribution of back-propagated energy using the known or estimated propagation model, find the maximum.

For surface waves in a 1-D medium, the propagation is governed by the 2-D Helmholtz equation at each frequency ω:

> [∇² + k²(ω)] u(x, ω) = −δ(x − x_s)

where k(ω) = ω / c_R(ω) is the wavenumber. Time reversal is equivalent to back-propagation: the complex-conjugate spectrum u*(x_i, ω) re-emitted from receiver i back-propagates to focus at x_s. The focused energy at candidate location x_s is:

> P_TR(x_s) = |∫ [Σ_i u*(x_i, ω) × G(x_s, x_i, ω)] dω|²

where G(x_s, x_i, ω) is the forward Green's function (from x_i to x_s). This is mathematically identical to matched-field processing with a full-waveform replica, but the physical intuition of "re-emission and focusing" motivates a different implementation route: numeric back-propagation on a grid rather than dictionary precomputation.

**Reverberant environment advantage.** In a reverberant medium (reflections from layer boundaries), each reflection acts as a virtual source, effectively increasing the array aperture. TR focusing sharpens with increased reverberation (Fink 1993, Physics Today). In a layered half-space, the Rayleigh wave is already the product of constructive interference between free-surface-reflected P and S waves — TR exploits this automatically.

**Earthquake-scale implementation.** Bozdag et al. (GJI 2018, doi:10.1093/gji/ggy291) applied surface-wave TR to regional earthquakes using membrane-wave propagation (2-D Helmholtz), computing phase velocity maps at multiple periods and back-propagating. The algorithm focused successfully at known epicentre locations across a 300-km region. The method is computationally cheap relative to 3-D FEM: membrane-wave back-propagation is a 2-D scalar field problem.

**On the 1-D testbed.** The implementation simplifies: since the medium is 1-D (spatially uniform in the horizontal direction), the Green's function G(x_s, x_i, ω) = G(|x_s − x_i|, ω) = the same pyprop8 replica used for MFP. TR and MFP are therefore computationally equivalent on this testbed — TR provides a different physical interpretation but the same algorithm. The distinction matters in the 3-D Devito testbed, where TR back-propagation on a 3-D grid may be more efficient than a brute-force MFP grid search.

**Expected accuracy.** For surface waves in a 1-D medium with a known velocity model: equivalent to MFP Bartlett, so sub-meter at field scale and 5–20 cm with full calibration. Plate-scale experiments with controlled sources: < 3% of aperture (PMC11359918, 2024). For the testbed (aperture = array length = 10–20 m), this implies < 0.3–0.6 m.

**Failure modes.**
1. Requires the same model as MFP — velocity model errors propagate identically.
2. Limited phase-coherence aperture: at high frequencies (short wavelength), the focusing spot is λ/2. At 50 Hz with c_R = 150 m/s, λ/2 = 1.5 m — the fundamental resolution limit.
3. Sidelobes: in a finite array, the back-propagated field has sidelobes at ±λ from the true focus. With fewer than 6 receivers and a short aperture, sidelobes can be as large as the main lobe.

**Data requirements.** Full-waveform recordings at all receivers (same as MFP). Velocity model (same as MFP). In the testbed, TR and MFP will be implemented as two presentations of the same computation.

**References.** Fink 1993, Physics Today; Bozdag et al. GJI 2018 (surface-wave TR for earthquakes); PMC11359918 (plate TR, < 3% aperture error); Larmat et al. 2006 GRL (first seismic TR localization).

---

### 2.6 Likelihood-Surface Fusion

**Principle.** Each method above produces a spatial "likelihood" or "cost" surface over the candidate source grid: P_MFP(x_s), P_TDOA(x_s), P_RSS(x_s), P_bearing(x_s). These surfaces can be combined to produce a joint posterior that is sharper and more robust than any individual method.

The fusion is Bayesian product-of-experts if the methods are conditionally independent given the source location (which they approximately are if their errors are driven by different physical mechanisms: timing noise vs. amplitude noise vs. polarization noise):

> P_joint(x_s | data) ∝ P_MFP(x_s) × P_TDOA(x_s) × P_RSS(x_s) × P_bearing(x_s) × P_prior(x_s)

where P_prior(x_s) is a flat prior over the surveyed area (no prior information about source location) or a grid-based prior (e.g., restricted to walkable surface).

In practice, each likelihood surface must be normalised and converted to a proper probability (or log-probability). The CSDM used in MFP and the cross-correlation peak in TDOA both represent Gaussian-like likelihoods in good operating conditions. The RSS likelihood is approximately lognormal.

**Value of fusion.** Each method fails in different conditions:
- TDOA fails at low SNR (noisy onset picks) and with unmodeled dispersion.
- RSS fails at high attenuation heterogeneity.
- Bearing fails at small source–receiver distance (< λ).
- MFP fails at high model mismatch.

Their failures are largely uncorrelated. The product-of-experts fusion inherits the best accuracy of the individual methods and degrades gracefully when one method fails (the other terms still constrain the posterior). This is the standard result from multi-sensor fusion theory and is verified in joint TDOA + AOA localization literature (Lohrasbipeydeh et al. 2015; Shen & Ding 2008).

**Implementation on the testbed.** Because the testbed is simulation-based, each individual surface can be evaluated on a 1-D or 2-D source grid and the fusion is a per-pixel product. The Cramér-Rao lower bound for the joint estimator is:

> J_joint = J_TDOA + J_RSS + J_bearing + J_MFP

(Fisher information matrices sum for independent measurements), giving a joint CRLB that is tighter than any individual bound. This can be computed analytically for the 1-D translation-invariant model and used to quantify the gain from fusion.

**Expected improvement.** In the MFP + TDOA fusion regime: the complementary frequency bands used by each method (MFP exploits coherent spectral structure, TDOA exploits onset timing) reduce the joint RMSE by 20–40% relative to the better individual method, based on analogous fusions in underwater acoustics. The gain is largest when individual methods are near their individual noise floors (i.e., in the calibration-blind or low-SNR regime).

**Data requirements.** All data required by constituent methods. No additional sensors. Computationally: the cost of running all methods and multiplying their output surfaces.

**References.** Baggeroer et al. 1993 (MFP coherent-incoherent fusion); Shen & Ding, IEEE Trans. Signal Process. 2008 (TDOA + AOA Bayesian fusion); Berdugo et al. 1999 (joint bearing + range); Knapp & Carter 1976 (GCC CRLB as Fisher information for TDOA).

---

## 3. CRLB Analysis for the 1-D Layered-Soil Testbed

### 3.1 TDOA timing CRLB

For a sensor pair (i, j) with effective bandwidth β_eff (RMS bandwidth in rad/s), observation window T, and pre-detection SNR ρ:

> σ²_τ ≥ 1 / (8π² β²_eff T ρ)

For a 20-Hz narrow-band (f = 30 Hz center, β_eff = 2π × 20 rad/s = 126 rad/s), T = 0.2 s, ρ = 20 dB (100):

> σ_τ ≥ 1/(8π² × 126² × 0.2 × 100)^{0.5} ≈ 0.25 ms

At c_R = 150 m/s: σ_range = c_R × σ_τ ≈ 0.038 m per pair. With N=4 sensors and GDOP ≈ 1.5 (see §4): σ_position ≈ 0.06 m — the timing-noise floor.

### 3.2 Velocity uncertainty floor

A systematic error Δc/c in the assumed velocity introduces a ranging bias of d × Δc/c per sensor pair. For d = 10 m, Δc/c = 10%: bias = 1.0 m. This dominates timing noise entirely. The formula for the combined RMSE (bias + random):

> RMSE ≈ √((d × Δc/c)² + (GDOP × c × σ_τ)²)

For d = 10 m, Δc/c = 5%, GDOP=1.5, c=150 m/s, σ_τ = 0.25 ms:
> RMSE ≈ √(0.5² + 0.06²) ≈ 0.50 m — velocity-dominated.

For Δc/c = 1% (careful calibration): RMSE ≈ √(0.1² + 0.06²) ≈ 0.12 m — velocity and timing comparable.

### 3.3 MFP CRLB

For the Bartlett MFP processor with L snapshots, the CRLB for source range r is (Baggeroer et al. 1993):

> σ²_r ≥ 1 / (L × SNR² × |∂w/∂r|²)

where ∂w/∂r is the derivative of the replica steering vector with respect to source range. This is related to the sensor array geometry and the effective wavenumber spread. For the 1-D testbed, compute this analytically using pyprop8 sensitivity kernels (DerivativeSwitches gives ∂G/∂source_position via chain rule through source–receiver offset).

### 3.4 Joint Fisher information

Assuming timing and amplitude observations are independent, the joint FIM is:

> J_joint = J_TDOA + J_RSS + J_bearing

which bounds the joint estimator RMSE below min(RMSE_individual). The gap-to-CRLB metric for each method (RMSE_method / σ_CRLB) quantifies how far the method is from optimal. A gap of 1.0 means the method is CRLB-efficient; a gap of 3–10 is typical for practical estimators in moderate-SNR conditions.

---

## 4. Experimental Protocol

### 4.1 Testbed configuration (fixed across all experiments)

**Layered model (default).** Three-layer model representative of soft alluvial soil over bedrock:
- Layer 1: h=2 m, v_P=300 m/s, v_S=120 m/s, ρ=1.8 g/cm³
- Layer 2: h=5 m, v_P=600 m/s, v_S=250 m/s, ρ=2.0 g/cm³
- Half-space: v_P=2000 m/s, v_S=1000 m/s, ρ=2.5 g/cm³

This gives Rayleigh phase velocities of approximately 110 m/s at 5 Hz, 180 m/s at 30 Hz, 230 m/s at 80 Hz — a ~2× dispersion spread over the footstep band.

**Source.** Vertical point force on the surface (z=0), flat spectrum over 1–100 Hz, duration 0.02 s (half-Gaussian pulse). This models a footstep-like impulsive surface load.

**Source positions tested.** A 5×5 grid on a 10×10 m survey area (25 positions), plus 4 positions outside the receiver array perimeter (near-field and far-field).

**Sampling rate.** 1000 Hz (captures Rayleigh up to 500 Hz Nyquist; 100 Hz is the actual signal band). Window: 2 s per event.

**pyprop8 computation.** `compute_seismograms()` with vertical-force source at each of the 29 source positions, recording at all N receivers simultaneously. Output: 3-component displacement at each receiver, 2000-sample time series. Add Gaussian white noise at specified SNR to each channel after computing.

### 4.2 Experimental variables

The following variables are varied systematically, one at a time (full-factorial for the key combinations):

**A. N sensors (primary variable)**

| N | Geometry | Notes |
|---|---|---|
| 3 | Triangle, equilateral, 8 m side | Minimum for 2-D localization |
| 4 | Square, 8 m side | Field-proven minimum (doc 02 §3.3) |
| 6 | 4 corners + 2 midpoints of long sides | Recommended for house rooms |
| 8 | 4 corners + 4 midpoints | Dense, high-redundancy |

For TDOA, the number of independent TDOAs is N−1 (non-redundant) and N(N−1)/2 (all pairs). Track accuracy improvement as a function of N.

**B. Geometry / GDOP (secondary variable)**

For fixed N=4, vary the array shape:
- Square (GDOP ≈ 1.4): sensors at corners of a 8×8 m square
- Rectangle 2:1 (GDOP ≈ 1.8): sensors at corners of 10×5 m
- L-shape (GDOP ≈ 2.5): three on a 10 m baseline, one at 90°
- Collinear (GDOP → ∞ for off-axis sources): degenerate, used to confirm GDOP blow-up

Compute GDOP analytically for each source position and sensor geometry (GDOP = trace(H^T H)^{−1} × c² where H is the TDOA Jacobian). Track how GDOP × σ_TDOA predicts the measured RMSE.

**C. Source positions relative to array**

- Interior (all 25 grid points inside the 8×8 m square): best GDOP
- Array-edge: source near one side
- Array exterior: source 3 m outside array perimeter (tests extrapolation)
- Near-field: source 0.5 m from nearest receiver (tests near-field breakdown of assumptions)

**D. SNR levels**

SNR = {0, 10, 20, 30} dB (noise added to each synthetic channel post-simulation). This directly controls the timing CRLB. At SNR = 0 dB, all onset-based methods should fail; at SNR = 30 dB, methods should approach their velocity-uncertainty floor.

**E. Velocity knowledge (blind-velocity tests — see §5)**

Five conditions:
1. Perfect (oracle): use exact pyprop8 model parameters for all methods
2. Calibrated ± 2%: add Δc/c = ±2% uniform error to each layer's v_S
3. Calibrated ± 10%: Δc/c = ±10% (mis-estimated)
4. Calibrated ± 20%: Δc/c = ±20% (poor calibration)
5. Dispersion-blind: assume constant velocity c_R = mean(c_R(f)) across all frequencies (ignores dispersion entirely)

### 4.3 Metrics

For each (method, N, geometry, SNR, velocity_condition, source_position) combination:

**Primary metric.** Horizontal position error: ε = ‖x̂_s − x_s‖ in metres. Report mean, median, 90th percentile, and RMSE over all 29 source positions.

**Secondary metric.** Gap to CRLB: G = RMSE / σ_CRLB(x_s), computed using the analytical Fisher information for that source position, N, SNR, and velocity uncertainty. G ≥ 1 always (CRLB is a lower bound). Track G across conditions to identify when each method is efficient (G ≈ 1) vs. wasteful (G >> 1).

**Tertiary metric.** Failure rate: fraction of source positions where ε > 1.0 m (the "hard failure" threshold for a practical localization system). A method that achieves 0.3 m median but 20% failure rate at > 1 m is not field-deployable.

**Quaternary metric (for fusion).** Relative improvement of fusion over the best individual method: ΔG = (RMSE_best_single − RMSE_fused) / RMSE_best_single. Track whether fusion provides > 10% improvement (meaningful) or < 5% (marginal — individual method already near-optimal).

### 4.4 Comparison protocol

To compare methods fairly:

1. **Same synthetic data.** All methods receive identical simulated seismograms (same pyprop8 realization, same noise draw). No method gets access to more data than another. The only allowed difference is in how much of the velocity model each method uses (per the velocity_condition variable).

2. **Same source grid.** The MFP/TR replica dictionary is evaluated on the same 1-D distance grid as the TDOA and RSS methods. Grid resolution: 0.05 m (finer than the expected 0.1 m CRLB floor).

3. **Same prior.** All methods use a flat uninformative prior over the 10×10 m survey area. No method uses knowledge of the source position during inference.

4. **Head-to-head at equal calibration.** The velocity-knowledge condition must be identical across all methods in any single comparison run. "MFP with perfect model vs. TDOA with blind model" is not a fair comparison.

5. **CRLB normalization.** Report all method errors normalized to the local CRLB σ_CRLB(x_s) to remove the confound of GDOP varying across source positions.

6. **Bootstrap confidence intervals.** At each (method, condition) cell, run 50 independent noise realizations. Report 95% CI on mean RMSE and on gap-to-CRLB G.

---

## 5. Blind-Velocity Rigor: The Honest vs. Cheating Distinction

This section operationalizes the distinction in doc 02 §4.3 for the 1-D soil testbed.

### 5.1 The two dimensions of blindness

**Dimension 1 — Dispersion blindness.** Does the method assume a dispersion curve c_R(f) or a single constant velocity c_0?

- *Dispersion-blind (constant-velocity)*: set c_R(f) = c_0 for all f. c_0 is taken as the group velocity at the dominant frequency of the source spectrum. This is the classic TDOA assumption and the baseline for showing dispersion's importance.
- *Dispersion-informed*: use the true c_R(f) from pyprop8 (in the "perfect" condition) or a calibrated estimate.

**Dimension 2 — Calibration blindness.** Is the velocity magnitude correct?

- *Calibration-blind (Δc/c = 10–20%)*: use a velocity that is systematically wrong by the specified fraction. This models the situation where the operator has assumed a soil type that is similar but not measured.
- *Calibration-informed (Δc/c = 2–5%)*: use a velocity measured from a hammer calibration shot, with realistic measurement uncertainty.

These two dimensions define a 2×2 matrix of scenarios, all of which must be tested:

|  | Dispersion-blind | Dispersion-informed |
|---|---|---|
| **Calibration-blind** | Worst case: constant wrong velocity | Partial: know the shape of c_R(f) but not absolute scale |
| **Calibration-informed** | Cheating with constant velocity: correct speed, wrong dispersion model | Gold standard: correct c_R(f) everywhere |

The *honest* TDOA result is calibration-informed + dispersion-informed. Reporting only the gold-standard result while claiming field applicability is cheating — every published result should be accompanied by its blind-velocity degradation.

### 5.2 The "cheating vs. honest" distinction

The key fraud risk in methods papers: a method is calibrated on the same data it is evaluated on, or the velocity model is taken from the simulation oracle rather than from a realistic calibration procedure. The testbed prevents this by construction:

- The "oracle" condition uses pyprop8 parameters directly — explicitly labeled as oracle.
- The "calibrated" conditions use a separate calibration shot (a synthetic hammer at a known location), from which the velocity is estimated. The estimation adds realistic uncertainty. The localization evaluation uses a different set of source positions — no overlap.
- All other conditions degrade calibration systematically.

### 5.3 Quantifying dispersion bias

For the constant-velocity TDOA method with true dispersion curve c_R(f), the expected bias in a TDOA estimate between receivers at distances r_i, r_j from the source is:

> Δτ_bias(f_dom) = (r_i − r_j) × (1/c_0 − 1/c_R(f_dom))

where f_dom is the dominant frequency of the cross-correlation window and c_0 is the assumed constant velocity.

For the default soil model, at r_i − r_j = 5 m, f_dom = 30 Hz (c_R = 180 m/s), c_0 = 150 m/s (assumed):

> Δτ_bias = 5 × (1/150 − 1/180) = 5 × 0.00056 = 2.8 ms → 2.8 × 150 = 0.42 m ranging bias

At 10% velocity mismatch (c_0 = 162 m/s vs. c_R = 180 m/s):

> Δτ_bias = 5 × (1/162 − 1/180) = 5 × 0.00062 = 3.1 ms → 0.50 m bias

These analytical predictions will be verified against the simulation results in the testbed.

### 5.4 Dispersion-blind MFP

MFP can also be run dispersion-blind: instead of using pyprop8 replicas (which correctly model dispersion), use a constant-velocity plane-wave replica. This degrades MFP to SRP (Steered Response Power) / conventional beamforming — a well-known degradation. Comparing MFP-oracle vs. MFP-dispersion-blind quantifies the value of the propagation model in the MFP context.

### 5.5 Velocity estimation from the array itself

A calibration-free approach: estimate c_R from the array data itself using the MASW (Multichannel Analysis of Surface Waves) technique. For a linear array of N geophones with spacing Δx:

1. Compute the N×N cross-spectral density matrix (CSDM) at each frequency f.
2. Apply beam-shift analysis: for each trial velocity c_trial and frequency f, the wavenumber k = ω/c_trial; the beam power peaks when the wave equation is satisfied.
3. The resulting "f-k" or "f-c" diagram gives c_R(f) directly from the data.

For the testbed: synthetic seismograms from a calibration shot at a known location give the empirical c_R(f) used in subsequent localization. This is the "array-self-calibration" approach, achievable with Δc/c ≈ 3–5% for an array of 8+ geophones with inter-sensor spacing of 1–2 m.

**MASW velocity resolution.** For an array of N sensors with spacing Δx and aperture A = N × Δx, the minimum resolvable velocity difference at wavenumber k is Δc/c = λ/(2A) = c/(2Af). For A = 8 m, f = 30 Hz, c = 180 m/s: Δc/c = 180/(2×8×30) = 0.38% — well below the 2–5% target. MASW is well-suited for the testbed.

---

## 6. Implementation Roadmap for the Testbed

This is a research design document, not an implementation. The sequence when implementation begins:

1. **Environment setup.** Install pyprop8 (pip install pyprop8). Verify against the documented examples (compute a seismogram, confirm Rayleigh wave group velocity matches known formula for the chosen soil model).

2. **Forward model.** Write a function `simulate_event(source_pos, receivers, model, snr_db)` that calls `pyprop8.compute_seismograms()` and adds noise. Validate that the Rayleigh arrival time scales as r / c_group(f).

3. **Replica dictionary.** Build the 1-D table G(d, omega) for distances d ∈ [0.5, 20] m at 0.05 m resolution and frequencies f ∈ [1, 100] Hz at 1 Hz resolution. This is O(400 × 100) = 40,000 pyprop8 calls — a one-time offline computation of ~minutes.

4. **Method implementations.** One module per method: MFP (Bartlett + MVDR-DL), SO-TDOA (sign-only hyperbola ordering), TDOA-full (GCC-PHAT + nonlinear least-squares), RSS (energy trilateration + LM), polarization bearing (covariance eigendecomposition), fusion (log-probability product). Each method takes as input the simulated seismograms and the velocity condition, and returns a 2-D likelihood surface.

5. **Protocol runner.** A loop over all experimental conditions, calling each method, recording ε and G metrics, saving results as a structured DataFrame.

6. **CRLB computation.** Implement the analytical TDOA FIM for each source position and sensor geometry. Compute σ_CRLB(x_s) for the timing-noise-only case and for the combined timing + velocity-uncertainty case.

7. **Blind-velocity sweeps.** Run the calibration-blind and dispersion-blind conditions. Plot RMSE vs. Δc/c for each method to generate the "degradation curve."

---

## 7. Expected Findings and Publication-Grade Conclusions

The following predictions are hypothesis-level — to be verified by the simulation:

1. **MFP (oracle model) will be the top performer** at all SNR levels, because it uses the most complete information. Expected: 0.05–0.15 m RMSE at SNR ≥ 20 dB.

2. **TDOA-dispersion-corrected will rank second**, within 1.5× of MFP at high SNR, degrading to 3–5× at dispersion-blind condition.

3. **RSS will rank third or fourth**, accurate to ~0.3–0.7 m with perfect amplitude calibration, degrading to > 1 m when attenuation heterogeneity is introduced.

4. **Polarization bearing will rank third or fourth depending on SNR** — competitive at high SNR (bearing error < 3° → position error < 0.5 m at 10 m range) but degrades faster than TDOA below SNR = 10 dB.

5. **Dispersion blindness will be the single biggest differentiator**: constant-velocity TDOA will be 3–10× worse than dispersion-corrected TDOA, matching the 3.08× improvement factor seen in the indoor concrete literature (EDMF study in doc 02 §1.3).

6. **Fusion will improve upon the best individual method by 15–35%** at calibration-limited SNR, and by < 10% at oracle SNR (where MFP alone is already near-CRLB).

7. **GDOP is the dominant geometric factor**: for the collinear and L-shape arrays, position error will blow up for sources on the degeneracy axis, verifiable analytically before simulation.

These predictions, once verified (or falsified) by the simulation, constitute the publication contribution of the 1-D testbed.

---

## Sources

- pyprop8 (Valentine & Woodhouse 2022): [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04217); [GitHub](https://github.com/valentineap/pyprop8); [docs](https://pyprop8.readthedocs.io/en/latest/seismograms.html)
- Baggeroer, Kuperman & Mikhalevsky 1993 (MFP review): IEEE J. Ocean. Eng. doi:10.1109/48.107143
- Seydoux et al. 2018 (MFP + MCMC dense geophone array): [GJI doi:10.1093/gji/ggy139](https://academic.oup.com/gji/article/218/2/1044/5490346)
- Bozdag et al. 2018 (surface-wave TR earthquake location): [GJI doi:10.1093/gji/ggy291](https://academic.oup.com/gji/article/215/1/1/5046453)
- Bahroun et al. 2014 (SO-TDOA, perceived propagation velocity, concrete): [arXiv:1211.3233](https://arxiv.org/abs/1211.3233); J. Sound Vib. doi:10.1016/j.jsv.2013.10.040
- Casaclang et al. 2024 (dispersion propagation operators, Sensors PMC11644851): [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644851/)
- PMC12473310 2026 (MEMS accelerometer outdoor Rayleigh localization, 0.095 m with PSO-GDOP): [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473310/)
- Mirshekari et al. 2016 (indoor TDOA, 0.34 m concrete): MSSP doi:10.1016/j.ymssp.2016.05.044
- Alajlouni & Tarazaga 2020 (RSS trilateration Kalman, 0.63 m): J. Vib. Control doi:10.1177/1077546319890520
- Casaclang et al. 2026 (reservoir computing, 0.65 m, energy comparison): [arXiv:2603.04610](https://arxiv.org/html/2603.04610v1)
- Jurkevics 1988 (3-C polarization analysis BSSA): doi:10.1785/BSSA0780010218
- PMC11359918 2024 (time reversal plate localization, < 3% aperture): [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11359918/)
- Chan & Ho 1994 (hyperbolic TDOA closed-form solution): IEEE Trans. Signal Process. doi:10.1109/78.295550
- Knapp & Carter 1976 (GCC CRLB): IEEE Trans. ASSP doi:10.1109/TASSP.1976.1162830
- Gesret et al. 2015 (velocity model uncertainty propagation to location): [GJI doi:10.1093/gji/ggu502](https://academic.oup.com/gji/article/200/1/52/742502)
- Velocity-free AE localization (PMC7349182 / PMC7828095): [PMC7349182](https://pmc.ncbi.nlm.nih.gov/articles/PMC7349182/); [PMC7828095](https://pmc.ncbi.nlm.nih.gov/articles/PMC7828095/)
- MFP for seismic sources, complex Earth: [GJI doi:10.1093/gji/ggac286](https://academic.oup.com/gji/article/231/2/1268/6619059)
- Six-component polarization analysis (arXiv:2212.11701): [arXiv](https://arxiv.org/pdf/2212.11701)
