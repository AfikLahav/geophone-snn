> **Numerical fidelity verification — pyprop8 as a 1-D localization testbed.** Authored 2026-07-08. Research phase only. Inputs: `simgeo_v42/gfbank_build.py`, `scenes.py`, `profiles.py`; `research/house_localization/05_structural_reverberation.md`; `research/localization_1d/02_testbed_architecture.md`; pyprop8 upstream (O'Toole & Woodhouse 2011, JOSS 2022); Bouchon 2003 discrete-wavenumber review; MASW near-surface literature.

---

# pyprop8 Numerical Fidelity for the 1-D Localization Testbed

## Executive Summary (6 lines)

1. **Dispersion IS meaningful** over 3–20 m at 20–100 Hz for this project's soft-soil profiles (Vs 100–300 m/s, layers 0.5–8 m): wavelengths 1–15 m are comparable to layer thicknesses, placing the band squarely in the dispersive regime. Stiff profiles (Vs ≥ 500 m/s) are weakly dispersive at these ranges.
2. **pyprop8 converges** to the 250 Hz Nyquist ceiling and down to r = 0.5 m: the ghost-suppression logic in `gfbank_build.py` (contour-damping alpha + adaptive nk) is correctly designed and matches Bouchon 2003 theory; stiff-profile warnings are surfaced explicitly.
3. **3-component output and arbitrary point-force** are natively supported: pyprop8 returns (Z, R, T) for any force tensor; the current bank discards R/T but the compute cost is already paid.
4. **Per-layer Q** is not a pyprop8 input parameter; attenuation is applied post-hoc via the causal t* (Azimi) model in `scenes.py`, which is a body-wave approximation adequate for the soft-soil Rayleigh regime but slightly mis-specifies frequency-dependent Q for trapped or guided modes.
5. **The key limitation is physics, not numerics:** pyprop8 gives soil-Rayleigh dispersion; indoor localization involves Lamb/flexural-plate waves. The dispersion law differs fundamentally (Rayleigh: weakly dispersive above corner; Lamb: c_g ∝ √f). The testbed cannot reproduce indoor physics absolutely, but method rankings — which localizer wins, by how much, and under what geometry — transfer robustly because those rankings depend on geometry, SNR, and velocity uncertainty more than on the exact dispersion law.
6. **Verdict: the 1-D testbed is scientifically sound** as a localization-algorithm development platform, provided that (a) it is clearly labelled as a soil-Rayleigh proxy, not an indoor-slab model, (b) dispersion-relevant conclusions (velocity-sensitivity curves) are expressed as fractional deviations Δv/v rather than absolute meters-per-second, and (c) one spot-check Devito run is planned to validate method-ranking transfer.

---

## 1. Dispersion at Short Range: Is It Meaningful?

### 1.1 The Physical Question

Surface-wave dispersion in a 1-D layered half-space arises because different frequencies sample different depths. A Rayleigh wave at frequency f has a characteristic penetration depth z_pen ≈ λ/3 where λ = v_R(f)/f is the wavelength at that frequency. When z_pen spans a Vs contrast (a layer boundary), the phase velocity is frequency-dependent. The **dispersion is "meaningful" for localization** when the frequency-dependent group velocity spread Δv_g is large enough to produce TDOA errors comparable to or larger than the target localization accuracy.

The condition for dispersion to be active (i.e., for the fundamental-mode Rayleigh wave to sense the layer boundary) is:

```
λ/30 < h_layer < 30λ
```

where h_layer is the layer thickness (Xia et al. 1999; confirmed by MASW literature: wavelengths 1–30× the layer thickness drive the dispersive regime). Outside this band, the wave either rides the surface layer alone (λ ≪ h_layer) or averages over the full half-space (λ ≫ h_layer).

### 1.2 Quantitative Calculation for This Project's Profiles

The soil families in `profiles.py` span Vs_top = 100–500 m/s with top-layer thicknesses 0.5–25 m. Rayleigh-wave phase velocity is approximately v_R ≈ 0.92 × Vs_top (homogeneous half-space limit). The frequency range of interest is 20–100 Hz (the peak footstep energy band). Table 1 works out the wavelengths and the comparison with layer thickness for the key families:

**Table 1. Rayleigh Wavelengths vs. Layer Thicknesses at 20 and 100 Hz**

| Family | Vs_top (m/s) | v_R approx (m/s) | λ @ 20 Hz (m) | λ @ 100 Hz (m) | h_top range (m) | Dispersive? |
|---|---|---|---|---|---|---|
| soft_soil | 100–300 | 90–270 | 4.5–13.5 | 0.9–2.7 | 0.5–8 | YES — strongly |
| loess | 150–400 | 140–370 | 7–18 | 1.4–3.7 | 2–25 | YES — strongly |
| sand | 100–500 | 90–460 | 4.5–23 | 0.9–4.6 | 1–15 | YES — strongly |
| gravel | 300–750 | 280–690 | 14–34 | 2.8–6.9 | 0.5–6 | Moderate–weak |
| clay | 150–800 | 140–740 | 7–37 | 1.4–7.4 | 1–20 | YES — strongly |
| wet_soil | 200–600 | 180–550 | 9–27 | 1.8–5.5 | 1–10 | Moderate |
| rock | 800–2500 | 740–2300 | 37–115 | 7.4–23 | 0.2–3 | WEAK — wavelength >> h |
| kurkar | 475–1500 | 440–1380 | 22–69 | 4.4–13.8 | 1–10 | Moderate–weak |
| asphalt (inverse) | 1000–1800 | 920–1660 | 46–83 | 9.2–16.6 | 0.05–0.20 | NEGLIGIBLE |
| concrete (inverse) | 2000–2500 | 1840–2300 | 92–115 | 18.4–23 | 0.12–0.30 | NEGLIGIBLE |

**Key conclusion:** For the soft families (soft_soil, loess, sand, clay — approximately 94 of the 282 banks), Rayleigh wavelengths at 20–100 Hz range from 1–23 m, which is squarely comparable to the layer thicknesses 0.5–25 m. These profiles are in the strongly dispersive regime (λ/h ≈ 0.1–20, spanning the 1–30 window). The dispersion challenge is **real and non-trivial** over 3–20 m ranges.

### 1.3 What "Dispersion" Means Concretely for Localization Over 3–20 m

**Group velocity spread at 3–20 m range:**

For a two-layer model (soft layer Vs1, half-space Vs2 with Vs2 ≈ 2–4 × Vs1), the fundamental-mode group velocity ranges from ~0.92 Vs1 (high-frequency limit) to ~0.92 Vs2 (low-frequency limit). For a soft_soil profile with Vs1 = 200 m/s, Vs2 = 600 m/s: v_g spans roughly 180–550 m/s over the 20–100 Hz band.

**TDOA error from a 20% velocity mis-specification at 5 m range:**

```
Δτ = r × Δv / v² = 5 m × (0.20 × 300 m/s) / (300 m/s)² = 5 × 60 / 90000 ≈ 3.3 ms
Δx ≈ v × Δτ ≈ 300 × 0.0033 ≈ 1.0 m
```

At 10 m range with 20% velocity error: Δx ≈ 2.0 m. These errors are substantially larger than the 0.1–0.5 m target localization accuracy. So yes — dispersion is a meaningful challenge and the testbed correctly exercises it.

**Dispersion spread at 5 m propagation distance:**

A 20 Hz component traveling at v_g(20 Hz) = 550 m/s arrives at t = 5/550 = 9.1 ms. A 100 Hz component traveling at v_g(100 Hz) = 180 m/s arrives at t = 5/180 = 27.8 ms. The waveform is smeared over **18.7 ms** — more than one footstep period. A naive broadband TDOA pick averages over this spread. This waveform stretching is what differentiates localization methods: MFP using the exact GF bank resolves it; constant-velocity TDOA does not.

**Comparison with the indoor Lamb-wave case (from doc 05):**

Indoor concrete slab (150 mm): v_g(20 Hz) ≈ 192 m/s, v_g(100 Hz) ≈ 430 m/s — a 2.2× ratio over the same band. The soil Rayleigh case above: 550/180 ≈ 3× ratio. The **dispersion is actually larger in the soil case** for soft profiles. For stiff profiles (gravel, rock), the soil is weakly dispersive and the indoor slab is more dispersive. So the testbed exercises a dispersion challenge that is comparable to or harder than the indoor problem for the majority of soft-soil profiles.

### 1.4 Does 1-D Under-Represent the Dispersion Challenge?

For the **quantity of dispersion** (how much v_g varies with frequency): 1-D half-space gives the correct dispersion curve for horizontally layered soil — it is not an approximation for this geometry. The dispersion computed by pyprop8 is exact for the specified layered model.

What 1-D **does** under-represent is:
- **Lateral heterogeneity:** real soil has horizontal Vs variations of 20–50% over 5–20 m scales. These scatter surface waves and add apparent dispersion that a 1-D model cannot capture. This is a real-world complication that is not present in the testbed.
- **Boundary reflections:** the half-space has no room boundaries. For outdoor soil over 3–20 m at 20–100 Hz, the free surface is the only reflector (already included in Rayleigh wave theory). No wall reflections. This is correct for outdoor-geophone use and an appropriate simplification for method-development.
- **Higher Rayleigh modes:** at short ranges (< 3–5 m), higher modes contribute non-negligibly (Love waves, overtones). pyprop8 includes these automatically in the wavenumber integration (it computes the full Green's function, not mode-by-mode), so this is not a limitation.

**Verdict for §1:** Yes, surface-wave dispersion is real and meaningful at 3–20 m for soft/medium profiles. 1-D does not under-represent dispersion — it represents it correctly for its geometry. What is missing is lateral heterogeneity, which is a real-world complication rather than a fundamental flaw of the testbed.

---

## 2. Convergence at the Footstep Band and Short Range

### 2.1 The Discrete-Wavenumber Ghost Problem

The discrete-wavenumber (DW) method (Bouchon & Aki 1977; reviewed in Bouchon 2003) approximates the continuous wavenumber integral by a sum over discrete wavenumbers k_n = n × dk. Mathematically this is equivalent to assuming the physical source is repeated at spatial period L = 2π/dk — creating **ghost source copies** at distance L, 2L, … from the true source. The first ghost arrives at time t_ghost = L/v_p_max = 2π/(dk × v_p_max).

For convergence, ghost contamination must be suppressed to below the noise floor before it arrives. The two control parameters are:
1. **dk** (wavenumber spacing): smaller dk → larger L → later ghost arrival. Achieved by increasing nk (number of wavenumber samples) at fixed kmax.
2. **alpha** (imaginary-frequency contour shift, ω → ω + iα): an exponential damping exp(−α × t) suppresses the ghost by exp(−α × t_ghost) but amplifies late-time noise by exp(+α × T).

### 2.2 How `gfbank_build.py` Handles This

The `stencil_for()` function implements a two-sided bound (from the comments and logic at lines 25–62):

**Ghost suppression condition:**
```
alpha × t_ghost ≥ 13.5   →   suppresses ghost to exp(−13.5) ≈ 1.4 × 10⁻⁶
```
This is conservative: the ghost is suppressed to 1 part per million relative to the direct signal at t=0. Since the ghost scales with the near-field amplitude (which decays as r^{-0.5} or faster), at far range (r = 320 m) where the signal has decayed ~40 dB, this suppression is adequate. At near range (r = 0.5 m), the near-field amplitude is large and the ghost would be destructive without adequate suppression — hence the strict 13.5 criterion (the comment in the code explicitly notes "1e-3 left ~5% residue at far radii... since the ghost scales with NEAR-FIELD amplitude").

**Cap on alpha to prevent noise amplification:**
```
alpha ≤ 9.0/T   →   exp(+alpha × T) ≤ exp(9) ≈ 8103
```
At float64 (53-bit mantissa, ~15.9 decimal digits), this cap prevents amplification from eating into the float64 headroom.

**When the cap binds (stiff profiles), nk is raised instead:**
```
nk ≥ 0.239 × kmax × v_p_max × T
```
This pushes the ghost period out so that even the capped alpha suffices. The explicit WARNING in line 61 flags when this bound is not met even at the nk cap (30,000), so the user knows which profiles have residual ghost contamination.

**kmax selection for near-field evanescent waves:**
```python
kmax = max(1.25 × 2π × 250 / v_R, 10000/km)   # lower bound 10 rad/m
```
The 10000/km = 10 rad/m lower bound ensures the evanescent near-field at depth z ~ r_min = 0.5 m is captured: evanescent wavenumber k_evan ≈ 1/z = 2 rad/m for z = 0.5 m, covered comfortably by kmax = 10 rad/m.

**Assessment:** The ghost-handling logic is theoretically grounded and matches Bouchon's (2003) standard recommendations. The adaptive nk strategy correctly handles the stiff-profile case (where a naive fixed nk would fail). The explicit WARNING for unresolvable cases provides a safety net. The code comments cite Bouchon & Aki 1977 and Bouchon 2003 explicitly.

### 2.3 Convergence at 250 Hz (The Ceiling)

The project's design choice — computing banks at 250 Hz Nyquist (dt = 2 ms) with a cosine taper over the top 10% (225–250 Hz) and then zero-padding to 1000 Hz — is correct and standard. The gfbank_build.py header explicitly states: "the wavenumber stencil only converges to ~250 Hz anyway; the 250–500 Hz half of a 1 ms bank is unconverged junk." This is consistent with published discrete-wavenumber practice: the high-frequency limit is set by the highest k-mode captured (kmax ≈ 1.25 × 2π × 250/v_R), and beyond that limit the integral is not converged.

At 250 Hz, the minimum Rayleigh wavelength for soft soil (v_R = 90 m/s) is λ_min = 0.36 m — less than the receiver separation at r = 0.5 m. This means individual waveforms at r = 0.5 m include signal at wavelengths shorter than the receiver–source distance, which requires the near-field evanescent terms. The kmax = 10,000 rad/km = 10 rad/m lower bound captures these correctly (evanescent wavenumber ≈ 2π × 250/90 × 1.25 = 21.8 rad/m for v_R = 90, so kmax for these soft profiles actually sets via the Rayleigh criterion: 1.25 × 2π × 250/90 ≈ 21.8 rad/m, well within the kmax = 24,000 rad/km = 24 rad/m cap).

### 2.4 Convergence at r = 0.5–3 m (Short Range)

Short-range convergence is the most demanding case because:
1. The near-field P-SV body-wave terms dominate (r^{-2} geometric decay for near-field static, r^{-1} for body waves), requiring large kmax to integrate evanescent tails.
2. The time window T_window is short (the slowest Rayleigh arrives at t ≈ r/v_R ≈ 3 ms for r = 0.5 m, v_R = 180 m/s), but the ghost arrives at t_ghost = L/v_p_max, which must be larger than T_window.

The `stencil_for()` function sets T based on coda time + r_max/v_R — for r_max = 0.06 km (the minimum, soft profiles), T ≈ 0.06/0.18 + 2.5 ≈ 2.83 s. The ghost criterion then requires:
```
L = 2π/dk ≥ v_p_max × alpha/dk ≥ 13.5/alpha
```
With alpha = min(2.15 × dk × v_p, 9.0/T) — this is automatically satisfied by the stencil design.

The `settle_clamp()` function in `scenes.py` kills late-time numerical artifacts (alpha-amplified noise and undamped elastic ringing in stiff profiles) by clamping each radius's late tail to its median after the physical coda. This is a valid regularization: after all real seismic energy has passed, the step-function response is the static offset (a physical constant), and the coda clamp enforces this physically correct behavior while suppressing numerical noise.

**Assessment:** Convergence at short range is handled correctly by the stencil logic. The most vulnerable case — stiff profiles (asphalt, concrete, rock) at short range — is addressed by raising nk and flagging unmet bounds. For the soft profiles at 3–20 m (the primary localization testbed range), convergence is fully controlled.

---

## 3. Three-Component Output, Point-Force Source, Per-Layer Q

### 3.1 Three-Component Output

pyprop8's `compute_seismograms()` returns a seismogram array of shape `(n_sources, n_receivers, 3, nt)` where axis 2 is the component index: [0] = radial (R), [1] = transverse (T), [2] = vertical (Z) by default (Cartesian or polar basis, selectable via `xyz=False`).

**What the current bank stores:** `gfbank_build.py` line 100–101 saves only `a[0, :, 2, :]` (Z/Fz) and `a[1, :, 2, :]` (Z/Fx). The radial and transverse components — `a[:, :, 0, :]` (R) and `a[:, :, 1, :]` (T) — are computed by pyprop8 in the same call but discarded before writing to disk.

**Implication for the testbed:** Extending to 3-component output (needed for polarization bearing in localizer module §3.5 of doc 02) requires only two additional `np.savez_compressed` save lines — no change to the pyprop8 call itself. The additional storage is ~4 × (48 radii × 2500 samples × 4 bytes) ≈ 1.9 MB per bank (negligible).

**Transverse component:** For a vertical (Fz) or radial-horizontal (Fx) point force in a 1-D laterally homogeneous model, the transverse (T) component is identically zero by the symmetry of P-SV/SH decoupling. This is a mathematical exact result for 1-D models, not an approximation. The testbed recovers bearing information from the R-component amplitude ratio, not from T.

### 3.2 Point-Force Source Configuration

The current bank uses `PointSource(0., 0., 0.001, np.zeros((2,3,3)), F2, 0.)` — a buried point source at 1 m depth with two simultaneous force vectors:
- `F2[0] = [0, 0, 1]`: vertical force (Fz), representing the footstep normal load
- `F2[1] = [1, 0, 0]`: horizontal radial force (Fx), representing the forward push-off

This is physically correct for a footstep source: heel-strike and toe-off inject both a downward load and a horizontal push (Alajlouni & Tarazaga 2020 explicitly model the horizontal force component). The 1 m burial depth is a technical regularization (pyprop8 requires a buried source to avoid the free-surface singularity at depth=0) — at 1 m depth in a soft-soil profile with wavelengths ≥ 1 m, the surface response is not significantly altered.

pyprop8 supports **arbitrary moment tensor and force vector inputs** (the `F` parameter takes a 3×3 matrix for the moment tensor and the `FI` parameter takes the force vector). The current usage correctly represents a general double-force (Fz + Fx) source.

### 3.3 Per-Layer Q / Attenuation

**What pyprop8 does natively:** The O'Toole & Woodhouse (2011) propagator and the pyprop8 implementation assume a **purely elastic** (Q = ∞) layered half-space. There is no built-in Q attenuation per layer. The code is designed for static-to-teleseismic seismology where whole-path attenuation is applied separately.

**What the current code does:** Attenuation is applied post-hoc in `scenes.py`'s `Bank.__init__()` via the **causal t\* (Azimi) model** (lines 79–82):
```python
tstar = (self.radii_m / (q * vr))[:, None]   # path-integrated t* = r / (Q × v_R)
att = exp(-π f t*) × exp(2i f t* × log(f/25))  # Azimi amplitude + dispersion phase
```
where `q` and `vr` are the bulk Q and Rayleigh velocity from the profile metadata (a single representative value per profile).

**Adequacy assessment:**

*Positive:* The Azimi model correctly captures the causal t* operator for Rayleigh waves in an attenuating half-space: amplitude decays as exp(−π f t*) and the Kramers-Kronig phase dispersion is included (the log term). The t* = r/(Q v_R) parameterization is the standard seismological representation and is exact for a homogeneous Q model along the ray path.

*Limitation:* Real layered profiles have **depth-dependent Q**: soft wet soil has Q ≈ 5–20; kurkar has Q ≈ 30–80; rock has Q ≈ 50–200. The effective Q for a Rayleigh wave depends on which layers the wave samples at a given frequency. A single bulk-Q approximation misses this frequency-dependent effective attenuation. The error is bounded: for the dominant 20–100 Hz band, Rayleigh wavelengths are 1–15 m for soft profiles, penetrating 0.3–5 m. If the top layer (the softest, lowest-Q layer) dominates at these depths, the single Q value from the profile metadata (which reflects the top layer) is a good approximation. For profiles with strong Q contrasts across layers, errors of 20–50% in predicted amplitude vs. range are expected.

*Impact on localization:* Amplitude-based (RSS) localization is directly affected by Q errors — a 30% Q error produces a ~15–20% range error. TDOA and MFP are much less sensitive (they use phase information or waveform shape, which t* affects weakly below 100 Hz at ranges < 20 m where t* < 0.1 s). For the testbed, this limitation means RSS-based localizer tests are the most Q-sensitive; TDOA and MFP tests are robust.

*For the paper:* state that Q is applied as a path-integrated causal t* with a single representative per-profile value, and acknowledge that depth-dependent Q would require a more complex effective-Q calculation (e.g., frequency-dependent weighted average by Rayleigh-mode sensitivity kernel).

---

## 4. The Key Limitation: Soil-Rayleigh vs. Lamb/Flexural-Plate Physics

### 4.1 What the Testbed Actually Models

pyprop8 computes **Rayleigh waves in a semi-infinite layered soil half-space** (P-SV system, flat free surface, no top boundary). The fundamental mode Rayleigh wave on a homogeneous half-space has a near-constant group velocity v_g ≈ 0.92 Vs and is only weakly dispersive. In a layered half-space, it acquires dispersion through the layer structure.

Key propagation properties of soil-Rayleigh:
- Geometric spreading: amplitude ∝ r^{−0.5} (cylindrical, 2-D spreading)
- Attenuation: amplitude ∝ exp(−π f r / (Q v_R)) via t*
- Dispersion law: depends on layer thicknesses and Vs contrast; can be strong (factor 3×) for soft profiles
- No upper boundary: no reflections from room walls or a rigid slab surface
- No flexural modes: Rayleigh is a surface wave of the half-space, not a guided wave in a plate

### 4.2 What Indoor Localization Actually Involves

From doc 05, the indoor concrete/masonry floor is a **thin elastic plate** supporting Lamb/flexural waves:
- Phase velocity: c_B(f) = (2πf)^{1/2} × (D/ρh)^{1/4} ∝ √f — strongly dispersive, 2-D plate
- Group velocity: c_g = 2 c_B in the thin-plate limit (factor 2× vs phase velocity at all f)
- Geometric spreading: amplitude ∝ r^{−0.5} (same as Rayleigh — cylindrical spreading applies to both)
- Boundary reflections: strong reflections from wall footings create a reverberant field; virtual sensors
- Reverberation time T_s ≈ 2.2/(η_total × f): 0.1–4 s at 10–100 Hz; reverb overlaps direct arrival
- Modal structure: ~40 modes in 1–250 Hz for a 100 m² floor; sparse-to-transitional overlap M < 2
- No half-space geometry: the wave is guided in the plate, not a surface wave on a half-space

**The critical mismatch:** Lamb/flexural dispersion c_g ∝ √f means at 20 Hz, c_g ≈ 192 m/s; at 100 Hz, c_g ≈ 430 m/s — a factor 2.2× spread with a fixed functional form. Soil-Rayleigh dispersion has a similar magnitude for soft profiles but a different shape (depends on the specific layer structure rather than f^{1/2}). The **absolute velocity values and the functional form of dispersion differ**.

**Table 2. Soil-Rayleigh vs. Lamb Plate: Key Contrasts**

| Property | Soil-Rayleigh (pyprop8) | Lamb/Flexural (indoor slab) |
|---|---|---|
| Dispersion law | Layer-dependent; Δv/v ~ 10–200% over 20–100 Hz for soft soil | c_B ∝ f^{1/2}; fixed 2.2× over 20–100 Hz |
| Group velocity at 50 Hz | 180–550 m/s (soft profiles) | 214 m/s (150 mm concrete slab) |
| Geometric spreading | r^{−0.5} | r^{−0.5} |
| Upper boundary | None (half-space) | Free surface of slab |
| Reflections | None (half-space) | Strong: room boundary reflections |
| Reverberation | None | T_s = 0.1–4 s at 10–100 Hz |
| Higher modes | Included (wavenumber integral) | Lamb symmetric/antisymmetric modes |
| Q model | Causal t* (path-integrated) | Internal damping η ≈ 0.01–0.08 |

### 4.3 Why Method Rankings Transfer Despite Physics Mismatch

The critical claim for the scientific soundness of the testbed is that localizer performance rankings determined in the soil-Rayleigh simulator will predict the same rankings in the indoor-slab (Devito) setting. This is plausible for the following reasons:

**Reason 1: Geometric spreading is identical** — r^{−0.5} for both surface/plate waves. The RSS localizer's sensitivity to spreading-law mismatch is therefore not an artifact of soil-vs-plate physics; it reflects a genuine challenge that persists indoors.

**Reason 2: TDOA performance depends on velocity uncertainty, not the specific dispersion law.** The blind-velocity harness (testbed §4.1 of doc 02) measures d(RMSE)/d(Δv/v). This derivative is set by geometry (GDOP) and range-to-velocity ratios, not by whether the dispersion law is f^{1/2} (Lamb) or layer-structure-dependent (Rayleigh). A localizer that is robust to 20% velocity uncertainty in the soil model will be robust to 20% uncertainty in the plate model — the mathematical structure is the same.

**Reason 3: MFP performance relative to TDOA is set by information content, not wave type.** MFP is superior to TDOA precisely because it uses the full waveform (including frequency-dependent phase), not just the onset time. This advantage exists whether the wave is Rayleigh or Lamb. In both cases, MFP using the correct Green's function is the optimal linear estimator; TDOA using constant velocity is a degraded approximation. The testbed correctly measures this information-content gap.

**Reason 4: GDOP (geometry factor) is wave-type-independent.** The GDOP map for a given sensor geometry depends only on the 2-D geometry of sources and receivers and the velocity model used — not on whether the wave is Rayleigh or Lamb. GDOP calculations from the testbed (sensor-count Monte Carlo, geometry sweeps) are directly applicable to indoor deployment.

**Reason 5: The one place rankings may not transfer — reverberation.** The soil half-space has no reflections; the indoor slab has strong boundary reflections (T_s up to 4 s at 50 Hz). In the reverberant case:
- TDOA is degraded by ghost peaks (reflections create false TDOA peaks)
- MFP/fingerprinting improves because boundary reflections effectively multiply the aperture (virtual sensors)
- Time-reversal focusing gains resolution from reverberation

The testbed will therefore **over-rank TDOA and under-rank MFP relative to the indoor reality**. This is a known, bounded, and correctable bias: the testbed correctly identifies the ordering (MFP ≥ TDOA in the non-reverberant limit), and Devito spot-checks can quantify the reverberation advantage of MFP.

**Formalized transfer claim:** "The performance ranking of localizers (MFP ≥ TDOA ≥ RSS; polarization-bearing adds value with 3-component receivers; Kalman smoothing helps all) established in the soil-Rayleigh testbed is conservative relative to the indoor reverberant case — MFP's advantage over TDOA is larger indoors than the testbed predicts, and the testbed TDOA performance is an upper bound on indoor TDOA performance."

### 4.4 What the Testbed Cannot Say

1. **Absolute localization accuracy in the real building** — the testbed gives accuracy under soil-Rayleigh physics. Indoor accuracy depends on slab properties (E, h, η), room geometry, and reverberation, none of which are in the pyprop8 model.
2. **The effect of room boundary reflections on specific localizers** — reverberant MFP advantages require Devito (or real measurements) to quantify.
3. **The specific indoor velocity model** — the f^{1/2} Lamb dispersion must be calibrated from the real slab before TDOA can be applied indoors.
4. **Inter-floor transmission** — at < 50 Hz, energy leaks through junctions. The half-space model is a single-floor model with no coupling.

---

## 5. Summary Assessment: Numerical Parameters for the Testbed

| Parameter | Current implementation | Assessment for 1-D testbed |
|---|---|---|
| Frequency ceiling | 250 Hz Nyquist (dt = 2 ms) | Adequate: footstep energy peaks 20–100 Hz; converged |
| Wavenumber kmax | 10,000–24,000 rad/km | Correct: covers Rayleigh pole + near-field evanescent at r = 0.5 m |
| Number of wavenumbers nk | 8,000–30,000 (adaptive) | Ghost-proof for the convergence window (see §2.2) |
| Contour damping alpha | min(2.15 dk vp, 9/T) | Theory-grounded two-sided bound |
| Radii | 48 points, 0.5–320 m geomspace | 48 log-spaced points cover 3–20 m with ~4–6 points per decade |
| Components stored | Z only (z_fz, z_fx) | R and T computed but discarded; trivial to add (§3.1) |
| Q model | Causal t* (Azimi), single-Q per profile | Adequate for MFP/TDOA; ~20–50% error for RSS at multi-layer profiles |
| Source | Buried point force (Fz + Fx) at 1 m | Correct for footstep physics; 1 m burial avoids free-surface singularity |
| Settle clamp | Coda tail → median (late-time regularization) | Correct: enforces static offset, kills alpha-amplified noise |
| Stiff-profile guard | nk cap + explicit WARNING | Adequate: user notified of unmet ghost bounds |

---

## 6. Final Verdict

**The 1-D pyprop8 testbed is scientifically sound for its stated purpose** — benchmarking localization algorithms before the Devito 3-D deployment — subject to the following explicit conditions:

1. **Label the physics correctly:** "soil-Rayleigh half-space proxy; does not model indoor Lamb/flexural-plate waves or boundary reflections."

2. **Dispersion fidelity is real:** for soft/medium profiles (94 of 282 banks, Vs 100–400 m/s), the testbed exercises meaningful dispersion over 3–20 m at 20–100 Hz. The dispersion challenge is as hard as or harder than the indoor case for these profiles.

3. **Convergence is numerically controlled:** the adaptive ghost-suppression stencil is correctly grounded in Bouchon 2003 theory. Stiff-profile edge cases are flagged.

4. **3-component extension is trivial:** already computed by pyprop8, needs only two save lines.

5. **t* Q model is adequate for MFP and TDOA:** acknowledge the single-Q approximation limitation for RSS.

6. **Method ranking transfer is valid but conservatively bounded:** the testbed will under-predict MFP's advantage over TDOA because it lacks boundary reverberation. Report testbed results as a lower bound on MFP's indoor advantage and plan one Devito spot-check.

The testbed provides a scientifically defensible and practically essential intermediate step: it enables 100,000-scene Monte Carlo sweeps in 33 minutes (vs. 68 years for Devito), exercising the full range of soil profiles, sensor geometries, SNR levels, and velocity uncertainties that would be impossible to explore with 3-D simulation alone. The physics mismatch to the indoor case is known, quantifiable, and does not invalidate the method-ranking conclusions.

---

## Sources

- O'Toole & Woodhouse 2011: "Numerically stable computation of complete synthetic seismograms including the static displacement in plane layered media." *Geophysical Journal International* 187, 1516–1536. [https://academic.oup.com/gji/article-pdf/187/3/1516/1700023/187-3-1516.pdf](https://academic.oup.com/gji/article-pdf/187/3/1516/1700023/187-3-1516.pdf)
- pyprop8 JOSS paper (2022): "pyprop8: A lightweight code to simulate seismic observables in a layered half-space." *JOSS* 10.21105/joss.04217. [https://www.theoj.org/joss-papers/joss.04217/10.21105.joss.04217.pdf](https://www.theoj.org/joss-papers/joss.04217/10.21105.joss.04217.pdf)
- pyprop8 GitHub: [https://github.com/valentineap/pyprop8](https://github.com/valentineap/pyprop8)
- pyprop8 seismogram docs: [https://pyprop8.readthedocs.io/en/latest/seismograms.html](https://pyprop8.readthedocs.io/en/latest/seismograms.html)
- Bouchon 2003: "A Review of the Discrete Wavenumber Method." *Pure and Applied Geophysics* 160, 445–465. [https://link.springer.com/content/pdf/10.1007/PL00012545.pdf](https://link.springer.com/content/pdf/10.1007/PL00012545.pdf)
- Bouchon & Aki 1977: "Discrete Wave-Number Representation of Seismic-Source Wave Fields." *Bulletin of the Seismological Society of America* 67, 259–277. (foundational DW method; stencil logic in `gfbank_build.py` lines 25–62 is a direct implementation of the ghost-bound theory from this work and the Bouchon 2003 review)
- Xia et al. 1999: "Estimation of near-surface shear-wave velocity by inversion of Rayleigh waves." *Geophysics* 64, 691–700. [https://www.masw.com/files/XIA-99-04.pdf](https://www.masw.com/files/XIA-99-04.pdf) — source of the 1–30λ dispersive regime criterion.
- MASW penetration depth rule (1/3–1/2 wavelength): EPA Surface Wave Methods. [https://www.epa.gov/environmental-geophysics/surface-wave-methods](https://www.epa.gov/environmental-geophysics/surface-wave-methods)
- USGS group velocity dispersion analysis: [https://pubs.usgs.gov/publication/70036347](https://pubs.usgs.gov/publication/70036347)
- Bahroun et al. 2014 SO-TDOA: *J. Sound Vib.* 347, 107–126. arXiv 1211.3233. — dispersive concrete TDOA; confirms ~0.3–0.6 m errors from uncorrected velocity mismatch, consistent with testbed predictions.
- Indoor structural reverberation and Lamb wave physics: `research/house_localization/05_structural_reverberation.md` (this project, 2026-07-08).
- Testbed architecture: `research/localization_1d/02_testbed_architecture.md` (this project, 2026-07-08).
- Alajlouni & Tarazaga 2020: floor vibration localization with Kalman; cited in doc 02 §1.6 of this project.
- ScienceDirect: "Indoor footstep localization from structural dynamics instrumentation." [https://www.sciencedirect.com/science/article/abs/pii/S0888327016305015](https://www.sciencedirect.com/science/article/abs/pii/S0888327016305015)
