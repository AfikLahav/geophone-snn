# Ground-Motion Sensitivity and the Range Limit of Seismic Footstep Detection

**Scope.** This report examines the physics and technology that set how far a buried 4.5 Hz vertical geophone can detect a walking human, for a seismic perimeter / intrusion-detection system. It covers (1) why distant footsteps are hard — geometric spreading, intrinsic attenuation (Q), scattering, and the ambient seismic noise floor; (2) the sensitivity of a standard coil geophone versus alternative sensors up to LIGO-class instruments; (3) real footstep detection ranges and the source strength of a footstep; and (4) the decisive question — whether the binding limit is **sensor self-noise** (a better sensor extends range) or the **ambient ground-noise floor** (a better sensor does *not*, and processing / arrays / siting matter more).

**The short answer (stated up front, honestly).** For a single geophone at a typical outdoor security site, the binding limit is the **ambient ground-noise floor**, not the sensor's self-noise. A standard 4.5 Hz coil geophone already resolves ground motion ~1–2 orders of magnitude below the quietest realistic outdoor noise floor in the footstep band (≈10–60 Hz). A "better" sensor (broadband seismometer, MET, optical, MEMS) buys little or no additional footstep range, because both the signal and the noise it would newly resolve are buried under cultural/wind/ambient ground noise. Range is instead extended by: lower-noise siting (away from roads, machinery, wind-coupled vegetation; burial/good coupling), array gain and beamforming, and signal processing (matched filtering, kurtosis/impulse detection, ML classifiers). The LIGO analogy fails for a specific, quantifiable reason developed in Section 5.

---

## 1. Why distant footsteps are hard

A footstep injects energy into the ground at one point. By the time that energy reaches a sensor at distance R, three loss mechanisms plus an irreducible noise background have degraded it.

### 1.1 Geometric (geometrical) spreading
Energy spreads over an expanding wavefront, so amplitude falls with distance even with zero dissipation.

- **Body waves (P, S)** spread over a hemispherical front: amplitude ∝ **1/R** (energy ∝ 1/R²).
- **Surface (Rayleigh) waves** spread over an expanding cylinder: amplitude ∝ **1/√R** (energy ∝ 1/R).

This is why footstep detection lives on the **Rayleigh wave**: it spreads more slowly with distance and, for a near-surface point source on the ground, carries the majority of the radiated energy. A commonly cited partition for a surface source is roughly **P ≈ 7%, S ≈ 26%, Rayleigh ≈ 67%** of the radiated energy (Miller & Pursey, widely reproduced; see Rayleigh-wave references below). The slower 1/√R geometric decay of Rayleigh waves is the single biggest reason any range at all is achievable.

### 1.2 Intrinsic attenuation (anelastic damping, the quality factor Q)
Real soil is not perfectly elastic; it converts wave energy to heat. Amplitude decays as

  A(R) = A₀ · (geometric term) · exp(−α R),  with α = π f / (Q · v_R),

where f is frequency, v_R the Rayleigh phase velocity, and Q the (dimensionless) quality factor. Critically, **α grows linearly with frequency** — high frequencies are killed first. Near-surface soils are lossy: shallow Q_s is often only ~**10–50** (a review of near-surface Q_s methods, Springer J. Seismology 2021, reports values in this range; soft/unconsolidated sites can be lower).

Worked numbers (soft soil, v_R ≈ 150 m/s, Q ≈ 20):
- At **20 Hz**: α = π·20/(20·150) ≈ 0.021 m⁻¹ → e-folding distance ≈ 48 m; at 50 m the anelastic factor alone is ~e⁻¹ ≈ 0.35.
- At **50 Hz**: α ≈ 0.052 m⁻¹ → e-folding ≈ 19 m; at 50 m the factor is ~e⁻²·⁶ ≈ 0.07.
- At **80 Hz**: α ≈ 0.084 m⁻¹ → e-folding ≈ 12 m.

So the **high-frequency content of a footstep is gone within a few tens of meters**, which is exactly why the optimal detection band shifts down to ~10–40 Hz at range, and why the literature reports the maximum-distance band as **10–40 Hz / 10–20 Hz** for normal walking (Sabatier & Ekimov; see Section 4). Detection at range is a low-pass-filtered version of the near-field footstep.

### 1.3 Scattering
The shallow subsurface is heterogeneous (rocks, roots, voids, layering, water-table contrasts). Scattering redistributes coherent footstep energy into incoherent coda, both reducing peak amplitude and smearing the impulsive signature that detectors rely on. Combined with anelastic loss this is often lumped into an **effective Q** that is lower than the intrinsic Q. Scattering also makes amplitude-vs-distance noisy and site-specific — the footstep literature repeatedly emphasizes that the seismic response is **"site-specific"** (Sabatier & Ekimov).

### 1.4 The ambient seismic noise floor — the real ceiling
Even a perfect, noiseless sensor sits in a ground that is **always moving**. This background sets the floor the footstep signal must exceed.

**The Peterson New Low/High Noise Models (NLNM/NHNM, USGS OFR 93-322).** Peterson (1993) compiled global station data into a New Low Noise Model (NLNM) and New High Noise Model (NHNM) bounding observed background noise, expressed in **dB relative to (1 m/s²)²/Hz** (acceleration PSD). The NLNM is the quietest ground on Earth — matched by only ~0.1% of Global Seismographic Network spectra — so it is a *floor*, not a typical value. The classic Peterson models stop at **10 Hz**; high-frequency baselines to 100 Hz were later established from millions of PSDs (USGS / IRIS MUSTANG; Pub 70208091, doi:10.1785/0120190123).

**Why this dominates at footstep frequencies.** Above ~1 Hz, the noise is overwhelmingly **cultural / anthropogenic** (traffic, machinery, footfall of others, HVAC, wind coupling through trees and structures) and shows strong day/night and weekday/weekend cycles. Measured urban ground-noise velocity amplitudes in the ~1–5 Hz band are about **100–240 nm/s during the working day and 50–90 nm/s at night** (D'Alessandro et al., spectral characterization of background noise in Italy, AGU Earth & Space Science 2021). Quiet rural sites are lower, but seldom by more than ~1–2 orders of magnitude, and wind alone can dominate at exposed sites. **This 10–240 nm/s ambient ground velocity is the number that actually limits footstep range** — and it is *physically present in the ground*, so no sensor improvement removes it.

---

## 2. Sensitivity of a standard 4.5 Hz coil geophone

A geophone is a velocity transducer: a coil moves through a magnet's field, generating V = G · ẋ above the corner frequency.

### 2.1 Specifications (Geospace GS-11D class, the typical 4.5 Hz unit)
- **Natural frequency:** 4.5 Hz (response is flat in velocity *above* ~4.5 Hz; falls off 12 dB/oct below).
- **Transduction sensitivity:** ~**32 V/(m/s)** per element for the GS-11D as commonly wired; the EarthScope/Geospace catalog and IRIS NRL list the high-output GS-11D variant at **~100 V/(m/s)** (coil/wiring dependent). The SM-24 (10 Hz) is **28.8 V/(m/s)**. (EarthScope EPIC; IRIS NRL; Geometrics response curve.)
- **Damping:** ~0.7 of critical (with shunt resistor).
- **Coil resistance:** typically ~hundreds to ~few kΩ depending on variant (sets Johnson noise; see below).

### 2.2 Self-noise — the geophone is already very quiet at footstep frequencies
Geophone self-noise is the incoherent sum of: (a) **suspension (Brownian/thermal-mechanical) noise** of the moving mass, (b) **Johnson (thermal) noise of the coil resistance**, and (c) **preamplifier** voltage/current noise. Two anchor measurements:

- A careful **huddle test of the Sercel L-4C** (a 1 Hz coil geophone) with low-noise OPA188 amplifiers reached a measured self-noise of **~10⁻¹¹ m/√Hz (displacement) at 1 Hz**, i.e. near the **Johnson-noise limit** over 0.01–100 Hz (arXiv:1711.05439). In velocity at 1 Hz that is ~6×10⁻¹¹ (m/s)/√Hz, and self-noise (in acceleration) *improves* toward higher frequency for a velocity sensor.
- Conventional **moving-coil and MEMS geophones exhibit acceleration self-noise of ~10 ng/√Hz** (≈ **1.0×10⁻⁷ m/s²/√Hz**) in the seismic band, per the optomechanical-MEMS comparison study (Nature Microsystems & Nanoengineering 2024, PMC11589752).

**Convert the sensor floor to ground velocity in the footstep band for comparison with ambient noise.** Take ~10 ng/√Hz = 1×10⁻⁷ m/s²/√Hz at 30 Hz. As velocity that is (1×10⁻⁷)/(2π·30) ≈ **5×10⁻¹⁰ (m/s)/√Hz**. Integrated over a ~30 Hz footstep band (√30 ≈ 5.5) gives a sensor-noise-equivalent ground velocity of roughly **~3 nm/s**.

Compare:
- **Sensor self-noise equivalent:** ~3 nm/s (band-integrated).
- **Ambient ground noise (quiet night):** ~50–90 nm/s.
- **Ambient ground noise (working-day urban):** ~100–240 nm/s.

**The ambient ground is 15–80× noisier than the geophone's own floor.** The geophone is *not* the bottleneck. (The corner frequency at 4.5 Hz does cut sensitivity below ~4–5 Hz, but footstep energy at range is concentrated at 10–40 Hz, above the corner, so this matters little.)

---

## 3. Alternative sensors — do they help?

| Sensor class | Self-noise / sensitivity (footstep band) | Helps footstep range? |
|---|---|---|
| **4.5 Hz coil geophone** | ~10 ng/√Hz ≈ 1×10⁻⁷ m/s²/√Hz; ~32–100 V/(m/s) | Baseline. Already below ambient. |
| **Broadband seismometer** (STS-2, Trillium 240) | Self-noise *below NLNM* from ~100 s to ~10 Hz; STS-2 reaches ≈ −178 dB rel (m/s²)²/Hz at long period (RG 258611321; EarthScope T240) | **No.** Its advantage is at long periods (<1 Hz); at 10–50 Hz a geophone is already ambient-limited. Adds cost/fragility, not footstep range. |
| **MEMS accelerometer** (consumer/industrial) | Often 100–1000× *worse* than geophone (~µg–mg/√Hz) | Worse; only high-end optomechanical MEMS competes. |
| **Optomechanical MEMS geophone** | **2.5 ng/√Hz (100–200 Hz)** ≈ 2.5×10⁻⁸ m/s²/√Hz; ~4× better than coil (Nature 2024, PMC11589752) | **Marginal.** Lower self-noise, but still far above the ambient floor — paper itself notes the limit is its own *thermal* noise, not ambient, only because it was bench-tested. In the field, ambient dominates. |
| **Molecular-electronic transducer (MET)** | Matches STS-2 / Trillium-240 self-noise to ~60 s period; SOI designs ~1.78×10⁻⁷ (m/s²)/√Hz at 1.2 Hz (PMC3673101; IEEE 7421798) | **No** for range; useful where ruggedness + broadband matter. Ambient-limited at footstep f. |
| **Optical / interferometric geophone** | fm/√Hz displacement readout; e.g. interferometric readout of a commercial geophone (arXiv:2109.03147) | **No** for range — improves *readout* noise, but the geophone is already ambient-limited, not readout-limited, at 10–50 Hz. |
| **Fiber-optic DAS** | ~tens of pε/√Hz to **~8 pε** noise floor; sub-nε (Sentek picoDAS; Wikipedia DAS) | **Different value proposition.** A single DAS point is *less* sensitive than a good geophone, but DAS turns a whole fiber into a **dense line array** (every ~1–10 m a channel) → spatial coverage, localization, tracking along perimeters. Range gain comes from *array processing*, not per-point sensitivity. Documented footstep localization at ~10 m from the fiber (Nature Comms 2022; PMC7713295). |
| **Atom-interferometry / quantum gravimeter** (LIGO-class gravity) | ~**10⁻⁸ m/s² (≈1 µGal)**, best ~4.3×10⁻⁹ m/s²/√Hz (Exail AQG; arXiv reviews) | **No — wrong physics** (Section 5). Senses Newtonian gravity of mass, which falls as 1/R² (field) and 1/R³–1/R⁴ (usable gradient), far steeper than seismic waves. |

**Takeaway:** Every "better" point sensor improves a quantity (self-noise, readout noise, bandwidth) that is **not** the binding constraint for outdoor footstep detection. The only architectural changes that move the range needle are **arrays** (DAS, geophone arrays → beamforming/coherent gain) and **siting/coupling** (lower ambient).

---

## 4. Real footstep detection ranges and footstep source strength

### 4.1 Source strength of a footstep
- **Ground reaction force:** a walking person delivers a peak vertical force of ~**1.0–1.5× body weight** (≈ 700–1100 N for a 70 kg adult); running raises this to **2–3× body weight** (Nilsson & Thorstensson, Acta Physiol Scand 1989). Only a *small fraction* of this quasi-static force couples into propagating seismic waves; most is reacted statically by the ground.
- **Seismic radiation:** the footstep impulse (heel strike, ~tens of ms) launches mostly Rayleigh waves; characteristic outdoor **frequency band 20–90 Hz near-field**, collapsing to **10–40 Hz at range** after high-f anelastic loss (Ekimov & Sabatier, *Vibration and sound signatures of human footsteps in buildings*, JASA 120(2):762, 2006; and SPIE 2006/2008).
- **Order-of-magnitude energy:** a footstep's *mechanical* impulse is ~hundreds of N·s, but the fraction radiated as seismic energy is small (commonly estimated at <<1 J of seismic energy per step; the bulk is absorbed/static). This tiny radiated energy, spread over expanding wavefronts and attenuated, is why ranges are short.

### 4.2 Detection ranges from the literature
- **Typical (single geophone, real outdoor security conditions):** **~5–20 m** for reliable single-sensor footstep detection. Stealthy/soft walking can be undetectable **even at 1 m** (Sabatier & Ekimov, *Range limitation for seismic footstep detection*, SPIE 6963, 69630V, 2008).
- **Favorable (quiet site, normal/heavy walking, good coupling, optimal band):** **~40–70 m** (multiple studies report bearing/tracking out to 10–70 m with arrays; field maxima quoted ~**48–94 m** under good conditions).
- **Exceptional / heavier sources:** elephants detected at **~155 m** (and ~15–40 m routine) — illustrating that *source strength* moves range far more than sensor choice (arXiv:2406.05140). A jumping person has been reported to ~**48.5 m**.
- **Indoor / hard-floor short range:** ~**2.5 m** typical for a casually placed sensor.

**Two limiting factors are stated explicitly in the primary literature** (Sabatier & Ekimov): **(1) walking style** (soft/stealthy gait radiates far less seismic energy — a *source* limit) and **(2) the background noise floor** (higher in urban than quiet areas — a *site/ambient* limit). Notably, **neither is sensor self-noise.**

---

## 5. The LIGO analogy — why "LIGO-class" sensitivity does NOT extend footstep range

The owner's intuition: LIGO measures displacements of ~**10⁻¹⁹ m** (strain ~10⁻²¹), so surely such sensitivity could detect a footstep from kilometers away. It cannot, for three independent reasons.

**(a) LIGO measures a different thing (strain between isolated masses), and only above ~10 Hz, *after* ferocious seismic isolation.** At a LIGO site, ground motion at 10 Hz is ~**10 orders of magnitude larger** than the gravitational-wave signal. LIGO does **not** out-sense the ground — it **isolates** from it: active platforms reduce motion to ~2×10⁻¹³ m, then multi-stage pendulum suspensions reduce it ~10⁶× further to reach 10⁻¹⁹ m (LIGO Caltech, vibration isolation). A footstep detector wants to *measure* ground motion; LIGO's entire achievement is *rejecting* ground motion. The 10⁻¹⁹ m figure is a *differential, isolated, high-frequency* number that does not describe sensitivity to absolute ground motion at all.

**(b) If you instead tried to detect a footstep *gravitationally* (the genuine "LIGO physics" — Newtonian attraction of the walker's mass), the signal falls off catastrophically with distance.** Thorne & Winstein, *Human Gravity-Gradient Noise in Interferometric Gravitational-Wave Detectors* (Phys. Rev. D 60, 082001, 1999; gr-qc/9810016) computed exactly this. The strongest gravity signal from a walking person is the jerk of weight-transfer (~20 ms timescale, ~2×/s), producing strain noise in LIGO of

  √S_h(f) ≈ 0.6×10⁻²³ Hz⁻¹ᐟ² · (f/10 Hz)⁻⁶ · [ Σᵢ (rᵢ/10 m)⁻⁶ ]^{1/2}.

The key feature is the **(r/10 m)⁻⁶ scaling**: the detectable gravity-gradient effect of a walking human falls off as **1/r⁶** (and as f⁻⁶ in frequency). Their operational conclusion: a person merely needs to stay **~10 m away** for their gravitational pull to be negligible even to LIGO. A footstep's gravity is undetectable beyond ~10 m **for the most sensitive instrument ever built** — so gravity-based ranging is hopeless. (Newtonian *field* ∝ 1/r²; the usable *gradient/jerk* signature falls even faster.)

**(c) Quantum gravimeters confirm the same ceiling.** Atom-interferometry gravimeters reach ~**1 µGal = 10⁻⁸ m/s²** (best ~4×10⁻⁹ m/s²/√Hz). A 70 kg person at 10 m exerts gravitational acceleration g = GM/r² = (6.67×10⁻¹¹·70)/(10²) ≈ **5×10⁻¹¹ m/s²** — already ~**100–1000× below** the best gravimeter's noise floor, and it drops as 1/r². So even the best gravity sensor can't feel a person past a few meters. "Newtonian noise" from walking humans is a real nuisance *only* for instruments where the source is meters away.

**Conclusion of the analogy:** LIGO's headline sensitivity is irrelevant to footstep ranging. Seismic (elastic-wave) detection wins by many orders of magnitude over gravitational detection precisely because elastic waves decay as 1/√R (Rayleigh) rather than 1/r²–1/r⁶, and carry far more of the footstep's energy.

---

## 6. The key question, answered honestly

**Is range limited by sensor self-noise or by the ambient ground-noise floor?**

**By the ambient ground-noise floor — decisively, for a single outdoor geophone.** The numbers:

- Geophone self-noise (band-integrated, 10–40 Hz): **~3 nm/s** equivalent ground velocity.
- Ambient ground velocity at footstep frequencies: **~50–90 nm/s (quiet night)** to **~100–240 nm/s (urban day)**.
- The ground itself is **~15–80× above** the sensor floor. The footstep signal must beat the *ambient* number, not the sensor number.

**Implications for the project:**

1. **A more sensitive sensor will NOT meaningfully extend range** at a typical site. The geophone already resolves motion well below ambient; replacing it with a broadband, MET, optical, or premium-MEMS sensor changes the self-noise (already irrelevant) but not the ambient floor. (Exception: a genuinely *self-noise-limited* deployment — e.g. a deep, vault-quality, wind-shielded, traffic-free borehole — where a lower-self-noise sensor could help. This is rare for perimeter security.)

2. **What *does* extend range:**
   - **Lower-noise siting & coupling:** distance from roads/machinery, burial with good ground contact, wind shielding, choosing firm ground (also raises v_R and Q, slightly reducing attenuation). This directly lowers the binding ambient term — often the single biggest lever.
   - **Arrays + beamforming:** N coherent sensors give up to ~√N (incoherent-noise) to ~N (coherent processing) SNR gain, plus localization/bearing and rejection of spatially-incoherent noise. This is why DAS (a dense line array) and multi-geophone nodes outperform single sensors for perimeters.
   - **Signal processing:** matched filtering to the footstep impulse, **kurtosis / higher-order-statistic** impulse detectors (robust in noise), spectral-band selection (push to 10–40 Hz at range), and ML classifiers (your SNN). These exploit the *structure* of footsteps (periodic, impulsive, characteristic spectrum) to beat a higher noise floor than a simple threshold could.
   - **Source-dependent reality:** stealthy gait and soft ground can defeat detection at any range — a *source/site* limit no sensor fixes.

3. **For an article (publication rigor):** frame the range budget as
   SNR(R) = [ A₀ · (R₀/R)^{1/2} · e^{−π f (R−R₀)/(Q v_R)} ] / N_ambient(f, site),
   and be explicit that the denominator N_ambient (not sensor self-noise) sets the threshold. Report ranges with the **gait, band, site-class, and ambient level** stated — these, not the sensor, explain the spread from ~2.5 m to ~94 m in the literature. Avoid quoting a single "detection range" without these confounds (this is exactly the kind of selected/uncontrolled number a reviewer will target).

---

## Sources

- Peterson, J. (1993), *Observations and Modeling of Seismic Background Noise* (NLNM/NHNM), USGS OFR 93-322: https://pubs.usgs.gov/of/1993/0322/report.pdf
- USGS/IRIS high-frequency noise baselines to 100 Hz (doi:10.1785/0120190123): https://pubs.usgs.gov/publication/70208091
- USGS PQLX / PSD-PDF noise analysis (OFR 2005-1438): https://pubs.usgs.gov/of/2005/1438/pdf/OFR-1438.pdf
- D'Alessandro et al. (2021), spectral characterization of background seismic noise in Italy (urban day/night 100–240 / 50–90 nm/s), AGU Earth & Space Science: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EA001579
- Near-surface Q_s estimation review, J. Seismology (2021): https://link.springer.com/article/10.1007/s10950-021-10066-5
- Rayleigh-wave attenuation (microseism band), Nature Sci. Reports: https://www.nature.com/articles/s41598-021-89497-6
- Geophone self-noise huddle test (L-4C, ~10⁻¹¹ m/√Hz at 1 Hz, Johnson-limited), arXiv:1711.05439: https://arxiv.org/abs/1711.05439
- LIGO L-22D/L-4C geophone sensor-noise technical note (DCC T1600438): https://dcc.ligo.org/public/0138/T1600438/001/L-4C%20huddle%20test%20at%20the%20AEI.pdf
- Optomechanical MEMS geophone, 2.5 ng/√Hz; conventional coil/MEMS ~10 ng/√Hz, Nature Microsystems & Nanoengineering (2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11589752/
- GS-11D 4.5 Hz geophone specs (EarthScope EPIC): https://epic.earthscope.org/content/45hz-high-frequency-single-component-sensor — and IRIS NRL: https://ds.iris.edu/NRL/sensors/oyo_geospace/oyo_geospace_gs11d_sensors.htm — and Geometrics response curve: https://www.geometrics.com/wp-content/uploads/2021/05/4.5-Hz-Response-Curve_13-105-030D.pdf
- Molecular electronic transducer (MET) review, PMC3673101: https://pmc.ncbi.nlm.nih.gov/articles/PMC3673101/ ; MET planetary seismometer (IEEE 7421798): https://ieeexplore.ieee.org/document/7421798/
- Interferometric (optical) readout of a commercial geophone, arXiv:2109.03147: https://arxiv.org/pdf/2109.03147
- Broadband seismometer self-noise vs NLNM (STS-2 PDF self-noise), RG 258611321: https://www.researchgate.net/publication/258611321 ; Trillium 240 (EarthScope): https://epic.earthscope.org/content/instrumentation/sensors/broadband-sensors/t240-bb-sensor
- DAS overview and footstep/human locomotion ID: Nature Communications (2022) https://www.nature.com/articles/s41467-022-31681-x ; PMC7713295 (Rayleigh-enhanced DAS, human locomotion DNN): https://pmc.ncbi.nlm.nih.gov/articles/PMC7713295/ ; DAS (Wikipedia, noise floors): https://en.wikipedia.org/wiki/Distributed_acoustic_sensing ; Sentek picoDAS datasheet (8 pε floor): https://www.sentekinstrument.com/wp-content/uploads/2025/05/picoDAS-Datasheet.pdf
- Sabatier & Ekimov, *Range limitation for seismic footstep detection*, SPIE 6963, 69630V (2008): https://www.spiedigitallibrary.org/conference-proceedings-of-spie/6963/69630V/Range-limitation-for-seismic-footstep-detection/10.1117/12.785235.short
- Ekimov & Sabatier, *Vibration and sound signatures of human footsteps in buildings*, JASA 120(2):762 (2006): https://pubs.aip.org/asa/jasa/article-abstract/120/2/762/893348
- Ekimov & Sabatier, *Broad frequency acoustic response of ground/floor to human footsteps*, SPIE 6241 (2006): https://www.spiedigitallibrary.org/conference-proceedings-of-spie/6241/62410L/10.1117/12.663978.short
- Towards long-range detection of elephants using seismic signals (range to ~155 m), arXiv:2406.05140: https://arxiv.org/abs/2406.05140
- Nilsson & Thorstensson, *Ground reaction forces at different speeds of human walking and running*, Acta Physiol Scand (1989): https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1748-1716.1989.tb08655.x
- Thorne & Winstein, *Human Gravity-Gradient Noise in Interferometric Gravitational-Wave Detectors*, Phys. Rev. D 60, 082001 (1999) — the 1/r⁶ human-gravity scaling, gr-qc/9810016: https://arxiv.org/abs/gr-qc/9810016
- LIGO seismic isolation (ground motion ~10¹⁰× the signal; 10⁻¹³ → 10⁻¹⁹ m), Caltech: https://www.ligo.caltech.edu/page/vibration-isolation
- Atom-interferometry / quantum gravimeter sensitivity (~µGal, 4×10⁻⁹ m/s²): Exail AQG https://www.exail.com/product/quantum-gravimeters ; gravity cartography, Nature (2021): https://www.nature.com/articles/s41586-021-04315-3
