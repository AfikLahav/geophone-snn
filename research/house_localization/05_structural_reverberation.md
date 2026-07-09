> **Research survey — house-localization pivot ("wallhacks").** Background research agent (Sonnet), 2026-07-08. Literature survey; design input, not a validated experiment. Cross-references [02_indoor_localization_methods.md](02_indoor_localization_methods.md). Indexed in [README.md](README.md).

---

# Structure-Borne Reverberation and Multipath in Residential Concrete/Masonry Buildings: Localization Physics

## 1. Physics of Structure-Borne Reverberation

### 1.1 Flexural (bending) waves dominate

A footstep impact injects a broadband force impulse exciting multiple wave modes. In the 1–250 Hz geophone band the floor behaves as a plate carrying **flexural (bending) waves**. P (~3500–4000 m/s) and S (~1800–2200 m/s) body waves are too fast to give meaningful TDOA spread at room scales and carry far less energy from a surface footstep. The dominant energy carrier for localization is the flexural wave.

Flexural phase velocity in a thin plate: **c_B(f) = (2πf)^(1/2) × (D/ρh)^(1/4)**, D = Eh³/[12(1−ν²)]. **c_B ∝ √f — strongly dispersive; group velocity = 2 × phase velocity** in the thin-plate limit.

### 1.2 Dispersion curves for a 150 mm concrete slab (E=30 GPa, ρ=2400, ν=0.2)

D = 8.79×10⁶ N·m, ρh = 360 kg/m², (D/ρh)^(1/4) ≈ 12.5 m·s^(−1/2)

| f (Hz) | Phase c_B (m/s) | Group c_g = 2c_B (m/s) |
|---|---|---|
| 5 | 48 | 96 |
| 10 | 68 | 136 |
| 20 | 96 | 192 |
| 50 | 152 | 304 |
| 100 | 215 | 430 |
| 150 | 263 | 526 |
| 200 | 304 | 608 |
| 250 | 340 | 680 |

Thin-plate theory holds when wavelength ≫ thickness; at 250 Hz, λ_B ≈ 1.4 m ≫ 0.15 m → valid across 1–250 Hz. Measured perceived velocity in real buildings: **30–300 m/s** (lower end = heterogeneity). For a **200 mm slab**: D scales as h³ → phase velocity ×1.24 (c_B ≈ 267 m/s @ 100 Hz). Coincidence frequency f_c ≈ **92 Hz for 200 mm, ≈123 Hz for 150 mm** (measurements confirm 100–120 Hz for residential slabs).

### 1.3 Structural transfer function = vibration analog of room impulse response

Each source→sensor pair has **H(x_s, x_r, f)**. Measured signal: **p(t) = h(t) * f_source(t) + noise**, containing direct arrival (dispersed), wall reflections (image sources), mode resonances, and late reverberant field. First wall reflection ≈ **t_1st = 2L / c_g(f)**: for L=5 m at 50 Hz → 33 ms; at 100 Hz → 23 ms; L=10 m at 50 Hz → 66 ms.

### 1.4 Floor modes and modal density

Bending-plate modal density is **frequency-independent**: n(f) = (A/4)√(ρh/D) [modes/Hz]. For 150 mm concrete: 5×5 m room → 0.040 modes/Hz (1 per 25 Hz); 10×10 m → 0.160 (1 per 6 Hz); 20×10 m → 0.320. A 100 m² floor has ~**40 modes in 1–250 Hz**. Fundamental mode of a 5×5 m bay ≈ 8–31 Hz depending on boundary conditions; whole-building fundamentals often 4–8 Hz (code minimum ≥5 Hz vertical).

### 1.5 Structural reverberation time (T_s ≈ T60 analog)

**T_s ≈ 2.2 / (η_total × f)**. Loss factors: concrete/brick internal η ≈ 0.01–0.02; masonry up to 0.06–0.08; coupling loss 0.01–0.05 per coupled plate; radiation small below coincidence.

| f (Hz) | η_total | T_s low η (s) | T_s high η (s) |
|---|---|---|---|
| 10 | 0.01–0.02 | 22 | 11 |
| 50 | 0.02–0.05 | 4.4 | 0.88 |
| 100 | 0.03–0.08 | 2.2 | 0.28 |
| 200 | 0.04–0.10 | 1.1 | 0.11 |
| 250 | 0.05–0.10 | 0.88 | 0.088 |

ISO 10848 measurements: T_s = **0.1–0.3 s at 100–1000 Hz** in concrete frames. At 1–50 Hz coupling losses drop and **T_s ≈ 1–20 s** — the floor barely stops ringing between footsteps. **This is the core reverberation problem.**

### 1.6 Modal overlap factor M = n(f) × η_total × f

For a 100 m² floor, η_total = 0.05: M ≈ 0.008 (10 Hz), 0.40 (50 Hz), 0.80 (100 Hz), 1.60 (200 Hz), 2.00 (250 Hz). The entire band is **sparse-to-transitional (M < 2)** — individual modes dominate; SEA/diffuse-field assumptions break down. Transfer functions are highly structured and location-specific — **a curse for TDOA, a gift for fingerprinting.**

---

## 2. How Reverberation Corrupts Localization

- **Direct-arrival burial.** First wall reflection at ~20–70 ms; with T_s ~1–4 s at 50 Hz, reverberant energy overlaps the direct arrival, driving direct-to-reverberant ratio to ~0 dB within one footstep.
- **TDOA pick ambiguity from dispersion.** A broadband pick estimates an effective velocity averaged over bandwidth. Using a single assumed speed: at D_s=3 m, a broadband pick at 200 m/s vs true 136 m/s (10 Hz) → 7 ms systematic error → **1.4 m position error**. Reported: naive TDOA >1 m; dispersion-mitigated 0.38–0.65 m; 3 accel in 3.6×5.4 m ~50 cm; 11 sensors energy-based 0.99–1.58 m.
- **Timing precision needed.** 10 cm at c_g=300–600 m/s → 167–333 μs → ≥3–6 kHz sampling. GaitVibe+ used 2500 Hz for 10 cm resolution. For 250 Hz upper limit (c_g≈680): ≥1360 Hz for 50 cm, ≥6800 Hz for 10 cm.
- **Structural heterogeneity.** Varying slab thickness/rebar/beams → position-dependent apparent velocity varying 20–50% even at fixed frequency → must calibrate per structure; 20% speed error = 20% localization error at range.
- **Inter-floor / wall coupling.** At <50 Hz, junction transmission can be >50% in energy → the "room" for structural waves is the whole building; ghost sources from wall-transmitted energy.

---

## 3. Mitigation and Exploitation

### 3.1 Dereverberation / early-arrival isolation

- **Bandpass to high end (100–250 Hz):** dispersion makes HF arrive first (c_g ∝ √f) → concentrates direct-path energy earlier relative to reflections.
- **Cepstral / homomorphic:** reverberation appears as spectral ripple; the cepstrum lifts it to quefrency; liftering removes reverberant tails; can estimate first-arrival from spectral-null periodicity.
- **Short-time envelope:** windowing to first 5–30 ms isolates direct + first reflection. A <10 ms window keeps mostly direct-path energy at most spacings, at the cost of spectral resolution.

### 3.2 Matched-Field Processing (MFP): foe → friend

MFP replaces naive TDOA with a full-wave forward model: **P_MFP(x) = |w^H d|²**, w = replica from H (Bartlett), or MVDR/Capon for interferer suppression. **In a bounded reverberant medium the full transfer function (direct + all reflections) is more information-rich than the direct path** — each reflection is a virtual sensor at an image source, multiplying the aperture. Demonstrated on beams/plates (Jacquelin et al.). **For the simulation-trained project:** the 3D elastic-wave simulator computes the exact Green's function at every candidate footstep location → MFP is the *optimal linear estimator* with known multipath; the reverberant field is the full measurement kernel, not a nuisance.

### 3.3 Time-reversal focusing

Record → time-reverse → backpropagate through the known model → energy refocuses at the source. In bounded structures, boundary reflections act as virtual sensors, so a sparse N-geophone array gains a huge effective aperture; more reflections **improve** focus. TR-SBL demonstrated 0.37 m error at 125–250 Hz in a reverberant room (vs 0.50 m standard TR). Architecturally equivalent to cross-correlating observed data with simulated transfer functions across a source grid — i.e., MFP.

### 3.4 Fingerprinting / transfer-function learning

In the sparse modal regime (M<2), two locations >λ_B/4 apart (~0.55 m at 100 Hz) produce substantially uncorrelated transfer functions → location is encoded in a high-dimensional feature space. **A reverberant building is *easier* to fingerprint than free space** (transfer-function spatial diversity). FEEL: 98.6% within 63.5 cm (3 accel). Reservoir computing: 0.86 m RMSE (11 sensors, 16 m corridor), 38% better than energy-based.

### 3.5 Direct-path SNR vs late reverberation

Within one reflection cycle at 100 Hz (η_total=0.05, Δt=30 ms), ~**61% of energy density is already reverberant**. Direct-path isolation is clean only at very short windows (<5 ms) and higher frequencies. Below ~50 Hz the direct path is swamped for any sensor >2 m from the source → **fingerprinting/MFP is the only viable strategy** there.

---

## 4. Frequency-Band Strategy

| Band (Hz) | Primary method | Limiting factor | Accuracy (indicative) |
|---|---|---|---|
| 1–20 | Fingerprinting/MFP | Long T_s, low SNR | Room-level (1–3 m) |
| 20–80 | MFP / matched filter | High reverb, dispersion | 0.5–1.5 m with good model |
| 80–180 | Hybrid TDOA + fingerprint | Dispersion, coincidence | 0.3–0.7 m |
| 180–250 | TDOA (dispersion-corrected) + MFP | Sampling rate, heterogeneity | 0.1–0.4 m |

- **<10 Hz:** whole-building modes, T_s 10–50 s; avoid TDOA; building-level mode fingerprinting only.
- **10–50 Hz:** high footstep energy but T_s 1–10 s → unusable for TDOA without MFP; **best band for MFP** (well-separated, location-specific modes).
- **50–150 Hz:** T_s 0.3–1.5 s, first reflection 20–40 ms → short-window TDOA marginally viable; **recommended primary band for hybrid TDOA + fingerprinting.**
- **150–250 Hz:** T_s 0.1–0.4 s, direct-path window (5–10 ms) extractable; **best for direct-path TDOA with dispersion compensation.**

FEEL used 35.5–200 Hz; GaitVibe+ used 5–2500 Hz for 10 cm resolution.

---

## 5. Concrete Numbers (residential concrete/masonry)

| Parameter | Value |
|---|---|
| Flexural phase speed (150 mm, 50/100 Hz) | 152 / 215 m/s |
| Flexural group speed (150 mm, 100 Hz) | 430 m/s |
| Measured floor wave speed (footstep) | 30–300 m/s |
| Coincidence frequency (150/200 mm) | 123 / 92 Hz |
| Modal density (100 m², 150 mm) | 0.16 modes/Hz |
| Mode count 1–250 Hz (100 m²) | ~40 |
| Fundamental (5×5 m bay) | 8–31 Hz |
| η (concrete / masonry) | 0.01–0.02 / 0.02–0.08 |
| η_total (in-situ, coupled) | 0.03–0.10 |
| T_s @ 100 / 50 / 10 Hz | 0.1–0.3 / 0.5–2 / 2–20 s |
| Modal overlap M @ 100 Hz (100 m²) | 0.4–0.8 |
| First wall reflection (5 m, 100 Hz / 10 m, 50 Hz) | 23 / 66 ms |
| Loc. RMSE (3 sensors, TDOA) | 0.38–0.65 m |
| Loc. RMSE (fingerprinting, 3 sensors) | <0.3 m |
| Loc. RMSE (reservoir, 11 sensors) | 0.86 m |
| Loc. RMSE (energy baseline) | 1.43–2.3 m |
| Time-reversal vs direct TDOA | +26–38% |

---

## 6. Interface with Localization Estimators (Handoff Notes)

1. **Wave-speed model.** Any TDOA estimator must account for dispersive c_g(f) (thin-plate model with measured E,ρ,h, or warped frequency transform). Constant speed → O(1 m) systematic error below 100 Hz.
2. **Reverberant transfer function.** The 3D simulator's Green's function enables **MFP as the primary estimator** (grid-search maximizing simulated-vs-measured transfer-function correlation) — theoretically optimal here.
3. **Band selection.** Generate training data with separate sub-bands as features; a joint multi-band estimator weighting 20–80 Hz (fingerprinting/MFP) and 100–250 Hz (TDOA/warped-FT) beats single-band.
4. **Sparse-modal caution for deep learning.** M<2 → small floor-load changes (furniture, occupants) shift modal frequencies → transfer functions trained on one loading may degrade. **Include training-data variability** (loads, temperature) or use physics-based features less sensitive to modal shifts (energy envelopes, group-velocity-filtered TDOA). *(This is the indoor analog of our ground-corpus domain randomization.)*
5. **Short window for direct-path TDOA** at 150–250 Hz (~5–10 ms around the first arrival), triggered by the geophone network's broadband energy onset.
6. **Boundary reflections as virtual sensors.** A 5×5 m room provides ~10+ virtual sensors per physical geophone from 5 reflection orders — exploit explicitly in the MFP formulation.

---

## Sources

- [Structure-borne sound in buildings (Liverpool)](https://livrepository.liverpool.ac.uk/3077422/1/author%20version.pdf) · [Structural reverberation times (Liverpool)](https://livrepository.liverpool.ac.uk/3109444/1/Structural%20reverberation%20times%20-%20Copy%20for%20Elements.pdf) · [RC structure type on low-freq impact sound (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0003682X18311058)
- [Dispersion of Flexural Waves (Penn State)](https://www.acs.psu.edu/drussell/Demos/Dispersion/Flexural.html) · [SO-TDOA (arXiv:1211.3233)](https://arxiv.org/abs/1211.3233) · [Occupant localization (NSF PAR)](https://par.nsf.gov/servlets/purl/10057760)
- [Ubiquitous Gait Analysis (PMC11053483)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11053483/) · [FEEL (PMC9886238)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9886238/) · [Building-as-Reservoir (arXiv:2603.04610)](https://arxiv.org/html/2603.04610v1) · [Energy vibro-localization (PMC10708826)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10708826/) · [Dispersive propagation (PMC11644851)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644851/)
- [MFP for structural vibration (ResearchGate)](https://www.researchgate.net/publication/243520672) · [Time reversal in plates (PubMed)](https://pubmed.ncbi.nlm.nih.gov/28253676/) · [TR in reverberant environments (PMC11124834)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11124834/) · [Laser TR in stiffened plate (Frontiers)](https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2019.00030/full)
- [Damping in unreinforced stone masonry (ResearchGate)](https://www.researchgate.net/publication/245078328) · [Indoor footstep loc. from structural dynamics (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0888327016305015) · [Warped FT AE localisation (arXiv:2110.06457)](https://arxiv.org/pdf/2110.06457) · [Floor Vibrations (SteelConstruction.info)](https://www.steelconstruction.info/Floor_vibrations) · [Min slab thickness vs floor vibration (I-ASEM)](http://www.i-asem.org/publication_conf/asem13/178.M4I.3.SM129_543F.pdf)
