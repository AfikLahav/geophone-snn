# Multiple-Geophone Arrays for Seismic Perimeter Detection, Localization, and Classification

**Scope.** This report assesses what an *array* of geophones buys over the project's current single vertical 4.5 Hz geophone, for the perimeter/intrusion task of labelling 3 s windows as human / vehicle / animal / nothing. It covers (1) array SNR gain and detection-range extension, (2) localization and bearing, (3) array geometry design for the 1–100 Hz footstep/vehicle band on near-surface soil, (4) disambiguation of hard scenes (multiple people vs herds, counting, source separation), and (5) distributed UGS networks and DAS-as-a-dense-array. It ends with concrete design guidance and a realistic assessment of what is and is not achievable.

**Abstract.** A modest array (4–9 vertical geophones, ~5–20 m spacing) converts the single-channel detector into a *direction-finding* sensor. The honest, physics-bounded conclusions: array stacking gain is only √N in amplitude (≈3 dB for 4 sensors, ≈6 dB for 16, ≈10 dB for 100), and because footstep seismic energy is dominated by *geometric spreading* (amplitude ∝ R⁻¹ to R⁻² for surface vs body waves) plus strong soil attenuation, that gain extends footstep detection range only marginally — from a single-sensor reliable range of ~20–30 m to perhaps ~30–40 m, not a doubling. The real payoff of an array is *spatial information*: bearing to ~5–12° and TDOA-based location to sub-metre (controlled) up to ~1 m per 10 m of range (field), the ability to reject incoherent wind/cultural noise that no single sensor can separate, and the ability to spatially separate and *count* overlapping sources — the single hardest part of the multi-person-vs-herd problem. The dominant error source for all of this is the unknown, dispersive surface-wave velocity of near-surface soil (Rayleigh phase velocity ~100–400 m/s, frequency-dependent), which must be measured on-site, not assumed.

---

## 1. Array SNR Gain and Detection-Range Extension

### 1.1 The √N stacking law (coherent vs incoherent)

The foundational mechanism: when M co-located-in-wavefield sensors record the same signal S(t) plus *uncorrelated* noise n(t), a delay-and-sum beam aligns the signal so it adds coherently (×M) while noise adds incoherently (standard deviation grows as √M). The amplitude SNR therefore improves by:

> **G = √M**, i.e. array gain in dB = 10·log₁₀(√M) = **5·log₁₀(M)**.

Concretely (Wikipedia *Seismic array*; NMSOP Ch. 9):

| Sensors M | Amplitude SNR gain | dB gain |
|-----------|--------------------|---------|
| 4         | 2.0×               | ~3 dB   |
| 9         | 3.0×               | ~4.8 dB |
| 16        | 4.0×               | ~6 dB   |
| 30        | 5.5×               | ~7.4 dB |
| 100       | 10×                | ~10 dB  |

Empirical confirmation: a 30-element array (GRA30) improved SNR up to 5.4× for white noise vs 3.2× for 10 sensors — close to the √30 = 5.48 and √10 = 3.16 ideals (Gibbons & Ringdal, *GJI* 2006). The detection threshold of an array is lower than a co-located single 3-channel station "by a factor proportional to the square root of the number of array elements," and a 16-element array gives a 4× threshold gain.

**Critical caveat — coherence.** The √M law *only holds if the signal is coherent across the aperture and the noise is not.* "The failure of conventional beamforming over arrays to result in an SNR gain is almost invariably the result of incoherence between the signals at the different receiver sites." For near-surface footstep signals at 20–80 Hz, wavelengths are short (see §3) and scattering is severe, so full-aperture waveform coherence degrades quickly with spacing. This is *the* practical limiter: a 100-sensor footstep array will **not** reach 10 dB of coherent gain unless the inter-sensor spacing is small enough to keep signals coherent, which conflicts with the wide aperture wanted for resolution. Non-linear stacks (N-th root, phase-weighted/Schimmel–Paulssen 1997) suppress incoherent noise *better* than linear beamforming at the cost of waveform distortion — useful for detection, not for waveform-preserving classification.

### 1.2 Why the range extension is modest

Footstep energy reaching a geophone is bounded by propagation losses, which dominate over the array's √M gain:

- **Geometric spreading.** Body waves (vertical/longitudinal) decay as amplitude ∝ R⁻² (energy ∝ R⁻²... amplitude ∝ R⁻¹ for body-wave energy; the footstep literature quotes intensity ∝ R⁻² for body and R⁻¹ for the Rayleigh surface wave). The surface (Rayleigh) wave, which carries most footstep energy at range, spreads as amplitude ∝ R⁻⁰·⁵ in energy terms but is quoted as intensity ∝ R⁻¹ (Succi et al., *Range limitation for seismic footstep detection*, SPIE 6963).
- **Anelastic / scattering attenuation.** Soil adds strong *non-geometric* excess attenuation, exponential in distance and frequency (e⁻^(πfR/Qv)), preferentially killing the high-frequency footstep content first. This is "a special feature of seismic measurements in soil."

The consequence: detection range is set by an exponential-times-power-law wall. Adding √M of linear gain shifts that wall only logarithmically. Worked estimate using surface-wave intensity ∝ R⁻¹ as the locally dominant term: a 6 dB (16-sensor) gain buys ×4 in power ≈ ×4 in range *if attenuation were purely geometric R⁻¹* — but with the dominant exponential soil term, the realized gain is far smaller, typically **+30–60% range, not ×2–4×.**

### 1.3 Single-sensor baselines from the field literature (perimeter-relevant)

- Footsteps reliably detected to **~20–30 m** with single vertical geophones (Koç & Yegin, *Field Tests*, 2013; Pakhomov/Succi personnel-tracking, SPIE 2001). Koç & Yegin: footsteps tested 4–20 m, **100% classification with <5% false alarm at ≤10 m, 5K gain**; vehicles tested 10–30 m, **68% true / 0% false at 30 m**.
- "Stealthy" footstep detection evaluated to **~51.5 m** under favourable (quiet, well-coupled) conditions.
- DAS field study: empirical site-specific **max footstep detection ~≤24 m** on alluvial soil — even with dense spatial processing, footstep detection is fundamentally short-range.
- UGS vendor figures (defense): seismic detects **personnel to ~25 m, vehicles to ~100 m**; full multi-modal (seismic+acoustic) UGS quote **personnel/vehicle to ~300 m** (acoustic and magnetic carry the long-range vehicle detection, not seismic alone).

**Takeaway for the perimeter:** an array does *not* dramatically extend footstep range. Plan the geometry around a per-node reliable footstep radius of ~25–35 m and overlap nodes accordingly. The array's value is bearing, location, noise rejection, and disambiguation — not reach.

---

## 2. Localization and Bearing

Two distinct capabilities, with very different hardware needs.

### 2.1 Single 3-component geophone → bearing only (polarization)

A *single* triaxial geophone gives azimuth (not range) from **Rayleigh-wave particle motion**: the surface wave is elliptically (retrograde) polarized in the vertical–radial plane, so the horizontal components' covariance points back along the source azimuth. This is the cheapest "array-like" upgrade.

- Back-azimuth from 3-C polarization is a "classical method… based on Rayleigh-wave polarity evaluation, typically assuming retrograde motion."
- Accuracy is frequency-band-dependent: "the degree of polarization of Rayleigh waves varies across frequency bands, and the band with the strongest energy is not necessarily the one with the lowest azimuth error." Optimal-band selection (reciprocal ellipse rate, flatness, semi-minor-axis angle) is needed.
- Reported field bearing performance for walking persons: **bearing-error standard deviation < 12° over 10–70 m** (single/multi-geophone footstep tracking work).
- A single sensor cannot range; it must be paired with motion models (EKF/UKF/particle filters) to convert a bearing-rate time series into a track.

### 2.2 Multi-sensor → range + position (TDOA / hyperbolic)

With ≥3–4 separated vertical geophones, **time-difference-of-arrival** of the footstep impulse across sensor pairs defines hyperbolae whose intersection is the source:

- Method: band-pass filter → cross-correlate each receiver pair → traveltime differences → hyperbolic location estimator.
- **Controlled / structural (concrete floor) accuracy:** average localization error **0.34 m over a 20 m² area**; other indoor experiments **~0.61 m**; "tens of centimetres" typical.
- **Field (soil, outdoor) accuracy:** Pakhomov/Succi personnel tracking with **4 vertical geophones** — footsteps detected to 30 m, **localization error grows ~linearly with range, mean ≈ 8.4 m**. So expect **~1 m of position error per ~3–4 m of range** outdoors, dominated by velocity uncertainty and pick error, *much* worse than the lab numbers.

### 2.3 Beamforming / f-k / MUSIC → bearing + slowness (small array)

For a compact array, delay-and-sum **beamforming** (and high-resolution variants MVDR/Capon, MUSIC, Root-MUSIC, ESPRIT) scans backazimuth and **slowness** (s = 1/v), peaking at the wave's arrival direction and apparent velocity simultaneously. Slowness is the bonus: it estimates the local surface-wave velocity *from the data*, partly self-calibrating the velocity ambiguity (§2.4).

- A **tripartite (3-element triangular) array**, aperture ~100 m, with one 3-C centre + 3 vertical satellites, "can increase detection and location capabilities with a small effort vs single-station networks" by yielding backazimuth + apparent slowness from inter-station cross-correlation delays (NMSOP; Tripoli TRISAR).
- Angular resolution scales with aperture/wavelength: a 32-sensor dense line gave **3.75° resolution**; small-aperture arrays have correspondingly coarser slowness/backazimuth resolution ("poor array resolution… restricted aperture and small number of sensors," Tripoli).
- For a network of small triads, *azimuthal coverage from spatially separated arrays* dramatically improves location: a mesh of 1,030 three-element triads located regional events to **~0.2° (~22 km at teleseismic scale)** — the scale-invariant lesson is that **crossing bearings from ≥2 separated small arrays beats one large array of the same sensor count for 2-D positioning.**

### 2.4 The surface-wave-velocity ambiguity (the dominant error)

Every TDOA/beamforming location requires a propagation velocity, and near-surface Rayleigh velocity is **both uncertain and dispersive**:

- Typical near-surface S-wave velocities: **soft surface soils 180–360 m/s; stiff/denser 360–760 m/s; Vs30 commonly 250–640 m/s** (MASW field studies, e.g. Riyadh 444–640 m/s, Hawassa 249–371 m/s). Rayleigh phase velocity ≈ 0.9·Vs, so footstep surface waves propagate at roughly **~100–400 m/s** depending on soil and frequency.
- **Dispersion:** higher frequencies sample shallower, slower layers → phase velocity *decreases with frequency* (above ~5 Hz, Vs dominates the dispersion curve). A footstep is broadband (1–100 Hz), so different frequency components travel at different speeds, smearing the TDOA pick.
- In bounded/damped media the "perceived propagation velocity decreases as source–sensor distance increases," so a constant-velocity TDOA model is systematically biased — this is exactly the indoor-concrete problem, and soil is worse (heterogeneous, moisture-dependent: Koç & Yegin noted wet-soil dependence).

**Mitigations:** (a) on-site MASW/active-source calibration to measure the dispersion curve V(f); (b) band-pass to a narrow footstep band (e.g. 20–40 Hz) so one velocity applies; (c) use array slowness estimates (§2.3) to track V in situ; (d) weighted-least-squares/robust TDOA estimators (e.g. WLS with cone-tangent-plane constraint, *Sensors* 2018) that down-weight inconsistent picks. Without on-site velocity calibration, expect tens-of-percent range bias.

---

## 3. Array Geometry Design for the 1–100 Hz Footstep/Vehicle Band

Two independent constraints set the geometry: **aperture** controls resolution/lowest frequency; **spacing** controls spatial aliasing/highest frequency. They pull in opposite directions, forcing a band-targeted design.

### 3.1 Wavelengths in the footstep/vehicle band

With Rayleigh velocity V ≈ 100–400 m/s and λ = V/f:

| Frequency | λ at V=150 m/s | λ at V=300 m/s |
|-----------|----------------|----------------|
| 5 Hz      | 30 m           | 60 m           |
| 10 Hz     | 15 m           | 30 m           |
| 20 Hz     | 7.5 m          | 15 m           |
| 40 Hz     | 3.75 m         | 7.5 m          |
| 80 Hz     | 1.9 m          | 3.75 m         |

Footstep energy peaks ~**20–90 Hz** (impulsive heel strike); vehicles add strong **low-frequency 1–30 Hz** (engine/tracks) plus harmonics. So the design band of interest is roughly **λ ≈ 2–30 m**.

### 3.2 Spacing → spatial aliasing (upper-frequency limit)

> **Spatial Nyquist rule: sensor spacing d ≤ λ_min / 2.**

"Spatial aliasing occurs when a propagating waveform is sampled at spatial intervals larger than half the wavelength… dense spacing no greater than ½ wavelength eliminates aliasing at all wavelengths of interest" (ScienceDirect *Spatial Aliasing*; SEG *Geophysics* 86(4)). Some designs accept slight aliasing at d ≈ 0.75λ.

For footsteps to 80 Hz at V = 150 m/s, λ_min ≈ 1.9 m → **d ≤ ~0.95 m** to be alias-free at the top of the band. That is impractically dense for a wide perimeter array, so the standard compromise is to **beamform a band-limited sub-band** (e.g. 20–40 Hz, λ ≈ 4–8 m → d ≤ 2–4 m) and rely on incoherent/energy methods above that. Aliasing above the spatial Nyquist creates *false backazimuth peaks* (grating lobes) — acceptable if you only need detection, dangerous if you need unambiguous bearing.

### 3.3 Aperture → resolution and lowest resolvable frequency

> Backazimuth/slowness resolution ∝ λ / Aperture. "The larger the aperture, the more sensitive the array is to lower-frequency signals."

To resolve bearing well you want aperture ≳ a few wavelengths. For a footstep sub-band at 20–40 Hz (λ ≈ 4–15 m), an aperture of **~15–40 m** gives usable bearing resolution (single-digit to low-tens of degrees). Larger apertures sharpen the beam but, past the coherence length of scattered surface waves, the signals decorrelate and the √M gain collapses (§1.1). So aperture is bounded *above* by coherence, not just by cost.

### 3.4 Number of sensors and layout

- **Minimum useful:** **3 sensors (triangle)** → 2-D backazimuth + slowness with no front/back ambiguity; **4 sensors** → robust hyperbolic TDOA with redundancy (the field-proven personnel-tracking configuration).
- **Diminishing returns:** gain is √M, so going 4→16 buys only ×2 amplitude (3 dB→6 dB). Beyond ~9–16 sensors per node, added value is in sidelobe suppression / aliasing control, not raw SNR.
- **Geometry:** avoid uniform grids (strong grating lobes). Mature seismic arrays use **concentric log-periodic rings** (NORES/ARCES: R_n = R_min·2.15ⁿ, R_min = 150 m) to suppress sidelobes across a wide band with few elements; the scaled-down equivalent for a footstep node is a **small log-spaced spiral or nested triangles** spanning d ≈ 1–4 m (anti-alias) out to aperture ≈ 15–40 m (resolution). Odd numbers of elements per ring break symmetry and reduce aliasing.

### 3.5 Three-component vs vertical

The project's sensor is vertical 4.5 Hz. For an array, **mostly vertical geophones + one 3-C reference** is the efficient mix (matches the tripartite design): verticals do the TDOA/beamforming, the 3-C centre adds polarization backazimuth and a sanity check, and resolves the front/back ambiguity of a small/linear array.

---

## 4. Disambiguating Hard Scenes (Multi-Person vs Herd, Counting, Source Separation)

This is where an array changes *qualitative* capability, because a single sensor sees only the *superposition* of all sources and cannot in general tell "3 people" from "a herd."

### 4.1 Spatial filtering = separating overlapping sources

Beamforming is a **spatial filter**: by steering nulls and beams it can "discriminate waveforms emitted simultaneously by several sources" and, combined with hyperbolic triangulation, recover each source's 2-D coordinates. High-resolution DOA (MUSIC/ESPRIT) can resolve **multiple simultaneous directions of arrival** when sources are angularly separated by more than the beamwidth (~λ/aperture). Practically: two intruders separated in *azimuth* by more than the array's angular resolution (single-digit to ~15° for a compact node) appear as **two distinct slowness/backazimuth peaks**; a single sensor cannot do this at all.

### 4.2 Counting and multi-target tracking

- TDOA + association (assign detected footstep impulses to tracks across sensors) supports **multi-target localization and counting**: each consistent hyperbola-intersection is a target; cadence/stride periodicity per track separates walkers.
- Bearing-only multi-target trackers (EKF/UKF/particle filter; cubature information filters) maintain separate azimuth tracks even from a single 3-C sensor, but **counting robustly needs the range/position that only a multi-sensor array provides.**
- DAS near-field array processing (Nature Comms 2022) demonstrates blind separation and localization of multiple near-field seismic sources and "can easily separate multiple directions of arrival" — the dense-array limit of this capability.

### 4.3 Humans vs animals vs herds — what the array adds

Single-sensor classification (the project's current approach) already separates human/vehicle/animal *by waveform/spectral features* in the 3 s window. The array adds three discriminators the single sensor lacks:

1. **Spatial extent / count.** A herd is *many spatially distributed* impact sources spanning a wide azimuth fan; a squad of people is a *small number of discrete*, trackable point sources with regular bipedal cadence. The array sees the difference as "many diffuse bearings vs few sharp tracks."
2. **Cadence per track.** Once sources are spatially separated, per-target stride rate (~1.5–2.5 Hz human pace vs quadruped gait) becomes measurable — impossible when all footfalls overlap at one sensor.
3. **Velocity/heading consistency.** Coordinated humans move on a coherent heading; herds spread. Slowness vectors per source reveal this.

**Honest limit:** "footstep data with only wild animals is difficult to collect," so animal-vs-human discrimination remains the weakest, least-validated link even with an array; the array improves it mainly by enabling *counting and per-target cadence*, not by a magic spectral separator. For the publication's confound concern, the array's spatial features are exactly the kind of *physically grounded, hard-to-spoof* discriminators reviewers will favour over single-channel spectral features alone.

---

## 5. Distributed Sensor Networks (UGS) and DAS-as-a-Dense-Array

### 5.1 UGS networks (defense context — Elbit-relevant)

- **E-UGS / Pathfinder (ARA):** expendable seismic nodes emplaced in seconds, ML footstep/off-road-vehicle classifiers; U.S. Army procured **~48,000 sensors**. Multi-modal UGS (seismic + acoustic + PIR + magnetic) quote **personnel/vehicle detection to ~300 m**, battery life **30–45 days**, RF-relayed alerts (Bertin, Sensoguard, DSIAC UGS survey).
- Network architecture: many cheap nodes, each a single sensor or tiny array, fused at a base station. **Two paradigms:** (a) *isolated nodes* — cheap, give detection + per-node bearing only; (b) *clustered mini-arrays* — each node is a 3–4-element array giving local bearing/slowness, and **crossing bearings from ≥2 nodes give 2-D position** (the mesh-of-triads lesson from §2.3 at perimeter scale). For a perimeter fence-line, (b) is strongly preferred.

### 5.2 DAS — fibre-optic distributed acoustic sensing as a dense linear array

DAS turns a buried telecom fibre into thousands of strain-rate channels — effectively a continuous line array, ideal for a fence-line perimeter:

- **Channel spacing ~1 m** (10 m gauge length, super-sampled); **thousands of channels** (8,621 over an 8.7 km cable in one study); 100 Hz+ sampling (Copernicus *SE* 12, 2021).
- Acts as a dense array for beamforming with the same physics as geophone arrays, but at **~1 m spacing it is alias-free to much higher frequencies / shorter wavelengths** than any practical discrete geophone array.
- **Footstep localization demonstrated at ~10 m from the fibre** using small linear sub-sections (Nature Comms 2022); empirical footstep detection range **~≤24 m** on alluvium.
- **Limitation:** DAS measures *strain along the fibre axis only* — directional sensitivity ∝ cos2θ (P/SV) or sin2θ (SH), so it is blind to broadside motion and to the vertical particle motion a geophone sees best. Converting DAS strain to particle velocity "dramatically improves coherence and beamforming, on par with a nodal array." DAS is excellent for a **linear fence-following geometry**, weaker for area coverage.

**Relevance:** for a long linear perimeter (border fence), a single DAS fibre is a compelling dense-array alternative/complement to discrete geophone nodes; for area/point protection, clustered geophone mini-arrays are the better fit.

---

## 6. Concrete Design Guidance for a Perimeter Array

**Recommended node ("seismic picket"):**
- **Sensors:** 4 vertical 4.5 Hz geophones + 1 triaxial (3-C) geophone at centre (5 channels). The 4 verticals do TDOA/beamforming; the 3-C resolves front/back ambiguity and gives independent polarization backazimuth.
- **Geometry:** small nested/triangular layout — inner triad at **d ≈ 2–4 m** (anti-alias for the 20–40 Hz beamforming sub-band, λ ≈ 4–15 m) expanding to an outer **aperture ≈ 15–25 m** (bearing resolution). Avoid a regular square grid; use a log-spaced triangle/spiral to suppress grating lobes.
- **Per-node footstep coverage:** design for a reliable radius of **~25–35 m**; place nodes so coverage circles overlap (node spacing ≈ 1.5× the reliable radius, ~40–50 m along the line) so any intruder is seen by ≥2 nodes → crossing bearings → 2-D position.

**Processing pipeline:**
1. **Per-channel detection** (energy/STA-LTA or the existing CNN/SNN on 3 s windows) → candidate events.
2. **Band-split:** beamforming sub-band 20–40 Hz (footsteps), low band 1–30 Hz (vehicles).
3. **Array bearing+slowness** via delay-and-sum beamforming (upgrade to MVDR/MUSIC for multi-source); the slowness output *measures local Rayleigh velocity in situ* — feed it back to calibrate TDOA.
4. **TDOA hyperbolic location** within a node; **cross-node bearing intersection** for perimeter-wide 2-D position and tracking (EKF/UKF/particle filter; multi-target association by cadence).
5. **Disambiguation layer:** count spatially separated tracks, per-track stride cadence, azimuth spread → human-count vs herd-diffuse decision; fuse with the single-window human/vehicle/animal classifier.
6. **On-site calibration (one-time + periodic):** active-source MASW to measure V(f) dispersion; recompute when soil moisture changes (wet-soil velocity dependence is real).

**Velocity calibration is mandatory, not optional** — it is the dominant error term for every localization claim.

### What the array realistically buys (and does not)

| Capability | Single vertical geophone | Recommended array | Reality check |
|---|---|---|---|
| Footstep detection range | ~20–30 m | ~30–40 m | **Only +30–60%** — propagation-limited, not gain-limited |
| Bearing to target | None (or ~12° with 3-C polarization) | **~5–12°** (beamforming) | Aperture/coherence limited |
| 2-D position | None | sub-m (ideal) → **~1 m per 3–4 m range** (field) | Velocity ambiguity dominates |
| Reject wind/cultural noise | Hard | **Yes** — incoherent noise stacks down √M | Needs signal coherence across aperture |
| Separate overlapping sources | No | **Yes**, if Δazimuth > beamwidth | The key new capability |
| Count targets / herd-vs-squad | No | **Yes** — per-track cadence + azimuth spread | Best physically-grounded discriminator |
| Animal-vs-human spectral ID | Marginal | Marginally better | Data-starved problem regardless |

**Bottom line for the project.** Do not buy the array to see farther — buy it to *know where* and *how many*, to *reject noise* that defeats a single channel, and to *separate and count* overlapping intruders. A 5-channel node (4 vertical + 1 triaxial, 2–25 m nested aperture) with band-limited beamforming and cross-node bearing fusion is the right, cost-effective unit. The single largest determinant of localization accuracy is on-site measurement of the dispersive surface-wave velocity — assume nothing, calibrate everything. For a long linear border, evaluate a DAS fibre as a dense-array alternative; for point/area protection, clustered geophone mini-arrays are superior.

---

## Sources

- Seismic array (theory, √N gain, NORES/ARCES geometry, f-k, slowness): https://en.wikipedia.org/wiki/Seismic_array
- IASPEI NMSOP Ch. 9, *Seismic Arrays* (Schweitzer et al.): https://gfzpublic.gfz.de/rest/items/item_43213_8/component/file_56079/content
- Gibbons & Ringdal, array-based detection / SNR gain, *GJI* 165(1) 2006: https://academic.oup.com/gji/article/165/1/149/785425
- Incoherent/partially-coherent array processing (coherence limits), *GJI* 172(1) 2008: https://academic.oup.com/gji/article/172/1/405/588396
- CTBT coherent vs incoherent array processing, *J. Seismology* 2021: https://link.springer.com/article/10.1007/s10950-021-10026-z
- Spatial aliasing (½λ rule), ScienceDirect overview: https://www.sciencedirect.com/topics/engineering/spatial-aliasing
- Spatial aliasing and 3-C seismic sensors, *Geophysics* 86(4): https://library.seg.org/doi/10.1190/geo2020-0172.1
- Single & three-axis geophone — footstep detection, bearing, localization, tracking (bearing σ<12°): https://www.researchgate.net/publication/252385693
- Range limitation for seismic footstep detection (Succi et al., SPIE 6963): https://www.researchgate.net/publication/252566889
- Koç & Yegin, footstep & vehicle detection in WSN — field tests: https://journals.sagepub.com/doi/full/10.1155/2013/120386
- Personnel tracking using seismic sensors (4 geophones, ~8.4 m mean error), SPIE 2001: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/4393/1/Personnel-tracking-using-seismic-sensors/10.1117/12.441276.short
- New algorithm for footstep localization (TDOA, dispersive velocity), arXiv 1211.3233: https://arxiv.org/pdf/1211.3233
- Robust TDOA localization (WLS, cone tangent-plane), *Sensors* 18(3) 778, 2018: https://www.mdpi.com/1424-8220/18/3/778
- Survey: seismic-sensor target detection/localization/ID/activity, *ACM CSUR*: https://dl.acm.org/doi/fullHtml/10.1145/3568671
- Occupant localization via footstep vibration (0.34 m / 0.61 m): https://par.nsf.gov/servlets/purl/10057760
- Mesh of three-element arrays to detect/locate disparate sources, *GJI* 215(2) 2018: https://academic.oup.com/gji/article/215/2/942/5061121
- Backazimuth/slowness by sparsity-constrained array analysis, *GJI* 216(1) 2019: https://academic.oup.com/gji/article/216/1/1/5104381
- DAS seismic beamforming (1 m channels, dense array), *Solid Earth* 12 (2021): https://se.copernicus.org/articles/12/915/2021/
- DAS near-field array signal processing / source separation, *Nature Comms* 2022: https://www.nature.com/articles/s41467-022-31681-x
- Near-surface Vs / Rayleigh velocity (MASW, Vs30 ranges): https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2024.1395431/full ; https://onlinelibrary.wiley.com/doi/10.1155/2022/7588306
- Rayleigh wave / dispersion fundamentals: https://en.wikipedia.org/wiki/Rayleigh_wave ; https://www.usgs.gov/publications/estimation-near-surface-shear-wave-velocity-inversion-rayleigh-waves
- Single 3-C geophone DOA via Rayleigh polarization: https://www.sciencedirect.com/science/article/abs/pii/S0003682X23000981 ; single-station polarization tutorial: https://www.sciencedirect.com/science/article/abs/pii/S0065268718300025
- Bearing-only ground-target tracking, single 3-C seismic sensor (EKF/UKF/PF), *Arab. J. Geosci.* 2019: https://link.springer.com/article/10.1007/s12517-019-4368-2
- Unattended ground sensors (E-UGS Pathfinder, ~48,000 units; UGS survey): http://pathfinder.ara.com/industry-insights/unattended-ground-sensors ; https://dsiac.dtic.mil/technical-inquiries/notable/unattended-ground-sensor-survey/ ; https://www.bertin-technologies.com/products-range/multisensors-network/ ; https://sensoguard.com/ugs-system-unattended-ground-sensors/
- Moving source localization using seismic signal processing, *J. Sound Vib.* 2014: https://www.sciencedirect.com/science/article/abs/pii/S0022460X14007573
