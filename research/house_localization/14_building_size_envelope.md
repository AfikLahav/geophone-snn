> **Research synthesis — house-localization pivot ("wallhacks").** Inline research (Sonnet), 2026-07-09. Web research + physics derivation. Cross-references docs 02, 04, 05, 11. Indexed in [README.md](README.md).

---

# Building Size Envelope for Exterior-Ring Geophone Localization

**Scope.** This document answers: given N geophones bonded to the *exterior* of a building (walls, foundation slab edge, or roof parapet), what range of building sizes can the system usefully cover for indoor footstep localization via structure-borne vibration? The answer is bounded below by physics (too small → problem is trivial, array is overconstrained) and above by signal attenuation and localization geometry.

---

## 1. Geophone Sensing Range for a Footstep Through Structure

### 1.1 Signal budget on a concrete slab

A footstep heel-strike on concrete produces a broadband vertical force impulse of roughly 700–1,400 N (1.0–2.0× body weight), with dominant energy from 5 to 250 Hz. The resulting floor-surface velocity at distance r from the source follows two attenuation mechanisms operating simultaneously:

**Geometric (cylindrical) spreading.** Flexural waves radiate from the footstep source as a cylindrical wavefront expanding in the floor plane. For a 2-D surface wave, amplitude ∝ r^(−1/2), so the velocity amplitude decays by:

> ΔL_spread(r₁→r₂) = 10 log₁₀(r₂/r₁) dB [amplitude reference: 20 dB per decade]

From 1 m to 10 m: −10 dB. From 1 m to 20 m: −13 dB.

**Material damping (internal loss).** For a flexural wave in a concrete plate with loss factor η_total (internal concrete + energy leakage at boundaries), the spatial attenuation coefficient is:

> α(f) = η_total × π × f / c_B(f)   [Np/m]   →   α_dB = 8.686 × α   [dB/m]

Using the 150 mm slab flexural velocity from doc 05 and in-situ η_total values (0.03–0.08):

| f (Hz) | c_B (m/s) | η_total low (0.03) | α_dB/m low | η_total high (0.08) | α_dB/m high |
|--------|-----------|-------------------|------------|---------------------|-------------|
| 20 | 96 | 0.03 | 0.17 | 0.08 | 0.46 |
| 50 | 152 | 0.03 | 0.27 | 0.08 | 0.73 |
| 100 | 215 | 0.05 | 0.63 | 0.10 | 1.27 |
| 200 | 304 | 0.07 | 1.25 | 0.12 | 2.15 |

**Combined loss (spreading + damping) at key distances:**

| Distance r | 50 Hz, η=0.03 | 100 Hz, η=0.05 | 200 Hz, η=0.07 |
|------------|---------------|----------------|----------------|
| 5 m | −7 + 1.4 = −8.4 dB | −7 + 3.2 = −10.2 dB | −7 + 6.3 = −13.3 dB |
| 10 m | −10 + 2.7 = −12.7 dB | −10 + 6.3 = −16.3 dB | −10 + 12.5 = −22.5 dB |
| 15 m | −11.8 + 4.0 = −15.8 dB | −11.8 + 9.4 = −21.2 dB | −11.8 + 18.8 = −30.6 dB |
| 20 m | −13 + 5.4 = −18.4 dB | −13 + 12.5 = −25.5 dB | −13 + 25.0 = −38.0 dB |

### 1.2 Reference signal level and noise floor

Measured indoor footstep velocities on concrete: ~200–2,000 nm/s at 1 m source-sensor distance in the 20–80 Hz band (Mirshekari et al. 2018; NSF PAR 10057760). Geophone self-noise: ~10 ng/√Hz above corner frequency, equivalent to ~2–5 nm/s RMS in a 60 Hz processing band — many orders of magnitude below the footstep signal at indoor ranges. The binding SNR limit is therefore *ambient structural noise* (HVAC, traffic, appliance vibration), not geophone self-noise.

**Typical ambient structural noise on a quiet residential floor:** 10–50 nm/s in the 20–80 Hz band, rising to 100–500 nm/s near HVAC or mechanical rooms. Using a conservative 50 nm/s ambient floor, and requiring SNR ≥ 10 dB (3.2× amplitude) for reliable TDOA pick:

> Minimum detectable signal: 50 × 3.2 = 160 nm/s

Starting from 500 nm/s at 1 m, this buys roughly 10 dB of margin before hitting the detection floor. At 50 Hz with η=0.03, 10 dB occurs at ~13 m. At 100 Hz with η=0.05, 10 dB occurs at ~7–8 m.

**However, detection range and localization range are not the same.** TDOA-quality arrival picks require the direct-path pulse to be temporally resolvable from the first wall reflection, which demands SNR ≥ 15–20 dB (5×) at the sensor. This halves the effective usable range to ~5–8 m for the 80–150 Hz band that gives the sharpest TDOA picks.

### 1.3 Wall junction losses for exterior-ring sensors

Wall junction vibration reduction index Kij (EN 12354, ISO 10848 measured values for concrete rigid T/X-junctions):

- **Floor slab → exterior wall, rigid T-junction (bare concrete):** Kij = 4–8 dB
- **Floor slab → exterior wall → adjacent room floor (two junctions, through-wall):** 8–16 dB
- **Floor slab → interior partition wall → adjacent bay floor:** 12–20 dB
- **Elastomeric/acoustic isolation joint at junction:** 15–25 dB additional

An exterior-ring sensor bonded to the slab edge at the wall-floor junction (low-ring position per doc 11) receives the floor wave *before* it crosses a junction — the 1-junction loss is only 4–8 dB and occurs at the coupling point, not on the propagation path. This is why low-ring (floor slab level) mounting is the recommended configuration: the sensor sits at the structural node where the wave exits the floor into the wall, capturing near-full wave energy rather than the attenuated transmitted fraction.

For a footstep on the second floor traversing down through a wall to an exterior ground-level sensor: the path crosses at minimum 2 junctions (floor 2 → wall top → wall → wall base → foundation), totalling 8–20 dB additional loss beyond in-floor propagation. This is the inter-floor detection challenge.

### 1.4 Practical sensing range summary

| Localization use | Max source-sensor distance | Limiting factor |
|-----------------|---------------------------|-----------------|
| TDOA-quality arrival pick (SNR ≥ 20 dB) | 5–8 m (80–150 Hz band) | Reverberation + dispersion blur the direct-path peak |
| Detection only (SNR ≥ 10 dB) | 10–15 m | Ambient noise floor |
| Energy-based estimation | 8–12 m | Attenuation heterogeneity, coupling variability |
| MFP / fingerprinting (no direct-path requirement) | 10–15 m per sensor | Model mismatch and modal shift; no direct-path requirement |

**The practical rule from published experiments** (Mirshekari 2016/2018, NSF PAR 10057760) is that for accurate localization on concrete, each footstep location should be within ~4–5 m of at least one sensor, and within ~8–10 m of all sensors. The "geophone sensing radius of 4 m" cited for concrete tile floors is a localization accuracy limit, not a detection limit — detection extends to 10–15 m in quiet environments.

---

## 2. The Building-Size Envelope

### 2.1 Upper bound: sensor-to-interior distance constraint

The central constraint for an exterior ring is: **every interior point must be within the effective TDOA range of at least 3 sensors** (to get overdetermined 2-D position). With a TDOA-quality range of ~5–8 m per sensor and sensors placed on the exterior perimeter, the maximum distance from any interior point to the nearest sensor is approximately half the building's shorter dimension (for a rectangular floor).

For a rectangular building of width W and length L (W ≤ L), the maximum interior distance to the nearest perimeter sensor is:
> r_max ≈ W/2 (for a sensor mid-point on the long wall)

Setting r_max = 5 m (conservative, 80–150 Hz TDOA quality): **W_max ≈ 10 m**
Setting r_max = 8 m (aggressive, with strong MFP or careful calibration): **W_max ≈ 16 m**

For GDOP, the diagonal dimension matters: sources near the building center receive signals from all perimeter sensors at large angles. For a 10 × 15 m building with sensors at 4 corners + 2 mid-walls, the worst-case GDOP (center of floor) ≈ 1.3–1.8 — acceptable. For a 15 × 20 m building, the floor center is ~12 m from all corners; GDOP there is 2.5–4, and the TDOA SNR is marginal.

**Upper bound on usable floor dimension: ~10–15 m (shorter axis), ~20 m (longer axis) with MFP.**

This corresponds to a single-storey footprint of roughly **100–300 m²** for single-story, elongated or rectangular forms. For square footprints, **100–200 m²** is the comfortable range.

### 2.2 Lower bound: minimum meaningful problem

At very small scales the localization problem becomes trivial (few candidate positions, small GDOP) and the simulation cost is disproportionate to the physical richness. More importantly, below ~3 × 3 m (9 m²), the TDOA delay between any two sensors is less than 20 ms at 150 m/s group velocity, comparable to the TDOA pick uncertainty from reverberation — so the problem becomes largely fingerprinting-only.

Minimum meaningful floor dimension for TDOA-based localization: ~3 m (one room width). Smallest meaningful test case: a single room or studio of **15–30 m²**, with 3 sensors at a 3–5 m baseline.

### 2.3 Multi-storey: additional constraints

Each floor adds its own propagation problem. An exterior sensor at ground level receives signals from a footstep on floor 2 after traversing the floor-to-wall junction chain (net additional 8–20 dB loss per floor crossed). For a 3-storey building, a footstep on floor 3 arrives at a ground-level sensor with ~16–40 dB *more* attenuation than the same footstep on floor 1 — typically pushing the signal below the TDOA-quality threshold.

**Consequence:** effective floor-level discrimination requires sensors at *each storey level* (the "top ring" approach in doc 11). For a 3-storey building, the exterior ring must include 3 height levels to maintain TDOA-quality signals from each floor. This scales the sensor count as N_total = N_per_floor × N_floors.

---

## 3. Sensor Count Scaling with Building Size

### 3.1 Derivation from TDOA and GDOP requirements

For 2-D localization on a single floor:
- **Minimum:** 3 sensors (2 independent TDOAs → unique position, no outlier rejection)
- **Field-proven minimum:** 4 sensors (3 TDOAs, overdetermined; corner placement)
- **Recommended:** 5–6 sensors for a rectangular single-storey house (4 corners + 1 mid-wall + 1 symmetry breaker)

Sensor spacing should be ≤ 2 × (TDOA-quality range) ≈ 8–12 m to ensure each sensor "sees" any floor point with adequate SNR. For a rectangular floor of perimeter P, the number of exterior sensors needed is:

> N_min ≈ P / (2 × r_TDOA)

where r_TDOA ≈ 6 m (conservative mid-range estimate).

| Floor footprint | Perimeter | N_min (spacing 12 m) | Recommended N | Expected GDOP (center) |
|----------------|-----------|----------------------|---------------|------------------------|
| 5 × 5 m (25 m²) | 20 m | 2 (degenerate) | 3–4 | 1.0–1.5 |
| 8 × 10 m (80 m²) | 36 m | 3 | 4–6 | 1.2–1.8 |
| 10 × 12 m (120 m²) | 44 m | 4 | 5–6 | 1.3–2.0 |
| 10 × 20 m (200 m²) | 60 m | 5 | 6–8 | 1.5–2.5 |
| 15 × 20 m (300 m²) | 70 m | 6 | 8–10 | 2.0–3.5 |
| 20 × 30 m (600 m²) | 100 m | 8 | 12–16 | 3–6 (degraded) |

### 3.2 "Exterior ring stops working" threshold

The exterior ring fails when the building's shorter width W exceeds ~2 × r_TDOA_quality ≈ 10–16 m. Beyond this, the floor center is beyond TDOA-quality range from any perimeter sensor. The system can still *detect* footsteps from the center, but cannot localize them accurately without interior sensors or MFP with a well-calibrated structural model.

For MFP with full structural model: the effective range per sensor extends to 10–15 m, pushing the "stops working" threshold to W ≈ 20–30 m — sufficient for most single-family housing but not for large commercial buildings.

**Practical rule:** Exterior-ring localization works without interior sensors up to roughly **W = 12 m** (TDOA), **W = 20 m** (MFP with calibrated model). Beyond these, add interior sensors or tile the building into sub-zones.

---

## 4. Extremes: Very Small and Very Large Buildings

### 4.1 Very small: studio / single room (~15–30 m²)

**Floor dimensions:** 3.5–5.5 m per side.

**Attenuation at 5 m, 100 Hz:** −10.2 dB combined (spread + material). A footstep at any interior point is within 2–3 m of at least one perimeter sensor. TDOA delays are 5–15 ms, well resolved at 2,000 SPS.

**Challenges:**
- Modal problem: a 5 × 5 m floor has fundamental bending mode at ~8–31 Hz (boundary-dependent). With modal density 0.04 modes/Hz and only ~1–2 modes per 50 Hz bin, the transfer function is highly modal-peaked and sparse. TDOA is unreliable below ~80 Hz; fingerprinting works well because each location produces a unique modal superposition.
- Reverberation time at 50 Hz: T_s ≈ 4–10 s (doc 05 §1.5) — the floor rings for seconds between footsteps. The direct-path TDOA pick window is swamped by reverberant energy from previous steps. Narrow high-frequency bandpass (100–250 Hz) is essential.
- Sensor count: 3–4 sensors suffice. GDOP ≈ 1.0–1.5 with sensors at all 4 corners.
- Performance expectation: FEEL/fingerprinting dominant; 5–20 cm accuracy achievable with calibration; TDOA alone 20–50 cm.
- **Verdict:** Technically the easiest case for accuracy, but modal physics makes it a distinct simulation regime. Worth including as the minimum archetype.

### 4.2 Very large / multi-unit: apartment block (3–6 storeys, multiple flats, shared structural slabs)

**Floor dimensions:** 40–80 m per building face; individual flat: ~40–80 m² each, 4–8 flats per floor.

**Key new physics — shared structural slab:**

In a reinforced concrete frame apartment block, all flats on the same floor share the same structural slab. A footstep in flat A at 10 m from the party wall generates a wave that propagates through the shared slab, crosses the party wall (Kij ≈ 6–12 dB for a bare concrete wall), and arrives at sensors on the exterior of flat B with only 6–12 dB extra loss. This is *much less* than the isolation designers intend, and much more than the building-acoustics literature sometimes implies — because impact sound (which the standards are optimizing to *suppress*) and localization (which needs *more* of it) are engineering opposites.

**Shared slab: does it help or hurt?**

| Effect | Direction |
|--------|-----------|
| Wave propagates across party walls → exterior sensors on flat B receive signals from flat A | Ambiguity: which flat is the person in? |
| More propagation paths → more measurements → in principle more information for MFP | Helps MFP if the model covers the full building slab |
| Multiple junctions each adding 6–12 dB loss | Hurts SNR at distant sensors |
| Modal coupling between bays → transfer functions in flat B affected by footstep in flat A at the same frequency | Hurts fingerprinting (transfer functions not unique per flat if slab is monolithic) |
| Each flat has a unique modal configuration (different walls, room geometry) | Helps fingerprinting *within* a flat if inter-flat transmission is modeled correctly |

**The "which flat" problem.** In a 4-flat-per-floor plan, a footstep anywhere on the shared slab is detectable from exterior sensors of all flats. The localization problem is: given sensors on the building *exterior* only, determine which flat the person is in *and* their position within that flat. This is fundamentally a disambiguation problem with as many hypotheses as flats per floor.

For a 3-flat row (A-B-C) with exterior sensors on each side:
- Footstep in flat B: signals arrive at sensors on the A-side with delay through 1 party wall (Kij ≈ 8 dB) and sensors on the C-side similarly.
- A naive TDOA localizer sees signals at 4 exterior sensors and estimates a position somewhere between flat A and C — correct only if it knows the wall loss and models the geometry correctly.
- MFP with a full-building model can in principle disambiguate, but requires a calibrated model of the entire building, not just one flat.

**Inter-floor ("which storey") problem.** As established, each floor crossed adds 8–20 dB. With sensors at 3 height levels, storey determination is feasible. Without them (sensors only at ground level), floors above 2 are below TDOA-quality SNR at the exterior.

**Practical verdict on large multi-unit buildings:**
- Shared slab propagation *enables* signals from multiple flats to reach exterior sensors, but creates a localization ambiguity that cannot be resolved without modeling the full building.
- For single-story or two-story buildings where all flats are on the same slab plane (row houses, terrace housing), the problem can be tiled: each flat is an independent sub-problem with its own exterior sensor ring, and inter-flat leakage is treated as a noise source (signal arriving from the neighboring flat).
- For multi-storey apartment blocks (3+ floors), the problem changes character: it requires (a) sensors at each storey level, (b) a building-scale structural model, and (c) a localization algorithm that operates at the building level rather than the flat level. This is a qualitatively harder problem and should be treated as a separate architecture, not an extension of single-flat localization.

**Simulation treatment:** a large apartment block is *best modeled as a tiled set of sub-problems*. Each flat + its shared-wall boundary conditions is one simulation unit. The inter-flat coupling is a boundary condition (partial transmission / partial reflection at party walls), not a full-building wave propagation problem. This reduces the 3-D elastic simulation domain from the full building to one representative structural bay per storey type.

---

## 5. Recommended Size Classes to Simulate

Five archetypes span the useful range from trivially small to the boundary of exterior-ring viability. Each is described with floor area, sensor configuration, simulation domain, difficulty rating, and primary localization challenge.

### Archetype A — Single Room / Studio (15–30 m²)

| Parameter | Value |
|-----------|-------|
| Floor area | 15–30 m² (e.g., 4 × 6 m = 24 m²) |
| Storeys | 1 |
| Walkable interior | ~12–20 m² after wall margin |
| Sensor count | **4 nodes** (3 corners + 1 mid-wall) |
| Sensor placement | Low-ring, slab-bonded at floor-wall junction |
| Max source-sensor distance | 4–5 m |
| Primary localization band | 80–250 Hz (high reverb at lower f) |
| Expected TDOA accuracy | 15–30 cm (narrow-band + dispersion model) |
| Expected fingerprinting accuracy | 5–15 cm (FEEL-class, well-calibrated) |
| Primary challenge | Sparse modal regime; T_s 4–10 s; short source-sensor delay (3–8 ms) demands precise timing |
| Simulation domain | One 4–6 m × 4–6 m × 2.8 m box; ~4 M voxels at 150 mm |
| Role in dataset | Lower bound; calibrate modal physics; baseline for accuracy ceiling |

### Archetype B — Single-Family House, Small (60–80 m²)

| Parameter | Value |
|-----------|-------|
| Floor area | 60–80 m² (e.g., 8 × 9 m = 72 m²) |
| Storeys | 1 (or 1 + partial loft) |
| Walkable interior | ~50–65 m² |
| Sensor count | **5–6 nodes** (4 corners + 1–2 mid-wall, all low-ring) |
| Max source-sensor distance | 5–7 m |
| Primary localization band | 50–200 Hz |
| Expected TDOA accuracy | 20–40 cm |
| Expected fingerprinting/MFP accuracy | 15–30 cm |
| Primary challenge | Structural heterogeneity (partitions, door openings break the wave field); 2–3 interior rooms means inter-room transmission losses vary |
| Simulation domain | Full house 8 × 9 × 2.8 m with 2–4 interior walls; ~28 M voxels at 150 mm |
| Role in dataset | The core archetype; matches existing published experiments; best validated against Mirshekari 0.34 m result |

### Archetype C — Single-Family House, Full (100–150 m²) — Primary Target

| Parameter | Value |
|-----------|-------|
| Floor area | 100–150 m² (e.g., 10 × 12 m = 120 m²) |
| Storeys | 1 or 2 |
| Walkable interior per floor | ~80–110 m² |
| Sensor count | **6 nodes per floor** (4 corners + 1 mid-wall long axis + 1 top-ring for 2-storey) = 6 (1-storey) or 10–12 (2-storey) |
| Max source-sensor distance | 7–9 m |
| Primary localization band | 20–200 Hz |
| Expected TDOA accuracy | 25–50 cm |
| Expected MFP accuracy | 20–40 cm |
| Primary challenge | 2-storey: inter-floor path loss (8–20 dB/floor traversal); storey disambiguation requires top-ring sensors; GDOP near floor center rises to 1.8–2.5 |
| Simulation domain | 10 × 12 × 5.8 m (2-storey); ~83 M voxels at 150 mm → GPU-limited at 250 Hz, feasible at 100 Hz |
| Role in dataset | Primary target; matches doc 11 design (9 × 11 m walkable); most design effort here; publication-anchor archetype |

### Archetype D — Row House / Terraced House (80–100 m² per unit, shared party wall)

| Parameter | Value |
|-----------|-------|
| Floor area | 80–100 m² per unit (e.g., 6 × 14 m = 84 m²), 3 units sharing side walls |
| Storeys | 2 |
| Sensor count | **6 per unit** (each unit instrumented independently) + optional 2 on party walls |
| Max source-sensor distance | 6–8 m within-unit; cross-unit signals detectable but attenuated 8–16 dB |
| Primary localization band | 40–200 Hz |
| Expected accuracy (within target unit) | 25–45 cm |
| Primary challenge | Cross-unit leakage: footstep in unit B reaches unit A sensors at −8 to −16 dB → ghost localizations at unit B positions in unit A's coordinate system. Must be modeled to train the classifier to reject them |
| Simulation domain | One full-building simulation (3 × 6 × 14 × 5.8 m); or independent per-unit simulations with party-wall transmission modeled as a stochastic boundary loss. The tiled approach is recommended |
| Role in dataset | Tests inter-flat ambiguity; publication targets this for the "cross-unit ghost" result; critical for publication rigor |

### Archetype E — Mid-Rise Apartment Block (300–600 m² per floor, 3–6 storeys)

| Parameter | Value |
|-----------|-------|
| Floor area | 300–600 m² per floor (e.g., 15 × 25 m = 375 m²) |
| Storeys | 3–6 |
| Individual flat area | 40–80 m² (4–8 flats per floor) |
| Sensor count | **8–12 per storey × 3–6 storeys = 24–72 total** |
| Max source-sensor distance | 12–18 m (exterior ring only); center of building is 7–12 m from the nearest perimeter sensor |
| Primary localization band | 20–100 Hz (higher frequencies attenuated over long in-floor distances) |
| Expected accuracy | 50–100 cm per flat (exterior ring only); 30–50 cm with interior sensors or per-flat sub-ring |
| Primary challenge | (a) GDOP degradation at floor center; (b) multi-flat "which-flat" ambiguity; (c) inter-floor discrimination needs sensors at each storey; (d) simulation domain is 5–20× larger than Archetype C |
| Simulation strategy | Do NOT simulate the full building. Simulate one representative flat bay (one structural bay of ~50 m² per floor) with the shared slab as a periodic boundary condition; tile the localization dictionary. Cross-flat signals are modeled as spatially shifted copies of the bay's Green's function with Kij-weighted amplitude |
| Role in dataset | Establishes upper bound; tests where exterior-ring breaks down; not the primary publication target but needed to characterize the method's scope statement |

---

## 6. Summary of Attenuation-Driven Size Bounds

The table below integrates all constraints into a single reference:

| Constraint | Bound on building shorter dimension W |
|------------|--------------------------------------|
| TDOA-quality pick requires SNR ≥ 20 dB from nearest sensor | W ≤ 2 × 5 m = **10 m** |
| TDOA with MFP (SNR ≥ 10 dB adequate, model compensates dispersion) | W ≤ 2 × 8 m = **16 m** |
| Detection only (identify footstep, no accurate position) | W ≤ 2 × 12 m = **24 m** |
| Exterior ring becomes blind at building center (no sensor coverage) | W > 20–25 m → must add interior sensors |
| Minimum for meaningful TDOA (3 ms delay between sensors) | W ≥ **3–4 m** |
| Minimum for distinct fingerprints (> λ_B/4 separation of calibration points at 100 Hz) | W ≥ **1–2 m** (trivially satisfied above room scale) |

**The sweet spot for exterior-ring geophone localization is single buildings or single flats with shorter floor dimension 5–15 m, floor area 25–200 m², up to 2 storeys with a top-ring sensor at the upper slab.** Above this range the problem changes character (multi-flat, multi-floor) and requires a building-scale structural model and sensor count that grows as O(N_floors × N_perimeter).

---

## 7. What the Large-Building Case Teaches

The apartment block case is instructive precisely because it is the failure mode of the exterior-ring assumption. Key lessons:

1. **Shared slab is a double-edged sword.** The propagation path that enables detection from a distant flat is the same path that creates ghost localization in the wrong flat. This is not just a nuisance — it is a fundamental ambiguity that cannot be resolved without knowing which flat is occupied *a priori*, or without a cross-flat discriminant (time-of-arrival ordering, amplitude ratio, structural model covering party-wall loss).

2. **The "which-flat problem" is topologically similar to the "which sensor cluster" problem in outdoor arrays.** The same beam-steering / matched-field techniques apply, but operating on the building's full slab Green's function rather than on a single room.

3. **Tiling is the natural decomposition.** Each flat is a localization sub-problem with its own sensor ring and Green's function dictionary. Cross-flat signals are noise at the flat level but signal at the building level. The localization algorithm should operate in two stages: (a) detect and classify the event as originating from each flat (using amplitude ratios and arrival ordering across the full building sensor set), then (b) fine-localize within the identified flat using that flat's local Green's function. This two-stage architecture is not in the existing literature and would be a novel contribution.

4. **3–6 storey mid-rise is the hardest unsolved problem in this space.** No published paper has demonstrated meter-scale floor-localization in a multi-flat, multi-storey building using only exterior sensors. This represents the frontier of the method, and explicitly scoping the current work to Archetypes A–C (with Archetype D as a stretch goal) is the defensible publication framing.

---

## Sources

- Mirshekari et al. (2018) — 0.34 m accuracy, 4–8 geophones, concrete/wood/steel floors, 20 m²: [NSF PAR](https://par.nsf.gov/servlets/purl/10057760)
- Mirshekari et al. (2016) — Indoor footstep localization structural dynamics: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327016305015)
- FEEL Algorithm, 3 sensors, 3.05×4.38 m RC room (PMC9886238): [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9886238/)
- GaitVibe+ 0.22 m vibration-only (arXiv:2212.03377): [arXiv](https://arxiv.org/pdf/2212.03377)
- Reservoir computing, 11 sensors, 16 m corridor (arXiv:2603.04610): [arXiv](https://arxiv.org/html/2603.04610v1)
- Dispersion propagation operators, concrete + Al plate (PMC11644851): [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644851/)
- SO-TDOA, perceived velocity, Kelvin-Voigt (arXiv:1211.3233): [arXiv](https://arxiv.org/pdf/1211.3233)
- Energy-based vibro-localization with Byzantine elimination (PMC10708826): [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10708826/)
- Practical inter-floor noise sensing, TDOA + classification (PMC6749573): [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6749573/)
- Vibration reduction index Kij, EN 12354, concrete junctions: [ResearchGate](https://www.researchgate.net/publication/259574153_The_vibration_reduction_index_Kij_laboratory_measurements_versus_predictions_EN_12354-1_2000); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0003682X10000083)
- Estimation of vibration attenuation through building junctions: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0003682X95000255)
- Structure-borne sound in buildings, Liverpool: [Liverpool Repo](https://livrepository.liverpool.ac.uk/3077422/1/author%20version.pdf)
- Structural reverberation times, Liverpool: [Liverpool Repo](https://livrepository.liverpool.ac.uk/3109444/)
- Range limitation for seismic footstep detection, Sabatier & Ekimov SPIE 2008: [SPIE](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/6963/69630V/Range-limitation-for-seismic-footstep-detection/10.1117/12.785235.short)
- Occupant localization framework using slab vibration (Frontiers 2019): [Frontiers](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2019.00063/full)
- Floor vibration localization framework (ScienceDirect 2021): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327021008177)
- RC structure type and low-freq impact sound (ScienceDirect): [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0003682X18311058)
- Comparison direct vs flanking floor impact sound in bearing-wall apartments (ScienceDirect 2025): [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360132325004834)
- EN 12354 building acoustics prediction guide: [AcousPlan](https://acousplan.com/blog/en-12354-building-acoustics-guide)
- Outdoor MEMS accelerometer vibration localization: [PMC12473310](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12473310/)
- Human footstep-induced floor vibration (ResearchGate): [ResearchGate](https://www.researchgate.net/publication/5326491_Human_footsteps_induced_floor_vibration)
- Docs 02, 04, 05, 11 in this research folder (primary internal references)
