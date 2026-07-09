# Class List & Variable Specification (v0.2)

> **AUTHORITY: `GENERATION_PLAN.md` v1.1 supersedes this file where they conflict** (2026-06-12
> review). Key syncs: footstep band 10–40 → **20–90 Hz** (§4, corrected below); walk GRF
> asymmetric; run strike-pattern branch; animal double-hump WALK-ONLY + fore/hind asymmetry;
> adult mass 45–110 / child 18–42 kg; rain onset 60–80 Hz. See GENERATION_PLAN Appendix S.

Full taxonomy. Single vertical 4.5 Hz geophone (+tolerance). Output = `time_s, amplitude`
(1000 Hz) + a manifest of all labels/parameters. Designed to be **appended to** over time.

> **v0.2 changelog:** K locked at **300** (Israel-weighted); terrain catalog expanded for
> Israel (loess, kurkar, snow cover, sabkha, frozen, dirt road, paving stones, clay, rock);
> window 1 s → **3 s**; mains/ambient noise now **fitted to the measured rig**, not assumed;
> path model rebuilt around **r(t) with radial speed decoupled from gait speed**; scene
> duration = **residence time in detectable range** + lead-in/out; **composite events** added;
> coarse **output taxonomy** (human/vehicle/animal/nothing) pinned; footstep wavelet now
> **requires the sharp impact transient** (likeness-probe finding, see §4).

Conventions: BW = body weight (N) = mass·9.81. Cadence = events/s. All "how simulated"
entries are the forcing function `F(t)`; recorded `v(t) = F(t) ⊛ G(r,t)` (Green's function),
then through the sensor chain. **Terrain selects the propagation branch** (half-space vs modal).

---

## 0. Label hierarchy (stored per sample)

```
class        → human | vehicle | animal | environment
subclass     → e.g. human:walk, vehicle:car, animal:dog, environment:wind
gait/mode    → e.g. walk/run/crawl, walk/trot/gallop  (parameter, not a separate class)
+ full parameter vector (every sampled variable below)
```
Coarse (class) and fine (subclass) labels both stored → granularity chosen at train time.

**Output taxonomy (decided):** the deployed model outputs the coarse 4 —
**human / vehicle / animal / nothing**. Everything finer (subclass, gait, footwear,
composite-event membership) is retained in the manifest: it is available as optional
**auxiliary supervision** during training, and "what to do with a detection" is a
**post-processing / policy layer**, not a model class. *Train fine-grained, deploy coarse.*

---

## 1. Terrain axis (selects propagation physics) — **K = 300 profiles, Israel-weighted**

Each terrain *family* below is sampled into multiple distinct **soil profiles** (a profile =
a specific layer stack: per-layer thickness, Vs, Vp, ρ, Q — plus moisture/frozen state). One
profile → one cached GF bank. **K = 300 total** (decided; was 150) ≈ **~20 profiles/family
weighted**, so each family spans its internal range (Vs spread, 1–4 layers, depth-to-bedrock
when within ~30 m, velocity gradient, dry/wet) AND leaves **~4–5 held-out profiles/family**
in val/test for a per-family unseen-site generalization test.

| Family | Vs (m/s) | Vp (m/s) | ρ (kg/m³) | Qs | Branch | Coupling f_c (Hz) | ~profiles |
|---|---|---|---|---|---|---|---|
| Soft soil/loam | 100–300 | 300–700 | 1300–1700 | 5–20 | half-space | 30–150 | ~30 |
| **Loess (N. Negev)** | 150–400 | 400–900 | 1300–1600 | 5–25 | half-space | 30–150 | ~22 |
| Dry sand (coastal/dune) | 100–500 | 400–1200 | 1400–1700 | 10–40 | half-space | 30–150 | ~28 |
| Gravel / reg / wadi | 300–750 | 500–1500 | 1700–2000 | 20–60 | half-space | 60–250 | ~26 |
| Clay/silt (terra rossa, rendzina, vertisol) | 200–800 | 600–1800 | 1600–2000 | 5–30 | half-space | 50–200 | ~30 |
| Wet/saturated soil | 200–600 (Vp→1500) | 1500–2000 | 1900–2100 | 20–60 | half-space | 80–300 | ~20 |
| Rock / weathered→bedrock (limestone, chalk, basalt, hamada) | 800–3300 | 2000–5500 | 2000–2700 | 50–200 | half-space | 100–400 | ~28 |
| **Kurkar (cemented calc. sandstone)** | 600–1500 | 1500–3000 | 1900–2200 | 30–80 | half-space | 80–300 | ~10 |
| Dirt/unpaved road (compacted top) | 250–600 | 500–1200 | 1600–1900 | 15–50 | half-space | 60–250 | ~14 |
| Asphalt (inverse profile: stiff thin top over soft base) | 1000–1800 | 2500–3500 | 2200–2400 | 20–60 | half-space (stiff) | 150–400 | ~20 |
| Concrete slab-on-grade | 2000–2500 | 3000–4500 | 2200–2400 | 50–150 | half-space (stiff) | 150–500 | ~16 |
| Paving stones/brick over bedding sand | 800–1800 | 2000–3500 | 2000–2300 | 30–80 | half-space (stiff) | 100–400 | ~6 |
| Frozen ground (any soil family, Vs ×2–4) | 400–2000 | 1500–4000 | as base | ×2–5 | half-space | 100–400 | ~12 |
| **Snow-covered ground** (low-ρ LP surface layer over base/frozen) | layer: 100–500 | 300–1500 | 100–500 | 5–20 | half-space | 20–100 | ~6 |
| **Sabkha / saline flat (salt crust over soft saturated mud)** | crust 400–800 / mud 80–200 | →1500 | 1800–2100 | 5–20 | half-space | 50–200 | ~4 |
| **Suspended concrete floor** | modal | — | — | ζ 2–5% | **modal** | f₀ 5–15 Hz | ~10 |
| **Wooden floor** | modal | — | — | ζ 2–8% | **modal** | f₀ 10–25 Hz | ~8 |
| | | | | | | **Total** | **~290 → round to 300** |

**Weighting rationale (Israel deployment):** desert + rocky ≈ 40% (loess, reg/gravel, sand,
hamada/limestone), Mediterranean soils ≈ 25% (terra rossa/rendzina/vertisol clay, loam),
coastal ≈ 20% (dune sand, kurkar, hamra→soft soil), paved/urban ≈ 12%, floors + snow +
frozen ≈ 8% (snow/frozen = Hermon/Golan/Jerusalem winters only; Negev exceptional).
Counts are tunable; the learning curve (train-plan) confirms.

**Layering note (Israel-specific):** sample plenty of **thin-soil-over-fast-rock** stacks
(terra rossa/rendzina on limestone/chalk, hamada) — the strong impedance contrast is the
signature of Israeli hill terrain. Asphalt is the **inverse** (stiff top over soft base) —
propagates differently from natural ground; treat as its own family, never as "fast soil."

### 1b. Profile sampler — how the K=300 profiles vary (BINDING, amended 2026-06-11)

Each family's ~20 profiles are independent draws over these axes:

| Axis | Rule |
|---|---|
| top-layer Vs | **log-uniform** in the family range (uniform overweights the stiff end) |
| layer count | 2–4 layers; **uniform single-layer half-spaces ≤ ~5%** (geologically rare) |
| top-layer thickness | **per-family geologic prior** (loess 2–25 m, sand 1–15, clay 1–20, weathered rock 0.2–3, dirt-road cap 0.1–0.5 …) — a sub-meter soft veneer puts a quarter-wave site resonance Vs/4h in-band, which is real physics but the wrong prior for thick-deposit families |
| Vs growth with depth | ×1.3–2.2 per layer, **plus ~15–20% of profiles get one "bedrock contact" jump ×3–8** (the thin-soil-over-limestone Israeli-hills signature — terra rossa/rendzina/hamada) |
| Vp/Vs | 1.6–2.5 dry; saturated Vp pulled to ≥1500 m/s (Vs ~unchanged) |
| density | Gardner(Vp) **±10% jitter**; **snow overrides Gardner entirely (ρ 100–500)** |
| Q | log-uniform in family range, frequency-independent (v1 simplification); **plus ±30% per-SCENE jitter at render** — Q is applied at render time, so within-site Q variety is free (no bank recompute) |
| coupling f_c / Q_c | f_c log-uniform per family; Q_c **1–20** |
| inverse branch | asphalt, concrete, paving stones, **sabkha** (salt crust over soft mud) — stiff-over-soft: longer ring-down window + settle-clamp (elastic solver rings forever where real material Q damps in ~Q/πf s — documented approximation) |
| frozen | base-family stack with Vs ×2–4, Q ×2–5 |
| splits | **~15% val + ~15% test held out PER FAMILY** (≈3+3 of 20) — the per-family unseen-site generalization test |

Modal floors (susp. concrete + wood, ~18 profiles) need no GF bank → **~282 pyprop8 banks**.
Documented simplifications: no Q(f), Vs–Q correlation only family-implicit, V_R taken
from the top layer.

**Half-space branch:** Rayleigh wave, V_R ≈ 0.92·Vs; `A(r,f) ∝ coupling · r^−0.5 · exp(−π f r /(Q·V_R))`.
**Modal branch (bounded floors):** impulse response `h(t) = Σ_n A_n e^(−ζ_n ω_n t) sin(ω_n√(1−ζ_n²) t)`,
modes `f_n` from floor type (n=3–10), + 0–5 reflection echoes. (This is the "floor ring" we measured: ~65 Hz, Q≈17.)
Branch rule: **bonded to ground underneath = half-space** (incl. slab-on-grade); **air
underneath (bounded, energy trapped) = modal**.

Coupling efficiency (source→ground) is **inverse** with surface stiffness: soft soil ≈1.0, rock/concrete ≈0.1
(rigid surfaces barely deform → less injected energy, sharper/ringing signals).

---

## 2. Compatibility matrix (which classes occur on which terrain) — enforces plausibility

| Class \ Terrain | soft soil | sand | gravel | asphalt | concrete slab | susp. floor | wood floor |
|---|---|---|---|---|---|---|---|
| Human (all gaits) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Car / truck | ✅(dirt road) | ⚠️ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Motorbike | ✅ | ⚠️ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Bicycle | ⚠️ | ⚠️ | ✅ | ✅ | ✅ | ❌ | ⚠️(smooth) |
| E-scooter | ❌ | ❌ | ⚠️ | ✅ | ✅ | ❌ | ✅(smooth) |
| Dog | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Deer/sheep | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ❌ | ❌ |
| Cattle/horse | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Elephant | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ❌ | ❌ |
| Environment/nothing | ✅ all terrains (ambient + weather, no source) |

✅ valid · ⚠️ rare/edge (low weight) · ❌ excluded. The randomizer samples class first, then a valid terrain.

**v0.2 families inherit the column of their nearest mechanical analog:** loess & dirt road →
soft soil; kurkar & frozen → gravel/asphalt-like (stiff); snow-covered → the underlying
terrain's column (vehicles ⚠️ on deep snow); sabkha → soft soil (vehicles ⚠️); paving
stones → asphalt; clay/rock → gravel column. Override per-cell in config if needed.

---

## 3. Shared variables & how each is calculated

| Variable | Symbol | Range / value | How simulated |
|---|---|---|---|
| Sample rate | fs | 1000 Hz (fixed) | matches rig + training data |
| Window length | — | **3.0 s (3000 samples), 50% train hop** | cadence needs 2–3 steps; deployment hop tuned later |
| **Path** | r(t) | see §6 | 2-D trajectory → range-over-time (single sensor: azimuth ignored) |
| Geometric spread | — | r^−0.5 (Rayleigh) | half-space branch |
| Attenuation | α | exp(−π f r/(Q·V_R)) | per-frequency, from terrain Q |
| **Geophone** | H(s) | f₀=4.5 Hz ±2%, h=0.6 ±0.05, G=28.8 ±5% V/(m/s) | `H(s)=G·s²/(s²+2hω₀s+ω₀²)`; tolerance = manufacturing spread |
| Coupling resonance | H_c(s) | f_c per terrain, Q_c 1–20 | 2nd-order LP in series before geophone |
| Johnson noise | e_n | 2.46 nV/√Hz | white, at coil |
| Amp noise (AD620) | — | 9 nV/√Hz + 1/f (0.28 µV pp, 0.1–10 Hz) | white + 1/f, post-geophone |
| Mains/tonal lines | — | **fit to measured rig** (rig shows 44–47 & 55 Hz, NOT 50 Hz) + randomized site lines incl. classic 50 Hz+harmonics, 1–10% FS, drifting phase | additive sinusoids; per-scene line set sampled |
| Ambient seismic | — | **fitted to rig's measured floor** (0.08 mV RMS); NLNM…NHNM retained as the randomization *range* across sites | colored noise (FFT × √PSD), added to v_ground |
| Preamp gain | A_v | ~100–1000× (randomized) | scalar; verify no clip |
| ADC | — | 0.125 mV/LSB, ±0.256 V, 12-bit | `round(v/LSB)`, clip to FSR |
| SNR | — | 0–40 dB | sets source-vs-noise level |

---

## 4. SOURCE CLASS SPECS — humans

`F(t)` = impulse train of per-foot GRF wavelets at cadence; each footfall radiated from its
position along the path (so successive steps have slightly different r).

**BINDING [v1.1 — corrected band & mechanism; see GENERATION_PLAN Part H]:** the smooth
weight curve alone is **wrong** (concentrates energy <10 Hz). The discriminative band is
**20–90 Hz, site-dependent** (NOT 10–40), generated by the **sharp NORMAL-force heel-strike
impact transient, amplified/shaped by the site transfer function** — this is why the centroid
moves with terrain (sabkha ~19 → rock ~77 Hz). The **tangential/friction force (Fx, 0.10–0.20
BW) is a SECONDARY additive contributor at its TRUE magnitude — do NOT inflate it to co-equal**
(its own energy is mostly >100 Hz). Generate Fz (weight + sharp impact) and Fx at true
magnitudes; the terrain bank produces the band. **Gate 3 (binding, pre-generation): ≥50% of the
footstep's 5–120 Hz energy in 20–90 Hz, centroid 20–90 Hz (stiff terrain to 110).**

**Footwear = a temporal-sharpness knob, not an amplitude knob:** total impulse is set by
mass/gait (footwear-independent); footwear is a spring/damper that low-pass filters the
impact. Boot = fast rise, HF-rich, higher transient peak; barefoot/cushioned = slow rise,
LF, damped; flip-flop adds a secondary "slap." (Contact *area* is irrelevant: a ~0.2 m
patch ≪ wavelength below ~500 Hz → point force is exact for our band.)

| Subclass | GRF shape | Peak/foot | Contact (s) | Cadence (Hz) | How simulated / distinguishing feature |
|---|---|---|---|---|---|
| walk | double-hump (M) | 1.0–1.2 BW | 0.6–0.7 | 1.4–2.5 | two-peak wavelet; alternating L/R |
| run | single-hump | 2.0–2.9 BW | 0.12–0.30 | 2.6–3.0+ | sharp single peak, shorter contact, higher kurtosis |
| slow/stealth | double-hump ×0.3–0.6 | 0.5–0.8 BW | 0.7–1.0 | 0.8–1.5 | reduced amplitude + longer contact → hardest to detect |
| crawl | low broadband + drag | <0.5 BW | irregular | 0.5–1.0 | intermittent limb impacts + continuous low-amplitude drag term |
| irregular/injured | asymmetric double-hump | 0.7–1.3 BW alternating | varies | irregular | L≠R amplitude & interval (limp) — breaks cadence detectors |
| climb/jump | impulsive bursts | 1.5–3 BW | spikes | sporadic | high-amplitude transients, no steady cadence |
| group (N people) | superposed walkers | — | — | overlapping | sum of 2–5 walk sources, decorrelated phases |
| tamper/dig | repetitive impacts | varies | short | 0.5–3 | periodic non-locomotion impacts near sensor |
| loiter/stand | micro-tremor | tiny | — | ~0 | near-nothing; weight shifts |

Body mass: 50–100 kg (child option 20–40 kg). Speed 0.5–4 m/s. Source model option: bipedal
SLIP for GRF realism, or parametric double/single-hump wavelet (cheaper, sufficient).

---

## 5. SOURCE CLASS SPECS — vehicles, animals, environment

### Vehicles (moving load = quasi-static axle + roughness-dynamic + engine harmonics)
| Subclass | Mass (kg) | Forcing | Dominant band | Distinguishing feature |
|---|---|---|---|---|
| car | 1000–2500 | axles + quarter-car roughness + engine | 10–20 Hz | bell envelope; wheel-hop 10–17 Hz |
| truck | 3500–40000 | multi-axle, high load | 0–20 Hz | strongest, lowest; soil sets freq, mass sets amplitude |
| motorbike | 200–400 | light load + strong engine harmonics | tonal tens–100s Hz + 10–15 Hz | rpm-swept tonal lines |
| bicycle | 80–100 | quasi-static + tire roughness, no engine | low/broadband, weak | continuous, low-kurtosis, featureless |
| e-scooter | 75–120 | tiny-wheel roughness, silent drive | low WBV + HF roughness | non-tonal, pavement-dependent |

Speed: 0–130 km/h (cars), per-type. Engine harmonic lines: f = (rpm/60)·order, swept with speed.

### Animals (allometric — parametrize by mass M; gait = walk 4-beat / trot 2-beat / gallop)
Peak GRF/foot ≈ (1.0–1.2 BW walk, 1.5–2.5 BW fast)·M·9.81; stride freq ∝ M^−0.14; contact ∝ M^0.14;
dominant vib freq ∝ 1/contact (↓ with size). Footfall sequence = 4 impacts/stride (walk).

**Israel-first set (primary weight):** dog, jackal, boar, gazelle, sheep/goat, cattle, horse.
Others retained at low weight for generality.

| Subclass | Mass (kg) | Stride (Hz) | Peak/foot | Dominant freq | Notes |
|---|---|---|---|---|---|
| dog | 10–30 | 2–3.5 | 1.0–1.2 BW (100–350 N) | higher, <100 Hz | 60/40 fore/hind |
| jackal/fox | 5–15 | 2.5–4 | 1.0–1.2 BW | high, light | hardest animal to detect |
| wild boar | 50–120 | 1.5–2.5 | 1.2–1.8 BW | mid | heavy trot; common Israel intruder |
| gazelle/deer | 15–90 | 1.5–2.5 | 1.0–1.5 BW (300–1300 N) | mid | trot/bound; pronk = impulsive bursts |
| sheep/goat (flock option) | 30–90 | 1.5–2.5 | 1.0–1.5 BW | mid | flock = superposed, decorrelated |
| cattle/horse | 400–700 | 0.8–1.5 | 1.0–1.4 BW (4–9 kN), double-hump | low | |
| elephant (low weight, generality) | 2000–6000 | 0.8–1.2 | tens of kN, long contact | 10–40 Hz (peak ~24) | Rayleigh ~250 m/s |
Key discriminator vs human: **4 footfalls/stride vs 2** (temporal pattern), not raw frequency.

### Environment / nothing (no source; ambient + modifier)
| Subclass | How simulated |
|---|---|
| quiet | NLNM…NHNM colored noise only |
| wind | + colored noise gated >~3 m/s, +5 dB/(m/s), incoherent; optional 0.25 Hz tree / 30–50 Hz structure lines |
| rain | + impact noise >50 Hz, PSD ∝ R^1.1–1.9; + slow saturation (Vs↓) |
| hail | + sparse Poisson heavy-tailed impulses, lower-freq-skewed |
| machinery | + tonal line(s) at fixed freq (e.g. 50–70 Hz, like the real floor-ring source) |
| mains | + strong 50 Hz + harmonics |
Weather is **also** a modifier applied to *every* sample (snow = LP filter + dv/v; frozen = Vs×1–4 + Q shift; temperature drift).

---

## 6. Path model (single sensor → r(t)) — radial speed is the driver

**Principle:** the signal is driven by **r(t)** (range over time), not by gait speed. Gait
speed sets cadence + footfall spacing; **path geometry sets r(t)** → the amplitude envelope
and the scene duration. The two are sampled **decoupled**: brisk walker on a meandering path
= fast cadence + slow radial closure (dr/dt ≪ gait speed). For one vertical geophone,
azimuth is irrelevant — r(t) IS the geometry.

| Path type | Parameters | r(t) shape / envelope |
|---|---|---|
| straight pass-by | closest-approach d (0.5 m–class max range, up to 300 m), speed, direction | r(t)=√(d² + (v(t−t₀))²) — clean bell |
| diagonal/oblique | entry/exit points, speed | piecewise linear — asymmetric bell |
| curved/meandering | waypoints, varying radial speed | **multi-bump irregular envelope** |
| approach–stop–leave | d_min, dwell time | rise → plateau → fall |
| loiter/pace | center, radius, slow | bounded random within radius — long duration |
| partial traverse | enters range, turns back | one-sided bump |
| random walk | step size, bias | accumulated |
| direct approach/retreat | start r, speed | monotonic |
Footfalls placed along the path → each impact at its own r (per-footfall GF; vehicles =
retarded-time superposition, see PLAN 4.1).

**Scene duration (binding rule):** duration = **residence time in detectable range**,
computed from the **radial** speed — `≈ (2 × detectable range) / |dr/dt|` — plus lead-in/out
margins. Walking human at 75 m range ≈ ~2 min; car at 300 m ≈ ~40 s; slow/stealth, loiter,
congested traffic, multi-event timelines = minutes. **Never pad a fast event with
steady-state quiet** (redundant correlated windows); long scenes must contain evolving
structure. Spans ~10 s (fast pass) → minutes (loiter/traffic/multi-event).

**Lead-in / lead-out (every scene):** scenes start with the source below the noise floor,
rise through detectability, and recede back below it. This (a) gives realistic onsets/offsets,
(b) **generates the per-window deployment labels automatically** (faint edges → `nothing`,
peak → class), and (c) supplies free, class-balanced `nothing` windows.

---

## 6b. Composite events (scene-composition templates)

Multi-source scenes are **cheap by superposition** (linear small-strain regime): sum the
per-source ground motions, then ONE sensor pass + ONE noise + ONE ADC. They are **robustness
data for the primitive classes, not new output classes** — labeled per-window by *what is
present* (multi-label in manifest; coarse class for the model).

| Template | Composition | Purpose |
|---|---|---|
| group (2–5 people) | superposed walkers, decorrelated phases; in-step variant | the classic hard case |
| crowd converging | N walkers, paths converging on a point | rising multi-source density |
| crowd scattering | N walkers, radial dispersal from a point | sudden onset, decaying density |
| bidirectional traffic | 2+ vehicles, opposite directions, offset timing | overlapping Doppler bells |
| herd/flock | N animals (sheep/goat/cattle), correlated wander | animal masking/confusion |
| mixed timeline | e.g. car passes, 30 s later walker, then dog | realistic multi-event minutes |
Kept a **balanced minority** of the corpus; every member source still individually present
in the manifest parameter vector.

---

## 6c. Corpus policy (pre-generated, stratified)

- **Pre-generate a stratified corpus** (decided) — N scenes per terrain-family × class ×
  condition cell, so coverage is **provable by query**, not hoped-for from RNG. The corpus is
  the inspectable, versioned, reproducible dataset.
- **On-the-fly is reduced to a thin train-only layer:** fresh noise realization + small
  shift/scale per epoch (anti-memorization). **Val/test fully frozen to disk.**
- **Scale (estimate, learning-curve-tunable):** ~30k–100k scenes → ~0.4M–1.3M 3 s windows
  ≈ 1–3 weeks-equivalent of recording ≈ 5–16 GB. Generation after GF banks exist is
  O(S·N log N) FFT work (S scenes, N≈4k samples/scene-segment) — minutes, not hours; the
  K=300 GF-bank library (~hours, across-model parallel) is the only expensive step.

---

## 7. Verification checklist (run before trusting any batch)

- **Per-stage:** geophone freq response shows 4.5 Hz corner (match real data); noise PSD correct; ADC steps = 0.125 mV.
- **Physics invariants (automated over the manifest):** RMS decreases monotonically with labeled distance; doubling r drops amplitude by predicted factor; no NaNs; no silent ADC clipping; energy in labeled band dominates.
- **Closed-form:** fixed-distance point source vs Lamb's problem agree.
- **Compatibility:** no samples violate §2 (assert class–terrain validity).
- **Compare-to-real:** domain-classifier test + spectral overlay (sim footsteps land in real bands).
- **Reproducibility:** fixed seed → identical output (regression test).
- **Visual:** plot one of each subclass — cadence visible? bell envelope on pass-by? floor-ring on bounded surfaces?

---

## 8. Open / to finalize
- Per-class parameter *distributions* (uniform vs log-normal) — default per §master table in SIMULATION_RESEARCH.md.
- Source-model choice per class: parametric wavelet (cheap) vs SLIP/biomechanical (richer) —
  default parametric, **but the wavelet must satisfy the §4 impact-transient requirement.**
- Weights for ⚠️ (rare) compatibility cells.
- Exact footstep-wavelet impact/friction parametrization + its Gate 3 band-fraction threshold.
- Final dataset scale — set by the learning curve in the training plan.
- ~~Generation throughput benchmark~~ — DONE (Phase 0: across-model parallel, ~1 min/model
  effective; confirm scaling in orchestrator).
- ~~K~~ — DONE: **K = 300**, Israel-weighted (§1).
