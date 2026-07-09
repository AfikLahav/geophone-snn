# Roof-Budget Verification — Inputs Table & Acceptance Criteria (Workplan)

> **Work outline — house-localization pivot ("wallhacks").** Planning agent, 2026-07-09.
> **Purpose:** Define the frozen input set, the canonical test geometry, and the pass/fail criteria for the Step-1 analytical verification of the claim that a ground-slab footstep reaches the flat RC roof of a single-storey house attenuated ~20–40 dB (20–200 Hz).
> **Status:** WORKPLAN ONLY — no computation performed. Companion outlines: `01_*` (junction transmission theory, other agent) and `02_*` (path/spreading budget, other agent). This document is the third leg: inputs + criteria + deliverable spec.

---

## 1. Material property table

### 1.1 What the project has already decided (doc 09, §2.5 / §3.1 / §3.2)

| Item | Decided value (doc 09) | Status for this calculation |
|---|---|---|
| Roof slab | 200 mm RC flat slab (180–220), "same as floor slab"; flat roof IS the top boundary, no attic | **Use as-is** |
| Ground/floor slab | 200 mm RC (180–220) | **Use as-is** |
| Exterior wall | 200 mm hollow block (180–220); +10–30 mm insulation/plaster → 260–310 mm total | Use 200 mm block core; plaster mass is a second-order correction |
| Interior partition | 150 mm hollow block (120–180) | Use as-is (secondary paths only) |
| RC columns | 200×400 mm (200×350–250×500), grid 3.5–5 m | **Use as-is — but load-path role is undecided, see §1.3 (M5)** |
| Clear height | 2.75 m (2.70–2.85); floor-to-floor 2.95–3.10 m | Use as-is |
| RC elastic | ρ = 2400 kg/m³, Vp = 3800 m/s, Vs = 2000 m/s → E_dyn ≈ 28 GPa | Use as nominal; add explicit E/ν form below |
| Hollow block eff. medium | ρ = 1100 kg/m³, Vp = 2200 m/s, Vs = 1000 m/s (E_dyn ≈ 5–10 GPa) | Use as-is |
| Soil | ρ = 1800, Vp = 500, Vs = 200 m/s (Vs 150–300 = key diversity axis) | Needed only for ground-slab energy-leakage loss (§1.3 M6) |
| Damping | **Q = ∞ for the training corpus** ("omit Q; diversity axis later") | **NOT usable here — η is mandatory for this calculation** |

### 1.2 Required property table (to be frozen in the verification note)

| # | Property | Nominal | Range | Source |
|---|---|---|---|---|
| P1 | RC static E (C25/30–C30/37 ≈ Israeli B30) | 31 GPa | 30–33 GPa | [Eurocode 2 EN 1992-1-1, Table 3.1 (Ecm)](https://eurocodes.jrc.ec.europa.eu/EN-Eurocodes/eurocode-2-design-concrete-structures) |
| P2 | RC dynamic E (small-strain, ×1.1–1.25 static) | 34 GPa | 28–39 GPa | Consistency check vs doc 09 Vp/Vs (E_dyn ≈ 28 GPa); [Lee 2017, dynamic modulus via shear wave](https://onlinelibrary.wiley.com/doi/10.1155/2017/1651753) |
| P3 | RC density ρ | 2400 kg/m³ | 2300–2500 | doc 09 (decided) |
| P4 | RC Poisson ν | 0.20 | 0.15–0.25 | doc 05 uses 0.2; standard for dynamic concrete |
| P5 | **RC internal loss factor η_int** — MISSING in doc 09 | 0.01 | 0.005–0.02 | Lab cast concrete η ≈ 0.005 ([Hopkins & Robinson, structural reverberation times, Liverpool](https://livrepository.liverpool.ac.uk/3109444/1/Structural%20reverberation%20times%20-%20Copy%20for%20Elements.pdf) use 0.005 for 200 mm cast in-situ plates); in-situ RC is higher because of rebar interfaces + micro-cracking + finish layers ([Craik, "The internal damping of building materials", Applied Acoustics 1992](https://www.sciencedirect.com/science/article/abs/pii/0003682X9290028Q)) |
| P6 | **RC total loss factor η_tot(f) in situ** (internal + coupling + edge leakage) | EN formula | η_int + m′/(485·√f) | [EN ISO 12354-1:2017, Annex C](https://www.iso.org/standard/59114.html) empirical in-situ formula; for m′ = 480 kg/m²: η_tot ≈ 0.15 @ 50 Hz, 0.11 @ 100 Hz, 0.08 @ 200 Hz — NOTE this exceeds doc 05's η_total = 0.03–0.10 band; treat [doc-05 low, EN-formula high] as the uncertainty bracket |
| P7 | Hollow-block wall η | 0.04 | 0.02–0.08 | doc 05 §1.5 (masonry 0.02–0.08); [damping in unreinforced masonry](https://www.researchgate.net/publication/245078328) |
| P8 | Hollow-block wall E_dyn, ρ | 7 GPa, 1100 kg/m³ | 5–10 GPa, 1000–1200 | doc 09 §3.2 (decided) |
| P9 | **Roof screed** (sloped lightweight concrete, "mede") — MISSING in doc 09 | 70 mm avg, ρ 900 | 40–120 mm, ρ 600–1200, E 2–8 GPa | [Laterlite, roof screed laid to falls](https://www.laterlite.com/applications/roofs/roof-screed-laid-to-falls-lightweight-insulating/); [Perlite Institute, insulating concrete roofs](https://www.perlite.org/perlite-lightweight-insulating-concrete-roofing/); Israeli practice: SI 1752 flat-roof series, [Standards Institution of Israel catalog](https://www.sii.org.il) |
| P10 | Roof waterproofing (bitumen membrane) — MISSING | 5 kg/m², 5 mm | 4–10 kg/m² | Mass only; negligible stiffness, small added damping. SI 1752 series / manufacturer data |
| P11 | Floor finish on source slab (tile + bedding) — MISSING | 60 mm, ρ 2200 → +130 kg/m² | 50–70 mm | Standard Israeli terrazzo/porcelain on sand-cement bed; raises source-slab m′ and shifts its c_B slightly |
| P12 | Soil (for ground-slab edge leakage) | Vs 200 m/s, ρ 1800 | Vs 150–300 | doc 09 (decided) |
| P13 | Kij junction defaults (cross-check target) | T: 5.7 dB, X: 8.7 dB @ equal masses | ±3 dB vs measured | [EN ISO 12354-1 Annex E formulas](https://www.iso.org/standard/59114.html); [Crispin et al. 2006, lab Kij for rigid junctions](https://journals.sagepub.com/doi/10.1260/135101006777630427); [Kij measured vs EN 12354-1 predictions](https://www.researchgate.net/publication/259574153_The_vibration_reduction_index_Kij_laboratory_measurements_versus_predictions_EN_12354-1_2000); overview: [AcousPlan EN 12354 guide](https://acousplan.com/blog/en-12354-building-acoustics-guide) |

**Key physics note on P5/P6 (why in-situ ≠ lab):** the 20–40 dB verdict is roughly linear in η over the wall + roof propagation legs. Lab concrete η ≈ 0.005 would predict far less path damping than in-situ measured decay (ISO 10848 structural reverberation times T_s = 2.2/(η·f) of 0.1–0.3 s at 100 Hz imply η_tot ≈ 0.07–0.22). The verification MUST run both a low-η and high-η corner, not a single value. Measurement standard: [ISO 10848-1:2017](https://www.iso.org/standard/67226.html).

### 1.3 Missing-input register (decided nowhere in docs 03/09/14)

| ID | Missing item | Why it matters here | Where to get it |
|---|---|---|---|
| M1 | η for every element (doc 09 explicitly sets Q = ∞) | Dominates the wall-leg and roof-leg attenuation | P5–P7 sources above |
| M2 | Roof build-up (screed + membrane + optional gravel/pavers) | Adds 50–150 kg/m² to roof slab → shifts f_c, mass-law, modal frequencies, and adds constrained-layer damping | P9/P10 sources; SI 1752 series via [SII](https://www.sii.org.il) |
| M3 | Parapet ("ma'akeh") | Israeli flat roofs carry an RC parapet ≥ 1.05 m (accessible roofs; SI 1142 guardrail heights) — stiffens/mass-loads the roof edge exactly where the "corner receiver" sits; also changes the wall→roof junction from a T to an X-like geometry | [SII catalog](https://www.sii.org.il) (SI 1142); typical detail 150–200 mm RC; treat as X-junction variant in the junction workplan |
| M4 | Floor finish on ground slab | +~130 kg/m² on the source plate changes injected velocity level and c_B | P11 |
| M5 | **Vertical load path: block walls vs RC columns + ring beam** | Doc 09's voxel stacker rests the roof slab on continuous block walls + columns. Real Israeli practice: roof slab is cast on an RC ring/tie beam over the block walls, and columns carry most load. Junction Kij for slab–blockwall (mass ratio 480/220 kg/m²) differs by several dB from slab–RC-wall; columns are a separate low-loss longitudinal channel | Decide both variants and compute both: (a) plate–plate chain (walls), (b) column longitudinal path. Cite [seismic RC practice in Israeli buildings](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8837140/) for frame-plus-infill typology |
| M6 | Ground-slab boundary condition (slab-on-grade energy leakage into soil) | Sets the source-plate η_tot → the reference "floor sensor level" against which the roof deficit is quoted | EN 12354-1 Annex C coupling-loss terms + doc 09 soil values |
| M7 | Window/door openings in the wall leg | Openings cut the transmitting wall cross-section (up to 20–40% of facade) → less wall area, slightly higher effective loss | doc 09 generator places 1 window/room; use ±0 / −30% wall-length sensitivity case |

---

## 2. Canonical geometry

**Proposal: use doc 14 Archetype C (primary target), single-storey variant — 10 × 12 m footprint** — so the roof-budget number is directly comparable with every other number in the project (doc 14 uses the same house for sensor-count, GDOP, and range tables; doc 11's 9 × 11 m walkable interior is the same class).

| Parameter | Value | Source |
|---|---|---|
| Plan | 10 × 12 m rectangle | doc 14 Archetype C |
| Storeys | 1, flat RC roof, no attic | doc 09 §2.5 |
| Clear height | 2.75 m; roof slab soffit at +2.75 m, roof top at +2.95 m | doc 09 §3.1 |
| Ground slab / roof slab | 200 mm RC each | doc 09 §3.1 |
| Exterior walls | 200 mm hollow block + RC columns 200×400 @ 4 m + RC ring beam (M5 variants) | doc 09 §3.1 + M5 |
| Interior walls | 150 mm block; assume 2 partitions crossing the plan (from generator statistics) — secondary paths only | doc 09 §1.4 |
| Parapet | 1.05 m × 150 mm RC (M3; sensitivity: none vs full) | SI 1142 |
| Roof build-up | 70 mm lightweight screed + membrane (M2; sensitivity: bare slab vs full build-up) | P9/P10 |

**Source positions (footstep, vertical force on ground slab):**
- S1 — room center, (5.0, 6.0) m: max distance to all walls; the mean-field reference case.
- S2 — near-wall, 0.5 m from the mid-point of a long exterior wall, (0.5, 6.0) m: minimal in-slab spreading before the first junction; upper-bound roof coupling.

**Receiver positions (vertical velocity on roof top surface):**
- R1 — roof center, (5.0, 6.0): junction chain + maximal in-roof propagation. Primary claim test point.
- R2 — roof corner, (0.5, 0.5): directly above a wall–wall–roof junction; minimal in-roof propagation, near-parapet. Upper-bound roof signal.
- R3 — directly above S2, (0.5, 6.0): shortest possible structural path (slab → wall → roof, one wall leg of 2.75 m); the adversarial "best-case roof sensor" the claim must survive.

Path definition shared with the other two workplans: **source slab (spreading + η) → floor/wall junction (Kij→Dv) → wall leg 2.75 m (bending, η_wall) → wall/roof junction (Kij→Dv) → roof slab (spreading + η) → receiver**, plus the parallel RC-column longitudinal channel (M5b) summed energetically.

---

## 3. Claims under test and acceptance criteria

### 3.1 Claim register

| ID | Claim (verbatim source) | Tested by | Falsified if |
|---|---|---|---|
| C1 | **Roof signal 20–40 dB below** floor-level signal, "multiple structural joints between source and sensor" — doc 04 §4.4 / §4.2 mounting table | Full chain budget (this note + workplans 01, 02), S1→R1 and S1→R2, 20–200 Hz | Central estimate < 15 dB or > 45 dB in the majority of 50–150 Hz third-octave bands with the uncertainty band also excluding [20, 40] dB |
| C2 | Wind-induced roof vibration **10–100× footstep signal** — doc 04 §4.4 | Literature wind-response spectra of flat roofs vs the budget's predicted roof footstep level. NOTE a band mismatch to resolve: doc 04 puts wind energy at 0.1–10 Hz but the localization band is 20–200 Hz — the claim as written may conflate bands | Predicted roof footstep velocity (20–200 Hz) exceeds literature wind-induced levels *in that band* by > 10 dB for typical (≤10 m/s) wind — i.e., wind would NOT mask footsteps in-band. Partially testable analytically; flag as literature-dependent |
| C3 | **Roof structural modes 2–15 Hz dominate** — doc 04 §4.4 | Thin-plate modal calc of the roof slab: f11 = (π/2)·√(D/ρh)·(1/a² + 1/b²). Preliminary sanity: full 10×12 m simply-supported span → f11 ≈ 5–6 Hz (claim OK); 4×4 m column-supported bay → f11 ≈ 40 Hz (claim fails) | With the M5-realistic support grid, f11 > 20 Hz — the claim then only holds for unsupported spans and doc 04 wording needs qualification |
| C4 | **8–20 dB per floor crossed** (junction chain, ≥2 junctions) — doc 14 §1.3/§2.3 | Junction workplan (01): sum of two Kij→Dv legs + wall-leg damping for the 2.75 m storey height | Chain total outside 8–20 dB by > 3 dB systematically across 50–150 Hz |
| C5 | Junction transmission **> 50% energy below 50 Hz** — doc 05 §2 ("the 'room' for structural waves is the whole building") | Junction workplan (01): transmission coefficient τ(f) from the Kij/Dv model at 31.5–50 Hz for the rigid floor–wall T | τ < 0.5 at 31.5 and 50 Hz bands for the rigid T with project mass ratio |
| C6 | Floor→exterior-wall rigid-T **Kij = 4–8 dB** — doc 14 §1.3 | EN 12354-1 Annex E with the ACTUAL mass ratio (RC slab 480 kg/m² vs block wall 220 kg/m², lg-ratio ≠ 0 shifts Kij up) | Computed Kij outside 4–8 dB by > 3 dB — likely for the slab→light-block direction; would force doc 14 correction |

### 3.2 Acceptance criteria for "verified" (Step-1 gate)

1. **Primary criterion (C1):** the analytical mean-field budget for S1→R1, evaluated per third-octave band (25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200 Hz), lands inside **20–40 dB in at least 4 of the 6 bands 50–160 Hz**, AND the low-η/high-η uncertainty band overlaps [20, 40] dB in **all** of those bands. If the central estimate is outside but the band overlaps → verdict "consistent, revise claim wording". If the band excludes [20,40] → claim falsified, docs 04/14 get corrected numbers.
2. **Junction cross-check:** each per-junction loss used in the chain agrees with the EN 12354-1 Annex E default for that junction type and mass ratio within **±3 dB** (the documented lab-vs-prediction scatter, [Crispin 2006](https://journals.sagepub.com/doi/10.1260/135101006777630427)).
3. **Cross-method consistency:** the junction-theory chain (workplan 01) and the path/spreading budget (workplan 02) must agree within **6 dB** per band where they overlap; a larger gap blocks the verdict and forces reconciliation before any doc edits.
4. **Adversarial case survives:** S2→R3 (shortest path) must still show ≥ 15 dB loss for the 20–40 dB claim to stand as a *mounting-guidance* claim (doc 04's table is about whether a roof sensor is useful anywhere, not just above the roof center).
5. **Every claim C1–C6 gets an explicit verdict** (verified / revise-wording / falsified) with the number that replaces it if revised.

---

## 4. Deliverable format

**Location:** `research/house_localization/roof_budget/` (this folder), alongside the two theory workplans:

```
roof_budget/
  01_junction_theory_workplan.md      (other agent)
  02_path_budget_workplan.md          (other agent)
  03_inputs_and_criteria.md           (this file)
  04_verification_note.md             (the Step-1 result)
  roof_budget.py                      (single script, stdlib+numpy, reproduces every table)
  BUDGET_v1.json                      (machine-readable per-band budget + corner cases)
```

**Structure of `04_verification_note.md`:**
1. Header: date, inputs hash (frozen table §1.2 values actually used), scope.
2. Frozen inputs table (copy of §1.2 with chosen values, deviations flagged).
3. Geometry diagram (ASCII) + source/receiver coordinates.
4. **Per-third-octave budget table** — rows = 10 bands; columns = source-slab spreading+damping to junction, Kij→Dv junction 1, wall leg, Kij→Dv junction 2, roof spreading+damping to receiver, column-path parallel contribution, **total ΔL (dB)** — one table per (source, receiver) pair (S1→R1, S1→R2, S2→R3).
5. Uncertainty: low-η/low-Kij vs high-η/high-Kij corner rows per band (or a small Monte-Carlo over the §1.2 ranges — 1,000 samples × 10 bands × 3 paths is O(10⁴) evaluations of closed-form expressions, < 1 s compute, < 10 MB memory; corner cases suffice if MC adds nothing).
6. Junction cross-check table vs EN 12354 Annex E (criterion 2).
7. Method-agreement table 01-chain vs 02-budget (criterion 3).
8. **Verdict table C1–C6** with replacement numbers and the exact doc 04/05/14 sentences to edit.
9. Limitations: mean-field/SEA validity in the sparse-modal regime (doc 05: M < 2 below 200 Hz — quote per-band modal overlap so the reader knows where the mean-field number is a smoothed estimate, not a modal prediction), and the follow-up hook (one Devito spot-check run, out of Step-1 scope).

---

## 5. Effort estimate (once workplans 01 and 02 exist)

| Task | Effort |
|---|---|
| Freeze inputs (§1.2) incl. resolving M1–M7 decisions (mostly picking nominal + range, 2 short source lookups: SI 1752 roof build-up, parapet detail) | 0.5 day |
| Implement `roof_budget.py` (closed-form: plate spreading, η attenuation, Annex E Kij, chain sum, corner cases; ~200–300 lines) | 0.5–1 day |
| Cross-checks (criteria 2–3) + reconciliation with agents 01/02 if > 6 dB gap | 0.5 day |
| Write `04_verification_note.md` + edit list for docs 04/05/14 | 0.5 day |
| **Total** | **2–2.5 person-days**, negligible compute |

Main schedule risk: M5 (load-path variant) doubling the junction bookkeeping, and C2 (wind) needing a short literature pull for in-band wind-induced roof vibration levels.
