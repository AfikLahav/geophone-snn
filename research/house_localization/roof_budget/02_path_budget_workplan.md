> **Work outline — roof-attenuation verification ("wallhacks").** Planning agent, 2026-07-09. **Status: WORKPLAN — no computation performed yet.** Purpose: define the calculation steps to verify, from first principles, the claim that a footstep on the ground-floor slab of a single-storey flat-roof RC house arrives at the roof slab attenuated ~20–40 dB total in 20–200 Hz. Scope: source term, in-plate propagation, assembly, modal-fluctuation band, cross-checks. **Junction transmission losses are an external input** (separate agent, expected as per-band dB values with ranges); this plan treats them as given numbers. Companion doc: `01_*` (junction agent output). Geometry/material defaults from [09_procedural_house_generation.md](../09_procedural_house_generation.md); modal context from [05_structural_reverberation.md](../05_structural_reverberation.md).

---

# Work Plan: Source-to-Roof Vibration Path Budget (excluding junction theory)

## 0. Fixed conventions (decide once, use everywhere)

- **Reference quantity:** third-octave-band RMS velocity level `L_v = 20·log10(v_rms / 1 nm/s)` of a **single footstep event**, evaluated at **r_ref = 1 m** from the impact point on the 200 mm ground slab. The budget output is the **attenuation** `A(f) = L_v(1 m, slab) − L_v(roof slab, above-footprint point)` per band. The claim to test: `A(f) ≈ 20–40 dB` across 20–200 Hz.
- **Bands:** IEC third-octave centers 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200 Hz (11 bands). Rationale: matches building-acoustics convention (EN ISO 12354 / ISO 10848 data are third-octave), wide enough to average 0–2 modes/band, matches doc 05's modal-overlap framing.
- **Geometry (from doc 09):** 200 mm RC slab and roof slab (E=30 GPa, ρ=2400 kg/m³, ν=0.2); walls = 200 mm hollow-block (ρ≈1100, Vs≈1000 m/s) **plus RC columns 200×400 mm on a 3.5–5 m grid**; clear height 2.75 m; footprint 10×12 m nominal. η_material(RC) = 0.01–0.02; η_total(in-situ) = 0.03–0.10 (doc 05 §1.5).
- **Nominal path:** slab (1–5 m horizontal) → slab/wall junction → wall (2.75 m vertical) → wall/roof junction → roof slab (1–3 m horizontal). Junction losses = external input.

---

## Step 1 — Rebuild and verify the dispersion/material tables (do this FIRST)

**Goal.** Correct per-band `c_B(f)`, `c_g(f)`, bending stiffness `D`, surface mass `m"` for: 200 mm RC slab, 200 mm RC roof slab, 200 mm hollow-block wall, and longitudinal speed for the RC column path.

**Inputs.** Doc 09 material table; thin-plate formulas `c_B = (2πf)^{1/2}(D/m")^{1/4}`, `c_g = 2c_B`, `D = Eh³/12(1−ν²)`; equivalent form `c_B = √(1.8·h·c_L·f)` with `c_L = √(E/ρ(1−ν²))`.

**Outputs.** One table: 11 bands × 4 elements × {c_B, c_g, λ_B, k·r at 1 m}.

**Verification hook (mandatory).** Coincidence-frequency consistency: `c_B(f_c) = 343 m/s`. For 200 mm RC (c_L≈3600 m/s) f_c ≈ 93 Hz; for 150 mm ≈ 124 Hz.

**⚠ Known issue found while planning:** doc 05's dispersion table (§1.2) is internally inconsistent — it lists c_B(150 Hz)=263 m/s for a 150 mm slab while also stating f_c=123 Hz, but c_B at coincidence must equal 343 m/s. Recomputation gives c_B(100 Hz)≈310 m/s (150 mm) and ≈360 m/s (200 mm), i.e. doc 05 understates c_B by ~1.44×. Since damping loss and spreading-law radii scale with c_B/c_g, fix this before anything downstream. Also flag the correction back to doc 05.

**Effort.** 1–2 h. **Could go wrong:** hollow-block wall is not a homogeneous plate — effective E_dyn (2–3 GPa from Vs≈1000 m/s effective medium) has ±50% uncertainty; carry as parameter range, not point value.

---

## Step 2 — Source term: footstep force spectrum in third-octave bands

**Goal.** Band-resolved force excitation of a single footstep on stiff concrete, F²(f) per band, for a nominal 700–750 N walker at ~2 Hz step rate.

**Method.** Two-component model, both citable:
1. **Walking-harmonic (Fourier/DLF) component** — `F(t) = W[1 + Σ αᵢ sin(2πi·f_s·t + φᵢ)]`, α₁≈0.4–0.56 at 2 Hz, α₂–α₄≈0.05–0.1 (Kerr's dataset, Young's frequency-dependent fits). **Key realization: harmonics reach only ~8–10 Hz — below the 20–200 Hz band.** Include it only to show it contributes nothing in-band; do not let it silently dominate the budget.
2. **Heel-strike transient (in-band energy carrier)** — model as an effective impulse. Use the Arup/Willford–Young effective-impulse formulation (CCIP-016; `I_eff` empirical fit vs pace rate and mode frequency, valid where floor modes > 4× pace rate — true for all bands here) to set the impulse magnitude (~1–10 N·s scale), and Ekimov & Sabatier's measured footstep spectra to set the spectral shape/rolloff of the transient above 20 Hz (heel-strike duration 0–33 ms → energy concentrated below ~100–300 Hz).

**Citable sources (2–3 minimum):**
- Kerr/Young DLFs, as tabulated in Živanović, Pavić & Reynolds 2005 JSV review — https://wrap.warwick.ac.uk/id/eprint/40683/1/WRAP_Zivanovic_0872117-es-211211-2005_jsv_sz.pdf
- Willford, Field & Young effective-impulse model / CCIP-016 "A Design Guide for Footfall Induced Vibration of Structures"; accessible summary: Brownjohn & Pavić, "Simplified Methods for Estimating the Response of Floors to a Footfall" — https://www.researchgate.net/profile/James-Brownjohn/publication/228759176_Simplified_Methods_for_Estimating_the_Response_of_Floors_to_a_Footfall/links/0deec53c946ce980cc000000/Simplified-Methods-for-Estimating-the-Response-of-Floors-to-a-Footfall.pdf ; parameter tables also in SCIA CCIP-016 doc — https://help.scia.net/26.0/en/analysis/modal_analysis_and_dynamics_and_seismicity/dynamics_basics/footfall/ccip_016_coeff_phase.htm and https://www.steelconstruction.info/Floor_vibrations
- Ekimov & Sabatier 2006 JASA, "Vibration and sound signatures of human footsteps in buildings" — https://www.researchgate.net/publication/6849688_Vibration_and_sound_signatures_of_human_footsteps_in_buildings
- Supporting: single-footstep Fourier decomposition — https://www.researchgate.net/publication/314704949_Analysis_of_Vertical_Walking_Force_of_Single_Footstep ; pedestrian load-model review — https://www.mdpi.com/2571-631X/2/1/1

**Outputs.** Per-band footstep force ESD (energy spectral density, N²·s per band) + declared convention (per-event, not walking-average PSD).

**Effort.** ~0.5 day (mostly literature value extraction). **Could go wrong:** (a) mixing per-step ESD with continuous-walking PSD conventions → silent ~5–10 dB error; (b) DLF-only model gives ≈0 in-band energy and the whole budget degenerates — the transient component is mandatory; (c) hard/soft footwear spread ~±5–10 dB above ~100 Hz — carry as source uncertainty.

## Step 3 — Injection: power into the slab and the 1 m reference level

**Goal.** Injected bending-wave power per band and `L_v(1 m)` — the anchor of the whole budget.

**Method.**
- Driving-point mobility of an infinite thin plate (real, frequency-flat): `Y = 1/(8√(D·m"))` (Cremer & Heckl, Structure-Borne Sound — https://books.google.com/books/about/Structure_Borne_Sound.html?id=9DzvCAAAQBAJ). For 200 mm RC: Y ≈ 1.25×10⁻⁶ m/s/N. Injected band power `P = F²_rms·Re(Y)`.
- Velocity at 1 m: far-field cylindrical direct field `v²(r) = P/(2π·r·ρh·c_g)` **plus** near-field correction — at 20 Hz, λ_B ≈ 8 m so k·r ≈ 0.8 at 1 m: the 1 m point is inside the near field for bands ≤ 50 Hz. Use the exact point-driven infinite-plate response (Hankel-function form, Cremer & Heckl §V) rather than the far-field law for those bands.
- Add the reverberant-floor contribution at 1 m (from Step 4's energy balance) — at low bands it is not negligible even at 1 m.

**Empirical anchor (mandatory).** Compare predicted `L_v(1 m)` to measured indoor footstep velocities on concrete, ~200–2,000 nm/s at 1 m in 20–80 Hz (Mirshekari et al., https://par.nsf.gov/servlets/purl/10057760 ; doc 14 §1.2). If the prediction is outside ~±10 dB of this window, stop and debug before proceeding.

**Outputs.** Per-band P_in and L_v(1 m) with uncertainty range. **Effort.** 2–3 h. **Could go wrong:** finite-slab mobility fluctuates ±5 dB around infinite-plate Y in the sparse-modal regime (handled by Step 6, but note it applies to injection too); slab-on-grade ground contact adds real (radiation-into-soil) loading that lowers effective mobility at low f — flag as a systematic, bound with a soil-spring estimate.

---

## Step 4 — Propagation within each plate segment

**Goal.** Per-band attenuation across (a) slab from 1 m to the wall line (nominal 2–5 m, use 3 m; sweep 1–6 m), (b) wall over 2.75 m height, (c) roof slab 1–3 m to evaluation point.

**Method — compute BOTH regimes and take the correct one per band:**
1. **Direct field:** cylindrical spreading `ΔL = 10·log10(r₂/r₁)` dB (3 dB per doubling, velocity-squared basis) + damping `Δ = 27.3·η·f/c_g` dB/m (temporal decay ηω converted to space at **group** velocity). With η_material=0.01–0.02 this is small: ~0.06 dB/m at 100 Hz on the 200 mm slab (~0.3 dB over 5 m); wall (η=0.02–0.08, slower c_g): ~0.3–1 dB over 2.75 m.
2. **Direct-field radius / reverberant takeover:** from the plate energy balance `P_in = ω·η·E_rev` with `E_rev = ρhA·⟨v²_rev⟩` vs direct intensity `⟨v²_dir⟩(r) = P/(2πr·ρh·c_g)`, equate to get **`r_d = η·f·A/c_g`** (linear in A and η — *not* the sqrt form sketched in earlier notes; derive cleanly and sanity-check against Craik's SEA-of-buildings treatment). Numbers: 100 m² slab at 100 Hz → r_d ≈ 2 m (η=0.015) to 7 m (η_total=0.05); at 20 Hz → r_d < 0.5 m. **Conclusion to verify: at most bands and distances the field a receiver sees is reverberant, and per-meter attenuation saturates** — the mean level difference between plates, not dB/m within a plate, is what governs the budget.
3. **Reverberant (SEA) segment levels:** within one plate, no further distance decay beyond r_d; level set by the subsystem energy balance (Step 5).

**Outputs.** Per-band, per-segment: r_d, direct-path loss, and the flag "direct" or "reverberant" for the nominal geometry. **Effort.** ~0.5 day.

**⚠ Two known traps (found while planning; both present in doc 14 §1.1):**
- **c_B vs c_g in the damping formula.** `27.3·η·f/c` with phase velocity overstates damping loss ×2. Doc 14 used c_B *and* the understated doc 05 c_B values — its α_dB/m is ~3× too high with material η conventions.
- **η double-counting.** η_total(in-situ 0.03–0.10) already *includes* coupling/edge losses. Using η_total for in-plate damping **and** adding explicit junction losses counts the same energy twice. Budget rule: use η_material (0.01–0.02 RC) for propagation + explicit junction losses; use η_total only inside the SEA balance where coupling is otherwise unmodeled. State the rule in the results doc.

## Step 5 — Assembly: total source→roof budget

**Goal.** Combine Steps 2–4 with the junction agent's per-band transmission losses into `A(f)` with uncertainty bands.

**Method — two independent estimates, reported side by side:**
1. **Ray/segment chain (upper-bound level at roof):** `L_v(roof) = L_v(1 m) − [spreading+damping slab] − ΔL_junction1 − [wall] − ΔL_junction2 − [roof spreading]`. Single shortest path.
2. **SEA power-balance chain (mean-field reverberant estimate):** 6 subsystems (slab, 4 walls, roof); coupling loss factors from the junction agent's transmission coefficients via `η_ij = (c_g·L_ij·τ_ij)/(π·ω·A_i)` (L_ij = junction line length); solve the linear SEA matrix per band for E_roof/E_slab → velocity level difference. This captures what the ray chain misses: the roof receives power from **all four walls in parallel** (expect 3–6 dB less attenuation than the single-path chain), and re-injection from reverberant buildup.
3. **RC column short-circuit path (mandatory add-on):** columns connect slab to roof quasi-directly (longitudinal/flexural waves, c_L≈3600 m/s, low damping, small cross-section but stiff). Estimate power collected by a column base from the slab reverberant field, transmitted up 2.75 m, re-injected into the roof (column–plate mobility mismatch at both ends). If this path is within 10 dB of the plate path, the plate-only budget is invalid and the claim verification must include it.
- **Uncertainty:** Monte Carlo (~10⁴ draws) over: η_material ∈ [0.01, 0.02] (log-uniform), wall E_dyn ±50%, slab h ∈ [180, 220] mm, source-to-wall distance 1–6 m, junction losses at the range the junction agent supplies, source ESD ±5 dB. Report per-band median and 10–90%.

**Verdict format vs the claim:** per band, state `A_median` and `A_10–90%`; claim "confirmed" for a band if [20, 40] dB overlaps the 10–90% interval; report which bands (if any) fall outside and why.

**Outputs.** Per-band budget table (each term separately + total), waterfall decomposition, verdict row. **Effort.** ~1 day incl. a small Python script (`roof_budget/path_budget.py` emitting JSON + MD table — same datasheet pattern as `gap_study/v4_plan/`). **Could go wrong:** SEA formally invalid at M<2 (this is exactly why Step 6 exists — SEA supplies the *mean*, Step 6 the spread); junction agent's numbers may be defined per-junction-crossing vs per-path — reconcile definitions in writing before combining; flat-roof parapet/edge beams not in the model (note as unmodeled systematic, ±2–3 dB).

## Step 6 — Modal-sparseness fluctuation band (the answer must be a band, not a point)

**Goal.** Bound position/frequency fluctuation of `A(f)` around the mean-field estimate, given doc 05's finding that M < 2 everywhere in-band (M ≈ 0.4–0.8 at 50–100 Hz for a 100 m² floor).

**Theory to apply:**
- **Lyon's SEA variance results** (Lyon & DeJong, *Theory and Application of Statistical Energy Analysis* — https://books.google.com/books/about/Theory_and_Application_of_Statistical_En.html?id=pw9EAQAAIAAJ): relative variance of band-averaged energy scales as ~1/M → `σ_dB ≈ 4.34·√(1/M)` per subsystem as the first-order structure. At M=0.4 → σ≈7 dB; M=1 → σ≈4.3 dB; M=2 → σ≈3 dB.
- **Langley & Cotoni 2004** (JASA 115(2):706–718, "Response variance prediction in the statistical energy analysis of built-up systems" — https://pubs.aip.org/asa/jasa/article-abstract/115/2/706/546403/ ; https://pubmed.ncbi.nlm.nih.gov/15000183/): proper variance prediction for chained subsystems incl. point-response variance (Gaussian Orthogonal Ensemble statistics, α parameter for point loading); use their formulas rather than naive per-subsystem RSS if effort allows. Experimental validation: Andrade et al. 2019 — https://journals.sagepub.com/doi/abs/10.1177/0954406219843571
- Combine variances along the chain: injection (source sits on modes/antinodes), slab, wall, roof, plus receiver-position variance on the roof. Expect total 2σ of roughly ±8–12 dB at 20–63 Hz shrinking to ±5–8 dB at 100–200 Hz — these are prior guesses to be replaced by the computed values.

**Outputs.** Per-band ±2σ fluctuation band appended to Step 5's table; final claim statement of the form "A(100 Hz) = X dB (mean-field) ± Y dB (parametric 10–90%) ± Z dB (modal fluctuation, 2σ)". **Effort.** ~0.5 day. **Could go wrong:** GOE/Poisson mode-statistics assumptions are themselves questionable at M≪1 (lowest 2–3 bands) — if formulas blow up, fall back to stating those bands as "modal, deterministic per-house; mean-field number indicative only" and recommend checking against the Devito FDTD model when available.

## Step 7 — Cross-checks

1. **Doc 14 internal consistency:** recompute its −10.2 dB (5 m, 100 Hz, slab) figure with the Step 1 corrected c_B/c_g and the Step 4 η rules. Expected outcome: spreading −7 dB survives; the damping term (−3.2 dB) shrinks to ~−0.3…−1 dB with η_material and c_g, but the reverberant floor (r_d < 5 m) caps effective attenuation anyway — document the reconciliation and correct doc 14 if warranted.
2. **Published field data, storey-to-storey / floor-to-wall level differences in RC/masonry buildings:** Type-A (concrete floor + masonry wall) junction velocity-level decreases < 6 dB per junction (see Svantek building-vibration overview — https://svantek.com/applications/building-vibrations/); direct vs flanking heavy-impact paths ~10 dB (measured) / ~13 dB (SEA) in bearing-wall apartments — https://www.sciencedirect.com/science/article/abs/pii/S0360132325004834; structure-borne transmission measurement/prediction survey (Liverpool/Hopkins) — https://livrepository.liverpool.ac.uk/3077422/1/author%20version.pdf; multilayered concrete structure-borne transmission — https://www.sciencedirect.com/science/article/abs/pii/S0022460X17306910. Sanity requirement: our per-junction + per-segment numbers must not contradict the <6 dB/junction and ~10–13 dB direct-vs-flanking field observations after accounting for geometry.
3. **Total-claim check:** two junctions (junction-agent numbers) + spreading + damping + SEA parallel-path correction should land inside or near 20–40 dB; if the mean-field total lands *below* ~15 dB, the roof may be more useful as a sensor location than doc 04 §4.4 assumes (which claims 20–40 dB and "not useful") — that would be a project-relevant finding, not an error, but demands the column-path and wind-noise caveats be re-examined before any conclusion.
4. **Optional (if findable in ≤1 h):** ISO 10848 / EN ISO 12354-1 flanking `Dv,ij` measured datasets for concrete slab–masonry wall junctions to benchmark the junction-agent inputs independently.

**Effort.** 2–3 h.

## Step 8 — Write-up and integration

Produce `research/house_localization/roof_budget/03_path_budget.md`: per-band budget table, waterfall figure, verdict vs claim, corrections fed back to docs 05/14, and a one-paragraph handoff to the sensor-placement discussion (doc 04 §4.4 roof row). Keep the calculation in a runnable script with pinned inputs (JSON), matching the project's datasheet convention. **Effort.** 2–3 h.

---

## Effort total and order

Steps must run in order 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 (5 blocks on the junction agent's numbers, but 1–4 are independent of it and can start immediately). Total: **~3 working days** of analysis + scripting.

## Dealbreaker watchlist (things that would invalidate the whole exercise)

1. **RC column short-circuit** (Step 5.3): if columns carry comparable power to the plate path, plate-theory budgets alone cannot verify the claim.
2. **η bookkeeping** (Step 4): double-counting coupling losses inside η_total while also charging junction losses can fabricate 10+ dB of attenuation.
3. **Reverberant saturation** (Step 4.2): extrapolating dB/m direct-field laws across a 10 m slab (as doc 14 does) overstates attenuation; if uncorrected, the comparison against the 20–40 dB claim is meaningless.
4. **Doc 05 dispersion-table error** (Step 1): propagates into every c_g-dependent term; must be fixed first.
5. **Junction-loss definition mismatch** with the other agent (velocity-level difference Dv vs transmission coefficient τ vs Kij normalization — Kij is junction-normalized and is NOT directly a path loss); reconcile in writing before assembly.

---

## External sources cited (all URLs)

| Topic | Source |
|---|---|
| Walking DLFs (Kerr, Young fits) | Živanović, Pavić & Reynolds 2005, JSV review — https://wrap.warwick.ac.uk/id/eprint/40683/1/WRAP_Zivanovic_0872117-es-211211-2005_jsv_sz.pdf |
| Pedestrian load models review | https://www.mdpi.com/2571-631X/2/1/1 |
| Single-footstep Fourier decomposition | https://www.researchgate.net/publication/314704949_Analysis_of_Vertical_Walking_Force_of_Single_Footstep |
| Effective impulse (CCIP-016, Willford–Young) | Brownjohn & Pavić — https://www.researchgate.net/profile/James-Brownjohn/publication/228759176_Simplified_Methods_for_Estimating_the_Response_of_Floors_to_a_Footfall/links/0deec53c946ce980cc000000/Simplified-Methods-for-Estimating-the-Response-of-Floors-to-a-Footfall.pdf ; https://help.scia.net/26.0/en/analysis/modal_analysis_and_dynamics_and_seismicity/dynamics_basics/footfall/ccip_016_coeff_phase.htm ; https://openlibrary.org/books/OL25547064M/A_design_guide_for_footfall_induced_vibration_of_structures ; https://www.steelconstruction.info/Floor_vibrations |
| Footstep spectra / heel-strike transient | Ekimov & Sabatier 2006 JASA — https://www.researchgate.net/publication/6849688_Vibration_and_sound_signatures_of_human_footsteps_in_buildings |
| Plate mobility, bending-wave power, near-field | Cremer & Heckl, *Structure-Borne Sound* — https://books.google.com/books/about/Structure_Borne_Sound.html?id=9DzvCAAAQBAJ |
| SEA variance (mean + spread) | Lyon & DeJong — https://books.google.com/books/about/Theory_and_Application_of_Statistical_En.html?id=pw9EAQAAIAAJ ; Langley & Cotoni 2004 JASA 115(2):706 — https://pubs.aip.org/asa/jasa/article-abstract/115/2/706/546403/ ; https://pubmed.ncbi.nlm.nih.gov/15000183/ ; validation Andrade et al. 2019 — https://journals.sagepub.com/doi/abs/10.1177/0954406219843571 ; numerical/experimental validation — https://www.sciencedirect.com/science/article/abs/pii/S0022460X05004578 |
| 1 m footstep velocity anchor | Mirshekari et al. — https://par.nsf.gov/servlets/purl/10057760 |
| Field level differences, RC buildings | https://svantek.com/applications/building-vibrations/ ; https://www.sciencedirect.com/science/article/abs/pii/S0360132325004834 ; https://livrepository.liverpool.ac.uk/3077422/1/author%20version.pdf ; https://www.sciencedirect.com/science/article/abs/pii/S0022460X17306910 |
