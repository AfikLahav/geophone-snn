# Junction Transmission Theory — Workplan

> **Work outline — roof-attenuation verification ("wallhacks").** Planning agent, 2026-07-09.
> **Status: WORKPLAN — no computation performed yet.**
> **Purpose:** Define the calculation steps that produce, from first principles (wave approach for thin-plate junctions), the per-band junction transmission losses for the slab→wall and wall→roof junctions of the canonical house (doc `03_inputs_and_criteria.md` §2), in the exact quantities workplan `02_path_budget_workplan.md` consumes. This document also **resolves workplan 02's dealbreaker #5** (Kij vs Dv,ij vs τ definition mismatch) with exact conversion formulas (§0).
> Companions: `02_path_budget_workplan.md` (propagation/assembly), `03_inputs_and_criteria.md` (frozen inputs, criteria). Band set, geometry, and material table are taken from those two documents and not redefined here.

---

## 0. Quantity definitions and exact conversion chain (fixes the definition mismatch)

Three distinct quantities appear in the literature. They are NOT interchangeable, and the budget must consume the right one.

**(Q1) Angular-averaged transmission coefficient τ_ij(f)** — ratio of transmitted to incident bending-wave *power* for a diffuse (angle-averaged) bending field on semi-infinite plates meeting at a line junction. Direction-dependent (τ_ij ≠ τ_ji). This is what the wave theory produces (Step 1).

**(Q2) Velocity level differences.** Per ISO 10848-1:2017 §3.10–3.11 (definitions read verbatim from the standard's public preview, https://cdn.standards.iteh.ai/samples/67226/d725452310cf4618ad5baf4c194310cf/ISO-10848-1-2017.pdf):
- `Dv,ij` = difference of average velocity levels of elements i and j when only i is excited (dB). **This — not Kij — is the per-junction path loss the budget chain needs**, because it depends on the receiving element's total damping and area (it is situation-dependent).
- Direction-averaged: `D̄v,ij = (Dv,ij + Dv,ji)/2`.

**(Q3) Vibration reduction index Kij** — the *invariant* junction quantity of EN 12354-1 / ISO 10848-1 §3.13:

```
Kij = D̄v,ij + 10·lg( l_ij / sqrt(a_i · a_j) )          [ISO 10848-1:2017, 3.13]
a_j = (2.2·π²·S_j / (T_s,j · c_0)) · sqrt(f_ref / f),   f_ref = 1000 Hz   [ISO 10848-1:2017, 3.12]
T_s,j = 2.2 / (f · η_tot,j)   →   a_j = π²·S_j·η_tot,j·sqrt(f·f_ref) / c_0
```

with l_ij = junction line length (m), S_j = element area (m²), a_j = "equivalent absorption length" (m), c_0 = 343 m/s. Kij is junction-normalized: **using Kij directly as a chain loss is wrong** unless it is first converted back to the in-situ Dv,ij:

```
Dv,ij(situ) = Kij − 10·lg( l_ij / sqrt(a_i,situ · a_j,situ) )        [EN 12354-1 §4.2]
```

**Conversion (Q1)→(Q2), the formula the budget uses.** From first-order SEA of two coupled plates (source in i, weak coupling), with E = m″·S·⟨v²⟩ and the line-junction coupling loss factor:

```
η_ij = c_g,i · l_ij · τ_ij / (π · ω · S_i)                     (Craik; Hopkins eq. 4.x [VERIFY eq. no.])
⟨v_j²⟩/⟨v_i²⟩ = (η_ij / η_tot,j) · (m″_i·S_i)/(m″_j·S_j)
Dv,ij = 10·lg( π·ω·η_tot,j·S_j·m″_j / (c_g,i·l_ij·τ_ij·m″_i) )
```

**Conversion (Q1)→(Q3), for the Annex-E cross-check only.** Substituting the two lines above plus the a_j formula into the Kij definition, all S, η, m″ terms cancel and (using c_g = 2·c_0·sqrt(f/f_c)):

```
Kij = −10·lg( sqrt(τ_ij·τ_ji) ) + 5·lg( sqrt(f_c,i·f_c,j) / f_ref )
```

This closed form was derived while planning and **must be re-derived and unit-tested in Step 4** (numeric self-check given there) before use — it is the load-bearing identity of the whole cross-check. [VERIFY against Hopkins (2007) §5.2, which gives the τ↔Kij relation for semi-infinite plates.]

**Handoff rule to workplan 02:** this workstream delivers per-band `Dv,ij(situ)` (with low/high η_tot corners) AND per-band `τ_ij` (for 02's SEA matrix, Step 5.2 there). Kij appears only in the cross-check table (Step 7). State this in the deliverable header.

---

## Step 1 — Assemble the wave-theory formula set (bending + in-plane, L/T/X)

**Goal.** The explicit boundary-value problem and closed-form (or numerically solvable) transmission coefficients τ(θ, f) for a plane bending wave hitting a line junction of 2 (L), 3 (T), or 4 (X) semi-infinite thin plates, including mode conversion to quasi-longitudinal (L) and transverse shear (T) in-plane waves.

**Theory structure (what to extract from the sources):**
- Incident bending wave at angle θ on plate i; trace-wavenumber matching along the junction line (`k_sinθ` continuity) determines transmitted/reflected angles per wave type; evanescent bending near-field terms must be kept.
- Junction conditions at the line (rigid junction model): continuity of translational velocities and angular velocity, equilibrium of forces and moments — 4 continuity + 4 equilibrium relations per junction, giving a linear system in the unknown wave amplitudes (bending propagating + bending near-field + longitudinal + shear, per plate).
- τ_ij(θ,f) = transmitted/incident power ratio per receiving wave type; angular average over diffuse incidence: `τ̄_ij = ∫₀^{π/2} τ_ij(θ)·cosθ dθ / ∫₀^{π/2} cosθ dθ` — **state the cosθ-weighting convention explicitly in code and docs**; it must match the convention embedded in η_ij = c_g·l·τ/(π·ω·S), or the SEA chain is silently off by a constant.
- Parameter dependence to expose in the implementation: thickness ratio / surface-mass ratio ψ = m″_j/m″_i, c_L ratio (material), frequency (enters via k_B ∝ √f — for same-material plates the bending-only τ of a rigid junction is frequency-flat; frequency dependence enters through mode conversion and dissimilar materials).

**Sources (in order of preference):**
1. Hopkins, *Sound Insulation* (2007), ch. 5 — wave approach, angular-averaged bending transmission with conversion to in-plane waves, L/T/X junctions. (Book; library/PDF access.)
2. **Open-access substitute with the full derivation and formulas:** Yin Jia, *Structure-borne sound transmission between isotropic, homogeneous plates*, PhD thesis (Liverpool, Hopkins group, 2012) — https://livrepository.liverpool.ac.uk/8193/4/YinJia_Nov2012_8193.pdf — contains the angular-average transmission coefficients with wave conversion; use as the primary implementable reference if Hopkins' book is not at hand. [VERIFY it covers T and X, not only L — skim §2 first; if only L, fall back to Hopkins/Cremer for T/X.]
3. Cremer, Heckl & Petersson, *Structure-Borne Sound* (3rd ed.), plate-junction chapter — canonical derivation, normal-incidence special cases for validation.
4. Wave conversion validation data: Experimental SEA of coupled plates with wave conversion — https://www.sciencedirect.com/science/article/abs/pii/S0022460X08009061

**Inputs.** None beyond doc 03 §1.2 material table. **Outputs.** A one-page formula memo (equations + variable glossary + sign/normalization conventions) checked against two sources. **Effort.** 0.5–1 day (the conventions comparison is the slow part). **Risks.** Sign/normalization conventions differ between Cremer and Hopkins (velocity vs displacement amplitudes, power normalization); mixing them is the classic silent error — the memo must commit to one set.

## Step 2 — Implement and unit-test the junction solver

**Goal.** `roof_budget/junction_tau.py` (stdlib + numpy): input = per-plate {h, ρ, E, ν, η}, junction type (L/T/X) and geometry; output = τ̄_ij(f) matrix per third-octave band (11 bands, doc 02 §0), per wave-type pair (B→B, B→L, B→T, L→B, L→L, ...).

**Mandatory unit tests (gate before any downstream use):**
1. Energy conservation: Σ_j Σ_types τ_ij(θ) + r_i(θ) = 1 at every θ, every band, to 1e-10.
2. SEA reciprocity (consistency of τ_ij vs τ_ji with the η_ij·n_i = η_ji·n_j identity, n = modal density).
3. Limiting cases from Cremer/Heckl: identical-plate X and L junction bending-only τ at normal incidence vs the book's closed forms.
4. Annex E reproduction: for equal-mass rigid concrete T and X, computed Kij (via §0 conversion) within ±3 dB of 5.7 dB (T) / 8.7 dB (X) — the doc 03 P13 anchors.

**Effort.** 1 day. **Risks.** Root-finding/branch selection for evanescent wavenumbers above/below in-plane cut-on angles; complex arithmetic bugs pass test 1 but fail test 3 — keep both. Compute cost is trivial: linear solves of ~8–16 unknowns × 11 bands × ~90 angles ≈ O(10⁴) 16×16 solves, <1 s, <10 MB.

## Step 3 — Junction typing for OUR two junctions (compute bounds, not one number)

**Junction 1 — ground slab → exterior wall (perimeter).** Complicated by the foundation and soil. Prescribe THREE variants, reported side by side:
- (1a) **Rigid T**, slab straight-through (slab-on-grade continuous under the wall), wall as branch. Prior expectation from Annex E branch formula: M = lg(m″_wall/m″_slab) = lg(220/480) ≈ −0.34 → K12 ≈ 5.7 + 5.7·M² ≈ 6.4 dB (prior only — recompute).
- (1b) **Rigid X** — add the foundation/stem-wall below as a fourth semi-infinite leg: upper bound on junction loss for the slab→wall path.
- (1c) **T + soil-leakage term**: model soil contact as added junction absorption — implement as a semi-infinite lossy leg with soil-derived impedance (doc 03 P12: Vs = 200 m/s, ρ = 1800) or, simpler, as an additive imaginary term in the junction line impedance. Bounds the claim "some incident power leaks to ground instead of the wall". If (1c) differs from (1a) by <2 dB in all bands, drop it from the budget and note it.

**Junction 2 — wall → roof slab.** Two variants as bounds (doc 03 M3):
- (2a) **L junction** (wall terminates at roof edge, no parapet): prior K12 ≈ 15·|M| − 3 with M = lg(480/220) ≈ 0.34 → ≈ 2.1 dB [VERIFY corner formula coefficients, §7]. Note this is *lower* loss than a T — the no-parapet case is the adversarial one for the 20–40 dB claim.
- (2b) **T junction** (RC parapet continues above roof line, 150 mm; wall→parapet straight, roof as branch — per doc 03 §2 canonical parapet 1.05 m × 150 mm). Apply Annex E's exact path-element convention for which element is "perpendicular" (m″_⊥) — get this from the standard text, do not guess.
- Also run (2b′) with the ring-beam/RC-column varia­nt flagged as doc 03 M5: the wall→roof junction in Israeli practice passes through an RC ring beam; if time allows, bound it by re-running (2b) with the wall leg's properties replaced by RC over a 200 mm strip. Otherwise carry M5 as an unmodeled systematic.

**Inputs.** Doc 03 §1.2 (P3, P8, P9, P13), §2 geometry. **Outputs.** τ̄ matrices + Dv,ij + Kij for variants 1a–1c, 2a–2b, per band; the budget consumes the (min, max) envelope per junction. **Effort.** 0.5 day (solver exists after Step 2). **Risks.** Soil-leg impedance model is crude — mark 1c as an order-of-magnitude bound, not a prediction; hollow-block wall treated as homogeneous plate (±50% E_dyn per doc 02 Step 1) — propagate both corners through τ.

## Step 4 — Execute the conversion chain and verify the Kij identity

**Goal.** Turn Step 3's τ̄ into the two deliverable quantities per §0.

1. **Derive-and-test the identity** `Kij = −10·lg(sqrt(τ_ij·τ_ji)) + 5·lg(sqrt(f_c,i·f_c,j)/f_ref)` symbolically (half a page, from the four §0 equations). **Numeric self-check:** identical 200 mm RC plates (f_c ≈ 91 Hz), rigid T: identity ⇒ τ̄ ≈ 0.08 corresponds to Kij = 5.7 dB; verify the Step-2 solver's equal-mass T yields τ̄ in that neighborhood. If the identity or the solver misses by >3 dB, STOP — either the a_j substitution, the cosθ convention, or the CLF formula is wrong; reconcile before anything is handed to workplan 02.
2. **Compute Dv,ij(situ)** per band for both junctions with in-situ η_tot corners from doc 03 P5–P7 (low-η and high-η rows). Illustrative prior of why Dv ≠ Kij for our house: slab S = 120 m², η_tot = 0.1 @ 100 Hz gives a_slab ≈ 110 m; with l_ij = 12 m and a_wall ≈ 30 m, the normalization term 10·lg(l/√(a_i·a_j)) ≈ −7 dB, i.e. **Dv,ij(situ) ≈ Kij + 7 dB** — a 7 dB systematic that silently corrupts the budget if Kij is used raw. (Prior — recompute with frozen inputs.)
3. Emit `JUNCTIONS_v1.json`: per band × junction variant × direction: {τ̄ by wave-type pair, η_ij, Dv,ij(situ) low/high, Kij} + input hash. This file is workplan 02 Step 5's external input.

**Effort.** 0.5 day. **Risks.** η_tot double-counting at the interface with workplan 02 (its Step 4 trap #2): Dv,ij(situ) already contains the receiving element's η_tot — workplan 02's SEA route must use τ/η_ij, NOT Dv, inside the matrix (Dv is for the ray-chain route only). Write this rule into the JSON header.

## Step 5 — In-plane bypass bound (mode-conversion parallel path)

**Goal.** Bound the path: bending (slab) → B→L/T conversion at junction 1 → in-plane wave up the wall (in-plane damping and spreading are far lower; over 2.75 m the in-plane leg loses ≈0 dB) → L/T→B re-conversion at junction 2 → bending (roof). This parallel path can bypass the nominal two-junction bending attenuation.

**Method (worst-case, no new theory needed — Steps 1–2 already produce the cross-type τ):**
1. From the solver: τ_B→L and τ_B→T at junction 1 (slab→wall), τ_L→B and τ_T→B at junction 2 (wall→roof), per band.
2. Worst-case bypass attenuation: `A_bypass = −10·lg(τ_B→IP(j1) · τ_IP→B(j2))` assuming zero in-plane propagation loss in the wall (conservative) and no in-plane re-reflection bookkeeping.
3. Compare per band with the nominal chain `A_nominal = −10·lg(τ_BB(j1)·τ_BB(j2)) + wall bending leg (from workplan 02 Step 4)`. **Decision rule:** if A_bypass < A_nominal + 10 dB in any band, the bypass is significant → it must enter workplan 02's Step 5 assembly as a parallel energy path (add τ_B→IP, τ_IP→B rows to the JSON); otherwise document as negligible with the numbers.
4. Secondary check: in-plane → in-plane through both junctions with radiation into roof bending distributed over the roof's other junctions — only if (3) triggers.
5. Literature sanity anchor: in-plane contribution to flanking in masonry/concrete buildings becomes important after 1–2 junctions (Craik's SEA-of-buildings work on in-plane waves [VERIFY exact reference — Craik & Thancanamootoo, Applied Acoustics, ~1992; or Craik, *Sound Transmission Through Buildings Using SEA*, ch. 6]); the wave-conversion experiment https://www.sciencedirect.com/science/article/abs/pii/S0022460X08009061 for magnitudes.

**Effort.** 0.5 day. **Risks.** This is the most likely dealbreaker for the 20–40 dB claim (together with workplan 02's RC-column path, which is the *other* bypass; they are separate paths — do not merge the bookkeeping). Overly conservative worst-case may "trigger" spuriously — if step 3 triggers by <3 dB, refine with the wall's actual in-plane reverberant balance before escalating.

## Step 6 — Validity limits of the junction model

1. **Thin-plate check (do first, it's arithmetic):** require h < λ_B/6 at 200 Hz. Priors: 200 mm RC slab: c_B(200 Hz) ≈ √(1.8·h·c_L·f) ≈ 509 m/s → λ_B ≈ 2.5 m → λ_B/6 ≈ 0.42 m > 0.2 m ✓; 200 mm block wall (c_L ≈ 2600 m/s): λ_B(200 Hz) ≈ 2.2 m ✓. Recompute with Step-1-corrected tables from workplan 02 (its doc-05 dispersion fix) and tabulate the margin per band. If any element fails, apply the thick-plate (Mindlin) correction from Hopkins §2 or restrict the affected bands.
2. **Semi-infinite assumption vs modally sparse plates:** the wave approach gives ensemble-mean τ; real finite plates with M < 2 fluctuate around it (±10 dB scale). **Do NOT quantify here** — workplan 02 Step 6 (Lyon/Langley-Cotoni variance) owns that; this doc only states that τ̄/Dv are mean-field quantities and points there.
3. **Rigid vs pinned junction sensitivity:** re-run Step 3 with the pinned (moment-released) junction condition — Cremer/Hopkins give both. Report ΔKij(rigid−pinned) per junction; if > 5 dB, the junction-detail uncertainty (mortar joint quality, ring beam) dominates and the budget envelope must include both. Blockwork junctions in practice fall between rigid and pinned; Crispin's rigid-junction lab data (§7) anchors the rigid end.
4. **Low-frequency limit:** below ~50 Hz the junction "sees" the foundation and whole-building modes; flag bands 20–40 Hz as indicative-only for junction theory (consistent with doc 05's modal framing) — the Devito FDTD spot-check (out of scope here) is the eventual arbiter there.

**Effort.** 0.5 day (item 3 is a solver re-run). **Risks.** None structural; this step mainly produces caveat text and one sensitivity table.

## Step 7 — Cross-checks (pass/fail per doc 03 criteria)

**Reference set:**
1. **EN 12354-1 Annex E empirical formulas** (rigid junctions, M = lg(m″_⊥,i/m″_i)) — as commonly cited [VERIFY all coefficients against an actual copy of EN 12354-1:2000/2017 Annex E or the numerical catalogue paper below before use]:
   - Rigid T: K13 = 5.7 + 14.1·M + 5.7·M² dB (straight); K12 = 5.7 + 5.7·M² dB (branch)
   - Rigid X: K13 = 8.7 + 17.1·M + 5.7·M² dB; K12 = 8.7 + 5.7·M² dB
   - Corner (L): K12 = 15·|M| − 3 dB, minimum −2 dB [VERIFY — least certain of the three]
   - Also apply the standard's minimum-Kij clause (there is a normative floor value dependent on l_ij and a_i [VERIFY clause]).
   - Backup/extension source with FEM-derived formulas incl. non-standard junctions: "Catalogue of vibration reduction index formulas for heavy junctions based on numerical simulations" — https://www.researchgate.net/publication/324994689_Catalogue_of_vibration_reduction_index_formulas_for_heavy_junctions_based_on_numerical_simulations
2. **Crispin et al. 2006 lab measurements** (rigid concrete junctions: T ≈ 8 dB, X ≈ 7 dB measured) — https://journals.sagepub.com/doi/abs/10.1260/135101006777630427 ; prediction-vs-measurement study for brick/concrete rigid junctions — https://www.sciencedirect.com/science/article/abs/pii/S0003682X10000083
3. **Doc 04's 5–10 dB per-wall-junction claim** (project-internal).

**Pass/fail exactly per doc 03 §3.2 criterion 2:** each per-junction Kij used in the chain must agree with the Annex E default for that junction type and actual mass ratio within **±3 dB** (the documented lab-vs-prediction scatter). Additionally settle doc 03 claims C5 (τ > 0.5 below 50 Hz for the rigid floor–wall T — our prior τ̄ ≈ 0.08–0.3 says this likely FAILS as written; report the computed number) and C6 (floor→wall Kij = 4–8 dB at the actual 480/220 mass ratio). For doc 04's 5–10 dB/corner: compare against Dv,ij(situ) — not Kij — since doc 04 speaks of observed level drops; note in the verdict if the L-junction (no-parapet wall→roof, prior ≈ 2–3 dB Kij) undercuts the claimed range, because that lowers total roof attenuation below the claim's implicit budget.

**Outputs.** Cross-check table (junction × variant × band: ours vs Annex E vs Crispin, Δ, pass/fail) feeding doc 03's criterion-2 gate and workplan 02 Step 7. **Effort.** 0.5 day.

## Step 8 — Deliverables, order, effort

```
roof_budget/
  junction_tau.py        Step 2–5 solver + unit tests (numpy only)
  JUNCTIONS_v1.json      per-band τ̄ (all wave-type pairs), η_ij, Dv,ij(situ) corners, Kij, input hash
  01b_junction_note.md   formula memo (Step 1) + variants table (Step 3) + identity proof (Step 4)
                         + bypass verdict (Step 5) + validity table (Step 6) + cross-check table (Step 7)
```

Order: 1 → 2 → {3, 4} → 5 → 6 → 7 (6.1 can run any time after workplan 02's Step 1 tables exist). Total **~3.5–4 working days**, negligible compute (all closed-form/small linear solves; O(10⁴–10⁵) 16×16 solves total, <10 s, <100 MB).

---

## Dealbreaker watchlist (junction piece)

1. **Kij used as a path loss** (§0/Step 4): skipping the Kij→Dv,ij(situ) back-conversion injects a ~5–10 dB systematic (prior: ≈7 dB for our slab at 100 Hz). This is workplan 02's risk #5 — resolved by the §0 formulas, enforced by the Step 4 self-check.
2. **In-plane bypass** (Step 5): if τ_B→IP·τ_IP→B is within 10 dB of the bending chain, two-junction bending attenuation does not govern and the claim verification must include the in-plane path.
3. **L-junction wall→roof (no parapet)** (Step 3): prior Kij ≈ 2–3 dB — if confirmed, the "each junction costs 5–10 dB" claim fails for that junction type and the total budget shifts low.
4. **Convention mismatch** (Steps 1–2): cosθ-weighting of τ̄ vs the CLF formula, and Cremer-vs-Hopkins amplitude normalizations — caught only by unit tests 2 and 4; do not skip them.
5. **Rigid-junction assumption** (Step 6.3): if rigid-vs-pinned spread > 5 dB, junction detail uncertainty dominates the whole roof budget envelope.

## Sources

| Topic | Source |
|---|---|
| Kij, D̄v,ij, a_j exact definitions (quoted §0) | ISO 10848-1:2017 preview PDF — https://cdn.standards.iteh.ai/samples/67226/d725452310cf4618ad5baf4c194310cf/ISO-10848-1-2017.pdf ; standard page — https://www.iso.org/standard/67226.html |
| Annex E empirical Kij formulas | EN ISO 12354-1:2017 — https://www.iso.org/standard/59114.html [coefficients marked VERIFY]; FEM catalogue — https://www.researchgate.net/publication/324994689_Catalogue_of_vibration_reduction_index_formulas_for_heavy_junctions_based_on_numerical_simulations |
| Wave approach, τ(θ) with in-plane conversion, L/T/X | Hopkins, *Sound Insulation* (2007) ch. 5; open-access: Yin Jia PhD thesis (Liverpool 2012) — https://livrepository.liverpool.ac.uk/8193/4/YinJia_Nov2012_8193.pdf |
| Canonical junction derivations, limiting cases | Cremer, Heckl & Petersson, *Structure-Borne Sound* — https://books.google.com/books/about/Structure_Borne_Sound.html?id=9DzvCAAAQBAJ |
| Measured rigid-junction Kij (T ≈ 8, X ≈ 7 dB) | Crispin et al. 2006 — https://journals.sagepub.com/doi/abs/10.1260/135101006777630427 ; lab-vs-EN-12354 — https://www.researchgate.net/publication/259574153_The_vibration_reduction_index_Kij_laboratory_measurements_versus_predictions_EN_12354-1_2000 |
| Kij prediction for brick/concrete rigid junctions | https://www.sciencedirect.com/science/article/abs/pii/S0003682X10000083 |
| Wave conversion at junctions (experimental SEA) | https://www.sciencedirect.com/science/article/abs/pii/S0022460X08009061 |
| In-plane wave contribution to flanking | Craik, *Sound Transmission Through Buildings Using SEA*; Craik & Thancanamootoo, Applied Acoustics ~1992 [VERIFY exact citation] |
