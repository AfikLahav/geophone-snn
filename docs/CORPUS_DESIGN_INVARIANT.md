# The Corpus Design Invariant (CDI)

*2026-07-07. The established formula governing every corpus revision (v4.2+). A proposed
change is ON TRACK iff it preserves the factorization and passes the gates — no ad-hoc
debate per revision. Literature verification: `research/v42_verify_diversity_design.md`.
Empirical verification: `gap_study/v4_plan/invariant_check.py` → `INVARIANT_CHECK.json`.*

---

## 1. The formula

Every scene is drawn from the factored distribution:

```
P_train(scene) = U(class) · U(subkind | class) · W(snr) · U(stratum) · R(physics | terrain) · Π_j Q_j(rig_j)
```

| Term | Definition | Rationale (established result) |
|---|---|---|
| **U(class)** | uniform over {human, vehicle, animal, nothing} | train balanced, inject deployment priors at threshold time — Fisher-consistent (logit adjustment, Menon 2021; prior-shift EM, Saerens 2002) |
| **U(subkind\|class)** | mass(class)/n_subkinds(class), equal per subkind — **measured at the WINDOW level**, not scene count | the "likelihood normalized by scene count" rule; the q=0 point of the established `p ∝ n^q` sampling family (Kang 2020; q=0.5 tempering allowed if declared) |
| **W(snr)** | declared difficulty density (default ~uniform-in-dB over [τ_lo−15, +40]; any W permitted **if recorded** for inversion) | GW/speech synthetic-injection practice; importance weights invertible at calibration (Dal Pozzolo 2015; O'Kelly 2018) |
| **U(stratum)** | ~uniform scene mass across terrain-physics strata (~6–10 clusters of the 340 profiles); LHS within stratum | stratified/LHS coverage (McKay 1979); prevents cheap profiles dominating |
| **R(physics\|terrain)** | REAL-ANCHORED: propagation banks, GRF biomechanics, masses, speeds — never uniformized | realism belongs in physics; diversity in nuisances (structured DR, Prakash 2019) |
| **Q_j(rig_j)** | each sensor-chain nuisance j independent of (class, subkind, snr): gain ~ U-in-dB bracketing all measured rigs+margin; quantization/rail categorical mixtures; mains Bernoulli×log-uniform; spur sparse; coupling a MIXTURE OVER BOTH filter forms | domain randomization for classification (Tobin 2017, Peng 2018, Chen 2022); shapes per DR practice; form-mixture = structural DR for model-form uncertainty |

**Documented physical couplings (the only permitted dependencies):** coupling-fc ~ surface
stiffness (real physics); vehicle exclusion on vs < 95 m/s (physical scope). Each must be
listed here; anything else is a shortcut.

## 2. The gates (a revision passes iff ALL hold)

| Gate | Statistic | Target |
|---|---|---|
| **B** balance | 1 − mean-class TV(realized window-mass per subkind, uniform) | ≥ 0.85 (or declared tempering) |
| **C** coverage | min-class fraction of 5 dB SNR bins in [τ_lo, +30] with ≥1% of present windows | = 1.0 |
| **N** neutrality | max Cramér's V(class, nuisance_j) over all nuisances (strata, condition, rig params), excluding documented couplings | ≤ 0.05 |
| **F** floors/caps | every evaluable cell (subkind × SNR-bin) within [300, cap] windows | pass |
| **X** crossings | 2-way combinatorial coverage of (subkind × stratum × condition × SNR-bin) | = 100% |
| **P** probes (post-train) | linear-probe AUC feature→nuisance ≈ chance; label-from-nuisance-only ≈ chance | ≤ 0.55 |
| **D** datasheet | per-cell π_train + W + Q_j spans recorded (enables threshold-time inversion + reviewer audit) | present |

Deployment correction (the other half of the contract): decisions use
`f_y(x) + log π_deploy(y) − log π_train(y)` plus per-site operating-point adaptation
(BN-stats/threshold on a short unlabeled snippet — Nado 2020). The simulator is never
re-calibrated to a site; the *decision layer* is.

### 2b. The CDI scalar

```
CDI = min over gates of subscore_g,   subscore_g ∈ [0,1] with 1 = target met
  s_B = min(B/0.85, 1) · s_C = C · s_N = min(0.05/V, 1) · s_F, s_X = pass fractions ·
  s_P = clip((1−AUC)/0.45, 0, 1) · s_D ∈ {0,1}
```
**Min, not mean** — a single failed gate must sink the score; a mean hides a critical
defect behind six passing gates. The arg-min IS the actionable output (it names the
bottleneck). Unmeasured gates are excluded but flagged: unmeasured ≠ passing. The scalar
is a dashboard; the gates remain the authority — never optimize the scalar by reshaping a
subscore normalization (Goodhart guard).

Measured (2026-07-07, post checker-fix): **CDI(v3) = 0.333** (bottleneck C — exactly its
historical defect); **CDI(v4) = 0.229** (bottleneck N_condition — the weather→class leak).
Note a mean would have hidden v4's shortcut behind its passing gates; the min surfaces it.
A compliant v4.2 (N fixed, F floors met, DR spans in) should score ≥ 0.85 at every gate.

## 3. Verification

**(a) Literature** — every term above is an established result (full citations in
`research/v42_verify_diversity_design.md`); nothing in the formula is invented here except
the composition of the pieces.

**(b) Empirical retrodiction** — the diagnostics computed on our own corpora must track the
transfer outcomes we already measured (`invariant_check.py`):

| corpus | B balance | C coverage | N(class,stratum) | N(class,condition) | **CDI** | bottleneck |
|---|---|---|---|---|---|---|
| v3 | 0.889 ✓ | **0.333 ✗** | 0.013 ✓ | n/a (partial) | **0.333** | coverage |
| v4 | 0.873 ✓ | 1.000 ✓ | 0.023 ✓ | **0.218 ✗** | **0.229** | weather leak |

*(Correction 2026-07-07: an earlier revision reported B ≈ 0.22 for both corpora — that was
a checker bug (dictionary-encoded phantom subkind categories inflated the TV denominator),
caught by inspecting the per-subkind mass tables. True B ≈ 0.87–0.89: the window-level
normalization broadly holds. The bug and fix are in `invariant_check.py` history.)*

- **The retrodiction is now exact**: v3's CDI bottleneck is C = 0.333 — precisely the
  coverage starvation that was v3's historically measured defect. The diagnostic would have
  flagged v3 before a single training run.
- **Live defects the gates caught:**
  1. **N(class, condition) = 0.218 in v4** — a genuine shortcut risk: weather is drawn
     differently for "nothing" scenes (wind/rain subkinds get high draws) than for subject
     scenes (wind U(0,12)), so noise condition partially predicts class ("windy ⇒ nothing").
     A real intruder walks in wind too. v4.2 must draw weather identically across classes.
  2. **F floors (measured preview): violated for quiet vehicle subkinds** — thinnest
     (subkind × 5 dB detectable bin) cells: bicycle 44 windows, motorbike 74 (< 300 floor).
     Note the qualifier: floors apply to PHYSICALLY REACHABLE cells (a bicycle may not be
     able to produce the loudest bins at range); unreachable cells are documented, not
     force-filled.
  3. **B residual structure (passes, but document):** the deviation that remains is (a) the
     DESIGNED weights (walk×2, slow_quad×2, confuser×1.8) — declared tempering that belongs
     in the datasheet, and (b) an undesigned duration effect: slow subkinds (stealth
     0.2–0.6 m/s) hit the 60 s scene cap and over-contribute windows (stealth 19% vs 12.5%
     design share). Fix or declare in v4.2.

## 4. Usage protocol for any vX.Y

1. **Pre-render:** express the change as edits to the factorization terms (a wider Q_j span,
   a new subkind — automatically re-normalized by U(subkind|class), a reshaped W). If the
   change cannot be expressed in the factorization, it is out-of-formula and needs explicit
   discussion.
2. **Pilot:** render ~600 scenes → run `invariant_check.py` + `amplitude_gate.py` +
   `coverage_report.py` → all gates green.
3. **Full render:** gates re-run on the full corpus (exit-1 on fail).
4. **Post-train:** probe gate P + the standard transfer evals.
5. **Record:** the datasheet (D) travels with the corpus.

## 5. v4.2 adjustment list (from §3b, priority order)

1. **N-condition (0.218 → ≤0.05), the CDI bottleneck:** one wind/rain draw distribution
   shared by ALL scene types; weather-as-confuser stays a *subkind* of nothing, but ambient
   weather must be class-blind.
2. **F floors for quiet subkinds** (bicycle 44 / motorbike 74 in thinnest detectable bins →
   ≥300 where physically reachable): per-subkind SNR-target quotas, with unreachable cells
   documented instead of force-filled.
3. **Q_j DR spans** (the diversity core): gain across the full measured ~50 dB (not ±6 dB
   points), quantization LSB mixture, rail mixture, mains span, spur sparse, coupling
   form-mixture — per `research/v42_verify_diversity_design.md`.
4. **B tightening (0.87, passes):** cap slow-subkind window inflation (duration-aware
   quotas) or declare it; record the designed weights (walk×2, slow_quad×2, confuser×1.8)
   in the datasheet.
5. **Unmeasured gates get their tools:** X (2-way covering check), P (post-train nuisance
   probes), D (datasheet generator with per-cell π_train, W per class, Q_j spans).
