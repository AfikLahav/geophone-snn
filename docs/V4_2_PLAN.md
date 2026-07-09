# v4.2 — diversity corpus (weather-axis fix + sensor DR + coverage), CDI-gated

*Detailed execution plan for the v4.2 synthetic-geophone training corpus. Written to be
picked up with zero context loss. Scope: corpus generator + evaluation scripts only — no
hardware, no model-architecture change, 4 classes frozen, subkind stays metadata (never a
label). Rev 2 (2026-07-07): the six review findings are folded in-line and marked **[R#]**.*

## Context

Synthetic-geophone training corpus for a 4-class classifier (human/vehicle/animal/nothing).
Acceptance instrument = the Corpus Design Invariant (`CORPUS_DESIGN_INVARIANT.md`): 8 gates,
scalar CDI = min of normalized subscores. Production-candidate corpus **v4** scores
**CDI = 0.227**, bottlenecked by the **P gate**: on physically-ambient windows a probe recovers
scene class at 0.66–0.74 AUC. Proven root cause (`gap_study/v4_plan/_probe_conditional.py`):
weather is drawn by DIFFERENT laws per scene type AND double-encoded as nothing subkinds, so
ambient "feel" leaks the label. N-condition (Cramér's V = 0.217) is a coarse projection of the
same defect; the within-condition probe stays 0.66–0.86, proving the leak is **continuous
within tiers** — equalizing tier *frequency* is not enough, the continuous draw law must be
shared across classes.

**This requires a re-render** (weather values are baked into synthesized noise waveforms; no
post-hoc patch). One render carries ALL v4.2 changes. Everything env-flagged; defaults
reproduce v4 bit-for-bit. Acceptance = CDI ≥ 0.85 at every gate, THEN beat v4 (the E2 build)
on the transfer gauntlet.

**Conceptual reframe vs v4.1 [R6]:** v4.1 tried to *calibrate away* nuisances (eliminate
clipping, pin one gain). v4.2 is the opposite mindset — nuisances become **randomized axes the
model must be invariant to**. So a controlled fraction of clipped/low-gain/quantized scenes is
DESIRED, not a defect. Gates that in v4.1 said "minimize X" must in v4.2 say "X is present,
bounded, and class-independent." Do not carry v4.1's minimize-clipping criterion forward.

---

## CRITICAL INVARIANTS — honor every one (do not skip, do not "improve away")

1. **New package: copy `simgeo_v4/` → `simgeo_v42/`.** Never edit `simgeo_v4/` (frozen v4
   generator + baseline). All code changes live in `simgeo_v42/`. New output roots:
   `G:/geophone_synth/corpus_v42`, `labels_v42`, `features_v42_3s`,
   `config/snr_maps_v42.npz`.
   **[R2] Path guard — exact whitelist, NOT substring.** The current guard `assert "v4" in
   path` (init_db_v4:416, main:477) is a data-loss hazard: `"v4" in "corpus_v42"` is True, so
   an inherited `"v4"` guard ALSO accepts `corpus_v4`/`v4b`/`v41` — and `init_db` deletes
   `path`,`-wal`,`-shm` (:417-419). Replace with a positive basename whitelist that rejects
   every other corpus:
   ```python
   base = os.path.basename(os.path.dirname(path))
   assert base == "corpus_v42" or base.startswith("corpus_v42_pilot"), f"refusing non-v42 path: {path}"
   ```
   (G: blocks PowerShell deletes but Python `os.remove` may not — the guard is load-bearing.)
2. **Never touch on disk:** corpus_150k, corpus_v2, corpus_v3, corpus_v4, corpus_v4b,
   corpus_v41 and their feature/label dirs, `simgeo_banks/*.npz`, the real CSVs,
   feature_analysis sqlites. (v3 deletion is the SEPARATE final task §7 — only after v4.2 is
   fully validated.)
3. **Worker cap = 8 concurrent, ONE heavy CPU job at a time** (`GEO_V4_POOL=8`). Before
   launching any render/label/extract, verify no other heavy python job is running
   (`Get-Process python` CPU check — task-completion notifications LAG). Never run two
   8-worker jobs at once (standing user rule — makes the machine unusable).
4. **Commits attributed to AfikLahav ONLY — NO "Co-Authored-By: Claude" trailer** (standing
   user rule).
5. **Reproducibility ordering:** the FIRST rng consumer in `sensor.render_hp` is the parameter
   dict `p` (keeps clean_mv reproducible) — do NOT reorder. In `gen_scene_v42` the weather
   draw MUST stay before the content branch (the SNR inversion needs `cond`).
6. **D2 answer-key invariant:** per-class clean blobs must reconstruct the summed clean
   exactly (`Σ clean_cls == clean_full`, err≈0). The `gap_study/v4_plan/_check_d2.py` assert
   (20 mixed scenes, float32 tol) is a MANDATORY pilot gate. **Quantization/rail (Change 3)
   apply to the MODEL-INPUT SUM ONLY** — never to the stored per-class clean blobs (those stay
   unquantized/unclipped so D2 and the SNR labels stay exact).
7. **Gain-invariance of SNR maps:** keep the v4.1 line-level = dB-above-floor rule → per-window
   SNR is gain-invariant, so the gain span does NOT change the maps. **[R4]** Coupling FORM
   does change in-band SNR, so the maps MUST be rebuilt under the v42 sensor env (see Change 3
   / Exec step 2 for how the mixture is handled).
8. **Pilot path hygiene:** the generator RESUMES from committed scene_ids (skips them). Always
   render a pilot to a FRESH path (`corpus_v42_pilot1`, `_pilot2`, ...) — reusing a path with
   committed scenes silently reuses stale scenes; G: blocks deletes so you can't clear a path.
9. **Run the FULL CDI gate suite on the pilot BEFORE the full render, and again on the full
   corpus.** Any pilot gate fail ⇒ fix + re-pilot (fresh path); do NOT proceed.

---

## Change 1 — Weather = class-blind global axis (fixes P + N-condition; THE bottleneck)

**Exact defect location** (in the copied `simgeo_v42/generate_corpus_v4.py`):
- `gen_scene` weather block, lines **229–237**: per-coarse branches (nothing→subkind-keyed
  dict; else→wind U(0,12), rain U(0,4)@0.15).
- `NOTHING_V4` (:53): `[("calm",1),("wind",2),("rain",1),("machinery",1),("overflight",1),
  ("traffic",1)]` — weather double-encoded as subkinds.

**Fix — implement exactly:**
- **Tier assignment in `build_plan_v42`:** every scene gets a weather TIER from
  {calm, wind_low, wind_mid, wind_high, rain_light, rain_heavy}. Extend the plan tuple to 6
  elements: `(sid, pid, split, coarse, sub, tier)`.
  **[R3] Deal mechanics — the load-bearing detail (REVISED at implementation; smoke-proven).**
  A probability marginal `WEATHER_MARGINAL = {calm:0.35, wind_low:0.20, wind_mid:0.15,
  wind_high:0.10, rain_light:0.12, rain_heavy:0.08}` becomes a deterministic **largest-remainder
  deal**, NOT a per-scene weighted i.i.d. draw (i.i.d. reintroduces the exact by-luck class
  imbalance we are removing). **The deal is done PER COARSE CLASS on the class total, not per
  cell.** The originally-planned per-cell deal was *insufficient*: cells differ in size across
  classes (subj cells `pc≈8` vs nothing/mixed cells much larger), and `deal_tiers(8)` rounds the
  marginal coarsely to `{calm .375, others .125}` while large cells land near WEATHER_MARGINAL —
  leaving a residual class↔condition association (the smoke measured **TV 0.10** between subj and
  nothing, which would keep N-condition ≈ 0.1, defeating the fix). Per-class dealing removes it:
  for each coarse class, `deal_tiers(N_class)` (largest-remainder on the *class total* → marginal
  == WEATHER_MARGINAL to within one scene per tier for large N), then a **deterministic shuffle**
  (`rng(seed+777)`) before assigning tiers to the class's scenes (the shuffle decorrelates tier
  from pid/subkind so the X gate's subkind×condition pair stays covered). Result: **every class's
  tier marginal == WEATHER_MARGINAL exactly** (smoke: TV 0.0000 for all of human/vehicle/animal/
  nothing/mixed). `deal_tiers(count)` = floor(`WEATHER_MARGINAL × count`) + the `rem` largest
  fractional remainders (tie-break by fixed WEATHER_TIERS order). Spec tuple grows to 6:
  `(sid, pid, split, coarse, sub, tier)`. **Pilot verify:** every class marginal ≈ WEATHER_MARGINAL,
  `rain_heavy count > 0`, and N-condition (invariant_check) ≈ 0.
- **Continuous draw inside tier, class-independent** (`weather_from_tier(tier, rng) ->
  (wind, rain)`, replacing 229–237, applied to ALL coarse types identically):
  calm wind U(0,3) rain 0 · wind_low U(3,6) · wind_mid U(6,10) · wind_high U(10,16) ·
  rain_light wind U(0,6) rain U(0.6,5) · rain_heavy wind U(0,8) rain U(5.5,30). (rain lows nudged
  0.5→0.6 and 5→5.5 so every draw incl. the numpy `[low,high)` endpoint sits STRICTLY inside
  condition_of's bins → the `assert cond==tier` in gen_scene is provably always true; verified 0
  mismatches over 30 000 draws.) Assert `r3_noise.condition_of(wind,rain) == tier` per scene.
- **`NOTHING_V42`** = `[("ambient",3),("machinery",1),("overflight",1),("traffic",1)]`.
  "ambient" = pure background under the dealt weather (weight 3 keeps pure background ~half of
  nothing). In the `coarse=="nothing"` branch (:260-270): `if sub=="ambient": pass` (v_sig
  stays zeros; noise bed carries the weather); machinery/overflight/traffic unchanged and now
  ALSO receive the global weather (bed = `ground_noise(wind,rain)` + their `extra_ground`).
- **Downstream unchanged:** `r3_noise.ground_noise`, `sensor.rain_noise`, `condition_of`,
  the `noise_condition` scene column. `tier` and `noise_condition` become redundant (both
  encode weather) — keep both, the assert verifies they match.

**Predicted:** N-condition 0.217→~0.00 (round-robin); P 0.74→the non-weather residual
(terrain-mix). If residual P > 0.55, that is the documented vehicle-terrain-exclusion pathway
— decide mitigate-vs-document THEN, do not pre-empt.

---

## Change 2 — Subkind-aware SNR inversion (fixes F: bicycle/motorbike/jackal loud-bin starvation)

**Defect:** `snr_to_distance` (:80) inverts ONE per-class SNR(r) map for all subkinds; a
bicycle (~20+ dB below the car reference the vehicle map was built with) placed at "target
+20 dB" renders far fainter → loud bins never fill (F worst cells: bicycle@15–25 dB =
117–169 windows vs 300 floor; motorbike@20, jackal@15 similar).

**[R1] Fix — apply the offset at the CALL SITE, keyed on the REAL subkind (not `cap_key`).**
The bug the naive reading would hit: `_emit_moving` computes `cap_key = csub if c=="vehicle"
else c` (:141) and passes `cap_key` to `snr_to_distance` (:143) — for humans that's the string
`"human"`, so a `SUBKIND_SNR_OFFSET_DB[subkind]` inside `snr_to_distance` would receive
`"human"` and leave every gait at offset 0 (stealth/child still starved). Keep `cap_key` for
`R_MAX` (the 3rd arg's only use, :88) and apply the offset to `snr_t` at the call site keyed on
the real `csub`:
```python
snr_map_target = snr_t - SUBKIND_SNR_OFFSET_DB.get(csub, 0.0)
r_t, clipped = snr_to_distance(pid, c, cap_key, snr_map_target, cond, rng)
```
**Sign (verify, do not guess):** map is car-referenced; bicycle `offset = −22 dB` relative to
it; to make the bicycle *realize* `snr_t`, invert the car map at `snr_t − offset = snr_t + 22`
(closer). So `− SUBKIND_SNR_OFFSET_DB` with bicycle=−22 is correct.
**Record `target_snr_db = snr_t`** (intended realized), NEVER `snr_map_target`.
**Three call sites:** `_emit_moving` (:143), `two_vehicle` per-member (:180, key drawn `k`),
`idle_exit` (:274, key `"idle_exit"`≈−4). **Convoy caveat:** `r_t` is computed at :143 with
`cap_key="convoy"` before the member `kind` is drawn (:169) → convoy uses a convoy-level offset
(≈0, car-dominated); acceptable, documented — do not reorder convoy's draw for this.

`SUBKIND_SNR_OFFSET_DB` (dB rel. each class map's reference, starting values):
- vehicle (ref=car): car 0, truck +4, tractor +2, tracked +10, motorbike −10, bicycle −22,
  idle_exit −4, convoy 0, two_vehicle per member
- human (ref=walk): walk 0, run +3, child −4, stealth −6, loaded +1, group/march per n_members
- animal (ref=horse): horse 0, boar −2, sheep −4, dog −8, jackal −12, herd per k, slow_quad −2,
  horse_rider 0

Also compute per-subkind reachability ceiling (max achievable SNR at MIN_STANDOFF on the
loudest terrain) → write `config/subkind_ceilings_v42.json`; `invariant_check.py` F-gate
consumes it to exempt physically-unreachable bins (it already supports exemption via a p99.5
proxy — feed true ceilings instead).

---

## Change 3 — Sensor-chain domain-randomization spans (the diversity core)

All in `simgeo_v42/sensor.py`, class-independent (drawn in `render_hp`, independent of the
scene/label). Design verified in `research/v42_verify_diversity_design.md`. Flags default to v4.
- **gain** `GEO_GAIN_LOG10 = "lo,hi"` (TWO numbers): uniform-in-dB spanning ~50 dB, bracketing
  the 3 measured real rigs + margin (nothing-window RMS ~0.1–30 mV). Bounds set by the
  calibration pilot. NOT a point ±6 dB.
- **quantization** NEW `GEO_QUANT_MIX`: per-scene categorical LSB ∈ {0=continuous, 0.125, 0.2,
  0.25 mV}; apply `round(x/LSB)*LSB` to the MODEL-INPUT SUM only (Invariant 6). Default
  all-continuous (=v4).
- **rail** NEW `GEO_RAIL_MIX`: per-scene rail ∈ {inf=never-clips, 512, 256 mV}; `ADC_CLIP_MV`
  becomes per-scene. Default 256 (=v4). **[R6] Clipping is now a randomized nuisance, not a
  defect** — a controlled fraction of clipped scenes is intended (rail-invariance).
- **mains**: keep v4.1 `GEO_LINES_MODE=v41` (dB-above-floor), span 0–25 dB, Bernoulli
  terrain-gated as today.
- **spur**: keep v4.1 `GEO_SPUR_P=0.2`.
- **coupling FORM MIXTURE** NEW `GEO_COUPLING_MODE=mix`: per-scene choose lowpass OR bump
  (50/50), fc log-uniform ~30–150 Hz, keep `anchor_fc` per-surface conditioning as the
  documented physical coupling. `coupling_response` already has both forms; add a "mix" mode
  that picks per call.
- **[R5] Record ALL Q_j draws per scene** as new `SCENE_COLS_V42` columns
  (`gain_log10, quant_lsb, rail_mv, coupling_form`; `coupling_fc` already exists) — required by
  the datasheet AND fed to the N gate to confirm class-independence.

---

## Change 4 — B tightening (passes 0.87, remove the undesigned skew)

Slow subkinds (stealth 0.2–0.6 m/s) hit the 60 s `DUR_CAP_V4` and over-contribute windows
(stealth 19% vs 12.5% design). Fix: make the per-scene window quota `K = ceil(2*T_COVER/W)` in
`label_windows.py` per-subkind duration-aware so total windows per subkind are ~equal (cap slow
subkinds' K). Acceptable simpler alternative: cap subject-scene `dur` by subkind so window
yield is ~equal. **Re-verify the C-coverage gate on the pilot** — changing per-subkind K shifts
the per-bin window counts C depends on. Record the DESIGNED weights (walk×2, slow_quad×2,
confuser×1.8) in the datasheet as declared tempering.

---

## Change 5 — Real-side (not corpus): reporting protocol

Adopt seg30+signal-only as the fold reporting protocol (already in
`gap_study/v4_plan/exp_folds.py`). `GEO_REAL_SCALE` stays 25.4 for E2-comparability unless the
transfer eval shows ×1000 is better for the chosen corpus. No corpus impact.

---

## Files to create / edit (explicit)

- **Copy** `simgeo_v4/` → `simgeo_v42/` (whole package).
- **Edit** `simgeo_v42/generate_corpus_v4.py`: `build_plan_v42` (tier round-robin per **[R3]**,
  NOTHING_V42, 6-tuple spec), `gen_scene_v42` (weather_from_tier, ambient branch, **[R1]**
  call-site subkind offset at the 3 sites, Q_j draws + recording), `SCENE_COLS_V42`
  (+tier, +gain_log10, +quant_lsb, +rail_mv, +coupling_form), `run_shard_v42`, `init_db_v42`
  (**[R2]** exact-basename guard), `main` (honor GEO_V4_POOL). New constants:
  `WEATHER_MARGINAL`, `WEATHER_TIER_RANGES`, `SUBKIND_SNR_OFFSET_DB`.
- **Edit** `simgeo_v42/sensor.py`: `GEO_QUANT_MIX`, `GEO_RAIL_MIX`, `GEO_COUPLING_MODE=mix`;
  keep v4.1 flags.
- **Edit** `simgeo_v42/coupling.py`: expose the form-mixture mode.
- **Edit** `simgeo_v42/snr_maps.py`: output `snr_maps_v42.npz`; **[R4]** render under the
  IDENTICAL v42 sensor env (`GEO_COUPLING_MODE=mix`, gain span) with `N_REP≥3` — the per-rep
  median IS the expected-SNR-under-mixture; do NOT build per-form maps (the residual per-scene
  form scatter is the ±5 dB the coverage gate already absorbs). Emit
  `config/subkind_ceilings_v42.json`.
- **Edit** `simgeo_v42/label_windows.py`: Change 4 per-subkind K (or keep a flag; v4 default).
- **Edit** `gap_study/v4_plan/datasheet_gen.py` **[R5]**: add `DECLARED["v42"]`; extend the
  column read to `gain_log10/quant_lsb/rail_mv/coupling_form/coupling_fc` and report realized
  spans (min/max/histogram) per Q_j axis.
- **Edit** `gap_study/v4_plan/amplitude_gate.py` **[R6]**: replace the "nothing RMS median ∈
  [8,30] mV" check with a SPAN check (p5 ≲ 0.5, p95 ≳ 20 mV, brackets rigs 0.08/7.4/25.2), and
  reframe the clip check from "minimize" to "present, bounded, and class-independent" (clip
  median ~0 expected because most scenes draw inf/512 rail; verify class-independence).
- **Edit** `gap_study/v4_plan/invariant_check.py`: feed each Q_j axis into the N gate
  (Cramér's V(class, Q_j) ≈ 0 confirms neutrality).
- **Gates otherwise run as-is** (corpus-agnostic CLI): `invariant_check.py`,
  `coverage_crossings.py`, `nuisance_probe.py`, `coverage_report.py` — pass `v42` tag + paths.

## Execution order (STOP-AND-REPORT at every gate)

1. Copy package; implement Changes 1–4; smoke-test each new source/function (pattern
   `gap_study/v4_plan/_smoke_gen.py`) — assert `cond==tier`, D2 reconstruction (`_check_d2.py`),
   Q_j recorded, subkind offset applied to the RIGHT key.
2. Rebuild SNR maps under the v42 sensor env (**[R4]** mix coupling, N_REP≥3) →
   `snr_maps_v42.npz` + `subkind_ceilings_v42.json`; verify vs the calib map (±4 dB where
   comparable).
3. **Calibration pilot** (fresh `corpus_v42_pilot1`, 600–900 scenes): `amplitude_gate.py` sets
   `GEO_GAIN_LOG10` bounds so the nothing-RMS SPAN brackets [~0.1, 30] mV; run FULL CDI suite.
   **STOP-REPORT.** Fix + re-pilot (new path) on any gate fail. Targets: CDI ≥ 0.85,
   N-condition ~0, P as low as achievable, rain_heavy populated, C still 1.0 after Change 4.
4. **Full render** `corpus_v42` (`GEO_V4_POOL=8`) → label 3 s → CDI suite on full corpus
   (exit-1 gates). **STOP-REPORT** with the full 8-gate table vs v4.
5. Extract `features_v42_3s` → train frozen protocol (`GEO_GATED_EVAL=1 GEO_ORDINAL_MASK=1`,
   sampler OFF) → `snn_v4_out/E5_v42`.
6. Transfer gauntlet vs E2: `eval_synth_to_real.py` (threshold-free + signal-only) +
   `real_label_audit.py`; if ≥ E2, `finetune_5fold.py` + `stratified_auroc.py`. **STOP-REPORT.**
7. Registry: add row **E5 (v4.2)** to `V4_EXPERIMENTS.md`; update `CORPUS_DESIGN_INVARIANT.md`
   §3/§5 with the v42 CDI; commit per milestone (AfikLahav only).

## Acceptance / promotion

- CDI(v4.2) ≥ 0.85 at EVERY gate (the formula IS corpus acceptance).
- Transfer ≥ E2 on threshold-free AUROCs AND signal-only acc ⇒ v4.2 becomes production
  candidate. Falsifiable prediction: killing the P shortcut should improve zero-shot FAR
  behavior; if overall transfer DROPS, that measures how much v4 leaned on the shortcut
  (publishable either way) — v4.2 is still the honest corpus even if not promoted.

## §7 — FINAL TASK (only after v4.2 fully validated & committed): remove truly-unused old data

Add a task to delete ONLY `corpus_v3` + `features_v3` + v3 labels — verified unused: v3 is
superseded; v3.1 was retrained into `snn_v3_1_out` from features_v3 but that experiment is
closed and its verdict recorded. DO NOT delete: corpus_v2 (paper's working production model;
feature_analysis_v2.sqlite lineage), corpus_150k (v1), corpus_v4 (production candidate until
v4.2 promoted). G: blocks deletes (protected path) — the task must first confirm the delete
mechanism, list exact paths + sizes, and get explicit user go-ahead before removing anything.
corpus_v4b/v41 are lower-priority candidate deletes (reproducible from flags) — propose, don't
assume.

## Verification summary

Pilot CDI suite → full-corpus CDI suite (exit-1) → D2 assert → span-aware amplitude gate →
transfer vs E2 → A1/A2 if promoted. Every stage stop-and-reports. ~1 day code, ~4 h compute at
8 workers. All six review findings (**[R1]** offset threading + sign, **[R2]** exact path
guard, **[R3]** round-robin per-cell largest-remainder, **[R4]** SNR map under mix coupling,
**[R5]** datasheet Q_j reads + N-gate, **[R6]** span-aware amplitude gate + clipping-as-nuisance)
are folded into the sections above.
