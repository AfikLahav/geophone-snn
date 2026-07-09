# Scene Catalog (v0.2 — agent-reviewed; for user review)

> **AUTHORITY: `GENERATION_PLAN.md` v1.1 supersedes this file where they conflict** (2026-06-12
> review). Key syncs: three-zone labeling (band 5–90 Hz, marginal −20 dB); +shortcuts S-G…S-K;
> precedent honesty. See GENERATION_PLAN Appendix S.

> **v0.2 changelog:** 3-agent review (coverage / realism+hyperparameters / ML-design) folded
> in. +16 subject archetypes (Israeli-fauna & operational gaps: porcupine, jackal pack,
> horse-cart, picker crews, night workers, tracked APC, forklift, golf cart, agri-robot,
> drone landing…), +12 noise archetypes (livestock-pen feeding, valve slam, gate slam,
> sprinkler surge, sabkha thermal cracking, DST micro-seismicity, Bedouin generator,
> grove-specific wind…). ~20 realism fixes applied in-line. §7 anti-shortcut rules extended
> (duration cue, terrain-class correlation, window-position). §10 labeling rules for
> ambiguous archetypes. §11 QA assertions. Durations & distributions now in
> **SCENE_HYPERPARAMS.md** (companion, binding).

Conventions: R_det = class×terrain detection range. **START rule:** r₀ ∈ [0.15, 1.1]·R_det,
random azimuth/heading; **≥30% of subject scenes open mid-zone (r₀ ≤ 0.4·R_det, onset-free)**
[raised from 15% — window-position shortcut]. Near-static archetypes (H19, A11, A12, N-pen)
constrain r₀ ≤ 0.4–0.6·R_det (outer-annulus placement would yield all-nothing scenes).
Durations per SCENE_HYPERPARAMS (residence-time for transits; dwell-distributions for
static/loiter where the radial formula diverges; hard caps per family). Difficulty E/M/H/A.

---

## 1. Path-function library

| ID | Function | Definition | Used by |
|---|---|---|---|
| P1 | straight transit | line at closest-approach d, speed v | all |
| P2 | oblique transit | line at angle θ | all |
| P3 | waypoint meander | spline through k~Poisson(4) waypoints, speed jitter ±30% | humans, animals |
| P4 | approach–dwell–retreat | in at v₁, dwell T_d, out at v₂, exit azimuth ≠ entry | humans, vehicles |
| P5 | patrol loop | closed loop, n laps | humans, robots |
| P6 | correlated random walk | exp step lengths, wrapped-normal turning, optional drift | animals |
| P7 | startle–flee | P6 until t*, then straight burst at v_max | animals |
| P8 | stop-and-go | transit segmented by exp(τ) pauses | humans, vehicles |
| P9 | perimeter arc | near-constant r ± wander (duration from angular speed!) | humans, animals |
| P10 | road corridor | polyline road at offset d_road, speed profile | vehicles |
| P11 | junction maneuver | P10 + decel → idle/turn → accel (rpm follows) | vehicles |
| P12 | partial entry | enters annulus, turns back | all |
| P13 | converge/disperse | N paths to/from common point; spread σ ~ U(0.5,5) m | groups, herds |
| P14 | follower | subject 2 tracks subject 1, lag U(0.5,4) s, offset U(0.3,2) m | pairs, convoys |

## 2. HUMAN archetypes

| ID | Narrative | Paths | Notes (fixes in-line) | Diff |
|---|---|---|---|---|
| H01 | single walker transits | P1 70%/P2 30% | all footwear | E |
| H02 | jogger/runner pass | P1/P2 | cadence **2.6–3.5** Hz (ceiling raised) | E |
| H03 | brisk walker, meandering | P3 | low radial speed; duration cap 240 s | M |
| H04 | stealth approach | P4-in/P3 | GRF ×0.3–0.6; **speed locked 0.2–0.6 m/s**, soft soles only | H |
| H05 | crawl approach | P4 slow + P3 evasive | **speed 0.3–0.8 m/s**; limb impacts + drag | H |
| H06 | casual group 2–5, out of step | P3 ×N | decorrelated phases | M |
| H07 | column/patrol pair, near in-step | P1 ×2–4 | **re-rated M** (summed cadence = stronger cue); QA-A3 energy check | M |
| H08 | crowd converges | P13 in, N 5–10 | | M |
| H09 | crowd scatters | P13 out, N 5–10 | run+walk mix | M |
| H10 | worker digs/tampers near sensor | P4 long-dwell | **re-rated A**; confuser pair w/ A13 porcupine dig | A |
| H11 | approach–inspect–retreat | P4 | | M |
| H12 | perimeter walker (constant range) | P9 | **mid-zone start by default; duration from dwell/angular speed** | H |
| H13 | stop-and-go stroller | P8 | pauses 5–30 s | M |
| H14 | heavy-carry / loaded hiker | P1/P3 | +20–40 kg; speed −15–30%, cadence −10% | M |
| H15 | limping/irregular gait | P1 | L/R amp 0.5–0.85, interval 0.7–0.95 | H |
| H16 | child / light person | P3 | 18–42 kg, cadence ↑ | H |
| H17 | runner interval training | P5/P8 | run/walk alternation | M |
| H18 | fence climb + jump + run | P12→burst | impulse burst → run train | H |
| H19 | standing loiterer (long) | P4 dwell-only | **policy §10: honest per-window SNR labels; r₀ ≤ 0.4·R_det; cap 180 s** | A |
| H20 | mid-zone start variant rule | any | now a 30% RULE across classes, kept as explicit archetype too | M |
| H21 | fruit-picker crew in crop row (4–10) | P13 drift along row | tight lateral spacing, common slow drift; the "legitimate group" case | M |
| H22 | night worker (irrigation check) | P8/P4 | 01:00–04:00 low cultural floor; looks like H04 | H |
| H23 | gear-laden patrol column 2–4 | P1/P5 ×N (P14) | 90–120 kg eff. mass, heavy boots, column spacing | M |
| H24 | mobility scooter on paved path | P10 slow | **coarse label: vehicle**; electric, small wheels; paved only | H |

## 3. VEHICLE archetypes

| ID | Narrative | Paths | Notes | Diff |
|---|---|---|---|---|
| V01 | car transits road | P10 | | E |
| V02 | car on dirt road, slow | P10 rough | **speed floor 5 km/h** (axle-impact cadence overlaps walking) | E |
| V03 | car arrives, idles, departs | P11 | idle dwell exp(90 s) cap 300 s | M |
| V04 | truck pass | P10 | | E |
| V05 | tractor working field / **combine variant** | P5/P3 slow | combine adds 8–15 Hz header/thresher oscillation | M |
| V06 | motorcycle pass | P10 | | E |
| V07 | motorcycle aggressive | P11 | rpm sweeps, Poisson(3) accel events | M |
| V08 | bicycle commuter | P10 + **P3 dirt-track variant** | 10–30 km/h | H |
| V09 | mountain bike on track | P3 fast | 8–25 km/h | H |
| V10 | e-scooter | P10 | **paved terrains ONLY (matrix)**; load↔speed corr. | H |
| V11 | ATV / quad off-road | P3 fast | 10–60 km/h | M |
| V12 | bus / municipal heavy | P10 | | E |
| V13 | convoy 2–4 spaced | P10 ×N (P14) | gap exp(8 s) | M |
| V14 | bidirectional traffic stream | P10 ×Poisson | **hard cap 180 s; count-limited (cost)** | M |
| V15 | car stops; doors; footsteps away | P11 + walker | **labeling rule §10 (transition windows flagged)** | H |
| V16 | tracked agricultural crawler | P10 slow | **needs track-pitch harmonic comb (speed/sprocket-pitch) — [gen+] source term** | M |
| V17 | golf/utility electric cart | P10 | 250–600 kg, no tones; kibbutz internal roads | H |
| V18 | forklift/telehandler at yard | P4+P8 | stop-go load cycles; diesel tonal or electric silent | M |
| V19 | tracked APC / military vehicle | P10 fast | track-pitch comb + strong tonals, huge amplitude, >300 m | E |
| V20 | autonomous agri-robot (night rows) | P5/P3 slow | electric near-silent or diesel; systematic row passes | M |
| V21 | agricultural drone landing *(speculative)* | impulse @ r_land | 40–65 kg touchdown impulse + motor wind-down comb | H |

## 4. ANIMAL archetypes

| ID | Narrative | Paths | Notes | Diff |
|---|---|---|---|---|
| A01 | dog free-running | P6/P7 | trot/gallop | E |
| A02 | dog + owner (leash) | P14 pair | **labeling policy §10 — DECISION NEEDED** | H |
| A03 | jackal cautious night crossing | P6 slow + P12 | **re-rated A; r₀ ≤ 0.6·R_det** | A |
| A04 | boar group rooting | P6 ×2–5 | rooting impacts Poisson(0.8)/s/animal | M |
| A05 | gazelle crossing + pronk | P1 fast + bursts | **pronk = 4-leg simultaneous landing, 6–10 BW total, 0.05–0.10 s contact — [gen+] forcing mode** | M |
| A06 | sheep/goat flock + shepherd | P13 + walker | per-source SNR labeling §10 | H |
| A07 | cattle herd grazing drift | P6 ×5–10 | dwell-duration; cap 300 s | M |
| A08 | horse + rider walk→trot | **P3 primary**/P10 | rider = added mass per hoof, NOT separate gait | E |
| A09 | donkey/mule on track | P10 slow | **border/PA-adjacent zones annotation** | M |
| A10 | startle-flee past sensor | P7 | burst 5–15 m/s | M |
| A11 | stray dog loiters/digs | P4 dwell | scratch bursts 2–6 Hz | H |
| A12 | grazing single animal near-static | P9/P6 tiny | **dwell-duration; r₀ ≤ 0.5·R_det; §10 honest labels** | H |
| A13 | crested porcupine forage + dig | P6 slow + P4 dig | 11–18 kg nocturnal; dig bursts mimic H10 — the #1 animal false-alarm gap | H |
| A14 | golden jackal PACK crossing 2–4 | P13 loose | the animal→human-group confusion case | H |
| A15 | mongoose fast transit *(speculative)* | P6 fast | 1.7–4 kg; boundary-detectable on stiff ground | A |
| A16 | hyrax colony on rocky terrain *(speculative)* | P6 ×3–8 | weak chaotic multi-source on high-coupling rock | A |
| A17 | horse + cart (border-agricultural) | P10 slow | hoof train + wooden-wheel thump; **coarse label DECISION (animal vs vehicle)** | M |

## 5. MIXED / TIMELINE archetypes

| ID | Narrative | Composition | Diff |
|---|---|---|---|
| X01 | car passes; later a walker | V01 → gap U(30,**90**) s → H01 | M |
| X02 | walker during distant traffic | H01 + V14 background | H |
| X03 | dog chases vehicle | A01 + V01, dog lags 2–8 s | H |
| X04 | group meets, splits | H06 → P13 disperse | M |
| X05 | pump **cycles repeatedly** during walk | H01 + N18 duty cycle | A |
| X06 | storm building + intruder | H04 + rising weather; **SNR guard: intruder >6 dB for first 20% of windows** | A |
| X07 | walker + perimeter animal | H01 + A12 | H |
| X08 | quiet hour: 3 unrelated events | any 3, gaps exp(60 s) [10,300]; cap 360 s | M |
| X09 | vehicle arrives, crew disperses to field | V03 → H21-style ×3–5 P13 | H |
| X10 | tractor masks approaching walker | V05 fixed + H01 opposite side | A |

## 6. NOTHING archetypes (50% of scenes)

**Ambient & weather:** N01 calm night · N02 calm day + cultural hum · N03 breeze 3–6 ·
N04 windy 7–12 + vegetation · N05 gale/khamsin 13–20 + saltation · **N06 gust train at
1.5–2.5 s period (≈0.4–0.67 Hz envelope — deliberate cadence-band mimic) [A]** · N07 steady
rain 2–10 mm/h · N08 convective burst (ramp→15–30→decay) · N09 hail [H] · N10 thunderstorm
sequence [H] · N11 snow-quiet · **N12 frost-crack pops (frozen) [A][gen+]** · N13 hot-day
drift + expansion ticks · **N14 wadi flash flood — sustained broadband 2–25 Hz dominant,
slow envelope, minutes (re-described) [H]** · N15 high-surf (coastal, **dist_to_coast ≤
500 m gate**) [M] · **N38 sabkha thermal-crack pops (daytime warming; Israeli analog of
N12) [A][gen+]**

**Infrastructure & machinery:** N16 mains-rich · **N17 transformer 100 Hz hum [gen+]** ·
N18 irrigation pump duty cycle · **N19 water-pipe flow hum [gen+]** · **N35 valve-slam /
water hammer (sharp irregular thumps — most footstep-like nothing) [A][gen+]** ·
**N36 gate/metal-door slams (Poisson with shift-change bursts) [M][gen+]** ·
**N37 sprinkler-zone activation surges (timed sequence ≈ slow walker!) [M][gen+]** ·
N20 generator near · **N31 Bedouin-camp distant generator (lower fundamental, load drift)
[H]** · N21 distant pile driver [A] · N22 jackhammer bursts [H] · N23 mixed construction
day [H] · **N39 idling military vehicle at 200–500 m (steady tonal, labeled nothing) [H]**

**Biological & geophysical (non-subject):** **N32 animal burrowing/digging near sensor
(porcupine/tortoise; nothing-class counterpart of A13/H10) [A][gen+]** · **N34 livestock-pen
feeding frenzy at fixed 10–30 m (twice-daily heavy-hoof burst — looks like converging group)
[H]** · **N33 grove wind signatures (date palm sparse / eucalyptus broad 1–10 Hz / olive
intermediate) [M]** · **N40 Dead-Sea-Transform micro-earthquake (ML 1.5–3, broadband 0.5–20 Hz
burst 5–30 s, no envelope logic) [M][gen+]**

**Overflights & distant transport:** N24 airliner swell · N25 prop plane comb ·
**N26 helicopter — BPF 8–30 Hz comb (corrected; parametrize blades×RPM; Doppler minor) [A]** ·
**N41 train with bogie-thud cadence 1.5–3 Hz + rumble (expanded from N27) [H]** ·
N28 sonic boom · N29 distant road hum · N30 sensor pathologies ·
**N42 wind-turbine blade harmonics 1–3 Hz (Golan/Carmel sites) *(speculative)* [M]**

## 7. Anti-shortcut rules (extended; binding)

1. **Energy overlap:** per-class window-RMS 10–90th percentiles overlap ≥1 decade (incl.
   nothing — its loud members N05/N10/N14/N20 close the gap). [QA-A1]
2. **Periodicity on both sides:** periodic/impulsive nothing (N06/N12/N18/N19/N21/N34/N35/
   N37/N38/N41) vs aperiodic subjects (H13/H15/A12…).
3. **Tonality on both sides:** tonal nothing (N17/N18/N20/N26/N31/N39) + toneless vehicles
   (V08/V09/V10/V17/V20-electric) at boosted counts. [QA-G1/G2]
4. **SNR continuum:** peak-SNR sampled per difficulty tier (E:10–25 dB, M:3–15, H:0–8,
   A:−3–5 above threshold) — the boundary is populated by design.
5. **Onset-free ≥30%** of subject scenes (mid-zone starts), all classes. [QA-C3]
6. **No duration cue:** subject-window fraction per scene must overlap across classes;
   long vehicles exist (V03/V05/V14/V18/V20), short humans exist (H18, runners). [QA-C1]
7. **No terrain→class cue:** every coarse class ≥200 train windows in EVERY terrain family;
   no terrain group >65% one class; humans/animals get paved scenes, vehicles get dirt.
   [QA-B1/B2/B3]
8. **No archetype dominance:** no archetype >15% of its class's windows. [QA-A2]
9. **Cadence overlap across species:** fast humans (3.5 Hz) vs slow animals (cattle 0.8 Hz)
   — distributions overlap.
10. **Difficulty mix:** ~45/30/18/7 E/M/H/A.
11. Confuser ownership rule unchanged (sole content → nothing; under subject → background).

## 8. Stratification (provisional — see open questions)

- 50% nothing / 50% subject (scene-level). Subject split **human 37 / vehicle 40 / animal 23**
  (vehicle raised from 35: window-level arithmetic showed vehicles land at only ~5.5% of
  windows under the old split — agent estimate; plus vehicle durations extended via
  V03/V05/V14/V18/V20). Mixed X* ≈ 12% of subject scenes.
- Per-archetype tiers: E 500–800 / M 250–400 / H 150–250 / A 80–150 scenes; cost-capped
  archetypes (V14 ≤200 scenes). Target ~50k scenes → ~5–6M windows (window-level ≈
  70% nothing / 11% human / 7% vehicle / 8% animal / rest X*) — **window ratios tracked in
  manifest; documented in data card; class-weighted loss expected downstream.**

## 9. Labeling rules for ambiguous archetypes — see §10. QA assertions — see §11.

## 10. Labeling rules (binding; PLAN 4.4 extension)

- **HUMAN-PRIORITY RULE (user decision 2026-06-12, binding):** in any window where a HUMAN
  source is above the detection threshold, the deployment label is `human` — regardless of
  co-present animals/vehicles. Rationale: a missed human is the costly error; richer
  multi-detection outputs are a policy-layer/output-contract topic (manifest keeps all
  per-source SNRs + fine labels, so any future output design works without regeneration).
- **A02 dog+owner:** RESOLVED by human-priority → windows label `human` while the owner is
  detectable; `animal` only where the dog alone clears threshold.
- **A06 flock+shepherd:** human-priority; among non-human sources, dominant SNR wins;
  none above → nothing.
- **V15 handoff:** human-priority (tie-zone → human); `transition_flag=True` windows go to
  the hard-negatives eval slice, <10% of scene. [QA-D4; QA-D1 updated accordingly]
- **H19 / A12 boundary archetypes:** honest per-window SNR labels (windows flip
  human↔nothing as micro-motion fluctuates — that IS the training signal); assert ≥20%
  (H19) / ≥15% (A12) subject-labeled windows else archetype is non-functional. [QA-D2/D3]
- **Sequential X\*:** one active source per window (dominant-or-nothing).
  **Simultaneous X\*:** dominant SNR wins; `multi_source_flag=True`; both SNRs stored.
  **Confuser-background X\* (X05/X06):** confuser never labels; target labels when above
  threshold.
- Label: **three-zone [v1.1], band 5–90 Hz** — detectable ≥0 dB / marginal −20…0 dB soft / nothing <−20 dB; ±3 dB sensitivity audit (<15% of
  windows may flip) — full definition in SCENE_HYPERPARAMS §C.

## 11. QA assertions (corpus gate; testable)

Energy: QA-A1 RMS-overlap decade · QA-A2 archetype ≤15%/class · QA-A3 H07 ≤ H14+4 dB.
Terrain: QA-B1 ≤65% single class per terrain group · QA-B2 ≥200 windows class×family ·
QA-B3 KS-test on GF fingerprint across classes.
Position/duration: QA-C1 subject-window-fraction overlap (vehicle mean >0.35, human >0.45) ·
QA-C2 corr(window-index, nothing) < 0.40 · QA-C3 ≥30% mid-zone starts.
Labels: QA-D1 A02 policy enforced · QA-D2/D3 boundary-archetype functionality ·
QA-D4 V15 transitions <10% · QA-D5 multi-source label = argmax SNR (1% sample audit).
Splits: QA-E1/E2 no profile/scene crosses splits · QA-E3 cross-split near-duplicate scan ·
QA-E4 X*-profiles disjoint from constituent-class profiles across splits.
Balance: QA-F1 vehicle test windows ≥15k · QA-F2 class window counts within 3× ·
QA-F3 ≥500 windows per tier per class in test.
Spectral: QA-G1 ≥8% toneless vehicle windows · QA-G2 ≥5% tonal nothing windows ·
QA-G3 ≥10% aperiodic human windows.

## 12. Decisions log + remaining open items

**RESOLVED (user, 2026-06-12):**
- (1) Human-priority labeling — see §10. Richer output design (multi-detection, detection
  strength, distance estimation post-process, event aggregation) = the OUTPUT-CONTRACT
  sitting, scheduled before the training plan (Gate-7 precondition). Manifest already
  carries everything needed; no regeneration whatever is decided.
- (7) Window balance: corpus stays ~70% nothing (training mix ≠ deployment prior).
  BINDING for downstream: deployment prior is ~99%+ nothing → val/test reporting must
  include deployment-prior-adjusted metrics; threshold + N-of-M persistence calibration is
  an explicit training-plan stage (at 1.5 s hop a day ≈ 57,600 windows — per-window FAR
  must reach ~1e-5 via calibration, not raw balance). Documented in data card.
- Class representation: sufficiency FLOORS per class×terrain×tier (QA-F), not equality —
  generate generously; balance is a training-time choice (downsampling is free, missing
  data is not).

**STILL OPEN:**
2. A17 horse-cart coarse label (animal or vehicle?). Proposal: animal (hooves dominate).
3. Subject ratios 37/40/23 (vehicle boosted) — approve?
4. New [gen+] generators (track-pitch comb, pronk impulse, valve slam, gate slam, sprinkler
   surge, burrowing, livestock pen, sabkha cracks, DST micro-quake, transformer, pipes,
   frost pops, helicopter comb parametrization) — approve?
5. Speculative archetypes (A15 mongoose, A16 hyrax, V21 drone landing, N42 wind turbine):
   token counts or drop?
6. Difficulty re-ratings applied (H07→M, H10→A, A03→A) — objections?
