# Noise Model — per-condition generator specs (v0.2, 2026-06-12)

> **AUTHORITY: `GENERATION_PLAN.md` v1.1 supersedes this file where they conflict.** v1.1
> syncs: pump = BPF formula (NOT fixed 75 Hz); pile driver = impulsive 7–50 Hz comb (NOT a
> 0.5–2 Hz tone); rain onset 60–80 Hz; wind-AM σ/mean 0.5–0.9 (body fixed below).

> **v0.2 — R3 FITTED VALUES (from the fetched corpus: ~4,000 condition-labeled hours,
> hour-level ERA5 relabeling; r3_fits.npz holds the per-bin median PSDs + p10/p90):**
> - **Wind gust-AM intensity RAISED: envelope sigma/mu = 0.5–0.9 (measured; v0.1's
>   literature 0.15–0.30 was far too tame).** Modulation timescale tau = 4–20 s,
>   consistent with T_L ~ 100/U. Sample sigma/mu ~ U(0.5, 0.9) per scene.
> - **Diurnal cultural curve (measured, IS 2021):** day/night amplitude swing x3.1–x8.1
>   by station (urban-adjacent higher); quietest ~23:00–00:00, loudest 05:00–11:00.
>   Use the per-station curves in r3_fits (diurnal__*) as the N02 level template family.
> - Rain hours tilt spectra HF-ward (slope +0.25…+0.66 vs calm baseline at the nodes),
>   confirming the shot-noise band design; LASSO (500 Hz) anchors 125–250 Hz.
> - Per-bin PSD shapes (the real fitted product) live in r3_fits.npz — generators
>   validate against THEM, compared at WINDOW/SCENE granularity (3 s + 20–300 s
>   statistics), not hour-average vs hour-average.
> - Coverage caveats: wind_high thin outside Israel (34 h IS, ~20 YW); ZG hourly ERA5
>   winds suspiciously low (verify grid-cell coords before trusting its tier edges).
> - NEW (catalog v0.2 [gen+] queue, pending approval): valve slam, gate slam, sprinkler
>   surge, burrowing, livestock pen, sabkha/frost cracks, DST micro-quake, transformer,
>   track-pitch comb, pronk impulse, helicopter-comb parametrization.
> - NEW: long scenes (>120 s) get a slow within-scene level drift drawn from the fitted
>   hour-to-hour variation (meso-scale gap closure).

Extends spec §5 (environment) and PLAN Phase 1.3 with **fitted spectral shapes and time
structure** from the R1 research sprint (5 agents, web-sourced, citations below). Confidence:
**H** = fitted/published, **M** = published but different conditions, **L** = extrapolated —
L values are randomization ranges, not constants. All ground-borne generators inject at
`v_ground` (pre-sensor, so they render through any sensor profile); electronics stay
chain-specific.

**Supersedes/refines in spec §5:** wind level law (now bilinear + terrain/vegetation-dependent
slope), rain band (onset **60–80 Hz**, not ">50 Hz"), and adds: dust storm, surf, wadi flash
flood, agricultural tonals, diurnal cultural cycle.

---

## 1. Wind (the dominant weather noise)

**Level vs speed U** — slope is TERRAIN/VEGETATION dependent (this reconciles our old
+5 dB/(m/s) with the new bilinear fit; both are real, different sites):
- Bare hard ground (bedrock, Antarctica fit): bilinear — **0.4 dB/(m/s)** below 6 m/s,
  **1.4 dB/(m/s)** above (Frankinet 2021, H).
- Vegetated/structured sites (trees, fences, posts re-radiating): up to **~5 dB/(m/s)**
  (Johnson 2019, M). → **Sample the slope 0.4–5 dB/(m/s), correlated with vegetation/structure
  density.** Onset threshold ~3 m/s (existing).

**Coupling/terrain offset** (level ∝ 1/μ, Sorrells, H): rock −20…−30 dB; gravel −10…−15;
loess/sand = 0 (reference); clay +3…+5; soft coastal +5…+8.

**Spectral shape** (velocity PSD, M): ~flat (slope 0…−0.5) from 1–10 Hz, then **f^−1.5…−2**
above 10 Hz. Vegetation adds a 1–10 Hz bump: shrubs +3–5 dB, dense maquis +8–12 dB, trees
+15–20 dB (+0.25 Hz sway line — existing). Burial −20…−40 dB (existing).

**Time structure (the new piece, H-M):** multiplicative AM envelope on the shaped noise —
log-normal envelope E(t), von-Karman-shaped modulation spectrum, integral timescale
**T_L ≈ 100/U seconds**, turbulence intensity **σ/mean ≈ 0.5–0.9 [v1.1, R3-measured; was 0.15–0.30]**. Implementation: low-pass
white noise at f_c = 1/(2πT_L) → exponentiate/normalize → multiply. Result: 2–5 dB variation
within a 3 s window, 10–25 dB over 10 min, waveform kurtosis 4–7 (non-Gaussian — matches
desert wind-noise reports). **Wind noise must NOT be stationary across a scene.**

## 2. Rain

**Band (H — refines spec):** negligible below **~60 Hz** at the surface; onset rises with
burial (~125 Hz at 0.3 m). Model gate: sigmoid((f − f_on)/10 Hz), f_on ≈ 60 + 650·depth_m.
Main energy 60–200 Hz, gentle rise then roll-off above ~200–300 Hz. **Fully in OUR band
(1000 Hz) but invisible to 100 sps reference data** — and cleanly separated from the
footstep band (10–40 Hz): rain is a high-band texture, not a footstep confuser.

**Level (H/M):** PSD(100–200 Hz) ∝ **R^α, α ≈ 1.2–1.9** (sample the range; Mediterranean fit
leans 1.9). +5…+24 dB over ambient for R = 1.5–20 mm/h. Detectable from ~1.5 mm/h. Burial
−8 dB/0.1 m (existing).

**Time structure (M):** compound Poisson shot noise — only drops >3 mm matter seismically
(90% of power); per-impact wavelet ≈ Gaussian-enveloped ~100 Hz burst, ~5–20 ms, log-normal
amplitudes. Stratiform: near-stationary with slow (10–30 min) log-normal rate drift.
Convective (Israeli pattern): bursty — Weibull inter-burst (~5 min), bursts 0.5–5 min at
10–100× base rate. Onset ramp ~30–90 s; cessation ~15–30 s.

## 3. Hail (L throughout — flag in data card)

Same band as rain; fewer, ~16× stronger impacts (0.01–1 impacts/m²/s); intermittent shafts
(duty 20–60%) with Poisson impacts inside bursts. Discriminant vs rain: excess
PSD(100–200 Hz) − 1.16·log R of +3–5 dB.

## 4. Thunder (confuser event — labeled when primary, per 1.3/3.5 rule)

Air-to-ground coupling **C ≈ 2 µm/s per Pa** (site range 1–13, H). Event recipe (H/M):
N-wave onset (0.2–1 s broadband) → reverberant decay at the near-surface resonance
**4–7 Hz**, τ ≈ 3–10 s; total duration log-normal (median ~12 s, range 5–30 s); peak energy
6–12 Hz (overlaps the footstep band → genuine confuser); optional 20–130 Hz secondary at
−10…−20 dB. Inter-event Weibull (k≈0.8, mean ~30 s) during active storm. Arrives at
acoustic speed (~340 m/s).

## 5. Dust storm / khamsin (NEW; L — no published seismic recordings exist)

Composite of existing generators, flagged as extrapolated: wind generator at U = 15–25 m/s
(its bilinear law + AM structure) **+ saltation component** = rain-like shot noise, smaller
per-impact amplitude, sustained for hours, band ~40–150 Hz. Estimated total elevation
+10–25 dB over quiet baseline at 1–20 Hz. **Real anchor wanted:** Negev IS-station days with
dust-storm weather labels (see FETCH_PLAN).

## 6. Coastal surf (NEW; M) — site background, not an event

Classic 0.1–0.5 Hz microseism is invisible to a 4.5 Hz geophone (H). What matters: breaking
waves raise **5–30 Hz** background near the shore (H), modulated by sea state; attenuates
rapidly inland (no fitted dB/km found — L; treat reach as ~hundreds of m to ~km). Implement
as a coastal-terrain background elevation with a sea-state knob; anchor against an IS coastal
station (storm vs calm days).

## 7. Wadi flash flood (NEW; H for mechanism, L for Negev numbers) — environment EVENT

Spindle/cigar envelope, emergent onset (precursor tens of s before the front), minutes-long.
Band: 2–7 Hz (water flow) + **5–15 Hz peak** + 15–45 Hz (bedload). Near-channel RMS up to
0.4–28 µm/s; up to +20 dB at 1–20 Hz at peak discharge (Alpine/Himalayan proxies). Worth
having: a strong, rare, Negev-realistic `environment` event the model should NOT call a
vehicle (it overlaps the vehicle band).

## 8. Agricultural / machinery tonals (NEW; H — best find of the sprint)

- **Irrigation pump: narrowband 75 Hz line** with a hard ON/OFF duty cycle (measured: ~5 min
  ON / 17–34 min OFF; night cycles longer). The strongest single rural noise source in the
  published field study (Dean & Al Hasani 2020). Slot into the machinery-line generator with
  duty-cycle gating — a perfect "real site has a mystery line" robustness feature (our own
  rig's 44–47/55 Hz lines are the same phenomenon).
- Tractor/farm vehicle: 10–25 Hz moving tonal+broadband — already covered by the vehicle
  class; farm context just raises its prior.

## 9. Diurnal/weekly cultural cycle (NEW; H) — scene-level modulation

Cultural noise (1–25 Hz body, also >50 Hz) swings **~11 dB day vs night** (5–15 Hz: 6–20 dB);
weekday vs weekend ~5–6 dB; rural vs urban 16 dB (day) / 30 dB (night); traffic ~50%
amplitude per doubled distance from road (10–40 Hz). → Add **time_of_day + day_type +
dist_to_road** scene variables driving the cultural-noise level; cheap and makes "nothing"
windows honest (a 3am desert "nothing" ≠ a noon roadside "nothing").

---

## 10. New scene variables introduced

`time_of_day`, `day_type` (weekday/weekend), `dist_to_road`, `dist_to_coast` + `sea_state`,
`dist_to_farm` (pump line + duty phase), `storm_state` (drives thunder rate + convective
rain), `dust_storm_flag`. All manifest-recorded.

## 11. Considered and excluded (negligible for our band/floor)

- Classic ocean microseism 0.1–0.5 Hz — below the geophone corner (H).
- Dune booming (70–105 Hz) — active dune faces only; not flat loess/gravel (M).
- Lightning EMP on electronics — documented only on interferometers, not field geophones;
  optionally a rare ms-scale glitch (L). Schumann resonances — magnetic, pT-level (H).

## 12. Real-data anchors (see datasets/FETCH_PLAN.md)

| Generator | Anchor |
|---|---|
| Wind shape+AM | YW Oklahoma 5 Hz nodes (wind tiers) + San Jacinto ZG desert (Johnson papers) |
| Rain | LASSO 2A (9 radar-confirmed rain events, Clements & Denolle 2023) |
| Thunder | PNW-Exotic templates (have) + LASSO storm days |
| Khamsin/dust | IS network Negev (HRFI) dust-labeled days |
| Surf | IS coastal station, storm vs calm |
| Cultural cycle | any IS station, hour-of-day split |
| Coupling resonance | Parker & Miltenberger: surface-planted 5 Hz nodes show a 20–40 Hz
  mechanical resonance absent when buried — direct published anchor for our coupling stage |

## Key citations

Frankinet et al. 2021 (Cryosphere 15:5007); Naderyan et al. 2016 (JGR 2015JB012478) & 2020
(GJI ggaa362); Sorrells 1971; Withers et al. 1996 (BSSA 86:1507); Johnson et al. 2019
(JGR 2018JB017151) & 2020 (GRL 088353); Coughlin 2022 (arXiv:2205.04079); Rindraharisaona
et al. 2022 (ESS 2022EA002328); Dean 2018 (ASEG rain); Clements/Larose et al. 2023 (Sci Rep
13:11384); Sens-Schönfelder et al. 2023 (Sci Rep 13:1802); Lin & Langston 2007/2009; Zhu &
Stensrud 2019; Burtin et al. 2008 (JGR 2007JB005034); Coviello et al. 2019 (JGR-ES);
Poppeliers & Mallinson 2015 (GRL); Dean & Al Hasani 2020 (TLE 39:639); Groos & Ritter 2009
(GJI 179:1213); Díaz et al. 2017; Lecocq et al. 2020 (Science 369:1338); Parker &
Miltenberger 2018 (SRL); Anthony et al. 2019 (SRL).
