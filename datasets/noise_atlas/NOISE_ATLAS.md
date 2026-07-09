# Noise Atlas — real-data references for the simulation (built 2026-06-11)

Mined from the previously downloaded external datasets (`datasets/DATASETS.md`) to ground the
sim's noise generators and Gate 3 source checks in measured reality instead of assumptions.
Built by `build_atlas.py` (seeded, reproducible). **Everything here is shape/parameter
reference — the absolute amplitude anchor remains our own rig (Gate 1).**

## A. FootprintID — the Gate 3 footstep reference (the big result)

SM-24 geophone @ 1000 Hz (same sensor type + rate as ours), 10 people × 8 speeds, indoor
floor. 297 footstep-active segments + 1,587 quiet segments analyzed
(`footprintid_footstep_stats.csv`, `footprintid_noise_psds.npz`).

| Metric (1–120 Hz basis) | p10 | median | p90 |
|---|---|---|---|
| **10–40 Hz band fraction** | **0.84** | **0.92** | **0.96** |
| spectral centroid (Hz) | 21.6 | 25.6 | 30.1 |
| cadence (steps/s) | 1.60 | 1.68 | 1.74 |

**Use:** real geophone footsteps put ~84–96% of their 1–120 Hz energy in the 10–40 Hz band,
centroid ~26 Hz. Our likeness-probe sim footstep had **1% / 5.9 Hz** — confirming the smooth
GRF wavelet is missing the impact transient (PLAN 3.1 BINDING). This table is the Gate 3
reference distribution.

**Caveat:** FootprintID is an *indoor floor* (structure-borne propagation) — the band
fraction partly reflects floor physics, not open ground. Set the Gate 3 threshold generously
below the p10 (e.g., sim footsteps on hard terrain should reach tens of %, not necessarily
≥84%) and refine with open-ground literature values.

## B. PNW-Noise — ambient-noise randomization range

600 windows (150 s, Z component) stratified over 390 stations, `pnw_noise_psds.npz` +
per-window fits in `pnw_noise_fits.csv`.

**Data quality finding:** 229/600 windows are band-limited (native 40/50 Hz channels
resampled to 100 Hz → spectrum collapses above ~20 Hz). Flagged `highband_collapsed`;
fits use **1–20 Hz** where all windows are real.

| Colored-noise fit, 1–20 Hz (full-band windows only) | p10 | median | p90 |
|---|---|---|---|
| log-log slope (P ∝ f^b) | −2.0 | −1.15 | +0.04 |
| level spread across sites | — | **~2.2 decades** at 10 Hz | — |

**Use (Phase 1.3):** the site-ambient generator should sample its colored-noise slope from
roughly **f⁰ (white) to f⁻² (red), centered near f⁻¹**, with a **~2-decade level spread**
across sites — instead of assuming one Peterson curve. Absolute level is still anchored to
our rig's measured 0.08 mV RMS floor; this gives the *cross-site randomization shape/range*.

## C. PNW-Exotic — confuser band-shape templates

Median normalized PSD templates (Z, 100 Hz) from full-band windows only:
**thunder** (13 windows) and **sonic boom** (26), `exotic_templates.npz`, valid ~1–45 Hz.
Use as Gate 3 *band-shape* references for the 3.5 confuser classes (by-analogy rule:
shape + structure, not absolute dB).

## Not mined (deliberate)

- **SCEDC miniSEED** — needs ObsPy (not installed); PNW already covers ambient. Optional later.
- **SensIT** — FFT features only, no waveforms; usable as a vehicle feature-space reference, not for noise.
- **MD-Vibe** — zipped clinical footsteps (500 Hz); redundant with FootprintID for this purpose. Optional later.

## How the sim consumes this

1. **Phase 1.3 noise generators:** sample (slope, level-offset) from the §B distribution;
   keep rig-fitted lines/floor as the anchor site.
2. **Gate 3 footstep assertion:** sim footstep 10–40 Hz band fraction compared against §A
   (threshold TBD below p10, terrain-dependent).
3. **Hybrid mode (optional, later):** real PNW noise beds (1–20 Hz, instrument-corrected,
   re-rendered through our sensor chain) + synthetic sources — the classic seismology
   augmentation. Not required for v1.
