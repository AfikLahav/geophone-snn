# Real-Noise Fetch Plan (Stage R2) — APPROVED 2026-06-11, budget 100–150 GB

> **Approved changes:** budget bumped from ~25 GB to **100–150 GB** (adds rare-event tails,
> continuous multi-week stretches for non-stationarity + hybrid-mode insurance, more nodes
> for coupling spread). Storage: **`N:\geophone_real_noise\`** (4.7 TB free). Design moved
> from sampled 10-min windows to **full station-days** (windows cut later from continuous):
> YW ~16 nodes × ~25 days, LASSO ~12 × 15 (rainy + dry control), ZG ~10 × 12 (windy + calm),
> IS 3 stations × ~120 days (khamsin season + winter storms) ≈ **~100 GB** as float32 npz.

Goal: ~20–30 GB of **condition-labeled** noise windows from open archives, anchoring the
NOISE_MODEL.md generators. Weather labels via **Open-Meteo historical API** (ERA5, hourly,
any lat/lon, no key, CC-BY) joined on station coordinates × UTC day. Every window stored
with: net/sta/cha, UTC span, instrument response (StationXML), condition label + weather
values, license. **Windows without a response or a defensible label are discarded.**

Prereq: `pip install obspy openmeteo-requests requests-cache retry-requests` (system Python).

## Targets (priority order)

### T1 — YW: IRIS Community Wavefield, Oklahoma, 2016 — BEST SENSOR MATCH
- 363 Fairfield ZLand **5 Hz** 3C nodes @ 250 Hz (GS-30CT), rural.
- **VERIFIED (smoke):** lives in the **PH5 archive** (`service.iris.edu/ph5ws/`), NOT the
  standard FDSN archive (the YW code there = Tonga/Antarctica epochs — agent claim wrong);
  actual deployment **2016-06-21 → 2016-07-26** (still thunderstorm season). 16 nodes ×
  25 days, weather-labeled via Open-Meteo.
- **Anchors: wind shape+AM (primary), rain (secondary), diurnal.** ≈ 32 GB.

### T2 — 2A: LASSO, Oklahoma, Apr–May 2016 — BEST RAIN LABELS
- 1,825 ZLand **10 Hz** nodes @ 500 Hz; collocated rain gauge + radar; the 9 precipitation
  events are already identified in Clements & Denolle 2023.
- PH5 services (NOT standard FDSN): `http://service.iris.edu/ph5ws/dataselect/1/` net=2A
  (mostly DPZ vertical only; numeric station names).
- Sample: ~15 nodes × the 9 rain-event days + matched dry-control days.
- **Anchors: rain (primary — radar-confirmed), hail-candidate windows, thunder.**
- Caveat: 10 Hz corner → trust ≥20 Hz; rain band 60–250 Hz unaffected. Budget ≈ 8 GB.

### T3 — ZG: San Jacinto / Sage Brush Flats, May–Jun 2014 — DESERT TERRAIN (Israel-like)
- 1,108 ZLand **10 Hz** nodes @ 500 Hz, arid scrubland; open FDSN; wind behavior already
  published (Johnson 2019/2020) → known which station-days are windy.
- Sample: ~10 nodes × wind tiers + calm controls (Open-Meteo; nearest met ~50 km — label
  from ERA5 model values, mark confidence).
- **Anchors: wind-on-desert-terrain (validates terrain offsets), coupling resonance
  (surface-planted node 20–40 Hz ring, Parker & Miltenberger).** Budget ≈ 5 GB.

### T4 — IS: Israel Seismic Network via GEOFON — THE ISRAELI ANCHOR
- **VERIFIED (smoke):** the agent's "HH @ 100 sps" claim was wrong for this archive —
  GEOFON's open IS holdings are **SHZ @ 40 sps** (18 stations, → 1–20 Hz band) and BHZ @
  20 sps; **waveforms are OPEN only through 2021** (2022+ = TRUAA era, restricted/EIDA-auth;
  probes returned 204). GEOFON's cert chain also needs certifi (fixed in fetch_core).
- Revised: **year 2021** — Jan–Feb (winter storms/surf) + Apr–May (khamsin) — 3 probed-open
  stations (Negev / coastal / north), SHZ preferred.
- **Band caveat:** 1–20 Hz only — fine for khamsin/surf/cultural (low-band phenomena);
  the rain band (60–200 Hz) is anchored by LASSO/YW instead.
- **ZW (GEO-DESIRE 2006) checked and dropped:** metadata open (23 × HHZ @ 100 sps, Dead
  Sea) but waveforms 204 at IRIS — not actually downloadable.
- **Anchors: khamsin (only possible source — nothing published), surf, cultural cycle,
  Israeli absolute floors (1–20 Hz).** Budget ≈ 4 GB.

### Skipped (reasons logged)
- PoroTomo/Brady (industrial geothermal noise, no weather data); Long Beach (permission +
  urban); Ridgecrest 3J/7Q (redundant with ZG); GEO-DESIRE ZW_2006 (2006-era, keep as
  backup Israeli source if IS proves too restricted).

## Method (one script per target, shared core)

1. Station inventory + responses (StationXML) → pick the spatial subset.
2. Open-Meteo hourly join per station-day → condition label (wind tier / rain mm/h / dust
   flag / calm) with the labeling thresholds recorded in the manifest.
3. Fetch 6 × 10-min windows per station-day (FDSN bulk POST / PH5 GET; merge, gap-audit,
   discard windows spanning gaps).
4. `remove_response` → ground velocity (pre_filt per sample rate; counts never stored raw).
5. Store npz (velocity @ native rate) + manifest row; per-archive citation recorded.
6. QA notebook: PSD per condition bin vs the NOISE_MODEL prediction — the actual anchoring.

## Estimated totals

~6,000 windows ≈ 25 GB raw → ~8 GB as float32 velocity npz. Compute trivial; wall-clock
dominated by polite-rate downloads (overnight). AWS egress n/a (all HTTP services).

## Output layout

```
datasets/real_noise/
  manifest.parquet          (one row per window: source, label, weather, license, span)
  yw_oklahoma/   *.npz
  lasso_2a/      *.npz
  zg_sanjacinto/ *.npz
  is_israel/     *.npz
  qa/            per-bin PSD overlays vs NOISE_MODEL
```
