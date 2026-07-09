# External datasets — manifest

> **2026-06-11:** these sets have now been mined into **`noise_atlas/`** (real footstep
> band-stats from FootprintID, ambient-noise fit ranges from PNW-Noise, confuser templates
> from PNW-Exotic) — see `noise_atlas/NOISE_ATLAS.md`.

Fetched 2026-06-09 to support the geophone car/person/nothing classifier. Goal: extra
labeled footstep/vehicle waveforms + a diverse "nothing"/noise corpus for noise-robustness
pre-training.

> **Read this before using any of it.** Every set below is from a *different sensor / sample
> rate / unit / band* than your 4.5 Hz geophone → AD620 → ADS1015 chain. None can be merged
> raw into your training set. They are only useful after reconciliation (see "Before use").

---

## What was fetched

| Dir | Dataset | Contains | Sensor | Rate | Units | Size | License |
|---|---|---|---|---|---|---|---|
| `footprintid/` | FootprintID | Footsteps (10 people × 8 speeds) | **geophone SM-24** | 1000 Hz | velocity→V | 216 MB | CC-BY-4.0 |
| `md-vibe/` | MD-Vibe | Footsteps / gait (clinical) | **geophone SM-24** | 500 / 25600 Hz | velocity→V | ~1.0 GB | CC-BY-4.0 |
| `sensit/` | SensIT (SITEX'02) | **Vehicles** (tracked/wheeled) + noise | geophone (UGS) | 4960 Hz raw | 100-dim FFT **features** (not raw) | 49 MB | open (DARPA-origin) |
| `pnw-exotic/` | PNW-ML Exotic | Surface events, explosions, **thunder, sonic booms** | broadband | 100 Hz | velocity | 3.8 GB | CC-BY-4.0 |
| `pnw-noise/` | PNW-ML Noise | Ambient pre-P noise (>50k windows) | broadband | 100 Hz | velocity | 17.3 GB | CC-BY-4.0 |
| `scedc-noise/` | SCEDC sample | Continuous "nothing", 6 station-days (day 2017-010) | broadband | 100 Hz (HHZ) | counts | 56 MB | SCEDC public (cite SCSN) |

Relevance ranking to **your** task: FootprintID ≈ MD-Vibe (true geophone footsteps) > SensIT
(only open vehicle seismic, but features-only) > PNW-Exotic (impulsive non-EQ transients) >
PNW-Noise / SCEDC (ambient noise — weak transfer, band/sensor mismatch).

---

## Sources (verified URLs)

- FootprintID — https://zenodo.org/records/4691144  (SM-24, 1000 Hz floor vibration)
- MD-Vibe — https://zenodo.org/records/8125744  (filenames have spaces; use the `/api/.../content` endpoint)
- SensIT — https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html  (`combined.bz2` train, `combined.t.bz2` test)
- PNW-ML (Exotic/Noise) — https://github.com/niyiyu/PNW-ML ; files on SeisBench dCache: `https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/pnwexotic|pnwnoise/`
- SCEDC on AWS Open Data — `s3://scedc-pds` (anonymous, region us-west-2): https://scedc.caltech.edu/data/cloud.html

## File formats
- `.mat` (FootprintID, MD-Vibe): MATLAB; load with `scipy.io.loadmat` or `h5py` (v7.3).
- `.bz2` (SensIT): LIBSVM text after `bunzip2`; 100 features = 50 acoustic + 50 seismic FFT bins, 3 classes (AAV / DW / noise).
- `.hdf5` (PNW): SeisBench format — `metadata.csv` indexes traces in `waveforms.hdf5`. Easiest via `pip install seisbench` then `seisbench.data.PNWNoise()` / `PNWExotic()`, or read HDF5 directly with `h5py`.
- `.ms` (SCEDC): miniSEED, 24 h single channel; read with ObsPy `obspy.read()`.

---

## Before use — reconciliation each set needs

The current SNN pipeline hardcodes **1000 Hz**, **20–100 Hz bands**, and a **volt-scale**
normalization (`geophone_snn_classifier.ipynb`). Foreign data run through it unmodified
produces garbage. Minimum steps before any of this helps:

1. **Resample** every source to one common rate (e.g. your deployment rate). PNW/SCEDC are
   100 Hz; FootprintID 1000 Hz; MD-Vibe 500 Hz. Decimate with an anti-alias filter.
2. **Band-limit** all sources to the shared physical band (~1–50 Hz for footsteps). Broadband
   sets carry energy outside your geophone's response — filter it out so the model can't cheat
   on instrument signature.
3. **Per-window normalize** (z-score or peak) — do NOT mix absolute amplitudes across sensors.
   This is the single biggest leak: different gains/units let a classifier separate datasets by
   scale instead of by physics.
4. **Use as pre-training, not as merged labels.** Pre-train a feature encoder on the pooled,
   normalized data (esp. the noise) for robustness, then **fine-tune the classifier head on
   your own matched car/person/nothing recordings**. Merging foreign "nothing" raw into your
   training set teaches "this is a different sensor", which inflates val accuracy and hurts the
   field. (See conversation notes on the dataset-bias trap.)
5. SensIT is features-only (no waveform) — it can't be filtered/segmented; treat it as a
   separate feature-space reference for the vehicle class, not as raw signal.

## Not fetched (available if wanted)
- INSTANCE-noise (17.7 GB), LenDB (15 GB, 20 Hz), STEAD noise chunk (~14.6 GB), TXED (70 GB) —
  more broadband EQ-noise; skipped for weak transfer / budget.
- Comprehensive DAS Event-Classification (~46 GB, Figshare, CC-BY-NC-ND) — has walk/run/car
  classes but is DAS *strain*, a different modality; ND license restricts derived redistribution.
- NCEDC (`s3://ncedc-pds`, 2,640 stations / ~184 TB) and IRIS/EarthScope FDSN (~2,600 stations)
  — the likely "100 TB / 2600 locations" archives; pull subsets via `aws s3 cp --no-sign-request`
  or ObsPy FDSN. SCEDC sample here demonstrates the pattern.
- Geophone **vehicle** set (Kyushu, 250 Hz, Sci. Reports 2025) — best vehicle match but
  email-request only (github.com/MohamedHassanSaad/Vehicle-Classification has code only).
