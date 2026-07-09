# Geophone Synthetic-Sensing System

A synthetic-geophone **digital-twin** pipeline: it simulates a single vertical
4.5 Hz geophone's response to human / vehicle / animal activity over realistic
terrain, generates a large physics-grounded training corpus, and trains a
feature-input spiking neural network (SNN) to classify **human / vehicle /
animal / nothing** from that one sensor. A separate research track extends the
idea to **indoor footstep localization** ("house localization") from exterior
geophones.

> **Source-available, Noncommercial.** Licensed under [PolyForm Noncommercial
> 1.0.0](LICENSE) — free for academic/research/evaluation use; commercial use
> requires a separate license (afiklahav@gmail.com). Third-party components: see
> [`THIRD_PARTY.md`](THIRD_PARTY.md).

## What's here

| Path | What it is |
|---|---|
| `simgeo/` | The generator. Active version in `simgeo/simgeo_v42/`; frozen predecessors in `simgeo/Legacy/{simgeo_v4, simgeo_v2}` (reproduce `corpus_v4` / `corpus_v2`). |
| `terrain_models/` | Layered-earth soil-profile definitions (`library*.json`) + provenance. Green's-function (GF) banks regenerate into here. |
| `dataset_validation/` | The **Corpus Design Invariant (CDI)** acceptance gates + the real-vs-synthetic discrepancy analysis. The instrument that proves a corpus is class-unbiased. |
| `scripts/` | Training / evaluation / analysis tooling (`scripts/eval/` = synth→real transfer). |
| `research/` | Literature & design research, incl. the house-localization direction (`research/house_localization/`). |
| `datasets/` | Real-noise atlas (`noise_atlas/r3_fits.npz` — the fitted noise model), external-dataset docs, and fetch scripts. |
| `webui/` | GeoSense web-UI prototype (simulated backend). |
| `Arduino sketch/` | Geophone data-acquisition firmware. |
| `snn_*_out/` | Trained model weights + evaluation results. |
| `docs/` | Reference specifications & primers (corpus / scene / feature / model). |
| `notebooks/` | Jupyter notebooks (classical-separability demo; v1/v2 training). |

## Pipeline

```
terrain_models/library*.json                 (soil-profile definitions)
        │  gfbank_build.py        → terrain_models/*.npz   (Green's-function banks)
        │  snr_maps.py            → SNR(range) maps
        ▼
generate_corpus_v4.py            → corpus            (synthetic scenes)
        │  label_windows.py       → per-window labels
        │  extract_features_v4.py → 132-feature vectors
        ▼
scripts/…                        → train the SNN
scripts/eval/… + dataset_validation/ (CDI gates)  → validate + transfer-test
```

## Reproducing

1. **Point at your data location:** set `GEO_DB_ROOT` (defaults to `G:/geophone_synth`).
   Heavy artifacts (GF banks, corpora, features) live there, not in the repo — they
   are regenerable.
2. **Install deps:** `pyprop8` (GF banks), `numpy`/`scipy` (features), `torch` +
   `spikingjelly` (model). Use the system Python interpreter.
3. **Regenerate + run** (per-folder READMEs give the exact commands):
   `gfbank_build.py` → `snr_maps.py` → `generate_corpus_v4.py` → `label_windows.py`
   → `extract_features_v4.py` → train.

The only external data needed to *generate* is `datasets/noise_atlas/r3_fits.npz`
(the distilled real-noise model, ~1 MB) — it's included. The raw recordings and
`datasets/real_noise_fetch/` scripts are only needed to *re-fit* that model.

## Reference docs

All in [`docs/`](docs/). Corpus & scene spec: `CORPUS_DESIGN_INVARIANT.md`,
`SCENE_CATALOG.md`, `SCENE_HYPERPARAMS.md`, `CLASS_LIST_AND_VARIABLES.md`,
`FEATURE_FORMULAS.md`, `NOISE_MODEL.md`. Model: `SNN_PRIMER.md`,
`SNN_ARCHITECTURE_RECO.md`, `SNN_TRAINING.md`. Data map: `DATA_LOCATIONS.md`.
Third-party attributions: [`THIRD_PARTY.md`](THIRD_PARTY.md).
