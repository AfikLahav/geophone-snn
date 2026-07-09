# simgeo_v42/ — the active generator

The current synthetic-geophone generator. (Frozen predecessors: [`../Legacy/`](../Legacy/).)

**Pipeline order (each is a runnable entry point):**
```
gfbank_build.py       # 1. Green's-function banks (pyprop8) → ../../terrain_models/*.npz
snr_maps.py           # 2. SNR(range) maps for target-SNR sampling
generate_corpus_v4.py # 3. render the corpus (scenes + sensor chain + target-SNR placement)
label_windows.py      # 4. per-window presence labels
extract_features_v4.py# 5. 132-feature vectors
```

**Core modules:** `scenes.py` (GF convolution + path sampling + windowing), `sources_v2.py`
(human/vehicle/animal/confuser force models), `wavelet.py` (footstep wavelet), `sensor.py`
(coupling → geophone → noise → ADC), `r3_noise.py` (real-anchored ambient noise), `coupling.py`,
`profiles.py` (soil-profile sampler), `gates.py`/`sample_gates.py` (physics QA).

**Paths:** banks/profiles resolve at `../../terrain_models/`, noise fits at `../../datasets/`,
gate thresholds at `../../dataset_validation/`. External I/O roots derive from **`GEO_DB_ROOT`**
(default `G:/geophone_synth`). Module-by-module notes: [`../README.md`](../README.md).
