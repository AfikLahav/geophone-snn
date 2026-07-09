# simgeo/ — the synthetic-geophone generator

The physics engine that turns soil profiles + source models into a synthetic
single-geophone training corpus.

- **`simgeo_v42/`** — the **active** generator. All current work happens here.
- `Legacy/simgeo_v4/` — frozen; reproduces `corpus_v4` (production). See `Legacy/README.md`.
- `Legacy/simgeo_v2/` — frozen; reproduces `corpus_v2` (the paper's model).

## Active generator (`simgeo_v42/`) — module map

| module | role |
|---|---|
| `gfbank_build.py` | builds Green's-function banks (via **pyprop8**) into `terrain_models/` |
| `snr_maps.py` | precomputes SNR(range) maps for target-SNR sampling |
| `generate_corpus_v4.py` | production generator — scenes, sensor chain, target-SNR placement |
| `label_windows.py` | decoupled per-window labeling pass |
| `extract_features_v4.py` | 132-feature extraction from the corpus |
| `scenes.py` | scene assembly (GF convolution, path sampling, windowing) |
| `sources_v2.py` | source force models (human / vehicle / animal / confusers) |
| `wavelet.py` | footstep force wavelet (Hertzian heel-strike) |
| `sensor.py` | sensor chain: coupling → geophone → noise → ADC |
| `r3_noise.py` | real-anchored ambient/weather noise (uses `datasets/noise_atlas/r3_fits.npz`) |
| `coupling.py` | ground-coupling resonance model |
| `profiles.py` | soil-profile sampler (regenerates `terrain_models/library*.json`) |
| `gates.py`, `sample_gates.py` | physics QA gates, run before generation |

Analysis/validation helpers (`screen_features`, `analyze_*`, `validate_noise_*`,
`scene_ceiling`) sit alongside — tools, not pipeline steps.

## Paths
Bank/profile data resolves at `../../terrain_models/`, noise fits at
`../../datasets/`. External output locations derive from **`GEO_DB_ROOT`**
(default `G:/geophone_synth`).
