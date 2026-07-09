# Data locations (assets NOT in this repo)

This repo tracks code, docs, configs, metrics, and small model checkpoints.
The heavy data lives outside git. Map as of 2026-07-05:

## G:\geophone_synth (~288 GB) — synthetic corpora + features

| Dir | Size | What |
|---|---|---|
| `corpus_150k/` | 38.5 GB | v1 corpus (150k scenes). **Do not touch/regenerate.** |
| `corpus_v2/` | 155 GB | v2 corpus, full 4-blob schema. **Do not touch** — source of the production v2 model. |
| `corpus_v3/` | 84.5 GB | v3 lean corpus (167,960 scenes; clean_mv + noise_mv blobs only). |
| `features/` | 2.5 GB | v1 feature shards. |
| `features_v2/` | 3.6 GB | v2 feature parquet shards (default `GEO_FEAT_DIR`). |
| `features_v3/` | 3.9 GB | v3 feature parquet shards (7.2M windows). |
| `config/` | 0.1 GB | Run configs. |

Trainers read shards via `GEO_FEAT_DIR` and write to `GEO_OUT` (see HANDOFF.md for
the exact command chain).

## In-tree but gitignored (regenerable or archival)

- `geophone_data.db` (89 MB) — synthetic corpus SQLite; regenerable.
- `simgeo_banks/*.npz` (454 banks, 156 MB) — regenerate with
  `simgeo_banks/provenance_gfbank_build.py` + `provenance_profiles.py`
  (pyprop8; ~90 min for the 340-bank v3 delta on a 9800X3D).
  `library*.json` / `splits*.json` ARE tracked.
- `datasets/` corpora (~22.9 GB: pnw-noise, pnw-exotic, md-vibe, footprintid,
  sensit, scedc-noise, vehicle_ref) — refetch per `datasets/FETCH_PLAN.md`.
  Docs, `compat/`, `noise_atlas/` code + `r3_fits.npz` (required by
  `simgeo_v3/r3_noise.py`) ARE tracked.
- `פרויקט גמר/` (200 MB) — archival originals (duplicate DB/CSVs, Elbit PPTXs).
- `Goephone-Project/` (139 MB) — local clone of the GitHub repo
  (github.com/roeeds3/Geophone-Project); intentionally not nested into this repo.
- `webui/server/data/*.csv` (69 MB) — replay copies of the real recordings.
- `tools/` (53 MB) — arduino-cli binaries.
- `simgeo_v3_{src,scene,noise,terrain,dist,label}/` — superseded v3 dev-stage
  snapshots (NOT byte-identical to final; kept on disk as history).

## Real recordings (tracked in this repo)

`geophone_2026*.csv` at root (11 files, ~2.4 MB) — raw field sessions.
**Off-limits for training** per project policy; kept for provenance/diagnostics.
