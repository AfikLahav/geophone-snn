# scripts/eval/ — synth→real transfer & confound decomposition

Transfer metrics and the confound-decomposition experiments (the *dataset*-quality CDI gates
are separate, in [`../../dataset_validation/`](../../dataset_validation/); accuracy tables /
diagnostics live in [`../analysis/`](../analysis/)).

- `eval_synth_to_real.py` — the primary zero-shot synth→real transfer metric (AUROC + accuracy).
- `stratified_auroc.py` — SNR-stratified AUROC (v2 vs v3+ comparison).
- `exp_folds.py` — fold-scheme contamination study.
- `real_label_audit.py` — decompose the accuracy gap (label noise vs calibration).
- `terrain_match.py`, `make_terrain_subset.py` — terrain-matching confound experiment (ET).
- `domain_gap_v4.py` — domain-classifier AUROC + per-feature separability.
- `car_scale_sweep.py` — vehicle gain-mismatch sweep.
- `exp_coupling.py`, `coupling_verify.py` — coupling-model experiment + real-rig fc measurement.
- `gated_rescore.py` — re-score under the detectability gate (apples-to-apples baseline).

Read external features + real data (`GEO_DB_ROOT`, `Goephone-Project`) — provenance, not turnkey.
