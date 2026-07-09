# scripts/analysis/ — accuracy reports & one-off diagnostics

Accuracy tables, detectability curves, and root-cause / dataset-gap analyses (not part of the
production pipeline; kept as provenance).

**Accuracy & metrics**
- `accuracy_analysis.py`, `accuracy_compare.py` — accuracy tables (threshold/readout ablation, synth+real).
- `metrics_report.py` — per-head accuracy / balanced-acc / F1 / AUROC table.
- `detectability_curve.py` — empirical Pd(SNR) curves per class (SNR50/90).
- `error_analysis.py` — real-data error taxonomy + confidence analysis.
- `hysteresis_sweep.py` — per-fold hysteresis sweep.

**Dataset-gap diagnostics**
- `gap_analysis.py` — localizes the dataset gaps behind faint-human misses.
- `domain_feature_audit.py` — per-feature sim-vs-real domain AUC.
- `missed_window_analysis.py` — characterizes missed real windows vs background.
- `chunk_sensitivity.py` — real-data chunk-length sensitivity.
- `compare_15s.py` — PSD/feature comparison of a specific recording.

Most hardcode external data paths (`GEO_DB_ROOT`, `Goephone-Project`) — provenance, not turnkey.
