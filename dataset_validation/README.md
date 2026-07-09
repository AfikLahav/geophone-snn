# dataset_validation/ — Corpus Design Invariant (CDI) acceptance gates

The instrument that **proves a generated corpus is class-unbiased and complete** —
the rigor that makes the synthetic dataset defensible.

## The six gates

| gate | file | checks |
|---|---|---|
| B/C/N | `invariant_check.py` | balance, coverage, nuisance-neutrality; emits the CDI scalar |
| X | `coverage_crossings.py` | pairwise combinatorial coverage |
| P | `nuisance_probe.py` | label leakage on zero-signal windows (the usual bottleneck) |
| A | `amplitude_gate.py` | RMS / clipping / kurtosis / line sanity + D2 reconstruction |
| C3 | `coverage_report.py` | loudness & occupancy coverage |
| D | `datasheet_gen.py` | Gebru-style corpus datasheet |

`derive_gates.py` computes the matched-filter thresholds (`gates*.json`) that the
gates, trainer, and eval all consume.

## Records & analysis
- `*_v42.json`, `AMPLITUDE_GATE.json`, `corpus_v4_coverage.json`, `DATASHEET_v42.*`
  — the acceptance record for the shipped corpus.
- `discrepancy/` — the real-vs-synthetic gap analysis (structure / waveform /
  spectral / feature-space).
