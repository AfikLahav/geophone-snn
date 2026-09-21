# Dataset validation

These checks measure scene coverage, class balance, signal levels, and whether background properties reveal the labels. They identify weaknesses in a generated dataset; a good result on one check does not establish that the dataset is free of bias.

| Script | Check |
|---|---|
| [invariant_check.py](invariant_check.py) | Class balance, coverage, and associations between class and sampling conditions |
| [coverage_crossings.py](coverage_crossings.py) | Coverage of pairs of scene properties |
| [nuisance_probe.py](nuisance_probe.py) | Whether a classifier can infer scene class from windows with no active source |
| [amplitude_gate.py](amplitude_gate.py) | Signal levels, clipping, shape, narrow frequency peaks, and input reconstruction |
| [coverage_report.py](coverage_report.py) | Coverage across source strength and activity levels |
| [datasheet_gen.py](datasheet_gen.py) | Dataset summaries from the generated labels and scene metadata |

The scripts read generated data outside the repository. Check each script's arguments before running it; some diagnostic defaults refer to earlier dataset versions. Running `datasheet_gen.py` also writes a Markdown summary, which would replace the edited datasheet at the same output path.

## Recorded v4.3.1 results

The [datasheet](DATASHEET_v431.md) summarizes the saved measurements. [INVARIANT_CHECK_v431.json](INVARIANT_CHECK_v431.json) records 3,292,768 windows and reports that 96.4% of the assessed source-and-strength groups meet the minimum count.

The [background-only probe](NUISANCE_PROBE_v431.json) reports a maximum area under the ROC curve of 0.6538, above its target of 0.55 or less. Here, 0.5 means chance-level separation. Some class information therefore remains in windows without an active source. The combined validation score is 0.585, with this probe as its limiting check. These records do not show that every validation target passed.

[COVERAGE_CROSSINGS_v431.json](COVERAGE_CROSSINGS_v431.json), [AMPLITUDE_GATE.json](AMPLITUDE_GATE.json), and [dataset_v4_coverage.json](dataset_v4_coverage.json) contain the detailed coverage and amplitude records. The last file retains its older filename. [discrepancy](discrepancy) contains comparisons between synthetic and real signals. Historical records are kept separate from the files explicitly marked `v431`.
