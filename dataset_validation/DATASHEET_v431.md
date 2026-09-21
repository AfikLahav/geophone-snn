# Synthetic dataset v4.3.1

The dataset combines simulated people, vehicles, animals, background scenes, and mixtures of sources. It was built to train detection models across varied ground conditions without recording every combination in the field. The full feature set also includes measurements that may be useful for more detailed classification.

This page summarizes the saved [datasheet measurements](DATASHEET_v431.json) and [validation results](README.md). Those records describe a generated dataset; they are not measurements from a new run of the current repository.

## Size and labels

| Property | Value |
|---|---|
| Generator's default scene count | 171,700 |
| Windows in the saved validation records | 3,292,768 |
| Declared random seed | 20260706 |
| Sampling rate | 1,000 Hz |
| Window length / hop | 3 s / 1.5 s |
| Extracted features / final model inputs | 132 / 77 |

Presence labels follow source activity. Signal-to-noise ratio is recorded separately for each source class, so a source can be present without being easy to detect. Adjacent windows overlap; they must not be treated as independent observations when splitting a recording for evaluation.

An earlier repository summary gave 3,298,170 windows. The included v4.3.1 datasheet and invariant records agree on 3,292,768. The difference of 5,402 windows has not been reconciled against the full generated files.

## Scene composition

These are the recorded proportions of windows, rather than the requested proportions of scenes. Scene duration and source activity affect the resulting window counts.

| Class | Windows | Included source types |
|---|---:|---|
| No target | 55.52% | Ambient background, machinery, overflight, distant traffic |
| Animal | 13.16% | Dog, jackal, boar, horse, sheep, herd, slow quadruped, horse with rider |
| Vehicle | 11.73% | Car, truck, motorbike, bicycle, tractor, tracked vehicle, convoy, two vehicles, idle and departure |
| Human | 10.08% | Walking, running, quiet walking, child, group, marching, carrying a load |
| Mixed | 9.51% | Human with vehicle, human with animal, vehicle with animal |

The [JSON record](DATASHEET_v431.json) retains the proportions of individual source types. The ground library contains 340 ground profiles and 18 modal floor profiles. Vehicle scenes exclude profiles with shear-wave speed below 95 m/s.

## Weather and source strength

Weather is assigned independently of the target class at scene generation. The recorded window proportions are 35.47% calm, 19.96% light wind, 14.67% moderate wind, 9.78% strong wind, 12.06% light rain, and 8.05% heavy rain. Small differences between classes remain after windowing.

Source placement uses signal-to-noise ratio versus distance maps, with corrections for source type. The declared sampling distribution favors intermediate signal levels while retaining weak examples, including a 15% tail below the lower detection gate. This gate is a simulation sampling reference, not a measured detection limit for a trained model.

## Sensor settings and reproduction

The saved metadata describes a wider range of recording conditions than the current default settings:

| Setting | Saved dataset record | Current default |
|---|---|---|
| Base-10 logarithm of gain | 0.3–2.9 | 2.3–2.9 |
| Quantization step | 0, 0.125, 0.2, or 0.25 mV, roughly equally represented | No quantization |
| Clipping limit | 256 mV, 512 mV, or no clipping, roughly equally represented | ±256 mV |
| Coupling-filter metadata | Roughly equal `lowpass` and `bump` entries | `lowpass` entry |

`GEO_GAIN_LOG10`, `GEO_QUANT_MIX`, and `GEO_RAIL_MIX` control the first three settings. `GEO_COUPLING_MODE` controls the stored coupling entry. However, the current v4.3 rendering path bypasses both the coupling filter and the high-frequency resonance filter. Their presence in metadata does not establish that those filters were applied. The saved declaration also names the `v41` mains-noise setting, while `GEO_LINES_MODE` currently defaults to `v4`.

Exact reproduction therefore requires checking the original run settings as well as the seed and code version. The retained metadata is preserved in JSON; its older design notes should not be read as a description of every operation in the current code.

## Validation limits

The saved coverage check reports that 96.4% of the assessed source-and-strength groups meet the minimum count. The probe on windows without active sources reaches 0.6538 ROC area, above the stated target of 0.55. This indicates that some class information remains in background properties. The dataset covers the intended combinations well by these checks, but the records do not support claiming that it is free of unintended class cues.
