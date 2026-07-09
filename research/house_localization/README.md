# House-Localization Research Campaign ("wallhacks")

Research surveys supporting the pivot from the outdoor single-geophone detection system to an **indoor footstep-localization** system: N geophones on/around a concrete/brick house that classify (human/animal/nothing) and **locate** a person's position via structural vibration.

**Provenance & status.** Each document below was produced by a background research agent (Sonnet, 2026-07-08) under a rigor mandate, reading our existing `research/` docs first. These are **literature/tooling surveys — design input, not validated experiments.** Numbers are as reported in the cited sources; our own sim-to-real path carries additional risk not captured here.

## Index

| # | Document | Status | Headline finding |
|---|---|---|---|
| 01 | [Model architecture](01_model_architecture.md) | ✅ | LightGBM classifier + matched-field/fingerprint localizer; **drop the SNN**; models 100–500× smaller than the 220K SNN |
| 02 | [Indoor localization methods](02_indoor_localization_methods.md) | ✅ | **Transfer-function/matched-field beats TDOA**; ~0.3–0.6 m achievable, CRLB floor ~5–15 cm (velocity-calibration limited) |
| 03 | [House architecture datasets](03_house_architecture_datasets.md) | ✅ | Use concrete-frame refs (ResPlan/RPLAN/OSM-Israel), **not** US timber datasets; Israeli construction defaults given |
| 04 | [Sensor hardware + MEMS hybrid](04_sensor_hardware_hybrid.md) | ✅ | 4.5 Hz geophone confirmed optimal; **geophone+ADXL355 hybrid = RS4D**; **ADS1115 cannot do TDOA — must swap** |
| 05 | [Structural reverberation & multipath](05_structural_reverberation.md) | ✅ | Flexural dispersion (c∝√f); reverberation is **foe for TDOA, friend for matched-field/fingerprinting** |
| 06 | [Sim construction + cost](06_sim_construction_cost.md) | ✅ | **Devito voxel-FDTD** (1–2 wk) + shell-FEM validation; reciprocity 60×; 500 houses @100 Hz = 7.5 hr on one RTX 3090; GF library ~0.8–1.9 GB |
| 07 | [Sim-engine verification + plan](07_sim_engine_verification_and_plan.md) | ✅ | **Devito confirmed — but OSS has NO elastic free surface** (build it, ~1 wk, validate vs Lamb+SW4); Salvus = paid fallback; full install/MWE/reciprocity/scaling plan |
| 08 | [Elastic free-surface implementation](08_free_surface_implementation.md) | ✅ | **Stress-image (Kristek 2002) is required** — vacuum gives ~10–15% Rayleigh amplitude error at 9 PPW, stress-image gives ~5%; complete Devito SubDomain recipe (~100–130 lines); Lamb validation protocol (P-arrival ±1%, |Vz/Vr|=1.47 ±10%, SW4 cross-check ±5%); effort 1–1.5 wk; Salvus is the fallback |
| 09 | [Procedural house generation](09_procedural_house_generation.md) | ✅ | **Parametric generator + OSM footprints + RPLAN/ResPlan priors** recommended over neural (House-GAN) or CityEngine; rasterio `all_touched=True` wall rasterization; 1-voxel minimum at dx=0.15 m (150 mm wall); 200–500 houses; soil Vs is the #1 diversity axis |
| 10 | [Devito numerics, GPU & visualization](10_devito_numerics_gpu_viz.md) | ✅ | **Thin walls need Moczo harmonic/arithmetic averaging** (reduces staircasing bias from ~25% to ~5–10%; walls still under-resolved at 100 Hz); footstep source = τ_zz injection (not explosive); reciprocity: 1 shot/sensor with f_z → record v_z; RTX 3090 OpenACC ~5 GP/s → 7.7 s/sim at 100 Hz, 5 min/sim at 250 Hz; 100→250 Hz = 39× cost; sponge (nbl=20) + stress-image corner is stable; PyVista `ImageData.add_volume` for wavefield animation |

## Synthesis

A combined cross-agent synthesis (best-achievable accuracy + the sim/sensor/method/model stack to reach it, and where agents disagree) will be added here once report 06 lands. The convergent thread so far: **the simulated Green's-function corpus does triple duty** — it trains the classifier, *is* the matched-field localization dictionary, and serves as the fingerprint database; the sim must nail **flexural dispersion in a realistically-dimensioned Israeli concrete slab**, and the reverberation it produces is the *signal* for the winning localization methods.
