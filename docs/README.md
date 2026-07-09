# docs/ — reference specifications & primers

The authoritative specs behind the generator, corpus, features, and model.

**Corpus & scene design**
- `CORPUS_DESIGN_INVARIANT.md` — the CDI acceptance formula + the 8 gates (the corpus-quality instrument; enforced by `dataset_validation/`).
- `SCENE_CATALOG.md` — the ~90 scene archetypes (human/vehicle/animal/nothing).
- `SCENE_HYPERPARAMS.md` — binding numerical distributions per archetype.
- `CLASS_LIST_AND_VARIABLES.md` — full class taxonomy + parameter spec.
- `NOISE_MODEL.md` — the R3-anchored noise generator spec.

**Features**
- `FEATURE_FORMULAS.md` — canonical formula for all 132 features (source of truth for `simgeo/.../features.py`).

**Model**
- `SNN_PRIMER.md`, `SNN_TRAINING.md`, `SNN_ARCHITECTURE_RECO.md` — spiking-neural-net pedagogy, training method, and cited architecture rationale.
- `SPIKINGJELLY.md` — the SpikingJelly framework guide.

**Pipeline / data map**
- `DATA_LOCATIONS.md` — where the heavy external data lives (the `GEO_DB_ROOT` map).
- `V4_EXPERIMENTS.md` — the v4 training experiment registry.
- `V4_2_PLAN.md` — the v4.2 diversity-corpus execution plan.

House-localization research is separate — see [`../research/`](../research/).
