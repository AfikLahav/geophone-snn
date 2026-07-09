# Legacy/ — frozen predecessor generators

Kept for **reproducibility**, not active development.

- `simgeo_v4/` — reproduces `corpus_v4` (the production corpus).
- `simgeo_v2/` — reproduces `corpus_v2` (the paper's working model).

**Do not edit these** — they exist to regenerate their exact corpora bit-for-bit.
The active generator is [`../simgeo_v42/`](../simgeo_v42/).

Their bank/data paths were adjusted for this nested layout: they resolve
`terrain_models/` and `datasets/` at the repo root (`HERE/../../../…`).
