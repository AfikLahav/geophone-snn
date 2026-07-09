# Third-Party Components & Attributions

The [`LICENSE`](LICENSE) (PolyForm Noncommercial 1.0.0) covers **only the Owner's
own original work and derivatives**. The components below are used under their own
licenses and are attributed here as those licenses require. All are **CC-BY-4.0
or permissive** — commercial use and derivatives are permitted **with attribution**;
no non-commercial or copyleft/share-alike terms apply to anything shipped here.

## Software

### pyprop8
- **Author:** Andrew P. Valentine (Australian National University / Durham University)
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Source:** https://github.com/valentineap/pyprop8
- **Citation:** Valentine, A. P. (2022). *pyprop8: A lightweight code to simulate
  seismic observables in a layered half-space.* Journal of Open Source Software,
  7(76), 4217. https://doi.org/10.21105/joss.04217
- **Use:** imported as a dependency to compute layered-half-space Green's functions
  in `simgeo/simgeo_v42/gfbank_build.py`. Not modified; not redistributed (installed via pip).

### Runtime libraries
NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), PyTorch (BSD-3-Clause), SpikingJelly
(MIT) — standard dependencies used under their respective permissive licenses.

## Datasets (used to fit the noise model)

The shipped `datasets/noise_atlas/r3_fits.npz` is an aggregate **statistical
derivative** (fitted median PSD shapes + per-condition level ratios) of the datasets
below. **Raw recordings are not redistributed** in this repository.

| Dataset | License | Source | Use |
|---|---|---|---|
| FootprintID | CC-BY-4.0 | https://zenodo.org/records/4691144 | footstep band-statistics reference |
| PNW-ML — Noise | CC-BY-4.0 | https://github.com/niyiyu/PNW-ML | ambient-noise PSD fit ranges |
| PNW-ML — Exotic | CC-BY-4.0 | https://github.com/niyiyu/PNW-ML | confuser band-shape templates (thunder / sonic boom) |

A CC-BY-NC-ND DAS dataset was deliberately **excluded** from the atlas to avoid
non-commercial/no-derivatives restrictions. See `datasets/DATASETS.md` for the full
manifest of sources considered.
