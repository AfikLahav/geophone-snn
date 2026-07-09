> **Sim-engine verification + adoption plan — house-localization pivot.** Background research agent (Sonnet) + direct web checks, 2026-07-08. Skeptical verification of Devito for 3-D elastic house+soil simulation, alternatives, and a concrete getting-started plan. Indexed in [README.md](README.md). Supersedes the engine recommendation in [06_sim_construction_cost.md](06_sim_construction_cost.md) on the **free-surface** point.

---

# HEADLINE

**Devito is the right primary engine — BUT open-source Devito does not provide an elastic free surface out of the box.** `06` assumed the free surface was free; it is not. Since footstep signals at exterior/perimeter sensors depend on **Rayleigh/surface waves + traction-free building faces**, the free surface is the crux. You must **implement it yourself (~80–120 lines, vacuum + Kristek stress-image method) and validate it against Lamb's problem + an SW4 cross-check BEFORE generating any data.** That is the mandatory first milestone. If it isn't working in ~1 week, fall back to **Salvus** (commercial, academic license by application).

---

## PART 1 — Devito verification (skeptical, per criterion)

| Criterion | Verdict | Notes |
|---|---|---|
| 3-D elastic (velocity–stress / Virieux) | **PASS** | `VectorTimeFunction`+`TensorTimeFunction` are dimension-agnostic; `examples/seismic/elastic` tests 3-D. Demonstrated at 5.9 B grid points (SLIM 2020). *Docs lag: all tutorials are 2-D.* |
| **Free surface (elastic, traction-free)** | **FAIL (open-source)** | `freesurface()` is **acoustic-only**. `devitoboundary` **archived Feb 2023**; successor `Schism` is DevitoPRO-only ("OSS support ended"). Elastic immersed-boundary free surface is a **DevitoPRO experimental** feature, validated only on *acoustic* topography (no published Rayleigh validation). **Solvable in ~80–120 lines yourself** (vacuum / Kristek 2002 stress-image), but adds a validation burden. |
| GPU 3-D elastic | **CONDITIONAL PASS** | OSS uses **OpenACC via NVIDIA HPC SDK `nvc`** (not stock CUDA) → ~40–60% slower than hand-CUDA. CUDA/mixed-precision is DevitoPRO. Works on RTX 3090; **Docker image recommended**. |
| Strong velocity contrast / heterogeneity | **PARTIAL PASS** | Per-cell NumPy arrays for λ/μ/ρ; 10:1 concrete/soil is stable. **Thin-wall staircasing is real** (0.15 m wall at dx=0.1 m = 1–2 cells → amplitude error). Mitigate with effective-medium averaging, or document the bias. |
| Absorbing boundaries (elastic) | **PARTIAL PASS** | Cosine-taper **sponge (`model.damp`) works** and is fine for training data; CPML is DevitoPRO / user-implemented. |
| Maintenance / license | **STRONG PASS** | v4.8.22 (Jun 2026), regular cadence, **MIT**, Imperial/GT-SLIM/DevitoCodes. |

**Net:** Devito passes on everything that matters for *batch scripting and GPU*, and fails only on the elastic free surface — which is a **one-time implementation cost**, not a recurring one.

---

## PART 2 — Alternatives

| Tool | 3-D elastic | Free surface | Python API | GPU | Batch 1000s | License | Verdict |
|---|---|---|---|---|---|---|---|
| **Devito (OSS)** | Yes | **No (build it)** | **Excellent** | OpenACC (ok) | **Excellent** | MIT | **PICK** |
| SW4 | Yes | **Excellent** (curvilinear) | **None (config files)** | RAJA (unstable) | None | BSD | Eliminated: no Python automation |
| SpecFEM3D | Yes (SEM) | Excellent | Partial | CUDA/HIP (good) | **Poor** (re-mesh/house) | BSD | Eliminated: meshing bottleneck |
| **Salvus** | Yes (SEM) | Excellent | **Excellent** | CUDA (2 min/run) | **Excellent** | **Commercial** | **Paid fallback** |
| OpenSWPC / SOFI3D | Yes | Yes | None (Fortran/C) | None | None | BSD/GPL | No Python/GPU |
| jwave | **Acoustic only** | — | Excellent | Excellent | Excellent | MIT | Hard fail (not elastic) |
| FEniCSx shell-FEM | Yes | Excellent (natural) | Excellent | **Limited** | Feasible | LGPL | Validation tool only |

**Why Devito still wins:** *automation is the non-negotiable requirement* — thousands of house variants = just re-filling NumPy material arrays, **no re-meshing**. SW4 (no Python) and SpecFEM (a new mesh per house = weeks) can't batch. **Salvus does everything** (SEM + free surface + Python + GPU + batch) but is commercial — it's the fallback if the free-surface build slips.

---

## PART 3 — Concrete getting-started plan

### A. Install (GPU via Docker — recommended)
```bash
# CPU baseline
conda create -n devito python=3.11 && conda activate devito
pip install "devito[extras]"

# GPU (NVIDIA) — Docker + nvidia-container-toolkit
docker pull devitocodes/devito:nvidia-nvc-latest
docker run --gpus all --rm devitocodes/devito:nvidia-nvc-latest \
  python -c "from devito import configuration; print(configuration)"
# Native GPU alt: install NVIDIA HPC SDK (free), then:
#   export DEVITO_PLATFORM=nvidiaX DEVITO_ARCH=nvc DEVITO_LANGUAGE=openacc
```

### B. Minimal 3-D elastic example (CPU first, GPU by env-var, no code change)
```python
import numpy as np
from examples.seismic import demo_model, setup_geometry
from examples.seismic.elastic import ElasticWaveSolver

shape, spacing, nbl = (600,600,300), (0.1,0.1,0.1), 30
model = demo_model('layers-elastic', ndim=3, shape=shape, spacing=spacing, nbl=nbl,
                   vp=[3600.,400.], vs=[2078.,230.], rho=[2400.,1800.])  # concrete/soil
src = np.array([[30.,30.,0.]]); rec = np.array([[x,30.,0.] for x in (10,15,20,25,35,40)])
geometry = setup_geometry(model, tn=300., src_coordinates=src, rec_coordinates=rec, f0=50.)
solver = ElasticWaveSolver(model, geometry, space_order=4)
rec_vx, rec_vy, rec_vz, *_ = solver.forward()   # rec_vz.data = (n_time, n_rec)
```

### C. Build up to house + soil — geometry is just array slicing (no mesh)
```python
vp=np.full(shape,400.,np.float32); vs=np.full(shape,230.,np.float32); rho=np.full(shape,1800.,np.float32)
vp[:,:,:2]=3600.; vs[:,:,:2]=2078.; rho[:,:,:2]=2400.     # floor slab z=0–0.2 m
vp[:2,:,:]=3600.; vs[:2,:,:]=2078.; rho[:2,:,:]=2400.     # exterior wall (N face)
model.update('vp',vp); model.update('vs',vs); model.update('rho',rho)
```

### D. **The critical piece — elastic free surface (do this FIRST)**
Simple vacuum layer (fast, 1st-order): set the top 2 cells to near-vacuum and drop the top ABL.
```python
eps=1e-6; vp[:,:,-2:]=eps; vs[:,:,-2:]=eps; rho[:,:,-2:]=eps   # "air" layer at top
model.update('vp',vp); model.update('vs',vs); model.update('rho',rho)  # + model.damp fs=True
```
For accurate Rayleigh amplitudes, add the **antisymmetric stress-image (Kristek et al. 2002)** via a `SubDomain` over the top cells (~80–120 lines). Reference: [Kristek 2002, Geophysics](https://library.seg.org/doi/abs/10.1190/1.1512752).

### E. Reciprocity — Green's-function library (N_geophone shots, not N_floor)
```python
import h5py, json
geo = [[x,y,0.] for x,y in [(5,5),(5,55),(55,5),(55,55),(30,5),(30,55),(5,30),(55,30)]]
floor = [[x,y,0.] for x in np.arange(2,59,2) for y in np.arange(2,59,2)]  # ~870 pts
with h5py.File('greensfn.h5','w') as f:
    for i,g in enumerate(geo):
        geometry.src.coordinates.data[:]=[g]; geometry.rec.coordinates.data[:]=floor
        rec_vz,*_=solver.forward()
        f.create_dataset(f'g{i}/vz', data=rec_vz.data, compression='gzip', compression_opts=4)
```

### F. Validation (MANDATORY before any data generation)
**Lamb's problem** on a homogeneous half-space (Vs=300, Vp=520, ρ=2000), vertical point force, surface receivers at 2–10 m. Pass if:
- P & Rayleigh arrival times within **1%** (Rayleigh ≈ 0.9194·Vs for ν=0.25),
- Rayleigh Vz/Vr amplitude ratio within **10%** of −1.467,
- no spurious ABL reflections.
- **Cross-check vs SW4** (free, curvilinear free surface): accept if Rayleigh matches within **5%**. Larger discrepancy → the vacuum layer needs the stress-image correction.
Analytic code: [github.com/ktkimit/lamb_2dhalf_surface](https://github.com/ktkimit/lamb_2dhalf_surface).

### G. Scale to thousands of houses (embarrassingly parallel)
- Parametric grid: `itertools.product(concrete_vp, wall_thickness, floor_area, soil_vs)` → NumPy arrays per variant, one HDF5 out per house.
- **Throughput (RTX 3090, OpenACC, realistic):** full 250 Hz grid (~157 M cells, 12 k steps) ≈ **6–10 min/run** → 1000 runs ≈ 4–7 days sequential. Knock it down with: building-only smaller domain (~30 s/run → ~8 h), 100 Hz coarser grid (~75 s/run → ~21 h), 4 GPUs, or cloud A100 spot.
- **Storage:** ~50–150 MB/house gzipped → ~**100 GB for 1000 houses**.

---

## Decision tree

```
Implement + validate elastic free surface (vs Lamb + SW4) in ~1 week?
├─ YES → Devito (OSS), Docker GPU image. Everything else (3-D elastic, GPU,
│         batch, reciprocity) already works. ← target path
├─ NO, budget exists → apply for Salvus academic license (only turnkey
│         Python+GPU+free-surface+batch tool).
└─ NO, no budget → SpecFEM3D for a small reference set to validate a
          simplified Devito model, then Devito for the 1000-run campaign.
```

## Key links
- Devito: [github](https://github.com/devitocodes/devito) · [download/GPU](https://www.devitoproject.org/download.html) · [SLIM elastic examples](https://slimgroup.github.io/Devito-Examples/tutorials/06_elastic/) · [3-D elastic at scale (arXiv:2004.10519)](https://arxiv.org/abs/2004.10519)
- Free surface: [Kristek 2002 stress-image](https://library.seg.org/doi/abs/10.1190/1.1512752) · [DevitoPRO free-surface blog](https://www.devitocodes.com/blog/IM/) · [Schism (successor, DevitoPRO)](https://github.com/EdCaunt/schism)
- Validation: [Lamb analytic code](https://github.com/ktkimit/lamb_2dhalf_surface) · [SW4](https://github.com/geodynamics/sw4)
- Alternatives: [SpecFEM3D](https://github.com/SPECFEM/specfem3d) · [Salvus](https://www.mondaic.com/get-salvus) · [ABC methods in Devito (GMD 2022)](https://gmd.copernicus.org/articles/15/5857/2022/)
