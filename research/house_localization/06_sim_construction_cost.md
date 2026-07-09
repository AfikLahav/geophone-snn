> **Research survey — house-localization pivot ("wallhacks").** Background research agent (Sonnet), 2026-07-08. Literature/tooling survey; design input, not a validated experiment. Extends `research/seismic_simulators.md`. Indexed in [README.md](README.md).

---

# House + Soil Elastic-Wave Forward Simulator: Build Recipe and Cost Model

## What the Prior Survey Already Covered

`seismic_simulators.md` addressed open-field footstep/vehicle detection at metres-to-hundreds-of-metres range with a 1-D layered model (pyprop8), concluding propagation physics is not the binding constraint for sim-to-real transfer in that outdoor context.

The indoor building-localization problem differs in three ways: (1) the medium is a heterogeneous structure with two wildly different velocity regimes (concrete ~2000 m/s S-wave, soil ~150–400 m/s S-wave) at cm-to-m scale; (2) receivers are on/around the building, so structural modes matter; (3) the goal is a Green's-function library over a dense 2-D floor grid — exactly where reciprocity gives 60–100× cost reduction. This report starts where the survey ended.

---

## 1. Material Parameters

| Material | Vp (m/s) | Vs (m/s) | ρ (kg/m³) |
|---|---|---|---|
| Reinforced concrete | 3,500–4,000 | 1,800–2,200 | 2,200–2,500 |
| Brick masonry | 2,200–3,000 | 1,200–1,600 | 1,800–2,000 |
| Dense gravel/soft soil | 400–700 | 150–250 | 1,700–1,900 |
| Clay/silt (stiff) | 700–1,500 | 250–400 | 1,800–2,100 |

Identities: Vs = √(μ/ρ), Vp = √((κ+4μ/3)/ρ), ν ≈ 0.15–0.25 concrete, 0.33–0.45 saturated soil.

**The velocity contrast Vp_concrete / Vs_soil ≈ 3600 / 150 ≈ 24 governs the CFL "waste" factor in uniform-grid FDTD and is the central cost driver.**

---

## 2. Domain Sizing and Resolution

- Building 10×12×3 m (+3 m/floor); soil buffer 5 m/side; PML/dashpot 10×dx/face → total FDTD domain ~24×26×10 m.
- **Resolution rule (slowest medium, highest frequency):** dx = Vs_min / (f_max × PPW). 10 PPW min (body waves), 15–25 for surface-wave fidelity.

| Scenario | f_max | Vs_min | PPW | dx (m) | CFL dt (Vp_max=3600, 3D) |
|---|---|---|---|---|---|
| Low-cost | 100 Hz | 150 | 10 | 0.150 | 24 μs |
| High-fidelity | 250 Hz | 150 | 10 | 0.060 | 9.6 μs |

CFL 3D staggered: **dt ≤ dx / (Vp_max·√3)**. Note: dt is set by the *fastest* medium (concrete), dx by the *slowest* (soil) → 24× waste from the velocity contrast on a uniform grid.

---

## 3. Quantitative Compute + Storage Model

### Grid sizes
| Scenario | Domain | dx | Grid | Cells |
|---|---|---|---|---|
| 100 Hz | 24×26×10 | 0.150 | 160×173×67 | 1.85 M |
| 250 Hz | 24×26×10 | 0.060 | 400×433×167 | 28.9 M |
Ratio 15.6 = (250/100)³ ✓ (spatial scaling f³).

### DOF / memory (3D elastic staggered: 21 float32 arrays)
100 Hz → 155 MB; 250 Hz → 2.43 GB. Both fit an RTX 3090 (24 GB).

### Time steps (T_sim = 0.5 s)
100 Hz → 20,833; 250 Hz → 52,083.

### f_max⁴ scaling (cell-updates = cells × steps)
- 100 Hz: 38.5 B · 250 Hz: 1,505 B → ratio **39.1× = 2.5⁴** ✓. (Going 100→250 Hz is 39× more expensive, not 2.5×.)

### GPU throughput
Devito elastic order-8 ≈ 122 bytes/cell/step. RTX 3090 (936 GB/s, ~75% eff) → **~5.75 GP/s**.

| Scenario | Cell-updates | RTX 3090 |
|---|---|---|
| 100 Hz | 38.5 B | **6.7 s** |
| 250 Hz | 1,505 B | **261 s = 4.35 min** |

Cloud 8× A100 (2039 GB/s each, MPI ~75%): ~**75 GP/s** effective → 13× the RTX 3090.

### Reciprocity savings
N_sensors=8, N_floor_positions = 10×12 m @ 0.5 m = 480. Without reciprocity: 480 sims/house. With: **8 sims/house** (source at each sensor, record all 480 positions). **Savings 60×.**

### Full cost (with reciprocity, RTX 3090)
| Scenario | Per house | 50 houses | 500 houses | 500 on 8×A100 | AWS cost |
|---|---|---|---|---|---|
| 100 Hz | 54 s | 45 min | **7.5 hr** | 34 min | ~$18 |
| 250 Hz | 35 min | 29 hr | **12.4 days** | 22.8 hr | **~$748** |

### Green's-function library storage (N_sens × N_pos × N_time × N_comp × 4 B)
| Scenario | Samples | Per house | 500 houses |
|---|---|---|---|
| 100 Hz | 100 | 1.54 MB | 770 MB |
| 250 Hz | 250 | 3.84 MB | **1.9 GB** |
3-component ×3: 500 houses @ 250 Hz = **5.8 GB**. Trivially manageable.

---

## 4. Path A: FEniCSx / ElastodynamiCSx Shell-FEM

Walls/floors as 2D shell/plate elements (Reissner-Mindlin / Kirchhoff-Love); soil as 3D solid; shell-solid coupling via shared DOFs or Nitsche penalty. Physically natural because concrete walls (0.15–0.30 m) are thin vs seismic wavelength below 250 Hz.

Libraries: `dolfinx>=0.10`, `fenicsx-shells 0.10`, `elastodynamicsx 0.4 (MIT)`, `gmsh>=4.12`, `meshio>=5.3`, `build123d>=0.6`, `petsc4py`, `mpi4py`.

Pipeline: build123d parametric geometry → STEP; Gmsh OCC import + soil box + `fragment` for shared faces + physical groups + mesh; FEniCSx-Shells Reissner-Mindlin on wall submesh; ElastodynamiCSx isotropic soil + Lysmer-Kuhlemeyer dashpot; shell-to-solid coupling (shared node or Nitsche).

DOF counts: ~5.57M at 100 Hz (~18K shell + ~5.55M solid); ~35M at 250 Hz. Memory for explicit Newmark ≈ **50 GB at 250 Hz** — does not fit RTX 3090; needs CPU + 128 GB RAM (or matrix-free).

**Risks:** shell-to-solid coupling (no ready demo, 2–4 wks custom); transient shell dynamics (combine manually); elastic PML (adapt from acoustics, or use dashpot); DOLFINx GPU experimental. **Effort: 6–10 engineer-weeks to first simulation.**

---

## 5. Path B: Devito Voxel-FDTD

Voxel grid; per-cell (λ, μ, ρ); first-order velocity-stress (Virieux 1986) on a staggered grid; Devito generates optimized C/CUDA from a Python symbolic description. No mesh.

Libraries: `devito>=4.8.8`, numpy/scipy/sympy, vtk/h5py; for GPU: CUDA>=11.8, nvcc.

Build recipe:

```python
from devito import Grid, Function, VectorTimeFunction, TensorTimeFunction, Eq, div, grad, diag
import numpy as np
nx, ny, nz = 160, 173, 67; dx = 0.15; dt = 24e-6
grid = Grid(shape=(nx,ny,nz), extent=(nx*dx, ny*dx, nz*dx)); so = 8
lam = Function(name='lam', grid=grid, space_order=0)
mu  = Function(name='mu',  grid=grid, space_order=0)
rho = Function(name='rho', grid=grid, space_order=0)
# fill soil everywhere, then overwrite wall voxels with concrete
lam.data[:]=lam_soil; mu.data[:]=mu_soil; rho.data[:]=rho_soil
lam.data[wall_mask]=lam_concrete; mu.data[wall_mask]=mu_concrete; rho.data[wall_mask]=rho_concrete
# Virieux velocity-stress
v   = VectorTimeFunction(name='v',   grid=grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name='tau', grid=grid, space_order=so, time_order=1)
u_v   = Eq(v.forward,   v   + dt/rho * div(tau))
u_tau = Eq(tau.forward, tau + dt*(lam*diag(div(v.forward)) + mu*(grad(v.forward)+grad(v.forward).T)))
```

- **Source:** vertical point force (Ricker-derivative or empirical GRF) injected into τzz at the floor cell.
- **Absorbing:** damped sponge region (`damp` field ramping to d_max over ~10 cells); PML possible via complex-stretched coordinates (manual).
- **GPU:** `DEVITO_PLATFORM=nvidiaX DEVITO_LANGUAGE=openacc python run_sim.py`.
- **Receivers (reciprocity):** `SparseTimeFunction` recording vz at all 480 floor positions simultaneously.
- **Geometry → voxels:** build123d STEP → numpy mask (analytical axis-aligned boxes, or pyvista/skimage voxelizer) → material arrays.

**Risks:** staircased diagonal walls (axis-aligned exact; fine dx); 24× CFL waste from velocity contrast; sponge (adequate, not geophysics-grade); Devito CUDA production-ready. **Effort: 1–2 engineer-weeks to first simulation.**

---

## 6. Shell/Plate FEM: Validity vs Frequency

For a plate thickness t, S-wave speed Vs: Kirchhoff-Love valid t<λ/20 (f < Vs/(20t)); Reissner-Mindlin valid t<λ/5–10; through-thickness resonance f_res = Vs/(2t).

Concrete wall (t=0.20 m, Vs=2000): K-L valid <500 Hz, R-M <1000 Hz, f_res=5000 Hz — **shell theory valid up to ~500–1000 Hz, comfortably above 250 Hz.** Floor slab (t=0.15 m): K-L <667 Hz. **Shell elements are physically appropriate for this band.**

Shell-to-solid coupling: (A) shared-node (Gmsh OCC `fragment` → conformal mesh, displacement continuity automatic); (B) Lysmer-Kuhlemeyer dashpot at soil boundary (σn=−ρ·Vp·vn, σt=−ρ·Vs·vt — ~100% for body waves >30° incidence, 60–80% for Rayleigh/near-vertical; fine for training data).

---

## 7. Absorbing Boundaries

| Method | Absorption | Complexity | Wall reflection |
|---|---|---|---|
| Sponge (Devito default) | Fair (~10% with 10-cell layer) | Trivial | ~10% |
| Lysmer-Kuhlemeyer dashpot | Good >30° incidence; poor Rayleigh | Moderate | 5–20% |
| CPML / PML | Excellent all angles | Complex | <0.1% |
| LC-PML in FEniCSx | Excellent | Available (acoustics; adapt for elastic) | <0.1% tuned |

Use sponge/dashpot for dataset generation (tolerates ~10% artifacts); PML for validation runs.

---

## 8. Reciprocity: Green's Function Library

Elastodynamic reciprocity (Knopoff-Gangi 1959; Wapenaar 2004): G_ij(x_A,x_B;t)=G_ji(x_B,x_A;t). Instead of 480 source positions, simulate 8 runs with sensors as sources and record at all floor positions.

**Correctness conditions:** (1) linear elasticity (strain≪10⁻⁵, holds >0.5 m from foot); (2) reciprocal medium (isotropic soil/concrete OK); (3) identical BCs in forward and reciprocal; (4) same source character (vertical force at sensor → record vz at floor); (5) no near-source nonlinearity/contact (model near-field as linear, OK for ranges >1 m).

**Pitfalls:** component mismatch (for vertical sensor + vertical force, one reciprocal shot/sensor suffices); moving walker = a sequence of static positions synthesized in post-processing by convolving the GRF time history with the appropriate Green's-function slice; background medium must be static (build separate GF libraries per parameter set, or domain-randomize).

---

## 9. Footstep Source Injection

GRF: sequence of vertical transient pulses at ~1–2 Hz. Each pulse — heel-strike peak ~1.1–1.5×BW (rise 50–100 ms, energy 1–30 Hz), mid-stance valley ~0.8×BW, toe-off ~1.0–1.2×BW. Full content to 250 Hz from hard heel-strike transients.

```python
def ricker_deriv(t, f0=20, t0=0.05):
    u = np.pi*f0*(t-t0); return -2*u*np.exp(-u**2)
# moving walker synthesis from GF library:
seismogram_k = np.zeros(N_time)
for (xi, yi, ti) in footfalls:
    gf = GF_library[k, position_index(xi, yi), :]
    seismogram_k += np.convolve(grf_wavelet, gf)[ti:ti+N_time]
```
Unlimited training samples from a small GF library; each (trajectory, weight, step rate, footwear) is one sample synthesized in microseconds.

---

## 10. Path Comparison and Recommendation

| Criterion | A: FEniCSx Shell-FEM | B: Devito FDTD |
|---|---|---|
| Effort to first sim | 6–10 weeks | **1–2 weeks** |
| Concrete representation | Physically correct 2D shell | Voxelized (staircase; wastes resolution in fast concrete) |
| CFL waste | Avoided for shell DOFs | Full 24× |
| GPU | Experimental (DOLFINx 0.10) | **Production CUDA** |
| Reciprocity | Yes | Yes |
| Memory @ 250 Hz | 50 GB (CPU RAM) | **2.4 GB (RTX 3090)** |
| Absorbing BC | Good (dashpot/PML) | Fair (sponge) |
| Shell-solid coupling | Weak spot (2–4 wk) | N/A |
| Correctness-bug risk | High | **Low** |

**Recommendation: start with Path B (Devito FDTD)** — faster, GPU-native, physically correct for this band; axis-aligned walls voxelize with negligible staircase error; the 24× CFL waste is acceptable (500 houses @ 100 Hz = 7.5 hr on one RTX 3090; @ 250 Hz use 8×A100 for ~$748). **Use Path A (shell-FEM) as a validation tool** — 5–10 reference sims to confirm the voxelized concrete behaves correctly at the sensors (1–2 wk).

---

## 11. Summary Numbers Table

| Parameter | 100 Hz | 250 Hz |
|---|---|---|
| dx (m) | 0.150 | 0.060 |
| dt (μs) | 24 | 9.6 |
| Grid cells | 1.85 M | 28.9 M |
| FDTD memory (f32) | 155 MB | 2.43 GB |
| Time steps (0.5 s) | 20,833 | 52,083 |
| Cell-updates/sim | 38.5 B | 1,505 B |
| f_max scaling | — | 39.1× (2.5⁴) |
| RTX 3090 throughput | 5.75 GP/s | 5.75 GP/s |
| Wall-clock/sim (3090) | **6.7 s** | **261 s** |
| Wall-clock/house (N=8, recip) | **54 s** | **35 min** |
| 50 houses (3090) | 45 min | 29 hr |
| 500 houses (3090) | **7.5 hr** | **12.4 days** |
| 8×A100 speedup | 13× | 13× |
| 500 houses (8×A100) | 34 min | 22.8 hr |
| AWS p4d.24xlarge, 500 houses | ~$18 | **~$748** |
| GF library, 500 houses (1-comp) | 770 MB | **1.9 GB** |

---

## 12. Key References

- [Devito elastic tutorial (Virieux staggered grid)](https://www.devitoproject.org/examples/seismic/tutorials/06_elastic.html) · [Devito framework (SLIM 2020)](https://slim.gatech.edu/Publications/Public/TechReport/2020/louboutin2020SCsta/louboutin2020SCsta.html) · [DevitoPRO elastic benchmark](https://www.devitocodes.com/elastic)
- [FEniCSx-Shells (DOLFINx 0.10)](https://fenics-shells.github.io/fenicsx-shells/) · [ElastodynamiCSx (MIT)](https://github.com/Universite-Gustave-Eiffel/elastodynamicsx) · [FEniCSx Newmark elastodynamics](https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html)
- [build123d](https://build123d.readthedocs.io/en/latest/) · [LC-PML in FEniCSx (Undabit 2025)](https://undabit.com/locally-conformal-perfectly-matched-layer-implementation-in-fenicsx) · [Lysmer-Kuhlemeyer dashpot (SCIRP)](https://www.scirp.org/html/1-8102254_49496.htm) · [PPW for surface waves (GJI 2010)](https://academic.oup.com/gji/article/178/1/282/644205)
- [Elastodynamic reciprocity (Wapenaar, PRL 93, 2004)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.93.254301) · [Reciprocity & Green's functions (GeoScienceWorld)](https://pubs.geoscienceworld.org/books/book/2035/chapter/107744154/)
- [Footstep GRF seismic signature (JASA 2005)](https://ui.adsabs.harvard.edu/abs/2005ASAJ..118.2021S/abstract) · [GRF on vibrating surfaces (Eng Struct 2018)](https://www.sciencedirect.com/science/article/abs/pii/S0141029618305455) · [GRF biomechanics (arXiv:2503.16455)](https://arxiv.org/pdf/2503.16455)
- [Seismic material velocities (UNLV)](https://pburnley.faculty.unlv.edu/GEOL452_652/seismology/notes/SeismicNotes10RVel.html) · [GPG geosci.xyz](https://gpg.geosci.xyz/content/physical_properties/seismic_velocity_duplicate.html) · [Concrete Vs (Wiley 2017)](https://onlinelibrary.wiley.com/doi/10.1155/2017/1651753) · [Shell theory validity (arXiv:1504.01267)](https://arxiv.org/pdf/1504.01267)
