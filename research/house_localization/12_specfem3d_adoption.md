> **SPECFEM3D adoption analysis — house-localization pivot.** Research (Sonnet, 2026-07-09).
> Builds on `engine_decision/specfem3d.md` (steelman), `06`, `07`, `08`, `10`, and the pyprop8
> source in `simgeo_v42/gfbank_build.py` + `scenes.py`. Thorough inline research; no sub-agents.
> Indexed in [README.md](README.md).

---

# SPECFEM3D Adoption Analysis: Head-to-Head, Work-List, Compute, and Risks

---

## 0. What This Document Adds

`engine_decision/specfem3d.md` is an accurate steelman. This document adds four things it
did not cover:

1. A rigorous **pyprop8 vs SPECFEM3D head-to-head** — where they overlap, where they
   diverge, and how to use the overlap as a bridge validation.
2. A precise **feature-by-feature readiness map** ("ready / config / build") from the
   perspective of our pipeline, not a generic feature list.
3. A **concrete adoption work-list** with effort estimates tied to our actual pipeline
   (scenes.py + sensor-chain + reciprocity architecture).
4. **Building-scale risks** that are invisible at the regional/global scale SPECFEM3D was
   designed for — and which the steelman document under-weighted.

---

## 1. pyprop8 vs SPECFEM3D — Head-to-Head

### 1a. What pyprop8 actually does (from the source)

`gfbank_build.py` and `scenes.py` reveal the complete pyprop8 pipeline:

- **Physics**: Thompson-Haskell propagator matrix method (O'Toole & Woodhouse 2011/2012),
  1-D horizontally layered half-space, arbitrary Vp/Vs/rho/Q per layer. Semi-analytic:
  wavenumber integration over a discrete-wavenumber sum (Bouchon & Aki 1977 / Bouchon 2003),
  not time-stepping.
- **Sources**: vertical point force (Fz) + horizontal force (Fx) simultaneously in one call,
  sharing propagator matrices — the code comment explicitly states "shares the propagator
  matrices" and verifies bit-identical results vs separate calls.
- **Receivers**: `ListOfReceivers` at arbitrary radii on the surface (depth=0), z-component
  only in the current usage (`a[0, :, 2, :]` = z-component of Fz response).
- **Output**: step-force displacement in km per 1e15 N, stored in `.npz` banks indexed by
  radius; `scenes.py` converts to impulse-velocity via time differencing + (iω) chain +
  causal t* attenuation (Azimi) + optional plate resonance filter.
- **Geometry**: no structure. The "house" does not appear in the simulation at all. The
  only geometry is the 1-D layer stack (soil profile) and the source-receiver radius. The
  building is implicitly modelled as the soil profile seen by the exterior geophone —
  specifically, the `plate` parameter (SDOF resonance filter applied in the frequency domain)
  captures the coupling between footstep force and soil at the surface, but this is a
  phenomenological filter, not a simulation of the concrete structure.
- **Domain randomization**: `profiles.py` builds a library of 1-D soil profiles with
  varying Vs, Q, and layer structure; each profile gets a separate bank. The parametric
  diversity comes from soil variation, not from 3-D building geometry.
- **Scale**: radii 0.5 m to ~320 m, geomspaced with 36-48 points, computed at 250 Hz
  Nyquist (dt=2ms), upsampled to 1000 Hz by zero-padding.

**The physical model pyprop8 encodes**: a homogeneous or layered elastic half-space with
a free horizontal surface, excited by a buried or surface point force, received at the
surface. No building. No concrete-soil interface. No structural modes.

### 1b. What SPECFEM3D does that pyprop8 cannot

| Property | pyprop8 | SPECFEM3D |
|---|---|---|
| Geometry | 1-D horizontal layers only | Full 3-D arbitrary geometry |
| Building structure | Not present (absorbed into soil profile) | Explicit concrete shell: walls, floor, slab |
| Concrete-soil interface | Absent | Fully resolved: impedance contrast, reflection/transmission |
| Structural modes of the building | Absent | Present: the building has its own resonant modes (breathing modes, floor modes) |
| Rayleigh wave along building face | Absent (no building surface) | Present: traction-free building walls guide interface waves |
| 3-D geometric spreading | 1/√r × azimuthal factor (analytic) | Full 3-D divergence, no approximation |
| Reciprocity | Applied analytically in `scenes.py` at render time | Run as separate forward simulations (or analytically post-processed) |
| Attenuation (Q) | causal t* (Azimi formula) in frequency domain | Constant-Q memory-variable (Liu-Anderson-Kanamori 1976 standard linear solids) — frequency-dependent in time domain |
| Speed | Milliseconds per scene (convolution at render) | Minutes per shot |
| Cost at 500 houses | Essentially zero (convolution from cached banks) | Hours on A100 cluster |

### 1c. Where they agree (the bridge validation)

**The most important overlapping case**: open-field (no building), homogeneous or layered
soil half-space, surface receivers, vertical point force source. In this limit:

- pyprop8 is semi-analytic and exact to numerical precision of the wavenumber integral.
- SPECFEM3D solves the same problem numerically with SEM.

This is the **bridge validation**: run both on an identical layered soil model (no house)
with the same source and receiver geometry. Expected agreement:

- P-arrival times: < 0.1% (body wave, dominated by Vp, no dispersion).
- Rayleigh phase velocity: < 0.3% for SEM at 8 PPW (literature: Komatitsch & Tromp 1999).
- Rayleigh amplitude: < 1% for SEM at 10 PPW (vs pyprop8 analytic = reference).

This cross-check serves two purposes simultaneously:
1. Validates the SPECFEM3D setup (mesh, material assignment, absorbing BCs, source
   injection) before any building is added.
2. Provides a "soil-only" anchor point in the Green's function library — exterior sensor
   readings at large offsets (> 5 m from the building) should approach the 1-D
   halfspace response, which pyprop8 already knows exactly.

**Practical bridge procedure**:

```
Step 1: Choose a soil profile from profiles.py (e.g. profile "s01": Vs=200, Vp=400, rho=1800, Q=30).
Step 2: Run pyprop8 on this profile, vertical force at surface, receivers at r=2,5,10,20 m.
         Output: z-component displacement time series.
Step 3: Build a SPECFEM3D homogeneous halfspace mesh (use the shipped example
         EXAMPLES/homogeneous_halfspace_HEX27_elastic_no_absorbing/).
         Assign the same Vp/Vs/rho (no Q for this first check; add later).
         Run with FORCESOLUTION vertical, same receiver offsets.
Step 4: Compare: P-arrival times (< 0.5%), Rayleigh arrival time (< 0.5%),
         peak vz amplitude ratio (< 5%). If all pass, SPECFEM3D validated on soil.
Step 5: Add the concrete shell mesh. Now SPECFEM3D diverges from pyprop8
         (building effects), but pyprop8 gives the "no-building" baseline.
```

The bridge validation costs ~0.5 engineering days (step 3 uses the shipped example;
step 4 is a Python comparison script). It simultaneously cross-validates pyprop8's
soil physics against a fully independent code.

### 1d. Where they physically diverge and by how much

At building scale the open-field 1-D model (pyprop8) diverges from the 3-D reality
(SPECFEM3D) in at least four ways:

**Divergence 1 — Building scattering / impedance contrast.**
A reinforced concrete building (Vp≈3600, Vs≈2000, rho≈2400) in soil (Vs≈150-300,
rho≈1800) has a velocity contrast of ~10:1 in S-wave speed. This contrast reflects and
refracts incoming waves, producing shadow zones and amplification zones around the
building that do not exist in a 1-D halfspace. The scattered field is roughly
proportional to the velocity contrast squared → ~100× stronger scattered energy than
in a typical 2:1 layer contrast. This is **not captured by pyprop8** in any soil-profile
parametrisation: no choice of 1-D layers reproduces the 3-D shadow zone created by a
concrete box.

**Divergence 2 — Structural modes.**
The building has its own natural frequencies: floor slab bending modes (f_floor ≈ Vs_concrete
/ (2L_floor) for a simply-supported slab, roughly 1000–3000 Hz for a 0.15 m slab), wall
breathing modes (~500–800 Hz), and rigid-body rocking (~2–10 Hz). In the 10–250 Hz band
(our operating range), only the rigid-body and first-order wall modes are relevant. These
modes are **absent in pyprop8**, which models the building implicitly as a simple SDOF
resonance filter on the source force (the `plate` parameter). The SDOF filter is a
phenomenological approximation that was validated for a pavement plate in an outdoor
context; it is not a physics simulation of a house.

**Divergence 3 — Coupling path to exterior sensor.**
In the outdoor geophone use case (the original simgeo application), the energy path is:
source → soil → geophone (all in the same medium). In the house-localization application,
the path is: source (floor) → building structure → coupling zone at foundation → soil →
exterior geophone. The building-to-soil coupling at the foundation perimeter is a
radiation problem that depends on the foundation geometry, embedment depth, and
impedance contrast — none of which are in pyprop8.

**Divergence 4 — Near-field 3-D spreading.**
pyprop8 uses exact 3-D spreading for the halfspace (encoded in the wavenumber integral),
but this spreading law assumes no boundaries except the flat free surface. Near a building
corner (< 1 m), the 3-D cylindrical/spherical spreading law does not apply. SPECFEM3D
handles this naturally.

**Practical consequence for the localization dictionary**: a Green's function library built
from pyprop8 (treating the building as a soil profile) will be wrong for localization
inside the building, where the structural coupling path dominates. It will be approximately
correct only for positions where the source is far from the building walls (> 5 m) AND the
building effect on the exterior sensor is weak (sensor is far from the building and building
scattering is small). For the "wallhacks" indoor localization application, that limit is
never satisfied — the source is *inside* the building, always less than 5 m from at least
one wall.

---

## 2. SPECFEM3D Feature Readiness Map

For our pipeline: parametric boxy house → Gmsh hex mesh → material assignment (Vp/Vs/rho/Q
per block) → SPECFEM3D run → extract 3-component seismograms → sensor-chain + reciprocity.

| Feature | Status | Notes |
|---|---|---|
| **3-D elastic wave solver** | **READY** | SPECFEM3D Cartesian v4.1.0; full anisotropic + isotropic elastic; CUDA/HIP GPU. |
| **Free surface (soil top)** | **READY** | Natural SEM Neumann BC; no special coding; validated in shipped examples and 25 years of literature. |
| **Free surface (building walls facing air)** | **CONFIG** | Interior air rooms: do not mesh interior air; leave inner wall faces as unloaded element faces (free BC). This is physically correct for f < 500 Hz (air coupling negligible). Set in mesh design, zero code. |
| **Vertical point-force source** | **READY** | `FORCESOLUTION` file with Z-component force direction vector. Edit by Python script. |
| **3-component receiver output** | **READY** | All three Cartesian components recorded simultaneously. Choose SAC-binary or ASCII via Par_file. |
| **Multiple receivers in one run** | **READY** | All 480 floor positions in one `STATIONS` file; solver cost is constant in N_stations. |
| **Reciprocity (shoot from sensor, record at floor)** | **CONFIG** | Place sensors in FORCESOLUTION, all floor points in STATIONS. Requires generating STATIONS file with 480 entries (Python, trivial). |
| **Attenuation (Q)** | **CONFIG** | Enable `ATTENUATION = .true.` in Par_file; set Q per material in `nummaterial_velocity_file`. Standard linear solid (Liu et al. 1976). |
| **Heterogeneous materials (concrete + soil)** | **CONFIG** | Each Gmsh physical volume gets a material index; `nummaterial_velocity_file` assigns Vp/Vs/rho/Q. |
| **Absorbing boundaries (Stacey)** | **READY** | Default; first-order Clayton-Engquist; fast; some grazing-angle leakage (~5–10%). |
| **Absorbing boundaries (C-PML)** | **CONFIG** | Enable `PML_CONDITIONS = .true.`; set PML thickness to ~10 elements. Better than Stacey for near-grazing incidence; carries a known late-time instability risk at sharp velocity contrasts (concrete/soil). Recommend Stacey for data-generation runs; PML for validation. |
| **GPU (CUDA)** | **READY** | CUDA supported since v3.0; validated on P100/V100/A100/H100. |
| **GPU (HIP/AMD)** | **READY** | Added in v4.0; initial production status. |
| **NUMBER_OF_SIMULTANEOUS_RUNS** | **READY** | Runs multiple earthquake sources across MPI ranks sharing one mesh read. Directly maps to our 8 reciprocal shots: set NUMBER_OF_SIMULTANEOUS_RUNS=8, one run0001..run0008 directory per sensor shot. Net: single mesh decomposition, 8 shots batched. |
| **Gmsh hex mesh import** | **BUILD** | No native Gmsh→SPECFEM3D converter for 3D. Must write a Python conversion script to produce the 10 required files (nodes, connectivity, material flags, boundary surface tags). ~100–200 lines one-time. See section 3 for details. |
| **Parametric mesh automation** | **BUILD** | The Python Gmsh API (`gmsh.model`, `gmsh.mesh`) allows fully scripted geometry. One-time script `build_house_mesh(params)` ~200 lines. |
| **HDF5 GF library extraction** | **BUILD** | SAC or ASCII output → ObsPy or direct NumPy parse → HDF5. ~100 lines. Mirrors the `.npz` bank in `gfbank_build.py`. |
| **Sensor-chain convolution (scenes.py compatibility)** | **BUILD** | SPECFEM3D outputs raw 3-component velocity seismograms. The `scenes.py` Bank.emit() pipeline expects pre-convolved impulse-velocity spectra from pyprop8 banks. Need a SPECFEM3D-compatible Bank class that reads from HDF5 and provides the same interface. ~100 lines. |
| **Docker / container build** | **BUILD** | No official maintained Docker image from SPECFEM team. Must build from Fortran source (gfortran/Intel ifx + CUDA). One-time ~0.5–1 day. |

**Count**: Ready: 9 | Config (edit text files or Par_file): 6 | Build (new code): 5

---

## 3. The Adoption Work-List

This is the concrete pipeline we would build, ordered by execution sequence.

### Stage 0: Environment (0.5–1 day)

- Install SPECFEM3D from source: `cmake -DWITH_CUDA=ON -DCUDA_VERSION=12 ..` + make.
- Verify CUDA GPU path: run shipped example `EXAMPLES/homogeneous_halfspace_HEX27_elastic_no_absorbing/` on GPU.
- Write a Dockerfile (optional but recommended for reproducibility).
- **Effort: 0.5–1 day.** Risk: CUDA version mismatch with local driver; mitigated by using a cloud instance with a clean CUDA 12 image.

### Stage 1: Bridge validation — soil-only (0.5 day)

Run the shipped halfspace example with soil parameters from `profiles.py`, compare z-component
surface displacement to pyprop8 output at matched radii.

- **Pass criterion**: P-arrival < 0.5%, Rayleigh arrival < 0.5%, peak amplitude < 5%.
- **Why first**: validates the entire SPECFEM3D pipeline (build, source injection, receiver
  recording, output parsing) before any building geometry complexity is added. If this
  fails, all subsequent work is suspect.
- **Effort: 0.5 day** (shipped example; Python comparison script using existing `gfbank_build.py`
  output as the reference).

### Stage 2: Gmsh mesh script for a single boxy house (~3–5 days)

Write `mesh/build_house_mesh.py` using the Gmsh Python API. The function signature:

```python
def build_house_mesh(
    length: float,      # exterior floor-plan length (m)
    width: float,       # exterior floor-plan width (m)
    height: float,      # wall height per floor (m)
    n_floors: int,      # number of floors
    wall_t: float,      # wall thickness (m)
    floor_t: float,     # floor/slab thickness (m)
    soil_buffer: float, # soil buffer on each side (m)
    soil_depth: float,  # soil depth below foundation (m)
    lc_soil: float,     # Gmsh mesh size for soil elements (m)
    lc_concrete: float, # Gmsh mesh size for concrete elements (m)
    output_dir: str,    # where to write the 10 SPECFEM3D mesh files
) -> None
```

**Gmsh geometry** (Python API calls):
1. `gmsh.model.occ.addBox(...)` for the outer soil domain.
2. `gmsh.model.occ.addBox(...)` for the exterior building volume.
3. `gmsh.model.occ.addBox(...)` for the interior air void (building hollow).
4. `gmsh.model.occ.cut(building_vol, interior_void)` → concrete shell.
5. `gmsh.model.occ.fragment(soil_vol, concrete_shell)` → conforming shared faces at the
   concrete-soil interface (critical: creates shared DOFs at the interface so the impedance
   BC is automatic).
6. Assign physical groups: `soil_volume`, `concrete_volume`, `free_surface` (top soil faces),
   `abs_xmin`, `abs_xmax`, `abs_ymin`, `abs_ymax`, `abs_zmin` (absorbing boundaries).
7. `gmsh.option.setNumber('Mesh.RecombineAll', 1)` to force hexahedral elements (transfinite
   or Delaunay-recombine). For simple boxes, `gmsh.model.mesh.setTransfiniteVolume()` on
   each sub-volume is the cleanest path.
8. `gmsh.model.mesh.generate(3)` → generate 3D hex mesh.

**Conversion to SPECFEM3D format** (~100–150 lines of Python):
SPECFEM3D Cartesian requires 10 files:
- `nodes_coords_file`: node x,y,z coordinates.
- `mesh_file`: element connectivity (8 nodes per HEX8 element).
- `materials_file`: element → material index.
- `nummaterial_velocity_file`: material index → Vp,Vs,rho,Q,flag.
- `absorbing_surface_file_{xmin,xmax,ymin,ymax,zmin}` (5 files): face node lists for each
  absorbing boundary side.
- `free_surface_file`: face node lists for the free surface.

Gmsh physical groups provide the element-to-material mapping and boundary face lists. The
conversion reads Gmsh's mesh via `gmsh.model.mesh.getNodes()` / `getElements()` APIs and
writes the above files. This conversion is the main one-time engineering effort.

**No official Gmsh→SPECFEM3D 3D converter exists in the SPECFEM repo** (only a 2D one
in `utils/Gmsh/LibGmsh2Specfem.py`). The 3D conversion is user-written; the ResearchGate
thread confirms this has been done by users. The format is well documented in the SPECFEM3D
manual (Chapter 3).

**Effort: 3–5 engineering days** for writing + debugging the mesh script and converter.
Risk: hex mesh quality (Jacobian < 0 elements) on the concrete-soil interface corners.
Mitigation: use `gmsh.model.mesh.setTransfiniteVolume` on each box volume separately;
check with `xcheck_mesh_quality` (SPECFEM3D utility).

### Stage 3: Material assignment and first house sim (1–2 days)

- Edit `nummaterial_velocity_file`:
  ```
  1  elastic  Vp_soil    Vs_soil    rho_soil    Q_soil    0
  2  elastic  Vp_concrete Vs_concrete rho_concrete Q_concrete 0
  ```
- Write `FORCESOLUTION` (sensor location 1 of 8) and `STATIONS` (480 floor grid points) via Python.
- Run `xdecompose_mesh` (SCOTCH MPI partitioner) → `DATABASES_MPI/`.
- Run `xspecfem3D` on GPU → `OUTPUT_FILES/*.semd` (velocity seismograms).
- Parse output: `ObsPy.read()` or direct NumPy parse of ASCII columns.

**Effort: 1–2 days.** First sim confirms the full pipeline end-to-end.

### Stage 4: Reciprocity batch loop (2–3 days)

Write `run_campaign.py`:

```python
import subprocess, json, h5py, numpy as np, os

def run_house(house_params, sensor_coords, floor_coords, out_dir):
    mesh_dir = build_house_mesh(**house_params)        # Stage 2 script
    subprocess.run(['xdecompose_mesh', '...'])         # SCOTCH
    gf = np.zeros((8, 480, 3, N_time), dtype=np.float32)
    for k, sensor_xyz in enumerate(sensor_coords):
        write_forcesolution(sensor_xyz, mesh_dir)      # vertical force at sensor k
        write_stations(floor_coords, mesh_dir)         # all 480 floor positions
        subprocess.run(['mpirun', '-np', '8', 'xspecfem3D', ...])
        gf[k] = parse_seismograms(mesh_dir)            # (480, 3, N_time)
    with h5py.File(os.path.join(out_dir, f'{house_id}.h5'), 'w') as f:
        f.create_dataset('gf', data=gf, compression='gzip', compression_opts=4)
        f.attrs['params'] = json.dumps(house_params)
```

Alternatively, use `NUMBER_OF_SIMULTANEOUS_RUNS=8` to run all 8 sensor shots in one
MPI invocation, sharing the mesh read and decomposition. This halves I/O and likely
reduces wall-clock by ~30% vs 8 sequential runs.

**Effort: 2–3 days.** Includes job-scheduler integration (Slurm array or cloud batch).

### Stage 5: Sensor-chain compatibility layer (1–2 days)

Write `SpecFEMBank` class that mirrors `scenes.py`'s `Bank` interface:

```python
class SpecFEMBank:
    """Wraps an HDF5 GF file from SPECFEM3D. Provides .emit() compatible with Bank.emit()."""
    def __init__(self, h5_path: str): ...
    def emit(self, floor_pos_xy, sensor_k, fz_wavelet) -> np.ndarray:
        """Returns v_z(t) at sensor k due to footstep force fz at floor_pos_xy."""
        gf = self.data[sensor_k, floor_pos_idx, 2, :]  # 3rd comp = z
        return np.convolve(gf, fz_wavelet, mode='full')[:N_time]
```

The existing `generate_corpus*.py` scripts then call `SpecFEMBank.emit()` in place of
`Bank.emit()`, with no changes to the training or SNN stack.

**Effort: 1–2 days.**

### Stage 6: Scale to house campaign (1–2 days setup + compute)

- Define parametric house grid (floor plan L×W, n_floors, wall_t, soil_vs, soil_Q).
- For each house, run Stages 3–4. If geometry is fixed (same footprint, varying materials
  only), re-use the mesh from Stage 2; mesh decomposition runs once; only `nummaterial_velocity_file`
  changes between house variants. Per-house overhead: 0 meshing + solver time.
- If floor plans vary: full Stage 2 per house (~2–5 CPU-min meshing + ~3–5 min solver at
  100 Hz on A100 × 8 shots = ~30–45 min total per house).
- **Recommended**: fix footprint to one representative house; vary wall_t, soil_vs, soil_Q, floor_t
  as parametric materials. This reduces house campaign to pure solver time (no re-meshing).

**Effort: 1–2 days setup; compute per section 4.**

### Summary effort table

| Stage | Task | One-time effort | Per-house cost |
|---|---|---|---|
| 0 | Environment + GPU build | 0.5–1 day | 0 |
| 1 | Bridge validation (soil-only, pyprop8 cross-check) | 0.5 day | 0 |
| 2 | Gmsh mesh script + SPECFEM3D converter | 3–5 days | 0 (or ~2–5 CPU-min if geometry varies) |
| 3 | Material files + first full sim | 1–2 days | ~2–5 min (materials-only) or 0 (same mesh) |
| 4 | Reciprocity batch loop | 2–3 days | ~25–45 min (100 Hz, A100, 8 shots) |
| 5 | SpecFEMBank sensor-chain adapter | 1–2 days | 0 |
| 6 | Campaign + HDF5 GF library | 1–2 days setup | compute per section 4 |
| **Total** | | **~9–15 engineering days** | varies |

This is ~2–3 weeks to a fully operational pipeline, vs ~1–1.5 weeks for Devito (with
free-surface implementation overhead). The extra week is mesh automation; the free-surface
burden is zero.

---

## 4. Compute: SEM Efficiency, GPU Maturity, Per-Run Cost

### 4a. SEM efficiency vs FDTD

SEM with GLL nodes at polynomial order N=4 (SPECFEM3D default) requires ~5 PPW for the
same dispersion error that FDTD needs 10–15 PPW (Komatitsch & Tromp 1999; Tromp et al.
2008). The key advantage for the house problem is **element size adapts to local wave speed**:

- Soil elements: dx_soil = Vs_soil / (f_max × PPW) = 150 / (100 × 5) = **0.30 m**
- Concrete elements: dx_concrete = Vs_concrete / (f_max × PPW) = 2000 / (100 × 5) = **4.0 m**

Wait — concrete elements at 4 m are far too coarse for a 0.15 m wall. The PPW constraint
is on the **slowest wave in each region**, but the geometric constraint (wall thickness)
overrides it. A 0.15 m wall needs at least 3 elements → dx_concrete ≤ 0.05 m in wall
regions. This is a geometry-driven constraint that overrides the PPW limit and is the key
compute penalty at building scale (detailed in section 5).

In practice: for a 100 Hz simulation, the mesh uses dx_soil ≈ 0.14–0.30 m in the soil
bulk (SEM needs fewer cells than FDTD there), but dx_concrete must be ≤ 0.05 m in wall
regions. The wall mesh is much finer than needed for wave propagation but is necessary for
geometric resolution. This partially offsets SEM's DOF advantage.

**Net DOF advantage vs Devito FDTD at 100 Hz**: soil bulk is ~3× fewer DOFs (SEM 5 PPW
vs FDTD 10 PPW); concrete walls are ~3× more DOFs per unit thickness (SEM 0.05 m vs
FDTD's wall averaging at 0.15 m). Across the whole domain (mostly soil), SEM is still
lighter in total DOFs, but the margin is smaller than the steelman suggested.

**Timestep advantage**: SEM's dt is set per-element by the local CFL:
dt_soil = C_CFL × dx_soil / Vs_soil ≈ 0.30 / 150 ≈ **2 ms** (vs FDTD's dt = 24 μs from
concrete CFL). This is an **80× timestep advantage** in the soil-only elements. The
concrete elements at dx=0.05 m: dt_concrete = 0.05 / (3600 × √3) ≈ **8 μs** — similar
to FDTD's concrete-driven dt. However, SEM allows a non-uniform mesh: the solver timestep
is the minimum over all elements, which is set by the concrete wall elements (dt ≈ 8–10 μs).
So the timestep advantage is actually only ~2–3× vs Devito's dt=24 μs, not 80×.

**Revised compute estimate**: SEM vs Devito at 100 Hz — similar total cell-updates (soil
savings offset by wall refinement); ~2–3× larger timestep (8–10 μs SEM vs 24 μs FDTD).
Net: SEM is ~2–3× faster per sim than Devito at the same accuracy, not 4× as the steelman
estimated. Both estimates carry ±50% uncertainty without a concrete benchmark.

### 4b. OpenBenchmarking.org data point

The SPECFEM3D 4.1.1 benchmark (Homogeneous Halfspace example, unspecified hardware) shows
an average run-time of ~4 minutes. That example uses a domain of approximately 134 km × 134 km
× 60 km (regional seismology scale) at 1 Hz. Our domain is 24 m × 26 m × 10 m at 100 Hz.
Scaling: frequency ratio 100:1 → grid cells scale as f³ = 10^6; but our domain is
(24×26×10)/(134,000×134,000×60,000) = 7.4×10⁻¹² of the volume. The two effects nearly
cancel for a constant-f/wavelength ratio. The 4-minute benchmark is on unspecified hardware
and is not directly informative for our case. NVIDIA reports 38× GPU speedup (P100 vs
unspecified CPU), which anchors the GPU efficiency.

### 4c. Revised per-house wall-clock estimates

Using: 8 reciprocal shots, A100 GPU (estimated ~38 × (2039/732) × P100 ≈ 105× vs the
NVIDIA-cited CPU; or equivalently, approximately similar to or 2× faster than Devito
OpenACC on A100 per sim). Based on Devito's validated 7.7 s/sim at 100 Hz on RTX 3090
(from doc 10), and SPECFEM3D's ~2–3× fewer effective timesteps:

| Scenario | Per sim (A100) | Per house (8 shots) | 50 houses | 200 houses |
|---|---|---|---|---|
| 100 Hz, single A100 | ~5–15 s | ~40–120 s | ~0.5–1.6 hr | ~2–6.5 hr |
| 100 Hz, 8-shot simultaneous (NUMBER_OF_SIMULTANEOUS_RUNS=8) | — | ~30–80 s | ~25–70 min | ~1.6–4.5 hr |
| 250 Hz (if needed), single A100 | ~3–10 min | ~25–80 min | ~21–65 hr | — |

These estimates are rough (±50%) because no published SPECFEM3D benchmark exists for a
house-scale elastic domain with concrete-wall resolution. The simultaneous-runs option
is the recommended path: share one mesh read, run all 8 sensor shots in parallel across
8 GPU partitions.

### 4d. Cloud cost

At AWS p3.2xlarge (1× V100, ~$3.06/hr spot) or p4d instance (8× A100):
- 50 houses at 100 Hz (single A100, ~1 hr) ≈ **$3–5**.
- 200 houses at 100 Hz (~6 hr) ≈ **$18–25**.
- 200 houses at 250 Hz (~65 hr on single A100; ~8 hr on 8×A100) ≈ **$25–200** depending
  on instance.

These are comparable to Devito's estimates from doc 06.

---

## 5. Honest Gaps and Risks at Building Scale

The steelman document identifies risks accurately but under-weights three issues specific
to the building-scale, high-frequency, sharp-contrast regime. Here they are with revised
severity ratings.

### Risk 1 — Thin-wall geometric resolution (MEDIUM → HIGH at this scale)

SPECFEM3D was designed for regional seismology where the smallest features (faults, sediment
layers) are 100–1000 m. A 0.15 m concrete wall is 3–4 orders of magnitude smaller than
the design target. The hex mesh must place 3–5 elements through the wall thickness for
physical accuracy. With dx_concrete = 0.03–0.05 m, the concrete shell mesh has
element sizes 30–50× smaller than the soil bulk. Gmsh's `TransfiniteVolume` handles this
by setting different element counts per direction, but it creates a high element-count
zone localised in the walls.

**Effect on element count**: If the wall has 4 elements through 0.15 m (dx = 0.0375 m),
the wall's element count is approximately (house perimeter × height / element area):
perimeter = 44 m, height = 3 m, face area per element = 0.0375² m² → ~94,000 wall
elements. The soil domain at dx = 0.30 m is ~24×26×10 / 0.30³ ≈ 720,000 soil elements.
Total: ~814,000 elements — manageable, but 10× more than the steelman's rough estimate
that used dx_concrete = 0.40 m (which would leave walls completely unresolved at 1 element
through a 0.15 m wall).

**Conclusion**: at 100 Hz, the wall-resolution requirement dominates the mesh design.
Plan for ~800K–1.5M elements per house (vs the steelman's ~1.5M total claim, which was
based on incorrect assumption that concrete elements can be as coarse as soil). This still
fits on one A100.

### Risk 2 — PML instability at sharp velocity contrasts (LOW → MEDIUM)

SPECFEM3D's C-PML implementation at the domain boundary is known to have late-time
instability issues when the domain contains high-velocity-contrast material near the
absorbing boundary. In our setup: the concrete house sits in a ~24×26×10 m domain.
The absorbing boundaries are 5 m of soil buffer from the building. The PML sees only
soil at its faces (concrete is in the interior) → the concrete/soil contrast does not
directly destabilize the PML. Risk is LOW for our geometry as long as the building
is not within ~3 elements of the absorbing boundary.

Mitigation: use **Stacey** absorbing boundaries (first-order; no instability; built-in
to SPECFEM3D) for data-generation runs. Accept ~5–10% reflection from Rayleigh waves
at near-normal incidence. Only use C-PML for validation runs where reflection artifacts
must be < 1%. The Stacey reflection artifact is systematic (consistent across all simulations)
and does not destroy relative Green's function patterns used for localization.

### Risk 3 — Site-city interaction literature mismatch (INFORMATIONAL)

Published SPECFEM3D site-city interaction studies (e.g. Viña del Mar, Chile study;
Tokyo study 2025) operate at frequencies of 0.5–12 Hz with buildings treated as continuum
elastic solids at mesh sizes of 0.5–2 m. Our target is 10–250 Hz with individual wall
resolution at 0.03–0.05 m. This puts us in a regime that is **3–20× higher frequency**
than any published SPECFEM3D building-scale application found in the literature search.
This is not a blocker — SEM has no frequency ceiling for physical accuracy — but it means
we cannot directly cite a precedent paper as our validation anchor. The bridge validation
(Section 1c) compensates for this: we cross-check against pyprop8 (exact) for the
soil-only case, and use FEniCSx shell-FEM (doc 06, Path A) as a reference for the
building-interaction case.

### Risk 4 — Gmsh hex mesh for boxy buildings with the SPECFEM3D material model (LOW)

The SPECFEM3D material model per element (not per-node) means every element must be
assigned to exactly one material. The concrete-soil interface is represented by the boundary
between adjacend elements of different material indices. With a conforming hex mesh (shared
nodes at the interface, produced by Gmsh's `BooleanFragments`/`fragment`), the interface
condition is automatically a welded contact (displacement and traction continuity). This
is physically correct for a building embedded in soil (no gap or slip). No special
interface element or coupling code is needed. Risk is LOW.

### Risk 5 — Reciprocity completeness for 3-component sensors (INFORMATIONAL)

The current sensor-chain uses vertical geophones: G_zz reciprocity (doc 10, Section 2c).
If the project later upgrades to 3-component sensors, 3 force directions × N_sensors shots
are needed (24 shots for N_sensors=8). `NUMBER_OF_SIMULTANEOUS_RUNS` can batch these.
The SPECFEM3D architecture handles this cleanly. No risk for the current vertical-only plan.

### Risk 6 — Compile complexity (LOW, consistent with steelman)

SPECFEM3D requires Fortran source compilation. On a clean Ubuntu 22.04 + CUDA 12.1 instance,
the CMake build takes ~20–40 minutes. No official Docker image from the SPECFEM team exists
(vs Devito's `devitocodes/devito:nvidia-nvc-latest`). Budget 0.5 days for the build
environment. This is a one-time cost.

### Risk 7 — Not validated at 250 Hz building scale by the community

At f_max = 250 Hz, the minimum Rayleigh wavelength in soil is:
λ_R = 0.9194 × 150 / 250 = **0.55 m**.
At 5 PPW (SEM minimum): dx_soil = 0.55 / 5 = **0.11 m**. At 10 PPW: 0.055 m.
The concrete wall (dx = 0.05 m) is in the right range for both the 10 PPW soil constraint
and the geometric 3-element wall constraint at 250 Hz. The mesh is consistent.
However, at 250 Hz the total element count increases by (250/100)³ ≈ 15× in the soil bulk.
A 250 Hz full house simulation would require ~10–20M elements and ~30–50 min per shot on A100.
The 100 Hz run is preferred for the initial campaign; 250 Hz is the high-fidelity option
if position ambiguity > 0.7 m is a problem.

---

## 6. Licensing Recap (Concise)

**SPECFEM3D**: GPL v3. Use as a compute engine (run on company infrastructure → output GF
libraries consumed by proprietary localization software) does not trigger GPL copyleft. Only
modifying and redistributing SPECFEM3D binaries would. Standard SaaS pattern; LOW risk.
Verify with a lawyer for the specific commercial model.

**Gmsh**: LGPL. Freely usable in proprietary code; no copyleft on linking. Clean for
commercial use.

**pyprop8**: MIT license (github.com/valentineap/pyprop8). No restrictions.

**CUBIT**: NOT NEEDED — the Gmsh path (LGPL) fully replaces CUBIT for boxy geometry.
This eliminates the steelman's "CUBIT commercial license for automated commercial meshing"
risk entirely.

---

## 7. Summary (6 Lines)

1. **pyprop8 vs SPECFEM3D**: pyprop8 is exact for 1-D layered soil but physically absent
   on the building — the house does not exist in the model, only a soil profile and a
   phenomenological SDOF plate filter. SPECFEM3D is the only way to correctly simulate the
   concrete shell, its coupling to the soil, its structural modes, and the 3-D scattered
   wavefield reaching exterior sensors. The bridge validation (run both on a soil-only
   halfspace) takes 0.5 days and simultaneously cross-validates both engines.

2. **Feature readiness**: 9 features are ready out-of-the-box; 6 require config-file edits;
   5 require new code. The 5 BUILD items (Gmsh→SPECFEM3D converter, mesh script, batch
   driver, HDF5 extraction, SpecFEMBank adapter) total ~9–15 engineering days of one-time
   work. The free surface costs zero because it is a natural BC in SEM.

3. **Adoption work-list**: 6 stages from environment to campaign. Critical path: Gmsh mesh
   script + converter (Stage 2, 3–5 days) is the longest single task. Total to first
   validated PoC (bridge + first house sim): ~5–8 days. Total to production pipeline: ~9–15
   days.

4. **Compute**: 100 Hz campaign at 50–200 houses with 8 reciprocal shots on a single A100
   takes ~0.5–6.5 hours (±50%); cloud cost < $30 for 200 houses. Use
   NUMBER_OF_SIMULTANEOUS_RUNS=8 to batch all 8 sensor shots in one MPI invocation.
   250 Hz is 15–39× more expensive; use cloud 8×A100 spot for large 250 Hz runs.

5. **Honest building-scale risks**: (a) thin-wall resolution requires dx_concrete ≤ 0.05 m
   in wall regions, producing ~800K–1.5M mesh elements per house — fine for A100 but larger
   than the steelman's rough estimate; (b) no published SPECFEM3D precedent at 250 Hz
   building scale — bridge validation against pyprop8 (soil) + FEniCSx (building) is the
   compensating strategy; (c) use Stacey (not C-PML) for data-generation runs to avoid
   late-time instability risk near sharp velocity contrasts.

6. **Decision**: SPECFEM3D is the correct long-term engine for this application. It is the
   only open-source tool that simultaneously provides: exact free surface, 3-D elastic with
   heterogeneous concrete-soil geometry, GPU CUDA, reciprocity via STATIONS file, and
   publishable-quality Rayleigh wave accuracy. The 9–15 day adoption cost is the price of
   correct physics. Recommend starting SPECFEM3D Stage 0–2 in parallel with Devito's
   free-surface implementation (doc 07/08 plan): if Devito's stress-image implementation
   succeeds in ~1 week, use Devito for the initial 100 Hz data campaign; use SPECFEM3D for
   the high-accuracy 250 Hz campaign and all publication-quality validation runs.

---

## Concrete Work-List (Quick Reference)

```
STAGE 0  │ 0.5–1 day   │ Build SPECFEM3D from source; verify GPU; write Dockerfile
STAGE 1  │ 0.5 day     │ Bridge validation: soil-only halfspace vs pyprop8 (3 quantities, 3 radii)
STAGE 2  │ 3–5 days    │ build_house_mesh(params) in Gmsh Python API + 10-file SPECFEM3D converter
STAGE 3  │ 1–2 days    │ Material files + FORCESOLUTION + STATIONS scripts; first full house sim
STAGE 4  │ 2–3 days    │ run_campaign.py: reciprocity loop (or NUMBER_OF_SIMULTANEOUS_RUNS=8)
STAGE 5  │ 1–2 days    │ SpecFEMBank class: HDF5 GF → scenes.py-compatible .emit() interface
STAGE 6  │ 1–2 days    │ Parametric house campaign; HDF5 GF library at 100 Hz, 50–200 houses
─────────┼─────────────┼──────────────────────────────────────────────────────────────────────
TOTAL    │ 9–15 days   │ To production pipeline; bridge validation at end of Stage 1
```

---

## Key Sources

- SPECFEM3D GitHub (v4.1.0, GPL v3): [github.com/SPECFEM/specfem3d](https://github.com/SPECFEM/specfem3d)
- SPECFEM3D User Manual (Mesh generation): [specfem3d.readthedocs.io/en/latest/03_mesh_generation](https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/)
- SPECFEM3D User Manual (Running solver, FORCESOLUTION): [specfem3d.readthedocs.io/en/latest/05_running_the_solver](https://specfem3d.readthedocs.io/en/latest/05_running_the_solver/)
- SPECFEM3D benchmark (OpenBenchmarking.org, 4 min avg): [openbenchmarking.org/test/pts/specfem3d](https://openbenchmarking.org/test/pts/specfem3d)
- Site-city interaction study (Viña del Mar, SEM, ScienceDirect): [sciencedirect.com/science/article/abs/pii/S0141029619355269](https://www.sciencedirect.com/science/article/abs/pii/S0141029619355269)
- Tokyo site-city interaction, 2025 (arXiv:2502.09976): [arxiv.org/pdf/2502.09976](https://arxiv.org/pdf/2502.09976)
- SPEED (SEM + DG non-conforming for multi-scale SSI, ResearchGate): [researchgate.net/publication/257235167](https://www.researchgate.net/publication/257235167_SPEED_SPectral_Elements_in_Elastodynamics_with_Discontinuous_Galerkin_a_non-conforming_approach_for_3D_multi-scale_problems)
- Gmsh ResearchGate thread (Gmsh→SPECFEM3D 3D mesh conversion): [researchgate.net/post/Can-anyone-explain-how-to-construct-a-mesh-in-Gmsh-for-import-into-SPECFEM3D](https://www.researchgate.net/post/Can-anyone-explain-how-to-construct-a-mesh-in-Gmsh-for-import-into-SPECFEM3D)
- pyprop8 (O'Toole & Woodhouse, JOSS): [joss.theoj.org/papers/10.21105/joss.04217](https://joss.theoj.org/papers/10.21105/joss.04217)
- pyprop8 GitHub: [github.com/valentineap/pyprop8](https://github.com/valentineap/pyprop8)
- Komatitsch & Tromp (1999): Introduction to the SEM for 3D seismic wave propagation. *GJI* 139(3), 806–822. — Free surface accuracy (< 0.5% at 8 PPW).
- Stacey BC vs C-PML discussion (SPECFEM3D GitHub issue #50): [github.com/geodynamics/specfem3d/issues/50](https://github.com/geodynamics/specfem3d/issues/50)
- NVIDIA SPECFEM3D GPU (38× P100 speedup): [nvidia.com/es-la/data-center/gpu-accelerated-applications/specfem3d-cartesian](https://www.nvidia.com/es-la/data-center/gpu-accelerated-applications/specfem3d-cartesian/)
- Prior engine steelman: `engine_decision/specfem3d.md`
- Prior Devito analysis: `07_sim_engine_verification_and_plan.md`, `08_free_surface_implementation.md`, `10_devito_numerics_gpu_viz.md`
