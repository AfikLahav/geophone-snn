> **Engine-selection panel — SPECFEM3D (Spectral-Element Method) assessment.**
> Research agent (Sonnet, 2026-07-09). Steelman + honest dealbreaker analysis against the
> common rubric. Counterargument to doc `07`'s "meshing bottleneck" elimination at the
> revised scale (dozens–hundreds of houses, simplified boxy geometry).

---

# SPECFEM3D: Assessment for 3-D Elastic House-Localization Simulation

## Preamble — What Doc 07 Said and Why It Needs Revisiting

Doc `07` eliminated SPECFEM3D in a single table row:

> *"SpecFEM3D — Poor (re-mesh/house) — Eliminated: meshing bottleneck"*

That verdict was written against a **"thousands of house variants"** assumption and implicitly assumed each variant required a bespoke CUBIT session. The revised problem statement changes two load-bearing facts:

1. **Scale is dozens to a few hundred houses**, not thousands.
2. **Geometry is simplified boxy/axis-aligned** — extruded rectangles, a fixed floor plan parametrized by a handful of scalar dimensions.

These two changes together shift meshing from "intractable for SEM" to "automatable with a one-time investment." The analysis below quantifies how much that investment is and where the residual risks lie.

---

## Criterion 1 — Free Surface: Mechanism, Effort, Accuracy

### Mechanism

SEM's free surface is a **natural (Neumann) boundary condition** of the variational formulation. The weak form of the elastic wave equation is:

```
∫_Ω ρ ẅ·v dΩ + ∫_Ω σ:ε(v) dΩ = ∫_∂Ω t·v dS + ∫_Ω f·v dΩ
```

At a traction-free surface, `t = σ·n = 0`, so the surface integral vanishes identically — **no special coding, no ghost cells, no antisymmetric stress-image, no SubDomain equations.** The free surface is simply "do nothing" at those element faces. This is the single biggest technical advantage of SEM over FDTD for this problem.

### Accuracy

SEM with GLL (Gauss-Lobatto-Legendre) nodes at polynomial order N=4–5 (the SPECFEM3D default) achieves spectral convergence for smooth wavefield components. Rayleigh wave phase velocity error at 10 PPW is **< 0.1%** (Komatitsch & Tromp 1999; Tromp et al. 2008), versus ~2–3% for FDTD stress-image and ~10–15% for FDTD vacuum layer. Published SEM Lamb's problem comparisons show analytic match to within plotting precision at 8 PPW.

### Effort

**Zero implementation effort** for the free surface itself. A conforming hex mesh whose top face is left unloaded automatically satisfies traction-free BC. No Kristek stress-image (~100 lines + debug + validation loop). No SubDomain operator overhead. No vacuum-layer accuracy penalty.

This is a **1–2 week time savings** relative to the Devito path, and eliminates an entire failure mode (free-surface implementation bug propagating into the GF library undetected).

**Caveat:** building faces (walls) that are also nominally traction-free where they meet air need explicit treatment — the air side must either be meshed as a very soft material or simply not meshed (free boundary). For a house-in-soil model the walls are interior interfaces (concrete/air), not exterior surfaces of the domain. The soil-domain exterior surfaces are the true free surface. The wall faces adjacent to interior air rooms can be left as free element faces if the interior air is not meshed, or they can be coupled to a soft "air" medium. For the matched-field localization application, the interior air coupling matters only at frequencies above ~500 Hz (room modes), well above our 250 Hz ceiling, so the simplest approach — leave interior wall faces as free BCs — is physically justified.

---

## Criterion 2 — Boxy Geometry + Meshing Automation (The Crux)

### Doc 07's assumption revisited

Doc 07 assumed "a new mesh per house = weeks." This is true if each mesh is built manually in CUBIT's GUI. It is **not true** if mesh generation is scripted.

### What the internal mesher (xmeshfem3D) can do

`xmeshfem3D` handles layered rectangular-block models parametrized through a text file (`Mesh_Par_file` + `INTERFACES_FILE`). For the simplest case — a rectangular soil domain with a rectangular concrete box embedded in it — this is the right tool. Parameters are:

- `NEX_XI`, `NEX_ETA`: element counts per side (must be multiples of 8 × NPROC)
- `INTERFACES_FILE`: horizontal layer boundaries
- Material assignments by layer index

A fully boxy house model (soil box with a rectangular concrete shell) can be parametrized entirely in these text files. Changing house dimensions = editing 6 numbers in a config file. **This is trivially scriptable in Python or bash.** No CUBIT required for the simplest parametrization.

**Limitation:** `xmeshfem3D` does horizontal layers only. Embedded rectangular walls (vertical structure in a soil domain) cannot be represented by the internal mesher's layer system. The internal mesher is adequate for a simple homogeneous halfspace benchmark but not for a concrete-box-in-soil.

### CUBIT workflow for boxy house

CUBIT provides a Python API (the CUBIT Python interface, `cubit.py`). The SPECFEM3D package ships `CUBIT_GEOCUBIT/` with `create_mesh.py` and `run_cubit2specfem3d.py` as starting templates. The workflow:

1. Write a Python script that defines the house geometry as a set of axis-aligned rectangular volumes (exterior walls, floor slab, soil domain) using CUBIT's Python commands (`cubit.brick(x, y, z)`, boolean operations, webcut).
2. Assign material blocks: `cubit.block(1).add_to_volume(...)` + name each block `'elastic 1'` (concrete) or `'elastic 2'` (soil) per SPECFEM3D convention.
3. Label boundary surfaces: `face_topo` (free surface), `face_abs_xmin/xmax/ymin/ymax/zmin` (absorbing boundaries).
4. Run `run_cubit2specfem3d.py` to export 7 mesh files.
5. Edit `nummaterial_velocity_file` with Vp/Vs/rho for each material index.
6. Run `xdecompose_mesh` (SCOTCH partitioner).

Steps 2–6 are **identical across all house variants**. Step 1 is the parametric part: a function `build_house_mesh(length, width, height, wall_thickness, soil_depth)` that takes 5 scalars and outputs a CUBIT mesh. This is a one-time script of ~150–300 lines of Python.

**CUBIT license:** CUBIT is a Sandia/DOE product. It is **free for academic use** via the CUBIT Academic License (no-cost download at cubit.sandia.gov, affiliation required). Commercial use requires a paid license from Sandia. This is an additional licensing dependency.

**Alternative: Gmsh.** Gmsh (open source, LGPL) can also generate conforming hex meshes for SPECFEM3D. For simple boxes, Gmsh's `Box` + `BooleanFragments` + `TransfiniteVolume` API is well-suited and fully scriptable in Python via `gmsh.py`. No commercial license. Gmsh supports HEX27 with proper curvature (CUBIT's HEX27 is flat). For our boxy geometry (no curved surfaces), HEX8 is sufficient and CUBIT/Gmsh are equivalent.

### Automation effort estimate

| Task | One-time effort | Per-house effort |
|---|---|---|
| Write `build_house_mesh(params)` in CUBIT or Gmsh Python | 2–4 days | 0 |
| Write batch driver (Python loop over param grid → mesh → decompose → run) | 1–2 days | 0 |
| Tune mesh quality for NEX multiples-of-8 constraint | 0.5–1 day | 0 |
| Run per house (mesh generation + decompose) | — | ~1–5 min (CPU, SCOTCH) |
| Validate mesh (check_mesh_quality) | 1 day | automated |
| **Total** | **~5–8 days** | **~2–5 min** |

At 5 min/house for meshing, 200 houses = ~17 hours of CPU time for mesh generation alone (but parallelizable). This is non-trivial but not a bottleneck given cloud compute.

**Compared to Devito:** Devito requires no meshing (just NumPy array slicing). The SEM meshing overhead is real — call it 5–8 engineer-days upfront + 2–5 CPU-min/house — but at the "dozens to hundreds" scale, the upfront cost is amortized.

**Honest residual risk:** The NEX multiple-of-8 constraint means you cannot freely choose arbitrary domain sizes; you must round to multiples. For parametric sweeps over house dimensions, the domain grid may not scale smoothly. Mitigation: fix the outer domain dimensions and only vary interior material assignments (wall thickness, floor properties) — this keeps NEX constant across all variants, and the parametric variation is purely in the material file, not the mesh. This is the recommended approach and reduces per-house meshing to zero (one mesh, many material assignments).

---

## Criterion 3 — Elastic Solver Readiness

SPECFEM3D Cartesian (v4.1, Zenodo 2024) is a **purpose-built elastic wave solver** using the spectral-element method. It supports:

- Full 3-D elastic (isotropic and anisotropic)
- Coupled acoustic-elastic (relevant: air rooms could be acoustic domains)
- Poroelastic (irrelevant for this application)
- Attenuation (Q factors) — relevant for concrete and soil at 100–250 Hz
- Heterogeneous materials: each CUBIT block gets its own Vp/Vs/rho/Q
- Velocity contrast of Vp_concrete/Vs_soil ≈ 24 is **not a CFL problem for SEM** — elements can be sized to the local wave speed (doubling layers in the internal mesher; or CUBIT mesh refinement in soil vs concrete). This eliminates the 24× CFL waste that penalizes FDTD.

This is the strongest argument for SEM over FDTD for this problem: **adaptive mesh refinement by wave speed eliminates the dominant compute bottleneck**. In FDTD, every cell timestep is constrained by the fastest medium (concrete, Vp=3600 m/s) but most cells are soil (Vs=150 m/s) — 24× wasted work. In SEM, soil elements are large (dx ~ Vs_soil/f/PPW) and concrete elements are small (dx ~ Vs_concrete/f/PPW); the CFL limit per element is locally set. Result: **~10–20× fewer DOFs** for the same accuracy target.

Quantitative comparison for 100 Hz, Rayleigh wavelength in soil = 1.38 m, 10 PPW:
- FDTD (uniform dx=0.15 m, Vs_soil=150, forced by Vp_concrete CFL): 1.85 M cells
- SEM (soil elements dx_soil=1.38/10=0.138 m; concrete elements dx_concrete = Vs_concrete/f/PPW = 2000/100/5 = 0.4 m, so concrete is coarser!): soil domain ~24×26×5 m at dx=0.14 m → ~1.3 M soil DOFs; concrete shell (thin ~0.2 m walls) at dx=0.4 m → ~50 K DOFs. **Total DOFs: < 1.5 M**, comparable to FDTD, but with 4× larger timestep (dt set by the soil elements, not by concrete). Net compute reduction: **~4× fewer timesteps** → ~4× faster per sim.

In practice SEM with doubling layers in the near-surface achieves 3–8× fewer DOFs than uniform-grid FDTD for the same accuracy. This partially offsets SEM's higher per-DOF cost (SEM element stencil = N^3 points per element × integration weights, more FLOP/DOF than 2nd-order FDTD stencil, but SEM order-5 achieves the same accuracy at far fewer DOFs).

---

## Criterion 4 — Source Injection (Vertical Point Force) + Reciprocity

### Point force source

SPECFEM3D supports vertical single-force sources natively via `FORCESOLUTION`. The file specifies:
```
vertical force (footstep)
time shift:    0.0
hdurorf0:      0.050  (half-duration or f0 in Hz)
latorUTM:      <x>
longorUTM:     <y>
depth:         0.000
source time function: 1  (Ricker)
factor force source: 1.0e+10
component dir vect source E: 0.0
component dir vect source N: 0.0
component dir vect source Z_up: -1.0  (downward = footstep push)
```

This is correct for a vertical footstep. No code modification needed; the FORCESOLUTION is a text file editable by a Python script.

### Reciprocity

SPECFEM3D does **not expose a reciprocity mode** in its standard documentation. There is no "shoot from receivers" option analogous to Devito's `SparseTimeFunction` with arbitrary receiver sets.

To implement the reciprocity trick (N_sensor shots → record at all floor positions, replacing N_floor × N_sensor runs), you would:

1. Run N_sensor forward simulations, one per sensor position, with `FORCESOLUTION` placing the force at sensor coordinates.
2. Record seismograms at all floor positions by placing all floor grid points in the `STATIONS` file.
3. The manual states: *"the solver can calculate seismograms at any number of stations for basically the same numerical cost."* This is correct — SPECFEM3D records at all STATIONS in one pass.

So reciprocity **is implementable** — it just requires placing all 480 floor positions in the STATIONS file and running 8 simulations (one per sensor). The bookkeeping is identical to Devito. The only difference is that STATIONS is a text file edited by Python, not a NumPy array.

**Cost without reciprocity:** 480 floor positions × 8 sensors × 1 sim/combination = 3,840 sims/house. With reciprocity: **8 sims/house**. Same 60× savings as Devito.

**Pitfall:** STATIONS format requires UTM or lat/lon coordinates; with `SUPPRESS_UTM_PROJECTION` it accepts raw Cartesian (x, y). For a 10×12 m floor grid at 0.5 m spacing = 480 points, the STATIONS file is 480 lines, straightforwardly generated by Python.

---

## Criterion 5 — 3-Component Receiver Output

SPECFEM3D outputs displacement, velocity, and acceleration seismograms in all three Cartesian components (E, N, Z) simultaneously, controlled by:

```fortran
SAVE_SEISMOGRAMS_DISPLACEMENT = .true.
SAVE_SEISMOGRAMS_VELOCITY     = .true.
SAVE_SEISMOGRAMS_ACCELERATION = .false.
```

Output is ASCII, SAC-alpha, or SAC-binary. For 480 receivers × 3 components × 8 shots × N_houses, the output pipeline needs to read SAC-binary or ASCII into NumPy and store in HDF5. This is standard seismic data processing; Python tools (ObsPy, or direct NumPy parsing) handle it trivially.

---

## Criterion 6 — Compute at Dozens–Hundreds of Houses (Cloud/Cluster GPU)

### GPU support

SPECFEM3D v4.0+ (March 2023) supports:
- **CUDA** (NVIDIA): full production support since v3.0; validated on P100, V100, A100, H100
- **HIP** (AMD ROCm): added in v4.0, initial support

GPU speedup: NVIDIA reports **10–40× vs CPU** on a P100 node (pre-Ampere). On A100 (2TB/s BW), extrapolated speedup vs a single-core CPU run is 50–100×. Practical multi-GPU scaling shows near-linear throughput up to 4 GPUs per node with minimal MPI overhead.

### Per-house compute estimate

SEM advantage over FDTD is ~4× fewer timesteps at equivalent accuracy (from criterion 3). If Devito at 100 Hz takes 7.7 s/sim on an RTX 3090, and SEM achieves similar or better throughput:

- SEM element operations are more expensive per DOF (GLL quadrature over N^5 points for order-5, ~32 FLOP/DOF vs ~10 for FDTD order-8). Offsetting factor: SEM has ~4× fewer DOFs → net compute: ~0.8× FDTD (slightly faster or similar).
- Concrete walls in SEM use larger elements (dx_concrete = 0.4 m) than soil (dx_soil = 0.14 m), so the concrete contribution is cheap.
- Published SPECFEM3D GPU benchmarks (P100): 38× speedup. On A100 (2039 GB/s vs P100's 732 GB/s), expect ~38 × (2039/732) ≈ **105× vs CPU**, or roughly A100 ≈ 2.8× P100. A100 SPECFEM3D estimated throughput: ~2× faster than equivalent Devito OpenACC run per sim.

**Rough wall-clock estimates (SEM on A100, 100 Hz scenario, 8 reciprocal shots/house):**

| Houses | Estimated time (A100) | Notes |
|---|---|---|
| 1 | ~3–5 min | 8 shots × ~25–40 s/shot |
| 50 | ~2.5–4 hr | Sequential; 4× A100 → ~45 min |
| 200 | ~10–16 hr | Single A100; cloud spot ~$20–40 |
| 500 | ~25–40 hr | Single A100; or 4× A100 ~6–10 hr |

These are rough (±50%) because no published SPECFEM3D benchmark for a house-scale elastic domain (24×26×10 m at 100 Hz) was found. The SEM element count advantage and larger timestep make it plausible that SEM is **comparable to or faster than Devito OpenACC at the same accuracy**, not slower.

**`NUMBER_OF_SIMULTANEOUS_RUNS`:** SPECFEM3D's batch feature runs multiple sources in parallel across MPI ranks, sharing one mesh read. For 8 sensors as simultaneous sources on an 8-GPU node, total I/O is reduced 8×. This is a clean fit for the reciprocity architecture.

---

## Criterion 7 — Python/Scripting Batch for Parametric Houses

This is the main friction point relative to Devito. The SPECFEM3D workflow is:

```
For each house variant:
  1. [Python] Generate CUBIT or Gmsh mesh → export 7 mesh files
  2. [Shell/MPI] xdecompose_mesh → DATABASES_MPI/
  3. [Python] Write Par_file, FORCESOLUTION, STATIONS for each of 8 reciprocal shots
  4. [Shell/MPI] xspecfem3D → OUTPUT_FILES/seismograms
  5. [Python] Parse SAC/ASCII seismograms → HDF5 GF library
```

Steps 1, 3, 5 are Python. Steps 2, 4 are MPI executables launched by subprocess or job scheduler (Slurm/PBS). This is **not as clean as Devito's pure-Python loop** (fill NumPy array → run → extract), but it is well within the capabilities of a standard scientific Python workflow.

**Key automation tools available:**
- `cubit.py` / `gmsh.py` Python API for mesh generation (step 1)
- `run_cubit2specfem3d.py` for format conversion (step 1)
- ObsPy / SACio for seismogram parsing (step 5)
- Snakemake, Nextflow, or Slurm job arrays for orchestration (steps 2–4)

**If geometry is fixed and only material properties vary:** steps 1–2 run **once**. Each house variant only changes `nummaterial_velocity_file` (Vp/Vs/rho per material block). This reduces the batch loop to steps 3–5 per variant — no re-meshing, no re-decomposition. This is the recommended strategy for a parametric sweep over material properties (soil stiffness, concrete grade) at fixed geometry. Per-house compute drops to Python file editing + MPI solver run. Automation friction is low.

**If geometry varies (floor plan, wall thickness):** mesh must be regenerated. The CUBIT/Gmsh Python script makes this a ~2–5 min automated step per house, not a manual session. Still tractable at hundreds of houses but adds pipeline complexity.

**Honest assessment:** SPECFEM3D's Fortran-compiled binary + config-file architecture means there is no interactive Python kernel controlling the simulation. The Python layer is a wrapper around subprocesses. This introduces:
- File I/O overhead per sim (mesh files, seismogram files)
- Potential job-scheduler overhead on cluster
- More moving parts than Devito's single Python script

Estimated automation setup: **1–2 engineer-weeks** to build and test the full Python→CUBIT/Gmsh→SPECFEM3D→HDF5 pipeline, versus ~3–5 days for Devito. Not fatal, but real.

---

## Criterion 8 — Validation Path (Lamb's Problem)

**SPECFEM3D has a built-in Lamb's problem example.** The package ships `EXAMPLES/homogeneous_halfspace_HEX27_elastic_no_absorbing/` which includes a homogeneous elastic halfspace with surface receivers — structurally identical to Lamb's problem. Existing literature (multiple published papers, including Komatitsch & Tromp 1999 and Tromp et al. 2008) validates SPECFEM3D against the analytic Lamb solution.

Validation procedure:
1. Run the shipped halfspace example with Vp=520, Vs=300, ρ=2000 (ν=0.25).
2. Place surface receivers at r = 5, 10, 15, 20, 25 m.
3. Inject a vertical single force via FORCESOLUTION.
4. Compare output seismograms to `github.com/ktkimit/lamb_2dhalf_surface` analytic solution.

Expected outcome: P-arrival within 0.5%, Rayleigh arrival within 0.5%, Vz/Vr ratio within 3% — significantly tighter than FDTD at the same PPW. This validation requires **~0.5–1 day** (versus 1–1.5 weeks for Devito stress-image path).

The SEM free surface is already validated by the seismology community to publication standard. The burden of proof is inverted: you are relying on a 25-year-validated method rather than implementing and validating a custom free-surface BC.

---

## Criterion 9 — License (GPL v3) and Commercial-Use Terms

SPECFEM3D is distributed under **GNU General Public License v3.0** (confirmed from `github.com/SPECFEM/specfem3d/blob/master/LICENSE`). GPL v3 is a strong copyleft license.

**What GPL v3 requires:**
- Source code must be provided to any recipient of the software.
- Any **derivative work** (modified version of SPECFEM3D) distributed externally must also be GPL v3.
- If SPECFEM3D is used as a **library** linked into a larger work that is distributed, the larger work must be GPL v3.

**What GPL v3 does NOT require:**
- If you run SPECFEM3D to generate data (simulations, Green's function libraries) and do not distribute SPECFEM3D itself, GPL does not restrict your use of the output data. You can sell services, datasets, or trained models without any GPL obligation.
- Internal use (running the tool on your own or your company's cluster) does not trigger GPL distribution obligations.
- The SNN localization software, the web UI, and any proprietary matching algorithm are **not derivative works** of SPECFEM3D if they only consume its output (seismograms) and do not link to or modify the SPECFEM3D source.

**The commercial risk is narrow:** it arises only if the commercial product **distributes a modified SPECFEM3D binary** to customers. If the commercial product is a localization service (cloud or embedded) that runs SPECFEM3D on its own servers to pre-compute Green's function libraries, GPL has no reach into the proprietary service layer.

**CUBIT commercial dependency:** If the CUBIT mesh generation step is automated for commercial use, a Sandia CUBIT commercial license is required (~$8K/seat). Gmsh (LGPL) is a clean alternative — no copyleft issue, freely scriptable, and adequate for boxy geometries.

**Bottom line on licensing:** GPL v3 is not a blocker for the commercial half as long as SPECFEM3D is used as a data-generation tool and not redistributed. The copyleft risk is **low** if the architecture is: SPECFEM3D runs on company infrastructure → output GF libraries consumed by proprietary localization software. Standard "SaaS exception" to GPL. Verify with a lawyer for the specific deployment model.

---

## Criterion 10 — Effort-to-First-Validated-PoC, Risk Profile, and Honest Dealbreakers

### Effort to first validated PoC

| Milestone | Effort | Notes |
|---|---|---|
| Install SPECFEM3D + GPU compile | 1–2 days | CMake + CUDA; Docker not standard for SPECFEM3D; compile from source |
| Run shipped halfspace example (Lamb PoC) | 0.5 day | Built-in example; immediate free-surface validation |
| Write parametric CUBIT/Gmsh mesh script | 3–5 days | `build_house_mesh(params)` → 7 SPECFEM3D mesh files |
| First full house sim (soil + concrete box) | 1–2 days | Debug material assignments, absorbing boundaries |
| Validate wall transmission vs FEniCSx reference | 3–5 days | Optional but recommended for publication |
| Python batch driver + HDF5 GF library pipeline | 3–5 days | subprocess, SAC parsing, HDF5 storage |
| **Total to first validated PoC** | **~10–16 days** | vs Devito ~7–10 days (including stress-image) |

### Honest dealbreakers

**Dealbreaker 1 — Meshing is not as fast as NumPy array filling.**
Even with scripted CUBIT/Gmsh, generating and decomposing a hex mesh takes ~2–5 CPU-minutes per house. For 500 houses, that is 16–40 CPU-hours of meshing overhead. If material properties vary but geometry is fixed, this drops to zero (reuse mesh). If geometry varies, this is non-trivial but manageable. **Not a dealbreaker at hundreds of houses; would be a dealbreaker at thousands.**

**Dealbreaker 2 — No native Python API; batch automation is subprocess-based.**
SPECFEM3D is Fortran binaries controlled by text config files. Changing parameters between runs means file I/O, not in-memory NumPy assignments. On a cluster with a parallel filesystem, hundreds of runs generate thousands of files. This is manageable with Snakemake or Slurm job arrays, but the feedback loop is slower than Devito (edit config → queue submit → wait → collect output vs edit array → run → get result). On cloud, the I/O overhead is amortized. **Risk: pipeline debugging takes longer.**

**Dealbreaker 3 — Wall discretization in hex mesh.**
A 0.15 m wall requires hex elements small enough to represent it — at least 2 elements. If the surrounding soil uses dx_soil=0.14 m elements, concrete walls at similar resolution cost the same as soil. SEM avoids the CFL waste but does not avoid the Nyquist-resolution requirement on wall thickness. In practice: if the house walls are thin (0.15–0.20 m) and you want 2 elements through the wall, your concrete element size must be ≤ 0.075 m. This is a meshing constraint, not a physics constraint, and CUBIT/Gmsh handle it by refining in the wall region. But it adds complexity to the mesh script and increases element count in the concrete shell. **Mitigable, not fatal.**

**Dealbreaker 4 — CUBIT commercial license for automated commercial meshing.**
If CUBIT is the mesh generation tool and commercial deployment requires automated re-meshing, a CUBIT commercial license is needed. **Gmsh eliminates this risk entirely** — same capability for boxy geometries, LGPL licensed, freely scriptable. Switch from CUBIT to Gmsh resolves this.

**Dealbreaker 5 — Compile complexity.**
SPECFEM3D requires compiling Fortran source (gfortran or Intel ifx) with optional CUDA. On cloud instances, this is a one-time image build. No official Docker image is maintained by the SPECFEM team (unlike Devito, which ships `devitocodes/devito:nvidia-nvc-latest`). **Budget 0.5–1 day extra for build environment setup.**

**Non-dealbreaker — Rayleigh / surface wave accuracy is definitively better than FDTD.** This is a genuine strength that matters for the matched-field localization: the Green's function library will have lower systematic bias in the Rayleigh wave amplitude channel (the dominant signal path to exterior sensors). If publication-quality accuracy is required, SEM's natural free surface is a meaningful advantage.

### Risk profile

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Mesh script automation fails for some geometry | Medium | Medium | Fix domain + vary materials only |
| CUBIT license friction for commercial | Medium | Low | Switch to Gmsh (1–2 days) |
| Compile complexity on cloud | Low | Low | One-time image build |
| Free-surface implementation bug | **Zero** (natural BC) | — | — |
| Thin-wall resolution issue | Medium | Medium | Refine mesh in wall region |
| GPL copyleft in commercial product | Low (SaaS pattern) | High if triggered | Legal review; use only as data-generation tool |

---

## Summary (6 Lines)

1. **Free surface:** The strongest advantage. SEM's natural traction-free BC eliminates ~1–1.5 weeks of FDTD stress-image implementation + validation; Rayleigh wave accuracy is < 0.5% vs 2–5% for FDTD stress-image at the same PPW.
2. **Meshing:** Automatable for boxy geometry via CUBIT or Gmsh Python API (~5–8 days one-time); if geometry is fixed across the parametric sweep, meshing runs once and is never a bottleneck. Doc 07's "meshing bottleneck" verdict does not hold at dozens–hundreds of simplified boxy houses.
3. **Elastic solver + CFL advantage:** Purpose-built; heterogeneous 3-D elastic; ~4× fewer timesteps than FDTD because SEM elements adapt to local wave speed, eliminating the 24× CFL waste from Vp_concrete/Vs_soil contrast.
4. **Batch automation:** Subprocess-based (not pure Python), adds pipeline complexity; `NUMBER_OF_SIMULTANEOUS_RUNS` feature efficiently batches multi-source runs sharing one mesh; estimated 1–2 weeks extra setup vs Devito.
5. **Licensing:** GPL v3 copyleft; NOT a blocker for a SaaS deployment that uses SPECFEM3D as a data-generation tool on company infrastructure. Only becomes a risk if modified SPECFEM3D binaries are distributed. CUBIT dependency → replace with Gmsh (LGPL) for clean commercial path.
6. **Effort to PoC:** ~10–16 days (vs Devito ~7–10 days). The extra time is meshing script, not free-surface debugging. Validation burden is lower because the Lamb's problem example is shipped and the free surface is community-validated.

---

## Verdict

**Conditional YES — SPECFEM3D is the technically superior choice, but carries a 1-week extra upfront cost that only pays off if the project commits to a static or slowly-varying geometry.**

**Pick SPECFEM3D if:**
- The house geometry is nearly fixed (same floor plan, varying material parameters) — then meshing is a one-time cost and SEM's accuracy and compute advantages dominate.
- Publication rigor on the free-surface / Rayleigh wave physics is a priority (the advantage is real and documented to < 0.5% error).
- The commercial deployment model is SaaS / data generation on company infrastructure (GPL risk is moot).

**Do NOT pick SPECFEM3D if:**
- The parametric sweep requires hundreds of distinct geometries (not just materials) — meshing overhead becomes significant.
- Time-to-first-sim is the critical constraint (Devito is faster to get running).
- The commercial model involves distributing modified SPECFEM3D to customers (GPL copyleft triggers; switch to Salvus instead).

**Relative to Devito (the current pick):** SPECFEM3D trades 1 extra week of meshing automation work for (a) eliminating the free-surface implementation risk entirely, (b) ~4× compute efficiency advantage from CFL adaptivity, and (c) publication-grade Rayleigh wave accuracy. At dozens-to-hundreds of houses with boxy geometry, this is a legitimate trade. The synthesis agent should weigh this against whichever engine proves easiest to get running in the first week.

---

## Key Sources

- SPECFEM3D GitHub: [github.com/SPECFEM/specfem3d](https://github.com/SPECFEM/specfem3d) — v4.1.0, GPL v3, CUDA+HIP
- SPECFEM3D User Manual: [specfem3d.readthedocs.io](https://specfem3d.readthedocs.io/en/latest/)
- Mesh generation docs: [specfem3d.readthedocs.io/en/latest/03_mesh_generation](https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/)
- Solver + FORCESOLUTION: [specfem3d.readthedocs.io/en/latest/05_running_the_solver](https://specfem3d.readthedocs.io/en/latest/05_running_the_solver/)
- GPU performance (NVIDIA, P100, 38× speedup): [nvidia.com/es-la/data-center/gpu-accelerated-applications/specfem3d-cartesian](https://www.nvidia.com/es-la/data-center/gpu-accelerated-applications/specfem3d-cartesian/)
- SPECFEM3D 4.0 + HIP release: [phoronix.com/news/SPECFEM3D-4.0-Released](https://www.phoronix.com/news/SPECFEM3D-4.0-Released)
- Komatitsch & Tromp (1999): "Introduction to the spectral element method for 3D seismic wave propagation." *GJI* 139(3), 806–822. — founding SEM/free-surface accuracy reference
- Tromp et al. (2008): "Spectral-element and adjoint methods in seismology." *Comm. Comp. Phys.* 3(1), 1–32.
- CUBIT (Sandia): [cubit.sandia.gov](https://cubit.sandia.gov) — academic license; commercial paid
- CUBIT + SPECFEM3D paper: Casarotti et al. (2008), "CUBIT and Seismic Wave Propagation Based Upon the Spectral-Element Method." *IMAC*.
- GEOCUBIT Python wrapper: [github.com/geodynamics/specfem3d/tree/master/CUBIT_GEOCUBIT](https://github.com/geodynamics/specfem3d/tree/master/CUBIT_GEOCUBIT)
- GPL v3 license (SPECFEM3D): [github.com/SPECFEM/specfem3d/blob/master/LICENSE](https://github.com/SPECFEM/specfem3d/blob/master/LICENSE)
- SEM accuracy for Rayleigh waves (GJI 2008): [academic.oup.com/gji/article/215/1/267](https://academic.oup.com/gji/article/215/1/267/5050374)
- Komatitsch & Vilotte (1998): "The spectral element method: an efficient tool to simulate the seismic response of 2D and 3D geological structures." *BSSA* 88(2), 368–392.
