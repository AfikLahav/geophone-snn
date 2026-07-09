> **Engine-selection panel — General FEM route (FEniCSx/Firedrake + Nektar++) assessment.**
> Written by a panel agent (Sonnet), 2026-07-09. Honest steelman with explicit dealbreaker
> disclosure. Part of a 4-engine comparison (Devito / Salvus / SpecFEM3D / FEniCSx+Nektar++).
> Context: 3-D elastic Green's-function dictionary for indoor footstep localization at exterior
> geophones; dozens to ~hundreds of simplified boxy houses; reciprocity architecture (N_sensor
> shots, not N_floor); cloud/cluster compute delegatable. Inputs: docs 06–08, 10 + targeted web
> research (Jul 2026).

---

# General FEM Route — FEniCSx / Firedrake / Nektar++: Assessment

## Scope of this document

"General FEM route" covers two distinct technology families:

- **FEniCSx (DOLFINx)** — Python-driven FEM problem-solving environment, LGPL-3.0. The primary framework assessed here; includes the community `elastodynamicsx` (MIT) layer on top.
- **Firedrake** — Imperial College / Exeter automated FEM framework, LGPL-3.0. Similar architecture to FEniCSx; the `spyro` wave solver (acoustic-only at present, 2022) is the most geophysics-proximate application.
- **Nektar++** — Cambridge/Imperial spectral/hp element framework, MIT license. Ships ten+ built-in solvers but **no elastic seismic time-domain solver out of the box**. v5.9.0 released Nov 2025.

Doc 07 called FEniCSx "validation tool only." The rebuttal asked for is: can it be more, given simplified boxy geometry and dozens–hundreds of houses? This document assesses honestly.

---

## Criterion 1 — Free Surface: Mechanism, Effort, Accuracy

### The genuine FEM advantage

In any weak-form FEM (FEniCSx, Firedrake, Nektar++), the traction-free condition
`(C:ε(u))·n = 0` is the **natural Neumann boundary condition** of the variational problem.
The surface integral over a traction-free face vanishes identically upon integration by parts.
The implementation effort is: **do nothing** — no special code, no ghost cells, no antisymmetric
stress-image, no SubDomain equations, no Lamb's-problem validation pass required for the free
surface itself.

This is the single strongest technical argument for the FEM route over FDTD. The Devito path
requires ~130 lines of SubDomain ghost-cell code plus a 1–1.5 week implementation and validation
loop (Kristek 2002 stress-image, Bohlen & Saenger 2006 accuracy characterization, SW4
cross-check) just to get a free surface that achieves ~5% Rayleigh amplitude accuracy at 9.2
PPW. FEM gets ~1% accuracy for free from the variational principle.

### Accuracy numbers

SEM-based free surface at 10 PPW gives < 0.1% Rayleigh phase error (Komatitsch & Tromp 1999,
GJI). Standard Galerkin FEM (Lagrange elements, order 2–4) achieves < 1% at the same
resolution. The comparison to Devito:

| Method | Rayleigh phase error at 10 PPW | Rayleigh amplitude error at 10 PPW | Implementation effort |
|---|---|---|---|
| FDTD vacuum layer | 2–3% | 10–15% | 5 lines (inaccurate) |
| FDTD stress-image (Kristek 2002) | 2–3% | ~5% | ~130 lines + 1–1.5 wk validation |
| FEM natural BC (FEniCSx/Nektar++) | < 1% | < 1% | **0 lines** (variational default) |

For a publication targeting Rayleigh-wave-sensitive Green's functions, the FEM free-surface
accuracy is a real and meaningful advantage.

### Internal free surfaces (wall-to-air)

For a house model with unoccupied rooms, interior wall faces bounding air are also traction-free.
In FEM this is again the natural BC — leave those faces unloaded, done. No "air layer"
approximation, no impedance contrast approximation. The structural model is exactly correct for
the elastic problem without coupling to interior acoustics (valid up to ~500 Hz where room modes
first matter — well above 250 Hz, so OK).

**Verdict: Strong pass. FEM's most decisive advantage over FDTD in this problem.**

---

## Criterion 2 — Simplified Boxy-House Geometry + Meshing Automation

### FEniCSx + Gmsh path

For axis-aligned boxy houses, the Gmsh Python API is the natural meshing tool:

```python
import gmsh

gmsh.initialize()
gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)        # soil domain
gmsh.model.occ.addBox(x0, y0, z0, hx, hy, hz)     # house exterior
gmsh.model.occ.addBox(x1, y1, z1, hx-2t, hy-2t, hz)  # interior air (subtract)
gmsh.model.occ.fragment([(3,1)], [(3,2), (3,3)])   # conformal interface
# Assign physical groups, mesh, export to .msh
gmsh.model.mesh.generate(3)
gmsh.write("house.msh")
```

For purely axis-aligned geometry with a few material regions, a parametric `build_house_mesh()`
function in Python is ~150–250 lines and runs once. Per-house mesh generation time in Gmsh for
this domain (~24×26×10 m at ~0.1 m element size): **~30–120 seconds** per house variant. At
hundreds of houses this is a real overhead (hundreds of minutes total) but is parallelizable and
not a blocker.

**The decisive difference from Devito:** Devito generates a new material via NumPy array slicing
in milliseconds with zero meshing. FEM requires a new conformal mesh per geometry variant — the
mesh pipeline (Gmsh → meshio → dolfinx mesh import → physical group assignment) adds 1–2 minutes
per house **plus** engineering effort to build and test the parametric Gmsh script.

**Mitigation strategy:** If parametric variation is primarily over material properties (soil Vs,
concrete grade) at **fixed geometry**, the mesh is generated once and reused across all
material-property variants. This is exactly the scenario where FEM overhead drops to near-zero
for the batch campaign. The residual overhead is the initial meshing setup (2–4 days) rather
than per-variant re-meshing.

**If geometry varies** (floor plan dimensions, wall thickness, number of floors): re-meshing is
required per house variant. At ~1 min/house for 200 houses: ~3 hr overhead. At 500 houses: ~8 hr
overhead. Non-trivial but manageable on cloud.

**Thin-wall meshing (the honest FEM cost for geometry-varying sweeps):**
A 0.15 m concrete wall requires at least 2 elements through its thickness for a meaningful
material response (order-2 Lagrange). With element size ≤ 0.075 m inside the wall, the
surrounding soil uses a larger element size (~0.1–0.14 m). Gmsh's `setSize` or `Field` API
handles this via refinement zones. The conformal interface between concrete (fine) and soil
(coarse) requires either a matching boundary layer or Gmsh's `fragment` + size field approach.
This is standard practice in FEM, but it adds ~1–2 days to the mesh script development and
can cause poor-quality elements at the concrete-soil interface if not handled carefully.

**Verdict: Moderate pass. Axis-aligned geometry is tractable in Gmsh Python; conformal meshing
eliminates staircasing at the cost of per-house mesh generation time and engineering setup.**

---

## Criterion 3 — Elastic Velocity-Stress Solver Readiness

### THIS IS THE CRITICAL WEAKNESS — HONEST ASSESSMENT

**Neither FEniCSx, Firedrake, nor Nektar++ ships a production-ready 3-D elastic seismic
wave solver out of the box. You must build your own.**

#### FEniCSx / DOLFINx

The FEniCSx documentation and `elastodynamicsx` (v0.4.0, MIT, Université Gustave Eiffel) provide
elastodynamic time-domain tutorials in 2-D and 3-D using the Newmark-β / generalized-α method.
The `elastodynamicsx` library is the closest thing to a ready-to-use FEniCSx wave solver:

```python
# elastodynamicsx 0.4.0 — conceptual example
from elastodynamicsx.pde import material, BoundaryCondition, PDESYS
from elastodynamicsx.solvers import TimeStepper

mat = material(cell='tetrahedron', type_='isotropic-linearly-elastic',
               rho=1800, lambda_=lam_soil, mu=mu_soil)
pde = PDESYS([mat], ...)
stepper = TimeStepper.build(pde, scheme='newmark-explicit', dt=dt, ...)
stepper.run(nt)
```

`elastodynamicsx` is a **research library**, not a production solver:
- v0.4.0 released (no date in search results; the GitHub was last active as of mid-2024 based
  on indirect evidence). No formal GPU support.
- No built-in absorbing boundary condition (dashpot or PML). The user must implement ABCs
  as weak-form Robin conditions — feasible but non-trivial (~50–100 additional lines for
  Lysmer-Kuhlemeyer dashpots; PML requires 150–300 lines with auxiliary variables).
- No built-in seismic source injection (point force at an interior node). Must use Dirac
  delta approximation (distribute source over a ball of 1–2 elements) or DSS (diagonal
  scaling source term).
- The FEniCSx tutorial at `bleyerj.github.io/comet-fenicsx` provides a Newmark
  elastodynamics example, but it is 2-D and uses implicit time-stepping — **implicit
  Newmark for 3-D is extremely expensive** (requires solving a large sparse linear system
  at each timestep, O(N^1.5) memory and O(N^2) time for a direct solver, or iterated
  Krylov+preconditioner with convergence sensitivity).

**The build-your-own burden for FEniCSx:**

| Component | Status | Effort |
|---|---|---|
| Weak-form elastic operator (isotropic) | Available (dolfinx tutorial, elastodynamicsx) | 1–2 days |
| Explicit time-stepping (central difference or lumped Newmark-β) | Tutorial shows implicit; explicit requires mass lumping, ~50 extra lines | 2–3 days |
| Absorbing boundary (Lysmer-Kuhlemeyer dashpot) | Not in elastodynamicsx; must implement as Robin BC weak form | 2–3 days |
| PML (perfectly matched layer) | Not available; requires auxiliary variable formulation (~300 lines) | 1–2 weeks |
| Point-force source injection | Not in tutorials; requires delta-function approximation over patch | 1–2 days |
| Free surface | Zero effort (natural BC) | — |
| 3-component receiver extraction | SparseTimeFunction equivalent — `dolfinx.geometry.BoundingBoxTree` interpolation | 1 day |
| Lamb's-problem validation | Must run and compare to analytic — separate effort | 1–2 days |
| **Total build effort before any house sim** | **~3–5 weeks** | Engineer-weeks |

This is the honest cost. The FEM formulation is textbook, but assembling a validated 3-D explicit
elastic wave solver with correct BCs, source injection, and receiver extraction from FEniCSx
primitives requires 3–5 weeks of focused engineering — significantly more than Devito's 1–1.5
week free-surface path on top of an already-working solver.

#### Firedrake

Firedrake's `spyro` library (2022, published in GMD) provides an acoustic wave solver with FWI
capability. It does **not solve the elastic wave equation** — it uses the acoustic scalar wave
equation with the Cole-Cole attenuation model. The Firedrake docs (2026.4.2) include a full FWI
demo but, again, acoustic only. Building a 3-D elastic solver in Firedrake would require the same
effort as FEniCSx (the architecture is nearly identical — UFL-based variational form, PETSc
backend, Python scripting).

Firedrake has a notable advantage over FEniCSx in the FWI context (automatic differentiation via
the pyadjoint framework), but this is irrelevant for the forward Green's-function generation task
here.

#### Nektar++

Nektar++ ships a `LinearElasticSystem` class (documented in v4.4.0 doxygen). This is a static
elastic solver (frequency-domain or steady-state), NOT a time-domain wave propagation solver.
The `AcousticSolver` (Nektar++ major application area) solves the acoustic wave equation —
again, not elastic. No ready-to-use 3-D elastic time-domain solver exists in the Nektar++ release
tree.

Building an elastic time-domain solver in Nektar++ would require implementing the wave equations
as a custom `SolverBase` subclass or extending the existing linear elasticity system with time
marching — a much lower-level operation than in FEniCSx/Firedrake (requires C++ work, not just
Python UFL). This is a harder and longer route than FEniCSx.

The Nektar++ spectral/hp element basis functions (GLL basis at arbitrary polynomial order)
would give excellent accuracy per DOF for a wave simulation, but the development barrier is high.

**Bottom line on Criterion 3:** The build-your-own burden is the central weakness of the
general FEM route. This is what doc 07 flagged when labeling FEniCSx "validation tool only,"
and that assessment is correct. The FEniCSx/Firedrake build burden (3–5 weeks) is more than
twice the Devito free-surface implementation path (1–1.5 weeks), and Nektar++ would take even
longer to get a working time-domain elastic solver.

**Verdict: Weak pass — the physics formulation is correct, the tools exist, but there is no
ready production solver. Honest estimate: 3–5 engineer-weeks to a validated first sim.**

---

## Criterion 4 — Source Injection (Vertical Point Force) + Reciprocity Component Bookkeeping

### FEniCSx

A vertical point-force body load `f = (0, 0, F_z) δ(x - x_s)` enters the weak form as a
linear functional. In FEniCSx the approximation is a `PointSource` (available in legacy FEniCS
as `PointSource`; in DOLFINx, the recommended approach is distributing the load over a narrow
Gaussian or over nearest-node DOFs via `fem.assemble_vector` with a `DiracDelta` approximation).
The DOLFINx-native approach (v0.8+) uses interpolation onto a local patch to approximate the
delta function — this introduces a finite-support smearing error that is O(h) for linear elements
and O(h^p) for p-th order elements. At element size h=0.1 m and order-2 elements, this is a
~2% source smearing error — negligible for training-data generation.

**Reciprocity bookkeeping** is identical in principle to Devito: place the source at the sensor
location, record the displacement/velocity field at all 480 floor positions. In FEniCSx,
receiver extraction uses DOLFINx `pointcloud` interpolation (the `dolfinx.geometry` module's
`BoundingBoxTree` → `compute_collisions` → `eval` pipeline). This is ~20 lines of Python,
straightforward but requires setting up the geometry query tree once per mesh. No conceptual
difficulty; this is standard FEM post-processing.

**Component bookkeeping:** For vertical geophones (G_zz only), inject `f_z` at sensor, record
`u_z` (or `v_z`) at all floor positions in one pass — exactly as in Devito. For 3-component
geophones, 3 separate force directions × 1 shot each = 3 shots per sensor, 24 shots total for
8 sensors. The DOLFINx point-source API handles arbitrary force directions cleanly.

**Verdict: Pass. Slightly more engineering setup than Devito's SparseTimeFunction but no
conceptual obstacle.**

---

## Criterion 5 — 3-Component Receiver Output

FEniCSx vector function space solutions `u` are 3-component displacement fields. Extraction of
`u_x`, `u_y`, `u_z` at arbitrary receiver locations uses `dolfinx.geometry.compute_collisions_points`
+ `dolfinx.fem.Function.eval` — single-pass, all three components simultaneously. Velocity
components are obtained via time-stepping accumulators. Memory footprint: 480 receivers ×
3 components × 20,833 time steps × 4 bytes = **120 MB/shot** — trivially manageable.

**Verdict: Trivially available. No limitations.**

---

## Criterion 6 — Compute at Dozens–Hundreds of Houses on Delegated Cloud/Cluster

### The GPU situation for FEniCSx — Honest Assessment

GPU acceleration in FEniCSx (DOLFINx) is **not production-ready as of Jul 2026**. The situation:

- **cuDOLFINx** (`github.com/bpachev/cuda-dolfinx`): a research extension providing CUDA-
  accelerated assembly routines. Benchmarks show "up to 40× faster than CPU MPI assembly."
  However, this is assembly speedup only — the linear solver (PETSc GMRES/CG) also needs
  GPU support, which comes via PETSc's CUDA/HIP backend. The two do not yet integrate into
  a seamless end-to-end GPU pipeline out of the box as of the FEniCS 2024 conference.
- **dolfinx-gpu-solvers** (`github.com/FEniCS/dolfinx-gpu-solvers`): official FEniCS
  project repository for GPU-accelerated linear solvers on top of DOLFINx. Described as
  "under active development" with GPU support for CUDA and HIP. No production release noted.
- **dolfinx.sycl**: SYCL-based GPU extension via Intel's DPC++ or hipSYCL. Experimental.
- **Firedrake GPU strategy** (GitHub discussion #4586): as of mid-2024 the plan was to use
  IREE as a backend; "initial GPU functionality expected to merge in late 2024 if grant
  funded" — no confirmed production release found as of the search results available.

**What this means in practice:** For an explicit time-stepping elastic wave simulation, the
innermost loop is a matrix-vector product with the stiffness matrix (or a vector update for
truly explicit formulations). Without production-ready GPU assembly + solver integration,
FEniCSx runs on CPU (multi-core OpenMP/MPI) only. For this project's domain (28.9 M DOFs at
250 Hz), a 3-D FEM explicit Newmark solve running on CPU:

- DOF count at 100 Hz (h=0.15 m, order-2 tet): ~3M DOFs (comparable to Devito 1.85M cells).
- Per-step cost for an explicit mass-lumped solve: one matrix-vector product (~30 FLOP/DOF
  for order-2 tet) → ~90 GFLOP per step (with compute-bound execution, not memory-bound
  like FDTD). On a 64-core CPU @ ~3 TFLOPS (FP32): ~30 ms per step → ~620 s (10 min) for
  20,833 steps. One shot for 100 Hz: ~10 min. Eight shots/house: ~80 min. 50 houses: ~67 hr.
  500 houses: **27 days**. This is prohibitively slow without GPU acceleration.
- At 250 Hz (h=0.06 m, ~28M DOFs): ~16× more DOFs than 100 Hz → proportionally worse.

**Without GPU, FEniCSx is not viable as the primary production engine at this scale.**

**With cuDOLFINx + PETSc GPU (if working):** If GPU assembly and GPU linear solve are both
active, speedup vs CPU is potentially 10–40×. This would bring 100 Hz from ~80 min/house to
~4–8 min/house on an A100. But this pipeline is not documented as production-ready, and the
engineering overhead to get the GPU stack working is substantial (integrating cuDOLFINx +
PETSc CUDA + dolfinx-gpu-solvers into a coherent stack).

**Nektar++ GPU status:** Also "under active development" for GPU (accelerator support noted as
future work in user guide v5.5.0). No confirmed production CUDA backend for the built-in
solvers. A custom elastic time-domain solver in Nektar++ would not have automatic GPU support.

**Firedrake GPU status:** Plan-level as of 2024; no confirmed production release.

**Verdict: Weak pass at best. GPU acceleration in FEniCSx/Firedrake/Nektar++ is research-grade,
not production-ready. At this project's scale, CPU-only FEM is too slow for the batch campaign.
This is the second major weakness alongside the build-your-own solver burden.**

---

## Criterion 7 — Python Batch/Scripting for Parametric Houses

### FEniCSx — Strong Python Integration

FEniCSx (DOLFINx) is fully Python-native. The variational form, mesh, boundary conditions,
time-stepping loop, and receiver extraction are all first-class Python objects. The batch
loop for a parametric house sweep is:

```python
for wall_t, soil_vs, hx, hy in param_grid:
    mesh, cell_tags, facet_tags = build_house_mesh(wall_t, soil_vs, hx, hy)
    lam, mu, rho = build_material_functions(mesh, cell_tags, wall_t, soil_vs)
    # Rebuild FEM forms (material-dependent) — this DOES require re-assembly
    a, L, M_lumped = build_elastic_forms(mesh, lam, mu, rho)
    for k, sensor_pos in enumerate(sensor_positions):
        u_hist = run_elastic_sim(mesh, a, L, M_lumped, sensor_pos, nt, dt)
        save_to_hdf5(u_hist, house_id, k)
```

**Honest complexity:** Unlike Devito (where the symbolic operator is identical for all houses
and only material arrays change), in FEniCSx each new mesh requires **re-assembling the stiffness
and mass matrices** from scratch. This is because the element connectivity changes when the
mesh changes. Re-assembly at 100 Hz (~3M DOFs, order-2 tet): ~10–60 seconds per house (serial
CPU assembly). With cuDOLFINx this drops to ~1–3 seconds. For 200 houses: ~20–120 min of
assembly overhead on CPU alone. Non-trivial but not a blocker.

If geometry is fixed (only material properties vary): in FEniCSx, the form can be precompiled
via FFCx (FEniCS Form Compiler) and only the coefficient arrays updated per run — similar to
Devito's operator caching. This is achievable with `dolfinx.fem.Constant` for uniform materials
or `Function` for spatially varying coefficients. Re-assembly for a new material assignment at
fixed mesh: ~1–10 seconds (coefficient interpolation only). This is the efficient path for
material-only parametric sweeps.

### Nektar++ — NOT Python-native

Nektar++ is a C++ framework. The Python bindings (NekPy) expose some functionality but are
far from a complete Python API for building custom PDE solvers. Scripting a parametric house
simulation campaign in Nektar++ would require C++ solver development + Python/bash wrappers for
parameter injection — significantly more friction than FEniCSx. This makes Nektar++ the weaker
choice on this criterion despite its superior spectral accuracy.

**Verdict: FEniCSx/Firedrake — Strong pass (Python-native, comparable to Devito in scripting
ergonomics). Nektar++ — Weak pass (C++ core, limited Python API for custom solver development).**

---

## Criterion 8 — Validation Path (Lamb's Problem)

### FEniCSx Lamb's problem

Because the free surface is the natural BC, the Lamb's-problem validation becomes a check of
the interior wave equation accuracy and the absorbing boundaries — not the free surface itself.
The validation procedure:

1. Build a 50×50×25 m homogeneous halfspace mesh (soil only, Vp=520, Vs=300, ν=0.25).
2. Apply the vertical point force at the free surface center.
3. Run the time-stepping loop for 0.5 s.
4. Extract v_z and v_r at r = 5, 10, 15, 20, 25 m surface receivers.
5. Compare to `github.com/ktkimit/lamb_2dhalf_surface` analytic solution.

Pass criteria are identical to Devito (P-arrival ±1%, Rayleigh arrival ±1%, |Vz/Vr| ±10% of
1.47). Since the free surface is exact (variational), the expected error is primarily from grid
dispersion in the interior and the absorbing BC quality. FEM at order-2 with 10 PPW: < 2%
phase error; order-3 or higher: < 0.5%. The Rayleigh amplitude criterion should be met
comfortably.

**Effort for Lamb's problem after building the FEM solver:** ~1–2 days (once the solver exists).
**Effort before building the solver:** 3–5 weeks. The validation is not the bottleneck; the
solver build is.

**Verdict: Pass (once the solver is built, validation is easier than for Devito).**

---

## Criterion 9 — License (LGPL / MIT) + Commercial-Use Terms

### FEniCSx / DOLFINx — LGPL-3.0-or-later

- **DOLFINx**: LGPL-3.0+. FFCx: LGPL-3.0+. UFL: LGPL-3.0+. Basix: **MIT**.
- LGPL is NOT GPL. The LGPL-3.0 "library exception" allows proprietary software to link against
  LGPL libraries without becoming copyleft, as long as the LGPL library itself remains available
  as a dynamically linked shared object.
- **Commercial use: permitted.** A proprietary localization SaaS that uses DOLFINx to generate
  training data (runs on company infrastructure, outputs are datasets/models) carries zero LGPL
  obligation. Even if DOLFINx is dynamically linked into a custom solver, the LGPL permits this
  for commercial products as long as the DOLFINx source/library remains separately available
  to the end user (which it is — it's on GitHub).
- **Caution:** if the company **distributes** a statically linked binary that incorporates
  DOLFINx, the LGPL requires either dynamic linking or providing the DOLFINx source to the
  recipient. For a cloud/SaaS deployment, this is not triggered. For an embedded product that
  ships a simulator binary to customers: consult a lawyer about the static vs dynamic linking
  distinction.

**In practice for this project (batch generation on company/cloud cluster, output is a GF
dataset):** LGPL imposes zero operational restriction. Clean for commercial use.

### Firedrake — LGPL-3.0

Same LGPL-3.0 license; same analysis applies. PETSc (BSD-2) and SLEPc (LGPL) are dependencies.
Clean for cloud-based data generation.

### Nektar++ — MIT

MIT is the most permissive open-source license. Zero restrictions for any commercial use,
redistribution, or integration. No copyleft, no source-disclosure requirements. Nektar++'s
dependencies include BLAS/LAPACK (BSD), Boost (BSL-1.0), and PETSc (BSD). All permissive.

**Verdict: All three are clean for commercial use. FEniCSx/Firedrake (LGPL) are slightly more
complex than Nektar++ (MIT) but neither is a practical barrier for this project's deployment
model. Nektar++ wins on license simplicity; FEniCSx wins on practical relevance.**

---

## Criterion 10 — Effort-to-First-Validated-PoC + Risk Profile + HONEST DEALBREAKERS

### Effort timeline (FEniCSx path — the most tractable of the three)

| Task | Engineer-days | Notes |
|---|---|---|
| Install DOLFINx + FEniCSx + elastodynamicsx + Gmsh | 0.5 | Conda or Docker; well-documented |
| Write parametric Gmsh mesh script for boxy house | 2–4 | One conformal mesh + soil domain |
| Build 3-D isotropic elastic Newmark operator in FEniCSx | 2–3 | Weak form, lumped mass, central difference |
| Implement Lysmer-Kuhlemeyer absorbing BC (Robin) | 2–3 | Weak-form boundary integral; no PML yet |
| Implement vertical point-force source injection | 1 | DOFmap delta-function approximation |
| Implement 480-point receiver extraction (BoundingBoxTree) | 1 | dolfinx.geometry interpolation |
| Lamb's-problem validation (homogeneous halfspace) | 1–2 | Free surface is automatic; check dispersion + ABL |
| First full house sim (soil + concrete box) | 1–2 | Debug material assignment, check timing |
| Reciprocity GF library for one house (8 shots, HDF5 output) | 1 | SparseTimeFunction equivalent |
| GPU pipeline (cuDOLFINx + PETSc CUDA) setup | 3–7 | EXPERIMENTAL; may not achieve stable throughput |
| **Total to first CPU-validated PoC** | **~14–24 days** | 3–5 engineer-weeks |
| **Total to GPU-accelerated PoC** | **+3–7 additional days if GPU stack works** | High risk |

Compare to Devito (7–10 days to validated PoC including stress-image) and SPECFEM3D (10–16 days,
including mesh automation). FEniCSx takes 14–24 days — roughly twice Devito's path — mostly
because building the explicit elastic solver from FEniCSx primitives is 3–4× more work than
adapting Devito's existing elastic tutorial.

### Risk register

**Risk 1 — CPU-only simulation is too slow for the production campaign (probability: HIGH)**
Without production-ready GPU support, a 3-D FEniCSx explicit elastic solver runs on CPU. At the
project's required throughput (hundreds of houses), CPU performance is insufficient by ~10–30×.
This is not a "moderate" risk — it is a near-certainty unless GPU support is successfully
integrated. The mitigation (cuDOLFINx + PETSc CUDA) is itself a research-grade stack with
non-trivial integration complexity.

**Risk 2 — Explicit mass-lumping accuracy for FEM tetrahedral meshes (probability: MEDIUM)**
Explicit time-stepping with Newmark requires a lumped (diagonal) mass matrix. Standard Lagrange
elements on tetrahedral meshes do not yield spectrally accurate lumped masses — the HRZ
(Hinton-Rock-Zienkiewicz) lumping is used in practice, which reduces accuracy slightly vs
consistent mass. For wave propagation, lumped mass introduces dispersion that is approximately
equivalent to increasing PPW by 20–30%. This means 10 PPW with lumped mass performs like 7–8
PPW with consistent mass. Mitigations: use higher-order elements (order-3+), finer mesh, or
hexahedral elements (which have better lumping properties). This adds DOFs and cost.

**Risk 3 — Absorbing BC quality (probability: MEDIUM-HIGH)**
The Lysmer-Kuhlemeyer dashpot (~5–20% reflection for body waves) is significantly worse than
Devito's sponge or CPML. Implementing a proper elastic PML in FEniCSx requires auxiliary
differential equations (~300 lines of UFL/Python). Without PML, the sponge-equivalent in FEM
is a lossy-material absorbing zone (similar to Devito's model.damp), which achieves comparable
quality. The PML in FEM is harder to implement correctly than in FDTD due to the need for
auxiliary fields in the variational form — this is a research-level task, not a textbook one.

**Risk 4 — elastodynamicsx library maturity (probability: MEDIUM)**
`elastodynamicsx` is a research library from Université Gustave Eiffel. v0.4.0 is the current
release; it is not backed by a major organization with a strong maintenance commitment like
Devito (Imperial/GT-SLIM/DevitoCodes). If the library hits a bug or API break with a DOLFINx
update, the team absorbs the debugging cost. The alternative (building the solver from pure
FEniCSx primitives without elastodynamicsx) removes this dependency but adds development effort.

**Risk 5 — Nektar++ elastic time-domain solver does not exist (probability: CERTAIN)**
Nektar++'s `LinearElasticSystem` is a steady-state solver. A custom time-domain elastic wave
solver in Nektar++ requires C++ implementation of the wave equations within the Nektar++ solver
framework. This is a research-project-scale effort (4–8 weeks) and is not appropriate as a path
to a first PoC within the project timeline. Nektar++ is therefore only relevant as a framework
where someone has already built the solver — and no such seismological elastic solver exists in
the open-source Nektar++ ecosystem.

### DEALBREAKERS — conditions under which the general FEM route would not be chosen

**Dealbreaker 1 (HIGH PROBABILITY): GPU acceleration is unavailable or unreliable.**
Without GPU acceleration, FEniCSx cannot process hundreds of houses within a reasonable compute
budget. The cuDOLFINx + PETSc CUDA stack is research-grade, not a supported production pipeline.
If GPU speedup cannot be achieved within ~3–4 days of engineering effort, the FEM route is not
viable as the primary engine for a batch production campaign. This is the most likely reason to
reject the FEM route — and it is a serious structural weakness.

**Dealbreaker 2 (CERTAIN for Nektar++): No production elastic time-domain solver exists.**
Nektar++ requires building a custom C++ elastic wave solver from scratch. This is a 4–8 week
development effort. It is not appropriate as a primary engine given the other options. Nektar++'s
MIT license and spectral accuracy are attractive, but without an existing solver, it contributes
no practical advantage for this project.

**Dealbreaker 3 (MODERATE for FEniCSx): Build burden exceeds project timeline.**
At 14–24 engineer-days to a CPU-validated PoC (plus GPU integration), the FEM route does not
reach a production-capable state within a timeline competitive with Devito (7–10 days) or
SPECFEM3D (10–16 days). If the project has a 2-week-to-PoC constraint, FEM fails it.

**Dealbreaker 4 (LOW for FEniCSx): Validation tool-only verdict holds for reference use.**
The "validation tool only" framing in doc 07 is still essentially correct as a practical
assessment: FEniCSx is the right choice for running 5–10 carefully constructed reference
simulations to validate Devito's staircased wall transmission (as doc 06 §10 and the Devito
engine doc both recommend). For this role — a handful of high-accuracy reference runs at the
PoC stage — FEniCSx excels: free surface is exact, mesh can resolve the walls properly,
Python scripting is clean, and you do not need GPU acceleration for 5–10 runs. For the
production batch campaign (hundreds of houses, 8 shots each), it is not the right engine
unless GPU acceleration is verified to work.

---

## Summary Table Against Common Rubric

| # | Criterion | FEniCSx / Firedrake | Nektar++ | Notes |
|---|---|---|---|---|
| 1 | Free surface | **STRONG PASS** | **STRONG PASS** | Natural Neumann BC; zero code, < 1% error |
| 2 | Boxy geometry / meshing | **MODERATE PASS** | **MODERATE PASS** | Gmsh Python API; conformal mesh; 30–120 s/house; NekMesh for Nektar++ |
| 3 | Elastic solver readiness | **WEAK PASS** | **FAIL** | FEniCSx: build-your-own, 3–5 weeks. Nektar++: C++ solver doesn't exist |
| 4 | Vertical force + reciprocity | **PASS** | **N/A** | DOLFINx point source + BoundingBoxTree receiver extraction; ~20 lines |
| 5 | 3-component receiver output | **PASS** | **N/A** | Vector FunctionSpace, standard FEM post-processing |
| 6 | Cloud/cluster GPU | **WEAK PASS** | **FAIL** | cuDOLFINx experimental, not production; CPU-only is too slow at scale |
| 7 | Python batch scripting | **STRONG PASS** (FEniCSx) / **WEAK PASS** (Nektar++) | **WEAK PASS** | FEniCSx: Python-native, recomp needed per mesh. Nektar++: C++ core |
| 8 | Validation path | **STRONG PASS** | **N/A** | Free surface automatic; Lamb validation is solver-quality check only |
| 9 | License / commercial use | **PASS** (LGPL, data-gen use) | **STRONG PASS** (MIT) | LGPL fine for SaaS/data gen; MIT ideal. No copyleft trigger for either |
| 10 | Effort / risk / dealbreakers | **WEAK PASS** | **FAIL** | FEniCSx: 14–24 days to CPU PoC; GPU uncertain. Nektar++: no solver exists |

**Comparative column ratings (higher is better):**

| Criterion | Devito (for reference) | FEniCSx | Nektar++ |
|---|---|---|---|
| Free surface | Conditional (build + validate) | **Best** | **Best** |
| Geometry/meshing | **Best** (NumPy) | Middle | Middle |
| Elastic solver | **Best** (exists) | Weak | Fail |
| GPU | **Best** (OpenACC) | Fail | Fail |
| Python scripting | **Best** | Strong | Weak |
| Validation | Good | Good | N/A |
| License | MIT | LGPL (OK) | **Best** (MIT) |
| Effort to PoC | **Best** (7–10 d) | Weak (14–24 d) | Fail |

---

## 6-Line Summary + Honest Verdict

1. **Free surface is the one area where FEM definitively wins** — traction-free BC is the
   variational default, costs zero code, achieves < 1% Rayleigh amplitude error vs Devito's
   ~5%; this is a genuine and important advantage for the Rayleigh-wave-sensitive Green's
   function problem.

2. **No production elastic seismic time-domain solver exists in FEniCSx, Firedrake, or
   Nektar++ — you must build one** — 3–5 engineer-weeks for FEniCSx (the most tractable
   path), 4–8 weeks for Nektar++; doc 07's "validation tool only" verdict remains
   substantially correct for the production campaign.

3. **GPU acceleration is the deal-breaking structural weakness** — cuDOLFINx + PETSc CUDA
   is research-grade (2024 conference demos, not a production release); without GPU,
   CPU-only FEniCSx at this problem scale is ~10–30× slower than Devito on an A100, making
   hundreds-of-houses campaigns infeasible within any reasonable compute budget.

4. **FEniCSx is the right tool for its stated role in this project** — 5–10 validation
   reference runs to characterize Devito's wall-transmission staircasing bias (docs 06, 10);
   for this role GPU is not needed, solver build is a one-time 3-week cost, and free-surface
   accuracy is exactly the property that makes the comparison meaningful.

5. **Licensing is clean for both** — LGPL-3.0 (FEniCSx/Firedrake) and MIT (Nektar++) are
   both commercially usable for a data-generation pipeline on company infrastructure; LGPL
   imposes no obligation when FEniCSx is used as a cloud-hosted data-generation tool (not
   distributed to customers).

6. **Nektar++ should not be in this decision at all for this project** — its elastic
   time-domain solver does not exist, its GPU support is still in development, and its C++
   core makes Python-driven parametric campaigning significantly harder; its only advantage
   (MIT license) is irrelevant given that FEniCSx's LGPL causes no practical friction.

---

## Honest Verdict

**NO for primary engine; CONDITIONAL YES for reference/validation role.**

The general FEM route (FEniCSx/Firedrake) should NOT be chosen as the primary Green's-function
generation engine. The combination of (a) build-your-own solver burden (3–5 weeks on top of
what is already needed), (b) absent production GPU support, and (c) Devito being a faster path
to a running solver means FEM cannot compete for the batch campaign.

The case is **not** that FEM is technically inferior — the free surface accuracy advantage is
real, and the conformal meshing of wall interfaces eliminates the staircasing bias that Devito
accepts. In a world where FEniCSx had a production GPU-accelerated 3-D elastic solver
(analogous to Devito's existing elastic tutorial but on GPU), it would be a strong contender.
That world does not exist yet.

**What FEniCSx should be used for in this project:** exactly what doc 07 recommended — 5–10
reference simulations on carefully meshed house variants (proper wall resolution, conformal
interfaces, exact free surface) to generate a correction baseline for the Devito staircasing
and free-surface amplitude errors. Budget 3 weeks to build the FEniCSx reference solver once;
thereafter it takes ~80 min/run on CPU (acceptable for 5–10 runs) and provides the ground truth
for Devito bias characterization. This is its correct role, and doc 07 was right.

**Nektar++ has no role in this project as described.** If the project's academic publishing
trajectory later demands an independent SEM validation with MIT-licensed code, revisit Nektar++
at that point — but build the solver from scratch only if SPECFEM3D (already validated SEM,
BSD-licensed) is unavailable.

---

## Key References

- **FEniCSx / DOLFINx**: [github.com/FEniCS/dolfinx](https://github.com/FEniCS/dolfinx) —
  v0.8+, LGPL-3.0; [Releases](https://github.com/FEniCS/dolfinx/releases) — active as of 2026
- **elastodynamicsx (MIT)**: [universite-gustave-eiffel.github.io/elastodynamicsx/](https://universite-gustave-eiffel.github.io/elastodynamicsx/) —
  v0.4.0; Newmark time-stepping for elastic waves on FEniCSx
- **FEniCSx elastodynamics Newmark tutorial**: [bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/](https://bleyerj.github.io/comet-fenicsx/tours/dynamics/elastodynamics_newmark/elastodynamics_newmark.html)
- **DOLFINx next-gen paper**: [ResearchGate 2024](https://www.researchgate.net/publication/377241242_DOLFINx_The_next_generation_FEniCS_problem_solving_environment)
- **cuDOLFINx (CUDA extension)**: [github.com/bpachev/cuda-dolfinx](https://github.com/bpachev/cuda-dolfinx) —
  40× assembly speedup demo on GH200; not production-integrated
- **FEniCS GPU conference 2024**: [scientificcomputing.github.io/fenics2024/gpu-acceleration](https://scientificcomputing.github.io/fenics2024/gpu-acceleration) —
  shallow-water GPU demo; elastic wave not demonstrated
- **dolfinx-gpu-solvers**: [github.com/FEniCS/dolfinx-gpu-solvers](https://github.com/FEniCS/dolfinx-gpu-solvers) —
  CUDA/HIP GPU linear solver integration for DOLFINx; "under active development"
- **Firedrake project**: [firedrakeproject.org](https://www.firedrakeproject.org/) —
  LGPL, Python, UFL-based; GPU discussion #4586 notes planned IREE backend (2024–2025)
- **spyro (Firedrake acoustic FWI)**: [gmd.copernicus.org/articles/15/8639/2022/](https://gmd.copernicus.org/articles/15/8639/2022/) —
  acoustic-only; not elastic; published 2022
- **Nektar++**: [nektar.info](https://www.nektar.info/) — v5.9.0 (Nov 2025), MIT license;
  [LinearElasticSystem doxygen](https://doc.nektar.info/doxygen/4.4.0/class_nektar_1_1_linear_elastic_system.html) —
  static elastic solver only, NOT time-domain wave
- **Nektar++ CPC paper (2015)**: [Kirby-132.pdf](https://users.cs.utah.edu/~kirby/Publications/Kirby-132.pdf) —
  framework overview; GPU "under active development"
- **Komatitsch & Tromp (1999)**: "Introduction to the spectral element method for 3D seismic
  wave propagation." GJI 139(3), 806–822. — SEM free-surface accuracy benchmark (< 0.1% at 10 PPW)
- **FEniCSx governance / license page**: [github.com/FEniCS/governance](https://github.com/FEniCS/governance/blob/master/project-license.md)
- **Kristek, Moczo & Archuleta (2002)**: Stress-image free surface for FDTD staggered grid.
  Studia Geophysica et Geodaetica 46, 355–381 — the FDTD free-surface method FEniCSx avoids
- **Lamb analytic code**: [github.com/ktkimit/lamb_2dhalf_surface](https://github.com/ktkimit/lamb_2dhalf_surface) —
  Python, MIT; validation reference for any engine
