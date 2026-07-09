> **SPECFEM3D_Cartesian — end-to-end run + reciprocity + extraction BUILD SPEC.** House-localization pivot.
> Author-verified against the SPECFEM3D readthedocs manual and the `SPECFEM/specfem3d` repo `DATA/Par_file`
> (master, fetched 2026-07-09). Builds on `12_specfem3d_adoption.md` (adoption work-list, GPU numbers),
> `08_free_surface_implementation.md` (Lamb PoC), `11_dataset_design.md` (sensors/source-grid/reciprocity),
> `10_devito_numerics_gpu_viz.md` (source injection + reciprocity bookkeeping cross-check). This is a
> developer BUILD SPEC, not a summary. Indexed in ../README.md.

---

# SPECFEM3D Run + Reciprocity + Extraction Workflow

This document is the implementation authority for taking one parametric house mesh through the
SPECFEM3D_Cartesian pipeline, injecting the reciprocal vertical-force shots, extracting the 3-component
seismogram library, and adapting it into the existing `simgeo_v42` sensor-chain so the confuser + render
+ SNN stack is reused unchanged.

**Scope boundary.** Mesh *creation* (`build_house_mesh` Gmsh script + the 10-file converter) is Stage 2 of
doc 12 and is specified separately (doc 02 of this build series, TBD). This document starts from an existing
external mesh directory `MESH/` containing the SPECFEM3D-format files and ends at a populated `SpecFEMBank`.

---

## 0. Verified facts this spec is built on (primary sources)

All of the following were read directly from the manual / repo on 2026-07-09. Where a value is a *default*
from `DATA/Par_file@master` it is marked (default); we override many of them for our case.

| Fact | Value | Source |
|---|---|---|
| Element type / nodes-per-element | HEX8 (`NGNOD = 8`) or HEX27 | `DATA/Par_file`, manual ch.3 |
| GLL points per direction | `NGLLX = NGLLY = NGLLZ = 5` (polynomial degree 4) — **compile-time constant** in `setup/constants.h.in`, not a Par_file knob | `setup/constants.h.in` |
| Solver workflow | `xdecompose_mesh` → `xgenerate_databases` → `xspecfem3D` | manual ch.3–5 |
| `xdecompose_mesh` CLI | `./bin/xdecompose_mesh nparts input_dir output_dir` | manual ch.3 |
| External-mesh file set | `nodes_coords_file`, `mesh_file`, `materials_file`, `nummaterial_velocity_file`, `free_or_absorbing_surface_file_zmax`, `absorbing_surface_file_{xmin,xmax,ymin,ymax,bottom}` | manual ch.3 |
| `nummaterial_velocity_file` line | `domain_ID material_ID rho vp vs Qkappa Qmu anisotropy_flag`; `domain_ID`=1 acoustic, **2 elastic/viscoelastic** | manual ch.3 |
| Vertical point force | `FORCESOLUTION` file + `USE_FORCE_POINT_SOURCE = .true.` | manual ch.5, `DATA/Par_file` |
| FORCESOLUTION STF codes | 0 Gaussian, 1 Ricker, 2 Heaviside/step, 3 monochromatic, 4 Gaussian(Meschede), 5 Brune, 6 smoothed Brune; `hdurorf0` = half-duration (Gaussian/step) or **dominant f0 (Ricker)** | manual ch.5 |
| Force direction fields | `component dir vect source E`, `... N`, `... Z_up` (**not** required unit vector); `factor force source` in N | manual ch.5 |
| Reciprocity identity | `G_ij(x_A,x_B;t) = G_ji(x_B,x_A;t)` | Wapenaar 2004 (doc 10 §2c) |
| Simultaneous runs | `NUMBER_OF_SIMULTANEOUS_RUNS`; total MPI ranks = `NPROC * NUMBER_OF_SIMULTANEOUS_RUNS`; `run0001/`,`run0002/`… subdirs; `BROADCAST_SAME_MESH_AND_MODEL = .true.` reads mesh/model once and broadcasts | manual ch.7, `DATA/Par_file` |
| Seismogram output flags | `SAVE_SEISMOGRAMS_{DISPLACEMENT,VELOCITY,ACCELERATION,PRESSURE,STRAIN}` (independently settable); `USE_BINARY_FOR_SEISMOGRAMS`; all 3 Cartesian components always written per station | `DATA/Par_file` |
| GPU | `GPU_MODE = .true.` (only GPU knob currently exposed in master `DATA/Par_file`; CUDA/HIP chosen at `./configure`) | `DATA/Par_file` |
| Relevant defaults (we override) | `SIMULATION_TYPE=1`, `SAVE_FORWARD=.false.`, `NPROC=4`, `MODEL=default`, `ATTENUATION=.false.`, `STACEY_ABSORBING_CONDITIONS=.true.`, `PML_CONDITIONS=.false.`, `SUPPRESS_UTM_PROJECTION=.true.`, `USE_FORCE_POINT_SOURCE=.false.`, `USE_RICKER_TIME_FUNCTION=.false.` | `DATA/Par_file@master` |

Two facts flagged as **UNVERIFIED / must-check-in-implementation** (see §7): (a) whether SPECFEM3D writes a
station-side velocity trace that is exactly the impulse `G_zz` given a delta-like STF, or whether the STF is
baked in (it is baked in — see §2); (b) exact `GPU_RUNTIME`/`GPU_PLATFORM` Par_file keys in the version you
compile (master exposes only `GPU_MODE`; some tagged releases add `GPU_RUNTIME`).

---

## 1. The full command pipeline for ONE house

### 1.1 Directory layout (one house = one `RUN/` root)

```
RUN_house0001/
├── DATA/
│   ├── Par_file                       # the settings in §1.3
│   ├── FORCESOLUTION                  # rewritten per shot (§2)
│   └── STATIONS                       # all interior grid points (§3.2)
├── MESH/                              # external mesh (10 files, from Stage-2 converter)
│   ├── nodes_coords_file
│   ├── mesh_file
│   ├── materials_file
│   ├── nummaterial_velocity_file      # §1.4 (materials + Q)
│   ├── free_or_absorbing_surface_file_zmax
│   ├── absorbing_surface_file_xmin  (…xmax, ymin, ymax, bottom)
├── DATABASES_MPI/                     # output of decompose + generate_databases
└── OUTPUT_FILES/                      # solver output: *.sem[dvа], timestamp, etc.
```

### 1.2 Command order (single shot, single GPU MPI-decomposed run)

```bash
# ---- 0. one-time build (Stage 0, doc 12) -------------------------------------
#   ./configure --with-cuda=cuda12    # or --with-hip for AMD
#   make -j all                       # builds xdecompose_mesh, xgenerate_databases, xspecfem3D

NPROC=4                               # = number of MPI partitions = GPU sub-domains

# ---- 1. partition the external mesh for MPI (SCOTCH) -------------------------
./bin/xdecompose_mesh  $NPROC  ./MESH  ./DATABASES_MPI
#   in : MESH/*  (nodes_coords_file, mesh_file, materials_file, nummaterial_velocity_file,
#                 free/absorbing surface files)
#   out: DATABASES_MPI/proc000000_Database … proc000003_Database   (ASCII partition files)
#   NOTE: partitioning is INDEPENDENT of source/receiver → run ONCE per mesh, reuse for all 8 shots.

# ---- 2. build the solver databases (assign GLL, materials, MPI buffers) ------
mpirun -np $NPROC ./bin/xgenerate_databases
#   in : DATA/Par_file, DATABASES_MPI/proc*_Database, MESH/nummaterial_velocity_file
#   out: DATABASES_MPI/proc*_external_mesh.bin, proc*_ibool.bin, … (binary solver DB)
#        + OUTPUT_FILES/output_mesher.txt (min/max element size, resolved frequency, CFL/stability)
#   NOTE: also ONCE per mesh (material change → re-run this step only; geometry change → re-mesh + re-decompose).

# ---- 3. run the solver (one shot) -------------------------------------------
mpirun -np $NPROC ./bin/xspecfem3D
#   in : DATA/Par_file, DATA/FORCESOLUTION, DATA/STATIONS, DATABASES_MPI/*
#   out: OUTPUT_FILES/<net>.<sta>.<chan>.sem[v|d|a]  (one file per station per component)
#        + OUTPUT_FILES/output_solver.txt (timing, stability check)
```

**Reuse rule (critical for cost):** steps 1–2 are source/receiver-independent. For fixed geometry, run
`xdecompose_mesh` + `xgenerate_databases` once, then loop only step 3 across the 8 reciprocal shots
(or batch them with `NUMBER_OF_SIMULTANEOUS_RUNS`, §4). If only materials change between house variants
(same footprint), re-run step 2 only (cheap); if geometry changes, re-mesh + re-decompose.

### 1.3 Key `Par_file` settings for our case

Values below are the **overrides** from the master defaults, chosen for: elastic, GPU, ~250 Hz band,
reciprocal single-force shots, velocity output.

```ini
# --- simulation ---
SIMULATION_TYPE                 = 1          # forward
SAVE_FORWARD                    = .false.    # we are NOT doing adjoint kernels
NPROC                           = 4          # MPI partitions (= GPU subdomains); tune to VRAM
NGNOD                           = 8          # HEX8 external mesh
MODEL                           = default    # material from nummaterial_velocity_file
SUPPRESS_UTM_PROJECTION         = .true.     # Cartesian metres, NOT geographic — MANDATORY for our metre grid

# --- time stepping (see §1.5 CFL) ---
NSTEP                           = 25000      # 0.5 s @ DT=2e-5  (adjust to satisfy CFL from output_mesher.txt)
DT                              = 2.0d-5     # 20 µs; MUST be ≤ CFL limit reported by generate_databases

# --- source ---
USE_FORCE_POINT_SOURCE          = .true.     # read FORCESOLUTION (point force), NOT CMTSOLUTION (moment tensor)
USE_RICKER_TIME_FUNCTION        = .false.    # STF type is set per-source in FORCESOLUTION; see §2

# --- boundaries ---
STACEY_ABSORBING_CONDITIONS     = .true.     # Stacey on side+bottom faces; free surface on zmax by default
PML_CONDITIONS                  = .false.    # keep Stacey for data-gen; PML only for validation (doc 12 Risk 2)
# (top face zmax is a FREE surface automatically when both PML and Stacey are .false. there;
#  with the free_or_absorbing_surface_file_zmax tagged as free surface, zmax stays traction-free)

# --- attenuation (Q) ---
ATTENUATION                     = .true.     # constant-Q SLS (Liu-Anderson-Kanamori); reads Qkappa/Qmu per material
ATTENUATION_f0_REFERENCE        = 50.d0      # centre of our 20-250 Hz band (default is 18 Hz — WRONG for us)
# NOTE: our pyprop8 chain already applies a causal t* at render. Decide ONCE where Q lives:
#   Option A (recommended): ATTENUATION=.false. here → bake NO Q into GF → apply Azimi t* in the adapter
#     (mirrors scenes.py exactly, keeps one Q model across engines). See §5 + §7 gap Q1.
#   Option B: ATTENUATION=.true. here with per-material Qmu → GF already attenuated → adapter skips t*.
#   DO NOT do both (double attenuation). Default this spec: Option A for bridge-parity, Option B for physics runs.

# --- GPU ---
GPU_MODE                        = .true.     # requires ./configure --with-cuda (or --with-hip)

# --- seismogram output ---
SAVE_SEISMOGRAMS_DISPLACEMENT   = .false.
SAVE_SEISMOGRAMS_VELOCITY       = .true.     # we want particle VELOCITY (geophone measures v_z) → *.semv
SAVE_SEISMOGRAMS_ACCELERATION   = .false.
USE_BINARY_FOR_SEISMOGRAMS      = .true.     # binary is faster to parse for thousands of stations
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = 100000     # ≥ NSTEP → write once at the end (avoid mid-run flushes)

# --- simultaneous reciprocal shots (see §4) ---
NUMBER_OF_SIMULTANEOUS_RUNS     = 1          # set to 8 to batch all sensor shots (§4)
BROADCAST_SAME_MESH_AND_MODEL   = .true.     # share the single mesh/model read across the 8 shots
```

### 1.4 `nummaterial_velocity_file` (materials + Q)

Two isotropic elastic materials (soil, concrete). Format `domain_ID material_ID rho vp vs Qkappa Qmu aniso`:

```
2  1  1800.0  400.0  200.0   9999.0   30.0   0
2  2  2400.0  3600.0 2000.0  9999.0  100.0   0
#  ^  ^  rho    vp     vs      Qkappa   Qmu   aniso_flag
#  |  material_ID (matches materials_file element tags)
#  domain_ID: 2 = elastic
```
- Qkappa large (≈9999) = negligible bulk attenuation; Qmu carries the shear/ Rayleigh attenuation (soil Qmu≈20–40, concrete Qmu≈80–150). If **Option A** (§1.3) is chosen, set `ATTENUATION=.false.` and Qmu is ignored by the solver (Q applied later in the adapter).

### 1.5 Time step / CFL / band resolution

- **Band target.** Footstep useful band ≈ 20–250 Hz (doc 11). Resolve to f_max = 250 Hz.
- **Mesh resolution (from doc 12 §5 / doc 10):** dx_soil ≈ 0.11 m at 250 Hz (Vs_soil=200, ~5–8 PPW SEM);
  concrete walls geometry-limited to dx ≤ 0.05 m. `xgenerate_databases` prints the **resolved frequency**
  and **suggested DT** in `output_mesher.txt` — trust that number over hand estimates.
- **CFL / DT.** SEM stable DT is set by the *smallest* element / *fastest* wave: DT ≤ C·dx_min/Vp_concrete.
  For dx_min≈0.05 m, Vp_concrete=3600, GLL-clustered nodes push the effective spacing down ~4× → DT ≈ 1–2×10⁻⁵ s.
  **Do not hardcode DT; read the CFL-safe value from `output_mesher.txt` and set `NSTEP = ceil(0.5 s / DT)`.**
- 100 Hz variant: dx_soil≈0.28 m, DT≈5×10⁻⁵ s, ~10k steps for 0.5 s (doc 10/12 numbers).

---

## 2. Source injection — FORCESOLUTION for a vertical footstep

### 2.1 File layout (verbatim field order)

`DATA/FORCESOLUTION` for a single vertical downward force at a sensor location:

```
FORCE  001
time shift:                  0.0
hdurorf0:                    50.0
latorUTM:                    12.500000
longorUTM:                   9.300000
depth:                       0.0
source time function:        1
factor force source:         1.0
component dir vect source E:  0.0
component dir vect source N:  0.0
component dir vect source Z_up: -1.0
```

Field meaning (manual ch.5, verified):
- `time shift` = 0.0 (mandatory zero for a single source).
- `hdurorf0` = dominant frequency f0 in Hz when STF code = 1 (Ricker); = half-duration in s for code 0/2.
- `latorUTM` / `longorUTM` = **Cartesian X / Y in metres** because `SUPPRESS_UTM_PROJECTION=.true.`
  (the field names say lat/long but with UTM suppressed they are raw X,Y).
- `depth` = depth **in km below the top surface** (manual convention). For a surface-mounted exterior
  geophone, depth = 0.0; wall-mounted sensors sit at their real z (convert metres→km). **VERIFY sign/units:
  §7 gap S1** — some builds interpret `depth` in metres for Cartesian meshes.
- `source time function` = STF code (see §2.2).
- `factor force source` = force amplitude in N (reciprocity is linear → use 1.0 N, scale in post).
- `component dir vect source {E,N,Z_up}` = force direction; **need not be a unit vector**. Vertical
  **downward** footstep force → `Z_up = -1.0` (Z_up positive is up). This is the reciprocal shot direction
  for a vertical geophone (§3).

### 2.2 STF choice — Dirac vs. bake-in (DECISION)

**Chosen approach: shoot a NARROW, KNOWN wavelet, deconvolve to impulse, then convolve our footstep in post.**

SPECFEM3D always convolves the STF into the seismogram — there is no true delta output. Two options:

- **Bake the footstep STF in** (STF = our `wavelet.walk_footfall` Fz). *Rejected*: it destroys the
  linear-kernel property doc 11 §3.3 depends on (one GF library → arbitrary footsteps, confusers, source
  randomization, all by post-hoc convolution). We would have to re-run the solver for every source variant.
- **Shoot a known band-limited wavelet, store the near-impulse GF, convolve later** (*chosen*). Use STF
  code **1 (Ricker)** with `hdurorf0 = 50.0` Hz (f0 well above our 250 Hz? — no: Ricker f0 is the *dominant*
  frequency; pick f0 so the Ricker spectrum is flat-ish across 20–250 Hz → f0 ≈ 60–80 Hz, band to ~250 Hz).
  Then in the adapter **deconvolve the known Ricker spectrum** to recover the impulse `G_zz(ω)`, and convolve
  our `Fz(ω)` footstep. Deconvolution is division by a known analytic spectrum with a small water-level
  regulariser — stable because the Ricker is smooth and we control it exactly.

  *Alternative (simpler, slightly less clean):* use STF code **0 (Gaussian)** with a very short half-duration
  (the manual notes a zero-hdur Gaussian defaults to "five time steps"), treat the stored trace as the GF
  pre-smoothed by a known Gaussian, and deconvolve that Gaussian. Same idea, Gaussian instead of Ricker.

**Net rule for the adapter (§5):** store the raw `*.semv` velocity trace as `g_raw`; the true impulse GF is
`G(ω) = g_raw(ω) / STF_known(ω)` (water-level regularised); the sensor signal is
`irfft( G(ω) · Fz(ω) )`. This preserves the linear-kernel triple-duty of the library.

### 2.3 One FORCESOLUTION per shot

Python writes `DATA/FORCESOLUTION` before each solver call, substituting the sensor X,Y,Z and the fixed
STF block. For 3-component sensors (§3.4) the only change per force direction is the
`component dir vect source {E,N,Z_up}` triplet: (1,0,0), (0,1,0), (0,0,-1).

---

## 3. Reciprocity bookkeeping — precise

### 3.1 The identity we exploit

Elastodynamic reciprocity (Wapenaar 2004; doc 10 §2c):

```
G_ij(x_A, x_B; t) = G_ji(x_B, x_A; t)
```
`G_ij` = i-th velocity component at x_A from a unit j-direction force at x_B.

**Forward problem we ultimately want** (never simulated directly): response of vertical geophone at sensor
`s` to a vertical footstep force at floor grid point `p`:
```
response(s,p; t) = G_zz(x_s, x_p; t)  ⊛  Fz_footstep(t)
```
**Reciprocal problem we actually simulate** (shoot at the few sensors, record at the many floor points):
```
G_zz(x_s, x_p; t) = G_zz(x_p, x_s; t)      ← reciprocity
```
So: **inject a vertical force (Z_up = −1) at sensor `s`; record v_z at all interior grid points `p`.**
One solver run per sensor yields the entire column `G_zz(x_s, · )` over all `p` simultaneously. This turns
~hundreds–thousands of forward source positions into **N_sensor solver runs** (6–8), the dominant cost
reduction in the pipeline (doc 11 §2.2: ~250×).

### 3.2 STATIONS lists the interior receiver grid

The interior floor grid points become **receivers** (STATIONS), the sensors become **sources**
(FORCESOLUTION). `DATA/STATIONS`, one line per point:

```
#  STA     NET     Y(lat)        X(lon)         elevation(m)  burial(m)
P00001    HL      9.300000      12.500000      0.0            0.0
P00002    HL      9.550000      12.500000      0.0            0.0
...
```
- Columns (manual ch.5): `station_name  network  latitude  longitude  elevation  burial`.
- With `SUPPRESS_UTM_PROJECTION=.true.`, latitude=Y(m), longitude=X(m). **Column order is (Y, X)** — a common
  bug; verify against the shipped example STATIONS (§6). `burial` = depth below surface in metres; floor
  slab surface → 0.0, or the slab z if the floor is elevated.
- N_stations = number of interior grid points per floor (doc 11: ~1,500 at 0.25 m; doc 12 used ~480 at 0.5 m).
  Solver cost is ~constant in N_stations (receiver interpolation is cheap vs. the volume update), so use the
  dense grid. Generate `STATIONS` with a ~15-line Python loop over the walkable interior (wall-exclusion
  margin 0.3 m, and exclude points within 0.5 m of any sensor — doc 11 §6.3 Risk 6).

### 3.3 Vertical-geophone case (baseline): 1 shot per sensor

Only `G_zz` is needed. **N_sensor shots total** (6 per doc 11, 8 per doc 12), each: FORCESOLUTION with
`Z_up=−1` at the sensor, record `*.semv` at all STATIONS, keep the **Z component** (`.HXZ.semv`).
For N_sensor=8 → 8 solver runs per house.

### 3.4 3-component (ADXL355) sensors: 3 shots per sensor

A 3-axis sensor at `s` measures (v_x, v_y, v_z) from a vertical footstep at `p`, i.e. it needs
`G_xz`, `G_yz`, `G_zz` (i = x,y,z from j = z at the floor). By reciprocity `G_iz(x_s,x_p) = G_zi(x_p,x_s)`,
so we shoot **force direction i at the sensor** and record **v_z at the floor**:

| Sensor comp wanted | Forward `G_{i z}(s,p)` | Reciprocal shot at sensor `s` | Record at floor `p` |
|---|---|---|---|
| v_x from footstep | `G_xz` | force in **X** (E=1) | v_z |
| v_y from footstep | `G_yz` | force in **Y** (N=1) | v_z |
| v_z from footstep | `G_zz` | force in **Z** (Z_up=−1) | v_z |

So the recorded floor component is **always v_z**; the force direction rotates. → **3 shots per 3-C sensor**.
For a mixed array (geophone + ADXL355 co-located), the geophone reuses the sensor's Z-shot; a node needs
max 3 shots. Doc 12 Risk 5 confirms this: 3 force directions × N_sensor. For N_sensor=8, 3-C everywhere → 24 shots.
**Baseline campaign is vertical-only (8 shots); 3-C is a config flag that triples shot count** and is batched
identically with `NUMBER_OF_SIMULTANEOUS_RUNS` (§4).

Verified against the manual: FORCESOLUTION direction vector controls the force axis; STATIONS always records
all 3 components regardless, so we simply keep the Z column of the `.semv` output. Nothing about this is
uncertain except the depth-field units (§7 S1).

---

## 4. Batching + compute

### 4.1 `NUMBER_OF_SIMULTANEOUS_RUNS` to batch the reciprocal shots

Mechanism (manual ch.7, verified): setting `NUMBER_OF_SIMULTANEOUS_RUNS = R` launches R
embarrassingly-parallel forward runs inside one `mpirun`, each with its own `DATA/`+`OUTPUT_FILES/`
under `run0001/ … run000R/`. Total MPI ranks = **`NPROC * R`**. With
`BROADCAST_SAME_MESH_AND_MODEL = .true.` the mesh+model is read **once** and broadcast to all R runs —
the exact reciprocity use case (same house, only the source/receivers differ per shot).

Layout for R = 8 sensor shots (vertical-only):
```
RUN_house0001/
├── DATA/Par_file                 # NUMBER_OF_SIMULTANEOUS_RUNS=8, BROADCAST_SAME_MESH_AND_MODEL=.true.
├── run0001/DATA/{FORCESOLUTION,STATIONS}   # sensor 1 vertical shot
├── run0002/DATA/{FORCESOLUTION,STATIONS}   # sensor 2 …
│   …
├── run0008/DATA/{FORCESOLUTION,STATIONS}   # sensor 8
├── DATABASES_MPI/                # shared (read once, broadcast)
└── run0001/OUTPUT_FILES/ … run0008/OUTPUT_FILES/
```
Launch: `mpirun -np $((NPROC*8)) ./bin/xspecfem3D`. STATIONS is identical across runs; only FORCESOLUTION
differs. For 3-C (24 shots) set R=24 if the cluster has the ranks, else run in waves of R=8.

**Cross-check with the worker cap (user memory):** total concurrent heavy workers = `NPROC * R`. Keep this
consistent with the ≤8-heavy-worker rule on shared local hardware; on a dedicated cloud A100 node the cap
is the GPU/VRAM budget, not the 8-worker rule.

### 4.2 Partitioning for GPU

`NPROC` = number of MPI partitions = number of GPU subdomains. On a single A100, `NPROC=1`
(whole mesh on one GPU) is simplest if the mesh fits VRAM; use `NPROC=2–4` only if the mesh exceeds VRAM
or to overlap. With `NUMBER_OF_SIMULTANEOUS_RUNS=R`, total ranks `NPROC*R` must map onto available GPUs
(1 rank ↔ 1 GPU is the clean mapping; MPS allows over-subscription). For an 8×A100 node: `NPROC=1, R=8`
→ all 8 sensor shots run concurrently, one GPU each, one mesh read. This is the recommended production shape.

### 4.3 Realistic per-house wall-clock (cross-checked vs docs 10/12)

Anchored on doc 10's validated Devito 100 Hz = 7.7 s/sim and doc 12's estimate that SEM has ~2–3× fewer
effective timesteps, plus the f⁴ scaling (doc 10 §4) from 100→250 Hz (≈39×). All ±50% (no published
house-scale SPECFEM3D benchmark exists — doc 12 §4c).

| Scenario | Per shot (A100) | Per house | 50 houses | 200 houses |
|---|---|---|---|---|
| 100 Hz, single A100, sequential shots | ~5–15 s | ~8 shots → 40–120 s | ~0.5–1.6 hr | ~2–6.5 hr |
| 100 Hz, `R=8` on 8×A100 (1 mesh read) | — | ~30–80 s (concurrent) | ~25–70 min | ~1.6–4.5 hr |
| 250 Hz, single A100, sequential | ~3–10 min | ~25–80 min | ~21–65 hr | (use 8×A100) |
| 250 Hz, `R=8` on 8×A100 | — | ~3–10 min | ~2.5–8 hr | ~10–33 hr |

Cloud cost (doc 12 §4d): 200 houses @100 Hz ≈ $18–25; @250 Hz on 8×A100 spot ≈ tens–low-hundreds of $.

### 4.4 3-component GF-library HDF5 layout

Mirror doc 11 §5 storage. One HDF5 per house:

```
house0001.h5
├── /gf            float32  [N_sensor, N_force_dir, N_station, N_comp, N_time]
│                            e.g. [8, 3, 1500, 3, 1000]   (3-C); vertical-only → [8,1,1500,1,1000]
│                            axis meaning: (sensor s, force dir j, floor point p, recorded comp i, time)
│                            for the geophone product we index [s, 2(Z-force), p, 2(v_z), :] = G_zz
├── /station_xyz   float32  [N_station, 3]     (X,Y,Z of each floor grid point, metres)
├── /sensor_xyz    float32  [N_sensor, 3]
├── /t             float32  [N_time]           (time axis at stored SPS)
├── attrs: dt_stored, fs_stored, stf_type, stf_f0, house_params(json),
│          q_applied(bool), specfem_version, par_file_hash
```
- Store at the render SPS (1000 Hz, matching `simgeo_v42`) after resampling the solver's fine-DT trace;
  or store at solver DT and resample in the adapter. Recommend **store at 2000 SPS** (250 Hz Nyquist headroom,
  doc 11) then let the adapter resample to the 1000 Hz render grid, exactly as `scenes.py:Bank` upsamples
  the 250 Hz pyprop8 bank today.
- gzip level 4 (doc 11): ~36 MB/house (1-C) to ~108 MB/house (3-C) full time-trace; FFT-replica form ~4.6 MB.

---

## 5. The adapter — `SpecFEMBank` (drop-in for `simgeo_v42` `Bank`)

### 5.1 The interface it must match (read from `simgeo_v42/scenes.py`)

The existing pyprop8 `Bank` (scenes.py:35) exposes:
```python
class Bank:
    def __init__(self, profile_id): ...
    @classmethod
    def get(cls, profile_id): ...
    def emit(self, r_m, heading_xy, src_xy, fz, fx) -> np.ndarray   # ground velocity m/s, length self.n, dt=1e-3
```
`emit` returns a **ground-velocity wavelet at the sensor** (m/s, 1000 Hz, length `self.n`), which the
sensor chain (`sensor.py`: coupling → geophone H(iω) → preamp → ADC) then renders to counts. The confuser
pipeline (doc 11 §4) also calls `emit` with a confuser STF at a floor position. **We must reproduce this
exact contract** so nothing downstream changes.

### 5.2 The mismatch and how the adapter bridges it

| | pyprop8 `Bank` | SPECFEM3D `SpecFEMBank` |
|---|---|---|
| Geometry key | source–receiver **radius** `r_m` (1-D, azimuthally symmetric) | discrete **(sensor s, floor point p)** pair (full 3-D, no symmetry) |
| Stored quantity | impulse-velocity spectrum per radius, Q + plate baked in | raw `*.semv` per (s, p), STF known, Q optional (§1.3) |
| `heading_xy/src_xy/fx` | azimuth factor for horizontal force (§scenes.py:111) | **not needed for vertical geophone**; for 3-C, the x/y force GFs are stored explicitly (no azimuth trick) |

Because SPECFEM3D already encodes the true 3-D geometry, **the azimuth/radius machinery collapses**: there
is a distinct GF per (sensor, floor point), so `emit` is a table lookup by floor-point index, not a radius
search. The adapter therefore takes a **floor-point index** (or XY that it snaps to the nearest grid point),
not `r_m`. To keep the *call sites* identical we keep the signature and reinterpret:

```python
class SpecFEMBank:
    """Drop-in for simgeo_v42.scenes.Bank, backed by a SPECFEM3D per-house HDF5 GF library.
    emit() returns ground velocity (m/s) at a sensor for a footstep at a floor point, length self.n, dt=1e-3.
    """
    _cache = {}
    def __init__(self, h5_path, sensor_k):
        import h5py
        f = h5py.File(h5_path, "r")
        self.sensor_k = sensor_k
        self.station_xyz = f["station_xyz"][:]                 # (P,3)
        self.dt = 1e-3; self.n = <render length>               # match scenes.FS grid
        fs_stored = f.attrs["fs_stored"]
        # raw v_z GF for this sensor's vertical shot: g_raw[p, t] = G_zz ⊛ STF_known
        g_raw = f["/gf"][sensor_k, IDX_ZFORCE, :, IDX_VZ, :]   # (P, T_stored)
        # 1) resample to 1000 Hz render grid (spectrum zero-pad, exact for band-limited — like scenes.Bank)
        # 2) deconvolve the KNOWN STF (Ricker/Gaussian, water-level) → true impulse G_zz(ω) per floor point
        G = rfft(resample(g_raw)) / (STF_known(ω) + wl*max|STF_known|)
        # 3) OPTION A parity: apply causal Azimi t* here (reuse scenes.settle logic) if q_applied==False
        if not f.attrs["q_applied"]:
            G *= azimi_tstar(ω, r=|station-sensor|, q=..., vr=...)   # same formula as scenes.Bank
        self.Gz = G                                            # (P, n//2+1) impulse-velocity spectra, m/s per N
        f.close()

    @classmethod
    def get(cls, h5_path, sensor_k):
        key=(h5_path,sensor_k)
        if key not in cls._cache:
            if len(cls._cache)>12: cls._cache.clear()
            cls._cache[key]=cls(h5_path,sensor_k)
        return cls._cache[key]

    def emit(self, floor_xy, fz, fx=None, heading_xy=None, src_xy=None):
        """Ground-velocity wavelet (m/s, length self.n) at this sensor for footstep fz at floor_xy.
        Signature kept call-compatible with scenes.Bank.emit; r_m→floor_xy (snapped to nearest grid point).
        heading_xy/src_xy/fx are accepted for 3-C but ignored for the vertical-geophone product."""
        p = nearest_index(self.station_xyz[:, :2], floor_xy)   # snap to floor grid
        out = self.Gz[p] * np.fft.rfft(fz, self.n)             # convolve footstep Fz(ω) with impulse G_zz
        # 3-C extension: add self.Gx[p]*Fx + self.Gy[p]*Fy from the x/y-force GFs (stored explicitly)
        return np.fft.irfft(out, n=self.n) * self.dt
```

Key parity points with `scenes.Bank`:
- Returns **m/s at the sensor**, same units/length/dt → `sensor.py` chain unchanged.
- Same `.get()` LRU-cache pattern and 12-entry cap.
- Same **spectrum zero-pad upsample** trick used by `scenes.Bank.__init__` (2000→1000/ render grid).
- Q lives in exactly one place (§1.3 Option A/B); the Azimi t* helper is lifted verbatim from `scenes.Bank`
  so the soil-only case is bit-comparable in the bridge test (§6.2).

### 5.3 Call-site change

`generate_corpus*.py` / `scenes.py` construct a `Bank.get(profile_id)`; they instead construct
`SpecFEMBank.get(house_h5, sensor_k)` per sensor and call `emit(floor_xy, fz)`. Because the geometry is now
per-(sensor,point) the caller passes the **footstep XY** (which it already knows — it is the label) instead
of `r_m`. Everything after `emit` (superposition over a walk path, confuser ⊛ GF, sensor chain, windows,
labels) is unchanged. Confusers reuse the same GF library at a fixed floor point (doc 11 §4.2) — zero extra sims.

---

## 6. Validation

### 6.1 Gate 1 (MANDATORY FIRST): shipped homogeneous-halfspace example = Lamb PoC

Before any house geometry, prove the SPECFEM3D build + source + receiver + extraction chain on the
shipped example. The repo ships `EXAMPLES/applications/homogeneous_halfspace/` (elastic halfspace, the
canonical Lamb-problem setup; note the historical path `EXAMPLES/homogeneous_halfspace_HEX*_elastic_*` in
older tags).

```bash
cd EXAMPLES/applications/homogeneous_halfspace
./run_this_example.sh          # decompose → generate_databases → xspecfem3D (CPU by default)
# then flip GPU_MODE=.true. in DATA/Par_file and re-run to prove the GPU path
```
**Hello-world success = this example runs to completion on GPU and writes non-trivial `*.semv`/`*.semd`
seismograms at its stations, with `output_solver.txt` reporting a stable run (no NaN, CFL OK).** This is the
first-run milestone (see summary).

Then upgrade it to the Lamb quantitative gate (doc 08 §5): set the halfspace to soil params
(Vp=520, Vs=300, ρ=2000, ν=0.25), vertical `FORCESOLUTION` (Z_up=−1) at the surface, receivers at
r = 5,10,15,20,25 m, and check:
- P arrival within ±1% of r/Vp,
- Rayleigh arrival within ±1% of r/(0.9194·Vs),
- surface |v_z/v_r| within ±10% of 1.47 (Rayleigh eigenfunction ratio, ν=0.25).
SEM should pass these comfortably (free surface is exact in SEM — no stress-image burden, unlike Devito
doc 08). Any failure here means the build/source/units are wrong, not the physics.

### 6.2 Gate 2: bridge check vs pyprop8 on a soil-only case

Doc 12 §1c procedure. Pick a `profiles.py` soil profile (e.g. Vs=200, Vp=400, ρ=1800, Q=30), run pyprop8
(`gfbank_build.py`) → z-component GF at r = 2,5,10,20 m; build the same layered halfspace in SPECFEM3D
(soil only, **no concrete**), vertical FORCESOLUTION, matched receiver offsets, extract via the adapter.
Pass: P-arrival <0.5%, Rayleigh arrival <0.5%, peak v_z amplitude ratio <5%. Use **Option A** (Q in the
adapter) so both codes apply the *same* Azimi t* and the comparison isolates the wave physics, not the
attenuation model. This simultaneously validates the `SpecFEMBank` adapter end-to-end (it is the code path
that produces the SPECFEM3D trace being compared).

### 6.3 Gate 3 (later): add concrete, sanity-check vs FEniCSx shell-FEM (doc 12 Risk 3) — out of scope here.

---

## 7. HONEST gaps — must be verified in implementation (do not guess)

- **Q1 — Where Q lives (Option A vs B).** This spec recommends Option A (Q off in solver, Azimi t* in adapter)
  for bridge-parity and one-Q-model consistency, and Option B (constant-Q SLS in solver) for
  physics-fidelity house runs. Which one the publication uses must be decided and stated; **never both**
  (double attenuation). The SLS constant-Q band and `ATTENUATION_f0_REFERENCE` centring at ~50 Hz for a
  20–250 Hz band needs a convergence check (SPECFEM's SLS fits Q over a frequency band around f0_REFERENCE;
  verify the fit is flat enough across our band).
- **S1 — FORCESOLUTION `depth` and STATIONS `burial` units for a Cartesian mesh.** Manual states depth in km
  (geographic convention). With `SUPPRESS_UTM_PROJECTION=.true.` and a metre-scale Cartesian mesh, confirm
  whether `depth`/`burial` are metres or km in your compiled version — a km/m error is a 1000× vertical
  misplacement. Verify on the shipped example (its FORCESOLUTION/STATIONS values vs its known geometry).
- **S2 — STATIONS column order (Y before X).** With UTM suppressed, latitude=Y, longitude=X → the file
  order is (Y, X). Confirm against the shipped example STATIONS before generating 1,500-line files.
- **G1 — GPU Par_file keys.** master `DATA/Par_file` exposes only `GPU_MODE`. Some tagged releases add
  `GPU_RUNTIME`/`GPU_PLATFORM`. Confirm the exact keys in the version you `./configure` and whether CUDA vs
  HIP is a build-time (`./configure --with-cuda`) or runtime choice.
- **G2 — GLL is compile-time.** `NGLLX=5` (degree 4) is fixed in `setup/constants.h.in`, not a Par_file knob.
  If a different order is ever needed it requires a recompile. Assume degree 4 (the validated default).
- **STF1 — Deconvolution stability.** The "store near-impulse GF, deconvolve known STF" plan (§2.2) relies on
  the known Ricker/Gaussian spectrum being non-zero across 20–250 Hz. Verify the chosen f0 keeps the STF
  spectrum well above the water-level floor across the whole band; otherwise high-frequency GF content is
  amplified noise. Validate by round-tripping (convolve STF, deconvolve, compare to a direct short-Gaussian run).
- **B1 — Stacey reflection into late Rayleigh tail.** doc 12 recommends Stacey (not PML) for data-gen; verify
  the 5 m soil buffer keeps spurious reflections <5% of the Rayleigh peak in the post-Rayleigh window at 250 Hz
  (check on the Lamb gate). If not, thicken the buffer or switch that face to PML for validation runs only.
- **M1 — External-mesh surface-file free-vs-absorbing tagging.** The top face must be tagged as **free**
  (traction-free) in `free_or_absorbing_surface_file_zmax` while sides/bottom are absorbing. Confirm the
  Stage-2 converter tags zmax as free, and that `STACEY_ABSORBING_CONDITIONS=.true.` does not accidentally
  make zmax absorbing. (Free surface is the default on faces not listed as absorbing, per the Par_file note.)
- **R1 — Reciprocity amplitude/units.** With `factor force source = 1.0` N the stored trace is v/N. Confirm
  the reciprocal G_zz has the right units to convolve directly with `wavelet.walk_footfall` Fz (Newtons) and
  land in m/s at the sensor, matching `scenes.Bank` scale (which uses m/s per N). A one-off amplitude check
  against pyprop8 at one offset (Gate 2) settles this.

---

## 8. First-run milestone ("hello-world" success)

Run `EXAMPLES/applications/homogeneous_halfspace/run_this_example.sh` end-to-end
(`xdecompose_mesh` → `xgenerate_databases` → `xspecfem3D`) **on the GPU** (`GPU_MODE=.true.`), and confirm:
(1) `output_mesher.txt` reports a resolved frequency and CFL-stable DT; (2) `output_solver.txt` finishes
with no NaN and a passing stability check; (3) `OUTPUT_FILES/*.semv` velocity seismograms are written for
the example's stations with a physically sane vertical-force waveform (visible P + Rayleigh arrivals).
Parsing one `*.semv` into NumPy and plotting a v_z trace = the extraction path works. That is the gate to
begin the Lamb quantitative check (§6.1) and then the pyprop8 bridge (§6.2).
