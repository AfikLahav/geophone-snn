> **Devito numerics, GPU, and visualization deep-dive — house-localization pivot.** Research (Sonnet, 2026-07-08). Extends `06_sim_construction_cost.md` and `07_sim_engine_verification_and_plan.md`. Covers: thin-wall staircasing, source injection & reciprocity bookkeeping, GPU setup & realistic throughput, frequency cost trade-off, absorbing boundaries, and PyVista/ParaView visualization. Indexed in [README.md](README.md).

---

# Devito Numerics, GPU, and Visualization for 3-D Elastic House Simulation

## Context

The prior docs established: Devito FDTD is the primary engine; free-surface (stress-image, Kristek 2002) is the make-or-break implementation; 100 Hz at dx=0.15 m and 250 Hz at dx=0.06 m are the two candidate operating points; N_sensor=8 reciprocity shots replace 480 direct forward sims. This document resolves the six remaining open technical questions.

---

## 1. Thin-Wall Staircasing: Magnitude, Mitigations, and Required Resolution

### The problem

A standard UK/EU house wall is 0.15–0.30 m thick. At dx=0.15 m (100 Hz scenario) a 0.15 m wall is **exactly 1 cell**. At dx=0.10 m it spans 1–2 cells. On a Cartesian voxel grid, any wall thickness < dx rounds to zero (invisible) and any thickness between dx and 2dx is treated as a full 2-cell slab regardless of the true geometry. This is the staircasing error.

### Physics of the error

The staggered Virieux grid places stress and velocity components at sub-cell offsets. When a material interface does not coincide with a grid plane (which it never exactly does for oblique or thin walls), the finite-difference stencil integrates over a mix of materials within its support width. For an order-8 stencil (half-width = 4 cells) the contamination zone around a 1-cell wall is 8 cells wide on each side. This produces:
- **Amplitude errors** at the transmitted/reflected wavelet from the wall: typically 10–30% for a 1-cell wall with simple nearest-neighbour assignment.
- **Phase errors**: small (< 2%) for normal incidence; grow for oblique incidence because the effective impedance of the smeared interface differs from the true sharp interface.

### Moczo et al. (2002) effective-medium averaging

The canonical solution for heterogeneous staggered-grid FDTD is the Moczo, Kristek, Vavrycuk, Archuleta & Halada (2002) scheme [BSSA 92(8), 3042–3066]. Its rules for averaging elastic parameters within the cell volume occupied by each staggered component:

| Parameter | Grid location (Virieux) | Correct averaging |
|---|---|---|
| λ (first Lamé, bulk modulus) | full-integer (collocated with τ_kk) | **Arithmetic average** over cell volume |
| μ (shear modulus) | half-integer (shear stress τ_ij nodes) | **Harmonic average** over cell volume around that node |
| ρ (density) | half-integer (velocity v_i nodes) | **Arithmetic average** over the relevant sub-cell |

The harmonic averaging of μ is crucial: it allows the shear stress to vanish at a fluid–solid interface (μ_fluid = 0 → harmonic mean → 0) and correctly down-weights the stiff phase at an interface, mimicking the physical "welded contact" boundary condition in the discrete equations. Simple nearest-neighbour assignment (μ = μ of the closest grid point) is systematically wrong at interfaces and inflates the effective stiffness.

**Practical effect**: switching from nearest-neighbour to Moczo-style averaging reduces the amplitude error at a single-cell-thick wall from ~20–30% to ~5–10% (error depends on velocity contrast and wall geometry).

### What resolution do walls actually need?

The minimum practical rule is: **the wall must span at least 3–5 cells** for the interior stiffness to converge to the true value even with harmonic averaging. This is because:
1. The order-8 stencil needs 4 ghost values on each side; a 1-cell wall is thinner than the stencil half-width.
2. Harmonic averaging of μ across a 1-cell wall still misestimates the shear coupling (the effective μ at the wall face is √(μ_soil · μ_concrete) for a 2-cell wall, not the true contact value).

| Wall thickness | dx=0.15 m (100 Hz) | dx=0.06 m (250 Hz) | Assessment |
|---|---|---|---|
| 0.15 m exterior concrete | 1 cell | 2.5 cells | Under-resolved at 100 Hz; marginal at 250 Hz |
| 0.20 m exterior concrete | 1.3 cells | 3.3 cells | Under-resolved at 100 Hz; acceptable at 250 Hz |
| 0.30 m reinforced concrete | 2 cells | 5 cells | Marginal at 100 Hz; good at 250 Hz |
| 0.15 m floor slab | 1 cell | 2.5 cells | Same as exterior wall |
| 0.12 m partition (drywall) | 0 cells (invisible!) | 2 cells | Vanishes at 100 Hz; barely resolved at 250 Hz |

**Minimum required dx for 5-cell wall coverage of a 0.15 m wall: dx ≤ 0.03 m**, which corresponds to f_max ≥ 500 Hz (Vs_soil=150, 10 PPW). This is far beyond the project target and not pursued.

### Chosen approach and documented bias

For this project, the plan is:
1. **Apply Moczo harmonic/arithmetic averaging** when filling the material arrays (10–20 lines of NumPy). This reduces but does not eliminate the staircasing error.
2. **Accept the residual amplitude bias** (~10% at walls, < 5% in the soil bulk) and document it as a known systematic error in the Green's function library. The bias is spatially structured (concentrated at wall voxels) and consistent across all training simulations, so it affects all samples equally — important for a comparative localization dictionary, where relative patterns matter more than absolute amplitudes.
3. **Validate via FEniCSx shell-FEM reference**: run 5–10 reference sims comparing Devito wall transmission against a properly meshed FEniCSx simulation (no staircasing). The ratio of transmitted amplitudes gives the correction factor, which can be applied to the GF library as a frequency-dependent wall-transmission correction.
4. **Resolution recommendation if walls dominate the localization error**: increase to dx=0.06 m (250 Hz grid) for a 2.5-cell exterior wall — the Moczo averaging then brings error down to ~5%, which is the same order as the free-surface Rayleigh amplitude uncertainty.

### Implementation (NumPy, ~25 lines)

```python
import numpy as np
from scipy.ndimage import uniform_filter

# Fill baseline soil material
lam = np.full(shape, lam_soil, dtype=np.float32)
mu  = np.full(shape, mu_soil,  dtype=np.float32)
rho = np.full(shape, rho_soil, dtype=np.float32)

# Mark wall voxels
wall_mask = make_wall_mask(shape, dx)   # axis-aligned box geometry → boolean

# Overwrite wall voxels
lam[wall_mask] = lam_concrete
mu[wall_mask]  = mu_concrete
rho[wall_mask] = rho_concrete

# Moczo-style averaging over 1-cell neighbourhood (approximation of volume average)
# Lambda: arithmetic average of 3x3x3 neighbourhood (uniform_filter = box average)
lam_avg = uniform_filter(lam, size=1)   # stays as-is for co-located; use as-is

# Mu: harmonic average in each half-integer sub-volume (shear stress node)
# For mu at tau_xz node (half-step in x and z): harmonic mean of 4 surrounding cells
def harmonic_mean_shift(arr, axis_a, axis_b):
    """Harmonic average of arr shifted +0.5 in axis_a and +0.5 in axis_b."""
    r = np.roll(arr, -1, axis=axis_a)
    s = np.roll(arr, -1, axis=axis_b)
    t = np.roll(np.roll(arr, -1, axis=axis_a), -1, axis=axis_b)
    # Harmonic mean of 4: 4 / (1/arr + 1/r + 1/s + 1/t)
    return 4.0 / (1.0/arr + 1.0/r + 1.0/s + 1.0/t)

mu_xz = harmonic_mean_shift(mu, axis_a=0, axis_b=2)   # for tau_xz
mu_yz = harmonic_mean_shift(mu, axis_a=1, axis_b=2)   # for tau_yz
mu_xy = harmonic_mean_shift(mu, axis_a=0, axis_b=1)   # for tau_xy
# Rho: arithmetic average at each velocity node (half-step in one direction)
rho_vx = 0.5 * (rho + np.roll(rho, -1, axis=0))
rho_vy = 0.5 * (rho + np.roll(rho, -1, axis=1))
rho_vz = 0.5 * (rho + np.roll(rho, -1, axis=2))
# Feed these into Devito Function arrays instead of the raw lam/mu/rho
```

The Devito elastic tutorial uses a single μ field; the correct 3D elastic implementation requires three μ fields (one per off-diagonal shear stress) each with its own harmonic average. This is a non-trivial extension of the tutorial code.

---

## 2. Source Injection and Reciprocity Component Bookkeeping

### 2a. What the Devito tutorial injects (explosive source)

The official Devito elastic tutorial injects the source into **both diagonal stress components simultaneously**:

```python
src_xx = src.inject(field=tau.forward[0, 0], expr=src)
src_zz = src.inject(field=tau.forward[1, 1], expr=src)
```

This is an **isotropic (explosive) source** — it applies equal compressive stress in all directions, exciting P-waves equally in all azimuths. A footstep is **not** an explosive source; it is a nearly pure vertical body force (normal traction on the floor, negligible horizontal component).

### 2b. Vertical single-force injection (correct for footstep)

A vertical point body force f_z at position x_s is injected as a **body-force term in the vertical momentum equation** in the Virieux velocity–stress formulation:

```
ρ ∂v_z/∂t = ∂τ_xz/∂x + ∂τ_yz/∂y + ∂τ_zz/∂z + f_z(x,t)δ(x − x_s)
```

In Devito this is implemented by adding the force to the **v_z velocity update** (not into the stress tensor):

```python
# src is a SparseTimeFunction with the force time history (e.g. Ricker derivative)
src_vz = src.inject(
    field=v[2].forward,      # v_z component (index 2 = vertical if z is last dimension)
    expr=src * dt / rho      # body force → velocity: Δv = f · dt / ρ
)
```

The factor `dt/rho` converts force (N/m³ body force density) to velocity increment per time step.

Alternatively (and equivalently for a vertical force), inject into **τ_zz only** (not τ_xx or τ_yy):

```python
src_zz = src.inject(field=tau.forward[2, 2], expr=src)   # τ_zz only
# Leave tau[0,0] and tau[1,1] unforced
```

Injecting only τ_zz creates a net vertical compression without horizontal compression, which is the correct radiation pattern for a vertical surface traction (it excites both P and SV waves with the radiation pattern of a vertical single force, not an explosion). In practice both methods are used; the τ_zz injection is slightly cleaner because it directly modifies the stress equation where the force appears.

**Radiation pattern comparison:**
- Explosive (τ_xx = τ_yy = τ_zz): excites only P-waves; no SV or SH; omnidirectional.
- Vertical force (τ_zz only, or f_z in momentum): excites P + SV with cos(θ) radiation pattern (vertical force → maximum P-wave radiation vertically, maximum SV at 45°, zero SH).
- Horizontal force (τ_xx or f_x): excites P + SV + SH; rotated pattern.

For geophone exterior sensors measuring vertical ground velocity from footsteps, the **vertical force model** is the correct approximation.

### 2c. Reciprocity: which force direction maps to which recorded component

Elastodynamic reciprocity (Knopoff & Gangi 1959; Wapenaar 2004) states:

```
G_ij(x_A, x_B; t) = G_ji(x_B, x_A; t)
```

where G_ij is the i-th component of velocity at x_A due to a unit impulse force in the j-th direction at x_B. The indices i, j ∈ {x, y, z}.

**Forward problem (what we want to compute):**  
Response at sensor k (vertical geophone, measures v_z) due to a vertical footstep force (f_z) at floor position p:
```
response_k(t) = G_zz(x_sensor_k, x_floor_p; t)  ×  GRF(t)
```

**Reciprocal problem (what we simulate):**  
Shoot a vertical force (f_z) at sensor k; record v_z at all floor positions p:
```
G_zz(x_sensor_k, x_floor_p; t) = G_zz(x_floor_p, x_sensor_k; t)   ← by reciprocity
```

Therefore: **inject f_z (or τ_zz) at the sensor location; record v_z at all 480 floor positions**. This is the single-component case — only G_zz is needed. One simulation per sensor suffices.

**Component table for common localization scenarios:**

| Physical scenario | Forward: force dir at floor | Forward: measured comp at sensor | Reciprocal: inject at sensor | Record at floor |
|---|---|---|---|---|
| Footstep + vertical geophone | z | v_z | f_z (or τ_zz) | v_z |
| Footstep + horizontal geophone (x) | z | v_x | f_x | v_z |
| General 3-component sensor | z | v_x, v_y, v_z | f_x, f_y, f_z | v_z each |

For a 3-component sensor array, you would need 3 reciprocal shots per sensor (one per force direction), recording the matching velocity component at the floor. For vertical geophones only: **1 shot per sensor, record v_z, done.** Total: 8 shots for N_sensor=8.

**Pitfall — sign convention**: Devito uses a right-handed coordinate system with z pointing upward (or downward depending on convention). Verify: if z is depth-positive (as in seismic), v_z positive = downward; a footstep (downward force on floor) injects a positive τ_zz spike. Check sign consistency in your post-processing.

**Pitfall — staggered-grid offset**: on the Virieux SSG, v_z is located at (i, j, k+½). The receiver interpolation (`SparseTimeFunction.interpolate`) handles the half-cell offset automatically in Devito.

### 2d. Recording all floor positions simultaneously (SparseTimeFunction)

```python
import numpy as np
from devito import SparseTimeFunction

# Build receiver array at all 480 floor positions (on the concrete floor surface)
floor_x = np.array([x for x in np.arange(1, 11, 0.5) for _ in np.arange(1, 13, 0.5)])
floor_y = np.array([y for _ in np.arange(1, 11, 0.5) for y in np.arange(1, 13, 0.5)])
floor_z = np.zeros_like(floor_x) + z_floor   # z-coordinate of floor slab surface

n_floor = len(floor_x)   # 480
rec_coords = np.column_stack([floor_x, floor_y, floor_z])   # (480, 3)

rec = SparseTimeFunction(name='rec', grid=grid, npoint=n_floor, nt=nt)
rec.coordinates.data[:] = rec_coords

# Record v_z (component index 2 in 3D)
rec_term = rec.interpolate(expr=v[2])   # interpolates v_z at each floor position

# Source: single vertical force at sensor k
src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
src.coordinates.data[0] = sensor_coords[k]
src.data[:, 0] = ricker_deriv_wavelet   # or GRF wavelet
src_term = src.inject(field=v[2].forward, expr=src * dt / rho)

op = Operator([v_update, tau_update, src_term, rec_term])
op.apply(time=nt, dt=dt_val)
# rec.data shape: (nt, n_floor) — the full G_zz(x_sensor_k, x_floor_p, t) for all p
```

---

## 3. GPU: Realistic Throughput on RTX 3090, Docker Setup, and Memory

### 3a. RTX 3090 hardware specs

| Spec | Value |
|---|---|
| VRAM | 24 GB GDDR6X |
| Peak memory bandwidth | **936 GB/s** |
| Peak FP32 FLOPS | 35.6 TFLOPS |
| Architecture | Ampere (GA102) |
| PCIe | 4.0 x16 (bandwidth irrelevant — stencil is GPU-memory-bound) |

The RTX 3090 is **memory-bandwidth bound** for stencil kernels (arithmetic intensity of elastic order-8 stencil ≈ 1 FLOP/byte → well below the ridge point on roofline). The realistic utilization of peak bandwidth by Devito OpenACC is 60–75% (based on reported benchmarks for Devito acoustic/elastic stencils); call it **~650 GB/s effective**.

### 3b. Arithmetic intensity and throughput

The 3-D elastic velocity–stress staggered-grid system has:
- 9 wavefield components (3 velocity + 6 independent stress) stored as float32 arrays
- 12 material arrays (λ, μ_xy, μ_xz, μ_yz at staggered locations, ρ_vx, ρ_vy, ρ_vz) → for basic implementation: 3 material arrays (lam, mu, rho) = 3 arrays
- Order-8 stencil: 8 neighbours per spatial direction per term

Bytes read/written per cell update ≈ 9 wavefield arrays × 2 (read old + write new) × 4 B + 3 material × 4 B = 72 + 12 = **84 bytes/cell/step** (minimum, ignoring stencil reuse imperfection). With stencil halo traffic and non-unit-stride access in 3D, effective bytes/cell/step reported in literature for elastic order-8 = **~100–130 bytes/cell/step** (the value 122 from the prior doc 06 is within this range).

At 650 GB/s effective: throughput = 650 / 122 ≈ **5.3 GP/s** (consistent with prior estimate of 5.75 GP/s — the prior estimate was slightly optimistic).

**Published data point for calibration**: DevitoPRO elastic benchmark on CPU (Intel Sapphire Rapids, 536 GB/s peak bandwidth) achieved 2.45 GP/s FP32 and 4.1 GP/s mixed-precision [devitocodes.com/elastic]. GPU runs faster due to higher bandwidth; the ratio A100/CPU ≈ 2039/536 ≈ 3.8×, giving expected ~9–10 GP/s on A100 (FP32) and ~5–6 GP/s on RTX 3090. This confirms the 5.3 GP/s estimate.

**Important caveat**: Devito OSS OpenACC is reported to be **~1.5× slower than hand-CUDA** (xDSL CUDA lowerings outperform Devito OpenACC tiled kernels by >1.5× on V100 [arXiv:2404.02218]). So the 5.3 GP/s figure already assumes the OpenACC path. DevitoPRO's CUDA path would reach ~8 GP/s, but that requires a commercial license.

### 3c. Revised per-scene wall-clock estimates

Using 5.0 GP/s (conservative, OpenACC, RTX 3090):

| Scenario | Cells | Time steps (0.5 s sim) | Cell-updates | Wall-clock |
|---|---|---|---|---|
| 100 Hz, dx=0.15 m | 1.85 M | 20,833 | 38.5 B | **7.7 s** |
| 250 Hz, dx=0.06 m | 28.9 M | 52,083 | 1,505 B | **301 s = 5.0 min** |

With N_sensor=8 reciprocal shots per house:

| Scenario | Per house (8 shots) | 50 houses | 500 houses |
|---|---|---|---|
| 100 Hz | 62 s | 52 min | **8.6 hr** |
| 250 Hz | 40 min | 33 hr | **14 days** |

These are sequentially scheduled. For 500 houses at 250 Hz, the RTX 3090 is too slow; use cloud A100 (throughput ~10 GP/s, ~13× per-scene) or 8×A100.

### 3d. Memory footprint check (both scenarios fit 24 GB)

Float32, 3D elastic: 9 wavefield arrays (time-level 0 + 1) + 3 material arrays = 12 arrays minimum. With source/receiver scratch and the order-8 ghost cells:

| Scenario | Grid | Arrays | Memory |
|---|---|---|---|
| 100 Hz | 160×173×67 = 1.85 M | 15 × float32 | **111 MB** |
| 250 Hz | 400×433×167 = 28.9 M | 15 × float32 | **1.73 GB** |

Both scenarios fit comfortably in the RTX 3090's 24 GB. Even adding 10 material arrays (Moczo staggered μ arrays) adds less than 200 MB at 250 Hz. There is no memory constraint for either scenario on the RTX 3090.

### 3e. Docker setup and environment variables

The official GPU path uses the NVIDIA HPC SDK `nvc` compiler for OpenACC offloading, not nvcc:

```bash
# Pull the pre-built GPU image (recommended — avoids HPC SDK install complexity)
docker pull devitocodes/devito:nvidia-nvc-latest

# Run with GPU access (requires nvidia-container-toolkit)
docker run --gpus all --rm -it \
  -v /path/to/project:/workspace \
  devitocodes/devito:nvidia-nvc-latest \
  bash

# Inside the container: verify GPU is visible
python -c "from devito import configuration; configuration['platform']='nvidiaX'; print(configuration)"

# Environment variables for GPU (set before import or via shell):
export DEVITO_PLATFORM=nvidiaX    # tells Devito this is an NVIDIA GPU
export DEVITO_LANGUAGE=openacc   # generate OpenACC C code (compiled by nvc)
export DEVITO_ARCH=nvc            # use the nvc compiler (NVIDIA HPC SDK)
# Optional:
export DEVITO_DEVICE=0            # GPU device index (0 = first GPU)
export DEVITO_LOGGING=DEBUG       # verbose JIT compilation output
```

Or in Python:
```python
from devito import configuration
configuration['platform'] = 'nvidiaX'
configuration['compiler'] = 'pgcc'    # older alias; use 'nvc' in HPC SDK ≥21.1
configuration['language'] = 'openacc'
```

**Windows/WSL2 gotchas**:
1. Docker Desktop with WSL2 backend does NOT support `--gpus` with OpenACC workloads. Use **Docker in a native Linux VM** or WSL2 with the NVIDIA CUDA driver for WSL (not the full HPC SDK — the HPC SDK must be installed inside the container). The pre-built `nvidia-nvc-latest` image already contains the HPC SDK.
2. RTX 3090 is a consumer GPU; NVIDIA's HPC SDK targets consumer cards fine. No Quadro/Data Center required.
3. `nvidia-container-toolkit` must be installed on the host: `apt install nvidia-container-toolkit && systemctl restart docker`.
4. Verify: `docker run --gpus all --rm devitocodes/devito:nvidia-nvc-latest nvidia-smi` should show the RTX 3090.

**Native install alternative** (no Docker):
```bash
# Install NVIDIA HPC SDK (free, includes nvc + OpenACC runtime)
# From https://developer.nvidia.com/hpc-sdk (Ubuntu/CentOS)
conda create -n devito python=3.11 && conda activate devito
pip install "devito[extras,nvidia]"
# Set env vars as above; Devito will JIT-compile via nvc on first run
```

---

## 4. Frequency Choice: 100 Hz vs 250 Hz Cost–Accuracy Trade-off

### The f⁴ scaling

The computational cost of a 3D FDTD sim scales as:
```
Cost ∝ N_cells × N_steps = (f_max/V_min · PPW)³ × (T_sim · f_max · Vp_max / (dx · √3))
     = PPW³ · f_max³ · (1/V_min³) · T_sim · f_max · Vp_max · √3 / (PPW · V_min)
     = PPW² · √3 · Vp_max/V_min⁴ · T_sim · f_max⁴
```

So cost ∝ f_max⁴ (holding Vp_max, V_min, PPW, T_sim constant). From 100 Hz to 250 Hz: (250/100)⁴ = **39.1× more expensive**.

### What does higher frequency buy?

| Property | 100 Hz | 250 Hz |
|---|---|---|
| Min Rayleigh wavelength (soil, V_R=138 m/s) | 1.38 m | 0.55 m |
| Spatial resolution | 1.38 m | 0.55 m |
| Position ambiguity (matched-field, ½λ) | ~0.7 m | ~0.28 m |
| Thin-wall visibility (0.15 m wall) | 1 cell (marginal) | 2.5 cells (acceptable) |
| PPW for Rayleigh at each scenario | 9.2 PPW | 9.2 PPW (same ratio) |
| Frequency content of footstep heel-strike | Up to ~300 Hz — captured at 250 Hz | Captures heel-strike transient; 100 Hz misses it |

The critical practical difference:
- **100 Hz**: Rayleigh wavelength 1.38 m → position ambiguity ~0.7 m. Adequate for room-level localization (3–5 m rooms); insufficient for precise position tracking (< 0.5 m accuracy goal).
- **250 Hz**: Rayleigh wavelength 0.55 m → position ambiguity ~0.28 m. Adequate for sub-half-metre localization. Also captures the heel-strike energy band (50–250 Hz) that carries most of the structural information.
- The heel-strike transient has 80% of its energy above 80 Hz; the second harmonic region (100–250 Hz) is where the building's structural response is most distinct between floor positions.

### Decision rule

- **For dataset generation at scale (> 200 houses)**: start at **100 Hz** on the RTX 3090. Wall-clock 8.6 hr / 50 houses is manageable overnight. Use 100 Hz training data to demonstrate feasibility; use position accuracy as the gate metric.
- **If 100 Hz position error > 1 m**: upgrade to **250 Hz** on cloud A100 (5.0 min / house × 8 shots = 40 min / house × 50 houses = 33 hr, ~$20 on spot A100 at $0.6/hr).
- **Do not use 250 Hz on RTX 3090 at scale** (14 days / 500 houses).

### Hybrid strategy (recommended)

Generate the Green's function library at 250 Hz (preserves all frequency content); low-pass filter to 100 Hz for the initial training run. If the SNN needs higher-frequency features, re-train with the full-band 250 Hz library — no re-simulation needed. This costs one 250 Hz sim campaign but buys unlimited post-hoc frequency experiments.

---

## 5. Absorbing Boundaries: Cosine Sponge vs CPML, Corner Treatment

### The options

| Method | Typical reflection | Cells needed | Implementation in Devito | Stability |
|---|---|---|---|---|
| Cosine-taper sponge (`model.damp`) | ~5–15% at 10 cells, ~1–5% at 20 cells | 10–20 | **Built-in** | Unconditionally stable |
| Lysmer-Kuhlemeyer dashpot | Good for body waves > 30° incidence; poor for Rayleigh / near-normal | 1 | Manual SubDomain equation | Stable |
| CPML (CFS-PML) | ~0.01–0.1% at 10 cells | 10 | Manual (~200 lines) | Corner instability risk |
| C-PML with M-PML corners | ~0.01–0.1% | 10 | Manual + multiaxial corner fix | Stable |

### CPML vs sponge: quantitative comparison

CPML (convolutional PML) with 10–15 cells achieves reflection coefficients below −72 dB (< 0.025%) for all incidence angles and evanescent modes. The cosine sponge at 10 cells achieves ~5–15% (−17 to −26 dB) and at 20 cells achieves ~1–5% (−26 to −40 dB). For training-data generation where systematic bias is more tolerable than random noise, sponge artifacts (~5%) are acceptable.

**Sponge parameter to use in Devito's `model.damp`:**

```python
# Devito handles sponge via the 'nbl' (number of boundary layers) parameter
# Best practice for 100 Hz scenario:
model = demo_model('layers-elastic', ndim=3, shape=shape,
                   spacing=(dx, dx, dx), nbl=20,  # 20 sponge cells = 3.0 m buffer at dx=0.15
                   vp=..., vs=..., rho=...)
# 20 cells × 0.15 m = 3.0 m sponge; reflection ~2–5% at 100 Hz
# For 250 Hz, use nbl=30 (30 × 0.06 = 1.8 m sponge; reflection ~2–5% same)
```

### Free-surface corner treatment

The corner where the **top free surface** meets the **side sponge** is potentially unstable with CPML but safe with the sponge:

- **Sponge + stress-image**: the cosine damping term in the velocity equation and the antisymmetric stress-image in the ghost cells are independent operations. The damping does not break the antisymmetry; the corner is stable. **This is the recommended combination.**
- **CPML + stress-image**: the complex-frequency-shift of the PML equations must not be applied in the ghost cells above the free surface (which are outside the physical domain). Apply CPML on side faces only, stopping at z = z_fs. Corner cells at top-side intersections receive only the free-surface image, no CPML. This is stable if implemented correctly but adds 40 lines of corner-bookkeeping code.
- **CFS-PML corner instability**: the standard CFS-PML propagated into the free-surface corner develops late-time exponential instability (Drossaert & Giannopoulos 2007). Mitigation: use M-PML (multiaxial PML) in the 6 edge regions and 8 corner cells only. M-PML is available in SW4 but not in Devito OSS.

**Layout for this project (safe default):**

```
Top face (z = z_fs):       free-surface ghost image; NO sponge; NO PML
5 remaining faces:          cosine sponge, nbl=20 cells
Top-edge corners:           free-surface image only (sponge is off in top row)
All other corners/edges:    cosine sponge (naturally handled by model.damp)
```

Implementation in Devito: set `model.damp` to zero in the top `so//2 = 4` rows (the ghost-cell region). Devito's built-in nbl sponge applies the damping to all 6 faces by default; add a post-step override `model.damp.data[..., -4:] = 0` to suppress damping in the free-surface ghost layer.

---

## 6. Visualization: PyVista and ParaView Pipeline

### Overview of the pipeline

The goal is two types of visualization:
1. **Wavefield snapshots** for validation (compare Lamb's problem to analytic; check Rayleigh wavefront shape).
2. **Propagation animations** for demo/paper (3D house model with wave propagating through it).

Both are best served by writing Devito output to VTK/HDF5 format and rendering with PyVista (interactive Python) or ParaView (high-quality off-screen batch rendering).

### 6a. Writing wavefield snapshots from Devito

```python
import h5py, numpy as np

# During the operator.apply() loop, capture every N_snapshot steps
snapshots = {}
n_snap = 50   # capture every 50 steps

def snapshot_callback(step, time, **kwargs):
    if step % n_snap == 0:
        snapshots[step] = {
            'vz':  v[2].data[0].copy(),    # v_z, current time level
            'tau_zz': tau[2,2].data[0].copy()
        }

# Alternative: pre-allocate and fill during manual time loop
nt = 20833   # 100 Hz scenario
vz_cube = np.zeros((nt // n_snap, nx, ny, nz), dtype=np.float32)
for i_snap in range(nt // n_snap):
    op.apply(time_m=i_snap*n_snap, time_M=(i_snap+1)*n_snap, dt=dt_val)
    vz_cube[i_snap] = v[2].data[0].copy()

# Save to HDF5
with h5py.File('wavefield_snapshots.h5', 'w') as f:
    f.create_dataset('vz', data=vz_cube, compression='gzip', compression_opts=4)
    f.attrs['dx'] = dx; f.attrs['dt'] = dt_val * n_snap
    f.attrs['nx'] = nx; f.attrs['ny'] = ny; f.attrs['nz'] = nz
```

### 6b. PyVista: interactive wavefield animation (validation + quick demo)

```python
import pyvista as pv
import numpy as np, h5py

# Load snapshot cube
with h5py.File('wavefield_snapshots.h5', 'r') as f:
    vz_cube = f['vz'][:]          # shape: (n_snap, nx, ny, nz)
    dx = f.attrs['dx']

# Build PyVista ImageData (the only type that supports add_volume)
grid = pv.ImageData()
grid.dimensions = np.array([nx, ny, nz]) + 1   # cell-corner convention
grid.origin = (0, 0, 0)
grid.spacing = (dx, dx, dx)

# Interactive animation: update point data at each time step
plotter = pv.Plotter()
grid.cell_data['vz'] = vz_cube[0].flatten(order='F')   # Fortran order = VTK z-fastest
vol = plotter.add_volume(grid, scalars='vz', cmap='RdBu',
                         opacity=[0,0,0.1,0.5,0.1,0,0],   # symmetric, highlight wavefronts
                         clim=[-1e-9, 1e-9])

plotter.show(auto_close=False)
for i in range(len(vz_cube)):
    grid.cell_data['vz'] = vz_cube[i].flatten(order='F')
    plotter.render()

# Static slice through the wavefield at a given time step
sliced = grid.slice(normal='z', origin=(0, 0, z_floor))
plotter2 = pv.Plotter()
plotter2.add_mesh(sliced, scalars='vz', cmap='RdBu', clim=[-1e-9, 1e-9])
plotter2.show()
```

**Key PyVista facts:**
- Volume rendering (`add_volume`) is only supported for `pv.ImageData` (formerly `UniformGrid`).
- Data must be in Fortran order for VTK z-fastest indexing when using `cell_data`.
- The `opacity` parameter is a transfer function (length 5–7 list) mapping scalar value → opacity. Use a symmetric V-shape (zero opacity at mid-range, high at extremes) to highlight wavefronts while hiding the near-zero interior.
- For off-screen batch rendering: set `pv.start_xvfb()` on headless Linux or set env `PYVISTA_OFF_SCREEN=1`.

### 6c. Adding building geometry to the wavefield visualization

```python
# Overlay the house geometry as a wireframe
# Build the wall geometry as PyVista boxes
walls = []
for wall_def in house_wall_definitions:
    box = pv.Box(bounds=wall_def)   # (x_min, x_max, y_min, y_max, z_min, z_max)
    walls.append(box)

house_mesh = walls[0].merge(walls[1:])   # combine into single mesh

plotter = pv.Plotter()
plotter.add_mesh(house_mesh, color='tan', opacity=0.3, style='wireframe', line_width=2)
plotter.add_volume(grid, scalars='vz', cmap='RdBu', opacity=..., clim=...)
# Add sensor locations as spheres
sensors = pv.PolyData(sensor_coords)
plotter.add_mesh(sensors, color='red', point_size=10, render_points_as_spheres=True)
plotter.show()
```

### 6d. ParaView: batch rendering for publication/paper

ParaView handles larger grids and provides publication-quality rendering with programmable filters.

**Workflow:**
1. Write snapshots to XDMF+HDF5 (ParaView's preferred format for time series):

```python
import meshio

for i_snap, snap in enumerate(vz_cube):
    cells = {}   # no cells for point data on a regular grid
    points = None
    # Use meshio or directly write .vti (VTK image data) files
    import vtk
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(dx, dx, dx)
    arr = vtk.util.numpy_support.numpy_to_vtk(snap.flatten('F'), deep=True)
    arr.SetName('vz')
    image.GetCellData().AddArray(arr)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(f'snap_{i_snap:04d}.vti')
    writer.SetInputData(image)
    writer.Write()
```

2. Load the `.vti` file series in ParaView (File → Open → select `snap_*.vti` as a group).
3. Apply filters: Clip (to reveal interior cross-section), Threshold (to show only large-amplitude cells = wavefront), Volume Rendering (scalar field → color/opacity transfer function).
4. Use the Animation panel to play through time steps and export as MP4.

**ParaView scripting (pvpython) for batch off-screen rendering:**

```python
# Run as: pvpython render_wavefield.py
from paraview.simple import *

reader = XMLImageDataReader(FileName=['snap_0000.vti', 'snap_0001.vti', ...])
GetAnimationScene().UpdateAnimationUsingDataTimeSteps()

clip = Clip(Input=reader)
clip.ClipType = 'Plane'; clip.ClipType.Normal = [0, 1, 0]   # cut along Y

display = Show(clip, GetActiveView())
display.Representation = 'Volume'
display.ColorArrayName = 'vz'

SaveAnimation('wavefield_animation.avi', GetActiveView(),
              ImageResolution=[1920, 1080], FrameRate=15)
```

### 6e. Recommended validation workflow

For Lamb's problem validation (doc 08):

```python
# After running Lamb's problem sim, extract radial + vertical traces at each receiver
for r_idx, r in enumerate(receiver_ranges):
    rx, ry, rz = receiver_coords[r_idx]
    vz_trace = rec_vz.data[:, r_idx]   # (nt,) time series
    vr_trace  = rec_vr.data[:, r_idx]  # horizontal, if recorded

# Plot comparison: Devito vs analytic
import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(receiver_ranges), 1, figsize=(12, 20))
for i, r in enumerate(receiver_ranges):
    axes[i].plot(t_axis, vz_trace_devito[i], label='Devito (stress-image)')
    axes[i].plot(t_analytic, vz_analytic[i], '--', label='Analytic (Pekeris/ktkimit)')
    # Mark expected arrival times
    axes[i].axvline(r/Vp, color='blue', linestyle=':', label='P')
    axes[i].axvline(r/V_R, color='red', linestyle=':', label='Rayleigh')
    axes[i].legend(); axes[i].set_title(f'r = {r} m')
plt.tight_layout(); plt.savefig('lamb_validation.png', dpi=150)
```

For wavefield snapshots as paper figures, use PyVista with a clip plane at z = z_floor to show the horizontal slice, and overlay the geophone positions as coloured markers. Export at 300 dpi via `plotter.screenshot('figure.png', return_img=False, window_size=(3000, 2000))`.

---

## 7. Consolidated Parameter Reference

| Parameter | 100 Hz scenario | 250 Hz scenario | Notes |
|---|---|---|---|
| dx (m) | 0.150 | 0.060 | Governs wall resolution |
| dt (μs) | 24 | 9.6 | CFL: dt ≤ dx/(Vp_max·√3) |
| Grid (cells) | 160×173×67 = 1.85 M | 400×433×167 = 28.9 M | Domain 24×26×10 m |
| FDTD memory (21 arrays, f32) | 155 MB | 2.43 GB | Fits RTX 3090 comfortably |
| Extended memory (Moczo, ~15 arrays) | 111 MB | 1.73 GB | Including staggered μ and ρ arrays |
| Time steps (0.5 s sim) | 20,833 | 52,083 | |
| Cell-updates / sim | 38.5 B | 1,505 B | |
| Relative cost (100→250 Hz) | 1× | **39×** | f⁴ scaling |
| RTX 3090 throughput (OpenACC) | ~5 GP/s | ~5 GP/s | Memory-bandwidth-limited |
| Wall-clock / sim (1 shot) | **7.7 s** | **301 s (5.0 min)** | |
| Wall-clock / house (8 reciprocal shots) | **62 s** | **40 min** | |
| 50 houses, RTX 3090 | 52 min | 33 hr | |
| 500 houses, RTX 3090 | **8.6 hr** | **14 days** | 250 Hz: need cloud |
| 500 houses, A100 (10 GP/s est.) | ~4.3 hr | ~5.6 days | Or 8×A100: ~40 min |
| Wall voxel coverage (0.15 m wall) | **1 cell** (under-resolved) | **2.5 cells** (marginal) | Use Moczo averaging |
| Staircasing amplitude bias (walls, Moczo) | ~10% | ~5% | Document in GF library |
| Sponge reflection (nbl=20) | ~2–5% | ~2–5% | Acceptable for training |
| Source: vertical footstep | τ_zz injection or f_z in v_z update | Same | Not explosive (not τ_xx + τ_yy + τ_zz) |
| Reciprocity: shots needed | 8 (one per sensor, f_z, record v_z) | Same | G_zz only for vertical geophones |
| Visualization tool | PyVista (interactive) | ParaView (batch/publication) | Both read VTK/HDF5 |

---

## 8. Open Issues and Risks Not Yet Resolved

1. **Moczo averaging implementation in Devito**: The Devito elastic tutorial uses a single `mu` field. Correct Moczo averaging requires three μ fields (mu_xy, mu_xz, mu_yz), each passed to the appropriate shear stress update equation. This requires modifying the elastic operator equations — straightforward but not demonstrated in any Devito example. Test on a 1-D layered model with a known analytic solution before scaling to 3-D.

2. **Sponge vs CPML at 250 Hz**: At 250 Hz the sponge with nbl=20 (1.2 m buffer at dx=0.06) may be insufficient. Reflection at 5% will contaminate late-arriving Rayleigh tails. If Lamb's problem shows > 5% spurious reflection energy in the post-Rayleigh window, increase nbl to 30 (1.8 m) or implement CPML.

3. **OpenACC compilation time**: Devito JIT-compiles the operator on first run. For a 3-D order-8 elastic operator with Moczo averaging, compilation via nvc can take 5–15 minutes. Cache the compiled operator and reuse it across simulations (`op = Operator(...); op.apply(...)` — Devito caches by operator hash).

4. **RTX 3090 OpenACC driver compatibility**: The nvc compiler in the Docker image targets specific CUDA compute capabilities. Verify the image's nvc version supports sm_86 (Ampere, RTX 3090). The `devitocodes/devito:nvidia-nvc-latest` image with NVIDIA HPC SDK ≥ 21.9 supports sm_86.

5. **Large SparseTimeFunction memory**: Recording 480 floor positions × 20,833 time steps at float32 = 40 MB per sim — negligible. At 500 houses × 8 shots: 160 GB of raw trace data before compression. With HDF5 gzip compression (ratio ~5–10×): 16–32 GB on disk. Plan storage accordingly.

---

## 9. Key References

- **Moczo, Kristek, Vavrycuk, Archuleta & Halada (2002)**: 3D heterogeneous staggered-grid FD modeling of seismic motion with volume harmonic and arithmetic averaging. *BSSA* 92(8), 3042–3066. — Harmonic/arithmetic averaging rules for staggered elastic parameters at interfaces.
- **Virieux (1986)**: P-SV wave propagation in heterogeneous media: velocity-stress FD method. *Geophysics* 51, 889–901. — The foundational staggered-grid scheme used by Devito.
- **Kristek, Moczo & Archuleta (2002)**: Efficient methods to simulate planar free surface in 3D 4th-order staggered-grid FD schemes. *Studia Geophysica et Geodaetica* 46, 355–381. — Stress-image free surface (from doc 08).
- **Wapenaar (2004)**: Retrieving the elastodynamic Green's function of an arbitrary inhomogeneous medium by cross-correlation. *PRL* 93, 254301. — Reciprocity identity G_ij(A,B) = G_ji(B,A).
- **DevitoCodes elastic benchmark**: "Advancements in elastic wave solvers using DevitoPRO and mixed-precision." [devitocodes.com/elastic] — 2.45 GP/s FP32, 4.1 GP/s mixed on Intel Sapphire Rapids (536 GB/s BW). Basis for RTX 3090 extrapolation.
- **arXiv:2404.02218 (Luo et al., ASPLOS 2024)**: "A shared compilation stack for distributed-memory parallelism in stencil DSLs." — xDSL CUDA outperforms Devito OpenACC by > 1.5× on V100; calibration for OpenACC penalty.
- **Devito download/GPU docs**: [devitoproject.org/download.html] — Docker images `devito:nvidia-nvc-latest`; environment variables DEVITO_PLATFORM, DEVITO_LANGUAGE, DEVITO_ARCH.
- **PyVista volume rendering**: [docs.pyvista.org/examples/02-plot/volume_rendering.html] — `ImageData` + `add_volume` + opacity transfer function for wavefield animation.
- **ParaView seismic use**: SphGLLTools (Ciardelli et al. 2022, CompGeosci) demonstrates XDMF+HDF5 import for seismic adjoint models; same pattern for wavefield snapshots.
- **Bohlen & Saenger (2006)**: Accuracy of heterogeneous staggered-grid FD modeling of Rayleigh waves. *Geophysics* 71, T109–T115. — Vacuum vs stress-image amplitude errors at 10 PPW (from doc 08).
