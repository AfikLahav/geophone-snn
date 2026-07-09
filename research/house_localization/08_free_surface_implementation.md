> **Elastic free-surface deep-dive — house-localization pivot.** Research (Sonnet, 2026-07-08). Resolves the make-or-break technical question raised in `07_sim_engine_verification_and_plan.md`: how to implement and validate a traction-free elastic free surface in OSS Devito. Indexed in [README.md](README.md).

---

# Elastic Free Surface in Devito: Implementation, Validation, and Risk Assessment

## Why This Is the Make-or-Break Question

Footstep vibrations from a person walking inside a building travel from the concrete slab, through the building structure and soil, to exterior geophones. The dominant long-range path is **Rayleigh/surface waves**, which are guided along the traction-free building faces and the soil surface. If the free surface is wrong:

- Rayleigh-wave amplitudes are incorrect → Green's function library is wrong → matched-field localization fails
- Reflected body waves at the surface are wrong → waveform timing errors → TDOA and fingerprinting both degrade
- For training-data quality the entire corpus is systematically biased in exactly the channel (surface-wave coupling) that carries most of the usable energy to exterior sensors

OSS Devito's `freesurface()` call is **acoustic only**. The `devitoboundary` package that had elastic free surface was **archived February 2023**, and its successor `Schism` is **DevitoPRO-only** with no published Rayleigh validation. This document provides a complete build-and-validate recipe for doing it yourself.

---

## 1. The Physics: What "Traction-Free" Means in a Staggered Grid

In the first-order velocity–stress (Virieux 1986) formulation, the governing equations are:

```
ρ ∂vᵢ/∂t = ∂τᵢⱼ/∂xⱼ        (momentum)
∂τᵢⱼ/∂t = λδᵢⱼ ∇·v + μ(∂vᵢ/∂xⱼ + ∂vⱼ/∂xᵢ)  (Hooke)
```

The traction-free condition on a horizontal free surface at z = z_fs requires:

```
τzz = 0,   τxz = 0,   τyz = 0   at z = z_fs
```

Only τxx and τyy are non-zero (tangential in-plane stresses). On the Virieux staggered grid, the stress components are NOT co-located:
- τzz is on full-integer z-nodes (at the surface)
- τxz and τyz are staggered at z ± ½dx (below and above the surface)

This staggering creates the fundamental difficulty: you cannot directly set all three tractions to zero simultaneously because they live on different sub-grids.

---

## 2. Implementation Option A — Vacuum / Air Layer

### Mechanics

Set the top 2–4 cells of the domain to near-zero elastic moduli and density (ρ ≈ ε, λ ≈ ε, μ ≈ ε). The stiffness contrast between air and solid is ~10⁶, which effectively zeroes the stress above the surface. The particle velocity in the "air" layer is then set by the momentum equation with near-zero right-hand side.

### Devito code (5 lines)

```python
eps = 1e-6
# Assuming vp, vs, rho are 3-D numpy arrays, z-index last (or first, depending on convention)
vp[..., -nair:]  = eps
vs[..., -nair:]  = eps
rho[..., -nair:] = eps
model.update('vp', vp); model.update('vs', vs); model.update('rho', rho)
# Remove the top ABL (sponge) from the last nair cells — already treated as vacuum
```

No extra SubDomain, no ghost-cell equations, no symbolic manipulation of the stress tensor. The existing Virieux update loop handles everything.

### Accuracy vs stress-image

The vacuum formulation is **first-order accurate at the free surface** (O(dx) truncation error there vs O(dx⁴) in the interior). The practical consequence for Rayleigh waves:

- Rayleigh-wave phase velocity error: ~1–3% at 10 PPW (compared to <0.1% for stress-image at the same PPW)
- Rayleigh-wave amplitude error: **5–15% at 10 PPW** (literature: Mittet 2002, Bohlen & Saenger 2006)
- To match stress-image accuracy, vacuum needs **~25 PPW** (some sources cite 60 PPW for topographic surfaces)
- At our 100 Hz scenario (dx = 0.150 m, Vs_soil = 150 m/s) the minimum Rayleigh wavelength ≈ 150/100 × 0.9194 = 1.38 m → ~9 cells per wavelength — right at the danger zone

**Bottom line on vacuum:** Usable for a rough first test but will introduce systematic ~10% amplitude errors in Rayleigh wave Green's functions, which feeds directly into localization dictionary errors. **Not acceptable for the final training corpus.**

### Stability

Vacuum is unconditionally stable (the CFL limit is unchanged because the maximum wave speed is still Vp_concrete). No corner or long-run instability issues.

---

## 3. Implementation Option B — Antisymmetric Stress-Image (Kristek et al. 2002)

### Reference

Kristek, J., P. Moczo, and R. J. Archuleta (2002): "Efficient methods to simulate planar free surface in the 3D 4th-order staggered-grid finite-difference schemes." *Studia Geophysica et Geodaetica*, 46, 355–381. DOI: 10.1023/A:1019866422821 — the seminal paper cited by essentially all geophysical FDTD codes for free-surface treatment on SSG (standard staggered grids).

### Physical idea

Extend the computational domain by a "ghost" layer of 1–2 cells above the free surface. Enforce traction-free BC by **antisymmetrically imaging the stress components** in the ghost cells:

```
τzz(z_fs + k·dx) = −τzz(z_fs − k·dx)   for k = 1, 2, ...
τxz(z_fs + ½dx)  = −τxz(z_fs − ½dx)
τyz(z_fs + ½dx)  = −τyz(z_fs − ½dx)
```

This antisymmetry ensures that the finite-difference stencil computing ∂τzz/∂z at the surface (which spans across z_fs) naturally sees τzz = 0 at z_fs, and that the shear stress derivative is also correctly computed. τxx and τyy above the surface are zeroed (they don't appear in the vertical traction).

### How it works on the Virieux staggered grid (concrete mechanics)

The free surface sits between two z-planes of the grid. Specifically, Kristek et al. test two formulations:

- **H-formulation**: free surface at z = k·dx (full-integer z-node; τzz lives here)
- **W-formulation**: free surface at z = (k+½)·dx (half-integer node; τxz/τyz live here)

The H-formulation is simpler because τzz is directly at the surface:
1. At every time step, BEFORE the stress update: set τzz at index k_fs = 0 (or wherever the surface row is).
2. After the stress update in the interior: mirror-copy τzz from below the surface into ghost cells above: `tau_zz[k_fs+j] = -tau_zz[k_fs-j]` for j = 1, 2.
3. Mirror τxz and τyz: `tau_xz[k_fs+1] = -tau_xz[k_fs]` (these are half-integer, so only one ghost layer needed for 4th-order).
4. Set τxx and τyy in ghost cells to zero (they are non-zero at and below the surface but we need their values for the stencil, so leave them as computed — they naturally remain finite and do not affect traction).
5. On the VELOCITY update at the surface: ∂τzz/∂z at z_fs uses τzz(z_fs+½) and τzz(z_fs−½). Because τzz = 0 at z_fs and the antisymmetric image gives τzz(z_fs+½) = −τzz(z_fs−½), this derivative is correctly second-order-accurate without explicit treatment.

### PPW requirements (Kristek et al. 2002 results)

| Method | PPW for <1% Rayleigh phase error | PPW for <5% amplitude error |
|---|---|---|
| Vacuum (standard staggered grid) | ~20–25 | ~15 |
| Stress-image H/W (standard SSG) | **~10** | **~10** |
| AFDA (adjusted FD at surface nodes) | **~6** | **~6** |

The AFDA (adjusted finite-difference approximation) is a further refinement by Kristek et al. where the FD coefficients at and near the surface are recomputed to account for the one-sided stencil — it gives the best accuracy at the lowest PPW but requires modifying the update stencil at specific grid rows, which is more complex to implement in Devito's symbolic framework.

**Recommendation**: implement the standard H-formulation stress-image (not AFDA). At 10 PPW (our scenario: dx=0.15 m, λ_min_Rayleigh≈1.38 m → 9.2 cells — marginally tight) this gives ~2–3% phase error and ~5% amplitude error. Acceptable for training data if validated.

### Devito implementation — SubDomain approach (~100–130 lines)

The stress-image is most naturally expressed as a set of extra equations over a `SubDomain` covering the ghost layer above the free surface. Here is the complete pattern (Python pseudocode; variable names follow the OSS elastic tutorial):

```python
from devito import SubDomain, Eq, solve, Operator, Grid
from devito import VectorTimeFunction, TensorTimeFunction, Function
import numpy as np

# --- 1. Define SubDomain for the free surface ghost layer ---
class FreeSurfaceDomain(SubDomain):
    name = 'fsdomain'
    def __init__(self, so):
        super().__init__()
        self._so = so

    def define(self, dimensions):
        # Cover the top 'so//2' cells (ghost layer above + at the surface)
        x, y, z = dimensions
        return {x: x, y: y, z: ('left', self._so // 2)}

# --- 2. Build grid, functions, material ---
so = 8  # spatial order; ghost layer needs so//2 = 4 cells
nx, ny, nz = 160, 173, 67
grid = Grid(shape=(nx, ny, nz), extent=(nx*0.15, ny*0.15, nz*0.15))
fsd = FreeSurfaceDomain(so=so, grid=grid)

# VectorTimeFunction and TensorTimeFunction as in the elastic tutorial
v   = VectorTimeFunction(name='v',   grid=grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name='tau', grid=grid, space_order=so, time_order=1)

# Material fields (fill as in doc 06)
lam = Function(name='lam', grid=grid, space_order=0)
mu  = Function(name='mu',  grid=grid, space_order=0)
rho = Function(name='rho', grid=grid, space_order=0)

# --- 3. Interior update equations (standard Virieux) ---
x, y, z = grid.dimensions
dt = grid.stepping_dim.spacing

u_v = Eq(v.forward, v + dt/rho * div(tau))
u_tau = Eq(tau.forward, tau + dt*(lam*diag(div(v.forward))
                                  + mu*(grad(v.forward) + grad(v.forward).T)))

# --- 4. Free-surface stress-image equations over FreeSurfaceDomain ---
# The free surface is at z-index 0 (the top face of the domain, z dimension = last).
# Ghost cells: z-index -1, -2, ... (above surface = negative indices = wrap-around).
# Antisymmetric image: tau_zz[-j] = -tau_zz[j], tau_xz[-1] = -tau_xz[0]

# Zero out traction at surface directly (belt + suspenders)
fs_zero_tzz = Eq(tau[2, 2],  0,     subdomain=fsd)   # tau_zz = 0 at z=0 row

# Antisymmetric mirror into ghost cells (order-8 stencil needs 4 ghost rows)
# tau_zz: ghost rows j=1..4 above surface ← negative of rows j=1..4 below
ghost_eqs = []
for j in range(1, so//2 + 1):
    ghost_eqs.append(
        Eq(tau[2, 2].subs(z, -j),
           -tau[2, 2].subs(z,  j),
           subdomain=fsd)
    )
    # Shear: tau_xz, tau_yz (staggered at half-integer z)
    ghost_eqs.append(
        Eq(tau[0, 2].subs(z, -j),
           -tau[0, 2].subs(z,  j),
           subdomain=fsd)
    )
    ghost_eqs.append(
        Eq(tau[1, 2].subs(z, -j),
           -tau[1, 2].subs(z,  j),
           subdomain=fsd)
    )

# --- 5. Build operator: interior update → free-surface image ---
op = Operator([u_v, u_tau, fs_zero_tzz] + ghost_eqs,
              name='ElasticFreesurf')
op.apply(time=nt, dt=dt_val)
```

**Important notes on the above skeleton:**
- The exact SubDomain index substitution syntax (`subs(z, -j)`) needs to be adapted to Devito's indexed access API (use `tau._data[...][iz_surface - j]` or Devito's `AbstractFunction.indexed` form). The symbolic index substitution shown is conceptual; test against the scalar index API.
- In practice, the ghost-layer equations are applied AFTER the stress tensor update, BEFORE the velocity update (operator ordering: interior stress update → mirror image → velocity update).
- Devito v4.8+ has the `SubDomain` + equation approach fully supported for MPI; this is the recommended path per the Devito BCs tutorial.
- The ghost-layer needs to be outside the absorbing sponge region — the free surface and the ABL must not overlap (see Section 4).

### Line count

The above pattern is ~90–110 lines of Python (excluding imports and material setup). The AFDA variant would add ~40 additional lines for modified stencil coefficients. The stress-image H-formulation is the target.

---

## 4. Free-Surface vs Absorbing Boundary (ABL) Interaction at Corners

### The problem

The top face is the free surface (traction-free). The side and bottom faces are absorbing boundaries (sponge or PML). The **corners** where side ABL meets the top free surface are a known instability source.

For the sponge ABL (Devito default):
- Sponge acts by adding a damping term `d(x) * v` to the velocity equation, where `d(x)` ramps from 0 at the physical domain edge to `d_max` at the grid edge over `nbl` cells (typically 10–20 cells).
- The free surface must sit at the **physical domain boundary**, i.e., outside the sponge layer (or at the inner edge of the sponge, not inside it).
- Corner cells where the sponge is active AND the free-surface ghost image is applied receive both damping and antisymmetric stress mirroring. Because sponge damping is real-valued and multiplicative, it does not break the antisymmetry — **sponge + stress-image is stable**.

For CPML / C-PML (if using):
- The PML equations should NOT be applied in the free-surface ghost cells — the ghost cells are outside the physical domain and the PML stretch is undefined there.
- Apply PML on side faces only, stopping at z = z_fs. The top row of PML cells should be one row below the free surface.
- **Known instability**: if the CFS-PML (complex-frequency-shifted PML) is extended to the free-surface corner without the "surface correction" (Drossaert & Giannopoulos 2007, Correia & Jin 2004), the corner can go unstable at long run times. Mitigations: (a) use M-PML (multiaxial PML) in corners only, (b) use surface correction, (c) stay with the sponge which is unconditionally stable.
- **Recommendation for our use case**: use the cosine-taper sponge on all 5 non-surface faces. Sponge + stress-image is the simplest stable combination. Accept ~10% reflection from the sponge (acceptable for training data).

### Corner treatment in practice

```
Free surface (z=top): ghost image applied, NO sponge, NO PML
Side faces (x=0, x=Lx, y=0, y=Ly): sponge active in 10-20 cell border
Bottom (z=0): sponge active in 10-20 cell layer
Corners top-side: z=top row of sponge is omitted; corner cells receive free-surface image only
```

This is the conservative safe approach. Do not apply sponge damping at z = z_fs or the ghost cells above.

---

## 5. Validation Protocol: Lamb's Problem

### Why Lamb's problem

Lamb (1904) derived the exact surface displacement field for a vertical point force applied to a traction-free elastic half-space. It has analytic closed-form solutions (Pekeris 1955, Cagniard-de Hoop method) and is the canonical benchmark for any elastic free-surface implementation.

### Material parameters for the validation run

Use a homogeneous soil half-space (simple, no concrete buildings):

```
Vp  = 520 m/s     (= Vs * √3 for ν = 0.25)
Vs  = 300 m/s
ρ   = 2000 kg/m³
ν   = 0.25         (Poisson solid; simplest analytic case)
```

Grid: 50×50×25 m domain, dx = 0.15 m (same as 100 Hz house sim), dt from CFL.  
Source: vertical point force at surface center, Ricker derivative f₀ = 20 Hz.  
Receivers: radial array at r = 5, 10, 15, 20, 25 m on the surface.  
Record vz (vertical) and vr (radial = horizontal in 2D) at each receiver.

### Analytic pass criteria

#### 1. Arrival times (P and Rayleigh)

For a source at the origin and receiver at range r:

```
t_P  = r / Vp  = r / 520        [seconds]
t_S  = r / Vs  = r / 300        [seconds]
t_R  = r / V_R = r / (0.9194 × 300) = r / 275.8   [seconds]
```

For ν = 0.25: V_R = 0.9194 Vs (exact root of the Rayleigh secular equation ζ³ − 8ζ² + 24ζ − 16 = 0 → ζ = 0.8453, V_R = Vs·√ζ = 0.9194 Vs).

**Pass criterion: simulated P and Rayleigh arrival times within ±1% of analytic values** at all receivers.

#### 2. Rayleigh wave Vz/Vr amplitude ratio

At the free surface (z = 0), the Rayleigh wave particle displacement eigenfunctions for ν = 0.25 are (Richart, Woods & Hall 1970; confirmed from Purdue lecture notes and the Brazil paper):

```
U(0) = horizontal/radial component ∝ (−1 + 0.5773) = −0.4226  (normalized to W(0))
W(0) = vertical component         ∝ (0.8475 − 1.4679) = −0.6204
```

These come from the eigenfunction formulas:
```
U(z) ∝ −exp(−0.8475 k z) + 0.5773 exp(−0.3933 k z)
W(z) ∝  0.8475 exp(−0.8475 k z) − 1.4679 exp(−0.3933 k z)
```
where k is the Rayleigh wavenumber and z is depth.

The amplitude ratio at the surface:
```
|W(0)| / |U(0)| = 0.6204 / 0.4226 ≈ 1.469
```

In the signed (time-domain) convention with retrograde elliptical motion, the vertical displacement leads the radial by 90° and is larger; the peak velocity ratio Vz_peak/Vr_peak matches the displacement ratio ≈ 1.47. In some literature this is quoted as −1.467 (the negative sign indicates the retrograde orbit convention where upward is negative).

**Pass criterion: simulated |Vz_peak/Vr_peak| within ±10% of 1.47** in the far-field Rayleigh window (time window around t_R, beyond 5 λ_R from source).

A ±10% tolerance accommodates:
- Grid dispersion at ~9–10 PPW with the stress-image method (~3–5% phase error → ~5% amplitude error)
- Finite-domain effects and imperfect windowing (~3%)
- Source near-field contamination at short ranges (exclude r < 5 m)

If the vacuum formulation is used, expect 10–20% amplitude error (marginal pass) → this is why the stress-image method is required for this criterion to be reliably met.

#### 3. No spurious ABL reflections

After t_R (Rayleigh arrival), the trace should decay smoothly. A spurious reflection from the ABL would appear as a secondary pulse at `t_refl = (2·L_ABL + r) / V_R` where `L_ABL` is the distance from source to sponge. **Pass criterion: no visible spurious pulse > 5% of Rayleigh peak amplitude in the post-Rayleigh tail.**

### Analytic reference code

Use `github.com/ktkimit/lamb_2dhalf_surface` (Python, NumPy/SciPy/Matplotlib). This solves the 2.5-D Lamb problem (line force → point force by Abel transform) using convolution with the analytic Green's function. The repository computes uz(t) and ur(t) at the surface for arbitrary receiver offsets, material parameters, and source wavelet. Steps:

```bash
git clone https://github.com/ktkimit/lamb_2dhalf_surface
cd lamb_2dhalf_surface
pip install numpy scipy matplotlib
# Edit parameters.py: set vp=520, vs=300, rho=2000, nu=0.25
# Set receiver offsets r = [5, 10, 15, 20, 25] m
# Set source wavelet to Ricker derivative at f0=20 Hz
python lamb_2dhalf_surface.py
```

Output: `uz_analytic.npy`, `ur_analytic.npy` → compare directly to Devito output. Compute time-domain cross-correlation and amplitude ratio.

**Note**: The ktkimit repository solves the 2D (line-force / plane-strain) problem by default. For a 3D point force comparison, the geometric spreading differs (1/r vs 1/√r), but the phase (arrival times) and the Vz/Vr amplitude ratio are identical (ratio is a material property, not a geometry factor). Use the same code for arrival time validation; compute amplitude ratio from the 3D Devito simulation and compare to the analytic value 1.47.

### SW4 cross-check

SW4 (Seismic Waves, 4th Order, LLNL/CIG, BSD license) uses a curvilinear free surface with 4th-order accuracy and has a built-in Lamb's problem test case in `examples/lamb/`:

```bash
git clone https://github.com/geodynamics/sw4
cd sw4 && mkdir build && cd build && cmake .. && make -j8
# Run the lamb example (no Python, uses input file lamb-1.in)
./sw4 ../examples/lamb/lamb-1.in
# Output: surface seismograms at multiple receivers
```

The SW4 Lamb example uses Vp=1.732 m/s, Vs=1.0 m/s, ρ=1.0 kg/m³ (non-dimensional); rescale to physical units or compare via normalized traces.

**Cross-check pass criterion**: Devito Rayleigh amplitude within **5% of SW4** for the same material parameters and source geometry. If discrepancy > 5%, the vacuum formulation is suspected → must implement the stress-image correction.

### Summary: validation decision tree

```
1. Run Lamb's problem with vacuum free surface.
   → Check P-arrival time: within 1%?   [usually yes — body wave, not affected by FS]
   → Check R-arrival time: within 1%?   [sensitive to FS; vacuum usually within 2–3%]
   → Check |Vz/Vr| ratio: within 10%?  [vacuum often fails this at 10 PPW]
   
   If ALL pass → vacuum is adequate for this scenario (unexpected but lucky).
   If amplitude fails → implement stress-image method.

2. Run Lamb's problem with stress-image free surface.
   → Repeat the same checks; expect all to pass.
   → Run SW4 on the same problem; Rayleigh amplitude within 5% of SW4?

3. If stress-image passes all criteria → proceed to house simulations.
   If stress-image also fails → call Salvus fallback.
```

---

## 6. Honest Risk and Effort Assessment

### Implementation effort

| Task | Effort | Notes |
|---|---|---|
| Vacuum free surface | 0.5 day | Just zero out top cells; test with a single sim |
| Stress-image free surface (H-formulation) | 2–4 days | SubDomain + ghost equations + debug loop |
| AFDA (adjusted FD coefficients) | +3–5 days | More accurate but harder to implement symbolically in Devito |
| Analytic Lamb comparison (ktkimit) | 0.5–1 day | Install + parameter matching + plotting |
| SW4 install + cross-check | 1–2 days | CMake build, possibly needs MPI; input file adaptation |
| **Total: vacuum + Lamb + SW4** | **~3–5 days** | Minimum viable validation |
| **Total: stress-image + full validation** | **~1–1.5 weeks** | Full validation including amplitude criterion |

These are engineering days (focused work), not calendar weeks with interruptions.

### Realistic failure modes

1. **Devito symbolic indexing difficulties** (moderate probability): The ghost-layer substitution `tau.subs(z, -j)` may not map cleanly onto Devito's indexed tensor dimensions. Workaround: do the ghost-cell copy in NumPy directly on `tau.data` at each time step inside a Python loop (slower but guaranteed correct; then move to Devito Eq for performance).

2. **Order-8 stencil needs 4 ghost rows but ABL only removed 2**: At space_order=8, the FD stencil for ∂τ/∂z at the surface requires 4 ghost values. This means the physical domain must NOT start the sponge until 4 cells below the surface. Verify by checking that no sponge damping coefficient is non-zero in the top 4 rows.

3. **Amplitude criterion marginal at 9 PPW**: Our 100 Hz / dx=0.15 m scenario gives ~9.2 PPW for the minimum Rayleigh wavelength (soil, λ_R = 275.8/100 = 2.76 m; 2.76/0.15 = 18.4 PPW — wait, this is wrong: λ_R at 100 Hz in soil Vs=150 m/s is λ_R = 0.9194×150/100 = 1.38 m → 1.38/0.15 = 9.2 PPW). The stress-image method at 10 PPW has ~2–3% phase error and ~5% amplitude error, which is within the ±10% pass criterion. Marginally OK; document this bias.

4. **SW4 build complexity**: SW4 requires MPI and optionally FFTW/SuperLU; on Windows/WSL it can take 2–4 hours to build. Alternative: use a Linux Docker image.

### Go/No-Go verdict

**GO on Devito with stress-image free surface**, subject to:
- Completing the Lamb's problem validation (P-arrival ±1%, Rayleigh arrival ±1%, |Vz/Vr| ±10%) before running any house simulations
- SW4 cross-check Rayleigh amplitude within 5%
- Documenting the residual ~5% amplitude bias in the Green's function library (uncertainty box in the publication)

**Trigger for fallback to Salvus** (apply for academic license now, in parallel, as insurance):
- Stress-image fails the amplitude criterion after 1.5 weeks of debugging
- Devito symbolic indexing prevents a clean SubDomain implementation and the NumPy-loop workaround is too slow (>10× per-step overhead)
- SW4 cross-check shows >10% discrepancy that cannot be traced to a coding bug

Salvus academic license: apply via [mondaic.com/get-salvus](https://www.mondaic.com/get-salvus) (Academic License Agreement at mondaic.com/ala.pdf). License is free for academic/research use (non-commercial). Typical response time 1–5 business days. Salvus provides a full Python API with SEM-based free surface (exact, not approximate), GPU support, and batch automation — equivalent to the Devito target but without the free-surface implementation burden.

---

## 7. Appendix: Key Numbers Reference Card

| Quantity | Value | Source |
|---|---|---|
| Rayleigh velocity / Vs for ν=0.25 | 0.9194 | Secular equation root ζ=0.8453 |
| V_R at Vs=150 m/s (soil) | 137.9 m/s | 0.9194 × 150 |
| λ_R at 100 Hz, soil | 1.38 m | 137.9/100 |
| PPW for λ_R at dx=0.15 m, 100 Hz | **9.2** | 1.38/0.15 |
| λ_R at 250 Hz, soil | 0.55 m | 137.9/250 |
| PPW for λ_R at dx=0.06 m, 250 Hz | **9.2** | 0.55/0.06 — same ratio! |
| |Vz/Vr| at surface, ν=0.25 | **1.47** | Eigenfunction derivation |
| Vacuum amplitude error at 10 PPW | ~10–15% | Bohlen & Saenger 2006; Mittet 2002 |
| Stress-image amplitude error at 10 PPW | ~5% | Kristek et al. 2002 |
| AFDA amplitude error at 6 PPW | ~1–2% | Kristek et al. 2002 |
| Ghost rows needed for order-8 | 4 | Stencil half-width = so//2 |
| Lamb analytic code | github.com/ktkimit/lamb_2dhalf_surface | Python, MIT |
| SW4 code | github.com/geodynamics/sw4 | BSD |

---

## 8. Key References

- **Kristek, Moczo, Archuleta (2002)**: "Efficient methods to simulate planar free surface in the 3D 4th-order staggered-grid finite-difference schemes." *Studia Geophysica et Geodaetica* 46, 355–381. [SpringerLink](https://link.springer.com/article/10.1023/A:1019866422821) · [SEG](https://library.seg.org/doi/10.1190/1.1512752) — THE reference for stress-image + AFDA in SSG.
- **Virieux (1986)**: P-SV wave propagation in heterogeneous media: velocity-stress FD method. *Geophysics* 51, 889–901. — The underlying staggered-grid scheme.
- **Levander (1988)**: Fourth-order finite-difference P-SV seismograms. *Geophysics* 53, 1425–1436. — First stress-imaging technique (Levander's H/W formulations).
- **Bohlen & Saenger (2006)**: "Accuracy of heterogeneous staggered-grid finite-difference modeling of Rayleigh waves." *Geophysics* 71, T109–T115. — Systematic vacuum vs stress-image accuracy comparison.
- **Mittet (2002)**: "Free-surface boundary conditions for elastic staggered-grid modeling schemes." *Geophysics* 67, 1616–1623. [SEG](https://library.seg.org/doi/10.1190/1.1512752) — Quantified vacuum accuracy deficiency.
- **Lamb (1904)**: "On the propagation of tremors over the surface of an elastic solid." *Phil. Trans. R. Soc. A* 203, 1–42. — The original Lamb's problem.
- **Pekeris (1955)**: "The seismic surface pulse." *PNAS* 41, 469–480. — Closed-form analytic solution for vertical point force.
- **Richart, Woods & Hall (1970)**: *Vibrations of Soils and Foundations*. Prentice-Hall. — Standard reference for Rayleigh eigenfunction formulas including U(0)/W(0) = −0.4226/−0.6204.
- **Devito BC tutorial**: [devitoproject.org/examples/userapi/04_boundary_conditions.html](https://www.devitoproject.org/examples/userapi/04_boundary_conditions.html) — `FSDomain` SubDomain pattern (acoustic); elastic adaptation follows same pattern.
- **Devito elastic tutorial**: [devitoproject.org/examples/seismic/tutorials/06_elastic.html](https://www.devitoproject.org/examples/seismic/tutorials/06_elastic.html)
- **SW4 code + Lamb example**: [github.com/geodynamics/sw4](https://github.com/geodynamics/sw4) — see `examples/lamb/`
- **ktkimit Lamb code**: [github.com/ktkimit/lamb_2dhalf_surface](https://github.com/ktkimit/lamb_2dhalf_surface) — Python analytic solution
- **Salvus (Mondaic)**: [mondaic.com/get-salvus](https://www.mondaic.com/get-salvus) — paid fallback; academic license via mondaic.com/ala.pdf
- **Corner instability / M-PML**: [GJI 2014, CFS-M-PML](https://academic.oup.com/gji/article/198/1/140/608458) — treat with multiaxial PML in corners or use sponge ABL to avoid instability.
