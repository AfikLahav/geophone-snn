> **Research document — house-localization pivot ("wallhacks").** Deep web research synthesis (Claude Sonnet, 2026-07-09). Covers SPECFEM3D sim-build gaps: foundation representation in the hex mesh, and domain/absorbing-boundary sizing. Design input; not a validated experiment. Indexed in [README.md](README.md).

---

# Foundation Representation and Domain/Buffer Sizing for the SPECFEM3D House Simulation

**Scope.** Two build-blocking gaps for the footstep-localization elastic-wave simulation:
- **Topic A** — How to model the house foundation in the all-hex voxel mesh (structure, dimensions, physics, mesh recipe).
- **Topic B** — How large to make the soil domain and which absorbing boundary condition to use.

Context: ~14 × 7 m house (Israeli residential concrete frame, 1–2 storeys), exterior geophones on or near the foundation/slab, footstep band 20–250 Hz, soil Vs = 150–500 m/s, existing all-hex mesh of slab + walls + roof on a soil block.

---

## TOPIC A — House Foundations

### A.1 Foundation Types for 1–2 Storey Israeli/Mediterranean Concrete-Frame Residential Buildings

**The regional context.** Israel has no frost depth — the ground never freezes, so the primary drivers of foundation depth are: (a) bearing capacity of the near-surface soil layer, (b) the minimum below which near-surface soils become stable (topsoil/fill removal), and (c) seismic code IS 413 (mandatory since 1980, revised 1994) which requires a tied, ductile lateral-force-resisting system. There is no thermal insulation requirement.

**Soil types.** Israel's coastal plain (Tel Aviv, Hadera) sits on Hamra (red sandy loam) and alluvial silts. The Negev and inland areas have loess and calcareous soils (Nari). The north has heavier clays (Vertisol). Hamra typical allowable bearing pressure: 100–200 kPa; Nari: 150–300 kPa. Neither demands particularly deep embedment — bearing-capacity failure governs at depths of 0.5–1.0 m, unlike frost countries where 1.2–1.5 m is legally mandated.

**Most common foundation system for 1–2 storey RC-frame house in Israel:**

| Foundation type | When used | Prevalence in 1–2 storey Israeli houses |
|---|---|---|
| **Strip/continuous footing under bearing walls** | Masonry load-bearing walls; older (pre-1980s) construction | Declining; still common in small detached houses |
| **Isolated pad footings + grade beam ring** | Modern RC column-frame buildings; IS 413 seismic tie | **Most common** for new 1–2 storey RC frame; columns on pads, grade beam connects them at ±grade |
| **Raft/mat slab (continuous RC slab, thickened)** | Poor or variable soil; when footing area > 50% of footprint; multi-storey on soft ground | Common for 3+ storey; used for 1–2 storey on soft coastal Hamra; increasingly common for simplicity |
| **Slab-on-grade (thin unreinforced or lightly reinforced)** | Ground floor of any system; not a structural foundation | Always present as the ground-floor slab; rides on the strip/pad system or on the raft |
| **Pile foundation** | Soft soils (liquefiable Hamra, deep alluvial); tall buildings | Not typical for 1–2 storey residential; used in specific poor-soil sites |

For the simulation context, the dominant system to model is: **isolated RC pad footings under each column, connected by a continuous reinforced grade beam (ring beam) around the building perimeter, with a slab-on-grade ground floor sitting inside this ring**. For soft coastal sites, a **raft slab** replaces the pad + grade beam: one continuous 300–400 mm RC slab under the entire footprint with downstand beams.

### A.2 Typical Dimensions (Israeli/Mediterranean 1–2 Storey RC Frame)

**Isolated pad footings:**
- Plan size: 800–1,200 mm × 800–1,200 mm (1–2 storey column loads; soil qa = 100–200 kPa)
- Thickness: 300–500 mm of RC
- Depth of soffit below finished grade: **0.5–0.8 m** (no frost constraint; soil competence governs; 300 mm depth minimum to sound bearing material, plus footing thickness)
- Top of footing typically at **−0.4 to −0.7 m** from finished floor level

**Grade beam (perimeter foundation beam):**
- Width: 200–300 mm (matches or slightly wider than wall above)
- Depth: 400–600 mm (structural, must span between pad footings)
- Top of grade beam: flush with, or slightly below, finished floor level (used as the ring that retains the slab-on-grade fill)
- Bottom: tied into pad footings; overall soffit at ~0.6–1.0 m below grade

**Raft/mat slab (alternative):**
- Slab thickness: 300–400 mm overall
- Downstand (thickened) beams under walls/columns: 600–900 mm deep × 300–400 mm wide
- Top of raft at −100 to −200 mm (slightly below finish floor to accept topping)
- Bottom of raft (soffit) at: **0.4–0.6 m below finish grade** for residential; deeper if soft-soil site

**Slab-on-grade (ground floor):**
- 150–200 mm RC (separate from raft); sits on compacted fill inside the grade-beam ring
- Not a structural element — not carrying column loads
- In the simulation, this is the **floor slab** that receives footstep impacts

**Summary of below-grade concrete extent:**

| Element | Depth of soffit below finish grade |
|---|---|
| Pad footing soffit | 0.5–1.0 m |
| Grade beam soffit | 0.6–1.0 m |
| Raft slab soffit (residential) | 0.4–0.8 m |
| Ground-floor slab soffit | ~0 m (on grade) |

### A.3 How the Foundation Couples Structure to Soil — Physics for the Localization Problem

**The energy path for a footstep.** A heel-strike on the ground-floor slab injects a broadband force impulse (20–250 Hz) into the slab. Energy leaves the footstep source and travels along three parallel paths to exterior geophones:

1. **Direct plate path (dominant, lowest loss):** Flexural (bending) waves propagate radially outward through the floor slab → junction into the foundation ring beam / grade beam → into the foundation wall / exterior wall → geophone mounted on the foundation stem wall or slab edge. This path stays entirely in concrete (impedance homogeneous), with loss only at wall-to-foundation junctions (~3–8 dB per junction). This is the primary signal carrier in the 50–250 Hz band.

2. **Foundation-to-soil radiation path:** Compressional and shear body waves are radiated from the underside of the foundation into the soil below. The grade beam / raft slab acts as a horizontal source, exciting primarily Rayleigh waves in the near-surface soil that propagate outward. Energy then re-couples into the structure from any soil-embedded element (foundation walls). This path suffers a foundation-soil coupling loss of **2–15 dB** (frequency-dependent, higher loss at high frequency), plus geometric spreading in the soil (1/r for Rayleigh, 1/r² for body waves in 3D). This is the secondary path — less efficient at carrying the 100–250 Hz content but dominant for geophone deployments in the soil rather than on the wall.

3. **Airborne path** (negligible for concrete-frame structures in the 20–250 Hz band).

**Why foundation type matters for the simulation:**

- **Grade beam + pad footings:** The grade beam creates a continuous concrete ring at foundation level. Energy in the floor slab flows efficiently into this ring (same concrete material, bonded junction) and then around the building perimeter. Exterior geophones on the grade beam wall are in the most direct path. Soil radiation is from pad footings (discrete; acts like point radiators at low frequencies).

- **Raft slab:** The entire underside of the building is a large vibrating plate in contact with soil. It radiates Rayleigh and body waves over the full footprint — much stronger coupling per unit area than pad footings. The raft effectively radiates in all directions, making the soil-path contribution larger. This improves signal at geophones placed in the soil but also increases the reverberant soil field that mixes back into the structure.

- **Impedance at the concrete-soil interface:** Concrete Vp ≈ 3,500–4,000 m/s, ρ ≈ 2,400 kg/m³ → Z_c = ρV_P ≈ 8.4–9.6 MRayl. Soil: Vs ≈ 150–500 m/s, Vp ≈ 300–1,000 m/s, ρ ≈ 1,500–2,000 kg/m³ → Z_soil (P) ≈ 0.45–2.0 MRayl. **Impedance ratio Z_c/Z_soil ≈ 5–20.** This large contrast means: (a) most incident wave energy in concrete reflects back at the interface (high reflectance R = ((Z_c − Z_soil)/(Z_c + Z_soil))² ≈ 0.5–0.7 for typical Hamra sand); (b) only 30–50% of energy transmits into soil per crossing; (c) the foundation effectively traps energy in the building structure (the building is a resonant cavity for structural waves), which is precisely what makes MFP/fingerprinting viable.

**Why this matters for Green's-function quality:** A correctly modeled foundation carries the impedance boundary that controls how much energy leaks from structure to soil. Without foundation elements (if the slab just "floats" at grade), the simulation has the slab bottom directly touching soil — an incorrect boundary that over-radiates energy and produces a wrong structural-wave reverberant field. Adding realistic foundation depth adds only ~0.8 m of concrete below grade but changes the transmission/reflection physics at the most critical interface.

**Frequency dependence of coupling loss:** Foundation-soil coupling loss is significant from **10–80 Hz** (the band where footstep energy is strongest). Above 80 Hz, the coupling loss grows rapidly. This means:
- Below 80 Hz: exterior soil geophones receive meaningful signal from the soil-radiation path (grade beam or raft acting as distributed source).
- Above 80 Hz: exterior geophones on the concrete wall/foundation face are far more efficient than soil geophones.
- Raft foundations show **lower coupling loss** (i.e., more energy into soil) than isolated footings; pile foundations show the highest isolation (most energy stays in the structure).

### A.4 How to Represent the Foundation in an Axis-Aligned Voxel/Hex Mesh

**Design constraint from the existing mesh.** The all-hex mesh uses axis-aligned elements; walls, slabs, roof are already represented as rectangular concrete blocks. The foundation is simply an extension of the concrete geometry downward from the existing ground-floor slab.

**Recommended concrete meshing recipe for the foundation:**

The simplest physically-faithful model is a **raft slab + downstand beams**. This is the correct choice for simulation for four reasons: (a) it is common for 1–2 storey Israeli RC-frame houses on Hamra/alluvial soil; (b) it is trivially representable as axis-aligned hex elements (a rectangular concrete plate below grade); (c) it creates the correct boundary condition (full-footprint concrete-soil interface) for the soil-radiation path; and (d) it avoids the need to model individual pad footings (which would require sub-element geometry at the 0.3–0.5 m scale that is already near the element size).

**Concrete mesh recipe (numbers for a 14 × 7 m house):**

```
Layer A — Ground-floor slab (already in mesh)
  Material:  concrete (Vp=3800, Vs=2200, ρ=2400 kg/m³, Q=100)
  Thickness: 0.20 m (200 mm RC slab)
  Top face:  z = 0.00 m (finish floor level)
  Bottom:    z = −0.20 m

Layer B — Raft slab / grade beam zone (NEW — extend existing mesh downward)
  Material:  concrete (same as above)
  Plan:      full house footprint 14.0 × 7.0 m  (axis-aligned)
  Thickness: 0.35–0.40 m (350–400 mm to represent both raft + downstand average)
  Top:       z = −0.20 m  (bonded to underside of ground-floor slab, no gap)
  Bottom:    z = −0.55 to −0.60 m  (soffit of raft/grade beam)
  → Total concrete depth below finish grade at foundation level: 0.55–0.60 m

Layer C — Foundation stem wall zone (optional; under exterior walls only)
  If exterior walls extend below grade as foundation walls:
  Material:  concrete
  Width:     match wall thickness (0.20–0.30 m exterior wall)
  Height:    0.30–0.40 m below Layer B soffit (i.e., to z = −0.85 to −1.0 m)
  Only along the four perimeter walls
  → Adds lateral embedding into soil; improves seismic tie simulation
```

**Simplified 2-layer recommendation (minimum faithful):**
- Extend the concrete slab geometry downward by **0.40 m** below the existing slab bottom, covering the full footprint. This represents the raft + grade beam as a single uniform 400 mm concrete layer at z = −0.20 m to z = −0.60 m.
- Set all elements in this zone to concrete material (Vp/Vs/ρ same as slabs above).
- Below z = −0.60 m: soil only.
- No need to model pad footings individually — the raft representation is correct at the simulation's spatial resolution.

**Element sizing.** At the existing mesh resolution (~0.05 m in concrete walls), the 400 mm foundation layer needs 8 elements through thickness at 0.05 m — acceptable. If the foundation zone uses the same element size as the slab (0.05 m), this adds 8 element layers × 14 × 7 / 0.05² = ~8 × 280 × 140 ≈ 314,000 hex elements for the foundation zone. This is a moderate addition (the full mesh is ~0.8–1.5 M elements per `00_what_specfem3d_needs.md`).

**Alternative (strip-footing model):** If you want to represent isolated pad footings + grade beam:
- Model the grade beam as a continuous 250 × 500 mm concrete strip around the four perimeter walls (extend exterior walls downward by 0.50 m, narrowed to wall width).
- Leave the interior as soil below the ground-floor slab.
- This is geometrically more accurate for the most common Israeli single-house type but has a lower-area concrete-soil interface → less soil radiation signal at exterior geophones for low-frequency energy.
- **Not recommended** as the default — it is harder to mesh (non-uniform below grade), and the raft is more common for multi-room 1–2 storey buildings.

**Material interface handling.** At the concrete-soil boundary (bottom of the foundation layer), SPECFEM3D requires conforming elements (shared faces between concrete and soil elements). Because the hex mesh is axis-aligned and the foundation bottom is flat (z = const), this is straightforward: the top row of soil elements shares faces with the bottom row of foundation elements. No special treatment is needed — SPECFEM3D handles elastic-elastic material discontinuities automatically at conforming element faces.

---

## TOPIC B — Domain Size and Absorbing Boundary Conditions

### B.1 SPECFEM3D Absorbing Boundary Options: Stacey vs. CPML

**Stacey (Clayton-Engquist paraxial) conditions:**
- Implemented as a local paraxial approximation: approximate absorbing condition on the boundary itself, no additional elements required.
- Efficient (no extra memory or elements) and computationally cheap.
- Works well for body waves at non-grazing incidence (typically <45° from boundary normal).
- **Significant failure mode:** At grazing incidence (waves traveling nearly parallel to the boundary), Stacey conditions produce large spurious reflections. Rayleigh waves — the dominant surface-wave mode in our near-surface soil — travel parallel to the lateral boundaries (grazing incidence relative to vertical walls). This is the key deficiency: **Stacey absorbs Rayleigh waves poorly**, with reflection coefficients of the order 10–30% or worse at grazing angles.
- In SPECFEM3D, this is controlled by `STACEY_ABSORBING_CONDITIONS = .true.` in Par_file. The SPECFEM3D documentation explicitly states: "In almost all cases it is much better to use CPML absorbing layers."
- **Stability:** Stacey is unconditionally stable for long simulations with elastic media (no divergence risk).

**CPML (Convolutional Perfectly Matched Layer):**
- A layer of elements (volume, not just a boundary condition) surrounding the computational domain on the lateral and bottom faces. The free surface (ground top) is NOT covered by PML — it remains a free surface (traction-free), which is the natural Neumann condition in SPECFEM3D.
- PML achieves near-zero theoretical reflection coefficient for waves at any angle of incidence. In practice, reflection coefficient R ≈ 10⁻³ (specified in the damping profile design).
- CPML specifically improves the classical PML at **grazing incidence**, which is critical for Rayleigh waves. The shifted-CPML variant reduces spurious reflections from surface/grazing waves by factors of 10⁴–10¹³ compared to classical non-shifted PML.
- CPML requires a layer of conforming, preferably regular (non-deformed cube) elements. In SPECFEM3D, these elements are tagged in the mesh and the code applies the CPML damping profile.
- Thickness: industry practice (SPECFEM2D example, Komatitsch & Martin 2007, OpenSWPC docs) is **10–20 grid points**, corresponding to roughly **1–3 element layers** at SPECFEM's larger element sizes. The recommendation from the Komatitsch (2003 GJI) paper is 2 spectral elements (each with 11 GLL points in the damping direction); the OpenSWPC package uses 10–20 FDTD grid points. For SPECFEM3D with SEM elements at 0.30 m (soil bulk) or 0.05 m (wall), **3–5 elements in the PML zone** (total PML thickness ≈ 0.9–1.5 m at soil resolution) is standard practice.
- **Key parameter: `f0_FOR_PML`** — must be set to the dominant source frequency. The SPECFEM3D documentation marks this as "VERY IMPORTANT": a mismatch causes PML instability. For our footstep simulation (source dominant frequency ~20 Hz per the Ricker-derivative STF centered at 20 Hz), set `f0_FOR_PML = 20.0`.

**PML/CPML known instabilities:**
- **Material heterogeneity inside PML:** CPML cannot handle velocity/density gradients along the direction normal to the PML boundary inside the PML layer. The velocity model must be 1D (constant or laterally homogeneous) within the PML zone. For our simulation, the soil properties in the PML zone should be uniform (extend the outermost soil column values laterally into the PML without gradients). This is straightforward since the PML zone is outside the building and contains soil only.
- **Viscoelastic / poroelastic material in PML:** SPECFEM3D does not support viscoelastic or poroelastic material inside PML elements. If attenuation (Q) is enabled, SPECFEM3D auto-converts PML elements to elastic. This causes slight mismatch at the PML interface but is "sufficient in practice" per the documentation.
- **Corner regions (PML-PML overlap):** At the 12 edges and 8 corners where two or three PML layers meet, the damping is applied in multiple directions simultaneously. CPML handles corners correctly (unlike split-field PML). No special treatment is needed in SPECFEM3D — the code manages corner elements automatically.
- **Long-simulation stability:** CPML can exhibit long-time instabilities if `f0_FOR_PML` is set incorrectly. With the correct frequency, shifted CPML is stable for 10⁵ time steps. M-PML (multiaxial PML) is even more stable for very long anisotropic simulations but is not the default in SPECFEM3D Cartesian. For our 0.5 s simulation (T_sim = 0.5 s, dt ≈ 10–50 μs → 10,000–50,000 time steps), CPML with correctly set f0 is stable.
- **Free-surface corner:** The intersection of the free surface (top) with the lateral PML walls is the most sensitive region. SPECFEM3D handles this by making the top face a free surface and starting the PML at the sides from z = 0 downward. The corner elements at top-side intersection are present in the CPML tag set. Research shows that CPML remains stable at free-surface corners for typical Poisson's ratios (ν ≈ 0.2–0.35) of soil and concrete. Deformed elements at corners can trigger instabilities — keep corner elements as regular cubes.

**Verdict for our elastic building sim:**
Use **CPML** (`PML_CONDITIONS = .true.`, `STACEY_ABSORBING_CONDITIONS = .false.`) for all lateral and bottom faces. Keep the top face as free surface. Set `f0_FOR_PML = 20.0` Hz. PML zone must contain soil only (no concrete foundation elements inside PML).

### B.2 How Much Soil Buffer Is Needed

**The physics problem.** When a boundary reflection occurs (because the PML is imperfect or the domain is too small), the spurious wave travels back across the domain, arriving at the exterior geophone after some travel time. If this arrival is within the recording window (0.5 s), it contaminates the Green's function. The domain must be large enough that either: (a) the PML absorbs residual reflections to machine zero before they re-enter, or (b) any residual reflection travels far enough that it arrives after the recording window ends.

**Rayleigh wave — the hardest case.** In our footstep band (20–250 Hz) with soil Vs = 150–500 m/s:
- Rayleigh wave phase velocity c_R ≈ 0.92 × Vs = 140–460 m/s
- At 20 Hz: λ_R = c_R / f = 7–23 m (soft to stiff soil)
- At 250 Hz: λ_R = 0.56–1.84 m
- The recording window is T_sim = 0.5 s

**Round-trip contamination time.** For a geophone at distance d_geo from the edge, and domain lateral padding L_pad beyond the geophone:
```
t_return = (L_pad + L_pad) / c_R = 2 × L_pad / c_R
```
We need t_return > T_sim:
```
L_pad > T_sim × c_R / 2 = 0.5 × (140–460) / 2 = 35–115 m
```
This "pure distance" approach without PML would require enormous domains. With CPML, the effective reflection coefficient from the PML is R ≈ 10⁻³ or better, so we only need the PML to reduce the residual to below the noise floor of the simulation (typically 10⁻⁶ of the incoming amplitude). The two methods (pure distance + CPML) together set the required buffer.

**Wavelength-based rule from PML literature.** For CPML with theoretical R = 10⁻³, the required PML thickness is approximately **0.5–1.0 × λ_min**, where λ_min is the shortest wavelength in the medium (S-wave: λ_S = Vs/f_max). However, this rule applies to the PML layer itself, not to the soil buffer between the building and the PML. The literature practice (Komatitsch & Martin 2003, 2007; OpenSWPC; the Frontiers SSI study at 243 × 243 × 24 m for a 4–50 Hz problem) consistently shows that the total distance from the structure edge to the PML boundary should be **≥ 1–3 longest Rayleigh wavelengths** at the minimum frequency.

At f_min = 20 Hz, λ_R,max = 7–23 m. For a conservative 2× longest wavelength: **L_pad = 14–46 m** lateral from the building exterior wall to the PML inner face. For the stiffest expected soil (Vs = 500 m/s, λ_R = 23 m), 2λ gives 46 m — too large for a cost-efficient simulation. For the softer (but more typical Israeli coastal Hamra, Vs ≈ 150–300 m/s), λ_R = 7–14 m → 14–28 m buffer is achievable.

**FEM/SSI practice.** The soil-structure interaction literature places the domain lateral boundary at 3× building width from the building center: for our 14 m wide house, 3 × 14/2 = 21 m lateral from the house axis → ~14 m from the exterior wall. One major SSI study used 243 × 243 m for a 51 m building at 4.5 Hz, giving a ratio of ~2.4× the maximum expected wavelength. Another rule stated: distance from structure boundary = soil layer depth for equivalent unbounded problem.

**Depth below foundation.** Downgoing body waves (P, S) and evanescent Rayleigh tails are the concern at the bottom. At 20 Hz, Vs = 150 m/s → λ_S = 7.5 m. The Rayleigh wave is exponentially damped below one wavelength (amplitude ≈ e⁻¹ at depth λ_R). So significant Rayleigh energy is within the top 7–23 m of soil. Body waves radiated downward need to not reflect back in the recording window. For the source at floor level (z = 0), a direct P wave radiated straight down at Vp = 300–800 m/s reaches a bottom boundary at depth D_b in time t = D_b/Vp, and reflects back in 2D_b/Vp. For this round-trip to exceed 0.5 s: D_b > 0.25 × 300 = 75 m (soft) or 0.25 × 800 = 200 m (stiff). This is again too large for pure distance. With CPML at the bottom, much of the downward energy is absorbed — the buffer below foundation only needs to ensure the CPML has sufficient distance from any structure-modified soil region. In practice, **D_depth = 10–15 m below foundation** is sufficient with CPML.

**The recording-window trick.** Even without perfect PML, if a spurious reflection takes longer than T_sim = 0.5 s to arrive at the geophone, it is outside the recording window and does not corrupt the Green's function. For Rayleigh waves at Vs = 150 m/s (slowest expected): time for reflection to travel from geophone → PML boundary → back to geophone: with L_pad = 20 m lateral buffer, t = 2 × 20 / 140 = 0.29 s. This is within the 0.5 s window. To use the recording-window trick alone, you need L_pad ≥ c_R × T_sim / 2 = 35 m (soft soil). With CPML reducing the reflection by 10³, the effective "virtual distance" the residual must travel to become negligible is much shorter.

**Summary table — buffer sizes for our simulation:**

| Soil class | Vs (m/s) | c_R (m/s) | λ_R at 20 Hz | λ_S at 250 Hz | Recommended lateral pad | Recommended depth below foundation |
|---|---|---|---|---|---|---|
| Soft (coastal Hamra) | 150 | 138 | 6.9 m | 0.6 m | **15–20 m** | **8–10 m** |
| Medium (alluvial clay) | 250 | 230 | 11.5 m | 1.0 m | **20–25 m** | **10–12 m** |
| Stiff (Nari calcareous) | 500 | 460 | 23 m | 2.0 m | **30–40 m** | **12–15 m** |
| Simulation default (conservative) | 200–300 | 185–275 | 9–14 m | 0.74–1.1 m | **25 m** | **12 m** |

**Element count implication.** At soil resolution h_soil = 0.30 m (required for 250 Hz with Vs = 150 m/s: λ_min = 0.6 m → h = 0.6/3 = 0.20 m; at Vs = 300 m/s: λ_min = 1.2 m → h = 0.40 m; target h_soil = 0.20–0.30 m):
- Domain plan: (14 + 2×25) × (7 + 2×25) = 64 × 57 m ≈ 3,648 m² plan area
- Depth: 3.5 m (house height) + 0.6 m (foundation) + 12 m (soil depth below foundation) = 16.1 m total depth
- Total volume: 3,648 × 16.1 ≈ 58,700 m³
- Elements at h = 0.30 m (soil bulk): 58,700 / 0.027 ≈ 2.2 M elements (soil only)
- Concrete elements (walls, slabs, foundation): ~0.5–0.8 M elements at 0.05 m
- **Total: ~2.5–3 M elements** for the 25 m lateral pad + 12 m depth configuration

This is a large but manageable mesh for a GPU-equipped workstation or modest cloud node (16–32 GB GPU VRAM at NGLL=5). Using the multi-resolution meshing strategy (0.05 m concrete, 0.20–0.30 m soil, coarsening with depth), the actual element count is closer to 1.5–2 M with doubling-brick transitions.

### B.3 Interaction: CPML + Buffer Together

**The correct mental model:** CPML and physical buffer are complementary, not alternatives.
- CPML is the primary absorber. A well-configured CPML (R = 10⁻³) reduces the outgoing wave amplitude by 10³ before it even reaches the PML outer face.
- The physical soil buffer between the building and the CPML inner face provides:
  1. **Wave attenuation by geometric spreading:** Rayleigh wave amplitude ∝ 1/√r in 2D (cylindrical spreading); body waves ∝ 1/r. At 25 m from the building exterior, Rayleigh wave amplitude is reduced by additional factor √(r_geo/r_PML) ≈ √(3/25) ≈ 0.35 relative to at the first exterior sensor (assuming sensor at 3 m from wall).
  2. **Material attenuation:** Even with Q = 30–50 for soil (typical for alluvial deposits), attenuation at 50 Hz over 25 m is exp(−π × 50 × 25 / (Q × 140)) ≈ exp(−1.4) ≈ 0.25. Modest but not negligible.
  3. **Time margin for the recording-window trick:** With 25 m of buffer and c_R = 200 m/s, round-trip time to PML boundary = 2 × 25 / 200 = 0.25 s. Even a residual reflection that leaks back from the PML must traverse another 25 m + sensor distance before arriving. Total round trip ≈ 0.5–0.7 s — right at or beyond the recording window.

**Trade-off against compute cost:** Every 5 m of additional lateral buffer adds ~5/0.30 × 57/0.30 × 16.1/0.30 × 5 ≈ 300,000 additional soil elements per side. The recommended 25 m pad is the practical compromise: CPML absorbs >99.9% of outgoing energy, the 25 m buffer provides another ~0.35 amplitude reduction, and the recording-window trick covers any residual.

**If compute is tight (RTX 3090 with <24 GB VRAM):** Reduce to **15 m lateral pad** with CPML. At 15 m, Rayleigh wave round-trip time = 2 × 15 / 200 = 0.15 s — reflection arrives at 0.15–0.25 s into the 0.5 s window. At this shorter pad, the CPML must work harder. Ensure `f0_FOR_PML = 20.0`, regular cubic elements in PML zone (no distortion), and 3–5 element layers in the PML. Validate with a sensitivity test: run 10 m vs 15 m vs 25 m pad and check that the recorded geophone waveforms are identical within numerical precision for the first 0.2 s (before any possible reflection arrives even from the smallest domain).

---

## TOPIC A — Concrete Recommendations

**Foundation representation — use this recipe:**

Extend the existing concrete geometry **0.40 m below the current ground-floor slab** across the entire building footprint (14 × 7 m plan), creating a uniform raft slab layer at z = −0.20 m to z = −0.60 m. Material: concrete (Vp = 3800 m/s, Vs = 2200 m/s, ρ = 2400 kg/m³, Q_p = 100, Q_s = 50). Below z = −0.60 m: soil. This raft layer:

1. Is geometrically trivial (axis-aligned rectangular block, no new complexity).
2. Correctly places the structural concrete-to-soil impedance boundary 0.6 m below grade (consistent with Israeli 1–2 storey practice).
3. Provides the full-footprint concrete-soil interface that controls both the energy trapping in the structure (the resonant cavity) and the soil-radiation coupling efficiency.
4. Requires approximately 8 element layers at h_concrete = 0.05 m, adding ~200,000–300,000 hex elements.

Do NOT model pad footings individually — the raft is the correct approximation at mesh resolution, and is the more common system for multi-room houses on Hamra/alluvial soils.

---

## TOPIC B — Concrete Recommendations

**Absorbing BC: use CPML.** Set in Par_file:
```
PML_CONDITIONS = .true.
STACEY_ABSORBING_CONDITIONS = .false.
f0_FOR_PML = 20.0   # dominant source frequency in Hz
```
Ensure PML elements are regular (non-deformed) cubes, containing soil material only (no concrete in PML zone). Use 3–5 element layers in the PML zone at soil resolution (≈ 0.9–1.5 m PML thickness at h_soil = 0.30 m). Do not apply PML to the top face — leave as free surface.

**Buffer sizing:**
```
Lateral pad:        25 m from building exterior wall to PML inner face
  (on all 4 sides; extendable to 30 m for stiff soil Vs > 400 m/s)
Depth below foundation: 12 m below the foundation soffit (z = −0.60 m)
  → total soil depth: 12.6 m below finish grade
```

For the 14 × 7 m house:
- Domain plan (inside PML): (14 + 50) × (7 + 50) = 64 × 57 m
- Domain depth: 0.60 m (foundation) + 12 m (soil) + 3.5 m (house height above grade) ≈ 16 m total
- CPML zone adds another ~1.5 m (5 elements × 0.30 m) on all 5 non-free sides

**If compute is constrained:** Reduce to 15 m lateral pad but validate with a sensitivity test (waveforms should be identical within 0.2 s recording window).

---

## Summary of Sources

- SPECFEM3D documentation — absorbing BCs and CPML guidance: [specfem3d.readthedocs.io](https://specfem3d.readthedocs.io/en/latest/04_creating_databases/), [specfem3d.readthedocs.io/mesh_generation](https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/)
- Komatitsch & Tromp (2003) PML for 2nd-order seismic wave equation — GJI 154(1):146: [academic.oup.com](https://academic.oup.com/gji/article/154/1/146/603527) — 2 spectral elements PML, R = 10⁻³ design, Rayleigh wave absorption, no instabilities at ν=0.38
- Martin, Komatitsch et al. (2009) CPML improved at grazing incidence: [academic.oup.com](https://academic.oup.com/gji/article/179/1/333/738594) — shifted CPML reduces spurious energy by factor 25,000× vs non-shifted; stable to 10⁵ time steps
- Festa & Nielsen (2003) surface wave interaction with absorbing boundaries: [agupubs.onlinelibrary.wiley.com](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2005GL024091) — Rayleigh waves handled well by PML; paraxial/Stacey has large grazing-incidence reflection
- OpenSWPC documentation — 10–20 grid points for PML thickness: [earth-planets-space.springeropen.com](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-017-0687-2)
- Peng & Toksöz (2014) M-PML — long-time stability for elastic wave: [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0926985113002838)
- Frontiers — Numerical modeling and validation of earthquake SSI 12-storey building (Ventura, CA): [frontiersin.org](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2023.1249550/full) — domain 4–13× building width; soil depth matching simulation depth
- ScienceDirect — coupling loss at building foundation 10–80 Hz (2–15 dB range): [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0141029616302917) — raft foundation shows lowest coupling loss (most energy to soil); pile highest isolation
- theconstructor.org / gharpedia — minimum foundation depth 0.5 m practical minimum, clay 0.8–1.2 m: [theconstructor.org](https://theconstructor.org/geotechnical/foundation-construction/1392/), [gharpedia.com](https://www.gharpedia.com/blog/minimum-depth-of-foundation-for-new-home/)
- MDPI SSI damping review — radiation damping 20–50% effective; frequency-dependent impedance functions: [mdpi.com](https://www.mdpi.com/2571-631X/8/1/5)
- BigRentz/BuildConstruct — raft foundation dimensions 150–300 mm typical, up to 400 mm residential: [build-construct.com](https://build-construct.com/structural-engineering/raft-foundations/)
- IsraeliCode IS 413 seismic requirements (mandatory 1980): [iisee.kenken.go.jp](https://iisee.kenken.go.jp/net/seismic_design_code/israel/israel.htm)
- Times of Israel — Israeli buildings and earthquake risk (seismic design standard background): [timesofisrael.com](https://www.timesofisrael.com/israeli-buildings-face-major-earthquake-risk-despite-efforts-to-upgrade-them/)
- GJI 2007 Komatitsch & Martin — unsplit CPML improved at grazing incidence for seismic equation: [geophysics.geoscienceworld.org](https://pubs.geoscienceworld.org/seg/geophysics/article-abstract/72/5/SM155/308300/An-unsplit-convolutional-perfectly-matched-layer)
- Train-induced floor vibration and structure-borne wave study: [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0141029622000724) — energy path via columns; impedance at junctions
