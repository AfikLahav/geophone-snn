> **Research survey — house-localization pivot ("wallhacks").** Background research agent (Sonnet), 2026-07-08. Literature/tooling survey; design input, not a validated experiment. Indexed in [README.md](README.md).

---

# Procedural 3-D Voxel House Generation for Devito FDTD Simulation

**Scope.** This document covers the complete geometry pipeline: where floorplan data comes from, how layouts are generated procedurally at scale, how a 2-D floorplan becomes the 3-D per-cell material arrays (λ/μ/ρ) that Devito's voxel-FDTD consumes, Israeli construction parameterization, and how many houses are needed and which variation axes matter most.

Builds on: [doc 03](03_house_architecture_datasets.md) (datasets), [doc 06](06_sim_construction_cost.md) (Devito cost model, 100 Hz → dx=0.15 m), [doc 07](07_sim_engine_verification_and_plan.md), [doc 08](08_free_surface_implementation.md).

---

## 1. Geometry Sources

### 1.1 Real Footprints — OpenStreetMap and Derivatives

**OSMnx** (Python, BSD) downloads OSM building polygons into GeoPandas GeoDataFrames in a single function call:

```python
import osmnx as ox
gdf = ox.features_from_place("Tel Aviv-Yafo, Israel", tags={"building": ["residential", "apartments", "house"]})
```

What you get: Shapely Polygon footprint, `building:levels` (integer), sometimes `height` (metres). The OSM Israel building dataset on HDX contains ~752 K footprints under the ODbL licence (commercial-compatible). Overture Maps (1.39 B global buildings, Parquet/GeoJSON, ODbL) adds estimated height where OSM is missing it. Neither source provides interior layout.

**Practical use.** Real footprints anchor the outer boundary of each simulated house to an Israeli statistical distribution of shapes (area, aspect ratio, L/U/irregular plan) without inventing them. A sample of 500–1 000 distinct Israeli residential footprints supplies the outer shell; the interior layout is synthesised separately.

**Limits.** OSM footprints frequently mis-classify apartment blocks and single-family villas. Filter: `building:levels` ≤ 5, footprint area 60–250 m², polygon convexity > 0.7 (roughly rectangular) to exclude high-rise towers.

### 1.2 Real Floorplan Datasets as Statistical Priors

Three sources (ranked) provide the interior layout statistics needed to drive a parametric generator. Meshes from these sets are **not** used directly — they are mined for priors.

| Source | N plans | Key content | Relevance to Israeli concrete | Licence |
|---|---|---|---|---|
| **RPLAN** [USTC DeepLayout] | 80,788 Chinese RC apartments | 256×256 raster, 13 room types, footprint 63–201 m² (mean 102 m²), mean 2.48 bedrooms | High — RC frame construction identical to Israeli dominant type | Non-commercial research |
| **ResPlan** [arXiv 2508.14006] | 17,000 plans | Vector JSON: Shapely polygons per room type, `wall_depth` explicitly stored (typically 0.15 m), door/window polygons, room-adjacency graph | Moderate — region unstated; wall_depth field is uniquely valuable | Permissive open-source |
| **CubiCasa5K** [GitHub] | 5,000 Finnish | SVG vector with room, wall, door, window polygons | Moderate-high — Nordic masonry/RC; only fully MIT-licensed vector set | MIT |

From RPLAN extract: footprint-area CDF, bedroom-count histogram, room-type frequency, room-aspect ratios.
From ResPlan extract: `wall_depth` distribution (→ confirms 0.15 m interior), room areas per type, door widths, adjacency-graph structure (which rooms border which).
From OSM Israel extract: footprint area and storey-count distributions specific to the Israeli market.

### 1.3 Neural Layout Generators

These produce synthetic floorplans on demand from a bubble diagram (adjacency graph). They are useful for generating layout *diversity beyond* what the static datasets contain.

**House-GAN / House-GAN++** [arXiv 2003.06988, ECCV 2020]: GAN conditioned on a graph (nodes = room types, edges = adjacency). Input: noise vector per room + Conv-MPN. Output: axis-aligned bounding boxes per room. Trained on RPLAN. Generates diverse layouts in milliseconds. Limits: bounding-box output is approximate (IoU ~0.65 vs ground truth); needs post-processing to build watertight wall polygons.

**Graph2Plan** [arXiv 2004.13204]: retrieves nearest RPLAN graph then refines bounding boxes via GNN. More faithful to the training distribution; less generative diversity than House-GAN.

**HouseTune / GFLAN (2024–2025)**: newer LLM-assisted methods that can also generate layouts from text prompts or room-count specs; output formats vary. Less critical for our use — we need geometry, not text-to-plan.

**Assessment.** For this project, House-GAN++ provides the cheapest path to synthetic layout diversity, but the bounding-box output still needs a wall-polygon synthesis step. If diversity is the goal, the parametric generator below is simpler to audit and control.

### 1.4 Parametric / Procedural Generator (Recommended Primary)

A hand-written parametric generator is more transparent, faster to audit for physical correctness, and avoids neural artifacts (rooms that overlap, zero-area rooms, disconnected graphs). The approach:

1. **Sample footprint** from OSM-Israel CDF (or Gaussian: area ~N(110, 20²) m², aspect ratio ~N(1.4, 0.3²), rectangular or simple L-shape).
2. **Sample room program** from RPLAN histogram: N_bed ~ Cat({1:5%, 2:55%, 3:35%, 4:5%}), plus living, kitchen, bathrooms, corridors.
3. **Lay out rooms** via a Constrained Growth / squarified-treemap algorithm: recursively partition the footprint rectangle into sub-rectangles proportional to required room areas. No GNN needed; runs in milliseconds in pure Python (NumPy/Shapely). See: [Constrained Growth GAMEON'10, TU Delft].
4. **Place doors** at shared wall midpoints (one door per room-pair in the adjacency graph); **place windows** on exterior walls, one per room except bathrooms, centred.
5. **Assign materials** from §3.

This generator produces watertight, analytically clean geometry with zero staircase ambiguity in wall placement, which is the primary requirement for the voxelizer in §2.

### 1.5 CGA/CityEngine Shape Grammars and Blender

**CityEngine CGA** (Esri) is a rule-based parametric modeller: footprint → 3-D building via split/extrude/texture rules. It produces full 3-D exterior geometry including roof type, facade division, floor heights. Useful for *exterior* appearance and roof geometry (gabled vs flat), which affects the top boundary condition. However CityEngine is proprietary (commercial licence required) and its output is mesh-only, needing voxelization. It is **not recommended** as the primary interior pipeline.

**Blender** with Geometry Nodes or add-ons (Prokitektura, the wojtryb procedural generator) can extrude walls and cut openings from a floor-plan polygon. **Infinigen Indoors** (Princeton, Blender-based, open-source, 2024–2025) generates complete photorealistic interiors including architectural elements. However Blender's output is a mesh (.blend, .obj, .fbx), and converting a mesh with furniture/objects to a clean structural-only voxel grid requires extensive mesh cleanup. For our application — structural elastic simulation where only walls/slabs matter — the parametric generator in §1.4 is simpler and cleaner.

**Conclusion on geometry sources.** The recommended stack: OSM-Israel footprints (outer boundary statistical distribution) + RPLAN/ResPlan priors (room-layout statistics) → parametric generator (§1.4) → voxelizer (§2). Neural generators (House-GAN++) serve as a diversity augmentation layer if the base set proves too homogeneous.

---

## 2. Floorplan → 3-D Voxel Conversion

### 2.1 Coordinate and Resolution Choices

At 100 Hz (dx = 0.15 m per doc 06), a 150 mm interior hollow-block wall occupies exactly **1 voxel**. That is the minimum representation that is physically distinguishable from soil. This is acceptable because the wall's transmission/reflection coefficient at 100 Hz depends primarily on acoustic impedance (ρ·Vp) and layer thickness relative to wavelength — and λ_concrete ≈ 3600/100 = 36 m >> 0.15 m — so the 1-voxel wall behaves as a thin impedance boundary, which is physically correct.

At 250 Hz (dx = 0.06 m), the same 150 mm wall occupies **2.5 voxels** (rounds to 2 or 3). Exterior 270 mm walls → 4–5 voxels; RC columns 200 mm → 3 voxels. This is a reasonable representation.

Rule: **each structural element must span at least 1 voxel.** At dx = 0.15 m, a 150 mm wall is right at the limit. Use dx = 0.10 m (intermediate case: f_max = 150 Hz, Vs_min = 150 m/s, PPW = 10) if thin-wall fidelity is the priority.

Voxelization of FDTD head-related transfer functions (acoustic; JASA 2016) showed that when wall thickness < 1 voxel, errors in the simulated field increase significantly. The minimum is 1 voxel per structural layer.

### 2.2 Rasterization Pipeline (2-D per floor)

**Step 1 — Set up the 2-D grid.**

```python
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import mapping

dx = 0.15  # m per voxel, 100 Hz scenario
# house bounding box in metres, padded by soil buffer
x0, y0, x1, y1 = 0.0, 0.0, 12.0, 10.0
nx = int((x1-x0)/dx) + 1
ny = int((y1-y0)/dx) + 1
transform = rasterio.transform.from_bounds(x0, y0, x1, y1, nx, ny)
```

**Step 2 — Rasterize room interiors (interior air / floor cavity).**

```python
# rooms is a list of (shapely_polygon, room_id) pairs
room_mask = rasterize(
    [(mapping(poly), room_id) for poly, room_id in rooms],
    out_shape=(ny, nx), transform=transform,
    fill=0, all_touched=False, dtype=np.int16
)
# room_mask[i,j] = room_id if inside a room, 0 otherwise
```

**Step 3 — Derive wall mask by difference.**

The wall polygons come directly from the generator (ResPlan-style: each room polygon is already the interior face; the wall is the buffer zone between adjacent rooms). Alternatively, derive walls analytically: the footprint exterior polygon buffered inward by wall thickness.

```python
from shapely.ops import unary_union

# footprint exterior, buffered inward by exterior wall thickness
exterior_wall_ring = footprint.buffer(-exterior_t, join_style=2)
# interior walls: each room shrunk by half interior wall thickness
interior_void = unary_union([r.buffer(-interior_t/2, join_style=2) for r in room_polys])

wall_zone = footprint.difference(interior_void)  # everything that is wall
wall_mask_2d = rasterize([(mapping(wall_zone), 1)], out_shape=(ny,nx), transform=transform,
                          fill=0, all_touched=True, dtype=np.uint8)
```

`all_touched=True` is critical here: it ensures thin walls register at least one voxel wide even when the wall polygon is sub-voxel in one direction.

**Step 4 — Identify RC columns/shear walls.**

Columns are inserted at corners of the structural frame (typically at 3–5 m grid spacing in Israeli RC construction). They are small rectangles (0.20 × 0.40–0.50 m) overwriting part of the wall mask with `MAT_RC_COLUMN`.

**Step 5 — Extrude to 3-D.**

```python
# material_ids[z, y, x] — full 3-D volume
Nz = int(total_height / dx)
material_ids = np.zeros((Nz, ny, nx), dtype=np.uint8)

for z in range(Nz):
    h = z * dx
    if h < slab_thickness:                        # floor slab
        material_ids[z] = np.where(footprint_mask, MAT_RC_SLAB, MAT_SOIL)
    elif h < slab_thickness + clear_height:       # storey interior
        material_ids[z] = np.where(wall_mask_2d, MAT_HOLLOW_BLOCK, MAT_AIR)
        material_ids[z][column_mask] = MAT_RC_COLUMN
    elif h < slab_thickness + clear_height + slab_thickness:  # ceiling slab
        material_ids[z] = np.where(footprint_mask, MAT_RC_SLAB, MAT_SOIL)
    # repeat for each storey
```

For multi-storey houses, the per-floor layout can differ: repeat steps 2–4 per floor with a (possibly different) floor plan, stacked vertically. ResPlan notes that multi-floor homes appear as separate per-floor plans, so linking floors (placing stairs, matching exterior walls) is a generator responsibility.

**Step 6 — Openings (doors and windows).**

Doors: zero out (set to AIR) the wall voxels spanning the door width (typically 0.80–0.90 m = 5–6 voxels at dx=0.15 m) at the sill-to-lintel height range.
Windows: similarly, but only in the height band sill_height (0.90 m) to head_height (2.10 m).

```python
for door in doors:
    ix0, ix1, iy0, iy1 = door_voxel_indices(door, transform, dx)
    material_ids[z_sill:z_lintel, iy0:iy1, ix0:ix1] = MAT_AIR
```

**Step 7 — Soil fill and PML sponge.**

Everything outside the footprint at storey height is MAT_SOIL. Below foundation and around perimeter: MAT_SOIL. A separate `damp` array (float32, same shape) is filled with a sponge ramp in the outer ~10 voxels.

### 2.3 Material → Devito Parameter Arrays

```python
# Material property table (scalar approximation; no anisotropy)
# lam = K - 2/3*mu; mu = rho*Vs^2; lam = rho*(Vp^2 - 2*Vs^2)
MATERIALS = {
    MAT_AIR:          dict(Vp=343,  Vs=0,    rho=1.2),    # near-vacuum (set mu~0)
    MAT_RC_SLAB:      dict(Vp=3800, Vs=2000, rho=2400),
    MAT_RC_COLUMN:    dict(Vp=3800, Vs=2000, rho=2400),
    MAT_HOLLOW_BLOCK: dict(Vp=2200, Vs=1000, rho=1100),   # see §3
    MAT_SOIL:         dict(Vp=500,  Vs=200,  rho=1800),   # Israeli loam/sandy; calibrate
}

lam_arr = np.zeros(grid_shape, dtype=np.float32)
mu_arr  = np.zeros(grid_shape, dtype=np.float32)
rho_arr = np.zeros(grid_shape, dtype=np.float32)

for mat_id, props in MATERIALS.items():
    mask = (material_ids == mat_id)
    rho_arr[mask] = props['rho']
    mu_arr[mask]  = props['rho'] * props['Vs']**2
    lam_arr[mask] = props['rho'] * (props['Vp']**2 - 2*props['Vs']**2)

# Then assign to Devito Function objects (as in doc 06 code)
lam.data[:] = lam_arr
mu.data[:]  = mu_arr
rho.data[:] = rho_arr
```

**Air cavities**: in a stress-velocity Virieux staggered-grid scheme, setting μ = 0 and λ small (but non-zero) inside rooms is the standard "near-vacuum" approach. Vp_air ≈ 343 m/s is slower than concrete by 10×, so it does not worsen the CFL condition. Setting λ=ρ(Vp_air²) and μ=0 gives a pure acoustic air cavity — adequate for this application because footstep energy at the floor is primarily structural (S-wave/flexural), not airborne.

### 2.4 Thin-Wall Representation Warning

At dx = 0.15 m the 150 mm hollow-block interior wall is exactly 1 voxel wide. This means:
- The wall has the correct impedance product (ρ·Vp) if the material parameters are right.
- Transmission/reflection at such a wall at 100 Hz is dominated by the impedance contrast (not resonance of the wall itself — resonance frequency of a 0.15 m wall in S-wave is Vs/(2t) ≈ 1000/(0.3) ≈ 3 kHz, far above our band).
- Quantitative risk: acoustic FDTD studies of thin walls (1–2 voxels) find amplitude errors of 5–15% relative to analytical solutions. This is acceptable for training-data diversity; it is **not** acceptable for a precision validation benchmark. Use dx = 0.06 m (250 Hz scenario) for validation runs.
- `all_touched=True` in rasterize is essential: with `all_touched=False`, a 1-voxel wall may vanish if the polygon centerline falls between grid cell centers.

### 2.5 Multi-Storey Stacking

Each storey adds one floor slab (height `slab_thickness`), one clear-height band (height `clear_height`), and one ceiling slab. For a 2-storey house:

```
z = 0 to slab_thickness           → RC floor slab (foundation/ground floor)
z = slab_thickness to slab+H1     → storey 1 walls + air
z = slab+H1 to slab+H1+slab       → intermediate RC slab
z = slab+H1+slab to slab+H1+slab+H2 → storey 2 walls + air
z = ... to top+slab               → roof slab (flat RC, common in Israel)
```

Standard Israeli flat-roof construction: no pitched-roof attic; the roof slab IS the top boundary. Roof slab thickness ≈ 200 mm (same as floor slab).

---

## 3. Israeli Construction Parameterization

### 3.1 Wall Thickness and Type (from doc 03 + field standards)

Hardcode these as the central values; then sample with ±σ for diversity:

| Element | Central (mm) | Range (mm) | Notes |
|---|---|---|---|
| Interior hollow-block partition | 150 | 120–180 | Typical 10 cm or 15 cm block + 10–20 mm plaster both sides |
| Exterior hollow-block envelope | 200 | 180–220 | Sometimes 25 cm block; add 10–30 mm insulation + plaster → total 260–310 mm |
| RC shear wall | 200 | 180–250 | Used at stairwells and elevator shafts |
| RC column cross-section | 200×400 | 200×350 to 250×500 | Spaced 3.5–5 m on structural grid |
| RC floor/roof slab | 200 | 180–220 | Flat slab (common) or 250–300 mm waffle; use flat for the generator |
| Clear ceiling height | 2,750 | 2,700–2,850 | Floor-to-ceiling; floor-to-floor ≈ clear + slab = 2,950–3,100 mm |

### 3.2 Elastic Wave Parameters for Israeli Materials

**Hollow concrete block (infill, non-structural):**
- Block alone: ρ ≈ 1,000–1,200 kg/m³, Vp ≈ 2,000–2,500 m/s, Vs ≈ 900–1,100 m/s (from ASCE measurements of similar CMU; dynamic modulus E_dyn ≈ 5–10 GPa).
- With plaster fill: effective medium ρ ≈ 1,100 kg/m³, Vp ≈ 2,200 m/s, Vs ≈ 1,000 m/s. (Use these as nominal.)
- Hollow-block masonry E_dyn rises ~11% from 12 → 102 MPa confining pressure; at typical in-situ wall loads (< 1 MPa), use unconfined values.

**RC slab and column (structural):**
- ρ = 2,400 kg/m³, Vp = 3,500–4,000 m/s, Vs = 1,800–2,200 m/s.
- Nominal: Vp = 3,800 m/s, Vs = 2,000 m/s → E_dyn = ρ·Vs²·(3Vp²-4Vs²)/(Vp²-Vs²) ≈ 28 GPa (consistent with 30 GPa design value + dynamic stiffening).

**Soil (Israeli urban loam/sandy clay; Mediterranean coast):**
- ρ = 1,700–1,900 kg/m³, Vp = 300–700 m/s, Vs = 100–300 m/s (NEHRP site class C–D typical for coastal Israel).
- Nominal: ρ = 1,800 kg/m³, Vp = 500 m/s, Vs = 200 m/s. **This is the calibration-critical parameter.** Vs_soil should be measured or inverted per deployment site.

**Damping (attenuation):** Devito's standard stress-velocity scheme does not include intrinsic Q (quality factor). Implement via a simple Kelvin-Voigt viscous damping term if needed, or add a Q-frequency-proportional viscosity to μ:

```python
# approximate constant-Q viscosity (Carcione 2007):
# mu_eff = mu * (1 + i*omega/(2*Q*omega_ref)) -- frequency domain
# time domain: add velocity-proportional stress increment
# For training data, omit Q (Q→∞); use as a diversity axis in the sensitivity study.
```

For the training corpus Q = ∞ is acceptable. Add Q randomization (Q_concrete ~ U[30, 100], Q_soil ~ U[15, 40]) as a later augmentation axis.

### 3.3 Parameter Sampling for Dataset Diversity

Sample these per-house:

```python
import numpy as np
rng = np.random.default_rng(seed)

# Footprint (rectangular, Israeli statistics from OSM)
L = rng.normal(11, 2)        # metres, longer axis; clip to [7, 18]
W = rng.normal(9, 1.5)       # metres, shorter axis; clip to [6, 14]
n_storeys = rng.choice([1,2,3], p=[0.15, 0.60, 0.25])

# Ceiling height per storey (metres)
h_clear = rng.normal(2.75, 0.05)         # clip to [2.60, 2.90]
h_slab  = rng.normal(0.20, 0.02)         # RC slab thickness
h_floor = h_clear + h_slab               # floor-to-floor

# Wall thicknesses (metres)
t_int = rng.normal(0.15, 0.015)          # interior hollow block
t_ext = rng.normal(0.20, 0.02)           # exterior hollow block (block only)

# Room program
n_bed = rng.choice([1,2,3,4], p=[0.05, 0.55, 0.35, 0.05])

# Material parameters
Vs_soil = rng.uniform(150, 300)          # KEY diversity axis
Vp_soil = rng.uniform(350, 700)
rho_soil = rng.uniform(1700, 1900)
Vs_block = rng.uniform(900, 1100)
Vp_block = rng.uniform(2000, 2500)
```

---

## 4. Scale and Diversity — How Many Houses and Which Axes Matter

### 4.1 What the Literature Says

**ProcTHOR (Allen AI, 2022)** generated 10,000 procedural houses and showed remarkable generalization in navigation tasks — but their metric is object-manipulation accuracy, not wave-physics simulation quality, so this number does not translate directly.

**SubsurfaceGen (2025, arXiv 2605.30541)** found that 42 field-scale 3D geological models → 4,276 2D slices were sufficient for cross-geology generalization in full-waveform inversion, when diversity was structured across six distinct geological settings (not just random parameter variation). The key finding: **structured between-setting diversity matters more than raw count.**

**Seismic sim-to-real** practice (50,000 synthetic events → real fine-tuning on 2,435 events) suggests that at least 1–2 orders of magnitude more synthetic data than real data is needed, and real data can be used for fine-tuning.

**Acoustic indoor localization**: A floorplan-localization study used 118 distinct indoor environments (100 train / 9 val / 9 test) and achieved adequate generalization — but these had very different geometry from a restricted residential set.

**Domain randomization** literature consensus: models trained with sufficient variability in the simulation parameters that differ between sim and real (primarily: soil Vs, sensor coupling, material damping) outperform models trained on fewer, more precisely-specified simulations. Randomizing soil Vs ± 50% and wall thickness ± 20% is more valuable per-compute-dollar than doubling the number of unique footprints.

### 4.2 Recommended Dataset Size

**Minimum viable corpus:** 50 houses (5 distinct footprint shapes × 2 storey counts × 5 layout variants × soil-Vs randomized). This is enough to check that the localization model does not overfit to a single house topology.

**Research paper corpus:** 200–500 houses, sampled along the axes in §4.3. With reciprocity (8 sims/house) and 100 Hz, 500 houses = 4,000 Devito runs = ~7.5 hr on one RTX 3090 (doc 06). This is the recommended scale before any publication claim about generalization.

**Upper bound / ablation study:** 2,000 houses (32 hr on RTX 3090 at 100 Hz). Run this only if the 500-house model shows identifiable failure modes tied to unseen geometries.

**Note on storey count.** A 2-storey house doubles the domain height → ~2× more voxels but the 3-D CFL dt is unchanged. Memory stays within budget (155 MB at 100 Hz for a 1-storey, roughly 280 MB for a 2-storey). Storey count is a **high-impact diversity axis** because flexural wave coupling between slabs changes the transfer function signature dramatically.

### 4.3 Diversity Axes Ranked by Expected Impact on Localization Generalization

| Rank | Axis | Why it matters | How to sample |
|---|---|---|---|
| 1 | **Soil Vs** (150–300 m/s) | Controls geophone sensitivity, coupling, and Rayleigh wave content — the dominant channel for exterior sensors | Uniform(150,300); 5–10 distinct values per footprint |
| 2 | **Storey count** (1–3) | Slab resonances, inter-storey coupling, and source-to-receiver path geometry are qualitatively different | Sample Cat([1,2,3], p=[0.15,0.60,0.25]) |
| 3 | **Footprint shape / area** (60–200 m²) | Changes the set of allowed floor positions and sensor-position geometry | Sample from OSM-Israel CDF |
| 4 | **Interior layout** (room arrangement) | Changes local impedance paths; high impact in matched-field / fingerprint methods | 5–10 layout variants per footprint; House-GAN++ or constrained-growth |
| 5 | **Wall thickness ±20%** | Direct effect on wave transmission amplitude across walls | Normal(t_nominal, 0.2*t_nominal) |
| 6 | **Sensor placement** (N=4–8, exterior positions) | Determines the localization geometry; randomize across 2–3 sensor-placement configs per house | Fixed set of K configs; not a model diversity axis but training-set diversity |
| 7 | **Soil Vp / density** | Secondary after Vs | Correlated sampling with Vs |
| 8 | **Material damping (Q)** | Affects long-time reverb; secondary for matched-field based on early waveform | Add Q later, not in first corpus |

---

## 5. Full Pipeline Summary

```
[OSM Israel / Overture]
   ↓  footprint polygon (Shapely) + n_storeys + h_floor
[Parametric Layout Generator]
   Inputs: footprint, n_bed, h_clear, t_int, t_ext, t_slab
   Algorithm: constrained-growth rectangle partition per floor
   Output: {rooms: [(poly, type)], doors: [...], windows: [...]}
   (optionally augmented with House-GAN++ layouts)
        ↓
[2-D Rasterizer per floor]  (rasterio.features.rasterize, dx=0.15 m, all_touched=True)
   → material_id_2d[floor] array
        ↓
[3-D Stacker]
   slab → storey-1-walls → slab → storey-2-walls → ... → roof-slab
   → material_ids[Nz, Ny, Nx]
        ↓
[Material LUT]  (lam, mu, rho per material_id)
   → lam_arr, mu_arr, rho_arr  (float32)
   → damp_arr (sponge ramp, outer 10 voxels)
        ↓
[Devito FDTD + stress-image free surface]
   8 reciprocal simulations per house (sensor → 480 floor positions)
   → Green's function library: GF[house, sensor, floor_pos, time, component]
        ↓
[Training data synthesis]
   Convolve GF with GRF wavelet → unlimited (house, trajectory, speed) samples
```

**Estimated compute cost (100 Hz, RTX 3090, 500 houses):**
- Rasterization: < 1 s/house, negligible.
- 8 Devito sims × 54 s/sim × 500 houses = ~60 hr. (Doc 06 gives 54 s/house; that assumed 8 sims × 6.7 s = 53.6 s.)
- **Matches doc 06: 7.5 hr** — confirmed consistent.
- GF library (1-component, 100 Hz, 500 houses): 770 MB.

---

## 6. Key Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| 1-voxel wall disappears under `all_touched=False` | High | Always use `all_touched=True` for wall rasterization |
| Rooms overlap or gap appears between rooms | Medium | Use Shapely `buffer(-eps)` shrink + `unary_union` validation before rasterizing; assert no room pixels also flagged as wall |
| Footprint from OSM has MultiPolygon or holes (courtyards) | Low-Medium | Filter to simple Polygon; buffer(-0.5m) to remove degenerate inner rings |
| Soil Vs wrong → sim-to-real gap dominates | High | Treat Vs_soil as the primary calibration parameter; provide per-deployment Vs estimation routine (e.g. Rayleigh-wave group velocity from the sensor array) |
| Interior layout unrealistic (neural generator artifacts) | Medium | Use parametric generator as default; neural generator only for augmentation; assert N_rooms ≥ 2, all rooms area > 5 m², all rooms connected |
| Air-cavity coupling dominates over structural path | Low | Confirm: at 10–100 Hz, acoustic air path from floor to exterior sensor is negligible vs structural path (impedance of air is 415 Pa·s/m vs hollow block ~2.4 M Pa·s/m — 6,000× weaker). Air-cavity boundary condition (Neumann free surface) handles this correctly in Virieux scheme with μ=0. |

---

## 7. Source References

- [ResPlan arXiv 2508.14006](https://arxiv.org/abs/2508.14006) · [HTML](https://arxiv.org/html/2508.14006v1) · [Kaggle dataset]
- [RPLAN (USTC DeepLayout)](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html)
- [OSMnx docs](https://osmnx.readthedocs.io/en/stable/) · [OSM Israel buildings (HDX)](https://data.humdata.org/dataset/hotosm_isr_buildings) · [Overture Buildings](https://gee-community-catalog.org/projects/overture_buildings/)
- [House-GAN arXiv 2003.06988](https://arxiv.org/abs/2003.06988) · [House-GAN++ CVPR 2021]
- [Graph2Plan arXiv 2004.13204](https://arxiv.org/abs/2004.13204)
- [Constrained Growth Procedural Floor Plans, GAMEON 2010, TU Delft](https://graphics.tudelft.nl/~rafa/myPapers/bidarra.GAMEON10.pdf)
- [Infinigen Indoors arXiv 2406.11824](https://arxiv.org/abs/2406.11824) · [CG Channel 2025](https://www.cgchannel.com/2025/06/open-source-tool-infinigen-indoors-generates-procedural-3d-interiors/)
- [VoxCity arXiv 2504.13934](https://arxiv.org/abs/2504.13934) · [GitHub](https://github.com/kunifujiwara/VoxCity) · [PyPI](https://pypi.org/project/voxcity/)
- [SubsurfaceGen arXiv 2605.30541](https://arxiv.org/abs/2605.30541)
- [Dynamic properties of concrete block masonry, ASCE JMCE 2024](https://ascelibrary.org/doi/10.1061/JMCEE7.MTENG-22541)
- [Evaluating dynamic elastic modulus of concrete via shear-wave (Lee 2017, Wiley)](https://onlinelibrary.wiley.com/doi/10.1155/2017/1651753)
- [Influence of voxelization on FDTD HRTF simulations, JASA 2016](https://pubs.aip.org/asa/jasa/article/139/5/2489/838418/)
- [Rasterio features API](https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html) · [PyVista voxelize](https://docs.pyvista.org/examples/01-filter/voxelize.html)
- [CubiCasa5K MIT](https://github.com/CubiCasa/CubiCasa5k)
- [ProcTHOR-10K (AllenAI)](https://procthor.allenai.org/)
- [CityEngine CGA manual](http://cehelp.esri.com/help/topic/com.procedural.cityengine.help/html/manual/cga/basics/toc.html)
- [City Maker OSM 3D pipeline, MDPI ISPRS 2019](https://www.mdpi.com/2220-9964/8/7/298)
- [3D-GIS parametric modelling with CityEngine, Tandfonline 2022](https://www.tandfonline.com/doi/full/10.1080/19475683.2022.2037019)
- [Seismic reinforcement of Israeli RC buildings, PMC 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8837140/)
- [Computer-aided layout generation review arXiv 2504.09694](https://arxiv.org/abs/2504.09694)
