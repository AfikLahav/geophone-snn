> **Research survey — house-localization pivot ("wallhacks").** Background research agent (Sonnet), 2026-07-09. Literature/tooling survey; design input, not a validated experiment. Indexed in [README.md](README.md).

---

# House Model Sourcing for 3-D Elastic-Wave Simulation

**Scope.** Where do the 3-D house geometries come from? This document deepens [doc 09](09_procedural_house_generation.md) (which recommended parametric generation from OSM footprints + RPLAN/ResPlan priors) by surveying every realistic source of ready-made 3-D building geometry, diagnosing the structural completeness gap between "visual 3-D model" and "sim-ready structural model," and laying out the constraints our voxel-FDTD (Devito) and Gmsh-hex (SPECFEM3D) pipelines impose. The conclusion reinforces doc 09 but with clearer evidence for _why_ ready-made 3-D sources fail and exactly _where_ the floorplan→3-D route is the right path.

Builds on: [doc 03](03_house_architecture_datasets.md) (datasets), [doc 09](09_procedural_house_generation.md) (parametric generator + voxelizer), [doc 06](06_sim_construction_cost.md) (compute cost), [doc 07](07_sim_engine_verification_and_plan.md) (sim engine), [doc 10](10_devito_numerics_gpu_viz.md) (numerics).

---

## 1. Ready-Made 3-D Building Model Sources

### 1.1 CityGML / CityJSON National Datasets (LoD1 and LoD2)

The richest open source of 3-D building exterior geometry is the family of national and city-scale CityGML/CityJSON datasets. These are maintained by surveying agencies as part of digital-twin and energy-simulation programs.

**Key available datasets:**

| Dataset | Coverage | LoD | Buildings | License | Format | Download |
|---|---|---|---|---|---|---|
| **3DBAG** (Netherlands) | Entire country | LoD1.2, LoD1.3, LoD2.2 | ~10 M | Open | CityJSON, OBJ, GPKG | [3dbag.nl](https://3dbag.nl/en/download) |
| **LoD2-DE** (Germany) | All 16 states | LoD2 | ~58 M | Open (fee/free by state) | CityGML | [ADV / state portals](https://www.adv-online.de/Products/3D-Building-Models/) |
| **PLATEAU** (Japan) | 250+ cities | LoD1–LoD2 (LoD4 limited) | ~2 M+ | Open | CityGML, 3D Tiles | [geospatial.jp](https://www.geospatial.jp/ckan/dataset/plateau) |
| **Bavaria LoD2** | State (part of LoD2-DE) | LoD2 | ~7 M | Open | CityGML | [geodaten.bayern.de](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2) |
| **SwissBUILDINGS3D 3.0** | Nationwide | LoD2 | ~3 M | Mixed (fee/free) | CityGML, OBJ | [swisstopo.admin.ch](https://www.swisstopo.admin.ch/de/landschaftmodell-swissbuildings3d-3-0-beta) |
| **PLATEAU Helsinki/Espoo** (Finland) | City | LoD1–3, textured | ~500 K | Open | CityGML | [kartta.hel.fi/3d](https://kartta.hel.fi/3d/#/) |
| **GlobalBuildingAtlas** | Worldwide | LoD1 only | 2.75 B | Open (ODbL/CC) | GeoJSON tiles | [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.LoD1) |
| **opencitymodel** (USA) | 125 M buildings | LoD1 | 125 M | Open | CityGML, GeoJSON | [GitHub](https://github.com/opencitymodel/opencitymodel) |

**Comprehensive index:** The "Awesome CityGML" repository (github.com/OloOcki/awesome-citygml) lists 65+ regions and 210+ million semantic 3-D building models across 21 countries.

**What LoD1 gives you.** A box: footprint polygon extruded to a flat roof at measured or estimated mean height. Attributes: building ID, function, storey count, height. No wall surfaces, no roof shape, no interior. Adequate only for building envelope (outer boundary), not structural detail.

**What LoD2 gives you.** Exterior shell with thematic surfaces: GroundSurface, WallSurface, RoofSurface. Roof shape is modeled (gabled, hipped, flat, etc.). Still **no interior geometry, no wall thickness, no floor slabs, no interior walls.** All surfaces are zero-thickness shell polygons. This is the fundamental LoD2 constraint: the building is a hollow closed shell.

**What LoD3 gives you.** Architectural exterior: windows, doors, balconies added as openings in the wall surfaces. Still exterior-only. LoD3 data is rare (Ingolstadt, Poznan, Helsinki/Espoo are examples). LoD4 would add interior geometry (rooms, furniture) but is essentially non-existent at scale in any open dataset.

**The LoD2+ extension (TU Delft, 2015):** An automatic method (CGAL + Nef polyhedra) inflates LoD2 exterior shells into per-storey solid volumes by intersecting the shell with horizontal cutting planes at each storey level. Output: solid volumes per storey, not individual walls. This gives the _outer envelope_ of each storey as a solid, but still no interior walls, no wall thickness, no partition layout. Freely available at [github.com/tudelft3d/lod2plus](https://github.com/tudelft3d/lod2plus). Validated against Rotterdam dataset. Useful as a starting point for storey-height bounding boxes; does NOT replace interior layout.

**Verdict on CityGML datasets for our sim:** LoD1/LoD2 give accurate exterior footprint + roof geometry and correct building height — useful for calibrating the outer-boundary conditions and geophone placement geometry. They do NOT provide wall thickness, interior partition walls, floor slabs, or room layout. They are input to the footprint layer of the parametric generator (replacing OSM-only footprints with richer exterior geometry where available), not a replacement for the generator itself.

### 1.2 BIM / IFC Sources

IFC (Industry Foundation Classes) is the building data format that can encode full structural detail: wall layer assemblies with material and thickness, floor slabs, columns, rooms, openings. This is the format closest to "sim-ready structural model."

**Open IFC repositories:**

| Repository | Scale | Content | License |
|---|---|---|---|
| **OpenIFC Model Repository** (Auckland) | ~130 models, ~182 IFC files | Mixed types; includes residential examples (LTU_AHouse, Duplex); wall layers in most models; CC-BY-3.0 | Open (CC-BY-3.0) |
| **BIMData R&D collection** | 40 models / 100 IFC files (~3.7 GB) | Architecture + MEP; mixed types including residential; sorted download list | Open |
| **BIMNet** | 25 real-world scan-to-BIM IFC models; 382 rooms; 8,700 m² | Full IFC: walls with material layer thicknesses, floor slabs, structural elements; point clouds paired | Open access |
| **Kaggle IFC example files** | Small (~few models) | Example IFC files; primarily for parser/viewer testing | Open |
| **OSArch community** | ~50 contributed models | Community-contributed; variable quality; some residential | Mixed |
| **BOT Duplex house** (GitHub MadsHolten) | 1 model | Classic 2-storey duplex; full wall/floor IFC; reference standard | Open |

**IfcOpenShell (Python) for extraction:**
```python
import ifcopenshell
import ifcopenshell.util.element

ifc = ifcopenshell.open("house.ifc")
for wall in ifc.by_type("IfcWall"):
    material_set = ifcopenshell.util.element.get_material(wall)
    # If IfcMaterialLayerSet: layer.LayerThickness gives thickness in mm
    # Geometry bounding box from ifcopenshell.geom gives exterior dimensions
    thickness = sum(layer.LayerThickness for layer in material_set.MaterialLayers)
```
When `IfcMaterialLayerSet` is attached to a wall type, layer thicknesses are stored explicitly. When only a simple material is attached (common in older or simpler models), thickness must be computed from the geometric bounding box.

**Critical limitation:** The OpenIFC repository contains ~130 models total across all building types (commercial, institutional, residential). Even if 30% are usable residential houses with concrete-frame construction, that is ~40 models — far below the 200–500 needed for the sim corpus. Scaling from IFC sources alone is not feasible without either: (a) a large proprietary BIM database (unavailable open-access), or (b) using IFC as a format for storing the output of the parametric generator.

**IFC as output format (recommended use):** The parametric generator in doc 09 can write its output as IFC using IfcOpenShell's write API. This gives the benefits of the IFC schema (explicit wall types, storey hierarchy, openings) while generating at the scale needed. IFC output also enables downstream use of IFC-to-CityGML converters for cross-validation.

### 1.3 Blender / Mesh-Based Architectural Model Repositories

These sources provide visual 3-D building meshes (OBJ, FBX, BLEND, GLB), typically produced for rendering, gaming, or film. They are NOT structurally detailed models.

**Key repositories:**
- **TurboSquid, Sketchfab, CGTrader**: Commercial; residential house meshes available; typically surface-only, non-manifold, no wall thickness, no material semantics. Licenses vary (most non-commercial).
- **3D Warehouse** (Trimble): Free download; SketchUp format; architectural models at variable LOD; most are exterior-only surfaces; non-manifold issues common.
- **BlenderKit**: Free + paid; similar limitations.
- **BuildingNet** (2,000 annotated building OBJ models): Exterior shells only — annotated by component (roof, wall, window, door) but no interior, no thickness, non-manifold at junctions. Academic non-commercial.
- **Infinigen Indoors** (Princeton, 2024–2025): Blender-based open-source procedural interior generator; produces photorealistic rooms with furniture. Architectural elements (walls, floors) are present as Blender mesh objects but: (a) wall thickness is typically 0 (single-face walls), (b) no material elastic properties, (c) heavy mesh complexity (furniture, props) obscures structural skeleton. Not useful for elastic wave sim without major surgery.
- **ProcTHOR-10K**: JSON scene format; US timber-frame style; room polygons and furniture; no wall thickness; Apache 2.0. Covered in doc 03 (Tier 3 — wrong construction type).

**Verdict on Blender/mesh sources:** Architectural meshes are designed for visual realism, not structural accuracy. The systematic problems are: (1) zero-thickness walls (surface models, not solids), (2) non-manifold junctions at wall corners and wall-floor intersections, (3) no material semantic tags, (4) high polygon counts from decorative geometry that is irrelevant to wave simulation. Converting a visual mesh to a sim-ready voxel grid requires resolving all non-manifold geometry, assigning material tags, and thickening zero-thickness surfaces — effectively rebuilding the model from scratch. This is more expensive than generating from a floorplan.

### 1.4 National Scan-Based 3-D Datasets (LiDAR-Derived)

**HM3D / Matterport3D / Structured3D**: Covered in doc 03 (Tier 3). Scan noise, non-watertight, US-biased construction type, academic-only. Not suitable.

**LiDAR point clouds (e.g., AHN4 Netherlands, IGN France, ISPRS Vaihingen)**: LiDAR gives outdoor point clouds. Building interiors require indoor scanning, which is rare at scale. Exterior-only LiDAR → LoD2 exterior shell (same as 3DBAG result).

**TUM2TWIN** (2025, arXiv 2505.07396): Large-scale urban digital twin benchmark for Munich including semantic building models. Primarily useful for urban-scale modeling, not residential interior structural detail.

### 1.5 Summary Table — Source Fitness for Our Sim

| Source class | Scale accessible | Wall thickness | Interior walls | Floor slabs | Material tags | Watertight | Effort to sim-ready |
|---|---|---|---|---|---|---|---|
| CityGML LoD1 (3DBAG, GlobalBuildingAtlas) | Millions | No | No | No | No | Yes (exterior) | Cannot provide interior — use as footprint only |
| CityGML LoD2 (Germany, Netherlands, Japan) | Millions | No | No | No | No | Yes (exterior) | Cannot provide interior — use as footprint only |
| CityGML LoD2+ (TU Delft method) | Same as LoD2 | No | No | Storey volume only | No | Yes | Storey bounding box — still needs interior layout |
| IFC/BIM (OpenIFC, BIMNet) | ~130–200 models | **Yes** | **Yes** | **Yes** | **Yes** | Yes (if well-authored) | High quality; too few residential concrete models |
| Blender/mesh repos (TurboSquid, 3D Warehouse) | Thousands | No (surface) | No | No | No | No (non-manifold) | Very high (surface repair + thickening + tag assignment) |
| Infinigen Indoors (Blender) | Unlimited (procedural) | No (surface) | Partial | Partial | No | No | High (furniture clutter; surface walls) |
| Floorplan → parametric generator (doc 09) | Unlimited | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Minimal (generator outputs sim-ready by design)** |

---

## 2. Floorplan-Dataset → 3-D Route: Pragmatic Pipeline

### 2.1 Source Datasets for the Pipeline

Building on doc 03 and 09, the three-tier source stack:

**Tier 1 — ResPlan (17,000 residential floorplans, JSON vector):**
- Stores `wall_depth` (uniform per plan, typically 0.15 m interior partitions).
- Room polygons in Shapely-compatible format; door and window polygons; room adjacency graph.
- **2-D only; single-floor only** (multi-floor homes are separate per-floor entries not linked).
- Direct 3-D extrusion: extrude wall polygons to `h_clear` (~2.75 m), add floor slab at base and ceiling slab at top, cut door/window openings. No additional assumption needed for wall thickness.
- Ceiling height must be supplied externally (from OSM/Overture `height`/`levels` ratio; Israeli default 2.75 m clear per doc 09 §3.1).
- Licence: permissive open-source (confirm before publication use).

**Tier 2 — RPLAN (80,788 raster floorplans, 256×256 PNG):**
- Room type per pixel (13 categories); no explicit wall polygon or wall thickness.
- To use: vectorize room regions with Shapely/rasterio, shrink each room polygon by half the target wall thickness (Israeli default: 75 mm = half of 150 mm) to define the interior face, then the gap between adjacent shrunk polygons is the wall.
- Statistical use: footprint area CDF, bedroom count histogram, room type frequency (as in doc 09 §1.2). Raster noise means vectorized wall positions are ±1 pixel (±4 mm at 256px/10m scale) — acceptable for training data diversity.
- Licence: non-commercial research.

**Tier 3 — CubiCasa5K (5,000 Finnish floorplans, SVG vector, MIT):**
- Only fully MIT-licensed vector set; 80+ semantic categories including wall polygons with measurable thickness.
- Finnish masonry/RC construction is geographically adjacent to Israeli concrete (similar structural approach, different wall materials).
- Use: validate wall-thickness priors; supplement room dimension statistics for single-family detached houses (Finnish dataset is more detached-house-heavy than Chinese apartment-dominated RPLAN).

**OSM-Israel footprints (752 K, ODbL):** Outer boundary + storey count. These anchor the exterior dimensions of the procedural generator to the actual Israeli residential stock distribution.

### 2.2 The 2-D → 3-D Extrusion Pipeline

The pipeline (extending doc 09 §2) adds explicit treatment of wall solids vs surface shells:

```
ResPlan JSON (per floor)
   ↓
[1] Parse room polygons (Shapely)
    wall_polygon = footprint.difference(unary_union([room.buffer(-wall_depth/2)]))
    # Result: polygon with holes (rooms) = wall cross-section
   ↓
[2] Rasterize per-floor (rasterio, dx=0.10–0.15 m, all_touched=True)
    material_2d[y,x] ∈ {AIR, HOLLOW_BLOCK, RC_COLUMN, RC_SLAB}
   ↓
[3] Extrude to 3-D (NumPy stacking, doc 09 §2.2)
    for z in range(Nz):
        if z in slab_range:   material_3d[z] = RC_SLAB where footprint_mask
        else:                  material_3d[z] = material_2d  (storey interior)
   ↓
[4] Multi-storey stacking (doc 09 §2.5)
    link per-storey material arrays; exterior walls are continuous column
   ↓
[5] Assign elastic parameters (doc 09 §2.3, doc 03 §3.2)
    lam, mu, rho arrays → Devito or SPECFEM3D input
```

**Key correctness requirements for step 1:**
- `buffer(-wall_depth/2)` with `join_style=2` (mitre) for wall corners; `join_style=1` (round) creates incorrect curved corners.
- `unary_union` before difference to avoid double-wall artefacts at room adjacencies.
- Assert: `wall_polygon.area > 0`; `wall_polygon.is_valid`; no room polygons overlap after shrink.
- For exterior walls (footprint boundary): use `exterior_buffer = footprint.buffer(-t_ext)` independently; do not compute from rooms (rooms end at interior face of exterior wall).

### 2.3 Ready-Made 3-D vs Generate-from-Floorplan: Head-to-Head

| Criterion | Ready-made 3-D (LoD2 CityGML) | IFC/BIM models | Generate from floorplan |
|---|---|---|---|
| **Interior walls** | Absent | Present | Present (by construction) |
| **Wall thickness** | Absent | Present (if MaterialLayerSet) | Explicit (wall_depth parameter) |
| **Floor slabs** | Absent | Present | Explicit (slab_thickness parameter) |
| **Material tags** | Absent | Present | Explicit (MATERIALS dict) |
| **Watertight / manifold** | Yes (exterior) | Yes (if well-authored) | Yes (parametric = analytically clean) |
| **Scale (number of models)** | Millions (exterior only) | ~130–200 (open-access) | Unlimited (generator) |
| **Israeli/concrete construction bias** | None (geometry only) | Depends on source | Enforced by parameter priors |
| **Diversity control** | None (fixed real data) | None | Full (seed-based sampling) |
| **Audit trail** | None | IFC schema | Generator code |
| **Sim-ready without cleanup** | No | Mostly | Yes |

**Conclusion:** For structural elastic-wave simulation where the inside of the building is the domain of interest, ready-made 3-D sources are architecturally valuable (real exterior shapes) but structurally blind. IFC is the right format for structural detail but cannot reach the required scale (~200–500 models) from open repositories alone. The floorplan → parametric generator route (doc 09) is the correct primary path. CityGML LoD2 data contributes as the outer-boundary input (replacing simple rectangular footprints with real building polygons) but does not change the need for the generator.

---

## 3. Sim Constraints: What the Model Actually Needs

### 3.1 Geometry Constraints (Devito Voxel Path)

The Devito FDTD engine (doc 06, 07, 10) consumes 3-D arrays of elastic parameters (λ, μ, ρ) on a regular Cartesian grid. It does not deal with meshes. The geometry constraints are therefore:

**C1. Minimum wall thickness ≥ 1 voxel.**
At dx = 0.15 m (100 Hz scenario), the 150 mm interior hollow-block partition is exactly 1 voxel. At dx = 0.10 m (150 Hz), it is 1.5 voxels → rounds to 1–2. The critical implementation requirement is `all_touched=True` in `rasterio.features.rasterize` (doc 09 §2.4). Thinner walls are physically invisible to the sim.

**C2. Watertight / closed exterior envelope.**
The exterior footprint must be a simple closed polygon (no MultiPolygon, no inner rings, no self-intersections). OSM footprints occasionally violate this; filter: `polygon.is_valid and polygon.geom_type == 'Polygon'`. Apply `.buffer(0)` to repair minor self-intersections.

**C3. Axis-aligned / rectangular wall segments.**
The rasterization step discretizes wall polygons onto the Cartesian grid. Non-axis-aligned walls (diagonal walls, curved walls) produce staircase artefacts. At dx = 0.10–0.15 m, a wall at 45° produces a 1-voxel zigzag that is physically equivalent to a somewhat thicker (diagonal) wall. This is acceptable for training-data diversity but is a known error source; quantified in the FDTD HRTF voxelization study (amplitude errors 5–15% for thin walls). For the parametric generator, using only axis-aligned room partitions eliminates this error entirely.

**C4. Homogeneous material per voxel (single-material assignment).**
The Devito material arrays are single-valued per voxel. At wall-material boundaries, the sub-voxel composition is not tracked. The Moczo harmonic/arithmetic averaging (doc 10) handles velocity contrasts at interfaces correctly for 1-voxel-wide walls, as long as the wall has a different material tag than the adjacent air or soil.

**C5. Continuous floor slab across entire footprint.**
The RC slab must span continuously from exterior wall to exterior wall, including over room boundaries. This is automatically satisfied by the voxelizer (the slab layer is applied to all footprint voxels at slab height regardless of room boundaries).

**C6. Soil fill and PML sponge.**
Everything outside the footprint at storey height is MAT_SOIL. The PML sponge (nbl = 20 voxels, doc 10) wraps the entire domain. The house must be fully inside the sponge boundary.

### 3.2 Geometry Constraints (Gmsh / SPECFEM3D Hex Path)

SPECFEM3D Cartesian uses conforming hexahedral meshes (structured or unstructured). The mesh must be generated by tools such as Gmsh, CUBIT, or Trelis, then converted to SPECFEM3D internal format using `LibGmsh2Specfem.py` or similar. For a building-scale structural simulation, this creates specific mesh requirements:

**C7. Conforming surfaces at material boundaries.**
In a conforming hex mesh, the mesh faces must align exactly with material boundaries (wall-air interfaces, wall-slab interfaces). This means the building geometry must be defined as a CSG (Constructive Solid Geometry) or BRep model in Gmsh before meshing, not as a voxel array. Interior walls, floor slabs, and columns must all be explicit geometric entities.

**C8. Minimum 5–10 elements across each structural element.**
SPECFEM3D requires a minimum of ~5 PPW (points per wavelength) for fourth-order spectral elements. At 100 Hz and Vs_hollow_block = 1,000 m/s, λ_min = 10 m — which requires elements of ≤ 2 m for 5 PPW. This is coarse enough that a hex mesh is feasible. However, a 150 mm interior wall at this element size is a single sub-element feature and requires a local mesh refinement region around the wall. With mesh refinement, elements at the wall boundary must be 30–50 mm for ≥ 5 PPW across the wall at 100 Hz — a 5× reduction in element size that increases the element count inside the building domain dramatically.

**C9. Hex mesh generation for building geometry is non-trivial.**
Unstructured hexahedral meshing for general 3-D domains with thin flat features (walls, slabs) is an open problem in mesh generation. Gmsh's `Recombine2D` + `Extrude` can produce a structured hex mesh for a single prismatic domain, but a multi-room building with internal walls and slabs requires either: (a) a partition of the building into non-overlapping hexahedral sub-domains, meshed independently and merged; or (b) a tetrahedral mesh converted to hex via subdivision algorithms. Option (a) is laborious for each building geometry; option (b) produces poor-quality hexahedra at thin walls.

**Practical consequence for SPECFEM3D:** The voxel-FDTD (Devito) path is far simpler to implement for this application because it side-steps the mesh generation problem entirely. SPECFEM3D is the better engine for validation runs on a single carefully-modeled house (where a human-crafted hex mesh is feasible) but is not the right engine for a 200–500-house automated simulation campaign. Doc 09/06/07 recommendation (Devito as primary) is confirmed by this analysis.

### 3.3 The "Architectural Visual 3-D vs Sim-Ready Structural" Gap

This gap is the central problem with all ready-made 3-D sources. It has three dimensions:

**Gap A — Zero-thickness surfaces.**
Architectural visual models represent walls as 2-D surfaces (zero thickness) because this is correct for rendering. For elastic wave simulation, a zero-thickness surface is a mathematical singularity — it has no volume, hence no mass, no stiffness, and no effect on wave propagation in a voxel grid. A surface representation must be converted to a volume (solid wall) before voxelization. This requires explicit wall thickness information (not present in LoD2 CityGML, mesh repos, or scan-based datasets).

**Gap B — Non-manifold topology.**
Visual 3-D models frequently contain non-manifold edges (edges shared by >2 faces) and T-junctions (a vertex of one polygon touches the edge of another without being shared). These are acceptable for rendering but prevent volumetric operations (solid body Boolean, voxelization, FEM mesh generation). The TU Delft City4CFD pipeline reported 5–10% invalid buildings with self-intersections and non-manifold edges in urban CityGML datasets, requiring an alpha-wrapping repair step that changes the geometry. For individual architectural CAD models (IFC, SketchUp), non-manifold geometry is less common but still present at wall-floor junctions if the model was assembled from independently-drawn components.

**Gap C — Missing material semantics.**
Visual models assign materials for appearance (texture maps, diffuse colors). Elastic simulation requires materials assigned by structural function (RC concrete: Vp/Vs/ρ specific to C25/C30; hollow block: different set; soil: third set). Even when a CityGML dataset tags building function, it does not tag individual structural elements with elastic properties. The material semantic assignment must be done either by rule (all walls → hollow_block; all slabs → RC) or by structural type (RC columns → RC; infill walls → hollow_block), which requires structural classification not present in LoD1/LoD2.

**Which sources satisfy the constraints:**

| Constraint | CityGML LoD2 | IFC/BIM | Parametric generator |
|---|---|---|---|
| Wall solid (non-zero thickness) | **Fails** | Pass (if MaterialLayerSet) | **Pass** |
| Manifold/watertight geometry | Pass (exterior) | Pass | **Pass** |
| Interior wall partitions | **Fails** | Pass | **Pass** |
| Floor slabs explicit | **Fails** | Pass | **Pass** |
| Material semantic tags | **Fails** | Pass | **Pass** |
| Scale to 200–500 models | Pass | **Fails** | **Pass** |
| Israeli concrete construction bias | **Fails** | **Fails** (mostly European) | **Pass** |
| Diversity controllable | **Fails** | **Fails** | **Pass** |

---

## 4. Recommended Sourcing Strategy

### 4.1 Primary Path (Devito voxel campaign, 200–500 houses)

**Step 1 — Download OSM-Israel footprints (OSMnx / HDX dataset, ~752 K buildings, ODbL).**
Filter: `building:levels` 1–5, footprint area 60–250 m², `Polygon.is_valid`, convexity > 0.7. Sample N_footprint = 500 distinct Israeli residential footprints as outer-boundary templates.

**Step 2 — Optionally enrich with LoD2 exterior geometry (3DBAG for validation, German LoD2-DE for diversity of roof shapes).**
For the validation subset of the paper (e.g., 10–20 houses), use real LoD2 exterior shells (correct roof slopes, precise wall-face positions) as the outer boundary fed into the generator, instead of the rectangular-footprint approximation. This tests sensitivity to exterior geometry accuracy. For the main training corpus, the OSM rectangular footprint is sufficient because exterior shape variation is a lower-priority diversity axis than interior layout and soil Vs (doc 09 §4.3).

**Step 3 — Run parametric generator (doc 09 §1.4)** to produce per-house 2-D floorplans using:
- ResPlan priors: `wall_depth` distribution → central 0.15 m ± 0.015 m; room area per type; adjacency patterns.
- RPLAN priors: bedroom count histogram; footprint area distribution.
- Israeli hardcoded defaults (doc 03 §4, doc 09 §3.1): exterior wall 0.20 m, slab 0.20 m, h_clear 2.75 m.
- CubiCasa5K: supplement for single-family detached house layout statistics (larger rooms, fewer bedrooms, more corridor area).

**Step 4 — Voxelize and run Devito** as in doc 09 §2 + doc 10. 200 houses at 100 Hz ≈ 3.0 hr on RTX 3090; 500 houses ≈ 7.5 hr.

**Step 5 — Use IFC/BIM models for validation only.** Download the 10–20 most structurally-complete residential IFC files from the OpenIFC repository + BIMNet. Extract wall thicknesses and storey heights using IfcOpenShell. Voxelize via `ifcopenshell.geom` → triangle mesh → PyVista voxelize. These provide a ground-truth structural model (actual measured wall dimensions) to validate the parametric generator's approximations.

### 4.2 Secondary Path (SPECFEM3D hex mesh, 5–10 reference houses)

For the 5–10 reference-house validation runs (used to calibrate the matched-field dictionary and validate the Devito results):

1. Select 3 geometrically distinct Israeli house types (1-storey villa, 2-storey villa, 3-storey apartment).
2. Build each as an explicit BRep geometry in Gmsh using the exact same parametric generator parameters — but output as `.geo` (Gmsh geometry file) rather than voxel array.
3. Assign material regions as Gmsh `Physical Volumes`.
4. Generate structured hex mesh via Gmsh `Transfinite Surface` + `Transfinite Volume` for each rectangular sub-domain (rooms, walls, slabs modeled as separate rectangular blocks).
5. Mesh element size: 30–50 mm inside walls and slabs (≥ 5 PPW at 100 Hz in hollow block), 150–200 mm in soil/air.
6. Export to SPECFEM3D with `LibGmsh2Specfem.py`.

This is feasible for 5–10 houses at manageable human effort per house. It is not scalable to 200+ houses without automated Gmsh scripting.

### 4.3 What NOT to Pursue

- **Downloading CityGML LoD2 data as structural models:** The shell geometry is available but requires gap-filling the interior — equivalent effort to running the parametric generator, with worse control over material parameters and no Israeli construction priors.
- **Using Blender/3DWarehouse models:** Surface geometry, non-manifold, no thickness, no material tags. More cleanup effort than building from scratch.
- **Using VoxCity:** Voxelizes exterior building envelopes from OSM/Overture height data. No interior. Not relevant for indoor wave propagation.
- **Using ProcTHOR / HSSD / Matterport:** US timber-frame construction; wrong impedance contrast profile; structural wave paths are fundamentally different from Israeli RC concrete.
- **IFC as the primary large-scale source:** Only ~130 open-access IFC models exist across all building types; too few residential concrete-frame examples.

---

## 5. Source-Specific Access Notes

### 5.1 3DBAG (Netherlands, LoD2, CityJSON)

```bash
# Download a tile covering a Dutch residential neighborhood as CityJSON
# https://3dbag.nl/en/download — tile-based download or API
# CityJSON files: GroundSurface, WallSurface, RoofSurface per building
# No interior. Use as footprint-with-roof-shape template.
pip install cjio
cjio rotterdam_tile.json info  # inspect attributes
```

Building attributes in 3DBAG: `b3_bag_bag_pand_id`, `b3_h_dak_50p` (median roof height), `b3_h_dak_70p`, `b3_puntdichtheid_ahn` (point density), roof type, construction year. No wall thickness, no interior.

### 5.2 Germany LoD2-DE

Available from state geodata portals (Bavaria: open without fee; other states: varies). Data format: CityGML 2.0. Attributes per building: height (m), number of floors, building function code, roof type. No wall thickness, no interior. Bavaria portal: [geodaten.bayern.de](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2).

### 5.3 PLATEAU (Japan, CityGML/3D Tiles)

LoD1–LoD2 data for 250+ Japanese cities; LoD4 interior data available for a very small subset of public buildings (not residential). Japan RC frame construction is similar to Israeli concrete. The LoD2 exterior geometry is available as open data via [geospatial.jp](https://www.geospatial.jp/ckan/dataset/plateau). Not a source for interior geometry, but useful as an alternative footprint statistics source for Asian RC apartment buildings (similar to RPLAN priors, but at the city-model scale with GPS-accurate footprint positions).

### 5.4 OpenIFC Repository (Auckland)

CC-BY-3.0. Direct download at [openifcmodel.cs.auckland.ac.nz](https://openifcmodel.cs.auckland.ac.nz/). Approximately 130 models; subset residential: the `LTU_AHouse`, `Duplex`, and several others. For each IFC:
```python
import ifcopenshell, ifcopenshell.geom
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)
# Extract wall solids
for wall in ifc.by_type("IfcWall"):
    shape = ifcopenshell.geom.create_shape(settings, wall)
    # shape.geometry → triangulated solid
```

### 5.5 BIMNet

Open-access at [buildingsmart.org gallery](https://awards.buildingsmart.org/gallery/NpJQxbDr/ArNdXglG). 25 real-world scan-to-BIM IFC models; 382 rooms; >8,700 m². Primarily large commercial/institutional (hospitals, offices) — useful for validating wall-thickness extraction pipeline and elastic parameter assignment, but NOT representative of Israeli residential construction.

---

## 6. Integration with Devito and SPECFEM3D Pipelines

### 6.1 Devito Voxel Path

```
[Footprint source]
  OSM Israel (OSMnx) → Shapely polygon → outer boundary
  OR 3DBAG LoD2 CityJSON → cjio → building polygon (richer roof info, ignored for interior sim)
        ↓
[Parametric generator (doc 09 §1.4)]
  Inputs: footprint polygon, n_storeys, room program (RPLAN priors), wall_depth (ResPlan priors)
  Output: {rooms: [(poly, type)], doors: [...], windows: [...], wall_depth, h_clear, n_storeys}
        ↓
[2-D rasterizer per floor (doc 09 §2.2, rasterio)]
  dx = 0.10–0.15 m, all_touched=True
  → material_2d[floor][y, x] (uint8)
        ↓
[3-D stacker + material LUT (doc 09 §2.2–2.3, doc 10)]
  → lam_arr, mu_arr, rho_arr (float32)
  → damp_arr (sponge ramp, nbl=20)
        ↓
[Devito FDTD + stress-image free surface (doc 08)]
  8 reciprocal sims/house → Green's function library
```

**IFC integration (validation subset only):**
```
[OpenIFC / BIMNet IFC file]
  → ifcopenshell.geom → triangle mesh per wall/slab element
  → PyVista.voxelize(mesh, density=dx) → binary volume per element
  → assign material tag by IfcWall/IfcSlab/IfcColumn element type
  → assemble into material_3d array
  → same Devito path from material_3d onward
```

### 6.2 SPECFEM3D Hex Path (Validation Only)

```
[Parametric generator (doc 09 §1.4)]
  → Same geometry parameters as Devito case
        ↓
[Gmsh .geo script (auto-generated from generator output)]
  Each room: one Box(x0, y0, 0, w, d, h_clear)
  Each wall: BooleanDifference(footprint_box, union_of_rooms)
  Each slab: Box(x0, y0, -slab_t, W, D, slab_t)  ×  n_storeys+1
  BooleanFragments to make all volumes conformal
  Physical Volume("HOLLOW_BLOCK") = {wall_volumes}
  Physical Volume("RC_SLAB") = {slab_volumes}
  Physical Volume("AIR") = {room_volumes}
  Physical Volume("SOIL") = {surrounding soil box}
  Mesh.Algorithm3D = 4 (Frontal-Delaunay)
  Mesh.RecombineAll = 1 (force hex)
  Characteristic length: 0.05 m in walls/slabs, 0.30 m in soil
        ↓
[LibGmsh2Specfem.py (from SPECFEM3D wiki)]
  → mesh.materials, mesh.nodes, mesh.connectivity → SPECFEM3D input
        ↓
[SPECFEM3D Cartesian]
  Elastic wave simulation for reference validation
```

Note: `BooleanFragments` in Gmsh (OpenCASCADE kernel) is essential for producing conformal interfaces between wall, slab, and air volumes. Without it, mesh vertices at wall-air boundaries are not shared and the simulation will have inter-material gaps. This is a non-trivial step that requires the Gmsh OpenCASCADE kernel (`SetFactory("OpenCASCADE")`), not the built-in Gmsh kernel.

---

## 7. Key Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Wall disappears at 1-voxel thickness under `all_touched=False` | High | Always `all_touched=True`; add assertion that wall pixel count ≥ expected (footprint perimeter / dx) |
| CityGML footprint has self-intersection or MultiPolygon | Medium | Filter: `.is_valid`, `.geom_type == 'Polygon'`; `.buffer(0)` repair; fallback to OSM rectangle |
| LoD2 exterior geometry mismatches interior layout (real building ≠ parametric model) | Medium-Low | Not a problem for training corpus; only relevant if real survey data is used as validation — then compare GF dict to measured response |
| IFC wall has no MaterialLayerSet → thickness unknown | Medium | Compute from geometric bounding box (`ifcopenshell.geom` bounding box y-axis for IfcWall) |
| Non-manifold IFC geometry from OpenIFC models | Low-Medium | Run `ifcopenshell.validate`; repair with PyVista `.clean()` or MeshLib before voxelization |
| Gmsh BooleanFragments fails on complex room layouts | Medium | Simplify layout (rectangular rooms only); avoid L-shaped rooms in SPECFEM3D path; use parametric generator's axis-aligned constraint |
| Over-reliance on RPLAN Chinese apartment statistics for Israeli houses | Medium | Weight RPLAN statistics by the OSM-Israel CDF: Israeli houses tend to be more rectangular and single/double storey than Chinese high-rise apartments; clip RPLAN room areas to Israeli range |
| LoD2-DE Germany buildings used without Israeli construction priors | High | LoD2-DE gives exterior footprint + roof type; always apply Israeli interior construction parameters (wall thickness, material properties) from doc 09 §3.1 regardless of footprint source |

---

## 8. Source References

- [3DBAG Netherlands](https://3dbag.nl/en/download) · [docs](https://docs.3dbag.nl/en/) · [CityJSON delivery](https://docs.3dbag.nl/en/delivery/cityjson/) · [IJGI 2022 paper on automated LoD2 reconstruction](https://www.researchgate.net/publication/359006698)
- [Awesome CityGML (OloOcki)](https://github.com/OloOcki/awesome-citygml) — comprehensive index of open 3D city datasets
- [LoD2-DE Germany (European Open Data)](https://data.europa.eu/data/datasets/31bedca5-1843-4254-a168-1acda618c0b4) · [ADV official](https://www.adv-online.de/Products/3D-Building-Models/)
- [Bavaria LoD2 open download](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2)
- [Project PLATEAU Japan](https://www.mlit.go.jp/plateau/en/) · [data portal](https://www.geospatial.jp/ckan/dataset/plateau)
- [GlobalBuildingAtlas (TUM 2025)](https://essd.copernicus.org/articles/17/6647/2025/) · [GitHub](https://github.com/zhu-xlab/GlobalBuildingAtlas) · [HuggingFace dataset](https://huggingface.co/datasets/zhu-xlab/GBA.LoD1)
- [TU Delft open geodata (incl. 3DBAG, CityJSON datasets)](https://3d.bk.tudelft.nl/opendata/)
- [LoD2+ automatic indoor generation (Biljecki 2015, IJGIS)](https://filipbiljecki.com/publications/2015_ijgis_citygml_lod2.pdf) · [GitHub lod2plus](https://github.com/tudelft3d/lod2plus)
- [CityGML SIG3D Modeling Guide for Buildings (LoD1–LoD3)](https://en.wiki.quality.sig3d.org/index.php/Modeling_Guide_for_3D_Objects_-_Part_2:_Modeling_of_Buildings_(LoD1,_LoD2,_LoD3))
- [LOD2ES for energy simulation, ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S2352710223018958) — proposes IFC-aligned LOD2ES for simulation use
- [IFC2CityGML (NUS)](https://ifc2citygml.github.io/) · [TUM ifc-to-citygml3 (GitHub)](https://github.com/tum-gis/ifc-to-citygml3) · [TU Delft ifc2citygml (GitHub)](https://github.com/tudelft3d/ifc2citygml)
- [OpenIFC Model Repository (Auckland)](https://openifcmodel.cs.auckland.ac.nz/) · [re3data entry](https://www.re3data.org/repository/r3d100012443) · [BOT Duplex House (GitHub)](https://github.com/MadsHolten/BOT-Duplex-house)
- [BIMNet scan-to-BIM dataset (buildingSMART)](https://awards.buildingsmart.org/gallery/NpJQxbDr/ArNdXglG)
- [IfcOpenShell Python docs](https://docs.ifcopenshell.org/) · [wall material layer extraction (Medium)](https://medium.com/@talha.siddiqui767/extracting-information-about-material-layers-and-thickness-from-ifc-file-using-python-e86ee001dfde)
- [VoxCity (arXiv 2504.13934)](https://arxiv.org/abs/2504.13934) · [GitHub](https://github.com/kunifujiwara/VoxCity) — confirmed exterior-envelope only; not useful for interior sim
- [BuildingNet (2,000 annotated building OBJ)](https://buildingnet.org/) — exterior shells only
- [JASA 2016 voxelization HRTF study](https://pubs.aip.org/asa/jasa/article/139/5/2489/838418/) — 5–15% amplitude errors for sub-voxel thin walls
- [City4CFD building reconstruction for CFD (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0360132324008205) — reports 5–10% invalid buildings in CityGML due to non-manifold; alpha-wrapping repair
- [ResPlan arXiv 2508.14006](https://arxiv.org/abs/2508.14006) — confirmed 2D-only, single-floor; wall_depth is uniform per plan
- [RPLAN (USTC DeepLayout)](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html)
- [CubiCasa5K MIT](https://github.com/CubiCasa/CubiCasa5k)
- [OSM Israel Buildings (HDX)](https://data.humdata.org/dataset/hotosm_isr_buildings) · [Overture Buildings](https://gee-community-catalog.org/projects/overture_buildings/)
- [SPECFEM3D Cartesian GitHub](https://github.com/SPECFEM/specfem3d) — confirms Gmsh, CUBIT, Abaqus, Salome as supported mesh generators
- [Gmsh reference manual](https://gmsh.info/doc/texinfo/gmsh.html) — BooleanFragments, Transfinite meshing, Recombine
- [Automatic high-detailed building reconstruction for urban microscale sims (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0360132324008205)
