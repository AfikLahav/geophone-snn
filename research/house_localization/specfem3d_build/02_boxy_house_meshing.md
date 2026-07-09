> **BUILD SPECIFICATION — parametric boxy-house → conforming all-hex mesh for SPECFEM3D.**
> Research + primary-source Gmsh API verification (Opus, 2026-07-09). Builds on `12_specfem3d_adoption.md`
> (SPECFEM3D adoption/converter), `09_procedural_house_generation.md` (voxel/geometry side),
> `10_devito_numerics_gpu_viz.md` (wall-resolution + element-count analysis), `14_building_size_envelope.md`
> (house archetypes/sizes). This is the meshing half of SPECFEM3D Stage 2. Sibling to (future) `01_*`
> in this folder. Indexed in `../README.md`.
>
> **This is an implementable recipe, not a summary.** API calls are verified against Gmsh 4.15.x and
> the SPECFEM3D Cartesian manual (sources at end). Where a claim is unverified it is flagged UNVERIFIED.

---

# Parametric Boxy-House → Conforming All-Hex Mesh (Gmsh → SPECFEM3D)

## 0. The one paragraph that decides everything

SPECFEM3D needs an **all-hexahedral, conforming** mesh (no hanging nodes — the SEM assembly is
node-shared). The mesh is **multi-resolution**: dx ≤ 0.05 m in 0.15 m concrete walls (≥3 elements
through the wall — the established constraint from doc 10 §Risk 1), dx ≈ 0.30 m in soil. Gmsh has
exactly **two** ways to make all-hex: (a) **transfinite + recombine** (structured, high quality, but
each meshed volume must be a topological box — **5 or 6 faces only**), and (b) **`SubdivisionAlgorithm=2`**
(splits any tet/prism mesh into hexes — works on anything, but produces **skewed hexes that fail
SPECFEM3D's skewness < 0.75 gate**). The naive "one OCC box per wall + `fragment` + `RecombineAll`"
pipeline **does not work**, because `fragment` turns the soil into a single volume with >6 faces
(a block with a house-shaped notch), which transfinite refuses. The workable recipe is therefore a
**hand-decomposed multi-block transfinite model**: partition the whole domain into axis-aligned
6-faced boxes (walls, slabs, and the soil split into a grid of blocks by the wall planes), share
faces by construction, grade element size *within* each block, and use SPECFEM3D **doubling bricks**
(a conforming 3:1 coarsening layer) to bridge the wall-fine zone to the soil-coarse zone. This is
**scriptable but non-trivial** — verdict in §5.

---

## 1. Geometry construction

### 1.1 What is a solid, what is a void, what is a free surface

Decision (physics, from doc 09 §2.3 + doc 10 and confirmed by doc 14): **do NOT mesh interior air.**
Air coupling to structure below ~500 Hz is negligible (impedance of air 415 Pa·s/m vs hollow block
~2.4 M Pa·s/m — 6000× weaker, doc 09 §6). So interior rooms are **empty (unmeshed) voids**, and the
inner faces of walls / undersides of slabs that bound those voids are **traction-free surfaces**,
handled by SEM's natural Neumann BC at **zero cost and zero code** (doc 12 readiness map: "free
surface (building walls facing air) — CONFIG"). Concretely:

| Entity | Meshed as | Boundary treatment |
|---|---|---|
| Soil block (buffer + depth) | HEX, material = soil | top = free surface; sides+bottom = absorbing |
| Foundation / floor slab | HEX, material = RC slab | top face inside rooms = free; bottom welded to soil |
| Exterior walls (0.20 m) | HEX, material = block/RC | outer & inner faces = free surface |
| Interior partition walls (0.15 m) | HEX, material = block | both faces = free surface |
| Ceiling / storey slab / roof slab | HEX, material = RC slab | under/over faces = free surface |
| Interior air rooms | **NOT meshed** (void) | its bounding wall/slab faces are the free surfaces above |
| Door/window openings | **gap in the wall block** (wall block simply not created there) | new free faces at the reveal |

This is a big simplification win: the mesh is a **thin concrete shell of boxes sitting on/in a soil
block**, not a solid house. The "hollow" is literally absence of elements.

### 1.2 The naive OCC-boolean approach — and why it fails for all-hex

The obvious script (and the one sketched in doc 12 §Stage 2, steps 1–8) is:

```python
import gmsh
gmsh.initialize()
gmsh.model.add("house")
soil     = gmsh.model.occ.addBox(x0,y0,z0, Lx,Ly,Lz)          # outer soil domain
outer    = gmsh.model.occ.addBox(hx0,hy0,hz0, hlx,hly,hlz)     # building envelope
inner    = gmsh.model.occ.addBox(...)                          # interior void
shell,_  = gmsh.model.occ.cut([(3,outer)], [(3,inner)])        # concrete shell
gmsh.model.occ.fragment([(3,soil)], shell)                     # <-- make interface conforming
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
gmsh.model.mesh.generate(3)
```

`occ.fragment` (= `BooleanFragments`) **is** the right tool to make the concrete↔soil interface
*conforming* — it splits both operands along their intersection so they share the interface surface
(this part of doc 12 is correct; verified: fragment produces shared boundary entities, which is
exactly what welds displacement/traction continuity). **BUT** two things break all-hex:

1. **`Recombine3DAll` on an unstructured tet mesh does not reliably give all-hex.** Gmsh's 3D
   recombination (Blossom-quad-only in 2D; the 3D "recombine tets→hex" path) is **hex-*dominant*, not
   all-hex** — it leaves pyramids/prisms/tets in the transition regions. SPECFEM3D requires **pure
   HEX8/HEX27**; mixed elements are rejected. (Verified: manual says structured constraints are
   honored first, unstructured recombination fills the rest and is not guaranteed pure-hex.)

2. **After `fragment`, the soil is one volume with >6 faces.** A box with a rectangular building
   footprint pressed into it becomes a single volume whose boundary is the outer box faces **plus**
   the many interface faces of the shell — far more than 6. `setTransfiniteVolume` is **hard-limited
   to 5- or 6-faced volumes** (verified, Gmsh issue #2439 and mailing list: "the transfinite
   algorithm interpolates from one face across to the opposite; it can only do this for 5- and
   6-sided volumes"). So the structured path is unavailable on the fragmented soil.

Conclusion: **`fragment` is kept for the concrete↔soil *interface welding only*, but the soil must be
pre-partitioned by hand into 6-faced sub-boxes so every meshed volume is transfinite-able.** This is
the pivot from "let Gmsh figure it out" to "hand-authored multi-block", detailed in §2.

### 1.3 The multi-block construction (the approach that actually meshes to all-hex)

Because the house is **axis-aligned and boxy**, the domain decomposes cleanly into a 3-D grid of
axis-aligned boxes. Every structural element and every soil region between structural planes is a
6-faced box. Build them as OCC boxes, then `fragment` the whole collection **once** so all coincident
faces are shared (conforming), then apply transfinite+recombine per volume.

```python
import gmsh
gmsh.initialize()
gmsh.model.add("house")
occ = gmsh.model.occ

boxes = []   # list of (dimtag, material_name)

# --- concrete shell as individual wall/slab boxes (NOT one cut shell) ---
# each exterior/interior wall segment is its own addBox; openings = omit that sub-box
for w in wall_segments:      # from the parametric layout generator, doc 09 §1.4
    t = occ.addBox(w.x, w.y, w.z, w.dx, w.dy, w.dz)
    boxes.append(((3, t), w.material))     # 'RC' or 'BLOCK'
for s in slab_boxes:         # floor slab, storey slabs, roof slab
    t = occ.addBox(s.x, s.y, s.z, s.dx, s.dy, s.dz)
    boxes.append(((3, t), 'RC_SLAB'))

# --- soil partitioned into a grid of boxes by the wall/footprint planes ---
# split-planes = every wall face plane + footprint edges + a buffer ring.
# Each cell of that grid is one axis-aligned soil box (6 faces).
for sb in soil_grid_boxes:   # see §2.2 for how the grid is built
    t = occ.addBox(sb.x, sb.y, sb.z, sb.dx, sb.dy, sb.dz)
    boxes.append(((3, t), 'SOIL'))

# --- ONE fragment over everything: makes ALL coincident faces shared/conforming ---
all_dimtags = [b[0] for b in boxes]
frag_out, frag_map = occ.fragment(all_dimtags, [])
occ.synchronize()
```

`occ.fragment(objects, [])` with an empty tool list = "fragment this set against itself" =
`removeAllDuplicates`-style welding: every pair of touching boxes now shares the touching face, edge,
and vertices. Because the boxes were built on a common grid of planes, **no face is partially
overlapping** — each shared face is whole-to-whole, so **each resulting volume is still a 6-faced
box** (this is the crucial property that a hand-built grid guarantees and a `cut`-shell does not).
`frag_map` tells you which output volumes came from which input box, so you can re-attach the material
label after fragmentation (input order preserved; iterate `frag_map`).

Openings (doors/windows): simply **do not emit** the wall sub-box where the opening is, or split the
wall column into below-sill / above-lintel boxes and omit the middle. The reveal faces become free
surfaces automatically. No boolean subtraction needed.

Multiple storeys: repeat the wall/slab box set per storey, stacked in z, sharing the storey slab.
The soil grid is only under/around the ground floor.

---

## 2. The hard part — one conforming multi-resolution all-hex mesh

### 2.1 Honest assessment of the four candidate strategies

| Strategy | All-hex? | Conforming? | Multi-res dx? | Quality (SPECFEM skew<0.75)? | Scriptable to 100s? | Verdict |
|---|---|---|---|---|---|---|
| **A. `fragment` + `Recombine3DAll`** | ❌ hex-dominant, leaves prisms/pyramids | ✅ | ✅ via fields | mixed elems rejected outright | ✅ | **Fails: not pure hex** |
| **B. `SubdivisionAlgorithm=2`** (tet→hex) | ✅ pure hex | ✅ | ✅ | ❌ many hexes skew>0.8 near interfaces | ✅ | **Fails quality gate** |
| **C. Multi-block transfinite + recombine, uniform-graded** | ✅ | ✅ by construction | ✅ via `setTransfiniteCurve` node counts | ✅ excellent (axis-aligned → near-perfect hexes) | ⚠️ needs the auto-decomposer | **Works; the recommended core** |
| **D. C + SPECFEM doubling bricks** for wall→soil coarsening | ✅ | ✅ | ✅ true 3:1 jump without hanging nodes | ✅ good except within doubling brick | ⚠️ hardest to script | **Works; needed if graded transition is too many elements** |

The reason **C** is viable *specifically here* and not in general hex-meshing (which is famously a
"swamp" — see "Hex Me If You Can", PMC9828569): **the geometry is axis-aligned boxes.** Transfinite
interpolation on an axis-aligned box with matched edge node-counts produces **perfect rectangular
hexes** (Jacobian ≈ 1, skewness ≈ 0). All the difficulty of hex meshing is in curved/branching
geometry; we have none. Our only real difficulty is the **fine↔coarse size transition**, addressed
by grading (§2.3) or doubling bricks (§2.4).

### 2.2 Building the transfinite-able box decomposition automatically

The decomposer is the piece of real engineering. Input: parametric house (wall planes, slab planes,
footprint, buffer, depth). Output: a set of axis-aligned boxes that (i) tile the whole domain, (ii)
are each 6-faced, (iii) share whole faces with neighbors.

Algorithm (a **3-D grid slab / "box complex"** — the classic multiblock trick):

1. Collect **all split-plane coordinates** in each axis:
   - x-planes: every wall face x-coordinate, footprint x-edges, buffer-ring x, domain x-bounds.
   - y-planes: same in y.
   - z-planes: soil bottom, foundation top/bottom, each storey floor/ceiling slab top/bottom,
     roof slab top, free surface.
2. This defines a **structured grid of cells** (a "box lattice"). Every cell is an axis-aligned box.
3. **Classify each cell** by its centroid: inside a wall → BLOCK/RC; inside a slab → RC_SLAB;
   inside an interior room void → **DROP the cell (do not emit)**; otherwise → SOIL.
4. Emit one `occ.addBox` per non-dropped cell. Because the lattice is shared, all faces match
   whole-to-whole → after `fragment`, all volumes are 6-faced. ✅

This is O(Nx·Ny·Nz) cells where Nx,Ny,Nz = number of split planes per axis — for a boxy house that is
~10–20 planes/axis → a few thousand cells before dropping room voids and merging (merge coplanar
same-material adjacent cells to cut count). Cheap. This is the same idea as the voxel stacker in
doc 09 §2.2, but at the *block* level (few thousand blocks) instead of the voxel level (millions),
and it yields a clean CAD model rather than a staircased raster.

### 2.3 Grading element size from wall (0.05 m) to soil (0.30 m) — no hanging nodes

Within the transfinite framework, element size is set by **node count on each curve**
(`setTransfiniteCurve(curveTag, nPoints, "Progression", coef)`), and **conformity is automatic**
because shared curves must be given the **same node count from both sides** (this is the multi-block
matching rule, verified in the matveichev recipe and the FEniCS thread). The strategy:

- **Concrete boxes:** set node counts so element size ≈ 0.0375 m (4 elements through a 0.15 m wall,
  per doc 10 §Risk 1). E.g. a wall face 0.15 m thick → `setTransfiniteCurve(edge, 5)` (5 nodes =
  4 elements). Along-wall and vertical edges: node count = ceil(length/0.0375).
- **Soil boxes far from house:** element size ≈ 0.30 m → node count = ceil(length/0.30)+1.
- **Transition soil boxes (the ring touching the house):** use **geometric progression**
  (`"Progression", coef`) so the box has 0.05 m elements on the wall-facing edge grading to 0.30 m on
  the outer edge. A single box with `setTransfiniteCurve(edge, N, "Progression", 1.15)` graduates
  the size smoothly. **Because the wall-facing edge node count must equal the wall box's edge node
  count, the meshes are conforming across the interface with zero hanging nodes.**

The catch: a graded soil box shares its **outer** edge with the next soil box out — that edge's node
count is fixed by the graded box, forcing the next box's inner edge to match, and so the fine size
"leaks" outward. To reach true 0.30 m you either (a) accept a graded ramp over a few boxes (simplest;
costs extra elements), or (b) insert a **doubling brick** (§2.4) to make a genuine 3:1 or 2:1 jump.

**Element-count cost of pure grading (no doubling):** the ~0.05 m size in the near-wall ring persists
for the width of the graded ring. If the ring is ~1 m wide (≈ shortest Rayleigh wavelength / a few)
before reaching 0.30 m, the near-house soil is over-refined. This is the main penalty of the
"grading-only" (strategy C) path and the reason doubling bricks (D) exist.

### 2.4 SPECFEM3D doubling bricks — the conforming 3:1 coarsening (the key enabler)

**This is the single most important fact for the multi-resolution requirement, and it flips the
outlook.** SPECFEM3D's own philosophy is *conforming coarsening via doubling bricks*: a special
hexahedral macro-cell that connects **one coarse face to a 3×3 (or 2×2) set of fine faces without any
hanging node** — all nodes are shared, so it is fully conforming and SEM-legal. Verified: the manual's
internal mesher `xmeshfem3D` exposes `NDOUBLINGS`, `NZ_DOUBLING_*`, and v4.0 introduced an efficient
symmetric "doubling brick" assembled from four base bricks to double in both horizontal directions in
one layer (ResearchGate fig, SPECFEM3D v4.0). Doubling layers must be ≥2 layers apart.

Implication for our design: **you do not need Gmsh to grade continuously from 0.05 m to 0.30 m.** You
keep the concrete shell + a thin ring of soil at ~0.05 m, insert one doubling layer to jump to
~0.15 m, and (optionally) a second doubling layer (≥2 layers later) to jump to ~0.30–0.45 m for the
far soil. Each jump is a factor of ~2–3 in size, conforming, all-hex.

Two ways to get the doubling bricks:

- **(D1) Use SPECFEM3D's internal mesher `xmeshfem3D` for the soil**, and Gmsh only for the concrete
  shell + near-ring, then merge. This is clean *for the soil* (the internal mesher's doubling is
  battle-tested) but merging two meshes conformingly at the ring interface is itself fiddly (node
  matching). UNVERIFIED whether a clean internal-mesher↔Gmsh merge is routine; flag to prototype.
- **(D2) Build the doubling brick pattern in Gmsh as explicit transfinite sub-boxes.** The doubling
  brick is a fixed topological template of hexes (the "27→8" or "9→3" transition). It can be authored
  once as a parametric block and instantiated in the ring. More work up front; fully in Gmsh.
  UNVERIFIED that a Gmsh-authored doubling brick passes SPECFEM's skewness gate; **must prototype.**

**Recommended:** start with **strategy C (grading only)** for the first working house — it is the
least code and axis-aligned grading gives excellent quality — measure the element count, and only add
doubling bricks (D) if the near-house over-refinement blows the budget. Given the count estimate in §4
(~0.8–1.5M elements with grading), C alone is likely sufficient for a single-storey house; doubling
becomes worthwhile at 250 Hz or 2–3 storeys.

### 2.5 Failure modes to guard against (name them, test for them)

1. **Volume with >6 faces after fragment** → transfinite silently not applied → that volume meshes as
   tets → generate(3) yields mixed elements → SPECFEM rejects. **Guard:** after synchronize, for every
   volume call `getBoundary` and assert exactly 6 surfaces; abort with the offending dimtag if not.
2. **Mismatched node counts on a shared curve** → Gmsh error "curve X: transfinite meshing with N and M
   points" or a non-conforming interface. **Guard:** build a curve→nodecount map keyed by the *geometric*
   line, set once per unique line after fragment (fragment renumbers entities — set transfinite
   *after* synchronize, resolving shared curves via `getEntities`/coordinates, not pre-fragment tags).
3. **`Recombine` not applied to every surface of a transfinite volume** → volume meshes as tets even
   though transfinite. **Guard:** `setRecombine(2, s)` on **all** surfaces, or set `Mesh.RecombineAll=1`
   globally, *before* generate.
4. **`SubdivisionAlgorithm=2` used as a shortcut** → all-hex but skew>0.8 hexes at concrete corners →
   fails `xcheck_mesh_quality`. **Guard:** do not use it except as a diagnostic; verify skewness.
5. **Doubling brick distortion** → the transition hexes are the worst-quality in the mesh. **Guard:**
   run `xcheck_mesh_quality` and inspect the skewness histogram tail specifically in doubling layers.

---

## 3. Material tagging

After meshing, assign **physical volume groups** so the doc-12 converter can map each to (Vp,Vs,ρ,Q).
Tag by iterating the (dimtag → material_name) list you kept through `frag_map`:

```python
from collections import defaultdict
vol_by_mat = defaultdict(list)
for (dim, tag), mat in resolved_boxes:      # resolved via frag_map after fragment
    vol_by_mat[mat].append(tag)

phys = {}
for mat, tags in vol_by_mat.items():
    g = gmsh.model.addPhysicalGroup(3, tags)          # 3 = volume
    gmsh.model.setPhysicalName(3, g, mat)             # 'SOIL','RC_SLAB','BLOCK','RC_WALL'
    phys[mat] = g

# Boundary surface physical groups for the SPECFEM 10-file converter:
# free surface (soil top + all air-facing concrete faces), and 5 absorbing faces.
gmsh.model.addPhysicalGroup(2, free_surface_face_tags,  name="free_surface")
gmsh.model.addPhysicalGroup(2, xmin_face_tags,          name="abs_xmin")
gmsh.model.addPhysicalGroup(2, xmax_face_tags,          name="abs_xmax")
gmsh.model.addPhysicalGroup(2, ymin_face_tags,          name="abs_ymin")
gmsh.model.addPhysicalGroup(2, ymax_face_tags,          name="abs_ymax")
gmsh.model.addPhysicalGroup(2, zmin_face_tags,          name="abs_zbottom")
```

Face classification is easy for axis-aligned geometry: get each boundary surface's centroid/normal
(`getBoundary` + `getCenterOfMass`), a face on the domain x-min plane with outward −x normal →
`abs_xmin`; any concrete/soil-top face at the free surface z or bounding an interior void → `free_surface`.

Material property table (from doc 09 §3.2 — the physical numbers are already established there):

| Physical group | Vp (m/s) | Vs (m/s) | ρ (kg/m³) | Q (initial) |
|---|---|---|---|---|
| SOIL | 500 (350–700) | 200 (150–300) | 1800 | ∞ first, then 15–40 |
| RC_SLAB / RC_WALL | 3800 | 2000 | 2400 | 30–100 |
| BLOCK (hollow-block infill) | 2200 | 1000 | 1100 | 30–100 |

These become lines in `nummaterial_velocity_file` (doc 12 §Stage 3). Note: the concrete↔soil interface
is a **welded contact automatically** because fragment made it node-shared (doc 12 §Risk 4 — LOW,
confirmed correct).

---

## 4. Quality + verification

### 4.1 Element-count estimate (cross-check doc 10's ~0.8–1.5M)

Representative single-storey Archetype C house (doc 14): footprint 10×12 m, wall height 3 m, wall
0.20 m ext / 0.15 m int, slab 0.20 m; soil buffer 5 m each side → domain ~20×22×10 m; f_max=100 Hz.

- **Soil at dx=0.30 m (far), grading to 0.05 m near house:** bulk soil volume ≈ 20·22·10 − house ≈
  4360 m³. At uniform 0.30 m: 4360/0.027 ≈ **161k** elements. The graded near-house ring roughly
  doubles the local density over ~1 m ring → add ~**200–400k**. Soil total ≈ **0.35–0.55M**.
  (Note: doc 12 §Risk 1 quoted ~720k soil elements using the full 24×26×10 domain at strict 0.30 m
  with no SEM PPW relief; with a 20×22 domain and grading, 0.35–0.55M is the honest range.)
- **Concrete shell at dx=0.0375 m:** wall perimeter ≈ 2·(10+12)=44 m, height 3 m, thickness ~0.18 m
  avg. Wall volume ≈ 44·3·0.18 ≈ 23.8 m³ + slabs (2·10·12·0.20 ≈ 48 m³ counting floor+roof). Shell
  volume ≈ 72 m³. At 0.0375³=5.3e−5 m³/elem → **1.36M** elements. Walls alone (23.8 m³) → ~450k;
  slabs dominate. If slabs are meshed at 0.0375 m they blow the budget — **relax slab in-plane dx to
  ~0.10–0.15 m and keep only through-thickness at 0.05 m** (3 elements through 0.20 m), cutting slab
  elements ~10×. With that relaxation shell ≈ **0.4–0.6M**.
- **Total ≈ 0.8–1.1M elements** for the single-storey 100 Hz house — **consistent with doc 10's
  0.8–1.5M**. ✅ Two-storey or 250 Hz pushes toward and past the upper end (doc 12 §Risk 7: 250 Hz
  ≈ 10–20M — needs doubling bricks to stay tractable).

Design lever confirmed: **anisotropic element sizing** (fine through-thickness, coarse in-plane on
slabs/walls) is essential and is trivially available in transfinite (different node count per edge).
This is a capability the voxel/Devito path (doc 09/10) does **not** have — a genuine SPECFEM/Gmsh
advantage.

### 4.2 Hex quality / Jacobian / conformity checks (in order)

1. **In Gmsh, pre-export:** `gmsh.model.mesh.getElementQualities(..., "minSICN")` (signed inverse
   condition number; >0 means valid, no inverted hexes). Assert min > 0. For axis-aligned transfinite
   boxes this should be ~1.0 everywhere except doubling bricks.
2. **Conformity / watertightness:** assert no volume was left tet (`getElements(3)` returns only type
   5 = HEX8, or type 12 = HEX27; **no** type 4/6/7 tet/prism/pyramid). This is the pure-hex gate.
3. **In SPECFEM3D, post-convert:** run `xcheck_mesh_quality` → skewness histogram. **Gate: max
   skewness < 0.75** (verified manual threshold; >0.80 = must fix). Doubling-brick hexes are the
   tail; inspect them.
4. **Stability estimate:** `xcheck_mesh_quality` also reports the CFL-implied min stable dt; confirm
   it matches the doc-12 estimate (~8–10 μs, set by 0.05 m concrete elements).

### 4.3 Visual QA (before spending GPU-hours)

- **Gmsh GUI:** `gmsh.fltk.run()` — clip through the house, color by physical group, confirm walls are
  solid boxes and rooms are empty (no elements in voids), confirm free surfaces are where expected.
- **PyVista** (doc 10 §6 already has the pipeline): read the `.msh` via `meshio`, `pv.read`, clip at
  a room plane, `.plot(show_edges=True)`. Visually confirm: (i) 3–4 elements through each wall,
  (ii) node-matched (conforming) interface at concrete↔soil, (iii) no orphan/floating elements,
  (iv) doubling-brick transition looks clean. Overlay sensor positions as spheres (doc 10 §6c).
- **Automated regression:** for the parametric campaign, dump per-house (n_elements, min_SICN,
  max_skew, n_hex, n_nonhex, watertight_bool) to a CSV and **gate the whole campaign** on it — never
  simulate a house that failed the mesh gate.

---

## 5. HONEST assessment — CLEAN or SWAMP?

**Verdict: CLEAN-*ish* — a "paved road with two potholes", not a swamp, *specifically because the
houses are axis-aligned boxes*.** Reasoning:

- General 3-D hex meshing is a genuine research swamp (curved/branching geometry, "Hex Me If You
  Can"). **We are exempt from all of it.** Axis-aligned boxes are the one case where structured
  transfinite meshing is trivially high-quality and fully deterministic. Perfect rectangular hexes,
  Jacobian ≈ 1, no optimization needed. That is the CLEAN part, and it is a strong, real advantage
  over the Devito voxel path (which staircases walls and needs Moczo averaging, doc 10 §1).
- **Pothole 1 — the auto-decomposer must be written.** You cannot use the naive
  `cut`-shell + `fragment` + `RecombineAll` pipeline; it produces >6-face volumes and hex-dominant
  (not pure-hex) meshes. You must write the **box-lattice decomposer** (§2.2) that emits one 6-faced
  transfinite box per grid cell and matches shared-curve node counts. This is **~300–500 lines of
  careful Python**, most of the risk in bookkeeping entity renumbering after `fragment` (set
  transfinite/recombine/physical-groups by *geometry* lookup post-synchronize, not by pre-fragment
  tags). It is deterministic and scriptable once written — no per-house manual intervention — but it
  is real work and the single largest failure surface.
- **Pothole 2 — the fine↔coarse transition.** Grading-only (strategy C) is clean and probably
  sufficient at 100 Hz single-storey (§4.1 count fits), but over-refines the near-house soil. True
  multi-resolution wants **SPECFEM doubling bricks** (§2.4), and whether a *Gmsh-authored* doubling
  brick meshes all-hex at skewness<0.75 is **UNVERIFIED — this is the #1 thing to prototype.** The
  fallback (D1: internal mesher for soil) is proven but adds a mesh-merge step of unverified ease.

**Net:** for **single-storey, axis-aligned, boxy houses at 100 Hz**, auto-meshing is **CLEAN enough
to scale to hundreds of houses** with a one-time ~1 week decomposer build (on top of doc 12's 3–5 day
Stage 2 estimate — call it **the upper end, 5 days, of that range, plus ~2 days if doubling bricks are
needed**). It does **not** become a per-house swamp: same footprint → mesh once, vary only materials
(doc 12 §Stage 6). The swamp risk appears **only** if (a) footprints become non-axis-aligned
(L-shapes at angles, curved walls — then transfinite decomposition gets hard and you drift toward the
general hex-meshing swamp), or (b) you must run 250 Hz / 2–3 storeys where element count forces
doubling bricks and their quality is unproven here.

### What MUST be prototyped before committing (in priority order)

1. **A single-house box-lattice decomposer** → generate(3) → assert **pure HEX8** + **min SICN > 0**.
   This proves the core recipe. ~2 days. If this fails, the whole SPECFEM path is in doubt.
2. **`xcheck_mesh_quality` skewness < 0.75** on that mesh after the doc-12 converter. Proves SPECFEM
   accepts it. ~0.5 day (needs the converter, doc 12 Stage 2).
3. **A grading-only near-house transition, element count measured** vs the §4.1 estimate. Confirms the
   budget. ~0.5 day.
4. **A doubling-brick transition (D2 in Gmsh, or D1 internal-mesher merge)** meshing all-hex at
   skew<0.75. **Highest-uncertainty item.** Only needed if step 3 shows over-refinement blows the
   budget (i.e. at 250 Hz / multi-storey). ~2–3 days; may fail → fall back to CUBIT (paid) or accept
   grading-only + 100 Hz.
5. **The soil↔concrete `fragment` weld verified conforming** by a bridge check: run the shipped
   SPECFEM homogeneous halfspace vs a Gmsh soil-only box (doc 12 Stage 1 bridge) — if they agree, the
   Gmsh→SPECFEM converter and BC tagging are correct before any house is added. ~0.5 day.

---

## 6. Verified-API quick reference (Gmsh 4.15.x)

```python
# --- structured all-hex on ONE axis-aligned box (the atomic operation) ---
for c in box_curves:  gmsh.model.mesh.setTransfiniteCurve(c, n_nodes[c])   # per-edge node count
for s in box_surfs:   gmsh.model.mesh.setTransfiniteSurface(s)             # needs 3-4 corner surf
for s in box_surfs:   gmsh.model.mesh.setRecombine(2, s)                   # tri->quad on the face
gmsh.model.mesh.setTransfiniteVolume(v)                                    # ONLY if v has 5-6 faces
# global equivalents:
gmsh.option.setNumber("Mesh.RecombineAll", 1)          # recombine every surface
gmsh.option.setNumber("Mesh.Recombine3DAll", 1)        # 3D recombine (hex-DOMINANT, not pure)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)# 1=Blossom (best 2D quad)
# do NOT rely on:
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # all-hex but skewed -> fails SPECFEM gate
gmsh.model.mesh.generate(3)
```

**Load-bearing verified facts** (sources §7): `setTransfiniteVolume` is limited to **5–6 face**
volumes; `Recombine3DAll` is **hex-dominant not pure-hex**; `SubdivisionAlgorithm=2` gives **pure hex
but poor skewness**; `occ.fragment` **welds coincident faces into shared (conforming) entities**;
SPECFEM3D requires **conforming pure-hex, skewness < 0.75**, supports **HEX8/HEX27**, and supports
**doubling bricks** for conforming fine→coarse coarsening; interior air is **not meshed** (free-surface
BC, valid <500 Hz).

---

## 7. Sources (primary, verified)

- **Gmsh 4.15.2 Reference Manual** — transfinite/recombine/subdivision/OCC API:
  https://gmsh.info/doc/texinfo/gmsh.html
- **Gmsh issue #2439 — "Transfinite Volume with > 5-6 faces"** (the 5–6 face hard limit):
  https://gitlab.onelab.info/gmsh/gmsh/-/issues/2439 (also Gmsh mailing list "Transfinite volume",
  onelab.info/pipermail/gmsh/2013/008682.html)
- **matveichev, "Building hexahedral meshes with Gmsh"** (multi-block transfinite+recombine recipe,
  6-face surface-loop requirement): https://matveichev.blogspot.com/2013/12/building-hexagonal-meshes-with-gmsh.html
- **FEniCS discourse — structured hex via transfinite curves/surfaces** (4-corner requirement, call
  sequence): https://fenicsproject.discourse.group/t/.../13383
- **J. Dokken, "Using the GMSH Python API to generate complex meshes"** (OCC + physical groups
  patterns): https://jsdokken.com/src/tutorial_gmsh.html
- **SPECFEM3D Cartesian Manual — 03 Mesh Generation** (HEX8/HEX27, skewness<0.75/0.80,
  `xcheck_mesh_quality`, external-mesh 10 files, conforming requirement, doubling layers):
  https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/
- **SPECFEM3D v4.0 doubling brick** (symmetric doubling in both horizontal directions, one layer):
  ResearchGate fig, doi via "In version 4.0 of SPECFEM3D, the mesh doubling ... one ..." (search).
- **"Hex Me If You Can" (Reberol et al.), PMC9828569** — context that general hex meshing is hard;
  our axis-aligned case is the easy exception: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9828569/
- Internal: `12_specfem3d_adoption.md` (converter, 10 files, risks), `09_procedural_house_generation.md`
  (geometry/materials), `10_devito_numerics_gpu_viz.md` (wall-resolution + element counts),
  `14_building_size_envelope.md` (archetypes).

**UNVERIFIED / must-prototype flags:** (a) Gmsh-authored doubling brick passing skewness<0.75;
(b) ease of merging SPECFEM internal-mesher soil with a Gmsh concrete shell conformingly;
(c) exact element count once the decomposer + anisotropic slab sizing is real (§4.1 is an estimate).
