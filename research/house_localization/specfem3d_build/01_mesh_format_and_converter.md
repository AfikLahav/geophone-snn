> **BUILD SPECIFICATION — SPECFEM3D_Cartesian external-mesh format + `gmsh2specfem3d.py` converter.**
> Implementable spec for a developer. Builds on `research/house_localization/12_specfem3d_adoption.md`
> (which established: no official Gmsh→SPECFEM3D 3D converter exists; the external-mesh path needs
> ~10 files). Do NOT re-read the adoption analysis here — this doc is the concrete format + algorithm.
> Primary sources cited inline; all verified against the SPECFEM3D `master` source tree and manual
> (fetched 2026-07-09). Items that could NOT be verified from primary sources are flagged
> **MUST VERIFY IN IMPLEMENTATION**.

---

# 0. Scope and the two mesh paths (read first)

SPECFEM3D_Cartesian has **two** ways to get a mesh into the solver. Do not confuse them:

- **Internal mesher (`xmeshfem3D`)** — reads a `Mesh_Par_file` + interface files, generates a
  regular deformed-brick mesh itself, and writes the partitioned `DATABASES_MPI/` directly. This
  path CANNOT represent our concrete shell embedded in soil (it only does layered/deformed bricks).
  **Not our path.** (This is the path the shipped `EXAMPLES/homogeneous_halfspace_*` use, and the
  reason their example files are sometimes named `free_surface_file` — see the naming trap in §1.)

- **External mesh + `xdecompose_mesh` (SCOTCH)** — you supply an all-hexahedra mesh as a set of
  ASCII files in a `MESH/` directory; `xdecompose_mesh` partitions it; then `xgenerate_databases`
  builds `DATABASES_MPI/`. **This is our path** (Gmsh → these files → decompose → generate → solve).

This document specifies the external-mesh file set that `xdecompose_mesh` reads, and a converter
that produces it from a Gmsh `.msh`.

Verified file list that `xdecompose_mesh` opens (readthedocs 03_mesh_generation; confirmed against
`src/decompose_mesh/module_mesh.f90` `open(...file=localpath_name//'/...')` statements):

```
MESH/
  nodes_coords_file                     (required)
  mesh_file                             (required)
  materials_file                        (required)
  nummaterial_velocity_file             (required)
  free_or_absorbing_surface_file_zmax   (required — top face; free OR Stacey per Par_file flag)
  absorbing_surface_file_xmin           (required if that side is absorbing)
  absorbing_surface_file_xmax
  absorbing_surface_file_ymin
  absorbing_surface_file_ymax
  absorbing_surface_file_bottom
  absorbing_cpml_file                   (only if PML_CONDITIONS = .true.)
  nummaterial_poroelastic_file          (only if any poroelastic material — NOT us)
  moho_surface_file                     (optional — NOT us)
```

That is the "10 files": 4 mesh/material files + 6 surface files (1 top + 5 absorbing). The CPML,
poroelastic, and moho files are conditional and not needed for our concrete/soil/air runs.

**Source note:** the exporter `CUBIT_GEOCUBIT/geocubitlib/cubit2specfem3d.py` writes exactly these
filenames (with `free_or_absorbing_surface_file_zmax` for the top) — it is the canonical reference
implementation for the format and the converter below mirrors it. The internal-mesher examples use
the older name `free_surface_file`; **for the decompose_mesh path the top-surface file MUST be named
`free_or_absorbing_surface_file_zmax`.**

Units throughout: **SI — metres, m/s, kg/m³.** All integer indices are **1-based**.

---

# 1. Exact format of every external-mesh file

Formats below are taken from the actual Fortran read statements in
`src/decompose_mesh/module_mesh.f90` (fetched from `github.com/SPECFEM/specfem3d` `master`) and the
write statements in `CUBIT_GEOCUBIT/geocubitlib/cubit2specfem3d.py`. Where the manual and the source
agree they are quoted together.

Throughout, `NGNOD` = nodes per hex = **8** (HEX8) or 27 (HEX27); `NGNOD2D` = nodes per boundary
face = **4** (for HEX8) or 9 (for HEX27). We use **HEX8 / NGNOD=8 / NGNOD2D=4** (justified in §2).

## 1.1 `nodes_coords_file`

Reader (`module_mesh.f90`):
```fortran
read(IIN_DB,*) nnodes_glob                       ! line 1: node count
read(IIN_DB,*) num_node, x_coord, y_coord, z_coord   ! one per node
```
- **Line 1:** total number of nodes `N_nodes` (integer).
- **Lines 2..N+1:** `node_id  x  y  z` — `node_id` is a **1-based** integer; x,y,z are doubles (metres).
- Node ids should run 1..N_nodes. The reader stores by the `num_node` value, so ids need not be
  contiguous in file order, but emit them contiguous and sorted to be safe.

Minimal example (2-element mesh = 2 stacked unit cubes sharing a face → 12 nodes):
```
12
 1   0.0 0.0 0.0
 2   0.0 1.0 0.0
 3   1.0 1.0 0.0
 4   1.0 0.0 0.0
 5   0.0 0.0 1.0
 6   0.0 1.0 1.0
 7   1.0 1.0 1.0
 8   1.0 0.0 1.0
 9   0.0 0.0 2.0
10   0.0 1.0 2.0
11   1.0 1.0 2.0
12   1.0 0.0 2.0
```
(The corner order used to *number* these nodes is arbitrary; what matters is the order they are
*cited* inside each element in `mesh_file` — see §1.2 and §2.)

## 1.2 `mesh_file` (hex connectivity)  ← #1 failure point (node order)

Reader:
```fortran
read(IIN_DB,*) nspec_long                                  ! line 1: element count
read(IIN_DB,*,iostat=ier) num_elmnt, (elmnts(inode), inode=1,NGNOD)  ! one per element
```
- **Line 1:** total number of elements `N_elem`.
- **Lines 2..:** `element_id  n1 n2 n3 n4 n5 n6 n7 n8` — `element_id` 1-based; then exactly `NGNOD`(=8)
  node ids, cited in **SPECFEM3D's hex corner order** (§2, critical).

**SPECFEM3D HEX8 corner convention** — VERIFIED from the comment block in `module_mesh.f90` /
`cubit2specfem3d.py` (reference cube in local coords, unit cube [0,1]³):
```
  bottom face (z=0), counter-clockwise:
    n1 = (0,0,0)   n2 = (0,1,0)   n3 = (1,1,0)   n4 = (1,0,0)
  top face (z=1), same (x,y) walk:
    n5 = (0,0,1)   n6 = (0,1,1)   n7 = (1,1,1)   n8 = (1,0,1)
```
This is **not** the same as Gmsh's order — see §2 and §3 for the exact remap. This mismatch is the
single most common reason a converted mesh runs but gives garbage (negative Jacobians or scrambled
elements).

Example (the two stacked cubes; nodes as numbered in §1.1). Cube A uses bottom nodes 1-4, top 5-8;
cube B uses bottom nodes 5-8, top 9-12. Node ids are written in SPECFEM order (0,0,0),(0,1,0),
(1,1,0),(1,0,0) then top:
```
2
1   1 2 3 4 5 6 7 8
2   5 6 7 8 9 10 11 12
```
**MUST VERIFY IN IMPLEMENTATION:** that the 8 ids per line, after the Gmsh→SPECFEM remap, actually
yield a positive Jacobian. Check with the shipped `xcheck_mesh_quality` utility (or inspect
`OUTPUT_FILES` for the "negative Jacobian" abort). See §3 step 7 and §5-risk-A.

## 1.3 `materials_file` (element → material index)

Reader:
```fortran
read(IIN_DB,*) num_mat, mat_flag        ! one line per element
```
- One line per element: `element_id  material_id`.
- `element_id` 1-based, matches `mesh_file`. Lines need not be pre-sorted (reader stores by id).
- `material_id` **> 0** → a material defined in `nummaterial_velocity_file`.
  `material_id` **< 0** → tomographic/interpolated model (not us; negative ids index tomo files).
- (Advanced: a fully-general form allows domain flags per element; for our uniform-per-domain case
  the two-column `element_id material_id` form is what the CUBIT exporter writes and is sufficient.)

Example (cube 1 = soil=material 1, cube 2 = concrete=material 2):
```
1  1
2  2
```

## 1.4 `nummaterial_velocity_file` (material index → physical properties)

Reader (elastic/acoustic branch):
```fortran
read(line,*) idomain_id, num_mat, rho, vp, vs, qkappa, qmu, aniso_flag
```
- One line per material id. Column meaning:
  `domain_id  material_id  rho  vp  vs  Qkappa  Qmu  anisotropy_flag`
  - `domain_id`: **1 = acoustic**, **2 = elastic/viscoelastic**. (Poroelastic uses a separate file.)
  - `material_id`: matches the positive ids in `materials_file`.
  - `rho` kg/m³, `vp` m/s, `vs` m/s.
  - `Qkappa`, `Qmu`: quality factors. Use **9999.0** to disable attenuation for that material
    (valid working range 1–9000 when attenuation is on). `ATTENUATION` toggle is in `DATA/Par_file`;
    reference frequency is `ATTENUATION_f0_REFERENCE`.
  - `anisotropy_flag`: **0 = isotropic** (our case).
- Lines may be prefixed with `#` comments; blank/comment lines are skipped by the reader.

See §4 for our concrete/soil/air numbers.

Example (matching the manual's own sample line format `2 1 2300 2800 1500 9999.0 9999.0 0`):
```
2  1  1800.0   400.0   200.0  9999.0  9999.0  0     # soil
2  2  2400.0  3600.0  2000.0  9999.0  9999.0  0     # concrete
```

## 1.5 Boundary-surface files (`free_or_absorbing_surface_file_zmax`, `absorbing_surface_file_*`)  ← #2 trap

ALL six surface files share **one identical format** (verified against `module_mesh.f90` reads for
`ibelm_xmin/…/top` and `cubit2specfem3d.py` writes):

Reader (shown for xmin; identical for xmax, ymin, ymax, bottom, and zmax/top):
```fortran
read(IIN_DB,*) nspec2D_xmin                                    ! line 1: face count
read(IIN_DB,*) ibelm_xmin(i), (nodes_ibelm_xmin(inode,i), inode=1,NGNOD2D)  ! one per face
```
- **Line 1:** number of boundary faces on this side, `N_faces`.
- **Lines 2..:** `element_id  c1 c2 c3 c4` where:
  - `element_id` = the **volume hex** (1-based, from `mesh_file`) that owns this boundary face.
    NOT a separate 2D element id — it points back into the 3D element list.
  - `c1 c2 c3 c4` = the **4 corner node ids** of that face (NGNOD2D=4 for HEX8), 1-based, referencing
    `nodes_coords_file`. They must be the four nodes of ONE face of that hex.
- **Face node order / orientation:** the four corners must be listed so the face is a coherent quad.
  The CUBIT exporter enforces an outward-consistent winding via a `normal_check()` that, if the
  computed face normal is wrong-signed, reverses to `nodes[0], nodes[3], nodes[2], nodes[1]`.
  **MUST VERIFY IN IMPLEMENTATION:** whether `xgenerate_databases` strictly requires a specific
  (outward vs inward) winding for absorbing faces, or only that the 4 nodes are the correct face set.
  Empirically the CUBIT exporter always writes outward-consistent winding; safest to replicate it
  (see §3 step 6). Getting the 4 nodes right but the winding wrong is the classic §2 "absorbing
  boundary silently reflects" bug.
- `free_or_absorbing_surface_file_zmax` uses the **same** format; whether the solver treats it as a
  free surface or a Stacey absorbing top is set by `STACEY_INSTEAD_OF_FREE_SURFACE` in `DATA/Par_file`,
  not by the file contents. For our houses: **free surface** (`.false.`) at zmax (soil top + air).

Example — bottom face of cube 1 (nodes 1,2,3,4) is the model's zmin absorbing face:
```
# absorbing_surface_file_bottom
1
1   1 2 3 4
```
Example — top face of cube 2 (nodes 9,10,11,12) is the free surface:
```
# free_or_absorbing_surface_file_zmax
1
2   9 10 11 12
```
(The four xmin/xmax/ymin/ymax files are built the same way from the corresponding side faces of the
outer soil box; each has its own count header.)

## 1.6 `absorbing_cpml_file` (only if PML)

First line = number of CPML spectral elements; then one line per CPML element: `element_id  cpml_flag`
where the flag encodes which PML region (X/Y/Z/XY/XZ/YZ/XYZ). Per adoption doc §5, we use **Stacey,
not PML**, for data-generation runs → this file is omitted. Documented here only for completeness.
**MUST VERIFY IN IMPLEMENTATION** the exact CPML flag integer codes if PML is ever enabled.

---

# 2. The Gmsh side and the node-order mapping

## 2.1 Element type

SPECFEM3D needs **all-hexahedra**, either HEX8 (`NGNOD=8`) or HEX27 (`NGNOD=27`). We use **HEX8**:
- Boxy axis-aligned houses mesh cleanly into HEX8 via transfinite volumes (§ adoption doc Stage 2).
- HEX27 buys higher geometric order (curved elements) we don't need for boxy geometry, at ~3× the
  node bookkeeping and a 27-node reorder map.
- SPECFEM3D's SEM accuracy comes from the internal GLL polynomial order (`NGLLX=5` in the solver),
  **not** from NGNOD. HEX8 elements still run at full spectral accuracy. NGNOD only controls the
  geometric (shape-function) order of the element mapping.

Gmsh element type code for 8-node hex = **type 5**. (HEX27 = type 12; not used.)

## 2.2 Gmsh HEX8 corner convention (VERIFIED, gmsh.info §10.2 Node ordering)

Gmsh local node order for an 8-node hex, reference cube [0,1]³:
```
  g1 = (0,0,0)  g2 = (1,0,0)  g3 = (1,1,0)  g4 = (0,1,0)     ! bottom, CCW starting +x
  g5 = (0,0,1)  g6 = (1,0,1)  g7 = (1,1,1)  g8 = (0,1,1)     ! top, same walk
```

## 2.3 The reorder map (Gmsh → SPECFEM3D)  ← verify this FIRST in implementation

Put the two conventions side by side by reference-cube coordinate:

| coord (x,y,z) | Gmsh local # | SPECFEM local # |
|---|---|---|
| (0,0,0) | 1 | 1 |
| (1,0,0) | 2 | 4 |
| (1,1,0) | 3 | 3 |
| (0,1,0) | 4 | 2 |
| (0,0,1) | 5 | 5 |
| (1,0,1) | 6 | 8 |
| (1,1,1) | 7 | 7 |
| (0,1,1) | 8 | 6 |

So to build a SPECFEM `mesh_file` line from a Gmsh hex whose node ids (in Gmsh order) are
`[g1..g8]`, place them into SPECFEM slots `[s1..s8]` as:

```
SPECFEM_order = [ g1, g4, g3, g2, g5, g8, g7, g6 ]
                   ^s1 ^s2 ^s3 ^s4 ^s5 ^s6 ^s7 ^s8
```
i.e. **swap the 2nd and 4th corners of each face** (positions 2↔4 on the bottom, 6↔8 on the top);
positions 1,3,5,7 are unchanged. This is exactly a reversal of the in-plane winding direction of
each face (Gmsh bottom is CCW looking down +z from above; SPECFEM's listed order (0,0,0)→(0,1,0)→
(1,1,0)→(1,0,0) is the opposite in-plane sense), which flips the sign of the geometric Jacobian back
to positive under SPECFEM's convention.

**MUST VERIFY IN IMPLEMENTATION — this is THE riskiest single line of the whole converter.** Two
independent things can each be off (the corner permutation, and whether the Jacobian ends up +),
and the failure mode is a solver that either aborts with "negative Jacobian" or, worse, runs and
produces subtly wrong wavefields. Verification recipe:
1. Build the trivial 2-cube mesh from §1, run `xdecompose_mesh` + `xgenerate_databases`, and confirm
   no "negative/zero Jacobian" error and that reported element volumes are positive.
2. Run the soil-only bridge case (adoption doc Stage 1) and confirm P/Rayleigh arrivals match
   pyprop8. A wrong reorder that still passes the Jacobian check will fail the bridge amplitudes.
3. Cross-check against the CUBIT exporter: mesh one box in Gmsh AND in CUBIT/Trelis, export both,
   diff a few `mesh_file` lines after mapping node ids by coordinate. (Optional but decisive.)

## 2.4 Gmsh face (quad) node order for the boundary files

A Gmsh quadrangle (`.msh` element type 3) lists 4 corners CCW. For the boundary files SPECFEM wants
the 4 corner node ids of the owning hex's face. Two viable strategies (§3 covers both):
- **(A) From physical surface groups:** if the Gmsh model tags each outer face as a physical surface,
  Gmsh emits quad elements for those faces; each quad's 4 node ids ARE `c1..c4`. You then find the
  owning hex (the unique volume element sharing all 4 nodes) to get `element_id`, and fix winding.
- **(B) Derived from the hex:** iterate hex faces directly (6 faces per hex, known local-node
  quadruples), keep only faces on the domain boundary planes (x=xmin, etc.), emit `element_id` +
  the 4 face nodes. This avoids needing physical surface groups for the outer box but you must know
  the face→plane assignment (trivial for an axis-aligned outer box by coordinate test).

## 2.5 MSH format v2 vs v4 (what the converter must parse)

- **MSH v2 (`$MeshFormat 2.2`)**: flat lists.
  - `$Nodes`: `count`, then `node_id x y z` per line.
  - `$Elements`: `count`, then `elm_id  elm_type  n_tags  tag1 tag2 …  node_id…`. For our meshes
    `tag1 = physical group id`, `tag2 = elementary/geometric entity id`. Physical group → material
    is read straight off `tag1`.
- **MSH v4 (`$MeshFormat 4.1`)**: block-structured by geometric entity.
  - `$Nodes`: header `numEntityBlocks numNodes …`; then per entity block a header
    `entityDim entityTag parametric numNodesInBlock`, a list of node tags, then a list of xyz.
  - `$Elements`: per entity block `entityDim entityTag elementType numElemsInBlock`, then
    `elm_id node…` per element. **Physical groups are NOT on the element line in v4** — they are
    attached to geometric entities via the `$Entities` section (each entity lists its physical
    tags), so you must join element → entityTag → physical group.

**Recommendation:** don't hand-parse. Use one of:
- the **`gmsh` Python API** (`gmsh.model.mesh.getNodes`, `getElements`,
  `gmsh.model.getPhysicalGroups`, `getEntitiesForPhysicalGroup`, `getElementsByType`) — version-proof,
  since you already build geometry with it (adoption doc Stage 2); OR
- **`meshio`** (`meshio.read`), which normalizes v2/v4 and exposes `mesh.cells` (`"hexahedron"`,
  `"quad"`) and `mesh.cell_data["gmsh:physical"]` uniformly.
The converter below is written against these APIs so the v2/v4 distinction disappears.

---

# 3. Converter algorithm — `gmsh2specfem3d.py`

Pseudocode-level but implementation-precise. Assumes the mesh was built by the Gmsh Python API
(adoption doc Stage 2) with these **physical groups** defined:

- Physical **volumes**: `soil` (→ material 1), `concrete` (→ material 2), optionally `air`
  (→ material 3, acoustic, domain_id 1) — see §4 for the air decision.
- Physical **surfaces** (only needed if using strategy A; otherwise derived): `zmax_free`,
  `xmin_abs`, `xmax_abs`, `ymin_abs`, `ymax_abs`, `zmin_abs`.

```
CONFIG:
  MAT_OF_PHYS = { "soil":1, "concrete":2, "air":3 }        # physical-volume-name → material id
  GMSH2SPECFEM = [0, 3, 2, 1, 4, 7, 6, 5]                  # 0-based reorder (§2.3): slot i takes gmsh[GMSH2SPECFEM[i]]
  SIDE_PLANES  = detect from global bbox: xmin,xmax,ymin,ymax,zmin,zmax

STEP 1 — load mesh.
  gmsh.open(mshfile)   OR   m = meshio.read(mshfile)
  Collect:
    nodes:  dict node_id -> (x,y,z)          # 1-based ids as Gmsh gives them
    hexes:  list of (gmsh_elem_id, [8 gmsh node_ids], phys_group_id)
    quads:  list of ([4 gmsh node_ids], phys_group_id)   # only if strategy A
  Build phys_group_id -> name map from gmsh.model.getPhysicalGroups()/getPhysicalName
  (meshio: physical id is in cell_data["gmsh:physical"]; names via field_data).

STEP 2 — renumber nodes to a clean contiguous 1..N (optional but recommended).
  Gmsh node ids are usually already 1..N and contiguous; if not, build
  remap = { old_id: new_id } sorted, and apply to every reference below.
  Write nodes_coords_file:
      line1 = N_nodes
      for nid in 1..N_nodes:  "%d %.10g %.10g %.10g" % (nid, x, y, z)

STEP 3 — assign each hex an element_id (1..N_elem in iteration order) and its material.
  for k, (gid, gnodes, phys) in enumerate(hexes, start=1):
      elem_id[k]     = k
      material_id[k] = MAT_OF_PHYS[ name_of[phys] ]       # 1=soil,2=concrete,3=air
      # remember mapping gid -> k for boundary-face owner lookup
      hex_by_gid[gid] = k
      spec_nodes[k]  = [ gnodes[GMSH2SPECFEM[i]] for i in 0..7 ]   # <-- §2.3 REORDER

STEP 4 — write mesh_file.
      line1 = N_elem
      for k in 1..N_elem:  "%d " + 8 node ids  ->  "%d %d %d %d %d %d %d %d %d" % (k, *spec_nodes[k])

STEP 5 — write materials_file and nummaterial_velocity_file.
      materials_file:            for k in 1..N_elem: "%d %d" % (k, material_id[k])
      nummaterial_velocity_file: for each material in use, emit
          "%d %d %g %g %g %g %g %d" % (domain_id, mat_id, rho, vp, vs, Qk, Qmu, aniso)
          (domain_id: 2 for soil/concrete elastic; 1 for air if modeled acoustic — §4)

STEP 6 — build boundary faces (the 6 surface files).
  Precompute the SPECFEM hex face table (which of the 8 local corners form each of the 6 faces,
  IN SPECFEM local numbering). For SPECFEM's convention (§1.2) the 6 faces are:
      bottom(zmin_local): [1,2,3,4]      top(zmax_local): [5,6,7,8]
      (the 4 side faces follow from the corner coordinates; enumerate by coordinate, see below)
  ---- Strategy B (recommended for the outer box; no surface phys-groups needed): ----
  for k in 1..N_elem:
      for face in 6 faces of hex k:
          fc = the 4 node ids of that face (global)
          if all 4 nodes lie on plane X==xmin (within tol):  add to list_xmin as (k, fc)
          elif ... xmax -> list_xmax ; ymin ; ymax ; zmin -> list_bottom
          elif all 4 on plane Z==zmax (global top):          add to list_zmax_free
      # interior faces (concrete/soil interface, room walls) are NOT boundary faces -> skip
  ---- Strategy A (if you tagged physical surfaces): ----
  for (qnodes, phys) in quads:
      owner_k = the hex whose node set contains all 4 qnodes (index via a node->hexes map)
      route by name_of[phys] into the matching list; fc = qnodes
  ---- winding fix (both strategies): ----
  for each (owner_k, fc) in each side list:
      n = face_normal(fc)                    # from node coords, right-hand rule on listed order
      want = outward normal of that side     # xmin:(-1,0,0), xmax:(+1,0,0), ymin:(0,-1,0),
                                             # ymax:(0,+1,0), bottom:(0,0,-1), zmax:(0,0,+1)
      if dot(n, want) < 0:  fc = [fc[0], fc[3], fc[2], fc[1]]   # reverse (mirror cubit2specfem3d)

STEP 7 — write the 6 surface files.
  for (fname, lst) in [("free_or_absorbing_surface_file_zmax", list_zmax_free),
                       ("absorbing_surface_file_xmin", list_xmin), ... , 
                       ("absorbing_surface_file_bottom", list_bottom)]:
      write line1 = len(lst)
      for (owner_k, fc) in lst:  "%d %d %d %d %d" % (owner_k, fc[0], fc[1], fc[2], fc[3])

STEP 8 — (optional) sanity self-checks before handing to xdecompose_mesh:
  - every node id referenced in mesh_file/surface files is in 1..N_nodes
  - every element_id in materials_file/surface files is in 1..N_elem
  - each hex has exactly 6 faces accounted (interior + boundary); boundary counts look sane
  - compute signed volume of each hex under SPECFEM ordering; assert > 0  (catches §2.3 error early)
```

**On "the room-facing wall faces should be free surface":** interior air-void wall faces are neither
outer boundary nor (if air is not meshed) shared with another solid element — they are simply
element faces with nothing on the other side, which SEM treats as a **traction-free (free) surface**
automatically. You do NOT list them in any surface file. Only the 5 outer soil sides (absorbing) and
the model top (free) go into the surface files. This matches adoption-doc §2 ("leave inner wall faces
as unloaded free faces, zero code").

**MUST VERIFY IN IMPLEMENTATION:** the exact local-corner quadruples for the 4 *side* faces under
SPECFEM numbering. Rather than hardcode them, Strategy B derives faces by coordinate-plane membership
(robust, convention-independent) — recommended. Only the reorder in STEP 3 depends on §2.3.

---

# 4. Material / velocity file for concrete / soil / air

Per material, one line in `nummaterial_velocity_file`:
`domain_id  material_id  rho  vp  vs  Qkappa  Qmu  aniso_flag`

Representative values (adoption doc §1d/§5 and standard geotechnical/concrete references — treat as
**defaults to be swept**, not fixed truth; the campaign randomizes soil):

```
# domain  id   rho     vp      vs     Qkappa   Qmu    aniso
  2       1    1800.0   400.0   200.0   50.0    30.0    0     # soil (elastic; Q on for realism)
  2       2    2400.0  3600.0  2000.0  200.0   150.0    0     # reinforced concrete (elastic)
```

Notes:
- **domain_id = 2** for both soil and concrete (elastic/viscoelastic). `aniso_flag = 0` (isotropic).
- **Attenuation:** to run with Q, set `ATTENUATION = .true.` in `DATA/Par_file` and set
  `ATTENUATION_f0_REFERENCE` to the band center (e.g. 50 Hz). To run elastic-lossless (e.g. the
  first bridge check), set both Q columns to **9999.0** (the reader's "no attenuation" sentinel) or
  set `ATTENUATION = .false.`. Keep the columns present either way.
- **Air:** three options, in order of preference for our problem:
  1. **Do NOT mesh the interior air (recommended).** Rooms are voids; inner wall faces become free
     surfaces (§3). No air material, no acoustic domain. Physically valid for f < ~500 Hz where
     air–structure coupling is negligible (adoption doc §2). Simplest and fastest.
  2. Mesh air as an **acoustic** domain: `domain_id = 1`, and the acoustic branch of
     `nummaterial_velocity_file` expects `rho vp` (vs ignored / set 0), e.g.
     `1  3  1.2  340.0  0.0  9999.0  9999.0  0`. Requires acoustic–elastic coupling surfaces — more
     files/complexity. **MUST VERIFY IN IMPLEMENTATION** the exact acoustic-line column expectation
     (some builds read only rho,vp for domain_id 1). Only pursue if option 1 proves insufficient.
  3. Mesh air as a very slow elastic solid — **not recommended** (crushes the CFL timestep for no
     physical gain).
- **material_id** must match the positive ids used in `materials_file`.

---

# 5. Honest gaps — MUST VERIFY IN IMPLEMENTATION

Ranked by how likely they are to silently corrupt results:

**A. (HIGHEST) The HEX8 node reorder map (§2.3).** Verified from the SPECFEM source comment and the
Gmsh docs *independently*, and they are self-consistent (each is a per-face winding reversal → sign
of Jacobian flips to +). But I could not run a mesh through the solver here. The permutation
`[g1,g4,g3,g2,g5,g8,g7,g6]` (0-based `[0,3,2,1,4,7,6,5]`) is my derived answer; **prove it** with the
§2.3 recipe (trivial 2-cube Jacobian check + soil-only pyprop8 bridge) before meshing any house. A
wrong map can pass decomposition and still produce wrong physics.

**B. Boundary-face winding requirement (§1.5).** The CUBIT exporter enforces outward-consistent
winding via `normal_check()`, so replicating it is safe. What I could NOT verify from source is
whether `xgenerate_databases` *rejects* wrong-wound absorbing faces or *silently* mis-applies the
Stacey traction. Replicate the outward-winding fix (STEP 6) and, in the bridge test, confirm
absorbing boundaries don't reflect (check late-time energy at an interior receiver).

**C. Top-surface filename.** Confirmed the decompose_mesh path wants
`free_or_absorbing_surface_file_zmax` (not `free_surface_file`, which is the internal-mesher name).
If you copy an internal-mesher example, you WILL hit a "file not found" or a mis-read. Verified from
readthedocs + `cubit2specfem3d.py`; still, diff against your installed SPECFEM version's
`module_mesh.f90` open() statements once, since filenames have drifted across versions.

**D. `materials_file` column count.** I verified the 2-column `element_id material_id` form (the
CUBIT exporter and the manual both use it). Some documentation shows an extended per-element form
with additional domain/property columns. For uniform-per-domain materials the 2-column form is
correct; **verify** your version's reader accepts it (it does in `master`).

**E. HEX27 (if ever needed).** The 27-node reorder map (Gmsh type 12 → SPECFEM 27-node order in
`doc/mesh_numbering_convention/numbering_convention_27_nodes.jpg`) is NOT worked out in this doc. We
use HEX8; if HEX27 becomes necessary, the corner-8 map above still applies to the first 8 but the 12
edge + 6 face + 1 center nodes need their own verified permutation. **Do not assume.**

**F. Acoustic air line format (§4 option 2).** Column expectation for `domain_id = 1` unverified
against source; avoid by using option 1 (don't mesh air).

**G. Node/element id contiguity.** The reader stores by the id field, so non-contiguous ids *should*
work, but `xdecompose_mesh`/`xgenerate_databases` and downstream tools are happiest with contiguous
1..N. The converter emits contiguous ids (STEP 2/3) to sidestep this — keep it that way.

---

# 6. Primary sources

- SPECFEM3D external-mesh file list, formats, `nummaterial_velocity_file` syntax, free-vs-absorbing
  top and `STACEY_INSTEAD_OF_FREE_SURFACE`:
  https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/
- Exact Fortran read statements + default filenames (`open(file=localpath_name//'/…')`), NGNOD/NGNOD2D,
  and the HEX8 corner comment (point1=(0,0,0), point2=(0,1,0), point3=(1,1,0), point4=(1,0,0), top 5–8):
  `src/decompose_mesh/module_mesh.f90`,
  https://github.com/SPECFEM/specfem3d/blob/master/src/decompose_mesh/module_mesh.f90
- Reference write formats for all files incl. the surface `element_id + 4 face nodes` line and the
  `normal_check()` winding fix (`nodes[0],nodes[3],nodes[2],nodes[1]`):
  `CUBIT_GEOCUBIT/geocubitlib/cubit2specfem3d.py`,
  https://github.com/SPECFEM/specfem3d/blob/master/CUBIT_GEOCUBIT/geocubitlib/cubit2specfem3d.py
- CUBIT_GEOCUBIT export workflow (define blocks for materials + absorbing boundaries; call
  `export2SPECFEM3D` last): https://github.com/SPECFEM/specfem3d/blob/master/CUBIT_GEOCUBIT/README.md
- Gmsh HEX8 (type 5) local node ordering and MSH v2/v4 structure + physical groups:
  https://gmsh.info/doc/texinfo/gmsh.html (§10.2 Node ordering; §10.1 file formats)
- Confirmation of Gmsh↔other-tool ordering differences (winding), general node-order caveat:
  https://gitlab.onelab.info/gmsh/gmsh/-/issues/1478 ,
  https://gmsh.geuz.narkive.com/7DeWOpMF/local-referential-for-the-hexahedron-and-prism-element-family
- Prior: `research/house_localization/12_specfem3d_adoption.md` (path decision, mesh scale, Stacey
  vs PML, air-void free faces, bridge validation).
