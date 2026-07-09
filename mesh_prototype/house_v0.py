"""E1.0 prototype v0 — parametric boxy house -> mesh -> 3D render.
Purpose: validate the gmsh->pyvista toolchain + SEE the house. (tets for now; all-hex is v1.)"""
import gmsh, numpy as np, pyvista as pv

# ---- parametric house (meters) ----
rw, rl, rh, t = 4.0, 5.0, 2.7, 0.20      # room W,L,ceiling height, wall/slab thickness
Hw, Hl = rw + 2*t, rl + 2*t               # house footprint incl. walls
pad, depth = 1.5, 2.0                      # soil extends around + below

gmsh.initialize()
gmsh.model.add("house")
occ = gmsh.model.occ
defs = []
def box(mat, x, y, z, dx, dy, dz):
    tag = occ.addBox(x, y, z, dx, dy, dz); defs.append((mat, tag))

box("soil", -pad, -pad, -depth, Hw+2*pad, Hl+2*pad, depth)   # ground
box("slab", 0, 0, 0, Hw, Hl, t)                               # floor slab
box("wall", 0, 0, t, Hw, t, rh)                               # S wall
box("wall", 0, Hl-t, t, Hw, t, rh)                            # N wall
box("wall", 0, t, t, t, rl, rh)                               # W wall
box("wall", Hw-t, t, t, t, rl, rh)                            # E wall
box("slab", 0, 0, t+rh, Hw, Hl, t)                            # roof slab
occ.synchronize()

mats = {}
for mat, tag in defs: mats.setdefault(mat, []).append(tag)
for i, (mat, tags) in enumerate(mats.items(), 1):
    pg = gmsh.model.addPhysicalGroup(3, tags, i); gmsh.model.setPhysicalName(3, pg, mat)

gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
gmsh.model.mesh.generate(3)

# ---- extract per-material submeshes into pyvista ----
ntags, ncoords, _ = gmsh.model.mesh.getNodes()
pts = np.array(ncoords).reshape(-1, 3)
tag2idx = {int(tg): i for i, tg in enumerate(ntags)}
VTK = {4: 10, 5: 12}; NPE = {4: 4, 5: 8}  # gmsh tet=4/hex=5 -> VTK tetra=10/hex=12
elem_types = {}
grids = {}
for mat, tags in mats.items():
    cells, ctypes = [], []
    for vt in tags:
        ets, _, enodes = gmsh.model.mesh.getElements(3, vt)
        for et, en in zip(ets, enodes):
            n = NPE[et]; elem_types[et] = elem_types.get(et, 0) + len(en)//n
            for row in np.array(en).reshape(-1, n):
                idx = [tag2idx[int(x)] for x in row]
                cells += [n] + idx; ctypes.append(VTK[et])
    if ctypes:
        grids[mat] = pv.UnstructuredGrid(np.array(cells), np.array(ctypes), pts)

gmsh.write("S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/mesh_prototype/house_v0.msh")
print("=== MESH STATS ===")
print("nodes:", len(pts))
print("element types (gmsh 4=tet,5=hex):", {k: v for k, v in elem_types.items()})
for mat, g in grids.items(): print(f"  {mat}: {g.n_cells} cells")

# ---- render ----
col = {"soil": "#8a6d3b", "slab": "#9aa6ad", "wall": "#c3c8cc"}
op  = {"soil": 0.22, "slab": 1.0, "wall": 1.0}
p = pv.Plotter(off_screen=True, window_size=(1300, 950))
p.set_background("white")
for mat, g in grids.items():
    p.add_mesh(g, color=col.get(mat, "pink"), opacity=op.get(mat, 1.0),
               show_edges=True, edge_color="#555555", line_width=0.4)
p.add_axes()
p.camera_position = 'iso'; p.camera.azimuth = 35; p.camera.elevation = 15
out = "S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/mesh_prototype/house_v0.png"
p.screenshot(out)
print("saved:", out)
