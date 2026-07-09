"""House #1 — real ResPlan plan 14450 -> 3D (soil + floor slab + walls + translucent roof) -> render.
Dims per docs 03/14: wall 0.20 m, slab 0.20 m, ceiling 2.7 m. Coarse mesh (viz, not sim).
Usage: build_house1.py [show]   (show = interactive window; default = save PNG)"""
import sys, pickle, json, numpy as np, gmsh, pyvista as pv

MP = r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/mesh_prototype"
meta = json.load(open(MP + "/house1_idx.json")); IDX = meta["idx"]; SCALE = meta["scale"]
plans = pickle.load(open(r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/datasets/resplan/ResPlan.pkl", "rb"))
plan = plans[IDX]
def parts(g):
    if g is None or getattr(g, "is_empty", True): return []
    return list(g.geoms) if hasattr(g, "geoms") else [g]

inner = plan["inner"]; cx, cy = inner.centroid.x, inner.centroid.y
def xf(coords): return [((x - cx) * SCALE, (y - cy) * SCALE) for x, y in coords]
inner_polys, wall_polys = parts(inner), parts(plan["wall"])
x1, y1, x2, y2 = inner.bounds; W = (x2 - x1) * SCALE; L = (y2 - y1) * SCALE
SLAB, CEIL, ROOF, SOIL_D, PAD = 0.20, 2.70, 0.20, 2.0, 1.5

gmsh.initialize(); gmsh.model.add("house1"); occ = gmsh.model.occ
def ring(coords, z):
    c = xf(coords)
    if c[0] == c[-1]: c = c[:-1]
    pts = [occ.addPoint(x, y, z) for x, y in c]
    ls = [occ.addLine(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
    return occ.addCurveLoop(ls)
def prism(poly, z0, z1):
    loops = [ring(list(poly.exterior.coords), z0)] + [ring(list(r.coords), z0) for r in poly.interiors]
    s = occ.addPlaneSurface(loops)
    return [t for d, t in occ.extrude([(2, s)], 0, 0, z1 - z0) if d == 3]

matvol = {"soil": [], "slab": [], "wall": [], "roof": []}
matvol["soil"].append(occ.addBox(-W/2-PAD, -L/2-PAD, -SOIL_D, W+2*PAD, L+2*PAD, SOIL_D))
for ip in inner_polys:
    matvol["slab"] += prism(ip, -SLAB, 0.0)
    matvol["roof"] += prism(ip, CEIL, CEIL + ROOF)
for wp in wall_polys:
    matvol["wall"] += prism(wp, 0.0, CEIL)
occ.synchronize()
for i, (m, vols) in enumerate(matvol.items(), 1):
    if vols: gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, vols, i), m)

gmsh.option.setNumber("Mesh.MeshSizeMax", 0.45); gmsh.option.setNumber("Mesh.MeshSizeMin", 0.12)
gmsh.model.mesh.generate(3); gmsh.write(MP + "/house1.msh")

ntags, nc, _ = gmsh.model.mesh.getNodes(); pts = np.array(nc).reshape(-1, 3)
t2i = {int(t): i for i, t in enumerate(ntags)}; VTK = {4: 10, 5: 12}; NPE = {4: 4, 5: 8}
grids = {}; tot = 0
for m, vols in matvol.items():
    cells, ct = [], []
    for v in vols:
        ets, _, en = gmsh.model.mesh.getElements(3, v)
        for et, e in zip(ets, en):
            n = NPE[et]
            for row in np.array(e).reshape(-1, n):
                cells += [n] + [t2i[int(x)] for x in row]; ct.append(VTK[et])
    if ct: grids[m] = pv.UnstructuredGrid(np.array(cells), np.array(ct), pts); tot += len(ct)
print(f"House #1 (plan {IDX}): {W:.1f} x {L:.1f} m | nodes {len(pts)} | elements {tot}")
for m, g in grids.items(): print(f"   {m}: {g.n_cells}")

col = {"soil": "#9b7d54", "slab": "#8f979d", "wall": "#c7ccd0", "roof": "#aeb4b9"}
op  = {"soil": 0.13, "slab": 1.0, "wall": 1.0, "roof": 0.15}
SHOW = (len(sys.argv) > 1 and sys.argv[1] == "show")
p = pv.Plotter(off_screen=not SHOW, window_size=(1500, 1050)); p.set_background("white")
for m in ("soil", "slab", "wall", "roof"):
    if m in grids:
        p.add_mesh(grids[m], color=col[m], opacity=op[m], show_edges=(m not in ("soil", "roof")),
                   edge_color="#6a6f74", line_width=0.3)
p.add_axes(); p.camera_position = 'iso'; p.camera.azimuth = 40; p.camera.elevation = 28
p.add_text(f"House #1 - ResPlan {IDX}  ({W:.1f}x{L:.1f} m, 0.20 m walls, translucent roof)",
           font_size=11, color="black")
if SHOW:
    p.add_text("drag = rotate   scroll = zoom", position="lower_right", font_size=9, color="gray")
    p.show()
else:
    p.screenshot(MP + "/house1.png"); print("saved:", MP + "/house1.png")
