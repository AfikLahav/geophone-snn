"""Interactive 3D house viewer — rotate/zoom with the mouse. Close the window when done."""
import gmsh, numpy as np, pyvista as pv
rw, rl, rh, t = 4.0, 5.0, 2.7, 0.20
Hw, Hl = rw + 2*t, rl + 2*t
pad, depth = 1.5, 2.0
gmsh.initialize(); gmsh.model.add("house"); occ = gmsh.model.occ
defs = []
def box(mat, x, y, z, dx, dy, dz):
    occ.addBox(x, y, z, dx, dy, dz); defs.append((mat, occ.getMaxTag(3)))
box("soil", -pad,-pad,-depth, Hw+2*pad, Hl+2*pad, depth)
box("slab", 0,0,0, Hw,Hl,t)
box("wall", 0,0,t, Hw,t,rh); box("wall", 0,Hl-t,t, Hw,t,rh)
box("wall", 0,t,t, t,rl,rh); box("wall", Hw-t,t,t, t,rl,rh)
box("slab", 0,0,t+rh, Hw,Hl,t)
occ.synchronize()
mats = {}
for mat, tag in defs: mats.setdefault(mat, []).append(tag)
for i,(m,tags) in enumerate(mats.items(),1):
    gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3,tags,i), m)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4); gmsh.model.mesh.generate(3)
ntags, nc, _ = gmsh.model.mesh.getNodes(); pts = np.array(nc).reshape(-1,3)
t2i = {int(tg):i for i,tg in enumerate(ntags)}; VTK={4:10,5:12}; NPE={4:4,5:8}
grids={}
for mat,tags in mats.items():
    cells,ct=[],[]
    for vt in tags:
        ets,_,en = gmsh.model.mesh.getElements(3,vt)
        for et,e in zip(ets,en):
            n=NPE[et]
            for row in np.array(e).reshape(-1,n):
                cells+=[n]+[t2i[int(x)] for x in row]; ct.append(VTK[et])
    if ct: grids[mat]=pv.UnstructuredGrid(np.array(cells),np.array(ct),pts)
col={"soil":"#8a6d3b","slab":"#9aa6ad","wall":"#c3c8cc"}; op={"soil":0.22,"slab":1.0,"wall":1.0}
p=pv.Plotter(window_size=(1300,950)); p.set_background("white")
for mat,g in grids.items():
    p.add_mesh(g,color=col.get(mat,"pink"),opacity=op.get(mat,1.0),show_edges=True,edge_color="#555555",line_width=0.4)
p.add_axes(); p.camera_position='iso'
p.add_text("Boxy house v0  (drag=rotate, scroll=zoom)", font_size=11)
p.show()
