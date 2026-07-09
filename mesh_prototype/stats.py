import pickle, json, numpy as np
MP=r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/mesh_prototype"
meta=json.load(open(MP+"/house1_idx.json")); IDX=meta["idx"]; SC=meta["scale"]
plans=pickle.load(open(r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/datasets/resplan/ResPlan.pkl","rb"))
p=plans[IDX]
def parts(g):
    if g is None or getattr(g,"is_empty",True): return []
    return list(g.geoms) if hasattr(g,"geoms") else [g]
A=SC*SC  # canvas area -> m2

print("========== ResPlan plan %d ==========" % IDX)
print("net_area (listed): %.1f m2   | area (gross): %.1f m2" % (p.get("net_area",0), p.get("area",0)))
inner=p["inner"]; x1,y1,x2,y2=inner.bounds
print("footprint: %.2f x %.2f m  (inner area %.1f m2)" % ((x2-x1)*SC,(y2-y1)*SC, inner.area*A))
print("wall_depth: %.3f m" % (p.get("wall_depth",0)*SC))
print("\n-- rooms --")
for k in ["living","bedroom","bathroom","kitchen","storage","stair"]:
    ps=parts(p.get(k))
    if ps:
        ars=sorted([g.area*A for g in ps], reverse=True)
        print("  %-9s x%d : %s m2 (total %.1f)" % (k,len(ps)," + ".join("%.1f"%a for a in ars), sum(ars)))
print("\n-- openings & walls --")
print("  doors: %d  windows: %d  front_door: %d" % (len(parts(p.get("door"))),len(parts(p.get("window"))),len(parts(p.get("front_door")))))
wl=parts(p.get("wall"))
wlen=sum(g.length*SC for g in wl)/2  # perimeter/2 ~ centerline length approx
warea=sum(g.area*A for g in wl)
print("  wall segments: %d   wall footprint area: %.2f m2   (~%.1f m3 concrete at 2.7m)" % (len(wl), warea, warea*2.7))

# ----- 3D model stats -----
print("\n========== 3D model ==========")
CEIL=2.7; SLAB=0.2; ROOF=0.2
foot=inner.area*A
print("building height: %.1f m (floor slab .20 + walls 2.70 + roof .20)" % (SLAB+CEIL+ROOF))
print("concrete volume: floor %.1f + roof %.1f + walls %.1f = %.1f m3" % (foot*SLAB, foot*ROOF, warea*CEIL, foot*SLAB+foot*ROOF+warea*CEIL))
print("concrete mass ~ %.0f kg (2400 kg/m3)" % ((foot*SLAB+foot*ROOF+warea*CEIL)*2400))

# ----- mesh stats -----
import re
msh=open(MP+"/house1.msh").read()
print("\n========== mesh (the polygons you see) ==========")
# count nodes/elements from gmsh via reload
import gmsh
gmsh.initialize(); gmsh.open(MP+"/house1.msh")
nt,_,_=gmsh.model.mesh.getNodes(); print("nodes: %d" % len(nt))
ets,etags,_=gmsh.model.mesh.getElements()
tot=0
for et,tg in zip(ets,etags):
    name={1:"line",2:"triangle",4:"tetrahedron"}.get(et,"type%d"%et)
    print("  %-12s : %d" % (name, len(tg))); tot+=len(tg) if et==4 else 0
gmsh.finalize()
