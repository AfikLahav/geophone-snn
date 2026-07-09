"""House #1 selection from ResPlan (inner is MultiPolygon). docs 03/14 -> Archetype B, axis-aligned."""
import pickle, numpy as np, json
PKL=r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/datasets/resplan/ResPlan.pkl"
plans=pickle.load(open(PKL,"rb"))
def parts(g):
    if g is None or getattr(g,"is_empty",True): return []
    return list(g.geoms) if hasattr(g,"geoms") else [g]
def npart(p,k): return len(parts(p.get(k)))
def axis_frac(g,tol=0.75):
    tot=ok=0.0
    for pp in parts(g):
        for r in [pp.exterior,*pp.interiors]:
            c=np.asarray(r.coords);d=np.diff(c,axis=0);L=np.hypot(d[:,0],d[:,1])
            m=(np.abs(d[:,0])<tol)|(np.abs(d[:,1])<tol);tot+=L.sum();ok+=L[m].sum()
    return ok/tot if tot>0 else 0.0
cands=[]
for i,p in enumerate(plans):
    inner=p.get("inner")
    if inner is None or inner.is_empty or inner.geom_type not in ("Polygon","MultiPolygon"): continue
    if not parts(p.get("wall")): continue
    if any(parts(p.get(k)) for k in ("balcony","balacony","pool","garden","parking")): continue
    nbed,nbath,nkit,nliv=npart(p,"bedroom"),npart(p,"bathroom"),npart(p,"kitchen"),npart(p,"living")
    if not (2<=nbed<=3 and nbath>=1 and nkit==1 and nliv==1): continue
    if not(parts(p.get("door")) and parts(p.get("window")) and parts(p.get("front_door"))): continue
    na=p.get("net_area")
    if na is None or not (55<=na<=125): continue
    scale=float(np.sqrt(na/inner.area)); wd_m=(p.get("wall_depth") or 0)*scale
    if not (0.12<=wd_m<=0.30): continue
    aaf=axis_frac(p.get("wall"))
    if aaf<0.98: continue
    x1,y1,x2,y2=inner.bounds; env=(x2-x1)*(y2-y1); rect=inner.area/env if env>0 else 0
    nwall=len(parts(p.get("wall"))); ninner=len(parts(inner)); span=max(x2-x1,y2-y1)*scale
    score=rect*2 - (ninner-1)*0.3 - abs(nwall-9)*0.02 - abs(na-92)*0.004 + aaf
    cands.append((score,i,na,wd_m,rect,nwall,ninner,nbed,nbath,span,scale))
cands.sort(reverse=True)
print(f"candidates: {len(cands)}")
print(f"{'rk':>3}{'idx':>7}{'net':>6}{'wall_m':>7}{'rect':>6}{'nwall':>6}{'ninr':>5}{'bed':>4}{'bth':>4}{'span':>6}")
for r,c in enumerate(cands[:10]):
    s,i,na,wd,rc,nw,ni,nb,nba,sp,sc=c
    print(f"{r:>3}{i:>7}{na:>6.0f}{wd:>7.3f}{rc:>6.2f}{nw:>6}{ni:>5}{nb:>4}{nba:>4}{sp:>6.1f}")
if cands:
    c=cands[0]
    json.dump({"idx":c[1],"scale":c[10],"net_area":c[2],"wall_m":c[3],"span_m":c[9]},
              open(r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/mesh_prototype/house1_idx.json","w"))
    print("CHOSEN idx=%d  net=%.0f m2  wall=%.3f m  span=%.1f m"%(c[1],c[2],c[3],c[9]))
