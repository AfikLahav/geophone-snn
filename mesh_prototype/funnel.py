import pickle, numpy as np
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
g_inner=g_wall=g_extra=g_rooms=g_conn=g_na=g_wd=g_axis=0
aaf_pass=[]; wdm=[]
for p in plans:
    inner=p.get("inner")
    if inner is None or inner.is_empty or inner.geom_type!="Polygon": continue
    g_inner+=1
    if not parts(p.get("wall")): continue
    g_wall+=1
    if any(parts(p.get(k)) for k in ("balcony","balacony","pool","garden","parking")): continue
    g_extra+=1
    nbed,nbath,nkit,nliv=npart(p,"bedroom"),npart(p,"bathroom"),npart(p,"kitchen"),npart(p,"living")
    if not (2<=nbed<=3 and nbath>=1 and nkit==1 and nliv==1): continue
    g_rooms+=1
    if not(parts(p.get("door")) and parts(p.get("window")) and parts(p.get("front_door"))): continue
    g_conn+=1
    na=p.get("net_area")
    if na is None or not (55<=na<=125): continue
    g_na+=1
    scale=float(np.sqrt(na/inner.area)); wd_m=(p.get("wall_depth") or 0)*scale; wdm.append(wd_m)
    if not (0.12<=wd_m<=0.30): continue
    g_wd+=1
    aaf=axis_frac(p.get("wall")); aaf_pass.append(aaf)
    if aaf<0.985: continue
    g_axis+=1
print("funnel: inner=%d wall=%d noextra=%d rooms=%d conn=%d na=%d wd=%d axis=%d"%(g_inner,g_wall,g_extra,g_rooms,g_conn,g_na,g_wd,g_axis))
if wdm: print("wall_depth_m at na-gate: p5 %.3f med %.3f p95 %.3f"%(np.percentile(wdm,5),np.median(wdm),np.percentile(wdm,95)))
if aaf_pass: print("axis_frac at wd-gate: p5 %.3f med %.3f max %.3f  (>=0.985 needed)"%(np.percentile(aaf_pass,5),np.median(aaf_pass),np.max(aaf_pass)))
