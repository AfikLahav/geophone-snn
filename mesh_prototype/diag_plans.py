import pickle, numpy as np
from shapely.geometry import Polygon
PKL = r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/datasets/resplan/ResPlan.pkl"
plans = pickle.load(open(PKL,"rb"))
def parts(g):
    if g is None or g.is_empty: return []
    return list(g.geoms) if hasattr(g,"geoms") else [g]

# distributions
na = np.array([p.get("net_area") or np.nan for p in plans], float)
ar = np.array([p.get("area") or np.nan for p in plans], float)
wd = np.array([p.get("wall_depth") or np.nan for p in plans], float)
innerA = np.array([p["inner"].area if isinstance(p.get("inner"),Polygon) else np.nan for p in plans], float)
print("net_area   : min %.1f  med %.1f  max %.1f" % (np.nanmin(na),np.nanmedian(na),np.nanmax(na)))
print("area       : min %.1f  med %.1f  max %.1f" % (np.nanmin(ar),np.nanmedian(ar),np.nanmax(ar)))
print("wall_depth : min %.2f  med %.2f  max %.2f (canvas units)" % (np.nanmin(wd),np.nanmedian(wd),np.nanmax(wd)))
print("inner.area : min %.0f  med %.0f  max %.0f (canvas^2)" % (np.nanmin(innerA),np.nanmedian(innerA),np.nanmax(innerA)))
# is net_area already m2? implied scale = sqrt(net_area / inner.area)
sc = np.sqrt(na/innerA)
print("implied scale sqrt(net/inner): med %.4f  -> wall_depth*scale med %.3f m" % (np.nanmedian(sc), np.nanmedian(wd*sc)))
# funnel
def parts_n(p,k): return len(parts(p.get(k)))
c_inner = c_wall = c_rooms = c_conn = c_na = 0
for p in plans:
    inner=p.get("inner")
    if not isinstance(inner,Polygon) or inner.is_empty: continue
    c_inner+=1
    if not parts(p.get("wall")): continue
    c_wall+=1
    nbed,nbath,nkit,nliv = parts_n(p,"bedroom"),parts_n(p,"bathroom"),parts_n(p,"kitchen"),parts_n(p,"living")
    if not (2<=nbed<=3 and nbath>=1 and nkit==1 and nliv==1): continue
    c_rooms+=1
    if not(parts(p.get("door")) and parts(p.get("window")) and parts(p.get("front_door"))): continue
    c_conn+=1
    x=p.get("net_area")
    if x is not None and 55<=x<=130: c_na+=1
print("funnel: inner=%d wall=%d rooms(2-3bed)=%d +conn=%d +netarea55-130=%d"%(c_inner,c_wall,c_rooms,c_conn,c_na))
