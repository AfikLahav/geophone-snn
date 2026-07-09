"""E1.0 — all-hex simulation-grade mesh for House #1 (ResPlan 14450).

Tensor-banded structured hex grid for axis-aligned geometry (doc 02 box-lattice,
simplified): x/y/z breakpoints from wall positions -> per-interval subdivision
(fine through wall thickness, coarse in soil/in-plane) -> perfect rectangular
HEX8 bricks, conforming by construction. Air cells omitted (unlisted faces =
free surface, per specfem3d_build/01). Output: house1_hex.npz + cutaway PNG.

Usage: hexmesh_house1.py [coarse|prod] [show]
  coarse: wall 0.10 m through-thickness (2 elems) — fast iteration
  prod  : wall 0.05 m through-thickness (4 elems) — doc-12 sim grade
"""
import sys, json, pickle
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely import prepared

MP = r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/mesh_prototype"
MODE = "prod" if "prod" in sys.argv[1:] else "coarse"
SHOW = "show" in sys.argv[1:]

# ---- target element sizes (m) ----
T_WALL = 0.05 if MODE == "prod" else 0.10   # through wall thickness
T_PLANE = 0.30                               # concrete in-plane / air spans
T_SOIL = 0.40                                # soil bulk
T_SLABZ = 0.05 if MODE == "prod" else 0.10   # through slab/roof thickness
SNAP = 0.01                                  # coordinate snap grid
MERGE = 0.06                                 # merge breakpoints closer than this

# ---- load House #1 geometry (meters, centered) ----
meta = json.load(open(MP + "/house1_idx.json")); IDX, SC = meta["idx"], meta["scale"]
plans = pickle.load(open(r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/datasets/resplan/ResPlan.pkl", "rb"))
plan = plans[IDX]
def parts(g):
    if g is None or getattr(g, "is_empty", True): return []
    return list(g.geoms) if hasattr(g, "geoms") else [g]
from shapely.affinity import scale as shp_scale, translate as shp_translate
inner = plan["inner"]; cx, cy = inner.centroid.x, inner.centroid.y
def to_m(g):
    return shp_scale(shp_translate(g, -cx, -cy), SC, SC, origin=(0, 0))
inner_m = to_m(inner)
walls_m = unary_union([to_m(w) for w in parts(plan["wall"])])
foot_m = unary_union([inner_m, walls_m])            # slab/roof footprint
x1, y1, x2, y2 = foot_m.bounds
SLAB, CEIL, ROOF, SOIL_D, PAD = 0.20, 2.70, 0.20, 2.0, 1.5

# ---- breakpoints ----
def snap(v): return round(round(v / SNAP) * SNAP, 4)
def breakpoints(axis):
    bp = set()
    for poly in parts(walls_m) + parts(foot_m):
        for ring in [poly.exterior, *poly.interiors]:
            for c in ring.coords: bp.add(snap(c[0] if axis == 0 else c[1]))
    lo = snap((x1 if axis == 0 else y1) - PAD); hi = snap((x2 if axis == 0 else y2) + PAD)
    bp |= {lo, hi}
    bp = sorted(b for b in bp if lo - 1e-9 <= b <= hi + 1e-9)
    out = [bp[0]]                                    # merge near-duplicates
    for b in bp[1:]:
        if b - out[-1] < MERGE: out[-1] = out[-1] if len(out) == 1 else out[-1]
        else: out.append(b)
    out[-1] = hi
    return out
XB, YB = breakpoints(0), breakpoints(1)
ZB = [-SOIL_D - SLAB, -SLAB, 0.0, CEIL, CEIL + ROOF]  # soil | slab band | walls | roof

# ---- per-interval subdivision ----
wall_prep = prepared.prep(walls_m.buffer(0.001))
foot_prep = prepared.prep(foot_m.buffer(0.001))
def subdivide(a, b, target):
    n = max(1, int(np.ceil((b - a) / target)))
    return list(np.linspace(a, b, n + 1))[1:]
def axis_coords(B, axis):
    coords = [B[0]]
    for a, b in zip(B[:-1], B[1:]):
        w = b - a
        # fine if this interval is a wall through-thickness band (narrow interval inside wall zone)
        mid = (a + b) / 2
        probe = Point(mid, (y1+y2)/2) if axis == 0 else Point((x1+x2)/2, mid)
        fine = w <= 0.35 and a >= (x1 if axis == 0 else y1) - 0.01 and b <= (x2 if axis == 0 else y2) + 0.01
        t = T_WALL if (fine and w <= 0.35) else (T_PLANE if foot_prep.contains(probe) else T_SOIL)
        coords += subdivide(a, b, t)
    return np.array(coords)
xs, ys = axis_coords(XB, 0), axis_coords(YB, 1)
zs = np.array([ZB[0]] + subdivide(ZB[0], ZB[1], T_SOIL)                    # soil
             + subdivide(ZB[1], ZB[2], T_SLABZ)                            # slab
             + subdivide(ZB[2], ZB[3], T_PLANE)                            # wall storey
             + subdivide(ZB[3], ZB[4], T_SLABZ))                           # roof
nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1

# ---- material per cell (0=void/air, 1=soil, 2=concrete) ----
xc, yc = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
zc = (zs[:-1] + zs[1:]) / 2
in_wall = np.zeros((nx, ny), bool); in_foot = np.zeros((nx, ny), bool)
for i, xv in enumerate(xc):
    for j, yv in enumerate(yc):
        p = Point(xv, yv)
        in_wall[i, j] = wall_prep.contains(p)
        in_foot[i, j] = foot_prep.contains(p)
mat = np.zeros((nx, ny, nz), np.int8)
for k, zv in enumerate(zc):
    if zv < -SLAB:                    mat[:, :, k] = 1                       # soil layer
    elif zv < 0:                      mat[:, :, k] = np.where(in_foot, 2, 1) # slab | soil ring
    elif zv < CEIL:                   mat[:, :, k] = np.where(in_wall, 2, 0) # walls | air
    else:                             mat[:, :, k] = np.where(in_foot, 2, 0) # roof | air

# ---- extract hexes (drop void) ----
solid = np.argwhere(mat > 0)
nid = -np.ones((nx + 1, ny + 1, nz + 1), np.int64)
def node_id(i, j, k):
    if nid[i, j, k] < 0: nid[i, j, k] = node_id.n; node_id.n += 1
    return nid[i, j, k]
node_id.n = 0
hexes = np.empty((len(solid), 8), np.int64); hmat = np.empty(len(solid), np.int8)
for c, (i, j, k) in enumerate(solid):
    hexes[c] = [node_id(i, j, k), node_id(i+1, j, k), node_id(i+1, j+1, k), node_id(i, j+1, k),
                node_id(i, j, k+1), node_id(i+1, j, k+1), node_id(i+1, j+1, k+1), node_id(i, j+1, k+1)]
    hmat[c] = mat[i, j, k]
pts = np.empty((node_id.n, 3))
for (i, j, k), v in np.ndenumerate(nid):
    if v >= 0: pts[v] = (xs[i], ys[j], zs[k])

# ---- stats + gates ----
dxs = np.diff(xs); dys = np.diff(ys); dzs = np.diff(zs)
print(f"=== E1.0 all-hex mesh — House #1, mode={MODE} ===")
print(f"grid: {nx} x {ny} x {nz}  |  solid hexes: {len(hexes):,}  (soil {int((hmat==1).sum()):,}, concrete {int((hmat==2).sum()):,})")
print(f"nodes: {node_id.n:,}   GLL points (NGLL5): {len(hexes)*125:,}")
print(f"dx: {dxs.min():.3f}-{dxs.max():.3f}  dy: {dys.min():.3f}-{dys.max():.3f}  dz: {dzs.min():.3f}-{dzs.max():.3f} m")
print(f"max aspect ratio: {max(dxs.max(),dys.max(),dzs.max())/min(dxs.min(),dys.min(),dzs.min()):.1f}")
print("GATES: pure HEX8 = True (by construction) | conforming = True (tensor grid) | SICN = 1.0 (rectangular bricks)")
wall_bands = [ (a,b) for a,b in zip(XB[:-1],XB[1:]) if b-a <= 0.35 and a >= x1-0.01 and b <= x2+0.01 ]
print(f"wall x-bands: {len(wall_bands)} (elems through-thickness ~{int(np.ceil(0.20/T_WALL))})")
np.savez_compressed(MP + f"/house1_hex_{MODE}.npz", points=pts, hexes=hexes, mat=hmat,
                    xs=xs, ys=ys, zs=zs)
print(f"saved: house1_hex_{MODE}.npz")

# ---- cutaway render ----
import pyvista as pv
cells = np.hstack([np.full((len(hexes), 1), 8, np.int64), hexes]).ravel()
grid = pv.UnstructuredGrid(cells, np.full(len(hexes), 12, np.uint8), pts)
grid.cell_data["mat"] = hmat
clipped = grid.clip(normal=[0, 1, 0], origin=[0, 0.3, 0])   # cut through the house
p = pv.Plotter(off_screen=not SHOW, window_size=(1500, 1050)); p.set_background("white")
p.add_mesh(clipped, scalars="mat", cmap=["#9b7d54", "#c7ccd0"], show_edges=True,
           edge_color="#5f6468", line_width=0.25, show_scalar_bar=False)
p.add_axes(); p.camera_position = "iso"; p.camera.azimuth = 35; p.camera.elevation = 22
p.add_text(f"House #1 all-hex mesh ({MODE}) — {len(hexes):,} HEX8, cutaway", font_size=11, color="black")
if SHOW: p.show()
else: p.screenshot(MP + f"/house1_hex_{MODE}.png"); print(f"saved: house1_hex_{MODE}.png")
