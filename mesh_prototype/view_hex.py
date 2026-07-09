"""Translucent viewer for the all-hex House #1 mesh.
Loads house1_hex_<mode>.npz and renders concrete semi-transparent so the room
layout is visible through it (like the earlier translucent-roof view).
Usage: view_hex.py [coarse|prod] [show] [clip]"""
import sys, numpy as np, pyvista as pv
MP = r"S:/ALL PROJECTS/geophone sensor/finals project/geophone_clean/mesh_prototype"
mode = "prod" if "prod" in sys.argv[1:] else "coarse"
SHOW = "show" in sys.argv[1:]; CLIP = "clip" in sys.argv[1:]
d = np.load(MP + f"/house1_hex_{mode}.npz")
pts, hexes, mat = d["points"], d["hexes"], d["mat"]
cells = np.hstack([np.full((len(hexes), 1), 8, np.int64), hexes]).ravel()
grid = pv.UnstructuredGrid(cells, np.full(len(hexes), 12, np.uint8), pts)
grid.cell_data["mat"] = mat
soil = grid.extract_cells(np.where(mat == 1)[0])
conc = grid.extract_cells(np.where(mat == 2)[0])
if CLIP:
    conc = conc.clip(normal=[0, 1, 0], origin=[0, 0.3, 0])
    soil = soil.clip(normal=[0, 1, 0], origin=[0, 0.3, 0])
p = pv.Plotter(off_screen=not SHOW, window_size=(1500, 1050)); p.set_background("white")
p.add_mesh(soil, color="#9b7d54", opacity=0.10, show_edges=False)
p.add_mesh(conc, color="#b7bdc2", opacity=0.42, show_edges=True,
           edge_color="#4a4f54", line_width=0.3)
p.add_axes(); p.camera_position = "iso"; p.camera.azimuth = 38; p.camera.elevation = 30
p.add_text(f"House #1 all-hex ({mode}) — translucent concrete, {len(hexes):,} HEX8",
           font_size=11, color="black")
if SHOW: p.show()
else:
    out = MP + f"/house1_hex_{mode}_translucent.png"; p.screenshot(out); print("saved:", out)
