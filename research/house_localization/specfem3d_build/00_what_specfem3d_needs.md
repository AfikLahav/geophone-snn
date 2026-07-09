# What SPECFEM3D Actually Needs — and Why "Polygons" Are the Whole Point

> Foundational primer for the house-localization sim. Written to resolve a common,
> reasonable confusion: *"I built a 3-D model of the house — isn't that what the simulator
> needs?"* Short answer: **no.** A 3-D model is a *shape*. SPECFEM3D needs that shape
> **cut into hexahedral elements**, because it solves the wave equation *inside* those
> elements. The elements are not decoration — they are the computational substrate.
> Sources: SPECFEM3D docs, Komatitsch & Tromp SEM monograph, and the SEM literature (§Sources).

---

## 1. The core reframe: a "3-D model" is not a "mesh"

There are two very different objects that both look like "a 3-D house":

| | **A 3-D model (what we built so far)** | **A mesh (what SPECFEM3D eats)** |
|---|---|---|
| What it is | The *shape* — walls, slabs, boundaries | The shape **filled with small solid elements** |
| Describes | *Where the surfaces are* | *Every cubic chunk of material, and its properties* |
| Analogy | An empty balloon (just the skin) | A balloon packed solid with tiny bricks |
| Usable by SPECFEM3D? | **No** | **Yes** |

The house model in `mesh_prototype/` is currently the *left* column — geometry + a coarse
visualization mesh. SPECFEM3D needs the *right* column: the interior volume packed with
**hexahedral elements** (little deformed cubes), each carrying material properties.

**The "polygons you see" are the beginning of that packing.** They are not a graphics choice —
they are the physics grid, just currently at low resolution and the wrong element type.

---

## 2. WHY the elements are the input — the spectral-element method

SPECFEM3D uses the **Spectral-Element Method (SEM)**. This is the single fact that explains
everything about its input requirements.

SEM solves the elastic wave equation **not on the whole house at once**, but **element by
element**. Inside *each* hexahedral element it represents the wavefield as a **high-degree
polynomial** (degree 4 by default) sampled at special interior points — the **Gauss-Lobatto-
Legendre (GLL) points**. With the SPECFEM default `NGLL = 5`, that is 5×5×5 = **125 GLL points
per element**.

So the chain is:

```
mesh element  →  125 GLL points inside it  →  a degree-4 polynomial wavefield  →  the physics
```

**No elements ⇒ nowhere to place the GLL points ⇒ nowhere to compute the wavefield ⇒ no
simulation.** The mesh is not a preprocessing convenience; it *is* where the solver lives. This
is why "just give it the 3-D model" cannot work — there is no shortcut that skips the
discretization, because the discretization is the numerical method itself.

(Bonus: placing the nodes at GLL points makes the mass matrix **diagonal**, which is what lets
SPECFEM march forward in time cheaply and explicitly. It is a deep, load-bearing choice, not an
implementation quirk.)

---

## 3. WHY it must be hexahedra (not the tetrahedra we have now)

Our current mesh is **tetrahedra** (4-cornered pyramids). SPECFEM3D **cannot use them** — it
requires an **all-hexahedral** mesh. This is not a limitation to grumble about; it is forced by
§2:

- The 125 GLL points are built as a **tensor product** of 1-D GLL points along the three local
  axes of a **cube** (ξ, η, ζ). That tensor-product structure — and the diagonal mass matrix it
  produces — **only exists for hexahedra.** A tetrahedron has no three-axis tensor structure, so
  the standard SEM polynomial + diagonal-mass machinery does not apply.
- Consequently, converting our tet mesh is **not** an option; the mesh must be **regenerated as
  all-hex.** (This is exactly the E1.0 "all-hex at sim resolution" step.)

**Takeaway:** tets = for viewing / general FEM; **hexes = mandatory for SPECFEM3D.**

---

## 4. WHY the *number* of polygons matters — points per wavelength

This is the heart of "why do polygons matter at all." It comes down to one rule:

> **A wave must be sampled by enough elements to be represented. Too few elements per
> wavelength ⇒ the polynomials cannot draw the wave ⇒ the result is numerically wrong
> (dispersion, wrong speeds, garbage).**

Because each SEM element already carries a degree-4 polynomial (not just a flat value), SEM is
*efficient*: it needs only about **~3–5 elements per minimum wavelength** for high accuracy
(vs ~10–15 for finite-difference). Quantitatively (SEM literature): a degree-4 (NGLL=5) element
reaches ~0.1% error at **~3 elements per shortest wavelength**.

So the element *size* is fixed by physics:

```
shortest wavelength  λ_min = V_min / f_max
element size         h ≈ λ_min / (elements-per-wavelength ≈ 3–5)
```

For our house at 100 Hz with slow structural waves (~150 m/s), λ_min ≈ 1.5 m → element size ~0.3
m in the *soil bulk*. **That is why polygon size is a hard physics number, not an aesthetic
one** — pick elements too big and the wave literally cannot exist on the grid.

**The competing constraint — geometry (walls):** a 0.20 m wall must be resolved by **≥3
elements through its thickness** for the wave to "feel" the wall correctly → element size
**≤ ~0.05 m in walls.** This wall constraint is *finer* than the wavelength constraint, so
**walls set the smallest element size**, and the mesh is **multi-resolution**: ~0.05 m in walls,
~0.30 m in soil.

**Net:** at real simulation resolution one house is **~0.8–1.5 million hexahedra** — roughly
**100× more, much smaller** polygons than the ~32k coarse tets we render for viewing. The
polygon count is dictated by (a) the shortest wave and (b) the thinnest wall — both physics,
neither cosmetic.

---

## 5. The complete list of what SPECFEM3D needs as input

Beyond the mesh, the solver needs the following (all detailed in docs `01`/`03`):

**A. The mesh — an all-hexahedral, conforming mesh**, delivered as SPECFEM3D's external-mesh
**10-file format** (node coordinates, hex connectivity, the 6 boundary-surface files, and the
material files). Our Gmsh mesh must be *translated* into these files — the `gmsh2specfem3d.py`
converter (doc 01), whose #1 risk is the hex node-ordering map.
- *Conforming* = adjacent elements share whole faces (no "hanging nodes"). SEM requires this.

**B. Material properties per region** — for each material (soil, concrete slab, walls):
**Vp, Vs, ρ** (P-speed, S-speed, density) and **Q** (attenuation). Declared in
`nummaterial_velocity_file`. This is where "concrete vs soil vs different floors" enters — it is
just different numbers per element region (the material-library / terrain-profile idea).

**C. A source** — the footstep as a **point force** (`FORCESOLUTION`), vertical, with a
source-time function.

**D. Receivers** — the sensor locations (`STATIONS`), where 3-component seismograms are written.
(In the reciprocity scheme, source and receivers swap — doc 03.)

**E. Run configuration** (`Par_file`) — elastic mode, GPU on/off, simulation length, time step
(set by the CFL stability condition from the *smallest* element), attenuation on/off, output
sampling rate (we choose **1000 Hz** to match the real geophone; see the sampling note).

**F. The free surface** — SEM's big win: the traction-free top/faces are the **natural** ("do
nothing") boundary condition — no Kristek stress-image hack needed (unlike the Devito/FDTD path).
Absorbing boundaries (Stacey sponge) close the domain sides.

---

## 6. Where our House #1 stands against this list

| Requirement | Status |
|---|---|
| Geometry (real ResPlan house, meters, materials as regions) | ✅ done (`build_house1.py`) |
| A viewable mesh | ✅ done (coarse tets) |
| **All-hexahedral** mesh | ❌ currently tets — must regenerate |
| **Sim resolution** (~0.05 m walls / ~0.30 m soil, ~1M hex) | ❌ currently ~0.45 m coarse |
| Conforming (no hanging nodes) at the multi-resolution transition | ❌ to verify (doubling bricks, doc 02) |
| 10-file SPECFEM3D format | ❌ converter not written (doc 01) |
| Material Vp/Vs/ρ/Q table | ⬜ trivial (material library) |
| Source / receivers / Par_file | ⬜ config (doc 03) |
| SPECFEM3D solver environment | ❌ WSL/cloud pending |

So the geometry work is real progress, but **three things stand between the model and a
simulation**, in order:
1. **Regenerate as all-hex at sim resolution** (the E1.0 decider; §3–§4). ← the hard one
2. **Write the 10-file converter** (doc 01).
3. **Stand up the solver** (WSL/cloud) + config (doc 03).

---

## 7. The one-paragraph answer to "why do polygons matter at all?"

Because SPECFEM3D does not simulate "a house" — it simulates the wave equation **inside a grid
of hexahedral elements**, using polynomials sampled at 125 points per element. The polygons
(elements) *are* that grid: they are where the physics is computed, they must be **hexahedra**
(the method's tensor-product math demands it), and their **size is fixed by physics** — small
enough that the shortest wave has ~3–5 elements to live on, and small enough that a 0.20 m wall
holds ≥3 elements. A "3-D model" gives the *shape*; the mesh turns that shape into the
**computational reality** the solver runs on. Get the polygons wrong (too big, or tets instead of
hexes) and there is simply nothing for the wave to propagate through.

---

## Sources
- [SPECFEM3D_Cartesian — Mesh Generation](https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/) · [Creating Databases](https://specfem3d.readthedocs.io/en/latest/04_creating_databases/)
- [Komatitsch & Tromp — The Spectral-Element Method in Seismology (monograph PDF)](https://igpppublic.ucsd.edu/~shearer/227C/spec_elem_monograph.pdf)
- [Spectral element method — Wikipedia](https://en.wikipedia.org/wiki/Spectral_element_method) · [SEM overview — ScienceDirect](https://www.sciencedirect.com/topics/mathematics/spectral-element-method)
- NGLL=5 / 125 points-per-element, degree-4 default, ~3 elements/wavelength for 0.1% error — from the SEM resolution literature and SPECFEM docs above.
- Internal: `specfem3d_build/01_mesh_format_and_converter.md`, `02_boxy_house_meshing.md`, `03_run_reciprocity_workflow.md`; `house_localization/10_devito_numerics_gpu_viz.md` (wall-resolution + element-count analysis).
