# engine_decision/ — the 3-D sim engine bake-off

A 4-way steelman panel choosing the elastic-wave engine for the house Green's-function sim.
Each doc argues the strongest *honest* case for one engine against a common rubric (free
surface, meshing, solver readiness, reciprocity, GPU, license, effort-to-PoC, dealbreakers).

| Doc | Engine | Verdict (short) |
|---|---|---|
| `specfem3d.md` | SPECFEM3D (open SEM) | **Leading** — natural free surface, <0.5% Rayleigh, GPL+Gmsh clean; costs ~1 wk meshing automation |
| `devito.md` | Devito (open FDTD) | Fastest PoC + MIT license, but must build+validate the elastic free surface (~1.5 wk) and homogenize thin walls |
| `salvus.md` | Salvus (commercial SEM) | Turnkey + best for a pure-academic PoC, but commercial license ($10–30k/yr est.) taxes the commercial half |
| `fem.md` | FEniCSx/Firedrake/Nektar++ | Validation tool only — no ready elastic solver, immature GPU |

**Context that reshaped the decision:** scale is dozens–hundreds of houses (not thousands, so
SEM meshing is tractable), geometry is simplified/boxy, compute is delegatable, and **wall
thickness — not wavelength — sets resolution** (0.15 m wall needs dx ≤ 0.05 m), which favors
SEM's adaptive mesh over Devito's uniform grid.

**Decision gate:** the choice hinges on one experiment — **can a parametric boxy house be
auto-meshed in Gmsh cleanly?** (roadmap E1.0). If yes → SPECFEM3D; if it's a swamp → Devito.
Deeper adoption analysis: [`../12_specfem3d_adoption.md`](../12_specfem3d_adoption.md).
Full plan: [`../../ENGINEERING_ROADMAP.md`](../../ENGINEERING_ROADMAP.md).
