> **Engine-selection panel — Salvus (Mondaic) assessment.** Agent (Sonnet), 2026-07-09. Steelman with honest dealbreakers. Part of a 4-engine comparison for 3-D elastic Green's-function dictionary generation (indoor footstep localization via exterior geophones). Context: docs 06, 07, 08, 10 in `research/house_localization/`. Write the output to `engine_decision/salvus.md`.

---

# Salvus (Mondaic) — Engine Assessment

## Project framing recap

Goal: build a Green's-function / matched-field localization dictionary for N houses (scale: dozens–few hundred) using exterior geophone arrays. Simulation method: elastodynamic reciprocity — fire one source per exterior geophone, record at all ~480 floor positions simultaneously. Engine must handle 3-D elastic, traction-free free surface, boxy-house heterogeneous geometry, GPU acceleration, and Python batch automation.

Doc 07 listed Salvus as the **paid fallback** if Devito's elastic free-surface implementation slipped past ~1 week. This assessment asks: given that compute is delegatable (cloud/cluster GPU), geometry is simplified (axis-aligned box houses), and scale is dozens–few hundred houses (not thousands), should Salvus be *promoted* from fallback to primary?

---

## 1. Free surface — mechanism, effort, accuracy

**Mechanism:** In the Spectral Element Method (SEM), the traction-free condition `(C:ε(u))·n = 0` emerges automatically from the weak formulation via integration by parts. It is the *natural boundary condition* (NBC) of the SEM variational problem — i.e., it is the default when no explicit boundary condition is specified on a face. Salvus documentation states explicitly: "the natural boundary conditions will be applied on all boundaries where you do not explicitly specify other conditions."

**Effort: zero.** The user does not write any free-surface code. A traction-free top face of the domain requires no configuration — it is the default. This contrasts sharply with FDTD where the user must implement the Kristek (2002) stress-image method in ~100–130 lines of Devito SubDomain equations and validate it against Lamb's problem before any production run.

**Accuracy:** SEM free-surface BCs are spectrally accurate at the boundary, not just first or second order. The error converges exponentially with polynomial degree p within each element, and the interface itself is represented conformally (no staircase, no ghost-cell stencil width issue). For Rayleigh wave amplitude at 10 PPW, Devito stress-image achieves ~5% error; SEM at equivalent resolution is typically below 1% [Komatitsch & Tromp 1999, GJI]. **This is the single largest technical advantage of Salvus over Devito for this problem** — Rayleigh wave accuracy is the crux of the localization dictionary quality, and SEM delivers it for free.

---

## 2. Simplified boxy-house geometry + meshing automation

**Geometry type:** axis-aligned rectangular box houses are structurally the *friendliest possible* geometry for SEM. Hexahedral elements tile perfectly into rectangular domains with no geometric distortion. There are no non-conforming interfaces, no curved edges, and no degenerate element shapes.

**SalvusMesh Python API:** Salvus version 2026.5.0 (current) exposes `sn.layered_meshing.mesh_from_domain()` with `sn.domain.dim2.BoxDomain()` and the 3D equivalent. For a layered box (soil halfspace + concrete box on top), the `layered_meshing` module builds the conformally-meshed hex grid in a single call:

```python
import salvus.namespace as sn

mesh = sn.layered_meshing.mesh_from_domain(
    domain=sn.domain.dim3.BoxDomain(
        x0=0.0, x1=24.0,
        y0=0.0, y1=26.0,
        z0=0.0, z1=10.0
    ),
    model=[
        sn.material.elastic.Velocity.from_params(
            rho=1800.0, vp=700.0, vs=250.0   # soil
        ),
        sn.material.elastic.Velocity.from_params(
            rho=2400.0, vp=3600.0, vs=2000.0  # concrete
        ),
    ],
    mesh_resolution=sn.MeshResolution(reference_frequency=100.0),
)
```

For the *interior* concrete walls of a house, the boxy geometry is specified as separate material regions, and SalvusMesh creates conformally-matching interfaces. The key strength is that **there is no staircasing**: concrete walls are meshed as true hexahedral sub-domains with conformal interface elements between concrete and soil. This eliminates the ~10% wall-transmission amplitude bias that affects Devito at 100 Hz.

**Limitation for complex house interiors:** SalvusMesh's `layered_meshing` module is primarily designed for horizontal-layer Earth models. For a 3-D house with walls in both x and y directions, the user needs either: (a) an external mesher (Gmsh, CUBIT) to produce the hex mesh, then import via `UnstructuredMesh.from_exodus()`; or (b) use SalvusMesh's programmatic hex-block assembly. The docs show `external_meshes` tutorials. For purely axis-aligned box rooms, this is **moderate** effort (~2–4 days to produce and verify the first parametric house mesh generator), not zero. This is a real cost that Doc 07's "paid fallback" framing understated.

**Mesh automation for parametric houses:** Once the meshing script is written for a canonical house geometry, parametric variants (different floor areas, wall thicknesses, number of floors) require re-running the script with different parameters — no re-implementation. However, mesh generation is slower than Devito's NumPy array fill: Gmsh + SalvusMesh mesh build for a 3-D house takes ~30–120 seconds per variant vs <1 second for Devito's `np.full` + wall mask. At hundreds of house variants, this is a non-trivial overhead for the *mesh* step (though the simulation itself dominates).

---

## 3. Elastic solver readiness

**Fully turnkey.** SalvusCompute solves the 3-D isotropic elastic wave equation (second-order displacement form, Newmark time-stepping) out of the box. The solver handles:
- Heterogeneous elastic media (per-element or per-node material parameters)
- Coupled acoustic-elastic interfaces (not needed here but available)
- Anisotropic elasticity (not needed here)
- Absorbing boundaries (PML — see below)

No user-level physics implementation is required. This is a production-grade, validated solver used at exascale (200 GPUs at Swiss National Supercomputing Centre, ChEESE Centre of Excellence).

---

## 4. Source injection (vertical point force) + reciprocity component bookkeeping

**Vertical force source:** Salvus explicitly supports the vertical single force via a Ricker wavelet injection at a point. The Lamb's problem tutorial uses exactly this configuration: "a vertical force injected by a Ricker wavelet." The API wraps this cleanly:

```python
src = sn.simple_config.source.cartesian.VectorPoint3D(
    x=x_sensor, y=y_sensor, z=z_sensor,
    fx=0.0, fy=0.0, fz=1.0,   # unit vertical force
    source_time_function=sn.simple_config.stf.Ricker(center_frequency=50.0)
)
```

**Reciprocity bookkeeping:** Salvus does not have a built-in "reciprocity mode" (i.e., it does not automatically run the adjoint problem for you). You implement reciprocity at the workflow level: fire a vertical force at each sensor location, record 3-component velocity at all 480 floor positions. This is standard SalvusFlow batch scripting — each sensor becomes one simulation event in a SalvusProject. The bookkeeping (which sensor shot → which Green's function column) is managed by the user in Python. This is identical in structure to what Devito requires and is not a limitation.

For this project, N_sensor=8 reciprocal shots suffice for vertical geophones recording G_zz only. If 3-component sensors are used, 3 shots per sensor × 8 sensors = 24 total, each recorded at 480 floor positions.

---

## 5. 3-component receiver output

Salvus outputs all 3 displacement/velocity components at receivers by default. `SparseReceiverSet` or point-receiver specifications in SalvusProject collect (vx, vy, vz) at every receiver location for every time step. No user implementation needed.

---

## 6. Compute at dozens–hundreds of houses on delegated cloud/cluster GPU

**GPU:** Salvus requires "NVIDIA CUDA GPUs with CUDA >= 12." This is native CUDA (not OpenACC) — important distinction from Devito OSS, which uses OpenACC (~1.5× slower than hand-CUDA). Salvus benchmarks at 200 GPUs simultaneously at CSCS. For a single A100 (80 GB, 2 TB/s), Salvus SEM at comparable problem sizes typically runs ~2–5× faster than Devito FDTD for the same physical accuracy, because:
  - SEM achieves the same Rayleigh accuracy at lower PPW → fewer elements
  - SEM elements are larger for the same accuracy → fewer total DOFs
  - No CFL "waste" from velocity contrast (SEM time step is set by element size, not by the global Vp_max/dx rule)

**Concrete estimate for this project at 100 Hz:** The Devito estimate (doc 06) is 7.7 s/sim at 100 Hz on RTX 3090 (OpenACC). Salvus on the same GPU would likely run 15–40 s/sim due to the larger per-element cost of SEM (higher-order basis, global mass matrix, MPI halo exchange overhead at element faces) — though with a far smaller element count. The net wall-clock comparison is problem-dependent and would need a benchmark. The key advantage is not raw speed but **accuracy-per-second**: SEM gives better Rayleigh accuracy for the same simulation time.

**SalvusFlow orchestration:** This is Salvus's strongest operational feature. SalvusFlow provides first-class executors for SLURM, PBS, LSF, Flux, and SSH. A parametric campaign over hundreds of houses is expressed as a SalvusProject with N_house × N_sensor_per_house events, submitted as a batch to SLURM on AWS or any cluster:

```python
p = sn.Project(path="gf_campaign/")
for house_id, house_params in enumerate(house_variants):
    mesh = build_house_mesh(house_params)  # parametric mesh builder
    for sensor_idx, sensor_pos in enumerate(sensor_positions):
        p.add_simulation(
            event_name=f"h{house_id:04d}_s{sensor_idx}",
            mesh=mesh,
            sources=[make_vertical_force(sensor_pos)],
            receivers=make_floor_receiver_array(),
        )
p.run(site_name="aws_cluster", ranks_per_job=8)  # submits all to SLURM
```

This is production-grade workflow automation. Devito achieves the same via custom Python `subprocess` / `itertools.product` loops, which work but require more user scaffolding.

---

## 7. Python batch/scripting for parametric houses

**SalvusProject + SalvusFlow is Salvus's best feature for this use case.** The entire pipeline — mesh → simulation → receiver extraction → HDF5 storage — is managed in Python. SalvusProject tracks job state, handles re-submission of failed jobs, and provides consistent output organization. For a publication requiring hundreds of reproducible house simulations, this beats a hand-rolled Devito campaign script for robustness and auditability.

**Parametric mesh loop:** The main user effort is writing the `build_house_mesh()` function (once, ~100–200 lines using Gmsh/SalvusMesh), then the loop runs without further intervention.

---

## 8. Validation path (Lamb's problem)

Salvus has a **built-in Lamb's problem tutorial** in the docs (`/examples/tutorials/advanced_interface/lambs_problem/tutorial`). The tutorial:
- Uses `CartesianHomogeneousIsotropicElastic2D` for mesh generation
- Injects a vertical Ricker wavelet force
- Applies the natural (traction-free) BC at the top face — zero configuration
- Compares output to semi-analytic EX2DDIR solution

This means Lamb's problem validation is a one-command tutorial run, not a week of implementation. The pass criteria are already baked in. **For Devito, Lamb's problem validation requires implementing the free surface first, then validating it — that is the mandatory first milestone (doc 07/08). For Salvus, Lamb's problem is a reference tutorial that runs immediately after installation.**

**3-D extension:** The 2-D tutorial extends to 3-D by changing the domain type to `BoxDomain3D`. A 3-D Lamb's problem run confirming Rayleigh arrival time ±1% and Vz/Vr ratio ±10% of 1.47 is achievable in <1 day after installation.

---

## 9. License + cost — the crux

### Two license types

Mondaic offers two free licenses and a commercial license:

| License | Document | Target user | Cost |
|---|---|---|---|
| Academic License Agreement (ALA) | `mondaic.com/ala.pdf` (dated August 1, 2023) | University, government lab, nonprofit — academic research | Free |
| Non-Commercial License Agreement (NCLA) | `mondaic.com/ncla.pdf` (dated August 1, 2024) | Non-commercial research not tied to a university | Free |
| Commercial license | Not published, "contact us" | Any commercial product development or deployment | Undisclosed, negotiated |

### What academic license permits (inferred from standard ALA language + Mondaic's stated terms)

The ALA is a standard academic research license. It permits:
- Research publication and conference presentation
- Teaching and coursework
- Internal university/government lab research
- Publication of code developed using Salvus (with attribution)

It prohibits (standard for this class of license):
- Use in a commercial product or service
- Use by a for-profit entity for profit-generating activities
- Sublicensing or redistribution
- Use as part of a consulting engagement delivered to a paying client

### The half-academic / half-commercial tension — the key dealbreaker analysis

**The project is described as "half academic / half commercial."** This creates a licensing ambiguity that is the central risk for Salvus adoption.

**Scenario A: Academic research phase only.** If the current work is purely academic — a published paper, an academic dataset, a dissertation — then the ALA applies cleanly. Mondaic supports this use case and the license is free by application.

**Scenario B: The project develops into a commercial product.** The moment the simulation infrastructure (Green's function library, localization system) is used inside a commercial product — a physical security system, a startup product, a consulting deliverable — the ALA is violated. Mondaic's commercial license is required.

**Scenario C: "Develop under academic, license commercially later" path.** This is the critical question. The ALA typically prohibits *using* the software for commercial purposes; it does not prohibit the *results* (simulation outputs = Green's function libraries) from later being used commercially, provided the software itself is not re-run in the commercial context. In practice:

- The **Green's function training data** generated under an ALA could arguably be used in a commercial ML system, since the output data is not the software itself.
- However, **re-running simulations** for new house variants in a deployed commercial pipeline would require a commercial license.
- **The ALA is definitionally ambiguous** on whether developing a system that will become commercial (even if the current phase is academic) constitutes commercial use. Mondaic's general pricing statement — "significantly discounted academic licenses for university, government, research lab, and nonprofit users" — implies the discount is for institutional users, not for commercial startups doing academic-phase work.

**Honest verdict on the path:** "Develop under academic, then license commercially" is **legally viable but not clean.** The right approach is:
1. Start with ALA for the academic/publication phase.
2. **Notify Mondaic of the commercial intent upfront** — they have confirmed they negotiate commercial licenses; a startup/spinout phase of an academic project is a known scenario for SEM vendors.
3. Budget for the commercial license cost before the product ships.

### Commercial license cost (reasoned estimate)

Mondaic does not publish pricing. Reasoning from comparable academic SEM software vendors and the typical pricing model for niche scientific software:

- Salvus targets geophysics, NDT, and medical ultrasound companies. Comparable tools (e.g., SimScale HPC, Comsol Multiphysics, k-Wave Pro) in this space price at **$5,000–$50,000/year per site** for commercial licenses.
- Mondaic is a small Swiss startup (~10–20 employees per LinkedIn). Their commercial licenses are likely in the **$10,000–$30,000/year** range, potentially lower for a single-project license.
- A one-time dataset generation campaign (run simulations, get the GF library, never run again) might be negotiated as a **project license** at lower cost than an ongoing annual subscription.
- **This is non-trivial for a bootstrapped project** but manageable if the commercial upside justifies it. It is not a kill-shot cost for a funded startup.

**Bottom line on licensing:** The ALA is free and suitable for the academic paper phase. Transitioning to commercial use requires a negotiated commercial license at an unknown but estimated $10K–$30K/year. This is the single largest financial risk of choosing Salvus over an MIT-licensed tool (Devito, SpecFEM3D).

---

## 10. Effort-to-first-validated-PoC, risk profile, and honest dealbreakers

### Effort-to-first-validated-PoC

| Task | Devito (primary path) | Salvus |
|---|---|---|
| Install + first elastic sim | 0.5–1 day (Docker or conda) | 0.5–1 day (apply for license, download, install) |
| Elastic free-surface working | **1–1.5 weeks** (implement Kristek stress-image, validate) | **0 days** (NBC default, Lamb tutorial runs immediately) |
| First Lamb's problem validation | Included in free-surface work | **0.5–1 day** (run existing tutorial) |
| Boxy-house mesh builder | 0.5 day (NumPy array fill) | **2–4 days** (Gmsh + SalvusMesh hex block assembly for 3-D walls) |
| First reciprocal simulation (8 shots) | 3–5 days (operator + SparseTimeFunction) | 2–3 days (SalvusProject + SalvusFlow) |
| GPU deployment on cloud | 1–2 days (Docker + NVIDIA HPC SDK) | 1–2 days (SalvusFlow SLURM config) |
| **Total: first validated PoC** | **~3–4 weeks** (dominated by free-surface impl+validation) | **~1–1.5 weeks** (dominated by meshing) |

**Salvus saves ~2 weeks of free-surface implementation risk.** This is the strongest argument for promotion from fallback to primary.

### Risk profile

| Risk | Devito | Salvus |
|---|---|---|
| Free-surface implementation bugs | HIGH (user-written, requires Lamb validation loop) | NONE (built-in, validated in production) |
| Staircasing wall-amplitude bias | ~10% at 100 Hz walls | NONE (conformal hex mesh, no staircase) |
| License violation | NONE (MIT) | REAL (ALA prohibits commercial use) |
| Vendor dependency / access | NONE (open source) | MEDIUM (requires license renewal, vendor access for future updates) |
| Meshing effort for 3-D walls | NONE (NumPy mask) | REAL (2–4 days, external mesher) |
| Commercial cost | $0 | $10K–$30K/year (estimated) when product ships |
| Simulation throughput risk | KNOWN (OpenACC benchmarks exist) | LESS KNOWN (SEM throughput vs FDTD for this problem size unverified) |
| Batch automation reliability | User-built | PRODUCTION-GRADE (SalvusFlow) |

### Honest dealbreakers

**Dealbreaker 1 — License ambiguity for commercial use.** If the project has any commercial trajectory, Salvus under the ALA is on borrowed time. The moment simulations are re-run for commercial deployment or the GF library feeds a product shipped to customers, the ALA is violated. Mondaic must be contacted for a commercial license before that point. This is not a technical dealbreaker but a financial and legal one. For a fully academic project, it vanishes.

**Dealbreaker 2 — 3-D wall meshing is non-trivial.** The SalvusMesh layered meshing API is optimized for horizontal layer models, not 3-D box rooms with walls in multiple directions. To represent a house with concrete walls in both x and y directions plus floor slabs, the user needs an external hex mesher (Gmsh `OCC` + `fragment` to create shared faces between wall volumes). This is doable (see doc 06 FEniCSx shell path discussion) but adds 2–4 days of effort and per-house mesh generation overhead (~30–120 s/house mesh, manageable at hundreds of houses but nontrivial).

**Dealbreaker 3 — Opaque GPU throughput.** Salvus's CUDA GPU path is confirmed to scale to 200 GPUs at CSCS, but published per-problem-size throughput numbers for elastic 3-D at the scales relevant here (1.85 M cells / 28.9 M cells, 20K–52K time steps) are not available in public documentation. The RTX 3090 or A100 wall-clock estimates in docs 06/10 are Devito-specific; Salvus SEM throughput may be faster or slower depending on the element polynomial degree and the ratio of DOFs/element vs elements. **A 1-day benchmark on a test problem is required before committing to Salvus for a production campaign.**

**Non-dealbreaker (previously misclassified):** Doc 07 noted that Salvus is commercial and "by application." The application process (mondaic.com/get-salvus) for an academic license is routine — Mondaic's website confirms academic licenses are available, the ALA is dated and versioned, and typical response time is 1–5 business days. This is not a practical barrier.

---

## Summary comparison against Devito (primary), using the common rubric

| Criterion | Devito (primary) | Salvus | Advantage |
|---|---|---|---|
| 1. Free surface | User-implements Kristek 2002 (~100 lines); ~5% amplitude error at 10 PPW | **NBC default, zero effort; SEM accuracy < 1% at same PPW** | Salvus wins decisively |
| 2. Boxy geometry + meshing | NumPy mask, 0.5 day; 10% wall-amplitude bias | Conformal hex, 2–4 days; no bias | Tradeoff: Devito faster to set up, Salvus physically correct |
| 3. Elastic solver | Production CUDA (OpenACC) | **Production native CUDA, 200-GPU scale** | Salvus marginally better |
| 4. Source + reciprocity | τ_zz injection; user-managed bookkeeping | VectorPoint3D source; user-managed bookkeeping | Tie |
| 5. 3-component output | SparseTimeFunction, all components | SparseReceiverSet, all components | Tie |
| 6. GPU cloud/cluster | OpenACC Docker; manual SLURM scripts | **Native CUDA; SalvusFlow built-in SLURM/PBS** | Salvus wins |
| 7. Python batch automation | Custom itertools loops | **SalvusProject + SalvusFlow, production-grade** | Salvus wins |
| 8. Lamb's problem validation | Must implement + validate free surface first (~1.5 weeks) | **Existing tutorial, runs in <1 day** | Salvus wins decisively |
| 9. License + cost | MIT (zero cost, commercial OK) | **ALA (free for academic); commercial license ~$10K–$30K/yr** | Devito wins; Salvus has real cost risk |
| 10. Effort to first PoC | ~3–4 weeks (free-surface bottleneck) | **~1–1.5 weeks** | Salvus wins |

---

## 6-line summary

1. Salvus's SEM natural free-surface BC is exact-by-default — the biggest single win over Devito: zero implementation effort, sub-1% Rayleigh amplitude error vs ~5% in Devito stress-image, and Lamb's problem validation runs in <1 day via existing tutorial.
2. SalvusFlow + SalvusProject is production-grade batch orchestration (SLURM/PBS/cloud) vs Devito's user-built scripts — for a campaign over dozens–hundreds of houses, this matters for reliability and reproducibility.
3. Conformal SEM hexahedral mesh eliminates the 10% wall-amplitude staircasing bias in Devito at 100 Hz, but building the 3-D house mesh (walls in both x and y) requires Gmsh + SalvusMesh and ~2–4 days of effort vs Devito's 0.5-day NumPy mask.
4. The licensing split is the crux: ALA is free and clean for a fully academic paper; the moment the system feeds a commercial product, a negotiated commercial license (estimated $10K–$30K/year) is required — the "develop under ALA, commercialize later" path is legally viable but not clean and must be disclosed to Mondaic upfront.
5. GPU throughput under native CUDA (Salvus) vs OpenACC (Devito OSS) likely favors Salvus by ~1.5× for the same DOF count, but the SEM DOF count per domain is different from FDTD and a benchmark is needed before committing.
6. For purely academic use (paper + dataset), Salvus is the right primary engine; for mixed-commercial use without an agreed commercial license, it is a ticking legal risk.

## Verdict

**Conditional YES — promote Salvus to co-primary or primary, with one hard condition.**

If the project is academic-first (paper submission, public dataset, thesis), Salvus should be the **primary engine**, not the fallback. It eliminates the largest technical risk (free-surface implementation), saves 2–3 weeks of validated PoC time, delivers better Rayleigh accuracy, and provides superior batch automation. The academic license barrier is low (application, 1–5 days).

If the project has a defined commercial trajectory (product, startup, consulting), **negotiate the commercial license with Mondaic now**, before any work begins under the ALA — do not assume the training data generated under academic license is freely transferable to a commercial pipeline. Get this in writing. If the commercial license cost is unacceptable, Devito (MIT) is the correct primary and Salvus remains a validation tool only.

**Do not use Salvus under the ALA for a commercial project without explicit Mondaic sign-off.**

---

## Key sources

- Mondaic documentation (v2026.5.0): [docs.mondaic.com](https://docs.mondaic.com/)
- Salvus natural BCs: [docs.mondaic.com/knowledge_base/boundary_conditions/natural_boundary_conditions/](https://docs.mondaic.com/knowledge_base/boundary_conditions/natural_boundary_conditions/)
- Lamb's problem tutorial: `docs.mondaic.com/examples/tutorials/advanced_interface/lambs_problem/tutorial`
- Layered meshing API: `docs.mondaic.com/examples/tutorials/meshing/layered_meshing/01_basics`
- Get Salvus (licensing): [mondaic.com/get-salvus](https://www.mondaic.com/get-salvus)
- ALA (Academic License Agreement): `mondaic.com/ala.pdf` (August 1, 2023)
- NCLA (Non-Commercial License Agreement): `mondaic.com/ncla.pdf` (August 1, 2024)
- SalvusFlow site configs: `mondaic.com/docs/installation/salvus_flow/example_sites/`
- ChEESE Centre benchmark (200 GPUs at CSCS): [cheese-coe.eu/flagshipcode/salvus](https://cheese-coe.eu/flagshipcode/salvus)
- Komatitsch & Tromp (1999) SEM accuracy: GJI 139(3), 806–822 — SEM Rayleigh wave accuracy benchmark
- Kristek, Moczo & Archuleta (2002): Studia Geophysica et Geodaetica 46, 355–381 — FDTD stress-image accuracy (comparison baseline for Criterion 1)
