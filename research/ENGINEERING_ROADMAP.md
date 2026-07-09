# Engineering Roadmap — Interior-Movement Localization Simulation

> **Status: discovery → execution.** The research phase (docs in `house_localization/`
> 01–14, `localization_1d/` 01–03, `house_localization/engine_decision/`) resolved the open
> *unknowns*. This document turns them into concrete **engineering tasks**: what to build,
> the approach, inputs/outputs, reuse, effort, dependencies, and the validation gate for each.
> Load-bearing technical claims are verified against primary sources (§2), 2026-07-09.

## 1. Goal & scope

Pinpoint a person's movement **inside** a building using **N geophones on the exterior**
(walls / foundation / flat roof), via structure-borne footstep vibration + matched-field /
TDOA localization on a **simulated Green's-function (GF) dictionary**.

**Two tracks, run concurrently:**
- **Track 2 (pyprop8 testbed)** — cheap, exact, CPU. Develops + validates the localization
  *methods* on flat layered soil. Independent of the engine decision. **Ready to build now.**
- **Track 1 (house sim)** — 3-D elastic sim of real house geometry to produce the GF
  dictionary. Gated on one decision (§4). Supplies the *house physics* the methods run on.

Track 2 de-risks *method viability* before Track 1's expensive physics is built. The same
localizer code re-points from the pyprop8 dictionary onto the house dictionary later.

## 2. Verified technical foundation (primary sources)

| Claim | Verdict | Source |
|---|---|---|
| pyprop8 = layered half-space, exact free surface, 3-comp, point force, CC-BY-4.0 | ✅ | [JOSS 10.21105/joss.04217](https://joss.theoj.org/papers/10.21105/joss.04217) · in-repo `simgeo_v42/gfbank_build.py` |
| Stress-image / traction-image free surface is the **standard** elastic-FDTD method (antisymmetric ghost cells) | ✅ | [Traction image method, GJI](https://academic.oup.com/gji/article/167/1/337/606655) · [SEG 1.1512752](https://library.seg.org/doi/10.1190/1.1512752) |
| OSS Devito free surface is **acoustic-only**; elastic must be built (`SubDomain`) | ✅ | [Devito BC tutorial](https://www.devitoproject.org/examples/userapi/04_boundary_conditions.html) · [arXiv:2004.10519](https://arxiv.org/abs/2004.10519) |
| SEM (SPECFEM3D) free surface is **natural** (unloaded element face); ~5 PPW vs FDTD ~10–15 | ✅ | Komatitsch & Tromp 1999; [SPECFEM3D docs](https://specfem3d.readthedocs.io/) |
| **SPECFEM3D has NO direct Gmsh importer** — must write the 10-file converter (CUBIT has one, but CUBIT is commercial) | ✅ | [SPECFEM3D mesh docs](https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/) |
| SPECFEM3D `FORCESOLUTION` = tilted point force, Dirac STF (convolve any STF), 3-comp disp/vel/acc output, reciprocity standard | ✅ | [SPECFEM3D Par_file](https://github.com/SPECFEM/specfem3d/blob/master/EXAMPLES/homogeneous_halfspace_HEX27_elastic_no_absorbing/DATA/Par_file) |
| Ready-made 3-D building datasets are **exterior shells only** — no interior walls/thickness/materials → geometry must be **generated** | ✅ | doc 13 (CityGML/3DBAG/OSM-3D survey) |
| **Wall thickness (not wavelength) sets resolution**: 0.15 m wall needs ≥3 cells → dx ≤ 0.05 m in walls | ✅ | doc 12 §element-sizing |
| Geophone TDOA-quality range ≈ **5–8 m** to nearest sensor; exterior ring sweet spot **5–15 m short-dim, 25–200 m², ≤2 storeys** | ✅ | doc 14 |

**Materials:** treated as a **lookup table** (the terrain-profile pattern already used in
`profiles.py` for soil) — literature (Vp, Vs, ρ, Q) per building material, domain-randomized.
**Not a research task; data entry inside the pipeline.**

---

## 3. Track 2 — pyprop8 localization testbed (ready now)

Reuses the existing pyprop8 pipeline (`simgeo_v42/gfbank_build.py`, `scenes.py emit()`,
`wavelet.py`). See `localization_1d/01_methods_and_protocol.md`, `02_testbed_architecture.md`,
`03_pyprop8_validity.md`.

| ID | Task | Approach / reuse | Effort | Validation gate |
|---|---|---|---|---|
| **E2.1** | 3-component GF banks | Extend `gfbank_build.py` to persist H (R) + Z (already computed; ~2 lines). T=0 by P-SV/SH decoupling. | 0.5 d | Z matches current banks bit-for-bit; R non-zero, physically sane |
| **E2.2** | 2-D N-sensor scene harness | Place N sensors + walking path in 2-D; synthesize each (sensor, source) waveform via `emit()` distance-lookup + azimuth + wavelet + t*. Ground-truth positions. | 2–3 d | Round-trip: known source → recovered by an oracle localizer at ~0 error |
| **E2.3** | Localizer modules | MFP (Bartlett/MVDR on the distance-dictionary), SO-TDOA / GCC-PHAT+dispersion, RSS, 3-C polarization bearing, likelihood-surface fusion, Kalman track. Ranked bake-off order per `localization_1d/01`. | 5–8 d | Each reproduces its literature accuracy on a clean case |
| **E2.4** | Blind-velocity + Monte-Carlo harness | 2×2 blind matrix (calibration-blind × dispersion-blind); sweep N∈{3,4,6,8}, GDOP geometries, SNR, velocity-knowledge; measure position error + gap-to-CRLB. | 3–4 d | Reproduces the CRLB floor; honest-worst-case degradation quantified |

**Track 2 deliverable:** a ranked, CRLB-anchored answer to *"which methods localize, how well,
and how much does velocity-blindness cost"* — the core method-viability result, at ~10⁶× lower
cost than the house sim. **~11–16 engineer-days.**

---

## 4. E1.0 — Engine decision (the decider experiment)

The house-sim engine (SPECFEM3D vs Devito vs Salvus) is **not yet decided**; the steelman panel
(`engine_decision/`) tilted toward **SEM** after the wall-resolution finding (§2) — SEM's
adaptive mesh resolves 0.05 m walls locally while keeping soil coarse, whereas Devito's uniform
grid must homogenize (accuracy loss) or refine globally (~27× cost). But the decision hinges on
**one experiment**:

> **E1.0 — prototype: parametric boxy house → Gmsh hex mesh → SPECFEM3D 10-file format.**
> Build one simplified axis-aligned house, script the Gmsh mesh (walls ≥3 elements thick, soil
> coarse), write/adapt the 10-file converter, run one elastic sim, confirm it produces sane
> 3-component seismograms. **~3–5 days.**

- **Meshing clean + automatable** → SPECFEM3D (natural free surface, <0.5% Rayleigh, GPL+Gmsh
  clean for the commercial half). **Recommended if it passes.**
- **Meshing a swamp** → Devito (meshless voxel batch) + the ~1.5-week stress-image free-surface
  build (`house_localization/08`).
- **Salvus** = fast academic-license PoC fallback, but commercial license ($10–30k/yr est.) is a
  recurring tax → poor production base for the commercial half.

**E1.0 gates all of E1.1–E1.5.** Run it before deep-planning the house pipeline.

---

## 5. Track 1 — house simulation (post-E1.0)

Effort/approach below assume **SPECFEM3D** (the leading candidate); Devito variant noted where
it differs. See `house_localization/09` (generation), `12` (SPECFEM3D adoption), `13` (sourcing),
`14` (sizes).

| ID | Task | Approach | Effort | Validation gate |
|---|---|---|---|---|
| **E1.1** | Procedural house generator | Parametric floorplan (OSM footprint boundary + RPLAN/ResPlan priors + Israeli defaults) → extrude to 3-D → material tags. Materials LUT (§2). Axis-aligned/boxy, windows/openings as geometry, appliances as sources (not geometry). | 5–8 d | **Visual QA gate** (PyVista/ParaView) + automated geometry check (watertight, walls ≥3 cells, storeys stacked); **manual review per house** |
| **E1.2** | Geometry → mesh → sim pipeline | Gmsh hex mesh (walls 0.05 m, soil 0.30 m) → the **10-file converter** (E1.0 output) → material assignment → SPECFEM3D. *(Devito: voxel material array, no mesh.)* | 5–8 d (folds in E1.0) | Bridge validation: soil-only case vs pyprop8 (P & Rayleigh <0.5%, amp <5%) |
| **E1.3** | Free-surface / Lamb validation | SEM: *validate* the natural BC on Lamb's problem (analytic half-space). Devito: *build* the stress-image + validate. | SEM 0.5–1 d / Devito 1–1.5 wk | Lamb: arrival times <1%, Vz/Vr ratio within 10%; SW4 cross-check |
| **E1.4** | Reciprocity GF campaign + storage | Shoot 1 reciprocal force per sensor (`FORCESOLUTION`), record 3-comp at all interior grid positions; `NUMBER_OF_SIMULTANEOUS_RUNS` batches the shots. HDF5 GF library → adapter to the existing `emit()`/`scenes.py` interface. | 3–4 d | Reciprocity self-check G(a,b)=G(b,a); GF library loads into the sensor-chain architecture |
| **E1.5** | Sensor model + confusers | Reuse `sensor.py` (geophone chain); re-tune for **concrete coupling**, add **bounded tilt** (≤~15–20°, vertical-referenced) + **MEMS/ADXL355** sub-model; confusers (AC/appliances/mains) as source-STF ⊛ GF (re-point `r3_noise`). | 3–5 d | Engine-agnostic: same chain consumes pyprop8 *and* house GF; confuser hard-negatives present |

**Track 1 deliverable:** a validated house GF dictionary that trains the classifier, *is* the
matched-field dictionary, and serves as the fingerprint DB. **~17–26 engineer-days** post-E1.0
(SEM path), plus compute (50–200 houses @100 Hz ≈ hours on one A100, <$30 — doc 12).

---

## 6. Critical path & sequencing

```
NOW ─┬─ Track 2 (E2.1→E2.4)  ── independent, cheap, tests method viability ── START FIRST
     │
     └─ E1.0 Gmsh decider ── settles engine ──┐
                                              ▼
                              Track 1 (E1.1 ∥ E1.2→E1.3→E1.4→E1.5)
                                              ▼
                              Bridge validation (pyprop8 soil ↔ house sim)
                                              ▼
                              Re-point Track-2 localizers onto house GF dictionary
```

- **E2 and E1.0 run in parallel** (no shared dependency).
- **E1.1 (generator) is engine-independent** — can start alongside E1.0.
- **Everything else in Track 1 waits on E1.0** (the engine decision).

## 7. Validation gates (nothing proceeds past a red gate)

1. **E2 round-trip** — oracle localizer recovers known sources (harness is correct).
2. **E1.0 mesh** — a boxy house meshes + runs (engine decision).
3. **Lamb's problem** — free surface correct vs analytic (E1.3). *The make-or-break for the physics.*
4. **Bridge** — house sim reproduces a pyprop8 layered case on soil (both engines cross-checked).
5. **Per-house manual + visual QA** — no house enters the GF campaign unreviewed.

## 8. Open decisions (need a call)

- **Engine** — resolved by E1.0. (Leaning SEM/SPECFEM3D.)
- **Multi-unit "which-flat" scope** — in (frontier research, doc 14 archetypes D/E) or out
  (v1 = single unit, archetypes A–C). Shapes the house-archetype set, not the pyprop8 spec.
- **Frequency** — 100 Hz (cheap, coarser) vs 250 Hz (finer localization, ~15–39× cost;
  cloud A100). Can decide after E2 shows the accuracy sensitivity to bandwidth.

## 9. Risk register (honest)

| Risk | Severity | Mitigation |
|---|---|---|
| **Sim-to-real transfer** (dominant, un-reducible pre-real-data) | High | Domain-randomize velocities/junction loss; Tier-2 ridge calibration on real steps; state plainly as the key limitation |
| Method doesn't localize honestly (Track 2) | High | Cheap to test first — that's E2's whole purpose; fail fast |
| Free surface wrong (Devito path) | High | SEM avoids it; else Lamb+SW4 gate before any data |
| Gmsh→SPECFEM3D converter friction | Medium | E1.0 de-risks it up front; CUBIT (commercial) is a fallback |
| Wall under-resolution | Medium | ≥3 cells/wall enforced in mesh/voxel gate |
| Which-flat ambiguity (multi-unit) | Medium | Scope out of v1 unless explicitly targeted |

---

*Referenced research: `house_localization/01–14`, `house_localization/engine_decision/{devito,
specfem3d,salvus,fem}.md`, `localization_1d/01–03`. This roadmap supersedes scattered "next step"
notes in those docs with a single execution plan.*
