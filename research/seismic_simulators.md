# Higher-Fidelity Seismic Forward Simulators for Near-Surface Footstep / Vehicle Synthesis

**Scope:** Evaluate higher-fidelity forward simulators as alternatives/upgrades to **pyprop8** for generating synthetic seismic training data for an edge classifier that detects footsteps and vehicles at metres-to-hundreds-of-metres range, ~1–250 Hz, near-surface, near-field.

**Bottom line (stated up front):** For this specific task the dominant sim→real error is **not** in the wave-propagation physics. It is overwhelmingly in (a) the **source model** (footstep ground-reaction force; multi-wheel/track vehicle loading) and (b) the **sensor/coupling/site/noise** chain. A higher-fidelity wave solver improves the part of the pipeline that is already the *least* wrong, while leaving the two binding constraints untouched. The honest recommendation is **do not switch the wave solver first**; invest in source realism, a stochastic site/coupling/noise layer, and real-data calibration/domain adaptation. A better propagation engine becomes worthwhile only later, and even then a 2-D/2.5-D viscoelastic FD or SEM code (free) is a better fit than a heavyweight 3-D global SEM or a commercial license.

---

## 1. What pyprop8 is (the baseline) and where its physics ends

pyprop8 is a pure-Python (NumPy/SciPy only) implementation of the O'Toole & Woodhouse (2011) propagator-matrix / wavenumber-integration algorithm for **complete synthetic seismograms in a 1-D plane-layered elastic half-space**, including the static offset, plus source-parameter derivatives (O'Toole, Valentine & Woodhouse 2012). It handles point-force and moment-tensor sources, is numerically stable from 0 Hz to high frequency, and is GPL-3.0 ([GitHub](https://github.com/valentineap/pyprop8), [JOSS paper, Valentine & Sambridge 2022, doi:10.21105/joss.04217](https://joss.theoj.org/papers/10.21105/joss.04217), [docs](https://pyprop8.readthedocs.io/en/latest/seismograms.html)).

This is a *semi-analytic, exact* solution **for the model it assumes**. Its physics envelope:

- **Includes:** exact P-SV + SH response of arbitrarily many flat layers; Rayleigh/Love surface waves *for a laterally homogeneous stack*; intrinsic attenuation if complex velocities are supplied; near-field + far-field + static terms; point-force source (well suited to a vertical footstep/wheel load).
- **Structurally cannot represent:** lateral velocity variation, 3-D heterogeneity, topography/free-surface relief, scattering/diffraction off buried objects or lateral contrasts, and the resulting **coda** that real near-surface records are full of. It also has no built-in moving-source / distributed-load support — a vehicle or a walking person must be synthesized as a *sequence/array of point forces* by the user.

So the "missing physics" relative to higher-fidelity codes is real but specific: **3-D/lateral heterogeneity, topography, and scattering-generated coda**, plus convenience features for distributed/moving sources.

---

## 2. The main simulator families

| Method / Code | Dimensionality | Key physics | Openness / License | Compute cost (relative) |
|---|---|---|---|---|
| **Discrete-wavenumber / propagator matrix** (pyprop8; also Hisada, EDGRN/EDCMP, CPS `hspec96`) | 1-D layered | Exact layered elastic response, attenuation, static term | GPL-3.0; free | **Cheapest.** Seconds on a laptop CPU; embarrassingly parallel over receivers/frequencies |
| **Spectral-element (SEM)** — SPECFEM2D / SPECFEM3D Cartesian / SPECFEM3D_GLOBE | 2-D / 3-D | (An)elastic, anisotropy, poroelasticity, attenuation (Q), topography via conforming hexahedral mesh, full 3-D heterogeneity, coupled acoustic-elastic | GPL (v2/v3); free; MPI + CUDA/OpenCL GPU | **High.** Needs mesh generation + MPI cluster (or GPU) for 3-D; 2-D runs on a workstation |
| **Spectral-element (commercial)** — Salvus (Mondaic) | 2-D / 3-D | Same SEM physics, polished meshing/workflow, FWI built in | **Commercial** license; academic license available; no public pricing | High; but turnkey, GPU-ready |
| **Finite-difference time-domain (FDTD)** — SW4, OpenSWPC, SOFI2D/3D, fdelmodc, AWP-ODC, Devito-based | 2-D / 3-D | Visco-elastic (GSLS/Zener Q), anisotropy, topography (SW4 curvilinear), point force + moment tensor; regular grids | SW4 GPL-v2+; OpenSWPC **MIT**; SOFI GPL; all free; MPI/OpenMP, some GPU | **Medium–high.** Regular-grid, easy to script; 2-D cheap, 3-D needs a cluster/GPU |
| **Finite-element (FEM)** — Abaqus, COMSOL, custom; 2.5-D FEM, FEM/infinite-element | 2-D / 2.5-D / 3-D | Arbitrary geometry, soil constitutive models (plasticity, poroelastic/saturated), contact, *moving loads* natively | Mostly **commercial** (Abaqus/COMSOL); some open FE | Medium–high; strong for soil-structure & moving-load problems |
| **Boundary-element (BEM) / 2.5-D BEM, TLM/FVM** | surface/interface mesh | Layered/homogeneous half-space, moving surface loads (railway/traffic), radiation to infinity handled exactly | Mostly research/commercial | Medium; efficient for half-space + surface loads |

Notes and sources:

- **SPECFEM** family: open-source SEM, GPL, GitHub-hosted; SPECFEM2D does acoustic/(an)elastic/poroelastic with CPML; SPECFEM3D Cartesian does the same on structured/unstructured hexahedral meshes; SPECFEM3D_GLOBE adds non-blocking MPI and GPU (CUDA/OpenCL) via source-to-source transformation; the codes are reference HPC benchmarks with excellent strong/weak scaling ([specfem.org](https://specfem.org/), [SPECFEM2D repo](https://github.com/SPECFEM/specfem2d), [SPECFEM3D repo](https://github.com/SPECFEM/specfem3d), [Tromp group software](https://tromp.princeton.edu/software)). Applicability to near-surface/geotechnical and exploration scales is established ([Earthquake Science, 2014, doi:10.1007/s11589-014-0069-9](https://link.springer.com/article/10.1007/s11589-014-0069-9)).
- **SW4** (Seismic Waves, 4th order): 3-D visco-elastic + anisotropy, **curvilinear grid conforming to topography** so the free-surface BC is applied at the true surface, super-grid absorbing boundaries, arbitrary point-force/moment-tensor sources; GPL-v2+; C++/Fortran, MPI/OpenMP ([SW4 repo](https://github.com/geodynamics/sw4)).
- **OpenSWPC**: 3-D/2-D FDM, generalized-Zener frequency-independent Q, MPI+OpenMP, **MIT license** (most permissive of the FD codes — relevant for a startup) ([OpenSWPC repo](https://github.com/OpenSWPC/OpenSWPC), [Earth Planets Space 2017, doi:10.1186/s40623-017-0687-2](https://link.springer.com/article/10.1186/s40623-017-0687-2)).
- **SOFI2D/3D**: viscoelastic staggered-grid FD (generalized standard linear solid), C, MPI; GPL ([opentoast.de](https://www.opentoast.de/Forward-modelling-code_SOFI.php)).
- **Salvus (Mondaic)**: modular SEM in 2-D/3-D, SalvusForward (forward only) vs SalvusComplete (FWI/custom workflows), academic license available, **commercial license required for company use, pricing not public — contact required** ([Mondaic get-salvus](https://www.mondaic.com/get-salvus), [docs](https://docs.mondaic.com/)).
- **Moving-load / traffic-vibration FE & BE** literature: railway/traffic ground vibration is "a typical low-frequency problem controlled by the Rayleigh wave in the top soil layer," solved with 2.5-D FEM, FEM/infinite-element, and TLM/FVM-BEM on layered half-spaces, often with saturated-soil poroelastic and plasticity constitutive models ([Investigation of ground vibrations from moving loads, Eng. Struct. 2006](https://www.sciencedirect.com/science/article/abs/pii/S0141029605002324), [2.5-D FEM, Sci China 2008](https://link.springer.com/article/10.1007/s11433-008-0060-3), [Numerical modeling of traffic-induced ground vibration, Comput. Geotech. 2011](https://www.sciencedirect.com/science/article/abs/pii/S0266352X11001091)).

---

## 3. What higher-fidelity codes buy that a 1-D layered model misses

A 3-D FD/FE/SEM/BE code can add, in roughly decreasing relevance to *this* task:

1. **Scattering and coda from lateral heterogeneity** — real near-surface ground is heterogeneous at the metre scale; the resulting scattered coda dominates the late part of footstep/vehicle records and is exactly what a 1-D model omits. This is the single most physically important thing a higher-fidelity solver adds.
2. **Topography and lateral velocity variation** — slopes, fill, buried utilities, water-table contrasts; change surface-wave amplitude/dispersion and arrival times.
3. **More realistic surface-wave (Rayleigh/Love) generation in 3-D** — pyprop8 already produces surface waves for a flat stack; 3-D codes add mode conversion and lateral trapping. For footsteps/vehicles, the signal is overwhelmingly Rayleigh-wave energy in the top soil layer, so getting dispersion right matters — but a *layered* model already captures the first-order dispersion.
4. **Realistic attenuation (Q)** — available in pyprop8 (complex velocities) and in all the FD/SEM codes; not a differentiator.
5. **Distributed / moving sources natively** — FE/BE/2.5-D codes model a moving multi-wheel or tracked load directly; in pyprop8 you must build it from point forces (doable, just manual).
6. **Nonlinearity / saturated-soil poroelasticity** — relevant very close to a heavy source; SPECFEM and FE soil codes support poroelastic/plastic soil, which a linear elastic layered model cannot. Usually second-order for detection-range signals.

These are genuine gains. The question (Section 4) is whether any of them is the *binding* constraint on classifier transfer.

---

## 4. The honest tradeoff: is propagation physics the binding constraint? (No.)

The sim→real gap for a learned footstep/vehicle classifier decomposes into three error sources. Evidence says the propagation term is the smallest:

**(A) Source model — LARGE, and a better wave solver does not help it.**
The recorded signal is, to first order, *source force* × *site/path transfer function*. The transfer function is what a wave solver computes; the *source force* is an independent input you must supply.
- **Footsteps:** broadband from a few Hz to ultrasonic, but seismic detection lives below ~100 Hz; the low-frequency content (<~200 Hz) comes from the **normal ground-reaction force** of heel-strike and toe-off, and depends on gait, footwear, body mass, surface, and stride — i.e., it is highly variable and person/surface-specific ([Vibration signature of human footsteps, JASA 2005](https://ui.adsabs.harvard.edu/abs/2005ASAJ..118.2021S/abstract), [Broad-frequency acoustic response to footsteps](https://www.researchgate.net/publication/271498947), [footstep GRF / floor-vibration biomechanics, arXiv:2503.16455](https://arxiv.org/pdf/2503.16455)).
- **Vehicles:** the seismic spectrum is **harmonic line series** set by engine firing and (for tracked vehicles) **track-pad periodicity × speed**; tracked vehicles' track component dominates and is >10× stronger than wheeled at close-medium range; classification works off the relative powers of the first ~15 harmonics ([Ground vibrations from tracked vehicles: theory and applications](https://www.researchgate.net/publication/282158486), [Detection of various vehicles using wireless seismic sensor network](https://www.researchgate.net/publication/235738325)).
- A flat-layer code and a 3-D SEM, fed the *same wrong source*, both produce wrong waveforms. **Upgrading the propagation engine cannot fix a mis-specified source spectrum/force history.** The leverage here is realistic, randomized GRF and multi-wheel/track loading templates — not the solver.

**(B) Sensor / coupling / site / noise — LARGE, and a better wave solver does not help it.**
- **Geophone–ground coupling** alters amplitude and phase above the coupling resonance; coupling resonances of 100–500 Hz are reported and vary with soil firmness, burial, and moisture — i.e., directly inside the band of interest and **site-dependent** ([Geophone ground coupling, Geophysics 1984, doi:10.1190/1.1441700](https://library.seg.org/doi/10.1190/1.1441700)).
- **Site response / near-surface conditions** cause frequency-dependent amplification and vary over a few metres laterally; soil moisture and freezing change the response materially.
- **Real-world false alarms** are dominated by **wind** (noise below a few hundred Hz rises sharply in wind), **rain/soil-moisture and freezing**, and other environmental transients; operational seismic intrusion systems must auto-calibrate per-site thresholds to the local noise ([seismic intrusion detection false-alarm behavior, US Pat. 4,107,660](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4107660); [Magus GeoPS](https://miratechnologies.at/products-and-services/magus-geops/)).
- None of coupling transfer, site amplification, or environmental noise is produced by the elastic wave solver; they are added *after* it. A 3-D SEM that nails propagation but is convolved with a wrong coupling response and trained without realistic noise will still transfer poorly.

**(C) Propagation physics — the part pyprop8 already does decently.**
For near-field footstep/vehicle signals, the energy is dominated by **direct + Rayleigh-wave arrivals in the top soil layer** — which a layered model captures to first order. The 1-D model's real deficiency is **coda from lateral heterogeneity/scattering**. That coda matters for some discriminative features, but it is (i) partially mimickable by stochastic perturbation of the layered model and added reverberation/noise, and (ii) second-order compared to getting the source and the sensor/noise chain right.

Broader ML evidence agrees: sim→real failure is usually an *appearance/content* distribution mismatch fixed by **domain randomization** and **real-data adaptation/calibration**, not by adding solver fidelity ([sim2real domain-gap survey, arXiv:2311.11039](https://arxiv.org/pdf/2311.11039); [MLReal, bridging synthetic↔real in ML geophysics, Leading Edge / ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666544122000260); urban-seismic footstep CNN demonstrating real-noise dominance, [Footstep detection with a CNN](https://www.academia.edu/97604131/Footstep_detection_in_urban_seismic_data_with_a_convolutional_neural_network)).

**Conclusion of the tradeoff:** propagation physics is *not* the binding constraint. The two binding constraints are the **source force model** and the **sensor/coupling/site/noise model**. A higher-fidelity wave solver improves neither.

---

## 5. Cost / effort to adopt each, and recommendation

**Effort to adopt (engineering + compute), for a synthetic-data pipeline:**

- **Stay on pyprop8 / add a 1-D DW alternative (CPS, Hisada):** ~0 new infra; seconds/CPU per shot; trivially parallel; GPL — fine for internal use, but GPL distribution constraints matter if you ship the simulator with a product. **Lowest cost.**
- **2-D viscoelastic FD (OpenSWPC — MIT, or SOFI2D, or fdelmodc):** moderate effort (build, scripting, absorbing-boundary tuning); a 2-D run is workstation-scale (minutes); lets you inject lateral heterogeneity + topography + coda cheaply. **OpenSWPC's MIT license is the friendliest for a startup.** **Low–moderate cost, best fidelity-per-dollar upgrade if you upgrade at all.**
- **SW4 (3-D FD, GPL-v2+):** higher effort (mesh/topography setup); 3-D needs a multicore/cluster or GPU; minutes-to-hours per scenario; excellent topography handling. **Moderate–high cost.**
- **SPECFEM2D (free, GPL):** moderate effort; 2-D on a workstation; full (an)elastic/poroelastic + topography; good if you want exact surface-wave physics in 2-D. **SPECFEM3D:** high effort (hexahedral meshing is the real cost) + MPI/GPU cluster; hours and real HPC budget per 3-D scenario — **disproportionate for generating a *training set* of thousands of scenarios.**
- **Salvus (commercial):** lowest *engineering* effort (turnkey meshing/workflow, GPU), but **license cost + vendor dependency**; academic license won't cover a startup product. **High monetary cost; only justified if 3-D SEM is genuinely required at scale and you want to avoid maintaining a solver.**
- **FE (Abaqus/COMSOL) or 2.5-D FE/BE for moving loads:** best *native* moving-load/soil-constitutive fidelity; commercial license; heavy per-run cost. Useful as a **one-off generator of high-quality source/near-field templates**, not as the bulk data engine.

**Recommendation (ordered):**

1. **Do not switch the wave solver as the first move.** Keep pyprop8 (or swap to an MIT/permissive 1-D DW code if GPL shipping is a concern) as the propagation core.
2. **Invest first in the source model.** Build realistic, *randomized* footstep ground-reaction-force histories (heel-strike/toe-off, gait/mass/footwear variation) and vehicle sources as harmonic-line + multi-wheel/track moving point-force arrays. This is where the largest, cheapest sim→real reduction lives. Optionally use one FE/BE moving-load study to generate ground-truth source/near-field templates.
3. **Invest second in the sensor/coupling/site/noise layer.** Convolve outputs with a stochastic geophone-coupling transfer function (resonance 100–500 Hz, soil-dependent), apply per-site amplification, and **train with real environmental noise** (wind, rain, traffic, soil-moisture states). Use **domain randomization** over all of these.
4. **Calibrate against real recordings and use domain adaptation** (e.g., MLReal-style statistics matching, or fine-tuning on a small labelled real set). This closes the residual gap far more cheaply than any solver upgrade.
5. **Only then, if coda/heterogeneity features are demonstrably limiting** (verify by ablation against real data), add **2-D viscoelastic FD via OpenSWPC (MIT)** — or SW4/SPECFEM2D for topography — to inject scattering. Reserve 3-D SEM (SPECFEM3D) or commercial Salvus for special, low-volume high-fidelity studies, not bulk training-set generation, because the per-scenario HPC and meshing cost does not scale to thousands of training scenarios.

**One-line verdict:** Upgrading the simulator is the wrong first lever for this task — the propagation physics is already adequate; the money should go to source realism, a stochastic sensor/coupling/site/noise model, and real-data calibration. If/when a solver upgrade is justified, choose a free permissive 2-D viscoelastic FD code (OpenSWPC, MIT), not a 3-D SEM or a commercial license.

---

## Sources

- pyprop8: [GitHub](https://github.com/valentineap/pyprop8) · [JOSS, Valentine & Sambridge 2022](https://joss.theoj.org/papers/10.21105/joss.04217) · [docs](https://pyprop8.readthedocs.io/en/latest/seismograms.html)
- SPECFEM: [specfem.org](https://specfem.org/) · [SPECFEM2D](https://github.com/SPECFEM/specfem2d) · [SPECFEM3D](https://github.com/SPECFEM/specfem3d) · [Tromp software](https://tromp.princeton.edu/software) · [SEM near-surface/geotechnical, Earthquake Science 2014](https://link.springer.com/article/10.1007/s11589-014-0069-9)
- SW4: [GitHub geodynamics/sw4](https://github.com/geodynamics/sw4)
- OpenSWPC (MIT): [GitHub](https://github.com/OpenSWPC/OpenSWPC) · [Earth Planets Space 2017](https://link.springer.com/article/10.1186/s40623-017-0687-2)
- SOFI2D/3D: [opentoast.de](https://www.opentoast.de/Forward-modelling-code_SOFI.php)
- Salvus (commercial): [get-salvus](https://www.mondaic.com/get-salvus) · [Mondaic docs](https://docs.mondaic.com/)
- Moving-load / traffic vibration: [Eng. Struct. 2006](https://www.sciencedirect.com/science/article/abs/pii/S0141029605002324) · [2.5-D FEM, Sci China 2008](https://link.springer.com/article/10.1007/s11433-008-0060-3) · [Comput. Geotech. 2011](https://www.sciencedirect.com/science/article/abs/pii/S0266352X11001091)
- Footstep source/signature: [JASA 2005](https://ui.adsabs.harvard.edu/abs/2005ASAJ..118.2021S/abstract) · [Broad-frequency response](https://www.researchgate.net/publication/271498947) · [GRF biomechanics, arXiv:2503.16455](https://arxiv.org/pdf/2503.16455) · [Footstep CNN](https://www.academia.edu/97604131/Footstep_detection_in_urban_seismic_data_with_a_convolutional_neural_network)
- Vehicle source/signature: [Tracked-vehicle ground vibrations](https://www.researchgate.net/publication/282158486) · [Wireless seismic sensor network vehicle detection](https://www.researchgate.net/publication/235738325)
- Sensor/coupling/site/noise: [Geophone ground coupling, Geophysics 1984](https://library.seg.org/doi/10.1190/1.1441700) · [Seismic intrusion false alarms, US Pat. 4,107,660](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4107660) · [Magus GeoPS](https://miratechnologies.at/products-and-services/magus-geops/)
- Sim→real / domain gap: [Sim2Real survey, arXiv:2311.11039](https://arxiv.org/pdf/2311.11039) · [MLReal](https://www.sciencedirect.com/science/article/pii/S2666544122000260)
