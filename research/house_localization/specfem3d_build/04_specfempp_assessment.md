# SPECFEM++ Assessment — Can It Replace Fortran SPECFEM3D (and WSL) for the House PoC?

> Primary-source verification (2026-07-09) against the actual repo
> [PrincetonUniversity/SPECFEMPP](https://github.com/PrincetonUniversity/SPECFEMPP) —
> tree, CI workflows, CMake options, the 3-D benchmark, and open issues/merged PRs.
> Motivation: SPECFEM++ documents a **native Windows (MSVC + CUDA)** build path, which
> would eliminate the WSL/virtualization requirement of Fortran SPECFEM3D entirely.

---

## 1. Project status (verified)

| Fact | Evidence |
|---|---|
| Active, fast-moving | pushed **2026-07-09** (same day as this check); v0.5.0 released 2026-01; 116 open issues |
| Self-declared **"not ready for production use"** | README/docs |
| Unified rewrite of SPECFEM2D/3D/GLOBE in C++/Kokkos | README |
| GPU via Kokkos: CUDA + HIP presets (`release-cuda`, `debug-cuda`, `release-hip`) | `CMakePresets.json` |
| **MPI is OPTIONAL — `SPECFEM_ENABLE_MPI` default OFF** | top-level `CMakeLists.txt` |
| CI builds/tests are **Ubuntu-only** — no Windows job | `.github/workflows/compilation.yml`, `unittests.yml` |
| 2D is the mature side (`examples/` ships only `dim2`) | repo tree |

## 2. 3-D capability today (the part that matters for the house)

**Present and validated:**
- 3-D **elastic isotropic** solver — `benchmarks/src/dim3/homogeneous_halfspace` with
  `FORCESOLUTION` (point force), `STATIONS`, 3-component reference seismograms → the 3-D
  path is real and regression-checked.
- **Free surface** (Neumann) in 3-D — the benchmark runs "Neumann BCs on all edges".
- **Stacey absorbing + free-surface BCs in 3-D with tests — merged 2026-06-15** (very fresh).
- 3-D acoustic + kernels, wavefield IO, SIMD for 3-D (June 2026 PR train).
- GPU via Kokkos presets; deprecated separate executables → unified `specfempp` binary.
- Bundled **`fortran/meshfem3d`** = the classic *internal* structured mesher (writes the
  databases the C++ reader consumes: `read_materials`, `read_boundaries`,
  `read_pml_boundaries`, …). Its `Mesh_Par_file` keeps the classic `NMATERIALS`/`NREGIONS`
  (materials assigned to element-index boxes on a structured NEX_XI × NEX_ETA grid,
  vertical doublings).

**Missing today — the blockers for our use case (all confirmed as OPEN issues):**
- **No external 3-D mesh input.** `decompose_mesh` (the CUBIT/Gmsh 10-file route) exists
  only in 2-D; porting it to 3-D is **open issue [#1692](https://github.com/PrincetonUniversity/SPECFEMPP/issues/1692)**.
- **No Gmsh→3-D converter**: `gmsh2meshfem dim3` is **open issue [#1950](https://github.com/PrincetonUniversity/SPECFEMPP/issues/1950)**.
  (`scripts/gmshlayerbuilder` is a *layered-topography* helper for the internal mesher —
  not an arbitrary-geometry importer.)
- **No 3-D MPI** yet ([#1697](https://github.com/PrincetonUniversity/SPECFEMPP/issues/1697)) —
  single-process (+GPU) only. Fine for a single-house PoC; limits multi-GPU scaling.
- **Attenuation (Q) in the 3-D C++ solver: unclear/likely absent** (only the Fortran
  mesher-side attenuation model file exists). Concrete/soil Q matters for our GF library.
- Nonconforming (multi-resolution) 3-D meshes: in active development (#1941).

## 3. The Windows-native question (the "no WSL" prize)

- **For it:** MPI optional (the classically painful Windows dependency is simply OFF);
  CUDA presets exist; the docs explicitly give the MSVC+CUDA Kokkos flag
  (`-DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON`); Kokkos itself officially supports
  MSVC+CUDA. Local toolchain is already complete (VS2022, CUDA 12.8, CMake 3.31, Ninja,
  RTX 3090 → `Kokkos_ARCH_AMPERE86`).
- **Against it:** **no Windows CI** — nobody continuously builds this on Windows, so
  MSVC-specific breakage is undiscovered territory; VTK is ON by default (heavy on
  Windows — disable via `SPECFEM_ENABLE_VTK=OFF`).
- **Verdict: plausible but unproven.** A 1–2 h build attempt is a cheap experiment with a
  real chance of success; treat failure as expected-case, not surprising.

## 4. Could the *internal* mesher host a house? (the no-converter loophole)

The bundled MESHFEM3D supports `NMATERIALS`/`NREGIONS` — materials assigned to
element-index **boxes** on a **structured grid**. An axis-aligned house *could in
principle* be voxelized as material boxes (walls/slabs/soil) with **no external mesh and
no converter**. Costs and caveats:
- Structured grid = **uniform horizontal resolution** (no local wall refinement): 0.05 m
  walls force ~0.05 m everywhere → ~2–4 M elements (vs ~1 M multi-res) → SEM's
  mesh-adaptivity advantage is lost; VRAM likely exceeds the 3090 for full domains.
- Needs verification that the bundled port accepts many regions and that region
  boundaries land exactly on wall positions.
- **Interesting fallback/experiment — not the plan of record.**

## 5. Bottom line

| Question | Answer |
|---|---|
| Does SPECFEM++ have all the tools we need **today**? | **No.** The decisive gap: it **cannot ingest our Gmsh house mesh in 3-D** (no external-mesh/decompose_mesh path — open issues #1692/#1950). Also: no 3-D attenuation (likely), no 3-D MPI. |
| Is the no-WSL native Windows build real? | **Plausible, unproven** (MPI-off default helps a lot; no Windows CI). Worth a cheap attempt. |
| Is the 3-D core sound? | Yes — elastic + force source + free surface + Stacey (new) are present and benchmarked. It's the *geometry input* that's missing, which is exactly what a house needs. |
| Decision | **PoC stays on Fortran SPECFEM3D** (external hex mesh fully supported; our build docs 01–03 target it). **SPECFEM++ = watch-and-switch.** |

**Concrete switch trigger:** when **#1692 (decompose_mesh 3-D)** or **#1950 (gmsh2meshfem
dim3)** merges, re-evaluate immediately — at the current 3-D development pace (Stacey-3D
landed 3 weeks ago; weekly 3-D PRs) this could be **months, not years**. At that point
SPECFEM++ would offer: native Windows candidate (no WSL), single unified binary, Kokkos
CUDA on the local 3090, and no MPI requirement — a materially simpler ops story than
Fortran SPECFEM3D.

**Optional cheap side-bet (independent of the PoC):** attempt the native Windows
`release-cuda` build now and run the 3-D homogeneous-halfspace benchmark on the 3090.
Success = a working no-WSL SEM engine in hand for familiarization + the internal-mesher
voxel-house experiment (§4), and a head start for the switch. Failure costs ~1–2 hours.

---

*Related: `00_what_specfem3d_needs.md` (mesh requirements), `01–03` (Fortran SPECFEM3D
build specs — unchanged as plan of record), `12_specfem3d_adoption.md`.*
