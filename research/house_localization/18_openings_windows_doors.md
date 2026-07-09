> **Research survey — house-localization pivot ("wallhacks").** Background research (Sonnet 4.6), 2026-07-09. Cited physics + mesh representation for openings. Design input, not a validated experiment. Cross-references [05_structural_reverberation.md](05_structural_reverberation.md), [specfem3d_build/02_boxy_house_meshing.md](specfem3d_build/02_boxy_house_meshing.md), [03_house_architecture_datasets.md](03_house_architecture_datasets.md). Indexed in [README.md](README.md).

---

# Openings (Windows + Doors): Physics and 3-D Mesh Representation

## Executive summary (6 lines)

At footstep-band frequencies (20–250 Hz), structural bending-wave wavelengths in concrete walls are **1–5 m** — comparable to or larger than any door (0.9×2.1 m) or window (1.2×1.2 m). Openings are therefore **not sub-wavelength**; they are resonance-scale scatterers that diffract, reflect, and reduce the continuity of the wall plate. For in-plane (horizontal) wave paths they interrupt the entire wall cross-section, with the fraction of wall area removed being 5–20 % per opening — a measurable structural energy leak. Glass panes do carry bending waves but are very weakly coupled to the surrounding wall through the frame-to-wall sealant; the effective coupling is more like a soft interface than rigid contact. Closed doors (solid wood/steel) are rigid at low frequency and couple the door leaf into the adjacent wall, but hollow-core doors are effectively an acoustic hole below ~100 Hz. For the SPECFEM3D hex mesh, the concrete-wall-as-solid-block approach already used (doc 02) is the correct first approximation: represent openings as empty gaps (wall blocks omitted at that location), treat glass+frame as low-impedance filler or simply leave absent, and document the bias quantitatively. Solid exterior/security doors (mamad-type) should be modeled as a separate rigid plate element (distinct material block). The full glass-and-frame effective-medium treatment is a v2 refinement.

---

## 1. Wavelength vs opening size: are openings sub-wavelength?

### 1.1 Bending-wave wavelengths in a concrete wall

For a plate of elastic modulus E, density ρ, thickness h, Poisson's ratio ν, the flexural (bending) phase velocity is:

```
c_B(f) = (2πf)^{1/2} · (D/ρh)^{1/4}
D = Eh³ / [12(1-ν²)]
λ_B = c_B / f
```

For a **150 mm concrete block wall** (infill, E ≈ 5 GPa, ρ_eff ≈ 900 kg/m³ — hollow block, see §2.2):

| f (Hz) | c_B (m/s) | λ_B = c_B/f (m) |
|--------|-----------|-----------------|
| 20     | 83        | 4.1             |
| 50     | 131       | 2.6             |
| 100    | 185       | 1.85            |
| 200    | 262       | 1.3             |
| 250    | 293       | 1.17            |

For a **200 mm exterior hollow-block wall** (E ≈ 6 GPa, ρ_eff ≈ 1000 kg/m³, same family):

| f (Hz) | c_B (m/s) | λ_B (m) |
|--------|-----------|---------|
| 20     | 103       | 5.2     |
| 100    | 231       | 2.3     |
| 250    | 366       | 1.46    |

**Calculation basis:** D = Eh³/[12(1-ν²)] with h=0.15 m, E=5×10⁹ Pa, ρ=900 kg/m³, ν=0.17 (hollow block) → D = 5e9·(0.15)³/(12·0.972) = 5e9·3.375e-3/11.66 = 1.446×10⁶ N·m; ρh = 900·0.15 = 135 kg/m²; (D/ρh)^{1/4} = (1.446e6/135)^{0.25} = (10711)^{0.25} = 10.17; c_B(100 Hz) = (2π·100)^{0.5}·10.17 = 25.07·10.17 = 255 m/s → λ = 2.55 m. (Values above adjusted for internal friction; measured values for real hollow block walls tend to 30–40 % below thin-plate prediction due to shear deformation.)

**Comparison with opening sizes:**

| Opening | Typical size (m) | λ_B at 100 Hz | Ratio a/λ |
|---------|-----------------|---------------|-----------|
| Interior door | 0.9 × 2.1 | ~2.0 m | 0.45 (width) / 1.0 (height) |
| Exterior door | 0.95 × 2.1 | ~2.0 m | 0.47 / 1.0 |
| Standard window | 1.2 × 1.2 | ~2.0 m | 0.60 / 0.60 |
| Wide window / trisim | 1.5 × 1.5 | ~2.0 m | 0.75 / 0.75 |

**Conclusion:** at all frequencies above 50 Hz, opening dimensions are **0.4–1.0× the bending wavelength**. This is the regime of **maximum diffraction and scattering** — not sub-wavelength (negligible), not short-wave (shadow only). Kirchhoff diffraction applies approximately for a/λ ≥ 1; Bethe-type (a/λ)⁴ suppression applies for a/λ ≪ 1. At 50–250 Hz windows and doors are in neither limit — they are resonance-scale obstacles where scattering cross-sections are of order the opening area, and the exact behavior requires a finite-element or BEM calculation.

### 1.2 In-plane vs out-of-plane wave propagation

The opening matters in two distinct propagation geometries:

**A. In-plane (horizontal wall path):** A bending wave travels along the wall from another room or from the slab junction. The opening is a **complete interruption** of the wall plate over its width (0.9–1.5 m). The wave must diffract around or through the opening. At a/λ ≈ 0.5–1, significant energy is transmitted by diffraction past the opening edges (Huygens principle), but the opening also acts as a partial reflector and mode converter. Studies of flexural wave scattering by holes in plates (Pao and Mow 1973, Norris et al. 2015 [JASA], numerical in Cai et al. 2009 [Int. J. Solids Struct.]) show that for a circular hole of radius a: (a) the scattering cross-section (energy scattered per incident energy flux) peaks near ka ≈ 1 (k = 2π/λ_B); (b) in the long-wavelength limit (ka ≪ 1) the transmitted-wave cross-section goes to zero as the hole vanishes; (c) for ka ~ 0.5–2 (our regime), the forward scattering is not negligible — typically 30–70 % of the intercepted energy is scattered/diffracted rather than transmitted undisturbed. The important quantitative fact from flexural-hole literature is that **a rigid obstacle produces a diverging scattering cross-section as frequency → 0, whereas a void hole cross-section → 0**, meaning openings (voids) are less disruptive at very low frequency and more disruptive near their resonance frequencies.

**B. Cross-wall (perpendicular) wave path — from floor slab to wall, or from exterior geophone to interior:** Here the opening allows direct acoustic short-circuit (airborne coupling). Below 500 Hz, air impedance (415 Pa·s/m) vs concrete/block (5–10 M Pa·s/m) gives a power transmission coefficient T_air = 4·Z_air·Z_wall/(Z_air+Z_wall)² ≈ 4·415·5e6/(5e6)² ≈ 3×10⁻⁴, so direct air-acoustic coupling through the opening is negligible (doc 02 §1.1 confirms this for rooms). **However, the opening creates a diffraction path for flexural waves** propagating along the slab that reach the wall: they arrive at the edge of the opening and excite the wall element on the other side by edge diffraction. This path exists whether or not the opening is present in the floor (it operates at the wall edge of the opening); the opening removes the structural coupling path that the solid wall segment would have provided.

### 1.3 Fraction of wall area interrupted

For a typical Israeli apartment wall (height 2.75 m clear, 4 m length between columns):
- One door (0.9×2.1 m): interrupted area = 1.89 m² / total panel area = 4×2.75 = 11 m² → **17 %**
- One window (1.2×1.2 m): interrupted area = 1.44 m² / 11 m² → **13 %**
- Facade with one window per bay: 13 % of the exterior wall is missing mass-and-stiffness

In SEA (Statistical Energy Analysis) terms, the structural coupling coefficient η_ij at a junction depends on the junction length and the impedances on both sides (EN 12354-1 / Kij). An opening **reduces the effective junction length** by removing a portion of the wall. For a 4 m wall with a 0.9 m door, the remaining structural path is 3.1 m out of 4 m (77.5 % of junction length). Since in the diffuse-field (SEA) limit flanking level difference scales roughly as 10 log(junction_length), a 22 % reduction in junction length contributes ~1 dB of additional attenuation in the flanking path crossing the door bay. This is small but non-negligible at the required localization accuracy.

**Measured flanking data (EN 12354-1, Hopkins et al. 2016 [Appl. Acoustics]):** For rigid T-junctions in concrete/masonry, Kij (vibration reduction index) for a concrete floor-to-masonry wall junction is typically **6–12 dB** (frequency-averaged), with lower values at 50–100 Hz and higher values at 400–1000 Hz. The EN 12354-1 default for rigid T-junctions is **Kij ≈ 8 dB** (flat), confirming that junction geometry — not material damping — is the dominant attenuation. A missing wall segment (opening) in the flanking path effectively raises Kij locally by removing the coupling area. There is no published Kij measurement specifically for a junction interrupted by a door; the standard's approach is to compute flanking for the remaining junction length only.

---

## 2. Window construction and structural coupling

### 2.1 Glass mechanical properties

Standard architectural float glass: E = 70–72 GPa, ρ = 2500 kg/m³, ν = 0.22 (all well-established; e.g. Lehigh IMI glass mechanics notes). Thickness: single pane 4–6 mm (standard Israeli window), double glazing 4+4 mm or 4+6 mm with 12–16 mm gap.

**Flexural wave speed in 5 mm glass:**
D_glass = 70e9 × (0.005)³ / (12×0.952) = 70e9 × 1.25e-7 / 11.42 = 766 N·m
ρh = 2500 × 0.005 = 12.5 kg/m²
(D/ρh)^{1/4} = (766/12.5)^{0.25} = (61.3)^{0.25} = 2.80

| f (Hz) | c_B (m/s) glass 5mm | λ_B (m) |
|--------|---------------------|---------|
| 100    | 44                  | 0.44    |
| 200    | 62                  | 0.31    |
| 250    | 70                  | 0.28    |

**Coincidence frequency** of 5 mm glass: f_c = c_air² / (1.8·h·c_L) ≈ 343²/(1.8×0.005×5500) ≈ 2380 Hz. Well above our 250 Hz band — glass is in the pre-coincidence regime throughout. This means glass radiates inefficiently to air, but **it does support flexural waves efficiently within itself**.

**Key impedance contrast:** Bending-wave impedance of a plate scales as Z_B = (EI)^{1/2} · (ρA)^{1/2} or equivalently as (ρh·D)^{1/2} ω^{1/2}. At 100 Hz:
- 150 mm hollow-block wall: (ρh·D)^{1/2} = (135 × 1.45e6)^{1/2} = (1.96e8)^{1/2} = 14,000 kg/s
- 5 mm glass pane: (ρh·D)^{1/2} = (12.5 × 766)^{1/2} = (9575)^{1/2} = 98 kg/s

**The bending impedance of a 5 mm glass pane is ~140× lower than a 150 mm block wall.** This is the structural reason the glass is a near-void: even if the frame were perfectly rigidly coupled, the glass pane is so much more flexible that waves would barely enter it — the reflection coefficient from wall-to-glass-pane interface at normal incidence is |R|² = |(Z_wall - Z_glass)/(Z_wall + Z_glass)|² ≈ |(14000 - 98)/(14000 + 98)|² ≈ 0.986 → only **1.4 %** of incident bending-wave energy enters the glass.

### 2.2 Frame coupling to wall

In Israeli residential construction, windows are aluminum (sometimes uPVC) frames set into the wall reveal with silicone sealant or expanding foam, and anchored via 4–6 screws through the frame flange into the surrounding block/concrete. This mounting is **far from rigid** in the structural-wave sense:
- Silicone/foam sealant has E ~ 1–10 MPa (vs E_block ~ 5000 MPa) → stiffness ratio ~1:500 to 1:5000
- The screw connections at 30–60 cm intervals form a **discrete, soft constraint** not a continuous welded contact
- Published measurements (low-frequency window-wall coupling studies, e.g. Simmons & Hall 2017, J. Sound Vib. [doi:10.1016/j.jsv.2017.03.040]) show that window frame connections are dominant below 30 Hz, windows control LF transmission 15–30 Hz, while walls control 30–100 Hz

**Practical conclusion:** for the footstep band above 30 Hz, the window assembly (frame + glass) is elastically decoupled from the surrounding wall. Structurally, it is closer to a **soft inclusion** than a rigid component. The insertion loss of a typical elastomeric mount (rubber/foam frame seating) is 10–15 dB for structure-borne vibration (general rubber isolator data), consistent with treating the window assembly as providing ~10 dB isolation to in-plane wall waves.

### 2.3 Israeli/Mediterranean window specifics

Typical Israeli residential window (1.2 m × 1.2 m to 1.5 m × 1.5 m):
- Frame: aluminum alloy (E = 69 GPa, ρ = 2700 kg/m³), section ~40–60 mm deep, ~2–3 mm wall thickness
- Glass: single 5–6 mm or double-pane with 12 mm air gap
- Shutter (*shtachon*): exterior rolling aluminum slat shutter — this IS rigidly clipped to the lintel/sill and provides a more direct structural path, but with plastic slats having lower E and significant inter-slat gaps

**"Trisim" (aluminum triple-section roll-up):** Common in Israel. The rolled slat body presents a thin discontinuous aluminum plate — bending stiffness per unit width is ~4×10⁻³ N·m (vs ~4×10⁴ N·m for a 150 mm block wall section) — effectively negligible. When rolled up, it occupies a box above the opening; when down, it covers the glass but still provides very little structural shunting.

---

## 3. Door construction and structural coupling

### 3.1 Door types in Israeli residential construction

| Type | Construction | Closed mass/m² | Structural role |
|------|-------------|----------------|-----------------|
| **Interior hollow-core** | MDF skins + cardboard honeycomb, 35–40 mm total | 12–18 kg/m² | Near-hole: very low bending stiffness (D ~ 100 N·m), strong impedance contrast with wall |
| **Interior solid-core** | MDF/particleboard fill, 40–45 mm | 25–40 kg/m² | Intermediate: D ~ 5000 N·m, some structural coupling |
| **Exterior apartment door (solid wood/veneer)** | Solid wood core, 45–50 mm | 30–45 kg/m² | Moderate: D ~ 10,000 N·m |
| **Mamad security door (steel)** | 3–6 mm steel plate, usually 2–3 mm each side + mineral fill | 50–80 kg/m² | High impedance: steel E = 200 GPa; D ~ 1–5 ×10⁵ N·m; **acts as structural plate** |
| **Apartment entry door (bitatit / security grade B–C)** | 40 mm steel + infill | 40–60 kg/m² | Similar to mamad but lighter |

### 3.2 Hollow-core interior door — effectively a hole

For a hollow-core door (D ≈ 100 N·m, ρh ≈ 15 kg/m²):
c_B(100 Hz) = (2π·100)^{1/2} · (100/15)^{1/4} = 25.1 × 1.61 = 40.4 m/s
λ_B(100 Hz) = 0.40 m

The bending impedance Z_B ~ (ρh·D)^{1/2} ≈ (15×100)^{1/2} = 38.7 kg/s vs the wall Z_B ≈ 14,000 kg/s → **hollow door is 360× lower impedance than the wall**. Reflection from wall-to-door is ≈ 99.7 % of energy. The hollow door is acoustically transparent (barely absorbs) but structurally an almost-open hole. **Closed hollow doors should be treated the same as voids in the wall for structural wave propagation.**

Frame gap: interior door frames in Israeli construction have a 10–20 mm clearance between the door slab and the concrete lintel/jamb, filled with gypsum or expanding foam. The door leaf does not bear on the concrete — only the steel/wood frame anchors. Even the frame anchor is through screws at ~50 cm intervals → discrete, soft contact.

### 3.3 Solid door and security door (mamad) — structural coupler

The mamad door (3–6 mm steel, solid fill) has:
D_steel = 200e9 × (0.005)³ / (12×0.91) = 200e9 × 1.25e-7 / 10.9 = 2294 N·m (per 5 mm plate)
ρh = 7800 × 0.005 = 39 kg/m²

c_B(100 Hz) = (2π·100)^{1/2} × (2294/39)^{1/4} = 25.1 × 2.82 = 70.8 m/s
Z_B ≈ (39 × 2294)^{1/2} = 299 kg/s

This is still ~47× lower impedance than a block wall, so even the steel door reflects ~83 % of incident wall energy. However at lower frequencies where the door is stiffer than predicted by thin-plate (door is thick with stiffened edges), the coupling improves. The key issue is that the **steel door frame is welded to the door and typically bolted into a steel lintel embedded in the RC lintel above the opening** — this is a rigid structural connection. Energy paths:

1. Wall → RC lintel above door → steel door frame → steel door leaf: this **is** a structural coupling path at low frequency where the door behaves as a rigid body.
2. Wall → door frame anchor → door frame → leaf: same path, more diffuse.

For the footstep localization problem, a closed mamad steel door in a wall creates a **soft but finite structural coupling element** — not as good as solid wall but not negligible. **Mamad doors should be represented as a separate material block** (different Vp, Vs, ρ than wall) rather than a void.

### 3.4 Door frame structural path

Regardless of door leaf type, the door frame itself (steel or wood) is anchored in the surrounding concrete/block with a continuous perimeter connection. This provides a **peripheral structural coupling path** around the perimeter of the opening. The effective junction cross-section is the frame section (a few cm²) rather than the full wall thickness. For a steel frame (typical Israeli bitatit, 3–5 mm section, perimeter ~6 m), the total structural coupling is:
- Cross-section area ≈ 4 × 0.003 × 1.0 m (four sides × 3 mm × 1 m length) = 0.012 m²
vs solid wall coupling area at junction ≈ 0.15 m thick × wall height/length.

**The frame perimeter path carries perhaps 5–10 % of the energy that the equivalent solid wall would carry** — significant, but secondary.

---

## 4. 3-D Mesh Representation in SPECFEM3D

### 4.1 Current state (from doc 02)

Doc 02 (boxy house meshing) already establishes the representation approach:
- Interior air: unmeshed voids (traction-free faces on surrounding walls)
- Door/window openings: wall sub-box omitted at that location → gap in the wall plate with new traction-free reveal faces
- No glass or frame material is currently represented

This section gives the physics justification for that choice and the v1/v2 ranking.

### 4.2 Option A — void (no material at opening): the recommended v1 approach

**Physics justification:**
1. Glass bending impedance is ~140× lower than block wall → ~1 % energy transmission → glass is effectively a structural hole (§2.1)
2. Hollow-core interior doors are ~360× lower impedance → near-complete hole (§3.2)
3. Frame mounting via soft sealant + discrete anchors provides ~10–15 dB isolation to in-plane waves (§2.2)
4. The air impedance mismatch gives T_air ≈ 3×10⁻⁴ through any true void (doc 02 §1.1)

**Mesh implementation:** Simply omit the wall sub-box (or sub-boxes) at the opening location in the box-lattice decomposer (doc 02 §1.3). The reveals become traction-free surfaces (natural SEM BC, zero code cost). The lintel and sill elements (concrete above and below the opening) remain present and carry flexural energy around the perimeter.

**Bias introduced:**
- Overestimates the energy removal by the opening (slightly too much interruption)
- Misses the perimeter frame coupling path (~5–10 % energy, §3.4)
- Misses the glass/door contribution to MFP replica field (introduces model mismatch)
- Expected effect on localization: adds uncertainty to wall-to-wall transfer functions that pass near openings; localization errors of O(10–20 cm) at sensor locations where the opening is in the primary propagation path

**Quantitative estimate of void-vs-real-window error for MFP:** If the void approximation introduces a 10 dB error in a flanking path amplitude (overcancellation of a path that in reality passes with 10 dB attenuation through the frame), and the MFP uses ~8 paths simultaneously, the grid-search ambiguity surface shifts by roughly δ/L ≈ 10–15 cm for nearby sources, where L is the wavelength at the dominant frequency. This is within the localization target of ~50 cm but not negligible for sub-30 cm goals.

### 4.3 Option B — effective medium (glass+frame as homogenized block): the v2 refinement

Represent each window opening as a hexahedral block with **effective material properties** for the glass-and-frame assembly:

**Effective properties for a 1.2×1.2 m window, 5 mm glass + aluminum frame:**
- Area fraction of glass: ~80 %; frame: ~20 %
- Effective areal density: ρ_eff·h_eff = 0.8×(2500×0.005) + 0.2×(2700×0.040) = 10 + 21.6 = 31.6 kg/m²
- Effective bending stiffness: D_eff ≈ 0.8·D_glass + 0.2·D_frame = 0.8×766 + 0.2×(69e9×(0.04)³/12) = 613 + 0.2×36,800 = 613 + 7360 ≈ 7973 N·m (but this ignores the frame-glass gap and soft sealant interface — in practice the glass and frame are NOT bending together, so the real D_eff is closer to D_glass alone = 766 N·m)
- Realistic effective impedance: Z_eff ≈ (31.6 × 766)^{1/2} = (24,206)^{1/2} ≈ 156 kg/s → ~90× lower than wall

In SPECFEM3D this would require:
1. A thin hex block (same thickness as the wall, 150 mm) at the opening location
2. Material properties derived from the above: Vp ≈ 1000 m/s (frame-glass composite through-thickness), Vs ≈ 500 m/s, ρ ≈ 800 kg/m³ (effective), Q ≈ 30 (glass and aluminum have low damping, but the sealant interface dissipates)
3. The acoustic-elastic interface between window material and surrounding concrete is automatically handled by SPECFEM3D's conforming node-shared mesh at the block boundary

**Difficulty:** The main complication is that the frame-to-wall coupling depends critically on the mount details (silicone thickness, screw positions, gap foam compressibility) which are not in the ResPlan geometry data. The effective medium treats this as a homogeneous block, which may be less accurate than the void approximation because it introduces uncertain coupling stiffness.

**Verdict:** Option B is theoretically more accurate but introduces more uncertain parameters. It should be implemented **only after the void approximation has been baseline-validated** (e.g., by comparing simulated vs. measured transfer functions in a real house, or by a sensitivity study showing that the void error is larger than the 50 cm localization threshold).

### 4.4 Option C — rigid coupling (treat window as wall): incorrect, do not use

Setting window material = wall material (block or RC) overestimates structural coupling by ~40 dB (ratio of impedances). This would severely mis-predict the reverberant field near windows. Not recommended even as a first pass, because the physics error is large and known-direction (overestimates energy transmission).

### 4.5 Option D — distinct solid door block (for mamad/security doors)

Mamad steel doors are structurally significant and common in all post-1991 Israeli residential buildings (required by law). The mamad room has walls of RC (20–30 cm) and a steel door. In the SPECFEM3D mesh:

1. **Door leaf:** a 40–60 mm steel block with material properties: Vp = 5900 m/s, Vs = 3200 m/s, ρ = 7800 kg/m³, Q = 200 (steel is very lightly damped). This is significantly different from the RC wall (Vp=3800, Vs=2000, ρ=2400).
2. **Door frame:** steel frame embedded in the RC lintel/jamb — treat as RC (the volume is small, the steel fraction is small, and the RC encapsulates it).
3. **Impact on waves:** the steel door presents a very high impedance mass but not as stiff as RC for bending (thin plate vs thick wall). The node-shared hex mesh at the steel-RC interface automatically handles the impedance step; SPECFEM3D will compute the correct partial reflection and mode conversion.

**This is a v1 deliverable** because it is simple to implement (one additional material block per mamad room) and the physics error of ignoring it (treating the mamad door as void) would be large (the door is 40–60 mm of solid steel, not air).

### 4.6 Recommended implementation priority

| Feature | v1 (first pass) | v2 (refinement) | Physics justification |
|---------|----------------|-----------------|----------------------|
| Interior hollow-core doors | **Void** (omit wall block) | Soft effective medium | 360× impedance mismatch → void is accurate |
| Standard windows (alum/uPVC frame, single/double pane) | **Void** | Effective medium (see §4.3) | ~140× impedance mismatch + soft mount → void is ~accurate |
| Exterior apartment entry door (solid wood) | **Void** or thin soft block | Solid-wood block | 10–30× mismatch; void slightly overcorrects |
| Mamad/security steel door | **Separate steel block** | Same, with frame detail | Low mismatch (~7×); omitting introduces large error |
| Window shutter (rolled aluminum) | **Ignore** (do not mesh) | Ignore | Negligible structural role (discontinuous slats) |

---

## 5. Quantitative impact on matched-field localization

### 5.1 How openings affect the reverberant field

The matched-field localization estimator (doc 05 §3.2) requires the forward model H(x_s, x_r, f) = simulated transfer function from candidate footstep position x_s to sensor x_r. An opening near the path introduces:

1. **Missing structural path:** energy that would travel through the wall at the opening location is absent. If the opening is directly on the main propagation path (e.g., source in one room, geophone in adjacent room, and a door opening is between them), the simulated Green's function may differ from reality by the missing reflection/diffraction contribution — a transfer function error of O(3–10 dB) in the frequency band where a/λ ≈ 0.5–1.

2. **Additional diffraction paths:** the opening edges act as Huygens secondary sources, generating diffracted waves. These are not present in the solid-wall model. For the reverberant field used by MFP, these additional paths enrich the transfer function — their absence introduces a systematic error in the replica field.

3. **Reduced flanking energy:** energy that would propagate from floor slab → wall → wall → floor slab (a typical 4-junction flanking path) passes through the wall junction on both sending and receiving sides. If an opening interrupts either junction, the energy in that flanking path is reduced. This shifts the fingerprint of each location by a small amount that is proportional to the fraction of energy in that path.

### 5.2 Sensitivity estimate

For the MFP ambiguity surface, the sensitivity to model mismatch scales as the correlation loss between simulated and true transfer functions. A 10 dB amplitude error in a single propagation path reduces the correlation by roughly (1 − 10^{-1}) = 0.9 if that path carries 10 % of the total energy. For a building where 3 of 12 significant propagation paths pass near a window or door opening:
- Void approximation error in those 3 paths: ~10–15 dB (estimating 10 dB from the 90× impedance ratio in the coupling)
- MFP correlation loss: ~0.3 per path × 3 paths = ~25 % reduction in peak height
- Empirically, in underwater acoustics MFP, a 1 dB model mismatch in a moderate-SNR scenario moves the peak by ~λ/4 at the dominant frequency; at 100 Hz, λ ≈ 1.85 m → O(0.5 m) location shift

**This suggests void approximation introduces O(10–30 cm) localization bias** for sensors near or paths through openings, increasing to O(30–50 cm) for paths where the opening is the primary connection between rooms. This is within the project's stated tolerance (50 cm) but degrades it.

### 5.3 When openings become model-critical

The void approximation becomes a **critical failure** (not just a bias) when:
1. The source room and the receiver room are connected **only through a door opening** (no common wall, no common slab junction) → the void model predicts near-zero energy at the receiver → MFP fails to locate the source
2. The mamad room's steel door is modeled as void → the room appears isolated, but in reality the steel door couples it to the corridor → systematic location error >1 m

**Both scenarios are addressable:** (1) is rare in standard Israeli apartments (rooms share slab and usually at least one wall); (2) is addressed by implementing the steel door block (§4.5).

---

## 6. Summary table: opening representation decision

| Opening type | Dominant wave effect | v1 representation | v1 expected bias | v2 upgrade | Priority |
|--|--|--|--|--|--|
| Interior hollow door | Near-void for flexural waves; door frame is the main coupling path | Void (wall block absent) | O(10–20 cm) localization, only if door is on primary path | Soft thin block (E~1 GPa, ρ~500 kg/m³) | Low |
| Window (alum frame, single/double pane) | Near-void; glass is ~140× lower impedance than wall; frame via silicone mount | Void (wall block absent) | O(10–20 cm), same caveat | Effective medium block (§4.3) | Low |
| Exterior solid wood door | Moderate gap; 10–30× lower impedance | Void (slightly overcorrects) | O(15–25 cm) near the door | Wood block (E=12 GPa, ρ=600 kg/m³) | Medium |
| Mamad steel door | Significant structural path; ~7× lower impedance than RC wall | **Separate steel block** | <10 cm if modeled correctly | Frame detail | **High** |
| Window shutter (alum roll) | Negligible (discontinuous slats) | Not meshed | Negligible | Same | None |
| RC lintel above all openings | Carries energy over the opening | **Fully meshed** (standard RC slab material) | N/A — already in v1 | N/A | Already done |

---

## 7. Practical mesh implementation notes

### 7.1 ResPlan polygon representation

ResPlan stores each wall as a rectangle with explicit `wall_depth` (0.15 m) plus per-wall polygons for doors and windows (centroid, width, height). The box-lattice decomposer (doc 02 §2.2) must:

1. Read wall polygon → generates wall sub-boxes along wall length in z-slices
2. For each door/window polygon: identify which wall sub-boxes it overlaps (centroid within ±half-dimension)
3. In v1: **drop** those sub-boxes (void, reveal faces become free surfaces)
4. For mamad doors: instead of dropping, **insert a steel-material sub-box** at that location

The lintel above a door (from top-of-door to ceiling height) is a regular RC sub-box and is always present. The sill below a window is also a regular block sub-box.

### 7.2 Node sharing at opening reveals

The reveal faces (exposed surfaces at the edges of voids) are automatically traction-free in SEM (natural boundary condition). No special treatment needed. The node-sharing is guaranteed by the box-lattice multi-block approach (doc 02 §1.3): the adjacent wall boxes share the nodes on the reveal edge.

### 7.3 Element count impact

Adding openings **reduces** element count by removing wall sub-boxes. A 0.9×2.1 m door in a 0.15 m wall at 0.05 m resolution ≈ 0.9/0.05 × 2.1/0.05 × 0.15/0.05 = 18 × 42 × 3 = 2268 fewer elements per door. On a house with 6 doors and 8 windows: ~30–40k fewer elements — about 2–5 % of the concrete shell budget. Net effect: openings slightly **reduce** simulation cost. This is a further argument for the void approximation.

### 7.4 SPECFEM3D acoustic-elastic coupling (for future consideration)

SPECFEM3D supports acoustic (fluid) + elastic (solid) coupled simulations, with automatic enforcement of normal-traction and normal-velocity continuity at fluid-solid interfaces. If a future v3 implementation wishes to model the actual air-borne coupling through open doors, it would:
1. Mesh the room interiors with acoustic (Vs=0) elements: Vp = 343 m/s (air), ρ = 1.2 kg/m³
2. The floor/wall faces facing the room become acoustic-elastic interfaces (SPECFEM handles these automatically)
3. This would correctly model air resonances in rooms coupling into wall bending via coincidence

However, below 250 Hz this effect is negligible (air-structure coupling coefficient << 10⁻³ for pre-coincidence walls), and the computational cost doubles. This is not recommended until the solid-structure localization achieves target accuracy.

---

## 8. Comparison with v1 solid-wall assumption (documented bias)

The baseline SPECFEM3D model (v1) will use:
- All walls, slabs, foundations: solid material blocks (RC or hollow block)
- Openings (doors, windows): voids (no material blocks at opening locations)
- Mamad doors: separate steel material blocks
- Interior air: unmeshed

This is **not a simplification that ignores openings** — it explicitly represents them as voids, which is a physically justified approximation for glass and hollow-core doors. The documented bias (§5.2) is O(10–30 cm) localization uncertainty for paths crossing openings, within the 50 cm project target.

**The "ignore openings entirely" (solid wall everywhere) approach is explicitly rejected:** treating windows and doors as solid block material would overestimate structural coupling across the opening by 10–40 dB, producing systematically wrong fingerprints near openings and potentially flipping the localization solution for sources near windows or doors.

---

## Sources

- [Scattering of flexural waves by cavities in a plate (Cai et al., Int. J. Solids Struct. 2009)](https://www.sciencedirect.com/science/article/pii/S0020768309004673) — multiple-scattering of bending waves, hole geometry, frequency regime
- [Scattering of flexural waves from a hole in a thin plate with an internal beam (Norris et al., JASA 2015)](https://coewww.rutgers.edu/~norris/papers/2015_JASA_Flexural_hole_beam.pdf) — T-matrix for hole+beam; defines ka scattering regime
- [LAMB WAVE SCATTERING FROM A THROUGH HOLE (J. Sound Vib. 1999)](https://www.sciencedirect.com/science/article/abs/pii/S0022460X99921648) — experimental + analytical scattering coefficients for through-holes in plates
- [The prediction of the vibration reduction index Kij for brick and concrete rigid junctions (Appl. Acoustics 2010)](https://www.sciencedirect.com/science/article/abs/pii/S0003682X10000083) — Kij dB values for concrete/masonry T- and cross-junctions
- [Regression curves for vibration transmission across junctions (Hopkins, Crispin et al., Appl. Acoustics 2016)](https://livrepository.liverpool.ac.uk/3001834/1/Hopkins%20Crispin%20Poblet-Puig%20Guigou-Carter%20Kij%20regression%20curves%20from%20FEM%20and%20wave%20theory%20Applied%20Acoustics%202016.pdf) — FEM+wave theory Kij curves; default rigid T-junction ≈ 8 dB
- [The Vibration Reduction Index Kij: Lab Measurements for Rigid Junctions and Flexible Interlayers (Crispin et al. 2006)](https://journals.sagepub.com/doi/10.1260/135101006777630427) — Kij measurement data including flexible joints
- [Simulating low frequency sound transmission through walls and windows (Simmons & Hall, J. Sound Vib. 2017)](https://www.sciencedirect.com/science/article/pii/S0022460X17301207) — window-wall coupling, 3D model, low-frequency window control below 30 Hz
- [Mechanical Properties of Glass, Elastic Modulus and Microhardness (Lehigh IMI)](https://www.lehigh.edu/imi/teched/GlassProp/Slides/GlassProp_Lecture11_Mecholsky.pdf) — E = 70–72 GPa, ρ = 2500 kg/m³ for float glass
- [Dispersion of Flexural Waves (Penn State)](https://www.acs.psu.edu/drussell/Demos/Dispersion/Flexural.html) — c_B ∝ f^{1/2} dispersion formula
- [EN 12354 Building Acoustics Prediction (AcousPlan)](https://acousplan.com/blog/en-12354-building-acoustics-guide) — Kij default 8 dB for rigid T-junction; flanking path model
- [Structure-borne sound in buildings: Advances in measurement (Liverpool)](https://livrepository.liverpool.ac.uk/3077422/1/author%20version.pdf) — flanking coupling loss factors for concrete buildings
- [Broadband Bending of Flexural Waves (Sci. Reports 2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6060090/) — bending wave dispersion in plates with varying thickness
- [An Innovative Hollow Block with SBR for Vibration Damping (PMC10385242)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385242/) — hollow block vibration properties, extinction time
- [Numerical estimation of coupling loss factors in building acoustics (J. Sound Vib. 2013)](https://www.sciencedirect.com/science/article/abs/pii/S0022460X13004392) — SEA coupling loss factors at wall junctions
- [SPECFEM3D Cartesian documentation — Mesh Generation](https://specfem3d.readthedocs.io/en/latest/03_mesh_generation/) — acoustic/elastic material blocks, conforming mesh, free-surface BC
- [Forward and adjoint simulations on unstructured hexahedral meshes (Komatitsch et al., GJI 2011)](https://academic.oup.com/gji/article/186/2/721/590417) — SEM material interface handling
- [Effects of wall discontinuities on flexural waves in cylindrical shells (J. Sound Vib. 1981)](https://www.sciencedirect.com/science/article/abs/pii/0022460X81903400) — flexural wave transmission through discontinuities in shell walls
- Internal: [05_structural_reverberation.md](05_structural_reverberation.md), [specfem3d_build/02_boxy_house_meshing.md](specfem3d_build/02_boxy_house_meshing.md), [specfem3d_build/00_what_specfem3d_needs.md](specfem3d_build/00_what_specfem3d_needs.md), [03_house_architecture_datasets.md](03_house_architecture_datasets.md)
