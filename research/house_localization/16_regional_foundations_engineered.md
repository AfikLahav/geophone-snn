> **Research document — house-localization pivot ("wallhacks").** Deep web research synthesis (Claude Sonnet 4.6, 2026-07-09). Covers engineered (code-compliant) residential foundation practice in Israel and Lebanon — dimensions, materials, soil context — as design input for SPECFEM3D hex-mesh parametric archetypes. Not a validated experiment. Builds on [15_foundation_and_domain_buffer.md](15_foundation_and_domain_buffer.md) (coupling physics + raft recommendation), [03_house_architecture_datasets.md](03_house_architecture_datasets.md) (Israeli construction defaults), [05_structural_reverberation.md](05_structural_reverberation.md). Indexed in [README.md](README.md).

---

# Engineered Residential Foundations in Israel and Lebanon: Regional Practice for SPECFEM3D Mesh Design

**Scope.** Two countries, three soil regimes each, 1–4 storey RC-frame residential. Deliverable: 2–3 parametric foundation archetypes (dimensions + material assignment) for the hex mesh generator, cited and honest about where hard numbers are available vs. inferred from codes + general practice.

---

## 1. Israel: Seismic Code Context

### 1.1 IS 413 (SI 413) — the governing seismic standard

Israeli Standard **SI 413** (first mandatory 1980, revised 1994, current version 2009: SI 413:2009) defines reference peak ground acceleration (PGA) and spectral shapes for all structural design. Key parameters for the coastal plain and typical residential sites:

- **PGA range across Israel:** 0.06 g – 0.28 g at 475-year return period (10 % probability of exceedance in 50 years), depending on distance from the Dead Sea Transform Fault (DSF) and local geology. [[DLUBAL SI 413 tool]](https://www.dlubal.com/en/load-zones-for-snow-wind-earthquake/seismic-si-413.html)
- **Coastal plain (Tel Aviv, Haifa, Ashdod):** PGA ≈ 0.10–0.15 g (relatively distant from the DSF).
- **Jordan Rift / Dead Sea region (Jericho, Beit She'an):** PGA ≈ 0.20–0.28 g — highest seismic exposure in Israel.
- **Northern Galilee:** PGA ≈ 0.15–0.20 g (Sea of Galilee proximity to DSF).
- **Soil site classes:** SI 413 adopts the NEHRP five-class system (A–E) using V_{s,30} — a direct copy from NEHRP 1997 per Israeli research. [[Questioning NEHRP applicability in Israel, SCIRP 2012]](https://www.scirp.org/journal/paperinformation?paperid=21662) Fa and Fv amplification factors are specified per site class. However, researchers note that V_{s,30} alone is unreliable in Israel due to complex layered geology (kurkar / Hamra alternation); site-specific investigations are recommended. Hamra (red sandy loam) is laterally very variable even within a single site.

**Concrete code IS 466** (Israeli concrete code, analogous to Eurocode 2 / ACI 318) governs material grades and detailing. It is referenced by IS 413 for all RC structural elements. [[IS 466-1 Scribd reference]](https://www.scribd.com/doc/163429335/IS-466-1) The minimum concrete grade for structural RC elements under IS 466 in current practice is **C20** (fck = 20 MPa cylinder), with C25 typical for new construction. Older Israeli buildings (1970s–1990s vintage) commonly used C20 concrete with plain or deformed rebars at 260–280 MPa yield. [[Israeli seismic retrofitting study, PMC8837140]](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837140/) New construction since ~2005 tends to C25–C30 for structural elements.

### 1.2 Israeli Soil Types and Their Foundation Implications

| Soil type | Location | Vs30 approx. | Site class (NEHRP) | Allowable bearing pressure | Foundation implication |
|---|---|---|---|---|---|
| **Hamra** (red sandy loam, Quaternary) | Coastal plain, Sharon, Tel Aviv to Haifa belt | 150–300 m/s | C–D | 100–200 kPa | Pad + grade beam standard; raft on soft Hamra; slab-on-grade atop |
| **Alluvial clay / Vertisol** | Jezreel Valley, coastal lowland depressions, northern plains | 100–200 m/s | D–E | 80–150 kPa | Raft preferred; piles if very soft; grade beams mandatory for seismic tie |
| **Nari / calcareous crust** (calcrete) | Negev, inland foothills, semi-arid zones | 300–500 m/s | B–C | 150–300 kPa | Shallow pad footings; rock-cut strip footings common in thin Nari |
| **Kurkar** (aeolianite sandstone, coastal ridge) | Coastal ridge adjacent to Hamra zones | 300–500 m/s | B–C | 150–250 kPa | Rock-like bearing; footing at 0.3–0.5 m depth; minimal grade beam |
| **Limestone / Dolomite** (Judean/Carmel highlands) | Jerusalem, Carmel, Galilee hill sites | 500–1500 m/s | A–B | >300 kPa | Very shallow footings; sometimes directly on prepared rock; no frost constraint |
| **Fill / liquefiable coastal sand** | Marina areas, reclaimed land, beach strip | < 150 m/s | E | < 80 kPa | Piles or deep rafts; not typical for residential |

*Hamra Vs30 range and site class from [[GJI Beirut paper]](https://academic.oup.com/gji/article/199/2/894/619969) and [[Israeli coastal plain geology]](https://grokipedia.com/page/Israeli_coastal_plain) (kurkar/Hamra described as Quaternary sandy loam with gravel ridges). Bearing pressures from general geotechnical practice; no direct Israeli standard citation available for qa values.*

**No frost depth in Israel.** Israel's ground never freezes. The single primary driver of foundation depth is bearing — reaching competent soil below loose fill, topsoil, and disturbed near-surface material. The practical minimum embedment is **0.5 m** below finish grade, ensuring the footing soffit rests on undisturbed, vegetable-matter-free bearing soil. On rock/kurkar, footings can be shallower (0.3 m), provided the rock surface is clean and properly prepared.

### 1.3 Foundation Types by Storey Count and Soil — Israel

#### 1.3.1 Isolated (Pad) Footings + Grade Beam (Ring Beam)

**Most common system for 1–3 storey RC column-frame buildings on Hamra / Nari / kurkar.** Mandatory under IS 413 for seismic resistance: the grade beam (tie beam at foundation level) interconnects all column pads and ensures no independent column movement under lateral load.

**Typical dimensions — 1–2 storey RC frame on medium Hamra (qa ≈ 150 kPa):**

| Element | Plan size | Depth/thickness | Soffit depth below finish grade |
|---|---|---|---|
| Isolated pad footing | 0.9 × 0.9 m to 1.2 × 1.2 m | 300–450 mm RC | 0.50–0.80 m |
| Grade beam (perimeter) | 200–300 mm wide × 400–600 mm deep | RC | soffit at 0.60–1.0 m below grade |
| Grade beam (interior, if needed) | 200–250 mm × 400 mm | RC | same as perimeter |
| Ground-floor slab (on-grade) | full footprint | 150–200 mm RC | soffit at ~0 m (on compacted fill) |

*Pad size per general structural engineering practice (M20 concrete, 150 kPa soil, 1–2 storey column loads ≈ 200–600 kN): A_footing = P / qa. For P = 300 kN, qa = 150 kPa → A = 2 m² → ~1.4 × 1.4 m. Thumb-rule tables (CivilSir) give 1.0–1.2 m for 1–2 storey on good-bearing gravel/sand. [[CivilSir footing sizes]](https://civilsir.com/column-footing-size-for-1-2-3-4-and-5-storey-building/) For 3–4 storey: 1.5–1.8 m. These are minimum estimates; actual sizes require geotechnical report.*

**Depth driver:** On Hamra, competent bearing (qa ≥ 120 kPa) is typically encountered at 0.5–0.8 m depth, where the loose upper soil grades into medium-dense Hamra. Israeli practice (no frost) targets the soffit at 0.5–0.6 m for 1–2 storey, with the pad thickness adding another 0.3–0.45 m, so the top of the pad (grade beam junction) is at approximately −0.2 to −0.3 m.

**Grade beam dimensions:** Grade beams in IS 413-compliant design serve as seismic ties and must meet: (a) minimum width = column dimension or wall width, (b) depth providing adequate flexural capacity to span between pads (span = column spacing, typically 4–6 m). For 200 mm column spacing and 5 m span, a 250 × 500 mm grade beam is typical. The grade beam top is usually at or just below finish floor, so it doubles as the retaining element for the slab-on-grade fill. [[Design for Seismic Tie Beams (Zstructures)]](https://www.zstructures.com/2012/09/14/design-for-seismic-tie-beams/)

#### 1.3.2 Raft / Mat Slab

**Common for soft Hamra, clayey alluvial sites, 3+ storeys, and increasingly preferred by contractors for simplicity.** A continuous reinforced slab under the full building footprint; downstand beams (thickened ribs) under walls and columns.

**Typical dimensions:**

| Element | Thickness | Soffit depth below finish grade |
|---|---|---|
| Raft slab (flat zone) | 300–400 mm | 0.40–0.60 m |
| Downstand rib (under walls/columns) | 600–900 mm deep × 300–400 mm wide | 0.70–1.0 m |
| Blinding / lean concrete below raft | 50–100 mm | (below raft soffit) |

*Raft thickness from general practice: [[build-construct.com raft dimensions]](https://build-construct.com/structural-engineering/raft-foundations/). Israeli residential raft: 350–400 mm typical; older builds as thin as 250 mm on stiff soils. Downstand at column lines adds 300–500 mm below raft slab. Blinding: 50–75 mm C15 lean concrete or sand blinding, below raft.*

#### 1.3.3 Strip/Continuous Footings (Under Load-Bearing Walls)

**Older system (pre-1980s) and still used under masonry load-bearing walls in low-rise detached houses (1–2 storey), or where stone / dense kurkar makes strip excavation easier than discrete pads.** Rarely the primary system in post-1994 IS 413-compliant RC column-frame buildings (which need discrete footings + grade beam per the seismic code).

**Typical dimensions:**

| Element | Width | Depth | Soffit depth |
|---|---|---|---|
| Strip footing under bearing wall | 600–900 mm | 300–400 mm | 0.50–0.80 m |
| Reduced strip on rock / kurkar | 400–600 mm | 200–300 mm | 0.20–0.50 m |

*UK and Mediterranean minimum strip footing practice (no frost) gives ≥ 450 mm depth soffit to avoid near-surface seasonal moisture variation. [[Strip Foundations guide]](https://www.reinforcementproductsonline.co.uk/news/strip-foundations/)*

#### 1.3.4 Pile Foundations

Rare in 1–4 storey residential. Used on liquefiable coastal sands (beach-front, reclaimed areas) or very soft clay layers where bearing capacity cannot be achieved at shallow depth. For simulation purposes: **exclude piles from the parametric generator for 1–4 storey residential on typical Israeli sites.** Mention as a limiting edge case for soft fill / liquefiable zone only.

### 1.4 Below-Grade Concrete Extent — Israel Summary

| Foundation type | Total concrete thickness below finish grade | Notes |
|---|---|---|
| Pad + grade beam, 1–2 storey | 0.60–1.0 m (pad soffit) + 0.3–0.45 m pad thickness above | Top of grade beam at ~ −0.20 to −0.30 m; pad soffit at −0.5 to −0.8 m |
| Raft slab (no downstand), 1–2 storey | 0.40–0.60 m (raft soffit) | Uniform slab, full footprint |
| Raft + downstand rib | 0.70–1.0 m (rib soffit) | Ribs under walls/columns; flat zone at 0.40–0.60 m |
| Strip footing | 0.50–0.80 m (soffit) | Under bearing wall lines only |

---

## 2. Lebanon: Seismic Code Context

### 2.1 Lebanese Building Code and Seismic Practice

Lebanon is seismically active and sits astride major branches of the Dead Sea Transform Fault system: the **Yammouneh Fault** (main left-lateral transform), **Rachaya Fault**, **Serghaya Fault**, and the **Mount Lebanon Thrust**. The expected PGA at 475-year return period is **0.20–0.30 g** for most sites within ~20 km of the active faults — substantially higher than coastal Israel. [[ResearchGate Lebanon seismic hazard]](https://www.researchgate.net/publication/226662491_Evaluation_of_the_seismic_hazard_of_Lebanon) [[NHESS Levant fault 2025]](https://nhess.copernicus.org/articles/25/3397/2025/)

**Seismic code history:**
- Before 2005: No mandatory seismic design code. Most Beirut RC buildings (pre-2005) were designed for **gravity loads only**, with no lateral-force-resisting system. This is the single most important fact about Lebanese building practice. [[L'Orient Today — Lebanese buildings and earthquakes]](https://today.lorientlejour.com/article/1329979/what-will-it-take-to-make-lebanon-s-buildings-safe-from-earthquakes.html) [[Lebanon case study PMC6279946]](https://pmc.ncbi.nlm.nih.gov/articles/PMC6279946/)
- 2005: Lebanese Standard **NL 135:2012** ("Protection from Earthquakes: General Rules") adopted by LIBNOR (Lebanese Standards Institution). [[LIBNOR NL 135]](http://www.libnor.gov.lb/CatalogDetails.aspx?id=4152&language=en) References ACI 318-08 and either UBC 97 or IBC 2009 at the designer's choice. French PS 92 is also permitted. Enforcement remains inconsistent, particularly outside Beirut.
- Post-2005 (new construction): RC frames theoretically designed to ACI 318 seismic provisions. Tie beams (grade beams) between isolated footings are required per ACI 318 Section 18.13 for Seismic Design Category C and above. Given Lebanon's PGA ≥ 0.2 g, nearly all sites fall in SDC D–E under ACI 318 criteria.

**Key structural finding:** Over **50 % of Beirut's RC building stock** was built without seismic design requirements and is classified as highly vulnerable. [[Seismic response of Beirut buildings, Springer]](https://link.springer.com/article/10.1007/s10518-016-9920-9) This split — pre-2005 gravity-only vs. post-2005 seismic-designed — is the dominant partition for any parameterization.

### 2.2 Beirut and Coastal Lebanon Soil Profile

A GJI study on the Beirut River alluvial plain provides the most detailed geotechnical profile: [[Shear wave velocity structure of Beirut, GJI 2014]](https://academic.oup.com/gji/article/199/2/894/619969)

| Layer | Description | Thickness | Vs (m/s) | Vp (m/s) | Site class |
|---|---|---|---|---|---|
| Gravelly upper alluvium | Sandy gravel with pebbles | 4–14 m | 300–400 | 600–800 | C |
| Soft clay lens | CL / SC (low-plasticity clay) | 2–12 m, laterally discontinuous | 150–200 | 400–600 | D–E locally |
| Lower sand / weathered limestone | Sandy deposits or marly limestone | variable | 300–500 | 700–1000 | C |
| Sound limestone bedrock | Tertiary marly limestone | — | > 500 | > 1500 | A–B |

Depth to bedrock: **14–60 m** depending on location. Vs30 for most Beirut alluvial sites: **200–250 m/s** → Site Class D (stiff soil per IBC/NEHRP). The soft clay lens, when present, creates a local Site Class E zone.

**Coastal Lebanon mountain sites** (village construction, e.g., Mount Lebanon): Tertiary to Cretaceous limestone at or near the surface. Very shallow bedrock (0–2 m) at many village sites. Foundation practice shifts entirely to rock-bearing shallow footings.

**Bearing capacity for Beirut alluvium:** Fine sand/gravel at 0.5–1.5 m depth: qa ≈ 150–200 kPa (SPT N = 11 blows at 0–3 m depth in the case study [[PMC6279946]](https://pmc.ncbi.nlm.nih.gov/articles/PMC6279946/), N = 21 at 4–10 m; angle of friction 33–35°, bearing capacity 3 kg/cm² = 294 kPa at tested depth).

### 2.3 Foundation Types — Lebanon

#### 2.3.1 Pre-2005 Lebanese Construction (Gravity-Load Only Design)

The typical Beirut RC building constructed pre-2005 (and still the majority of the building stock) has:

- **Isolated (spread) footings** under individual columns — no grade beams or tie beams connecting them, because lateral-force design was not required.
- **Footing dimensions (case study data):** One documented Beirut building has a **230 × 230 cm spread footing at a test pit depth of ~70 cm** (i.e., 2.3 × 2.3 m plan, approximately 0.7 m embedment depth). [[Lebanon building case study, PMC6279946]](https://pmc.ncbi.nlm.nih.gov/articles/PMC6279946/) Concrete grade: fck ≈ 20 MPa (C20) conservatively; rebar: mild steel plain bars, fy ≈ 260 MPa. This is consistent with typical pre-code RC practice: relatively large plan footing, moderate depth, minimal seismic detailing.
- **Column sizes:** 120 × 30 cm, 60 × 60 cm, and 30 × 60 cm columns documented in the same case study — typical of Mediterranean gravity-dominated RC frames.
- **Ground floor:** Ribbed slab or flat slab, 300 mm total depth (including ribs).
- **No tie beams:** Isolated footings are independent; no perimeter ring beam.

#### 2.3.2 Post-2005 Lebanese Construction (ACI 318 Seismic)

Under NL 135:2012 + ACI 318-08, a post-2005 Beirut building in SDC D requires:
- **Isolated footings + tie beams** (grade beams) per ACI 318 Section 18.13.
- Tie beam minimum cross-section per ACI 318 §18.13.3.2: smallest dimension ≥ (clear column spacing) / 20, ≥ 300 mm; and the tie must be able to develop in tension or compression = 10 % of the larger column axial force.
- **Tie beam (grade beam) dimensions:** Practical minimum for a 4–5 m column spacing: 300 × 500 mm (width × depth), similar to Israeli practice. Top of grade beam near finish grade.
- **Footing depth:** ACI 318 §18.13 does not explicitly mandate a minimum embedment depth beyond general bearing requirements. For Beirut alluvial sand/gravel with qa ≈ 150–200 kPa achievable at 0.8–1.0 m, the design embedment is typically 0.8–1.2 m to the footing soffit.
- **Concrete grade:** ACI 318 §26.4.2 minimum f'c = 17 MPa (2,500 psi) for structural members; in practice, post-2005 Lebanese RC uses f'c = 25–30 MPa (C25–C30), with new construction often specifying 30 MPa for seismic members.

#### 2.3.3 Mountain Village Construction (Lebanon)

Lebanese mountain village residential (typical 1–2 storey stone / RC frame on Jurassic–Cretaceous limestone):
- **Foundation type:** Short rock-bearing strip footing or pad footing directly on the limestone surface or on very thin (<0.3 m) soil overburden.
- **Depth:** 0.2–0.5 m below grade; sometimes a clean-cut rock face with 50–100 mm blinding concrete and the RC footing poured directly on rock.
- **Width:** 400–600 mm for strip under wall; 600–800 mm for pad under column.
- **Grade beam:** Often absent in traditional construction; a perimeter stem wall (150–200 mm stone or concrete) doubles as the foundation wall.
- **Concrete grade:** C20–C25 in new builds; older stone construction uses mortared local limestone with no RC footing at all.
- **Seismic implication:** Rock-bearing foundations (direct on Vs > 600 m/s limestone) have extremely low coupling loss — the structure–soil impedance contrast is smaller (concrete Zc vs. rock Z_rock both high), so energy transfer is more efficient in both directions. The foundation impedance boundary is fundamentally different from alluvial-soil sites.

### 2.4 Comparison: Israel vs. Lebanon

| Parameter | Israel (coastal/typical) | Lebanon (Beirut alluvial) | Lebanon (mountain) |
|---|---|---|---|
| Seismic code mandatory since | 1980 (IS 413) | 2005 (NL 135) | 2005 (NL 135), rarely enforced |
| Dominant foundation system | Pad + grade beam (RC frame); raft on soft soil | Pre-2005: isolated pads, no tie beams; post-2005: pads + tie beams | Rock-bearing strip/pad; no grade beam |
| Footing plan size (1–2 storey) | 0.9–1.2 m | 1.5–2.3 m (pre-2005, oversized for gravity) | 0.4–0.8 m |
| Footing depth (soffit, 1–2 storey) | 0.5–0.8 m | 0.6–1.0 m | 0.2–0.5 m |
| Grade beam present? | Yes (post-1980) | Pre-2005: No; Post-2005: Yes | Rarely |
| Concrete grade | C20 (old) / C25–C30 (new) | C20 (old) / C25–C30 (new) | C20–C25 |
| Raft common? | Yes (3+ storey; soft Hamra) | Less common; isolated footings preferred | Very rare |
| PGA design level | 0.06–0.28 g | 0.20–0.30 g | 0.20–0.30 g |

---

## 3. Concrete Material Properties for Simulation

### 3.1 Structural RC Foundation Concrete (Grades Used in Region)

| Concrete grade | fck (MPa, cylinder) | E (MPa) | ν | ρ (kg/m³) | Vp (m/s) | Vs (m/s) | Z_P (MRayl) |
|---|---|---|---|---|---|---|---|
| C20 (old Israeli/Lebanese) | 20 | 29,000 | 0.20 | 2,350 | 3,510 | 2,100 | 8.2 |
| **C25 (current standard)** | **25** | **31,000** | **0.20** | **2,400** | **3,590** | **2,250** | **8.6** |
| C30 (new/seismic members) | 30 | 33,000 | 0.20 | 2,400 | 3,700 | 2,330 | 8.9 |

*Vp and Vs derived from elastic relations: Vp = √[E(1−ν) / (ρ(1+ν)(1−2ν))], Vs = √[E / (2ρ(1+ν))]. For C25, E = 31 GPa, ν = 0.20, ρ = 2400 kg/m³: Vp = √[31×10⁹ × 0.8 / (2400 × 1.2 × 0.6)] = √[24.8×10⁹ / 1728] ≈ 3,788 m/s. Correcting for reinforcement and crack microstructure (reduces effective E ~5–10 %): Vp ≈ 3,500–3,800 m/s. Measured range for sound structural RC: 3,500–4,500 m/s for Vp, 2,000–2,500 m/s for Vs. [[PMC: P,S,R wave velocities in RC slabs]](https://www.hindawi.com/journals/amse/2016/1548215/) [[ConcCalc tool example: E=30 GPa, ρ=2400 → Vp=3543, Vs=2271]](https://www.calculatorultra.com/en/tool/concrete-wave-speed-calculation-p-wave-s-wave-velocities.html)*

**Recommended simulation values for C25 structural RC (foundations, slabs, columns):**
- Vp = 3,800 m/s, Vs = 2,200 m/s, ρ = 2,400 kg/m³
- Q_p = 100, Q_s = 50 (concrete internal attenuation; consistent with doc 15)

**Blinding / lean concrete (C10–C15, 50–100 mm layer below raft):**
- Not structurally significant for wave propagation; in the mesh, assimilate into the soil layer or treat as C25 (the 50–100 mm thickness is sub-element at typical mesh resolution).

### 3.2 Reinforcement

Israeli and Lebanese RC foundations use deformed rebar (ribbed bar):
- Old stock (pre-1990s Israel, pre-2005 Lebanon): plain round bars, fy ≈ 260 MPa (mild steel), diameter 12–20 mm.
- New construction: deformed high-yield bars, fy ≈ 500 MPa (Israeli) / 420–500 MPa (Lebanese ACI 318 Grade 60).
- **Simulation note:** Reinforcement is not explicitly modeled in the SPECFEM3D hex mesh. It is implicitly captured in the effective concrete properties (density increase ~1–2 %; negligible effect on Vp/Vs at the volume fractions used). Standard concrete properties as above are appropriate.

---

## 4. 3-D Mesh Parametrization — Engineered Foundation Archetypes

For a 14 × 7 m plan house (as in doc 15), three parametric archetypes cover the engineered-foundation design space for Israel and Lebanon. All are axis-aligned rectangular concrete blocks — trivially representable in the all-hex mesh.

---

### Archetype A — Raft Slab (Dominant for Soft Soil, Multi-Room Israeli / Post-2005 Lebanese)

**Applies to:** Israeli coastal Hamra sites; alluvial clay / Jezreel Valley; multi-storey (3+) anywhere; any site where isolated footings would cover > 50 % of footprint; post-2005 Lebanese coastal RC.

```
Concrete zone (below ground-floor slab):
  Plan:        full building footprint, e.g., 14.0 × 7.0 m
  Layer B:     raft slab body
               thickness = 0.35 m (350 mm)
               top:    z = −0.20 m  (bonded to ground-floor slab underside)
               bottom: z = −0.55 m
  Layer C:     downstand ribs (optional; under all wall/column lines)
               width = 0.30–0.40 m (match wall width)
               additional depth = 0.25 m below Layer B
               bottom: z = −0.80 m
               ← only at exterior walls + interior load-bearing walls; soil elsewhere

Material (both layers): C25 concrete
  Vp = 3800 m/s, Vs = 2200 m/s, ρ = 2400 kg/m³, Qp = 100, Qs = 50

Below z = −0.55 m (no downstand) or z = −0.80 m (with downstand):  soil

Blinding: 50 mm lean concrete below raft; assimilate into soil block or C25
           (sub-element scale at mesh resolution h ≥ 0.05 m)

Mesh note: Layer B adds 7 element layers at h = 0.05 m (concrete zone).
           Layer C (downstand) adds local geometry under wall columns; implement
           as wall-width-wide concrete columns extending to z = −0.80 m.
           Total new hex elements: ~200,000–300,000 for full footprint.
```

**Seismic coupling note:** This is the full-footprint radiator. The raft underside (at z = −0.55 m) creates a 14 × 7 = 98 m² continuous concrete-soil interface — the strongest possible soil-path coupling (minimum coupling loss = 2–5 dB). Exterior geophones receive maximum structure-to-soil energy via this interface.

---

### Archetype B — Pad Footings + Perimeter Grade Beam (Dominant New Israeli RC Frame; Post-2005 Lebanese)

**Applies to:** Israeli 1–3 storey RC frame on Hamra / Nari (post-1994 IS 413); Lebanese post-2005 RC frame on Beirut alluvium or coastal plain; any site where individual column loads are moderate and soil qa ≥ 120 kPa at 0.5–0.8 m.

```
Geometry:
  A. Perimeter grade beam (ring beam):
     Width:      0.25 m (matches 200–250 mm exterior wall)
     Depth:      0.50 m
     Top:        z = −0.20 m (just below floor slab soffit)
     Bottom:     z = −0.70 m
     Plan:       four exterior wall lines only (L-shaped perimeter in plan)

  B. Pad footings under each column:
     Plan:       1.0 × 1.0 m  (1–2 storey, qa=150 kPa, column load ~200 kN)
                 or 1.2 × 1.2 m  (2–3 storey, column load ~400 kN)
     Thickness:  0.40 m
     Top:        z = −0.35 m  (tied into grade beam)
     Bottom:     z = −0.75 m  (soffit; within the grade beam depth zone)
     Position:   at each column location (typically 4–5 m column spacing
                 → 6–12 pads for a 14 × 7 m plan)

  C. Ground-floor slab (on-grade, already in mesh):
     Thickness:  0.18 m
     z:          0.00 m to −0.18 m
     Material:   concrete (same as above)
     Sits on compacted fill inside the grade beam ring; no structural function.

Material: C25 concrete (same as Archetype A)
          Vp = 3800, Vs = 2200, ρ = 2400, Qp = 100, Qs = 50

Between column pads (interior), below ground-floor slab:  soil
The grade beam line is concrete; interior is soil to z = 0 (surface grade).

Mesh note: Grade beam = thin concrete strip along four perimeter lines.
           Pad footings = discrete 1.0 × 1.0 × 0.40 m concrete blocks
           at column grid intersections.
           Interior soil elements reach from z = −0.18 m down through the
           domain; only the perimeter has concrete below the slab.
```

**Seismic coupling note:** The concrete-soil interface area is much smaller than Archetype A. Perimeter grade beam underside area ≈ 2(14 + 7) × 0.25 = 10.5 m². Pad footing undersides ≈ 9 pads × 1 m² = 9 m². Total ≈ 20 m² vs. 98 m² for the raft. Soil-path radiation is roughly 5× weaker; structure retains more energy → longer reverberant field in the building. Grade beam–pad system is more sensitive to geophone placement (on concrete vs. on soil).

---

### Archetype C — Rock-Bearing Shallow Strip Footing (Mountain Sites; Israeli Nari/Limestone)

**Applies to:** Israeli hill/mountain sites (Jerusalem environs, Carmel, Galilee highlands) on Nari calcrete or limestone outcrop; Lebanese mountain village construction (Jurassic–Cretaceous limestone at surface); any site where bedrock is at < 0.5 m depth.

```
Geometry:
  A. Strip footing (under all load-bearing walls and exterior frame lines):
     Width:      0.50–0.60 m (wider than wall above)
     Thickness:  0.25–0.30 m (minimum structural depth for RC strip)
     Top:        z = −0.10 m  (very shallow; near-surface bearing)
     Bottom:     z = −0.35 to −0.40 m
     Plan:       continuous strip under all four walls + any internal bearing wall

  B. No grade beam separate from the footing (the strip IS the grade beam).
  C. Ground-floor slab:
     same as above; poured directly on rock or on 50 mm sand blinding atop rock.

  D. Rock bearing layer:
     Below z = −0.35 m: rock material (limestone)
     Vs_rock = 600–1500 m/s, Vp_rock = 1500–4000 m/s, ρ_rock = 2500 kg/m³
     (Use Vs = 800, Vp = 2000 as representative limestone weathered surface)

Material:
  Footing concrete:  C20–C25, same Vp/Vs as above
  Rock:              separate material block below footing soffit
                     Vs = 800 m/s, Vp = 2000 m/s, ρ = 2500 kg/m³, Q = 200

Mesh note: Strip footings are concrete lines (0.55 × 0.28 m section) along
           all wall lines, extending 0.25 m below the wall base.
           Rock layer replaces the soil block below z = −0.40 m across the
           entire domain (or zoned rock under the building footprint with
           soil in the surrounding buffer — site-specific).
```

**Seismic coupling note:** The concrete–rock impedance contrast is much lower than concrete–soil. Z_concrete / Z_rock ≈ 8.6 / 5.0 = 1.7 (for Vs_rock = 800 m/s). Reflection coefficient R = ((1.7−1)/(1.7+1))² ≈ 0.07. Only 7 % of incident wave energy reflects at the concrete-rock interface → **most energy transmits into rock** → rapid diffusion into the rock halfspace → exterior geophone signal on rock is dominated by body waves, not Rayleigh. Very different Green's function character vs. Archetypes A and B.

---

### Archetype D (Optional) — Pre-2005 Lebanese Isolated Footings, No Tie Beams

**Applies to:** Pre-seismic-code Lebanese buildings (majority of Beirut stock). Same geometry as Archetype B but without the perimeter grade beam. Use for simulating the large vulnerable building population that dominates signal in any Lebanese urban survey.

```
Geometry:
  Isolated pads under each column:
    Plan:      1.5–2.3 m × 1.5–2.3 m (larger than Israeli; sized for gravity
               loads without the grade beam to redistribute)
    Thickness: 0.50–0.70 m (thicker; pre-code conservative sizing)
    Soffit:    z = −0.70 to −1.0 m (deeper; Beirut alluvium requires
               reaching N > 15 SPT layer at 0.8–1.0 m depth)
    No perimeter grade beam.
    Interior soil between pads.
  Ground-floor slab:
    Same; 180–200 mm, sits on fill between columns.

Material: C20 concrete (pre-2005)
  Vp = 3500, Vs = 2100, ρ = 2350, Qp = 80, Qs = 40
  (Slightly lower Vp/Vs to reflect older, potentially lower-quality concrete)
```

**Seismic coupling note:** Even larger footprint pads than Israeli pads → comparable total concrete-soil interface area to Archetype B (no grade beam but larger pads). The absence of a perimeter concrete ring (no grade beam) means the lateral wave path from interior footstep → foundation boundary has to traverse soil between columns rather than concrete — significantly higher path attenuation for structure-borne energy reaching exterior sensors.

---

## 5. Mesh Generator Sampling Guide

For a procedural generator randomizing over the Israel–Lebanon residential parameter space:

| Parameter | Archetype A (Raft) | Archetype B (Pad+Grade Beam) | Archetype C (Rock Strip) | Archetype D (Pre-code pads) |
|---|---|---|---|---|
| Soil type | Hamra, alluvial clay, soft | Hamra, Nari, medium | Limestone, Nari outcrop | Beirut alluvial / coastal |
| Country | Israel (coast, valley) | Israel, post-2005 Lebanon | Israel (hills), Lebanon (mountain) | Lebanon (pre-2005) |
| Raft slab thickness | 0.35 m | — | — | — |
| Downstand rib depth | 0.25 m optional | — | — | — |
| Raft soffit depth | 0.55–0.80 m | — | — | — |
| Grade beam width | — | 0.25 m | — (strip = grade beam) | — |
| Grade beam depth | — | 0.50 m | — | — |
| Grade beam soffit depth | — | 0.70 m | — | — |
| Pad plan size | — | 1.0–1.2 m | — | 1.5–2.3 m |
| Pad thickness | — | 0.40 m | — | 0.50–0.70 m |
| Pad soffit depth | — | 0.75 m | — | 0.70–1.0 m |
| Strip footing width | — | — | 0.50–0.60 m | — |
| Strip footing thickness | — | — | 0.28 m | — |
| Strip soffit depth | — | — | 0.35–0.40 m | — |
| Below-foundation medium | Soil (Vs=150–300 m/s) | Soil | Rock (Vs=600–1500 m/s) | Soil (Vs=200–350 m/s) |
| Concrete Vp / Vs | 3800 / 2200 | 3800 / 2200 | 3600 / 2100 | 3500 / 2100 |
| Concrete ρ | 2400 | 2400 | 2400 | 2350 |
| Concrete Qp / Qs | 100 / 50 | 100 / 50 | 100 / 50 | 80 / 40 |

**For the simulation default (Israel, coastal, 1–2 storey):** Use Archetype A (raft) with 350 mm slab, soffit at z = −0.55 m. This is the recommendation from doc 15 and it stands: it correctly represents the full-footprint impedance boundary, is geometrically trivial to mesh, and is the most common system for multi-room Israeli residential buildings on Hamra. Archetype B is the second-priority variant to generate (more common in 1–2 storey single-column-frame buildings on stiffer soil; harder to mesh but captures the grade-beam ring-beam physics correctly).

---

## 6. Key Uncertainties and Caveats

1. **IS 466 full text not publicly accessible.** The minimum concrete grade requirements for Israeli foundation elements per IS 466 are inferred from: (a) the Israeli seismic retrofitting study showing C20 in 1970s–1980s buildings, (b) general regional practice and regional comparability with Eurocode 2 (minimum C20 for foundations). The actual IS 466:2003 specification may differ from the Eurocode 2 / ACI 318 cited values. [[IS 466-1 scribd (paywalled)]](https://www.scribd.com/doc/163429335/IS-466-1)

2. **Lebanese foundation dimensions come from a single case study.** The 2.3 × 2.3 m spread footing [[PMC6279946]](https://pmc.ncbi.nlm.nih.gov/articles/PMC6279946/) is one documented building — a 6-storey structure, not the 1–4 storey range targeted here. For a 2-storey Lebanese gravity-designed building, the footing would be proportionally smaller (≈ 1.5 × 1.5 m). Use the stated range (1.5–2.3 m) as the realistic bracket.

3. **Beirut soil Vs profile from one study.** The GJI (2014) data covers the Beirut River alluvial plain — representative for the central coastal districts but not the southern suburbs (different geology) or the Achrafieh / Ashrafieh ridge (shallower rock).

4. **No frost depth anywhere in the region.** This is confirmed; no Israeli or Lebanese code specifies a frost-protection depth. The 0.5 m minimum depth is a soil-quality / bearing-capacity requirement, not a thermal requirement.

5. **Hamra bearing capacity.** The 100–200 kPa range for Hamra is inferred from general Mediterranean sandy loam practice and Israeli geotechnical engineering literature (the SCIRP paper on SI 413 and general geotechnical references). No direct Israeli standard value is cited because Israeli bearing capacity standards are not available in English-language web sources. A site-specific geotechnical investigation is always required per IS 413 for all but the simplest structures.

6. **Code enforcement in Lebanon.** The NL 135:2012 foundation requirements are design-code provisions. Actual enforcement in rural mountain villages and areas outside Beirut is reportedly minimal. The practical foundation type for Lebanese mountain village construction is empirical local practice (mason-built shallow strip on rock), not code-compliant RC detailing.

---

## Sources

- DLUBAL SI 413:2009 seismic load zones — Israel PGA map and zone tool: [https://www.dlubal.com/en/load-zones-for-snow-wind-earthquake/seismic-si-413.html](https://www.dlubal.com/en/load-zones-for-snow-wind-earthquake/seismic-si-413.html)
- SCIRP: Questioning applicability of NEHRP site coefficients in Israel (SI 413 soil classes A–E, Fa/Fv, Vs30): [https://www.scirp.org/journal/paperinformation?paperid=21662](https://www.scirp.org/journal/paperinformation?paperid=21662)
- IISEE: Israeli seismic code IS 413 overview (mandatory 1980, Cd, ductility classes): [https://iisee.kenken.go.jp/net/seismic_design_code/israel/israel.htm](https://iisee.kenken.go.jp/net/seismic_design_code/israel/israel.htm)
- Times of Israel — Israeli buildings and earthquake risk: [https://www.timesofisrael.com/israeli-buildings-face-major-earthquake-risk-despite-efforts-to-upgrade-them/](https://www.timesofisrael.com/israeli-buildings-face-major-earthquake-risk-despite-efforts-to-upgrade-them/)
- PMC8837140 — Seismic retrofitting of typical Israeli residential building (C20 concrete, 30×30 cm columns, 2.6 m storey height, 1970s–1980s type): [https://pmc.ncbi.nlm.nih.gov/articles/PMC8837140/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8837140/)
- PMC6279946 — Lebanese building case study (230×230 cm spread footing, C20, rebar fy=260 MPa, 21m tall 6-storey RC frame, Beirut alluvial sand, SPT N=11–21, bearing 3 kg/cm²): [https://pmc.ncbi.nlm.nih.gov/articles/PMC6279946/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6279946/)
- GJI 2014 — Shear wave velocity structure of Beirut alluvial plain (Vs gravel 300–400 m/s, clay 150–200 m/s, bedrock >500 m/s, Vs30 ≈ 200–250 m/s, site class D, bedrock 14–60 m): [https://academic.oup.com/gji/article/199/2/894/619969](https://academic.oup.com/gji/article/199/2/894/619969)
- Springer BEE: Seismic response of Beirut buildings from ambient vibrations — 330 RC buildings, >50% pre-code vulnerable: [https://link.springer.com/article/10.1007/s10518-016-9920-9](https://link.springer.com/article/10.1007/s10518-016-9920-9)
- L'Orient Today — Lebanon building safety and seismic code: [https://today.lorientlejour.com/article/1329979/what-will-it-take-to-make-lebanon-s-buildings-safe-from-earthquakes.html](https://today.lorientlejour.com/article/1329979/what-will-it-take-to-make-lebanon-s-buildings-safe-from-earthquakes.html)
- LIBNOR NL 135:2012 — Lebanese seismic code (ACI 318-08 or IBC 2009 reference codes): [http://www.libnor.gov.lb/CatalogDetails.aspx?id=4152&language=en](http://www.libnor.gov.lb/CatalogDetails.aspx?id=4152&language=en)
- NHESS 2025 — Levant fault system PSHA (PGA 0.20–0.30 g for Lebanon): [https://nhess.copernicus.org/articles/25/3397/2025/](https://nhess.copernicus.org/articles/25/3397/2025/)
- ResearchGate — Seismic hazard evaluation Lebanon (fault systems, PGA map): [https://www.researchgate.net/publication/226662491_Evaluation_of_the_seismic_hazard_of_Lebanon](https://www.researchgate.net/publication/226662491_Evaluation_of_the_seismic_hazard_of_Lebanon)
- Zstructures — Seismic tie beam design (ACI 318 §18.13, grade beam sizing): [https://www.zstructures.com/2012/09/14/design-for-seismic-tie-beams/](https://www.zstructures.com/2012/09/14/design-for-seismic-tie-beams/)
- CivilSir — Column footing sizes by storey count (1–4 storey thumb rule, M20, 1.0–1.8 m plan): [https://civilsir.com/column-footing-size-for-1-2-3-4-and-5-storey-building/](https://civilsir.com/column-footing-size-for-1-2-3-4-and-5-storey-building/)
- CalculatorUltra — Concrete wave speed (Vp=3543, Vs=2271 for E=30 GPa, ρ=2400): [https://www.calculatorultra.com/en/tool/concrete-wave-speed-calculation-p-wave-s-wave-velocities.html](https://www.calculatorultra.com/en/tool/concrete-wave-speed-calculation-p-wave-s-wave-velocities.html)
- Hindawi AMSE 2016 — P, S, R wave velocities in RC slabs: [https://www.hindawi.com/journals/amse/2016/1548215/](https://www.hindawi.com/journals/amse/2016/1548215/)
- build-construct.com — Raft foundation dimensions (150–400 mm residential): [https://build-construct.com/structural-engineering/raft-foundations/](https://build-construct.com/structural-engineering/raft-foundations/)
- Strip foundation minimum depth UK/Mediterranean (≥0.45 m, no frost): [https://www.reinforcementproductsonline.co.uk/news/strip-foundations/](https://www.reinforcementproductsonline.co.uk/news/strip-foundations/)
- Grokipedia — Israeli coastal plain geology (kurkar ridges, Hamra red sandy loam): [https://grokipedia.com/page/Israeli_coastal_plain](https://grokipedia.com/page/Israeli_coastal_plain)
- AUB ScholarWorks — Seismic collapse assessment mid-rise RC buildings Beirut: [https://scholarworks.aub.edu.lb/items/0fa5befb-8b86-4d79-9d12-526aef0baa28](https://scholarworks.aub.edu.lb/items/0fa5befb-8b86-4d79-9d12-526aef0baa28)
- Rock bearing minimum depth (0.05–0.5 m on rock vs. 0.8–1.0 m on sand): [https://testbook.com/question-answer/the-minimum-depth-of-foundation-for-the-load-beari--62389988d4e2fd6a11da44cf](https://testbook.com/question-answer/the-minimum-depth-of-foundation-for-the-load-beari--62389988d4e2fd6a11da44cf)
