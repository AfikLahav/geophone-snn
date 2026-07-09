> **Research document — house-localization pivot ("wallhacks").** Deep web research synthesis (Claude Sonnet, 2026-07-09). Covers informal/non-engineered/vernacular foundations for the Levant/Arab-Israeli region. Design input for the SPECFEM3D procedural generator; not a validated experiment. Indexed in [README.md](README.md).

---

# Informal / Non-Engineered / Vernacular Residential Foundations — Levantine and Arab-Israeli Context

**Scope.** Companion to [15_foundation_and_domain_buffer.md](15_foundation_and_domain_buffer.md), which covers the engineered-RC-frame baseline (pad footings + grade beam, raft slab). This document covers the **informal and vernacular end of the foundation quality spectrum**: self-built RC-frame construction, unreinforced masonry (stone and concrete block), older stone-masonry houses, and buildings directly on rock. The purpose is to produce 2–3 parameterized archetypes for the mesh generator so the simulation corpus samples a realistic range of foundation coupling conditions.

---

## 1. Context: Why Foundation Quality Varies So Much in This Region

### 1.1 Pre-Code and Non-Permitted Construction

Israel's mandatory seismic code (IS 413) took effect in 1980; the West Bank and Gaza Strip have no unified national building code enforced in practice. A large fraction of the Arab-Israeli and Palestinian building stock predates 1980 or was built without permits regardless of era. As documented in Israeli reporting, approximately 80,000 structures nationally were built before 1980; in Arab towns in the Galilee, Negev, and Triangle, this proportion is disproportionately high because Israeli planning policy historically restricted permit issuance for Arab communities — meaning large-scale informal construction was the only option [(Times of Israel, 2023)](https://www.timesofisrael.com/israeli-buildings-face-major-earthquake-risk-despite-efforts-to-upgrade-them/); [(Human Rights Watch, 2008, on Negev Bedouin)](https://www.hrw.org/reports/2008/iopt0308/iopt0308.htm).

The seismic research on deficient RC frames in Ramallah (West Bank) identifies the primary causes of vulnerability as: "lack of appropriate seismic design guidelines and limited control of construction technologies" and buildings "designed by individuals without formal training in seismic design and built by inadequately skilled construction workers, resulting in non-engineered and vulnerable buildings." Additional stories are "frequently built without permits and typically neglect the added demands on the existing RC frame and its foundations." [(Springer Discover Civil Engineering, 2024)](https://link.springer.com/article/10.1007/s44290-024-00099-3).

This pattern — incremental vertical expansion ignoring foundation capacity — is the key structural signature of informal construction in the region. It is not limited to Palestine; it describes a broad class of "incrementally constructed" buildings across the Mediterranean and Middle East [(ScienceDirect, non-engineered RC frame framework, 2023)](https://www.sciencedirect.com/science/article/pii/S2352012423002588).

### 1.2 Regional Building Materials

The Levant has a strong tradition of **limestone masonry** driven by the availability of quarried stone in hilly areas (Jerusalem hills, Galilee, Samaria, Judean hills) and **concrete block (hollow CMU)** in coastal and lower-elevation areas where hard stone is unavailable. On the coast (Haifa–Tel Aviv–Jaffa corridor), the traditional local stone was **kurkar** — an aeolian carbonate-cemented quartz sandstone (calcarenite), used historically from Caesarea through Acre [(Wikipedia: Kurkar)](https://en.wikipedia.org/wiki/Kurkar). Kurkar compressive strength varies widely (3.7–34.4 MPa), is highly sensitive to weathering, and is much softer than hard limestone (which has UCS 50–150 MPa for Cenomanian varieties). Both materials have been used as building and foundation stone in vernacular construction.

The British Mandate-era (1920–1948) construction in Palestinian towns introduced reinforced concrete but also continued the stone tradition; British Mandate regulations in many areas compelled stone cladding for new buildings. Self-built housing from the 1950s onward increasingly used hollow concrete blocks (CMU, 100–200 mm thick), often combined with a minimal RC skeleton or load-bearing wall configuration.

---

## 2. Typology A: Unreinforced Stone-Masonry House on Shallow Rubble Strip Footing

### 2.1 Construction Description

This typology is the **oldest surviving vernacular layer**: limestone or kurkar stone-masonry load-bearing walls built by a mu'allim (master mason) with local quarried stone, lime or cement mortar. Common construction era: Ottoman through British Mandate (pre-1948) and continuing into the 1960s in villages.

**Structural system:** Load-bearing stone masonry walls (both wythe and rubble-core). The wall consists of two outer wythes of roughly dressed or coursed stone with an interior core of rubble and weak mortar (lime or mud in older buildings, cement-sand in post-1940 construction). Per the World Housing Encyclopedia global review of stone masonry construction, walls in rubble stone buildings "are supported either by strip footings of uncoursed rubble masonry or there are no footings at all" [(WHE Stone Masonry reports, summarized in Frontiers review)](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.590520/full).

**Foundation:** A continuous strip of uncoursed rubble stone, set in lime or weak cement mortar, running under all bearing walls. Depth is typically shallow — the international survey of similar vernacular stone masonry in seismic zones (Nepal, India, Iran, Turkey, Mediterranean) consistently shows depths of **0.5–0.8 m below finish grade**; no frost-depth constraint exists in Israel/Palestine. In hill-village sites in the Galilee and West Bank, where bedrock outcrops at or near the surface, the rubble strip is reduced to 1–3 courses of stone set directly into a hand-excavated trench or directly on exposed bedrock with minimal excavation.

Key quote from WHE Review (confirmed via search, WHE global stone masonry dataset): "Foundations consist of strip footings, which vary in depth from 0.5 m to 2.0 m depending on the number of stories and the local soil conditions. In some cases, footings do not exist at all." On rocky terrain: "in case of outcropping rock at the surface, the platform of dry stone masonry is directly erected onto ground without any embedded foundation."

**Mortar in foundation:** Historically lime-mud in Ottoman-era buildings; cement-sand (1:5 to 1:6) in Mandate-era and post-1940 buildings. The low mortar strength of the rubble core means the foundation is effectively a granular medium rather than a monolithic plate.

**Wall geometry:** Total wall thickness 450–600 mm (as documented in WHE reports for similar rubble stone building stock globally, consistent with Levantine masonry); outer wythes ~150–200 mm of cut/dressed stone; inner rubble core ~150–250 mm. Heights typically 1–2 storeys (3.0–6.0 m above grade). Ceilings: vaulted arches (stone domes common in older Levantine rural houses) or flat concrete/timber slabs added in 20th-century updates.

**Floor:** In older buildings: stone-flagged earthen floor on compacted fill; in 20th-century renovations: thin concrete screed or unreinforced 100–150 mm slab poured directly on compacted fill without isolation from the foundation walls.

### 2.2 What "No Foundation" Means Physically

In hilly Levantine construction (Galilee hill villages, Ramallah area, Hebron hills, Jerusalem neighborhood edges), the foundation situation is frequently **wall-on-rock with minimal interface material**:

- The mason excavates shallow trenches following the rock contour, removing the soil-weathering horizon (typically 0.1–0.3 m of residual terra rossa soil on hard Cenomanian limestone).
- The first course of wall stone is set directly on the exposed bedrock surface with a thin layer of mortar or sometimes dry-laid.
- There is no concrete footing, no reinforcement, and no engineered bearing surface — the wall rests on the structural competence of the exposed rock.

This is structurally defensible on competent limestone bedrock (allowable bearing capacity of hard limestone: 500–2,000 kPa, far exceeding the 50–100 kPa load from 1–2 storey masonry) but it creates a **very different elastic-wave coupling** than any soil-embedded foundation (see §4).

On softer soil sites (Hamra sand in coastal areas, valley alluvium), the same builder tradition produced a shallow rubble trench 0.5–0.8 m deep — enough to reach competent bearing but no engineered footing.

### 2.3 Regional Citations

- **Palestinian vernacular architecture:** The traditional Palestinian house used locally quarried stone ("collected from their own lands in the process of digging cisterns or flattening fields"), with the mu'allim constructing the key structural elements [(Arab America)](https://www.arabamerica.com/the-traditional-architecture-of-palestine/). The earliest Palestinian houses from ~9,000 years ago had "stone foundations with a superstructure made of mud-brick" [(Wikipedia: Architecture of Palestine)](https://en.wikipedia.org/wiki/Architecture_of_Palestine). Limestone materials ranged from meleke (highest quality, Cenomanian), to mizzi ahmar and yahudi (durable local varieties), with kurkar used on the coast [(Scribd/Use of Stones in Palestinian Architecture)](https://www.scribd.com/document/859125479/Use-of-Stones-in-the-Vernacular-Architecture-of-Palestinian-State).
- **Turkish analogue (eastern Turkey rural stone masonry):** Masonry structures "still erected in accordance with long-established traditions, frequently without the benefit of formal engineering supervision" [(MDPI Applied Sciences, 2025, Türkiye masonry)](https://www.mdpi.com/2076-3417/15/10/5490). The Frontiers review of rubble stone codes notes that Middle Eastern countries either "lack codes or follow international standards that exclude traditional rubble techniques" [(Frontiers Built Environment, 2020)](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.590520/full).
- **WHE World Housing Encyclopedia global dataset:** Walls of uncoursed rubble stone with mud mortar in Nepal (WHE Report 74) are supported by "strip foundations of uncoursed rubble masonry"; in some cases "footings do not exist at all." Wall thickness 450–600 mm. [(World Housing Encyclopedia)](https://world-housing.net/report-74-uncoursed-rubble-stone-masonry-walls-with-timber-floor-and-roof/).
- **Unreinforced masonry foundation seismic behavior:** "Unreinforced masonry — brick, concrete block, or stone — foundations often cannot resist earthquake shaking, may break apart or be too weak to hold anchor bolts, and homes may shift off such foundations" [(Earthquake Country Alliance)](https://www.earthquakecountry.org/step4/urmfoundations/).

---

## 3. Typology B: Non-Engineered RC Frame with Monolithic Slab-on-Grade (No Separate Footing)

### 3.1 Construction Description

This is the **dominant construction type for self-built housing since the 1960s–1980s** across Arab-Israeli towns and Palestinian territories. Concrete became preferred over stone as costs dropped; Israeli concrete was "the most popular building material" in Palestinian construction by the 2000s [(Arab News, 2021)](https://www.arabnews.com/node/1874316/amp). This building type is characterized by:

- A basic RC column-frame skeleton (columns ~200×200 to 250×300 mm, often poorly detailed, with minimal or absent stirrups in the joint zone).
- Infill of hollow concrete block (CMU, 100–200 mm thickness; hollow cavity not grouted).
- **Incremental construction**: built as one or two storeys, with the columns and slab designed and built to allow future vertical additions. Structural deficiencies include "no transverse reinforcement in the beam-column joint area, inadequate development length, and inadequate shear reinforcement" [(Springer/Ramallah case study)](https://link.springer.com/article/10.1007/s44290-024-00099-3). Upper storeys are added "without permits and typically neglect the added demands on the existing RC frame and its foundations."
- Hollow block walls: individual unit dimensions typically 200 mm wide × 200 mm high × 400 mm long, assembled in running bond with thin-bed or conventional mortar. Load-bearing walls minimum 240 mm [(eqrguides.com unreinforced CMU)](https://eqrguides.com/MasonryConcreteBlock/PlainConcreteBlockMasonry.htm).

**Foundation in the non-engineered version:** The critical departure from engineered practice is at the foundation. In non-engineered self-built construction in the region, the common practice is:

1. **Thin unreinforced or minimally reinforced slab-on-grade poured directly on compacted fill** without a separate strip footing. The slab acts as both the ground floor and the "foundation." Typical thickness: **100–150 mm** (compared to 180–220 mm for an engineered slab-on-grade and 300–400 mm for a raft). Reinforcement, when present, is a single layer of light mesh (4–6 mm diameter, 200 mm grid) with no bottom cover protection. Often absent entirely.

2. Alternatively: **an unreinforced or minimally reinforced plain-concrete strip under each bearing wall only**, without the perimeter grade beam and without pad footings under columns. The strip is typically 200–300 mm wide × 300–500 mm deep, poured into hand-excavated trenches with no formwork (trench sides serve as form). This is the dominant practice internationally for low-cost non-engineered residential construction per general strip foundation practice: "Strip footings don't necessarily need reinforcement; most strip footings have been designed unreinforced" with concrete ≥15 MPa [(general strip footing practice)](https://www.structuralbasics.com/strip-foundation-clay-concrete/). Minimum depth: "at least 400 mm below ground level" per South African practice (analogous to Israeli no-frost climate) [(SANS10400)](https://www.sans10400.co.za/concrete-foundations/).

3. **No tie between column base and footing:** In the most deficient cases, the RC columns simply sit on or are cast into the unreinforced slab, with no pad footing. The column bars may extend 200–300 mm into the slab as a stub but without the engineered development length or pad dimensions that is required.

**Wall in block typology:** Hollow CMU infill or load-bearing walls. Wall thickness: 150 mm interior, 200 mm exterior. Load-bearing walls may be 200–250 mm.

**Floor slabs:** 150–180 mm RC flat slab at each floor level, sometimes poured with a joist-and-block (waffle) system using hollow blocks as permanent formwork — a common Middle Eastern practice reducing slab weight and RC volume.

### 3.2 Regional Citations

- Tehran informal settlement study: "None [of 160 surveyed buildings in Farahzad informal settlement] were categorized as engineered buildings. 97.5% were formed of heavy construction materials. All sampled dwellings had been constructed with heavy materials assembled without any structural engineering design." [(PMC/Iranian Journal of Public Health, 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7152627/).
- Erbil (Iraq) empirical seismic survey: 45% of 1,470 surveyed buildings were non-engineered, "primarily made of brick masonry and concrete blocks with flexible to semi-rigid diaphragms." [(ResearchGate/Empirical Seismic Assessment Erbil, 2024)](https://www.researchgate.net/publication/382465658_Empirical_Seismic_Assessment_Of_Urban_Settlements_In_Developing_Countries_Case_Study_Erbil_Iraq).
- Palestinian refugee camp study: Identified six construction types in the West Bank. Camp buildings "represent a brick wall with columns and beams mostly designed to work separated and to resist vertical loads only." All construction defects noted were "soft storey, short column, lack of verticality and continuity of vertical structural elements." [(Sci Alert / Journal of Applied Sciences, 2008)](https://scialert.net/fulltext/?doi=jas.2008.1371.1382).
- Unrecognized Negev Bedouin settlements: Structures typically use "corrugated metal sheets, concrete blocks, and salvaged materials, resulting in rudimentary housing that expands organically without adherence to zoning or safety regulations." [(Wikipedia: Unrecognized Bedouin Villages in Israel)](https://en.wikipedia.org/wiki/Unrecognized_Bedouin_villages_in_Israel); [(HRW, 2008)](https://www.hrw.org/reports/2008/iopt0308/iopt0308.htm).
- Israeli Arab town informal expansion: "Building without permits spread widely as a direct result of not being able to receive building permits, typically involving two-family houses." [(Arab America research reported via +972 Magazine context)](https://www.972mag.com/israel-issuing-palestinian-building-permits-to-further-west-bank-land-grab/).

---

## 4. Physics: How Informal Foundations Change Structure-to-Soil Coupling

This section contrasts the three foundation scenarios against the engineered raft baseline from doc [15_foundation_and_domain_buffer.md](15_foundation_and_domain_buffer.md).

### 4.1 Impedance Values

| Material | Vp (m/s) | Vs (m/s) | ρ (kg/m³) | Z_P (MRayl) | Z_S (MRayl) |
|---|---|---|---|---|---|
| Engineered concrete (raft/grade beam) | 3,800–4,000 | 2,200–2,400 | 2,400 | 9.1–9.6 | 5.3–5.8 |
| Hard limestone bedrock (Cenomanian Israel) | 3,000–6,000 | 1,600–3,500 | 2,300–2,600 | 6.9–15.6 | 3.7–9.1 |
| Kurkar sandstone (coastal, variable) | 800–2,500 | 400–1,300 | 1,800–2,100 | 1.4–5.3 | 0.7–2.7 |
| Rubble stone masonry (mortar-bound) | 1,500–2,500 | 700–1,400 | 1,800–2,200 | 2.7–5.5 | 1.3–3.1 |
| Hamra (red sandy loam, coastal) | 300–600 | 100–200 | 1,600–1,900 | 0.48–1.14 | 0.16–0.38 |
| Alluvial clay/silt (inland, Israeli) | 400–900 | 150–300 | 1,800–2,000 | 0.72–1.80 | 0.27–0.60 |
| Hard bedrock (seismic site class A) | 1,500–3,000 (Vs) | | 2,200–2,600 | | 3.3–7.8 |

Sources: limestone Vp/Vs ranges from seismic site classification literature (Vs 760–1,500 m/s less weathered limestone, >1,500 m/s hard limestone) [(PMC seismic site classification, 2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6306271/); kurkar UCS 3.7–34.4 MPa [(Haaretz/Stones Park)](https://www.stones-park.com/en/Kurkar+Stone); general limestone P-wave literature [(Nature Scientific Reports, 2021)](https://www.nature.com/articles/s41598-021-03524-0).

### 4.2 Coupling Physics by Foundation Type

**A. Engineered raft slab (baseline, doc 15):**
- Full footprint concrete-to-soil contact: 14×7 = 98 m² interface area.
- Impedance ratio Z_concrete/Z_soil ≈ 5–20 (Hamra) → 30–50% energy transmits per crossing.
- This creates the resonant cavity in the building and the soil-radiation path described in doc 15.

**B. Rubble strip footing on soil (Typology A, shallow site):**
- The rubble strip has lower Vp/Vs than engineered concrete (porous, cracked mortar matrix, Vp ≈ 1,500–2,500 m/s vs 3,800 m/s for concrete).
- Foundation-soil impedance ratio Z_rubble/Z_soil ≈ 1.5–5 (Hamra) — **much lower contrast than engineered concrete-to-soil**. This means:
  - More energy transmits into soil per crossing (reflection coefficient R = ((Z_r − Z_s)/(Z_r + Z_s))² ≈ 0.03–0.50, compared to 0.50–0.70 for concrete-to-Hamra).
  - The building leaks more energy into soil → the interior structural wave reverberant field decays faster → shorter reverberation times.
  - Exterior geophones receive more energy via the soil path (more radiation), but the structural cavity quality factor Q is lower.
  - The energy difference between locations is compressed (worse localization SNR from structural reverb).
- Geometrically: strip under each wall only, not full footprint. Radiation area ≈ 2 × (14 + 7) × 0.4 m width = ~17 m² (perimeter strip underside) vs. 98 m² for raft. **Much smaller contact area → weaker absolute coupling to soil, despite lower impedance contrast.**

**C. Slab-on-grade without footing, minimal reinforcement (Typology B):**
- A thin (100–150 mm) concrete plate sitting on compacted fill or natural ground.
- Material properties: concrete Vp ≈ 3,500–4,000 m/s (close to engineered concrete).
- Coupling area = full footprint (98 m² for 14×7 building) — same as raft.
- **Key difference from raft:** No embedded foundation below grade. The slab soffit is at grade (z = 0) or slightly above, not at −0.4 to −0.6 m. This means:
  - The structural concrete-soil interface is at the surface, not buried — there is no surrounding-soil confinement that would stiffen the soil column above the slab soffit.
  - The thin slab has much lower bending stiffness: EI scales as t³ → at 120 mm vs 350 mm, bending stiffness is (120/350)³ = 0.04× lower. This matters because low-frequency flexural modes of the slab (below ~30 Hz for a 14 m span, thin slab) couple more energy into long-wavelength soil radiation.
  - Radiation impedance is determined by the soil beneath (Z_soil, not Z_concrete as the dominant path), so the slab behaves more like a flexible plate vibrating on soil — a different boundary condition than the rigid raft.
  - **Net: more energy leaks into soil at low frequencies, less structural wave trapping in the building.** Exterior geophones benefit at low frequency (10–50 Hz); structural-wave fingerprint differences between rooms are weaker because the cavity is leakier.

**D. Wall on bedrock (Typology A, hill-village on Cenomanian limestone):**
- The stone masonry wall rests directly on hard limestone bedrock (Vs = 1,500–3,500 m/s, ρ ≈ 2,300–2,600 kg/m³).
- Foundation-rock impedance ratio Z_stone/Z_rock ≈ 0.3–0.7 (rubble stone Z ≈ 2.7–5.5 MRayl vs limestone Z ≈ 6.9–15.6 MRayl). This is an **inverted impedance contrast**: stone is softer than the substrate.
- Physics of inverted contrast: waves going from building (stone masonry, lower impedance) into bedrock (higher impedance) have transmission coefficient T = 2Z₂/(Z₁+Z₂) > 1 for amplitude (but energy T_E = 4Z₁Z₂/(Z₁+Z₂)² ≤ 1). For Z_stone = 3.5 MRayl and Z_rock = 10 MRayl: R = ((10−3.5)/(10+3.5))² = (6.5/13.5)² ≈ 0.23. So ~77% of wave energy transmits into the rock per crossing. This is much higher energy transfer than concrete-to-Hamra (30–50%).
- The key finding from the vibration transmission literature: "For a building supported directly on rock, the coupling loss is 0 dB" [(ScienceDirect foundation coupling review)](https://www.sciencedirect.com/science/article/abs/pii/S0141029613001612). This means there is essentially **no energy loss at the structure-rock interface**: footstep energy injected into the masonry wall couples nearly perfectly into the bedrock.
- The bedrock is a **much larger radiating body** than a soil block: seismic wave energy from a footstep propagates with low geometric attenuation and low material damping in intact limestone (Q_rock ≈ 100–300 vs Q_soil ≈ 20–50 for soft deposits). Exterior geophones on or near the rock surface can receive very strong signals at large stand-off distances.
- However, the **structural resonant cavity behavior is fundamentally different**: the building walls (low-impedance) are bounded below by high-impedance bedrock. The building barely traps energy — it is like a soft layer on stiff substrate. Structural resonant frequencies shift because the boundary condition at the base is no longer reflective (as with concrete-to-soil) but transmissive.
- **Important seismic site effect analogy:** Bedrock amplification at the surface is de-amplified relative to soft soil, but for interior→exterior energy path, rock coupling is efficient — geophone near the building on exposed rock surface may pick up very high-amplitude signals at frequencies above the bedrock fundamental resonance. The signal-to-noise ratio at exterior geophones could actually be **higher** for this typology, but the structural-wave fingerprint pattern is more spatially uniform (less room-to-room differentiation).

**Summary table — informal foundation coupling vs. engineered baseline:**

| Foundation type | Contact area | Z contrast (struct/soil or rock) | Energy to soil/rock per crossing | Structural cavity quality | Exterior geophone signal strength | Localization fingerprint quality |
|---|---|---|---|---|---|---|
| Engineered raft (doc 15 baseline) | ~98 m² (full footprint, buried 0.4–0.6 m) | High (Z_c/Z_soil = 5–20) | 30–50% | High Q (resonant cavity) | Moderate-strong via soil path | Strong spatial differentiation |
| Rubble strip on shallow soil | ~17 m² (perimeter only, 0.5–0.8 m deep) | Medium (Z_rubble/Z_soil = 1.5–5) | 50–80% per crossing | Medium Q (leaky) | Weaker absolute level (smaller area) | Moderate differentiation |
| Thin slab-on-grade, no footing | ~98 m² (at grade, not buried) | High (Z_c/Z_soil = 5–20) | 30–50% but low-frequency leaky (thin plate) | Low Q (leaky slab, less trapping) | Similar to raft at high freq, more leakage at low freq | Weaker differentiation |
| Stone wall on bedrock | Direct wall-rock contact, small area per wall | Inverted (Z_stone/Z_rock ≈ 0.3–0.7) | 70–80% into rock | Very low Q (energy bleeds into rock) | Very strong on rock surface | Weak spatial differentiation (flat field) |

---

## 5. Three Parametric Archetypes for the Mesh Generator

The following archetypes define the parametric range for **informal/non-engineered** buildings. They are designed to be representable in the existing axis-aligned hex mesh, consistent with the building footprint family from doc [14_building_size_envelope.md](14_building_size_envelope.md) (~10–16 m × 6–10 m), and complement the **Archetype E** (engineered raft) defined in doc 15.

---

### Archetype F — "Rubble Stone Masonry on Shallow Strip" (Hill-Village Vernacular)

**Physical description:** 1–2 storey unreinforced stone masonry load-bearing wall house. Walls of rough-cut limestone or rubble-core stone. Foundation is a continuous rubble-stone or plain-concrete strip under all bearing walls, 0.5–0.7 m below finish grade. No RC frame. Floor is stone-flagged or thin concrete screed.

**Mesh recipe (for 14 × 7 m house):**

```
Superstructure — Stone masonry walls (all bearing walls):
  Material:    stone_masonry (Vp=2200, Vs=1100, ρ=2100 kg/m³, Q=30)
  Thickness:   0.50 m (two wythes + rubble core)
  Height:      0 to +3.0 m (single storey) or 0 to +6.0 m (double)
  Footprint:   all four exterior walls; interior cross-walls at ~3–4 m spacing

Foundation strip — rubble or plain concrete under each bearing wall:
  Material:    rubble_foundation (Vp=1800, Vs=800, ρ=2000 kg/m³, Q=20)
               (or plain_concrete_unreinforced: Vp=3200, Vs=1800, ρ=2300, Q=50 if
                owner poured a concrete strip instead of rubble)
  Geometry:    continuous strip only under the wall above;
               Width:  0.60 m (approximately 0.50 m wall + 0.05 m overhang each side)
               Depth:  0.60 m (height in z dimension)
               Bottom: z = −0.60 m below finish grade
  NOTE: This is NOT a full-footprint layer — only under bearing walls.
        Interior of footprint below grade is: soil from z = 0 down.

Ground floor:  thin concrete screed or slab
  Material:    concrete (Vp=3500, Vs=2000, ρ=2400, Q=80)
  Thickness:   0.10 m (100 mm unreinforced screed on fill)
  Top:         z = 0.00 m (finish floor)
  Bottom:      z = −0.10 m
  Sits on compacted fill below (not bonded to foundation strip → treat as
  separate element with soil contact at bottom)

Soil fill (inside perimeter, below screed):
  Material:    compacted_fill (Vp=500, Vs=200, ρ=1800, Q=15)
  From z = −0.10 m to z = −0.60 m (inside the foundation strip)

Soil (native, outside perimeter and below foundation strip):
  Material:    soil_native (Vp=400–700, Vs=150–300 per site class)
  From z = −0.60 m downward
```

**Coupling note:** The discontinuous strip (not full-footprint) reduces soil radiation area by ~80% vs. raft. Energy enters soil as a distributed line source under each perimeter wall, not as a planar radiator. At low frequencies (20–50 Hz), this radiates as near-cylindrical waves outward, with a 1/√r geometric spreading — different from the 2D-plate radiation of a raft.

**Sampler parameters:**
- Strip depth D_f: uniform sample in [0.40, 0.80] m
- Strip width W_f: uniform in [0.50, 0.70] m (1.0–1.4× wall thickness)
- Foundation material: discrete choice {rubble (60%), plain concrete (40%)}
- Wall thickness t_wall: uniform in [0.40, 0.60] m
- Storeys: discrete {1 (60%), 2 (40%)}

---

### Archetype G — "Thin Slab-on-Grade, No Footing" (Non-Engineered RC-Frame or Load-Bearing Block)

**Physical description:** Self-built 1–3 storey RC-frame or load-bearing hollow CMU block house. The owner/builder pours a thin concrete slab directly on compacted fill as the "foundation" — no separate strip footing, no grade beam, no pad under columns. Slab may be nominally reinforced (single mesh layer) or unreinforced.

**Mesh recipe (for 14 × 7 m house):**

```
Superstructure — CMU block walls or thin RC-frame + CMU infill:
  Exterior walls:  Material: hollow_block (Vp=1800, Vs=900, ρ=1100 kg/m³, Q=25)
                   Thickness: 0.20 m (200 mm hollow block)
  Interior walls:  same material, 0.15 m (150 mm)
  RC columns:      Material: concrete_weak (Vp=3500, Vs=2000, ρ=2300, Q=60)
                   Dimensions: 0.20 m × 0.20 m (non-engineered RC columns)
                   At perimeter and major junctions

Foundation — thin unreinforced or lightly reinforced slab at grade:
  Material:    concrete_weak (Vp=3200, Vs=1800, ρ=2300 kg/m³, Q=60)
               (Use lower Vp/Vs than engineered RC: poorer mix design, higher w/c ratio,
               aggregate void pockets; strength class C15–C20 typical in self-built construction
               vs C25–C35 engineered)
  Geometry:    full footprint slab (14.0 × 7.0 m plan)
  Thickness:   0.12 m (120 mm — typical for thin unreinforced slab-on-grade)
  Top:         z = 0.00 m (finish floor)
  Bottom:      z = −0.12 m (soffit contacts compacted fill directly at near-grade level)
  NO separate footing below this layer.
  NOTE: This slab rests directly on compacted fill — not embedded. No perimeter stem wall.
        Below z = −0.12 m: compacted fill or native soil.

Compacted fill:
  Material:    compacted_fill (Vp=500, Vs=200, ρ=1800, Q=15)
  From z = −0.12 m downward (full footprint, no structure below)

Soil:
  Native soil below
```

**Coupling note:** This is a **full-footprint** slab (same contact area as a raft) but it is thin and at grade (not buried). The low bending stiffness means the slab flexes more freely → more energy at low frequencies radiates into soil. The slab-soil impedance contrast is the same as raft (concrete to soil), so the per-unit-area coupling efficiency is similar, but the thinner slab has different flexural resonance modes. The net effect: similar radiation area to raft, but less energy trapping in the structure — more of the footstep energy goes into soil at low frequencies.

**Concrete properties note:** Self-built construction in Palestine and Arab-Israeli towns uses locally purchased ready-mix of variable quality. Research documents "material variability" as a key characteristic of non-engineered construction [(ScienceDirect non-engineered RC frame framework)](https://www.sciencedirect.com/science/article/pii/S2352012423002588). For simulation: reduce concrete Vp by 10–15% and Q by 30–40% relative to engineered concrete to account for lower strength and higher porosity/cracking.

**Sampler parameters:**
- Slab thickness t_slab: uniform in [0.10, 0.15] m
- Concrete quality: discrete {weak (Vp=3200, Vs=1800, Q=60, 60%) or standard (Vp=3800, Vs=2200, Q=100, 40%)}
- Interior columns: discrete {present — 0.20×0.20 m RC (70%)} or {absent — pure load-bearing block (30%)}
- Storeys: discrete {1 (30%), 2 (50%), 3 (20%)}

---

### Archetype H — "Stone Masonry Wall on Exposed Bedrock" (Hill-Village on Limestone)

**Physical description:** Historic 1–2 storey stone masonry house on a hill site where limestone bedrock outcrops at or near the surface. The wall base is set directly on bedrock with no concrete footing — only a mortar bedding course or dry-laid first course on the rock surface. This is the limiting case of "zero foundation depth."

**Mesh recipe (for 14 × 7 m house on rock site):**

```
Superstructure — Stone masonry walls:
  Material:    stone_masonry (Vp=2200, Vs=1100, ρ=2100 kg/m³, Q=30)
  Thickness:   0.50 m
  Height:      z = 0 to +3.0 m (or +6.0 m for 2 storeys)

Foundation "interface" — mortar bedding course only:
  Material:    mortar_interface (Vp=1500, Vs=700, ρ=1900, Q=15)
  Geometry:    under each bearing wall, width = wall thickness (0.50 m),
               Height: 0.05 m (50 mm bedding course only — 1 element thick at min)
  Top:         z = 0.00 m (finish floor = rock surface, or ~100 mm above)
  Bottom:      z = −0.05 m (contacts bedrock directly)
  NOTE: No concrete footing. No excavation below bedrock surface.

Bedrock (replaces soil entirely, or from z = −D_soil downward):
  Option 1: Bedrock at surface (z = 0):
    Material:  limestone_bedrock (Vp=3500, Vs=2000, ρ=2500 kg/m³, Q=150)
    From z = 0 downward through entire domain.
    Soil layer absent or very thin (0–0.1 m terra rossa topsoil — negligible, treated as bedrock)

  Option 2: Thin soil veneer over bedrock (common on Cenomanian hillsides):
    Soil layer: 0.1–0.3 m of residual terra rossa (Vp=400, Vs=150, ρ=1800, Q=15)
    Bedrock from z = −0.3 m downward (but foundation strip already seated on rock at z ≈ 0)

Ground floor:  stone flagging or thin screed on bedrock
  Thickness:   0.05–0.10 m
  Material:    stone_masonry or mortar_interface
  Top:         z = 0.00 m
  Bottom:      z = −0.05 m, contacts bedrock

Kurkar variant (coastal hill sites):
  Replace limestone_bedrock with kurkar_rock:
  Material:    kurkar (Vp=1500, Vs=700, ρ=2000 kg/m³, Q=50)
               (softer, more porous; Vp/Vs ratio similar but lower absolute values)
```

**Coupling note:** This is the **strong-coupling / low-quality-factor** end. The building couples nearly perfectly into the bedrock (energy transmission coefficient ~0.75–0.85 per wall-rock crossing). Footstep energy in the building propagates as seismic body waves and surface waves in the rock, not as soil Rayleigh waves. The bedrock Rayleigh wave speed is 1,800–3,200 m/s (vs. 140–460 m/s for soft soil) — the same footstep energy at 50 Hz travels λ_R ≈ 36–64 m in bedrock vs. 2–9 m in soft soil. Exterior geophones at 5–10 m distance from the building could receive very strong signals from the rock path. However, the signal field will be less spatially differentiated (bedrock is high-Q, does not scatter or attenuate within the domain), making room-level localization based on arrival time differences harder. The fingerprinting approach must rely on building-interior structural modes more than on soil-path amplitude differences.

**Sampler parameters:**
- Bedrock depth D_br: discrete {at surface (z=0), 50% of samples} or {shallow at z=−0.2 m, 50%}
- Bedrock type: discrete {limestone (Vp=3500, Vs=2000, ρ=2500, Q=150, 70%)} or {kurkar (Vp=1500, Vs=700, ρ=2000, Q=50, 30%)}
- Wall thickness: uniform in [0.40, 0.60] m
- Storeys: discrete {1 (60%), 2 (40%)}
- Floor at surface: stone flag or screed, 0.05–0.10 m thick

---

## 6. Integrating Informal Archetypes into the Generator

### 6.1 Quality-Range Sampling

The three informal archetypes (F, G, H) plus the engineered baseline (Archetype E = raft, from doc 15) form a **4-point quality spectrum**:

| Archetype | Foundation type | Soil-coupling efficiency | Structural cavity Q | Building era |
|---|---|---|---|---|
| E (engineered raft) | RC raft, 0.40–0.60 m deep | Medium (30–50%, full footprint) | High | Post-1980 |
| F (rubble strip) | Rubble/plain strip, 0.5–0.8 m, perimeter only | Medium (50–80%, small area) | Medium | Pre-1970 |
| G (slab-on-grade, no footing) | Thin slab 0.10–0.15 m, at grade | Medium-High low-f, less trapping | Low | 1960s–2000s |
| H (wall on bedrock) | No footing, direct rock contact | Very high (0 dB coupling loss on rock) | Very low | Pre-1950 and recent on rocky sites |

The generator should sample from {E, F, G, H} with approximate frequency weights reflecting regional prevalence:
- **E (engineered raft):** ~25% of houses (post-1980 legal construction)
- **F (rubble strip):** ~20% (older stone-masonry housing stock)
- **G (slab-on-grade/non-engineered RC):** ~40% (dominant post-1960s self-build)
- **H (bedrock):** ~15% (hill village sites; less common in flat terrain)

These weights are estimates based on the regional context described in §1. They have not been validated against a census; treat as informed priors.

### 6.2 Material Property Ranges (Summary for Generator)

| Material label | Vp (m/s) | Vs (m/s) | ρ (kg/m³) | Q_p | Q_s |
|---|---|---|---|---|---|
| concrete_engineered | 3,800 | 2,200 | 2,400 | 100 | 50 |
| concrete_weak | 3,200 | 1,800 | 2,300 | 60 | 30 |
| stone_masonry | 2,200 | 1,100 | 2,100 | 30 | 15 |
| rubble_foundation | 1,800 | 800 | 2,000 | 20 | 10 |
| mortar_interface | 1,500 | 700 | 1,900 | 15 | 8 |
| hollow_block | 1,800 | 900 | 1,100 | 25 | 12 |
| limestone_bedrock | 3,500–5,000 | 2,000–3,000 | 2,500 | 150 | 75 |
| kurkar_rock | 1,500 | 700 | 2,000 | 50 | 25 |
| compacted_fill | 500 | 200 | 1,800 | 15 | 8 |
| soil_hamra | 400–600 | 150–200 | 1,700 | 20 | 10 |
| soil_alluvial_clay | 600–900 | 200–300 | 1,900 | 25 | 12 |

Note: Q values for masonry and rubble are broad estimates. The literature does not provide systematic Q measurements for vernacular rubble masonry; values above are derived by analogy with similar low-quality masonry studied for seismic purposes. They should be treated as representative midpoints with ±50% uncertainty.

### 6.3 Mesh Implementation Notes

1. **Archetype F (rubble strip):** Model the strip as 3–5 element columns at mesh resolution under each exterior wall, from z = 0 to z = −D_f. No elements below grade in the building interior. This creates the correct perimeter-only coupling geometry.

2. **Archetype G (slab-on-grade):** A single thin concrete plate at z = −0.10 to 0.00 m, full footprint. No elements below grade. This is actually the simplest mesh case of the four.

3. **Archetype H (bedrock):** Replace the soil material from z = −D_br downward with the bedrock material. Optionally model a thin soil veneer (1–2 element layers at mesh resolution). The mortar interface course (0.05 m thick) under the first wall course can be treated as the lowest row of wall elements with lower Vp/Vs, avoiding a sub-mesh-resolution element.

4. **Interface treatment:** SPECFEM3D handles elastic-elastic material discontinuities at conforming element faces automatically. No special treatment is needed at any of these interfaces beyond ensuring conforming meshes.

---

## 7. Honest Assessment of Uncertainties

1. **No direct foundation survey for Arab-Israeli towns or Palestinian territories was found in the literature.** The documentation of foundation types for this specific building stock is inferred from: (a) regional seismic vulnerability studies for similar construction in Palestine (West Bank), (b) informal settlement surveys for analogous non-engineered construction in Iran and Iraq, (c) the global World Housing Encyclopedia stone masonry dataset, and (d) general construction practice knowledge for the region. The typologies described here are well-grounded but should be treated as engineering judgment, not a validated census.

2. **Q values for rubble and stone masonry in the 20–250 Hz band are not well-established in the literature.** Seismic attenuation measurements for these materials at low frequency are sparse. The values given (Q_s = 10–15 for rubble) are conservative (high attenuation); real values may be higher if the mortar quality is better.

3. **Bedrock depth distribution on specific sites is unknown.** The Galilee and Judean hills have extremely variable cover: from 0 to >5 m terra rossa over limestone within 100 m. For the simulation this is treated as a sampled parameter.

4. **Hollow block material properties vary widely** depending on block void fraction, block strength, and mortar quality. The values given (Vp = 1800 m/s, Vs = 900 m/s) are for 200 mm hollow CMU in cement mortar; they should be varied ±20–30% in the generator to sample realistic uncertainty.

---

## 8. Summary of Sources

- [Times of Israel — Israeli building earthquake risk (2023)](https://www.timesofisrael.com/israeli-buildings-face-major-earthquake-risk-despite-efforts-to-upgrade-them/)
- [Springer / Discover Civil Engineering — Seismic performance of deficient RC frames, Ramallah (2024)](https://link.springer.com/article/10.1007/s44290-024-00099-3)
- [ScienceDirect — Framework for non-engineered masonry infilled RC frame performance (2023)](https://www.sciencedirect.com/science/article/pii/S2352012423002588)
- [ScienceDirect — Non-engineered incrementally constructed URM RC frames (2025)](https://www.sciencedirect.com/science/article/abs/pii/S2212420925005278)
- [Human Rights Watch — Off the Map: Bedouin villages Israel (2008)](https://www.hrw.org/reports/2008/iopt0308/iopt0308.htm)
- [Wikipedia — Unrecognized Bedouin villages in Israel](https://en.wikipedia.org/wiki/Unrecognized_Bedouin_villages_in_Israel)
- [Arab America — Traditional Architecture of Palestine](https://www.arabamerica.com/the-traditional-architecture-of-palestine/)
- [Wikipedia — Architecture of Palestine](https://en.wikipedia.org/wiki/Architecture_of_Palestine)
- [Wikipedia — Kurkar (aeolian sandstone)](https://en.wikipedia.org/wiki/Kurkar)
- [Stones Park — Kurkar stone properties](https://www.stones-park.com/en/Kurkar+Stone)
- [Frontiers Built Environment — Rubble Stone Masonry codes worldwide (2020)](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.590520/full)
- [Frontiers Built Environment — School buildings in rubble stone masonry (2019)](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2019.00013/full)
- [World Housing Encyclopedia — Stone Masonry report index](https://world-housing.net/category/masonry/stone-masonry/)
- [World Housing Encyclopedia — Report 74 (Nepal, rubble stone)](https://world-housing.net/report-74-uncoursed-rubble-stone-masonry-walls-with-timber-floor-and-roof/)
- [Sci Alert / JAS — Rapid assessment seismic vulnerability Palestinian refugee camps (2008)](https://scialert.net/fulltext/?doi=jas.2008.1371.1382)
- [PMC / Iranian Journal of Public Health — Non-engineered buildings Tehran informal settlement (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7152627/)
- [ResearchGate — Empirical seismic assessment Erbil informal housing (2024)](https://www.researchgate.net/publication/382465658_Empirical_Seismic_Assessment_Of_Urban_Settlements_In_Developing_Countries_Case_Study_Erbil_Iraq)
- [Earthquake Country Alliance — Unreinforced masonry foundations](https://www.earthquakecountry.org/step4/urmfoundations/)
- [eqrguides.com — Unreinforced concrete block masonry](https://eqrguides.com/MasonryConcreteBlock/PlainConcreteBlockMasonry.htm)
- [eqrguides.com — Unreinforced stone masonry](https://eqrguides.com/MasonryStone/PlainStoneMasonry.htm)
- [SANS10400 — Concrete foundations (non-frost climate minimum depth 400 mm)](https://www.sans10400.co.za/concrete-foundations/)
- [ScienceDirect — Foundation coupling loss for structure-borne vibration (coupling loss = 0 dB for building on rock)](https://www.sciencedirect.com/science/article/abs/pii/S0141029613001612)
- [PMC — Seismic site classification and shallow bedrock sites (limestone Vs 760–3000 m/s)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6306271/)
- [MDPI Applied Sciences — Seismic vulnerability masonry dwellings eastern Turkey (2025)](https://www.mdpi.com/2076-3417/15/10/5490)
- [Arab News — Palestinian self-build housing (2021)](https://www.arabnews.com/node/1874316/amp)
- [Haaretz — Rare sandstone (kurkar) ridges endangered](https://www.haaretz.com/.premium-rare-sandstone-ridges-endangered-1.5350749)
- [Nirtopper.com — Kurkar as building material history](https://www.nirtopper.com/post/kurkar)
- [Nature Scientific Reports — Limestone P-wave velocity correlation (2021)](https://www.nature.com/articles/s41598-021-03524-0)
- [Springer Journal of Structural Engineering — Seismic response stone masonry on soft soil vs SSI (2025)](https://www.sciencedirect.com/science/article/pii/S2772467025000132)
- [geotech.hr — Types of shallow foundations](https://www.geotech.hr/en/types-of-shallow-foundations/)
