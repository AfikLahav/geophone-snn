> **Research survey — house-localization pivot ("wallhacks").** Background research agent (Sonnet), 2026-07-08. Literature/tooling survey; design input, not a validated experiment. Indexed in [README.md](README.md).

---

# Architectural Datasets for Procedural Generator Reference — Seismic Simulation Context

**Use case:** Extract statistical priors (room dimensions, wall configs, footprint area, ceiling height, storey count, door/window placement) to drive a procedural 3-D generator of concrete-frame/masonry residential buildings for elastic-wave footstep-localization simulation. Deployment context: Israel (Mediterranean). Meshes from these datasets are NOT used directly.

---

## 1. Comparison Table

| Dataset | Scale | Format | Arch. Data Available | License | Commercial? | Geometric Quality | Construction Bias |
|---|---|---|---|---|---|---|---|
| **3D-FRONT** | 6,813 houses / 14,629–18,968 rooms | JSON (1 file/house) | Room polygons, types, furniture; NO explicit wall thickness or ceiling height fields | CC-BY-NC-4.0; email request to Alibaba | No | CAD-clean synthetic | Chinese apartments → concrete frame. Directly relevant |
| **Structured3D** | 3,500 houses / 21,835 rooms, avg 5.79 rooms/scene | JSON + panoramic images | Wall junctions, planes, room cuboids, semantics; 3D bounding boxes | Custom Terms of Use; form signature required | No | CAD-clean synthetic | Chinese/generic interior; concrete assumption |
| **ProcTHOR-10k** | 12,000 procedural houses, 1–10 rooms | JSONL (1 house/line) | Floor polygons, room type, door/window positions; NO wall thickness; ceiling implicit | Apache 2.0 | Yes (fully open) | Ideal procedural; not real architecture | US single-family, timber-frame style; low relevance for concrete |
| **HM3D** | 1,000 large-scale 3D scans (residential + commercial) | GLB/OBJ meshes | 3D mesh geometry; rooms inferable; no explicit dim annotations | Matterport Terms of Use | No (academic only) | Real scan; scan noise; NOT watertight | US suburban; mixed construction |
| **HM3D-Sem** | 216 HM3D scenes annotated; 3,100 rooms; 142,646 object instances | Mesh + JSON semantic labels | Room semantic labels, object instances; same geometry as HM3D | Academic non-commercial only | No | Same as HM3D | Same as HM3D |
| **Matterport3D** | 90 buildings; 10,800 panoramas; 194,400 RGB-D images | PLY meshes + JSON | Room regions, object segments; ceiling height annotations (~2.98m avg) | Matterport Terms of Use | No (academic only) | Real scan with noise; NOT watertight | US spaces; mixed types |
| **BuildingNet** | 2,000 building models; 513K annotated mesh primitives | OBJ + PLY | **Exterior shells only** — no interior, no rooms, no wall thickness | 3D Warehouse derivatives; form required | No | CAD-clean exterior OBJ | Mixed types (houses, churches, etc.); exterior only |
| **HSSD** | 211 houses; 18,656 objects | glTF (walls/floors/ceilings decomposed) | Walls, floors, ceilings, openings as separate elements; based on real floor plans | Permissive academic research license | No (academic) | CAD-clean synthetic | US/Floorplanner.com plans; mixed |
| **Replica** | 18 scenes | JSON + binary mesh | Dense mesh geometry; semantic segmentation; glass/mirror labels | Meta Research License (allows commercial employees with compliance) | Conditionally yes | Near-watertight, high quality | US office/apartment; mixed |
| **CubiCasa5K** | 5,000 floor plan images | Raster PNG + SVG vector | Rooms, walls, doors, windows as polygons; 80+ categories | MIT License | Yes | 2D vector; no 3D | Finnish residential; **masonry/concrete** — geographically relevant |
| **RPLAN** | 80,788 Asian residential floor plans | 256×256 4-ch PNG raster | 13 room types inferred from pixel labels; footprint 63–201 m², mean 102 m² | Non-commercial research only; no redistribution | No | 2D raster; no wall vector | Chinese apartments → **concrete frame**; highly relevant to Israeli context |
| **LIFULL HOME'S** | 5.31M Japanese listings; 83M images | JPEG images + TSV metadata | Area (m²), floor, structure in metadata; floor plans as raster images only | Academic only via NII Japan | No | Raster JPEG; no vector annotation | Japanese RC frame (similar to Med.); hard to extract priors programmatically |
| **ZInD** (Zillow Indoor) | 1,524 US homes; 71,474 panoramas | JSON (room polygons, W/D/O) + images | Room polygons, door/window/opening positions, 2D+3D floor plans | Academic use only (Zillow) | No | 2D vector + real-world data | US residential; timber frame |
| **ResPlan** (Aug 2025) | 17,000 residential floor plans | Compact JSON vector + PNG | Rooms (labels, centroid, area), walls (wall_depth=0.15 m stored explicitly), doors, windows, balconies, room adjacency graph | Permissive open-source | Yes (likely) | 2D vector with consistent wall_depth; direct 3D extrusion supported | Real-estate listings; region unstated |
| **IFC-Bench** (HuggingFace) | Small collection of IFC models | IFC/STEP | Full BIM: structural elements, wall layers, materials, floor slabs | Mostly CC BY 4.0 / MIT | Yes | CAD-clean BIM | Varies; European models present |
| **BIMNet** | 25 real-world IFC scans; 382 rooms; 8,700 m² | IFC + point clouds | Full BIM with wall thickness, structural type, material layers, openings | Open-access | Yes | High-fidelity; IFC-aligned geometry | European real-world buildings |
| **OSM Buildings (Israel)** | ~752K Israel buildings | Shapefile/GeoJSON/PBF | Footprint polygon, `building:levels`, occupancy type | ODbL (open) | Yes | 2D footprints only; no wall thickness | **Israel-specific footprints** — directly useful |
| **Overture Maps Buildings** | 1.39B global footprints | Parquet/GeoJSON | Footprint, height (m), occupancy, source | ODbL | Yes | 2D footprints + estimated height | Global; covers Israel; height data present |
| **GHS-OBAT** | 2.3B footprints from Overture | CSV/GeoPackage | Construction epoch, height, residential/non-residential, compactness | Open (JRC EU) | Yes | 2D footprints + thematic attributes | Global including Mediterranean |

---

## 2. Extractable Architectural Priors

**A. Footprint Area Distribution** — Best: **RPLAN** (63–201 m², mean 102±20 m²). Secondary: **OSM Israel** (actual Israeli footprints via `building=residential/apartments` + `building:levels`, ODbL).

**B. Room Count / Room Type** — Best: **RPLAN** (80K, mean 2.48 bedrooms, 13 room categories). Secondary: **3D-FRONT** (14,629 rooms). Also **ResPlan** (typed adjacency graph).

**C. Room Dimensions / Aspect Ratios** — Best: **3D-FRONT** (room polygons in JSON). **Structured3D** (room cuboids). **RPLAN** (areas from pixel count).

**D. Wall Thickness** — **ResPlan** is the only large-scale set storing `wall_depth` explicitly (e.g., 0.15 m). **IFC-Bench / BIMNet** give actual material-layer thicknesses. For Israel, hardcode: 150 mm interior hollow block, 200 mm exterior, 200–300 mm RC columns/shear walls (see §4).

**E. Ceiling Height / Storey** — **Matterport3D** (`layoutHeight` ~2.98 m). **OSM/Overture** (`building:levels` + height). Israeli standard: 2.7–3.0 m clear per floor.

**F. Door and Window Placement** — **ResPlan** (coplanar door/window polygons). **ZInD** (real-world W/D/O measurements). **CubiCasa5K** (SVG annotations).

**G. Room Adjacency / Connectivity** — **ResPlan** (explicit NetworkX-format graph, typed edges). Best for layout generation.

---

## 3. Construction-Type / Regional Realism

### Why wall/floor material thickness matter for elastic-wave simulation

- Concrete (E ≈ 30 GPa, ρ ≈ 2400 kg/m³) → compressional wave speed ~3,500–4,000 m/s; low damping.
- Hollow concrete block infill (E ≈ 5–10 GPa, ρ ≈ 1000–1200 kg/m³) → lower wave speed, higher attenuation.
- Timber frame (E ≈ 12 GPa, ρ ≈ 500 kg/m³) → very different impedance/anisotropy; **all US-sourced dataset statistics are invalid for Israeli concrete buildings.**

Israeli residential defaults (hardcode into the generator):
- Interior partition: 150 mm hollow block + 10–15 mm plaster both sides → ~175–180 mm
- Exterior envelope: 200 mm hollow block + insulation + plaster → ~280–320 mm
- RC columns/shear walls: 200 mm × 500 mm, or 200–250 mm shear walls
- RC floor slabs: 180–220 mm (flat slab or joist+block waffle)
- Clear floor-to-floor: 2.9–3.2 m

### Dataset relevance by construction type

| Dataset | Timber-Frame Bias (US) | Concrete/Masonry Relevance |
|---|---|---|
| RPLAN | No | High — Chinese RC frame |
| 3D-FRONT | No | High — Chinese RC frame |
| ResPlan | Unknown | Moderate |
| CubiCasa5K | No | Moderate-High — Finnish masonry/RC |
| ProcTHOR-10k | Yes | Low |
| HM3D / Matterport3D | Partial | Low-Moderate |
| ZInD | Yes | Low |
| LIFULL HOME'S | No | High — Japanese RC frame (hard access) |
| OSM Israel | N/A | Direct (footprints only) |

---

## 4. Ranked Recommendations

### Tier 1 — Use these

**1. ResPlan** ([arXiv:2508.14006](https://arxiv.org/abs/2508.14006)) — only large set with explicit `wall_depth`; adjacency graph; permissive; direct 3D extrusion. Extract: wall-thickness distribution, room area per type, adjacency patterns, door/window dims. Caveat: single-floor; no ceiling height; region unstated.

**2. RPLAN** ([USTC DeepLayout](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html)) — 80,788 Chinese RC apartments ≈ Israeli morphology. Extract: footprint area distribution, bedroom-count histogram, room-type frequency, aspect ratio. Caveat: raster-only, no wall thickness at pixel level, non-commercial.

**3. OSM Buildings Israel + Overture** — actual Israeli footprints, `building:levels`, height; ODbL (commercial-compatible). Extract: footprint area distribution, storey counts, floor-to-floor height. Caveat: no interior/wall thickness.

### Tier 2 — Supplements

**4. 3D-FRONT** — 3D room polygons + furniture (door heuristics); Chinese/concrete; non-commercial.
**5. CubiCasa5K** (MIT) — only freely-licensed set with SVG wall polygons; Finnish masonry; full commercial use.
**6. BIMNet** — 25 IFC scans with layer-by-layer wall assemblies + measured thicknesses; validate wall priors.

### Tier 3 — Avoid or low priority

| Dataset | Reason |
|---|---|
| BuildingNet | Exterior shells only |
| HM3D / Matterport3D | Scan noise, not watertight, US-biased, non-commercial |
| Replica | Only 18 scenes |
| ProcTHOR-10k | US timber; wrong geometry for concrete sim |
| HSSD | 211 scenes; academic-only; US plans |
| LIFULL HOME'S | Raster only; NII academic access |
| Structured3D | Useful but 3D-FRONT is a better prior from same ecosystem |
| Google Open Buildings V3 | Does not cover Israel |
| ZInD | US-only; academic-only |

---

## 5. Practical Extraction Pipeline

```
RPLAN (80K plans)      → footprint area CDF, room count histogram, aspect ratio
ResPlan (17K, JSON)    → wall_depth dist (→ 0.15 m interior), room area per type,
                          door width dist, adjacency graph → layout Markov chain
OSM Israel Buildings   → footprint area dist (Israeli), building:levels (storeys),
                          height/levels ratio (→ floor-to-floor height)
BIMNet / IFC-Bench     → validate wall layer thicknesses (interior/exterior/RC slab)
Hardcode (Israeli std) → interior 180 mm, exterior 270 mm, slab 180–220 mm,
                          ceiling 2.70–2.80 m clear, floor-to-floor ~2.95–3.10 m
```

---

## Source URLs

- [3D-FRONT (Tianchi)](https://tianchi.aliyun.com/dataset/65347) · [arXiv](https://arxiv.org/abs/2011.09127) · [HF huanngzh](https://huggingface.co/datasets/huanngzh/3D-Front)
- [Structured3D](https://structured3d-dataset.org/) · [GitHub](https://github.com/bertjiazheng/Structured3D)
- [ProcTHOR-10K](https://github.com/allenai/procthor-10k) · [site](https://procthor.allenai.org/)
- [HM3D (Meta)](https://ai.meta.com/blog/introducing-the-habitat-matterport-3d-research-data-set-for-training-embodied-ai/) · [HM3D-Sem](https://aihabitat.org/datasets/hm3d-semantics/)
- [Matterport3D](https://github.com/niessner/Matterport) · [BuildingNet](https://buildingnet.org/) · [HSSD arXiv](https://arxiv.org/abs/2306.11290) · [HSSD site](https://3dlg-hcvc.github.io/hssd/)
- [Replica](https://github.com/facebookresearch/Replica-Dataset) · [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k)
- [RPLAN (USTC)](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) · [LIFULL HOME'S](https://www.nii.ac.jp/dsc/idr/en/lifull/) · [ZInD](https://github.com/zillow/zind)
- [ResPlan arXiv](https://arxiv.org/abs/2508.14006) · [HTML](https://arxiv.org/html/2508.14006v1)
- [IFC-Bench (HF)](https://huggingface.co/datasets/sylvainHellin/ifc-bench) · [GHS-OBAT](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12221504/)
- [OSM Israel Buildings (HDX)](https://data.humdata.org/dataset/hotosm_isr_buildings) · [Overture Buildings](https://gee-community-catalog.org/projects/overture_buildings/)
- [Hollow block / Israel seismic construction (Springer)](https://link.springer.com/article/10.1007/s10518-024-02054-0) · [European housing dimensions (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8073340/)
- [BlenderProc 3D-FRONT docs](https://dlr-rm.github.io/BlenderProc/examples/datasets/front_3d/README.html)
