> **Reference catalog — geophone hardware.** Research survey (Sonnet), 2026-07-09. Comprehensive hardware reference for sensor-model parameterization in two systems: (a) single-geophone human/vehicle/animal classifier (~5–250 Hz), (b) indoor house-localization array. Extends `research/geophone_sensitivity.md`, `research/mems_detection.md`, `research/complementary_sensors.md`, `research/house_localization/04_sensor_hardware_hybrid.md`. Indexed in README.md.

---

# Geophone Hardware Reference Catalog

## Scope and purpose

This document catalogs commercially available and scientifically deployed moving-coil geophones across the full frequency spectrum (1–100 Hz natural frequency) and sensitivity range (~11–260 V/m/s). Its purpose is to support **domain-randomized sensor modeling** for two simulation systems:

- **System A** — single-geophone seismic classifier (footstep / vehicle / animal), deployed outdoors in soil, band 5–250 Hz.
- **System B** — N-geophone house-localization array (TDOA on structural Lamb waves), band 5–250 Hz, sensors bonded to building slab/wall.

For both systems the sensor model must randomize over the real population of geophones that might be fielded, not just one nominal part. This catalog provides the empirical distributions to draw from.

---

## 1. Master Table — Real Geophone Models and Verified Specifications

Notes on table columns:
- **f₀** = natural frequency (corner frequency), Hz.  
- **G_oc** = open-circuit sensitivity, V/(m/s). Tolerance typically ±2.5–10% depending on manufacturer.  
- **R_coil** = coil DC resistance, Ω. Sets Johnson noise floor.  
- **ζ_oc** = open-circuit damping ratio (fraction of critical). Operational damping ζ ≈ 0.6–0.7 is achieved by adding a shunt resistor.  
- **Spur.** = lowest spurious (parasitic resonance) frequency, Hz. Usable band is roughly f₀ to 0.5 × spur.  
- **m_mov** = moving mass, g.  
- **Wt** = total element weight, g.  
- **Axes** = 1C (single component, usually vertical or horizontal) or 3C (orthogonal triaxial package).  
- **Price** = rough order-of-magnitude per element at low-to-mid volume. ¢ = cents, $ = ~$10–100, $$ = ~$100–500, $$$ = ~$500–2000.

### 1.1 Standard-sensitivity land geophones — SM-4/GS-20DX class (~25–40 V/m/s)

These are the workhorses of reflection seismic surveys. Coil resistance ~375–570 Ω, sensitivity ~27–40 V/m/s depending on frequency. The low coil resistance lowers Johnson voltage noise but the moderate sensitivity means a front-end amplifier with low current noise is still needed.

| Model | Mfr | f₀ (Hz) | G_oc (V/m/s) | R_coil (Ω) | ζ_oc | Shunt R (Ω) for ζ=0.6–0.7 | Distortion | Spur. (Hz) | m_mov (g) | Wt (g) | Dim (mm) | Axes | Price | Application |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SM-24** (ST-10N equiv.) | ION/Sercel/Seis-Tech | 10 | 28.8 | 375 | 0.25 | 1339 (→ζ=0.60) | ≤0.1% | >240 | 11.0 | 74 | 25.4 × 32 | 1C | $ | Land seismic reflection; SparkFun dev boards |
| **SM-4** equiv. ST-8N | Seis-Tech | 8 | 28.8 | 375 | 0.337 | — (→ζ=0.60) | ≤0.2% | ≥240 | 11.3 | 74 | 25.4 × 32 | 1C | $ | Shallow refraction |
| **SM-4** equiv. ST-10N | Seis-Tech | 10 | 28.8 | 375 | 0.271 | — (→ζ=0.60) | ≤0.2% | ≥240 | 11.3 | 74 | 25.4 × 32 | 1C | $ | Standard land survey |
| **SM-4** equiv. ST-14N | Seis-Tech | 14 | 28.8 | 375 | 0.200 | — (→ζ=0.60) | ≤0.2% | ≥240 | 11.6 | 74 | 25.4 × 32 | 1C | $ | Shallow high-res survey |
| **SM-4** equiv. ST-40N | Seis-Tech | 40 | 32.5 | 540 | 0.620 | — | ≤0.2% | ≥300 | 9.3 | 80 | 26 × 32 | 1C | $ | Coal-mine shallow survey |
| **GS-20DX** (10 Hz std.) | Geospace | 10 | 27.6 | 395 | 0.30 | ~1000 (→ζ=0.70) | ≤0.20% | >250 | 11.0 | 87 | 25.4 × 33 | 1C | $ | Reflection survey; externally damped for temp. stability |
| **GS-32CT** (equiv.) | Geospace/Seis-Tech | 10 | 27.5 | 395 | 0.316 | 1000 (→ζ=0.70) | ≤0.10% | ≥250 | 11.2 | 89 | 25.4 × 33 | 1C | $ | Close-tolerance land; same pin-out as GS-20DX |
| **GS-30CT** | Geospace | 10 | ~27.5 | 395 | ~0.32 | ~1000 | ≤0.03% | >250 | ~11 | ~87 | 25.4 × ~32 | 1C | $ | Predecessor to GS-32CT; very low distortion |
| **GS-14-L3** (28 Hz) | Geospace | 28 | 11.4 | 570 | 0.18 | N.S. | N.S. | N.S. | 2.15 | 19 | 16.8 × 17.3 | 1C | $ | Embedded/industrial vibration; very small can |
| **GS-14-L9** (28 Hz) | Geospace | 28 | 23.6 | 1500 | 0.31 | N.S. | N.S. | N.S. | 2.15 | 27 | 19 × 18 | 1C | $ | High-temp industrial; 1000 G shock rating |
| **ST-20DX 28 Hz** | Seis-Tech | 28 | 39.0 | 395 | 0.27 | 1000 (→ζ=0.55) | ≤0.2% | ≥350 | 11.0 | 100 | 25.4 × 33 | 1C | $ | Coal mine shallow; high spurious band |
| **ST-20DX 40 Hz** | Seis-Tech | 40 | 32.0 | 395 | ~0.35 | — | ≤0.2% | ≥450 | ~9 | ~95 | 25.4 × 33 | 1C | $ | High-res shallow survey |
| **ST-20DX 60 Hz** | Seis-Tech | 60 | ~28.0 | ~395 | ~0.40 | — | ≤0.2% | ≥500 | ~7 | ~95 | 25.4 × 33 | 1C | $ | Very shallow / engineering |
| **ST-20DX 100 Hz** | Seis-Tech | 100 | 23.0 | 570 | 0.45 | — | ≤0.2% | ≥600 | 5.0 | 95 | 27 × 33.5 | 1C | $ | Coal mine; highest-res shallow |

**Source notes:** SM-24 confirmed from ION/Sercel OEM datasheet (SparkFun CDN PDF) and Seis-Tech equivalent ST-10N page. GS-20DX from Geospace product page. GS-32CT from Seis-Tech page. GS-14 from Geospace GS-14 product page. ST-20DX series from Seis-Tech 20DX catalog page.

---

### 1.2 SM-6 / SM-6-class long-coil-travel geophones (~28–82 V/m/s, 4.5 Hz)

The SM-6 family (manufactured by ION/Sensor Nederland, cloned by many Chinese makers) differs from SM-4 by having a **longer coil travel (4 mm vs 1.5–2 mm)** and is available in both standard-sensitivity (B-coil, ~28.8 V/m/s / 375 Ω) and high-sensitivity (H-coil, ~80–82 V/m/s / 3400 Ω) configurations. The omni-directional variants tolerate any tilt up to ±360° and are used in horizontal/3C packages.

| Model | Mfr | f₀ (Hz) | G_oc (V/m/s) | R_coil (Ω) | ζ_oc | Distortion | Spur. (Hz) | m_mov (g) | Wt (g) | Dim (mm) | Axes | Coil excursion | Price |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SM-6 B-coil 4.5 Hz** | ION/Sensor Nederland (equiv.) | 4.5 | 28.8 | 375 | 0.60 | ≤0.2% | ≥160 | 11.3 | 86 | 25.4 × 36 | 1C | 4 mm p-p | $ |
| **SM-6 H-coil 4.5 Hz** (ST-4.5N equiv.) | Seis-Tech | 4.5 | 82.0 | 3400 | 0.58 | ≤0.3% | ≥90 | 11.3 | 80 | 25.4 × 36 | 1C | 4 mm p-p | $ |
| **SM-6 B-coil 10 Hz** | ION equiv. | 10 | ~28.8 | 375 | ~0.27 | ≤0.2% | ≥240 | ~11 | ~80 | 25.4 × 36 | 1C | 4 mm p-p | $ |
| **SM-6 H-coil 10 Hz** | Seis-Tech / ION | 10 | ~80 | ~3400 | ~0.55 | ≤0.2% | ≥160 | ~11 | ~82 | 25.4 × 36 | 1C | 4 mm p-p | $ |
| **SM-6 Omni 14 Hz** | ION (omni-directional) | 14 | ~28 | ~375 | 0.18–0.19 | ≤0.2% | ≥240 | ~11 | ~80 | 25.4 × 36 | 1C omni | 4 mm p-p | $ |

**Source notes:** SM-6 B-coil 4.5Hz from Seis-Tech SM-6 B-coil page. SM-6 H-coil (ST-4.5N H-B style) from Seis-Tech SM-6/H-B page. ION SM-6 omni specs from ION product page (documents.pub mirror). The SM-6 omni 14Hz open-circuit damping of 0.18–0.19 is intentionally low; designed to reach ζ≈0.6 with an external shunt.

---

### 1.3 GS-11D / GS-11D-class 4.5 Hz geophones (standard and high-sensitivity)

The GS-11D (OYO Geospace, now Geospace Technologies) is the canonical 4.5 Hz research-grade land geophone used in EarthScope/IRIS deployments. It is available in multiple coil-resistance options giving very different sensitivities. The 380 Ω variant is the "standard" (32 V/m/s); the 4000 Ω coil reaches ~97 V/m/s. The GS-ONE LF is the modern high-sensitivity replacement at 100.4 V/m/s.

| Model | Mfr | f₀ (Hz) | G_oc (V/m/s) | R_coil (Ω) | ζ_oc | Spur. (Hz) | m_mov (g) | Wt (g) | Dim (mm) | Axes | Price | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GS-11D** (380 Ω coil) | Geospace (OYO) | 4.5 | 32.0 | 380 | 0.34 | N.S. | 23.6 | 111 | 31.8 × 33.5 | 1C | $$ | IRIS/EarthScope standard; large moving mass gives low Brownian noise |
| **GS-11D** (4000 Ω coil) | Geospace (OYO) | 4.5 | 97.4 | 4000 | ~0.40 | N.S. | 23.6 | 111 | 31.8 × 33.5 | 1C | $$ | High-sensitivity variant; same mechanical can, more turns |
| **GS-ONE LF** (4.5 Hz) | Geospace | 4.5 | 100.4 | 2450 | 0.42 | ~120 | 25.2 | 131 | 30.5 × 40.7 | 1C | $$ | Modern high-sens. 4.5 Hz; spurious at only 120 Hz — limits high-f use |
| **GS-ONE LF** (5 Hz vert.) | Geospace | 5.0 | 100.4 | 2450 | 0.31–0.45 | ~160 | 25.2 | 131 | 30.5 × 40.7 | 1C | $$ | Higher spur. than 4.5Hz variant |
| **ST-4.5A** (high-sens.) | Seis-Tech | 4.5 | 100.0 | 3800 | 0.595 | ≥90 | 11.8 | 81 | 25.4 × 36 | 1C | $ | Compact clone of GS-ONE LF class; note lower spur. (90 Hz) |
| **Sunfull PS-4.5B** | Sunfull | 4.5 | 28.8 | 375 | ~0.60 | N.S. | 11.3 | ~80 | 25.4 × ~32 | 1C | $ | Raspberry Shake RS1D internal geophone |
| **RTC 4.5 Hz 375Ω** | R.T. Clark | 4.5 | ~28.8 | 375 | ~0.60 | N.S. | ~11 | ~75 | 25.4 × ~32 | 1C | $ | US-branded equivalent; vertical and horizontal available |
| **RGI-4.5Hz** | Racotech | 4.5 | ~28.8 | 375 | ~0.60 | N.S. | ~11 | ~75 | 25.4 × ~32 | 1C | $ | Chinese OEM; used in Raspberry Shake RS1D alternatives |

**Source notes:** GS-11D 380Ω from DESY vibration lab page and IRIS NRL. GS-11D 4000Ω from IRIS NRL variant table (97.39 V/m/s). GS-ONE LF from Geospace product page. ST-4.5A from Seis-Tech high-sensitivity 4.5 Hz page. Sunfull PS-4.5B from Raspberry Shake documentation.

---

### 1.4 Sercel / ION high-sensitivity scientific geophones (SG-5, SG-10, SG-10HS)

These use rare-earth magnets with more turns and heavier moving masses to achieve 80–86 V/m/s while maintaining the standard SM-4 form factor diameter (32 mm). They represent the high end of the standard geophone family for seismic surveys requiring better noise floor.

| Model | Mfr | f₀ (Hz) | G_oc (V/m/s) | R_coil (Ω) | ζ_oc | Distortion | Spur. (Hz) | m_mov (g) | Wt (g) | Dim (mm) | Axes | Price | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SG-5** | Sercel | 5.0 | 80.0 | 1850 | 0.60 | ≤0.075% | ≥150 | 22.7 | 170 | 32 × 43 | 1C | $$ | Very low distortion; Sercel's 5 Hz high-sens flagship |
| **SG-10** (std.) | Sercel | 10 | ~28.8 | ~375 | ~0.25 | ≤0.1% | >240 | ~11 | ~80 | ~25.4 × 32 | 1C | $ | Standard 10 Hz; rare-earth magnet vs SM-24 |
| **SG-10HS** | Sercel | 10 | 85.8 | 1800 | 0.56 | ≤0.10% | >240 | ~14 | ~120 | ~27–30 × ~35 | 1C | $$ | High-sensitivity 10 Hz; same spec as GS-ONE 10Hz |
| **SmartSolo DT-SOLO 5Hz** | DTCC/SmartSolo | 5.0 | 80.0 | 1850 | 0.60 | ≤0.10% | >170 | 22.7 | 170 | 32 × 43 | 1C | $$ | IGU-16 nodal system internal sensor; certified per SG-5 spec |
| **SmartSolo DT-SOLO 10Hz** | DTCC/SmartSolo | 10 | 85.8 | 1800 | 0.56 | ≤0.10% | >240 | ~14 | ~120 | ~27 × ~35 | 1C | $$ | IGU-16 nodal system 10Hz option |

**Source notes:** SG-5 confirmed from Sercel official specifications PDF (via search). SG-10HS from search result confirmed specs. SmartSolo IGU-16 from product page — lists 5Hz 80 V/m/s and 10Hz 85.8 V/m/s. Note SG-5 coil travel is only 3 mm p-p vs SM-6's 4 mm.

---

### 1.5 GS-ONE (10 Hz high-sensitivity) — Geospace flagship for vibration monitoring

| Model | Mfr | f₀ (Hz) | G_oc (V/m/s) | R_coil (Ω) | ζ_oc | Distortion | Spur. (Hz) | m_mov (g) | Wt (g) | Dim (mm) | Axes | Price | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GS-ONE** (10 Hz, vert.) | Geospace | 10 | 85.8 | 1800 | 0.48–0.54 | ≤0.05% | >240 | 14 | 130 | 30.5 × 40.7 | 1C | $$ | Designed to replace 3×2 or 6×1 arrays; very low distortion |
| **GS-ONE** (10 Hz, OMNI) | Geospace | 15 | 69.2 | 1800 | 0.70 | ≤0.20% | >160 | 14 | 130 | 30.5 × 40.7 | 1C omni | $$ | OMNI config: any-tilt use; raised f₀ and lower spur. penalty |
| **ST-10PA** (GS-ONE equiv.) | Seis-Tech | 10 | 85.8 | 1800 | 0.48–0.54 | <0.10% | >240 | 14 | 104 | 27 × 34 | 1C | $ | Clone of GS-ONE 10Hz spec; compact housing |

**Source notes:** GS-ONE vert. from Geospace product page. GS-ONE OMNI from same page. ST-10PA from Seis-Tech high-sensitivity 10Hz page.

---

### 1.6 Low-frequency geophones / short-period seismometer boundary (1–4 Hz)

Below ~4 Hz the term "geophone" blurs into "short-period seismometer." Physically these use the same moving-coil principle but with much heavier moving masses (60–770 g vs 11–25 g for standard geophones), longer coil travel, and much higher sensitivity (200–276 V/m/s). They are larger, heavier, and more expensive.

| Model | Mfr | f₀ (Hz) | G_oc (V/m/s) | R_coil (Ω) | ζ_oc | Spur. (Hz) | m_mov (g) | Wt (g) | Dim (mm) | Axes | Price | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ST-2A** | Seis-Tech | 2.0 | 260 | 6400 | 0.70 | ≥90 | 60 | 264 (vert) | 38 × 47 | 1C | $$ | Ultra-high sens. for deep seismic; long coil 3 mm p-p |
| **ST-1N** | Seis-Tech | 1.0 | 200 | 3300 | 0.36 | N.S. | 770 | 1400 | 65 × 106 | 1C | $$ | Bridges geophone/seismometer; huge mass for low Brownian noise |
| **HS-1** (2 Hz std.) | Geospace | 2.0 | 18.1 | 225 | 0.61 | N.S. | 23 | 247 | 41 × 51 | 1C | $$ | Lower sensitivity but good coupling; 460–2000 mV/ips options |
| **GS-1** (1.0 Hz) | Geospace | 1.0 | ~75–480 | varies | N.S. | N.S. | large | 1960 | 76 × 164 | 1C | $$$ | True short-period seismometer; 6.35 mm coil travel |
| **L-4C** (1 Hz) | Sercel/Mark Products | 1.0 | 83.5–276.8 | 500–5500 | 0.28 (oc) | N.S. | 500–1000 | 1700–2150 | 76 × 130 | 1C | $$$ | LIGO seismic isolation sensor; studied Johnson-noise limit |

**Source notes:** ST-2A from Seis-Tech 2Hz low-frequency page. ST-1N from Seis-Tech 1Hz page. HS-1 from Geospace HS-1 page. GS-1 from Geospace GS-1 page. L-4C from IRIS Sercel NRL entry and LIGO noise study (arXiv:1711.05439).

**The geophone/seismometer boundary:** There is no sharp physical boundary — the same electromagnetic principle operates across all frequencies. The practical distinction is:
- **Geophones** (typically ≥4.5 Hz): compact (25–32 mm diameter), light (74–170 g), suitable for dense arrays, spike-coupled into soil, tolerant of mild misorientation.
- **Short-period seismometers** (1–4 Hz): large (40–76 mm diameter), heavy (250 g – 2+ kg), require careful leveling, vault installation preferred, individually calibrated.
- The crossover is conventionally at ~2–4 Hz. Between 4–4.5 Hz (e.g., GS-11D at 4.5 Hz, ST-2A at 2 Hz) both labels are used.

---

### 1.7 High-frequency geophones — 14 Hz, 28 Hz, 40 Hz, 60 Hz, 100 Hz

| Model | Mfr | f₀ (Hz) | G_oc (V/m/s) | R_coil (Ω) | ζ_oc | Spur. (Hz) | m_mov (g) | Wt (g) | Application | Axes |
|---|---|---|---|---|---|---|---|---|---|---|
| **ST-14N** (SM-4 equiv.) | Seis-Tech | 14 | 28.8 | 375 | 0.20 | ≥240 | 11.6 | 74 | Shallow refraction, medium depth | 1C |
| **ST-20DX 14Hz** | Seis-Tech | 14 | ~28.8 | 395 | ~0.30 | ≥300 | ~11 | ~90 | 2D/3D reflection | 1C |
| **HS-1 14 Hz** option | Geospace | 14 | ~18 | 225 | ~0.60 | N.S. | ~23 | ~247 | Custom HS-1 can; industrial | 1C |
| **GS-14-L3** | Geospace | 28 | 11.4 | 570 | 0.18 | N.S. | 2.15 | 19 | Compact embedded sensing | 1C |
| **GS-14-L9** | Geospace | 28 | 23.6 | 1500 | 0.31 | N.S. | 2.15 | 27 | High-temp industrial | 1C |
| **ST-20DX 28Hz** | Seis-Tech | 28 | 39.0 | 395 | 0.27 | ≥350 | 11.0 | 100 | Coal mine survey | 1C |
| **Seis-Tech 28Hz reinf.** | Seis-Tech | 28 | 28–39 | 395 | 0.27–0.55 | ≥350 | ~11 | ~100 | Shallow coal survey; land case | 1C |
| **High-sens. 14Hz** | Seis-Tech | 14 | ~80 | ~1800 | ~0.55 | ≥160 | ~14 | ~110 | Vibration monitoring | 1C |
| **Omni 14Hz** | Seis-Tech | 14 | ~80 | ~1800 | ~0.55 | ≥160 | ~14 | ~110 | Omni-directional / 3C use | 1C omni |
| **ST-20DX 40Hz** | Seis-Tech | 40 | 32.0 | ~395 | ~0.35 | ≥450 | ~9 | ~95 | Engineering, NDT | 1C |
| **ST-20DX 60Hz** | Seis-Tech | 60 | ~28 | ~395 | ~0.40 | ≥500 | ~7 | ~95 | Shallow | 1C |
| **ST-20DX 100Hz** | Seis-Tech | 100 | 23.0 | 570 | 0.45 | ≥600 | 5.0 | 95 | Coal mine, NDT | 1C |
| **Seis-Tech 100Hz** | Seis-Tech | 100 | 23.0 | 570 | 0.45 | ≥600 | 5.0 | 95 | Highest res. shallow | 1C |

**Notes on high-frequency geophones for our applications:** 28–100 Hz geophones are **not suitable** for footstep detection (4.5–250 Hz). They cut off all energy below their f₀ — a 28 Hz unit sees only 28–350 Hz, missing the dominant 10–80 Hz footstep band. Their only relevance is if used as a *high-frequency supplement* above 100 Hz (rare) or in a multi-sensor stack. Avoid for System A and System B.

---

### 1.8 Borehole / downhole 3C geophones

Borehole geophones are armored in a pressure casing (typically 38–60 mm outer diameter), with an electromechanical wall-lock arm to clamp against the borehole wall. They are used in VSP, crosshole, and microseismic monitoring. Same moving-coil physics; rugged sealing; often higher operating temperature.

| Model | Mfr | f₀ (Hz) | G_oc (V/m/s) | R_coil (Ω) | ζ_oc | m_mov (g) | Housing OD (mm) | Temp. (°C) | Axes |
|---|---|---|---|---|---|---|---|---|---|
| Borehole 3C 10Hz (ST-BH10-3C) | Seis-Tech | 10 | 10.0 (=1000 mV/cm/s) | 3400 | 0.55 | ~13 | 60 | −40 to +70 | 3C |
| Borehole 3C 2Hz | Seis-Tech | 2.0 | 20.0 (=2 V/m/s) | 6040 kΩ (error; likely 6040 Ω?) | 0.70 | 60 | 38.5 | −25 to +55 | 3C |
| Borehole 3C 4.5–100 Hz variants | Seis-Tech | 4.5/10/14/28/40/100 | varies by freq | varies | 0.5–0.7 | 5–60 | 38.5–60 | −40 to +70 | 3C |
| Wall-lock borehole 3C (BHG-2) | Geostuff | 10 or 28 | ~28 | ~395 | ~0.60 | ~11 | 46 | −20 to +120 | 3C |
| HS-1 3C (surface package) | Geospace | 2.0 | 26.4 (1.04 V/in/s) | 1250 | N.S. | ~23 × 3 | package: 149 mm | −40 to +100 | 3C |

**Note on borehole sensitivity units:** Borehole datasheets sometimes state sensitivity as mV/cm/s or V/cm/s rather than V/m/s. Conversion: 1 V/cm/s = 100 V/m/s; 1000 mV/cm/s = 10 V/m/s.

**Source notes:** Borehole 3C 10Hz from Seis-Tech borehole triaxial 10Hz page. Borehole 3C variants from Seis-Tech borehole 3C page. HS-1 3C from Geospace HS-1 3C page.

---

### 1.9 Nodal system geophones (integrated sensor + ADC + battery)

These are complete acquisition nodes that integrate the geophone element, 24–32-bit ADC, battery, GPS, and storage. The sensor inside is the same moving-coil element; the system specs determine usable dynamic range and timing precision.

| System | Mfr | Internal sensor | f₀ (Hz) | G_oc (V/m/s) | ADC bits | Sample rates | Weight (system) | Axes | Price |
|---|---|---|---|---|---|---|---|---|---|
| **SmartSolo IGU-16** | DTCC | DT-SOLO (SG-5 class) | 5 or 10 | 80 (5Hz) / 85.8 (10Hz) | 32 | 500–2000 SPS | 1.1 kg | 1C | $$$ |
| **SmartSolo IGU-16HR 3C** | DTCC | DT-SOLO (3 elements) | 5 | 80 | 32 | 500–2000 SPS | ~1.5 kg | 3C | $$$+ |
| **Raspberry Shake RS1D** | RS | Sunfull PS-4.5B | 4.5 | ~28.8 | 24 | 100 SPS | ~200 g | 1C | $$ |
| **Raspberry Shake RS4D** | RS | Sunfull PS-4.5B (1×V) + MEMS (3C) | 4.5 geo / DC MEMS | 28.8 (geo) / 25 µg/√Hz (MEMS) | 24 | 100 SPS | ~250 g | 1C geo + 3C MEMS | $$ |

**Source notes:** SmartSolo from product page. RS1D/RS4D from Raspberry Shake documentation and mems_detection.md.

---

## 2. Three-Component (3C) Geophones — Dedicated Section

### 2.1 What 3C buys

A single vertical geophone gives only a scalar velocity measurement. Three orthogonal elements (one vertical Z + two horizontal X, Y) add:
- **Polarization / particle motion**: P-wave arrivals are longitudinal (radial direction), S-waves are transverse — the 3C particle-motion ellipse reveals the **bearing to the source** from a single station.
- **S-wave separation**: P vs S arrival time difference provides range estimate without knowing velocity.
- **Horizontal ground motion**: relevant for structural / lateral footstep energy, which couples strongly into horizontal plate bending modes.
- **First-arrival direction**: STA/LTA onset pick on Z-channel for TDOA; H-channels for direction.

For the **localization array (System B)**: each node is a vertical geophone for onset timing + either (a) the same node has horizontal geophones from a 3C package for bearing, or (b) TDOA across the N-node array substitutes for single-node bearing. A 3C node at each corner of a 4-corner array provides both TDOA and per-node bearing — overdetermined, robust.

For **System A** (single-sensor classifier): adding horizontal channels enriches the feature set (quadrupeds show different H/V ratio signatures than bipeds), but the binding limit is detection range (which is set by ambient noise, not axis count).

### 2.2 Real 3C geophone models

| Model | Mfr | f₀ (Hz) | G_oc each element (V/m/s) | R_coil each (Ω) | Package weight (kg) | Package dim (mm) | Mounting | Price | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **4.5 Hz 3C** (ST-4.5N case) | Seis-Tech | 4.5 | 28.8 | 375 | ~0.30 | case + spikes | Spike into soil or bonded | $ | 3× SM-4 B-coil in sealed case with 75mm steel spikes |
| **GS-1 3C Seismonitor** | Geospace | 1.0 | ~75–480 | 450 / 4550 / 17400 | ~3 kg | Al case, carry handle | Leveled, IP67 | $$$ | Scientific portable seismograph; up to 100 Hz BW |
| **HS-1 3C** | Geospace | 2.0 | 26.4 (1.04 V/in/s) | 1250 or 3810 | 3.6 kg | 149 mm dia | Leveled, IP67 | $$$ | Seismological; 500 Hz BW |
| **SmartSolo IGU-16HR 3C** | DTCC | 5.0 | 80.0 | 1850 | 1.5 kg | 95×103×118 mm | Spike | $$$ | Fully integrated nodal 3C; self-contained |
| **Borehole 3C 10Hz** | Seis-Tech | 10 | 10.0 | 3400 | — | 60 mm OD | Wall-lock in borehole | $$$ | VSP / microseismic / crosshole |
| **Borehole 3C 4.5 Hz** | Seis-Tech | 4.5 | 28.8 | 375 | — | 60 mm OD | Wall-lock | $$$ | Borehole VSP |
| **SeniMax 3D Wireless** | Resensys | 14–40 Hz (typical) | ~25–35 | ~395 | 0.7 kg | 90 × 70 mm | Surface / embedded | $$$+ | Integrated 3C + wireless; structural health monitoring |

### 2.3 Cost/size penalty of 3C vs 1C

- **Cost**: a 3C package costs roughly 3× the single element cost (3 elements) + case + connector + cable. Typical 3C packages run $100–500 vs $10–80 for a 1C.
- **Size/weight**: 3C packages are 2–5× heavier than a 1C element.
- **Leveling requirement**: all elements must be within ~10–20° of their design orientation; horizontal elements require separate leveling. This makes borehole 3C and scientific 3C installations labor-intensive.
- **For arrays**: if TDOA from N nodes provides bearing, individual 1C vertical geophones are sufficient and cheaper. If N < 3 or the geometry is poor, per-node 3C bearing is valuable.

---

## 3. The Spec Dimensions That Matter and Why

### 3.1 Corner frequency f₀

The geophone has a flat velocity response **above f₀** and rolls off at 12 dB/octave **below f₀**. For footstep detection in the 5–250 Hz band:
- **4.5 Hz**: captures everything — floor bending modes (5–8 Hz), low-frequency heel-strike energy (10–20 Hz), and high-frequency structural resonances (50–250 Hz). Best choice.
- **10 Hz**: misses 5–10 Hz floor modes; acceptable for TDOA on arrivals above 10 Hz.
- **14/28 Hz**: misses the most energetic part of the structural footstep signature. Avoid unless high-frequency surface-wave dispersion is the only target.
- **1–2 Hz**: overkill for footsteps; used when building sway / very low structural modes need capturing. Larger, heavier, more expensive.

The **self-noise** of a geophone is flat in velocity above f₀ but rises steeply below (see §4.2). So a lower f₀ also means lower acceleration noise at frequencies above f₀ (the noise converts from velocity to acceleration as v_noise / (2π f)).

### 3.2 Sensitivity G (V/m/s)

Sensitivity is the transduction gain: G = (output voltage) / (velocity). A higher G means more output voltage for the same ground motion. This affects:
- **Minimum detectable signal** relative to electronics noise (preamplifier input voltage noise, not geophone thermal noise). With G = 28.8 V/m/s and front-end 10 nV/√Hz amplifier noise, the amplifier contributes ~0.35 nm/s/√Hz equivalent velocity noise. With G = 100 V/m/s the amplifier contributes ~0.1 nm/s/√Hz — a 3.5× improvement, but both are well below the geophone's own thermal noise floor (~0.5 nm/s/√Hz above f₀).
- **Headroom / clipping**: higher G → lower clip level (output saturates at lower velocity). For footstep amplitudes of 1–10 µm/s at range, this is never the binding constraint in outdoor soil. For a near-field node at 0.5 m from footfall on concrete, velocities can be mm/s — a 100 V/m/s geophone delivers 0.1–10 V peak, which can clip a 24-bit ADC with ±2V input range.
- **Johnson noise contribution**: high-G geophones have high coil resistance (G scales roughly with turns count, R scales with turns²). Higher R_coil means higher Johnson voltage noise but also higher G, so the signal-to-noise ratio of the coil itself (its equivalent noise velocity) does not improve proportionally. The Johnson noise floor in velocity terms is approximately sqrt(4 k_B T R_coil) / G. For R=375Ω, G=28.8: ~0.89 nm/s/√Hz. For R=1850Ω, G=80: ~0.79 nm/s/√Hz. For R=2450Ω, G=100: ~0.73 nm/s/√Hz. The improvement is modest (~20%) — the dominant thermal noise term is the **mechanical Brownian noise** of the moving mass, which is independent of coil winding.

**Key insight**: Sensitivity is largely a gain factor. Two geophones with the same f₀, same moving mass, same spring, but different coil windings (different G, different R_coil) have nearly the same **equivalent noise velocity**. The differences in f₀, moving mass, and spring design are what actually change the noise floor.

### 3.3 Coil resistance R_coil

- Sets the **Johnson voltage noise**: V_J = sqrt(4 k_B T R) ≈ 4 nV/√Hz at 300K for R=1kΩ; 2 nV/√Hz for R=250Ω.
- Determines the **shunt resistor needed** to reach ζ=0.6–0.7: R_shunt ≈ (damping constant) / (ζ_target - ζ_oc). Higher R_coil requires higher R_shunt, which is fine — it just sets loading.
- Affects **cable loading**: high R_coil geophones (1800–6400 Ω) are more susceptible to stray capacitance of long cables (creates a pole in the high-frequency response).

### 3.4 Damping ratio ζ

Open-circuit ζ_oc is always below 0.5 for standard geophones (typically 0.2–0.6). Operational ζ ≈ 0.6–0.7 is obtained by shunting with R_ext. The importance:
- **Flat response**: ζ ≈ 0.7 gives flattest amplitude response above f₀ (within ±3 dB from ~1.1 f₀ upward). Lower ζ produces a resonant peak near f₀.
- **Onset picking**: a lightly damped geophone (ζ < 0.5) rings at f₀ after an impulse — this creates **waveform artifacts** after a footstep impulse that blur STA/LTA onset picks. Damping to ζ ≈ 0.6–0.7 suppresses this.
- **Domain randomization**: ζ should be sampled around 0.6–0.7 for a properly installed/shunted geophone; for an unshunted or poorly configured sensor, ζ can be 0.25–0.50 (the open-circuit values). Both cases should be modeled.

### 3.5 Self-noise (acceleration equivalent)

Geophone self-noise has two components:
1. **Mechanical (Brownian/thermal) noise** of the moving mass suspension: S_accel ≈ 4 k_B T (2π f₀)² / (m × Q) ≈ constant above f₀ in velocity; rises as f² in acceleration below f₀. This dominates at low frequencies near f₀. Larger moving mass m reduces this noise.
2. **Electrical (Johnson) noise** of the coil: converts to velocity noise as V_J / G. Dominates at mid-to-high frequencies above f₀.

Published measured values:
- **L-4C (1 Hz, 500 g mass)**: measured ~10⁻¹¹ m/√Hz displacement at 1 Hz (~6×10⁻¹¹ m/s/√Hz velocity), near Johnson noise limit (arXiv:1711.05439, LIGO AEI huddle test).
- **SM-24 class (10 Hz, 11 g mass)**: equivalent acceleration noise ~10 ng/√Hz above 10 Hz (PMC11589752; CREWES thesis).
- **GS-11D class (4.5 Hz, 23 g mass)**: similar acceleration noise but effective down to 4.5 Hz; ~5–10 ng/√Hz in 5–30 Hz band.
- **High-sensitivity 4.5/5 Hz (25 g mass, G=80–100 V/m/s)**: Brownian noise reduced ~2× vs 11 g standard; effective noise ~4–7 ng/√Hz.
- **Optomechanical MEMS** (2024, Nature): 2.5 ng/√Hz at 100–200 Hz — best reported for geophone class (PMC11589752).

At mid-band (10–50 Hz), all standard moving-coil geophones are dominated by **ambient ground noise** (50–240 nm/s or ~50,000–240,000 ng/s/√Hz at typical outdoor sites), which is 10–50× above the geophone noise floor. Self-noise is only the binding limit in a vault/borehole site on very quiet ground.

### 3.6 The crossover to MEMS accelerometers

The practical crossover from geophone to MEMS depends on application:
- **Self-noise crossover**: ADXL355 at 25 µg/√Hz ≈ 245,000 ng/√Hz. A standard geophone ~10 ng/√Hz. The geophone wins by ~4 orders of magnitude in self-noise at low frequencies. Even the best commodity MEMS do not catch up in the 5–50 Hz band.
- **Indoor strong-motion crossover**: at 5 m indoor on concrete, footstep velocities are 100–10,000 nm/s — both sensors are far above self-noise. MEMS is adequate, geophone is better but not necessary.
- **DC / tilt**: MEMS wins — no corner frequency, flat from DC. Geophone is AC-coupled.
- **3-axis**: MEMS is naturally 3-axis for pennies. A 3C geophone package is 3× size/cost.
- **Practical verdict**: use geophone for detection range and low-noise onset timing; use MEMS for 3-axis bearing, DC tilt, and strong-motion anti-clip. This is the RS4D hybrid architecture.

---

## 4. Sensitivity vs the Rest — What Is Really Independent

**Sensitivity (G, V/m/s) is primarily a gain factor.** It can be changed by varying the number of coil turns without changing f₀, moving mass, or spring. Two geophones with the same mechanical design but different coil windings have:
- Different G (proportional to turns)
- Different R_coil (proportional to turns²)
- Essentially the same mechanical Brownian noise floor (independent of coil)
- Different Johnson noise voltage (proportional to √R), but when referred to velocity units the improvement is partially offset by the higher G: net velocity noise ≈ √(4kT R) / G ≈ constant (scales only weakly with winding).

**Parameters that are NOT gain factors** (they change the underlying physics):
- **f₀**: changes where the response begins; changes the Brownian noise floor frequency profile; changes the transient ringing duration.
- **Moving mass m**: heavier mass lowers Brownian noise (S_Brownian ∝ 1/m), lowers f₀ (f₀ ∝ 1/√m), and makes the sensor physically larger.
- **Damping ζ**: changes flatness of response, resonance behavior, and transient ring-down.
- **Spurious frequency**: hard upper limit of flat-response bandwidth.

### 4.1 Realistic parameter ranges for domain randomization

Based on the full catalog above, here are the distributions a domain-randomization model should sample from, for a geophone that could plausibly be used in System A or System B:

**Corner frequency f₀:**
- Footstep-appropriate range (usable): 1–14 Hz (with 4.5 Hz most common, 10 Hz common, 1–4 Hz heavy seismometer regime)
- Recommended simulation range: **f₀ ~ Uniform[4.5, 10] Hz** for standard deployments; extend to [2, 14] Hz for sensitivity analysis
- Note: 14–100 Hz geophones are excluded as they miss the footstep band

**Open-circuit sensitivity G_oc:**
- Standard (375 Ω coil): 27–39 V/m/s (varies with f₀; higher f₀ can have slightly higher G due to mechanical design)
- High-sensitivity coil (1800–3800 Ω): 80–100 V/m/s
- Low-frequency (1–2 Hz, heavy): 18–280 V/m/s
- **Recommended simulation range**: **G_oc ~ LogUniform[28, 100] V/m/s** (covers standard through high-sens 4.5–10 Hz class)

**Operational damping ζ (with shunt resistor):**
- Well-configured: 0.60–0.70
- Poorly configured or open-circuit: 0.25–0.50
- **Recommended simulation range**: **ζ ~ Uniform[0.55, 0.72]** for properly shunted; extend to [0.25, 0.70] for adversarial/poorly-configured scenarios

**Coil resistance R_coil:**
- Standard: 375–570 Ω (SM-4/SM-24/GS-20DX class)
- High-sensitivity: 1800–3800 Ω (GS-ONE, SG-5, SM-6 H-coil)
- Very high (heavy seismometers): 500–6400 Ω
- **Recommended simulation range**: **R_coil ~ LogUniform[375, 2450] Ω**

**Self-noise (equivalent acceleration, mid-band 10–50 Hz):**
- Standard geophone (375 Ω, 11 g mass): ~8–15 ng/√Hz
- High-sensitivity (1800 Ω, 14–25 g mass): ~4–8 ng/√Hz
- Heavy seismometer (500–5500 Ω, 500–1000 g mass): ~0.5–3 ng/√Hz
- Note: for outdoor deployment the ambient noise floor (~50,000–240,000 ng/s/√Hz equivalent) completely dominates; self-noise is irrelevant outdoors.
- **Recommended simulation range**: **a_noise ~ LogUniform[4, 15] ng/√Hz** (geophone class); note this is rarely the binding parameter in simulation since ambient noise exceeds it by 4 orders of magnitude

**Spurious frequency f_spur:**
- Standard: ≥160–300 Hz
- High-sensitivity (heavy coil): ≥90–160 Hz (lower spur. is the penalty for high G)
- Compact: ≥240–600 Hz
- **Recommended range**: **f_spur ~ Uniform[120, 300] Hz** for 4.5–10 Hz footstep-band geophones

**Moving mass m_mov:**
- Standard SM-4/SM-24 class: 9–12 g
- High-sensitivity GS-ONE/SG-5 class: 14–25 g
- Heavy seismometer: 23–1000 g
- **Footstep-band geophones**: **m_mov ~ Uniform[11, 25] g**

---

## 5. Recommendations for Our Two Use-Cases

### 5.1 System A: Single-sensor seismic classifier (outdoor soil)

**Goal**: detect and classify human / vehicle / animal footstep signatures, 5–250 Hz band, buried in soil.

**Primary recommendation — 4.5 Hz geophone:**
- **Best value**: any SM-6 B-coil 4.5 Hz clone (~$10–25) or GS-11D class (~$50–100). Captures the full footstep band. The SM-4/SM-24 at 10 Hz is acceptable but loses the 5–10 Hz floor-mode band and lower-frequency long-range content.
- **Sensitivity**: the 28.8 V/m/s standard vs 80–100 V/m/s high-sensitivity difference matters **only if** front-end amplifier noise is the binding limit — which it is not at typical outdoor sites (ambient noise dominates). Either works.
- **Coil resistance**: 375 Ω standard is better for long cable runs (lower impedance). High-sensitivity 3400 Ω is better for very short cable runs with high-quality preamps.
- **Damping**: shunt to ζ ≈ 0.6–0.7 for clean impulse response.

**Tier 1 — Standard/budget ($10–25 per element):**  
SM-24 / SM-4 B-coil 4.5 Hz (any SM-6 B-coil clone or Seis-Tech ST-4.5N). Used in virtually all oil/gas seismic surveys. Adequate noise floor for outdoor footstep detection — ambient noise limits range, not the sensor.

**Tier 2 — High-sensitivity ($30–80 per element):**  
GS-ONE LF 4.5 Hz (100.4 V/m/s), SG-5 (80 V/m/s, 5 Hz), or ST-4.5A (100 V/m/s). Useful when: (a) the deployment site is exceptionally quiet (night, rural, vaulted), so the geophone noise floor starts to matter; (b) the front-end ADC has poor input-referred noise and the gain margin is needed; (c) very small footstep signals (soft gait, dry loose soil) are expected.

**3C for System A**: adds horizontal channels for gait classification (lateral sway signatures differ between human/animal/vehicle). **Worth adding** if the SNN has horizontal-channel inputs trained. The ADXL355 MEMS (25 µg/√Hz) is a cheaper alternative to a 3C geophone for horizontal bearing — adequate at outdoor ranges where the signal is strong (see RS4D hybrid architecture in `04_sensor_hardware_hybrid.md`).

**Not recommended for System A:**
- 10 Hz or higher (misses 5–10 Hz content)
- 1–2 Hz heavy seismometers (overkill, fragile, orientation-sensitive)
- 28–100 Hz geophones (miss most of the footstep band)

### 5.2 System B: House-localization array (N sensors on building)

**Goal**: TDOA-based localization of indoor footsteps on structural Lamb waves, 5–250 Hz band, sensors bonded to slab/wall.

**Primary recommendation — 4.5 Hz geophone (bonded):**
- Same rationale as System A but now the coupling to the structure (epoxy/stud) is critical — uncoupled (free-standing) sensors give poor high-frequency coupling above ~50 Hz.
- **4.5 Hz** strongly preferred over 10 Hz because indoor floor bending modes at 5–8 Hz carry strong structural energy from footsteps (Ekimov & Sabatier 2006).
- **Orientation**: vertical geophone detects normal-to-surface (out-of-plane) structural velocity; the dominant Lamb-wave mode for a floor slab couples strongly to vertical.

**Number of sensors**: minimum 3, recommended 4–6 for a house (see `04_sensor_hardware_hybrid.md` §3).

**Per-node hybrid**: geophone (vertical, lowest noise for TDOA onset) + ADXL355 (3-axis MEMS, horizontal components for bearing, DC tilt, anti-clip). This is the RS4D architecture validated at >1000 nodes. Cost: +$25–65/node.

**Which 4.5 Hz geophone for System B:**

| Tier | Model | Cost/sensor | Notes |
|---|---|---|---|
| Budget | Seis-Tech 4.5Hz B-coil (ST-4.5N) / SM-6 clone | $10–25 | Adequate; bond to slab; 375Ω for standard preamp |
| Research | GS-ONE LF 4.5Hz | $80–150 | 100 V/m/s, 25 g mass, lower noise floor; better for quiet building at night |
| Research | SG-5 (5 Hz) | $100–200 | 80 V/m/s, 5 Hz; slightly higher f₀ but lower spur. risk |

**Is a 3C geophone per node worth it for System B?**  
Only if the array has fewer than 4 nodes. With 4+ sensors the TDOA geometry already provides 2D position from the vertical-only channels. Single-node bearing from H-channels is a useful redundancy check but not essential. The ADXL355 MEMS (3-axis, $25–65) provides horizontal bearing more cheaply than a 3C geophone package ($100–500). **Recommendation: 1C geophone + ADXL355 per node**, not a 3C geophone package.

**ADC for System B**: replace ADS1115 with MCP3912 (4-channel simultaneous 24-bit, ~$6) or similar — see `04_sensor_hardware_hybrid.md` §5.

---

## 6. Honest Gaps

The following specs were not found in publicly accessible datasheets and are flagged:

| Gap | Models affected | What is unknown | Impact |
|---|---|---|---|
| GS-11D / GS-20DX / GS-32CT **distortion and spurious frequency** | GS-11D (all variants), GS-20DX | Geospace datasheets accessed via PDF were binary-encoded; distortion and spur. not extracted from product pages | Moderate — these are needed to verify the upper usable bandwidth. Industry rule of thumb: spur. ≥ 250 Hz for these models (from equivalent model specs). |
| **GS-11D open-circuit damping** exact value | GS-11D 380Ω | DESY page gives "damping constant 762" (in Ω units, not the ratio) and oc damping 0.34 ±20% | Low — 0.34 oc is consistent with the standard geophone family; shunting to 0.7 is straightforward |
| **SM-7 specifications** | SM-7 (Sensor Nederland/Sercel) | No datasheet found; SM-7 appears to be a lower-frequency variant of the SM-6 family, possibly 2 Hz | Moderate — SM-7 may be equivalent to ST-2A class specs; treat as 2 Hz / 260 V/m/s proxy |
| **SG-10 (standard) exact specs** | SG-10 (non-HS variant) | Sercel catalog PDF was binary-encoded; SG-10 standard sensitivity not confirmed | Low — likely ~28.8 V/m/s / 375 Ω, same as SM-24 |
| **GS-30CT distinct specs vs GS-32CT** | GS-30CT | Confirmed to be predecessor of GS-32CT with same footprint; distortion spec is <0.03% (better than 32CT). Full datasheet not extracted. | Low |
| **Borehole 3C coil resistance clarification** | Seis-Tech borehole 3C 2Hz | Page states "6040 kΩ" which is almost certainly a typo for 6040 Ω. 2 Hz, 260 V/m/s, 6040 Ω would be consistent with ST-2A | Low |
| **Self-noise measured values for GS-ONE LF and SG-5** | GS-ONE LF, SG-5 | No published huddle-test or NLNM-comparison self-noise curves found for these specific models. Estimated from Brownian noise formula using known mass and f₀. | Moderate — for publication, add a note that self-noise is estimated, not measured |
| **RTClark geophone full specs** | RTC 4.5Hz 375Ω | Product page only confirms 4.5 Hz / 375 Ω; full datasheet is a linked PDF not accessible to web scraping | Low — almost certainly identical to SM-4 B-coil class |
| **Sunfull PS-4.5B full specs** | PS-4.5B | Datasheet PDF was binary-encoded; sensitivity confirmed as 28.8 V/m/s / 375 Ω from Raspberry Shake documentation | Low |
| **GS-20DX 8Hz variant specs** | GS-20DX 8Hz | Confirmed to exist; assume similar to 10Hz but lower G (per physics) and ζ_oc different | Low |
| **Sercel L-22 geophone specs** | L-22 | Not found in search; may be obsolete or OEM-only model | Moderate — L-22 appears in LIGO documentation but no public datasheet |
| **Prices for most models** | All | List prices not confirmed for most models; research/survey pricing differs from volume pricing. All price tiers are order-of-magnitude estimates from distributor listings | Moderate |

---

## 7. Sources

**Manufacturer product pages (primary, accessed July 2026):**
- [Geospace GS-ONE product page](https://www.geospace.com/products/sensors-and-geophones/gs-one/) — GS-ONE 10Hz, 85.8 V/m/s, 1800Ω
- [Geospace GS-ONE LF product page](https://www.geospace.com/products/sensors-and-geophones/gs-one-lf/) — GS-ONE LF 4.5/5Hz, 100.4 V/m/s, 2450Ω
- [Geospace GS-20DX product page](https://www.geospace.com/products/sensors-and-geophones/gs-20dx/) — 10Hz, 27.6 V/m/s, 395Ω, ζ_oc=0.30
- [Geospace GS-14 product page](https://www.geospace.com/products/sensors-and-geophones/gs-14/) — L3 (570Ω, 11.4 V/m/s) and L9 (1500Ω, 23.6 V/m/s) variants
- [Geospace HS-1 Seismometer page](https://www.geospace.com/products/sensors-and-geophones/hs-1-seismometer/) — 2Hz, 18.1 V/m/s, 225Ω
- [Geospace HS-1 3C page](https://www.geospace.com/sensors/hs-1/) — 2Hz, 1.04 V/in/s, 1250Ω, IP67
- [Geospace GS-1 Seismometer page](https://www.geospace.com/products/sensors-and-geophones/gs-1-seismometer/) — 1.0Hz, 3–19 V/in/s, 1.96 kg
- [Geospace GS-1 3C Seismonitor page](https://www.geospace.com/products/sensors-and-geophones/gs-1-3c-seismonitor/) — 1.0Hz, triaxial, IP67
- [Sercel SG-5 specifications](https://www.sercel.com/sites/default/files/2024-08/geophones_specifications_sercel_en.pdf) — 5Hz, 80 V/m/s, 1850Ω, confirmed via DirectIndustry and search
- [Sercel SG-10HS specs](https://www.directindustry.com/prod/sercel-inc/product-137881-1920041.html) — 10Hz, 85.8 V/m/s, 1800Ω
- [SmartSolo IGU-16 product page](https://smartsolo.com/igu-16.html) — 5Hz 80 V/m/s / 10Hz 85.8 V/m/s, 32-bit ADC

**Seis-Tech product pages (clone/equivalent datasheets, all accessed July 2026):**
- [SM-4 equivalent (ST-8N/10N/14N/40N)](https://www.seis-tech.com/sm-4-equivalent-geophone-sensor/) — 8/10/14/40Hz, 28.8–32.5 V/m/s, 375–540Ω
- [SM-24 equivalent (ST-10N)](https://www.seis-tech.com/sm-24-equivalent-geophone-sensor/) — 10Hz, 28.8 V/m/s, 375Ω, ζ_oc=0.25
- [SM-6 B-coil 4.5Hz equivalent](https://www.seis-tech.com/4-5hz-geophone-sensor/) — 4.5Hz, 28.8 V/m/s, 375Ω, ζ_oc=0.60
- [SM-6 H-coil 4.5Hz equivalent (ST-4.5N H-B)](https://www.seis-tech.com/sm-6-h-b-style-4-5hz-geophone) — 4.5Hz, 82.0 V/m/s, 3400Ω
- [High-sensitivity 4.5Hz (ST-4.5A)](https://www.seis-tech.com/high-sensitivity-geophone-4-5hz/) — 4.5Hz, 100 V/m/s, 3800Ω
- [High-sensitivity 5Hz (ST-5NA)](https://www.seis-tech.com/geophone-5hz/) — 5Hz, 80 V/m/s, 1850Ω, ζ_oc=0.60
- [High-sensitivity 10Hz (ST-10PA)](https://www.seis-tech.com/high-sensitivity-geophone-10hz/) — 10Hz, 85.8 V/m/s, 1800Ω
- [GS-32CT equivalent](https://www.seis-tech.com/gs-32ct-equivalent-geophone-sensor/) — 10Hz, 27.5 V/m/s, 395Ω, ζ_oc=0.316
- [28Hz geophone (ST-20DX 28Hz)](https://www.seis-tech.com/geophone-28hz/) — 28Hz, 39.0 V/m/s, 395Ω
- [100Hz geophone (ST-20DX 100Hz)](https://www.seis-tech.com/100hz-geophone/) — 100Hz, 23.0 V/m/s, 570Ω
- [3C geophone (ST-4.5N case)](https://www.seis-tech.com/three-component-geophone-3c-geophone/) — 4.5Hz, 28.8 V/m/s, triaxial
- [4.5Hz 3C for HVSR/MASW](https://www.seis-tech.com/4-5hz-3c-geophone/) — 4.5Hz, 28.8 V/m/s, 375Ω
- [Borehole triaxial 10Hz](https://www.seis-tech.com/borehole-triaxial-geophone-10hz/) — 10Hz, 10 V/m/s, 3400Ω, 60mm OD
- [Borehole 3C full range](https://www.seis-tech.com/borehole-3c-geophone/) — 2–100Hz variants
- [Low-frequency 2Hz (ST-2A)](https://www.seis-tech.com/low-frequency-geophone-2hz/) — 2Hz, 260 V/m/s, 6400Ω, 60g mass
- [Low-frequency 1Hz (ST-1N)](https://www.seis-tech.com/low-frequency-geophone-1hz-2/) — 1Hz, 200 V/m/s, 3300Ω, 770g mass

**IRIS / EarthScope instrument libraries:**
- [OYO Geospace GS-11D Sensor Responses (IRIS NRL)](https://ds.iris.edu/NRL/sensors/oyo_geospace/oyo_geospace_gs11d_sensors.htm) — 380Ω: 32 V/m/s; 4000Ω: 97.4 V/m/s; 10Hz 380Ω: 32 V/m/s
- [EarthScope GS-11D 4.5Hz](https://epic.earthscope.org/content/45hz-high-frequency-single-component-sensor) — 100 V/m/s, ζ=0.707 noted
- [Sercel L-4C Sensor Responses (IRIS NRL)](https://ds.iris.edu/NRL/sensors/sercel/sercel_l4c_sensors.html) — 1Hz, 83.5–276.8 V/m/s, 500–5500Ω, 500–1000g mass

**Technical lab pages:**
- [DESY GS-11D page](https://vibration.desy.de/equipment/geophones/gs_11d/) — 4.5Hz, 380Ω, 32 V/m/s (0.81 V/in/s), ζ_oc=0.34, 23.6g mass, 111g weight, 31.8×33.5mm

**Noise and calibration literature:**
- [L-4C huddle test (arXiv:1711.05439)](https://arxiv.org/abs/1711.05439) — Johnson-noise-limited self-noise ~10⁻¹¹ m/√Hz at 1Hz
- [LIGO L-4C noise note DCC T1600438](https://dcc.ligo.org/public/0138/T1600438/001/L-4C%20huddle%20test%20at%20the%20AEI.pdf) — Johnson + Brownian limits
- [Optomechanical MEMS geophone 2.5 ng/√Hz (PMC11589752)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11589752/) — conventional geophone ~10 ng/√Hz comparison
- [SmartSolo IGU-16HR 3C (EarthScope)](https://epic.earthscope.org/content/instrumentation/all-one-systems/smart-solo) — 5Hz nodal system performance
- [SmartSolo seismological performance (EarthArXiv:3564)](https://eartharxiv.org/repository/view/3564/) — performs comparably to Lennartz 3D/5s down to 0.2 Hz

**Price references:**
- [SparkFun SM-24 (SEN-11744)](https://www.sparkfun.com/products/11744) — ~$60 retail
- [eBay GS-ONE LF listings](https://www.ebay.com/itm/146664993832) — ~$30–80 surplus
- Chinese OEM pricing (Made-in-China, Alibaba): 4.5–10Hz standard geophones ~$5–25/ea at MOQ ≥ 100 units

---

## 8. Quick-Reference Summary

**Domain-randomization parameter ranges (geophone family, footstep-band 4.5–10 Hz class):**

| Parameter | Representative "standard" value | Full simulation range | Note |
|---|---|---|---|
| f₀ (Hz) | 4.5–10 | [4.5, 14] (use [4.5, 10] for primary) | Below 4.5 Hz → seismometer regime; above 14 Hz misses footstep band |
| G_oc (V/m/s) | 28.8 (std) / 80–100 (high-sens) | [28, 102] | Largely a gain factor; sample log-uniform |
| R_coil (Ω) | 375 (std) / 1800 (high-sens) | [375, 3800] | Sets Johnson noise and cable loading; sample log-uniform |
| ζ_operational | 0.65 | [0.55, 0.72] | With proper shunt; extend to [0.25, 0.70] for adversarial |
| Self-noise (ng/√Hz, 10–50 Hz) | ~10 | [4, 15] | Rarely binding outdoors; always below ambient |
| f_spur (Hz) | ≥240 (std) / ≥120 (high-G) | [90, 350] | High-sensitivity geophones have lower spur. |
| m_mov (g) | 11 (std) / 14–25 (high-G) | [11, 25] | For footstep-band class; not seismometers |
| Wt (g) | 74–104 (std) / 130–170 (high-G) | — | Physical packaging consideration |
| Coil excursion (mm p-p) | 1.5–4 | — | SM-6 / GS-ONE LF class have larger travel |
