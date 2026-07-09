> **Research survey — house-localization pivot ("wallhacks").** Background research agent (Sonnet), 2026-07-08. Literature/tooling survey; design input, not a validated experiment. Extends `research/geophone_arrays.md` §3.5, `research/geophone_sensitivity.md`, `research/mems_detection.md`. Indexed in [README.md](README.md).

---

# Sensor Selection, Hybrid Architecture, Array Sizing, Mounting Coupling, and Timing for Indoor Structural Footstep Localization

## 1. Sensor Type for Indoor Structural Footstep Sensing

### 1.1 Why indoor structural differs from outdoor soil

- **Wave medium.** Concrete slab / brick wall carries **Lamb plate waves** whose phase velocity is a function of frequency × thickness, dispersive from the start. Measured propagation velocities in concrete floor slabs span **30–500 m/s** (low-frequency bending) up to compressional ~3000+ m/s. This range and its frequency dependence is the dominant localization error source.
- **Frequency band.** Footsteps on concrete excite structural resonances at **11, 53, 80, 112, 165, 200 Hz** (Ekimov & Sabatier, JASA 2006); floor fundamentals at **5–8 Hz**. Useful band ≈ **5–250 Hz**, with high-frequency energy persisting farther than in soil.
- **Signal level vs. noise.** Short indoor distances (3–20 m) + lower structural attenuation → **much better footstep-to-noise margin indoors**. This shifts the binding constraint: a MEMS accelerometer's higher self-noise is acceptable at indoor ranges.

### 1.2 Geophone corner-frequency comparison

| Sensor | f₀ (Hz) | Sensitivity (V·s/m) | Coil R (Ω) | Bandwidth | Self-noise (accel, mid-band) | Notes |
|---|---|---|---|---|---|---|
| 4.5 Hz (GS-11D class) | 4.5 | 32–100 | 375–2000 | 4.5 → 300+ Hz | ~10 ng/√Hz @ 30 Hz | Catches floor modes (5–8 Hz) + full footstep band |
| SM-24 (10 Hz) | 10 | 28.8 | 375 | 10–240 Hz | ~12 ng/√Hz | Misses floor fundamentals; OK for TDOA >10 Hz |
| 14 Hz | 14 | 28–40 | ~400 | 14–190 Hz | ~15 ng/√Hz | Misses floor modes |
| 28 Hz | 28 | 28–39 | 395 | 28+ Hz | ~25 ng/√Hz | Cuts 20–28 Hz footstep energy |
| 40/100 Hz | 40–100 | ~25–35 | ~400 | 40+ Hz | higher | Misses most low-freq floor modes |

**Verdict on 4.5 Hz for indoor use.** Well-suited — captures floor fundamentals (5–8 Hz), the full footstep band (20–250 Hz), and lowest self-noise in 5–100 Hz. 10 Hz (SM-24) acceptable but loses the floor-mode band; 14/28 Hz are worse; 40–100 Hz completely wrong for indoor. The one real limitation of any moving-coil geophone: single vertical axis, orientation-sensitive, velocity output (AC-coupled, no DC).

### 1.3 Full sensor type comparison (indoor footstep band, ~5–250 Hz)

| Sensor class | Part | Noise floor (accel) | Bandwidth | DR | Axes | DC | Size | Power | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Moving-coil geophone 4.5 Hz | GS-11D/SM-11 | **~10 ng/√Hz** (5–100 Hz) | 4.5–300+ Hz | 80–90 dB | 1 (V) | No | 26×36 mm | 0 mW | $10–50 | Excellent for weak signals; 1-axis, no DC |
| Geophone 10 Hz | SM-24 | ~12 ng/√Hz | 10–240 Hz | 80 dB | 1 | No | 25×32 mm | 0 mW | $15–40 | Good; loses 5–10 Hz |
| Geophone 28 Hz | Seis-Tech | ~25 ng/√Hz | 28+ Hz | 75 dB | 1 | No | 25×33 mm | 0 mW | $10–30 | Avoid for indoor |
| MEMS accel (mid) | ADXL355 | **25 µg/√Hz ≈ 245,000 ng/√Hz** | DC–1000 Hz | >90 dB | 3 | Yes | 3×3 mm | ~0.5 mW | $25–65 | Adequate indoors (strong signal); 3-axis + DC |
| MEMS (research) | Silicon Audio optical | **0.5 ng/√Hz @ 10 Hz** | 0.1–1k Hz | >183 dB | 1/3 | Yes | can | few mW | $1,000s | Overkill; geophysical |
| MEMS seismometer | Sercel WiNG | −136 dB BN (1–100 Hz) | 0.03–100+ Hz | >120 dB | 3 | Yes | nodal | days | $1,000s | Way above spec |
| IEPE piezo | PCB 352C33 | ~0.5–1 µg/√Hz, AC | 1 Hz–10 kHz | 80–100 dB | 1 | No | 10×12 mm | 2–4 mA | $200–800 | Excellent at footstep f; needs current source; pricey for arrays |

**Geophone vs ADXL355 crossover.** Geophone velocity self-noise ≈ 5×10⁻¹⁰ m/s/√Hz above corner → as acceleration ≈ 3 ng/√Hz @10 Hz, 10 ng/√Hz @30 Hz. ADXL355 ≈ 245,000 ng/√Hz flat. The crossover where MEMS matches geophone in acceleration noise is ≈ **78 kHz** — the geophone is quieter at every seismically relevant frequency. **MEMS never wins on raw self-noise in the footstep band.**

**Why MEMS is still adequate indoors:** for a footstep at 5 m on concrete the signal is large (hundreds–thousands of nm/s), so both sensors sit well above their self-noise floors. The MEMS just can't detect a faint distant signal the geophone still sees.

**Recommendation on sensor type.** Primary = **4.5 Hz moving-coil geophone**. For N ≥ 4 arrays where cost/wiring matter, **ADXL355/ADXL356** MEMS is practically adequate at indoor ranges and adds 3-axis + DC cheaply. 10 Hz (SM-24) is the pragmatic smaller/cheaper geophone; avoid 14/28 Hz.

---

## 2. Hybrid Geophone + MEMS

### 2.1 Precedents

Not novel — it is the **Raspberry Shake RS4D** architecture: one vertical 4.5 Hz geophone (weak-motion, low-noise) + a triaxial ±2 g MEMS accelerometer (strong-motion, 3-axis, DC), shared 24-bit digitizer, >1,000 units deployed globally. **IMS GmbH** hybrid nodes combine a geophone + IEPE accelerometer for "better coverage of a wider variety of seismic events." LIGO seismic isolation uses geophone+seismometer complementary fusion. The 2024 EarthDoc "Digital MEMS and Seismic Nodes — Technology Fusion" documents the oil-industry convergence.

### 2.2 Division of labor

| Frequency region | Geophone | MEMS |
|---|---|---|
| DC–4.5 Hz | −12 dB/oct rolloff | **Flat accel; DC/tilt; building sway; HVAC modes** |
| 4.5–100 Hz | **Primary: lowest noise, flat velocity** | Secondary: noisier but adequate for strong indoor signals |
| 100–500 Hz | Flat velocity (mechanical BW above 100 Hz) | Flat accel; calibrated |
| >500 Hz | Response falls | Flat to 1 kHz+ |
| Strong-motion | **Can clip** (coil hits stop) | ±2–40 g; never clips on footsteps |
| 3-axis bearing | Vertical only | **3-axis → azimuth from horizontals** |

### 2.3 Complementary-filter fusion

```
Geophone → high-pass (corner f₀ ≈ 4.5 Hz) → geo_hp
MEMS    → low-pass (1 − H(s))              → mems_lp
                            └─ sum ──→ fused (flat DC to ~250 Hz)
```
Below 4.5 Hz the MEMS provides the signal; above, the geophone. The MEMS supplies horizontal channels continuously.

For footstep localization specifically:
- Geophone channel for **TDOA onset picks** (lowest noise → sharpest STA/LTA trigger → least TDOA jitter)
- MEMS horizontals for **bearing** (polarization / first-arrival direction)
- MEMS as **strong-motion backup** (anti-clip)
- MEMS **DC** to monitor sensor tilt/orientation

### 2.4 Concrete recommendation

**1× 4.5 Hz geophone (vertical) + 1× ADXL355 (3-axis) per node**, both wired to the same 24-bit simultaneous-sampling ADC. Cost +$25–65/node, power +0.5 mW, size negligible. This is the RS4D architecture scaled down, validated by >1,000 deployed units. Run onset detection on the geophone (best SNR); use the 3 MEMS horizontals for single-node bearing/polarization.

---

## 3. How Many Sensors

- **Minimum (2D TDOA):** 3 sensors → unique 2D position (ambiguities in degenerate geometry).
- **Field-proven minimum:** 4 (3 TDOA pairs → overdetermined, outlier rejection).
- **3D (which floor):** sensors at different heights.

| Study | Sensors | Floor | Area | Accuracy |
|---|---|---|---|---|
| FEEL | **3** | RC | 305×438 cm | 98.6% within 63.5 cm (transfer-function, not TDOA) |
| SO-TDOA | **9** | concrete | 3.6×5.4 m | ~50 cm (sign-only TDOA) |
| NSF occupant loc. | ~4–6 | concrete/timber | 20 m² | ~0.34 m |
| Outdoor TDOA (Pakhomov) | **4** | soil | field | ~8.4 m @ 30 m |

**Diminishing returns:** 3→4 large gain (robustness); 4→6 moderate (coverage, >20 m²); 6→9 small; 9→16 marginal.

**Practical (10×10 m house):** minimum 4 (corners) → ~0.5–1 m; recommended **5–6** (corners + large-room centers) → ~0.3–0.7 m; extended 200 m² building → 8–12 across structural nodes; +1/floor for storey ID.

---

## 4. Mounting and Coupling

### 4.1 Coupling physics

Coupling determines what fraction of local particle velocity reaches the proof mass. A compliant layer acts as a mechanical low-pass; rigid coupling passes all modes up to the sensor resonance. A heavy geophone (~75 g) on a light partition wall creates impedance mismatch and a local mounting resonance — fix with rigid coupling (epoxy, threaded stud) to a massive element.

### 4.2 Mounting location table

| Location | Wave mode | Coupling | Amplitude | Freq content | TDOA utility | Edge-case |
|---|---|---|---|---|---|---|
| **Floor slab (bonded)** | Lamb plate waves | Excellent | **Highest** | Full 5–250+ Hz | **Best** | Floating floor on isolators → decoupled, high insertion loss |
| **Underfloor (soffit, bonded)** | Same, opposite face | Excellent | High | Cleaner (less heel-mass artifact) | **Very good** | Needs access below |
| **Floor surface (uncoupled)** | Lamb, partial | Poor | Low | Low-pass filtered | Weak | MEMS phones slide >0.3 g |
| **Wall (bonded)** | Bending in wall, mode-converted at junction | Moderate | Moderate | Low-f (5–20 Hz) preserved; >80 Hz attenuated | Detection + low-f bearing; poor high-f TDOA | Wall resonance 10–50 Hz |
| **Foundation / slab-on-grade** | Lamb + ground Rayleigh | Good if embedded | Moderate | + outdoor ground wave | Good for outdoor→indoor; less for above-ground footstep | Isolation joints add loss |
| **Roof / top slab** | Traverses all walls/floors | Very poor | **Lowest** | Modal-dominated, decorrelated | **Not useful** | Wind 10–100× footstep |
| **Structural column (bonded)** | Longitudinal + flexural; direct load path | Good | Moderate–high at base | Broadband | Good secondary | Flexural column resonance |

### 4.3 Soft-ground edge case

- Velocity drop (soft soil 50–150 m/s) + soil-foundation reflection/refraction losses.
- Attenuation: Q ≈ 10–20 → footstep gone within ~20 m.
- Geophone-soil resonance (5–30 Hz) if laid on surface without spike → ringing artifact.
- **Impact:** useful for detection of strong approaching footsteps; unreliable for TDOA (velocity unknown, dispersive, moisture-dependent, 2–3× wet vs dry).

### 4.4 Roof edge case

- Multiple structural joints between source and sensor → 20–40 dB lower.
- Wind-induced vibration (0.1–10 Hz) exceeds footstep signal.
- Roof structural modes (2–15 Hz) dominate.
- **TDOA useless:** arrival time bears little relation to source location. Only viable use: *detecting* very heavy impacts (fall, running), not localization.

### 4.5 Coupling method impact

| Method | Frequency response | Notes |
|---|---|---|
| Epoxy / structural adhesive | Best (rigid) | Permanent |
| Threaded stud / M5 tapped hole | Excellent | Preferred |
| Beeswax / wax | Very good low/mid f | Temporary; softens in heat |
| Rare-earth magnet on bonded steel | Good | Removable; slight HF loss |
| Double-sided foam tape | Adequate <50 Hz; lossy above | Quick |
| Free-standing on concrete | Poor | Only <20 Hz |
| Spiked into soft soil | Moderate | Poor above 30 Hz |

---

## 5. Multi-Sensor Timing: ADS1115 Limits and Replacement

### 5.1 Why the ADS1115 breaks TDOA

1. **Max 860 SPS**; when multiplexing 4 channels via I²C, ~**150–200 SPS/channel** (~5 ms resolution) — inadequate.
2. **Sequential, not simultaneous** — one ADC core, multiplexed inputs. Ch1 at t=0, Ch2 at t=1.16 ms → 0.23 m/channel spatial uncertainty at 200 m/s.
3. **Software-driven timing** — Raspberry Pi (non-RT Linux) task jitter 1–10 ms swamps TDOA.
4. **Required TDOA timing:** 0.5 m / 200 m/s = **2.5 ms** (or **1 ms** at 500 m/s). ADS1115 at 860 SPS = 1.16 ms quantization, and multiplexing pushes it to ~5 ms → ~1 m error. Off the table.
5. **No usable cross-board DRDY sync pin.**

### 5.2 Replacement ADCs

**Option A — single-board simultaneous 24-bit ADC (wired single node):**

| ADC | Ch | Max rate | Bits | Simultaneous | Cost |
|---|---|---|---|---|---|
| **MCP3912** | 4 | 125 kSPS | 24 | **Yes (<1 µs skew)** | ~$6 |
| ADS1256 | 8 (mux) | 30 kSPS | 24 | No (<33 µs jitter) | ~$5–15 |
| ADS1262 | 2 | 40 kSPS | 24 | No (25 µs) | ~$7 |
| AD7768-4 | 4 | 256 kSPS | 24 | **Yes** | ~$25 |
| MAX11254 | 6 | 64 kSPS | 24 | Near-simultaneous | ~$15 |

The **MCP3912** starts all 4 modulators from one clock → inter-channel skew < 1 µs → TDOA timing error ~0.0002 m; real error dominated by onset-pick jitter and velocity uncertainty.

**Option B — GPS PPS (distributed/wireless nodes):** u-blox NEO-M8N (~$15), 1PPS ±20 ns; inter-node uncertainty ~1–5 µs → ~0.0025 m. Needs sky view / external antenna.

**Option C — wired hardware trigger line:** master fires a rising edge to all nodes' ADC START/SYNC pins; skew <50 ns (for <20 m cable). Fully adequate.

**Option D — PTP (IEEE 1588) over Ethernet:** ~1 µs sync with hardware timestamping. Viable for networked nodes.

### 5.3 Minimum sample rate

- Nyquist for 250 Hz → 500 SPS min; **recommend 1000–2000 SPS**, preferred 2000–4000 SPS.
- At 1000 SPS: 1 ms → 0.2 m spatial (200 m/s). At 2000–4000 SPS: 0.05–0.1 m timing contribution.
- Dominant error is **wave-velocity uncertainty** and onset-pick jitter, not timing → on-site velocity calibration is required (same conclusion as the outdoor array analysis).

### 5.4 Verdict

**Replace the ADS1115 for any TDOA localization.** Use MCP3912 (single node), or GPS-PPS / hardware trigger to per-node 24-bit ADCs (distributed). Sample at 1000+ SPS simultaneous.

---

## Consolidated Recommendations

- **Sensor:** 4.5 Hz moving-coil geophone as primary; don't go to 14/28 Hz for indoor.
- **Hybrid:** add 1× ADXL355 (or ADXL356) per node (3-axis bearing, DC/tilt, strong-motion anti-clip) — the RS4D architecture, +$25–65/node. Geophone for TDOA onset; MEMS horizontals for bearing.
- **Count:** minimum 4; recommended **4–6** for a house (slab-bonded, corners + large-room centers); +1/floor for storey ID.
- **Mounting priority:** Floor slab (bonded) > Underfloor soffit (bonded) > Column base (bonded) > Load-bearing wall (bonded) > Foundation ≫ Roof. Soft ground = detection only; roof = detection only.
- **ADC/timing:** replace ADS1115 → MCP3912 (single node) or GPS-PPS / hardware trigger (distributed), 1000+ SPS simultaneous.

---

## Sources

- [GS-11D 4.5 Hz (EarthScope)](https://epic.earthscope.org/content/45hz-high-frequency-single-component-sensor) · [SM-24 datasheet](https://www.seis-tech.com/wp-content/uploads/2024/08/data-sheet-of-sm24-10hz-equivalent-geophone.pdf) · [SM-24 (SparkFun)](https://www.sparkfun.com/geophone-sm-24-with-insulating-disc.html)
- [Seis-Tech 28 Hz](https://www.seis-tech.com/geophone-28hz/) · [14 Hz](https://www.seis-tech.com/reinforced-14hz-geophone-for-land-desert/) · [multi-freq 4.5–100 Hz](https://seistech.en.made-in-china.com/product/VycmrUzubwkH/)
- [Optomechanical MEMS geophone 2.5 ng/√Hz (Nature 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11589752/) · [ADXL355 (ADI)](https://www.analog.com/en/products/adxl355.html) · [ADXL355 microtremor](https://www.researchgate.net/publication/353789426)
- [Silicon Audio optical MEMS](http://siaudioseismic.com/seismic-sensors/) · [Sercel WiNG QuietSeis](https://www.sercel.com/en/news/wing-solution-clear-choice) · [WiNG passive seismology](https://www.researchgate.net/publication/380080104)
- [Raspberry Shake RS4D](https://shop.raspberryshake.org/product/turnkey-iot-home-earth-monitor-rs-4d/) · [RS4D structural ID (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9268860/) · [IMS hybrid geophone+IEPE](https://www.imseismology.org/sensors/) · [Digital MEMS + nodes fusion (EarthDoc)](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202270122)
- [Ekimov & Sabatier footstep signatures (JASA 2006)](https://www.researchgate.net/publication/6849688) · [Footstep localization Lamb waves 30–300 m/s (PMC11644851)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644851/) · [FEEL (PMC9886238)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9886238/) · [SO-TDOA (arXiv:1211.3233)](https://arxiv.org/abs/1211.3233) · [Occupant loc. (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0888327018302280) · [GaitVibe+ (arXiv 2212.03377)](https://arxiv.org/pdf/2212.03377) · [WSN loc. optimization (IOP 2024)](https://iopscience.iop.org/article/10.1088/1361-6501/ad4810)
- [Geophone-ground coupling (flat bases)](https://www.researchgate.net/publication/280874569) · [coupling + attenuation compensation](https://www.researchgate.net/publication/255617142)
- [ADS1115 datasheet](https://www.ti.com/lit/ds/symlink/ads1115.pdf) · [ADS1256](https://www.ti.com/lit/ds/symlink/ads1256.pdf) · [MCP3912](https://ww1.microchip.com/downloads/aemDocuments/documents/MSLD/ProductDocuments/DataSheets/MCP3912-3V-Four-Channel-Analog-Front-End-DS20005348C.pdf)
- [GPS PPS for TDOA](https://www.researchgate.net/publication/280556880) · [GPS PPS + ADC STM32](https://community.st.com/t5/stm32cubemx-mcus/gps-pps-aligned-external-adc-sampling-in-stm32l4p5-for-multiple/td-p/586471)
- [Evans et al. low-cost MEMS accel (USGS)](https://pubs.usgs.gov/publication/70115921) · [CREWES geophone vs MEMS](https://www.crewes.org/Documents/GraduateTheses/2008/Hons-MSc-2008.pdf) · [MEMS vs geophone (OmniDots)](https://www.omnidots.com/en/vibration-monitoring-sensors-for-construction-mems-vs-geophone)
