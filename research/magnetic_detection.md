# Magnetic and Electromagnetic Detection of Ferrous/Metallic Objects (Weapons and Vehicles) for Security and Unattended Ground Sensors

*A scientific report prepared for the GeoSense multi-sensor unattended-ground-sensor (UGS) program. Companion to the buried 4.5 Hz geophone + edge-ML seismic classifier; this document scopes the magnetic/electromagnetic confirmation modality.*

**Date:** 2026-06-25

---

## Abstract

A buried geophone classifies *what is moving* (human, vehicle, animal) from substrate vibration, but it cannot by itself confirm that a target carries metal — i.e. that a person is armed, or that a moving mass is a metallic vehicle rather than livestock. Two distinct magnetic-domain techniques close this gap, and the project's informal phrase "electromagnets that detect weaponry/vehicles" conflates them. **(A) Passive Magnetic Anomaly Detection (MAD)** uses a sensitive magnetometer to measure the local distortion that a *ferromagnetic* object imposes on the Earth's static field; it transmits nothing. **(B) Active Electromagnetic Induction (EMI)** — the principle of conventional metal detectors — *transmits* an alternating primary field, induces eddy currents in any *conductive* object, and senses the secondary field they re-radiate. Both are governed by steep near-field range laws: the passive magnetic-dipole anomaly falls as **1/r³**, while the active transmitter→target→receiver EMI response falls roughly as **1/r⁶**. This report quantifies the underlying physics, sensor-technology sensitivities (nT to fT/√Hz), realistic detection ranges (≈1–5 m for a personal weapon; tens to hundreds of metres for a vehicle for the *passive* modality; centimetres-to-~1 m for hand-portable *active* metal detection), the dominant false-alarm and clutter sources (geomagnetic diurnal variation, ULF micropulsations, ferrous clutter, the sensor's own platform) and their mitigation, and how a passive magnetometer channel complements the seismic node for armed/unarmed and vehicle-class confirmation.

---

## 1. Terminology: clarifying "electromagnets that detect weaponry/vehicles"

The colloquial phrase "electromagnets" is imprecise. Detection of ferrous/metallic threats is done by **magnetometers** and **electromagnetic-induction coils**, not by electromagnets in the sense of a powered lifting magnet. There are two fundamentally different operating modes, and a UGS designer must keep them separate because they sense different physical properties of the target, have different range laws, and have different power budgets:

| | **(A) Passive MAD (magnetometry)** | **(B) Active EMI (metal detection)** |
|---|---|---|
| Transmits a field? | **No** — listens only | **Yes** — radiates a primary AC field |
| Physical property sensed | **Ferromagnetism** (permanent + induced magnetisation that perturbs Earth's field) | **Electrical conductivity / permeability** (eddy currents in any metal) |
| Source of the measured signal | The Earth's field, distorted by the target's magnetic moment | The secondary field re-radiated by induced eddy currents |
| Range law (point target) | Anomaly ∝ **1/r³** | Response ∝ **1/r⁶** (≈ 1/r³ out, 1/r³ back) |
| Practical standoff | metres (weapon) to hundreds of metres (vehicle) | centimetres to ~1 m (hand-portable) |
| Detects non-ferrous metal? | No (needs ferromagnetic content) | Yes (aluminium, brass, copper, etc.) |
| Power | µW–mW (sensor only) | Higher (must drive transmit coil) |

For an **always-on, battery-powered, buried UGS**, technique **(A)** is the natural fit: it is passive, low-power, omnidirectional, and detects exactly the ferrous mass (steel weapons, vehicle bodies/engines) that characterises a military or intrusion threat. Technique **(B)** is the natural fit for **portal/walk-through, handheld, or close-approach** screening (and for buried-mine clearance), where the sensor can be brought within ~1 m of the target and the extra power for a transmit coil is acceptable.

---

## 2. Technique A — Passive Magnetic Anomaly Detection (MAD)

### 2.1 Physical principle

The Earth produces a quasi-static background field of magnitude **~25,000–65,000 nT (0.25–0.65 G)** at the surface, decreasing from poles to equator [USGS; Wikipedia *Earth's magnetic field*]. A ferromagnetic object placed in this field acquires an **induced magnetisation** (and usually carries a **remanent** magnetisation from its manufacture and history); a vehicle that has sat or driven in the Earth's field for a long time becomes magnetised and "is capable of disturbing the earth field by its magnetic moment, and this disturbed signal can be detected by a magnetometer" [Tan et al., *Magnetic Anomaly Detection Using a Three-Axis Magnetometer*]. The object thus behaves, when viewed from beyond roughly its own dimensions ("far field"), as a **magnetic dipole** of moment **m** (units A·m²). MAD is the measurement of this localised distortion ("anomaly") superimposed on the smooth geomagnetic background.

### 2.2 The magnetic-dipole field and the 1/r³ law

The field of a point dipole **m** at displacement **r** is

> **B**(r) = (µ₀ / 4π) · [ 3 (**m**·r̂) r̂ − **m** ] / r³

On the dipole axis the magnitude is **B_axis = µ₀ m / (2π r³)** and in the equatorial plane **B_eq = µ₀ m / (4π r³)** [Physics LibreTexts, *Magnetic Dipole Moment*]. The crucial feature for sensor design is the **1/r³ dependence**: the anomaly amplitude collapses with the cube of range. Field experiments confirm this directly — extracted anomaly amplitudes "exhibit a clear dipole-like decay with distance, approaching ΔB ∝ D⁻³ over the measured range" [arXiv 2601.08716, *Portable Single-Beam Atomic Total-Field Magnetometer for Stand-off Magnetic Sensing*].

**Detection-range scaling — the key design relation.** If a sensor can resolve a minimum anomaly **B_min** (set by its noise floor and the geomagnetic background), then for a target of moment **m** the maximum detection range follows by setting the dipole field equal to B_min:

> **r_max ≈ ( µ₀ m / (4π · B_min) )^(1/3)**

Two consequences dominate practical UGS performance:

1. **Range grows only as the cube root of target moment.** A target with 1000× the magnetic moment is detectable at only 10× the range. This is why a small carried weapon and a large vehicle — separated by orders of magnitude in moment — differ in detection range by only one to two orders of magnitude.
2. **Range grows only as the cube root of sensitivity improvement.** Buying a magnetometer 1000× quieter buys 10× more range. Beyond a point, range is capped not by the sensor but by **geomagnetic background noise** (Section 4), so chasing fT-class sensors yields diminishing returns for surface targets.

### 2.3 Quantitative target moments and detection ranges

**Vehicles.** A typical passenger automobile has a magnetic moment of order **m ≈ 100–300 A·m²** (a school bus ≈ 2000 A·m²) [Lenz & Edelstein and traffic-sensing literature, via *Dynamic Vehicle Detection via the Use of Magnetic Field Sensors*, PMC4732111]. Inserting m = 200 A·m² and a modest field UGS sensitivity B_min ≈ 1 nT gives

> r_max ≈ ( (4π×10⁻⁷ × 200) / (4π × 10⁻⁹) )^(1/3) = (200/10⁻⁹ × 10⁻⁷)^(1/3) ≈ (2.0×10⁴)^(1/3) ≈ **27 m**.

With a quieter (0.01–0.1 nT) sensor sited in a low-noise location, vehicle detection of **tens of metres up to ~100+ m** is realistic; large/armoured ferrous masses (tanks, trucks) extend this further. This is consistent with fielded systems: the U.S. Army REMBASS/IREMBASS family quotes overall engagement ranges of **wheeled 15–250 m and tracked 25–350 m** across its sensor suite [FAS, *REMBASS/IREMBASS*].

**Personal weapons.** A handgun or rifle behaves as a small "long dipole" characterised by its magnetisation, length, centre, azimuth and plunge [Bigman et al., *Suitability of magnetometry to detect clandestine buried firearms*, Forensic Sci. Int. 2020]. Controlled buried-firearm magnetometry gives concrete anomaly amplitudes: **a rifle produced ±20 nT at 0.6 m, while a handgun gave only ±2 nT at 1.8 m**, and the study concluded firearms can be detected to **~1.8 m** burial depth, recommending a 0.25 m survey grid [Bigman et al., 2020]. Because the 1/r³ law dominates, the practical standoff for confirming a **carried** weapon with a fixed UGS magnetometer is short — of order **1–3 m**, consistent with the REMBASS magnetic module which "detects targets containing ferrous metals at a distance of only ~3 m" and flags personnel only when they carry ferrous metal [FAS; Defense-Update, *REMBASS-II*]. Weapon orientation matters strongly: vertically oriented weapons produce significantly stronger anomalies than horizontal ones, and even slight tilts enhance the anomaly [Bigman et al., 2020].

**Implication for the GeoSense node:** a passive magnetometer can reliably *confirm a metallic vehicle out to tens of metres*, but can only *confirm a carried weapon when the target passes within a few metres* of the buried node — which is exactly the geometry of a perimeter choke-point or trail-crossing sensor.

### 2.4 Magnetometer sensor technologies and sensitivities

| Technology | Typical noise floor / sensitivity | Notes for UGS use |
|---|---|---|
| **Fluxgate** | ~**1–50 pT/√Hz**; resolves fields **< 0.1 nT**; advanced designs reach ~1 pT noise [ResearchGate, *1 pT-noise fluxgate magnetometer*] | Vector (3-axis), low power, rugged, mature. The workhorse of military UGS; REMBASS-II MAGID integrates "the proven technology of a flux gate magnetometer combined with advanced DSP" [SPIE *MAGID-II*; Defense-Update]. **Recommended baseline for GeoSense.** |
| **AMR / GMR / TMR (magnetoresistive)** | ~**nT-class** (typ. ~0.1–10 nT/√Hz for low-cost AMR; TMR better) | Chip-scale, very low cost/power, vector. Used in commercial vehicle-detection magnetometers; AMR sensors detect vehicles by the Earth-field distortion of their metal body [PMC4732111; Banner Engineering]. Good for the *vehicle-confirmation* channel; marginal for carried weapons. |
| **MEMS magnetometer** | ~nT–µT (lower sensitivity), tiny, ultra-low power | Smallest/cheapest; best for coarse vehicle presence, not weapon-grade anomalies. |
| **Optically-pumped / atomic (Cs, Rb)** | scalar total-field, ~**3.5 pT/√Hz at 1 Hz**, down to **fT-class** for SERF; ~0.01 nT operational [Frontiers; arXiv 2301.08437] | Highest sensitivity, scalar (heading-error-free), but higher power/complexity. Used for stand-off MAD and aeromagnetic survey; overkill and power-hungry for a buried node, but relevant for a fixed mains-powered gate sensor. |
| **Superconducting (SQUID)** | better than ~0.05 nT, fT-class | Requires cryogenics — impractical for field UGS. |

For a **buried, battery-powered GeoSense node**, a **3-axis fluxgate** (or a TMR magnetoresistive array as a low-power alternative) gives the best balance of sensitivity (~tens of pT/√Hz, i.e. sub-nT), vector information (bearing/direction-of-travel), power, and cost.

### 2.5 Use in unattended ground sensors and weapon screening

- **REMBASS / IREMBASS / REMBASS-II (AN/GSR-8):** the U.S. Army's fielded UGS family fuses **seismic, acoustic, magnetic, and passive-IR** modalities. Its **magnetic module** detects wheeled and tracked vehicles and personnel carrying ferrous metal at ~3 m, and provides **count and direction-of-travel** of objects through its zone — a "localized" confirm channel complementing the longer-range seismic/acoustic detector (personnel to ~75 m) and PIR (~30 m) [FAS; Defense-Update; SPIE 4743].
- **MAGID / MAGID-II:** purpose-built **magnetic UGS** using a fluxgate plus DSP for low false-alarm vehicle/personnel detection [SPIE 8388, *MAGID-II*].
- **Walk-through / handheld weapon screening:** passive **magnetometer gun-detection portals** (e.g. INEEL/INL-derived systems) sample the Earth's field and flag the local aberration produced by a ferrous gun or knife, using pattern-recognition/neural-net discrimination of the magnetic signature for weapon-vs-benign classification [NIJ, *Hands-off Frisking*; Kotter et al., SPIE 4708, *Detection and classification of concealed weapons using a magnetometer-based portal*]. Note that **passive magnetometers only flag ferrous (steel) weapons**; all-polymer or non-ferrous items are invisible to them — a key limitation versus active EMI.

---

## 3. Technique B — Active Electromagnetic Induction (EMI) / metal detection

### 3.1 Physical principle (and why it is "active")

EMI metal detection is **active**: a **transmit coil radiates an alternating primary magnetic field**. Where this field reaches a conductive object, Faraday induction drives **eddy currents** in the metal; those eddy currents in turn "trigger a secondary magnetic field" which a **receive coil** senses and the electronics evaluate [FOERSTER, *Detection of mines with EMI*; *Metal detector*, Wikipedia]. Because the detector supplies the energising field, EMI senses **electrical conductivity (and permeability) of any metal** — ferrous *and* non-ferrous (aluminium, brass, copper) — not just ferromagnetism. This is the defining contrast with passive MAD, which transmits nothing and senses only ferromagnetic distortion of the Earth's existing field.

### 3.2 Range law — why active EMI is short-range

The signal traverses a **dipole–dipole path twice**: the primary field falls toward the target as ~1/r³, and the secondary field falls on its way back to the receiver as ~1/r³ again, so the received response scales approximately as **1/r⁶** [Nelson, *Metal Detection and Classification Technologies*, JHU APL Tech. Digest 25(1); arXiv 1308.6027]. This sixth-power falloff is far steeper than the passive 1/r³ law and is the fundamental reason hand-portable metal detectors are **proximity** devices: detection of a handgun-sized object is limited to roughly tens of centimetres, and even large objects to of order **one foot or more** for pulse-induction units, "depending on soil conditions and the size of the object" [Garrett; metal-detecting literature]. Practically, EMI cannot be a stand-off perimeter sensor; its strength is **classification at close range**.

### 3.3 Modes and applications

- **VLF (very-low-frequency, frequency-domain, induction-balance):** continuous sinusoidal excitation; good **metal-type discrimination** (ferrous vs non-ferrous) via phase, useful for identifying specific landmine signatures, but degraded by mineralised soil [gearupgrades; hobby-hour].
- **PI (pulse induction, time-domain):** transmits pulses and measures eddy-current decay; **greater depth** and far less sensitive to soil mineralisation, but poor discrimination [Garrett, *Pulse Induction*].
- **Concealed-weapon / portal metal detectors:** each weapon has a characteristic EMI signature set by its size, shape and composition; modern systems extract and classify these signatures, detecting **both ferrous and non-ferrous** weapons that passive magnetometers would miss [LinkedIn/Kazantsev; Garrett].
- **Humanitarian/military demining:** EMI is "the technology of choice" for buried-metal landmine detection, increasingly combined with a fluxgate magnetometer to characterise the conductive object [FOERSTER; arXiv 2206.12187, *Detection and characterisation of conductive objects using EMI and a fluxgate magnetometer*]. Low-metal-content ("minimum-metal") mines remain the hard case and motivate sensor fusion (e.g. EMI + GPR).

### 3.4 Role in the GeoSense concept

Active EMI is **not** the right primary modality for an always-on buried node (power-hungry transmit coil, ~1/r⁶ range, sub-metre reach). It is the right modality for a **complementary close-range confirm** — e.g. a portal at a controlled entry, a handheld wand for a follow-up frisk, or a buried-mine/IED clearance role — and it adds the unique capability of detecting **non-ferrous** metal that passive MAD cannot see.

---

## 4. False alarms, clutter, and mitigation (passive MAD)

A passive magnetometer measures *everything* that perturbs the field, so clutter rejection is the central engineering problem. Dominant sources:

1. **Geomagnetic diurnal (Sq) variation.** On quiet days the field varies smoothly with local solar time by **tens of nT** (typical daily variation ≈ 25 nT) due to ionospheric currents [USGS; Wikipedia]. This is large compared with a weapon anomaly (single-digit nT at metres) and must be removed.
2. **ULF micropulsations (Pc/Pi pulsations).** Continuous (Pc, ~0.2–600 s) and irregular (Pi) geomagnetic pulsations add structured noise in the sub-Hz band where vehicle/personnel transits also appear; Pc5 amplitudes are small but degrade detection of weak anomalies [ScienceDirect, *Pi2 micropulsations*; *World-Wide Characteristics of Geomagnetic Micropulsations*]. Magnetic storms add tens-to-hundreds of nT and raise the false-alarm floor.
3. **Ferrous environmental clutter.** Fences, rebar, pipes, culverts, vehicles parked nearby, and even mineralised geology create static and slowly varying anomalies.
4. **The sensor's own platform.** Any ferrous fasteners, battery, electronics, or enclosure near the magnetometer impose a large fixed bias and motion-correlated noise — the "platform" or "ownship" effect well known in MAD.

**Mitigation strategies:**

- **Gradiometry (the primary tool).** Differencing two (or more) magnetometers on a short baseline cancels the *spatially uniform* geomagnetic background and its temporal variations (Sq, micropulsations are spatially coherent over the baseline), while preserving the *local* target gradient. Buried-firearm magnetometry and most weapon portals use **gradiometers** for exactly this reason [Bigman et al. used a two-sensor 0.55 m vertical-separation Overhauser gradiometer; NIJ portal]. Gradiometry exploits the fact that the target anomaly's gradient falls as **1/r⁴** (steeper than the field's 1/r³), sharpening localisation and suppressing distant clutter.
- **Reference/base-station subtraction.** A second magnetometer in a quiet location records the common geomagnetic variation for removal.
- **Band-pass / matched filtering and wavelet denoising** tuned to the transit-induced signature versus the slow diurnal drift and ULF bands [e.g. wavelet/energy-detection methods, PMC4208775].
- **Platform hygiene:** non-magnetic (austenitic stainless, brass, aluminium, polymer) enclosure and hardware near the sensor; characterise and subtract the fixed platform bias.
- **Vector + signature classification / ML:** 3-axis data plus pattern recognition (as in REMBASS DSP and portal neural nets) reject benign transients and yield the **very low false-alarm rate** that is the stated design goal of fielded magnetic UGS [Defense-Update; NIJ].

---

## 5. How passive magnetics complement a seismic geophone node

The seismic geophone is a **long-range, omnidirectional motion/identity** sensor; the magnetometer is a **short-range metal-confirmation** sensor. They are complementary precisely because their physics and failure modes differ:

| Question | Seismic geophone (4.5 Hz) | Passive magnetometer |
|---|---|---|
| Something is moving / approaching? | **Yes**, tens of metres, all targets | Only within metres (weapon) to tens of m (vehicle) |
| Is it human / vehicle / animal? | **Primary** (gait/track classification) | Confirms metallic vs non-metallic mass |
| Is the vehicle **metallic** (truck/armour) vs an animal herd? | Ambiguous (large animal can mimic) | **Resolves it** — only ferrous mass gives an anomaly |
| Is the person **armed** (carrying steel)? | **Cannot tell** | **Can confirm** a ferrous weapon at close pass |
| Direction of travel / count | Partial | **Yes** (vector anomaly polarity, REMBASS-style) |

**Fusion logic for intent/threat confirmation:**

- **Seismic "human" + magnetic anomaly at close pass → likely *armed* human.** This is the high-value discriminator: the geophone says "person," the magnetometer says "carrying significant steel," together raising confidence in an *armed intruder/combatant* over an unarmed civilian or animal. (Caveat: belt buckles, phones, tools also contain ferrous metal — discrimination/ML and anomaly magnitude thresholds are needed to separate a weapon from incidental metal, mirroring the false-alarm work in portal screening.)
- **Seismic "vehicle" + strong magnetic anomaly → confirmed *metallic vehicle*** (and, by anomaly magnitude, helps separate light wheeled from heavy tracked/armoured), rejecting large-animal or wind/cultural seismic false alarms.
- **Seismic detection but *no* magnetic anomaly → non-metallic target** (animal, person without significant steel), down-weighting the threat.
- **Bearing/timing fusion:** the magnetometer's direction-of-travel and the geophone's range/bearing cues cross-validate a single track and reject uncorrelated clutter in either channel.

This is the same multi-modal philosophy REMBASS/IREMBASS already embodies — seismic+acoustic+magnetic+PIR fused for low false-alarm classification — adapted to an edge-ML node where the magnetometer supplies the **metal/intent** feature that a geophone fundamentally cannot.

---

## 6. Summary and recommendations for GeoSense

1. The project needs **passive magnetometry (MAD)**, not "electromagnets," as the always-on UGS metal-confirm channel: passive, low-power, omnidirectional, detects the ferrous mass of weapons and vehicles via 1/r³ distortion of the Earth's field.
2. **Sensor choice:** a **3-axis fluxgate** (sub-nT, tens of pT/√Hz) as baseline — proven in REMBASS/MAGID UGS — with a **TMR/AMR magnetoresistive** array as a lower-power, lower-cost alternative for the vehicle channel. Reserve optically-pumped/atomic for fixed, mains-powered stand-off gates.
3. **Expect range:** ~**1–3 m** to confirm a carried weapon, **tens of metres up to ~100+ m** to confirm a metallic vehicle — set by the 1/r³ law, the target moment (weapon ≪ a few A·m²; car ≈ 100–300 A·m²), and the geomagnetic-background-limited noise floor.
4. **Implement gradiometry + reference subtraction + band-pass/ML signature classification** to suppress diurnal variation (~25 nT), ULF micropulsations, ferrous clutter, and platform bias, achieving the low false-alarm rate required for a deployed sensor.
5. **Add active EMI only for close-range/portal/clearance roles** where its 1/r⁶ short reach is acceptable and its unique non-ferrous-metal sensitivity is wanted.
6. **Fuse with the geophone** so that *seismic identity* (human/vehicle/animal) × *magnetic metal-confirm* (armed/metallic vs not) yields an **intent/threat** decision neither sensor can make alone.

---

## References

1. Tan, M. *et al.* **Magnetic Anomaly Detection Using a Three-Axis Magnetometer.** ResearchGate. https://www.researchgate.net/publication/224380190_Magnetic_Anomaly_Detection_Using_a_Three-Axis_Magnetometer
2. **Detection of vehicle tracks by a three-axis magnetometer.** *Sensors and Actuators A*, ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0924424717319301
3. **Magnetic anomaly detector.** Grokipedia. https://grokipedia.com/page/Magnetic_anomaly_detector
4. **Magnetic Dipole Moment and Magnetic Dipole Media** (axial/equatorial dipole field, 1/r³). Physics LibreTexts. https://phys.libretexts.org/Bookshelves/Electricity_and_Magnetism/Essential_Graduate_Physics_-_Classical_Electrodynamics_(Likharev)/05:_Magnetism/5.04:_Magnetic_Dipole_Moment_and_Magnetic_Dipole_Media
5. **Portable Single-Beam Atomic Total-Field Magnetometer for Stand-off Magnetic Sensing** (ΔB ∝ D⁻³ confirmation). arXiv 2601.08716. https://arxiv.org/pdf/2601.08716
6. **Magnetic moment.** Wikipedia. https://en.wikipedia.org/wiki/Magnetic_moment
7. **Earth's magnetic field** (25,000–65,000 nT; ~25 nT diurnal variation). Wikipedia. https://en.wikipedia.org/wiki/Earth%27s_magnetic_field
8. **Introduction to Geomagnetism.** U.S. Geological Survey. https://www.usgs.gov/programs/geomagnetism/introduction-geomagnetism
9. **Remote Battlefield Sensor System (REMBASS) / IREMBASS** (magnetic sensor ~3 m; personnel/wheeled/tracked ranges; ferrous-metal detection). Federation of American Scientists. https://man.fas.org/dod-101/sys/land/rembass.htm
10. **REMBASS II – Remotely Monitored Battlefield Sensor System** (AN/GSR-8; fluxgate + DSP; seismic/acoustic/magnetic/IR fusion; low false-alarm). Defense Update. https://defense-update.com/20060107_rembass-ii-remotely-monitored-battlefield-sensor-system.html
11. **MAGID-II: a next-generation magnetic unattended ground sensor (UGS).** SPIE 8388. https://www.spiedigitallibrary.org/conference-proceedings-of-spie/8388/1/MAGID-II--a-next-generation-magnetic-unattended-ground-sensor/10.1117/12.917501.short
12. **REMBASS-II: status and evolution of the Army's UGS system.** SPIE 4743. https://www.spiedigitallibrary.org/conference-proceedings-of-spie/4743/0000/REMBASS-II--the-status-and-evolution-of-the-Armys/10.1117/12.448390.short
13. **Dynamic Vehicle Detection via the Use of Magnetic Field Sensors** (AMR sensors; vehicle distorts Earth field; vehicle magnetic moments ~100–300 A·m²). PMC4732111. https://pmc.ncbi.nlm.nih.gov/articles/PMC4732111/
14. Bigman, D. P. *et al.* **Suitability of magnetometry to detect clandestine buried firearms from a controlled field site and numerical modeling** (rifle ±20 nT @ 0.6 m; handgun ±2 nT @ 1.8 m; long-dipole model; Overhauser gradiometer, 0.55 m separation; 0.25 m grid). *Forensic Sci. Int.*, 2020. https://pubmed.ncbi.nlm.nih.gov/32663720/ ; https://www.sciencedirect.com/science/article/abs/pii/S0379073820302589
15. Kotter, D. *et al.* **Detection and classification of concealed weapons using a magnetometer-based portal.** SPIE 4708. https://ui.adsabs.harvard.edu/abs/2002SPIE.4708..145K/abstract
16. **Hands-off Frisking: High-Tech Concealed Weapons Detection.** National Institute of Justice (INEEL passive-magnetic portal, neural-net discrimination). https://nij.ojp.gov/library/publications/hands-frisking-high-tech-concealed-weapons-detection
17. **Magnetometer Gun Detection: How It Works.** Garrett. https://garrett.com/magnetometer-gun-detection/
18. **1 pT-noise fluxgate magnetometer design and its performance in geomagnetic measurements.** ResearchGate. https://www.researchgate.net/publication/335722405
19. **A Multi-Pass Optically Pumped Rubidium Atomic Magnetometer with Free Induction Decay.** PMC9572103 / arXiv 2301.08437. https://pmc.ncbi.nlm.nih.gov/articles/PMC9572103/ ; https://arxiv.org/pdf/2301.08437
20. **Fast and robust optically pumped cesium magnetometer** (~3.5 pT/√Hz @ 1 Hz). Frontiers / Advanced Optical Technologies. https://www.frontiersin.org/articles/10.1515/aot-2020-0024/pdf
21. **Detection of mines with the EMI method.** FOERSTER Group (transmit coil → eddy currents → secondary field). https://blog.foerstergroup.com/en/foerster-group/clearance-of-mines-with-emi-method
22. **Metal detector.** Wikipedia (induction balance; VLF/PI/BFO; eddy-current secondary field). https://en.wikipedia.org/wiki/Metal_detector
23. **Detection and characterisation of conductive objects using electromagnetic induction and a fluxgate magnetometer.** arXiv 2206.12187. https://arxiv.org/pdf/2206.12187
24. **Detection and classification from electromagnetic induction data** (dipole–dipole EMI response, near-field falloff). arXiv 1308.6027. https://arxiv.org/pdf/1308.6027
25. Nelson, C. V. **Metal Detection and Classification Technologies** (active EMI physics; transmit/secondary field; ~1/r⁶ scaling; landmine detection). *Johns Hopkins APL Technical Digest* 25(1). https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/V25-N01/25-01-Nelson.pdf
26. **Pulse Induction vs VLF metal detectors** (depth vs discrimination; soil mineralisation). Garrett / gearupgrades. https://garrett.com/understanding-how-pulse-induction-metal-detectors-work/
27. **Geomagnetic micropulsations (Pi2/Pc) and world-wide characteristics** (ULF noise bands). ScienceDirect / ResearchGate. https://www.sciencedirect.com/science/article/abs/pii/0032063373900925 ; https://www.researchgate.net/publication/229514647_World-Wide_Characteristics_of_Geomagnetic_Micropulsations
28. **Energy Detection Based on Undecimated Discrete Wavelet Transform and Its Application in Magnetic Anomaly Detection** (denoising/clutter rejection). PMC4208775. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4208775/

---

*Notes on rigor and caveats:* Detection-range figures are order-of-magnitude estimates from the 1/r³ dipole law with representative target moments and sensor noise floors; real range depends strongly on target magnetisation history, orientation, local geology, and the geomagnetic-noise-limited B_min at the specific site. Vehicle moment values (100–300 A·m²) vary by vehicle and source. For a publishable claim, the GeoSense weapon-confirm and vehicle-confirm ranges should be measured empirically at the deployment site with the actual sensor/gradiometer baseline, not quoted from the literature alone.
