> **Research document — house-localization pivot ("wallhacks").** Targeted web research + quantitative analysis, 2026-07-09. Builds on doc-11 (confuser taxonomy §4), doc-05 (structural reverberation), doc-00 (SPECFEM3D inputs). Indexed in README.md.

---

# Appliances, Fixtures, and Plumbing: Vibration Sources and Structural Representation

**Scope.** This document addresses two questions: (1) which building appliances inject structure-borne vibration that competes with footstep signals (confuser sources), and (2) whether any appliance or fixture must be represented as meshed geometry or mass in the 3-D elastic simulation, or whether the doc-11 strategy (STF ⊛ Green's function, post-hoc) is sufficient.

**Bottom line up front.** The doc-11 approach — treat all appliances as located vibration sources convolved post-hoc with the precomputed Green's function — is correct for all Israeli residential appliances. No appliance needs to appear as meshed geometry. One appliance (the roof-mounted solar water heater, "dud shemesh") warrants representation as an added point mass on the roof slab node because its mass ratio is large enough to produce a measurable resonance shift; all others are negligible mass loads. The ranked priority confuser list is at §5.

---

## 1. Appliances as Vibration Sources (Confusers)

### 1.1 Split-Unit AC Compressor (Outdoor Unit, Wall- or Roof-Mounted)

**Prevalence in Israel.** Wall-mounted mini-split ("mitznan") systems dominate Israeli residential construction; >90% of apartments have at least one outdoor unit bolted to an exterior concrete or masonry wall, typically at 1.5–3 m height on the living-room or bedroom façade. Some units are roof-mounted on flat concrete roofs. Masonry/concrete mounting is the dominant case, making direct hard coupling the norm rather than the exception.

**Compressor type and speed.** Standard single-speed units run 2-pole motors at 2,900 RPM on the 50 Hz grid (Israel uses 50 Hz, 230 V per IEC standards). Inverter (variable-frequency) units sweep from ~900–4,500 RPM depending on load. The scroll compressor is the dominant type in residential mini-splits.

**Dominant vibration frequencies.** For a fixed-speed 2-pole/50 Hz scroll compressor running at 2,900 RPM:
- Shaft rotation frequency: 2,900 / 60 = **48.3 Hz** (fundamental)
- 2nd harmonic: **96.7 Hz**
- 3rd harmonic: **145 Hz**
- 4th harmonic: **193 Hz**

In practice the working frequency is slightly below 48.3 Hz under partial load. Measured spectra of residential compressors show the dominant tonal peak at the **63 Hz one-third-octave band** (covering 56–71 Hz), with significant energy in the 125 Hz band (second harmonic cluster), and a broad low-frequency hump from discharge-pulsation turbulence from 25–60 Hz. [Sources: Wiley Shock and Vibration 2021; ResearchGate condenser-compressor noise study; HVAC-Talk forum data.]

For **inverter units**, the fundamental sweeps continuously, producing a chirping broadband signature from ~15–75 Hz rather than fixed tonal peaks. This makes inverter compressors a harder confuser to notch-filter but also harder to synthesize precisely — the STF must be a frequency-swept sinusoid:

```python
def stf_inverter_ac(t, f_lo=15.0, f_hi=75.0, ramp_time=120.0):
    """Inverter AC sweep: slow thermal-load modulated FM."""
    f_inst = f_lo + (f_hi - f_lo) * 0.5 * (1 - np.cos(np.pi * t / ramp_time))
    phase = 2 * np.pi * np.cumsum(f_inst) / len(t) * (t[1] - t[0])
    return np.sin(phase)
```

**Structural coupling path.** Wall-bracket → concrete/masonry wall → floor slab via wall-slab junction. The junction transmission loss (concrete to concrete) is <6 dB per junction at <100 Hz (Liverpool structure-borne sound database). The compressor vibration therefore appears at the exterior-wall-mounted geophone sensors with low insertion loss, making it the highest-amplitude continuous confuser. Refrigerant lineset provides a secondary path (soft tubing acts as a vibration bridge to the interior where the lineset penetrates the wall).

**Key localization impact.** The outdoor unit is at a fixed known location on the exterior wall. Its vibration creates a quasi-continuous 48–100 Hz signal that a classifier must reject without false-positiving on a nearby footstep. It is the **primary hard-negative confuser** for the geophone array. Each unit's location is constrained to the building façade, so its position prior is the exterior wall grid cell nearest to the installation bracket.

**Amplitude relative to footstep.** Measured structure-borne vibration from split-unit compressors at 1–2 m from mounting bracket reaches 0.1–1.0 mm/s peak velocity in the 50–100 Hz band. Footstep-induced velocity at a wall sensor 3–5 m from the footstep is typically 0.01–0.1 mm/s in the same band (estimated from geophone sensitivity ~2.88 V/(mm/s) and typical amplifier output, footstep GRF ~700–1,200 N at 2,900 Hz bandwidth). The compressor is therefore a 10–40 dB stronger source than a distant footstep, making it a severe confuser when the geophone is near the AC bracket.

---

### 1.2 Refrigerator Compressor

**Type and location.** Hermetic reciprocating or rotary compressor, located in the base of the appliance, on the kitchen floor or recessed in cabinetry.

**Dominant vibration frequencies.** The refrigerator compressor in a 50 Hz grid runs at 1,450–2,900 RPM depending on pole count. A 2-pole compressor: fundamental **48 Hz**; 4-pole: fundamental **24 Hz**. Measured dominant structure-borne peaks in the literature occur at **50 Hz and 100 Hz** (Frontiers Energy Research 2022; Purdue ICEC). The hermetic housing transmits vibration to the appliance cabinet feet, thence to the floor.

**Coupling.** Cabinet rubber feet provide ~10–15 dB isolation above ~20 Hz; residual coupling gives 0.001–0.01 mm/s floor velocity at 1 m distance. This is below a typical footstep at 3–5 m range but becomes significant if a geophone is placed close to a kitchen wall.

**Character.** Intermittent: duty cycle 40–70%; on for 5–20 minutes, off for 5–15 minutes. On/off transitions produce a brief broadband transient (0.1–0.5 s) that can resemble a footstep onset.

**Key localization impact.** Fixed location (kitchen). Amplitude low relative to AC compressor. Primary risk: the on/off transient triggers false detection. Classified as a **secondary confuser** (continuous but low-amplitude; transient at cycle boundaries).

---

### 1.3 Washing Machine (Spin Cycle)

**Vibration profile.** A loaded drum (full wash: machine 70–100 kg + 5 kg laundry + water) at spin speed 600–1,200 RPM:
- 600 RPM → **10 Hz** fundamental unbalance rotation
- 800 RPM → **13.3 Hz**
- 1,200 RPM → **20 Hz**

During spin-up, the fundamental sweeps from 0 to the plateau frequency, passing through floor resonances and generating transient broadband bursts when the sweep rate matches a modal frequency (Purdue washing machine vibration analysis; COMSOL washing machine blog). Measured floor vibration from an unbalanced spin cycle reaches **0.015–0.5 m/s²** RMS acceleration on a concrete floor (ISO 2631-1 comfort threshold is 0.015 m/s² — spin cycles routinely exceed this). Converted to velocity: at 10 Hz, 0.1 m/s² = **1.6 mm/s** — this is the **strongest structure-borne source** among common residential appliances, exceeding the AC compressor at close range.

**Frequency spread.** During spin-up, unbalance force sweeps 0–20 Hz with harmonics to 60+ Hz. At plateau, dominant peak at 10–20 Hz (first harmonic) and strong 2nd, 3rd harmonics at 20–40 Hz and 30–60 Hz. Wall-vibration spectra peak in the 0–30 Hz range with a mode at ~1.67 Hz in the appliance frame. [Source: GreenBuildingAdvisor structural reports; ResearchGate washing machine unbalanced mass analysis.]

**Key localization impact.** Fixed location (laundry room or kitchen). Extremely strong at low frequencies (10–40 Hz). Spin-up chirp is transient and broadband — most likely appliance to be misclassified as a footstep sequence in the 10–40 Hz band. **Highest-priority hard-negative for the classifier in the low-frequency band.** The spin-up phase (30–90 s long) produces a continuous FM sweep that must be represented with a time-varying STF.

```python
def stf_washing_spinup(t, f_plateau=13.3, t_ramp=60.0, t_plateau_start=60.0):
    """
    Washing machine spin-up: linear frequency ramp from 0 to f_plateau,
    then steady state. Unbalance force ~ mass * r * omega^2 grows as f^2.
    """
    f_inst = np.where(t < t_ramp, f_plateau * t / t_ramp, f_plateau)
    phase = 2 * np.pi * np.cumsum(f_inst) * (t[1] - t[0])
    amplitude = np.clip((t / t_ramp) ** 2, 0, 1)   # force grows as omega^2
    return amplitude * np.sin(phase)
```

---

### 1.4 Water Pump / Circulation Pump

**Context in Israel.** Israeli apartment buildings commonly have: (a) a domestic hot water circulation pump (speeds water recirculation between dud shemesh tank and taps), 50 Hz operated; (b) a booster pump if municipal pressure is insufficient (common in upper floors of Israeli apartment buildings ≥5 stories).

**Vibration frequencies.** For a 50 Hz grid pump: motor fundamental **50 Hz**, dominant harmonic **100 Hz** (confirmed by multiple sources: Virgo site noise paper; residential condominium pump study; pump noise inspection database). The 100 Hz line is the dominant structure-borne signature because magnetostriction in the motor core produces vibration at twice the electrical frequency. Additional blade-pass frequency depends on impeller blade count × RPM (typically 200–800 Hz for centrifugal pumps, above our 250 Hz upper limit).

**Coupling.** Pipe-borne and structure-borne. In Israeli concrete construction, pipes are embedded in wall and floor slabs (cast-in conduits), providing a very stiff coupling path from pump to the whole building skeleton. Booster pumps in the basement or on roof plant rooms transmit via pipe network.

**Key localization impact.** Fixed known location (utility room, basement, or roof). Generates a persistent 50 Hz + 100 Hz tone in the structure. In combination with mains hum (see §1.7), the 50 and 100 Hz bins are the most contaminated in the geophone spectrum. **Secondary confuser; fixed signature makes notch filtering viable.**

---

### 1.5 Solar Water Heater — "Dud Shemesh" (Roof-Mounted)

**Prevalence.** By Israeli law (1980), solar water heaters are mandatory on all new residential buildings. Approximately 80% of Israeli households use a dud shemesh. The typical Israeli system comprises a flat-plate or evacuated-tube collector panel (2 m², dry mass ~35 kg) plus an insulated tank of 150–200 liters on the roof slab.

**Mass when full.**
- Tank (150 L): empty mass ~55 kg, full mass = 55 + 150 = **205 kg**
- Collector panel: ~35 kg
- Mounting frame + pipework: ~20 kg
- **Total system on roof: ~260 kg when full of water** [Source: GMOD150 product spec; DudaDiesel 150 L tank spec; various solar water heater vendor data.]

**Vibration sources.** The passive dud shemesh (no pump) generates vibration primarily from:
1. **Thermal expansion/contraction** of the collector panel and connecting pipes — slow creaking, <1 Hz, not a confuser in the geophone band.
2. **Circulation pump** (if active/forced-circulation type): 50 Hz + 100 Hz tones as in §1.4.
3. **Water-hammer transients** when solar-heated water is drawn: broadband 35–130 Hz pulse (see §3).

The passive dud shemesh (thermosiphon, no pump — most common in Israel) produces **no continuous vibration** once filled. Its significance is therefore primarily as a **mass load** on the roof slab, not as a vibration source.

---

### 1.6 Electric Water Heater (Dud Hashmal — Backup Element)

Israeli dud shemesh systems include an electric backup heating element that activates when solar input is insufficient. The element itself (resistive) produces no mechanical vibration. The only vibration is expansion noise in the tank when element cycles on — negligible below 250 Hz. **Not a confuser source.**

---

### 1.7 Mains Hum — 50 Hz Grid, Electromagnetic Coupling

**Character.** The Israeli grid operates at 50 Hz (IEC standard). Mains hum in geophones is predominantly **electromagnetic (EM) induction** into the moving-coil sensor. The dominant frequency is **100 Hz** (magnetostriction in transformers and motors operates at 2× the supply frequency; Wikipedia Mains hum article; HandWiki Physics:Mains hum). The 50 Hz fundamental and harmonics at 150, 200 Hz are also present but weaker in EM coupling.

**Critical distinction from structure-borne.** Mains hum enters the geophone signal through two independent pathways:
1. **EM induction**: directly into the coil/cable, independent of structural vibration. This component is spatially incoherent across sensors (depends on cable routing, proximity to transformers) and cannot be modeled by any STF ⊛ GF convolution. It must be added **additively post-hoc** as a stochastic tone: `A_em * sin(2π×100t + φ)` with `φ` randomized per sensor, `A_em` ~ 0.005–0.10 × signal RMS.
2. **Structural 50 Hz vibration**: pumps, compressors, and transformer mounting-feet inject 50 Hz tones into the structure, which then propagate to the geophone as mechanical vibration. This component can be modeled via STF ⊛ GF with a 50 Hz sinusoid STF.

The EM component is always additive. Both components must be included for a realistic geophone training signal.

---

### 1.8 Elevator (Apartment Buildings)

**Prevalence.** In Israel, most mid-rise apartment buildings (4+ stories) have one or more traction elevators. This is relevant because the house-localization project may be deployed in apartment buildings.

**Vibration frequencies.** Traction elevator machinery produces:
- Motor running at low RPM (100–500 RPM): dominant at **1.5–8 Hz** (car frame / guide rail excitation per MDPI Buildings 2022; Elevator Noise and Vibration MDPI 2020).
- Electromagnetic noise from traction motor stator: **up to 1,000 Hz**, concentrated in the 125–500 Hz band.
- Guide-rail excitation frequency: `f_r = v / L_rail` where v = car speed (1–2.5 m/s) and L_rail = 5 m rail segments → `f_r = 0.2–0.5 Hz` — sub-Hz, filtered by any HP cutoff.
- Measured dominant structure-borne component in apartments: **<63 Hz, concentrated <32 Hz** (BNAM 2018 paper; ResearchGate elevator noise multi-story).

**Key localization impact.** Elevator is intermittent (operates only during active use); vertical motion + machine room vibration transmit to building frame. In a 6-story apartment building: vibration couples from the machine room and guide rails through the concrete shaft into floor slabs. The elevator shaft acts as a line source; signals arrive at exterior geophones after multiple junction transmissions, giving significant insertion loss (>15 dB per junction). **Tertiary confuser**; significant only in apartment buildings, not single-family houses; priority below AC and washing machine.

---

## 2. Appliances as Mass / Structural Loading

### 2.1 Framework: When Does a Point Mass Matter?

For a simply supported plate with modal mass M_modal and added concentrated mass Δm at a modal antinode:

The frequency shift formula (Rayleigh quotient approximation) is:

```
f_loaded / f_bare = 1 / sqrt(1 + Δm/M_modal)
```

For small Δm/M_modal, this gives a frequency reduction of approximately `Δm / (2 × M_modal)` in relative terms.

For a plate, M_modal ≈ ρ_concrete × h × A × 0.25 (the effective modal mass is ~25% of total plate mass for the fundamental mode of a simply-supported plate). [Source: Extrica/ResearchGate plate concentrated mass study, which shows 1% mass ratio → -1.96% frequency reduction; 10% → -15.91% for mode-1 at center.]

**Concrete roof slab mass (10×12 m, h=0.15 m):**
```
M_slab = 10 × 12 × 0.15 × 2,400 = 43,200 kg ≈ 43 tonnes
```
This is for the roof panel alone. For a full residential building (walls + slabs + columns), the structural mass is easily 100–300 tonnes.

### 2.2 Appliance Mass Survey

| Appliance | Typical Mass (operating) | Location |
|---|---|---|
| Split AC outdoor unit (small, 9,000–12,000 BTU) | 30–45 kg | Exterior wall bracket |
| Split AC outdoor unit (large, 18,000–24,000 BTU) | 45–90 kg | Exterior wall bracket |
| Refrigerator | 60–120 kg | Kitchen floor |
| Washing machine (front-load, full) | 70–100 kg | Laundry floor |
| Dud shemesh (150 L, full + collector) | ~260 kg | Roof slab |
| Electric water heater (80 L, standing) | ~120 kg | Bathroom/utility wall |
| Booster pump assembly | 20–50 kg | Utility room floor |

[Sources: Appliance Analysts AC weight database; GreenBuildingAdvisor mini-split weight; Amazon/DudaDiesel solar tank specs.]

### 2.3 Quantitative Mass-Loading Assessment

**Dud shemesh on roof slab:**
- Added mass: Δm = 260 kg
- Roof slab effective modal mass (10×12 m, h=0.15 m): M_modal ≈ 0.25 × 43,200 = 10,800 kg
- Mass ratio: Δm / M_modal = 260 / 10,800 = **2.4%**
- Frequency reduction (mode 1, antinode location, from Extrica table): approximately **-2% to -4%**
- For a roof fundamental at 20 Hz (thin slab at 10 m span), this is a shift of **0.4–0.8 Hz**

This shift is detectable by the simulation (≈1–4 Hz for higher modes nearer the tank placement, from the Extrica table showing 10% mass ratio → ~16% reduction) and is large enough to meaningfully perturb the Green's function in the 15–30 Hz band where roof resonances occur.

**AC outdoor unit on exterior wall:**
- Added mass: 30–90 kg on a single wall panel
- Exterior wall panel (concrete, 3 m × 2.7 m × 0.20 m): mass = 3 × 2.7 × 0.2 × 2,400 = 3,888 kg; M_modal ≈ 970 kg
- Mass ratio (90 kg): 90 / 970 = **9.3%**
- Frequency reduction (9.3% mass at near-antinode): approximately **-8% to -12%** on wall panel modes
- Wall panel fundamentals in 15–80 Hz range → shift of 1–8 Hz

This is **non-negligible for the wall resonances but very local to the wall panel**. The slab GF is less affected because the slab has much higher mass. However, the wall-panel resonance shift does perturb the GF between a nearby geophone and sources through that wall. **This effect is secondary (affects only the wall-panel modes, not the slab modes used for primary TDOA) but non-trivial for sensors mounted close to the bracket.**

**Washing machine and refrigerator on floor slab:**
- Floor slab (10×12 m, h=0.15 m): M_slab = 43,200 kg; M_modal ≈ 10,800 kg
- Washing machine: 100 kg; mass ratio = 100 / 10,800 = **0.93%** → frequency reduction < **-1%**
- Refrigerator: 120 kg; mass ratio = 1.1% → < **-1.1%**
- At these mass ratios the Extrica table predicts < 2 Hz shift on any slab mode
- **Conclusion: negligible.** Floor slab modes are essentially unperturbed by individual domestic appliances. The 100-tonne structural mass dwarfs these loads.

**Summary table:**

| Appliance | Mass (kg) | Attached to | Mass ratio | Δf (freq shift) | Mesh? |
|---|---|---|---|---|---|
| Dud shemesh (full) | 260 | Roof slab | 2.4% | 0.4–0.8 Hz (roof modes) | Optional point mass |
| AC outdoor unit (large) | 90 | Exterior wall panel | 9.3% | 1–8 Hz (wall panel modes) | Optional point mass |
| Washing machine | 100 | Floor slab | 0.9% | <1 Hz | None |
| Refrigerator | 120 | Floor slab | 1.1% | <1 Hz | None |
| Electric water heater | 120 | Wall/floor | 1–3% | <2 Hz (local) | None |

**Verdict: No appliance needs to be meshed as geometry.** The only candidates for added-mass representation are the dud shemesh and a large AC unit, and even these can be handled as point-mass perturbations (described in §4).

---

## 3. Plumbing: Pipes, Water Hammer, and Material Inclusions

### 3.1 Embedded Pipes as Material Inclusions

Israeli residential construction buries water supply (copper, 15–22 mm diameter) and drainage pipes (PVC, 50–110 mm diameter) in cast-in conduits within concrete slabs and walls. These are voids or stiffness inclusions in the concrete matrix.

**Effect on elastic wave velocity.** NDT research on concrete with voids and inclusions shows: a 2.1% reduction in ultrasonic pulse velocity in inclusion areas vs. pristine concrete (NDT-E International 2025). For P-wave velocity in concrete (Vp ~ 3,500–4,200 m/s), a 2.1% change is 73–88 m/s. In the 10–250 Hz seismic band (much longer wavelengths than NDT frequencies), the effect is smaller: the pipe void occupies < 0.1% of the cross-sectional area of a typical slab, and effective-medium theory gives velocity perturbation ∝ (void volume fraction) × (velocity contrast factor). For a 22 mm copper pipe in a 150 mm slab:

```
Pipe cross-section area: π × (0.011)² = 3.8 × 10⁻⁴ m²
Slab cross-section (1 m strip): 0.150 m²
Volume fraction: 3.8×10⁻⁴ / 0.150 = 0.25%
Effective Vs perturbation: < 0.25% × (velocity contrast) ≈ < 0.5%
```

At 0.5% velocity perturbation, the arrival time error for a 3 m path is 3 / 0.995 × V_s - 3 / V_s ≈ 0.015 ms — negligible against a 0.5 ms localization target. **Embedded residential plumbing pipes do not need to be represented as material inclusions in the mesh. The homogeneous concrete medium is adequate.**

### 3.2 Water-Hammer Transients as Sources

Water hammer is triggered by fast valve closure (toilet flush, washing machine solenoid, any quick-acting valve). The Frontiers Energy Research 2022 FSI study of pipes embedded in concrete identifies dominant vibration frequencies transmitted from the pipe to the surrounding concrete:

| Mode | Frequency |
|---|---|
| Pressure wave in fluid | 35.2 Hz |
| Stress wave in pipe wall | 129.3 Hz |
| Stress wave transmitted to concrete | 99.6 Hz |
| Full modal range (8 modes) | 35–450 Hz |

The frequencies 35–130 Hz fall squarely in the footstep localization band. A water-hammer transient has a duration of 0.1–0.5 s and a broadband waveform (decaying sinusoid centered around the pipe resonance). [Source: Frontiers in Energy Research doi:10.3389/fenrg.2022.956209; ASCE Journal of Pipeline Systems Engineering 2024.]

**Critical localization risk.** A toilet flush or washing machine solenoid produces a 0.1–0.5 s broadband transient in the 35–130 Hz band. This temporal-spectral signature is similar to a footstep heel-strike transient. The key distinction: water hammer propagates along the pipe route (a line source, not a point), and the apparent TDOA from multiple sensors will not be consistent with any single floor-grid position. The classifier should learn this spatial-incoherence signature.

**STF for water hammer:**
```python
def stf_water_hammer(t, t0=0.05, f_center=35.0, tau=0.03, decay=0.10):
    """
    Water hammer: decaying broadband burst centered at f_center Hz.
    t0: valve closure time; tau: rise time; decay: exponential decay.
    """
    envelope = np.exp(-decay * np.abs(t - t0)) * (t >= t0 - tau)
    return envelope * np.sin(2 * np.pi * f_center * (t - t0))
```

**GF position.** The pipe enters the building from the utility shaft or exterior wall; branching occurs at each floor via risers. The nearest floor-grid cell to the riser (typically a corner of the kitchen or bathroom) is the representative injection point. Multiple positions along the pipe route can be included at negligible cost (each uses an already-precomputed GF column).

### 3.3 Drain Flow Noise

Running water in drain pipes (50–110 mm PVC or cast iron) produces turbulent flow noise in the 200–2,000 Hz band. Below 200 Hz the signal is weak (broadband noise floor, PSD ∝ f). At geophone sensitivity levels, drain flow below 250 Hz is below the noise floor of a 4.5 Hz geophone at typical drain distances (>1 m). **Not a significant confuser in the 1–250 Hz localization band.** Ignore in simulation.

---

## 4. 3-D Simulation Representation: What Goes in the Mesh vs. Post-Hoc

### 4.1 Decision Principle

A source or feature needs to be in the SPECFEM3D mesh if and only if it modifies the **Green's function** (transfer function from any floor position to any sensor) in a way that is too large to treat as a post-hoc perturbation. Everything else is a post-hoc additive signal: `signal_k(t) = STF_confuser * GF[k, pos_appliance, :]`.

### 4.2 What Must NOT Be in the Mesh (Post-Hoc Only)

All vibration sources go post-hoc. Specifically:

1. **AC compressor**: STF = band-limited quasi-sinusoid at 48 Hz (fixed-speed) or FM sweep (inverter); position = exterior wall grid cell; no meshing.
2. **Refrigerator**: STF = narrowband 48 Hz with on/off envelope; position = kitchen floor cell; no meshing.
3. **Washing machine**: STF = time-varying unbalance force (see §1.3 STF); position = laundry room floor cell; no meshing.
4. **Water hammer**: STF = decaying burst 35–130 Hz; position = nearest pipe-route floor cell; no meshing.
5. **Circulation pump**: STF = 50 Hz + 100 Hz sinusoids; position = utility/roof cell; no meshing.
6. **Mains hum (EM)**: additive sinusoid 100 Hz, per-sensor, NOT via GF.
7. **Elevator**: STF = low-frequency sweep 2–8 Hz; position = nearest shaft wall cell; no meshing.
8. **Pet footsteps**: STF = scaled-down GRF (as in doc-11 §4.2); floor grid position; no meshing.

### 4.3 What MAY Benefit From Mesh Representation (Point Mass)

SPECFEM3D Cartesian supports adding a scalar point mass to individual mesh nodes by scaling the density in the element or by using a lumped-mass diagonal term. Both approaches are O(1) cost additions; they do not require geometry changes.

**Dud shemesh (260 kg on roof slab).** The mass ratio of ~2.4% on the roof panel produces a meaningful shift in roof-slab resonances (0.4–0.8 Hz for the fundamental, more for higher modes). If the localization system uses fingerprint features from roof-slab resonances (i.e., if sensors are placed on or near the roof), this shift is worth capturing. **Recommendation: add a lumped point mass of 260 kg at the roof slab node nearest to the tank's footprint center.** In SPECFEM3D this is implemented by increasing the density of the 1–2 elements at that node to give the correct nodal mass in the diagonal mass matrix.

**Large AC outdoor unit (≥90 kg).** The 9% mass ratio on the local wall panel shifts wall-panel resonances by 8–12%. If sensors are placed on the same wall panel, this affects the wall-panel transfer function at 20–80 Hz. **Recommendation: add a lumped point mass of 60–90 kg at the wall bracket node.** This is optional and can be dropped from the baseline simulation; include it in the "high-fidelity" configuration variant.

### 4.4 What Must NOT Be Represented as Geometry

No appliance needs to appear as a structural element (solid mesh block). Reasons:
- The appliance bodies are not load-bearing; they are not mechanically integral to the building frame.
- Their stiffness contribution is negligible vs. the concrete skeleton.
- Voxelizing an appliance (e.g., a 60 cm × 60 cm × 85 cm refrigerator) in a SPECFEM3D mesh would consume mesh elements and require a cell size ≤ 10 cm to capture it accurately, increasing simulation cost without improving the Green's function at any point outside the immediate vicinity of the appliance.

---

## 5. Ranked Confuser Priority and Source Parameters

### 5.1 Priority 1 — Must Include in All Training Sets

| Source | STF character | Frequency range | Location prior | Why critical |
|---|---|---|---|---|
| **AC outdoor compressor (fixed-speed)** | Quasi-sinusoidal, 48 Hz + harmonics 96/145/193 Hz; continuous | 30–200 Hz | Exterior wall cell, 1.5–3 m height | Highest amplitude; continuous; hard negative for 50–100 Hz band; always present in Israeli houses with AC |
| **Washing machine (spin cycle)** | FM sweep 0→13 Hz over 60 s, then steady 13 Hz + harmonics; strong 0–40 Hz | 0–60 Hz | Laundry room / kitchen floor cell | Strongest amplitude source; sweep mimics footstep sequence; dominant false-positive risk in 10–40 Hz band |
| **Water hammer transient** | Decaying broadband 35–130 Hz burst, 0.1–0.5 s duration | 35–130 Hz | Kitchen / bathroom wall cell (pipe riser) | Temporal profile mimics footstep; occurs at predictable household events (toilet flush, tap); spatially incoherent signature must be learned |
| **Mains hum EM (100 Hz + 50 Hz)** | Pure additive sinusoid; constant; per-sensor | 50, 100, 150, 200 Hz | N/A (additive to all sensors independently) | Always present; notchable but must appear in training or filter will not be learned |

### 5.2 Priority 2 — Include in Baseline Training, Randomize Occurrence

| Source | STF character | Frequency range | Location prior |
|---|---|---|---|
| **AC outdoor compressor (inverter)** | FM sweep 15–75 Hz; slow modulation (minutes) | 15–100 Hz | Exterior wall cell |
| **Refrigerator compressor** | Narrowband 48 Hz + 96 Hz; on-cycle 5–20 min | 40–110 Hz | Kitchen floor cell |
| **Circulation pump / booster pump** | 50 Hz + 100 Hz narrowband; continuous when active | 50, 100 Hz | Utility room / roof plant cell |

### 5.3 Priority 3 — Include in Extended Corpus (Apartment Building Scenarios)

| Source | STF character | Frequency range | Location prior |
|---|---|---|---|
| **Elevator machinery** | Tonal 4–8 Hz (motor rotation) + broadband <32 Hz; intermittent | 2–32 Hz | Shaft wall exterior cell |
| **Neighbors' footsteps (upper floor)** | Scaled GRF through inter-slab junction; attenuated by junction TL | 5–250 Hz, -15 to -25 dB vs same-floor | Ceiling cell (inter-floor path) |

### 5.4 Confuser STF Reference Parameters

```python
CONFUSER_LIBRARY = {
    "ac_fixed_speed": {
        "type": "quasi_sinusoidal",
        "f0": 48.3,           # Hz, 2-pole 50 Hz grid
        "harmonics": [1, 2, 3, 4],
        "harmonic_amplitudes": [1.0, 0.5, 0.25, 0.12],  # approx roll-off
        "phase_jitter_std": 0.05,  # rad, slight frequency wobble
        "location": "exterior_wall",  # position prior
        "amplitude_rel_footstep": (5, 100),  # dB range: 14 to 40 dB above footstep at sensor
    },
    "ac_inverter": {
        "type": "fm_sweep",
        "f_lo": 15.0,  "f_hi": 75.0,  # Hz
        "sweep_period": 300.0,  # s (thermal modulation period)
        "location": "exterior_wall",
    },
    "washing_machine_spin": {
        "type": "unbalance_ramp",
        "f_plateau": 13.3,    # Hz at 800 RPM; randomize 10–20 Hz
        "ramp_time": 60.0,    # s
        "force_model": "quadratic",   # force ∝ omega^2
        "harmonics": [1, 2, 3],
        "location": "laundry_floor",
    },
    "water_hammer": {
        "type": "decaying_burst",
        "f_center": 35.0,     # Hz fluid pressure wave (Frontiers 2022)
        "f_pipe": 99.6,       # Hz concrete coupling mode
        "tau_rise": 0.01,     # s
        "tau_decay": 0.10,    # s
        "duration": 0.3,      # s
        "location": "pipe_riser_cell",
    },
    "refrigerator": {
        "type": "narrowband",
        "f0": 48.0,           # Hz (2-pole 50 Hz)
        "harmonics": [1, 2],
        "duty_cycle": (0.4, 0.7),  # on fraction
        "on_off_transient": 0.3,   # s transient duration
        "location": "kitchen_floor",
    },
    "mains_hum_em": {
        "type": "additive_sinusoid",  # NOT via GF
        "frequencies": [50, 100, 150, 200],  # Hz
        "amplitude_rel_signal_rms": (0.005, 0.10),  # per sensor, randomized
        "per_sensor_independent_phase": True,
    },
    "pump": {
        "type": "narrowband",
        "f0": 50.0,
        "harmonics": [1, 2],   # 50 + 100 Hz
        "location": "utility_room",
    },
    "elevator": {
        "type": "low_freq_tonal",
        "f0": 4.0,             # Hz, motor fundamental at ~240 RPM
        "harmonics": [1, 2, 3],
        "location": "shaft_wall_cell",
        "intermittent": True,
    },
}
```

---

## 6. Validation of the Doc-11 Strategy

**Doc-11 thesis (§4.2):** Each confuser = `STF_confuser(t) * GF_library[k, pos_confuser, :]`, no additional simulation needed.

**Is this correct?** Yes, under the following conditions, all satisfied for the appliances above:

1. **Linear source assumption:** The appliance vibration force is small enough that the structural response is linear. Washing machines are the worst case: 0.1–0.5 m/s² floor acceleration from an unbalanced spin still falls in the linear elastic regime for concrete (concrete nonlinear strain threshold ~10⁻⁶; structural strain from 0.5 m/s² at 10 Hz is ~8×10⁻⁷ — just at the boundary). For the purpose of the GF model, linear assumption holds.

2. **Source is a point force (or can be treated as one):** Appliances with hard coupling to a single floor or wall cell can be treated as point forces. Distributed coupling (e.g., washing machine on 4 rubber feet spanning 0.6 m) is smaller than one simulation cell (0.15 m grid) only if using a coarse mesh — at 0.10–0.15 m resolution the machine spans 4–6 cells. At these scales, distributing the force across the 4 feet cells introduces < 5% error in the 1–250 Hz band (wavelength >> appliance footprint). Single-cell approximation is adequate.

3. **Confuser location is time-invariant:** True for all appliances (they don't move). The GF column at the appliance cell is computed once and reused for all samples.

4. **The GF already covers the confuser location:** True because the GF library sweeps all 1,500 floor cells per house. If the appliance is on the floor, its position is already in the library. If on an exterior wall or roof, it is NOT on the floor grid — a separate set of GF evaluations is needed for wall and roof positions. **This is the one gap in the current doc-11 design.**

**Required extension to doc-11:** The reciprocal simulation records GF at all **floor** positions. AC compressors (wall), washing machines (floor), and pumps (floor/ceiling) need GF values at their actual 3-D attachment points, not the nearest floor cell. Two options:
- **Option A:** During each reciprocal simulation, additionally record velocity at a pre-specified list of ~20 "appliance attachment points" (wall bracket cells, roof cells). This costs < 1% extra storage (20 extra time traces per simulation vs 1,500 floor traces).
- **Option B:** Use the nearest floor cell as a proxy and apply an empirical junction transmission loss factor (-3 to -12 dB, frequency-dependent) to account for the floor-to-wall transmission loss. Less accurate but zero additional simulation cost.

**Recommendation: Option A for AC (wall-mounted) and dud shemesh (roof); Option B for refrigerator and washing machine (floor-mounted, < 3 dB error).**

---

## 7. Summary of Decisions

| Question | Answer |
|---|---|
| Does any appliance need to be meshed as structural geometry? | No. All are post-hoc STF ⊛ GF sources. |
| Does any appliance need a point mass in the mesh? | Dud shemesh: yes (260 kg, roof slab, shifts roof resonances ~2–4%). AC unit: optional (90 kg wall mass shifts wall-panel resonances 8–12%; relevant only if geophone is on same panel). |
| Primary hard-negative for classifier? | AC compressor (continuous 48 Hz + harmonics); washing machine spin-up (broadband 0–40 Hz FM). |
| Does water hammer need meshing? | No. Post-hoc burst STF at pipe-riser position. |
| Do embedded pipes need to be modeled as inclusions? | No. Velocity perturbation < 0.5%; below simulation uncertainty. |
| Does mains hum go through the GF? | Structural component (50/100 Hz): yes, via pump/compressor STF ⊛ GF. EM component (100 Hz): additive per-sensor, not via GF. |
| Gap in doc-11 to fix? | AC (wall) and dud shemesh (roof) positions are off the floor grid; add ~20 wall/roof attachment points to the reciprocal simulation recorder. |

---

## Sources

- [Frontiers in Energy Research: FSI pipe in concrete, water hammer 35–130 Hz (2022)](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.956209/full) — water hammer frequencies 35.2, 99.6, 129.3 Hz
- [Extrica/ResearchGate: effects of concentrated mass on plate natural frequency](https://www.extrica.com/article/15812) — 1% mass ratio → -1.96% frequency; 10% → -15.91%
- [PMC10490749: Additional mass effect on natural frequencies](https://pmc.ncbi.nlm.nih.gov/articles/PMC10490749/) — non-linear mass-frequency relationship, sharp initial reduction
- [ResearchGate: AC condenser-compressor noise in multi-story apartments](https://www.researchgate.net/publication/344192079_Air-conditioning_condenser-compressor_noise_in_multi-story_apartment_buildings) — 63 Hz band dominant, 125 Hz harmonic
- [GreenBuildingAdvisor: mini-split wall vibration](https://www.greenbuildingadvisor.com/question/wall-hung-mini-split-low-frequency-vibration-what-vibration-isolators-are-most-effective-for-the-lower-frequencies) — coupling path: bracket → masonry wall → slab
- [HVAC-Talk: 50 Hz compressor → 2,900 RPM](https://hvac-talk.com/vbb/threads/1788061-50-HZ-compressor) — grid frequency relationship
- [Wikipedia / HandWiki: Mains hum](https://handwiki.org/wiki/Physics:Mains_hum) — dominant frequency 100 Hz (2× supply); EM induction into coils
- [ResearchGate: washing machine unbalanced mass vibration analysis](https://www.researchgate.net/publication/372974699_Vibration_Analysis_of_Washing_Machine_under_Various_Unbalanced_Mass) — vibration dominated 0–30 Hz; floor resonance excitation
- [COMSOL: washing machine vibration simulation](https://www.comsol.com/blogs/simulating-vibration-and-noise-in-a-washing-machine) — resonance through floor during spin-up
- [MDPI Buildings 2020: elevator noise in high-rise residential](https://www.mdpi.com/2071-1050/12/21/8924) — dominant <63 Hz, <32 Hz
- [BNAM 2018: estimation of residential noise from service equipment (elevator)](https://events.artegis.com/urlhost/artegis/customers/1571/.lwtemplates/layout/default/events_public/12612/Papers/2027210_Hawkins_BNAM2018.pdf) — elevator vibration concentrated <63 Hz
- [Solaripedia / GMOD150 spec: dud shemesh 150 L mass](https://www.solaripedia.com/13/61/solar_boilers_for_hot_water_(israel).html) — tank empty 55 kg, full 205 kg; collector 35 kg
- [ISO 2631-1: human comfort threshold 0.015 m/s²](https://www.iso.org/standard/13143.html) — washing machine spin exceeds this
- [NDT-E International 2025: ultrasonic defect detection concrete slab](https://www.ndt.net/article/ndt-e-intl/papers/Ultrasonic-defect-detection-in-a-concrete-slab-assisted-b_2025_NDT---E-Inter.pdf) — 2.1% velocity reduction from inclusion
- [Liverpool: Structure-borne sound in buildings](https://livrepository.liverpool.ac.uk/3077422/1/author%20version.pdf) — junction transmission loss < 6 dB per concrete-concrete joint
- [Pump noise and vibration in high-rise condominium (ResearchGate)](https://www.researchgate.net/publication/267853557_PUMP_NOISE_AND_VIBRATION_MITIGATION_IN_HIGH-RISE_RESIDENTIAL_CONDOMINIUM) — 50 Hz pump → 100 Hz dominant tone
- [ASCE Journal Pipeline Systems Engineering: water hammer wavelet analysis (2024)](https://ascelibrary.org/doi/10.1061/JPSEA2.PSENG-1469) — transmission through soil/concrete media
