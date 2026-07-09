# Complementary Sensing Modalities for a Single-Geophone Perimeter Node

**Scope.** This report surveys sensors that complement a single vertical 4.5 Hz geophone in an unattended ground-sensor (UGS) perimeter/intrusion-detection node. The geophone is cheap, passive, buried, and strong at footstep/vehicle ground-motion detection, but has structural blind spots: (1) human-vs-animal classification ambiguity, (2) no bearing/localization from a single element, (3) blindness to above-ground/aerial threats (drones), (4) limited range on light/distant targets, and (5) no read on intent or identity. For each candidate modality we cover what it adds, detection range, cost/power/size (SWaP-C), maturity (TRL), and the *fusion value* with seismic. We close with a recommended low-power complementary stack for a buried node.

Date: 2026-06-23. Context: vertical 4.5 Hz geophone, synthetic (pyprop8) training, edge SNN classifier, Israel deployment, security/defense use.

---

## 0. Baseline: what the single geophone already gives you (and where it stops)

A geophone is a passive velocity transducer; the 4.5 Hz coil sensor is the classic UGS workhorse. Footsteps couple into the soil as body waves (amplitude falling ~1/R²) and a Rayleigh surface wave (~1/R, plus frequency-dependent intrinsic attenuation), so range is strongly **soil- and gait-dependent**.

Reported, quantitative limits from the literature:
- **Footstep range:** field experiments detect walkers from ~1 m out to ~17 m on hard campus ground and out to ~120 m on favorable (firm, low-loss) sites; "stealthy" walking can be undetectable within a few meters. ([Range limitation for seismic footstep detection, SPIE 6963](https://www.researchgate.net/publication/252566889_Range_limitation_for_seismic_footstep_detection_-_art_no_69630V))
- **Vehicle range:** commercial UGS quote ~25 m for footsteps and ~100 m for vehicles as planning numbers. ([SensoGuard UGS](https://sensoguard.com/ugs-system-unattended-ground-sensors/), [RBtec Seismo](https://www.rbtec.com/perimeter-sensors-and-underground-protection/buried-sensors/unattended-ground-sensor-security-system-ugs/))
- **Human-vs-animal failure mode:** because quadruped and biped gaits produce very similar rhythmic low-frequency seismic patterns, **frequency analysis of seismic alone cannot reliably separate human from animal footsteps.** ([HSAJ, Intelligent Recognition of Acoustic and Vibration Threats](https://www.hsaj.org/articles/72); [Exemplar Selection to Distinguish Human from Animal Footsteps](https://posenhuang.github.io/papers/Exemplar_Selection_HLVD2011.pdf))
- **No bearing from one element:** a single vertical channel gives detection and crude range, but **no direction**. Bearing requires either a 3-axis geophone (P-wave polarization) or multiple spatially separated elements/arrays. ([Single and three-axis geophone: footstep detection with bearing estimation](https://www.researchgate.net/publication/252385693_Single_and_three_axis_geophone_Footstep_detection_with_bearing_estimation_localization_and_tracking))
- **Aerial blindness:** drones, low aircraft, and most above-ground activity couple poorly into the ground; the geophone is essentially deaf to them.

These five gaps define what a complement must buy you. Below, modalities are ordered roughly by fusion value for a *buried, low-power* node.

---

## 1. Acoustic (audible microphone / small MEMS array) — the highest-value complement

**What it adds.** An air-coupled microphone hears what the ground misses: voices/speech (a near-decisive human cue), vehicle engine harmonics, gunshots, and — critically — it is the natural co-modality for separating human from animal. Seismic detects the *event*; acoustic helps *classify* it. A small array (3–4 MEMS mics) additionally gives **bearing** via time-difference-of-arrival, the localization the single geophone lacks.

**Why fusion is strong.** The ARL multimodal work is the canonical evidence: fusing acoustic + seismic (+ PIR) materially raises personnel detection/classification accuracy over any single modality, and the algorithms are cheap enough for on-node real-time use. Geophones are also *less sensitive to Doppler* than mics and work passively, while mics add the spectral content (speech formants, engine firing order) seismic lacks — the two are genuinely orthogonal. ([Multimodal Sensor Fusion for Personnel Detection, DTIC ADA565006](https://apps.dtic.mil/sti/pdfs/ADA565006.pdf); [Damarla, ARL — non-imaging detection of people and animals](https://apps.dtic.mil/sti/tr/pdf/ADA564999.pdf))

**Range.** Footstep audio is short-range (often shorter than seismic on soft ground because footfall is a quiet airborne source); voices ~tens of m; vehicles 100s of m; gunshots up to km-scale. Acoustic footstep+seismic fusion datasets (e.g., AFPILD) report person ID and localization from a single mic array. ([AFPILD dataset](https://www.sciencedirect.com/science/article/abs/pii/S1566253523004979))

**SWaP-C.** Excellent: MEMS mics cost cents–dollars, draw mW, are tiny. The cost is *compute* (audio sampling/feature extraction) and *power* if you run continuously — manage with a seismic/PIR wake trigger.

**TRL.** 9 (ubiquitous in fielded UGS: Exensor Mini Mk3, McQ RANGER all pair seismic + acoustic). ([Exensor Mini Mk3](https://www.exensor.com/products/flexnet-flexible-network-of-sensors/unattended-ground-sensors-ugs/mini-mk3/); [McQ RANGER](https://www.mcqinc.com/products/mcq-ranger-unattended-ground-sensor/))

**Tradeoff.** Wind/rain noise; privacy/legal concerns for recording speech in some jurisdictions; quiet/stealthy targets defeat it just as they defeat seismic.

---

## 2. Passive infrared (PIR) — cheapest, best low-power wake/confirm

**What it adds.** PIR confirms a *warm moving body*, which is a strong, **near-site-independent** discriminator that seismic lacks — it directly attacks human-vs-(cold object) and reduces false alarms. In ARL fusion studies, "PIR sensors are almost site-independent and achieve much higher accuracy than seismic sensors in both detection and classification," and seismic+PIR composite features improve results. ([Multimodal Sensor Fusion for Personnel Detection, DTIC](https://apps.dtic.mil/sti/pdfs/ADA565006.pdf))

**Range.** Modest: 5–12 m indoor, up to ~20 m outdoors under good conditions; FoV 90–180°. ([Roombanker PIR guide](https://www.roombanker.com/blog/what-is-pir-sensor/); [VideoExperts PIR](https://www.videoexpertsgroup.com/glossary/pir))

**SWaP-C.** Best in class. µA-class standby current, event-driven wake, dollar-level cost, tiny. Ideal as the **always-on tripwire that wakes the heavier modalities (mic/radar/camera).** ([Milesight, off-grid PIR](https://www.milesight.com/security/blog/edge-ai-pir-camera))

**TRL.** 9 (Exensor ships a dedicated PIR module in Flexnet). ([Exensor PIR module](https://www.exensor.com/products/flexnet-flexible-network-of-sensors/unattended-ground-sensors-ugs/pir/))

**Tradeoff.** "Something moved" only — no ID, can't separate human from large animal by itself, and outdoor false triggers from sun-heated surfaces, wind-blown hot/cold air, and ambient swings are real. Best used *as a gate*, not a classifier.

---

## 3. mmWave / Doppler radar — bearing, range, micro-Doppler classification

**What it adds.** Active radar gives **precise range, velocity, and angle**, plus **micro-Doppler** signatures: a human's swinging limbs and a vehicle's wheel motion produce distinct spectral micro-tracks, enabling human-vs-vehicle (and gait-based human-vs-animal) classification that seismic can't do well. It directly fills the *localization/bearing* and *light-target* gaps, and it works in darkness/fog/rain. ([Micro-Doppler human/vehicle classification](https://edwin-pan.github.io/project/cs598/); [Linpowave mmWave intrusion](https://linpowave.com/blog/millimeter-wave-radar-intrusion-detection))

**Range.** 77 GHz modules reliably detect medium targets at ~30–80 m; 4D automotive-class radars reach ~350 m. 60 GHz is short-range. For perimeter human detection, plan tens of meters; the value is *quality of track*, not raw range. ([Consumer mmWave range guide](https://linpowave.com/blog/consumer-mmwave-radar-range-guide))

**SWaP-C.** Good and improving: TI IWR / Infineon XENSIV 60–77 GHz chips are low-cost (bulk automotive parts ~$10–$32; modules $10–$500+) and low-power; specialized low-duty designs reach sub-mW averages. Antenna/RF integration and compute are the real costs. ([TI mmWave](https://www.ti.com/product-category/sensors/mmwave-radar/industrial/overview.html); [Infineon 60 GHz](https://www.infineon.com/products/sensor/radar-sensors/radar-sensors-for-iot/60ghz-radar); [LintechTT pricing](https://www.lintechtt.com/mmwave-radar-sensor-pricing-guide/))

**TRL.** 8–9 for intrusion (mature COTS), high for above-ground use.

**Tradeoff.** **Active emitter** — detectable/jammable, a real concern for covert defense use (the geophone's passivity is a key asset you partially surrender). Needs line-of-sight above ground, so it cannot be fully buried; mount low/concealed. Power higher than PIR/mic when continuously scanning — gate it.

---

## 4. Acoustic drone detection (audible array) and infrasound — the aerial-threat layer

The geophone is blind to drones; this is where a node earns C-UAS relevance, which matters for the Elbit/defense angle.

**4a. Audible-acoustic drone detection.** Passive arrays exploit propeller/motor tonal signatures. Combat-proven at scale: Ukraine's networks of **14,000+ acoustic sensors at <$500/unit**, and Latvia's border deployment, show this is a real, cheap, mass-deployable layer. It catches RF-silent (fiber-optic / pre-programmed autonomous) drones that RF detection misses. ([Drone-Warfare acoustic C-UAS](https://drone-warfare.com/counter-uas/acoustic-detection/); [Osinto acoustic drone detection](https://www.osinto.com/the-osborne-report/acoustic-drone-detection/))
- **Range:** ~300–500 m for small quadcopters; **degrades sharply in wind >5 m/s**; "quiet" drones (~15 dB at 1 km) are nearly inaudible. ([Osinto](https://www.osinto.com/the-osborne-report/acoustic-drone-detection/))
- **SWaP-C:** low (MEMS mics + edge ML, e.g., DroSonic-class). TRL 8–9, battle-tested.

**4b. Infrasound (<20 Hz).** Detects drones, helicopters, artillery, missile launches, and explosions at much longer ranges (1–20 Hz band; artillery/launch events 100s of km in arrays). A single infrasound channel plus the geophone gives a wide low-frequency aperture spanning ground and air. ([Infrasonic sensing array, Wikipedia](https://en.wikipedia.org/wiki/Infrasonic_sensing_array); [QuakeLogic SIS-1, 0.1–50 Hz, UAV detection km-scale](https://blog.quakelogic.net/uav-detection-using-infrasound-the-power-of-sis-1-infrasound-sensors/))
- **SWaP-C:** sensor itself is low-power/passive; needs wind-noise mitigation (porous hoses/rosettes) and arrays for bearing — that adds footprint. TRL 7–9 (mature in military arrays, less so as a single cheap node).

**Fusion value.** Both are passive (preserve covertness) and extend the node *upward* into the airspace the geophone cannot see. Infrasound also reinforces seismic on heavy events (blasts, large vehicles) via cross-modal confirmation.

---

## 5. Magnetometer — metal/vehicle/weapon confirmation, passive

**What it adds.** A fluxgate/MEMS magnetometer senses ferromagnetic mass: it confirms **vehicles, weapons, and armed personnel** and reports a magnetic disturbance magnitude (target-size proxy) plus crossing direction. This is a powerful complement because it answers a question seismic can't: *is the moving thing carrying metal?* — directly aiding armed-human vs animal and vehicle classification, with very low false-alarm rates. ([MAGID-II next-gen magnetic UGS, SPIE 8388](https://ui.adsabs.harvard.edu/abs/2012SPIE.8388E..0AW/abstract))

**Range.** Short and mass-dependent: a few meters for a handgun/armed person, tens of meters for a vehicle. Best as a close-in "gate-crossing" sensor along the buried line.

**SWaP-C.** Low power, passive, compact, low cost. TRL 9 (magnetic is one of the three classic UGS modalities; McQ RANGER and Exensor Flexnet both use it). ([McQ RANGER](https://www.mcqinc.com/products/mcq-ranger-unattended-ground-sensor/); [Exensor Flexnet](https://www.exensor.com/products/flexnet-flexible-network-of-sensors/))

**Tradeoff.** Detects only ferrous targets — a person in non-metallic clothing/shoes is invisible; geomagnetic and powerline noise need filtering; short range. High specificity, low sensitivity — use it to *confirm/upgrade* a seismic alarm to "armed/vehicle," not as a primary tripwire.

---

## 6. Higher-frequency geophones / MEMS accelerometers — same family, different band

**What it adds.** Adding a second seismic element of a different type, or going **3-axis**, buys (a) bearing via P-wave polarization / vector processing, and (b) a wider/different frequency band. A 3-axis geophone gives footstep DOA and tracking from a single emplacement — directly filling the localization gap without a separate modality. ([3-axis geophone bearing/localization](https://www.researchgate.net/publication/252385693_Single_and_three_axis_geophone_Footstep_detection_with_bearing_estimation_localization_and_tracking))

**MEMS accelerometer vs geophone — honest tradeoff.** MEMS are flat DC–5 kHz, tiny, digital, and cheap, but for *footstep* detection a **broad flat response is not an advantage**: a flat DC–5 kHz MEMS underperforms a band-matched commercial geophone, and low-noise MEMS that rival geophone noise floors are expensive. The coil geophone's resonance-shaped low-frequency response is well-matched to footstep energy. So MEMS is attractive for SWaP/integration but is a sideways move, not a strict upgrade, for the core footstep task. ([MEMS vs geophone for seismic monitoring](https://www.researchgate.net/publication/268585232_MEMS_accelerometer_vs_geophone_for_seismic_monitoring_and_survey); [Low-noise MEMS accelerometer for UGS, SPIE](https://www.researchgate.net/publication/252866012_A_low-noise_MEMS_accelerometer_for_unattended_ground_sensor_applications))

**Recommendation.** If you want bearing on the cheap and want to stay passive/buried, upgrading the single vertical 4.5 Hz to a **3-axis geophone** (or adding 1–2 spatially separated vertical elements for an array baseline) is the lowest-risk localization fix. TRL 9.

---

## 7. Fiber-optic Distributed Acoustic Sensing (DAS) — line, not node (architecture choice)

**What it adds.** DAS turns a buried telecom fiber into thousands of virtual seismic/acoustic channels via coherent Rayleigh backscatter. It detects footsteps, vehicles, digging, and tampering and **localizes along the cable**, covering an entire perimeter line rather than a point. It is essentially a *distributed geophone array* and is the natural scale-up of your single-node concept. ([AP Sensing DAS](https://www.apsensing.com/en/technology-and-products/distributed-acoustic-sensing); [RBtec DAS](https://www.rbtec.com/perimeter-sensors-and-underground-protection/fence-intrusion-detection-products/raysense-fiber-optic-perimeter-fence-security-system/what-is-distributed-acoustic-sensing-das/))

**Range.** Per-interrogator: tens of km (commonly 50–70 km, modern units to ~100 km with up to 50,000 ~2 m channels). Footsteps detected to ~10 m lateral offset from buried fiber. ([OptaSense QuantX](https://www.optasense.com/technology/quantx/); [DAS for linear infrastructure, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9572166/))

**SWaP-C.** This is the catch for a *node*: the **interrogator is a powered, mains-class box costing >$100k**; the fiber itself is cheap/passive/buried/EMI-immune. DAS is a *backbone/long-perimeter* solution, not a battery node. ([DOSITS DAS overview](https://dosits.org/distributed-acoustic-sensing-das-shedding-light-on-passive-acoustics/))

**TRL.** 9 for perimeter/border (AP Sensing, OptaSense fielded on borders and pipelines).

**Fusion value & strategic note.** DAS is a **product-tier complement, not a node sensor.** For the buried-node product, the relevant fusion is *node-DAS*: cheap nodes provide rich per-point classification and aerial/RF awareness, while a DAS spine provides continuous line coverage and localization between nodes. Worth positioning as the "scale" story for investors, but not part of the low-power node BOM.

---

## 8. EO/IR cameras — identity/intent, but power-hungry and not buried

**What it adds.** The only modality that delivers **identity and intent** (count, posture, weapon-in-hand, behavior) and, with thermal, robust human-vs-animal and day/night detection at long range. This is the natural *confirm-and-assess* layer once a buried sensor cues a direction. ([Infiniti EO/IR](https://www.infinitioptics.com/glossary/eo-ir-electro-optical-and-infrared-imaging); [Sirix thermal perimeter](https://sirixmonitoring.com/blog/thermal-security-cameras/))

**Range.** Uncooled LWIR human detection ~800–1,200 m (budget) to >1 km; cooled MWIR far longer. ([LightPath long-range thermal](https://www.lightpath.com/blog/long-range-thermal-surveillance-cameras-what-oems-need-to-know))

**SWaP-C.** The expensive, power-hungry end. Thermal cores + edge-AI compute draw watts continuously; thermal cameras are pricier than visible. Edge-AI cameras give on-device human/face/line-cross analytics with low latency, which helps. Cannot be buried — needs a mast/LoS, raising visibility. ([SightLogix edge AI](https://sightlogix.com/technology/ai-at-the-edge-detection/); [i-PRO edge AI](https://i-pro.com/products_and_solutions/en/surveillance/solutions/technologies/edge-ai-solutions))

**TRL.** 9. **Fusion value:** very high for assessment, but architecturally it belongs to a *cued/elevated* sub-node woken by the buried sensors — not the always-on buried element.

---

## 9. LiDAR — excellent 3D tracking, wrong form factor for a buried node

**What it adds.** 3D point-cloud volumetric detection with low false-alarm rates, immune to lighting, filters animals/vegetation/weather; one sensor covers ~15,000 m² / 360°. Great localization and object sizing. ([Quanergy 3D LiDAR PIDS](https://quanergy.com/3d-lidar-sets-new-benchmarks-in-perimeter-intrusion-detection/); [SDM Magazine](https://www.sdmmag.com/articles/103046-3d-lidar-changes-the-pids-landscape))

**SWaP-C.** ~10 W and PoE-class — far above a battery node; needs LoS mounting; eye-safe active laser. ([Ouster](https://ouster.com/insights/blog/powering-up-critical-infrastructure-security-with-lidar))

**TRL.** 8–9 for fixed-site PIDS. **Verdict:** a fixed-installation complement (substations, gates), **not** a low-power buried node sensor. Mentioned for completeness; deprioritize for the node BOM.

---

## 10. RF detection — passive C-UAS layer (system-level, not buried node)

**What it adds.** Passive RF sensing detects drone-to-controller links and IDs many COTS drones by protocol — long range, very low power, fully passive. Complements acoustic C-UAS (which catches RF-silent drones). ([Drone-Warfare RF detection](https://drone-warfare.com/counter-uas/rf-detection/); [Robin Radar 10 C-UAS technologies](https://www.robinradar.com/resources/10-counter-drone-technologies-to-detect-and-stop-drones-today))

**Tradeoff.** Blind to autonomous/fiber-controlled drones; antenna/spectrum coverage drives cost. Best as a *site-level* C-UAS sensor layered with node acoustics, not a per-node buried element. TRL 9.

---

## Comparative summary

| Modality | Primary gap filled | Detection range (human/relevant) | Power | Cost | Passive? | Buriable? | TRL | Node fit |
|---|---|---|---|---|---|---|---|---|
| **Geophone (baseline)** | footstep/vehicle ground motion | ~17–120 m footstep; ~100 m vehicle | µW–mW | $ | Yes | Yes | 9 | core |
| **Microphone / mic array** | classification (speech), bearing | voice 10s m; vehicle 100s m; gunshot km | mW (gate it) | ¢–$ | Yes | Near-surface | 9 | **top complement** |
| **PIR** | warm-body confirm, low-power wake | 5–20 m | µA standby | $ | Yes | Surface | 9 | **wake gate** |
| **mmWave radar** | range+bearing, micro-Doppler class | 30–80 m (to 350 m 4D) | mW–W | $–$$ | **No (emits)** | Surface/low | 8–9 | strong, gate it |
| **Acoustic drone array** | aerial threat | 300–500 m small UAS | mW | $ (<$500) | Yes | Surface | 8–9 | aerial layer |
| **Infrasound** | aerial/blast, long range | km–100s km (arrays) | low | $$ | Yes | Surface+hose | 7–9 | aerial/blast |
| **Magnetometer** | metal/vehicle/weapon confirm | few m (weapon) – 10s m (vehicle) | low | $ | Yes | Yes | 9 | confirm/upgrade |
| **3-axis geophone / MEMS** | bearing, band | as geophone | µW–mW | $ | Yes | Yes | 9 | localization fix |
| **DAS (fiber)** | line coverage + localization | 10 m offset; 50–100 km/interrogator | mains box | **>$100k interrogator** | fiber yes | Yes | 9 | product spine, not node |
| **EO/IR camera** | identity/intent | 0.8–>1 km (thermal) | W (continuous) | $$–$$$ | Yes (passive IR) | No | 9 | cued sub-node |
| **LiDAR** | 3D track, low FAR | ~70–100 m radius | ~10 W | $$ | No (laser) | No | 8–9 | fixed-site only |
| **RF detection** | drone link detect/ID | km-scale | low | $$ | Yes | Surface | 9 | site C-UAS layer |

(Cost tiers are order-of-magnitude: ¢=cents, $=tens, $$=thousands, $$$=tens of k+.)

---

## Recommended low-power complementary stack for a buried perimeter node

**Design constraints:** battery/solar, mostly passive (preserve covertness — a core geophone virtue), buriable or low-profile, edge-classifiable. The goal is to kill the five gaps with the least power and the least loss of passivity.

**Tier 1 — always-on, micro-power core (the "is something there?" layer)**
1. **Vertical 4.5 Hz geophone (keep).** Primary footstep/vehicle event detector. Passive, µW.
2. **Upgrade to 3-axis geophone** (or add a 2nd spaced vertical element). Buys **bearing/tracking** from a single buried emplacement — the cheapest fix for the localization gap, fully passive, no new modality. ([3-axis geophone bearing](https://www.researchgate.net/publication/252385693_Single_and_three_axis_geophone_Footstep_detection_with_bearing_estimation_localization_and_tracking))
3. **PIR module** as the µA-class **wake gate** and warm-body confirm. Near-site-independent classification lift, negligible power. ([ARL fusion, DTIC](https://apps.dtic.mil/sti/pdfs/ADA565006.pdf))

**Tier 2 — duty-cycled, woken by Tier 1 (the "what is it?" layer)**
4. **Small MEMS microphone array (3–4 elements).** The single highest-value classification complement: speech/engine/gunshot spectral cues that crack human-vs-animal, plus **acoustic bearing**. Run only when seismic/PIR trips, to bound power. This pairing is exactly what fielded UGS (Exensor Mini Mk3, McQ RANGER) and the ARL fusion literature validate. ([Damarla ARL](https://apps.dtic.mil/sti/tr/pdf/ADA564999.pdf); [Multimodal fusion, DTIC](https://apps.dtic.mil/sti/pdfs/ADA565006.pdf))
5. **Magnetometer (fluxgate/MEMS).** Passive, low-power confirm that upgrades a seismic+acoustic alarm to **"armed person / vehicle"** when ferrous mass is present — high-specificity intent cue. ([MAGID-II](https://ui.adsabs.harvard.edu/abs/2012SPIE.8388E..0AW/abstract))

**Tier 3 — aerial layer (defense/C-UAS differentiator, the "is it above ground?" layer)**
6. **Passive acoustic drone-detection channel** reusing the Tier-2 mic array with a UAS-tuned classifier (propeller tonals). Cheap, passive, combat-proven at scale, and it makes the node relevant to the drone threat the geophone is blind to — strategically aligned with the Elbit/defense angle. ([Drone-Warfare acoustic](https://drone-warfare.com/counter-uas/acoustic-detection/))
   - *Optional add for higher-end nodes:* a single **infrasound channel** for long-range blast/aircraft/large-vehicle confirmation. ([QuakeLogic SIS-1](https://blog.quakelogic.net/uav-detection-using-infrasound-the-power-of-sis-1-infrasound-sensors/))

**Deliberately excluded from the buried node** (kept at system tier):
- **mmWave radar** — superb micro-Doppler classification and bearing, but it *emits* (surrenders covertness, jammable) and isn't buriable. Offer it as a **higher-power surface node variant** for sites where active sensing is acceptable. ([Linpowave mmWave](https://linpowave.com/blog/millimeter-wave-radar-intrusion-detection))
- **EO/IR camera** — reserve for a **cued, elevated assessment sub-node** that the buried node wakes for identity/intent; too power-hungry and non-buriable for the core. ([SightLogix](https://sightlogix.com/technology/ai-at-the-edge-detection/))
- **LiDAR / RF detection / DAS** — **site/architecture-tier**, not node-tier. Position DAS as the *perimeter spine* and RF as the *site C-UAS layer* in the product story, not in the node BOM. ([OptaSense QuantX](https://www.optasense.com/technology/quantx/))

**Rationale in one line.** Keep the geophone passive core, fix bearing for free with 3-axis seismic, gate everything behind µA PIR, and spend the duty-cycled power budget on a MEMS mic array — the single modality that simultaneously cracks human-vs-animal, adds acoustic bearing, and (re-tuned) covers drones — with a passive magnetometer as the cheap "armed/vehicle" intent confirm. This closes all five geophone blind spots while keeping the node mostly passive, buriable, and battery-viable.

---

## Sources
- [Range limitation for seismic footstep detection (SPIE 6963)](https://www.researchgate.net/publication/252566889_Range_limitation_for_seismic_footstep_detection_-_art_no_69630V)
- [Intelligent Recognition of Acoustic and Vibration Threats (HSAJ)](https://www.hsaj.org/articles/72)
- [Exemplar Selection to Distinguish Human from Animal Footsteps](https://posenhuang.github.io/papers/Exemplar_Selection_HLVD2011.pdf)
- [Single and three-axis geophone: footstep detection, bearing, localization](https://www.researchgate.net/publication/252385693_Single_and_three_axis_geophone_Footstep_detection_with_bearing_estimation_localization_and_tracking)
- [Multimodal Sensor Fusion for Personnel Detection (DTIC ADA565006)](https://apps.dtic.mil/sti/pdfs/ADA565006.pdf)
- [Damarla, ARL — Detection of people and animals using non-imaging sensors (DTIC ADA564999)](https://apps.dtic.mil/sti/tr/pdf/ADA564999.pdf)
- [AFPILD acoustic footstep dataset (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1566253523004979)
- [MEMS accelerometer vs geophone for seismic monitoring](https://www.researchgate.net/publication/268585232_MEMS_accelerometer_vs_geophone_for_seismic_monitoring_and_survey)
- [Low-noise MEMS accelerometer for UGS (SPIE)](https://www.researchgate.net/publication/252866012_A_low-noise_MEMS_accelerometer_for_unattended_ground_sensor_applications)
- [PIR sensor guide (Roombanker)](https://www.roombanker.com/blog/what-is-pir-sensor/)
- [Off-grid PIR cameras (Milesight)](https://www.milesight.com/security/blog/edge-ai-pir-camera)
- [mmWave intrusion detection (Linpowave)](https://linpowave.com/blog/millimeter-wave-radar-intrusion-detection)
- [Consumer mmWave radar range guide](https://linpowave.com/blog/consumer-mmwave-radar-range-guide)
- [Micro-Doppler human/vehicle classification](https://edwin-pan.github.io/project/cs598/)
- [TI industrial mmWave radar](https://www.ti.com/product-category/sensors/mmwave-radar/industrial/overview.html)
- [Infineon 60 GHz radar](https://www.infineon.com/products/sensor/radar-sensors/radar-sensors-for-iot/60ghz-radar)
- [mmWave radar pricing (LintechTT)](https://www.lintechtt.com/mmwave-radar-sensor-pricing-guide/)
- [Acoustic drone detection C-UAS (Drone-Warfare)](https://drone-warfare.com/counter-uas/acoustic-detection/)
- [Acoustic drone detection (Osinto)](https://www.osinto.com/the-osborne-report/acoustic-drone-detection/)
- [RF drone detection (Drone-Warfare)](https://drone-warfare.com/counter-uas/rf-detection/)
- [10 counter-drone technologies (Robin Radar)](https://www.robinradar.com/resources/10-counter-drone-technologies-to-detect-and-stop-drones-today)
- [Infrasonic sensing array (Wikipedia)](https://en.wikipedia.org/wiki/Infrasonic_sensing_array)
- [QuakeLogic SIS-1 infrasound / UAV detection](https://blog.quakelogic.net/uav-detection-using-infrasound-the-power-of-sis-1-infrasound-sensors/)
- [MAGID-II next-gen magnetic UGS (SPIE 8388)](https://ui.adsabs.harvard.edu/abs/2012SPIE.8388E..0AW/abstract)
- [McQ RANGER UGS](https://www.mcqinc.com/products/mcq-ranger-unattended-ground-sensor/)
- [Exensor Flexnet UGS](https://www.exensor.com/products/flexnet-flexible-network-of-sensors/)
- [Exensor Mini Mk3 seismic+acoustic sensor](https://www.exensor.com/products/flexnet-flexible-network-of-sensors/unattended-ground-sensors-ugs/mini-mk3/)
- [Exensor PIR module](https://www.exensor.com/products/flexnet-flexible-network-of-sensors/unattended-ground-sensors-ugs/pir/)
- [SensoGuard UGS](https://sensoguard.com/ugs-system-unattended-ground-sensors/)
- [RBtec Seismo UGS](https://www.rbtec.com/perimeter-sensors-and-underground-protection/buried-sensors/unattended-ground-sensor-security-system-ugs/)
- [AP Sensing DAS](https://www.apsensing.com/en/technology-and-products/distributed-acoustic-sensing)
- [RBtec DAS](https://www.rbtec.com/perimeter-sensors-and-underground-protection/fence-intrusion-detection-products/raysense-fiber-optic-perimeter-fence-security-system/what-is-distributed-acoustic-sensing-das/)
- [OptaSense QuantX interrogator](https://www.optasense.com/technology/quantx/)
- [DAS for linear infrastructure (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9572166/)
- [DOSITS DAS overview](https://dosits.org/distributed-acoustic-sensing-das-shedding-light-on-passive-acoustics/)
- [3D LiDAR PIDS (Quanergy)](https://quanergy.com/3d-lidar-sets-new-benchmarks-in-perimeter-intrusion-detection/)
- [3D LiDAR PIDS landscape (SDM)](https://www.sdmmag.com/articles/103046-3d-lidar-changes-the-pids-landscape)
- [LiDAR critical infrastructure (Ouster)](https://ouster.com/insights/blog/powering-up-critical-infrastructure-security-with-lidar)
- [EO/IR imaging (Infiniti)](https://www.infinitioptics.com/glossary/eo-ir-electro-optical-and-infrared-imaging)
- [Long-range thermal cameras (LightPath)](https://www.lightpath.com/blog/long-range-thermal-surveillance-cameras-what-oems-need-to-know)
- [Thermal perimeter security (Sirix)](https://sirixmonitoring.com/blog/thermal-security-cameras/)
- [Edge-AI detection (SightLogix)](https://sightlogix.com/technology/ai-at-the-edge-detection/)
- [i-PRO edge AI solutions](https://i-pro.com/products_and_solutions/en/surveillance/solutions/technologies/edge-ai-solutions)
