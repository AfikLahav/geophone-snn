# MEMS Inertial Sensors for Detecting Footsteps, Vehicles, and Ground/Structural Vibration

*A technical assessment for a low-cost multi-sensor unattended-ground-sensor (UGS) node built around a buried 4.5 Hz geophone.*

---

## Abstract

Micro-electromechanical-systems (MEMS) inertial sensors — the accelerometers, gyroscopes, and integrated IMUs mass-produced for smartphones and automotive use — are attractive for unattended ground sensing because they are millimetre-scale, cost a few dollars to a few tens of dollars, draw sub-milliwatt to a few-milliwatt power, and provide a flat acceleration response from DC to hundreds of hertz across three axes. Historically they could not rival a 4.5 Hz geophone for weak-signal seismic work because their self-noise below ~10 Hz was orders of magnitude too high (early consumer parts ~hundreds of µg/√Hz to mg/√Hz, versus an effective geophone acceleration noise floor of a few ng/√Hz near resonance). Modern low-noise MEMS parts (e.g. the Analog Devices ADXL355 at 25 µg/√Hz, the ADXL354/356 family, and research-grade resonant/optomechanical devices reaching ng/√Hz) have narrowed but not eliminated this gap. Large MEMS/smartphone seismic networks — MyShake, the Quake-Catcher Network, the Community Seismic Network, and the geophone+MEMS Raspberry Shake — demonstrate that MEMS can detect M≈2.5–5 earthquakes and strong ground motion at densities and costs (<1% of a traditional network) impossible with research instruments, while published self-noise comparisons confirm they remain noisier than a geophone in the ~3–50 Hz band most relevant to footsteps and vehicles. For a UGS node the practical conclusion is that a 3-axis MEMS accelerometer is best used as a *complement* to the geophone — extending usable bandwidth above ~50–100 Hz, adding DC/tilt and 3-axis bearing information, and providing a strong-motion (un-clipped) channel — rather than as a wholesale substitute below 10 Hz, with ML classifiers (SVM, CNN, CNN-LSTM) reported at >95–99% accuracy when fed multi-axis/spectral features.

---

## 1. MEMS accelerometer physics and the specs that matter

### 1.1 How a MEMS capacitive accelerometer works

A typical MEMS accelerometer is a silicon proof mass suspended by micromachined springs between fixed electrodes, forming a differential capacitor. External acceleration displaces the proof mass; the resulting capacitance change is read out and conditioned. Unlike a geophone, the device responds to acceleration directly and is **DC-coupled** — it measures constant (gravity/tilt) and dynamic acceleration with a flat response from 0 Hz up to its bandwidth limit. Resonant ("vibrating-beam"/quartz) MEMS instead measure the shift in a beam's resonant frequency under inertial load, trading a more complex readout for very high resolution.

### 1.2 The specs that matter for seismic/vibration sensing

| Spec | Why it matters | What "good" looks like for ground/structural vibration |
|---|---|---|
| **Noise density / noise floor (µg/√Hz)** | Sets the smallest detectable signal; the single most important seismic spec | Tens of µg/√Hz (consumer low-noise) down to ng/√Hz (research grade) |
| **Bandwidth** | Footsteps concentrate energy ~20–90 Hz; vehicles ~1–50 Hz; structural modes ~0.1–50 Hz | DC–several hundred Hz flat |
| **Dynamic range** | Must span faint footsteps to near-field heavy vehicles/blasts without clipping | ≥90 dB |
| **Full-scale range** | Determines clip level; small range → better resolution but earlier saturation | ±2 g (resolution) to ±100 g (strong motion/impact) |
| **Sensitivity & offset drift** | Affects DC/tilt accuracy and long-term stability | Low 0 g offset temperature drift for DC use |
| **Power** | Battery-life-limited UGS node | Sub-mW to a few mW |

### 1.3 Representative low-noise MEMS parts

The **Analog Devices ADXL355** is the canonical low-noise, low-power seismic-class consumer MEMS accelerometer:

- Noise density **25 µg/√Hz**, selectable ranges **±2.048 / ±4.096 / ±8.192 g**, resolution **3.9 µg/LSB** (20-bit), nonlinearity 0.1%, **dynamic range > 90 dB**, current **< 200 µA** (≈0.5 mW at typical supply). It is explicitly marketed for IoT/seismic nodes; list price is roughly **US$25–65** depending on package/quantity (PMOD/eval boards ~US$50–65). [4][14][15]
- The related **ADXL354/356** family targets the same low-noise/low-drift/low-power niche with wider ranges (the ADXL356 reaching ±40 g) for combined precision + strong-motion use. [13]

Higher-bandwidth single-axis parts such as the **ADXL1002/1004** trade noise for bandwidth and full scale (e.g. the ADXL1004 to ±500 g, ~5 mW), useful for high-frequency machine/impact vibration rather than weak seismic detection. [4][9]

Research-grade MEMS push far lower: a **quartz resonant accelerometer** reported **14 µHz/√Hz** frequency resolution and 32 µHz bias instability; a low-noise DC seismic MEMS achieved **< 100 nG/√Hz** flat over DC–200 Hz; and an **optomechanical MEMS geophone** reached **2.5 ng/√Hz** over 100–200 Hz — approaching the best geophones — though these are not yet commodity parts. [1][2][7][8]

### 1.4 Head-to-head: low-noise MEMS vs a 4.5 Hz geophone

**The geophone.** A 4.5 Hz geophone is a coil-and-magnet velocity transducer with a mechanical resonance ("corner") at 4.5 Hz. Above resonance it has a flat *velocity* response; below resonance the velocity sensitivity rolls off at 12 dB/octave, so referred to *acceleration* the self-noise rises steeply toward low frequency. Its fundamental noise limits are the **Johnson (thermal) noise of the coil resistance** and the **mechanical Brownian noise** of the suspended mass, plus the front-end amplifier voltage noise (the amplifier and coil Johnson noise are typically comparable and dominate over most of the band). [LIGO/L-4C note; 19][20] With good electronics the *effective* in-band acceleration noise floor can be extremely low — e.g. a digital 4.5 Hz geophone with a sigma-delta modulator and velocity feedback reached **4.3 ng/√Hz at 1.0 Hz** over a 200 Hz band, and extended usable response down to 0.16 Hz. [11][12]

**Where MEMS lose.**
- **Low-frequency self-noise (the historical gap).** Below ~10 Hz a good geophone is far quieter in acceleration terms (ng/√Hz class) than commodity MEMS (tens of µg/√Hz for the ADXL355; hundreds of µg/√Hz to mg/√Hz for phone-grade parts). A widely cited rule of thumb from field comparisons is that a 10 Hz geophone is **less noisy than a digital MEMS accelerometer between roughly 3 and 50 Hz**, and noisier only outside that band. [search/CREWES] That 3–50 Hz window is exactly where footstep and vehicle energy concentrates, so the geophone retains a weak-signal advantage there. The physical reason MEMS historically lost below ~10 Hz is that a small proof mass + capacitive readout has high electronic/Brownian noise relative to the tiny accelerations involved, and 1/f electronic noise dominates the low-frequency end.

**Where MEMS win.**
- **High-frequency / wide flat band.** MEMS are flat in acceleration from **DC to hundreds of Hz (often quoted DC–500 Hz, some parts to ~kHz)**, with no velocity roll-off and no resonant peak to deconvolve. Footstep harmonics extend to ~30–90 Hz and the maximum footstep *range* often occurs in the 30–40 Hz band — region where a 4.5 Hz geophone's response is flat in velocity but a MEMS gives clean, calibrated acceleration. [search/MEMS DC–5 kHz; 16]
- **DC response and tilt/bearing.** Because MEMS are DC-coupled they give absolute **tilt/orientation** for free, and 3-axis parts give horizontal components for **bearing/azimuth** estimation — a geophone element gives one axis of velocity only.
- **Sub-resonance response.** Below the geophone's 4.5 Hz corner, the MEMS' flat acceleration response can actually have *better relative response*: field tests show a MEMS sensor ~2 dB stronger than a geophone sensor at 1.5 Hz, widening to ~8 dB at 1 Hz. [3][search/INZWA]
- **Cost, size, power, robustness.** A MEMS die is ~10 mm²/millimetre-scale and a few dollars; a geophone is a centimetre-scale electromechanical can that must be installed within ~2° of its design orientation and has moving parts. MEMS draw sub-mW–few-mW and have no orientation constraint.

**Net.** Modern low-noise MEMS (tens of µg/√Hz) have *closed the gap enough to be useful seismically*, and research MEMS (ng/√Hz) match geophones in narrow bands — but for a buried weak-signal sensor in the 3–50 Hz footstep/vehicle band, a 4.5 Hz geophone with good electronics still has the lower acceleration noise floor. MEMS' decisive advantages are bandwidth above ~50 Hz, DC/tilt, 3-axis bearing, cost, size, and power.

---

## 2. MEMS as low-cost seismic/vibration sensors in practice

### 2.1 Footstep, vehicle, and perimeter/intrusion detection

Miniature seismometer/accelerometer networks are an established approach to **perimeter protection, border security, and intruder detection**: they sense surface waves from footsteps and vehicles to detect and classify intruders in a guarded area. [16][17][18] Key practical results:

- A **vibrating-beam (quartz MEMS) seismometer node** designed for UGS intruder detection achieved **100 µg @ 500 Hz resolution, 3.5 mW power, in a ~10 mm² MEMS** — explicitly chosen because prior sensors had either limited range *or* limited battery autonomy. Its DC-coupled MEMS output also supplied **tilt orientation** for the buried nodes, and the wide flat response (cited DC–5 kHz) improved detection/classification. [16]
- Field tests of seismic UGS report representative ranges: for a ~215 lb walker in a desert (Yuma) test, a **90% probability of ≥2 "walker" detections at 140 m sensor spacing**. [16/search]
- Footstep characterization studies report **mean detection ranges around 60–64 m for normal walking**, with the **30–40 Hz band giving the maximum range** across regular/soft/stealthy gaits; *stealthy* walking can be undetectable beyond a few metres. Strong footstep signals showed **>11 dB SNR (primary) and >7 dB (harmonics) at 40 m**. Detection range is strongly governed by the local background-noise floor (quiet rural vs. urban). [17][search/range-limitation]

### 2.2 Large MEMS / smartphone seismic networks

These projects validate MEMS at scale and quantify their seismic performance:

**MyShake** (UC Berkeley). A global smartphone seismic network using the phone's built-in MEMS accelerometer. Reported characteristics: **50 sps** sampling, useful **0.5–10 Hz** band; **all phones sensitive to M5 at ≤10 km** in 1–10 Hz, with newer phones reaching **M3.5 sensitivity at 10 Hz** (sensor quality improves with phone release year). An on-device artificial neural network **correctly identified 98% of earthquake records (within 10 km) and rejected 93% of everyday motion (7% false-alarm)**. A key limitation: a phone resting freely **slides at horizontal accelerations ≳0.3 g above 3 Hz**, clipping recorded amplitude. MyShake reported smartphone noise floors around **−75 dB** average (variable −50 to −80 dB at low frequency, −78 to −85 dB at high frequency), and noted **high-quality MEMS can approach traditional force-balance strong-motion sensors**. [5][6][search/MyShake]

**Raspberry Shake (RS4D)** (raspberryshake.org). A hybrid that is directly relevant to a geophone+MEMS UGS node: it combines **one vertical 4.5 Hz geophone (weak-motion)** with a **triaxial ±2 g MEMS accelerometer (strong-motion)**, digitiser, and Raspberry Pi in one box. The geophone catches micro-tremors; the MEMS prevents saturation on strong motion. The global Raspberry Shake fleet is a crowdsourced network of **>1000 nodes**. This is essentially the commercial validation of the "geophone for sensitivity + MEMS for range and extra axes" architecture. [search/RS4D]

**Community Seismic Network (CSN)** (Caltech). A low-cost, cloud-based **strong-motion** network: a **3-axis MEMS accelerometer + Raspberry Pi**, sampled at **50 sps**, transmitting waveforms to the cloud, deployed by the hundreds in LA-region buildings (5–23 storeys) for shaking maps and structural health monitoring. [search/CSN]

**Quake-Catcher Network (QCN)** (Stanford/USGS). Used volunteers' laptop-internal or USB MEMS accelerometers; the data-gathering scheme cost **<1% of a traditional seismic network**, enabling very high station density. Reliability studies addressed its higher per-station noise and false detections. [search/QCN]

**Independent benchmarking.** Evans et al. (2014, *SRL*), "Performance of Several Low-Cost Accelerometers," classified these "Class C" MEMS sensors (~US$100–200, ~12–16-bit useful resolution over ±2 g ranges) and quantified their self-noise/clip behaviour against research-grade ("Class A") force-balance instruments and the USGS/Peterson noise models — establishing that consumer MEMS sit well above the New Low Noise Model but are adequate for strong-motion and near-field weak-motion at low cost. The Peterson **NLNM/NHNM** (units dB rel (1 m/s²)²/Hz) are the standard ambient-noise references; weak-signal seismic monitoring generally requires an instrument self-noise well **below ~−130 dB**, which commodity MEMS do not reach at low frequency but research MEMS and geophones can. [Evans 2014; search/Peterson]

---

## 3. Machine learning on MEMS/IMU signals for event classification

Classifying **footstep vs. vehicle vs. animal vs. noise** from inertial signals is a mature ML problem; reported approaches and accuracies:

- **Classical ML.** SVM classifiers on seismic footstep features are well established; cadence-frequency-only models cause high false-alarm rates, so spectral/statistical features are added. Wavelet-packet "manifold" features have been used for seismic target classification in UGS systems. [search/UGS-SVM; PMC3758609]
- **Deep learning (the current state of the art).**
  - A **CNN** separating **earthquake / vehicle / other-noise** from 3-channel, 10 s seismic traces (+ their spectra) achieved **>99% accuracy for all classes**. [search/Springer 2024]
  - A CNN using **log-scaled frequency cepstral coefficients (LFCC)** for **vehicle classification** from seismic data reported strong results in a 2025 *Scientific Reports* study. [search/Nature SR 2025]
  - A **CNN for footstep detection in urban seismic data** demonstrated robust detection against urban background noise. [search/footstep CNN]
  - On smartphone/wearable **IMU (accel + gyro) human-activity recognition**, hybrid **CNN-LSTM / LSTM** models report **~97–99% accuracy** distinguishing walking, running, stairs, etc. — directly transferable to discriminating gait-like (human) vs. continuous (vehicle) signatures on a node. [search/HAR CNN-LSTM]
- **Why MEMS suits ML here.** The flat DC–hundreds-of-Hz, **3-axis** acceleration (optionally + gyro) gives richer, calibrated, multi-channel input than a single-axis geophone — more features (per-axis spectra, cross-axis ratios for bearing, DC tilt context) for the classifier, which is one reason MEMS+ML is so widely used. The dominant practical problem is **false alarms** from environmental noise, road-surface and coupling variability, and limited on-board compute. [search/UGS challenges]

---

## 4. Detection ranges and limitations vs the ambient ground-noise floor

Detection range for any inertial seismic sensor is set by **signal vs. local ground noise**, not by the sensor alone — but a noisier sensor (MEMS at low frequency) raises the effective floor and shrinks range when ground noise is low.

- **Footsteps.** Normal walking is detectable to ~**60–64 m mean range** in favourable (quiet) conditions; the **30–40 Hz band** maximises range; signal drops sharply with distance (ground acts as a low-pass filter, pushing the usable peak toward ~10–16 Hz as range grows from 6→60 m). Stealthy gait may be undetectable beyond a few metres. [17]
- **Vehicles.** Lower-frequency (~1–50 Hz), higher-energy, continuous signatures detectable substantially farther than footsteps (often hundreds of metres for heavy vehicles in quiet ground), and easier to classify due to their sustained spectral character.
- **The noise-floor ceiling.** The **Peterson NLNM/NHNM** bound real-world ground noise; a sensor only helps to the extent its self-noise is below ambient. In **quiet rural ground**, ambient noise is low, so a low-self-noise sensor (geophone) extends range; in **urban ground**, cultural noise dominates and even a geophone is noise-limited, so the MEMS' higher self-noise matters less. Weak-signal work wants instrument self-noise **≲ −130 dB rel (m/s²)²/Hz**; commodity MEMS exceed (are worse than) this at low frequency, which is why MEMS networks target **strong motion** and **near-field** events, and why a buried geophone is preferred for **faint, distant footsteps in the 3–50 Hz band**. [search/Peterson; CREWES]
- **Coupling and decoupling.** Burial/coupling quality dominates real performance. MyShake's **0.3 g / >3 Hz sliding** clip is the consumer analogue of poor coupling; a properly coupled/buried MEMS avoids this but still cannot beat physics on self-noise below ~10 Hz.

---

## 5. Relevance to a cheap multi-sensor UGS node

### 5.1 Complement, not (sub-10 Hz) substitute

The evidence points to a **fused geophone + 3-axis MEMS** node, exactly the Raspberry Shake RS4D architecture, with a clear division of labour:

| Role | Geophone (4.5 Hz) | 3-axis MEMS accelerometer |
|---|---|---|
| **Weak/faint signals 3–50 Hz** | **Best** (lowest acceleration self-noise) | Worse below ~10 Hz |
| **Bandwidth > ~50 Hz** | Flat (velocity) but 1-axis | **Best** (flat accel, calibrated, multi-axis) |
| **Strong/near-field motion** | Can clip / over-range | **Best** (selectable ±2…±40 g, won't saturate) |
| **DC / tilt / install orientation** | None (AC-coupled velocity) | **Yes** (DC-coupled) |
| **Bearing / azimuth** | No (single vertical axis) | **Yes** (3-axis horizontal components) |
| **Cost / size / power** | Larger, orientation-sensitive can | **Best** (mm-scale, few $, sub-mW–few-mW) |

So the MEMS **complements** the geophone by (a) adding the **>50 Hz band** where footstep harmonics live, (b) providing an **un-clippable strong-motion channel**, (c) giving **DC tilt** for self-orientation/health, and (d) supplying **3-axis data for bearing** and richer ML features. It is a credible **substitute only above ~10 Hz / for strong motion / where cost-size-power dominate**, not for the faintest sub-10 Hz signals.

### 5.2 Integration and power

- **Part choice.** A low-noise consumer part like the **ADXL355 (25 µg/√Hz, ±2/4/8 g, >90 dB DR, <200 µA, DC–~1 kHz, SPI/I²C, ~US$25–65)** is the natural complement; the **ADXL356** adds a ±40 g strong-motion range. For pure high-frequency machine/impact vibration, an **ADXL1002/1004** single-axis part. [4][13][14][15]
- **Interfacing.** SPI/I²C straight into the same MCU/SBC that reads the geophone ADC; sample both at ≥200 sps to cover footstep harmonics. Raspberry Shake/CSN show a Raspberry-Pi-class SBC handles geophone + 3-axis MEMS + ML inference. [search/RS4D, CSN]
- **Power budget.** The MEMS adds only **~0.5 mW (ADXL355) to ~3.5 mW (resonant node)** — negligible beside radio/compute — so it does **not** materially change the node's battery life; duty-cycling and edge-classification (only transmit on detection) dominate the energy budget, consistent with QCN/CSN designs. [16][4]
- **ML on-node.** A small CNN/CNN-LSTM (or the existing SNN classifier) can ingest fused geophone + 3-axis MEMS channels; the extra axes and the >50 Hz band measurably help footstep-vs-vehicle separation and bearing, at the cost of more input channels. [§3]

### 5.3 Bottom line for the project

Adding a **low-noise 3-axis MEMS accelerometer (ADXL355-class)** alongside the buried 4.5 Hz geophone is **high-value, low-cost, low-power**: it widens usable bandwidth, prevents strong-motion clipping, adds DC/tilt and 3-axis bearing, and enriches the ML feature set — while the geophone remains the primary weak-signal sensor in the 3–50 Hz band. Replacing the geophone outright with a commodity MEMS would sacrifice faint, distant sub-10 Hz detection range; only research-grade ng/√Hz MEMS could do that, and they are not yet commodity priced.

---

## References

1. Hines, A. et al. "An optomechanical MEMS geophone with a 2.5 ng/Hz¹ᐟ² noise floor for oil/gas exploration." *Microsystems & Nanoengineering* (2024). https://www.nature.com/articles/s41378-024-00802-5 — also https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11589752/
2. "An Ultra-low noise MEMS accelerometer for Seismology / seismic imaging" (prototype ~2 µg/√Hz at 100 Hz). https://www.researchgate.net/publication/324601654_An_Ultra-low_noise_MEMS_accelerometer_for_Seismology and https://www.researchgate.net/publication/261153656
3. INZWA, "MEMS vs. Geophones — vibration-monitoring test results" (geophone ~2 dB lower than MEMS at 1.5 Hz, ~8 dB at 1 Hz). https://www.inzwa.io/mems-vs-geophone-vibration-monitoring-test-results/
4. Analog Devices, ADXL355 product page (25 µg/√Hz, ±2.048/4.096/8.192 g, >90 dB DR, <200 µA). https://www.analog.com/en/products/adxl355.html
5. Kong, Q., Allen, R. M., et al. "MyShake: A smartphone seismic network for earthquake early warning and beyond." *Science Advances* (2016). https://www.science.org/doi/10.1126/sciadv.1501055 — open access https://pmc.ncbi.nlm.nih.gov/articles/PMC4758737/
6. Kong, Q. et al. "Toward Global Earthquake Early Warning with the MyShake Smartphone Seismic Network." *SRL* (2020). https://rallen.berkeley.edu/pub/2020KongMartinShortAllen1/KongEtAl-MyShakeGlobalAlerting-Pt1-SRL-2020preprint.pdf
7. "A Low-Noise DC Seismic Accelerometer Based on MET/MEMS" (< 100 nG/√Hz, DC–200 Hz). https://www.researchgate.net/publication/270290694
8. "A 14 µHz/√Hz resolution and 32 µHz bias-instability MEMS quartz resonant accelerometer." *PMC* (2024). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11666743/
9. Analog Devices, ADXL354/ADXL355 and ADXL1002/1004 datasheets. https://www.mouser.com/datasheet/2/609/ADXL354_355-1026032.pdf
10. Evans, J. R., Allen, R. M., Chung, A. I., Cochran, E. S., Guy, R., Hellweg, M., Lawrence, J. F. "Performance of Several Low-Cost Accelerometers." *Seismological Research Letters* 85(1):147–158 (2014). https://pubs.usgs.gov/publication/70115921 — preprint https://rallen.berkeley.edu/pub/2014evans/EvansEtAl-ClassCSensors-SRL-2014.pdf
11. "A digital low-frequency geophone based on 4th-order sigma-delta modulator and single-coil velocity feedback" (4.3 ng/√Hz @ 1 Hz; response to 0.16 Hz). *Sensors & Actuators A* (2020). https://www.sciencedirect.com/science/article/abs/pii/S0924424719320345
12. "An Effective Method for Improving Low-Frequency Response of Geophone." *Sensors / MDPI* (2023). https://www.mdpi.com/1424-8220/23/6/3082 — https://ncbi.nlm.nih.gov/pmc/articles/PMC10059280
13. Analog Devices, ADXL356 product page (±40 g range; combined precision + strong motion). https://www.analog.com/en/products/adxl356.html
14. ADXL355 PMOD datasheet (sensitivity 256000 LSB/g; 3.9 µg/LSB; noise density 25 µg/√Hz). https://www.farnell.com/datasheets/2310319.pdf
15. Seeed Studio, "ADXL356 / ADXL345 vs ADXL335" comparison and pricing/power context. https://www.seeedstudio.com/blog/2019/11/26/adxl356-get-started-adxl345-and-adxl335-comparison-guide/
16. Lévy, C., Moras, J., et al. "Vibrating Beam MEMS Seismometer for Footstep and Vehicle Detection." *IEEE Sensors Journal* (2017) (100 µg @ 500 Hz, 3.5 mW, ~10 mm², DC tilt, 140 m @ 90% in Yuma test). https://ieeexplore.ieee.org/document/7990504/ and https://ieeexplore.ieee.org/document/7808973
17. Pakhomov, A., Goldburt, T. "Range limitation for seismic footstep detection." *Proc. SPIE* 6963, 69630V (mean range ~60–64 m; 30–40 Hz max range; >11/7 dB SNR at 40 m). https://www.researchgate.net/publication/252566889
18. "Footstep and Vehicle Detection Using Seismic Sensors in Wireless Sensor Network: Field Tests." https://www.researchgate.net/publication/258389659
19. LIGO/AEI, "Sensor noise of L-22D and L-4C Geophones" technical note T1600438 (geophone self-noise; Johnson + Brownian limits). https://dcc.ligo.org/public/0138/T1600438/001/L-4C%20huddle%20test%20at%20the%20AEI.pdf
20. CREWES, "Comparison of geophones and accelerometers" (10 Hz geophone quieter than digital MEMS between ~3 and 50 Hz). https://www.crewes.org/Documents/GraduateTheses/2008/Hons-MSc-2008.pdf
21. Peterson, J. "Observations and modeling of seismic background noise" — USGS NLNM/NHNM (units dB rel (1 m/s²)²/Hz; weak-signal needs self-noise ≲ −130 dB). USGS Open-File Report 93-322. (See also https://nhess.copernicus.org/articles/23/3219/2023/ for applied background-noise levels.)
22. "Effective deep learning aided vehicle classification approach using Seismic Data." *Scientific Reports* (2025). https://www.nature.com/articles/s41598-025-01684-x
23. "Deep learning based earthquake and vehicle detection algorithm" (CNN; >99% on earthquake/vehicle/noise, 10 s 3-channel). *Journal of Seismology* (2024). https://link.springer.com/article/10.1007/s10950-024-10267-8
24. "Footstep detection in urban seismic data with a convolutional neural network." https://www.researchgate.net/publication/344056607
25. "Seismic Target Classification Using a Wavelet Packet Manifold in Unattended Ground Sensors Systems." *Sensors* (PMC). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3758609/
26. "Deep CNN-LSTM With Self-Attention Model for Human Activity Recognition Using Wearable Sensor" (IMU accel+gyro; ~97–99% accuracy). *PMC* (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9252338/
27. Raspberry Shake RS4D product/specifications (1× vertical 4.5 Hz geophone + triaxial ±2 g MEMS; >1000-node network). https://manual.raspberryshake.org/_downloads/SpecificationsforRaspberryShake4DMEMSV4.pdf and https://shop.raspberryshake.org/product/turnkey-iot-home-earth-monitor-rs-4d/
28. Community Seismic Network (Caltech) — 3-axis MEMS + Raspberry Pi, 50 sps, strong-motion. http://csn.caltech.edu/ and http://csn.caltech.edu/sensor/
29. Cochran, E. S. et al., Quake-Catcher Network — USB/laptop MEMS, <1% cost of traditional network; reliability study https://www.researchgate.net/publication/276419887

---

*Notes on figures: numbers attributed to "search/..." in the text were obtained from the cited primary or secondary source's abstract/summary via literature search; for the few values where only the abstract was machine-readable (e.g. the L-4C self-noise PDF and the Evans 2014 full tables), the quantitative claims are stated at the level supported by the abstract/summary and should be confirmed against the full PDF before use in a publication. This matters for the project's publication-rigor bar: treat §1.4's geophone ng/√Hz figures and §2.2's dB noise-floor numbers as load-bearing and verify against the primary tables.*
