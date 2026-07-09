# Multi-Sensor Fusion Architectures and "World Models" for Perimeter / Threat Situational Awareness

*Research report — geophone-centric perimeter & intrusion-detection system. Compiled 2026-06-23.*

---

## Abstract

This report surveys the landscape of multi-sensor fusion for a seismic-centric perimeter security system whose end goal is a "world model" that fuses one-or-more geophones with cameras, radar, acoustic, and RF sensors to detect, classify, localize, track, and assess intent. It covers the classical fusion taxonomy (data/feature/decision-level; Bayesian; Kalman/EKF/UKF/particle filters; factor graphs; Dempster-Shafer; multi-object tracking via JPDA/MHT/PHD), the modern ML stack (cross-attention transformers, GNNs, and state-space/latent-dynamics "world models" such as Dreamer and JEPA), and the per-modality value-add of pairing seismic with each complementary sensor. It then maps the deployed prior art (REMBASS/Scorpion-class UGS, commercial PIDS, and AI battle-management platforms like Anduril Lattice) and closes with an honest separation of what is buildable on an edge/distributed budget today versus research frontier — and where the genuine economic/security moat actually lives. **Key honest finding: the ML "world model" paradigm (Dreamer/JEPA-style learned latent dynamics) is largely a buzzword for a sparse, discrete-event security sensing problem; the defensible engineering is classical probabilistic fusion + tracking + calibrated per-modality classifiers, with ML used surgically at the feature/classifier level.**

---

## 1. The classical sensor-fusion taxonomy

### 1.1 Where you fuse: data / feature / decision (early / intermediate / late / hybrid)

The canonical organizing axis is *the stage of the perception pipeline at which information from different sensors is combined* — this is consistent across the autonomous-driving, remote-sensing and defense literature [1][2][3].

- **Data-level (early) fusion** — combine raw or minimally-processed signals (e.g., a geophone waveform spatially/temporally aligned with a microphone waveform) before feature extraction. Highest information content, but requires tight time/space registration and is fragile to one sensor's noise/dropout.
- **Feature-level (intermediate / "deep") fusion** — extract features per modality (spectrogram, kurtosis, cadence, MFCCs, range-Doppler map), then mix in a shared feature space by concatenation, element-wise ops, or learned attention. Best accuracy/robustness trade-off for heterogeneous sensors and the natural home for ML fusion.
- **Decision-level (late) fusion** — each sensor runs its own classifier/tracker and only the *decisions* (labels, probabilities, tracks) are combined (Bayes, voting, Dempster-Shafer). Most robust to sensor failure, lowest bandwidth, easiest to certify — and the natural fit for a *distributed* edge network where each node has limited power.
- **Hybrid fusion** — cascades levels (e.g., one branch contributes a decision while another contributes features), to capture the strengths of each [1].

For a geophone perimeter system the practical recommendation is **decision-level (or hybrid) fusion across nodes, feature-level fusion within a node**: a seismic+acoustic node can afford feature fusion locally, but the wide-area picture should be assembled from compact per-node decisions/tracks to survive node loss and conserve radio/power.

Sources: [1] arXiv 2202.02703 *Multi-modal Sensor Fusion for Auto Driving Perception: A Survey*; [2] arXiv 2506.21885 *Integrating Multi-Modal Sensors*; [3] ACM Computing Surveys, *Deep Multimodal Data Fusion* (10.1145/3649447).

### 1.2 The JDL / situational-awareness reference model

The defense reference frame is the **JDL (Joint Directors of Laboratories) Data Fusion Model** (mid-1980s, revised by Steinberg/Bowman/White) [4][5]. Its levels map almost one-to-one onto the project's stated goals:

- **L0 – Source preprocessing** (signal/image processing, spatial/temporal alignment, filtering) → raw geophone/mic/radar conditioning.
- **L1 – Object refinement** (position + identity estimation) → *detect + classify + localize* a footstep/vehicle.
- **L2 – Situation refinement** (relationships among entities/events) → "three people moving line-abreast toward the fence."
- **L3 – Threat / impact refinement** (risk, intent, opportunity) → *assess intent*.
- **L4 – Process refinement** (resource/sensor management) → cue the camera, wake a node, retask radar.

This is the most useful spine for the system architecture, because it makes "world model" concrete: the owner's vision is essentially *climbing from L1 to L3* with L4 closing the loop. It also clarifies that "intent assessment" (L3) is a genuinely hard, under-determined inference, not a classifier output.

Sources: [4] DTIC ADA391479, Steinberg/Bowman/White *Revisions to the JDL Data Fusion Model*; [5] GlobalSpec ref. 4.4 *The JDL Model*.

### 1.3 Bayesian fusion and the recursive-estimation family (Kalman → particle)

For **tracking** (the L1 state-over-time problem), the workhorses are recursive Bayesian filters:

- **Kalman Filter (KF)** — optimal for linear-Gaussian dynamics/measurements. Constant-velocity/constant-acceleration target models on a fence line are nearly linear, so a KF or a small bank of KFs is often sufficient and extremely cheap.
- **Extended KF (EKF)** — linearizes nonlinear models (e.g., bearing-only or range measurements) via Jacobians.
- **Unscented KF (UKF)** — propagates a deterministic set of sigma points through the nonlinearity; better than EKF for strong nonlinearity, no Jacobians, still cheap.
- **Particle Filter (PF)** — represents arbitrary (multimodal, non-Gaussian) posteriors with samples; needed when the posterior is genuinely non-Gaussian (e.g., severe multipath, ambiguous bearings) but O(N_particles) per step and the most expensive of the four.

All are instances of recursive Bayesian estimation; the choice is dictated by how nonlinear/non-Gaussian the seismic-localization geometry is. For a fence-line geometry with a few nodes, EKF/UKF is the sweet spot.

### 1.4 Factor graphs (smoothing & mapping)

**Factor graphs** generalize filtering into *smoothing* — instead of only propagating the latest state, you optimize over a window (or all) of past states given all measurements, as a sparse nonlinear least-squares problem [6][7]. The reference implementation is **GTSAM** with the **iSAM/iSAM2** incremental solvers (Bayes-tree updates that keep it real-time). For a perimeter network this is the right formalism if you want to (a) fuse asynchronous, multi-rate measurements from heterogeneous nodes into one consistent spatiotemporal track, (b) jointly calibrate node positions/clock offsets, and (c) re-linearize as new evidence arrives. It is heavier than a KF but is the principled way to fuse *delayed/out-of-order* edge measurements — a real issue in a radio-linked sensor field.

Sources: [6] GTSAM tutorial, *Factor Graphs and GTSAM* (gtsam.org); [7] arXiv 2009.11097 *Factor Graph-Based Smoothing Without Matrix Inversion*.

### 1.5 Dempster-Shafer (evidential) fusion

**Dempster-Shafer (DS) / evidence theory** assigns "mass" to *sets* of hypotheses (e.g., {human}, {animal}, {human, animal} = "ambiguous"), explicitly modeling ignorance rather than forcing a prior [8][9]. Strength: it represents "I don't know" cleanly, which matters when a single seismic node genuinely cannot separate human from animal. Weakness: **Dempster's combination rule produces counter-intuitive results under highly conflicting evidence** (the classic Zadeh paradox), so naive use is dangerous; modern variants reweight evidence by a belief-entropy/credibility measure before combining [8]. For this project DS is a reasonable *decision-level* fuser across modalities (seismic says "ambiguous", camera says "human" → resolve), but it must use a conflict-aware combination rule, and in practice a well-calibrated Bayesian fuser is often simpler and equally effective.

Sources: [8] Sensors 19(21):4810 *Paradox Elimination in Dempster-Shafer Combination Rule*; [9] arXiv 1906.09769 *Fault Matters: Sensor Data Fusion … Dempster-Shafer … IoT*.

### 1.6 Multi-object tracking: JPDA, MHT, and Random-Finite-Set / PHD

When more than one target is present, the hard part is **data association** (which measurement belongs to which track) [10][11]:

- **GNN/Global Nearest Neighbor** — cheapest, assigns each measurement to its single best track; brittle in clutter.
- **JPDA (Joint Probabilistic Data Association, Bar-Shalom)** — soft-assigns measurements probabilistically across a *known/fixed* number of targets; robust in clutter, moderate cost.
- **MHT (Multiple Hypothesis Tracking, Reid 1978)** — defers association by maintaining a tree of hypotheses across time; very capable but combinatorial, requires pruning/gating to stay tractable.
- **Random Finite Set (RFS) / PHD filter (Mahler FISST)** — treats the whole set of targets as a single set-valued state, *avoids explicit association*, and naturally handles unknown/time-varying target count, births and deaths; the **PHD filter** is the computationally cheap first-moment approximation, with CPHD and labeled-RFS (GLMB) variants adding track identity [10][11].

For a perimeter line with a handful of simultaneous intruders, **JPDA or a PHD filter** is the realistic choice; full MHT is usually overkill for the target counts and node densities involved.

Sources: [10] ResearchGate *A Survey of PHD Filtering Method Based on Random Finite Set*; [11] Stanford notes, *Probability Hypothesis Density Filter Implementation and Application*.

---

## 2. The modern ML fusion stack — and an honest "world model" assessment

### 2.1 Cross-attention transformers / multimodal encoders

The dominant modern feature-level fuser is **cross-attention**: each modality is encoded into tokens, and attention lets one modality query another so the model learns *which* parts of each stream are relevant and aligns mis-registered modalities [12][13][14]. Representative defense-adjacent instances: AFTR (adaptive spatial cross-attention + spatial-temporal self-attention for multi-sensor 3D detection) [12]; cross-modal fusion transformers for multispectral pedestrian detection [13]; cross-channel attention for object detection in remote-sensing imagery [14]. **Relevance to this project:** cross-attention is a strong *within-node feature fuser* if you have a camera + radar + seismic co-located, and is the right tool to resolve human-vs-animal by jointly attending to seismic cadence and a radar micro-Doppler signature. It is *not* appropriate as the wide-area fuser on a low-power distributed field — transformers are memory/FLOP-hungry relative to a microcontroller budget (see §5).

Sources: [12] AFTR, PMC10611098; [13] *Multimodal fusion transformer … multispectral pedestrian detection*, Sci. Reports s41598-025-03567-7; [14] arXiv 2310.13876 *Multimodal Transformer Using Cross-Channel Attention … Remote Sensing*.

### 2.2 Graph neural networks for sensor networks

**GNNs** model the sensor field as a graph (nodes = sensors/tracks, edges = spatial/temporal/communication relationships) and aggregate relational information — a natural fit for a *distributed* perimeter where node geometry matters [15][16]. Demonstrated uses: GNN-based multi-sensor fusion for robust UAV tracking (UAVs as nodes, spatio-temporal relations as edges) [15]; spatial-temporal GNNs that aggregate multi-sensor time series across known sensor locations [16]. **Relevance:** a GNN is a credible *L1→L2* layer (turning per-node detections into a relational situation picture, exploiting that real intruders create *spatially-coherent* sequences of node activations while noise does not). This is one of the few places modern ML adds genuine, non-buzzword value over classical methods, because it directly encodes the sensor-network topology.

Sources: [15] *Graph neural network-tracker … multi-sensor fusion … UAV tracking*, PMC12267811; [16] *Attention-aware temporal-spatial GNN with multi-sensor information fusion*, ScienceDirect S095070512300641X.

### 2.3 State-space / latent-dynamics "world models" — Dreamer, RSSM, JEPA

This is the term the owner is excited about, so it deserves precision. In the ML literature a **"world model"** is a *learned, latent forward dynamics model* trained to predict the future of an environment, used to plan or train a policy by "imagination":

- **RSSM / Dreamer (DreamerV1-V3)** — a Recurrent State-Space Model with deterministic + stochastic latent state learns environment dynamics + reward from a replay buffer, then an actor-critic is trained entirely on *imagined* latent rollouts. DreamerV3 is a general MBRL algorithm that learns a compact latent world model for efficient policy optimization [17]. TransDreamer swaps the RNN for a transformer [17].
- **JEPA (Joint-Embedding Predictive Architecture, LeCun 2022)** — self-supervised; predicts the *representation* (not pixels) of a masked/target part from a context part, learning semantic structure while discarding unpredictable noise. Lineage: I-JEPA (images, 2023) → V-JEPA (video, 2024) → **V-JEPA 2 / V-JEPA 2-AC (2025)**, a ~1.2B video model whose 300M action-conditioned head does **zero-shot robot pick-and-place via model-predictive control in latent space** [18][19].

**Honest assessment — does this paradigm fit a sparse-sensor security system?**

Mostly **no**, and it is important to say so to a reviewer/investor:

1. **World models earn their keep when (a) the observation stream is high-dimensional and dense (video, proprioception), (b) you need to *plan actions* whose consequences must be imagined, and (c) you can collect millions of frames/hours of self-supervised data.** A geophone perimeter is the opposite: sparse, low-dimensional, mostly-idle, *discrete-event* sensing, and the system's "actions" (cue a camera, raise an alarm) have trivial, well-understood dynamics that need no learned simulator.
2. **The hard problems here — data association, false-alarm suppression, human/animal disambiguation, localization — are estimation/classification problems, not control problems.** Dreamer/JEPA solve control-by-imagination; they do not, by themselves, give you calibrated detection probabilities or track continuity, which is what the product actually sells.
3. **Where a "world-model-flavored" idea is legitimately useful:** (i) a *learned latent dynamics prior* over how targets move along the perimeter can regularize tracking (this is just a learned motion model inside a classical filter/factor graph — valuable, but not Dreamer); (ii) JEPA-style **self-supervised pretraining on unlabeled seismic** is a real, defensible win for the chronic problem of scarce labeled intrusion data — predicting masked future seismic representations is a sound way to learn features without labels. That is the honest, non-hype use of the JEPA idea for this domain.

**Bottom line:** "World model" as a *systems concept* (a fused, predictive situational picture across sensors → the JDL L2/L3 stack) is exactly the right product vision. "World model" as the *ML architecture* (Dreamer/JEPA learned latent dynamics for planning) is a poor fit for sparse security sensing and should be treated as a buzzword unless narrowly scoped to self-supervised seismic representation learning or a learned motion prior. Lead with classical probabilistic fusion + tracking; use ML at the classifier/feature level and (optionally) for self-supervised pretraining.

Sources: [17] *DreamerV3 Algorithm* (EmergentMind), arXiv 2202.09481 *TransDreamer*; [18] arXiv 2506.09985 *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* (Meta AI); [19] *V-JEPA-2-AC: Video World Modeling for Robotics* (EmergentMind).

---

## 3. Value-add of fusing seismic with each complementary modality

Seismic/geophone sensing is uniquely good at **covert, all-weather, day/night, line-of-sight-independent ground detection**: it picks up footsteps that *cannot be heard acoustically*, and footstep impulses are separable from vehicles/wind by their impulsive, high-kurtosis character [20][21]. Reported single-modality footstep/vehicle classification rates sit around **~86-89%** with SVM/GMM/CNN features [20][21] — good for detection, but with two structural weaknesses: (1) **human-vs-animal ambiguity** (quadruped and biped gait produce very similar rhythmic seismic patterns; spectral analysis alone fails) [21], and (2) **poor bearing/identity** from a single node. Every complementary modality is chosen to patch a specific seismic weakness:

| Complementary modality | What it fixes for the geophone | Mechanism / evidence |
|---|---|---|
| **Acoustic (mic / array)** | Adds **bearing/DoA** and a second cadence cue; near-free to co-locate. Seismic+acoustic regression localizes targets (seismic→range, acoustic→direction). | Acoustic/seismic UGS fusion; field arrays reach **~5° (compact) to 1-2° (extended)** bearing; shooter-localization fusion reaches **~1 m 3D** [22][23]. |
| **Camera / EO-IR** | **Identity & false-alarm kill**: turns "ambiguous footstep" into "confirmed human"; resolves human-vs-animal that seismic cannot. Best used *cued* by seismic to save power/bandwidth. | Radar/seismic-cued cameras + AI video analytics are the standard false-alarm-reduction layer in commercial PIDS [24]; "radar detects, camera identifies" is best practice [24]. |
| **Radar (FMCW / micro-Doppler)** | **Range + velocity + micro-Doppler class** (limb vs wheel vs rotor); works through some obscurants; resolves human/animal via limb-motion signature. | DCNNs on micro-Doppler reach **~97.6% human detection / ~90.9% activity classification**; micro-Doppler separates birds/UAVs [25][26]. |
| **RF / SIGINT** | **Aerial & comms-emitting threats**; detects drone↔controller links. | RF analyzers detect comms-emitting drones (but miss autonomous/tethered ones — complementary to radar micro-Doppler) [26]. |
| **PIR / magnetic (classic UGS plug-ins)** | Cheap **direction-of-movement** (PIR) and **ferrous/vehicle** confirmation (magnetic). | REMBASS uses seismic/acoustic + PIR (direction) + magnetic (ferrous) exactly this way [27][28]. |

The recurring theme: **fusion's dominant economic value is false-alarm reduction and human/animal disambiguation**, because false alarms are what make perimeter systems get switched off in the field. Cross-referencing radar + vibration + camera lets each sensor veto the others' nuisance triggers (wildlife, wind, rain) [24].

Sources: [20] *Footstep Detection and Classification Algorithms based Seismic Sensor* (ResearchGate); [21] *Robust discrimination of human footsteps using seismic signals* / *Exemplar Selection … Human from Animal Footsteps* (HLVD 2011); [22] NATO STO RTO-MP-SET-107-17 *Acoustic/Seismic Ground Sensors* (referenced via abstract — direct PDF access 403); [23] ScienceDirect S0924424716306458 *Sensor fusion … Shooter Localization*; [24] Magos Systems / Gato Security PIDS false-alarm fusion notes; [25] MDPI Electronics 14(24):4831 *Lightweight CNN … Micro-Doppler UAV*; [26] Robin Radar *How Micro-Doppler Radar Works* / arXiv 1901.07703 *Micro-UAV Detection from RF*; [27][28] REMBASS — see §4.

---

## 4. Deployed prior art (academic + commercial/defense)

### 4.1 Military UGS — REMBASS / I-REMBASS / Scorpion

The reference deployed system is the U.S. Army **AN/GSR-8 REMBASS-II (Remotely Monitored Battlefield Sensor System)** [27][28]: an all-weather, day/night UGS that **detects, classifies, and determines direction of movement** of personnel and vehicles. Architecture and numbers worth internalizing:

- **Sensor suite:** a base **seismic/acoustic multi-sensor** (personnel detection **~75 m**), a **PIR** module (direction-of-movement, **~30 m**), and a **magnetic** module (ferrous/vehicle, **~3 m**) — i.e., decision-level multi-modal fusion at the node, with each modality contributing a different L1 attribute [27][28].
- **Endurance/power:** passive sensors stay in ultra-low-power idle and run **unattended up to ~30 days**, waking on ambient-energy change [28]. This is the canonical edge power model the project must match.
- **Scorpion** is a later CF-UGS-family sensor (Northrop Grumman-Xetron lineage) in the same lineage of low-power, air/hand-emplaced unattended sensors [27].

These systems are decades-old proof that **seismic+acoustic+PIR+magnetic decision-level fusion on a multi-day power budget** is a fielded, validated capability — the project's classical core is *not* speculative.

Sources: [27] Wikipedia *Unattended ground sensor* / FAS & GlobalSecurity REMBASS pages; [28] Defense-Update *REMBASS II*.

### 4.2 Commercial PIDS (perimeter intrusion detection)

Commercial perimeter security has converged on **multi-sensor fusion + AI analytics for false-alarm reduction** as the differentiator [24]: fence-mounted vibration/fiber sensors + ground radar + PTZ cameras, fused so that radar *detects/tracks* and the camera *classifies/confirms*, with ML filtering wildlife/weather nuisance alarms (Magos, Navtech, Sintela, and integrator stacks). The explicit selling point is the same as the project's: **cut false alarms while keeping detection high** [24].

Source: [24] Magos Systems *Radar Sensor Fusion …*; Gato Security *PIDS false alarms*; Navtech/Sintela PIDS pages.

### 4.3 AI battle-management / situational-awareness platforms

At the high end, **Anduril Lattice** is the current archetype of a "world model"-as-product: an AI battle-management layer that **fuses thousands of heterogeneous sensors/effectors into a single track picture**, classifies and tracks targets, and closes the L4 loop (cueing, autonomy, fires) [29]. Notably it underpins **200+ autonomous surveillance towers on the U.S. southern border** ("virtual border wall") and is being adopted as a C-UAS fire-control core [29]. **Elbit America has publicly partnered with Anduril (Team SIGMA)** — directly relevant to the project's stated Elbit interest, and a signal that the buyer-side appetite is for *fusion software that turns sensors into an operational picture*, not for another raw sensor [29]. This is the commercial embodiment of the JDL L2/L3 vision and the realistic shape of "world model" the project should aim its narrative at.

Source: [29] Anduril *Lattice — Command & Control* (anduril.com); DefenseScoop / Army-Technology coverage; Defence-Blog *Team SIGMA* (Elbit America + Anduril).

---

## 5. Edge/distributed reality vs. research frontier — and the moat

### 5.1 What is realistic on an edge node today

The deployable envelope is well-characterized by **TinyML / TFLite-Micro** practice [30][31]:

- Microcontroller-class nodes: **~1 MB flash / ~256 KB SRAM**, INT8 post-training quantization shrinking models **4-8×** to **~286-536 KB** deployable footprints, with per-inference energy on the order of **~10-22 J** and multi-year coin-cell-to-battery operation for duty-cycled, wake-on-event sensing [30][31].
- This budget comfortably supports: classical DSP features (kurtosis, cadence, spectral), small CNNs/SVMs/GMMs for per-node seismic/acoustic classification, a Kalman/EKF/UKF or PHD tracker, and decision-level (Bayes/DS) fusion of compact messages. **REMBASS-class behavior is squarely achievable** [27][28].
- It does **not** support on-node cross-attention transformers, GNN-over-the-whole-field, or any Dreamer/JEPA latent-dynamics model. Those, if used at all, belong on a gateway/edge-server or in the cloud, on *features/decisions* streamed up from nodes.

**Recommended split:** heavy ML (cross-attention within a rich node, GNN situation layer, optional self-supervised seismic pretraining) trained offline / run at a gateway; nodes run cheap classical detection+classification+tracking and send tracks/decisions; a factor-graph or PHD fuser at the gateway assembles the wide-area picture (JDL L1→L2), with L3 intent assessment as a separate, clearly-bounded inference layer.

### 5.2 Frontier (real but not yet productizable here)

- Self-supervised (JEPA-style) pretraining on large unlabeled seismic corpora to beat the **label-scarcity** problem — promising, unproven at product scale for this modality.
- Learned latent **motion priors** inside classical filters (a thin, honest slice of the "world model" idea).
- Labeled-RFS (GLMB) trackers and learned data-association — capable but compute-heavy for an edge field.
- True multimodal "world model" planning (Dreamer/V-JEPA-2-AC) — relevant to *robotic/effector* platforms (a patrol drone), **not** to passive perimeter sensing.

### 5.3 Where the genuine moat is

The defensible value is **not** any single algorithm above — all are published. The moat is:

1. **A validated, labeled, multimodal field dataset** (seismic + acoustic + camera/radar, with ground truth, across terrains/seasons/animals). This is the scarce asset and the thing reviewers/buyers cannot replicate cheaply. (Ties directly to the project's publication-rigor and confound concerns — selected/unvalidated numbers will be the reviewer's first target.)
2. **Calibrated false-alarm performance and human/animal disambiguation in the field** — the metric customers actually pay for and the reason REMBASS-class systems get trusted or abandoned.
3. **The fusion + power-budget engineering** that makes decision-level fusion + tracking work across an unreliable, multi-day-endurance radio field (the JDL L1→L3 stack on a TinyML budget). This systems integration — not a novel net — is what Elbit/Anduril-type buyers actually procure.

Sources: [30] *Deploying TinyML for energy-efficient object detection … low-power edge AI* (Nature Sci. Reports s41598-025-27818-9); [31] *TinyML: Enabling Inference of Deep Learning Models on Ultra-Low-Power IoT Edge Devices* (PMC9227753).

---

## 6. One-paragraph recommendation

Architect the system on the **JDL spine** with **classical probabilistic fusion as the load-bearing core**: per-node DSP + small ML classifiers (seismic/acoustic) → decision-level Bayes/Dempster-Shafer fusion across nodes → an EKF/UKF or PHD/JPDA multi-target tracker (factor-graph smoothing at the gateway for asynchronous/delayed fusion) → a GNN or rule-based situation layer (L2) → a bounded intent-assessment layer (L3) → L4 sensor cueing (seismic wakes camera/radar). Use modern ML **surgically** — cross-attention for rich-node feature fusion, a GNN for the situation layer, and (the only legitimate "world-model" move) **JEPA-style self-supervised pretraining on unlabeled seismic** to fight label scarcity. Treat Dreamer/JEPA *latent-dynamics-for-planning* as out-of-scope buzzword for passive sensing. Compete on **dataset, calibrated field false-alarm/disambiguation performance, and edge-fusion engineering**, not on a novel architecture.

---

## Source list (URLs)

1. https://arxiv.org/html/2202.02703v3 — Multi-modal Sensor Fusion for Auto Driving Perception: A Survey
2. https://arxiv.org/pdf/2506.21885 — Integrating Multi-Modal Sensors: A Review of Fusion Techniques
3. https://dl.acm.org/doi/full/10.1145/3649447 — Deep Multimodal Data Fusion (ACM Computing Surveys)
4. https://apps.dtic.mil/sti/tr/pdf/ADA391479.pdf — Steinberg et al., Revisions to the JDL Data Fusion Model
5. https://www.globalspec.com/reference/50956/203279/4-4-the-jdl-model — The JDL Model
6. https://gtsam.org/tutorials/intro.html — Factor Graphs and GTSAM
7. https://arxiv.org/pdf/2009.11097 — Factor Graph-Based Smoothing Without Matrix Inversion
8. https://mdpi.com/1424-8220/19/21/4810/htm — Paradox Elimination in Dempster-Shafer Combination Rule
9. https://arxiv.org/pdf/1906.09769 — Fault Matters: Sensor Data Fusion using Dempster-Shafer (IoT)
10. https://www.researchgate.net/publication/317280863_A_Survey_of_PHD_Filtering_Method_Based_on_Random_Finite_Set — PHD Filtering Survey
11. https://web.stanford.edu/~blange/data/Probability%20Hypothesis%20Density%20Filter%20Implementation%20and%20Application.pdf — PHD Filter Implementation
12. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10611098/ — AFTR: Adaptive Fusion Transformer for 3D Object Detection
13. https://www.nature.com/articles/s41598-025-03567-7 — Multimodal fusion transformer for multispectral pedestrian detection
14. https://arxiv.org/html/2310.13876v3 — Multimodal Transformer Using Cross-Channel Attention (remote sensing)
15. https://pmc.ncbi.nlm.nih.gov/articles/PMC12267811/ — GNN-tracker: GNN-based multi-sensor fusion for UAV tracking
16. https://www.sciencedirect.com/science/article/abs/pii/S095070512300641X — Attention-aware temporal-spatial GNN with multi-sensor fusion
17. https://arxiv.org/html/2202.09481v2 — TransDreamer (transformer world model); DreamerV3 overview: https://www.emergentmind.com/topics/dreamerv3-algorithm
18. https://arxiv.org/abs/2506.09985 — V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (Meta)
19. https://www.emergentmind.com/topics/v-jepa-2-ac — V-JEPA-2-AC: Video World Modeling for Robotics
20. https://www.researchgate.net/publication/276566493_Footstep_Detection_and_Classification_Algorithms_based_Seismic_Sensor — Footstep Detection/Classification (seismic)
21. https://posenhuang.github.io/papers/Exemplar_Selection_HLVD2011.pdf — Exemplar Selection: Human vs Animal Footsteps
22. https://publications.sto.nato.int/publications/STO%20Meeting%20Proceedings/RTO-MP-SET-107/MP-SET-107-17.pdf — NATO STO, Acoustic/Seismic Ground Sensors (PDF 403 at fetch; cited via search abstract)
23. https://www.sciencedirect.com/science/article/abs/pii/S0924424716306458 — Sensor fusion … Shooter Localization Systems
24. https://magossystems.com/case-study/radar-sensor-fusion-a-game-changer-in-eliminating-false-alarms/ — Radar Sensor Fusion for false-alarm reduction (PIDS); https://www.gato-security.com/how-to-avoid-false-alarms-with-perimeter-intrusion-detection-systems-pids/
25. https://www.mdpi.com/2079-9292/14/24/4831 — Lightweight CNN Micro-Doppler UAV Detection; https://pmc.ncbi.nlm.nih.gov/articles/PMC10780901/ — Drone Detection/Classification review
26. https://www.robinradar.com/blog/how-micro-doppler-radar-works — How Micro-Doppler Radar Works; https://arxiv.org/pdf/1901.07703 — Micro-UAV Detection from RF
27. https://en.wikipedia.org/wiki/Unattended_ground_sensor — Unattended Ground Sensor (REMBASS/Scorpion); https://man.fas.org/dod-101/sys/land/rembass.htm
28. https://defense-update.com/20060107_rembass-ii-remotely-monitored-battlefield-sensor-system.html — REMBASS II
29. https://www.anduril.com/lattice/command-and-control — Anduril Lattice C2; https://defence-blog.com/silicon-valley-meets-the-cannon-anduril-joins-team-sigma/ — Elbit America + Anduril (Team SIGMA)
30. https://www.nature.com/articles/s41598-025-27818-9 — Deploying TinyML for energy-efficient edge AI
31. https://pmc.ncbi.nlm.nih.gov/articles/PMC9227753/ — TinyML: Inference of DL Models on Ultra-Low-Power Edge Devices

*Note on sourcing: figures quoted (classification rates, bearing accuracies, power/memory budgets) are drawn from the abstracts/summaries of the cited works as surfaced in search; the NATO STO PDF [22] returned HTTP 403 on direct fetch and is cited via its indexed abstract. For a publication-grade article, primary figures should be re-verified against the full PDFs before inclusion.*
