# Model Architectures and SNN Energy Cost for Single-Geophone Seismic Perimeter Classification

**Scope.** This report addresses two questions for a seismic perimeter system that classifies windows of single-geophone signal (1000 Hz) into human / vehicle / animal / nothing, currently using a feature-input PLIF spiking neural network (SpikingJelly), ~216K–936K params, T = 4, edge-deployable.

- **Part A** — architecture comparison (SNN vs CNN, RNN/LSTM/GRU, TCN, transformer, SSM/Mamba) for footstep/vehicle/animal seismic-vibration classification, plus a window-length recommendation (3 s → 10 s).
- **Part B** — the standard SNN energy-cost formula, the Horowitz (2014) 45 nm reference energies, how SpikingJelly / the `syops-counter` library report it, and a worked example for the project's MLP-style SNN.

Date: 2026-06-23.

---

## PART A — Architectures and Window Length

### A.0 What the task actually demands

Three properties dominate the architecture choice here:

1. **Temporal structure is the signal.** Footstep/vehicle/animal discrimination on a geophone is fundamentally about *cadence and periodicity* (step interval, gait frequency, harmonic structure of an engine), not a single instantaneous spectrum. The classic seismic-discrimination literature is explicit that the discriminating feature is the *temporal gait pattern / cadence*, extracted over a window long enough to contain multiple strides (Park & Dibazar, US 2010/0260011 A1; Pakhomov et al., footstep characterization).
2. **Hard edge budget.** Single MCU-class node, battery/solar, always-on. Energy-per-inference and latency are first-class constraints, not afterthoughts.
3. **Small, imbalanced, partly-synthetic data.** Real labeled footstep/animal seismic data is scarce, so data efficiency matters.

### A.1 Architecture-by-architecture

#### 1-D / 2-D CNN (spectrogram CNN)
- **Accuracy:** The strongest, most reproducible baseline in the seismic-vibration literature. CNNs separating earthquakes / vehicles / noise reach >99% per class on seismic data ([Springer J. Seismology 2024](https://link.springer.com/article/10.1007/s10950-024-10267-8)); deep-learning vehicle classification from seismic data and seismometer-based pedestrian classifiers report strong results ([Sci Rep 2025](https://www.nature.com/articles/s41598-025-01684-x); [ESE 2025](https://link.springer.com/article/10.1007/s41748-025-01003-4)). For urban footstep detection a CNN reached ~0.85 (high-SNR) / 0.80 (full set) ([The Leading Edge 2020](https://library.seg.org/doi/10.1190/tle39090654.1)). A 1-D CNN matched the 2-D spectrogram CNN almost exactly (81.0% vs 81.6%) while avoiding the cost of 2-D image encoding — relevant for the edge.
- **Temporal context:** Captured only within the receptive field of the window you feed it; a 2-D spectrogram CNN sees cadence as texture, which works but is window-bound.
- **Data efficiency:** Good with augmentation; the most "default" choice and the one with the most transfer-learning support.
- **Edge energy/latency:** Excellent on INT8 MCU kernels (CMSIS-NN); the de-facto efficient baseline that SNNs are measured *against*.

#### RNN / LSTM / GRU
- **Accuracy:** On multivariate time series LSTMs reach near-perfect accuracy on clean tasks (99% on some classification benchmarks; e.g. the edge-SNN study below found LSTM/MLP ~99% where the SNN was lower) ([arXiv:2510.20997](https://arxiv.org/abs/2510.20997)).
- **Temporal context:** Native stateful modeling of step intervals — a natural fit for cadence. But long windows at 1000 Hz mean long sequences (a 3 s window = 3000 samples; 10 s = 10000), and vanilla RNNs struggle with very long sequences and are sequential (poor parallelism).
- **Data efficiency:** Moderate; prone to overfitting on small sets.
- **Edge energy/latency:** Recurrent step-by-step inference is awkward on MCUs; usually you down-sample / feature-frame first. GRU is the lighter of the three.

#### Temporal Convolutional Network (TCN)
- **Accuracy:** Comparable to or better than LSTM on many sequence tasks; dilated causal convolutions give a large receptive field cheaply.
- **Temporal context:** Receptive field is fixed by depth/dilation — a known limitation ("struggle with limited context receptive fields built through stacked dilated 1-D conv layers," noted in the TSCMamba comparison, [arXiv:2406.04419](https://arxiv.org/html/2406.04419v2)). For cadence over a 10 s window you must size dilations deliberately.
- **Data efficiency:** Good; convolutional weight sharing.
- **Edge energy/latency:** Very edge-friendly — parallel, streamable, INT8-quantizable. A TCN is arguably the strongest *conventional* candidate for this task on an MCU.

#### Transformer
- **Accuracy:** Strong, especially for short-term dynamics; "Transformer is more effective at modeling short-term dynamics" while it underperforms SSMs on long-term structure ([Mamba/Transformer hybrid study](https://arxiv.org/html/2404.14757v1)).
- **Temporal context:** Global attention captures long-range step relationships well, but cost is O(L²) in sequence length — expensive for 3000–10000-sample raw windows; needs patching/framing.
- **Data efficiency:** Data-hungry; weak on small seismic sets without pretraining.
- **Edge energy/latency:** Highest compute/memory of the group; least attractive on a microcontroller unless heavily distilled.

#### State-space models (S4 / Mamba)
- **Accuracy:** State of the art on long-sequence time-series classification; Mamba "significantly outperforms RNN, LSTM, and Transformer across all evaluation metrics" on several TS tasks and "excels at capturing long-term structures" ([TSCMamba, arXiv:2406.04419](https://arxiv.org/html/2406.04419v2); [Mamba-360 survey](https://www.sciencedirect.com/science/article/abs/pii/S0952197625012801)).
- **Temporal context:** Best-in-class for *long* context with **linear** O(L) scaling — exactly the regime a 10 s, 10000-sample window lives in. This is the architecture most naturally suited to "needs longer context to count targets."
- **Data efficiency:** Competitive; fewer params than transformers for the same context.
- **Edge energy/latency:** Linear-time and streamable (recurrent inference form), but the kernels are immature on MCUs today; tooling/quantization is far behind CNN/TCN. A near-term risk for *this* deployment, a strong medium-term option.

#### Spiking Neural Network (current choice)
- **Accuracy:** Typically **1–2 points below** a matched ANN. EdgeSpike: SNN 91.4% vs CNN 92.6% mean across five sensing tasks (gap 1.2 pp) ([arXiv:2604.27004](https://arxiv.org/html/2604.27004v1)). On a vibration predictive-maintenance task an SNN reached 97.95% accuracy ([arXiv:2506.13416](https://arxiv.org/html/2506.13416v1)). On a strict static-threshold multivariate-TS-at-the-edge task, LSTM/MLP hit ~99% while the SNN was lower — the SNN's justification there was explicitly *energy, not accuracy* ([arXiv:2510.20997](https://arxiv.org/html/2510.20997v1)).
- **Temporal context:** Stateful membrane dynamics model temporal dependence at the neuron level "without storing full input histories (transformers) or special recurrent structures (RNNs)." **Caveat:** that advantage is real for *spike/event* input; a **feature-input SNN at T = 4** (the current design) replays the same static feature vector 4 times and behaves much more like a low-precision MLP than a true temporal model. It does *not* by itself extend temporal context — the context comes from the features you compute over the window, not from T = 4.
- **Data efficiency:** Comparable to a same-size MLP; surrogate-gradient training is finicky and architecture standards are immature ("appropriate design of SNNs is less understood").
- **Edge energy/latency:** This is the SNN's whole reason to exist — see Part B. On neuromorphic silicon, 18×–47× energy reduction vs CNN at ≤9.4 ms latency (EdgeSpike); 3–4 orders of magnitude vs CPU on Loihi for vibration ([arXiv:2506.13416](https://arxiv.org/html/2506.13416v1)). **But** those savings are *realized on neuromorphic hardware* (Loihi 2, SpiNNaker 2). On a plain Cortex-M4 with sparse kernels the same study reports only **4.6×–7.9×** — and only if the firing rate is genuinely low.

### A.2 Is the SNN genuinely well-suited here?

**Honest assessment: the SNN is defensible but not clearly optimal, and its advantage is conditional.**

- If the deployment target is a **conventional MCU** (Cortex-M class) with no neuromorphic accelerator, the SNN's energy edge collapses from ~30× to single-digit (4.6×–7.9×, EdgeSpike Cortex-M4), and you pay for it with a 1–2 point accuracy loss and harder training. A well-quantized **1-D CNN or TCN** would likely match or beat the current accuracy at comparable MCU energy, with far more mature tooling.
- The SNN is genuinely the right call **only if** (a) you deploy on or plan to deploy on neuromorphic hardware (Loihi 2 / SpiNNaker 2 / Akida), **or** (b) you re-architect to consume an event/spike-encoded *stream* (not a static feature vector at T = 4), so the event-driven sparsity actually pays off, **or** (c) the always-on duty-cycle energy budget is so tight that even the MCU-level 4–8× matters more than the accuracy gap.
- The **current feature-input PLIF SNN at T = 4** is the weakest form of the SNN argument: it sacrifices accuracy for an energy benefit that is small unless run on neuromorphic silicon, and it does not exploit temporal coding. For a publication-grade claim, the SNN should be compared head-to-head against (i) a matched-size INT8 1-D CNN/TCN baseline and (ii) ideally a small Mamba/TCN for the long-window case, reporting accuracy **and** measured/`syops`-estimated energy on the *actual* target hardware.

**Recommendation (architecture).** Keep the SNN as the energy-story contender, but treat a **quantized 1-D CNN or TCN as the primary accuracy baseline**, and evaluate a small **Mamba/S4 or TCN specifically for the long-window multi-target case** (Section A.3). Do not claim an SNN energy win without specifying the hardware it is realized on — on a generic MCU the win is modest.

### A.3 Window length: would 10 s (vs 3 s) help?

**What the literature says about window length and overlap:**

- The canonical human-vs-quadruped seismic-cadence method (**Park & Dibazar / USC, US 2010/0260011 A1 and WO2010118233A2**) uses a **3 s window with ~2 s overlap** (i.e. a new decision ~every 1 s), then *partitions each window into sub-windows of one gait-period length* and averages across strides. It reports **>95% overall**: horse 98.14%, single human walk 98.54%, multiple-people walk 98.02%, human running 94.81% (running confuses with horse trot). Crucially, **multiple walkers were discriminated because the randomness of peak locations in time makes the cadence feature space "flat"** vs the sharp periodicity of a single walker ([Google Patents US20100260011](https://patents.google.com/patent/US20100260011); [Semantic Scholar](https://www.semanticscholar.org/paper/4645b0f3df16837101ae7ef8ba9023938f345904)).
- Urban-footstep deep-learning work uses **3 s scan windows assuming the next ~9 s is the same source** (runners produce detectable seismic energy for ~12 s on average), and a **5 s analysis window with 95% overlap reached F1 > 0.92** in noisy urban settings ([The Leading Edge 2020](https://library.seg.org/doi/10.1190/tle39090654.1); [eartharxiv urban running](https://eartharxiv.org/repository/object/2026/download/4229/)). Seismic CNN pipelines commonly ingest **~10 s traces** ([academia.edu footstep CNN](https://www.academia.edu/97604131/Footstep_detection_in_urban_seismic_data_with_a_convolutional_neural_network)).
- Detection is consistently reported as **harder with multiple walkers than single** — the exact regime the owner wants to crack.

**Cadence arithmetic (why longer helps for counting/disambiguation):**

- Human walking cadence ≈ 1.6–1.9 Hz (≈100–115 steps/min), i.e. **step period ≈ 0.53–0.63 s**; running and slow/pathological gait spread this to roughly 1.4–2.2 Hz ([Sci Rep stride frequency](https://www.nature.com/articles/s41598-017-01972-1)). A full gait *cycle* (two steps) is ~1.0–1.3 s.
- A **3 s** window therefore contains only **~2–3 gait cycles** (≈5–6 steps). That is barely enough to estimate a single dominant cadence and is *fragile* for:
  - **Multi-person vs multi-animal**, where you must separate two or more interleaved periodicities and judge whether the inter-step-interval distribution is sharp (one source) or "flat"/multi-modal (several). Reliable estimation of a *distribution* of step intervals, or of two coexisting fundamental frequencies and their stability, needs many more cycles.
  - **Counting targets**, which depends on cadence-statistics stability and on bearing/peak-clustering over time.
- A **10 s** window contains **~8–12 gait cycles (~16–20 steps)** — a 3–4× longer observation of the periodic process. In frequency terms, a 10 s record gives ~3.3× finer spectral resolution (Δf ≈ 1/T ≈ 0.1 Hz vs ~0.33 Hz at 3 s), which is exactly what is needed to *resolve two nearby cadences* (e.g. two people, or person vs trotting animal) rather than blurring them into one peak. The cadence literature's own design — sub-dividing the window into per-gait-period sub-windows and averaging — only becomes statistically meaningful with more cycles.

**Recommendation (window length).**

- **Yes, moving from 3 s to ~10 s is well-motivated** for the specific hard cases (multi-person vs multi-animal, target counting): longer context gives more gait cycles, finer cadence-frequency resolution, and stabler inter-step-interval statistics. This is consistent with the broader seismic/audio practice of 5–10 s windows for footstep/runner analysis.
- Use **~8 s overlap (≈2 s hop)** as proposed — a new decision every ~2 s is operationally fine for a perimeter and keeps decision latency acceptable. (The cadence patent itself runs 3 s/2 s = 1 s hop, so ~2 s hop is comfortably in range.)
- **Mitigations / caveats to test, not assume:**
  - Longer windows raise the chance of **mixed/overlapping events** (two different intruders, or footstep + vehicle) inside one window — which can *hurt* clean single-source accuracy. Consider a **hierarchical/multi-window** scheme: a short window (≈3 s) for fast detect + coarse class, a long window (≈10 s) only for the disambiguation/counting head. This preserves your current strengths while adding long-context capability where it pays.
  - For the long window, the architecture that benefits most is one built for long context: **Mamba/S4 or a dilated TCN** over framed features, *or* a feature-extractor that explicitly computes cadence statistics (autocorrelation peaks, inter-step-interval histogram, dual-cadence detector) feeding the classifier. A plain T = 4 feature-MLP-SNN will only benefit from 10 s if the *features* you compute over those 10 s encode the extra periodicity — the network does not gain temporal context for free.
  - Validate on held-out **multi-person and multi-animal** recordings specifically; report whether the 10 s head actually reduces multi-target confusion vs the 3 s baseline, since this is the reviewer-facing claim.

---

## PART B — The SNN Energy-Cost Formula

### B.1 The standard estimate

SNN energy is estimated by **counting operations and weighting each by its hardware energy**. The two operation types differ fundamentally:

- An **ANN** does a **MAC** (multiply-accumulate) at every synapse every forward pass: `acc += w * x`. Both a multiply and an add.
- An **SNN** does an **AC** (accumulate-only) **only when a presynaptic spike arrives**: because the input is a binary spike (0/1), the multiply degenerates to a gated add, `acc += w`. No multiply, and skipped entirely when there is no spike (event-driven sparsity).

So SNN energy is driven by **how many accumulate operations actually fire**, which depends on the **firing rate** and the **fan-out** (number of outgoing synapses), summed over all **T** timesteps.

#### Synaptic-operation (SOP) count

For a layer / block `l`:

```
SOPs(l) = fr(l) × T × FLOPs(l)
```

- `FLOPs(l)` = the number of MAC operations that the *equivalent ANN layer* would do in one forward pass (for a linear layer, `FLOPs = in_features × out_features`; the connection / fan-out count).
- `fr(l)` = average firing rate of the **input** spike train to block `l` (fraction of neurons spiking, ∈ [0,1]).
- `T` = number of simulation timesteps.

Equivalent neuron-level form (Spikformer / analytical-estimation papers):

```
SOP = Σ_l Σ_j  f_out^l[j] · (Σ_t o_t^l[j])
```

i.e. for every neuron `j` in every layer `l`, multiply its fan-out `f_out` by the number of spikes it emits across all `T` steps. ([SOPs/firing-rate formula: Spikformer, arXiv:2209.15425](https://arxiv.org/pdf/2209.15425); [analytical estimation, arXiv:2210.13107](https://arxiv.org/pdf/2210.13107)).

#### Energy

```
E_SNN = E_AC × Σ_l SOPs(l)                  (pure-spiking layers)
E_ANN = E_MAC × Σ_l FLOPs(l)
```

If the first layer is a direct (non-spiking) encoder that does real MACs (common — the input features are real-valued), it is charged at the MAC rate and the rest at the AC rate:

```
E_SNN = E_MAC × FLOPs(layer 1)  +  E_AC × Σ_{l≥2} SOPs(l)
```

This is exactly how Spikformer reports it: "E_MAC × FLOPs of the first conv layer, plus E_AC × Σ SOPs of the remaining SNN conv/FC/attention layers" ([arXiv:2209.15425](https://arxiv.org/pdf/2209.15425)).

#### The ANN→SNN energy ratio

```
E_SNN / E_ANN = (Σ SOPs · E_AC) / (Σ FLOPs · E_MAC)
```

Two multiplicative factors make SNNs cheaper: **(i)** AC is ~5× cheaper than MAC (`E_AC/E_MAC = 0.9/4.6 ≈ 0.196`), and **(ii)** sparsity — `SOPs/FLOPs = fr × T`, which is < 1 whenever `fr × T < 1`. The SNN wins when `fr × T × (E_AC/E_MAC) < 1`, i.e. when the network is sparse enough relative to its timestep count.

### B.2 The Horowitz (2014) 45 nm reference energies

The per-operation energies universally cited in this literature come from **Mark Horowitz, "Computing's Energy Problem (and what we can do about it)," ISSCC 2014** ([gwern mirror PDF](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf); [ResearchGate](https://www.researchgate.net/publication/271463146)). For a **45 nm, 0.9 V** process:

| Operation (32-bit) | Energy |
|---|---|
| FP **MULT** | **3.7 pJ** |
| FP **ADD** | **0.9 pJ** |
| **MAC** (= MULT + ADD) | **3.7 + 0.9 = 4.6 pJ** |
| **AC** (= ADD only) | **0.9 pJ** |
| INT32 ADD | ~0.1 pJ |
| Cache access | ~10–100 pJ (size-dependent) |
| DRAM access | ~640 pJ to 1.3–2.6 nJ (off-chip) |

So the two constants the SNN literature plugs in are:

```
E_MAC = 4.6 pJ   (32-bit FP multiply-accumulate, 45 nm)
E_AC  = 0.9 pJ   (32-bit FP accumulate / add,    45 nm)
```

A MAC costs ~5× an AC, and Horowitz's larger point is that **memory movement dwarfs arithmetic** (DRAM access ≈ 100–1000× a MAC) — which is why SNN sparsity, by reducing both ops *and* the memory traffic that feeds them, matters in practice. ([Per-op derivation confirmed across SNN papers, e.g. arXiv:1903.06379, arXiv:2210.13107.](https://arxiv.org/pdf/2210.13107))

### B.3 How SpikingJelly / `syops-counter` report it

**SpikingJelly** does not embed a fixed energy number; it gives you the **firing-rate / spike-count instrumentation** you need, and you (or a counter library) apply the formula.

- `spikingjelly.activation_based.monitor.OutputMonitor(net, neuron.LIFNode)` records every spiking layer's output spike tensor `S[t]`. With a custom transform `lambda s: s.mean()` you get each layer's **firing rate** `fr(l)` (fraction of non-zero elements), averaged over neurons and the T steps. `AttributeMonitor('v_seq', ...)` records membrane potentials. Records are read back via `monitor.records`. ([SpikingJelly monitor docs / Sci. Adv. 2023](https://www.science.org/doi/10.1126/sciadv.adi1480)).
- From per-layer `fr(l)` and the layer's `FLOPs(l)` you compute `SOPs(l) = fr(l) × T × FLOPs(l)` and then `E_SNN = Σ E_AC·SOPs(l)`.

**`syops-counter`** ([github.com/iCGY96/syops-counter](https://github.com/iCGY96/syops-counter)) automates this. It is the `ptflops`-style counter for SNNs and natively supports SpikingJelly's `IF/LIF/PLIF` nodes. It traverses the model on a dataloader and reports, **per layer and total**:

- **ACs** — accumulate ops on spike inputs,
- **MACs** — multiply-accumulate ops on non-spike (real-valued) inputs,
- **Params**, and the measured **firing rate**.

Its energy model is exactly the Horowitz-weighted sum:

```
Energy = ACs × 0.9 pJ  +  MACs × 4.6 pJ      (45 nm)
```

API:

```python
from syops import get_model_complexity_info
ops, params = get_model_complexity_info(
        net, (input_shape), dataloader,
        as_strings=True, print_per_layer_stat=True)
# ops -> (ACs, MACs); apply E_AC=0.9pJ, E_MAC=4.6pJ
```

(Note: the counter only accounts for explicit `nn` layers, not `torch.nn.functional.*` calls — wrap functional ops in modules if you want them counted.)

### B.4 Worked example — 104 → 512 → 256 → 128 feature-MLP SNN, T = 4

This matches the project's feature-input design (≈104 input features, hidden 512/256/128, PLIF neurons, T = 4). I count the **synaptic MAC-equivalents (connections)** per linear layer, then apply firing rate, T, and the energy constants. (Bias and the small final classifier head are negligible; I include a 128→4 output layer for completeness.)

**Step 1 — connections (FLOPs / MACs per forward pass) per layer:**

| Layer | in × out | Connections (FLOPs) |
|---|---|---|
| L1 104 → 512 | 104×512 | 53,248 |
| L2 512 → 256 | 512×256 | 131,072 |
| L3 256 → 128 | 256×128 | 32,768 |
| L4 128 → 4 | 128×4 | 512 |
| **Total** | | **217,600** (~0.22M MACs) |

This ~218K-MAC, ~218K-param figure sits right at the low end of the stated 216K–936K range — consistent.

**Step 2 — ANN reference energy** (same MLP run once as a dense ANN):

```
E_ANN = E_MAC × FLOPs = 4.6 pJ × 217,600 ≈ 1,000,960 pJ ≈ 1.00 nJ / inference
```

**Step 3 — SNN energy.** Assume a plausible average firing rate **fr ≈ 0.15** (mid of the 0.1–0.3 "optimal" band reported for well-trained SNNs; the vibration-SNN paper observed ~0.1) and **T = 4**.

The **first layer** receives the *real-valued feature vector* (direct encoding), so it does true MACs and is charged at `E_MAC`, run T times:

```
SOPs/MACs(L1) = T × FLOPs(L1) = 4 × 53,248 = 212,992 MACs   (real input, no firing-rate gating)
E(L1) = 4.6 pJ × 212,992 ≈ 979,763 pJ ≈ 0.98 nJ
```

The **spiking layers L2–L4** receive binary spikes; apply `SOPs = fr × T × FLOPs`, charged at `E_AC`:

```
SOPs(L2) = 0.15 × 4 × 131,072 = 78,643
SOPs(L3) = 0.15 × 4 × 32,768  = 19,661
SOPs(L4) = 0.15 × 4 × 512     =    307
Σ SOPs(L2..L4) = 98,611
E(L2..L4) = 0.9 pJ × 98,611 ≈ 88,750 pJ ≈ 0.089 nJ
```

**Total SNN energy / inference:**

```
E_SNN = 0.98 nJ (L1, MAC) + 0.089 nJ (L2–L4, AC) ≈ 1.07 nJ
```

**Energy ratio (this configuration):**

```
E_SNN / E_ANN ≈ 1.07 nJ / 1.00 nJ ≈ 1.07   →  the SNN is NOT cheaper here.
```

**Why — and the key lesson for this project:** with a **direct real-valued feature input**, the first layer's MACs are repeated **T = 4** times and dominate everything (0.98 of the 1.07 nJ). The accumulate-only savings in L2–L4 (0.089 nJ vs 0.59 nJ if those layers were ANN MACs — an ~6.6× saving on those layers alone) are swamped by the encoder cost. The SNN energy advantage **only materializes if either (a) T is small and the first layer is also spike-coded / cheap, or (b) you do not pay T× for a real-valued encoder.**

**Contrast — if the input were spike-encoded (all layers AC)** at the same fr = 0.15, T = 4:

```
Σ SOPs(all) = 0.15 × 4 × 217,600 = 130,560
E_SNN = 0.9 pJ × 130,560 ≈ 117,504 pJ ≈ 0.118 nJ
E_SNN / E_ANN ≈ 0.118 / 1.00 ≈ 0.118   →  ~8.5× cheaper than the ANN.
```

So the **architecture of the input stage decides the energy story**: a fully spike-coded version of this net is ~8× cheaper than the dense ANN; the current **real-feature + T = 4** version is roughly energy-neutral with the ANN (and loses 1–2 accuracy points). General rule from the ratio formula: **SNN wins iff `fr × T × (E_AC/E_MAC) < 1`**, i.e. iff `fr × T < 5.1`. At T = 4 you need `fr < 1.28` (easy) for the *spiking* layers — but the direct-MAC encoder bypasses that math and is the real cost driver here.

### B.5 Energy-formula takeaways for the project

1. **Report energy on the actual target hardware.** Horowitz 45 nm `(0.9 pJ AC, 4.6 pJ MAC)` is the standard *theoretical* yardstick and is fine for the paper, but the realized advantage differs wildly: ~30× on Loihi 2 / SpiNNaker 2 vs ~5–8× on a Cortex-M4 (EdgeSpike) vs ~neutral on a generic MCU for a MAC-encoder design like the current one.
2. **Instrument firing rate with SpikingJelly's `OutputMonitor` and total ops with `syops-counter`** — don't hand-wave `fr`. The whole energy claim hinges on the *measured* per-layer firing rate.
3. **The current feature-input, T = 4 SNN is the least favorable case** for the energy argument: the real-valued first layer paid T times dominates. Either reduce the encoder cost / fold it into the feature extractor, move to spike-coded input, or accept that on a plain MCU the energy win is marginal and the SNN must be justified some other way.

---

## Consolidated recommendations

1. **Architecture:** Keep the SNN as the efficiency contender but benchmark it head-to-head against a **quantized 1-D CNN / TCN** (likely equal-or-better accuracy at comparable MCU energy, mature tooling). Reserve **Mamba/S4 or dilated TCN** for the long-window disambiguation head.
2. **Window length:** Adopt **~10 s windows with ~2 s hop**, ideally in a **hierarchical** scheme (short window for detect/coarse-class, long window for multi-person-vs-multi-animal and counting). Longer context is well-justified by cadence arithmetic (3 s ≈ 2–3 gait cycles vs 10 s ≈ 8–12) and by 0.1 Hz vs 0.33 Hz cadence resolution — but the gain is only realized if the long-window **features/architecture actually encode the extra periodicity**, not automatically by a T = 4 MLP.
3. **Energy:** Use `E_SNN = E_MAC·FLOPs(encoder) + E_AC·ΣSOPs`, `SOPs = fr·T·FLOPs`, with Horowitz `0.9/4.6 pJ`; measure `fr` with SpikingJelly monitors and total ops with `syops-counter`; **report on the real target chip**, and note that the current MAC-encoder design is roughly energy-neutral vs the ANN unless re-architected.

## Key sources

- Park & Dibazar (USC), *Cadence analysis of temporal gait patterns for seismic discrimination*: [US 2010/0260011 A1](https://patents.google.com/patent/US20100260011), [WO2010118233A2](http://www.google.com/patents/WO2010118233A2?cl=en), [Semantic Scholar](https://www.semanticscholar.org/paper/4645b0f3df16837101ae7ef8ba9023938f345904)
- Footstep detection in urban seismic data with a CNN — [The Leading Edge 2020](https://library.seg.org/doi/10.1190/tle39090654.1), [academia.edu](https://www.academia.edu/97604131/Footstep_detection_in_urban_seismic_data_with_a_convolutional_neural_network)
- Deep-learning earthquake/vehicle detection (seismic, >99%) — [J. Seismology 2024](https://link.springer.com/article/10.1007/s10950-024-10267-8); vehicle classification from seismic — [Sci Rep 2025](https://www.nature.com/articles/s41598-025-01684-x); seismometer pedestrian classifier — [ESE 2025](https://link.springer.com/article/10.1007/s41748-025-01003-4)
- Urban running detection, 12 s seismic / window design — [eartharxiv](https://eartharxiv.org/repository/object/2026/download/4229/)
- Human cadence / stride frequency — [Sci Rep 2017](https://www.nature.com/articles/s41598-017-01972-1)
- TCN/LSTM/Transformer/Mamba TS comparison — [TSCMamba, arXiv:2406.04419](https://arxiv.org/html/2406.04419v2); [Mamba-360 survey](https://www.sciencedirect.com/science/article/abs/pii/S0952197625012801); [Mamba+Transformer LS forecasting](https://arxiv.org/html/2404.14757v1)
- EdgeSpike (SNN vs CNN accuracy/energy/latency) — [arXiv:2604.27004](https://arxiv.org/html/2604.27004v1)
- SNN for vibration predictive maintenance (energy, fr≈0.1, Loihi) — [arXiv:2506.13416](https://arxiv.org/html/2506.13416v1)
- SNN binary classification of multivariate TS at the edge (accuracy vs LSTM/MLP, energy framing) — [arXiv:2510.20997](https://arxiv.org/html/2510.20997v1)
- SOPs / firing-rate energy formula — [Spikformer, arXiv:2209.15425](https://arxiv.org/pdf/2209.15425); [Analytical estimation of SNN energy efficiency, arXiv:2210.13107](https://arxiv.org/pdf/2210.13107)
- Horowitz, *Computing's Energy Problem*, ISSCC 2014 (45 nm: FP MULT 3.7 pJ, FP ADD 0.9 pJ → MAC 4.6 pJ, AC 0.9 pJ) — [gwern PDF](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf), [ResearchGate](https://www.researchgate.net/publication/271463146)
- `syops-counter` (SOP/AC/MAC + 0.9/4.6 pJ energy, SpikingJelly LIF/PLIF support) — [github.com/iCGY96/syops-counter](https://github.com/iCGY96/syops-counter)
- SpikingJelly (monitors: OutputMonitor / AttributeMonitor for firing rate) — [Science Advances 2023](https://www.science.org/doi/10.1126/sciadv.adi1480), [GitHub](https://github.com/fangwei123456/spikingjelly)
